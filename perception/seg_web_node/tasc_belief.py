"""Hierarchical TASC belief over segmented objects and grip contact points."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from .feature_cost import (
    FeatureConfig,
    MultiFeatureCost,
    clip_log_probs,
    normalize_log_probs,
    robust_standardize,
)
from .grip_candidates import (
    GripCandidate,
    GripCandidateConfig,
    generate_grip_candidates,
)


EPS = 1.0e-9


@dataclass(frozen=True)
class BeliefInputObject:
    object_id: str
    label: str
    score: float
    center_goal_m: np.ndarray
    points_goal_m: np.ndarray | None = None
    grip_candidates: list[GripCandidate] | None = None


@dataclass(frozen=True)
class EETarget:
    position: np.ndarray
    orientation_xyzw: np.ndarray
    object_id: str
    grip_index: int
    object_confidence: float
    joint_confidence: float


@dataclass(frozen=True)
class TascBeliefConfig:
    enabled: bool = True
    beta: float = 1.0
    w_sc: float = 0.95
    max_prob_any_goal: float = 0.99
    leak_kappa0: float = 0.3
    leak_gamma: float = 5.0
    grip_update_hz: float = 10.0
    w_geom_prior: float = 1.0
    feature_ema_alpha: float = 0.25
    grip_match_max_distance_m: float = 0.04
    grip_track_missing_ttl_frames: int = 3
    grip_switch_dwell_frames: int = 2
    grip_switch_margin: float = 0.0
    fallback_eef_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robust_min_signal_spread: float = 0.002
    robust_scale_floor: float = 0.01
    robust_clip: float = 2.0
    log_every_sec: float = 1.0
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    grip: GripCandidateConfig = field(default_factory=GripCandidateConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "TascBeliefConfig":
        data = dict(data or {})
        leak = dict(data.get("leak", {}))
        grip = dict(data.get("grip", {}))
        return cls(
            enabled=bool(data.get("enabled", True)),
            beta=float(data.get("beta", 1.0)),
            w_sc=float(data.get("w_sc", data.get("weight_sc", 0.95))),
            max_prob_any_goal=float(data.get("max_prob_any_goal", 0.99)),
            leak_kappa0=float(leak.get("kappa0", data.get("leak_kappa0", 0.3))),
            leak_gamma=float(leak.get("gamma", data.get("leak_gamma", 5.0))),
            grip_update_hz=float(
                grip.get("grip_update_hz", data.get("grip_update_hz", 10.0))
            ),
            w_geom_prior=float(grip.get("w_geom_prior", data.get("w_geom_prior", 1.0))),
            feature_ema_alpha=float(data.get("feature_ema_alpha", 0.25)),
            grip_match_max_distance_m=float(
                grip.get(
                    "match_max_distance_m",
                    data.get("grip_match_max_distance_m", 0.04),
                )
            ),
            grip_track_missing_ttl_frames=int(
                grip.get(
                    "track_missing_ttl_frames",
                    data.get("grip_track_missing_ttl_frames", 3),
                )
            ),
            grip_switch_dwell_frames=int(
                grip.get("switch_dwell_frames", data.get("grip_switch_dwell_frames", 2))
            ),
            grip_switch_margin=float(
                grip.get("switch_margin", data.get("grip_switch_margin", 0.0))
            ),
            fallback_eef_position_m=tuple(
                float(v) for v in data.get("fallback_eef_position_m", [0, 0, 0])
            ),
            robust_min_signal_spread=float(
                dict(data.get("robust_standardize", {})).get(
                    "min_signal_spread",
                    data.get("robust_min_signal_spread", 0.002),
                )
            ),
            robust_scale_floor=float(
                dict(data.get("robust_standardize", {})).get(
                    "scale_floor",
                    data.get("robust_scale_floor", 0.01),
                )
            ),
            robust_clip=float(
                dict(data.get("robust_standardize", {})).get(
                    "clip",
                    data.get("robust_clip", 2.0),
                )
            ),
            log_every_sec=float(data.get("log_every_sec", 1.0)),
            feature=FeatureConfig.from_dict(data),
            grip=GripCandidateConfig.from_dict(grip),
        )


def _softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return scores
    return normalize_log_probs(scores)


def _prob_dict(ids: list[str], log_probs: np.ndarray) -> dict[str, float]:
    probs = np.exp(normalize_log_probs(log_probs))
    return {ids[i]: float(probs[i]) for i in range(len(ids))}


def _view_direction(target: np.ndarray, ray_origin: np.ndarray | None) -> np.ndarray | None:
    if ray_origin is None:
        return None
    return (
        np.asarray(target, dtype=np.float64).reshape(3)
        - np.asarray(ray_origin, dtype=np.float64).reshape(3)
    )


class StableObjectIdTracker:
    """Assign stable label+3D-nearest IDs to per-frame detections."""

    def __init__(
        self,
        *,
        max_distance_m: float = 0.25,
        missing_ttl_frames: int = 5,
    ) -> None:
        self.max_distance_m = float(max_distance_m)
        self.missing_ttl_frames = int(missing_ttl_frames)
        self.tracks: dict[str, dict[str, Any]] = {}
        self.next_id = 0

    def assign(self, detections: list[dict[str, Any]]) -> dict[int, str]:
        assignments: dict[int, str] = {}
        used_tracks: set[str] = set()
        valid_detections: list[tuple[int, str, np.ndarray]] = []
        for detection in detections:
            try:
                index = int(detection["index"])
                label = str(detection["label"])
                center = np.asarray(
                    detection["center_goal_m"], dtype=np.float64
                ).reshape(3)
            except Exception:
                continue
            if np.isfinite(center).all():
                valid_detections.append((index, label, center))

        for index, label, center in valid_detections:
            best_track_id = ""
            best_distance = float("inf")
            for track_id, track in self.tracks.items():
                if track_id in used_tracks or str(track["label"]) != label:
                    continue
                distance = float(np.linalg.norm(center - np.asarray(track["center"])))
                if distance < best_distance:
                    best_track_id = track_id
                    best_distance = distance
            if best_track_id and best_distance <= self.max_distance_m:
                track_id = best_track_id
            else:
                track_id = f"{label}:track{self.next_id:04d}"
                self.next_id += 1
            self.tracks[track_id] = {
                "label": label,
                "center": center.copy(),
                "missing": 0,
            }
            used_tracks.add(track_id)
            assignments[index] = track_id

        for track_id in list(self.tracks):
            if track_id in used_tracks:
                continue
            self.tracks[track_id]["missing"] = int(self.tracks[track_id]["missing"]) + 1
            if self.tracks[track_id]["missing"] > self.missing_ttl_frames:
                del self.tracks[track_id]
        return assignments


class HierarchicalTascBelief:
    """Object and grip-point belief using TASC hindsight-advantage updates."""

    def __init__(self, config: TascBeliefConfig | dict | None = None) -> None:
        self.config = (
            config
            if isinstance(config, TascBeliefConfig)
            else TascBeliefConfig.from_dict(config)
        )
        self.feature_cost = MultiFeatureCost(self.config.feature)
        self.object_ids: list[str] = []
        self.log_object_distribution = np.empty(0, dtype=np.float64)
        self.log_grip_distributions: dict[str, np.ndarray] = {}
        self.object_advantage_ema: dict[str, float] = {}
        self.grip_advantage_ema: dict[str, np.ndarray] = {}
        self.previous_grip_candidates: dict[str, list[GripCandidate]] = {}
        self.previous_grip_ids: dict[str, list[str]] = {}
        self.grip_tracks: dict[str, dict[str, dict[str, Any]]] = {}
        self.next_grip_track_index: dict[str, int] = {}
        self.selected_grip_indices: dict[str, int] = {}
        self.pending_grip_switches: dict[str, tuple[int, int]] = {}
        self.last_grip_update_monotonic = 0.0
        self.last_log_monotonic = 0.0
        self.last_target: EETarget | None = None
        self.last_telemetry: dict[str, Any] = {
            "enabled": self.config.enabled,
            "objects": [],
            "target": None,
        }

    def _align_object_distribution(self, objects: list[BeliefInputObject]) -> np.ndarray:
        ids = [obj.object_id for obj in objects]
        if not ids:
            self.object_ids = []
            self.log_object_distribution = np.empty(0, dtype=np.float64)
            return self.log_object_distribution
        previous = {
            object_id: self.log_object_distribution[i]
            for i, object_id in enumerate(self.object_ids)
        }
        priors = np.asarray(
            [max(EPS, float(obj.score)) for obj in objects], dtype=np.float64
        )
        prior_log = _softmax_from_scores(np.log(priors))
        aligned = np.empty(len(ids), dtype=np.float64)
        for index, object_id in enumerate(ids):
            aligned[index] = previous.get(object_id, prior_log[index])
        self.object_ids = ids
        self.log_object_distribution = normalize_log_probs(aligned)
        return self.log_object_distribution

    def _initial_grip_distribution(self, candidates: list[GripCandidate]) -> np.ndarray:
        scores = np.asarray(
            [self.config.w_geom_prior * float(c.geom_score) for c in candidates],
            dtype=np.float64,
        )
        return _softmax_from_scores(scores)

    def _persistent_grip_candidates(
        self, object_id: str, candidates: list[GripCandidate]
    ) -> tuple[list[GripCandidate], list[str], int]:
        """Keep a short-lived stable candidate bank for one object.

        Geometry candidates are regenerated from a noisy point cloud every frame.
        Without a stable candidate identity, the grip posterior can be copied to
        the wrong physical contact point or reset when a candidate briefly
        disappears.  This bank assigns each contact a nearest-neighbor ID and
        keeps missing contacts for a small TTL, so the posterior's domain changes
        smoothly instead of frame-by-frame.
        """
        tracks = self.grip_tracks.setdefault(object_id, {})
        used_track_ids: set[str] = set()
        match_threshold = max(0.0, float(self.config.grip_match_max_distance_m))
        for candidate in candidates:
            center = np.asarray(candidate.center, dtype=np.float64).reshape(3)
            best_id = ""
            best_distance = float("inf")
            for track_id, track in tracks.items():
                if track_id in used_track_ids:
                    continue
                previous = np.asarray(track["candidate"].center, dtype=np.float64).reshape(3)
                distance = float(np.linalg.norm(center - previous))
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance
            if best_id and best_distance <= match_threshold:
                track_id = best_id
            else:
                next_index = int(self.next_grip_track_index.get(object_id, 0))
                track_id = f"{object_id}:grip{next_index:04d}"
                self.next_grip_track_index[object_id] = next_index + 1
            tracks[track_id] = {"candidate": candidate, "missing": 0}
            used_track_ids.add(track_id)

        ttl = max(0, int(self.config.grip_track_missing_ttl_frames))
        for track_id in list(tracks):
            if track_id in used_track_ids:
                continue
            tracks[track_id]["missing"] = int(tracks[track_id].get("missing", 0)) + 1
            if tracks[track_id]["missing"] > ttl:
                del tracks[track_id]

        ordered_ids = sorted(tracks)
        persisted = [tracks[track_id]["candidate"] for track_id in ordered_ids]
        return persisted, ordered_ids, len(candidates)

    def _match_previous_grips(
        self, object_id: str, candidates: list[GripCandidate]
    ) -> dict[int, int]:
        previous = self.previous_grip_candidates.get(object_id)
        if not previous:
            return {}
        mapping: dict[int, int] = {}
        used_previous: set[int] = set()
        for new_index, candidate in enumerate(candidates):
            best_index = -1
            best_distance = float("inf")
            for previous_index, previous_candidate in enumerate(previous):
                if previous_index in used_previous:
                    continue
                distance = float(np.linalg.norm(candidate.center - previous_candidate.center))
                if distance < best_distance:
                    best_index = previous_index
                    best_distance = distance
            if best_index >= 0 and best_distance <= self.config.grip_match_max_distance_m:
                mapping[new_index] = best_index
                used_previous.add(best_index)
        return mapping

    def _align_grip_distribution(
        self,
        object_id: str,
        candidates: list[GripCandidate],
        grip_match: dict[int, int],
    ) -> np.ndarray:
        prior = self._initial_grip_distribution(candidates)
        previous = self.log_grip_distributions.get(object_id)
        if previous is None:
            current = prior
        else:
            current = prior.copy()
            for new_index, previous_index in grip_match.items():
                if previous_index < previous.shape[0]:
                    current[new_index] = previous[previous_index]
        self.log_grip_distributions[object_id] = normalize_log_probs(current)
        return self.log_grip_distributions[object_id]

    def _smooth_object_advantages(
        self, objects: list[BeliefInputObject], raw_advantages: np.ndarray
    ) -> np.ndarray:
        alpha = float(np.clip(self.config.feature_ema_alpha, 0.0, 1.0))
        smoothed = np.asarray(raw_advantages, dtype=np.float64).copy()
        next_ema: dict[str, float] = {}
        for index, obj in enumerate(objects):
            raw = float(raw_advantages[index])
            previous = self.object_advantage_ema.get(obj.object_id, raw)
            value = alpha * raw + (1.0 - alpha) * previous
            smoothed[index] = value
            next_ema[obj.object_id] = value
        self.object_advantage_ema = next_ema
        return smoothed

    def _smooth_grip_advantages(
        self,
        object_id: str,
        raw_advantages: np.ndarray,
        grip_match: dict[int, int],
    ) -> np.ndarray:
        alpha = float(np.clip(self.config.feature_ema_alpha, 0.0, 1.0))
        raw = np.asarray(raw_advantages, dtype=np.float64).reshape(-1)
        previous = self.grip_advantage_ema.get(object_id)
        smoothed = raw.copy()
        if previous is not None:
            for new_index, previous_index in grip_match.items():
                if previous_index < previous.shape[0]:
                    smoothed[new_index] = (
                        alpha * raw[new_index] + (1.0 - alpha) * previous[previous_index]
                    )
        self.grip_advantage_ema[object_id] = smoothed.copy()
        return smoothed

    @staticmethod
    def _map_previous_index(
        previous_index: int | None, grip_match: dict[int, int], candidate_count: int
    ) -> int | None:
        if previous_index is None:
            return None
        for new_index, old_index in grip_match.items():
            if old_index == previous_index:
                return new_index
        return previous_index if 0 <= previous_index < candidate_count else None

    def _select_grip_index(
        self,
        object_id: str,
        best_index: int,
        grip_probs: np.ndarray,
        grip_match: dict[int, int],
    ) -> int:
        candidate_count = int(np.asarray(grip_probs).reshape(-1).shape[0])
        selected = self._map_previous_index(
            self.selected_grip_indices.get(object_id), grip_match, candidate_count
        )
        if selected is None:
            selected = int(best_index)
            self.selected_grip_indices[object_id] = selected
            self.pending_grip_switches.pop(object_id, None)
            return selected
        if int(best_index) == selected:
            self.selected_grip_indices[object_id] = selected
            self.pending_grip_switches.pop(object_id, None)
            return selected
        margin = float(self.config.grip_switch_margin)
        if float(grip_probs[best_index]) <= float(grip_probs[selected]) + margin:
            self.selected_grip_indices[object_id] = selected
            self.pending_grip_switches.pop(object_id, None)
            return selected

        previous_pending = self.pending_grip_switches.get(object_id)
        pending_index = self._map_previous_index(
            previous_pending[0] if previous_pending else None,
            grip_match,
            candidate_count,
        )
        pending_count = previous_pending[1] if previous_pending and pending_index is not None else 0
        if pending_index == int(best_index):
            pending_count += 1
        else:
            pending_index = int(best_index)
            pending_count = 1
        if pending_count >= max(1, int(self.config.grip_switch_dwell_frames)):
            selected = int(best_index)
            self.pending_grip_switches.pop(object_id, None)
        else:
            self.pending_grip_switches[object_id] = (int(pending_index), int(pending_count))
        self.selected_grip_indices[object_id] = selected
        return selected

    def update(
        self,
        objects: list[BeliefInputObject],
        *,
        eef_position_m: np.ndarray | None = None,
        user_translation_mps: np.ndarray | None = None,
        ray_origin_m: np.ndarray | None = None,
        ray_direction: np.ndarray | None = None,
        controller_orientation_xyzw: np.ndarray | None = None,
        now_monotonic: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not self.config.enabled:
            self.last_telemetry = {"enabled": False, "objects": [], "target": None}
            return self.last_telemetry
        valid_objects: list[BeliefInputObject] = []
        for obj in objects:
            try:
                center = np.asarray(obj.center_goal_m, dtype=np.float64).reshape(3)
            except Exception:
                continue
            if np.isfinite(center).all():
                valid_objects.append(obj)
        if not valid_objects:
            self.object_ids = []
            self.log_object_distribution = np.empty(0, dtype=np.float64)
            self.last_target = None
            self.last_telemetry = {"enabled": True, "objects": [], "target": None}
            return self.last_telemetry
        valid_object_ids = {obj.object_id for obj in valid_objects}
        for object_id in list(self.grip_tracks):
            if object_id not in valid_object_ids:
                del self.grip_tracks[object_id]
        for object_id in list(self.previous_grip_ids):
            if object_id not in valid_object_ids:
                del self.previous_grip_ids[object_id]

        eef_position = (
            np.asarray(self.config.fallback_eef_position_m, dtype=np.float64)
            if eef_position_m is None
            else np.asarray(eef_position_m, dtype=np.float64).reshape(3)
        )
        self._align_object_distribution(valid_objects)
        object_feature_breakdowns = [
            self.feature_cost.breakdown(
                obj.center_goal_m,
                eef_position=eef_position,
                user_translation=user_translation_mps,
                ray_origin=ray_origin_m,
                ray_direction=ray_direction,
                agreement_direction=_view_direction(obj.center_goal_m, ray_origin_m),
            )
            for obj in valid_objects
        ]
        raw_object_advantages = np.asarray(
            [
                float(breakdown["weighted_total"])
                for breakdown in object_feature_breakdowns
            ],
            dtype=np.float64,
        )
        object_advantages = self._smooth_object_advantages(
            valid_objects, raw_object_advantages
        )
        object_update = robust_standardize(
            object_advantages,
            min_signal_spread=self.config.robust_min_signal_spread,
            scale_floor=self.config.robust_scale_floor,
            clip=self.config.robust_clip,
        )
        log_obj = (
            self.config.w_sc * self.log_object_distribution
            + self.config.beta * object_update
        )
        log_obj = normalize_log_probs(log_obj)
        confidence = float(np.max(np.exp(log_obj)))
        kappa = self.config.leak_kappa0 / (
            1.0 + np.exp(self.config.leak_gamma * confidence)
        )
        if kappa > 0.0:
            uniform = np.full_like(log_obj, -np.log(log_obj.size))
            log_obj = normalize_log_probs((1.0 - kappa) * log_obj + kappa * uniform)
        log_obj = clip_log_probs(log_obj, self.config.max_prob_any_goal)
        self.log_object_distribution = log_obj

        should_update_grips = (
            now - self.last_grip_update_monotonic
            >= 1.0 / max(1.0e-6, self.config.grip_update_hz)
        )
        telemetry_objects: list[dict[str, Any]] = []
        best_target: EETarget | None = None
        object_probs = np.exp(self.log_object_distribution)
        for obj_index, obj in enumerate(valid_objects):
            candidates = obj.grip_candidates
            if candidates is None:
                candidates = (
                    generate_grip_candidates(obj.points_goal_m, self.config.grip)
                    if obj.points_goal_m is not None
                    else []
                )
            if not candidates:
                candidates = [
                    GripCandidate(
                        center=np.asarray(obj.center_goal_m, dtype=np.float64).reshape(3),
                        approach=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                        width=0.0,
                        geom_score=0.0,
                        source="center_fallback",
                    )
                ]
            candidates, grip_ids, generated_candidate_count = (
                self._persistent_grip_candidates(obj.object_id, candidates)
            )
            previous_grip_ids = self.previous_grip_ids.get(obj.object_id, [])
            grip_match = {
                new_index: previous_grip_ids.index(track_id)
                for new_index, track_id in enumerate(grip_ids)
                if track_id in previous_grip_ids
            }
            if not grip_match:
                grip_match = self._match_previous_grips(obj.object_id, candidates)
            log_grip = self._align_grip_distribution(
                obj.object_id, candidates, grip_match
            )
            raw_grip_advantages = np.zeros(len(candidates), dtype=np.float64)
            grip_advantages = self.grip_advantage_ema.get(
                obj.object_id, raw_grip_advantages
            )
            grip_feature_breakdowns = [
                self.feature_cost.breakdown(
                    candidate.center,
                    eef_position=eef_position,
                    user_translation=user_translation_mps,
                    ray_origin=ray_origin_m,
                    ray_direction=ray_direction,
                    agreement_direction=_view_direction(candidate.center, ray_origin_m),
                )
                for candidate in candidates
            ]
            if should_update_grips:
                raw_grip_advantages = np.asarray(
                    [
                        float(breakdown["weighted_total"])
                        for breakdown in grip_feature_breakdowns
                    ],
                    dtype=np.float64,
                )
                grip_advantages = self._smooth_grip_advantages(
                    obj.object_id, raw_grip_advantages, grip_match
                )
                grip_update = robust_standardize(
                    grip_advantages,
                    min_signal_spread=self.config.robust_min_signal_spread,
                    scale_floor=self.config.robust_scale_floor,
                    clip=self.config.robust_clip,
                )
                log_grip = (
                    self.config.w_sc * log_grip + self.config.beta * grip_update
                )
                log_grip = clip_log_probs(
                    normalize_log_probs(log_grip), self.config.max_prob_any_goal
                )
                self.log_grip_distributions[obj.object_id] = log_grip
            grip_probs = np.exp(normalize_log_probs(log_grip))
            best_grip_index = int(np.argmax(grip_probs))
            grip_index = self._select_grip_index(
                obj.object_id, best_grip_index, grip_probs, grip_match
            )
            self.previous_grip_candidates[obj.object_id] = list(candidates)
            joint_confidence = float(object_probs[obj_index] * grip_probs[grip_index])
            orientation = (
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                if controller_orientation_xyzw is None
                else np.asarray(controller_orientation_xyzw, dtype=np.float64).reshape(4)
            )
            candidate_target = EETarget(
                position=np.asarray(candidates[grip_index].center, dtype=np.float64).reshape(3),
                orientation_xyzw=orientation,
                object_id=obj.object_id,
                grip_index=grip_index,
                object_confidence=float(object_probs[obj_index]),
                joint_confidence=joint_confidence,
            )
            if best_target is None or candidate_target.joint_confidence > best_target.joint_confidence:
                best_target = candidate_target
            telemetry_objects.append(
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "score": float(obj.score),
                    "object_probability": float(object_probs[obj_index]),
                    "object_advantage_raw": float(raw_object_advantages[obj_index]),
                    "object_advantage_ema": float(object_advantages[obj_index]),
                    "object_advantage_standardized": float(object_update[obj_index]),
                    "object_feature_breakdown": object_feature_breakdowns[obj_index],
                    "generated_grip_candidate_count": int(generated_candidate_count),
                    "persistent_grip_candidate_count": int(len(candidates)),
                    "grip_track_ids": list(grip_ids),
                    "grip_probabilities": [float(v) for v in grip_probs],
                    "grip_advantages": [float(v) for v in grip_advantages],
                    "selected_grip_feature_breakdown": grip_feature_breakdowns[grip_index],
                    "selected_grip_index": grip_index,
                    "selected_grip_track_id": grip_ids[grip_index] if grip_index < len(grip_ids) else "",
                    "selected_contact_goal_m": candidates[grip_index].center.tolist(),
                    "grip_candidates": [
                        {
                            **candidate.to_dict(),
                            "track_id": grip_ids[index] if index < len(grip_ids) else "",
                            "probability": float(grip_probs[index]),
                            "advantage_raw": float(raw_grip_advantages[index]),
                            "advantage_ema": float(grip_advantages[index]),
                            "feature_breakdown": grip_feature_breakdowns[index],
                        }
                        for index, candidate in enumerate(candidates)
                    ],
                }
            )
            self.previous_grip_ids[obj.object_id] = list(grip_ids)
        if should_update_grips:
            self.last_grip_update_monotonic = now
        self.feature_cost.finish_tick(user_translation_mps)
        self.last_target = best_target
        target_dict = None
        if best_target is not None:
            target_dict = {
                "object_id": best_target.object_id,
                "grip_index": int(best_target.grip_index),
                "object_confidence": float(best_target.object_confidence),
                "joint_confidence": float(best_target.joint_confidence),
                "position_goal_m": best_target.position.tolist(),
                "orientation_xyzw": best_target.orientation_xyzw.tolist(),
            }
        self.last_telemetry = {
            "enabled": True,
            "eef_position_m": eef_position.tolist(),
            "eef_velocity_mps": (
                np.asarray(user_translation_mps, dtype=np.float64).reshape(3).tolist()
                if user_translation_mps is not None
                else None
            ),
            "ray_origin_m": (
                np.asarray(ray_origin_m, dtype=np.float64).reshape(3).tolist()
                if ray_origin_m is not None
                else None
            ),
            "ray_direction_m": (
                np.asarray(ray_direction, dtype=np.float64).reshape(3).tolist()
                if ray_direction is not None
                else None
            ),
            "object_probabilities": _prob_dict(self.object_ids, self.log_object_distribution),
            "objects": telemetry_objects,
            "target": target_dict,
        }
        return self.last_telemetry

    def should_log(self, now_monotonic: float | None = None) -> bool:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        return now - self.last_log_monotonic >= max(0.0, self.config.log_every_sec)

    def mark_logged(self, now_monotonic: float | None = None) -> None:
        self.last_log_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)

    def format_log_line(self) -> str:
        target = self.last_telemetry.get("target")
        if not target:
            return "[TASC-belief] no valid target"
        objects = self.last_telemetry.get("objects", [])
        obj = next(
            (item for item in objects if item.get("object_id") == target.get("object_id")),
            {},
        )
        probs = self.last_telemetry.get("object_probabilities", {})
        prob_text = ", ".join(
            f"{key}:{value:.2f}" for key, value in sorted(probs.items())
        )
        contact = np.asarray(target["position_goal_m"], dtype=np.float64)
        grip_breakdown = obj.get("selected_grip_feature_breakdown") or {}
        contributions = grip_breakdown.get("weighted_contributions") or {}
        grip_track_ids = obj.get("grip_track_ids") or []
        generated_count = int(obj.get("generated_grip_candidate_count", len(grip_track_ids)))
        persistent_count = int(obj.get("persistent_grip_candidate_count", len(grip_track_ids)))
        distance_m = grip_breakdown.get("distance_m")
        distance_text = (
            f"ee_d={float(distance_m):.3f}m "
            if isinstance(distance_m, (int, float))
            else ""
        )
        return (
            "[TASC-belief] "
            f"target={target['object_id']} label={obj.get('label', '?')} "
            f"joint_p={target['joint_confidence']:.3f} grip={target['grip_index']} "
            f"obj_p={target.get('object_confidence', 0.0):.3f} "
            f"contact={np.round(contact, 4).tolist()} "
            f"grips={generated_count}->{persistent_count} ids={grip_track_ids} "
            f"{distance_text}"
            f"feat[d={float(contributions.get('distance', 0.0)):+.3f},"
            f"r={float(contributions.get('ray', 0.0)):+.3f},"
            f"a={float(contributions.get('agreement', 0.0)):+.3f}] "
            f"objects=[{prob_text}]"
        )
