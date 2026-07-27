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
    confidence: float


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
    fallback_eef_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
            fallback_eef_position_m=tuple(
                float(v) for v in data.get("fallback_eef_position_m", [0, 0, 0])
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

    def _align_grip_distribution(
        self,
        object_id: str,
        candidates: list[GripCandidate],
    ) -> np.ndarray:
        previous = self.log_grip_distributions.get(object_id)
        if previous is None or previous.shape[0] != len(candidates):
            current = self._initial_grip_distribution(candidates)
        else:
            current = normalize_log_probs(previous)
        self.log_grip_distributions[object_id] = current
        return current

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
        object_advantages = np.asarray(
            [
                float(breakdown["weighted_total"])
                for breakdown in object_feature_breakdowns
            ],
            dtype=np.float64,
        )
        object_update = robust_standardize(object_advantages)
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
            log_grip = self._align_grip_distribution(obj.object_id, candidates)
            grip_advantages = np.zeros(len(candidates), dtype=np.float64)
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
                grip_advantages = np.asarray(
                    [
                        float(breakdown["weighted_total"])
                        for breakdown in grip_feature_breakdowns
                    ],
                    dtype=np.float64,
                )
                grip_update = robust_standardize(grip_advantages)
                log_grip = (
                    self.config.w_sc * log_grip + self.config.beta * grip_update
                )
                log_grip = clip_log_probs(
                    normalize_log_probs(log_grip), self.config.max_prob_any_goal
                )
                self.log_grip_distributions[obj.object_id] = log_grip
            grip_probs = np.exp(normalize_log_probs(log_grip))
            grip_index = int(np.argmax(grip_probs))
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
                confidence=joint_confidence,
            )
            if best_target is None or candidate_target.confidence > best_target.confidence:
                best_target = candidate_target
            telemetry_objects.append(
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "score": float(obj.score),
                    "object_probability": float(object_probs[obj_index]),
                    "object_advantage": float(object_advantages[obj_index]),
                    "object_advantage_standardized": float(object_update[obj_index]),
                    "object_feature_breakdown": object_feature_breakdowns[obj_index],
                    "grip_probabilities": [float(v) for v in grip_probs],
                    "grip_advantages": [float(v) for v in grip_advantages],
                    "selected_grip_feature_breakdown": grip_feature_breakdowns[grip_index],
                    "selected_grip_index": grip_index,
                    "selected_contact_goal_m": candidates[grip_index].center.tolist(),
                    "grip_candidates": [
                        {
                            **candidate.to_dict(),
                            "probability": float(grip_probs[index]),
                            "advantage": float(grip_advantages[index]),
                            "feature_breakdown": grip_feature_breakdowns[index],
                        }
                        for index, candidate in enumerate(candidates)
                    ],
                }
            )
        if should_update_grips:
            self.last_grip_update_monotonic = now
        self.feature_cost.finish_tick(user_translation_mps)
        self.last_target = best_target
        target_dict = None
        if best_target is not None:
            target_dict = {
                "object_id": best_target.object_id,
                "grip_index": int(best_target.grip_index),
                "confidence": float(best_target.confidence),
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
        distance_m = grip_breakdown.get("distance_m")
        distance_text = (
            f"ee_d={float(distance_m):.3f}m "
            if isinstance(distance_m, (int, float))
            else ""
        )
        return (
            "[TASC-belief] "
            f"target={target['object_id']} label={obj.get('label', '?')} "
            f"p={target['confidence']:.3f} grip={target['grip_index']} "
            f"contact={np.round(contact, 4).tolist()} "
            f"{distance_text}"
            f"feat[d={float(contributions.get('distance', 0.0)):+.3f},"
            f"r={float(contributions.get('ray', 0.0)):+.3f},"
            f"a={float(contributions.get('agreement', 0.0)):+.3f}] "
            f"objects=[{prob_text}]"
        )
