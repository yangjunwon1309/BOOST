"""Hierarchical TASC belief over segmented objects and grip contact points."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
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
    ContactFrame,
    GripCandidate,
    GripCandidateConfig,
    PersistentCylinderSurfaceTracker,
    PseudoContactConfig,
    SurfacePrimitive,
    contact_frame,
    extract_surface_primitives,
    generate_grip_candidates,
)
from .robot_geometry import (
    HAND_AXIS_LOCAL,
    hand_frame_profile,
    pelvis_to_world_rotation,
)

EPS = 1.0e-9
LOG = logging.getLogger(__name__)
POLICY_BY_PRIMITIVE = {
    "cylinder": "grasp_frame",
    "plane": "minimal_rotation",
    "circle": "minimal_rotation",
    "pca": "minimal_rotation",
    "point": "minimal_rotation",
    "gravity": "minimal_rotation",
}
GRAVITY_TRACK_UID = "__gravity__"


@dataclass
class AxisEstimator:
    """Exponentially weighted structure tensor for an undirected axis."""

    alpha: float = 0.30
    tensor: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    weight: float = 0.0
    observations: int = 0

    def update(self, normal: np.ndarray, weight: float = 1.0) -> None:
        axis = np.asarray(normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < 1.0e-9:
            return
        axis = axis / norm
        value = float(np.clip(weight, 0.0, 1.0))
        alpha = float(np.clip(self.alpha, 0.0, 1.0))
        self.tensor = (
            (1.0 - alpha) * self.tensor
            + alpha * value * np.outer(axis, axis)
        )
        self.weight = (1.0 - alpha) * self.weight + alpha * value
        self.observations += 1

    def decay(self, factor: float) -> None:
        """Shrink confidence while preserving the principal eigenvector."""
        value = float(np.clip(factor, 0.0, 1.0))
        self.tensor *= value
        self.weight *= value

    def axis_and_concentration(self) -> tuple[np.ndarray, float] | None:
        trace = float(np.trace(self.tensor))
        if not np.isfinite(trace) or trace < 1.0e-9:
            return None
        values, vectors = np.linalg.eigh(self.tensor)
        index = int(np.argmax(values))
        axis = vectors[:, index]
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < 1.0e-9:
            return None
        concentration = float(values[index] / max(1.0e-12, trace))
        return axis / norm, concentration


@dataclass(frozen=True)
class OrientationBankConfig:
    axis_ema_alpha: float = 0.30
    coast_ttl_frames: int = 24
    max_coast_seconds: float = 4.0
    coast_decay: float = 0.90
    assoc_max_axis_deg: float = 30.0
    assoc_base_distance_m: float = 0.10
    assoc_drift_mps: float = 0.15
    gravity_candidate_enabled: bool = True
    gravity_label_prior: float = -0.3
    contact_requires_active: bool = True
    goal_fallback_ttl_frames: int = 3

    @classmethod
    def from_dict(cls, data: dict | None) -> "OrientationBankConfig":
        values = dict(data or {})
        return cls(
            axis_ema_alpha=float(values.get("axis_ema_alpha", 0.30)),
            coast_ttl_frames=int(values.get("coast_ttl_frames", 24)),
            max_coast_seconds=float(values.get("max_coast_seconds", 4.0)),
            coast_decay=float(values.get("coast_decay", 0.90)),
            assoc_max_axis_deg=float(values.get("assoc_max_axis_deg", 30.0)),
            assoc_base_distance_m=float(
                values.get("assoc_base_distance_m", 0.10)
            ),
            assoc_drift_mps=float(values.get("assoc_drift_mps", 0.15)),
            gravity_candidate_enabled=bool(
                values.get("gravity_candidate_enabled", True)
            ),
            gravity_label_prior=float(values.get("gravity_label_prior", -0.3)),
            contact_requires_active=bool(
                values.get("contact_requires_active", True)
            ),
            goal_fallback_ttl_frames=int(
                values.get("goal_fallback_ttl_frames", 3)
            ),
        )


@dataclass(frozen=True)
class BeliefInputObject:
    object_id: str
    label: str
    score: float
    center_goal_m: np.ndarray
    points_goal_m: np.ndarray | None = None
    grip_candidates: list[GripCandidate] | None = None
    surface_primitives: list[SurfacePrimitive] | None = None


@dataclass(frozen=True)
class EETarget:
    position: np.ndarray
    orientation_xyzw: np.ndarray
    object_id: str
    grip_index: int
    object_confidence: float
    joint_confidence: float


@dataclass(frozen=True)
class OrientationCandidate:
    object_id: str
    pseudo_index: int
    relation: str
    position: np.ndarray
    normal: np.ndarray
    fit_quality: float
    track_id: str = ""
    label: str = ""
    role: str = "UNKNOWN"
    label_prior: float = 0.0
    source: str = ""
    primitive_kind: str = ""
    surface_axis: np.ndarray | None = None
    primitive: SurfacePrimitive | None = None


@dataclass
class OrientationTrack:
    """One persistent surface-axis hypothesis owned by the bank."""

    track_uid: str
    label: str
    kind: str
    axis_estimator: AxisEstimator
    anchor_world_m: np.ndarray
    anchor_stamp_us: int
    radius_m: float = 0.0
    half_height_m: float = 0.0
    extent_m: np.ndarray | None = None
    fit_quality_ema: float = 0.0
    detection_track_ids: set[str] = field(default_factory=set)
    last_detection_track_id: str = ""
    last_seen_us: int = 0
    missing_frames: int = 0
    total_observations: int = 0
    state: str = "active"
    persistent: bool = False
    frame_mode: str = "world"

    def effective_quality(self, coast_decay: float) -> float:
        resolved = self.axis_estimator.axis_and_concentration()
        if resolved is None:
            return 0.0
        _axis, concentration = resolved
        concentration_quality = max(0.0, (3.0 * concentration - 1.0) * 0.5)
        coast_quality = float(np.clip(coast_decay, 0.0, 1.0)) ** max(
            0, self.missing_frames
        )
        return float(
            np.clip(
                self.fit_quality_ema * concentration_quality * coast_quality,
                0.0,
                1.0,
            )
        )

    def to_primitive(
        self,
        rotation_world_pelvis: np.ndarray | None,
        coast_decay: float = 1.0,
    ) -> SurfacePrimitive | None:
        resolved = self.axis_estimator.axis_and_concentration()
        if resolved is None:
            return None
        axis_stored, _concentration = resolved
        if self.frame_mode == "world":
            if rotation_world_pelvis is None:
                return None
            axis_goal = rotation_world_pelvis.T @ axis_stored
            origin_goal = rotation_world_pelvis.T @ self.anchor_world_m
        else:
            axis_goal = axis_stored
            origin_goal = self.anchor_world_m
        return SurfacePrimitive(
            kind=self.kind,
            origin=np.asarray(origin_goal, dtype=np.float64).reshape(3),
            axis=np.asarray(axis_goal, dtype=np.float64).reshape(3),
            fit_quality=self.effective_quality(coast_decay),
            source=f"bank_{self.state}",
            radius=float(self.radius_m),
            half_height=float(self.half_height_m),
            extent=(
                None
                if self.extent_m is None
                else np.asarray(self.extent_m, dtype=np.float64).reshape(3)
            ),
            observation_timestamp_us=int(self.last_seen_us),
            observation_track_id=str(self.last_detection_track_id),
            observation_state=str(self.state),
        )


@dataclass(frozen=True)
class OrientationTarget:
    quaternion_xyzw: np.ndarray | None
    axis_world: np.ndarray
    relation: str
    object_id: str
    confidence: float
    source_position: np.ndarray
    policy: str = "minimal_rotation"
    frame_world: np.ndarray | None = None


@dataclass
class HandBeliefState:
    """Independent posterior state for one hand in parallel inference."""

    side: str
    log_object: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    log_contact: dict[str, np.ndarray] = field(default_factory=dict)
    log_orientation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    ema_adv_object: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    ema_adv_contact: dict[str, np.ndarray] = field(default_factory=dict)
    ema_adv_orientation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    armed: bool = False
    beta_current: float = 0.0
    lambda0_current: float = 0.0
    alpha: float = 0.0
    switch_dwell: int = 0
    quadratic_variation: float = 0.0


@dataclass(frozen=True)
class SharedGeometry:
    """Immutable hand-independent geometry cache for parallel belief updates."""

    objects: list[BeliefInputObject]
    grip_candidates: dict[str, list[GripCandidate]]
    surface_primitives: dict[str, list[SurfacePrimitive]]
    ray_origin: np.ndarray | None
    ray_direction: np.ndarray | None
    stamp_us: int
    up_goal: np.ndarray | None = None
    surface_status: dict[str, dict] = field(default_factory=dict)
    base_quat_wxyz: np.ndarray | None = None
    orientation_candidates: list[OrientationCandidate] = field(default_factory=list)
    orientation_bank_status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HandInput:
    eef_position: np.ndarray
    user_translation: np.ndarray | None
    hand_rotation: np.ndarray | None
    hand_axis: np.ndarray | None
    approach_axis: np.ndarray | None
    angular_velocity: np.ndarray | None
    button_pressed: bool
    shoulder_position: np.ndarray | None = None


@dataclass(frozen=True)
class TascBeliefConfig:
    enabled: bool = True
    beta: float = 1.0
    w_sc: float = 0.95
    max_prob_any_goal: float = 0.99
    leak_kappa0: float = 0.3
    leak_gamma: float = 5.0
    jump_enabled: bool = True
    jump_lambda0: float = 0.05
    jump_kappa: float = 2.0
    jump_switch_sigma_d: float = 0.15
    # Uniform-drift probability at one configured belief tick. The prediction
    # converts this to elapsed wall time so callback jitter cannot alter drift.
    persist_leak: float = 0.02
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
    orientation_enabled: bool = True
    orientation_beta: float = 1.0
    orientation_w_sc: float = 0.95
    orientation_max_prob: float = 0.99
    orientation_ema_alpha: float = 0.25
    orientation_switch_gamma: float = 4.0
    orientation_relations: tuple[str, ...] = ("parallel", "perpendicular")
    hand_model: str = "dex1"
    hand_axis_local: tuple[float, float, float] = tuple(HAND_AXIS_LOCAL["dex1"].tolist())
    orientation_match_max_distance_m: float = 0.08
    orientation_match_max_normal_deg: float = 25.0
    orientation_track_missing_ttl_frames: int = 5
    orientation_switch_dwell_frames: int = 15
    orientation_advantage_scale: float = 0.5
    orientation_gravity_snap_deg: float = 15.0
    orientation_ghost_decay: float = 0.35
    orientation_bank: OrientationBankConfig = field(
        default_factory=OrientationBankConfig
    )
    pseudo: PseudoContactConfig = field(default_factory=PseudoContactConfig)
    role_hard_filter: bool = False
    label_role_keywords: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "held": ("stick", "leg", "rod", "bar", "pole"),
        "target": ("hole", "circle", "surface", "top", "plate", "board"),
    })
    label_relation_prior: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "hole": {"parallel": 1.0, "perpendicular": -0.5},
        "circle": {"parallel": 1.0, "perpendicular": -0.5},
        "surface": {"parallel": -0.5, "perpendicular": 1.0},
        "top": {"parallel": -0.5, "perpendicular": 1.0},
        "board": {"parallel": -0.5, "perpendicular": 1.0},
        "gravity": {"parallel": -0.3, "perpendicular": -0.3},
    })
    label_prior_scale: float = 1.0
    per_hand_parallel: bool = True
    beta_armed: float = 1.0
    beta_idle: float = 0.2
    lambda0_armed: float = 0.05
    lambda0_idle: float = 0.30
    beta_ramp_tau_s: float = 0.2
    commit_boost_enabled: bool = False
    commit_boost_s: float = 0.3
    commit_boost_scale: float = 1.5
    dual_hand_exclusive: bool = False
    ori_switch_gamma_h: float = 3.0
    ori_switch_gamma_omega: float = 1.0
    ori_switch_omega_ref: float = 0.5
    pos_switch_sigma_d: float = 0.15
    pos_switch_gamma_u: float = 2.0
    geometry_update_hz: float = 8.0
    belief_update_hz: float = 30.0
    max_objects: int = 10
    max_contacts_per_object: int = 6
    max_orientation_candidates: int = 96
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    grip: GripCandidateConfig = field(default_factory=GripCandidateConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "TascBeliefConfig":
        data = dict(data or {})
        leak = dict(data.get("leak", {}))
        grip = dict(data.get("grip", {}))
        jump = dict(data.get("jump", {}))
        orientation = dict(data.get("orientation", {}))
        orientation_bank = dict(
            orientation.get("bank", data.get("orientation_bank", {}))
        )
        legacy_leak = "kappa0" in leak or "gamma" in leak or "leak_kappa0" in data or "leak_gamma" in data
        if legacy_leak and "persist_leak" not in jump and "persist_leak" not in data:
            LOG.warning("tasc_belief leak_kappa0/leak_gamma are deprecated; mapping kappa0 to persist_leak")
        role_keywords = dict(orientation.get("label_role_keywords", data.get("label_role_keywords", {})))
        relation_prior = dict(orientation.get("label_relation_prior", data.get("label_relation_prior", {})))
        bank_config = OrientationBankConfig.from_dict(orientation_bank)
        default_relation_prior = {
            "hole": {"parallel": 1.0, "perpendicular": -0.5},
            "circle": {"parallel": 1.0, "perpendicular": -0.5},
            "surface": {"parallel": -0.5, "perpendicular": 1.0},
            "top": {"parallel": -0.5, "perpendicular": 1.0},
            "board": {"parallel": -0.5, "perpendicular": 1.0},
            "gravity": {
                "parallel": bank_config.gravity_label_prior,
                "perpendicular": bank_config.gravity_label_prior,
            },
        }
        default_relation_prior.update(relation_prior)
        return cls(
            enabled=bool(data.get("enabled", True)),
            beta=float(data.get("beta", 1.0)),
            w_sc=float(data.get("w_sc", data.get("weight_sc", 0.95))),
            max_prob_any_goal=float(data.get("max_prob_any_goal", 0.99)),
            leak_kappa0=float(leak.get("kappa0", data.get("leak_kappa0", 0.3))),
            leak_gamma=float(leak.get("gamma", data.get("leak_gamma", 5.0))),
            jump_enabled=bool(jump.get("enabled", data.get("jump_enabled", True))),
            jump_lambda0=float(jump.get("lambda0", data.get("jump_lambda0", 0.05))),
            jump_kappa=float(jump.get("kappa", data.get("jump_kappa", 2.0))),
            jump_switch_sigma_d=float(jump.get("switch_sigma_d", data.get("jump_switch_sigma_d", 0.15))),
            persist_leak=float(jump.get("persist_leak", data.get("persist_leak", leak.get("kappa0", data.get("leak_kappa0", 0.02))))),
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
            orientation_enabled=bool(orientation.get("enabled", data.get("orientation_enabled", True))),
            orientation_beta=float(orientation.get("beta", data.get("orientation_beta", 1.0))),
            orientation_w_sc=float(orientation.get("w_sc", data.get("orientation_w_sc", 0.95))),
            orientation_max_prob=float(orientation.get("max_prob", data.get("orientation_max_prob", 0.99))),
            orientation_ema_alpha=float(orientation.get("ema_alpha", data.get("orientation_ema_alpha", 0.25))),
            orientation_switch_gamma=float(orientation.get("switch_gamma", data.get("orientation_switch_gamma", 4.0))),
            orientation_relations=tuple(str(v).lower() for v in orientation.get("relations", data.get("orientation_relations", ("parallel", "perpendicular")))),
            hand_model=str(data.get("hand_model", orientation.get("hand_model", "dex1"))).lower(),
            hand_axis_local=tuple(float(v) for v in orientation.get("hand_axis_local", data.get("hand_axis_local", HAND_AXIS_LOCAL["dex1"]))),
            orientation_match_max_distance_m=float(orientation.get("match_max_distance_m", data.get("orientation_match_max_distance_m", 0.08))),
            orientation_match_max_normal_deg=float(orientation.get("match_max_normal_deg", data.get("orientation_match_max_normal_deg", 25.0))),
            orientation_track_missing_ttl_frames=int(orientation.get("track_missing_ttl_frames", data.get("orientation_track_missing_ttl_frames", 5))),
            orientation_switch_dwell_frames=int(orientation.get("switch_dwell_frames", data.get("orientation_switch_dwell_frames", 15))),
            orientation_advantage_scale=float(orientation.get("advantage_scale", data.get("orientation_advantage_scale", 0.5))),
            orientation_gravity_snap_deg=float(orientation.get("gravity_snap_deg", data.get("orientation_gravity_snap_deg", 15.0))),
            orientation_ghost_decay=float(orientation.get("ghost_decay", data.get("orientation_ghost_decay", 0.35))),
            orientation_bank=bank_config,
            pseudo=PseudoContactConfig.from_dict({**dict(data.get("pseudo") or {}), **dict(orientation.get("pseudo") or {}), "gravity_snap_deg": orientation.get("gravity_snap_deg", data.get("orientation_gravity_snap_deg", 15.0))}),
            role_hard_filter=bool(orientation.get("role_hard_filter", data.get("role_hard_filter", False))),
            label_role_keywords={str(k): tuple(str(v) for v in values) for k, values in (role_keywords or {
                "held": ("stick", "leg", "rod", "bar", "pole"),
                "target": ("hole", "circle", "surface", "top", "plate", "board"),
            }).items()},
            label_relation_prior={str(k): {str(r): float(x) for r, x in values.items()} for k, values in default_relation_prior.items()},
            label_prior_scale=float(orientation.get("label_prior_scale", data.get("label_prior_scale", 1.0))),
            per_hand_parallel=bool(data.get("per_hand_parallel", True)),
            beta_armed=float(data.get("beta_armed", 1.0)), beta_idle=float(data.get("beta_idle", 0.2)),
            lambda0_armed=float(data.get("lambda0_armed", 0.05)), lambda0_idle=float(data.get("lambda0_idle", 0.30)),
            beta_ramp_tau_s=float(data.get("beta_ramp_tau_s", 0.2)),
            commit_boost_enabled=bool(data.get("commit_boost_enabled", False)), commit_boost_s=float(data.get("commit_boost_s", 0.3)), commit_boost_scale=float(data.get("commit_boost_scale", 1.5)),
            dual_hand_exclusive=bool(data.get("dual_hand_exclusive", False)),
            ori_switch_gamma_h=float(orientation.get("switch_gamma_h", data.get("ori_switch_gamma_h", 3.0))),
            ori_switch_gamma_omega=float(orientation.get("switch_gamma_omega", data.get("ori_switch_gamma_omega", 1.0))),
            ori_switch_omega_ref=float(orientation.get("switch_omega_ref", data.get("ori_switch_omega_ref", 0.5))),
            pos_switch_sigma_d=float(data.get("pos_switch_sigma_d", 0.15)), pos_switch_gamma_u=float(data.get("pos_switch_gamma_u", 2.0)),
            geometry_update_hz=float(data.get("geometry_update_hz", 8.0)), belief_update_hz=float(data.get("belief_update_hz", 30.0)),
            max_objects=int(data.get("max_objects", 10)), max_contacts_per_object=int(data.get("max_contacts_per_object", 6)), max_orientation_candidates=int(data.get("max_orientation_candidates", 96)),
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


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    return None if not np.isfinite(norm) or norm <= EPS else arr / norm


def classify_label(
    label: str,
    role_keywords: dict[str, tuple[str, ...]],
    relation_prior: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float]]:
    """Classify once in the geometry stage; longest keyword match wins."""
    text = str(label).strip().lower()
    role, best = "UNKNOWN", ""
    for configured_role, keywords in role_keywords.items():
        for keyword in keywords:
            needle = str(keyword).lower()
            if needle in text and len(needle) > len(best):
                best = needle
                role = "HELD" if str(configured_role).lower() == "held" else "TARGET"
    prior: dict[str, float] = {}
    best = ""
    for keyword, mapping in relation_prior.items():
        needle = str(keyword).lower()
        if needle in text and len(needle) > len(best):
            best = needle
            prior = {str(key).lower(): float(value) for key, value in mapping.items()}
    return role, prior


@dataclass(frozen=True)
class _OrientationObservation:
    detection_track_id: str
    label: str
    primitive: SurfacePrimitive
    axis_world: np.ndarray
    anchor_world_m: np.ndarray
    frame_mode: str


class OrientationCandidateBank:
    """Persistent, detection-independent world-frame orientation hypotheses."""

    def __init__(self, config: OrientationBankConfig | dict | None = None) -> None:
        self.config = (
            config
            if isinstance(config, OrientationBankConfig)
            else OrientationBankConfig.from_dict(config)
        )
        self.tracks: dict[str, OrientationTrack] = {}
        self.next_track_index = 0
        self.pelvis_to_world: np.ndarray | None = None
        self.frame_mode = "goal_fallback"
        self._seen_this_frame: set[str] = set()
        self.expired_this_frame = 0
        self.reassociations = 0
        self.new_tracks = 0
        self._fallback_warned = False
        self._last_now_us = 0

    def _associate(
        self, observation: _OrientationObservation, now_us: int
    ) -> str | None:
        """Associate by detector hint, then label, undirected axis, and position."""
        for uid, track in self.tracks.items():
            if (
                uid not in self._seen_this_frame
                and not track.persistent
                and track.frame_mode == observation.frame_mode
                and observation.detection_track_id in track.detection_track_ids
            ):
                return uid

        cos_limit = float(
            np.cos(np.deg2rad(self.config.assoc_max_axis_deg))
        )
        best_uid: str | None = None
        best_distance = float("inf")
        for uid, track in self.tracks.items():
            if (
                uid in self._seen_this_frame
                or track.persistent
                or track.label != observation.label
                or track.frame_mode != observation.frame_mode
            ):
                continue
            elapsed = max(0.0, (now_us - track.last_seen_us) / 1.0e6)
            if elapsed > self.config.max_coast_seconds:
                continue
            resolved = track.axis_estimator.axis_and_concentration()
            if resolved is None:
                continue
            axis, _concentration = resolved
            if abs(float(np.dot(axis, observation.axis_world))) < cos_limit:
                continue
            budget = (
                self.config.assoc_base_distance_m
                + self.config.assoc_drift_mps * elapsed
            )
            distance = float(
                np.linalg.norm(
                    observation.anchor_world_m - track.anchor_world_m
                )
            )
            if distance <= budget and distance < best_distance:
                best_uid = uid
                best_distance = distance
        return best_uid

    def observe(
        self,
        primitives: dict[str, SurfacePrimitive],
        labels: dict[str, str],
        *,
        base_quat_wxyz: np.ndarray | None,
        up_goal: np.ndarray | None,
        now_us: int,
    ) -> None:
        """Fuse the current frame's surface observations into the bank."""
        del up_goal  # Gravity candidates are recomputed by candidates().
        self._last_now_us = int(now_us)
        self._seen_this_frame = set()
        self.expired_this_frame = 0
        self.pelvis_to_world = pelvis_to_world_rotation(base_quat_wxyz)
        self.frame_mode = (
            "world" if self.pelvis_to_world is not None else "goal_fallback"
        )
        if self.frame_mode == "goal_fallback":
            if not self._fallback_warned:
                LOG.warning(
                    "orientation bank has no valid base quaternion; using "
                    "short-TTL goal-frame fallback"
                )
                self._fallback_warned = True
        else:
            self._fallback_warned = False

        for detection_track_id, primitive in primitives.items():
            try:
                axis_goal = np.asarray(
                    primitive.axis, dtype=np.float64
                ).reshape(3)
                anchor_goal = np.asarray(
                    primitive.origin, dtype=np.float64
                ).reshape(3)
            except (TypeError, ValueError):
                continue
            axis_norm = float(np.linalg.norm(axis_goal))
            if (
                not np.isfinite(axis_goal).all()
                or not np.isfinite(anchor_goal).all()
                or axis_norm < 1.0e-9
            ):
                continue
            axis_goal = axis_goal / axis_norm
            if self.pelvis_to_world is None:
                axis_stored = axis_goal
                anchor_stored = anchor_goal
            else:
                axis_stored = self.pelvis_to_world @ axis_goal
                anchor_stored = self.pelvis_to_world @ anchor_goal
            observation = _OrientationObservation(
                detection_track_id=str(detection_track_id),
                label=str(labels.get(detection_track_id, "object")),
                primitive=primitive,
                axis_world=axis_stored,
                anchor_world_m=anchor_stored,
                frame_mode=self.frame_mode,
            )
            uid = self._associate(observation, int(now_us))
            if uid is None:
                uid = f"orientation:{self.next_track_index:06d}"
                self.next_track_index += 1
                estimator = AxisEstimator(alpha=self.config.axis_ema_alpha)
                estimator.update(axis_stored, primitive.fit_quality)
                self.tracks[uid] = OrientationTrack(
                    track_uid=uid,
                    label=observation.label,
                    kind=str(primitive.kind),
                    axis_estimator=estimator,
                    anchor_world_m=anchor_stored.copy(),
                    anchor_stamp_us=int(now_us),
                    radius_m=float(primitive.radius),
                    half_height_m=float(primitive.half_height),
                    extent_m=(
                        None
                        if primitive.extent is None
                        else np.asarray(primitive.extent, dtype=np.float64)
                        .reshape(3)
                        .copy()
                    ),
                    fit_quality_ema=float(
                        np.clip(primitive.fit_quality, 0.0, 1.0)
                    ),
                    detection_track_ids={str(detection_track_id)},
                    last_detection_track_id=str(detection_track_id),
                    last_seen_us=int(now_us),
                    total_observations=1,
                    frame_mode=self.frame_mode,
                )
                self.new_tracks += 1
            else:
                track = self.tracks[uid]
                was_reassociation = (
                    track.state == "coasting"
                    or str(detection_track_id)
                    not in track.detection_track_ids
                )
                track.axis_estimator.update(
                    axis_stored, float(primitive.fit_quality)
                )
                alpha = float(
                    np.clip(self.config.axis_ema_alpha, 0.0, 1.0)
                )
                quality = float(np.clip(primitive.fit_quality, 0.0, 1.0))
                track.fit_quality_ema = (
                    (1.0 - alpha) * track.fit_quality_ema + alpha * quality
                )
                track.anchor_world_m = anchor_stored.copy()
                track.anchor_stamp_us = int(now_us)
                track.last_seen_us = int(now_us)
                track.label = observation.label
                track.kind = str(primitive.kind)
                track.radius_m = float(primitive.radius)
                track.half_height_m = float(primitive.half_height)
                track.extent_m = (
                    None
                    if primitive.extent is None
                    else np.asarray(primitive.extent, dtype=np.float64)
                    .reshape(3)
                    .copy()
                )
                track.detection_track_ids.add(str(detection_track_id))
                track.last_detection_track_id = str(detection_track_id)
                track.missing_frames = 0
                track.total_observations += 1
                track.state = "active"
                if was_reassociation:
                    self.reassociations += 1
            self._seen_this_frame.add(uid)

    def advance(self, now_us: int) -> None:
        """Age unobserved tracks without rotating their fused axes."""
        self._last_now_us = int(now_us)
        for uid in list(self.tracks):
            track = self.tracks[uid]
            if track.persistent or uid in self._seen_this_frame:
                continue
            track.missing_frames += 1
            track.state = "coasting"
            track.axis_estimator.decay(self.config.coast_decay)
            track.fit_quality_ema *= float(self.config.coast_decay)
            elapsed = max(0.0, (int(now_us) - track.last_seen_us) / 1.0e6)
            ttl = (
                self.config.goal_fallback_ttl_frames
                if track.frame_mode == "goal_fallback"
                else self.config.coast_ttl_frames
            )
            if (
                track.missing_frames > max(0, int(ttl))
                or elapsed > self.config.max_coast_seconds
            ):
                track.state = "expired"
                del self.tracks[uid]
                self.expired_this_frame += 1

    def _gravity_candidates(
        self,
        *,
        relations: tuple[str, ...],
        up_goal: np.ndarray | None,
        label_relation_prior: dict[str, dict[str, float]],
    ) -> list[OrientationCandidate]:
        if up_goal is None or not self.config.gravity_candidate_enabled:
            return []
        axis_goal = _unit(up_goal)
        if axis_goal is None:
            return []
        prior_map = label_relation_prior.get("gravity", {})
        output: list[OrientationCandidate] = []
        for relation in relations:
            if relation not in {"parallel", "perpendicular"}:
                continue
            output.append(
                OrientationCandidate(
                    object_id=GRAVITY_TRACK_UID,
                    pseudo_index=0,
                    relation=relation,
                    position=np.zeros(3, dtype=np.float64),
                    normal=axis_goal.copy(),
                    fit_quality=1.0,
                    track_id=f"{GRAVITY_TRACK_UID}:{relation}",
                    label="gravity",
                    role="UNKNOWN",
                    label_prior=float(
                        prior_map.get(
                            relation, self.config.gravity_label_prior
                        )
                    ),
                    source="gravity_live",
                    primitive_kind="gravity",
                    surface_axis=axis_goal.copy(),
                )
            )
        return output

    def candidates(
        self,
        *,
        relations: tuple[str, ...],
        base_quat_wxyz: np.ndarray | None,
        up_goal: np.ndarray | None,
        label_relation_prior: dict[str, dict[str, float]],
        role_keywords: dict[str, tuple[str, ...]],
        max_candidates: int,
    ) -> list[OrientationCandidate]:
        """Emit current goal-frame candidates plus the untruncated gravity pair."""
        rotation = pelvis_to_world_rotation(base_quat_wxyz)
        frame_mode = "world" if rotation is not None else "goal_fallback"
        ranked: list[tuple[float, OrientationCandidate]] = []
        for track in self.tracks.values():
            if track.frame_mode != frame_mode:
                continue
            primitive = track.to_primitive(rotation, self.config.coast_decay)
            if primitive is None:
                continue
            axis_goal = _unit(primitive.axis)
            if axis_goal is None:
                continue
            role, relation_prior = classify_label(
                track.label, role_keywords, label_relation_prior
            )
            if role == "HELD":
                role_penalty = -1.0
            else:
                role_penalty = 0.0
            quality = track.effective_quality(self.config.coast_decay)
            object_id = (
                track.last_detection_track_id
                if track.last_detection_track_id
                else track.track_uid
            )
            for relation in relations:
                if relation not in {"parallel", "perpendicular"}:
                    continue
                prior = float(relation_prior.get(relation, 0.0)) + role_penalty
                ranked.append(
                    (
                        quality,
                        OrientationCandidate(
                            object_id=object_id,
                            pseudo_index=0,
                            relation=relation,
                            position=np.asarray(
                                primitive.origin, dtype=np.float64
                            ).reshape(3),
                            normal=axis_goal.copy(),
                            fit_quality=quality,
                            track_id=f"{track.track_uid}:{relation}",
                            label=track.label,
                            role=role,
                            label_prior=prior,
                            source=primitive.source,
                            primitive_kind=track.kind,
                            surface_axis=axis_goal.copy(),
                            primitive=primitive,
                        ),
                    )
                )
        ranked.sort(key=lambda item: item[0], reverse=True)
        limit = max(0, int(max_candidates))
        selected = [candidate for _quality, candidate in ranked[:limit]]
        selected.extend(
            self._gravity_candidates(
                relations=relations,
                up_goal=up_goal,
                label_relation_prior=label_relation_prior,
            )
        )
        return selected

    def status(
        self,
        *,
        base_quat_wxyz: np.ndarray | None,
        up_goal: np.ndarray | None,
        now_us: int | None = None,
    ) -> dict[str, Any]:
        rotation = pelvis_to_world_rotation(base_quat_wxyz)
        frame_mode = "world" if rotation is not None else "goal_fallback"
        timestamp_us = self._last_now_us if now_us is None else int(now_us)
        rows: list[dict[str, Any]] = []
        for track in self.tracks.values():
            resolved = track.axis_estimator.axis_and_concentration()
            primitive = track.to_primitive(rotation, self.config.coast_decay)
            rows.append(
                {
                    "track_uid": track.track_uid,
                    "label": track.label,
                    "kind": track.kind,
                    "state": track.state,
                    "concentration": (
                        0.0 if resolved is None else float(resolved[1])
                    ),
                    "fit_quality_eff": track.effective_quality(
                        self.config.coast_decay
                    ),
                    "missing_frames": int(track.missing_frames),
                    "observations": int(track.total_observations),
                    "axis_goal": (
                        None
                        if primitive is None
                        else np.asarray(primitive.axis).reshape(3).tolist()
                    ),
                    "coast_seconds": max(
                        0.0, (timestamp_us - track.last_seen_us) / 1.0e6
                    ),
                }
            )
        return {
            "tracks": rows,
            "active_count": sum(row["state"] == "active" for row in rows),
            "coasting_count": sum(
                row["state"] == "coasting" for row in rows
            ),
            "expired_this_frame": int(self.expired_this_frame),
            "gravity_available": bool(
                up_goal is not None and self.config.gravity_candidate_enabled
            ),
            "frame_mode": frame_mode,
            "reassociations": int(self.reassociations),
            "new_tracks": int(self.new_tracks),
        }


def _quat_xyzw_to_mat(quaternion: np.ndarray) -> np.ndarray | None:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= EPS:
        return None
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _mat_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = 2.0 * np.sqrt(max(EPS, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
            q = np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
        elif idx == 1:
            s = 2.0 * np.sqrt(max(EPS, 1.0 + r[1, 1] - r[0, 0] - r[2, 2]))
            q = np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
        else:
            s = 2.0 * np.sqrt(max(EPS, 1.0 + r[2, 2] - r[0, 0] - r[1, 1]))
            q = np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s])
    return q / max(EPS, float(np.linalg.norm(q)))


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal Rodrigues rotation taking source to target, including antipodal case."""
    a = _unit(source)
    b = _unit(target)
    if a is None or b is None:
        return np.eye(3)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cosine > 1.0 - 1.0e-9:
        return np.eye(3)
    if cosine < -1.0 + 1.0e-9:
        basis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, basis))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    axis = cross / sine
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def grasp_frame_from_cylinder(
    axis: np.ndarray,
    normal: np.ndarray,
    reference_rotation: np.ndarray,
    profile: dict[str, object] | None = None,
) -> np.ndarray | None:
    """Construct an SO(3) Dex1 grasp frame from a cylinder contact frame."""
    cylinder_axis = _unit(axis)
    contact_normal = _unit(normal)
    if cylinder_axis is None or contact_normal is None:
        return None
    y_axis = -contact_normal
    x_axis = _unit(np.cross(cylinder_axis, contact_normal))
    if x_axis is None:
        return None
    reference = np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
    spread_local = np.asarray((profile or {}).get("spread_axis_local", [1.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    current_spread = _unit(reference @ spread_local)
    if current_spread is not None and float(np.dot(x_axis, current_spread)) < 0.0:
        x_axis = -x_axis
    z_axis = _unit(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    frame = np.column_stack((x_axis, y_axis, z_axis))
    u, _singular, vt = np.linalg.svd(frame)
    frame = u @ vt
    if float(np.linalg.det(frame)) < 0.0:
        u[:, -1] *= -1.0
        frame = u @ vt
    return frame


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
        self.surface_tracker = PersistentCylinderSurfaceTracker(self.config.grip)
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
        self.orientation_candidates: list[OrientationCandidate] = []
        self.log_orientation_distribution = np.empty(0, dtype=np.float64)
        self.orientation_advantage_ema = np.empty(0, dtype=np.float64)
        self.selected_orientation_index: int | None = None
        self.selected_orientation_track_id: str = ""
        self.pending_orientation_switch: tuple[str, int] | None = None
        self.last_orientation_target: OrientationTarget | None = None
        self.last_grip_update_monotonic = 0.0
        self.last_update_monotonic: float | None = None
        self.last_log_monotonic = 0.0
        self.last_target: EETarget | None = None
        self.last_telemetry: dict[str, Any] = {
            "enabled": self.config.enabled,
            "objects": [],
            "target": None,
        }

    def _classify_label(self, label: str) -> tuple[str, dict[str, float]]:
        return classify_label(label, self.config.label_role_keywords, self.config.label_relation_prior)

    def _switch_weights_position(
        self,
        candidates: list[BeliefInputObject],
        eef_position: np.ndarray,
        user_translation: np.ndarray | None,
    ) -> np.ndarray:
        if not candidates:
            return np.empty(0, dtype=np.float64)
        velocity = _unit(user_translation)
        if velocity is None or float(np.linalg.norm(user_translation)) < self.feature_cost.config.min_user_speed_mps:
            return np.full(len(candidates), -np.log(max(1, len(candidates))))
        centers = np.stack([np.asarray(obj.center_goal_m, dtype=np.float64).reshape(3) for obj in candidates])
        logits = -np.linalg.norm(centers - eef_position[None, :], axis=1) / max(EPS, self.config.pos_switch_sigma_d)
        to_goal = centers - eef_position[None, :]
        norms = np.linalg.norm(to_goal, axis=1)
        valid = norms > EPS
        alignment = np.zeros(centers.shape[0], dtype=np.float64)
        alignment[valid] = (to_goal[valid] / norms[valid, None]) @ velocity
        logits += self.config.pos_switch_gamma_u * alignment
        return normalize_log_probs(logits)

    def _jump_predict(
        self,
        log_pi: np.ndarray,
        advantage_std: np.ndarray,
        switch_log_weights: np.ndarray | None,
        dt: float,
        *,
        lambda0: float | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Adaptive IMM/change-point predict, distinct from the generative prior.

        Innovation variance estimates the unknown switching rate. Prediction is
        intentionally performed before the likelihood correction below.
        """
        pi = np.exp(normalize_log_probs(log_pi))
        if not self.config.jump_enabled or pi.size < 2:
            lam_dt = 0.0
            variance = 0.0
        else:
            mean = float(np.sum(pi * advantage_std))
            variance = float(np.sum(pi * (advantage_std - mean) ** 2))
            rate = float(self.config.jump_lambda0 if lambda0 is None else lambda0)
            rate += float(self.config.jump_kappa) * variance
            lam_dt = float(np.clip(rate * max(0.0, dt), 0.0, 1.0))
        # Preserve the configured probability at one nominal belief tick while
        # making the equivalent probability depend only on elapsed time.
        # At 30 Hz, persist_leak=0.02 remains exactly a 2% leak per tick.
        leak_per_tick = float(np.clip(self.config.persist_leak, 0.0, 0.999999))
        nominal_ticks = max(0.0, dt) * max(0.1, self.config.belief_update_hz)
        q0 = float(1.0 - (1.0 - leak_per_tick) ** nominal_ticks)
        uniform = np.full_like(pi, 1.0 / max(1, pi.size))
        persist = (1.0 - q0) * pi + q0 * uniform
        jump = uniform if switch_log_weights is None else np.exp(normalize_log_probs(switch_log_weights))
        pi_minus = (1.0 - lam_dt) * persist + lam_dt * jump
        return normalize_log_probs(np.log(np.maximum(pi_minus, EPS))), lam_dt, variance

    def _orientation_switch_weights(
        self,
        candidates: list[OrientationCandidate],
        hand_axis: np.ndarray,
        hand_omega: np.ndarray | None,
    ) -> np.ndarray:
        axis = _unit(hand_axis)
        if axis is None:
            return np.full(len(candidates), -np.log(max(1, len(candidates))))
        omega = np.zeros(3, dtype=np.float64) if hand_omega is None else np.asarray(hand_omega, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(omega)) < self.feature_cost.config.min_hand_omega_rps:
            return np.full(len(candidates), -np.log(max(1, len(candidates))))
        logits: list[float] = []
        for candidate in candidates:
            normal = _unit(candidate.normal)
            cosine = float(np.clip(np.dot(axis, normal), -1.0, 1.0))
            error = 1.0 - cosine * cosine if candidate.relation == "parallel" else cosine * cosine
            dc = float(np.dot(np.cross(omega, axis), normal))
            de = -2.0 * cosine * dc if candidate.relation == "parallel" else 2.0 * cosine * dc
            logits.append(
                self.config.ori_switch_gamma_h * (1.0 - error)
                + self.config.ori_switch_gamma_omega * (-de) / max(EPS, self.config.ori_switch_omega_ref)
            )
        # Rank-one jump kernel: it represents the direction the hand is taking,
        # not similarity to its previous orientation candidate.
        return normalize_log_probs(np.asarray(logits, dtype=np.float64))

    def _select_orientation_index(
        self, best: int, probabilities: np.ndarray, candidates: list[OrientationCandidate]
    ) -> int:
        """Dwell hysteresis keyed by stable track ID, never list index."""
        best_id = candidates[int(best)].track_id
        by_id = {candidate.track_id: index for index, candidate in enumerate(candidates)}
        current = by_id.get(self.selected_orientation_track_id, -1)
        if current < 0:
            self.selected_orientation_track_id = best_id
            self.pending_orientation_switch = None
            self.selected_orientation_index = int(best)
            return int(best)
        if best_id == self.selected_orientation_track_id:
            self.pending_orientation_switch = None
            self.selected_orientation_index = int(current)
            return int(current)
        pending = self.pending_orientation_switch
        count = pending[1] + 1 if pending is not None and pending[0] == best_id else 1
        if count >= max(1, self.config.orientation_switch_dwell_frames):
            self.selected_orientation_track_id = best_id
            self.pending_orientation_switch = None
            self.selected_orientation_index = int(best)
            return int(best)
        self.pending_orientation_switch = (best_id, int(count))
        self.selected_orientation_index = int(current)
        return int(current)

    def _match_orientation_candidates(self, candidates: list[OrientationCandidate]) -> dict[int, int]:
        """Map each new candidate to a previous candidate without K-reset."""
        previous = self.orientation_candidates
        if not previous:
            return {}
        by_track = {candidate.track_id: index for index, candidate in enumerate(previous) if candidate.track_id}
        matched: dict[int, int] = {}
        used: set[int] = set()
        for index, candidate in enumerate(candidates):
            prior = by_track.get(candidate.track_id)
            if prior is not None and prior not in used:
                matched[index] = prior
                used.add(prior)
        for index, candidate in enumerate(candidates):
            if index in matched:
                continue
            best, best_distance = -1, float("inf")
            for prior_index, other in enumerate(previous):
                if prior_index in used or other.relation != candidate.relation:
                    continue
                distance = float(np.linalg.norm(candidate.position - other.position))
                if distance <= self.config.orientation_match_max_distance_m and distance < best_distance:
                    best, best_distance = prior_index, distance
            if best >= 0:
                matched[index] = best
                used.add(best)
        return matched

    def _quaternion_from_relation(
        self, candidate: OrientationCandidate, hand_rotation_matrix: np.ndarray
    ) -> np.ndarray | None:
        normal = _unit(candidate.normal)
        rotation = np.asarray(hand_rotation_matrix, dtype=np.float64).reshape(3, 3)
        current_axis = _unit(rotation @ np.asarray(self.config.hand_axis_local, dtype=np.float64))
        cosine = float(np.clip(np.dot(current_axis, normal), -1.0, 1.0))
        if candidate.relation == "parallel":
            target_axis = normal if cosine >= 0.0 else -normal
        else:
            target_axis = _unit(current_axis - cosine * normal)
            if target_axis is None:
                return None
        return _mat_to_quat_xyzw(_rotation_between(current_axis, target_axis) @ rotation)

    def _update_orientation_belief(
        self,
        objects: list[BeliefInputObject],
        *,
        hand_rotation_matrix: np.ndarray | None,
        hand_angular_velocity: np.ndarray | None,
        dt: float,
        up_goal: np.ndarray | None = None,
        orientation_candidates: list[OrientationCandidate] | None = None,
    ) -> tuple[OrientationTarget | None, dict, float, float]:
        if not self.config.orientation_enabled or hand_rotation_matrix is None:
            return None, {"enabled": False, "candidates": []}, 0.0, 0.0
        rotation = np.asarray(hand_rotation_matrix, dtype=np.float64).reshape(3, 3)
        hand_axis = _unit(rotation @ np.asarray(self.config.hand_axis_local, dtype=np.float64))
        if hand_axis is None:
            return None, {"enabled": False, "candidates": []}, 0.0, 0.0
        candidates: list[OrientationCandidate] = []
        raw_advantages: list[float] = []
        breakdowns: list[dict] = []
        priors: list[float] = []
        roles: list[str] = []

        def append_candidate(candidate: OrientationCandidate) -> None:
            """Keep every candidate-indexed posterior array in lockstep."""
            breakdown = self.feature_cost.orientation_breakdown(
                surface_normal=candidate.normal,
                hand_axis=hand_axis,
                hand_angular_velocity=hand_angular_velocity,
                relation=candidate.relation,
            )
            candidates.append(candidate)
            raw_advantages.append(
                float(breakdown["weighted_total"]) * candidate.fit_quality
                + candidate.label_prior
            )
            breakdowns.append(breakdown)
            priors.append(float(candidate.label_prior))
            roles.append(candidate.role)

        if orientation_candidates is not None:
            for candidate in orientation_candidates:
                append_candidate(candidate)
        for obj in objects if orientation_candidates is None else []:
            role, relation_prior = self._classify_label(obj.label)
            if role == "HELD" and self.config.role_hard_filter:
                continue
            primitives = obj.surface_primitives or extract_surface_primitives(
                np.asarray(obj.points_goal_m if obj.points_goal_m is not None else [obj.center_goal_m]),
                self.config.pseudo,
                cylinder_model=self.surface_tracker.models.get(obj.object_id),
                label=obj.label,
                up_goal=up_goal,
            )
            for pseudo_index, primitive in enumerate(primitives[: self.config.max_contacts_per_object]):
                # Standalone use has no per-hand shared geometry.  The current
                # hand axis selects a deterministic primitive side here.
                frame = contact_frame(primitive, np.asarray(self.config.fallback_eef_position_m))
                if frame is None:
                    continue
                for relation in self.config.orientation_relations:
                    if relation not in {"parallel", "perpendicular"}:
                        continue
                    prior = self.config.label_prior_scale * relation_prior.get(relation, 0.0)
                    if role == "HELD":
                        prior -= 1.0
                    candidate = OrientationCandidate(
                        object_id=obj.object_id, pseudo_index=pseudo_index, relation=relation,
                        position=np.asarray(frame.point, dtype=np.float64).reshape(3),
                        normal=np.asarray(frame.normal, dtype=np.float64).reshape(3),
                        fit_quality=float(frame.fit_quality), track_id=f"{obj.object_id}:{relation}", label=obj.label,
                        role=role, label_prior=float(prior), source=frame.source,
                        primitive_kind=frame.primitive_kind,
                    )
                    append_candidate(candidate)
        limit = max(1, int(self.config.max_orientation_candidates))
        protected = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.object_id == GRAVITY_TRACK_UID
        ]
        ranked = [
            index for index in range(len(candidates)) if index not in protected
        ]
        if len(ranked) > limit:
            ranked.sort(key=lambda index: raw_advantages[index], reverse=True)
            keep = sorted(ranked[:limit] + protected)
            candidates = [candidates[index] for index in keep]
            raw_advantages = [raw_advantages[index] for index in keep]
            breakdowns = [breakdowns[index] for index in keep]
            priors = [priors[index] for index in keep]
            roles = [roles[index] for index in keep]
        raw = np.asarray(raw_advantages, dtype=np.float64)
        if not candidates:
            self.orientation_candidates = []
            self.log_orientation_distribution = np.empty(0, dtype=np.float64)
            self.last_orientation_target = None
            return None, {
                "enabled": True,
                "hand_axis_world": hand_axis.tolist(),
                "candidates": [],
                "orientation_valid": False,
                "target": None,
            }, 0.0, 0.0
        match = self._match_orientation_candidates(candidates)
        entry_log = (
            float(np.min(self.log_orientation_distribution))
            if self.log_orientation_distribution.size
            else -float(np.log(max(1, len(candidates))))
        )
        aligned_log = np.full(len(candidates), entry_log, dtype=np.float64)
        ema = raw.copy()
        alpha = float(np.clip(self.config.orientation_ema_alpha, 0.0, 1.0))
        for new_index, prior_index in match.items():
            if prior_index < self.log_orientation_distribution.size:
                aligned_log[new_index] = self.log_orientation_distribution[prior_index]
            if prior_index < self.orientation_advantage_ema.size:
                ema[new_index] = alpha * raw[new_index] + (1.0 - alpha) * self.orientation_advantage_ema[prior_index]
        self.orientation_advantage_ema = ema
        standardized = np.clip(ema / max(EPS, self.config.orientation_advantage_scale), -self.config.robust_clip, self.config.robust_clip)
        self.log_orientation_distribution = normalize_log_probs(aligned_log)
        switch = self._orientation_switch_weights(candidates, hand_axis, hand_angular_velocity)
        predicted, lambda_dt, variance = self._jump_predict(self.log_orientation_distribution, standardized, switch, dt)
        posterior = self.config.orientation_w_sc * predicted + self.config.orientation_beta * standardized
        self.log_orientation_distribution = clip_log_probs(normalize_log_probs(posterior), self.config.orientation_max_prob)
        probabilities = np.exp(self.log_orientation_distribution)
        index = self._select_orientation_index(int(np.argmax(probabilities)), probabilities, candidates)
        selected = candidates[index]
        held_probability = float(probabilities[index])
        leader_probability = float(np.max(probabilities))
        confidence = (
            held_probability
            if self.pending_orientation_switch is None
            else leader_probability
        )
        profile = hand_frame_profile(self.config.hand_model)
        frame_world = None
        policy = "minimal_rotation"
        if POLICY_BY_PRIMITIVE.get(selected.primitive_kind, "minimal_rotation") == "grasp_frame" and selected.relation == "parallel":
            axis = selected.primitive.axis if selected.primitive is not None else selected.surface_axis
            # The cylinder axis is carried in source metadata by v4 candidates.
            if axis is not None:
                frame_world = grasp_frame_from_cylinder(axis, selected.normal, rotation, profile)
                policy = "grasp_frame" if frame_world is not None else "minimal_rotation"
        target = OrientationTarget(
            quaternion_xyzw=(_mat_to_quat_xyzw(frame_world) if frame_world is not None else self._quaternion_from_relation(selected, rotation)), axis_world=np.asarray(selected.normal, dtype=np.float64),
            relation=selected.relation, object_id=selected.object_id, confidence=confidence, source_position=np.asarray(selected.position, dtype=np.float64),
            policy=policy, frame_world=frame_world,
        )
        self.orientation_candidates = candidates
        self.last_orientation_target = target
        telemetry = {
            "enabled": True, "hand_axis_world": hand_axis.tolist(),
            "hand_omega_rps": None if hand_angular_velocity is None else np.asarray(hand_angular_velocity, dtype=np.float64).reshape(3).tolist(),
            "candidates": [
                {"object_id": c.object_id, "label": c.label, "role": roles[i], "relation": c.relation, "track_id": c.track_id, "source": c.source,
                 "position_goal_m": c.position.tolist(), "normal": c.normal.tolist(), "fit_quality": c.fit_quality, "primitive_kind": c.primitive_kind, "probability": float(probabilities[i]),
                 "advantage_raw": float(raw[i]), "advantage_ema": float(ema[i]), "advantage_standardized": float(standardized[i]),
                 "label_prior": float(priors[i]), "breakdown": breakdowns[i], "matched_previous": bool(i in match)}
                for i, c in enumerate(candidates)
            ], "selected_index": int(index), "selected_track_id": self.selected_orientation_track_id,
            "switch_dwell_count": 0 if self.pending_orientation_switch is None else int(self.pending_orientation_switch[1]),
            "candidate_count": int(len(candidates)), "matched_count": int(len(match)), "advantage_scale": float(self.config.orientation_advantage_scale),
            "held_probability": held_probability, "leader_probability": leader_probability,
            "orientation_valid": True,
            "target": {
                "object_id": target.object_id,
                "relation": target.relation,
                "axis_goal": target.axis_world.tolist(),
                "confidence": float(target.confidence),
                "source_position_goal_m": target.source_position.tolist(),
            },
        }
        return target, telemetry, lambda_dt, variance

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
        shoulder_position_m: np.ndarray | None = None,
        user_translation_mps: np.ndarray | None = None,
        ray_origin_m: np.ndarray | None = None,
        ray_direction: np.ndarray | None = None,
        controller_orientation_xyzw: np.ndarray | None = None,
        hand_rotation_matrix: np.ndarray | None = None,
        hand_angular_velocity: np.ndarray | None = None,
        hand_approach_axis: np.ndarray | None = None,
        contact_frames: dict[str, ContactFrame] | None = None,
        up_goal: np.ndarray | None = None,
        beta_override: float | None = None,
        lambda0_override: float | None = None,
        orientation_candidates: list[OrientationCandidate] | None = None,
        surface_status: dict[str, dict] | None = None,
        now_monotonic: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not self.config.enabled:
            self.last_telemetry = {"enabled": False, "objects": [], "target": None}
            return self.last_telemetry
        dt = 0.0 if self.last_update_monotonic is None else float(np.clip(now - self.last_update_monotonic, 0.0, 0.2))
        self.last_update_monotonic = now
        valid_objects: list[BeliefInputObject] = []
        for obj in objects:
            try:
                center = np.asarray(obj.center_goal_m, dtype=np.float64).reshape(3)
            except Exception:
                continue
            if np.isfinite(center).all():
                valid_objects.append(obj)
        if not valid_objects:
            self.surface_tracker.finish_frame(set())
            self.object_ids = []
            self.log_object_distribution = np.empty(0, dtype=np.float64)
            self.last_target = None
            if hand_rotation_matrix is None and controller_orientation_xyzw is not None:
                hand_rotation_matrix = _quat_xyzw_to_mat(
                    controller_orientation_xyzw
                )
            orientation_target, orientation_telemetry, ori_lambda_dt, ori_variance = (
                self._update_orientation_belief(
                    [],
                    hand_rotation_matrix=hand_rotation_matrix,
                    hand_angular_velocity=hand_angular_velocity,
                    dt=dt,
                    up_goal=up_goal,
                    orientation_candidates=orientation_candidates,
                )
            )
            self.feature_cost.finish_tick(user_translation_mps)
            self.last_telemetry = {
                "enabled": True,
                "objects": [],
                "object_probabilities": {},
                "target": None,
                "orientation": orientation_telemetry,
                "orientation_only_target": (
                    None
                    if orientation_target is None
                    else {
                        "object_id": orientation_target.object_id,
                        "relation": orientation_target.relation,
                        "axis_goal": orientation_target.axis_world.tolist(),
                        "confidence": float(orientation_target.confidence),
                    }
                ),
                "jump": {
                    "object_lambda_dt": 0.0,
                    "object_advantage_variance": 0.0,
                    "orientation_lambda_dt": float(ori_lambda_dt),
                    "orientation_advantage_variance": float(ori_variance),
                },
            }
            return self.last_telemetry
        valid_objects = valid_objects[: max(1, int(self.config.max_objects))]
        valid_object_ids = {obj.object_id for obj in valid_objects}
        standalone_geometry = any(obj.grip_candidates is None for obj in valid_objects)
        if standalone_geometry:
            self.surface_tracker.finish_frame(valid_object_ids)
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
                shoulder_position=shoulder_position_m,
                contact_normal=(contact_frames or {}).get(obj.object_id).normal if (contact_frames or {}).get(obj.object_id) is not None else None,
                hand_approach_axis=hand_approach_axis,
                contact_fit_quality=float((contact_frames or {}).get(obj.object_id).fit_quality) if (contact_frames or {}).get(obj.object_id) is not None else 0.0,
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
        # Predict before correct. The adaptive jump rate uses the pre-update
        # posterior and current innovation, never the post-update belief.
        object_switch = self._switch_weights_position(
            valid_objects, eef_position, user_translation_mps
        )
        predicted_obj, object_lambda_dt, object_variance = self._jump_predict(
            self.log_object_distribution, object_update, object_switch, dt,
            lambda0=lambda0_override,
        )
        beta = self.config.beta if beta_override is None else float(beta_override)
        log_obj = self.config.w_sc * predicted_obj + beta * object_update
        log_obj = clip_log_probs(normalize_log_probs(log_obj), self.config.max_prob_any_goal)
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
                candidates = []
                if obj.points_goal_m is not None:
                    candidates = self.surface_tracker.update(
                        obj.object_id,
                        obj.label,
                        obj.points_goal_m,
                    )
                    if not candidates:
                        candidates = generate_grip_candidates(
                            obj.points_goal_m, self.config.grip, up_goal=up_goal
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
            candidates = list(candidates)[: max(1, int(self.config.max_contacts_per_object))]
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
            grip_feature_breakdowns = [
                self.feature_cost.breakdown(
                    candidate.center,
                    eef_position=eef_position,
                    user_translation=user_translation_mps,
                    shoulder_position=shoulder_position_m,
                    contact_normal=(contact_frames or {}).get(obj.object_id).normal if (contact_frames or {}).get(obj.object_id) is not None else candidate.surface_normal,
                    hand_approach_axis=hand_approach_axis,
                    contact_fit_quality=float((contact_frames or {}).get(obj.object_id).fit_quality) if (contact_frames or {}).get(obj.object_id) is not None else float(candidate.geom_score),
                )
                for candidate in candidates
            ]
            # Compute the raw diagnostic every tick, even though the posterior
            # correction remains rate-limited below.  This does not change the
            # update; it makes skipped grip ticks distinguishable from a true
            # zero feature signal in telemetry/CSV.
            raw_grip_advantages = np.asarray(
                [
                    float(breakdown["weighted_total"])
                    for breakdown in grip_feature_breakdowns
                ],
                dtype=np.float64,
            )
            previous_grip_advantages = self.grip_advantage_ema.get(
                obj.object_id
            )
            grip_advantages = raw_grip_advantages.copy()
            if previous_grip_advantages is not None:
                for new_index, previous_index in grip_match.items():
                    if previous_index < previous_grip_advantages.shape[0]:
                        grip_advantages[new_index] = previous_grip_advantages[
                            previous_index
                        ]
            grip_update = robust_standardize(
                grip_advantages,
                min_signal_spread=self.config.robust_min_signal_spread,
                scale_floor=self.config.robust_scale_floor,
                clip=self.config.robust_clip,
            )
            if should_update_grips:
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
                    self.config.w_sc * log_grip + beta * grip_update
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
                    "surface_model": (surface_status or {}).get(obj.object_id, self.surface_tracker.status(obj.object_id)),
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
                            "advantage_standardized": float(grip_update[index]),
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
        if hand_rotation_matrix is None and controller_orientation_xyzw is not None:
            hand_rotation_matrix = _quat_xyzw_to_mat(controller_orientation_xyzw)
        orientation_target, orientation_telemetry, ori_lambda_dt, ori_variance = (
            self._update_orientation_belief(
                valid_objects,
                hand_rotation_matrix=hand_rotation_matrix,
                hand_angular_velocity=hand_angular_velocity,
                dt=dt,
                up_goal=up_goal,
                orientation_candidates=orientation_candidates,
            )
        )
        self.last_target = best_target
        target_dict = None
        if best_target is not None:
            if (
                orientation_target is not None
                and orientation_target.object_id
                not in {best_target.object_id, GRAVITY_TRACK_UID}
            ):
                orientation_target = None
            target_quaternion = (
                best_target.orientation_xyzw
                if orientation_target is None
                else orientation_target.quaternion_xyzw
            )
            target_dict = {
                "object_id": best_target.object_id,
                "grip_index": int(best_target.grip_index),
                "object_confidence": float(best_target.object_confidence),
                "joint_confidence": float(best_target.joint_confidence),
                "position_goal_m": best_target.position.tolist(),
                "orientation_xyzw": (
                    None if target_quaternion is None
                    else np.asarray(target_quaternion, dtype=np.float64).reshape(4).tolist()
                ),
            }
            if orientation_target is not None:
                target_dict.update({
                    "orientation_object_id": orientation_target.object_id,
                    "orientation_relation": orientation_target.relation,
                    "orientation_axis_world": orientation_target.axis_world.tolist(),
                    "orientation_confidence": float(orientation_target.confidence),
                    "orientation_source_position_m": orientation_target.source_position.tolist(),
                    "orientation_policy": orientation_target.policy,
                    "orientation_frame_world": None if orientation_target.frame_world is None else orientation_target.frame_world.tolist(),
                })
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
            "orientation": orientation_telemetry,
            "jump": {
                "object_lambda_dt": float(object_lambda_dt),
                "object_advantage_variance": float(object_variance),
                "orientation_lambda_dt": float(ori_lambda_dt),
                "orientation_advantage_variance": float(ori_variance),
            },
            "update": {
                "dt_s": float(dt),
                "beta": float(beta),
                "lambda0": float(
                    self.config.jump_lambda0
                    if lambda0_override is None
                    else lambda0_override
                ),
                "w_sc": float(self.config.w_sc),
                "max_prob_any_goal": float(self.config.max_prob_any_goal),
                "grip_update_applied": bool(should_update_grips),
                "grip_update_hz": float(self.config.grip_update_hz),
                "feature_ema_alpha": float(self.config.feature_ema_alpha),
                "orientation_beta": float(self.config.orientation_beta),
                "orientation_w_sc": float(self.config.orientation_w_sc),
                "orientation_max_prob": float(self.config.orientation_max_prob),
                "orientation_advantage_scale": float(
                    self.config.orientation_advantage_scale
                ),
            },
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
        jump = self.last_telemetry.get("jump") or {}
        orientation_text = ""
        if target.get("orientation_relation"):
            orientation_text = (
                f" ori={target.get('orientation_object_id')}:{target.get('orientation_relation')}"
                f"/{float(target.get('orientation_confidence', 0.0)):.2f}"
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
            f"jump={float(jump.get('object_lambda_dt', 0.0)):.3f}/"
            f"{float(jump.get('orientation_lambda_dt', 0.0)):.3f}"
            f"{orientation_text} objects=[{prob_text}]"
        )


class ParallelTascBelief:
    """Two independent hand posteriors over one immutable shared geometry frame."""

    def __init__(self, config: TascBeliefConfig | dict | None = None) -> None:
        self.config = config if isinstance(config, TascBeliefConfig) else TascBeliefConfig.from_dict(config)
        self.geometry_surface_tracker = PersistentCylinderSurfaceTracker(self.config.grip)
        self.orientation_bank = OrientationCandidateBank(
            self.config.orientation_bank
        )
        self.trackers = {
            "left": HierarchicalTascBelief(self.config),
            "right": HierarchicalTascBelief(self.config),
        }
        self.states = {side: HandBeliefState(side=side, beta_current=self.config.beta_idle, lambda0_current=self.config.lambda0_idle) for side in self.trackers}
        self.last_geometry: SharedGeometry | None = None
        self.last_update_monotonic: float | None = None

    def build_shared_geometry(
        self,
        objects: list[BeliefInputObject],
        *,
        ray_origin: np.ndarray | None = None,
        ray_direction: np.ndarray | None = None,
        up_goal: np.ndarray | None = None,
        base_quat_wxyz: np.ndarray | None = None,
        stamp_us: int = 0,
    ) -> SharedGeometry:
        """Compute hand-independent candidates once, then atomically replace cache."""
        kept = list(objects)[: self.config.max_objects]
        active = {obj.object_id for obj in kept}
        self.geometry_surface_tracker.finish_frame(active)
        object_candidates: dict[str, list[GripCandidate]] = {}
        object_primitives: dict[str, list[SurfacePrimitive]] = {}
        shared_objects: list[BeliefInputObject] = []
        for obj in kept:
            candidates = obj.grip_candidates
            if candidates is None:
                candidates = self.geometry_surface_tracker.update(obj.object_id, obj.label, obj.points_goal_m) if obj.points_goal_m is not None else []
                if not candidates and obj.points_goal_m is not None:
                    candidates = generate_grip_candidates(
                        obj.points_goal_m, self.config.grip, up_goal=up_goal
                    )
            candidates = list(candidates or [])[: self.config.max_contacts_per_object]
            primitives = obj.surface_primitives
            if primitives is None:
                cloud = obj.points_goal_m if obj.points_goal_m is not None else np.asarray([obj.center_goal_m])
                primitives = extract_surface_primitives(cloud, self.config.pseudo, cylinder_model=self.geometry_surface_tracker.models.get(obj.object_id), label=obj.label, up_goal=up_goal)
            primitives = list(primitives or [])[:1]
            object_candidates[obj.object_id] = candidates
            object_primitives[obj.object_id] = primitives
            shared_objects.append(BeliefInputObject(obj.object_id, obj.label, obj.score, obj.center_goal_m, None, candidates, primitives))
        observed_primitives = {
            obj.object_id: object_primitives[obj.object_id][0]
            for obj in shared_objects
            if object_primitives.get(obj.object_id)
        }
        labels = {obj.object_id: obj.label for obj in shared_objects}
        self.orientation_bank.observe(
            observed_primitives,
            labels,
            base_quat_wxyz=base_quat_wxyz,
            up_goal=up_goal,
            now_us=int(stamp_us),
        )
        self.orientation_bank.advance(int(stamp_us))
        scaled_relation_prior = {
            label: {
                relation: self.config.label_prior_scale * float(value)
                for relation, value in priors.items()
            }
            for label, priors in self.config.label_relation_prior.items()
        }
        orientation_candidates = self.orientation_bank.candidates(
            relations=self.config.orientation_relations,
            base_quat_wxyz=base_quat_wxyz,
            up_goal=up_goal,
            label_relation_prior=scaled_relation_prior,
            role_keywords=self.config.label_role_keywords,
            max_candidates=self.config.max_orientation_candidates,
        )
        orientation_bank_status = self.orientation_bank.status(
            base_quat_wxyz=base_quat_wxyz,
            up_goal=up_goal,
            now_us=int(stamp_us),
        )
        surface_status = {object_id: self.geometry_surface_tracker.status(object_id) for object_id in active}
        geometry = SharedGeometry(
            objects=shared_objects,
            grip_candidates=object_candidates,
            surface_primitives=object_primitives,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            stamp_us=int(stamp_us),
            up_goal=up_goal,
            surface_status=surface_status,
            base_quat_wxyz=(
                None
                if base_quat_wxyz is None
                else np.asarray(base_quat_wxyz, dtype=np.float64)
                .reshape(4)
                .copy()
            ),
            orientation_candidates=orientation_candidates,
            orientation_bank_status=orientation_bank_status,
        )
        self.last_geometry = geometry
        return geometry

    def update_hands(
        self,
        shared: SharedGeometry,
        inputs: dict[str, HandInput],
        *,
        now_monotonic: float | None = None,
    ) -> dict[str, dict]:
        """Update both hand posteriors every tick; buttons only gate authority."""
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        dt = 0.0 if self.last_update_monotonic is None else float(np.clip(now - self.last_update_monotonic, 0.0, 0.2))
        self.last_update_monotonic = now
        output: dict[str, dict] = {}
        for side, tracker in self.trackers.items():
            hand = inputs.get(side)
            if hand is None:
                continue
            state = self.states[side]
            state.armed = bool(hand.button_pressed)
            beta_target = self.config.beta_armed if state.armed else self.config.beta_idle
            lambda_target = self.config.lambda0_armed if state.armed else self.config.lambda0_idle
            ramp = float(np.clip(dt / max(EPS, self.config.beta_ramp_tau_s), 0.0, 1.0))
            state.beta_current += ramp * (beta_target - state.beta_current)
            state.lambda0_current += ramp * (lambda_target - state.lambda0_current)
            hand_rotation = hand.hand_rotation
            if hand_rotation is None and hand.hand_axis is not None:
                # No orientation target can be synthesized from an axis alone.
                hand_rotation = None
            contact_frames: dict[str, ContactFrame] = {}
            for obj in shared.objects:
                primitives = shared.surface_primitives.get(obj.object_id, [])
                if not primitives:
                    continue
                primitive = primitives[0]
                if (
                    self.config.orientation_bank.contact_requires_active
                    and str(primitive.source).startswith("bank_")
                    and primitive.source != "bank_active"
                ):
                    # A coasting axis remains rotation-correct, but its anchor
                    # has unobservable pelvis-translation error and cannot be
                    # used as a contact position.
                    continue
                prior_normal = None
                for old in tracker.orientation_candidates:
                    if old.object_id == obj.object_id:
                        prior_normal = old.normal
                        break
                frame = contact_frame(primitive, hand.eef_position, prior_normal)
                if frame is None:
                    continue
                contact_frames[obj.object_id] = frame
            orientation_candidates: list[OrientationCandidate] = []
            for candidate in shared.orientation_candidates:
                if candidate.role == "HELD" and self.config.role_hard_filter:
                    continue
                active_frame = contact_frames.get(candidate.object_id)
                if (
                    active_frame is not None
                    and candidate.source == "bank_active"
                ):
                    orientation_candidates.append(
                        replace(
                            candidate,
                            position=np.asarray(
                                active_frame.point, dtype=np.float64
                            ).reshape(3),
                            normal=np.asarray(
                                active_frame.normal, dtype=np.float64
                            ).reshape(3),
                        )
                    )
                else:
                    orientation_candidates.append(candidate)
            telemetry = tracker.update(
                shared.objects,
                eef_position_m=hand.eef_position,
                shoulder_position_m=hand.shoulder_position,
                user_translation_mps=hand.user_translation,
                ray_origin_m=shared.ray_origin,
                ray_direction=shared.ray_direction,
                hand_rotation_matrix=hand_rotation,
                hand_angular_velocity=hand.angular_velocity,
                hand_approach_axis=hand.approach_axis,
                contact_frames=contact_frames,
                up_goal=shared.up_goal,
                orientation_candidates=orientation_candidates,
                surface_status=shared.surface_status,
                beta_override=state.beta_current,
                lambda0_override=state.lambda0_current,
                now_monotonic=now,
            )
            target = telemetry.get("target")
            confidence = float(target.get("joint_confidence", 0.0)) if isinstance(target, dict) else 0.0
            alpha_target = confidence if state.armed else 0.0
            # Use the configured hand-state time constant for authority too;
            # a button edge must not turn into a Cartesian reference step.
            state.alpha += ramp * (alpha_target - state.alpha)
            state.alpha = float(np.clip(state.alpha, 0.0, 1.0))
            telemetry["hand"] = {"side": side, "armed": state.armed, "beta": state.beta_current, "lambda0": state.lambda0_current, "alpha": state.alpha}
            telemetry["orientation_bank"] = shared.orientation_bank_status
            output[side] = telemetry
        if self.config.dual_hand_exclusive and len(output) == 2:
            left, right = output.get("left"), output.get("right")
            if left and right and left["hand"]["armed"] and right["hand"]["armed"]:
                left_target, right_target = left.get("target"), right.get("target")
                if left_target and right_target and left_target.get("object_id") == right_target.get("object_id"):
                    weaker = "left" if left_target.get("joint_confidence", 0.0) < right_target.get("joint_confidence", 0.0) else "right"
                    output[weaker]["hand"]["alpha"] = 0.0
        return output
