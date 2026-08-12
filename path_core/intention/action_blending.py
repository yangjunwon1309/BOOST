"""Task-space target contracts for the PATH second-order reference controller."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np


LEFT_WRIST_LINK = "left_wrist_yaw_link"
RIGHT_WRIST_LINK = "right_wrist_yaw_link"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedControlConfig:
    zeta_armed: float = 2.0
    zeta_released: float = 1.2
    omega_n_position_armed: float = 1.8
    omega_n_orientation_armed: float = 2.5
    omega_n_position_released: float = 10.0
    omega_n_orientation_released: float = 12.0
    max_speed_mps: float = 0.35
    max_rate_rps: float = 1.5
    resync_distance_m: float = 0.15
    resync_angle_rad: float = 0.6
    stale_after_s: float = 0.3
    d_max_m: float = 0.9
    orientation_min_confidence: float = 0.35
    orientation_max_angle_rad: float = 1.047
    orientation_max_step_rate_rps: float = 3.0
    orientation_accept_dwell: int = 3
    waist_mode: str = "sonic_passthrough"

    @classmethod
    def from_dict(cls, payload: dict | None) -> "SharedControlConfig":
        data = dict(payload or {})
        obsolete = {"rho1", "rho2", "alpha_rate_up", "alpha_rate_down", "v_transfer_max", "startup_hold_s"}
        if obsolete.intersection(data):
            LOG.warning("obsolete shared-control alpha parameters are ignored by second-order reference control")
        if "zeta" in data:
            LOG.warning("shared-control zeta is deprecated; applying it to zeta_armed and zeta_released")
            data.setdefault("zeta_armed", data["zeta"])
            data.setdefault("zeta_released", data["zeta"])
        values = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        for key in ("zeta_armed", "zeta_released"):
            if key in values and float(values[key]) < 1.05:
                LOG.warning("%s=%s is below the stable overdamped limit; clamping to 1.05", key, values[key])
                values[key] = 1.05
        return cls(**values)


@dataclass(frozen=True)
class EETarget:
    position: np.ndarray
    confidence: float
    orientation_wxyz: np.ndarray | None = None  # legacy diagnostic only
    object_id: str = ""
    grip_index: int = -1
    goal_uid: str = ""
    timestamp_us: int = 0
    orientation_axis_world: np.ndarray | None = None
    orientation_relation: str = ""
    orientation_confidence: float = 0.0
    position_point: str = "canonical_grasp_center"

    @property
    def has_orientation_constraint(self) -> bool:
        return self.orientation_axis_world is not None and self.orientation_relation in {"parallel", "perpendicular"}


def _finite_vec3(values: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if values is None:
        return None
    try:
        value = np.asarray(values, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value).all() else None


def explicit_orientation_wxyz(payload: dict | None) -> np.ndarray | None:
    if not isinstance(payload, dict) or payload.get("orientation_valid") is not True:
        return None
    try:
        quat = np.asarray(payload.get("quaternion", payload.get("target_quaternion")), dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(quat).all() or norm <= 1.0e-12:
        return None
    quat = quat / norm
    return -quat if quat[0] < 0.0 else quat


def quaternion_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    quat = explicit_orientation_wxyz({"orientation_valid": True, "quaternion": quaternion_wxyz})
    if quat is None:
        raise ValueError("invalid explicit target quaternion")
    w, x, y, z = quat
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]], dtype=np.float64)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a, b = np.asarray(source, dtype=np.float64).reshape(3), np.asarray(target, dtype=np.float64).reshape(3)
    cross = np.cross(a, b)
    sine, cosine = float(np.linalg.norm(cross)), float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1.0e-9:
        if cosine > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper); axis /= max(1.0e-9, float(np.linalg.norm(axis)))
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        return np.eye(3) + 2.0 * (skew @ skew)
    axis = cross / sine
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    angle = float(np.arctan2(sine, cosine))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotation_target_from_relation(reference_rotation: np.ndarray, axis_world: np.ndarray, relation: str, hand_axis_local: np.ndarray) -> np.ndarray | None:
    """Minimal reference-frame rotation satisfying the surface relation."""
    r_ref = np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
    normal = _finite_vec3(axis_world)
    if normal is None:
        return None
    normal /= max(1.0e-9, float(np.linalg.norm(normal)))
    hand_axis = r_ref @ np.asarray(hand_axis_local, dtype=np.float64).reshape(3)
    hand_axis /= max(1.0e-9, float(np.linalg.norm(hand_axis)))
    cosine = float(np.clip(np.dot(hand_axis, normal), -1.0, 1.0))
    if relation == "parallel":
        target_axis = normal if cosine >= 0.0 else -normal
    elif relation == "perpendicular":
        projected = hand_axis - cosine * normal
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm < 1.0e-3:
            return None
        target_axis = projected / projected_norm
    else:
        return None
    return _rotation_between(hand_axis, target_axis) @ r_ref
