"""Timestamped G1 waist geometry for camera points in pelvis_identity_fk."""

from __future__ import annotations

import collections
import math
import logging

import numpy as np

from .lcm_protocol import RobotState


LOG = logging.getLogger(__name__)


JOINT_NAMES_URDF = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


JOINT_DATA = {
    "left_hip_pitch_joint": ("pelvis", "left_hip_pitch_link", [0, 0.064452, -0.1027], [0, 0, 0], [0, 1, 0]),
    "left_hip_roll_joint": ("left_hip_pitch_link", "left_hip_roll_link", [0, 0.052, -0.030465], [0, -0.1749, 0], [1, 0, 0]),
    "left_hip_yaw_joint": ("left_hip_roll_link", "left_hip_yaw_link", [0.025001, 0, -0.12412], [0, 0, 0], [0, 0, 1]),
    "left_knee_joint": ("left_hip_yaw_link", "left_knee_link", [-0.078273, 0.0021489, -0.17734], [0, 0.1749, 0], [0, 1, 0]),
    "left_ankle_pitch_joint": ("left_knee_link", "left_ankle_pitch_link", [0, -9.4445e-5, -0.30001], [0, 0, 0], [0, 1, 0]),
    "left_ankle_roll_joint": ("left_ankle_pitch_link", "left_ankle_roll_link", [0, 0, -0.017558], [0, 0, 0], [1, 0, 0]),
    "right_hip_pitch_joint": ("pelvis", "right_hip_pitch_link", [0, -0.064452, -0.1027], [0, 0, 0], [0, 1, 0]),
    "right_hip_roll_joint": ("right_hip_pitch_link", "right_hip_roll_link", [0, -0.052, -0.030465], [0, -0.1749, 0], [1, 0, 0]),
    "right_hip_yaw_joint": ("right_hip_roll_link", "right_hip_yaw_link", [0.025001, 0, -0.12412], [0, 0, 0], [0, 0, 1]),
    "right_knee_joint": ("right_hip_yaw_link", "right_knee_link", [-0.078273, -0.0021489, -0.17734], [0, 0.1749, 0], [0, 1, 0]),
    "right_ankle_pitch_joint": ("right_knee_link", "right_ankle_pitch_link", [0, 9.4445e-5, -0.30001], [0, 0, 0], [0, 1, 0]),
    "right_ankle_roll_joint": ("right_ankle_pitch_link", "right_ankle_roll_link", [0, 0, -0.017558], [0, 0, 0], [1, 0, 0]),
    "waist_yaw_joint": ("pelvis", "waist_yaw_link", [0, 0, 0], [0, 0, 0], [0, 0, 1]),
    "waist_roll_joint": ("waist_yaw_link", "waist_roll_link", [-0.0039635, 0, 0.035], [0, 0, 0], [1, 0, 0]),
    "waist_pitch_joint": ("waist_roll_link", "torso_link", [0, 0, 0.019], [0, 0, 0], [0, 1, 0]),
    "left_shoulder_pitch_joint": ("torso_link", "left_shoulder_pitch_link", [0.0039563, 0.10022, 0.23778], [0.27931, 0, 0], [0, 1, 0]),
    "left_shoulder_roll_joint": ("left_shoulder_pitch_link", "left_shoulder_roll_link", [0, 0.038, -0.013831], [-0.27925, 0, 0], [1, 0, 0]),
    "left_shoulder_yaw_joint": ("left_shoulder_roll_link", "left_shoulder_yaw_link", [0, 0.00624, -0.1032], [0, 0, 0], [0, 0, 1]),
    "left_elbow_joint": ("left_shoulder_yaw_link", "left_elbow_link", [0.015783, 0, -0.080518], [0, 0, 0], [0, 1, 0]),
    "left_wrist_roll_joint": ("left_elbow_link", "left_wrist_roll_link", [0.100, 0.00188791, -0.010], [0, 0, 0], [1, 0, 0]),
    "left_wrist_pitch_joint": ("left_wrist_roll_link", "left_wrist_pitch_link", [0.038, 0, 0], [0, 0, 0], [0, 1, 0]),
    "left_wrist_yaw_joint": ("left_wrist_pitch_link", "left_wrist_yaw_link", [0.046, 0, 0], [0, 0, 0], [0, 0, 1]),
    "right_shoulder_pitch_joint": ("torso_link", "right_shoulder_pitch_link", [0.0039563, -0.10021, 0.23778], [-0.27931, 0, 0], [0, 1, 0]),
    "right_shoulder_roll_joint": ("right_shoulder_pitch_link", "right_shoulder_roll_link", [0, -0.038, -0.013831], [0.27925, 0, 0], [1, 0, 0]),
    "right_shoulder_yaw_joint": ("right_shoulder_roll_link", "right_shoulder_yaw_link", [0, -0.00624, -0.1032], [0, 0, 0], [0, 0, 1]),
    "right_elbow_joint": ("right_shoulder_yaw_link", "right_elbow_link", [0.015783, 0, -0.080518], [0, 0, 0], [0, 1, 0]),
    "right_wrist_roll_joint": ("right_elbow_link", "right_wrist_roll_link", [0.100, -0.00188791, -0.010], [0, 0, 0], [1, 0, 0]),
    "right_wrist_pitch_joint": ("right_wrist_roll_link", "right_wrist_pitch_link", [0.038, 0, 0], [0, 0, 0], [0, 1, 0]),
    "right_wrist_yaw_joint": ("right_wrist_pitch_link", "right_wrist_yaw_link", [0.046, 0, 0], [0, 0, 0], [0, 0, 1]),
}


JOINT_ORDER = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

G1_SHOULDER_LINK = {
    "left": "left_shoulder_pitch_link",
    "right": "right_shoulder_pitch_link",
}
G1_ARM_REACH_MAX_M = 0.55
G1_ARM_REACH_MIN_M = 0.18


def _rot(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _homogeneous(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _rpy_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rot("z", yaw) @ _rot("y", pitch) @ _rot("x", roll)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ], dtype=np.float64)


def g1_link_transforms(q: np.ndarray) -> dict[str, np.ndarray]:
    """Return pelvis-frame link transforms from the G1 29-DoF URDF chain."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] < 29 or not np.isfinite(q[:29]).all():
        raise ValueError("G1 q must contain at least 29 finite joints")
    q_by_name = {name: float(q[i]) for i, name in enumerate(JOINT_NAMES_URDF)}
    link_transforms = {"pelvis": np.eye(4, dtype=np.float64)}
    for joint_name in JOINT_ORDER:
        parent, child, xyz, rpy, axis = JOINT_DATA[joint_name]
        parent_transform = link_transforms.get(parent)
        if parent_transform is None:
            raise ValueError(f"FK parent link unavailable: {parent}")
        origin = _homogeneous(
            _rpy_rotation(*[float(v) for v in rpy]),
            np.asarray(xyz, dtype=np.float64),
        )
        joint_transform = parent_transform @ origin
        rotation = _homogeneous(
            _axis_angle_rotation(np.asarray(axis, dtype=np.float64), q_by_name[joint_name])
        )
        link_transforms[child] = joint_transform @ rotation
    return link_transforms


def g1_link_position(q: np.ndarray, link_name: str) -> np.ndarray:
    transforms = g1_link_transforms(q)
    transform = transforms.get(link_name)
    if transform is None:
        raise ValueError(f"unknown G1 link: {link_name}")
    return transform[:3, 3].copy()


def g1_shoulder_position(q: np.ndarray, side: str) -> np.ndarray:
    """Return the pelvis-frame shoulder pitch origin for one arm."""
    normalized = "left" if str(side).lower().startswith("l") else "right"
    return g1_link_position(q, G1_SHOULDER_LINK[normalized])


def g1_wrist_ee_position(
    q: np.ndarray,
    target_link: str = "left_wrist_yaw_link",
    ee_offset_m: float = 0.05,
) -> np.ndarray:
    """Return pelvis-frame EE point, matching the QP/FK viewer wrist offset."""
    transforms = g1_link_transforms(q)
    transform = transforms.get(target_link)
    if transform is None:
        raise ValueError(f"unknown G1 target link: {target_link}")
    local = np.array([float(ee_offset_m), 0.0, 0.0, 1.0], dtype=np.float64)
    return (transform @ local)[:3].copy()


_BRAINCO_HAND_BASE_ROTATION_FROM_WRIST = {
    # MuJoCo g1_29dof_with_revo2.xml: wrist_yaw_link -> {side}_base_link.
    # Quaternions are wxyz. BrainCo's spread axis is base-local +Y.
    "left": np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64),
    "right": np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64),
}
_BRAINCO_HAND_BASE_TRANSLATION_FROM_WRIST = {
    "left": np.array([0.0415, 0.0030, 0.0], dtype=np.float64),
    "right": np.array([0.0415, -0.0030, 0.0], dtype=np.float64),
}

# The wrist-to-base attachment is retained from the existing assembly model.
# Its exact Dex1-1 verification remains a hardware/URDF preflight item.
_DEX1_OFFSET_VERIFIED = False
_DEX1_OFFSET_WARNING_EMITTED = False
_DEX1_HAND_BASE_ROTATION_FROM_WRIST = {
    "left": np.eye(3, dtype=np.float64),
    "right": np.eye(3, dtype=np.float64),
}
_DEX1_HAND_BASE_TRANSLATION_FROM_WRIST = {
    # Measured usable Dex1 grasp-base reference: +X is forward from the
    # wrist-yaw axis, 45 mm in the plane normal to the yaw axis.
    "left": np.array([0.0450, 0.0, 0.0], dtype=np.float64),
    "right": np.array([0.0450, 0.0, 0.0], dtype=np.float64),
}

# Keep the existing 6 cm canonical EE offset while transitioning hardware.
# It remains a configurable base-local point; it is deliberately not replaced
# with the estimated Dex1 fingertip centre until hardware validation is done.
_BRAINCO_GRASP_CENTER_LOCAL = np.array([0.0, 0.0, 0.060], dtype=np.float64)
_DEX1_GRASP_CENTER_LOCAL = np.array([0.0, 0.100, 0.0], dtype=np.float64)

# The orientation relation is defined against this physical hand axis.  It is
# deliberately exported so perception and the PATH bridge cannot silently use
# different Dex1/BrainCo conventions.
HAND_AXIS_LOCAL = {
    "brainco": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "dex1": np.array([0.0, 1.0, 0.0], dtype=np.float64),
}

_HAND_FRAME_PROFILES = {
    "brainco": {
        "rotation": _BRAINCO_HAND_BASE_ROTATION_FROM_WRIST,
        "translation": _BRAINCO_HAND_BASE_TRANSLATION_FROM_WRIST,
        "grasp_center_local": _BRAINCO_GRASP_CENTER_LOCAL,
        "orientation_axis_local": HAND_AXIS_LOCAL["brainco"],
        "approach_axis_local": HAND_AXIS_LOCAL["brainco"],
        "spread_axis_local": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "stagger_axis_local": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    },
    "dex1": {
        "rotation": _DEX1_HAND_BASE_ROTATION_FROM_WRIST,
        "translation": _DEX1_HAND_BASE_TRANSLATION_FROM_WRIST,
        "grasp_center_local": _DEX1_GRASP_CENTER_LOCAL,
        # Dex1-1: +Y is finger extension; +/-X is the parallel-jaw spread.
        "orientation_axis_local": HAND_AXIS_LOCAL["dex1"],
        "approach_axis_local": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "spread_axis_local": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "stagger_axis_local": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "jaw_stroke_m": 0.089,
        "jaw_gap_open_m": 0.0951,
        "jaw_gap_closed_m": 0.0061,
    },
}


def hand_frame_profile(hand_model: str = "dex1") -> dict[str, object]:
    """Return copies of the selected physical hand-frame convention."""
    model = str(hand_model).strip().lower()
    if model not in _HAND_FRAME_PROFILES:
        raise ValueError(f"unsupported hand model: {hand_model!r}")
    profile = _HAND_FRAME_PROFILES[model]
    global _DEX1_OFFSET_WARNING_EMITTED
    if model == "dex1" and not _DEX1_OFFSET_VERIFIED and not _DEX1_OFFSET_WARNING_EMITTED:
        LOG.warning("Dex1 wrist/base offset is unverified; validate against the installed Dex1 URDF and hardware before tuning grasp offsets")
        _DEX1_OFFSET_WARNING_EMITTED = True
    return {
        "name": model,
        "rotation": {side: value.copy() for side, value in profile["rotation"].items()},
        "translation": {side: value.copy() for side, value in profile["translation"].items()},
        "grasp_center_local": profile["grasp_center_local"].copy(),
        "orientation_axis_local": profile["orientation_axis_local"].copy(),
        "approach_axis_local": profile["approach_axis_local"].copy(),
        "spread_axis_local": profile["spread_axis_local"].copy(),
        "stagger_axis_local": profile["stagger_axis_local"].copy(),
        "jaw_stroke_m": float(profile.get("jaw_stroke_m", 0.0)),
        "jaw_gap_open_m": float(profile.get("jaw_gap_open_m", 0.0)),
        "jaw_gap_closed_m": float(profile.get("jaw_gap_closed_m", 0.0)),
        "offset_verified": bool(_DEX1_OFFSET_VERIFIED) if model == "dex1" else True,
    }


def _hand_profile_for_link(hand_model: str, target_link: str) -> tuple[str, dict[str, object]]:
    side = "left" if str(target_link).startswith("left_") else "right"
    model = str(hand_model).strip().lower()
    if model not in _HAND_FRAME_PROFILES:
        raise ValueError(f"unsupported hand model: {hand_model!r}")
    return side, _HAND_FRAME_PROFILES[model]


def g1_hand_base_rotation(
    q: np.ndarray,
    target_link: str,
    hand_model: str = "dex1",
) -> np.ndarray:
    """Return pelvis<-selected-hand-base rotation from a wrist-yaw link."""
    side, profile = _hand_profile_for_link(hand_model, target_link)
    transforms = g1_link_transforms(q)
    wrist = transforms.get(target_link)
    if wrist is None:
        raise ValueError(f"unknown G1 target link: {target_link}")
    return wrist[:3, :3] @ profile["rotation"][side]


def g1_hand_base_transform(
    q: np.ndarray,
    target_link: str,
    hand_model: str = "dex1",
) -> np.ndarray:
    """Return the pelvis<-selected-hand-base transform from a wrist link."""
    side, profile = _hand_profile_for_link(hand_model, target_link)
    transforms = g1_link_transforms(q)
    wrist = transforms.get(target_link)
    if wrist is None:
        raise ValueError(f"unknown G1 target link: {target_link}")
    return wrist @ _homogeneous(profile["rotation"][side], profile["translation"][side])


def g1_grasp_center_position(
    q: np.ndarray,
    target_link: str = "left_wrist_yaw_link",
    grasp_center_local_m: np.ndarray | None = None,
    hand_model: str = "dex1",
) -> np.ndarray:
    """Return the pelvis-frame canonical grasp centre for the selected hand."""
    _, profile = _hand_profile_for_link(hand_model, target_link)
    base = g1_hand_base_transform(q, target_link, hand_model)
    local = (
        profile["grasp_center_local"]
        if grasp_center_local_m is None
        else np.asarray(grasp_center_local_m, dtype=np.float64).reshape(3)
    )
    return (base @ np.array([local[0], local[1], local[2], 1.0], dtype=np.float64))[:3].copy()


def g1_wrist_to_grasp_offset(
    q: np.ndarray,
    target_link: str,
    grasp_center_local_m: np.ndarray | None = None,
    hand_model: str = "dex1",
) -> np.ndarray:
    """Return the pelvis-frame canonical grasp-centre offset from the wrist."""
    wrist = g1_link_transforms(q).get(target_link)
    if wrist is None:
        raise ValueError(f"unknown G1 target link: {target_link}")
    return (
        g1_grasp_center_position(q, target_link, grasp_center_local_m, hand_model)
        - wrist[:3, 3]
    )


def gravity_up_in_pelvis(base_quat_wxyz: np.ndarray | None) -> np.ndarray | None:
    """Return world up expressed in the pelvis-identity goal frame."""
    if base_quat_wxyz is None:
        return None
    try:
        rotation = quaternion_rotation_wxyz(np.asarray(base_quat_wxyz, dtype=np.float64))
    except (TypeError, ValueError):
        return None
    up = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(up))
    return None if norm < 1.0e-6 else up / norm


BRAINCO_GRASP_CENTER_LOCAL = _BRAINCO_GRASP_CENTER_LOCAL.copy()
DEX1_GRASP_CENTER_LOCAL = _DEX1_GRASP_CENTER_LOCAL.copy()
HAND_BASE_ROTATION_FROM_WRIST = {
    name: {side: value.copy() for side, value in profile["rotation"].items()}
    for name, profile in _HAND_FRAME_PROFILES.items()
}
HAND_BASE_TRANSLATION_FROM_WRIST = {
    name: {side: value.copy() for side, value in profile["translation"].items()}
    for name, profile in _HAND_FRAME_PROFILES.items()
}


def angular_velocity_from_rotations(
    previous: np.ndarray,
    current: np.ndarray,
    dt_sec: float,
) -> np.ndarray | None:
    """World-frame angular velocity from two world<-hand rotations."""
    dt = float(dt_sec)
    if not (1.0e-4 <= dt <= 0.5):
        return None
    delta = np.asarray(current, dtype=np.float64).reshape(3, 3) @ np.asarray(previous, dtype=np.float64).reshape(3, 3).T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(math.acos(cosine))
    if angle < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) < 1.0e-6:
        values, vectors = np.linalg.eigh(delta + np.eye(3))
        axis = vectors[:, int(np.argmax(values))]
    else:
        axis = np.array([
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ], dtype=np.float64) / (2.0 * sine)
    norm = float(np.linalg.norm(axis))
    return None if not np.isfinite(norm) or norm < 1.0e-9 else axis / norm * (angle / dt)


def quaternion_rotation_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 0.5:
        raise ValueError("invalid IMU quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def waist_fk(q: np.ndarray) -> np.ndarray:
    """Official G1 29-DoF pelvis -> torso_link chain (joints 12,13,14)."""
    q = np.asarray(q, dtype=float)
    if q.size < 15:
        raise ValueError("G1 state has fewer than 15 joints")
    transform = np.eye(4)
    transform[:3, :3] = _rot("z", q[12])
    step = np.eye(4); step[:3, 3] = [-0.0039635, 0.0, 0.035]
    transform = transform @ step
    step = np.eye(4); step[:3, :3] = _rot("x", q[13])
    transform = transform @ step
    step = np.eye(4); step[:3, 3] = [0.0, 0.0, 0.019]
    transform = transform @ step
    step = np.eye(4); step[:3, :3] = _rot("y", q[14])
    return transform @ step


class RobotStateBuffer:
    def __init__(self, capacity: int = 250):
        self.states = collections.deque(maxlen=capacity)

    def add(self, state: RobotState) -> None:
        self.states.append(state)

    def nearest(self, timestamp_us: int, max_delta_ms: float) -> tuple[RobotState, float] | None:
        if not self.states:
            return None
        state = min(self.states, key=lambda item: abs(item.timestamp_us - timestamp_us))
        delta_ms = abs(state.timestamp_us - timestamp_us) / 1000.0
        return (state, delta_ms) if delta_ms <= max_delta_ms else None


def pelvis_to_torso(state: RobotState, mode: str = "dual_imu", max_disagreement_deg: float = 12.0) -> tuple[np.ndarray, dict]:
    fk = waist_fk(state.q)
    metadata = {
        "mode": "joint_fk", "imu_fk_disagreement_deg": None,
        "max_waist_speed_rad_s": float(np.max(np.abs(state.dq[12:15]))),
    }
    if mode == "identity":
        metadata["mode"] = "identity"
        return np.eye(4, dtype=float), metadata
    if mode == "joint_fk":
        return fk, metadata
    if mode != "dual_imu":
        raise ValueError(f"unknown torso estimator: {mode}")
    if not state.has_torso_imu:
        raise ValueError("G1 state has no torso IMU")
    r_base = quaternion_rotation_wxyz(state.base_quat_wxyz)
    r_torso = quaternion_rotation_wxyz(state.torso_quat_wxyz)
    relative = r_base.T @ r_torso
    residual = relative.T @ fk[:3, :3]
    angle = math.degrees(math.acos(np.clip((np.trace(residual) - 1) / 2, -1, 1)))
    if angle > max_disagreement_deg:
        raise ValueError(f"dual-IMU/FK disagreement {angle:.1f} deg exceeds {max_disagreement_deg:.1f} deg")
    result = fk.copy()
    result[:3, :3] = relative
    metadata = {
        "mode": "dual_imu", "imu_fk_disagreement_deg": angle,
        "max_waist_speed_rad_s": float(np.max(np.abs(state.dq[12:15]))),
    }
    return result, metadata


def camera_goal_transform(
    t_torso_cam: np.ndarray | None,
    torso_estimator: str,
    state_buffer: RobotStateBuffer,
    timestamp_us: int,
    max_delta_ms: float,
) -> np.ndarray | None:
    """Resolve T_goal_camera, avoiding an unnecessary state gate in identity mode."""
    if t_torso_cam is None:
        return None
    camera = np.asarray(t_torso_cam, dtype=np.float64)
    if camera.shape != (4, 4):
        raise ValueError("T_torso_camera must be 4x4")
    if torso_estimator == "identity":
        return camera.copy()
    match = state_buffer.nearest(timestamp_us, max_delta_ms)
    if match is None:
        return None
    state, _delta_ms = match
    t_pelvis_torso, _metadata = pelvis_to_torso(state, torso_estimator)
    return t_pelvis_torso @ camera
