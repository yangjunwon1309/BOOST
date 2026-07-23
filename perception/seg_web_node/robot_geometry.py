"""Timestamped G1 waist geometry for camera points in pelvis_identity_fk."""

from __future__ import annotations

import collections
import math

import numpy as np

from .lcm_protocol import RobotState


def _rot(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


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
