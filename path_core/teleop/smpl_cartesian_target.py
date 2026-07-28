"""Stable pelvis-local Cartesian targets from relative SMPL wrist motion.

Only body-relative wrist displacement is used.  SMPL root translation and
global pelvis motion never enter the robot target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEFT_WRIST = 20
RIGHT_WRIST = 21
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
PELVIS = 0
TORSO = 9
NECK = 12


@dataclass(frozen=True)
class SmplCartesianTargetConfig:
    scale_xyz: tuple[float, float, float] = (0.85, 0.85, 0.85)
    displacement_sign: float = -1.0
    ema_alpha: float = 0.35
    max_velocity_m_s: float = 0.25
    max_acceleration_m_s2: float = 1.5
    jump_reject_m: float = 0.07
    max_anchor_offset_m: float = 0.45
    stale_after_sec: float = 0.20
    left_workspace_min: tuple[float, float, float] = (-0.10, -0.12, -0.35)
    left_workspace_max: tuple[float, float, float] = (0.65, 0.65, 0.55)
    right_workspace_min: tuple[float, float, float] = (-0.10, -0.65, -0.35)
    right_workspace_max: tuple[float, float, float] = (0.65, 0.12, 0.55)


@dataclass
class SmplCartesianTargets:
    left_transform: np.ndarray
    right_transform: np.ndarray
    source_timestamp_us: int
    frame_index: int
    valid: bool
    held: bool
    reason: str
    left_raw_position: np.ndarray
    right_raw_position: np.ndarray


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1.0e-7:
        return fallback.copy()
    return vector / norm


def torso_frame(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return torso origin and rotation whose columns are forward/left/up."""

    points = np.asarray(joints, dtype=np.float64).reshape(24, 3)
    origin = points[TORSO]
    left = _normalize(
        points[LEFT_SHOULDER] - points[RIGHT_SHOULDER],
        np.array([0.0, 1.0, 0.0]),
    )
    up_hint = _normalize(
        points[NECK] - points[PELVIS],
        np.array([0.0, 0.0, 1.0]),
    )
    forward = _normalize(np.cross(left, up_hint), np.array([1.0, 0.0, 0.0]))
    up = _normalize(np.cross(forward, left), np.array([0.0, 0.0, 1.0]))
    left = _normalize(np.cross(up, forward), np.array([0.0, 1.0, 0.0]))
    return origin, np.column_stack((forward, left, up))


def torso_relative_wrists(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(joints, dtype=np.float64).reshape(24, 3)
    origin, rotation = torso_frame(points)
    left = rotation.T @ (points[LEFT_WRIST] - origin)
    right = rotation.T @ (points[RIGHT_WRIST] - origin)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("non-finite SMPL torso-relative wrist")
    return left, right


class SmplCartesianTargetTracker:
    """Convert relative SMPL wrist displacement into safe robot FK targets."""

    def __init__(self, config: SmplCartesianTargetConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._smpl_left0: np.ndarray | None = None
        self._smpl_right0: np.ndarray | None = None
        self._anchor_left_tf: np.ndarray | None = None
        self._anchor_right_tf: np.ndarray | None = None
        self._left_position: np.ndarray | None = None
        self._right_position: np.ndarray | None = None
        self._left_velocity = np.zeros(3, dtype=np.float64)
        self._right_velocity = np.zeros(3, dtype=np.float64)
        self._last_smpl_left: np.ndarray | None = None
        self._last_smpl_right: np.ndarray | None = None
        self._last_source_timestamp_us = 0
        self._last_frame_index = -1
        self._last_targets: SmplCartesianTargets | None = None

    @property
    def initialized(self) -> bool:
        return self._anchor_left_tf is not None

    def initialize(
        self,
        *,
        smpl_joints: np.ndarray,
        robot_left_transform: np.ndarray,
        robot_right_transform: np.ndarray,
        source_timestamp_us: int,
        frame_index: int,
    ) -> SmplCartesianTargets:
        left, right = torso_relative_wrists(smpl_joints)
        self._smpl_left0 = left.copy()
        self._smpl_right0 = right.copy()
        self._last_smpl_left = left.copy()
        self._last_smpl_right = right.copy()
        self._anchor_left_tf = np.asarray(robot_left_transform, dtype=np.float64).reshape(4, 4).copy()
        self._anchor_right_tf = np.asarray(robot_right_transform, dtype=np.float64).reshape(4, 4).copy()
        self._left_position = self._anchor_left_tf[:3, 3].copy()
        self._right_position = self._anchor_right_tf[:3, 3].copy()
        self._left_velocity.fill(0.0)
        self._right_velocity.fill(0.0)
        return self._make_targets(
            self._left_position,
            self._right_position,
            self._left_position,
            self._right_position,
            source_timestamp_us,
            frame_index,
            valid=True,
            held=False,
            reason="initialized",
        )

    def _clamp_position(self, side: str, raw: np.ndarray) -> np.ndarray:
        assert self._anchor_left_tf is not None and self._anchor_right_tf is not None
        if side == "left":
            lower = np.asarray(self.config.left_workspace_min, dtype=np.float64)
            upper = np.asarray(self.config.left_workspace_max, dtype=np.float64)
            anchor = self._anchor_left_tf[:3, 3]
        else:
            lower = np.asarray(self.config.right_workspace_min, dtype=np.float64)
            upper = np.asarray(self.config.right_workspace_max, dtype=np.float64)
            anchor = self._anchor_right_tf[:3, 3]
        clipped = np.clip(raw, lower, upper)
        offset = clipped - anchor
        distance = float(np.linalg.norm(offset))
        if distance > self.config.max_anchor_offset_m:
            clipped = anchor + offset * (self.config.max_anchor_offset_m / distance)
        return clipped

    def _advance_limited(
        self,
        current: np.ndarray,
        velocity: np.ndarray,
        desired: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = float(np.clip(dt, 1.0e-4, 0.1))
        requested_velocity = (desired - current) / dt
        speed = float(np.linalg.norm(requested_velocity))
        if speed > self.config.max_velocity_m_s:
            requested_velocity *= self.config.max_velocity_m_s / speed
        velocity_delta = requested_velocity - velocity
        max_velocity_delta = self.config.max_acceleration_m_s2 * dt
        acceleration_step = float(np.linalg.norm(velocity_delta))
        if acceleration_step > max_velocity_delta:
            velocity_delta *= max_velocity_delta / acceleration_step
        next_velocity = velocity + velocity_delta
        next_position = current + next_velocity * dt
        return next_position, next_velocity

    def update(
        self,
        *,
        smpl_joints: np.ndarray,
        robot_left_transform: np.ndarray,
        robot_right_transform: np.ndarray,
        source_timestamp_us: int,
        frame_index: int,
        dt: float,
    ) -> SmplCartesianTargets:
        if not self.initialized:
            return self.initialize(
                smpl_joints=smpl_joints,
                robot_left_transform=robot_left_transform,
                robot_right_transform=robot_right_transform,
                source_timestamp_us=source_timestamp_us,
                frame_index=frame_index,
            )

        assert self._smpl_left0 is not None and self._smpl_right0 is not None
        assert self._anchor_left_tf is not None and self._anchor_right_tf is not None
        assert self._left_position is not None and self._right_position is not None

        left, right = torso_relative_wrists(smpl_joints)
        jump = 0.0
        if self._last_smpl_left is not None and self._last_smpl_right is not None:
            jump = max(
                float(np.linalg.norm(left - self._last_smpl_left)),
                float(np.linalg.norm(right - self._last_smpl_right)),
            )
        self._last_smpl_left = left.copy()
        self._last_smpl_right = right.copy()
        if jump > self.config.jump_reject_m:
            return self.hold(
                source_timestamp_us=source_timestamp_us,
                frame_index=frame_index,
                reason=f"smpl_jump_{jump:.3f}m",
            )

        scale = np.asarray(self.config.scale_xyz, dtype=np.float64)
        sign = float(self.config.displacement_sign)
        left_raw = self._anchor_left_tf[:3, 3] + sign * scale * (left - self._smpl_left0)
        right_raw = self._anchor_right_tf[:3, 3] + sign * scale * (right - self._smpl_right0)
        left_safe = self._clamp_position("left", left_raw)
        right_safe = self._clamp_position("right", right_raw)

        alpha = float(np.clip(self.config.ema_alpha, 0.0, 1.0))
        left_filtered = self._left_position + alpha * (left_safe - self._left_position)
        right_filtered = self._right_position + alpha * (right_safe - self._right_position)
        self._left_position, self._left_velocity = self._advance_limited(
            self._left_position, self._left_velocity, left_filtered, dt
        )
        self._right_position, self._right_velocity = self._advance_limited(
            self._right_position, self._right_velocity, right_filtered, dt
        )
        return self._make_targets(
            self._left_position,
            self._right_position,
            left_raw,
            right_raw,
            source_timestamp_us,
            frame_index,
            valid=True,
            held=False,
            reason="tracking",
        )

    def hold(
        self,
        *,
        source_timestamp_us: int | None = None,
        frame_index: int | None = None,
        reason: str = "held",
    ) -> SmplCartesianTargets:
        if self._last_targets is None:
            identity = np.eye(4, dtype=np.float64)
            return SmplCartesianTargets(
                identity.copy(),
                identity.copy(),
                int(source_timestamp_us or 0),
                int(frame_index if frame_index is not None else -1),
                False,
                True,
                "not_initialized",
                np.zeros(3),
                np.zeros(3),
            )
        held = self._last_targets
        return self._make_targets(
            held.left_transform[:3, 3],
            held.right_transform[:3, 3],
            held.left_raw_position,
            held.right_raw_position,
            int(source_timestamp_us or held.source_timestamp_us),
            int(frame_index if frame_index is not None else held.frame_index),
            valid=True,
            held=True,
            reason=reason,
        )

    def _make_targets(
        self,
        left_position: np.ndarray,
        right_position: np.ndarray,
        left_raw: np.ndarray,
        right_raw: np.ndarray,
        source_timestamp_us: int,
        frame_index: int,
        *,
        valid: bool,
        held: bool,
        reason: str,
    ) -> SmplCartesianTargets:
        left_tf = (
            self._anchor_left_tf.copy()
            if self._anchor_left_tf is not None
            else np.eye(4, dtype=np.float64)
        )
        right_tf = (
            self._anchor_right_tf.copy()
            if self._anchor_right_tf is not None
            else np.eye(4, dtype=np.float64)
        )
        left_tf[:3, 3] = np.asarray(left_position, dtype=np.float64)
        right_tf[:3, 3] = np.asarray(right_position, dtype=np.float64)
        output = SmplCartesianTargets(
            left_transform=left_tf,
            right_transform=right_tf,
            source_timestamp_us=int(source_timestamp_us),
            frame_index=int(frame_index),
            valid=bool(valid),
            held=bool(held),
            reason=str(reason),
            left_raw_position=np.asarray(left_raw, dtype=np.float64).copy(),
            right_raw_position=np.asarray(right_raw, dtype=np.float64).copy(),
        )
        self._last_source_timestamp_us = output.source_timestamp_us
        self._last_frame_index = output.frame_index
        self._last_targets = output
        return output

