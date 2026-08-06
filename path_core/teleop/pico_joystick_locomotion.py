"""Safe PICO-controller locomotion mapping for SONIC planner commands."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


LOCOMOTION_MODE_IDLE = 0
LOCOMOTION_MODE_SLOW_WALK = 1
SONIC_DEFAULT_STANDING_HEIGHT = -1.0


@dataclass(frozen=True)
class PicoJoystickLocomotionConfig:
    deadzone: float = 0.18
    forward_max_mps: float = 0.30
    backward_max_mps: float = 0.15
    lateral_max_mps: float = 0.18
    minimum_nonzero_speed_mps: float = 0.08
    maximum_yaw_rate_rad_s: float = 0.30
    linear_acceleration_limit_mps2: float = 0.35
    linear_deceleration_limit_mps2: float = 0.70
    yaw_acceleration_limit_rad_s2: float = 0.60
    input_stale_timeout_sec: float = 0.15


@dataclass(frozen=True)
class SonicLocomotionCommand:
    mode: int
    movement: np.ndarray
    facing: np.ndarray
    speed: float
    height: float = SONIC_DEFAULT_STANDING_HEIGHT
    body_velocity_mps: np.ndarray | None = None
    yaw_rate_rad_s: float = 0.0
    active: bool = False
    reason: str = "idle"


def yaw_from_quaternion_wxyz(quaternion: Sequence[float]) -> float:
    """Return Z yaw from a scalar-first quaternion."""

    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size != 4 or not np.isfinite(q).all():
        return 0.0
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-9:
        return 0.0
    w, x, y, z = q / norm
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class PicoButtonGestureInterpreter:
    """Resolve individual X/Y releases without colliding with controller chords."""

    def __init__(self) -> None:
        self.previous = (False, False, False, False)
        self.x_active = False
        self.y_active = False
        self.x_chorded = False
        self.y_chorded = False

    def update(self, buttons: Sequence[bool]) -> list[str]:
        a, b, x, y = (bool(value) for value in buttons)
        pa, pb, px, py = self.previous

        if x and not px:
            self.x_active = True
            self.x_chorded = bool(a)
        if y and not py:
            self.y_active = True
            self.y_chorded = bool(b)

        if x and a:
            self.x_chorded = True
        if y and b:
            self.y_chorded = True
        if a and b and x and y:
            self.x_chorded = True
            self.y_chorded = True

        events: list[str] = []
        start_now = a and b and x and y
        start_previous = pa and pb and px and py
        ax_now = a and x
        ax_previous = pa and px
        by_now = b and y
        by_previous = pb and py
        if start_now and not start_previous:
            events.append("start_combo")
        elif ax_now and not ax_previous:
            events.append("ax")
        elif by_now and not by_previous:
            events.append("by")

        if px and not x:
            if self.x_active and not self.x_chorded:
                events.append("mode_q")
            self.x_active = False
            self.x_chorded = False
        if py and not y:
            if self.y_active and not self.y_chorded:
                events.append("mode_e")
            self.y_active = False
            self.y_chorded = False

        if (a, b, x, y) != self.previous:
            events.append("button_state")
        self.previous = (a, b, x, y)
        return events


class PicoJoystickLocomotionMapper:
    """Map raw thumbsticks to rate-limited SONIC high-level locomotion."""

    def __init__(self, config: PicoJoystickLocomotionConfig | None = None) -> None:
        self.config = config or PicoJoystickLocomotionConfig()
        self._body_velocity = np.zeros(2, dtype=np.float64)
        self._yaw_rate = 0.0
        self._yaw = 0.0
        self._initialized = False

    @property
    def yaw(self) -> float:
        return float(self._yaw)

    def reset(self, yaw_rad: float = 0.0) -> None:
        self._body_velocity.fill(0.0)
        self._yaw_rate = 0.0
        self._yaw = float(yaw_rad) if np.isfinite(yaw_rad) else 0.0
        self._initialized = True

    def idle(self, *, yaw_rad: float | None = None, reason: str = "disabled") -> SonicLocomotionCommand:
        if yaw_rad is not None and np.isfinite(yaw_rad):
            self._yaw = float(yaw_rad)
            self._initialized = True
        elif not self._initialized:
            self.reset(0.0)
        self._body_velocity.fill(0.0)
        self._yaw_rate = 0.0
        return self._command(reason=reason)

    def update(
        self,
        left_axis: Sequence[float],
        right_axis: Sequence[float],
        dt_sec: float,
        *,
        current_yaw_rad: float,
    ) -> SonicLocomotionCommand:
        if not self._initialized:
            self.reset(current_yaw_rad)

        left = self._axis(left_axis)
        right = self._axis(right_axis)
        dt = float(np.clip(dt_sec, 1.0e-4, 0.1))

        lx = self._shape_axis(left[0])
        ly = self._shape_axis(left[1])
        rx = self._shape_axis(right[0])

        forward_limit = (
            self.config.forward_max_mps
            if ly >= 0.0
            else self.config.backward_max_mps
        )
        # Body velocity order is [forward, left].  Positive PICO stick X is
        # right, hence the sign inversion for the robot-left component.
        target_velocity = np.array(
            [forward_limit * ly, -self.config.lateral_max_mps * lx],
            dtype=np.float64,
        )
        target_speed = float(np.linalg.norm(target_velocity))
        if target_speed > self.config.forward_max_mps:
            target_velocity *= self.config.forward_max_mps / target_speed
            target_speed = self.config.forward_max_mps
        if 1.0e-9 < target_speed < self.config.minimum_nonzero_speed_mps:
            target_velocity *= self.config.minimum_nonzero_speed_mps / target_speed

        current_speed = float(np.linalg.norm(self._body_velocity))
        target_speed = float(np.linalg.norm(target_velocity))
        accelerating = target_speed > current_speed + 1.0e-9
        linear_limit = (
            self.config.linear_acceleration_limit_mps2
            if accelerating
            else self.config.linear_deceleration_limit_mps2
        )
        self._body_velocity = self._rate_limit_vector(
            self._body_velocity,
            target_velocity,
            linear_limit * dt,
        )

        target_yaw_rate = -self.config.maximum_yaw_rate_rad_s * rx
        self._yaw_rate = self._rate_limit_scalar(
            self._yaw_rate,
            target_yaw_rate,
            self.config.yaw_acceleration_limit_rad_s2 * dt,
        )
        self._yaw += self._yaw_rate * dt
        self._yaw = math.atan2(math.sin(self._yaw), math.cos(self._yaw))
        return self._command(reason="tracking")

    def _command(self, *, reason: str) -> SonicLocomotionCommand:
        yaw = float(self._yaw)
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        left = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        world_velocity = (
            forward * self._body_velocity[0]
            + left * self._body_velocity[1]
        )
        speed = float(np.linalg.norm(world_velocity))
        moving = speed > 1.0e-6
        if moving:
            movement_xy = world_velocity / speed
            command_speed = max(speed, self.config.minimum_nonzero_speed_mps)
            mode = LOCOMOTION_MODE_SLOW_WALK
        else:
            movement_xy = np.zeros(2, dtype=np.float64)
            command_speed = -1.0
            mode = LOCOMOTION_MODE_IDLE
        facing = np.array([forward[0], forward[1], 0.0], dtype=np.float32)
        movement = np.array(
            [movement_xy[0], movement_xy[1], 0.0], dtype=np.float32
        )
        return SonicLocomotionCommand(
            mode=mode,
            movement=movement,
            facing=facing,
            speed=float(command_speed),
            body_velocity_mps=self._body_velocity.astype(np.float32).copy(),
            yaw_rate_rad_s=float(self._yaw_rate),
            active=bool(moving or abs(self._yaw_rate) > 1.0e-6),
            reason=reason,
        )

    def _shape_axis(self, value: float) -> float:
        value = float(np.clip(value, -1.0, 1.0))
        magnitude = abs(value)
        if magnitude <= self.config.deadzone:
            return 0.0
        normalized = (magnitude - self.config.deadzone) / (
            1.0 - self.config.deadzone
        )
        return math.copysign(normalized * normalized, value)

    @staticmethod
    def _axis(values: Sequence[float]) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if result.size < 2 or not np.isfinite(result[:2]).all():
            return np.zeros(2, dtype=np.float64)
        return np.clip(result[:2], -1.0, 1.0)

    @staticmethod
    def _rate_limit_vector(
        current: np.ndarray, target: np.ndarray, max_delta: float
    ) -> np.ndarray:
        delta = target - current
        norm = float(np.linalg.norm(delta))
        if norm <= max_delta or norm < 1.0e-12:
            return target.copy()
        return current + delta * (max_delta / norm)

    @staticmethod
    def _rate_limit_scalar(current: float, target: float, max_delta: float) -> float:
        return float(current + np.clip(target - current, -max_delta, max_delta))
