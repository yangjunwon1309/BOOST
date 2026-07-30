"""Arm-command post-processing matched to Unitree XR teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XrArmPostprocessResult:
    command_q: np.ndarray
    filtered_q: np.ndarray
    rate_limited: bool
    motion_scale: float
    max_command_step_rad: float


class XrArmCommandPostprocessor:
    """Replicate G1-29 XR teleop filtering and velocity clipping."""

    def __init__(
        self,
        *,
        weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1),
        velocity_limit_rad_s: float = 20.0,
        control_dt_s: float = 1.0 / 250.0,
        data_size: int = 14,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        if (
            self.weights.ndim != 1
            or self.weights.size == 0
            or not np.isclose(float(np.sum(self.weights)), 1.0)
        ):
            raise ValueError("XR arm filter weights must be non-empty and sum to 1")
        self.velocity_limit_rad_s = float(velocity_limit_rad_s)
        self.control_dt_s = float(control_dt_s)
        self.data_size = int(data_size)
        if self.velocity_limit_rad_s <= 0.0 or self.control_dt_s <= 0.0:
            raise ValueError("XR arm velocity limit and control dt must be positive")
        self._queue: list[np.ndarray] = []
        self._filtered = np.zeros(self.data_size, dtype=np.float64)

    def reset(self) -> None:
        self._queue.clear()
        self._filtered.fill(0.0)

    def _weighted_filter(self, raw_q: np.ndarray) -> np.ndarray:
        raw_q = np.asarray(raw_q, dtype=np.float64).reshape(self.data_size)
        if not self._queue or not np.array_equal(raw_q, self._queue[-1]):
            if len(self._queue) >= self.weights.size:
                self._queue.pop(0)
            self._queue.append(raw_q.copy())

        if len(self._queue) < self.weights.size:
            self._filtered = self._queue[-1].copy()
        else:
            data = np.stack(self._queue)
            self._filtered = np.array(
                [
                    np.convolve(data[:, index], self.weights, mode="valid")[-1]
                    for index in range(self.data_size)
                ],
                dtype=np.float64,
            )
        return self._filtered.copy()

    def apply(
        self,
        raw_q: np.ndarray,
        measured_q: np.ndarray,
    ) -> XrArmPostprocessResult:
        filtered_q = self._weighted_filter(raw_q)
        measured_q = np.asarray(measured_q, dtype=np.float64).reshape(self.data_size)
        delta = filtered_q - measured_q
        allowed_step = self.velocity_limit_rad_s * self.control_dt_s
        max_delta = float(np.max(np.abs(delta)))
        motion_scale = max_delta / allowed_step
        divisor = max(motion_scale, 1.0)
        command_q = measured_q + delta / divisor
        return XrArmPostprocessResult(
            command_q=command_q,
            filtered_q=filtered_q,
            rate_limited=motion_scale > 1.0,
            motion_scale=float(motion_scale),
            max_command_step_rad=float(np.max(np.abs(command_q - measured_q))),
        )
