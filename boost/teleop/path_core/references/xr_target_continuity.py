"""Per-wrist XR outlier rejection and weak SE(3) low-pass filtering."""

from __future__ import annotations

import math

import numpy as np

from .damped_se3 import so3_exp, so3_log


SIDES = ("left", "right")


def _valid_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(transform).all():
        raise ValueError("XR wrist transform must be finite")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
        raise ValueError("XR wrist transform has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
        raise ValueError("XR wrist rotation is not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2.0e-3:
        raise ValueError("XR wrist rotation is not proper")
    return transform.copy()


class XRTargetContinuity:
    """Accept plausible source-timestamped targets and filter them at tick rate."""

    def __init__(
        self,
        *,
        max_linear_speed_mps: float = 2.0,
        max_angular_speed_rps: float = 10.0,
        resync_after_sec: float = 0.3,
        cutoff_hz: float = 5.0,
    ) -> None:
        values = (
            max_linear_speed_mps,
            max_angular_speed_rps,
            resync_after_sec,
            cutoff_hz,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("XR continuity parameters must be finite and positive")
        self.max_linear_speed_mps = float(max_linear_speed_mps)
        self.max_angular_speed_rps = float(max_angular_speed_rps)
        self.resync_after_sec = float(resync_after_sec)
        self.cutoff_hz = float(cutoff_hz)
        self.accepted: dict[str, np.ndarray | None] = {side: None for side in SIDES}
        self.accepted_timestamp_us: dict[str, int | None] = {side: None for side in SIDES}
        self.filtered: dict[str, np.ndarray | None] = {side: None for side in SIDES}
        self.reject_started: dict[str, float | None] = {side: None for side in SIDES}
        self.reject_count = {side: 0 for side in SIDES}
        self.resync_count = {side: 0 for side in SIDES}

    def reset_session(self) -> None:
        for side in SIDES:
            self.accepted[side] = None
            self.accepted_timestamp_us[side] = None
            self.filtered[side] = None
            self.reject_started[side] = None

    def reset_filter_to_accepted(self, side: str | None = None) -> None:
        selected = SIDES if side is None else (side,)
        for hand in selected:
            accepted = self.accepted[hand]
            self.filtered[hand] = None if accepted is None else accepted.copy()

    def accept(self, side: str, transform, source_timestamp_us: int, now_s: float) -> str:
        """Return ``accepted``, ``rejected``, or ``resync`` for one XR frame."""
        if side not in SIDES:
            raise ValueError(f"unknown wrist side {side!r}")
        candidate = _valid_transform(transform)
        stamp = int(source_timestamp_us)
        accepted = self.accepted[side]
        previous_stamp = self.accepted_timestamp_us[side]
        if accepted is None or previous_stamp is None:
            self._store(side, candidate, stamp)
            return "accepted"

        delta_s = (stamp - previous_stamp) / 1.0e6
        if delta_s <= 0.0:
            return "rejected"
        linear_speed = float(np.linalg.norm(candidate[:3, 3] - accepted[:3, 3])) / delta_s
        angular_speed = float(np.linalg.norm(so3_log(accepted[:3, :3].T @ candidate[:3, :3]))) / delta_s
        is_outlier = (
            linear_speed > self.max_linear_speed_mps
            or angular_speed > self.max_angular_speed_rps
        )
        if not is_outlier:
            self.reject_started[side] = None
            self._store(side, candidate, stamp)
            return "accepted"

        self.reject_count[side] += 1
        if self.reject_started[side] is None:
            self.reject_started[side] = float(now_s)
        if float(now_s) - self.reject_started[side] < self.resync_after_sec:
            return "rejected"
        self.resync_count[side] += 1
        self.reject_started[side] = None
        self._store(side, candidate, stamp)
        return "resync"

    def _store(self, side: str, transform: np.ndarray, stamp: int) -> None:
        self.accepted[side] = transform.copy()
        self.accepted_timestamp_us[side] = stamp

    def tick(self, dt: float) -> dict[str, np.ndarray | None]:
        """Move each filtered transform toward the latest accepted target."""
        step = float(dt)
        if not math.isfinite(step) or step < 0.0:
            raise ValueError("dt must be finite and nonnegative")
        beta = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * step)
        for side in SIDES:
            accepted = self.accepted[side]
            current = self.filtered[side]
            if accepted is None:
                continue
            if current is None:
                self.filtered[side] = accepted.copy()
                continue
            updated = current.copy()
            updated[:3, 3] += beta * (accepted[:3, 3] - current[:3, 3])
            updated[:3, :3] = current[:3, :3] @ so3_exp(
                beta * so3_log(current[:3, :3].T @ accepted[:3, :3])
            )
            self.filtered[side] = updated
        return {
            side: None if self.filtered[side] is None else self.filtered[side].copy()
            for side in SIDES
        }
