"""Hold tracking losses and bound Revo2 jumps without a continuous low-pass."""

from __future__ import annotations

import math

import numpy as np


class HandCommandContinuity:
    def __init__(
        self, *, stale_sec: float = 0.20, hold_sec: float = 1.0,
        reacquire_gap_sec: float = 0.15, recovery_tau_sec: float = 0.10,
        recovery_speed: float = 2.0, minimum_recovery_sec: float = 0.25,
        tracking_speed: float = 5.0,
    ) -> None:
        values = (stale_sec, hold_sec, reacquire_gap_sec, recovery_tau_sec,
                  recovery_speed, minimum_recovery_sec, tracking_speed)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("hand continuity parameters must be finite and positive")
        if hold_sec < stale_sec or reacquire_gap_sec >= hold_sec:
            raise ValueError("hand hold must cover the stale timeout and reacquire gap")
        self.stale_sec = stale_sec
        self.hold_sec = hold_sec
        self.reacquire_gap_sec = reacquire_gap_sec
        self.recovery_tau_sec = recovery_tau_sec
        self.recovery_speed = recovery_speed
        self.minimum_recovery_sec = minimum_recovery_sec
        self.tracking_speed = tracking_speed
        self.source_time: float | None = None
        self.output: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.lost = False
        self.released = False
        self.recovery_started: float | None = None
        self.last_step: float | None = None

    @staticmethod
    def _six(value) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError("hand command must contain six finite normalized values")
        return np.clip(vector, 0.0, 1.0)

    def mark_lost(self, *, released: bool = False) -> None:
        self.lost = True
        self.released = self.released or released

    def needs_recovery(self, source_time: float) -> bool:
        return (self.output is None or self.lost or self.source_time is None
                or source_time - self.source_time > self.reacquire_gap_sec)

    def recovery_seed(self, measured, source_time: float) -> np.ndarray:
        if (self.released or self.output is None or self.source_time is None
                or source_time - self.source_time > self.hold_sec):
            return self._six(measured)
        return self.output.copy()

    def submit(self, target, source_time: float, now: float, *, measured=None) -> bool:
        """Use a new source sample; duplicate manager heartbeats do not refresh it."""
        goal = self._six(target)
        if not math.isfinite(source_time) or not math.isfinite(now):
            raise ValueError("hand timestamps must be finite")
        if self.source_time is not None and source_time <= self.source_time:
            return False
        recovering = self.needs_recovery(source_time)
        if recovering:
            if measured is not None:
                self.output = self.recovery_seed(measured, source_time)
            elif self.output is None:
                self.output = goal.copy()
            self.recovery_started = now
            self.last_step = now
        self.target = goal
        self.source_time = source_time
        self.lost = False
        self.released = False
        return True

    def command(self, now: float) -> tuple[np.ndarray | None, str]:
        if not math.isfinite(now):
            raise ValueError("hand command time must be finite")
        if self.source_time is None or self.output is None or self.target is None:
            return None, "waiting-hand"
        dt = min(0.05, max(0.0, now - (self.last_step if self.last_step is not None else now)))
        self.last_step = now
        age = max(0.0, now - self.source_time)
        if age > self.hold_sec:
            self.mark_lost(released=True)
            return None, "stale-stop"
        if self.released:
            return None, "waiting-recovery-seed"
        if self.lost or age > self.stale_sec:
            self.mark_lost()
            return self.output.copy(), "tracking-hold"
        if self.recovery_started is not None:
            previous = self.output.copy()
            delta = (1.0 - math.exp(-dt / self.recovery_tau_sec)) * (self.target - previous)
            self.output = previous + np.clip(
                delta, -self.recovery_speed * dt, self.recovery_speed * dt
            )
            if (now - self.recovery_started >= self.minimum_recovery_sec
                    and np.max(np.abs(self.target - self.output)) < 1.0e-4):
                self.recovery_started = None
            return self.output.copy(), "recovering"
        self.output += np.clip(
            self.target - self.output, -self.tracking_speed * dt, self.tracking_speed * dt
        )
        return self.output.copy(), "tracking"
