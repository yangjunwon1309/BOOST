"""Hold the waist yaw at an angle Thor asks for, hidden from the DreamWaQ policy.

The 12-action DreamWaQ exports were trained with the spine fixed at its default
by PD (manifest `spine_fixed: true`), so a turned waist is an observation the
policy has never seen.  This keeps the turn out of it: the writer commands the
waist to default + offset, and the state the policy reads has the same offset
taken back out, so the policy sees what the encoder would read with the waist
at its default -- tracking error and noise included, exactly as in training.

Thor requests the angle on rt/arm_sdk (see arm_target_source.WAIST_YAW_ID).
A missing, disengaged or stale request ramps the waist back to default.

`step` and `command` run on the LowCmd writer thread and `observe` on the
policy loop, so the offset and its rate are read and written under one lock.
"""

from __future__ import annotations

import math
import threading

import numpy as np

WAIST_YAW_SDK_INDEX = 12
# A small limit, not the joint's +-2.618: the request is a fixed turn of 45 deg
# (0.785 rad), and anything far past it is a bug on the sending side.
DEFAULT_LIMIT_RAD = 0.85
# Above Thor's own 0.35 rad/s ramp, so the turn follows it without lag; this
# bounds a jump if the sender ever steps instead of ramping.
DEFAULT_MAX_RATE_RAD_S = 0.5


class WaistYawOverride:
    def __init__(self, *, enabled, default_rad=0.0, limit_rad=DEFAULT_LIMIT_RAD,
                 max_rate_rad_s=DEFAULT_MAX_RATE_RAD_S):
        if not (limit_rad > 0.0 and max_rate_rad_s > 0.0):
            raise ValueError("waist yaw limit and rate must be positive")
        self.enabled = bool(enabled)
        self.default_rad = float(default_rad)
        self.limit_rad = float(limit_rad)
        self.max_rate_rad_s = float(max_rate_rad_s)
        self._lock = threading.Lock()
        self.offset = 0.0
        self.rate = 0.0
        self.target = 0.0
        self._last_ns = None
        self.ignored_requests = 0

    @property
    def active(self):
        return self.offset != 0.0 or self.target != 0.0

    def reset(self):
        """Back to default with no ramp: the caller has already moved the waist there."""
        with self._lock:
            self.offset = 0.0
            self.rate = 0.0
            self.target = 0.0
            self._last_ns = None

    def step(self, request_rad, now_ns):
        """Advance the applied offset toward `request_rad` (absolute, or None).

        Returns "begin" or "end" on the step the override starts or finishes.
        """
        with self._lock:
            was_active = self.active
            target = 0.0
            if request_rad is not None:
                if not self.enabled:
                    self.ignored_requests += 1
                elif math.isfinite(request_rad):
                    target = float(np.clip(request_rad - self.default_rad,
                                           -self.limit_rad, self.limit_rad))
            self.target = target
            dt = 0.0 if self._last_ns is None else max(0.0, (int(now_ns) - self._last_ns) * 1e-9)
            self._last_ns = int(now_ns)
            # A long gap (startup, a controller handoff) must not become one big step.
            max_step = self.max_rate_rad_s * min(dt, 0.1)
            previous = self.offset
            self.offset = previous + float(np.clip(target - previous, -max_step, max_step))
            self.rate = (self.offset - previous) / dt if dt > 0.0 else 0.0
            if self.active and not was_active:
                return "begin"
            if was_active and not self.active:
                return "end"
            return None

    def command(self, q_target):
        """The writer's 29-joint target with the waist yaw turned by the offset."""
        with self._lock:
            offset = self.offset
        if offset == 0.0:
            return q_target
        q = np.array(q_target, dtype=np.float32, copy=True)
        q[WAIST_YAW_SDK_INDEX] = self.default_rad + offset
        return q

    def observe(self, q29, dq29):
        """The state the policy reads: the offset and its rate taken back out."""
        with self._lock:
            offset, rate = self.offset, self.rate
        if offset == 0.0 and rate == 0.0:
            return q29, dq29
        q = np.array(q29, dtype=np.float32, copy=True)
        dq = np.array(dq29, dtype=np.float32, copy=True)
        q[WAIST_YAW_SDK_INDEX] -= offset
        dq[WAIST_YAW_SDK_INDEX] -= rate
        return q, dq
