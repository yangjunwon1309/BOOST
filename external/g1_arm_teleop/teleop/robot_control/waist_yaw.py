"""Waist-yaw turn requested from Thor and applied by DreamWaQ on the G1.

DreamWaQ owns the waist (rt/lowcmd) and reads only the arms from rt/arm_sdk, so
the turn is a REQUEST on that same message, which its onboard runtime ramps,
limits and hides from its own policy observation
(g1_deploy/thirdparty/dreamwaq_g1_29dof/deploy/waist_override.py).  The protocol, mirrored in
g1_deploy/thirdparty/dreamwaq_g1_29dof/deploy/arm_target_source.py:

    motor_cmd[WAIST_YAW_ID].q      requested waist yaw, rad (default pose = 0)
    motor_cmd[WAIST_ENABLE_ID].q   1.0 while a turn is requested or unwinding

The arm IK keeps its waist-zero model, so the arms turn with the torso.
"""

from __future__ import annotations

import math

import numpy as np

WAIST_YAW_ID = 12          # kWaistYaw
WAIST_ENABLE_ID = 30       # kNotUsedJoint1
# Positive waist yaw turns the torso to the robot's LEFT (URDF axis +z), so a
# turn to the right is negative.
WAIST_TURN_RIGHT_DEG = 45.0
WAIST_TURN_RIGHT_RAD = -math.radians(WAIST_TURN_RIGHT_DEG)
# Slow enough that the reaction torque stays small (~2.2 s for 45 deg); the
# G1 side limits again at 0.5 rad/s and +-0.85 rad.
WAIST_RATE_RAD_S = 0.35
WAIST_LIMIT_RAD = 0.85


class WaistYawRamp:
    """The requested angle, moved toward its target at a fixed rate."""

    def __init__(self, rate_rad_s=WAIST_RATE_RAD_S, limit_rad=WAIST_LIMIT_RAD):
        self.rate_rad_s = float(rate_rad_s)
        self.limit_rad = float(limit_rad)
        self.target = 0.0
        self.command = 0.0

    def set_target(self, target_rad):
        target = float(target_rad)
        if not math.isfinite(target) or abs(target) > self.limit_rad:
            raise ValueError(f"waist yaw target {target_rad!r} is outside +-{self.limit_rad} rad")
        self.target = target

    def step(self, dt):
        max_step = self.rate_rad_s * max(0.0, float(dt))
        self.command += float(np.clip(self.target - self.command, -max_step, max_step))
        return self.command

    @property
    def active(self):
        """Whether the G1 should apply the request: turned, turning or unwinding."""
        return self.target != 0.0 or self.command != 0.0

    @property
    def settled(self):
        return self.command == self.target
