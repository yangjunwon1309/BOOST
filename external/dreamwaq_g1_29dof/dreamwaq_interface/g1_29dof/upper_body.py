"""Fail-closed upper-body target multiplexing for the single LowCmd writer.

GR00T and XR teleoperation already publish the vendor ``rt/arm_sdk`` contract.
DreamWaQ consumes only its fourteen arm position targets.  It never accepts
leg or waist targets from that channel, and it never exposes upper-body targets
in the lower-body policy action-history observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .contract import JOINT_LOWER_SDK, JOINT_UPPER_SDK, JointOrderMap


@dataclass(frozen=True)
class ArmTarget14:
    """One timestamped arm target in Unitree SDK order (left seven, right seven)."""

    q_arm_sdk: np.ndarray
    authority_weight: float
    monotonic_ns: int
    sequence: int
    source: str = "arm_sdk"

    @classmethod
    def make(
        cls,
        q_arm_sdk: Any,
        *,
        monotonic_ns: int,
        sequence: int,
        authority_weight: float = 1.0,
        source: str = "arm_sdk",
    ) -> "ArmTarget14":
        target = np.asarray(q_arm_sdk, dtype=np.float32).reshape(-1)
        if target.shape != (14,) or not np.all(np.isfinite(target)):
            raise ValueError("arm target must contain exactly fourteen finite SDK-order values")
        stamp = int(monotonic_ns)
        seq = int(sequence)
        weight = float(authority_weight)
        if stamp <= 0 or seq < 0:
            raise ValueError("arm target identity is invalid")
        if not np.isfinite(weight) or not -0.05 <= weight <= 1.05:
            raise ValueError("arm_sdk authority weight is outside the tolerated [0, 1] envelope")
        label = str(source).strip()
        if label != "arm_sdk":
            raise ValueError("DreamWaQ accepts upper-body targets only from the pinned arm_sdk source")
        return cls(
            q_arm_sdk=target.copy(),
            authority_weight=float(np.clip(weight, 0.0, 1.0)),
            monotonic_ns=stamp,
            sequence=seq,
            source=label,
        )


class ArmTargetSource(Protocol):
    def sample(self) -> ArmTarget14 | None:
        ...

    def close(self) -> None:
        ...


class UpperBodyTargetMux:
    """Merge one optional arm source without changing lower-body ownership.

    Before the first valid source frame, the arms stay at the DreamWaQ default
    pose.  If a previously active source becomes stale, the last rate-limited
    target is held rather than initiating an uncommanded return motion.
    """

    def __init__(
        self,
        *,
        policy_dt: float = 0.02,
        max_target_age_ns: int = 100_000_000,
        velocity_limit_rad_s: float = 10.0,
        joint_limit_margin_rad: float = 0.0,
        default_q_sdk: Any,
        mapper: JointOrderMap | None = None,
    ) -> None:
        self.mapper = mapper or JointOrderMap()
        self.policy_dt = float(policy_dt)
        self.max_target_age_ns = int(max_target_age_ns)
        self.velocity_limit_rad_s = float(velocity_limit_rad_s)
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        if self.policy_dt <= 0.0 or self.max_target_age_ns <= 0 or self.velocity_limit_rad_s <= 0.0:
            raise ValueError("upper-body mux timing and velocity limits must be positive")
        if self.joint_limit_margin_rad < 0.0:
            raise ValueError("upper-body joint-limit margin must be non-negative")
        indices = self.mapper.arm_sdk_indices
        default = np.asarray(default_q_sdk, dtype=np.float32).reshape(-1)
        if default.shape != (29,) or not np.all(np.isfinite(default)):
            raise ValueError("FlashSAC default_q_sdk must contain 29 finite values")
        self._default = default[indices].astype(np.float32, copy=True)
        self._lower = (JOINT_LOWER_SDK[indices] + self.joint_limit_margin_rad).astype(np.float32)
        self._upper = (JOINT_UPPER_SDK[indices] - self.joint_limit_margin_rad).astype(np.float32)
        self._output = self._default.copy()
        self._last_sequence = -1
        self._ever_active = False
        self.active = False
        self.status = "default"

    def reset(self) -> None:
        self._output[:] = self._default
        self._last_sequence = -1
        self._ever_active = False
        self.active = False
        self.status = "default"

    def resolve(self, frame: ArmTarget14 | None, *, now_ns: int) -> np.ndarray:
        now = int(now_ns)
        if now <= 0:
            raise ValueError("upper-body mux requires a positive monotonic timestamp")
        if frame is None:
            self.active = False
            self.status = "hold_last_stale" if self._ever_active else "default"
            return self._output.copy()

        age = now - int(frame.monotonic_ns)
        if age < 0:
            raise ValueError("arm target is timestamped in the future")
        if age > self.max_target_age_ns:
            self.active = False
            self.status = "hold_last_stale" if self._ever_active else "default"
            return self._output.copy()
        if int(frame.sequence) < self._last_sequence:
            raise ValueError("arm target sequence moved backwards")

        requested = np.asarray(frame.q_arm_sdk, dtype=np.float32)
        outside = np.maximum(self._lower - requested, requested - self._upper)
        if float(np.max(outside)) > 0.05:
            raise ValueError("arm_sdk target exceeds a canonical G1 arm limit by more than 0.05 rad")
        bounded = np.clip(requested, self._lower, self._upper)
        desired = self._default * (1.0 - frame.authority_weight) + bounded * frame.authority_weight
        max_step = self.velocity_limit_rad_s * self.policy_dt
        self._output += np.clip(desired - self._output, -max_step, max_step)
        self._last_sequence = int(frame.sequence)
        self._ever_active = frame.authority_weight > 0.0 or self._ever_active
        self.active = frame.authority_weight > 0.0
        self.status = "arm_sdk" if self.active else "default_blend"
        return self._output.copy()


__all__ = ["ArmTarget14", "ArmTargetSource", "UpperBodyTargetMux"]
