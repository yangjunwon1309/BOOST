"""Simulator-independent, fail-closed G1 state and command checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from .contract import (
    JOINT_LOWER_SDK,
    JOINT_UPPER_SDK,
    JOINT_VELOCITY_LIMIT_SDK,
)
from .observation import RobotState29, quat_apply_inverse_wxyz


class SafetyFault(RuntimeError):
    """A latched condition that forbids a driving command."""


@dataclass(frozen=True)
class SafetyLimits:
    max_state_age_ns: int = 40_000_000
    # Permit a tethered G1 to recover through a substantial transient, while
    # still stopping well before the simulator's 60-degree fall criterion.
    max_tilt_rad: float = math.radians(40.0)
    measured_joint_velocity_multiplier: float = 1.25
    # MuJoCo clips commanded positions to the model joint range.  Keep the
    # command-side margin at zero so the target contract is identical; the
    # hardware profile still has an independent measured-state guard.
    target_joint_limit_margin_rad: float = 0.0
    measured_joint_limit_tolerance_rad: float = 0.0
    max_policy_target_error_rad: float = 0.75
    # MuJoCo has no separate target-error fault; the PD effort is clipped.
    # Keep this opt-in for strict bring-up diagnostics, not for parity mode.
    enforce_policy_target_error: bool = False
    max_predicted_effort_fraction: float = 1.0
    # MuJoCo saturates PD effort and continues stepping.  The physical writer
    # uses the same target-projection saturation by default.
    clip_pd_effort: bool = True
    max_inference_latency_ns: int = 15_000_000
    max_tick_lateness_ns: int = 8_000_000
    # HOMIE records Linux scheduling jitter but keeps publishing the newest
    # safe command.  State/mode/tilt/joint integrity stays fail-closed; only
    # software scheduling deadlines are diagnostic by default.
    enforce_timing_deadlines: bool = False

    def __post_init__(self) -> None:
        if self.max_state_age_ns <= 0 or self.max_inference_latency_ns <= 0 or self.max_tick_lateness_ns <= 0:
            raise ValueError("safety timing limits must be positive")
        if not 0.0 < self.max_tilt_rad < math.pi / 2.0:
            raise ValueError("max_tilt_rad is invalid")
        if min(
            self.measured_joint_velocity_multiplier,
            self.max_policy_target_error_rad,
            self.max_predicted_effort_fraction,
        ) <= 0.0:
            raise ValueError("safety motion limits must be positive")
        if self.target_joint_limit_margin_rad < 0.0:
            raise ValueError("target_joint_limit_margin_rad must be non-negative")
        if self.measured_joint_limit_tolerance_rad < 0.0:
            raise ValueError("measured_joint_limit_tolerance_rad must be non-negative")

    @classmethod
    def sim2sim_compatible(cls) -> "SafetyLimits":
        """Return the non-physical guard profile used by replay audits.

        MuJoCo records a fall at 60 degrees and permits a small joint-limit
        constraint penetration while continuing to step.  This profile is for
        mock/shadow parity only; physical ``real`` mode must keep the stricter
        hardware emergency envelope.
        """

        return cls(
            max_tilt_rad=math.radians(60.0),
            measured_joint_limit_tolerance_rad=0.01,
            target_joint_limit_margin_rad=0.0,
            clip_pd_effort=True,
            max_inference_latency_ns=50_000_000,
            max_tick_lateness_ns=50_000_000,
        )


class SafetyMonitor:
    """Latch the first state/target/deadline failure until explicit reset."""

    def __init__(
        self,
        *,
        limits: SafetyLimits = SafetyLimits(),
        expected_mode_machine: int | None = None,
        torque_limit_sdk: Any,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.limits = limits
        self.expected_mode_machine = expected_mode_machine
        self.torque_limit_sdk = np.asarray(torque_limit_sdk, dtype=np.float32).reshape(-1)
        if self.torque_limit_sdk.shape != (29,) or not np.all(np.isfinite(self.torque_limit_sdk)) or np.any(self.torque_limit_sdk <= 0.0):
            raise ValueError("FlashSAC torque limits must contain 29 positive finite values")
        self.clock_ns = clock_ns
        self.fault_reason: str | None = None

    def reset(self) -> None:
        self.fault_reason = None

    def _fault(self, reason: str) -> None:
        if self.fault_reason is None:
            self.fault_reason = str(reason)
        raise SafetyFault(self.fault_reason)

    @staticmethod
    def tilt_rad(state: RobotState29) -> float:
        gravity = quat_apply_inverse_wxyz(state.quaternion_wxyz, (0.0, 0.0, -1.0))
        return float(math.acos(max(-1.0, min(1.0, -float(gravity[2])))))

    def check_state(
        self,
        state: RobotState29,
        *,
        now_ns: int | None = None,
        check_age: bool = True,
    ) -> None:
        """Validate a decoded state.

        The DDS-owning parent always checks true LowState receipt freshness.
        A policy process sees only a latest-value shared-memory cache, so its
        scheduling delay must not be misclassified as a hardware LowState
        outage.  That path passes ``check_age=False`` while retaining every
        mode, tilt, position, and velocity guard.
        """

        if self.fault_reason is not None:
            raise SafetyFault(self.fault_reason)
        now = int(self.clock_ns() if now_ns is None else now_ns)
        age = now - int(state.monotonic_ns)
        if check_age and (age < 0 or age > self.limits.max_state_age_ns):
            self._fault("LowState is stale or timestamped in the future (age_ms=%.3f)" % (age / 1.0e6))
        if self.expected_mode_machine is not None and state.mode_machine != int(self.expected_mode_machine):
            self._fault("LowState mode_machine differs from the operator-pinned mode")
        tilt_rad = self.tilt_rad(state)
        if tilt_rad > self.limits.max_tilt_rad:
            self._fault(
                "base tilt exceeds the deployment safety bound "
                "(tilt_deg=%.2f limit_deg=%.2f)"
                % (math.degrees(tilt_rad), math.degrees(self.limits.max_tilt_rad))
            )
        tolerance = float(self.limits.measured_joint_limit_tolerance_rad) + 1.0e-5
        lower = JOINT_LOWER_SDK - tolerance
        upper = JOINT_UPPER_SDK + tolerance
        if np.any(state.q_sdk < lower) or np.any(state.q_sdk > upper):
            self._fault("measured joint position is outside the canonical G1 limit")
        velocity_bound = JOINT_VELOCITY_LIMIT_SDK * self.limits.measured_joint_velocity_multiplier
        if np.any(np.abs(state.dq_sdk) > velocity_bound):
            self._fault("measured joint velocity exceeds the deployment safety bound")

    def check_target(
        self,
        state: RobotState29,
        q_target_sdk: Any,
        *,
        enforce_tracking_error: bool = True,
    ) -> None:
        target = np.asarray(q_target_sdk, dtype=np.float32).reshape(-1)
        if target.shape != (29,) or not np.all(np.isfinite(target)):
            self._fault("command target is not exactly 29 finite values")
        inward_lower = JOINT_LOWER_SDK + self.limits.target_joint_limit_margin_rad
        inward_upper = JOINT_UPPER_SDK - self.limits.target_joint_limit_margin_rad
        if np.any(target < inward_lower) or np.any(target > inward_upper):
            self._fault("command target is outside the inward G1 joint limits")
        if (
            enforce_tracking_error
            and self.limits.enforce_policy_target_error
            and float(np.max(np.abs(target - state.q_sdk))) > self.limits.max_policy_target_error_rad
        ):
            self._fault("measured-to-command joint error exceeds the policy safety bound")

    def check_inference_latency(self, latency_ns: int) -> bool:
        """Return whether inference exceeded its budget, faulting only if strict."""

        late = int(latency_ns) < 0 or int(latency_ns) > self.limits.max_inference_latency_ns
        if late and self.limits.enforce_timing_deadlines:
            self._fault(
                "DreamWaQ inference missed its deployment deadline "
                "(latency_ms=%.3f limit_ms=%.3f)"
                % (float(latency_ns) / 1.0e6, self.limits.max_inference_latency_ns / 1.0e6)
            )
        return late

    def clip_pd_target(
        self,
        state: RobotState29,
        q_target_sdk: Any,
        kp_sdk: Any,
        kd_sdk: Any,
        tau_ff_sdk: Any = None,
    ) -> np.ndarray:
        """Project a target through MuJoCo's clipped position-PD semantics.

        Unitree position mode accepts q/kp/kd rather than a per-cycle torque.
        For positive Kp, reconstruct the equivalent q target whose PD effort
        equals the clipped MuJoCo effort.  The writer calls this at 500 Hz
        using the newest LowState, instead of applying a stale 50 Hz clamp.
        """

        target = np.asarray(q_target_sdk, dtype=np.float32).reshape(-1)
        kp = np.asarray(kp_sdk, dtype=np.float32).reshape(-1)
        kd = np.asarray(kd_sdk, dtype=np.float32).reshape(-1)
        tau_ff = np.zeros(29, dtype=np.float32) if tau_ff_sdk is None else np.asarray(tau_ff_sdk, dtype=np.float32).reshape(-1)
        if any(value.shape != (29,) for value in (target, kp, kd, tau_ff)):
            self._fault("PD command does not contain exactly 29 values")
        if not all(np.all(np.isfinite(value)) for value in (target, kp, kd, tau_ff)):
            self._fault("PD command contains NaN or infinity")

        lower = JOINT_LOWER_SDK + self.limits.target_joint_limit_margin_rad
        upper = JOINT_UPPER_SDK - self.limits.target_joint_limit_margin_rad
        bounded_target = np.clip(target, lower, upper).astype(np.float32)
        predicted = kp * (bounded_target - state.q_sdk) - kd * state.dq_sdk
        effort_limit = self.torque_limit_sdk * float(self.limits.max_predicted_effort_fraction)
        # Feed-forward torque is sent through the same Unitree motor command as
        # the position-PD effort.  Project the *sum*, not just the PD term.
        # That preserves the requested tau_ff whenever possible and leaves the
        # remaining actuator authority to position tracking.
        clipped_pd = np.clip(predicted + tau_ff, -effort_limit, effort_limit) - tau_ff
        result = bounded_target.copy()
        controllable = np.abs(kp) > 1.0e-8
        result[controllable] = (
            state.q_sdk[controllable]
            + (clipped_pd[controllable] + kd[controllable] * state.dq_sdk[controllable])
            / kp[controllable]
        )
        result = np.clip(result, lower, upper).astype(np.float32)
        residual = kp * (result - state.q_sdk) - kd * state.dq_sdk + tau_ff
        if np.any(np.abs(residual[~controllable]) > effort_limit[~controllable] + 1.0e-5):
            self._fault("uncontrollable damping effort exceeds the canonical G1 torque envelope")
        return result

    def check_pd_command(
        self,
        state: RobotState29,
        q_target_sdk: Any,
        kp_sdk: Any,
        kd_sdk: Any,
        tau_ff_sdk: Any = None,
    ) -> None:
        """Reject a position command whose unsaturated PD effort is unsafe."""

        target = np.asarray(q_target_sdk, dtype=np.float32).reshape(-1)
        kp = np.asarray(kp_sdk, dtype=np.float32).reshape(-1)
        kd = np.asarray(kd_sdk, dtype=np.float32).reshape(-1)
        tau_ff = np.zeros(29, dtype=np.float32) if tau_ff_sdk is None else np.asarray(tau_ff_sdk, dtype=np.float32).reshape(-1)
        if any(value.shape != (29,) for value in (target, kp, kd, tau_ff)):
            self._fault("PD command does not contain exactly 29 values")
        if not all(np.all(np.isfinite(value)) for value in (target, kp, kd, tau_ff)):
            self._fault("PD command contains NaN or infinity")
        predicted = kp * (target - state.q_sdk) - kd * state.dq_sdk + tau_ff
        bound = self.torque_limit_sdk * float(self.limits.max_predicted_effort_fraction)
        if np.any(np.abs(predicted) > bound + 1.0e-5):
            self._fault("predicted PD effort exceeds the canonical G1 torque envelope")
