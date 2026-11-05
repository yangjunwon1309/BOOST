"""The 50-Hz FlashSAC controller core.

Physical startup and DDS ownership live in the CLI parent process.  This file
only turns a canonical state and velocity command into the latest policy frame;
the HOMIE-style lower-body ramp and transport startup remain in the parent.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Protocol

import numpy as np

from .contract import DreamWaQ29Contract
from .control import LowerBodyActionCompiler
from .inference import InferenceResult
from .observation import DreamWaQObservationBuilder, FlashSACHistoryBuffer, RobotState29
from .recorder import SignalTraceRecorder, deployment_trace_shapes
from .safety import SafetyFault, SafetyMonitor
from .transport import CommandFrame29, CommandSink, StateSource
from .upper_body import ArmTargetSource, UpperBodyTargetMux


class InferenceBackend(Protocol):
    def infer(self, obs_frame: Any, obs_history: Any) -> InferenceResult:
        ...


@dataclass(frozen=True)
class RuntimeReceipt:
    sequence: int
    monotonic_ns: int
    command: np.ndarray
    state: RobotState29
    observation: np.ndarray
    history: np.ndarray
    inference: InferenceResult
    q_target_sdk: np.ndarray
    applied_lower_action: np.ndarray
    upper_body_status: str


class DreamWaQ29Runtime:
    """Stateful policy/observation controller shared by mock and policy child."""

    def __init__(
        self,
        *,
        contract: DreamWaQ29Contract,
        state_source: StateSource,
        command_sink: CommandSink,
        backend: InferenceBackend,
        safety: SafetyMonitor,
        kp_sdk: Any | None = None,
        kd_sdk: Any | None = None,
        motor_gain_scale: float = 1.0,
        policy_dt: float = 0.02,
        recorder: SignalTraceRecorder | None = None,
        arm_target_source: ArmTargetSource | None = None,
        arm_target_max_age_ns: int = 100_000_000,
        arm_velocity_limit_rad_s: float = 10.0,
        check_state_age: bool = True,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.contract = contract
        self.state_source = state_source
        self.command_sink = command_sink
        self.backend = backend
        self.safety = safety
        self.check_state_age = bool(check_state_age)
        self.policy_dt = float(policy_dt)
        self.clock_ns = clock_ns
        self.sleep_fn = sleep_fn
        if self.policy_dt <= 0.0:
            raise ValueError("policy_dt must be positive")
        nominal_kp = np.asarray(contract.kp_sdk if kp_sdk is None else kp_sdk, dtype=np.float32).reshape(-1)
        nominal_kd = np.asarray(contract.kd_sdk if kd_sdk is None else kd_sdk, dtype=np.float32).reshape(-1)
        if nominal_kp.shape != (29,) or nominal_kd.shape != (29,):
            raise ValueError("runtime gains must each contain 29 values")
        scale = float(motor_gain_scale)
        if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("motor_gain_scale must be finite and in [0, 1]")
        self.motor_gain_scale = scale
        self.kp_sdk = (nominal_kp * scale).astype(np.float32)
        self.kd_sdk = (nominal_kd * scale).astype(np.float32)
        self.observation_builder = DreamWaQObservationBuilder(contract=contract)
        self.history = FlashSACHistoryBuffer(length=contract.history_length, feature_dim=contract.measurable_observation_dim)
        self.action_compiler = LowerBodyActionCompiler(contract=contract)
        self.previous_lower_action = np.zeros(contract.action_dim, dtype=np.float32)
        self.upper_body_mux = UpperBodyTargetMux(
            mapper=self.action_compiler.mapper,
            policy_dt=self.policy_dt,
            max_target_age_ns=int(arm_target_max_age_ns),
            velocity_limit_rad_s=float(arm_velocity_limit_rad_s),
            default_q_sdk=contract.default_q_sdk,
        )
        self.arm_target_source = arm_target_source
        self.recorder = recorder
        self.sequence = 0
        self.closed = False

    def reset_policy_state(self) -> None:
        self.observation_builder.reset()
        self.history.reset()
        self.previous_lower_action.fill(0.0)
        self.upper_body_mux.reset()
        self.sequence = 0

    def _command(self, value: Any) -> np.ndarray:
        command = np.asarray(value, dtype=np.float32).reshape(-1)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise SafetyFault("velocity command must contain three finite values")
        if np.any(command < self.contract.command_lower - 1.0e-6) or np.any(command > self.contract.command_upper + 1.0e-6):
            raise SafetyFault("velocity command exceeds the policy training envelope")
        command[np.abs(command) < 1.0e-4] = 0.0
        return command

    def step(self, command: Any, *, state: RobotState29 | None = None) -> RuntimeReceipt:
        if self.closed:
            raise RuntimeError("DreamWaQ runtime is closed")
        state = self.state_source.sample() if state is None else state
        self.safety.check_state(state, check_age=self.check_state_age)
        command_array = self._command(command)
        observation = self.observation_builder.build(
            state, command=command_array, previous_lower_action=self.previous_lower_action
        )
        history = self.history.append(observation.measurable_current)
        inference = self.backend.infer(observation.policy_current, history)
        # Timing is recorded in the trace/parent diagnostics.  Do not print
        # from the policy thread: a terminal/SSH stdout flush can itself
        # block the 50-Hz loop.  The safety policy remains latest-value and
        # non-fatal, matching HOMIE.
        self.safety.check_inference_latency(inference.latency_ns)
        compiled = self.action_compiler.compile(inference.lower_action)
        arm_frame = self.arm_target_source.sample() if self.arm_target_source is not None else None
        q_target_sdk = compiled.q_target_sdk.copy()
        q_target_sdk[self.action_compiler.mapper.arm_sdk_indices] = self.upper_body_mux.resolve(
            arm_frame, now_ns=int(self.clock_ns())
        )
        if self.safety.limits.clip_pd_effort:
            q_target_sdk = self.safety.clip_pd_target(state, q_target_sdk, self.kp_sdk, self.kd_sdk)
        self.safety.check_target(
            state,
            q_target_sdk,
            enforce_tracking_error=(
                bool(getattr(self.command_sink, "is_hardware", False))
                and self.safety.limits.enforce_policy_target_error
            ),
        )
        self.safety.check_pd_command(state, q_target_sdk, self.kp_sdk, self.kd_sdk)
        stamp = int(self.clock_ns())
        frame = CommandFrame29.make(
            q_target_sdk=q_target_sdk,
            kp_sdk=self.kp_sdk,
            kd_sdk=self.kd_sdk,
            sequence=self.sequence,
            monotonic_ns=stamp,
        )
        self.command_sink.update(frame)
        if self.recorder is not None:
            self.recorder.append(
                monotonic_ns=np.asarray(stamp, dtype=np.int64),
                command=command_array,
                q_sdk=state.q_sdk,
                dq_sdk=state.dq_sdk,
                quaternion_wxyz=state.quaternion_wxyz,
                angular_velocity_body=state.angular_velocity_body,
                projected_gravity_body=self.observation_builder.projected_gravity(state),
                raw_control_frame=observation.raw_control,
                policy_current=observation.policy_current,
                cenet_history=history,
                v_hat=inference.v_hat,
                z=inference.z,
                actor_input=inference.actor_input,
                policy_mean=inference.policy_mean,
                lower_action=compiled.normalized_lower_action,
                previous_action_policy=compiled.previous_action_policy,
                q_target_sdk=q_target_sdk,
                inference_latency_ns=np.asarray(inference.latency_ns, dtype=np.int64),
            )
        receipt = RuntimeReceipt(
            sequence=self.sequence,
            monotonic_ns=stamp,
            command=command_array.copy(),
            state=state,
            observation=observation.policy_current.copy(),
            history=history.copy(),
            inference=inference,
            q_target_sdk=q_target_sdk.copy(),
            applied_lower_action=compiled.normalized_lower_action.copy(),
            upper_body_status=self.upper_body_mux.status,
        )
        self.previous_lower_action[:] = compiled.normalized_lower_action
        self.sequence += 1
        return receipt

    def run(
        self,
        *,
        command_source: Callable[[RobotState29], Any],
        duration_s: float,
        status_every_s: float = 5.0,
    ) -> int:
        duration = float(duration_s)
        if duration <= 0.0:
            raise ValueError("runtime duration must be positive")
        started = self.clock_ns()
        deadline = started + int(duration * 1.0e9)
        next_tick = started
        status_period_ns = int(float(status_every_s) * 1.0e9)
        next_status = None if status_period_ns <= 0 else started + status_period_ns
        steps = 0
        while self.clock_ns() < deadline:
            next_tick += int(self.policy_dt * 1.0e9)
            state = self.state_source.sample()
            command = command_source(state)
            receipt = self.step(command, state=state)
            steps += 1
            now = self.clock_ns()
            if next_status is not None and now >= next_status:
                print(
                    "[G1-29][runtime] step=%d command=(%.3f, %.3f, %.3f) inference_ms=%.3f max_action=%.3f gain_scale=%.3f upper_body=%s"
                    % (
                        receipt.sequence, receipt.command[0], receipt.command[1], receipt.command[2],
                        receipt.inference.latency_ns / 1.0e6,
                        float(np.max(np.abs(receipt.applied_lower_action))),
                        self.motor_gain_scale, receipt.upper_body_status,
                    ),
                    flush=True,
                )
                next_status = now + status_period_ns
            lateness = now - next_tick
            if lateness > self.safety.limits.max_tick_lateness_ns:
                if self.safety.limits.enforce_timing_deadlines:
                    raise SafetyFault("50-Hz policy loop missed its timing envelope")
            remaining = next_tick - self.clock_ns()
            if remaining > 0:
                self.sleep_fn(remaining / 1.0e9)
        return steps

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        errors: list[str] = []
        seen: set[int] = set()
        for obj in (self.command_sink, self.state_source, self.arm_target_source, self.recorder):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    errors.append("%s: %s" % (type(exc).__name__, exc))
        if errors:
            raise RuntimeError("runtime shutdown errors: " + "; ".join(errors))


def create_deployment_recorder(
    path: Any,
    contract: DreamWaQ29Contract,
    *,
    mode: str,
    motor_gain_scale: float = 1.0,
) -> SignalTraceRecorder:
    metadata = contract.metadata()
    metadata["runtime_mode"] = str(mode)
    metadata["motor_gain_scale"] = float(motor_gain_scale)
    return SignalTraceRecorder(
        path, signal_shapes=deployment_trace_shapes(contract.action_dim),
        metadata=metadata, chunk_steps=512,
    )
