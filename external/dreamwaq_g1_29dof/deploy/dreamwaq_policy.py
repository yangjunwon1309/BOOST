"""Small HOMIE-process adapter for the sealed FlashSAC DreamWaQ export.

The robot I/O, arm ownership, LowCmd writer, default-pose ramp, and gains stay
in the copied HOMIE deployment.  This module replaces only HOMIE's policy
math: DreamWaQ CENet + actor inference and its manifest-defined action compiler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dreamwaq_interface.g1_29dof.contract import DreamWaQ29Contract
from dreamwaq_interface.g1_29dof.control import LowerBodyActionCompiler
from dreamwaq_interface.g1_29dof.inference import TorchScriptDreamWaQ
from dreamwaq_interface.g1_29dof.observation import (
    DreamWaQObservationBuilder,
    FlashSACHistoryBuffer,
    RobotState29,
)


class DreamWaQPolicy:
    """Run DreamWaQ with the exact HOMIE policy-process interface."""

    def __init__(
        self,
        export: Any,
        *,
        device: str = "cpu",
        clip_lower_joint_targets: bool = True,
    ) -> None:
        self.contract = DreamWaQ29Contract.load(Path(export))
        self.backend = TorchScriptDreamWaQ(self.contract, device=device)
        self.backend.warmup(20)
        self.observation_builder = DreamWaQObservationBuilder(contract=self.contract)
        self.history = FlashSACHistoryBuffer(
            length=self.contract.history_length,
            feature_dim=self.contract.measurable_observation_dim,
        )
        self.compiler = LowerBodyActionCompiler(
            contract=self.contract,
            clip_joint_targets=clip_lower_joint_targets,
        )
        self.previous_lower_action = np.zeros(self.contract.action_dim, dtype=np.float32)
        self.last_lower_action = np.zeros(self.contract.action_dim, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.locomotion_mode = 0
        self._initialized = False

    def reset(self, q29: np.ndarray | None = None) -> None:
        self.observation_builder.reset()
        self.history.reset()
        self.previous_lower_action.fill(0.0)
        self.last_lower_action.fill(0.0)
        if q29 is not None:
            q = np.asarray(q29, dtype=np.float32).reshape(29)
            # Seed the action history from the measured lower pose, matching
            # HOMIE's warm-start behavior instead of injecting zero actions.
            indices = self.compiler.action_sdk_indices
            default = self.contract.default_q_sdk[indices]
            scale = self.compiler.physical_action_scale
            self.previous_lower_action[:] = np.clip(
                (q[indices] - default) / scale,
                -1.0,
                1.0,
            )

    def set_command(self, command: Any) -> None:
        value = np.asarray(command, dtype=np.float32).reshape(-1)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("DreamWaQ command must contain three finite values")
        # The HOMIE joystick limits are deliberately kept inside the exported
        # DreamWaQ command envelope. Keep this clamp here as a second boundary
        # for custom configs or future command sources.
        self.command[:] = np.clip(
            value, self.contract.command_lower, self.contract.command_upper
        )

    def set_locomotion_mode(self, mode: int) -> None:
        if mode not in (0, 1):
            raise ValueError("locomotion mode must be standing=0 or walking=1")
        self.locomotion_mode = int(mode)

    def update(
        self,
        q29: Any,
        dq29: Any,
        quaternion_wxyz: Any,
        angular_velocity_body: Any,
        previous_lower_action: Any | None = None,
    ) -> tuple[np.ndarray, str, int]:
        state = RobotState29.make(
            q_sdk=q29,
            dq_sdk=dq29,
            quaternion_wxyz=quaternion_wxyz,
            angular_velocity_body=angular_velocity_body,
            monotonic_ns=1,
            mode_machine=5,
        )
        if not self._initialized:
            self.reset(np.asarray(q29, dtype=np.float32))
            self._initialized = True
        externally_synchronized = previous_lower_action is not None
        if externally_synchronized:
            previous = np.asarray(previous_lower_action, dtype=np.float32).reshape(-1)
            if previous.shape != (self.contract.action_dim,) or not np.all(np.isfinite(previous)):
                raise ValueError("previous action must contain %d finite values" % self.contract.action_dim)
            if float(np.max(np.abs(previous))) > self.contract.clip_actions + 1.0e-5:
                raise ValueError("previous lower action exceeds the exported action bound")
            self.previous_lower_action[:] = previous
        observation = self.observation_builder.build(
            state,
            command=self.command,
            previous_lower_action=self.previous_lower_action,
            locomotion_mode=self.locomotion_mode,
        )
        history = self.history.append(observation.measurable_current)
        inference = self.backend.infer_action(observation.policy_current, history)
        compiled = self.compiler.compile(inference.lower_action)
        self.last_lower_action[:] = compiled.normalized_lower_action
        # Synchronous/sim2sim callers retain the traditional internal actor
        # history.  The isolated worker receives the parent-accepted actor
        # history on every request, so it must not commit a late/unaccepted
        # response by itself.
        if not externally_synchronized:
            self.previous_lower_action[:] = compiled.normalized_lower_action
        # HOMIE's writer owns the complete 15-joint leg/spine target. The
        # manifest decides whether DreamWaQ owns 12 leg joints or all 15.
        return (
            compiled.q_target_sdk[:15].copy(),
            ("walk" if self.locomotion_mode else "stand")
            if self.contract.explicit_locomotion_mode
            else ("walk" if float(np.linalg.norm(self.command)) >= 1.0e-6 else "stand"),
            int(inference.latency_ns),
        )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
