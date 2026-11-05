"""Lower-body action ownership and safe 29-joint target compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import (
    DreamWaQ29Contract,
    JOINT_LOWER_SDK,
    JOINT_UPPER_SDK,
    POLICY_JOINT_NAMES,
    SDK_JOINT_NAMES,
    JointOrderMap,
)


@dataclass(frozen=True)
class CompiledLowerBodyAction:
    normalized_lower_action: np.ndarray
    q_target_sdk: np.ndarray
    previous_action_policy: np.ndarray
    clipped_joint_names: tuple[str, ...]


class LowerBodyActionCompiler:
    """Expand the exported leg[/waist] action into an auditable 29-DoF target."""

    def __init__(
        self,
        *,
        contract: DreamWaQ29Contract,
        joint_limit_margin_rad: float = 0.0,
        clip_joint_targets: bool = True,
        mapper: JointOrderMap | None = None,
    ) -> None:
        self.contract = contract
        self.physical_action_scale = np.asarray(contract.physical_action_scale, dtype=np.float32).reshape(-1)
        self.clip_actions = float(contract.clip_actions)
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        self.clip_joint_targets = bool(clip_joint_targets)
        self.mapper = mapper or JointOrderMap()
        self.action_sdk_indices = self.mapper.sdk_indices(contract.action_joint_names)
        self.action_policy_indices = self.mapper.policy_indices(contract.action_joint_names)
        if self.physical_action_scale.shape != (contract.action_dim,) or np.any(self.physical_action_scale <= 0.0):
            raise ValueError("FlashSAC physical action scale does not match the action contract")
        if self.clip_actions <= 0.0:
            raise ValueError("action clip must be positive")
        if self.joint_limit_margin_rad < 0.0:
            raise ValueError("joint-limit margin must be non-negative")
        if np.any(JOINT_UPPER_SDK - JOINT_LOWER_SDK <= 2.0 * self.joint_limit_margin_rad):
            raise ValueError("joint-limit margin consumes a complete joint interval")

    def compile(self, lower_action: Any) -> CompiledLowerBodyAction:
        raw = np.asarray(lower_action, dtype=np.float32).reshape(-1)
        if raw.shape != (self.contract.action_dim,) or not np.all(np.isfinite(raw)):
            raise ValueError("policy action must contain %d finite values" % self.contract.action_dim)
        if float(np.max(np.abs(raw))) > self.clip_actions + 1.0e-5:
            raise ValueError("tanh lower-body policy action exceeds the exported action bound")
        bounded = np.clip(raw, -self.clip_actions, self.clip_actions)
        desired = self.contract.default_q_sdk.copy()
        desired[self.action_sdk_indices] += self.physical_action_scale * bounded

        if self.clip_joint_targets:
            inward_lower = JOINT_LOWER_SDK + self.joint_limit_margin_rad
            inward_upper = JOINT_UPPER_SDK - self.joint_limit_margin_rad
            target = np.clip(desired, inward_lower, inward_upper).astype(np.float32)
            clipped = tuple(
                name
                for name, before, after in zip(SDK_JOINT_NAMES, desired, target)
                if abs(float(before) - float(after)) > 1.0e-8
            )
        else:
            target = desired.astype(np.float32, copy=True)
            clipped = ()

        previous_policy = np.zeros(29, dtype=np.float32)
        previous_policy[self.action_policy_indices] = bounded

        # The policy must never acquire the spine or arm command channels.
        lower_set = set(self.contract.action_joint_names)
        for sdk_index, name in enumerate(SDK_JOINT_NAMES):
            if name not in lower_set and abs(float(target[sdk_index] - self.contract.default_q_sdk[sdk_index])) > 1.0e-6:
                raise RuntimeError("non-lower joint target escaped its dedicated hold owner: %s" % name)
        for policy_index, name in enumerate(POLICY_JOINT_NAMES):
            if name not in lower_set and previous_policy[policy_index] != 0.0:
                raise RuntimeError("upper/spine target leaked into policy action history: %s" % name)

        return CompiledLowerBodyAction(
            normalized_lower_action=bounded.astype(np.float32),
            q_target_sdk=target,
            previous_action_policy=previous_policy,
            clipped_joint_names=clipped,
        )
