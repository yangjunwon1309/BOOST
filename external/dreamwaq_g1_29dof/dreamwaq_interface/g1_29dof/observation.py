"""Exact FlashSAC DreamWaQ observation construction for G1 29-DoF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import DreamWaQ29Contract, JointOrderMap


def _finite_vector(values: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("%s must contain exactly %d finite values" % (label, size))
    return result


def _normalize_quaternion_wxyz(values: Any) -> np.ndarray:
    quaternion = _finite_vector(values, 4, "body-to-world quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-6 or abs(norm - 1.0) > 5.0e-2:
        raise ValueError("body-to-world quaternion is not near unit length")
    return quaternion / norm


def quat_apply_inverse_wxyz(quaternion_wxyz: Any, vector: Any) -> np.ndarray:
    """Apply the inverse of a body-to-world quaternion to one world vector."""

    q = _normalize_quaternion_wxyz(quaternion_wxyz)
    v = _finite_vector(vector, 3, "vector")
    w, xyz = float(q[0]), q[1:]
    return (
        v * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(xyz, v)
        + 2.0 * xyz * float(np.dot(xyz, v))
    ).astype(np.float32)


@dataclass(frozen=True)
class RobotState29:
    """One canonical Unitree state sample in SDK motor order."""

    q_sdk: np.ndarray
    dq_sdk: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity_body: np.ndarray
    monotonic_ns: int
    mode_machine: int
    source_tick: int = -1
    wireless_remote: bytes = b""
    reserved_q_raw: np.ndarray | None = None
    tau_est_sdk: np.ndarray | None = None
    temperature_sdk: np.ndarray | None = None

    @classmethod
    def make(
        cls,
        *,
        q_sdk: Any,
        dq_sdk: Any,
        quaternion_wxyz: Any,
        angular_velocity_body: Any,
        monotonic_ns: int,
        mode_machine: int,
        source_tick: int = -1,
        wireless_remote: Any = b"",
        reserved_q_raw: Any = None,
        tau_est_sdk: Any = None,
        temperature_sdk: Any = None,
    ) -> "RobotState29":
        stamp, mode = int(monotonic_ns), int(mode_machine)
        if stamp <= 0:
            raise ValueError("LowState monotonic timestamp must be positive")
        if not 0 <= mode <= 255:
            raise ValueError("mode_machine must be a uint8")
        remote = bytes(wireless_remote or b"")
        if remote and len(remote) != 40:
            raise ValueError("Unitree wireless_remote must be absent or exactly 40 bytes")
        return cls(
            q_sdk=_finite_vector(q_sdk, 29, "q_sdk").copy(),
            dq_sdk=_finite_vector(dq_sdk, 29, "dq_sdk").copy(),
            quaternion_wxyz=_normalize_quaternion_wxyz(quaternion_wxyz).copy(),
            angular_velocity_body=_finite_vector(angular_velocity_body, 3, "angular_velocity_body").copy(),
            monotonic_ns=stamp,
            mode_machine=mode,
            source_tick=int(source_tick),
            wireless_remote=remote,
            reserved_q_raw=None if reserved_q_raw is None else _finite_vector(reserved_q_raw, 6, "reserved_q_raw").copy(),
            tau_est_sdk=None if tau_est_sdk is None else _finite_vector(tau_est_sdk, 29, "tau_est_sdk").copy(),
            temperature_sdk=None if temperature_sdk is None else _finite_vector(temperature_sdk, 29, "temperature_sdk").copy(),
        )


@dataclass(frozen=True)
class FlashSACObservation:
    """All views of one measured state needed by the split FlashSAC actor."""

    raw_control: np.ndarray  # [ang, gravity, command, q, dq, previous_action] = 96
    policy_current: np.ndarray  # 99-D legacy frame, plus mode at index 99 for NEW3
    measurable_current: np.ndarray  # [ang, gravity, q, dq, action] = 93


class FlashSACHistoryBuffer:
    """Five-frame FlashSAC estimator history with term-major serialization."""

    def __init__(self, *, length: int = 5, feature_dim: int = 93) -> None:
        self.length, self.feature_dim = int(length), int(feature_dim)
        if (self.length, self.feature_dim) != (5, 93):
            raise ValueError("FlashSAC G1 CENet history is fixed at five 93-D frames")
        self._frames = np.zeros((self.length, self.feature_dim), dtype=np.float32)
        self._bootstrapped = False

    def reset(self) -> None:
        self._frames.fill(0.0)
        self._bootstrapped = False

    def append(self, measurable: Any) -> np.ndarray:
        frame = _finite_vector(measurable, 93, "FlashSAC measurable frame")
        if not self._bootstrapped:
            self._frames[:] = frame
            self._bootstrapped = True
        else:
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = frame
        return self.flat()

    def flat(self) -> np.ndarray:
        if not self._bootstrapped:
            raise RuntimeError("FlashSAC CENet history has not received its first frame")
        frames = self._frames
        return np.concatenate(
            (
                frames[:, :3].reshape(-1),
                frames[:, 3:6].reshape(-1),
                frames[:, 6:35].reshape(-1),
                frames[:, 35:64].reshape(-1),
                frames[:, 64:93].reshape(-1),
            )
        ).astype(np.float32, copy=False)

    @property
    def frames(self) -> np.ndarray:
        return self._frames.copy()


class DreamWaQObservationBuilder:
    """Build the FlashSAC raw, actor-current, and CENet-current observations."""

    def __init__(self, *, contract: DreamWaQ29Contract, mapper: JointOrderMap | None = None) -> None:
        self.contract = contract
        self.mapper = mapper or JointOrderMap()
        self._default_policy = self.mapper.sdk_to_policy(contract.default_q_sdk).astype(np.float32)
        self._previous_policy = np.zeros(29, dtype=np.float32)

    def reset(self) -> None:
        self._previous_policy.fill(0.0)

    @staticmethod
    def projected_gravity(state: RobotState29) -> np.ndarray:
        return quat_apply_inverse_wxyz(state.quaternion_wxyz, (0.0, 0.0, -1.0))

    def build(
        self, state: RobotState29, *, command: Any, previous_lower_action: Any,
        locomotion_mode: int = 0,
    ) -> FlashSACObservation:
        cmd = _finite_vector(command, 3, "velocity command")
        if self.contract.explicit_locomotion_mode:
            if locomotion_mode not in (0, 1):
                raise ValueError("locomotion mode must be standing=0 or walking=1")
            if locomotion_mode == 0:
                cmd = np.zeros(3, dtype=np.float32)
        lower = _finite_vector(previous_lower_action, self.contract.action_dim, "previous policy action")
        q_policy = self.mapper.sdk_to_policy(state.q_sdk).astype(np.float32)
        dq_policy = self.mapper.sdk_to_policy(state.dq_sdk).astype(np.float32)
        self._previous_policy.fill(0.0)
        self._previous_policy[self.mapper.policy_indices(self.contract.action_joint_names)] = lower
        raw = np.concatenate(
            (
                state.angular_velocity_body,
                self.projected_gravity(state),
                cmd,
                q_policy - self._default_policy,
                dq_policy * self.contract.joint_velocity_obs_scale,
                self._previous_policy,
            )
        ).astype(np.float32, copy=False)
        if raw.shape != (self.contract.raw_observation_dim,) or not np.all(np.isfinite(raw)):
            raise RuntimeError("FlashSAC raw control frame violates the sealed 96-D contract")
        raw = np.clip(raw, -self.contract.clip_observations, self.contract.clip_observations)
        policy = np.concatenate((np.zeros(3, dtype=np.float32), raw[:6], raw[9:], raw[6:9])).astype(np.float32)
        if self.contract.explicit_locomotion_mode:
            policy = np.concatenate((policy, np.asarray([locomotion_mode], dtype=np.float32)))
        measurable = np.concatenate((raw[:6], raw[9:])).astype(np.float32)
        if policy.shape != (self.contract.observation_dim,) or measurable.shape != (self.contract.measurable_observation_dim,):
            raise RuntimeError("FlashSAC observation split violates the sealed dimensions")
        return FlashSACObservation(raw, policy, measurable)


# The names are intentionally explicit in public imports; no GaitClock or
# generic frame-major HistoryBuffer remains in this FlashSAC-only package.
HistoryBuffer = FlashSACHistoryBuffer

__all__ = [
    "DreamWaQObservationBuilder", "FlashSACHistoryBuffer", "FlashSACObservation", "HistoryBuffer",
    "RobotState29", "quat_apply_inverse_wxyz",
]
