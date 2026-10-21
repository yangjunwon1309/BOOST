"""Binary LCM protocol between PATH on Thor and the G1 arm SDK bridge.

PATH sends only the 14 arm joints. The G1 bridge owns SDK publication and
robot-state telemetry; HOMIE/WBC remains an independent owner of legs/waist.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np


NUM_MOTORS = 29
ARM_START = 15
ARM_COUNT = 14

UPPER_COMMAND_CHANNEL = "G1_PATH_UPPER_COMMAND"
ROBOT_STATE_CHANNEL = "G1_ROBOT_STATE"

# ASCII "PDU1" in a little-endian uint32.
PATH_UPPER_COMMAND_MAGIC = 0x31554450
PATH_UPPER_COMMAND_VERSION = 1

FLAG_LEFT_GRIP = 1 << 1
FLAG_RIGHT_GRIP = 1 << 2

_UPPER_PACKET = struct.Struct("<IHHQq42f")
_STATE_PACKET = struct.Struct("<IHHQQqB107f")


def _float_vector(values: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def grip_flags(*, left_grip: bool, right_grip: bool) -> int:
    return (FLAG_LEFT_GRIP if left_grip else 0) | (FLAG_RIGHT_GRIP if right_grip else 0)


def pack_upper_command(
    *,
    sequence: int,
    timestamp_us: int,
    q_arm: Sequence[float] | np.ndarray,
    dq_arm: Sequence[float] | np.ndarray,
    tau_arm: Sequence[float] | np.ndarray,
    left_grip: bool = False,
    right_grip: bool = False,
) -> bytes:
    q = _float_vector(q_arm, ARM_COUNT, "q_arm")
    dq = _float_vector(dq_arm, ARM_COUNT, "dq_arm")
    tau = _float_vector(tau_arm, ARM_COUNT, "tau_arm")
    return _UPPER_PACKET.pack(
        PATH_UPPER_COMMAND_MAGIC,
        PATH_UPPER_COMMAND_VERSION,
        grip_flags(left_grip=left_grip, right_grip=right_grip),
        int(sequence),
        int(timestamp_us),
        *(q.tolist() + dq.tolist() + tau.tolist()),
    )


@dataclass(frozen=True)
class RobotState:
    sequence: int
    timestamp_us: int
    base_gyro: np.ndarray
    base_accel: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    base_quat_wxyz: np.ndarray


@dataclass(frozen=True)
class UpperCommand:
    sequence: int
    timestamp_us: int
    q_arm: np.ndarray
    dq_arm: np.ndarray
    tau_arm: np.ndarray
    left_grip: bool
    right_grip: bool


def unpack_upper_command(packet: bytes) -> UpperCommand:
    """Decode PATH's 14-joint PDU1 command packet."""
    if len(packet) != _UPPER_PACKET.size:
        raise ValueError(f"upper command packet size {len(packet)} != {_UPPER_PACKET.size}")
    values = _UPPER_PACKET.unpack(packet)
    magic, version, flags = values[:3]
    if magic != PATH_UPPER_COMMAND_MAGIC or version != PATH_UPPER_COMMAND_VERSION:
        raise ValueError(
            f"unexpected upper command header magic=0x{magic:08x} version={version}"
        )
    sequence = int(values[3])
    timestamp_us = int(values[4])
    arms = np.asarray(values[5:], dtype=np.float32)
    if not np.isfinite(arms).all():
        raise ValueError("upper command contains non-finite values")
    return UpperCommand(
        sequence=sequence,
        timestamp_us=timestamp_us,
        q_arm=arms[:ARM_COUNT].copy(),
        dq_arm=arms[ARM_COUNT : 2 * ARM_COUNT].copy(),
        tau_arm=arms[2 * ARM_COUNT :].copy(),
        left_grip=bool(flags & FLAG_LEFT_GRIP),
        right_grip=bool(flags & FLAG_RIGHT_GRIP),
    )


def unpack_robot_state(packet: bytes) -> RobotState:
    if len(packet) != _STATE_PACKET.size:
        raise ValueError(f"robot state packet size {len(packet)} != {_STATE_PACKET.size}")
    values = _STATE_PACKET.unpack(packet)
    magic, version = values[:2]
    if magic != 0x31545347 or version != 1:  # GST1
        raise ValueError(f"unexpected robot state header magic=0x{magic:08x} version={version}")
    sequence = int(values[3])
    timestamp_us = int(values[5])
    floats = np.asarray(values[7:], dtype=np.float64)
    return RobotState(
        sequence=sequence,
        timestamp_us=timestamp_us,
        base_quat_wxyz=floats[0:4].copy(),
        base_gyro=floats[4:7].copy(),
        base_accel=floats[7:10].copy(),
        q=floats[20:49].copy(),
        dq=floats[49:78].copy(),
        tau_est=floats[78:107].copy(),
    )
