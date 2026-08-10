"""Binary LCM packets for PATH's SONIC-free direct motor path.

The upper-command packet is intentionally smaller than the final 29-DoF
LowCmd packet.  Thor owns only the two IPOPT arm solutions; the process on the
G1 owns lower-body/waist damping today and will later mux a local lower-body
controller before publishing the existing direct-motor V2 packet.
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
MOTOR_COMMAND_CHANNEL = "G1_PATH_MOTOR_COMMAND"
ROBOT_STATE_CHANNEL = "G1_ROBOT_STATE"

# ASCII "PDU1" in a little-endian uint32.  This is distinct from the legacy
# GCA1 SONIC action packet and cannot be consumed by it accidentally.
PATH_UPPER_COMMAND_MAGIC = 0x31554450
PATH_UPPER_COMMAND_VERSION = 1
DIRECT_COMMAND_MAGIC = 0x31414347  # GCA1, shared with g1_lowlevel_bridge V2.
DIRECT_COMMAND_VERSION = 2

FLAG_LEFT_GRIP = 1 << 1
FLAG_RIGHT_GRIP = 1 << 2

_UPPER_PACKET = struct.Struct("<IHHQq42f")
_DIRECT_PACKET = struct.Struct("<IHHQq87f")
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


def pack_direct_motor_command_v2(
    *,
    sequence: int,
    timestamp_us: int,
    q_target: Sequence[float] | np.ndarray,
    dq_target: Sequence[float] | np.ndarray,
    tau_ff: Sequence[float] | np.ndarray,
    left_grip: bool = False,
    right_grip: bool = False,
) -> bytes:
    q = _float_vector(q_target, NUM_MOTORS, "q_target")
    dq = _float_vector(dq_target, NUM_MOTORS, "dq_target")
    tau = _float_vector(tau_ff, NUM_MOTORS, "tau_ff")
    return _DIRECT_PACKET.pack(
        DIRECT_COMMAND_MAGIC,
        DIRECT_COMMAND_VERSION,
        grip_flags(left_grip=left_grip, right_grip=right_grip),
        int(sequence),
        int(timestamp_us),
        *(q.tolist() + dq.tolist() + tau.tolist()),
    )


@dataclass(frozen=True)
class RobotState:
    sequence: int
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    base_quat_wxyz: np.ndarray


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
        q=floats[20:49].copy(),
        dq=floats[49:78].copy(),
        tau_est=floats[78:107].copy(),
    )
