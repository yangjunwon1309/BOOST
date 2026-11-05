"""Bounded fixed and Unitree-wireless velocity command sources."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .safety import SafetyFault


KEY_R1 = 0
KEY_L1 = 1
KEY_START = 2
KEY_SELECT = 3
KEY_A = 8
KEY_B = 9


def _deadband(value: float, threshold: float) -> float:
    value = float(value)
    threshold = float(threshold)
    if abs(value) <= threshold:
        return 0.0
    # Preserve full stick authority after the dead zone, exactly as HOMIE's
    # command source does.  A hard deadband otherwise creates a visible jump
    # at the first non-zero joystick sample.
    return float(np.sign(value) * (abs(value) - threshold) / (1.0 - threshold))


class FixedCommandSource:
    def __init__(self, command: Any = (0.0, 0.0, 0.0)) -> None:
        value = np.asarray(command, dtype=np.float32).reshape(-1)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("fixed velocity command must contain three finite values")
        self.command = value

    def __call__(self, _state: object = None) -> np.ndarray:
        return self.command.copy()


class WirelessCommandSource:
    """Decode the G1 wireless axes using HOMIE's command convention."""

    def __init__(
        self,
        *,
        max_vx: float = 0.6,
        max_vy: float = 0.4,
        max_yaw: float = 0.5,
        deadband: float = 0.1,
        require_deadman: bool = False,
    ) -> None:
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_yaw = float(max_yaw)
        self.deadband = float(deadband)
        self.require_deadman = bool(require_deadman)
        # A zero limit is intentional for a stationary real-transport check:
        # retain the wireless deadman/stop path but force that command axis to
        # zero.  Only negative or non-finite limits are unsafe/invalid.
        limits = np.asarray((self.max_vx, self.max_vy, self.max_yaw), dtype=np.float32)
        if (
            not np.all(np.isfinite(limits))
            or float(np.min(limits)) < 0.0
            or not np.isfinite(self.deadband)
            or not 0.0 <= self.deadband < 1.0
        ):
            raise ValueError("invalid wireless command limits")

    @staticmethod
    def buttons(payload: bytes) -> int:
        if len(payload) != 40:
            raise SafetyFault("wireless_remote is unavailable; real deadman cannot be proven")
        return int(struct.unpack("<H", payload[2:4])[0])

    @staticmethod
    def pressed(buttons: int, bit: int) -> bool:
        return bool(int(buttons) & (1 << int(bit)))

    def __call__(self, state: Any) -> np.ndarray:
        payload = bytes(getattr(state, "wireless_remote", b""))
        keys = self.buttons(payload)
        if self.pressed(keys, KEY_B) or self.pressed(keys, KEY_SELECT):
            raise SafetyFault("operator emergency/stop button is pressed")
        deadman = self.pressed(keys, KEY_L1) and self.pressed(keys, KEY_R1)
        if self.require_deadman and not deadman:
            return np.zeros(3, dtype=np.float32)
        # Unitree wireless_remote layout: lx, rx, ry, (gap), ly.
        lx = float(struct.unpack("<f", payload[4:8])[0])
        rx = float(struct.unpack("<f", payload[8:12])[0])
        ly = float(struct.unpack("<f", payload[20:24])[0])
        axes = np.asarray((lx, rx, ly), dtype=np.float32)
        if not np.all(np.isfinite(axes)) or float(np.max(np.abs(axes))) > 1.5:
            raise SafetyFault("wireless joystick contains invalid axes")
        vx = _deadband(ly, self.deadband) * self.max_vx
        vy = -_deadband(lx, self.deadband) * self.max_vy
        yaw = -_deadband(rx, self.deadband) * self.max_yaw
        return np.asarray((vx, vy, yaw), dtype=np.float32)
