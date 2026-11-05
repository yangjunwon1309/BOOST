"""Pure-Python cross-check for Unitree HG LowCmd/LowState CRC32."""

from __future__ import annotations

import struct
from typing import Any, Iterable


class UnitreeCRCError(RuntimeError):
    pass


def _field(value: Any) -> Any:
    return value() if callable(value) else value


def _get(obj: Any, name: str) -> Any:
    try:
        return _field(getattr(obj, name))
    except AttributeError as exc:
        raise UnitreeCRCError("Unitree HG message lacks field %s" % name) from exc


def _seq(value: Any, name: str, count: int) -> tuple[Any, ...]:
    try:
        result = tuple(value)
    except (TypeError, ValueError) as exc:
        raise UnitreeCRCError("Unitree HG field %s is not a sequence" % name) from exc
    if len(result) != count:
        raise UnitreeCRCError("Unitree HG field %s has %d values; expected %d" % (name, len(result), count))
    return result


def crc32_words(words: Iterable[int]) -> int:
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for word in words:
        current = int(word) & 0xFFFFFFFF
        bit = 1 << 31
        for _ in range(32):
            crc = (((crc << 1) & 0xFFFFFFFF) ^ polynomial) if crc & 0x80000000 else ((crc << 1) & 0xFFFFFFFF)
            if current & bit:
                crc ^= polynomial
            bit >>= 1
    return crc & 0xFFFFFFFF


def _words(payload: bytes) -> list[int]:
    if len(payload) % 4 != 0:
        raise UnitreeCRCError("packed Unitree HG message is not four-byte aligned")
    return [value[0] for value in struct.iter_unpack("<I", payload[:-4])]


def _pack_low_cmd(message: Any) -> bytes:
    payload = bytearray(struct.pack("<2B2x", int(_get(message, "mode_pr")), int(_get(message, "mode_machine"))))
    for motor in _seq(_get(message, "motor_cmd"), "motor_cmd", 35):
        payload.extend(
            struct.pack(
                "<B3x5fI",
                int(_get(motor, "mode")),
                float(_get(motor, "q")),
                float(_get(motor, "dq")),
                float(_get(motor, "tau")),
                float(_get(motor, "kp")),
                float(_get(motor, "kd")),
                int(_get(motor, "reserve")),
            )
        )
    payload.extend(struct.pack("<4I", *_seq(_get(message, "reserve"), "reserve", 4)))
    payload.extend(struct.pack("<I", int(_get(message, "crc"))))
    if len(payload) != 1004:
        raise UnitreeCRCError("packed HG LowCmd has %d bytes; expected 1004" % len(payload))
    return bytes(payload)


def _pack_low_state(message: Any) -> bytes:
    version = _seq(_get(message, "version"), "version", 2)
    imu = _get(message, "imu_state")
    payload = bytearray(
        struct.pack(
            "<2I2B2xI13fh2x",
            int(version[0]), int(version[1]),
            int(_get(message, "mode_pr")), int(_get(message, "mode_machine")),
            int(_get(message, "tick")),
            *[float(value) for value in _seq(_get(imu, "quaternion"), "quaternion", 4)],
            *[float(value) for value in _seq(_get(imu, "gyroscope"), "gyroscope", 3)],
            *[float(value) for value in _seq(_get(imu, "accelerometer"), "accelerometer", 3)],
            *[float(value) for value in _seq(_get(imu, "rpy"), "rpy", 3)],
            int(_get(imu, "temperature")),
        )
    )
    for motor in _seq(_get(message, "motor_state"), "motor_state", 35):
        payload.extend(
            struct.pack(
                "<B3x4f2hf7I",
                int(_get(motor, "mode")),
                float(_get(motor, "q")),
                float(_get(motor, "dq")),
                float(_get(motor, "ddq")),
                float(_get(motor, "tau_est")),
                *[int(value) for value in _seq(_get(motor, "temperature"), "temperature", 2)],
                float(_get(motor, "vol")),
                *[int(value) for value in _seq(_get(motor, "sensor"), "sensor", 2)],
                int(_get(motor, "motorstate")),
                *[int(value) for value in _seq(_get(motor, "reserve"), "reserve", 4)],
            )
        )
    payload.extend(struct.pack("<40B", *_seq(_get(message, "wireless_remote"), "wireless_remote", 40)))
    payload.extend(struct.pack("<4I", *_seq(_get(message, "reserve"), "reserve", 4)))
    payload.extend(struct.pack("<I", int(_get(message, "crc"))))
    if len(payload) != 2092:
        raise UnitreeCRCError("packed HG LowState has %d bytes; expected 2092" % len(payload))
    return bytes(payload)


class PurePythonHGCRC:
    def Crc(self, message: Any) -> int:
        if hasattr(message, "motor_cmd"):
            payload = _pack_low_cmd(message)
        elif hasattr(message, "motor_state"):
            payload = _pack_low_state(message)
        else:
            raise UnitreeCRCError("unknown Unitree HG message type")
        return crc32_words(_words(payload))
