"""Versioned binary frames for lossless teleoperation recording."""

from __future__ import annotations

import struct
from dataclasses import dataclass


MAGIC = b"TIM2"
VERSION = 1
_HEADER = struct.Struct("<4sHHQQQIII")
HEADER_SIZE = _HEADER.size
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class RecordingFrame:
    sequence: int
    capture_wall_time_us: int
    capture_monotonic_time_ns: int
    width: int
    height: int
    jpeg: bytes


def _validate_frame(frame: RecordingFrame) -> None:
    for name in ("sequence", "capture_wall_time_us", "capture_monotonic_time_ns"):
        value = getattr(frame, name)
        if not 0 <= value <= _UINT64_MAX:
            raise ValueError(f"{name} is outside uint64 range")
    if not 0 < frame.width <= _UINT32_MAX or not 0 < frame.height <= _UINT32_MAX:
        raise ValueError("image dimensions must be positive uint32 values")
    if not isinstance(frame.jpeg, bytes):
        raise TypeError("JPEG payload must be bytes")
    if len(frame.jpeg) > _UINT32_MAX:
        raise ValueError("JPEG payload is too large")
    if len(frame.jpeg) < 4 or not frame.jpeg.startswith(b"\xff\xd8") or not frame.jpeg.endswith(b"\xff\xd9"):
        raise ValueError("JPEG payload has invalid SOI/EOI markers")


def encode_recording_frame(frame: RecordingFrame) -> bytes:
    """Encode one capture and its metadata into a self-contained message."""
    _validate_frame(frame)
    return _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        frame.sequence,
        frame.capture_wall_time_us,
        frame.capture_monotonic_time_ns,
        frame.width,
        frame.height,
        len(frame.jpeg),
    ) + frame.jpeg


def decode_recording_frame(packet: bytes | bytearray | memoryview) -> RecordingFrame:
    """Decode and validate a recording message."""
    if len(packet) < HEADER_SIZE:
        raise ValueError("recording packet is smaller than its header")

    magic, version, _flags, sequence, wall_us, monotonic_ns, width, height, jpeg_size = _HEADER.unpack_from(packet)
    if magic != MAGIC:
        raise ValueError("recording packet magic mismatch")
    if version != VERSION:
        raise ValueError(f"unsupported recording packet version: {version}")
    if width == 0 or height == 0:
        raise ValueError("image dimensions must be positive")
    if len(packet) != HEADER_SIZE + jpeg_size:
        raise ValueError("recording packet JPEG length mismatch")

    frame = RecordingFrame(
        sequence=sequence,
        capture_wall_time_us=wall_us,
        capture_monotonic_time_ns=monotonic_ns,
        width=width,
        height=height,
        jpeg=bytes(packet[HEADER_SIZE:]),
    )
    _validate_frame(frame)
    return frame
