"""Small wire format for Thor-relayed, image-only wrist UVC frames."""

from __future__ import annotations

import time
from typing import Any

import msgpack


WRIST_RGB_FRAME_CHANNEL = "PATH_V11_WRIST_RGB_FRAME"
RGB_FRAME_VERSION = 2
SUPPORTED_RGB_FRAME_VERSIONS = frozenset((1, RGB_FRAME_VERSION))
CAMERA_IDS = {"left_wrist", "right_wrist"}


def now_us() -> int:
    return time.monotonic_ns() // 1_000


def pack_rgb_frame(
    *,
    camera_id: str,
    width: int,
    height: int,
    thor_receive_timestamp_us: int,
    jpeg: bytes,
    camera_sequence: int | None = None,
    g1_capture_wall_time_us: int | None = None,
    g1_capture_monotonic_time_ns: int | None = None,
) -> bytes:
    if camera_id not in CAMERA_IDS:
        raise ValueError(f"unsupported wrist camera id: {camera_id}")
    if min(int(width), int(height), int(thor_receive_timestamp_us)) <= 0:
        raise ValueError("RGB frame dimensions and receive timestamp must be positive")
    if not isinstance(jpeg, (bytes, bytearray)) or len(jpeg) < 4:
        raise ValueError("RGB frame must contain JPEG bytes")
    jpeg = bytes(jpeg)
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("RGB frame payload is not a complete JPEG")
    metadata = (
        camera_sequence,
        g1_capture_wall_time_us,
        g1_capture_monotonic_time_ns,
    )
    if any(value is not None for value in metadata) and not all(
        value is not None for value in metadata
    ):
        raise ValueError("TIM2 sequence and both capture timestamps must be supplied together")

    payload: dict[str, Any] = {
        "version": RGB_FRAME_VERSION if all(value is not None for value in metadata) else 1,
        "camera_id": camera_id,
        "width": int(width),
        "height": int(height),
        "thor_receive_timestamp_us": int(thor_receive_timestamp_us),
        "timestamp_source": "g1_tim2_capture" if metadata[0] is not None else "thor_zmq_receive",
        "jpeg": jpeg,
    }
    if metadata[0] is not None:
        sequence, capture_wall_us, capture_monotonic_ns = (int(value) for value in metadata)
        if min(sequence, capture_wall_us, capture_monotonic_ns) < 0:
            raise ValueError("TIM2 sequence and capture timestamps must be non-negative")
        payload.update(
            {
                "camera_sequence": sequence,
                "g1_capture_wall_time_us": capture_wall_us,
                "g1_capture_monotonic_time_ns": capture_monotonic_ns,
                "capture_clock_host": "g1",
                "thor_receive_clock_host": "thor",
            }
        )
    return msgpack.packb(payload, use_bin_type=True)


def unpack_rgb_frame(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict) or payload.get("version") not in SUPPORTED_RGB_FRAME_VERSIONS:
        raise ValueError("unsupported wrist RGB frame version")
    camera_id = payload.get("camera_id")
    if camera_id not in CAMERA_IDS:
        raise ValueError(f"unsupported wrist camera id: {camera_id}")
    if min(int(payload.get("width", 0)), int(payload.get("height", 0))) <= 0:
        raise ValueError("invalid wrist RGB frame dimensions")
    if int(payload.get("thor_receive_timestamp_us", 0)) <= 0:
        raise ValueError("invalid Thor receive timestamp")
    jpeg = payload.get("jpeg")
    if not isinstance(jpeg, (bytes, bytearray)):
        raise ValueError("invalid wrist RGB JPEG payload")
    jpeg = bytes(jpeg)
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("invalid wrist RGB JPEG payload")
    if payload["version"] == RGB_FRAME_VERSION:
        for field in (
            "camera_sequence",
            "g1_capture_wall_time_us",
            "g1_capture_monotonic_time_ns",
        ):
            value = payload.get(field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid TIM2 metadata field: {field}")
        if payload.get("capture_clock_host") != "g1" or payload.get(
            "thor_receive_clock_host"
        ) != "thor":
            raise ValueError("invalid wrist RGB timestamp clock-domain metadata")
    return payload
