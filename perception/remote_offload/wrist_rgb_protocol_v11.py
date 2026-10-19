"""Small wire format for Thor-relayed, image-only wrist UVC frames."""

from __future__ import annotations

import time
from typing import Any

import msgpack


WRIST_RGB_FRAME_CHANNEL = "PATH_V11_WRIST_RGB_FRAME"
RGB_FRAME_VERSION = 1
CAMERA_IDS = {"left_wrist", "right_wrist"}


def now_us() -> int:
    return time.monotonic_ns() // 1_000


def pack_rgb_frame(
    *, camera_id: str, width: int, height: int, thor_receive_timestamp_us: int, jpeg: bytes
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
    return msgpack.packb(
        {
            "version": RGB_FRAME_VERSION,
            "camera_id": camera_id,
            "width": int(width),
            "height": int(height),
            "thor_receive_timestamp_us": int(thor_receive_timestamp_us),
            "timestamp_source": "thor_zmq_receive",
            "jpeg": jpeg,
        },
        use_bin_type=True,
    )


def unpack_rgb_frame(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict) or payload.get("version") != RGB_FRAME_VERSION:
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
    return payload
