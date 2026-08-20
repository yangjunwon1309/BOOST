"""Wire contract for the direct, two-wrist D405 v11 perception path.

This module deliberately has no dependency on the head-D435i protocol.  A
dispatch is atomic: it always contains the newest independently captured frame
from both configured wrists, never a synthetic replacement for a missing side.
"""

from __future__ import annotations

import math
import time
import zlib
from typing import Any

import msgpack
import numpy as np


VERSION = 1
CAMERA_IDS = ("left_wrist", "right_wrist")
WRIST_BUNDLE_CHANNEL = "PATH_V11_WRIST_BUNDLE"
WRIST_RESULT_CHANNEL = "PATH_V11_WRIST_PERCEPTION_RESULT"
WRIST_HEARTBEAT_CHANNEL = "PATH_V11_WRIST_PERCEPTION_HEARTBEAT"
WRIST_ROBOT_STATE_CHANNEL = "PATH_V11_WRIST_ROBOT_STATE"
WRIST_CONCEPT_CONTROL_CHANNEL = "PATH_V11_WRIST_CONCEPT_CONTROL"


def now_us() -> int:
    return time.time_ns() // 1_000


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pack_depth_z16(depth_z16: np.ndarray) -> bytes:
    depth = np.asarray(depth_z16, dtype=np.uint16)
    if depth.ndim != 2:
        raise ValueError("D405 depth must be a HxW Z16 array")
    return zlib.compress(depth.tobytes(order="C"), level=1)


def unpack_depth_z16(data: bytes, *, width: int, height: int) -> np.ndarray:
    if min(int(width), int(height)) <= 0:
        raise ValueError("invalid D405 depth dimensions")
    expected = int(width) * int(height) * 2
    raw = zlib.decompress(data, bufsize=expected)
    if len(raw) != expected:
        raise ValueError(f"D405 depth expands to {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint16).reshape(int(height), int(width)).copy()


def pack_mask(mask: np.ndarray) -> bytes:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("D405 local mask must be HxW")
    return zlib.compress(np.packbits(array.reshape(-1), bitorder="little").tobytes(), level=1)


def _validate_intrinsics(value: Any, *, width: int, height: int) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise ValueError("D405 frame is missing intrinsics")
    result: dict[str, float | int] = {}
    for key in ("width", "height"):
        numeric = int(value.get(key, -1))
        if numeric != {"width": width, "height": height}[key]:
            raise ValueError("D405 intrinsics do not match the native frame grid")
        result[key] = numeric
    for key in ("fx", "fy", "ppx", "ppy"):
        numeric = value.get(key)
        if not _finite(numeric):
            raise ValueError(f"D405 intrinsic {key} must be finite")
        result[key] = float(numeric)
    return result


def validate_bundle(payload: dict[str, Any], *, require_two: bool = True) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist D405 bundle version")
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("wrist D405 bundle is missing cameras")
    ids = set(cameras)
    allowed = set(CAMERA_IDS)
    if not ids.issubset(allowed) or (require_two and ids != allowed):
        raise ValueError("wrist D405 bundle must contain exactly left_wrist and right_wrist")
    if not ids:
        raise ValueError("wrist D405 bundle contains no cameras")
    serials: set[str] = set()
    for camera_id in CAMERA_IDS:
        if camera_id not in cameras:
            continue
        frame = cameras[camera_id]
        if not isinstance(frame, dict) or frame.get("camera_id") != camera_id:
            raise ValueError(f"invalid {camera_id} frame identity")
        serial = str(frame.get("serial", "")).strip()
        width, height = int(frame.get("width", 0)), int(frame.get("height", 0))
        if not serial or serial in serials or min(width, height) <= 0:
            raise ValueError(f"invalid {camera_id} serial or dimensions")
        serials.add(serial)
        if not isinstance(frame.get("color_jpeg"), (bytes, bytearray)):
            raise ValueError(f"{camera_id} has no JPEG color payload")
        if not isinstance(frame.get("depth_z16_zlib"), (bytes, bytearray)):
            raise ValueError(f"{camera_id} has no Z16 depth payload")
        if not _finite(frame.get("depth_scale_m")) or float(frame["depth_scale_m"]) <= 0.0:
            raise ValueError(f"{camera_id} has invalid depth scale")
        _validate_intrinsics(frame.get("intrinsics"), width=width, height=height)
        for key in ("host_timestamp_us", "color_timestamp_us", "depth_timestamp_us"):
            if int(frame.get(key, 0)) <= 0:
                raise ValueError(f"{camera_id} has invalid {key}")
    return payload


def pack_bundle(payload: dict[str, Any]) -> bytes:
    validate_bundle(payload)
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_bundle(data: bytes, *, require_two: bool = True) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    return validate_bundle(payload, require_two=require_two)


def make_result(*, sequence: int, source_bundle_timestamp_us: int, objects: list[dict[str, Any]],
                discarded_seam_masks: int, valid: bool = True, error: str | None = None) -> dict[str, Any]:
    return {
        "version": VERSION,
        "sequence": int(sequence),
        "published_timestamp_us": now_us(),
        "source_bundle_timestamp_us": int(source_bundle_timestamp_us),
        "valid": bool(valid),
        "error": error,
        "discarded_seam_masks": int(discarded_seam_masks),
        "objects": list(objects),
    }


def pack_result(payload: dict[str, Any]) -> bytes:
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist D405 result version")
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_result(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist D405 result version")
    return payload
