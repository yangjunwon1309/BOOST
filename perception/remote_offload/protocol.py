"""Versioned MessagePack protocol for remote perception results.

This protocol is intentionally independent of the browser/webview. LCM carries
these payloads over Wi-Fi; the web UI is diagnostic only.
"""

from __future__ import annotations

import math
from typing import Any

import msgpack


PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 4 * 1024 * 1024
MAX_OBJECTS = 64
MAX_IMAGE_DIMENSION = 4096
MASK_ENCODING = "bitpack-zlib-row-major"


def _finite_vector(value: Any, size: int, field: str, *, nullable: bool = True) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must be a {size}-element list or null")
    if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value):
        raise ValueError(f"{field} contains non-finite values")


def validate_perception_result(payload: Any) -> dict[str, Any]:
    """Validate and return a decoded v1 remote-perception result."""
    if not isinstance(payload, dict):
        raise ValueError("result must be a map")
    required = {
        "version", "sequence", "published_timestamp_us", "source_rgb_timestamp_us",
        "source_depth_timestamp_us", "calibration_id", "image_width", "image_height",
        "backend", "inference_ms", "valid", "error", "objects",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"missing result fields: {sorted(missing)}")
    if payload["version"] != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {payload['version']}")
    for name in ("sequence", "published_timestamp_us", "source_rgb_timestamp_us", "source_depth_timestamp_us"):
        if not isinstance(payload[name], int) or payload[name] < 0:
            raise ValueError(f"invalid {name}")
    calibration_id = payload["calibration_id"]
    if calibration_id is not None and (not isinstance(calibration_id, int) or calibration_id < 0):
        raise ValueError("invalid calibration_id")
    width, height = payload["image_width"], payload["image_height"]
    if not isinstance(width, int) or not isinstance(height, int) or not (0 < width <= MAX_IMAGE_DIMENSION and 0 < height <= MAX_IMAGE_DIMENSION):
        raise ValueError("invalid image dimensions")
    if not isinstance(payload["backend"], str) or not payload["backend"]:
        raise ValueError("invalid backend")
    if not isinstance(payload["inference_ms"], (int, float)) or not math.isfinite(float(payload["inference_ms"])):
        raise ValueError("invalid inference_ms")
    if not isinstance(payload["valid"], bool) or (payload["error"] is not None and not isinstance(payload["error"], str)):
        raise ValueError("invalid valid/error fields")
    objects = payload["objects"]
    if not isinstance(objects, list) or len(objects) > MAX_OBJECTS:
        raise ValueError("invalid object list")
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"object {index} is not a map")
        for field in ("track_id", "label", "score", "box_xyxy", "mask_encoding", "mask_data", "mask_height", "mask_width", "median_depth_m", "center_pixel_uv", "center_camera_m", "center_goal_m", "valid_depth_points"):
            if field not in item:
                raise ValueError(f"object {index} missing {field}")
        if not isinstance(item["track_id"], str) or not isinstance(item["label"], str):
            raise ValueError(f"object {index} has invalid identity")
        if not isinstance(item["score"], (int, float)) or not math.isfinite(float(item["score"])):
            raise ValueError(f"object {index} has invalid score")
        _finite_vector(item["box_xyxy"], 4, f"object {index}.box_xyxy", nullable=False)
        if item["mask_encoding"] != MASK_ENCODING or not isinstance(item["mask_data"], bytes):
            raise ValueError(f"object {index} has invalid mask encoding")
        if len(item["mask_data"]) > MAX_PACKET_BYTES:
            raise ValueError(f"object {index} mask is too large")
        if item["mask_height"] != height or item["mask_width"] != width:
            raise ValueError(f"object {index} mask dimensions do not match image")
        if item["median_depth_m"] is not None and (not isinstance(item["median_depth_m"], (int, float)) or not math.isfinite(float(item["median_depth_m"]))):
            raise ValueError(f"object {index} has invalid median depth")
        _finite_vector(item["center_pixel_uv"], 2, f"object {index}.center_pixel_uv")
        _finite_vector(item["center_camera_m"], 3, f"object {index}.center_camera_m")
        _finite_vector(item["center_goal_m"], 3, f"object {index}.center_goal_m")
        if not isinstance(item["valid_depth_points"], int) or item["valid_depth_points"] < 0:
            raise ValueError(f"object {index} has invalid valid_depth_points")
    return payload


def encode_perception_result(payload: dict[str, Any]) -> bytes:
    validate_perception_result(payload)
    encoded = msgpack.packb(payload, use_bin_type=True)
    if len(encoded) > MAX_PACKET_BYTES:
        raise ValueError("perception result packet is too large")
    return encoded


def decode_perception_result(data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes) or len(data) == 0 or len(data) > MAX_PACKET_BYTES:
        raise ValueError("invalid perception result packet length")
    return validate_perception_result(msgpack.unpackb(data, raw=False, strict_map_key=False))
