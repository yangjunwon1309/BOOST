"""Versioned MessagePack protocol for remote perception results."""

from __future__ import annotations

import dataclasses
import math
import time
import zlib
from typing import Any

import msgpack
import numpy as np


VERSION = 1

LOCAL_RGB_CHANNEL = "G1_CAMERA_FRAME"
LOCAL_DEPTH_CHANNEL = "G1_DEPTH_FRAME"
LOCAL_CALIBRATION_CHANNEL = "G1_CAMERA_CALIBRATION"
LOCAL_ROBOT_STATE_CHANNEL = "G1_ROBOT_STATE"

REMOTE_RGB_CHANNEL = "PATH_REMOTE_RGB_FRAME"
REMOTE_DEPTH_CHANNEL = "PATH_REMOTE_DEPTH_FRAME"
REMOTE_CALIBRATION_CHANNEL = "PATH_REMOTE_CAMERA_CALIBRATION"
REMOTE_ROBOT_STATE_CHANNEL = "PATH_REMOTE_ROBOT_STATE"
REMOTE_RESULT_CHANNEL = "PATH_REMOTE_PERCEPTION_RESULT"
REMOTE_HEARTBEAT_CHANNEL = "PATH_REMOTE_PERCEPTION_HEARTBEAT"
REMOTE_OVERLAY_CHANNEL = "PATH_REMOTE_OVERLAY_JPEG"

MASK_ENCODING = "bitpack-zlib-row-major"
BACKENDS = {"sam3_pytorch", "sam3_tensorrt", "yolo11_pytorch", "mock"}


def now_us() -> int:
    return time.time_ns() // 1000


def encode_mask(mask: np.ndarray) -> bytes:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must be HxW")
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return zlib.compress(packed.tobytes(), level=1)


def decode_mask(data: bytes, height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("invalid mask dimensions")
    expected_bits = int(height) * int(width)
    expected_bytes = (expected_bits + 7) // 8
    raw = zlib.decompress(data, bufsize=expected_bytes)
    if len(raw) != expected_bytes:
        raise ValueError(f"mask expands to {len(raw)} bytes, expected {expected_bytes}")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    return bits[:expected_bits].reshape(int(height), int(width)).astype(bool)


def pack_result(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_result(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict):
        raise ValueError("remote result payload is not a map")
    return payload


def pack_heartbeat(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_heartbeat(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict):
        raise ValueError("remote heartbeat payload is not a map")
    return payload


def _finite_number(value: Any, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_vector(value: Any, length: int, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    return all(_finite_number(item) for item in value)


def make_empty_result(
    *,
    sequence: int,
    source_rgb_timestamp_us: int,
    source_depth_timestamp_us: int,
    image_width: int,
    image_height: int,
    backend: str = "sam3_pytorch",
    calibration_id: int | None = None,
    inference_ms: float = 0.0,
    valid: bool = True,
    error: str | None = None,
    published_timestamp_us: int | None = None,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "sequence": int(sequence),
        "published_timestamp_us": int(published_timestamp_us or now_us()),
        "source_rgb_timestamp_us": int(source_rgb_timestamp_us),
        "source_depth_timestamp_us": int(source_depth_timestamp_us),
        "calibration_id": None if calibration_id is None else int(calibration_id),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "backend": str(backend),
        "inference_ms": float(inference_ms),
        "valid": bool(valid),
        "error": error,
        "objects": [],
    }


def make_object(
    *,
    track_id: str,
    label: str,
    score: float,
    box_xyxy: list[float] | tuple[float, float, float, float],
    mask: np.ndarray,
    median_depth_m: float | None,
    center_pixel_uv: list[int] | tuple[int, int] | None,
    center_camera_m: list[float] | tuple[float, float, float] | None,
    center_goal_m: list[float] | tuple[float, float, float] | None,
    valid_depth_points: int,
    target_contact_goal_m: list[float] | tuple[float, float, float] | None = None,
    contact_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    height, width = np.asarray(mask).shape
    result = {
        "track_id": str(track_id),
        "label": str(label),
        "score": float(score),
        "box_xyxy": [float(value) for value in box_xyxy],
        "mask_encoding": MASK_ENCODING,
        "mask_data": encode_mask(mask),
        "mask_height": int(height),
        "mask_width": int(width),
        "median_depth_m": None if median_depth_m is None else float(median_depth_m),
        "center_pixel_uv": None if center_pixel_uv is None else [int(v) for v in center_pixel_uv],
        "center_camera_m": None if center_camera_m is None else [float(v) for v in center_camera_m],
        "center_goal_m": None if center_goal_m is None else [float(v) for v in center_goal_m],
        "valid_depth_points": int(valid_depth_points),
    }
    if target_contact_goal_m is not None:
        result["target_contact_goal_m"] = [float(v) for v in target_contact_goal_m]
    if contact_candidates is not None:
        result["contact_candidates"] = list(contact_candidates)
    return result


def encode_perception_result(payload: dict[str, Any]) -> bytes:
    """Validate and encode a remote perception result.

    This compatibility name is used by the Thor result receiver.  Contextual
    freshness/source-pair checks stay in the gateway, so this function only
    enforces the payload schema itself.
    """
    errors = validate_result(
        payload,
        ValidationContext(
            result_max_age_ms=float("inf"),
            now_timestamp_us=int(payload.get("published_timestamp_us", now_us())),
        ),
    )
    if errors:
        raise ValueError("; ".join(errors))
    return pack_result(payload)


def decode_perception_result(data: bytes) -> dict[str, Any]:
    payload = unpack_result(data)
    errors = validate_result(
        payload,
        ValidationContext(
            result_max_age_ms=float("inf"),
            now_timestamp_us=int(payload.get("published_timestamp_us", now_us())),
        ),
    )
    if errors:
        raise ValueError("; ".join(errors))
    return payload


@dataclasses.dataclass
class ValidationContext:
    result_max_age_ms: float = 800.0
    expected_rgb_timestamp_us: int | None = None
    expected_depth_timestamp_us: int | None = None
    expected_calibration_id: int | None = None
    expected_image_width: int | None = None
    expected_image_height: int | None = None
    previous_sequence: int | None = None
    now_timestamp_us: int | None = None


def validate_result(payload: dict[str, Any], context: ValidationContext) -> list[str]:
    errors: list[str] = []
    if payload.get("version") != VERSION:
        errors.append(f"unsupported version {payload.get('version')!r}")

    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or sequence < 0:
        errors.append("sequence must be a non-negative integer")
    elif context.previous_sequence is not None and sequence <= context.previous_sequence:
        errors.append("sequence is not monotonic")

    published_us = payload.get("published_timestamp_us")
    if not isinstance(published_us, int) or published_us <= 0:
        errors.append("published_timestamp_us is invalid")
    else:
        now_timestamp_us = context.now_timestamp_us or now_us()
        age_ms = (now_timestamp_us - published_us) / 1000.0
        if age_ms < -100.0:
            errors.append("published timestamp is in the future")
        if age_ms > context.result_max_age_ms:
            errors.append(f"result age {age_ms:.1f} ms exceeds limit")

    for key, expected in (
        ("source_rgb_timestamp_us", context.expected_rgb_timestamp_us),
        ("source_depth_timestamp_us", context.expected_depth_timestamp_us),
        ("calibration_id", context.expected_calibration_id),
        ("image_width", context.expected_image_width),
        ("image_height", context.expected_image_height),
    ):
        value = payload.get(key)
        if expected is not None and value != expected:
            errors.append(f"{key} mismatch: got {value!r}, expected {expected!r}")

    if payload.get("backend") not in BACKENDS:
        errors.append("backend is not one of the supported remote perception backends")
    if not isinstance(payload.get("valid"), bool):
        errors.append("valid must be bool")
    if payload.get("error") is not None and not isinstance(payload.get("error"), str):
        errors.append("error must be string or null")
    if not _finite_number(payload.get("inference_ms")):
        errors.append("inference_ms must be finite")

    width = payload.get("image_width")
    height = payload.get("image_height")
    objects = payload.get("objects")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        errors.append("image dimensions are invalid")
        width, height = 0, 0
    if not isinstance(objects, list):
        errors.append("objects must be a list")
        return errors

    for index, obj in enumerate(objects):
        prefix = f"objects[{index}]"
        if not isinstance(obj, dict):
            errors.append(f"{prefix} is not a map")
            continue
        if not isinstance(obj.get("track_id"), str) or not obj["track_id"]:
            errors.append(f"{prefix}.track_id is invalid")
        if not isinstance(obj.get("label"), str):
            errors.append(f"{prefix}.label is invalid")
        if not _finite_number(obj.get("score")):
            errors.append(f"{prefix}.score is not finite")
        if not _finite_vector(obj.get("box_xyxy"), 4):
            errors.append(f"{prefix}.box_xyxy is invalid")
        if obj.get("mask_encoding") != MASK_ENCODING:
            errors.append(f"{prefix}.mask_encoding is unsupported")
        if obj.get("mask_width") != width or obj.get("mask_height") != height:
            errors.append(f"{prefix}.mask dimensions do not match source RGB")
        if not isinstance(obj.get("mask_data"), (bytes, bytearray)):
            errors.append(f"{prefix}.mask_data is not bytes")
        else:
            try:
                decode_mask(
                    bytes(obj["mask_data"]),
                    int(obj.get("mask_height", 0)),
                    int(obj.get("mask_width", 0)),
                )
            except Exception as error:
                errors.append(f"{prefix}.mask_data decode failed: {error}")
        if not _finite_number(obj.get("median_depth_m"), allow_none=True):
            errors.append(f"{prefix}.median_depth_m is invalid")
        if not _finite_vector(obj.get("center_pixel_uv"), 2, allow_none=True):
            errors.append(f"{prefix}.center_pixel_uv is invalid")
        if obj.get("center_pixel_uv") is not None:
            u, v = obj["center_pixel_uv"]
            if not (0 <= int(u) < width and 0 <= int(v) < height):
                errors.append(f"{prefix}.center_pixel_uv is outside image")
        if not _finite_vector(obj.get("center_camera_m"), 3, allow_none=True):
            errors.append(f"{prefix}.center_camera_m is invalid")
        if not _finite_vector(obj.get("center_goal_m"), 3, allow_none=True):
            errors.append(f"{prefix}.center_goal_m is invalid")
        if not isinstance(obj.get("valid_depth_points"), int) or obj["valid_depth_points"] < 0:
            errors.append(f"{prefix}.valid_depth_points is invalid")
        if not _finite_vector(obj.get("target_contact_goal_m"), 3, allow_none=True):
            errors.append(f"{prefix}.target_contact_goal_m is invalid")
        contact_candidates = obj.get("contact_candidates", [])
        if contact_candidates is None:
            contact_candidates = []
        if not isinstance(contact_candidates, list):
            errors.append(f"{prefix}.contact_candidates must be a list")
            continue
        for candidate_index, candidate in enumerate(contact_candidates):
            candidate_prefix = f"{prefix}.contact_candidates[{candidate_index}]"
            if not isinstance(candidate, dict):
                errors.append(f"{candidate_prefix} is not a map")
                continue
            if not _finite_vector(candidate.get("center_goal_m"), 3):
                errors.append(f"{candidate_prefix}.center_goal_m is invalid")
            if not _finite_vector(candidate.get("approach_goal"), 3):
                errors.append(f"{candidate_prefix}.approach_goal is invalid")
            if not _finite_number(candidate.get("width_m")):
                errors.append(f"{candidate_prefix}.width_m is invalid")
            if not _finite_number(candidate.get("geom_score")):
                errors.append(f"{candidate_prefix}.geom_score is invalid")
            if "probability" in candidate and not _finite_number(candidate.get("probability")):
                errors.append(f"{candidate_prefix}.probability is invalid")
            if "track_id" in candidate and not isinstance(candidate.get("track_id"), str):
                errors.append(f"{candidate_prefix}.track_id is invalid")
            if "source" in candidate and not isinstance(candidate.get("source"), str):
                errors.append(f"{candidate_prefix}.source is invalid")
    return errors
