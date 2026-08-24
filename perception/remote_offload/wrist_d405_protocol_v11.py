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
WRIST_BELIEF_TELEMETRY_CHANNEL = "PATH_V11_WRIST_BELIEF_TELEMETRY"
# Direct state-machine -> laptop diagnostic stream.  This is intentionally
# separate from belief telemetry so an unavailable Thor-local ZMQ relay cannot
# hide the actual IPOPT reference from the wrist web viewer.
WRIST_IPOPT_VISUAL_CHANNEL = "PATH_V11_WRIST_IPOPT_VISUAL"


def now_us() -> int:
    return time.time_ns() // 1_000


def normalize_concepts(value: Any, *, max_concepts: int = 8) -> list[str]:
    """Validate the bounded text-prompt vocabulary used by the wrist worker."""
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        raise ValueError("concepts must be a string or list")
    concepts: list[str] = []
    for item in raw:
        concept = " ".join(str(item).strip().split())
        if not concept:
            continue
        if len(concept) > 80 or any(ord(char) < 32 for char in concept):
            raise ValueError("each concept must be printable and at most 80 characters")
        if concept not in concepts:
            concepts.append(concept)
    if not concepts:
        raise ValueError("at least one concept is required")
    if len(concepts) > max_concepts:
        raise ValueError(f"at most {max_concepts} concepts are allowed")
    return concepts


def pack_concept_control(payload: dict[str, Any]) -> bytes:
    concepts = normalize_concepts(payload.get("concepts"))
    revision = payload.get("revision")
    if not isinstance(revision, int) or revision < 0:
        raise ValueError("revision must be a non-negative integer")
    return msgpack.packb(
        {
            "version": VERSION,
            "published_timestamp_us": int(payload.get("published_timestamp_us", now_us())),
            "revision": revision,
            "concepts": concepts,
        },
        use_bin_type=True,
    )


def unpack_concept_control(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist concept-control payload")
    if not isinstance(payload.get("revision"), int) or payload["revision"] < 0:
        raise ValueError("wrist concept-control revision is invalid")
    if not isinstance(payload.get("published_timestamp_us"), int) or payload["published_timestamp_us"] <= 0:
        raise ValueError("wrist concept-control timestamp is invalid")
    payload["concepts"] = normalize_concepts(payload.get("concepts"))
    return payload


def pack_heartbeat(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_heartbeat(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist heartbeat payload")
    return payload


def pack_belief_telemetry(payload: dict[str, Any]) -> bytes:
    """Wire-safe v11 telemetry for laptop-only visualization."""
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_belief_telemetry(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist belief telemetry payload")
    return payload


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


def unpack_mask(data: bytes, *, width: int, height: int) -> np.ndarray:
    expected_bits = int(width) * int(height)
    expected_bytes = (expected_bits + 7) // 8
    raw = zlib.decompress(data, bufsize=expected_bytes)
    if len(raw) != expected_bytes:
        raise ValueError("D405 mask has an invalid packed length")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:expected_bits].reshape(height, width).astype(bool)


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


def _validate_vector(value: Any, *, length: int, name: str) -> None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} finite values") from error
    if vector.size != length or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {length} finite values")


def _validate_state_pairing(value: Any, *, camera_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{camera_id} state_pairing must be a mapping")
    for key in ("lookup_us", "state_arrival_us", "bias_us"):
        if not isinstance(value.get(key), int):
            raise ValueError(f"{camera_id} state_pairing.{key} must be an integer")
    if value["lookup_us"] <= 0 or value["state_arrival_us"] <= 0:
        raise ValueError(f"{camera_id} state_pairing has invalid timestamps")
    if value.get("method") not in {"nearest", "interpolated"}:
        raise ValueError(f"{camera_id} state_pairing method is invalid")
    for key in ("residual_ms", "wrist_speed_mps", "wrist_omega_rps", "estimated_error_m"):
        numeric = value.get(key)
        if numeric is not None and (not _finite(numeric) or float(numeric) < 0.0):
            raise ValueError(f"{camera_id} state_pairing.{key} must be a non-negative finite number or null")
    if not _finite(value.get("residual_ms")):
        raise ValueError(f"{camera_id} state_pairing.residual_ms is required")


def validate_bundle(payload: dict[str, Any], *, require_two: bool = True) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist D405 bundle version")
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("wrist D405 bundle is missing cameras")
    # Added after initial v11 deployment. Keep it optional so a laptop worker
    # can be restarted before Thor; the worker falls back to the separately
    # forwarded PATH_V11_WRIST_ROBOT_STATE stream during a rolling update.
    state_packet = payload.get("robot_state_packet")
    if state_packet is not None:
        if not isinstance(state_packet, (bytes, bytearray)):
            raise ValueError("wrist D405 bundle has an invalid paired robot-state packet")
        if int(payload.get("robot_state_host_received_us", 0)) <= 0:
            raise ValueError("paired wrist robot state has no host timestamp")
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
        # v14 pairs each wrist independently on Thor.  Keep all three fields
        # optional together for a rolling gateway/worker deployment, but never
        # accept a partial pairing because that silently falls back to a shared q.
        pairing_fields = (
            "q_at_exposure",
            "base_quat_at_exposure_wxyz",
            "state_pairing",
        )
        present = [key in frame for key in pairing_fields]
        if any(present) and not all(present):
            raise ValueError(f"{camera_id} has an incomplete v14 state pairing")
        if all(present):
            _validate_vector(frame["q_at_exposure"], length=29, name=f"{camera_id}.q_at_exposure")
            _validate_vector(
                frame["base_quat_at_exposure_wxyz"], length=4,
                name=f"{camera_id}.base_quat_at_exposure_wxyz",
            )
            _validate_state_pairing(frame["state_pairing"], camera_id=camera_id)
    return payload


def pack_bundle(payload: dict[str, Any]) -> bytes:
    validate_bundle(payload)
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_bundle(data: bytes, *, require_two: bool = True) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    return validate_bundle(payload, require_two=require_two)


def make_result(
    *,
    sequence: int,
    source_bundle_timestamp_us: int,
    source_bundle_sequence: int | None = None,
    objects: list[dict[str, Any]],
    discarded_seam_masks: int,
    geometry_status: dict[str, Any] | None = None,
    t_goal_cam: dict[str, list[float]] | None = None,
    transform_diagnostics: dict[str, dict[str, Any]] | None = None,
    valid: bool = True,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "sequence": int(sequence),
        "published_timestamp_us": now_us(),
        "source_bundle_timestamp_us": int(source_bundle_timestamp_us),
        "source_bundle_sequence": (
            None if source_bundle_sequence is None else int(source_bundle_sequence)
        ),
        "valid": bool(valid),
        "error": error,
        "discarded_seam_masks": int(discarded_seam_masks),
        "geometry_status": dict(geometry_status or {}),
        # Frame-level camera transforms are deliberately not copied into every
        # object. They are required even on a no-detection frame so the viewer
        # can project a pelvis-frame belief onto its newest camera image.
        "t_goal_cam": dict(t_goal_cam or {}),
        "transform_diagnostics": dict(transform_diagnostics or {}),
        "objects": list(objects),
    }


def _validate_transform_matrix(value: Any, *, name: str) -> None:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite 4x4 matrix") from error
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")


def validate_result(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported wrist D405 result version")
    if not isinstance(payload.get("objects"), list):
        raise ValueError("wrist D405 result objects must be a list")
    transforms = payload.get("t_goal_cam", {})
    diagnostics = payload.get("transform_diagnostics", {})
    if not isinstance(transforms, dict) or not isinstance(diagnostics, dict):
        raise ValueError("wrist D405 result transforms must be mappings")
    for camera_id, transform in transforms.items():
        if camera_id not in CAMERA_IDS:
            raise ValueError(f"unknown result camera transform {camera_id!r}")
        _validate_transform_matrix(transform, name=f"t_goal_cam[{camera_id}]")
        diagnostic = diagnostics.get(camera_id)
        if not isinstance(diagnostic, dict):
            raise ValueError(f"missing transform diagnostics for {camera_id}")
        diagnostic_transform = diagnostic.get("t_goal_camera")
        _validate_transform_matrix(
            diagnostic_transform, name=f"transform_diagnostics[{camera_id}].t_goal_camera"
        )
        if list(transform) != list(diagnostic_transform):
            raise ValueError(f"frame transform mismatch for {camera_id}")
    return payload


def pack_ipopt_visual(payload: dict[str, Any]) -> bytes:
    """Encode the state machine's current IPOPT wrist targets for the viewer."""
    wire = {"version": VERSION, **payload}
    return msgpack.packb(wire, use_bin_type=True, strict_types=False)


def unpack_ipopt_visual(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != VERSION
        or payload.get("schema") != "path.wrist_ipopt_visual.v1"
    ):
        raise ValueError("unsupported wrist IPOPT visual payload")
    return payload


def pack_result(payload: dict[str, Any]) -> bytes:
    validate_result(payload)
    return msgpack.packb(payload, use_bin_type=True, strict_types=False)


def unpack_result(data: bytes) -> dict[str, Any]:
    payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    return validate_result(payload)
