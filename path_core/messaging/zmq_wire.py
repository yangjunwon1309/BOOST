"""Stable PATH ZMQ binary wire-format builders.

The direct PATH runtime exchanges ``pose``, ``planner``, and ``command``
messages with XR sources using this format.  It intentionally preserves the
former SONIC wire bytes so existing PICO, Galaxy XR, and bridge processes can
be upgraded independently.
"""

from __future__ import annotations

import json
import struct
from typing import Sequence

import numpy as np


HEADER_SIZE = 1280


def _build_header(fields: list[dict], *, version: int = 1, count: int = 1) -> bytes:
    header_json = json.dumps(
        {"v": version, "endian": "le", "count": count, "fields": fields},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def build_command_message(
    start: bool,
    stop: bool,
    planner: bool,
    delta_heading: float | None = None,
    *,
    button_sequence: int | None = None,
    button_read_timestamp_us: int | None = None,
    bridge_receive_timestamp_us: int | None = None,
    bridge_forward_timestamp_us: int | None = None,
) -> bytes:
    """Build a byte-compatible legacy ``command`` ZMQ packet."""
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join(
        (struct.pack("B", bool(start)), struct.pack("B", bool(stop)), struct.pack("B", bool(planner)))
    )
    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))
    for name, value in (
        ("button_sequence", button_sequence),
        ("button_read_timestamp_us", button_read_timestamp_us),
        ("button_bridge_receive_timestamp_us", bridge_receive_timestamp_us),
        ("button_bridge_forward_timestamp_us", bridge_forward_timestamp_us),
    ):
        if value is not None:
            fields.append({"name": name, "dtype": "i64", "shape": [1]})
            payload += struct.pack("<q", int(value))
    return b"command" + _build_header(fields) + payload


def build_planner_message(
    mode: int,
    movement: Sequence[float],
    facing: Sequence[float],
    speed: float = -1.0,
    height: float = -1.0,
    upper_body_position: Sequence[float] | None = None,
    upper_body_velocity: Sequence[float] | None = None,
    left_hand_position: Sequence[float] | None = None,
    right_hand_position: Sequence[float] | None = None,
    vr_3pt_position: Sequence[float] | None = None,
    vr_3pt_orientation: Sequence[float] | None = None,
    vr_3pt_compliance: Sequence[float] | None = None,
) -> bytes:
    """Build a byte-compatible legacy ``planner`` ZMQ packet."""
    if len(movement) != 3:
        raise ValueError("movement must have length 3")
    if len(facing) != 3:
        raise ValueError("facing must have length 3")
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("<i", int(mode)),
            struct.pack("<fff", *(float(v) for v in movement)),
            struct.pack("<fff", *(float(v) for v in facing)),
            struct.pack("<f", float(speed)),
            struct.pack("<f", float(height)),
        )
    )
    for name, values in (
        ("upper_body_position", upper_body_position),
        ("upper_body_velocity", upper_body_velocity),
        ("left_hand_joints", left_hand_position),
        ("right_hand_joints", right_hand_position),
        ("vr_position", vr_3pt_position),
        ("vr_orientation", vr_3pt_orientation),
        ("vr_compliance", vr_3pt_compliance),
    ):
        if values is None:
            continue
        fields.append({"name": name, "dtype": "f32", "shape": [len(values)]})
        payload += b"".join(struct.pack("<f", float(value)) for value in values)
    return b"planner" + _build_header(fields) + payload


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    """Pack numpy pose fields into the established PATH ZMQ packet format."""
    fields: list[dict] = []
    binary_data: list[bytes] = []
    dtype_names = {
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32",
        np.dtype(np.int64): "i64",
        np.dtype(bool): "bool",
    }
    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        dtype_name = dtype_names.get(value.dtype)
        if dtype_name is None:
            dtype_name = "f32"
            value = value.astype(np.float32)
        fields.append({"name": key, "dtype": dtype_name, "shape": list(value.shape)})
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        binary_data.append(value.tobytes())
    return topic.encode("utf-8") + _build_header(fields, version=version) + b"".join(binary_data)
