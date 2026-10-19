"""Stable ZMQ pose wire format used by PICO and Galaxy XR inputs."""

from __future__ import annotations

import json

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
