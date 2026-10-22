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


def unpack_pose_message(raw: bytes, topic: str = "pose") -> dict[str, np.ndarray]:
    """Decode the packed PATH pose message produced by :func:`pack_pose_message`.

    The wire format is ``topic + fixed-size JSON header + contiguous field data``.
    Returned arrays own their buffers so callers can safely retain them after the
    ZeroMQ receive buffer is released.
    """
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        raise ValueError(f"unexpected topic for {topic!r}")

    offset = len(topic_bytes)
    if len(raw) < offset + HEADER_SIZE:
        raise ValueError("truncated PATH pose header")
    header_bytes = raw[offset : offset + HEADER_SIZE].split(b"\0", 1)[0]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid PATH pose header") from exc
    if header.get("endian") != "le":
        raise ValueError(f"unsupported PATH pose byte order: {header.get('endian')!r}")

    dtype_map = {
        "u8": np.dtype("u1"),
        "i32": np.dtype("<i4"),
        "i64": np.dtype("<i8"),
        "f32": np.dtype("<f4"),
        "f64": np.dtype("<f8"),
        "bool": np.dtype("?") ,
    }
    cursor = offset + HEADER_SIZE
    result: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        try:
            name = str(field["name"])
            dtype = dtype_map[field["dtype"]]
            shape = tuple(int(dimension) for dimension in field["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid PATH pose field declaration") from exc
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"negative dimension for PATH pose field {name!r}")
        byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        end = cursor + byte_count
        if end > len(raw):
            raise ValueError(f"truncated PATH pose field {name!r}")
        if name in result:
            raise ValueError(f"duplicate PATH pose field {name!r}")
        result[name] = np.frombuffer(raw[cursor:end], dtype=dtype).reshape(shape).copy()
        cursor = end
    if cursor != len(raw):
        raise ValueError(f"unexpected trailing data in PATH pose message ({len(raw) - cursor} bytes)")
    return result
