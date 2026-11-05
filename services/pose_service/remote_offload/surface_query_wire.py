"""Small PATH ZMQ pose-wire subset shared with the hand-MPPI client."""

from __future__ import annotations

import json

import numpy as np


HEADER_SIZE = 1280
SURFACE_QUERY_TOPIC = "hand_surface_query"
SURFACE_PROTOCOL_VERSION = 1
NUM_TIPS = 5
SIDE_TO_ID = {"left": 0, "right": 1}
ID_TO_SIDE = {value: key for key, value in SIDE_TO_ID.items()}


def pack_surface_query(fields: dict[str, np.ndarray]) -> bytes:
    declarations = []
    chunks = []
    dtype_names = {
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32",
        np.dtype(np.int64): "i64",
        np.dtype(bool): "bool",
    }
    for name, value in fields.items():
        array = np.asarray(value)
        dtype = dtype_names.get(array.dtype)
        if dtype is None:
            raise ValueError(f"unsupported surface-query wire dtype {array.dtype}")
        array = np.ascontiguousarray(array)
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        declarations.append({"name": name, "dtype": dtype, "shape": list(array.shape)})
        chunks.append(array.tobytes())
    header = json.dumps(
        {"v": 3, "endian": "le", "count": 1, "fields": declarations},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError("surface-query wire header is too large")
    return SURFACE_QUERY_TOPIC.encode() + header.ljust(HEADER_SIZE, b"\0") + b"".join(chunks)


def unpack_surface_query(raw: bytes) -> dict[str, np.ndarray]:
    prefix = SURFACE_QUERY_TOPIC.encode()
    if not raw.startswith(prefix) or len(raw) < len(prefix) + HEADER_SIZE:
        raise ValueError("truncated or mismatched surface-query packet")
    start = len(prefix)
    header = json.loads(raw[start : start + HEADER_SIZE].split(b"\0", 1)[0])
    if header.get("endian") != "le":
        raise ValueError("unsupported surface-query byte order")
    dtypes = {
        "i32": np.dtype("<i4"),
        "i64": np.dtype("<i8"),
        "f32": np.dtype("<f4"),
        "f64": np.dtype("<f8"),
        "bool": np.dtype("?"),
    }
    cursor = start + HEADER_SIZE
    output: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        name = str(field["name"])
        shape = tuple(int(item) for item in field["shape"])
        if any(item < 0 for item in shape) or name in output:
            raise ValueError("invalid surface-query field declaration")
        dtype = dtypes[field["dtype"]]
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        end = cursor + size
        if end > len(raw):
            raise ValueError(f"truncated surface-query field {name}")
        output[name] = np.frombuffer(raw[cursor:end], dtype=dtype).reshape(shape).copy()
        cursor = end
    if cursor != len(raw):
        raise ValueError("unexpected trailing surface-query bytes")
    return output
