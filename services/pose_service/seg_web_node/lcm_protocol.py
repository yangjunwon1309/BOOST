"""Byte-exact G1 RGB/depth LCM packet encoders and decoders."""

from __future__ import annotations

import dataclasses
import struct
import zlib

import numpy as np

CAMERA_MAGIC = 0x4D524143
DEPTH_MAGIC = 0x48545044
CALIBRATION_MAGIC = 0x4C414343
ROBOT_STATE_MAGIC = 0x31545347
CAMERA_HEADER = struct.Struct("<IIIIQ")
DEPTH_HEADER = struct.Struct("<IIIBBHIQI")
CALIBRATION_PACKET = struct.Struct("<IHHQQIIIII20d")
ROBOT_STATE_HEADER = struct.Struct("<IHHQQqB")
ROBOT_STATE_FLOATS = struct.Struct("<107f")
DEPTH_ENCODING_RAW_ZLIB = 0
DEPTH_SOURCE_ALIGNED_TO_COLOR = 0
DEPTH_SOURCE_RESIZE_ONLY = 1
DEPTH_SOURCE_RAW_DEPTH_OPTICAL = 2
ROBOT_STATE_HAS_TORSO_IMU = 1


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    width: int
    height: int
    timestamp_us: int
    jpeg: bytes


@dataclasses.dataclass(frozen=True)
class DepthFrame:
    width: int
    height: int
    encoding: int
    source: int
    depth_scale_um: int
    timestamp_us: int
    depth: np.ndarray


@dataclasses.dataclass(frozen=True)
class CameraCalibration:
    calibration_id: int
    timestamp_us: int
    depth_width: int
    depth_height: int
    color_width: int
    color_height: int
    depth_scale_um: int
    depth_intrinsics: np.ndarray
    color_intrinsics: np.ndarray
    rotation_color_depth: np.ndarray
    translation_color_depth_m: np.ndarray
    rectified: bool


@dataclasses.dataclass(frozen=True)
class RobotState:
    sequence: int
    lowstate_sequence: int
    timestamp_us: int
    mode_machine: int
    has_torso_imu: bool
    base_quat_wxyz: np.ndarray
    torso_quat_wxyz: np.ndarray
    q: np.ndarray
    dq: np.ndarray


def encode_camera(frame: CameraFrame) -> bytes:
    return CAMERA_HEADER.pack(
        CAMERA_MAGIC, frame.width, frame.height, len(frame.jpeg), frame.timestamp_us
    ) + frame.jpeg


def decode_camera(data: bytes) -> CameraFrame:
    if len(data) < CAMERA_HEADER.size:
        raise ValueError("camera packet is shorter than its header")
    magic, width, height, size, timestamp_us = CAMERA_HEADER.unpack_from(data)
    if magic != CAMERA_MAGIC:
        raise ValueError(f"invalid camera magic: 0x{magic:08x}")
    if width <= 0 or height <= 0 or len(data) != CAMERA_HEADER.size + size:
        raise ValueError("invalid camera dimensions or payload size")
    return CameraFrame(width, height, timestamp_us, data[CAMERA_HEADER.size:])


def encode_depth(
    depth: np.ndarray,
    timestamp_us: int,
    depth_scale_um: int = 1000,
    source: int = DEPTH_SOURCE_ALIGNED_TO_COLOR,
) -> bytes:
    array = np.ascontiguousarray(depth, dtype="<u2")
    if array.ndim != 2 or depth_scale_um <= 0:
        raise ValueError("depth must be HxW uint16 and scale must be positive")
    height, width = array.shape
    payload = zlib.compress(array.tobytes(), level=1)
    return DEPTH_HEADER.pack(
        DEPTH_MAGIC,
        width,
        height,
        DEPTH_ENCODING_RAW_ZLIB,
        source,
        0,
        depth_scale_um,
        timestamp_us,
        len(payload),
    ) + payload


def decode_depth(data: bytes, *, allow_unaligned: bool = False) -> DepthFrame:
    if len(data) < DEPTH_HEADER.size:
        raise ValueError("depth packet is shorter than its header")
    magic, width, height, encoding, source, reserved, scale, timestamp_us, size = (
        DEPTH_HEADER.unpack_from(data)
    )
    if magic != DEPTH_MAGIC:
        raise ValueError(f"invalid depth magic: 0x{magic:08x}")
    if width <= 0 or height <= 0 or width * height > 16_777_216:
        raise ValueError("invalid or excessive depth dimensions")
    if reserved != 0 or encoding != DEPTH_ENCODING_RAW_ZLIB or scale <= 0:
        raise ValueError("unsupported depth header fields")
    if source not in (DEPTH_SOURCE_ALIGNED_TO_COLOR, DEPTH_SOURCE_RAW_DEPTH_OPTICAL) and not allow_unaligned:
        raise ValueError(f"depth source {source} is not calibrated align-to-color")
    if len(data) != DEPTH_HEADER.size + size:
        raise ValueError("invalid depth payload size")
    expected = width * height * 2
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(data[DEPTH_HEADER.size:], expected + 1)
    if decompressor.unconsumed_tail:
        raise ValueError("compressed depth exceeds declared dimensions")
    raw += decompressor.flush()
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("invalid or excessive compressed depth stream")
    if len(raw) != expected:
        raise ValueError(f"depth expands to {len(raw)} bytes, expected {expected}")
    depth = np.frombuffer(raw, dtype="<u2").reshape(height, width).copy()
    return DepthFrame(width, height, encoding, source, scale, timestamp_us, depth)


def decode_calibration(data: bytes) -> CameraCalibration:
    if len(data) != CALIBRATION_PACKET.size:
        raise ValueError(f"calibration packet is {len(data)} bytes, expected {CALIBRATION_PACKET.size}")
    unpacked = CALIBRATION_PACKET.unpack(data)
    magic, version, flags, calibration_id, timestamp_us = unpacked[:5]
    if magic != CALIBRATION_MAGIC or version != 1:
        raise ValueError("unsupported camera calibration packet")
    dw, dh, cw, ch, scale = unpacked[5:10]
    if min(dw, dh, cw, ch, scale) <= 0:
        raise ValueError("invalid calibration dimensions or scale")
    values = np.asarray(unpacked[10:], dtype=np.float64)
    depth_k, color_k = values[:4], values[4:8]
    rotation = values[8:17].reshape(3, 3)
    translation = values[17:20]
    if np.any(depth_k[:2] <= 0) or np.any(color_k[:2] <= 0):
        raise ValueError("invalid focal length in calibration")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or np.linalg.det(rotation) < 0.99:
        raise ValueError("R_color_depth is not a proper rotation")
    return CameraCalibration(
        calibration_id, timestamp_us, dw, dh, cw, ch, scale,
        depth_k.copy(), color_k.copy(), rotation.copy(), translation.copy(),
        bool(flags & 1),
    )


def encode_robot_state(
    *,
    sequence: int,
    lowstate_sequence: int,
    timestamp_us: int,
    mode_machine: int = 0,
    base_quat_wxyz: np.ndarray | list[float] | tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    base_gyro: np.ndarray | list[float] | tuple[float, ...] = (0.0, 0.0, 0.0),
    base_accel: np.ndarray | list[float] | tuple[float, ...] = (0.0, 0.0, 0.0),
    torso_quat_wxyz: np.ndarray | list[float] | tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    torso_gyro: np.ndarray | list[float] | tuple[float, ...] = (0.0, 0.0, 0.0),
    torso_accel: np.ndarray | list[float] | tuple[float, ...] = (0.0, 0.0, 0.0),
    q: np.ndarray | list[float] | tuple[float, ...] = (),
    dq: np.ndarray | list[float] | tuple[float, ...] = (),
    tau: np.ndarray | list[float] | tuple[float, ...] = (),
    has_torso_imu: bool = True,
) -> bytes:
    q_array = np.asarray(q, dtype=np.float64).reshape(-1)

    def optional_29(value) -> np.ndarray:
        if value is None:
            return np.zeros(29, dtype=np.float64)
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        return array if array.size else np.zeros(29, dtype=np.float64)

    dq_array = optional_29(dq)
    tau_array = optional_29(tau)
    if q_array.shape[0] < 29 or dq_array.shape[0] < 29 or tau_array.shape[0] < 29:
        raise ValueError("robot state q/dq/tau must contain at least 29 values")

    def vector(value, length: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape[0] < length or not np.isfinite(array[:length]).all():
            raise ValueError(f"{name} must contain {length} finite values")
        return array[:length]

    values = np.concatenate(
        [
            vector(base_quat_wxyz, 4, "base_quat_wxyz"),
            vector(base_gyro, 3, "base_gyro"),
            vector(base_accel, 3, "base_accel"),
            vector(torso_quat_wxyz, 4, "torso_quat_wxyz"),
            vector(torso_gyro, 3, "torso_gyro"),
            vector(torso_accel, 3, "torso_accel"),
            q_array[:29],
            dq_array[:29],
            tau_array[:29],
        ]
    ).astype(np.float32)
    flags = ROBOT_STATE_HAS_TORSO_IMU if has_torso_imu else 0
    return (
        ROBOT_STATE_HEADER.pack(
            ROBOT_STATE_MAGIC,
            1,
            flags,
            int(sequence),
            int(lowstate_sequence),
            int(timestamp_us),
            int(mode_machine) & 0xFF,
        )
        + ROBOT_STATE_FLOATS.pack(*values.tolist())
    )


def decode_robot_state(data: bytes) -> RobotState:
    expected = ROBOT_STATE_HEADER.size + ROBOT_STATE_FLOATS.size
    if len(data) != expected:
        raise ValueError(f"robot state packet is {len(data)} bytes, expected {expected}")
    magic, version, flags, sequence, lowstate_sequence, timestamp_us, mode = (
        ROBOT_STATE_HEADER.unpack_from(data)
    )
    if magic != ROBOT_STATE_MAGIC or version != 1:
        raise ValueError("unsupported robot state packet")
    values = np.asarray(ROBOT_STATE_FLOATS.unpack_from(data, ROBOT_STATE_HEADER.size), dtype=np.float64)
    # Wire order: base quat/gyro/accel, torso quat/gyro/accel, q/dq/tau.
    return RobotState(
        sequence, lowstate_sequence, timestamp_us, mode,
        bool(flags & ROBOT_STATE_HAS_TORSO_IMU), values[0:4].copy(),
        values[10:14].copy(), values[20:49].copy(), values[49:78].copy(),
    )
