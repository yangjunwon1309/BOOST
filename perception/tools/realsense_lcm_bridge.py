#!/usr/bin/env python3
"""Publish a local Intel RealSense RGB-D stream using the G1 LCM RGB-D protocol."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import cv2
import lcm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_web_node.lcm_protocol import (  # noqa: E402
    CALIBRATION_MAGIC,
    CALIBRATION_PACKET,
    CameraFrame,
    DEPTH_SOURCE_ALIGNED_TO_COLOR,
    DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    encode_camera,
    encode_depth,
)


def _now_us() -> int:
    return time.monotonic_ns() // 1000


def _intrinsics4(intrinsics) -> tuple[float, float, float, float]:
    return (
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.ppx),
        float(intrinsics.ppy),
    )


def _calibration_id(
    depth_width: int,
    depth_height: int,
    color_width: int,
    color_height: int,
    depth_scale_um: int,
    values: tuple[float, ...],
) -> int:
    """Match the FNV-1a hash used by g1_ws/include/camera_calibration_packet.hpp."""

    h = 1469598103934665603

    def append(data: bytes) -> None:
        nonlocal h
        for byte in data:
            h ^= byte
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    append(struct.pack("<4i", depth_width, depth_height, color_width, color_height))
    append(struct.pack("<I", depth_scale_um))
    append(struct.pack("<20d", *values))
    return h


def encode_calibration(
    *,
    depth_width: int,
    depth_height: int,
    color_width: int,
    color_height: int,
    depth_scale_um: int,
    depth_intrinsics: tuple[float, float, float, float],
    color_intrinsics: tuple[float, float, float, float],
    rotation_color_depth: np.ndarray,
    translation_color_depth_m: np.ndarray,
    timestamp_us: int,
) -> bytes:
    rotation = np.asarray(rotation_color_depth, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_color_depth_m, dtype=np.float64).reshape(3)
    values = tuple(float(x) for x in (
        *depth_intrinsics,
        *color_intrinsics,
        *rotation.reshape(-1),
        *translation,
    ))
    calibration_id = _calibration_id(
        depth_width,
        depth_height,
        color_width,
        color_height,
        depth_scale_um,
        values,
    )
    return CALIBRATION_PACKET.pack(
        CALIBRATION_MAGIC,
        1,
        1,
        calibration_id,
        timestamp_us,
        depth_width,
        depth_height,
        color_width,
        color_height,
        depth_scale_um,
        *values,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=1")
    parser.add_argument("--rgb-channel", default="G1_CAMERA_FRAME")
    parser.add_argument("--depth-channel", default="G1_DEPTH_FRAME")
    parser.add_argument("--calibration-channel", default="G1_CAMERA_CALIBRATION")
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--raw-depth-optical",
        action="store_true",
        help="Publish unaligned depth plus true depth->color calibration.",
    )
    parser.add_argument("--calibration-resend-sec", type=float, default=1.0)
    parser.add_argument("--report-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    try:
        import pyrealsense2 as rs
    except Exception as error:
        raise SystemExit(
            "pyrealsense2 is required for D435i capture. Run this on the Thor "
            "host/container where librealsense Python bindings are installed."
        ) from error

    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width/height/fps must be positive")
    if not 0 <= args.jpeg_quality <= 100:
        raise SystemExit("jpeg-quality must be in [0, 100]")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_um = max(1, int(round(float(depth_sensor.get_depth_scale()) * 1_000_000)))
    align = None if args.raw_depth_optical else rs.align(rs.stream.color)
    bus = lcm.LCM(args.lcm_url)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()
    depth_to_color = depth_profile.get_extrinsics_to(color_profile)

    frames_sent = 0
    errors = 0
    last_calib = 0.0
    last_report = time.monotonic()
    started = time.monotonic()
    print(
        f"[thor-d435i-lcm] publishing RGB-D on {args.lcm_url} "
        f"{args.rgb_channel}/{args.depth_channel}/{args.calibration_channel}"
    )
    print(
        f"[thor-d435i-lcm] {args.width}x{args.height}@{args.fps} "
        f"depth_scale={depth_scale_um}um source="
        f"{'raw_depth_optical' if args.raw_depth_optical else 'aligned_to_color'}"
    )

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                errors += 1
                continue

            timestamp_us = _now_us()
            color_rgb = np.asanyarray(color_frame.get_data())
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            ok, jpeg = cv2.imencode(
                ".jpg",
                color_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if not ok:
                errors += 1
                continue

            depth = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()), dtype=np.uint16)
            if align is None:
                depth_source = DEPTH_SOURCE_RAW_DEPTH_OPTICAL
                calib_depth_intr = _intrinsics4(depth_intrinsics)
                calib_color_intr = _intrinsics4(color_intrinsics)
                rotation = np.asarray(depth_to_color.rotation, dtype=np.float64).reshape(3, 3)
                translation = np.asarray(depth_to_color.translation, dtype=np.float64)
            else:
                depth_source = DEPTH_SOURCE_ALIGNED_TO_COLOR
                aligned_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                calib_depth_intr = _intrinsics4(aligned_intrinsics)
                calib_color_intr = _intrinsics4(color_intrinsics)
                rotation = np.eye(3, dtype=np.float64)
                translation = np.zeros(3, dtype=np.float64)

            now = time.monotonic()
            if now - last_calib >= args.calibration_resend_sec:
                bus.publish(
                    args.calibration_channel,
                    encode_calibration(
                        depth_width=int(depth.shape[1]),
                        depth_height=int(depth.shape[0]),
                        color_width=int(color_frame.get_width()),
                        color_height=int(color_frame.get_height()),
                        depth_scale_um=depth_scale_um,
                        depth_intrinsics=calib_depth_intr,
                        color_intrinsics=calib_color_intr,
                        rotation_color_depth=rotation,
                        translation_color_depth_m=translation,
                        timestamp_us=timestamp_us,
                    ),
                )
                last_calib = now

            bus.publish(
                args.rgb_channel,
                encode_camera(
                    CameraFrame(
                        int(color_frame.get_width()),
                        int(color_frame.get_height()),
                        timestamp_us,
                        jpeg.tobytes(),
                    )
                ),
            )
            bus.publish(
                args.depth_channel,
                encode_depth(
                    depth,
                    timestamp_us,
                    depth_scale_um=depth_scale_um,
                    source=depth_source,
                ),
            )
            frames_sent += 1

            if now - last_report >= args.report_sec:
                elapsed = max(1e-6, now - started)
                print(
                    f"[thor-d435i-lcm] frames={frames_sent} "
                    f"fps={frames_sent / elapsed:.1f} errors={errors}"
                )
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
