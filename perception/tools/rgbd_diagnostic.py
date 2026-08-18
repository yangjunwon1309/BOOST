#!/usr/bin/env python3
"""Receive, validate, and visualize synchronized G1 RGB-D packets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import lcm
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_web_node.geometry import CameraIntrinsics, deproject
from seg_web_node.depth_alignment import align_depth_to_color, calibration_for_frame
from seg_web_node.lcm_protocol import (
    DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    decode_calibration,
    decode_camera,
    decode_depth,
)
from seg_web_node.rgbd_sync import RgbdSynchronizer, depth_at_rgb_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", default=None)
    parser.add_argument("--rgb-channel", default="G1_CAMERA_FRAME")
    parser.add_argument("--depth-channel", default="G1_DEPTH_FRAME")
    parser.add_argument("--calibration-channel", default="G1_CAMERA_CALIBRATION")
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/rgbd_diagnostic"))
    parser.add_argument("--allow-unaligned", action="store_true")
    parser.add_argument("--camera-info", type=Path, default=None)
    args = parser.parse_args()
    if args.camera_info is not None and not args.camera_info.is_file():
        raise SystemExit(f"Camera info does not exist: {args.camera_info}")
    sync = RgbdSynchronizer(args.sync_tolerance_ms)
    bus = lcm.LCM(args.lcm_url) if args.lcm_url else lcm.LCM()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = 0
    errors = 0
    calibration = None

    def process(pair):
        nonlocal pairs, errors
        jpeg = np.frombuffer(pair.rgb.jpeg, dtype=np.uint8)
        image = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        if image is None:
            errors += 1
            return
        frame_calibration = (
            None
            if calibration is None
            else calibration_for_frame(
                calibration,
                color_width=pair.rgb.width,
                color_height=pair.rgb.height,
                depth_width=(
                    pair.depth.width
                    if pair.depth.source == DEPTH_SOURCE_RAW_DEPTH_OPTICAL
                    else None
                ),
                depth_height=(
                    pair.depth.height
                    if pair.depth.source == DEPTH_SOURCE_RAW_DEPTH_OPTICAL
                    else None
                ),
            )
        )
        if pair.depth.source == DEPTH_SOURCE_RAW_DEPTH_OPTICAL:
            if calibration is None:
                raise ValueError("raw depth arrived before camera calibration")
            assert frame_calibration is not None
            depth = align_depth_to_color(pair.depth, frame_calibration)
        else:
            depth = depth_at_rgb_size(pair)
        h, w = depth.shape
        if frame_calibration is not None:
            intrinsics = CameraIntrinsics(
                frame_calibration.color_width, frame_calibration.color_height,
                *frame_calibration.color_intrinsics,
            )
        elif args.camera_info is None:
            intrinsics = CameraIntrinsics(
                w, h, 0.8 * w, 0.8 * w, (w - 1) / 2, (h - 1) / 2
            )
        else:
            values = yaml.safe_load(args.camera_info.read_text())
            intrinsics = CameraIntrinsics(
                int(values["width"]), int(values["height"]),
                float(values["fx"]), float(values["fy"]),
                float(values["cx"]), float(values["cy"]),
            )
            if (intrinsics.width, intrinsics.height) != (w, h):
                raise ValueError("camera-info resolution does not match received RGB")
        xyz = deproject(depth, intrinsics, pair.depth.depth_scale_um)
        center = xyz[h // 2, w // 2]
        valid = np.isfinite(xyz[..., 2])
        depth_m = depth.astype(np.float64) * pair.depth.depth_scale_um / 1_000_000.0
        nonzero = depth > 0
        in_range = nonzero & (depth_m >= 0.15) & (depth_m <= 3.0)
        nonzero_values = depth_m[nonzero]
        percentiles = (
            [float(value) for value in np.percentile(nonzero_values, [0, 50, 95, 100])]
            if nonzero_values.size
            else [None, None, None, None]
        )
        normalized = np.zeros_like(depth, dtype=np.uint8)
        if np.any(valid):
            values = xyz[..., 2][valid]
            lo, hi = float(values.min()), float(values.max())
            if hi > lo:
                normalized[valid] = np.clip((values - lo) * 255 / (hi - lo), 0, 255)
        color_depth = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(image, 0.55, color_depth, 0.45, 0)
        pairs += 1
        cv2.imwrite(str(args.output_dir / f"pair_{pairs:03d}.jpg"), overlay)
        print(
            f"[rgbd-diagnostic] pair={pairs} ts={pair.rgb.timestamp_us} "
            f"delta_ms={pair.delta_us / 1000:.3f} valid={valid.mean():.3f} "
            f"nonzero={nonzero.mean():.3f} in_range_0.15_3m={in_range.mean():.3f} "
            f"depth_m_min_p50_p95_max={percentiles} "
            f"center_xyz={center.tolist()} source={pair.depth.source} "
            f"rgb={pair.rgb.width}x{pair.rgb.height} "
            f"depth={pair.depth.width}x{pair.depth.height}"
        )

    def on_rgb(_channel, data):
        nonlocal errors
        try:
            pair = sync.add_rgb(decode_camera(data))
            if pair is not None:
                process(pair)
        except Exception as error:
            errors += 1
            print(f"[rgbd-diagnostic] RGB error: {error}")

    def on_depth(_channel, data):
        nonlocal errors
        try:
            frame = decode_depth(data, allow_unaligned=args.allow_unaligned)
            pair = sync.add_depth(frame)
            if pair is not None:
                process(pair)
        except Exception as error:
            errors += 1
            print(f"[rgbd-diagnostic] depth error: {error}")

    def on_calibration(_channel, data):
        nonlocal calibration, errors
        try:
            calibration = decode_calibration(data)
            print(f"[rgbd-diagnostic] calibration id={calibration.calibration_id} "
                  f"depth={calibration.depth_width}x{calibration.depth_height} "
                  f"color={calibration.color_width}x{calibration.color_height}")
        except Exception as error:
            errors += 1
            print(f"[rgbd-diagnostic] calibration error: {error}")

    bus.subscribe(args.rgb_channel, on_rgb)
    bus.subscribe(args.depth_channel, on_depth)
    bus.subscribe(args.calibration_channel, on_calibration)
    deadline = time.monotonic() + args.timeout_s
    while pairs < args.count and time.monotonic() < deadline:
        bus.handle_timeout(200)
    print(f"[rgbd-diagnostic] pairs={pairs} errors={errors} saved={args.output_dir}")
    return 0 if pairs >= args.count and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
