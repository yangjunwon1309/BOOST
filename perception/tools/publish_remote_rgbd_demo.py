#!/usr/bin/env python3
"""Publish sample RGB-D, calibration, and neutral robot state to remote LCM channels."""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import lcm
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.protocol import (  # noqa: E402
    REMOTE_CALIBRATION_CHANNEL,
    REMOTE_DEPTH_CHANNEL,
    REMOTE_RGB_CHANNEL,
    REMOTE_ROBOT_STATE_CHANNEL,
)
from seg_web_node.lcm_protocol import (  # noqa: E402
    CALIBRATION_MAGIC,
    CALIBRATION_PACKET,
    ROBOT_STATE_FLOATS,
    ROBOT_STATE_HAS_TORSO_IMU,
    ROBOT_STATE_HEADER,
    ROBOT_STATE_MAGIC,
    CameraFrame,
    DEPTH_SOURCE_ALIGNED_TO_COLOR,
    encode_camera,
    encode_depth,
)
from seg_web_node.mock_depth import make_mock_depth  # noqa: E402


def load_camera_info(path: Path, *, width: int, height: int) -> dict:
    if path.is_file():
        info = yaml.safe_load(path.read_text()) or {}
    else:
        info = {}
    return {
        "width": int(info.get("width", width)),
        "height": int(info.get("height", height)),
        "fx": float(info.get("fx", 512.0)),
        "fy": float(info.get("fy", 512.0)),
        "cx": float(info.get("cx", (width - 1) * 0.5)),
        "cy": float(info.get("cy", (height - 1) * 0.5)),
        "depth_scale_um": int(info.get("depth_scale_um", 1000)),
    }


def encode_mock_calibration(info: dict, *, timestamp_us: int, calibration_id: int) -> bytes:
    width = int(info["width"])
    height = int(info["height"])
    depth_scale_um = int(info["depth_scale_um"])
    intrinsics = [
        float(info["fx"]),
        float(info["fy"]),
        float(info["cx"]),
        float(info["cy"]),
    ]
    rotation_identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    translation_zero = [0.0, 0.0, 0.0]
    return CALIBRATION_PACKET.pack(
        CALIBRATION_MAGIC,
        1,
        1,
        int(calibration_id),
        int(timestamp_us),
        width,
        height,
        width,
        height,
        depth_scale_um,
        *(intrinsics + intrinsics + rotation_identity + translation_zero),
    )


def encode_neutral_robot_state(*, timestamp_us: int, sequence: int) -> bytes:
    values = np.zeros(107, dtype=np.float32)
    values[0:4] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    values[10:14] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return ROBOT_STATE_HEADER.pack(
        ROBOT_STATE_MAGIC,
        1,
        ROBOT_STATE_HAS_TORSO_IMU,
        int(sequence),
        int(sequence),
        int(timestamp_us),
        0,
    ) + ROBOT_STATE_FLOATS.pack(*[float(v) for v in values])


def read_depth(path: Path, *, width: int, height: int) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"Cannot read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    if depth.dtype != np.uint16:
        depth = np.clip(depth.astype(np.float32), 0, 65535).astype(np.uint16)
    return depth


def expand_images(image: Path, pattern: str | None) -> list[Path]:
    if pattern:
        paths = [Path(item) for item in sorted(glob.glob(pattern))]
        if not paths:
            raise SystemExit(f"No images matched --pattern: {pattern}")
        return paths
    return [image]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("output/g1_resize_only/pair_001.jpg"))
    parser.add_argument("--pattern", default=None, help="Optional glob for an RGB image sequence")
    parser.add_argument("--depth-image", type=Path, default=None)
    parser.add_argument("--camera-info", type=Path, default=Path("config/camera_info.mock.yaml"))
    parser.add_argument("--lcm-url", required=True)
    parser.add_argument("--rgb-channel", default=REMOTE_RGB_CHANNEL)
    parser.add_argument("--depth-channel", default=REMOTE_DEPTH_CHANNEL)
    parser.add_argument("--calibration-channel", default=REMOTE_CALIBRATION_CHANNEL)
    parser.add_argument("--robot-state-channel", default=REMOTE_ROBOT_STATE_CHANNEL)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--count", type=int, default=0, help="0 publishes forever")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--calibration-id", type=int, default=1)
    args = parser.parse_args()

    images = expand_images(args.image, args.pattern)
    bus = lcm.LCM(args.lcm_url)
    period = 1.0 / max(0.1, float(args.fps))
    published = 0
    print(
        "[remote-rgbd-demo] "
        f"images={len(images)} size={args.width}x{args.height} lcm={args.lcm_url}"
    )
    try:
        while args.count == 0 or published < args.count:
            started = time.monotonic()
            image_path = images[published % len(images)]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Cannot read image: {image_path}")
            image = cv2.resize(
                image,
                (args.width, args.height),
                interpolation=cv2.INTER_AREA,
            )
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise SystemExit("JPEG encoding failed")
            timestamp_us = time.monotonic_ns() // 1000
            camera_info = load_camera_info(
                args.camera_info,
                width=args.width,
                height=args.height,
            )
            if args.depth_image is None:
                depth = make_mock_depth(image)
            else:
                depth = read_depth(
                    args.depth_image,
                    width=args.width,
                    height=args.height,
                )
            bus.publish(
                args.calibration_channel,
                encode_mock_calibration(
                    camera_info,
                    timestamp_us=timestamp_us,
                    calibration_id=args.calibration_id,
                ),
            )
            bus.publish(
                args.robot_state_channel,
                encode_neutral_robot_state(
                    timestamp_us=timestamp_us,
                    sequence=published,
                ),
            )
            bus.publish(
                args.rgb_channel,
                encode_camera(
                    CameraFrame(
                        args.width,
                        args.height,
                        timestamp_us,
                        encoded.tobytes(),
                    )
                ),
            )
            bus.publish(
                args.depth_channel,
                encode_depth(
                    depth,
                    timestamp_us,
                    int(camera_info["depth_scale_um"]),
                    source=DEPTH_SOURCE_ALIGNED_TO_COLOR,
                ),
            )
            published += 1
            if published == 1 or published % max(1, round(args.fps * 5)) == 0:
                print(
                    "[remote-rgbd-demo] "
                    f"published={published} image={image_path} ts={timestamp_us}"
                )
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
