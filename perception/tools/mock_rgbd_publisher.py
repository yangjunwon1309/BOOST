#!/usr/bin/env python3
"""Publish a JPEG and matching synthetic aligned depth using the G1 protocol."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import lcm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_web_node.lcm_protocol import CameraFrame, encode_camera, encode_depth
from seg_web_node.mock_depth import make_mock_depth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--lcm-url", default=None)
    parser.add_argument("--rgb-channel", default="G1_CAMERA_FRAME")
    parser.add_argument("--depth-channel", default="G1_DEPTH_FRAME")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--count", type=int, default=0, help="0 publishes forever")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--depth-scale-um", type=int, default=1000)
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise SystemExit("JPEG encoding failed")
    depth = make_mock_depth(image)
    height, width = depth.shape
    bus = lcm.LCM(args.lcm_url) if args.lcm_url else lcm.LCM()
    period = 1.0 / args.fps
    published = 0
    print(f"[mock-rgbd] {width}x{height} image={args.image}")
    try:
        while args.count == 0 or published < args.count:
            started = time.monotonic()
            timestamp_us = time.monotonic_ns() // 1000
            bus.publish(
                args.rgb_channel,
                encode_camera(CameraFrame(width, height, timestamp_us, encoded.tobytes())),
            )
            bus.publish(
                args.depth_channel,
                encode_depth(depth, timestamp_us, args.depth_scale_um),
            )
            published += 1
            if published == 1 or published % max(1, round(args.fps * 5)) == 0:
                print(f"[mock-rgbd] published={published} ts={timestamp_us}")
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
