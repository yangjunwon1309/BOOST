#!/usr/bin/env python3
"""Offline Thor RGB-D validation without a G1 or multicast dependency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_web_node.geometry import CameraIntrinsics, deproject
from seg_web_node.lcm_protocol import (
    CameraFrame,
    decode_camera,
    decode_depth,
    encode_camera,
    encode_depth,
)
from seg_web_node.mock_depth import make_mock_depth
from seg_web_node.rgbd_sync import RgbdSynchronizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/mock_rgbd_overlay.jpg"))
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    depth = make_mock_depth(image)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    h, w = depth.shape
    timestamp_us = 123_456_789
    rgb = decode_camera(encode_camera(CameraFrame(w, h, timestamp_us, jpeg.tobytes())))
    depth_frame = decode_depth(encode_depth(depth, timestamp_us))
    sync = RgbdSynchronizer()
    assert sync.add_rgb(rgb) is None
    pair = sync.add_depth(depth_frame)
    assert pair is not None and pair.delta_us == 0
    intrinsics = CameraIntrinsics(w, h, 0.8 * w, 0.8 * w, (w - 1) / 2, (h - 1) / 2)
    xyz = deproject(pair.depth.depth, intrinsics, pair.depth.depth_scale_um)
    valid = np.isfinite(xyz[..., 2])
    assert valid.mean() > 0.9
    color_depth = cv2.applyColorMap(
        cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    overlay = cv2.addWeighted(image, 0.55, color_depth, 0.45, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(args.output), overlay)
    print(
        f"[mock-validation] PASS size={w}x{h} valid={valid.mean():.3f} "
        f"center_xyz={xyz[h // 2, w // 2].tolist()} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
