#!/usr/bin/env python3
"""Print one validated v11 dual-wrist D405 bundle from an LCM transport."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lcm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.wrist_d405_protocol_v11 import (
    CAMERA_IDS,
    WRIST_BUNDLE_CHANNEL,
    unpack_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", required=True)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args()
    bus = lcm.LCM(args.lcm_url)
    received = False

    def on_bundle(_channel: str, data: bytes) -> None:
        nonlocal received
        bundle = unpack_bundle(data)
        cameras = bundle["cameras"]
        summary = {
            camera_id: {
                "serial": cameras[camera_id]["serial"],
                "grid": f"{cameras[camera_id]['width']}x{cameras[camera_id]['height']}",
                "jpeg_bytes": len(cameras[camera_id]["color_jpeg"]),
                "depth_bytes": len(cameras[camera_id]["depth_z16_zlib"]),
                "rgb_depth_delta_ms": round(float(cameras[camera_id]["diagnostics"]["rgb_depth_delta_ms"]), 2),
            }
            for camera_id in CAMERA_IDS
        }
        print(
            f"VALID {WRIST_BUNDLE_CHANNEL} sequence={bundle['sequence']} "
            f"skew_ms={bundle['inter_wrist_skew_us'] / 1000.0:.2f} cameras={summary}",
            flush=True,
        )
        received = True

    bus.subscribe(WRIST_BUNDLE_CHANNEL, on_bundle)
    deadline = time.monotonic() + max(0.1, float(args.timeout_sec))
    while not received and time.monotonic() < deadline:
        bus.handle_timeout(100)
    if not received:
        raise SystemExit(f"no {WRIST_BUNDLE_CHANNEL} packet within {args.timeout_sec:.1f} seconds")


if __name__ == "__main__":
    main()
