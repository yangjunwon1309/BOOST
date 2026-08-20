#!/usr/bin/env python3
"""Capture a raw-mask D405 point for fixed-rotation translation calibration.

This calibration-only tool pairs a SAM3 result mask with its source RGB-D
bundle, computes the median raw D405 point under that mask, and writes a
known wrist-local reference point. It intentionally bypasses the production
100-point surface/belief filter: a compact red fiducial is enough to fit
translation when the CAD rotation remains fixed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import lcm
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.wrist_d405_protocol_v11 import (  # noqa: E402
    WRIST_BUNDLE_CHANNEL,
    WRIST_RESULT_CHANNEL,
    unpack_bundle,
    unpack_depth_z16,
    unpack_mask,
    unpack_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", required=True)
    parser.add_argument("--side", choices=("left_wrist", "right_wrist"), required=True)
    parser.add_argument("--wrist-point-m", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", help="Require this exact SAM3 label.")
    parser.add_argument("--min-depth-m", type=float, default=0.07)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    observations = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(observations, list):
        raise ValueError(f"{path} must contain observations: [...]")
    return observations


def _raw_mask_point(obj: dict[str, Any], frame: dict[str, Any], minimum_m: float, maximum_m: float) -> tuple[np.ndarray, int, list[float]] | None:
    try:
        width, height = int(frame["width"]), int(frame["height"])
        mask = unpack_mask(obj["mask_data"], width=width, height=height)
        depth = unpack_depth_z16(frame["depth_z16_zlib"], width=width, height=height)
        z = depth.astype(np.float64) * float(frame["depth_scale_m"])
        intrinsics = frame["intrinsics"]
        fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
        ppx, ppy = float(intrinsics["ppx"]), float(intrinsics["ppy"])
    except (KeyError, TypeError, ValueError):
        return None
    valid = mask & np.isfinite(z) & (z >= minimum_m) & (z <= maximum_m)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return None
    points = np.column_stack(((cols - ppx) * z[rows, cols] / fx, (rows - ppy) * z[rows, cols] / fy, z[rows, cols]))
    point = np.median(points, axis=0)
    pixel = [float(np.median(cols)), float(np.median(rows))]
    return point.astype(float), int(len(points)), pixel


def main() -> int:
    args = parse_args()
    wrist_point = np.asarray(args.wrist_point_m, dtype=np.float64).reshape(3)
    if not np.isfinite(wrist_point).all():
        raise SystemExit("--wrist-point-m must be finite")
    bundles: dict[int, dict[str, Any]] = {}
    captured: dict[str, Any] | None = None
    stats = {"bundles": 0, "results": 0, "matching_masks": 0, "raw_depth_points": 0}
    bus = lcm.LCM(args.lcm_url)

    def on_bundle(_channel: str, data: bytes) -> None:
        try:
            bundle = unpack_bundle(data)
        except (TypeError, ValueError):
            return
        bundles[int(bundle.get("sequence", -1))] = bundle
        while len(bundles) > 96:
            del bundles[next(iter(bundles))]
        stats["bundles"] += 1

    def on_result(_channel: str, data: bytes) -> None:
        nonlocal captured
        if captured is not None:
            return
        try:
            result = unpack_result(data)
        except (TypeError, ValueError):
            return
        stats["results"] += 1
        bundle = bundles.get(int(result.get("source_bundle_sequence", -1)))
        if bundle is None:
            return
        objects = [
            item for item in result.get("objects", [])
            if isinstance(item, dict)
            and str(item.get("camera_id", "")) == args.side
            and isinstance(item.get("mask_data"), (bytes, bytearray))
            and (args.label is None or str(item.get("label", "")) == args.label)
        ]
        if not objects:
            return
        objects.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        stats["matching_masks"] += len(objects)
        selected = _raw_mask_point(
            objects[0], bundle["cameras"][args.side], args.min_depth_m, args.max_depth_m
        )
        if selected is None:
            return
        point, count, pixel = selected
        stats["raw_depth_points"] = count
        captured = {
            "camera_point_m": point.tolist(),
            "wrist_point_m": wrist_point.astype(float).tolist(),
            "camera_id": args.side,
            "track_id": str(objects[0].get("track_id", "")),
            "label": str(objects[0].get("label", "")),
            "raw_mask_depth_points": count,
            "raw_mask_median_pixel_uv": pixel,
            "source_bundle_sequence": int(result.get("source_bundle_sequence", -1)),
        }

    bus.subscribe(WRIST_BUNDLE_CHANNEL, on_bundle)
    bus.subscribe(WRIST_RESULT_CHANNEL, on_result)
    deadline = time.monotonic() + max(0.1, float(args.timeout_sec))
    while captured is None and time.monotonic() < deadline:
        bus.handle_timeout(100)
    if captured is None:
        raise SystemExit(f"no raw calibration point; {stats}")
    observations = _load(args.output)
    observations.append(captured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump({"observations": observations}, sort_keys=False), encoding="utf-8")
    print(f"appended observation={len(observations)} camera_point_m={captured['camera_point_m']} raw_depth_points={captured['raw_mask_depth_points']} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
