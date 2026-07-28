#!/usr/bin/env python3
"""Measure EfficientTAM Tiny-512 central initialization and box tracking."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_web_node.efficienttam_backend import EfficientTamTi512Backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("output/sam3_demo.jpg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/models/efficienttam/efficienttam_ti_512x512.pt"))
    parser.add_argument("--track-iterations", type=int, default=20)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}; run tools/cache_efficienttam_ti_512.py")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    backend = EfficientTamTi512Backend(
        checkpoint=args.checkpoint, threshold=0.50, min_area_px=400,
        reinitialization="central", reinitialize_interval_frames=1000000, max_masks=12,
        central={"reference_width_px": 640, "reference_height_px": 360,
                 "roi_center_px": [320, 200], "roi_radius_px": [220, 125],
                 "prompt_spacing_px": 50, "lower_region_start_fraction": .60,
                 "lower_region_mask_fraction": .80, "merge_overlap_min_fraction": .50},
    )
    import torch
    torch.cuda.synchronize(); started = time.perf_counter()
    masks, _boxes, scores, _labels = backend.infer(image, ["object"])
    torch.cuda.synchronize()
    initialize_ms = (time.perf_counter() - started) * 1000.0
    track_ms: list[float] = []
    for _ in range(max(1, args.track_iterations)):
        torch.cuda.synchronize(); started = time.perf_counter()
        masks, _boxes, scores, _labels = backend.infer(image, ["object"])
        torch.cuda.synchronize()
        track_ms.append((time.perf_counter() - started) * 1000.0)
    print(json.dumps({
        "model": "efficienttam_ti_512x512", "image_size": [640, 360],
        "central_prompts": backend.last_central_prompt_count,
        "masks": int(len(masks)), "scores": scores.round(4).tolist(),
        "central_initialize_ms": round(initialize_ms, 3),
        "track_ms": {"mean": round(float(np.mean(track_ms)), 3),
                     "p50": round(float(np.percentile(track_ms, 50)), 3),
                     "p95": round(float(np.percentile(track_ms, 95)), 3)},
        "track_hz_mean": round(1000.0 / float(np.mean(track_ms)), 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
