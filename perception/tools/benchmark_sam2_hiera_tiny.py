#!/usr/bin/env python3
"""Standalone GPU smoke/latency test for the official SAM2.1 Hiera-Tiny image model.

This deliberately has no LCM, web server, SONIC, or EE-publisher dependency.
It measures a fixed central point prompt on the same demo image used by the
existing perception demos.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("output/sam3_demo.jpg"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/models/sam2/sam2.1_hiera_tiny.pt")
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("output/sam2_hiera_tiny_smoke.jpg"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}; run tools/cache_sam2_hiera_tiny.py")
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"cannot read image: {args.image}")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    image_bgr = cv2.resize(image_bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(args.checkpoint),
        device="cuda",
        mode="eval",
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model)
    point = np.asarray([[image_rgb.shape[1] * 0.5, image_rgb.shape[0] * 0.56]], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int32)

    def infer() -> tuple[np.ndarray, np.ndarray]:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point, point_labels=labels, multimask_output=False
        )
        return masks, scores

    for _ in range(max(0, args.warmup)):
        infer()
    torch.cuda.synchronize()
    elapsed_ms: list[float] = []
    masks = scores = None
    for _ in range(max(1, args.iterations)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        masks, scores = infer()
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    assert masks is not None and scores is not None

    mask = masks[0].astype(bool)
    overlay = image_bgr.copy()
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 220, 0])).astype(np.uint8)
    cv2.circle(overlay, tuple(point[0].astype(int)), 5, (0, 0, 255), -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)
    report = {
        "model": "sam2.1-hiera-tiny",
        "device": torch.cuda.get_device_name(0),
        "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "point_uv": point[0].tolist(),
        "score": float(scores[0]),
        "mask_pixels": int(mask.sum()),
        "iterations": len(elapsed_ms),
        "latency_ms": {
            "mean": round(float(np.mean(elapsed_ms)), 3),
            "p50": round(percentile(elapsed_ms, 50), 3),
            "p95": round(percentile(elapsed_ms, 95), 3),
        },
        "hz_mean": round(1000.0 / float(np.mean(elapsed_ms)), 3),
        "overlay": str(args.output),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
