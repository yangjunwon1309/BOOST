#!/usr/bin/env python3
"""Benchmark the COCO-pretrained YOLO11n-seg backend on one RGB frame."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_web_node.yolo11_backend import Yolo11SegBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("output/sam3_demo.jpg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/models/yolo11/yolo11n-seg.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.80)
    parser.add_argument(
        "--concepts", default="object",
        help="Comma-separated exact COCO labels, or object/all for all non-person classes.",
    )
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    if not args.checkpoint.is_file():
        raise SystemExit("missing YOLO11 checkpoint; run tools/cache_yolo11n_seg.py")
    backend = Yolo11SegBackend(
        checkpoint=args.checkpoint,
        threshold=0.0,
        min_area_px=100,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou_threshold=0.5,
        max_masks=20,
        quantize=16,
        track={"missing_ttl_frames": 0},
        excluded_classes=["person"],
    )
    import torch
    # One warm-up excludes CUDA/kernel/model initialization from steady state.
    concepts = [item.strip() for item in args.concepts.split(",") if item.strip()]
    backend.infer(image, concepts)
    torch.cuda.synchronize()
    elapsed_ms: list[float] = []
    masks = np.empty((0, *image.shape[:2]), dtype=bool)
    labels: list[str] = []
    scores = np.empty(0, dtype=np.float32)
    for _ in range(max(1, args.iterations)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        masks, _boxes, scores, labels = backend.infer(image, concepts)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    print(json.dumps({
        "model": "yolo11n-seg COCO-80",
        "input_size": [int(image.shape[1]), int(image.shape[0])],
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "concepts": concepts,
        "resolved_coco_labels": backend.last_resolved_labels,
        "unmatched_concepts": backend.last_unmatched_concepts,
        "masks": int(len(masks)),
        "labels": labels,
        "scores": scores.round(4).tolist(),
        "latency_ms": {
            "mean": round(float(np.mean(elapsed_ms)), 3),
            "p50": round(float(np.percentile(elapsed_ms, 50)), 3),
            "p95": round(float(np.percentile(elapsed_ms, 95)), 3),
        },
        "hz_mean": round(1000.0 / float(np.mean(elapsed_ms)), 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
