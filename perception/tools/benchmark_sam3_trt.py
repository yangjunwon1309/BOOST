#!/usr/bin/env python3
"""Benchmark the live TensorRT SAM3 adapter on one image and text concept."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg_web_node.sam3_backend import TensorRTSam3Backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("/models/sam3_trt/sam3_fp16_4out.engine"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="benchmark direct TensorRT enqueue instead of CUDA Graph replay",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    backend = TensorRTSam3Backend(
        engine_path=args.engine,
        model_id="facebook/sam3",
        threshold=args.threshold,
        min_area_px=400,
        use_cuda_graph=not args.no_cuda_graph,
    )
    for _ in range(max(0, args.warmup)):
        backend.infer(image, [args.prompt])
    elapsed = []
    counts = []
    for _ in range(max(1, args.iterations)):
        started = time.perf_counter()
        masks, _boxes, _scores, _labels = backend.infer(image, [args.prompt])
        elapsed.append((time.perf_counter() - started) * 1000.0)
        counts.append(len(masks))
    print(f"backend={backend.metadata}")
    print(f"image={image.shape[1]}x{image.shape[0]} prompt={args.prompt!r}")
    print(
        f"latency_ms mean={statistics.mean(elapsed):.2f} "
        f"p50={percentile(elapsed, 0.50):.2f} "
        f"p95={percentile(elapsed, 0.95):.2f} min={min(elapsed):.2f}"
    )
    print(f"instances min={min(counts)} max={max(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
