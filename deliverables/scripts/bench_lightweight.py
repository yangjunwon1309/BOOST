"""Benchmark one lightweight segmentation fallback at a time.

The script intentionally avoids installing packages. Install candidates inside
the container only after the relevant gate allows that phase.
"""

import argparse
import json
import time

import torch
from PIL import Image


def timed(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return {"median": ts[len(ts) // 2], "min": ts[0], "max": ts[-1]}


def bench_fastsam(image_path: str, prompt: str):
    from ultralytics import FastSAM

    model = FastSAM("FastSAM-s.pt")
    image = Image.open(image_path)
    return timed(lambda: model(image, device="cuda", retina_masks=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["fastsam"], required=True)
    parser.add_argument("--image", default="assets/test.jpg")
    parser.add_argument("--prompt", default="person")
    args = parser.parse_args()

    if args.candidate == "fastsam":
        latency = bench_fastsam(args.image, args.prompt)
    result = {
        "model": args.candidate,
        "prompt": args.prompt,
        "latency_s": latency,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
