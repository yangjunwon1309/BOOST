#!/usr/bin/env python3
"""Verify cached DINOv2 patch-token inference on the Thor GPU."""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = torch.hub.load(
        "facebookresearch/dinov2",
        args.model,
        trust_repo=True,
    ).eval().cuda()
    image = torch.rand(1, 3, 224, 224, device="cuda")

    with torch.inference_mode():
        features = model.forward_features(image)
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            features = model.forward_features(image)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

    patch_tokens = features["x_norm_patchtokens"]
    if not torch.isfinite(patch_tokens).all().item():
        raise RuntimeError("DINOv2 returned non-finite patch tokens")

    print(
        json.dumps(
            {
                "model": args.model,
                "patch_tokens": list(patch_tokens.shape),
                "embedding_dim": patch_tokens.shape[-1],
                "latency_ms_mean": sum(timings) / len(timings),
                "latency_ms_min": min(timings),
                "iterations": len(timings),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
