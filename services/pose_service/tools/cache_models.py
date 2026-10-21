#!/usr/bin/env python3
"""Download gated SAM3 and DINOv2 assets into the mounted persistent caches."""

from __future__ import annotations

import argparse
import gc

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    args = parser.parse_args()

    if not args.skip_sam3:
        # This no-argument call is the documented upstream API. It downloads the
        # gated checkpoint using the Hugging Face credentials in HF_HOME.
        from sam3.model_builder import build_sam3_image_model

        print("Caching the SAM3 image model checkpoint...")
        model = build_sam3_image_model()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_dinov2:
        print(f"Caching DINOv2 model {args.dinov2_model}...")
        model = torch.hub.load(
            "facebookresearch/dinov2",
            args.dinov2_model,
            trust_repo=True,
        )
        del model

    print("Model caches are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
