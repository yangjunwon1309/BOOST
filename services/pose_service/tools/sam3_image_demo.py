#!/usr/bin/env python3
"""Run SAM3 text-prompt inference on one image and save an overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Text prompt, for example: red cup")
    parser.add_argument("--output", required=True, type=Path, help="Output overlay path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local SAM3 checkpoint; default uses the Hugging Face cache",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unreported warm-up inferences before timing (default: 1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of CUDA-synchronized timed inferences (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Input image does not exist: {args.image}")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be >= 0 and --iterations must be >= 1")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")

    import numpy as np
    import sam3
    import torch
    from PIL import Image, ImageDraw
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available inside the SAM3 container")

    sam3_package = Path(sam3.__file__).resolve().parent
    bpe_path = sam3_package / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    build_kwargs = {
        "bpe_path": str(bpe_path),
        "device": "cuda",
        "eval_mode": True,
    }
    if args.checkpoint is not None:
        build_kwargs.update(
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("[sam3-demo] Loading SAM3 model...")
    model_started = time.perf_counter()
    model = build_sam3_image_model(**build_kwargs)
    processor = Sam3Processor(
        model,
        device="cuda",
        confidence_threshold=args.threshold,
    )
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - model_started) * 1000.0

    image = Image.open(args.image).convert("RGB")
    print(
        f"[sam3-demo] Running prompt={args.prompt!r} "
        f"image={args.image} size={image.width}x{image.height}"
    )
    def infer_once():
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            encode_started = time.perf_counter()
            state = processor.set_image(image)
            torch.cuda.synchronize()
            image_encode_ms = (time.perf_counter() - encode_started) * 1000.0

            processor.reset_all_prompts(state)
            prompt_started = time.perf_counter()
            state = processor.set_text_prompt(prompt=args.prompt, state=state)
        torch.cuda.synchronize()
        text_prompt_ms = (time.perf_counter() - prompt_started) * 1000.0
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        return state, image_encode_ms, text_prompt_ms, inference_ms

    for _ in range(args.warmup):
        infer_once()
    torch.cuda.reset_peak_memory_stats()
    samples = [infer_once() for _ in range(args.iterations)]
    state = samples[-1][0]
    image_encode_times = np.asarray([sample[1] for sample in samples])
    text_prompt_times = np.asarray([sample[2] for sample in samples])
    inference_times = np.asarray([sample[3] for sample in samples])
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024.0**3)

    masks = state["masks"].detach().to(device="cpu").numpy()[:, 0]
    boxes = state["boxes"].detach().to(device="cpu").float().numpy()
    scores = state["scores"].detach().to(device="cpu").float().numpy()

    palette = np.asarray(
        [
            [255, 64, 64],
            [64, 220, 96],
            [64, 128, 255],
            [255, 190, 48],
            [190, 80, 255],
            [48, 220, 220],
        ],
        dtype=np.float32,
    )
    overlay = np.asarray(image, dtype=np.float32).copy()
    for index, mask in enumerate(masks):
        color = palette[index % len(palette)]
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45

    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        color = tuple(palette[index % len(palette)].astype(int).tolist())
        xyxy = tuple(float(value) for value in box)
        draw.rectangle(xyxy, outline=color, width=3)
        label = f"{args.prompt} {float(score):.3f}"
        x0, y0 = xyxy[:2]
        draw.text((x0 + 3, max(0.0, y0 - 14)), label, fill=color)
        print(
            f"[sam3-demo] instance={index} score={float(score):.4f} "
            f"box_xyxy={[round(float(value), 1) for value in box]} "
            f"mask_pixels={int(masks[index].sum())}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"[sam3-demo] instances={len(scores)}")
    print(f"[sam3-demo] model_load_ms={model_load_ms:.1f}")
    print(
        f"[sam3-demo] benchmark warmup={args.warmup} iterations={args.iterations}"
    )
    print(f"[sam3-demo] image_encode_mean_ms={image_encode_times.mean():.1f}")
    print(f"[sam3-demo] text_prompt_mean_ms={text_prompt_times.mean():.1f}")
    print(
        f"[sam3-demo] inference_ms mean={inference_times.mean():.1f} "
        f"median={np.median(inference_times):.1f} "
        f"p95={np.percentile(inference_times, 95):.1f} "
        f"min={inference_times.min():.1f} max={inference_times.max():.1f}"
    )
    print(f"[sam3-demo] peak_vram_gib={peak_vram_gib:.3f}")
    print(f"[sam3-demo] saved={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
