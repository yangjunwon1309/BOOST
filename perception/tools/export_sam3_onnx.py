#!/usr/bin/env python3
"""Export the Hugging Face SAM3 image model for the TensorRT live backend.

This follows dataplayer12/SAM3-TensorRT commit
502b13d09d64d46e8f440b4c9caa99b4a8040eef, but retains every output required
for correct instance filtering and boxes in the live perception pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from PIL import Image
from transformers.models.sam3 import Sam3Model, Sam3Processor


OUTPUT_NAMES = (
    "pred_masks",
    "pred_boxes",
    "pred_logits",
    "presence_logits",
)


class Sam3OnnxWrapper(torch.nn.Module):
    def __init__(self, model: Sam3Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask):
        output = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return (
            output.pred_masks,
            output.pred_boxes,
            output.pred_logits,
            output.presence_logits,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/models/sam3_trt/runtime4/sam3_4out.onnx"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing ONNX graph: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SAM3-ONNX] loading {args.model_id} on {args.device}", flush=True)
    model = Sam3Model.from_pretrained(args.model_id).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.model_id)
    sample = Image.new("RGB", (640, 360), color=(96, 128, 160))
    inputs = processor(images=sample, text="object", return_tensors="pt")
    example = tuple(
        inputs[name].to(args.device)
        for name in ("pixel_values", "input_ids", "attention_mask")
    )
    print(
        "[SAM3-ONNX] inputs "
        + ", ".join(f"{tuple(value.shape)} {value.dtype}" for value in example),
        flush=True,
    )
    wrapper = Sam3OnnxWrapper(model).to(args.device).eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example,
            str(output),
            input_names=["pixel_values", "input_ids", "attention_mask"],
            output_names=list(OUTPUT_NAMES),
            opset_version=args.opset,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )

    graph = onnx.load(str(output), load_external_data=False)
    actual = tuple(item.name for item in graph.graph.output)
    if actual != OUTPUT_NAMES:
        raise RuntimeError(f"unexpected ONNX outputs: {actual}")
    print(f"[SAM3-ONNX] exported {output}", flush=True)
    for path in sorted(output.parent.glob(output.name + "*")):
        print(f"[SAM3-ONNX] artifact {path.name} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
