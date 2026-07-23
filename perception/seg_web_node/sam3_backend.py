"""SAM3 inference backends for the live RGB-D server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import warnings

import cv2
import numpy as np


class TorchSam3Backend:
    name = "pytorch"

    def __init__(self, *, checkpoint: Path | None, threshold: float, min_area_px: int):
        import sam3
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        bpe_path = (
            Path(sam3.__file__).resolve().parent
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        )
        kwargs: dict[str, Any] = {
            "bpe_path": str(bpe_path),
            "device": "cuda",
            "eval_mode": True,
        }
        if checkpoint is not None:
            kwargs.update(checkpoint_path=str(checkpoint), load_from_HF=False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        self.processor = Sam3Processor(
            build_sam3_image_model(**kwargs),
            device="cuda",
            confidence_threshold=threshold,
        )
        self.min_area_px = min_area_px

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": self.name, "device": "cuda"}

    def infer(
        self, image_bgr: np.ndarray, concepts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks_parts, boxes_parts, scores_parts = [], [], []
        labels: list[str] = []
        with self.torch.inference_mode(), self.torch.autocast(
            "cuda", dtype=self.torch.bfloat16
        ):
            state = self.processor.set_image(Image.fromarray(image_rgb))
            for concept in concepts:
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt=concept, state=state)
                masks = state["masks"].detach().cpu().numpy()[:, 0]
                boxes = state["boxes"].detach().cpu().float().numpy()
                scores = state["scores"].detach().cpu().float().numpy()
                if len(masks):
                    keep = (
                        masks.reshape(len(masks), -1).sum(axis=1)
                        >= self.min_area_px
                    )
                    masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
                if len(masks):
                    masks_parts.append(masks)
                    boxes_parts.append(boxes)
                    scores_parts.append(scores)
                    labels.extend([concept] * len(masks))
        return _merge(masks_parts, boxes_parts, scores_parts, labels, image_bgr.shape[:2])


class TensorRTSam3Backend:
    name = "tensorrt"
    REQUIRED_INPUTS = {"pixel_values", "input_ids", "attention_mask"}
    REQUIRED_OUTPUTS = {
        "pred_masks",
        "pred_boxes",
        "pred_logits",
        "presence_logits",
    }

    def __init__(
        self,
        *,
        engine_path: Path,
        model_id: str,
        threshold: float,
        min_area_px: int,
        mask_threshold: float = 0.5,
        use_cuda_graph: bool | None = None,
    ) -> None:
        import tensorrt as trt
        import torch
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        engine_path = engine_path.expanduser().resolve()
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        self.trt = trt
        self.torch = torch
        self.engine_path = engine_path
        self.threshold = float(threshold)
        self.mask_threshold = float(mask_threshold)
        self.min_area_px = int(min_area_px)
        if use_cuda_graph is None:
            use_cuda_graph = os.environ.get("G1_SAM3_CUDA_GRAPH", "1").lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        self._cuda_graph_requested = bool(use_cuda_graph)
        self._cuda_graph = None
        self._cuda_graph_state = (
            "pending" if self._cuda_graph_requested else "disabled"
        )
        self._cuda_graph_error: str | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self.stream = torch.cuda.Stream()
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if not shape or any(dim <= 0 for dim in shape):
                raise ValueError(f"TensorRT engine tensor {name} has dynamic/invalid shape {shape}")
            dtype = _torch_dtype(trt, torch, self.engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT tensor: {name}")
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = tensor
            else:
                self.outputs[name] = tensor
        missing_inputs = self.REQUIRED_INPUTS - self.inputs.keys()
        missing_outputs = self.REQUIRED_OUTPUTS - self.outputs.keys()
        if missing_inputs or missing_outputs:
            raise ValueError(
                f"incompatible TensorRT engine; missing inputs={sorted(missing_inputs)} "
                f"outputs={sorted(missing_outputs)}"
            )
        self.image_size = tuple(self.inputs["pixel_values"].shape[-2:])

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine": str(self.engine_path),
            "tensorrt_version": self.trt.__version__,
            "image_size": list(self.image_size),
            "cuda_graph": self._cuda_graph_state,
            "cuda_graph_error": self._cuda_graph_error,
        }

    def _pixels(self, image_bgr: np.ndarray):
        torch = self.torch
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pixels = torch.from_numpy(image_rgb).to(device="cuda", dtype=torch.float32)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        pixels = torch.nn.functional.interpolate(
            pixels, size=self.image_size, mode="bilinear", align_corners=False
        )
        return pixels.sub_(0.5).div_(0.5).contiguous()

    def _execute(self, pixels, concept: str) -> tuple[Any, Any, Any, Any]:
        torch = self.torch
        tokens = self.tokenizer(
            concept,
            return_tensors="pt",
            padding="max_length",
            max_length=self.inputs["input_ids"].shape[-1],
            truncation=True,
        )
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            self.inputs["pixel_values"].copy_(pixels)
            self.inputs["input_ids"].copy_(
                tokens["input_ids"].to(
                    device="cuda", dtype=self.inputs["input_ids"].dtype
                )
            )
            self.inputs["attention_mask"].copy_(
                tokens["attention_mask"].to(
                    device="cuda", dtype=self.inputs["attention_mask"].dtype
                )
            )
        self.stream.synchronize()

        if self._cuda_graph_state == "pending":
            self._capture_cuda_graph()

        with torch.cuda.stream(self.stream):
            if self._cuda_graph is not None:
                self._cuda_graph.replay()
            elif not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT SAM3 execution failed")
        self.stream.synchronize()
        return (
            self.outputs["pred_masks"],
            self.outputs["pred_boxes"],
            self.outputs["pred_logits"],
            self.outputs["presence_logits"],
        )

    def _capture_cuda_graph(self) -> None:
        """Warm up TensorRT once, then capture its fixed-shape enqueue path."""

        torch = self.torch
        try:
            # TensorRT may lazily allocate resources on its first enqueue. That
            # work is not capture-safe, so complete one normal execution first.
            with torch.cuda.stream(self.stream):
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT SAM3 warm-up execution failed")
            self.stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT SAM3 graph capture enqueue failed")
            self._cuda_graph = graph
            self._cuda_graph_state = "captured"
        except Exception as exc:
            # CUDA Graph support varies with TensorRT tactics/plugins. Preserve
            # the proven execute_async_v3 path when any operation is not
            # capture-safe.
            self._cuda_graph = None
            self._cuda_graph_state = "fallback"
            self._cuda_graph_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"SAM3 CUDA Graph capture failed; using direct TensorRT enqueue: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def infer(
        self, image_bgr: np.ndarray, concepts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        torch = self.torch
        height, width = image_bgr.shape[:2]
        pixels = self._pixels(image_bgr)
        masks_parts, boxes_parts, scores_parts = [], [], []
        labels: list[str] = []
        with torch.inference_mode():
            for concept in concepts:
                mask_logits, boxes, logits, presence = self._execute(pixels, concept)
                scores = logits.sigmoid() * presence.sigmoid()
                keep = scores[0] > self.threshold
                scores = scores[0, keep]
                boxes = boxes[0, keep]
                masks = mask_logits[0, keep]
                if len(masks):
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    mask_cutoff = float(np.log(self.mask_threshold / (1 - self.mask_threshold)))
                    masks = masks > mask_cutoff
                    areas = masks.flatten(1).sum(1)
                    area_keep = areas >= self.min_area_px
                    masks, boxes, scores = (
                        masks[area_keep],
                        boxes[area_keep],
                        scores[area_keep],
                    )
                if len(masks):
                    scale = torch.tensor(
                        [width, height, width, height], device=boxes.device
                    )
                    masks_parts.append(masks.cpu().numpy())
                    boxes_parts.append((boxes * scale).float().cpu().numpy())
                    scores_parts.append(scores.float().cpu().numpy())
                    labels.extend([concept] * len(masks))
        return _merge(masks_parts, boxes_parts, scores_parts, labels, (height, width))


def _torch_dtype(trt, torch, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT tensor dtype: {dtype}")
    return mapping[dtype]


def _merge(masks_parts, boxes_parts, scores_parts, labels, image_shape):
    if masks_parts:
        return (
            np.concatenate(masks_parts),
            np.concatenate(boxes_parts),
            np.concatenate(scores_parts),
            labels,
        )
    height, width = image_shape
    return (
        np.empty((0, height, width), dtype=bool),
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        [],
    )
