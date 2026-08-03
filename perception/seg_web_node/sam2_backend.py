"""Official SAM2.1 Hiera-Tiny generic-mask backend for the RGB-D pipeline.

SAM2 is used only for mask generation here.  Semantic selection remains the
robot-state/depth/belief responsibility of PATH, so every instance is labelled
``object`` and receives its persistent identity after RGB-D fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .efficientvit_backend import (
    _box_from_mask,
    central_point_grid,
)
from .depth_based_filter import DepthSelfFilterBackendMixin


class Sam2HieraTinyBackend(DepthSelfFilterBackendMixin):
    """SAM2.1 Tiny with periodic generic discovery and box-prompt tracking.

    The image predictor recomputes the image embedding per live RGB frame.
    Between generic discovery passes, each current object is refined from its
    own prior box, rather than collapsing all masks to one object.  Native
    SAM2 video-memory propagation is intentionally a separate next step: its
    file/video-state API needs a bounded live-frame adapter, while this backend
    first verifies the official image model and full RGB-D integration safely.
    """

    name = "sam2.1-hiera-tiny"

    def __init__(
        self,
        *,
        checkpoint: Path,
        threshold: float,
        min_area_px: int,
        reinitialization: str = "central",
        reinitialize_interval_frames: int = 15,
        automatic_points_per_side: int = 8,
        max_masks: int = 12,
        central: dict[str, Any] | None = None,
        depth_self_filter: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
        if reinitialization not in {"all", "central"}:
            raise ValueError("SAM2 supports reinitialization='all' or 'central'")
        self.torch = torch
        self.model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(checkpoint),
            device="cuda",
            mode="eval",
            apply_postprocessing=False,
        )
        self.predictor = SAM2ImagePredictor(self.model)
        automatic_kwargs: dict[str, Any] = {
            "points_per_batch": 64,
            "pred_iou_thresh": float(threshold),
            "stability_score_thresh": 0.90,
            "min_mask_region_area": int(min_area_px),
            "output_mode": "binary_mask",
        }
        if reinitialization == "central":
            automatic_kwargs.update(points_per_side=None, point_grids=[np.asarray([[0.5, 0.5]], dtype=np.float32)])
        else:
            automatic_kwargs["points_per_side"] = max(1, int(automatic_points_per_side))
        self.automatic = SAM2AutomaticMaskGenerator(self.model, **automatic_kwargs)
        self.checkpoint = checkpoint
        self.threshold = float(threshold)
        self.min_area_px = int(min_area_px)
        self.reinitialization = reinitialization
        self.reinitialize_interval_frames = max(1, int(reinitialize_interval_frames))
        self.max_masks = max(1, int(max_masks))
        central = dict(central or {})
        self.central_reference_width = float(central.get("reference_width_px", 640))
        self.central_reference_height = float(central.get("reference_height_px", 360))
        self.central_center_px = tuple(float(v) for v in central.get("roi_center_px", [320, 200]))
        self.central_radius_px = tuple(float(v) for v in central.get("roi_radius_px", [220, 125]))
        self.central_spacing_px = float(central.get("prompt_spacing_px", 50))
        self.track_missing_ttl_frames = max(0, int(central.get("track_missing_ttl_frames", 3)))
        self.track_fallback_score_decay = float(central.get("track_fallback_score_decay", 0.985))
        self.reinitialize_match_iou = float(central.get("reinitialize_match_iou", 0.15))
        self._init_depth_self_filter(depth_self_filter)
        self.frame_index = 0
        self.track_priors: list[dict[str, Any]] = []
        self.last_mode = "uninitialized"
        self.last_central_prompt_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": "sam2.1_hiera_tiny",
            "checkpoint": str(self.checkpoint),
            "device": "cuda",
            "reinitialization": self.reinitialization,
            "reinitialize_interval_frames": self.reinitialize_interval_frames,
            "last_mode": self.last_mode,
            "active_prompt_tracks": sum(int(track.get("missing", 0)) == 0 for track in self.track_priors),
            "fallback_prompt_tracks": sum(int(track.get("missing", 0)) > 0 for track in self.track_priors),
            "central_prompt_count": self.last_central_prompt_count,
            "depth_self_filter": self._depth_self_filter_metadata(),
        }

    @staticmethod
    def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
        union = int(np.logical_or(first, second).sum())
        return 0.0 if union == 0 else float(np.logical_and(first, second).sum()) / union

    def _discover(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        height, width = image_rgb.shape[:2]
        if self.reinitialization == "central":
            points = central_point_grid(
                width,
                height,
                reference_width=self.central_reference_width,
                reference_height=self.central_reference_height,
                center_px=self.central_center_px,
                radius_px=self.central_radius_px,
                spacing_px=self.central_spacing_px,
            )
            self.automatic.point_grids = [points]
            self.last_central_prompt_count = int(len(points))
        entries = self.automatic.generate(image_rgb)
        candidates: list[tuple[np.ndarray, float]] = []
        for entry in entries:
            mask = self._apply_depth_self_filter(np.asarray(entry["segmentation"], dtype=bool))
            score = float(entry.get("predicted_iou", 0.0))
            if mask is not None and score >= self.threshold and int(mask.sum()) >= self.min_area_px:
                candidates.append((mask, score))
        candidates.sort(key=lambda value: (value[1], int(value[0].sum())), reverse=True)
        return candidates[: self.max_masks]

    def _track(self) -> list[tuple[np.ndarray, float]]:
        output: list[tuple[np.ndarray, float]] = []
        updated: list[dict[str, Any]] = []
        for track in self.track_priors:
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(track["box"], dtype=np.float32), multimask_output=False
            )
            if len(masks) and float(scores[0]) >= self.threshold and int(masks[0].sum()) >= self.min_area_px:
                mask = self._apply_depth_self_filter(np.asarray(masks[0], dtype=bool))
                box = None if mask is None else _box_from_mask(mask)
                if box is not None:
                    updated.append({"box": box, "mask": mask, "score": float(scores[0]), "missing": 0})
                    output.append((mask, float(scores[0])))
                    continue
            missing = int(track.get("missing", 0)) + 1
            previous_mask = track.get("mask")
            if missing <= self.track_missing_ttl_frames and previous_mask is not None:
                fallback_score = float(track.get("score", self.threshold)) * (
                    self.track_fallback_score_decay ** missing
                )
                updated.append({**track, "missing": missing, "score": fallback_score})
                output.append((np.asarray(previous_mask, dtype=bool), fallback_score))
        self.track_priors = updated[: self.max_masks]
        return output

    def _reinitialize(self, image_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Refresh candidates without deleting a temporarily missed object track."""
        discovered = self._discover(image_rgb)
        previous = list(self.track_priors)
        unmatched_previous = set(range(len(previous)))
        refreshed: list[dict[str, Any]] = []
        for mask, score in discovered:
            best_index = None
            best_iou = self.reinitialize_match_iou
            for index in unmatched_previous:
                prior_mask = previous[index].get("mask")
                if prior_mask is None:
                    continue
                overlap = self._mask_iou(mask, np.asarray(prior_mask, dtype=bool))
                if overlap >= best_iou:
                    best_iou, best_index = overlap, index
            if best_index is not None:
                unmatched_previous.remove(best_index)
            box = _box_from_mask(mask)
            if box is not None:
                refreshed.append({"box": box, "mask": mask, "score": float(score), "missing": 0})
        # Discovery can miss a real object for several seconds. Preserve its
        # last valid mask until tracking or a later discovery re-acquires it.
        for index in unmatched_previous:
            prior = previous[index]
            missing = int(prior.get("missing", 0)) + 1
            if missing <= self.track_missing_ttl_frames and prior.get("mask") is not None:
                refreshed.append(
                    {
                        **prior,
                        "missing": missing,
                        "score": float(prior.get("score", self.threshold))
                        * (self.track_fallback_score_decay ** missing),
                    }
                )
        refreshed.sort(key=lambda item: (int(item["missing"]) > 0, -float(item["score"])))
        self.track_priors = refreshed[: self.max_masks]
        return [
            (np.asarray(track["mask"], dtype=bool), float(track["score"]))
            for track in self.track_priors
        ]

    def infer(
        self,
        image_bgr: np.ndarray,
        _concepts: list[str],
        *,
        depth_raw: np.ndarray | None = None,
        depth_scale_um: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        del depth_raw, depth_scale_um
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        reset = not self.track_priors or self.frame_index % self.reinitialize_interval_frames == 0
        if reset:
            candidates = self._reinitialize(image_rgb)
            self.last_mode = "central_reinitialize" if self.reinitialization == "central" else "all_reinitialize"
        else:
            candidates = self._track()
            self.last_mode = "box_track"
        self.frame_index += 1
        height, width = image_bgr.shape[:2]
        if not candidates:
            return (
                np.empty((0, height, width), dtype=bool),
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                [],
            )
        masks = np.stack([mask for mask, _ in candidates], axis=0)
        scores = np.asarray([score for _, score in candidates], dtype=np.float32)
        boxes = np.stack([_box_from_mask(mask) for mask in masks], axis=0).astype(np.float32)
        return masks, boxes, scores, ["object"] * len(masks)
