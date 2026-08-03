"""EfficientTAM-Ti-512 generic-mask backend sharing SAM2 RGB-D semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .sam2_backend import Sam2HieraTinyBackend


class EfficientTamTi512Backend(Sam2HieraTinyBackend):
    """EfficientTAM Tiny-512 with the same central discovery/box tracking API."""

    name = "efficienttam-ti-512"

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
        from efficient_track_anything.automatic_mask_generator import EfficientTAMAutomaticMaskGenerator
        from efficient_track_anything.build_efficienttam import build_efficienttam
        from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"EfficientTAM checkpoint does not exist: {checkpoint}")
        if reinitialization not in {"all", "central"}:
            raise ValueError("EfficientTAM supports reinitialization='all' or 'central'")
        self.torch = torch
        self.model = build_efficienttam(
            "configs/efficienttam/efficienttam_ti_512x512.yaml",
            str(checkpoint), device="cuda", mode="eval", apply_postprocessing=False,
        )
        self.predictor = EfficientTAMImagePredictor(self.model)
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
        self.automatic = EfficientTAMAutomaticMaskGenerator(self.model, **automatic_kwargs)
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
        metadata = super().metadata
        metadata.update(backend=self.name, model="efficienttam_ti_512x512")
        return metadata
