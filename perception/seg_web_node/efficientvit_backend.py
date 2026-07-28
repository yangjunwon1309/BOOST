"""EfficientViT-SAM-L0 generic RGB-D segmentation backend.

This adapter deliberately has no text classifier.  Every output is labelled
``object``; :class:`StableObjectIdTracker` assigns its persistent 3-D identity
after the live server deprojects the mask with the synchronized depth image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_WEIGHT_URL = (
    "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/"
    "efficientvit_sam_l0.pt"
)


def _empty(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    return (
        np.empty((0, height, width), dtype=bool),
        np.empty((0, 4), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        [],
    )


def _box_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    return 0.0 if union == 0 else float(np.logical_and(first, second).sum()) / union


def central_point_grid(
    width: int,
    height: int,
    *,
    reference_width: float,
    reference_height: float,
    center_px: tuple[float, float],
    radius_px: tuple[float, float],
    spacing_px: float,
) -> np.ndarray:
    """Return normalized SAM points on an ellipse in the current image.

    The YAML values are specified in pixels for a reference 640x360 camera.
    Scaling them here preserves the same camera-view cone for another stream
    resolution while SAM itself still receives the full RGB image.
    """
    sx, sy = width / float(reference_width), height / float(reference_height)
    cx, cy = float(center_px[0]) * sx, float(center_px[1]) * sy
    rx, ry = max(1.0, float(radius_px[0]) * sx), max(1.0, float(radius_px[1]) * sy)
    spacing = max(1.0, float(spacing_px) * min(sx, sy))
    points: list[tuple[float, float]] = []
    ys = np.arange(cy - ry, cy + ry + 0.5 * spacing, spacing)
    xs = np.arange(cx - rx, cx + rx + 0.5 * spacing, spacing)
    for y in ys:
        for x in xs:
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                points.append((float(np.clip(x / width, 0.0, 1.0)), float(np.clip(y / height, 0.0, 1.0))))
    return np.asarray(points, dtype=np.float32)


def filter_and_merge_central_masks(
    candidates: list[tuple[np.ndarray, float]],
    *,
    lower_region_start_fraction: float,
    lower_region_mask_fraction: float,
    merge_overlap_min_fraction: float,
    max_masks: int,
) -> list[tuple[np.ndarray, float]]:
    """Reject lower-view masks and merge duplicate object proposals.

    A floor/robot-body mask normally occupies most of the lower image.  A
    candidate is rejected only when at least ``lower_region_mask_fraction`` of
    *its own pixels* lie below ``lower_region_start_fraction`` of image height;
    therefore objects extending slightly into the lower region remain valid.

    SAM often returns nested masks for one object. IoU alone misses nested
    pairs, so the graph edge is intersection / min(mask area). Connected
    components of that graph are duplicate clusters; their masks are unioned.
    ``scipy.sparse.csgraph`` is used rather than a slower pixel clustering pass.
    """
    if not candidates:
        return []
    height = candidates[0][0].shape[0]
    lower_start = int(np.clip(round(height * lower_region_start_fraction), 0, height))
    retained: list[tuple[np.ndarray, float]] = []
    for mask, score in candidates:
        area = int(mask.sum())
        if area == 0:
            continue
        lower_coverage = float(mask[lower_start:, :].sum()) / area
        if lower_coverage >= lower_region_mask_fraction:
            continue
        retained.append((mask, score))
    if not retained:
        return []

    count = len(retained)
    adjacency = np.eye(count, dtype=np.uint8)
    areas = np.asarray([int(mask.sum()) for mask, _ in retained], dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            intersection = int(np.logical_and(retained[first][0], retained[second][0]).sum())
            coverage = intersection / max(1.0, min(areas[first], areas[second]))
            if coverage >= merge_overlap_min_fraction:
                adjacency[first, second] = adjacency[second, first] = 1

    # The candidate count is bounded (default 12), but sparse connected
    # components makes the intended duplicate clustering explicit and scalable.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    _, cluster_ids = connected_components(csr_matrix(adjacency), directed=False)
    merged: list[tuple[np.ndarray, float]] = []
    for cluster_id in np.unique(cluster_ids):
        indices = np.flatnonzero(cluster_ids == cluster_id)
        union = np.zeros_like(retained[0][0], dtype=bool)
        score = 0.0
        for index in indices:
            union |= retained[int(index)][0]
            score = max(score, float(retained[int(index)][1]))
        merged.append((union, score))
    merged.sort(key=lambda item: (item[1], int(item[0].sum())), reverse=True)
    return merged[:max_masks]


class EfficientViTSamL0Backend:
    """Prompted L0 backend with inexpensive tracking between reinitializations.

    ``all`` runs SAM's automatic-mask generator periodically. ``depth-only``
    obtains connected depth regions and refines each region using a point/box
    SAM prompt. ``central`` is an all-mask generator whose point grid is
    limited to the configured central camera-view ellipse.

    Reinitialization discovers multiple generic objects. Between resets every
    discovered object retains its own box prior and is decoded in one batched
    SAM decoder pass. This is intentionally different from the older single
    ``last_box`` path, which caused all but one object to disappear.
    """

    name = "efficientvit-l0"

    def __init__(
        self,
        *,
        threshold: float,
        min_area_px: int,
        weight_path: Path | None = None,
        weight_url: str = DEFAULT_WEIGHT_URL,
        reinitialization: str = "all",
        reinitialize_interval_frames: int = 15,
        automatic_points_per_side: int = 8,
        max_masks: int = 12,
        central: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from efficientvit.models.efficientvit.sam import (
            EfficientViTSamAutomaticMaskGenerator,
            EfficientViTSamPredictor,
        )
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if reinitialization not in {"all", "depth-only", "central"}:
            raise ValueError("reinitialization must be 'all', 'depth-only', or 'central'")
        source = str(weight_path.expanduser().resolve()) if weight_path else weight_url
        self.torch = torch
        self.model = create_efficientvit_sam_model(
            name="efficientvit-sam-l0", pretrained=True, weight_url=source
        ).cuda().eval()
        self.predictor = EfficientViTSamPredictor(self.model)
        automatic_kwargs: dict[str, Any] = {
            "points_per_batch": 64,
            "pred_iou_thresh": float(threshold),
            "stability_score_thresh": 0.90,
            "min_mask_region_area": int(min_area_px),
        }
        if reinitialization == "central":
            # This placeholder is replaced with resolution-correct points at
            # each central reset. The upstream constructor requires exactly
            # one of points_per_side or point_grids.
            automatic_kwargs["points_per_side"] = None
            automatic_kwargs["point_grids"] = [np.asarray([[0.5, 0.5]], dtype=np.float32)]
        else:
            automatic_kwargs["points_per_side"] = int(automatic_points_per_side)
        self.automatic = EfficientViTSamAutomaticMaskGenerator(self.model, **automatic_kwargs)
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
        self.central_max_mask_area_ratio = float(central.get("max_mask_area_ratio", 0.35))
        self.central_lower_region_start_fraction = float(
            central.get("lower_region_start_fraction", 0.60)
        )
        self.central_lower_region_mask_fraction = float(
            central.get("lower_region_mask_fraction", 0.80)
        )
        self.central_merge_overlap_min_fraction = float(
            central.get("merge_overlap_min_fraction", 0.50)
        )
        self.track_missing_ttl_frames = max(0, int(central.get("track_missing_ttl_frames", 3)))
        self.frame_index = 0
        self.track_priors: list[dict[str, Any]] = []
        self.last_central_prompt_count = 0
        self.robot_prompt_point: np.ndarray | None = None
        self.robot_prompt_box: np.ndarray | None = None
        self.last_mode = "uninitialized"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": "efficientvit-sam-l0",
            "device": "cuda",
            "image_size": list(self.model.image_size),
            "reinitialization": self.reinitialization,
            "reinitialize_interval_frames": self.reinitialize_interval_frames,
            "last_mode": self.last_mode,
            "active_prompt_tracks": sum(1 for track in self.track_priors if int(track["missing"]) == 0),
            "central_prompt_count": self.last_central_prompt_count,
            "central_filter": {
                "lower_region_start_fraction": self.central_lower_region_start_fraction,
                "lower_region_mask_fraction": self.central_lower_region_mask_fraction,
                "merge_overlap_min_fraction": self.central_merge_overlap_min_fraction,
            },
        }

    def set_robot_state_prompt(
        self, point_uv: np.ndarray | list[float] | None, box_xyxy: np.ndarray | list[float] | None = None
    ) -> None:
        """Set the next-frame projected wrist point/box prompt.

        Values outside the image are rejected by :meth:`infer`, because an EE
        behind the camera must not produce a synthetic prompt at an edge.
        """
        self.robot_prompt_point = (
            None if point_uv is None else np.asarray(point_uv, dtype=np.float32).reshape(2)
        )
        self.robot_prompt_box = (
            None if box_xyxy is None else np.asarray(box_xyxy, dtype=np.float32).reshape(4)
        )

    def _depth_proposals(self, depth_raw: np.ndarray | None, scale_um: int | None) -> list[tuple[np.ndarray, np.ndarray]]:
        if depth_raw is None or scale_um is None or depth_raw.ndim != 2:
            return []
        depth_m = depth_raw.astype(np.float32) * (float(scale_um) / 1_000_000.0)
        valid = (depth_m > 0.15) & (depth_m < 3.0)
        proposals: list[tuple[np.ndarray, np.ndarray]] = []
        # 8 cm slices avoid merging a foreground object with its background.
        quantized = np.floor(depth_m / 0.08).astype(np.int32)
        for bin_id in np.unique(quantized[valid]):
            component_input = ((quantized == bin_id) & valid).astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(component_input, 8)
            for index in range(1, count):
                area = int(stats[index, cv2.CC_STAT_AREA])
                if area < self.min_area_px:
                    continue
                x, y, width, height, _ = stats[index]
                if width * height > depth_raw.size * 0.85:
                    continue
                mask = labels == index
                point = np.asarray(centroids[index], dtype=np.float32)
                box = np.asarray([x, y, x + width, y + height], dtype=np.float32)
                proposals.append((point, box))
        proposals.sort(key=lambda proposal: float((proposal[1][2] - proposal[1][0]) * (proposal[1][3] - proposal[1][1])), reverse=True)
        return proposals[: self.max_masks]

    def _prompt_mask(self, point: np.ndarray | None, box: np.ndarray | None) -> tuple[np.ndarray, float] | None:
        if point is None and box is None:
            return None
        points = None if point is None else np.asarray(point, dtype=np.float32).reshape(1, 2)
        labels = None if points is None else np.ones(1, dtype=np.int32)
        masks, scores, _ = self.predictor.predict(
            point_coords=points, point_labels=labels, box=box, multimask_output=False
        )
        if not len(masks):
            return None
        return masks[0].astype(bool), float(scores[0])

    def _tracked_masks(self, width: int, height: int) -> list[tuple[np.ndarray, float]]:
        """Refine every live track from one image embedding and batched prompts."""
        stale = [
            {"box": track["box"], "missing": int(track["missing"]) + 1}
            for track in self.track_priors
            if 0 < int(track["missing"]) < self.track_missing_ttl_frames
        ]
        live = [track for track in self.track_priors if int(track["missing"]) == 0]
        if not live:
            return []
        boxes = np.stack([np.asarray(track["box"], dtype=np.float32) for track in live], axis=0)
        robot_point = self.robot_prompt_point
        robot_index: int | None = None
        if robot_point is not None and 0 <= robot_point[0] < width and 0 <= robot_point[1] < height:
            centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
            robot_index = int(np.argmin(np.linalg.norm(centers - robot_point.reshape(1, 2), axis=1)))

        torch = self.torch
        transformed_boxes = self.predictor.apply_boxes(boxes)
        box_torch = torch.as_tensor(transformed_boxes, dtype=torch.float32, device=self.predictor.device)
        points = np.zeros((len(live), 1, 2), dtype=np.float32)
        point_labels = np.full((len(live), 1), -1, dtype=np.int32)
        if robot_index is not None:
            points[robot_index, 0] = self.predictor.apply_coords(robot_point.reshape(1, 2))[0]
            point_labels[robot_index, 0] = 1
        with torch.inference_mode():
            point_torch = torch.as_tensor(points, dtype=torch.float32, device=self.predictor.device)
            label_torch = torch.as_tensor(point_labels, dtype=torch.int64, device=self.predictor.device)
            sparse, dense = self.model.prompt_encoder(
                points=(point_torch, label_torch), boxes=box_torch, masks=None
            )
            low_masks, iou = self.model.mask_decoder(
                image_embeddings=self.predictor.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            masks = self.model.postprocess_masks(
                low_masks, self.predictor.input_size, self.predictor.original_size
            ) > self.model.mask_threshold
        masks_np = masks[:, 0].detach().cpu().numpy().astype(bool)
        scores_np = iou[:, 0].detach().cpu().numpy().astype(np.float32)

        output: list[tuple[np.ndarray, float]] = []
        retained: list[dict[str, Any]] = []
        for track, mask, score in zip(live, masks_np, scores_np):
            box = _box_from_mask(mask)
            if box is not None and score >= self.threshold and int(mask.sum()) >= self.min_area_px:
                retained.append({"box": box, "missing": 0})
                output.append((mask, float(score)))
            elif int(track["missing"]) < self.track_missing_ttl_frames:
                retained.append({"box": track["box"], "missing": int(track["missing"]) + 1})
        # Keep an unmatched prior for its configured TTL; it remains hidden
        # from this frame but is available if a future reinitialization finds
        # the same 3-D object again.
        self.track_priors = (retained + stale)[: self.max_masks]
        return output

    def _mask_center_in_central_roi(self, mask: np.ndarray) -> bool:
        height, width = mask.shape
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return False
        sx, sy = width / self.central_reference_width, height / self.central_reference_height
        cx, cy = self.central_center_px[0] * sx, self.central_center_px[1] * sy
        rx, ry = self.central_radius_px[0] * sx, self.central_radius_px[1] * sy
        x, y = float(np.median(xs)), float(np.median(ys))
        return ((x - cx) / max(rx, 1.0)) ** 2 + ((y - cy) / max(ry, 1.0)) ** 2 <= 1.0

    def _automatic_masks(self, image_bgr: np.ndarray, *, central: bool = False) -> list[tuple[np.ndarray, float]]:
        if central:
            height, width = image_bgr.shape[:2]
            grid = central_point_grid(
                width, height,
                reference_width=self.central_reference_width,
                reference_height=self.central_reference_height,
                center_px=self.central_center_px,
                radius_px=self.central_radius_px,
                spacing_px=self.central_spacing_px,
            )
            self.automatic.point_grids = [grid]
            self.last_central_prompt_count = len(grid)
        generated = self.automatic.generate(image_bgr[:, :, ::-1])
        output: list[tuple[np.ndarray, float]] = []
        for item in sorted(generated, key=lambda entry: float(entry["predicted_iou"]), reverse=True):
            mask = np.asarray(item["segmentation"], dtype=bool)
            score = float(item["predicted_iou"])
            if mask.sum() < self.min_area_px or score < self.threshold:
                continue
            if central and (
                mask.mean() > self.central_max_mask_area_ratio
                or not self._mask_center_in_central_roi(mask)
            ):
                continue
            if not central and any(_mask_iou(mask, previous) >= 0.85 for previous, _ in output):
                continue
            output.append((mask, score))
            # Keep extra central candidates until graph clustering collapses
            # nested duplicate masks. Otherwise duplicates could consume all
            # available slots before distinct objects are considered.
            if len(output) >= (self.max_masks * 4 if central else self.max_masks):
                break
        if central:
            output = filter_and_merge_central_masks(
                output,
                lower_region_start_fraction=self.central_lower_region_start_fraction,
                lower_region_mask_fraction=self.central_lower_region_mask_fraction,
                merge_overlap_min_fraction=self.central_merge_overlap_min_fraction,
                max_masks=self.max_masks,
            )
        return output

    def infer(
        self,
        image_bgr: np.ndarray,
        concepts: list[str],
        *,
        depth_raw: np.ndarray | None = None,
        depth_scale_um: int | None = None,
        **_ignored: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        del concepts  # Generic masks intentionally have no semantic text prompt.
        height, width = image_bgr.shape[:2]
        self.frame_index += 1
        self.predictor.set_image(image_bgr, image_format="BGR")
        reinitialize = (
            not self.track_priors
            or not any(int(track["missing"]) == 0 for track in self.track_priors)
            or self.frame_index % self.reinitialize_interval_frames == 0
        )
        results: list[tuple[np.ndarray, float]] = []
        if not reinitialize:
            results = self._tracked_masks(width, height)
            self.last_mode = "multi-object-track-prompts"
        elif self.reinitialization == "depth-only":
            for point, box in self._depth_proposals(depth_raw, depth_scale_um):
                candidate = self._prompt_mask(point, box)
                if candidate is not None:
                    results.append(candidate)
            self.last_mode = "depth-only-reinitialize"
        else:
            central = self.reinitialization == "central"
            results = self._automatic_masks(image_bgr, central=central)
            self.last_mode = "central-mask-reinitialize" if central else "all-mask-reinitialize"

        kept: list[tuple[np.ndarray, float]] = []
        for mask, score in sorted(results, key=lambda pair: pair[1], reverse=True):
            if score < self.threshold or mask.sum() < self.min_area_px:
                continue
            if any(_mask_iou(mask, old_mask) >= 0.85 for old_mask, _ in kept):
                continue
            kept.append((mask, score))
            if len(kept) >= self.max_masks:
                break
        if not kept:
            return _empty(height, width)
        if reinitialize:
            # Stable *3-D* IDs remain the live server's responsibility because
            # only it owns calibrated depth. These are local 2-D prompt priors.
            self.track_priors = [
                {"box": _box_from_mask(mask), "missing": 0}
                for mask, _ in kept
            ]
        masks = np.stack([mask for mask, _ in kept], axis=0)
        boxes = np.stack([_box_from_mask(mask) for mask, _ in kept], axis=0)
        scores = np.asarray([score for _, score in kept], dtype=np.float32)
        return masks, boxes, scores, ["object"] * len(kept)
