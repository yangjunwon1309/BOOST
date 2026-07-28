"""COCO-pretrained YOLO11 instance-segmentation backend.

Unlike promptable SAM-family models, YOLO11 only emits its learned COCO object
classes.  This gives the RGB-D pipeline a deterministic semantic gate for
ordinary objects (bottle, cup, chair, box-like classes, tools where COCO has a
class) while retaining PATH's depth fusion, stable 3-D IDs and belief logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .efficientvit_backend import _box_from_mask
from .sam2_backend import Sam2HieraTinyBackend


class Yolo11SegBackend(Sam2HieraTinyBackend):
    """YOLO11n-seg with label-aware mask association and dropout persistence."""

    name = "yolo11n-seg-coco"
    # Friendly spellings still resolve to one exact COCO class.  This is not
    # text/semantic search: unknown concepts deliberately produce no masks.
    COCO_LABEL_ALIASES = {
        "tv": "tv",
        "television": "tv",
        "sofa": "couch",
        "table": "dining table",
        "plant": "potted plant",
        "phone": "cell phone",
        "mobile phone": "cell phone",
        "ball": "sports ball",
    }
    WILDCARD_CONCEPTS = {"", "*", "all", "object", "objects"}

    def __init__(
        self,
        *,
        checkpoint: Path,
        threshold: float,
        min_area_px: int,
        imgsz: int = 640,
        confidence: float = 0.80,
        iou_threshold: float = 0.50,
        max_masks: int = 20,
        quantize: int | None = 16,
        track: dict[str, Any] | None = None,
        allowed_classes: list[str] | None = None,
        excluded_classes: list[str] | None = None,
        robot_exclusion: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLO

        checkpoint = Path(checkpoint)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"YOLO11 segmentation checkpoint does not exist: {checkpoint}; "
                "run tools/cache_yolo11n_seg.py"
            )
        self.torch = torch
        self.model = YOLO(str(checkpoint), task="segment")
        self.checkpoint = checkpoint
        self.threshold = float(threshold)
        self.min_area_px = int(min_area_px)
        self.imgsz = max(32, int(imgsz))
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.max_masks = max(1, int(max_masks))
        self.quantize = None if quantize is None else int(quantize)
        if self.quantize not in {None, 16, 32}:
            raise ValueError("YOLO11 quantize must be 16 (FP16), 32 (FP32), or null")
        track = dict(track or {})
        self.track_missing_ttl_frames = max(0, int(track.get("missing_ttl_frames", 24)))
        self.track_fallback_score_decay = float(track.get("fallback_score_decay", 0.985))
        self.reinitialize_match_iou = float(track.get("match_iou", 0.15))
        self.class_names_by_id = {
            int(index): str(label).strip().lower()
            for index, label in self.model.names.items()
        }
        self.class_ids_by_name = {
            label: index for index, label in self.class_names_by_id.items()
        }
        self.allowed_classes = {str(value).strip().lower() for value in (allowed_classes or []) if str(value).strip()}
        # A generic COCO `person` mask is not a valid manipulation target and
        # is a useful extra guard for the robot/nearby operator.
        self.excluded_classes = {str(value).strip().lower() for value in (excluded_classes or ["person"]) if str(value).strip()}
        self.allowed_class_ids = {
            self.class_ids_by_name[label]
            for label in self.allowed_classes
            if label in self.class_ids_by_name and label not in self.excluded_classes
        }
        robot_exclusion = dict(robot_exclusion or {})
        self.robot_static_bottom_fraction = float(robot_exclusion.get("static_bottom_fraction", 0.16))
        self.robot_static_side_fraction = float(robot_exclusion.get("static_side_fraction", 0.035))
        self.robot_min_remaining_fraction = float(robot_exclusion.get("min_remaining_fraction", 0.30))
        self.boundary_crop_px = max(0, int(track.get("boundary_crop_px", 4)))
        self.robot_exclusion_mask: np.ndarray | None = None
        self.last_robot_filter = {"dynamic": False, "removed_masks": 0, "trimmed_masks": 0}
        self.track_priors: list[dict[str, Any]] = []
        self.frame_index = 0
        self.last_mode = "uninitialized"
        self.last_central_prompt_count = 0
        self.robot_prompt_point = None
        self.last_requested_concepts: list[str] = []
        self.last_resolved_labels: list[str] = []
        self.last_unmatched_concepts: list[str] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": "yolo11n-seg",
            "dataset": "COCO-80",
            "checkpoint": str(self.checkpoint),
            "inference_format": "pytorch",
            "device": "cuda",
            "imgsz": self.imgsz,
            "quantize": self.quantize,
            "confidence": self.confidence,
            "boundary_crop_px": self.boundary_crop_px,
            "iou_threshold": self.iou_threshold,
            "excluded_classes": sorted(self.excluded_classes),
            "allowed_classes": sorted(self.allowed_classes),
            "requested_concepts": list(self.last_requested_concepts),
            "resolved_coco_labels": list(self.last_resolved_labels),
            "unmatched_concepts": list(self.last_unmatched_concepts),
            "last_mode": self.last_mode,
            "active_tracks": sum(int(track.get("missing", 0)) == 0 for track in self.track_priors),
            "fallback_tracks": sum(int(track.get("missing", 0)) > 0 for track in self.track_priors),
            "robot_exclusion": dict(self.last_robot_filter),
        }

    def _requested_class_ids(self, concepts: list[str]) -> list[int] | None:
        """Resolve web concepts to COCO class IDs; ``None`` means all classes."""
        requested = [str(item).strip().lower() for item in concepts]
        self.last_requested_concepts = [item for item in requested if item]
        self.last_resolved_labels = []
        self.last_unmatched_concepts = []
        if not requested or any(item in self.WILDCARD_CONCEPTS for item in requested):
            if self.allowed_classes:
                resolved = sorted(self.allowed_class_ids)
            else:
                resolved = [
                    index for index, label in self.class_names_by_id.items()
                    if label not in self.excluded_classes
                ]
            self.last_resolved_labels = [self.class_names_by_id[index] for index in resolved]
            return resolved
        resolved: list[int] = []
        for raw_label in requested:
            if not raw_label:
                continue
            label = self.COCO_LABEL_ALIASES.get(raw_label, raw_label)
            class_index = self.class_ids_by_name.get(label)
            if class_index is None:
                self.last_unmatched_concepts.append(raw_label)
                continue
            if label in self.excluded_classes:
                self.last_unmatched_concepts.append(f"{raw_label} (excluded)")
                continue
            if self.allowed_classes and class_index not in self.allowed_class_ids:
                self.last_unmatched_concepts.append(f"{raw_label} (not allowed)")
                continue
            if class_index not in resolved:
                resolved.append(class_index)
                self.last_resolved_labels.append(label)
        return resolved

    def _detect(self, image_bgr: np.ndarray, concepts: list[str]) -> list[dict[str, Any]]:
        class_ids = self._requested_class_ids(concepts)
        # A non-empty invalid query should not burn GPU time on an all-class
        # inference that cannot produce an allowed result.
        if class_ids == []:
            return []
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_masks,
            classes=class_ids,
            device=0,
            quantize=self.quantize,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.masks is None or result.boxes is None or not len(result.boxes):
            return []
        raw_masks = result.masks.data.detach().to("cpu").numpy().astype(bool)
        confidences = result.boxes.conf.detach().to("cpu").numpy()
        classes = result.boxes.cls.detach().to("cpu").numpy().astype(np.int64)
        names = result.names
        height, width = image_bgr.shape[:2]
        output: list[dict[str, Any]] = []
        for mask, confidence, class_index in zip(raw_masks, confidences, classes):
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            label = str(names.get(int(class_index), class_index)).strip().lower()
            if label in self.excluded_classes:
                continue
            if self.allowed_classes and label not in self.allowed_classes:
                continue
            mask = self._exclude_robot_pixels(mask)
            mask = self._crop_mask_boundary(mask)
            box = None if mask is None else _box_from_mask(mask)
            if box is None:
                continue
            output.append({"label": label, "mask": mask, "box": box, "score": float(confidence), "missing": 0})
        output.sort(key=lambda item: float(item["score"]), reverse=True)
        return output[: self.max_masks]

    def _crop_mask_boundary(self, mask: np.ndarray | None) -> np.ndarray | None:
        """Erode an instance edge so RGB/depth boundary bleed is not a target."""
        if mask is None:
            return None
        result = np.asarray(mask, dtype=bool)
        if self.boundary_crop_px <= 0:
            return result if int(result.sum()) >= self.min_area_px else None
        diameter = self.boundary_crop_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        result = cv2.erode(result.astype(np.uint8), kernel, iterations=1).astype(bool)
        return result if int(result.sum()) >= self.min_area_px else None

    def _update_tracks(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unmatched_previous = set(range(len(self.track_priors)))
        updated: list[dict[str, Any]] = []
        for detection in detections:
            best_index = None
            best_iou = self.reinitialize_match_iou
            for index in unmatched_previous:
                previous = self.track_priors[index]
                if previous["label"] != detection["label"]:
                    continue
                overlap = self._mask_iou(detection["mask"], np.asarray(previous["mask"], dtype=bool))
                if overlap >= best_iou:
                    best_iou, best_index = overlap, index
            if best_index is not None:
                unmatched_previous.remove(best_index)
            updated.append(detection)
        active_labels = {str(item["label"]) for item in detections}
        for index in unmatched_previous:
            previous = self.track_priors[index]
            # If the detector sees the same class at a new image location,
            # the old mask is a stale viewpoint hypothesis, not a second
            # object. Drop it rather than producing a duplicate/ghost track.
            if str(previous["label"]) in active_labels:
                continue
            missing = int(previous.get("missing", 0)) + 1
            if missing > self.track_missing_ttl_frames:
                continue
            fallback_mask = self._crop_mask_boundary(
                self._exclude_robot_pixels(np.asarray(previous["mask"], dtype=bool))
            )
            fallback_box = None if fallback_mask is None else _box_from_mask(fallback_mask)
            if fallback_box is None:
                continue
            updated.append({
                **previous,
                "mask": fallback_mask,
                "box": fallback_box,
                "score": float(previous["score"]) * (self.track_fallback_score_decay ** missing),
                "missing": missing,
            })
        updated.sort(key=lambda item: (int(item["missing"]) > 0, -float(item["score"])))
        self.track_priors = updated[: self.max_masks]
        # Missing tracks are retained only for a later IoU re-association.
        # They are deliberately never rendered or sent into RGB-D/belief:
        # without a current detector score they cannot satisfy confidence gate.
        return [
            item for item in self.track_priors
            if int(item.get("missing", 0)) == 0 and float(item["score"]) >= self.confidence
        ]

    def infer(
        self,
        image_bgr: np.ndarray,
        concepts: list[str],
        *,
        depth_raw: np.ndarray | None = None,
        depth_scale_um: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        del depth_raw, depth_scale_um
        self.last_robot_filter = {"dynamic": False, "removed_masks": 0, "trimmed_masks": 0}
        tracks = self._update_tracks(self._detect(image_bgr, concepts))
        self.frame_index += 1
        self.last_mode = "yolo_detect_associate"
        height, width = image_bgr.shape[:2]
        if not tracks:
            return (
                np.empty((0, height, width), dtype=bool),
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                [],
            )
        return (
            np.stack([np.asarray(track["mask"], dtype=bool) for track in tracks]),
            np.stack([np.asarray(track["box"], dtype=np.float32) for track in tracks]),
            np.asarray([float(track["score"]) for track in tracks], dtype=np.float32),
            [str(track["label"]) for track in tracks],
        )
