"""External desktop YOLO11 worker for PATH-G1 Wi-Fi offload."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from seg_web_node.geometry import (
    CameraIntrinsics,
    transform_points,
)
from seg_web_node.grip_candidates import (
    GripCandidateConfig,
    PersistentCylinderSurfaceTracker,
    generate_grip_candidates,
)
from seg_web_node.geometry_fallback import LastValidGeometryFallback
from seg_web_node.rgbd_sync import RgbdPair
from seg_web_node.object_point_filter import (
    ObjectPointFilterConfig,
    extract_filtered_object_cloud,
)
from seg_web_node.yolo11_backend import Yolo11SegBackend

from .protocol import make_empty_result, make_object
from .remote_sam3_worker import RemoteSam3Worker, load_extrinsics, parse_concepts


LOG = logging.getLogger("remote_offload.remote_yolo11_worker")


def parse_csv(text: str | None) -> list[str]:
    if text is None:
        return []
    return list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))


def parse_quantize(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "fp32", "32"}:
        return None if text in {"", "none", "null"} else 32
    if text in {"fp16", "16"}:
        return 16
    raise ValueError("--quantize must be 16, 32, fp16, fp32, none, or null")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration file must contain a map: {path}")
    return payload


class RemoteYolo11Worker(RemoteSam3Worker):
    def __init__(
        self,
        *,
        yolo_options: dict[str, Any],
        depth_self_filter: dict[str, Any],
        object_point_filter: dict[str, Any],
        contact_candidates: bool,
        contact_config: GripCandidateConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.yolo_options = dict(yolo_options)
        self.depth_self_filter_options = dict(depth_self_filter)
        self.object_point_filter = ObjectPointFilterConfig.from_dict(object_point_filter)
        self.contact_candidates_enabled = bool(contact_candidates)
        self.contact_config = contact_config
        self.surface_tracker = PersistentCylinderSurfaceTracker(contact_config)
        track_options = dict(self.yolo_options.get("track", {}))
        fallback_ttl = int(
            track_options.get(
                "geometry_fallback_ttl_frames",
                track_options.get("missing_ttl_frames", 6),
            )
        )
        self.geometry_fallback = LastValidGeometryFallback(
            fallback_ttl,
            float(track_options.get("geometry_fallback_score_decay", 0.97)),
        )
        self.tracker.missing_ttl_frames = max(
            int(self.tracker.missing_ttl_frames),
            fallback_ttl,
        )

    def _backend_display_name(self) -> str:
        return "YOLO11"

    def _error_wire_backend_name(self) -> str:
        return "mock" if self.backend_name == "mock" else "yolo11_pytorch"

    def _load_backend(self):
        if self.backend_name == "mock":
            return super()._load_backend()
        if self.backend_name != "pytorch":
            raise ValueError(f"unknown YOLO11 backend: {self.backend_name}")
        options = self.yolo_options
        return (
            Yolo11SegBackend(
                checkpoint=Path(options.get("checkpoint", "/models/yolo11/yolo11n-seg.pt")),
                threshold=self.threshold,
                min_area_px=self.min_area_px,
                imgsz=int(options.get("imgsz", 640)),
                confidence=float(options.get("confidence", 0.80)),
                iou_threshold=float(options.get("iou_threshold", 0.50)),
                max_masks=int(options.get("max_masks", 20)),
                quantize=options.get("quantize", 16),
                track=dict(options.get("track", {})),
                allowed_classes=list(options.get("allowed_classes", [])),
                excluded_classes=list(options.get("excluded_classes", ["person"])),
                depth_self_filter=dict(
                    options.get("depth_self_filter", self.depth_self_filter_options)
                ),
            ),
            "yolo11_pytorch",
        )

    def _prepare_backend_for_frame(
        self,
        backend: Any,
        image_bgr: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics | None,
        pair: RgbdPair,
    ) -> None:
        if not hasattr(backend, "set_depth_filter_context"):
            return
        if intrinsics is None:
            backend.set_depth_filter_context(
                image_bgr=image_bgr,
                depth_raw=depth,
                depth_scale_um=pair.depth.depth_scale_um,
                intrinsics=None,
                t_goal_cam=None,
                q=None,
            )
            return
        try:
            t_goal_cam = self._goal_transform(pair.rgb.timestamp_us)
            match = self.state_buffer.nearest(
                pair.rgb.timestamp_us,
                self.robot_state_max_delta_ms,
            )
            if t_goal_cam is None or match is None:
                backend.set_depth_filter_context(
                    image_bgr=image_bgr,
                    depth_raw=depth,
                    depth_scale_um=pair.depth.depth_scale_um,
                    intrinsics=intrinsics,
                    t_goal_cam=None,
                    q=None,
                )
                return
            state, _delta_ms = match
            backend.set_depth_filter_context(
                image_bgr=image_bgr,
                depth_raw=depth,
                depth_scale_um=pair.depth.depth_scale_um,
                intrinsics=intrinsics,
                t_goal_cam=t_goal_cam,
                q=state.q,
            )
        except Exception as error:
            LOG.warning("depth self-filter context unavailable: %s", error)
            backend.set_depth_filter_context(
                image_bgr=image_bgr,
                depth_raw=depth,
                depth_scale_um=pair.depth.depth_scale_um,
                intrinsics=None,
                t_goal_cam=None,
                q=None,
            )

    def _build_result(
        self,
        pair: RgbdPair,
        backend_wire_name: str,
        inference_ms: float,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[str],
        depth: np.ndarray,
        intrinsics: CameraIntrinsics | None,
    ) -> dict[str, Any]:
        self.sequence += 1
        payload = make_empty_result(
            sequence=self.sequence,
            source_rgb_timestamp_us=pair.rgb.timestamp_us,
            source_depth_timestamp_us=pair.depth.timestamp_us,
            calibration_id=self.calibration.calibration_id if self.calibration else None,
            image_width=pair.rgb.width,
            image_height=pair.rgb.height,
            backend=backend_wire_name,
            inference_ms=inference_ms,
        )
        t_goal_cam = self._goal_transform(pair.rgb.timestamp_us)
        if t_goal_cam is not None:
            # The optical +Z axis is the image-center ray in the common goal
            # frame.  The Thor gateway uses it for the TASC ray feature.
            payload["camera_goal_origin_m"] = (
                np.asarray(t_goal_cam[:3, 3], dtype=np.float64).tolist()
            )
            payload["camera_goal_forward_m"] = (
                np.asarray(t_goal_cam[:3, :3], dtype=np.float64)
                @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            ).tolist()
        depth_m = depth.astype(np.float32) * (pair.depth.depth_scale_um / 1_000_000.0)

        detections: list[dict[str, Any]] = []
        for index, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            mask_bool = np.asarray(mask, dtype=bool)
            detection: dict[str, Any] = {
                "index": index,
                "label": str(label),
                "score": float(score),
                "center_camera_m": None,
                "center_goal_m": None,
                "center_pixel_uv": None,
                "valid_depth_points": 0,
                "median_depth_m": None,
                "target_contact_goal_m": None,
                "contact_candidates": [],
                "surface_model": None,
                "point_filter": None,
                "_points_goal_m": None,
            }
            cloud = None
            if intrinsics is not None:
                cloud = extract_filtered_object_cloud(
                    depth_m,
                    mask_bool,
                    intrinsics,
                    self.object_point_filter,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                    min_valid_points=self.min_depth_points,
                    max_points=self.contact_config.max_points,
                )
            if cloud is not None:
                detection["median_depth_m"] = cloud.median_depth_m
                detection["center_camera_m"] = cloud.center_camera_m.tolist()
                detection["center_pixel_uv"] = [
                    int(cloud.center_pixel_uv[0]),
                    int(cloud.center_pixel_uv[1]),
                ]
                detection["valid_depth_points"] = cloud.valid_depth_points
                detection["point_filter"] = cloud.stats
                if t_goal_cam is not None:
                    center_goal = transform_points(cloud.center_camera_m, t_goal_cam)
                    detection["center_goal_m"] = center_goal.tolist()
                    if self.contact_candidates_enabled:
                        detection["_points_goal_m"] = transform_points(
                            cloud.points_camera_m,
                            t_goal_cam,
                        )
            detections.append(detection)

        track_ids = self.tracker.assign(detections)
        active_track_ids: set[str] = set()
        wire_objects: list[dict[str, Any]] = []
        for index, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
            detection = detections[index]
            track_id = track_ids.get(index, f"{label}:untracked{index:04d}")
            points_goal = detection.pop("_points_goal_m", None)
            if self.contact_candidates_enabled and points_goal is not None:
                active_track_ids.add(track_id)
                fitted_candidates = self.surface_tracker.update(
                    track_id,
                    str(label),
                    points_goal,
                )
                if not fitted_candidates:
                    fitted_candidates = generate_grip_candidates(
                        points_goal,
                        self.contact_config,
                    )
                candidates = [candidate.to_dict() for candidate in fitted_candidates]
                detection["contact_candidates"] = candidates
                detection["surface_model"] = self.surface_tracker.status(track_id)
                if candidates:
                    detection["target_contact_goal_m"] = candidates[0]["center_goal_m"]
            wire_object = make_object(
                track_id=track_id,
                label=str(label),
                score=float(score),
                box_xyxy=[float(value) for value in box],
                mask=mask,
                median_depth_m=detection.get("median_depth_m"),
                center_pixel_uv=detection.get("center_pixel_uv"),
                center_camera_m=detection.get("center_camera_m"),
                center_goal_m=detection.get("center_goal_m"),
                valid_depth_points=int(detection.get("valid_depth_points", 0)),
                target_contact_goal_m=detection.get("target_contact_goal_m"),
                contact_candidates=detection.get("contact_candidates"),
                surface_model=detection.get("surface_model"),
            )
            wire_object["point_filter"] = detection.get("point_filter")
            wire_objects.append(wire_object)
        self.surface_tracker.finish_frame(active_track_ids)
        payload["objects"] = self.geometry_fallback.complete_frame(
            wire_objects,
            self.surface_tracker,
            image_width=pair.rgb.width,
            image_height=pair.rgb.height,
        )
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--backend", choices=["pytorch", "mock"], default="pytorch")
    parser.add_argument("--concepts", default="object")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--max-masks", type=int, default=None)
    parser.add_argument("--quantize", default=None)
    parser.add_argument("--allowed-classes", default=None)
    parser.add_argument("--excluded-classes", default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--min-area-px", type=int, default=None)
    parser.add_argument("--inference-hz", type=float, default=None)
    parser.add_argument("--sync-tolerance-ms", type=float, default=None)
    parser.add_argument("--min-depth-m", type=float, default=None)
    parser.add_argument("--max-depth-m", type=float, default=None)
    parser.add_argument("--min-depth-points", type=int, default=None)
    parser.add_argument("--robot-state-max-delta-ms", type=float, default=None)
    parser.add_argument("--torso-estimator", default=None)
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--publish-overlay-jpeg", action="store_true")
    parser.add_argument("--disable-contact-candidates", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    segmentation = dict(config.get("segmentation", {}))
    yolo_options = dict(segmentation.get("yolo11", {}))
    yolo_options["checkpoint"] = args.checkpoint or yolo_options.get(
        "checkpoint",
        "/models/yolo11/yolo11n-seg.pt",
    )
    if args.imgsz is not None:
        yolo_options["imgsz"] = args.imgsz
    if args.confidence is not None:
        yolo_options["confidence"] = args.confidence
    if args.iou_threshold is not None:
        yolo_options["iou_threshold"] = args.iou_threshold
    if args.max_masks is not None:
        yolo_options["max_masks"] = args.max_masks
    yolo_options.setdefault("quantize", 16)
    yolo_options["quantize"] = parse_quantize(
        args.quantize if args.quantize is not None else yolo_options.get("quantize")
    )
    if args.allowed_classes is not None:
        yolo_options["allowed_classes"] = parse_csv(args.allowed_classes)
    if args.excluded_classes is not None:
        yolo_options["excluded_classes"] = parse_csv(args.excluded_classes)

    depth_config = dict(config.get("depth", {}))
    cloud_config = dict(config.get("cloud", {}))
    target_config = dict(config.get("ee_target", {}))
    tasc_config = dict(config.get("tasc_belief", {}))
    contact_config_data = dict(config.get("grip", {}))
    contact_config_data.update(dict(tasc_config.get("grip", {})))
    contact_config_data["max_points"] = int(
        tasc_config.get(
            "object_max_points",
            contact_config_data.get(
                "max_points",
                contact_config_data.get("object_max_points", 8000),
            ),
        )
    )
    extrinsics_path = args.extrinsics or (
        Path(str(target_config["extrinsics"])) if target_config.get("extrinsics") else None
    )

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = RemoteYolo11Worker(
        wifi_lcm_url=args.wifi_lcm_url,
        concepts=parse_concepts(args.concepts),
        backend_name=args.backend,
        checkpoint=None,
        trt_engine=None,
        trt_model_id="",
        threshold=float(
            args.min_score
            if args.min_score is not None
            else segmentation.get("min_score", 0.5)
        ),
        min_area_px=int(
            args.min_area_px
            if args.min_area_px is not None
            else segmentation.get("min_area_px", 400)
        ),
        inference_hz=float(
            args.inference_hz
            if args.inference_hz is not None
            else segmentation.get("target_hz", 3.0)
        ),
        sync_tolerance_ms=float(
            args.sync_tolerance_ms
            if args.sync_tolerance_ms is not None
            else depth_config.get("sync_tol_ms", 50.0)
        ),
        min_depth_m=float(
            args.min_depth_m
            if args.min_depth_m is not None
            else depth_config.get("min_depth_m", 0.15)
        ),
        max_depth_m=float(
            args.max_depth_m
            if args.max_depth_m is not None
            else depth_config.get("max_depth_m", 3.0)
        ),
        min_depth_points=int(
            args.min_depth_points
            if args.min_depth_points is not None
            else target_config.get("min_depth_points", 100)
        ),
        robot_state_max_delta_ms=float(
            args.robot_state_max_delta_ms
            if args.robot_state_max_delta_ms is not None
            else target_config.get("robot_state_max_delta_ms", 40.0)
        ),
        torso_estimator=str(
            args.torso_estimator or target_config.get("torso_estimator", "dual_imu")
        ),
        t_torso_cam=load_extrinsics(extrinsics_path) if extrinsics_path else None,
        publish_overlay_jpeg=bool(args.publish_overlay_jpeg),
        yolo_options=yolo_options,
        depth_self_filter=dict(
            yolo_options.get(
                "depth_self_filter",
                segmentation.get("depth_self_filter", {}),
            )
        ),
        object_point_filter=dict(cloud_config.get("object_filter", {})),
        contact_candidates=not bool(args.disable_contact_candidates),
        contact_config=GripCandidateConfig.from_dict(contact_config_data),
    )
    worker.start()
    LOG.info("remote YOLO11 worker started on Wi-Fi LCM; publishes perception results only")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOG.info("stopping remote YOLO11 worker")
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
