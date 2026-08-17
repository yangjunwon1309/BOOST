"""External desktop YOLO11 worker for PATH-G1 Wi-Fi offload."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from seg_web_node.geometry import CameraIntrinsics
from seg_web_node.grip_candidates import GripCandidateConfig, PseudoContactConfig
from seg_web_node.object_point_filter import ObjectPointFilterConfig
from seg_web_node.rgbd_sync import RgbdPair
from seg_web_node.yolo11_backend import Yolo11SegBackend

from .protocol import make_empty_result
from .remote_geometry_pipeline import (
    RemoteRgbdGeometryPipeline,
    camera_geometry_metadata,
)
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
        pseudo_config: PseudoContactConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.yolo_options = dict(yolo_options)
        self.depth_self_filter_options = dict(depth_self_filter)
        self.geometry_requires_robot_state = True
        self._geometry_disabled_warned = False
        track_options = dict(self.yolo_options.get("track", {}))
        fallback_ttl = int(
            track_options.get(
                "geometry_fallback_ttl_frames",
                track_options.get("missing_ttl_frames", 6),
            )
        )
        self.geometry_pipeline = RemoteRgbdGeometryPipeline(
            object_point_filter=ObjectPointFilterConfig.from_dict(object_point_filter),
            contact_config=contact_config,
            pseudo_config=pseudo_config,
            contact_candidates=contact_candidates,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            min_depth_points=self.min_depth_points,
            tracker=self.tracker,
            fallback_ttl_frames=fallback_ttl,
            fallback_score_decay=float(
                track_options.get("geometry_fallback_score_decay", 0.97)
            ),
            fallback_reason="yolo_detection_missing",
        )
        # Compatibility handles for diagnostics/tests that inspect worker state.
        self.object_point_filter = self.geometry_pipeline.object_point_filter
        self.contact_candidates_enabled = self.geometry_pipeline.contact_candidates_enabled
        self.contact_config = contact_config
        self.surface_tracker = self.geometry_pipeline.surface_tracker
        self.geometry_fallback = self.geometry_pipeline.geometry_fallback

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
        try:
            super()._prepare_backend_for_frame(
                backend, image_bgr, depth, intrinsics, pair
            )
        except Exception as error:
            LOG.warning("depth self-filter context unavailable: %s", error)
            self.frame_geometry = None
            if hasattr(backend, "set_depth_filter_context"):
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
        context = self._current_frame_geometry(pair.rgb.timestamp_us)
        geometry_intrinsics = (
            intrinsics
            if context.state is not None and context.t_goal_cam is not None
            else None
        )
        if geometry_intrinsics is None:
            if not getattr(self, "_geometry_disabled_warned", False):
                LOG.warning(
                    "robot state or T_goal_cam unavailable; publishing masks only. "
                    "Thor will receive no contact candidates and no surface primitives"
                )
                self._geometry_disabled_warned = True
        else:
            self._geometry_disabled_warned = False
        payload.update(
            camera_geometry_metadata(context.t_goal_cam, geometry_intrinsics)
        )
        payload["gravity_up_goal"] = (
            None if context.up_goal is None else context.up_goal.tolist()
        )
        payload["objects"] = self.geometry_pipeline.build_object_geometry(
            masks=masks,
            boxes=boxes,
            scores=scores,
            labels=labels,
            depth_raw=depth,
            depth_scale_um=pair.depth.depth_scale_um,
            intrinsics=geometry_intrinsics,
            t_goal_cam=context.t_goal_cam if geometry_intrinsics is not None else None,
            up_goal=context.up_goal,
            source_robot_state_delta_ms=context.state_delta_ms,
            source_timestamp_us=pair.rgb.timestamp_us,
            image_width=pair.rgb.width,
            image_height=pair.rgb.height,
            allow_geometry_fallback=geometry_intrinsics is not None,
        )
        payload["geometry_status"] = {
            "state_available": context.state is not None,
            "t_goal_cam_available": context.t_goal_cam is not None,
            "up_goal_available": context.up_goal is not None,
            "state_delta_ms": context.state_delta_ms,
            "primitive_count": sum(
                len(obj.get("surface_primitives") or [])
                for obj in payload["objects"]
            ),
            "fallback_count": sum(
                1 for obj in payload["objects"] if obj.get("geometry_fallback")
            ),
        }
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
    pseudo_config_data = dict(tasc_config.get("pseudo", {}))
    orientation_config = dict(tasc_config.get("orientation", {}))
    pseudo_config_data.update(dict(orientation_config.get("pseudo", {})))
    pseudo_config_data["gravity_snap_deg"] = orientation_config.get(
        "gravity_snap_deg",
        pseudo_config_data.get("gravity_snap_deg", 15.0),
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
        pseudo_config=PseudoContactConfig.from_dict(pseudo_config_data),
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
