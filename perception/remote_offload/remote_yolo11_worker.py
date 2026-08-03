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
    deproject,
    robust_mask_center,
    transform_points,
)
from seg_web_node.grip_candidates import GripCandidateConfig, generate_grip_candidates
from seg_web_node.rgbd_sync import RgbdPair
from seg_web_node.robot_geometry import g1_link_transforms
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


def projected_robot_arm_mask(
    q: np.ndarray,
    t_goal_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    height: int,
    width: int,
    arm_radius_px: int,
    wrist_radius_px: int,
) -> np.ndarray:
    """Rasterize the FK arm/hand chain into the RGB camera image."""
    transforms = g1_link_transforms(q)
    transform = np.asarray(t_goal_cam, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    output = np.zeros((height, width), dtype=np.uint8)

    def project(link_name: str) -> tuple[int, int] | None:
        point_goal = transforms[link_name][:3, 3]
        point_camera = rotation.T @ (point_goal - translation)
        if not np.isfinite(point_camera).all() or point_camera[2] <= 0.05:
            return None
        u = intrinsics.fx * point_camera[0] / point_camera[2] + intrinsics.cx
        v = intrinsics.fy * point_camera[1] / point_camera[2] + intrinsics.cy
        if (
            u < -wrist_radius_px
            or u >= width + wrist_radius_px
            or v < -wrist_radius_px
            or v >= height + wrist_radius_px
        ):
            return None
        return int(round(u)), int(round(v))

    for side in ("left", "right"):
        links = [
            "torso_link",
            f"{side}_shoulder_pitch_link",
            f"{side}_shoulder_roll_link",
            f"{side}_shoulder_yaw_link",
            f"{side}_elbow_link",
            f"{side}_wrist_roll_link",
            f"{side}_wrist_pitch_link",
            f"{side}_wrist_yaw_link",
        ]
        points = [project(link) for link in links]
        for first, second in zip(points[:-1], points[1:]):
            if first is not None and second is not None:
                cv2.line(
                    output,
                    first,
                    second,
                    255,
                    max(1, int(arm_radius_px) * 2),
                    cv2.LINE_AA,
                )
        for index, point in enumerate(points):
            if point is not None:
                radius = wrist_radius_px if index >= len(points) - 3 else arm_radius_px
                cv2.circle(output, point, max(1, int(radius)), 255, -1, cv2.LINE_AA)
    return output.astype(bool)


def masked_goal_points(
    xyz_camera: np.ndarray,
    mask: np.ndarray,
    t_goal_cam: np.ndarray,
    *,
    max_points: int,
) -> np.ndarray:
    valid = mask.astype(bool) & np.isfinite(xyz_camera).all(axis=2)
    points = xyz_camera[valid]
    if points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max(1, int(max_points))))
        points = points[::step][:max_points]
    return transform_points(points, t_goal_cam)


class RemoteYolo11Worker(RemoteSam3Worker):
    def __init__(
        self,
        *,
        yolo_options: dict[str, Any],
        robot_exclusion: dict[str, Any],
        contact_candidates: bool,
        contact_config: GripCandidateConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.yolo_options = dict(yolo_options)
        self.robot_exclusion_options = dict(robot_exclusion)
        self.contact_candidates_enabled = bool(contact_candidates)
        self.contact_config = contact_config

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
                robot_exclusion=dict(options.get("robot_exclusion", {})),
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
        del depth
        if not hasattr(backend, "set_robot_exclusion_mask"):
            return
        backend.set_robot_exclusion_mask(None)
        if intrinsics is None:
            return
        try:
            t_goal_cam = self._goal_transform(pair.rgb.timestamp_us)
            match = self.state_buffer.nearest(
                pair.rgb.timestamp_us,
                self.robot_state_max_delta_ms,
            )
            if t_goal_cam is None or match is None:
                return
            state, _delta_ms = match
            robot_mask = projected_robot_arm_mask(
                state.q,
                t_goal_cam,
                intrinsics,
                height=image_bgr.shape[0],
                width=image_bgr.shape[1],
                arm_radius_px=int(self.robot_exclusion_options.get("arm_radius_px", 22)),
                wrist_radius_px=int(self.robot_exclusion_options.get("wrist_radius_px", 42)),
            )
            backend.set_robot_exclusion_mask(robot_mask)
        except Exception as error:
            LOG.warning("robot exclusion FK unavailable: %s", error)

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
        xyz_camera = None
        if intrinsics is not None and depth.shape == (intrinsics.height, intrinsics.width):
            xyz_camera = deproject(
                depth,
                intrinsics,
                pair.depth.depth_scale_um,
                self.min_depth_m,
                self.max_depth_m,
            )
        t_goal_cam = self._goal_transform(pair.rgb.timestamp_us)
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
            }
            valid_depth = (
                mask_bool
                & (depth > 0)
                & np.isfinite(depth_m)
                & (depth_m >= self.min_depth_m)
                & (depth_m <= self.max_depth_m)
            )
            values = depth_m[valid_depth]
            detection["median_depth_m"] = float(np.median(values)) if values.size else None
            if xyz_camera is not None:
                center = robust_mask_center(
                    xyz_camera,
                    mask_bool,
                    min_valid_points=self.min_depth_points,
                )
                if center is not None:
                    center_camera_array, pixel, valid_points = center
                    detection["center_camera_m"] = center_camera_array.tolist()
                    detection["center_pixel_uv"] = [int(pixel[0]), int(pixel[1])]
                    detection["valid_depth_points"] = int(valid_points)
                    if t_goal_cam is not None:
                        center_goal = transform_points(center_camera_array, t_goal_cam)
                        detection["center_goal_m"] = center_goal.tolist()
                        if self.contact_candidates_enabled:
                            points_goal = masked_goal_points(
                                xyz_camera,
                                mask_bool,
                                t_goal_cam,
                                max_points=self.contact_config.max_points,
                            )
                            candidates = [
                                candidate.to_dict()
                                for candidate in generate_grip_candidates(
                                    points_goal,
                                    self.contact_config,
                                )
                            ]
                            detection["contact_candidates"] = candidates
                            if candidates:
                                detection["target_contact_goal_m"] = candidates[0][
                                    "center_goal_m"
                                ]
            detections.append(detection)

        track_ids = self.tracker.assign(detections)
        for index, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
            detection = detections[index]
            payload["objects"].append(
                make_object(
                    track_id=track_ids.get(index, f"{label}:untracked{index:04d}"),
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
                )
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
    target_config = dict(config.get("ee_target", {}))
    tasc_config = dict(config.get("tasc_belief", {}))
    contact_config_data = dict(config.get("grip", {}))
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
        robot_exclusion=dict(yolo_options.get("robot_exclusion", {})),
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
