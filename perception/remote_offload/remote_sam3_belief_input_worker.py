"""Remote SAM3 worker that produces complete Thor belief geometry inputs."""

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

from .protocol import make_empty_result
from .remote_geometry_pipeline import (
    RemoteRgbdGeometryPipeline,
    camera_geometry_metadata,
)
from .remote_sam3_worker import RemoteSam3Worker, load_extrinsics, parse_concepts


LOG = logging.getLogger("remote_offload.remote_sam3_belief_input_worker")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration file must contain a map: {path}")
    return payload


def _pseudo_config(tasc_config: dict[str, Any]) -> PseudoContactConfig:
    data = dict(tasc_config.get("pseudo", {}))
    orientation = dict(tasc_config.get("orientation", {}))
    data.update(dict(orientation.get("pseudo", {})))
    data["gravity_snap_deg"] = orientation.get(
        "gravity_snap_deg", data.get("gravity_snap_deg", 15.0)
    )
    return PseudoContactConfig.from_dict(data)


class RemoteSam3BeliefInputWorker(RemoteSam3Worker):
    """SAM3 masks plus filtered object/contact geometry, with no control output."""

    def __init__(
        self,
        *,
        object_point_filter: ObjectPointFilterConfig,
        contact_config: GripCandidateConfig,
        pseudo_config: PseudoContactConfig,
        contact_candidates: bool,
        geometry_fallback_ttl_frames: int,
        geometry_fallback_score_decay: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.geometry_requires_robot_state = True
        self.geometry_pipeline = RemoteRgbdGeometryPipeline(
            object_point_filter=object_point_filter,
            contact_config=contact_config,
            pseudo_config=pseudo_config,
            contact_candidates=contact_candidates,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            min_depth_points=self.min_depth_points,
            tracker=self.tracker,
            fallback_ttl_frames=geometry_fallback_ttl_frames,
            fallback_score_decay=geometry_fallback_score_decay,
            fallback_reason="sam3_detection_missing",
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
        payload.update(camera_geometry_metadata(context.t_goal_cam))
        payload["gravity_up_goal"] = (
            None if context.up_goal is None else context.up_goal.tolist()
        )

        # A mask remains useful for diagnostics without calibration/state, but
        # Thor must never mistake camera-only or stale geometry for belief input.
        geometry_intrinsics = (
            intrinsics
            if context.state is not None and context.t_goal_cam is not None
            else None
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
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["pytorch", "tensorrt", "mock"], default="pytorch"
    )
    parser.add_argument("--concepts", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trt-engine", type=Path, default=None)
    parser.add_argument("--trt-model-id", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area-px", type=int, default=None)
    parser.add_argument("--inference-hz", type=float, default=None)
    parser.add_argument("--sync-tolerance-ms", type=float, default=None)
    parser.add_argument("--min-depth-m", type=float, default=None)
    parser.add_argument("--max-depth-m", type=float, default=None)
    parser.add_argument("--min-depth-points", type=int, default=None)
    parser.add_argument("--robot-state-max-delta-ms", type=float, default=None)
    parser.add_argument("--torso-estimator", default=None)
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--object-max-points", type=int, default=None)
    parser.add_argument("--geometry-fallback-ttl-frames", type=int, default=None)
    contacts = parser.add_mutually_exclusive_group()
    contacts.add_argument(
        "--enable-contact-candidates",
        dest="contact_candidates",
        action="store_true",
    )
    contacts.add_argument(
        "--disable-contact-candidates",
        dest="contact_candidates",
        action="store_false",
    )
    parser.set_defaults(contact_candidates=True)
    parser.add_argument("--publish-overlay-jpeg", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    segmentation = dict(config.get("segmentation", {}))
    sam_options = dict(segmentation.get("sam3", {}))
    tensorrt = dict(segmentation.get("tensorrt", {}))
    laptop_tensorrt = dict(sam_options.get("tensorrt", {}))
    tensorrt.update(laptop_tensorrt)
    depth_config = dict(config.get("depth", {}))
    cloud_config = dict(config.get("cloud", {}))
    target_config = dict(config.get("ee_target", {}))
    tasc_config = dict(config.get("tasc_belief", {}))
    track_config = dict(sam_options.get("track", {}))

    concepts_value = args.concepts
    if concepts_value is None:
        configured = sam_options.get("concepts", segmentation.get("concepts", ["object"]))
        concepts_value = (
            ",".join(str(value) for value in configured)
            if isinstance(configured, (list, tuple))
            else str(configured)
        )

    contact_data = dict(config.get("grip", {}))
    contact_data.update(dict(tasc_config.get("grip", {})))
    contact_data["max_points"] = int(
        args.object_max_points
        if args.object_max_points is not None
        else tasc_config.get(
            "object_max_points",
            contact_data.get("max_points", cloud_config.get("object_max_points", 8000)),
        )
    )
    fallback_ttl = int(
        args.geometry_fallback_ttl_frames
        if args.geometry_fallback_ttl_frames is not None
        else track_config.get(
            "geometry_fallback_ttl_frames",
            track_config.get("missing_ttl_frames", segmentation.get("miss_ttl", 10)),
        )
    )
    extrinsics_path = args.extrinsics or (
        Path(str(target_config["extrinsics"]))
        if target_config.get("extrinsics")
        else None
    )

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = RemoteSam3BeliefInputWorker(
        wifi_lcm_url=args.wifi_lcm_url,
        concepts=parse_concepts(concepts_value),
        backend_name=args.backend,
        checkpoint=args.checkpoint or (
            Path(str(sam_options["checkpoint"]))
            if sam_options.get("checkpoint")
            else (
                Path(str(segmentation["checkpoint"]))
                if segmentation.get("checkpoint")
                else None
            )
        ),
        # The top-level engine belongs to Thor and must never be deserialized
        # on a different GPU. A laptop engine requires an explicit CLI path or
        # segmentation.sam3.tensorrt.engine.
        trt_engine=args.trt_engine or (
            Path(str(laptop_tensorrt["engine"]))
            if laptop_tensorrt.get("engine")
            else None
        ),
        trt_model_id=str(
            args.trt_model_id or tensorrt.get("model_id", "facebook/sam3")
        ),
        threshold=float(
            args.threshold
            if args.threshold is not None
            else sam_options.get("threshold", segmentation.get("min_score", 0.5))
        ),
        min_area_px=int(
            args.min_area_px
            if args.min_area_px is not None
            else sam_options.get("min_area_px", segmentation.get("min_area_px", 400))
        ),
        inference_hz=float(
            args.inference_hz
            if args.inference_hz is not None
            else sam_options.get("target_hz", segmentation.get("target_hz", 3.0))
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
        depth_self_filter=dict(segmentation.get("depth_self_filter", {})),
        object_point_filter=ObjectPointFilterConfig.from_dict(
            dict(cloud_config.get("object_filter", {}))
        ),
        contact_config=GripCandidateConfig.from_dict(contact_data),
        pseudo_config=_pseudo_config(tasc_config),
        contact_candidates=bool(args.contact_candidates),
        geometry_fallback_ttl_frames=fallback_ttl,
        geometry_fallback_score_decay=float(
            track_config.get("geometry_fallback_score_decay", 0.97)
        ),
    )
    worker.start()
    LOG.info(
        "remote SAM3 belief-input worker started; Wi-Fi perception output only, "
        "no control publishers"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOG.info("stopping remote SAM3 belief-input worker")
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
