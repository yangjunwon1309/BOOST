"""External desktop SAM3 worker for PATH-G1 Wi-Fi offload."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import lcm
import numpy as np
import yaml

from seg_web_node.depth_alignment import align_depth_to_color
from seg_web_node.geometry import (
    CameraIntrinsics,
    deproject,
    robust_mask_center,
    transform_points,
)
from seg_web_node.lcm_protocol import (
    DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    CameraCalibration,
    decode_calibration,
    decode_camera,
    decode_depth,
    decode_robot_state,
)
from seg_web_node.rgbd_sync import RgbdPair, RgbdSynchronizer, depth_at_rgb_size
from seg_web_node.robot_geometry import RobotStateBuffer, camera_goal_transform
from seg_web_node.sam3_backend import TensorRTSam3Backend, TorchSam3Backend
from seg_web_node.visualization import render_segmented_rgbd

from .protocol import (
    REMOTE_CALIBRATION_CHANNEL,
    REMOTE_DEPTH_CHANNEL,
    REMOTE_HEARTBEAT_CHANNEL,
    REMOTE_OVERLAY_CHANNEL,
    REMOTE_RESULT_CHANNEL,
    REMOTE_RGB_CHANNEL,
    REMOTE_ROBOT_STATE_CHANNEL,
    make_empty_result,
    make_object,
    now_us,
    pack_heartbeat,
    pack_result,
)


LOG = logging.getLogger("remote_offload.remote_sam3_worker")


def parse_concepts(text: str) -> list[str]:
    concepts = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not concepts:
        raise ValueError("at least one concept is required")
    return concepts


def suppress_duplicate_masks(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    iou_threshold: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not len(masks):
        return masks, boxes, scores, labels
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    keep: list[int] = []
    for index in np.argsort(-scores):
        duplicate = False
        for kept in keep:
            intersection = np.logical_and(masks[index], masks[kept]).sum()
            union = areas[index] + areas[kept] - intersection
            if union > 0 and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(index))
    keep_array = np.asarray(keep, dtype=np.int64)
    return masks[keep_array], boxes[keep_array], scores[keep_array], [labels[i] for i in keep]


class Stable3DTracker:
    def __init__(self, *, max_distance_m: float = 0.25, missing_ttl_frames: int = 5) -> None:
        self.max_distance_m = float(max_distance_m)
        self.missing_ttl_frames = int(missing_ttl_frames)
        self.tracks: dict[str, dict[str, Any]] = {}
        self.next_id = 0

    def assign(self, detections: list[dict[str, Any]]) -> dict[int, str]:
        assignments: dict[int, str] = {}
        used: set[str] = set()
        for detection in detections:
            center = detection.get("center_goal_m") or detection.get("center_camera_m")
            if center is None:
                continue
            center_array = np.asarray(center, dtype=np.float64).reshape(3)
            if not np.isfinite(center_array).all():
                continue
            label = str(detection.get("label", "object"))
            best_id = ""
            best_distance = float("inf")
            for track_id, track in self.tracks.items():
                if track_id in used or str(track["label"]) != label:
                    continue
                distance = float(np.linalg.norm(center_array - np.asarray(track["center"])))
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance
            if best_id and best_distance <= self.max_distance_m:
                track_id = best_id
            else:
                track_id = f"{label}:track{self.next_id:04d}"
                self.next_id += 1
            self.tracks[track_id] = {
                "label": label,
                "center": center_array.copy(),
                "missing": 0,
            }
            used.add(track_id)
            assignments[int(detection["index"])] = track_id
        for track_id in list(self.tracks):
            if track_id in used:
                continue
            self.tracks[track_id]["missing"] = int(self.tracks[track_id]["missing"]) + 1
            if self.tracks[track_id]["missing"] > self.missing_ttl_frames:
                del self.tracks[track_id]
        return assignments


class MockSam3Backend:
    """Synthetic center mask backend for transport validation without SAM3 weights."""

    name = "mock"

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "mock", "device": "cpu"}

    def infer(self, image_bgr: np.ndarray, concepts: list[str]):
        height, width = image_bgr.shape[:2]
        mask = np.zeros((1, height, width), dtype=bool)
        y0, y1 = height // 4, 3 * height // 4
        x0, x1 = width // 4, 3 * width // 4
        mask[0, y0:y1, x0:x1] = True
        boxes = np.asarray([[x0, y0, x1 - 1, y1 - 1]], dtype=np.float32)
        scores = np.asarray([0.99], dtype=np.float32)
        return mask, boxes, scores, [concepts[0] if concepts else "object"]


def load_extrinsics(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    data = yaml.safe_load(path.read_text())
    key = "T_torso_cam" if data.get("T_torso_cam") is not None else "T_goal_cam"
    transform = np.asarray(data[key], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{key} must be a finite 4x4 matrix")
    return transform


class RemoteSam3Worker:
    def __init__(
        self,
        *,
        wifi_lcm_url: str,
        concepts: list[str],
        backend_name: str,
        checkpoint: Path | None,
        trt_engine: Path | None,
        trt_model_id: str,
        threshold: float,
        min_area_px: int,
        inference_hz: float,
        sync_tolerance_ms: float,
        min_depth_m: float,
        max_depth_m: float,
        min_depth_points: int,
        robot_state_max_delta_ms: float,
        torso_estimator: str,
        t_torso_cam: np.ndarray | None,
        publish_overlay_jpeg: bool,
    ) -> None:
        self.bus = lcm.LCM(wifi_lcm_url)
        self.concepts = concepts
        self.backend_name = backend_name
        self.checkpoint = checkpoint
        self.trt_engine = trt_engine
        self.trt_model_id = trt_model_id
        self.threshold = float(threshold)
        self.min_area_px = int(min_area_px)
        self.inference_period = 1.0 / max(0.1, float(inference_hz))
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_depth_points = int(min_depth_points)
        self.robot_state_max_delta_ms = float(robot_state_max_delta_ms)
        self.torso_estimator = torso_estimator
        self.t_torso_cam = t_torso_cam
        self.publish_overlay_jpeg = publish_overlay_jpeg
        self.sync = RgbdSynchronizer(sync_tolerance_ms, capacity=12)
        self.state_buffer = RobotStateBuffer()
        self.tracker = Stable3DTracker()
        self.lock = threading.Lock()
        self.pair_ready = threading.Condition(self.lock)
        self.pending_pair: RgbdPair | None = None
        self.calibration: CameraCalibration | None = None
        self.running = False
        self.sequence = 0
        self.backend = None
        self.backend_error: str | None = None
        self.last_result_age_ms: float | None = None
        self.status: dict[str, Any] = {
            "model_ready": False,
            "input_frames": 0,
            "processed_frames": 0,
            "dropped_frames": 0,
            "decode_errors": 0,
            "last_error": None,
            "last_source_timestamp_us": None,
            "last_inference_ms": None,
            "last_object_count": 0,
        }
        self.last_status_log_monotonic = 0.0
        self.threads: list[threading.Thread] = []
        self.bus.subscribe(REMOTE_RGB_CHANNEL, self._on_rgb)
        self.bus.subscribe(REMOTE_DEPTH_CHANNEL, self._on_depth)
        self.bus.subscribe(REMOTE_CALIBRATION_CHANNEL, self._on_calibration)
        self.bus.subscribe(REMOTE_ROBOT_STATE_CHANNEL, self._on_robot_state)

    def start(self) -> None:
        self.running = True
        self.threads = [
            threading.Thread(target=self._lcm_loop, name="remote-lcm", daemon=True),
            threading.Thread(target=self._inference_loop, name="remote-sam3", daemon=True),
            threading.Thread(target=self._heartbeat_loop, name="remote-heartbeat", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.pair_ready.notify_all()
        for thread in self.threads:
            thread.join(timeout=2.0)

    def _record_error(self, error: Exception | str) -> None:
        with self.lock:
            self.status["decode_errors"] += 1
            self.status["last_error"] = str(error)

    def _submit_pair(self, pair: RgbdPair | None) -> None:
        if pair is None:
            return
        with self.pair_ready:
            if self.pending_pair is not None:
                self.status["dropped_frames"] += 1
            self.pending_pair = pair
            self.status["input_frames"] += 1
            self.pair_ready.notify()

    def _on_rgb(self, _channel: str, data: bytes) -> None:
        try:
            self._submit_pair(self.sync.add_rgb(decode_camera(data)))
        except Exception as error:
            self._record_error(error)

    def _on_depth(self, _channel: str, data: bytes) -> None:
        try:
            self._submit_pair(self.sync.add_depth(decode_depth(data, allow_unaligned=True)))
        except Exception as error:
            self._record_error(error)

    def _on_calibration(self, _channel: str, data: bytes) -> None:
        try:
            calibration = decode_calibration(data)
            with self.lock:
                self.calibration = calibration
        except Exception as error:
            self._record_error(error)

    def _on_robot_state(self, _channel: str, data: bytes) -> None:
        try:
            state = decode_robot_state(data)
            with self.lock:
                self.state_buffer.add(state)
        except Exception as error:
            self._record_error(error)

    def _lcm_loop(self) -> None:
        while self.running:
            try:
                self.bus.handle_timeout(100)
            except Exception as error:
                self._record_error(error)
                time.sleep(0.1)

    def _load_backend(self):
        if self.backend_name == "mock":
            return MockSam3Backend(), "mock"
        if self.backend_name == "pytorch":
            return (
                TorchSam3Backend(
                    checkpoint=self.checkpoint,
                    threshold=self.threshold,
                    min_area_px=self.min_area_px,
                ),
                "sam3_pytorch",
            )
        if self.backend_name == "tensorrt":
            if self.trt_engine is None:
                raise ValueError("--trt-engine is required for --backend tensorrt")
            return (
                TensorRTSam3Backend(
                    engine_path=self.trt_engine,
                    model_id=self.trt_model_id,
                    threshold=self.threshold,
                    min_area_px=self.min_area_px,
                ),
                "sam3_tensorrt",
            )
        raise ValueError(f"unknown backend: {self.backend_name}")

    def _backend_display_name(self) -> str:
        return "SAM3"

    def _error_wire_backend_name(self) -> str:
        return "sam3_tensorrt" if self.backend_name == "tensorrt" else "sam3_pytorch"

    def _prepare_backend_for_frame(
        self,
        backend: Any,
        image_bgr: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics | None,
        pair: RgbdPair,
    ) -> None:
        del backend, image_bgr, depth, intrinsics, pair

    def _invalid_result(self, pair: RgbdPair, error: str) -> dict[str, Any]:
        self.sequence += 1
        return make_empty_result(
            sequence=self.sequence,
            source_rgb_timestamp_us=pair.rgb.timestamp_us,
            source_depth_timestamp_us=pair.depth.timestamp_us,
            calibration_id=self.calibration.calibration_id if self.calibration else None,
            image_width=pair.rgb.width,
            image_height=pair.rgb.height,
            backend=self._error_wire_backend_name(),
            valid=False,
            error=error,
        )

    def _prepare_depth_and_intrinsics(
        self,
        pair: RgbdPair,
        calibration: CameraCalibration | None,
    ) -> tuple[np.ndarray, CameraIntrinsics | None]:
        if pair.depth.source == DEPTH_SOURCE_RAW_DEPTH_OPTICAL:
            if calibration is None:
                raise ValueError("raw depth received before remote camera calibration")
            depth = align_depth_to_color(pair.depth, calibration)
            intrinsics = CameraIntrinsics(
                calibration.color_width,
                calibration.color_height,
                *calibration.color_intrinsics,
            )
            return depth, intrinsics
        depth = depth_at_rgb_size(pair)
        if calibration is None:
            return depth, None
        return (
            depth,
            CameraIntrinsics(
                calibration.color_width,
                calibration.color_height,
                *calibration.color_intrinsics,
            ),
        )

    def _goal_transform(self, timestamp_us: int) -> np.ndarray | None:
        return camera_goal_transform(
            self.t_torso_cam,
            self.torso_estimator,
            self.state_buffer,
            timestamp_us,
            self.robot_state_max_delta_ms,
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

        detections: list[dict[str, Any]] = []
        for index, (mask, label) in enumerate(zip(masks, labels)):
            center_camera = None
            center_goal = None
            if xyz_camera is not None:
                center = robust_mask_center(
                    xyz_camera,
                    mask,
                    min_valid_points=self.min_depth_points,
                )
                if center is not None:
                    center_camera_array, _pixel, _valid_points = center
                    center_camera = center_camera_array.tolist()
                    if t_goal_cam is not None:
                        center_goal = transform_points(center_camera_array, t_goal_cam).tolist()
            detections.append(
                {
                    "index": index,
                    "label": str(label),
                    "center_camera_m": center_camera,
                    "center_goal_m": center_goal,
                }
            )
        track_ids = self.tracker.assign(detections)
        for index, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
            detection = detections[index]
            payload["objects"].append(
                make_object(
                    track_id=track_ids.get(index, f"object:untracked{index:04d}"),
                    label=str(label),
                    score=float(score),
                    box_xyxy=[float(value) for value in box],
                    mask=mask,
                    median_depth_m=None,
                    center_pixel_uv=None,
                    center_camera_m=detection.get("center_camera_m"),
                    center_goal_m=detection.get("center_goal_m"),
                    valid_depth_points=0,
                )
            )
        # Fill depth fields from the detections in a second pass to keep the
        # tracking assignment independent of transport serialization.
        for index, obj in enumerate(payload["objects"]):
            mask = masks[index].astype(bool)
            depth_m = depth.astype(np.float32) * (pair.depth.depth_scale_um / 1_000_000.0)
            valid_depth = mask & (depth > 0) & np.isfinite(depth_m)
            values = depth_m[valid_depth]
            obj["median_depth_m"] = float(np.median(values)) if values.size else None
            if xyz_camera is not None:
                center = robust_mask_center(
                    xyz_camera,
                    mask,
                    min_valid_points=self.min_depth_points,
                )
                if center is not None:
                    center_camera_array, pixel, valid_points = center
                    obj["center_pixel_uv"] = [int(pixel[0]), int(pixel[1])]
                    obj["center_camera_m"] = center_camera_array.tolist()
                    obj["center_goal_m"] = (
                        transform_points(center_camera_array, t_goal_cam).tolist()
                        if t_goal_cam is not None
                        else None
                    )
                    obj["valid_depth_points"] = int(valid_points)
        return payload

    def _publish_overlay(
        self,
        image_bgr: np.ndarray,
        depth: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[str],
        depth_scale_um: int,
    ) -> None:
        if not self.publish_overlay_jpeg:
            return
        overlay, _objects = render_segmented_rgbd(
            image_bgr,
            depth,
            masks,
            boxes,
            scores,
            labels,
            depth_scale_um,
            self.min_depth_m,
            self.max_depth_m,
        )
        ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.bus.publish(REMOTE_OVERLAY_CHANNEL, encoded.tobytes())

    def _inference_loop(self) -> None:
        try:
            self.backend, backend_wire_name = self._load_backend()
            with self.lock:
                self.status["model_ready"] = True
        except Exception as error:
            backend_wire_name = self._error_wire_backend_name()
            self.backend_error = f"{self._backend_display_name()} load failed: {error}"
            with self.lock:
                self.status["model_ready"] = False
                self.status["last_error"] = self.backend_error
            LOG.exception("%s backend load failed", self._backend_display_name())

        last_started = 0.0
        while self.running:
            with self.pair_ready:
                self.pair_ready.wait_for(
                    lambda: self.pending_pair is not None or not self.running,
                    timeout=0.5,
                )
                if not self.running:
                    break
                pair = self.pending_pair
                self.pending_pair = None
                calibration = self.calibration
            if pair is None:
                continue
            delay = self.inference_period - (time.monotonic() - last_started)
            if delay > 0:
                time.sleep(delay)
            last_started = time.monotonic()

            try:
                if self.backend_error is not None or self.backend is None:
                    result = self._invalid_result(
                        pair,
                        self.backend_error
                        or f"{self._backend_display_name()} backend unavailable",
                    )
                    self.bus.publish(REMOTE_RESULT_CHANNEL, pack_result(result))
                    continue
                encoded = np.frombuffer(pair.rgb.jpeg, dtype=np.uint8)
                image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError("JPEG decoding failed")
                depth, intrinsics = self._prepare_depth_and_intrinsics(pair, calibration)
                self._prepare_backend_for_frame(
                    self.backend,
                    image_bgr,
                    depth,
                    intrinsics,
                    pair,
                )
                started = time.perf_counter()
                masks, boxes, scores, labels = self.backend.infer(image_bgr, self.concepts)
                inference_ms = (time.perf_counter() - started) * 1000.0
                if len(masks):
                    masks, boxes, scores, labels = suppress_duplicate_masks(
                        masks,
                        boxes,
                        scores,
                        labels,
                    )
                result = self._build_result(
                    pair,
                    backend_wire_name,
                    inference_ms,
                    masks,
                    boxes,
                    scores,
                    labels,
                    depth,
                    intrinsics,
                )
                result_bytes = pack_result(result)
                self.bus.publish(REMOTE_RESULT_CHANNEL, result_bytes)
                self._publish_overlay(
                    image_bgr,
                    depth,
                    masks,
                    boxes,
                    scores,
                    labels,
                    pair.depth.depth_scale_um,
                )
                with self.lock:
                    self.status["processed_frames"] += 1
                    self.status["last_source_timestamp_us"] = pair.rgb.timestamp_us
                    self.status["last_inference_ms"] = inference_ms
                    self.status["last_object_count"] = len(result.get("objects", []))
                    self.status["last_error"] = None
            except Exception as error:
                self._record_error(error)
                result = self._invalid_result(pair, str(error))
                self.bus.publish(REMOTE_RESULT_CHANNEL, pack_result(result))

    def _heartbeat_loop(self) -> None:
        while self.running:
            with self.lock:
                payload = {
                    "version": 1,
                    "published_timestamp_us": now_us(),
                    "model_ready": bool(self.status.get("model_ready")),
                    "backend": self.backend_name,
                    "input_frame_count": int(self.status.get("input_frames", 0)),
                    "processed_count": int(self.status.get("processed_frames", 0)),
                    "dropped_count": int(self.status.get("dropped_frames", 0)),
                    "decode_errors": int(self.status.get("decode_errors", 0)),
                    "inference_latency_ms": self.status.get("last_inference_ms"),
                    "latest_source_timestamp_us": self.status.get("last_source_timestamp_us"),
                    "last_error": self.status.get("last_error"),
                }
            now = time.monotonic()
            if now - self.last_status_log_monotonic >= 5.0:
                self.last_status_log_monotonic = now
                LOG.info(
                    "%s status ready=%s input=%s processed=%s dropped=%s "
                    "objects=%s latency_ms=%s last_error=%s",
                    self._backend_display_name(),
                    payload["model_ready"],
                    payload["input_frame_count"],
                    payload["processed_count"],
                    payload["dropped_count"],
                    self.status.get("last_object_count"),
                    payload["inference_latency_ms"],
                    payload["last_error"],
                )
            self.bus.publish(REMOTE_HEARTBEAT_CHANNEL, pack_heartbeat(payload))
            time.sleep(1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--backend", choices=["pytorch", "tensorrt", "mock"], default="pytorch")
    parser.add_argument("--concepts", default="object")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trt-engine", type=Path, default=None)
    parser.add_argument("--trt-model-id", default="facebook/sam3")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-px", type=int, default=400)
    parser.add_argument("--inference-hz", type=float, default=3.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--min-depth-points", type=int, default=100)
    parser.add_argument("--robot-state-max-delta-ms", type=float, default=40.0)
    parser.add_argument("--torso-estimator", default="dual_imu")
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--publish-overlay-jpeg", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = RemoteSam3Worker(
        wifi_lcm_url=args.wifi_lcm_url,
        concepts=parse_concepts(args.concepts),
        backend_name=args.backend,
        checkpoint=args.checkpoint,
        trt_engine=args.trt_engine,
        trt_model_id=args.trt_model_id,
        threshold=args.threshold,
        min_area_px=args.min_area_px,
        inference_hz=args.inference_hz,
        sync_tolerance_ms=args.sync_tolerance_ms,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_depth_points=args.min_depth_points,
        robot_state_max_delta_ms=args.robot_state_max_delta_ms,
        torso_estimator=args.torso_estimator,
        t_torso_cam=load_extrinsics(args.extrinsics) if args.extrinsics else None,
        publish_overlay_jpeg=args.publish_overlay_jpeg,
    )
    worker.start()
    LOG.info("remote SAM3 worker started on Wi-Fi LCM; publishes perception results only")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOG.info("stopping remote SAM3 worker")
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
