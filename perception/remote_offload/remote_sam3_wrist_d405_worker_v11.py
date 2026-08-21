"""Remote SAM3 worker for one vertically stacked pair of wrist D405 frames."""

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

from seg_web_node.lcm_protocol import decode_robot_state
from seg_web_node.grip_candidates import (
    GripCandidateConfig,
    PersistentCylinderSurfaceTracker,
    PseudoContactConfig,
)
from seg_web_node.geometry_fallback import LastValidGeometryFallback
from seg_web_node.object_point_filter import (
    ObjectPointFilterConfig,
    compute_depth_edge_mask,
)
from seg_web_node.robot_geometry import (
    RobotStateBuffer,
    g1_link_transforms,
    gravity_up_in_pelvis,
)
from seg_web_node.sam3_backend import TensorRTSam3Backend, TorchSam3Backend

from .wrist_d405_geometry_pipeline_v11 import (
    CAMERA_TO_WRIST_LINK,
    camera_transform_diagnostics,
    load_wrist_extrinsics,
    attach_surface_geometry,
    object_geometry,
)
from .wrist_d405_protocol_v11 import (
    CAMERA_IDS,
    WRIST_BUNDLE_CHANNEL,
    WRIST_CONCEPT_CONTROL_CHANNEL,
    WRIST_HEARTBEAT_CHANNEL,
    WRIST_RESULT_CHANNEL,
    WRIST_ROBOT_STATE_CHANNEL,
    make_result,
    now_us,
    pack_heartbeat,
    pack_mask,
    pack_result,
    unpack_bundle,
    unpack_concept_control,
    unpack_depth_z16,
)


LOG = logging.getLogger(__name__)


def parse_concepts(text: str) -> list[str]:
    concepts = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not concepts:
        raise ValueError("at least one SAM3 concept is required")
    return concepts


def load_surface_configs(path: Path) -> tuple[ObjectPointFilterConfig, GripCandidateConfig, PseudoContactConfig]:
    """Load the same robust cloud/surface settings as the prior SAM3 path."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"belief config must be a mapping: {path}")
    belief = data.get("tasc_belief", data.get("belief", {}))
    if not isinstance(belief, dict):
        raise ValueError(f"belief configuration must be a mapping: {path}")
    orientation = belief.get("orientation", {})
    return (
        ObjectPointFilterConfig.from_dict(belief.get("object_point_filter")),
        GripCandidateConfig.from_dict(belief.get("grip")),
        PseudoContactConfig.from_dict(orientation.get("pseudo") if isinstance(orientation, dict) else None),
    )


def stack_wrist_images(frames: dict[str, dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Return left-above-right BGR input, refusing mismatched camera grids."""
    decoded: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for camera_id in CAMERA_IDS:
        frame = frames[camera_id]
        image = cv2.imdecode(np.frombuffer(frame["color_jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"invalid JPEG from {camera_id}")
        expected = (int(frame["height"]), int(frame["width"]))
        if image.shape[:2] != expected:
            raise ValueError(f"JPEG grid mismatch from {camera_id}")
        if shape is None:
            shape = expected
        elif shape != expected:
            raise ValueError("left/right D405 images must have a common native grid")
        decoded[camera_id] = image
    assert shape is not None
    return np.vstack((decoded["left_wrist"], decoded["right_wrist"])), shape[0]


def attribute_stacked_masks(masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: list[str],
                            *, split_y: int, seam_guard_px: int = 0,
                            seam_minority_ratio: float = 0.05) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Attribute masks by dominant image, rejecting only true cross-camera merges."""
    output: dict[str, list[dict[str, Any]]] = {camera_id: [] for camera_id in CAMERA_IDS}
    discarded = 0
    del seam_guard_px  # Retained CLI compatibility; seam touching is normal for wrist views.
    minority_limit = float(np.clip(seam_minority_ratio, 0.0, 0.5))
    for mask, box, score, label in zip(np.asarray(masks, dtype=bool), boxes, scores, labels):
        if mask.ndim != 2 or mask.shape[0] != 2 * int(split_y):
            raise ValueError("SAM3 mask does not match the vertical stack")
        top, bottom = int(mask[:split_y].sum()), int(mask[split_y:].sum())
        total = top + bottom
        if total == 0:
            discarded += 1
            continue
        # A mask on the seam is valid. Only substantial support in both images
        # indicates that SAM3 merged two physically independent cameras.
        if min(top, bottom) / total > minority_limit:
            discarded += 1
            continue
        camera_id = "left_wrist" if top > bottom else "right_wrist"
        local_mask = mask[:split_y] if camera_id == "left_wrist" else mask[split_y:]
        local_box = np.asarray(box, dtype=np.float64).copy()
        if camera_id == "right_wrist":
            local_box[[1, 3]] -= split_y
        local_box[[0, 2]] = np.clip(local_box[[0, 2]], 0, local_mask.shape[1] - 1)
        local_box[[1, 3]] = np.clip(local_box[[1, 3]], 0, local_mask.shape[0] - 1)
        output[camera_id].append({"mask": local_mask, "box": local_box, "score": float(score), "label": str(label)})
    return output, discarded


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = np.asarray(first, dtype=np.float64).reshape(4)
    bx0, by0, bx1, by1 = np.asarray(second, dtype=np.float64).reshape(4)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = (
        max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        + max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        - intersection
    )
    return 0.0 if union <= 0.0 else intersection / union


def suppress_camera_duplicate_masks(
    detections: list[dict[str, Any]], *, iou_threshold: float = 0.80
) -> tuple[list[dict[str, Any]], int]:
    """Suppress SAM3 duplicates only within a physical camera image."""
    if len(detections) < 2:
        return detections, 0
    areas = [int(np.asarray(item["mask"], dtype=bool).sum()) for item in detections]
    kept: list[int] = []
    for index in sorted(range(len(detections)), key=lambda item: -float(detections[item]["score"])):
        duplicate = False
        for previous in kept:
            if _box_iou(detections[index]["box"], detections[previous]["box"]) < iou_threshold * 0.5:
                continue
            intersection = int(
                np.logical_and(detections[index]["mask"], detections[previous]["mask"]).sum()
            )
            union = areas[index] + areas[previous] - intersection
            if union > 0 and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    return [detections[index] for index in kept], len(detections) - len(kept)


class MockSam3Backend:
    name = "mock"

    def infer(self, image_bgr: np.ndarray, concepts: list[str]):
        height, width = image_bgr.shape[:2]
        mask = np.zeros((1, height, width), dtype=bool)
        mask[0, height // 8:height // 3, width // 4:3 * width // 4] = True
        return mask, np.asarray([[width / 4, height / 8, 3 * width / 4, height / 3]]), np.asarray([0.99]), [concepts[0]]


class WristD405Sam3Worker:
    def __init__(self, *, wifi_lcm_url: str, concepts: list[str], backend: Any, extrinsics: dict[str, np.ndarray],
                 seam_guard_px: int, min_depth_m: float, max_depth_m: float, min_depth_points: int,
                 robot_state_max_delta_ms: float, point_filter: ObjectPointFilterConfig | None = None,
                 contact_config: GripCandidateConfig | None = None, pseudo_config: PseudoContactConfig | None = None,
                 seam_minority_ratio: float = 0.05,
                 track_association_radius_m: float = 0.06,
                 geometry_fallback_ttl_frames: int = 3,
                 geometry_fallback_score_decay: float = 0.97) -> None:
        self.bus = lcm.LCM(wifi_lcm_url)
        self.concepts, self.backend, self.extrinsics = concepts, backend, extrinsics
        self.seam_guard_px, self.min_depth_m, self.max_depth_m = int(seam_guard_px), float(min_depth_m), float(max_depth_m)
        self.seam_minority_ratio = float(np.clip(seam_minority_ratio, 0.0, 0.5))
        self.min_depth_points, self.robot_state_max_delta_ms = int(min_depth_points), float(robot_state_max_delta_ms)
        self.track_association_radius_m = max(0.0, float(track_association_radius_m))
        self.states = RobotStateBuffer()
        self.point_filter = point_filter or ObjectPointFilterConfig()
        self.contact_config = contact_config or GripCandidateConfig()
        self.pseudo_config = pseudo_config or PseudoContactConfig()
        self.surface_trackers = {
            camera_id: PersistentCylinderSurfaceTracker(self.contact_config)
            for camera_id in CAMERA_IDS
        }
        self.geometry_fallbacks = {
            camera_id: LastValidGeometryFallback(
                geometry_fallback_ttl_frames,
                geometry_fallback_score_decay,
                fallback_reason="wrist_sam3_detection_missing",
            )
            for camera_id in CAMERA_IDS
        }
        self.lock = threading.Lock()
        self.latest_bundle: dict[str, Any] | None = None
        self.running = False
        self.sequence = 0
        self.last_geometry_log_monotonic = 0.0
        self.warned_unpaired_robot_state = False
        self.concept_revision = -1
        self.next_track_id = {camera_id: 0 for camera_id in CAMERA_IDS}
        self.tracks: dict[str, dict[str, tuple[np.ndarray, int]]] = {camera_id: {} for camera_id in CAMERA_IDS}
        self.stats: dict[str, Any] = {
            "state_age_rejects": 0,
            "suppressed_duplicate_masks": 0,
            "geometry_fallback_objects": 0,
        }
        self.bus.subscribe(WRIST_BUNDLE_CHANNEL, self._on_bundle)
        self.bus.subscribe(WRIST_ROBOT_STATE_CHANNEL, self._on_robot_state)
        self.bus.subscribe(WRIST_CONCEPT_CONTROL_CHANNEL, self._on_concept_control)

    def _publish_heartbeat(self) -> None:
        """Acknowledge the active prompt set for the legacy-compatible viewer UI."""
        with self.lock:
            concepts = list(self.concepts)
            revision = self.concept_revision
        self.bus.publish(
            WRIST_HEARTBEAT_CHANNEL,
            pack_heartbeat(
                {
                    "version": 1,
                    "published_timestamp_us": now_us(),
                    "active_concepts": concepts,
                    "pending_concepts": None,
                    "concept_revision": revision,
                    "backend": getattr(self.backend, "name", type(self.backend).__name__),
                }
            ),
        )

    def _on_bundle(self, _channel: str, data: bytes) -> None:
        try:
            with self.lock:
                self.latest_bundle = unpack_bundle(data)
        except Exception as error:
            LOG.warning("rejected wrist D405 bundle: %s", error)

    def _on_robot_state(self, _channel: str, data: bytes) -> None:
        try:
            with self.lock:
                self.states.add(decode_robot_state(data))
        except Exception as error:
            LOG.warning("rejected wrist robot state: %s", error)

    def _on_concept_control(self, _channel: str, data: bytes) -> None:
        try:
            control = unpack_concept_control(data)
            with self.lock:
                if control["revision"] <= self.concept_revision:
                    return
                self.concepts = list(control["concepts"])
                self.concept_revision = int(control["revision"])
            LOG.info("applied wrist SAM3 concepts revision=%d concepts=%s", self.concept_revision, self.concepts)
            self._publish_heartbeat()
        except Exception as error:
            LOG.warning("rejected wrist concept control: %s", error)

    def process_latest(self) -> dict[str, Any] | None:
        with self.lock:
            bundle = self.latest_bundle
            self.latest_bundle = None
            concepts = list(self.concepts)
        if bundle is None:
            return None
        frames = bundle["cameras"]
        paired_packet = bundle.get("robot_state_packet")
        paired_state = (
            decode_robot_state(paired_packet)
            if isinstance(paired_packet, (bytes, bytearray))
            else None
        )
        paired_state_received_us = int(
            bundle.get("robot_state_host_received_us", 0) or 0
        )
        state_age_by_camera: dict[str, float | None] = {}
        if paired_state is not None and paired_state_received_us > 0:
            state_age_by_camera = {
                camera_id: abs(
                    int(frames[camera_id]["host_timestamp_us"])
                    - paired_state_received_us
                )
                / 1000.0
                for camera_id in CAMERA_IDS
            }
        elif not self.warned_unpaired_robot_state:
            LOG.warning(
                "bundle has no timestamped paired robot state; rejecting geometry until Thor sends one"
            )
            self.warned_unpaired_robot_state = True

        state_age_ms = (
            max(value for value in state_age_by_camera.values() if value is not None)
            if state_age_by_camera
            else None
        )
        state_valid = (
            paired_state is not None
            and state_age_ms is not None
            and state_age_ms <= self.robot_state_max_delta_ms
        )
        if not state_valid:
            self.stats["state_age_rejects"] = int(
                self.stats.get("state_age_rejects", 0)
            ) + 1
            reason = (
                "robot_state_unavailable"
                if paired_state is None or state_age_ms is None
                else "robot_state_stale"
            )
            self.sequence += 1
            result = make_result(
                sequence=self.sequence,
                source_bundle_timestamp_us=int(bundle["bundle_timestamp_us"]),
                source_bundle_sequence=int(bundle.get("sequence", -1)),
                objects=[],
                discarded_seam_masks=0,
                geometry_status=self._geometry_status(
                    bundle,
                    state_available=False,
                    state_age_ms=state_age_ms,
                    primitive_count=0,
                    objects_with_valid_depth=0,
                    discarded_seam_masks=0,
                ),
                valid=False,
                error=reason,
            )
            self.bus.publish(WRIST_RESULT_CHANNEL, pack_result(result))
            return result

        # The paired state is mandatory above. One FK evaluation serves every
        # object in this atomic two-camera bundle.
        link_transforms = g1_link_transforms(np.asarray(paired_state.q, dtype=np.float64))
        t_goal_cameras: dict[str, np.ndarray] = {}
        transform_diagnostics: dict[str, dict[str, Any]] = {}
        for camera_id in CAMERA_IDS:
            goal_wrist = link_transforms[CAMERA_TO_WRIST_LINK[camera_id]]
            goal_camera = goal_wrist @ self.extrinsics[camera_id]
            t_goal_cameras[camera_id] = goal_camera
            transform_diagnostics[camera_id] = camera_transform_diagnostics(
                camera_id,
                self.extrinsics,
                goal_wrist=goal_wrist,
                goal_camera=goal_camera,
            )
        up_goal = gravity_up_in_pelvis(paired_state.base_quat_wxyz)
        stacked, split_y = stack_wrist_images(frames)
        masks, boxes, scores, labels = self.backend.infer(stacked, concepts)
        attributed, discarded = attribute_stacked_masks(
            masks,
            boxes,
            scores,
            labels,
            split_y=split_y,
            seam_guard_px=self.seam_guard_px,
            seam_minority_ratio=self.seam_minority_ratio,
        )
        suppressed_duplicates = 0
        for camera_id in CAMERA_IDS:
            attributed[camera_id], count = suppress_camera_duplicate_masks(
                attributed[camera_id]
            )
            suppressed_duplicates += count
        self.stats["suppressed_duplicate_masks"] = int(
            self.stats.get("suppressed_duplicate_masks", 0)
        ) + suppressed_duplicates
        objects: list[dict[str, Any]] = []
        for camera_id in CAMERA_IDS:
            frame = frames[camera_id]
            depth = unpack_depth_z16(
                frame["depth_z16_zlib"],
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
            depth_m = np.asarray(depth, dtype=np.float32) * float(
                frame["depth_scale_m"]
            )
            depth_edge_mask = compute_depth_edge_mask(
                depth_m,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
                threshold_m=self.point_filter.depth_edge_threshold_m,
            )
            active_surface_tracks: set[str] = set()
            camera_objects: list[dict[str, Any]] = []
            for detected in attributed[camera_id]:
                obj = object_geometry(
                    camera_id=camera_id, serial=str(frame["serial"]), mask=detected["mask"], box_xyxy=detected["box"].tolist(),
                    label=detected["label"], score=detected["score"], depth_z16=depth, depth_scale_m=float(frame["depth_scale_m"]),
                    intrinsics=frame["intrinsics"], source_timestamp_us=int(frame["host_timestamp_us"]),
                    calibration_id=str(frame.get("calibration_id", "factory")), state=paired_state,
                    state_age_ms=state_age_by_camera[camera_id],
                    min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m,
                    min_depth_points=self.min_depth_points, point_filter=self.point_filter,
                    depth_edge_mask=depth_edge_mask,
                    t_goal_cam=t_goal_cameras[camera_id],
                    gravity_up_goal=up_goal,
                    max_points=self.contact_config.max_points,
                )
                obj["track_id"] = self._assign_track(camera_id, obj)
                if attach_surface_geometry(
                    obj, track_id=obj["track_id"], label=str(obj["label"]),
                    source_timestamp_us=int(frame["host_timestamp_us"]),
                    surface_tracker=self.surface_trackers[camera_id], contact_config=self.contact_config,
                    pseudo_config=self.pseudo_config,
                    up_goal=up_goal,
                    t_goal_cam=t_goal_cameras[camera_id],
                ):
                    active_surface_tracks.add(obj["track_id"])
                obj.update(
                    mask_encoding="bitpack-zlib-row-major",
                    mask_data=pack_mask(detected["mask"]),
                )
                camera_objects.append(obj)
            self.surface_trackers[camera_id].finish_frame(active_surface_tracks)
            completed = self.geometry_fallbacks[camera_id].complete_frame(
                camera_objects,
                self.surface_trackers[camera_id],
                image_width=int(frame["width"]),
                image_height=int(frame["height"]),
            )
            for obj in completed:
                if bool(obj.get("geometry_fallback", False)):
                    # Goal-frame contact can bridge a short detector dropout;
                    # older surface normals and image masks must not be reused.
                    obj["surface_primitives"] = []
                    obj["fallback_contact_only"] = True
                    obj.pop("mask_data", None)
                    obj.pop("mask_encoding", None)
            objects.extend(completed)
        self.stats["geometry_fallback_objects"] = int(
            self.stats.get("geometry_fallback_objects", 0)
        ) + sum(bool(obj.get("geometry_fallback", False)) for obj in objects)
        self.sequence += 1
        primitive_count = sum(bool(obj.get("surface_primitives")) for obj in objects)
        valid_depth_count = sum(obj.get("center_goal_m") is not None for obj in objects)
        result = make_result(
            sequence=self.sequence,
            source_bundle_timestamp_us=int(bundle["bundle_timestamp_us"]),
            source_bundle_sequence=int(bundle.get("sequence", -1)),
            objects=objects,
            discarded_seam_masks=discarded,
            geometry_status=self._geometry_status(
                bundle,
                state_available=True,
                state_age_ms=state_age_ms,
                primitive_count=primitive_count,
                objects_with_valid_depth=valid_depth_count,
                discarded_seam_masks=discarded,
                up_goal_available=up_goal is not None,
                suppressed_duplicate_masks=suppressed_duplicates,
            ),
            t_goal_cam={
                camera_id: t_goal_cameras[camera_id].reshape(-1).astype(float).tolist()
                for camera_id in CAMERA_IDS
            },
            transform_diagnostics=transform_diagnostics,
        )
        self.bus.publish(WRIST_RESULT_CHANNEL, pack_result(result))
        now_monotonic = time.monotonic()
        if now_monotonic - self.last_geometry_log_monotonic >= 5.0:
            centers = sum(obj.get("center_goal_m") is not None for obj in objects)
            primitives = sum(bool(obj.get("surface_primitives")) for obj in objects)
            reasons: dict[str, int] = {}
            for obj in objects:
                reason = str((obj.get("geometry") or {}).get("reason") or "valid")
                reasons[reason] = reasons.get(reason, 0) + 1
            LOG.info(
                "WRIST_GEOMETRY sequence=%d masks=%d center_goal=%d primitives=%d "
                "state_age_ms=%s state_rejects=%d reasons=%s",
                self.sequence, len(objects), centers, primitives, state_age_ms,
                self.stats["state_age_rejects"], reasons,
            )
            self.last_geometry_log_monotonic = now_monotonic
        return result

    def _geometry_status(
        self,
        bundle: dict[str, Any],
        *,
        state_available: bool,
        state_age_ms: float | None,
        primitive_count: int,
        objects_with_valid_depth: int,
        discarded_seam_masks: int,
        up_goal_available: bool = False,
        suppressed_duplicate_masks: int = 0,
    ) -> dict[str, Any]:
        return {
            "state_available": bool(state_available),
            "state_age_ms": state_age_ms,
            "state_age_rejects": int(self.stats.get("state_age_rejects", 0)),
            "up_goal_available": bool(up_goal_available),
            "primitive_count": int(primitive_count),
            "objects_with_valid_depth": int(objects_with_valid_depth),
            "discarded_seam_masks": int(discarded_seam_masks),
            "suppressed_duplicate_masks": int(suppressed_duplicate_masks),
            "inter_wrist_skew_us": int(bundle.get("inter_wrist_skew_us", 0) or 0),
            "robot_state_arrival_interval_us": bundle.get("robot_state_arrival_interval_us"),
            "robot_state_max_arrival_interval_us": bundle.get("robot_state_max_arrival_interval_us"),
            # Host/G1 clocks are not synchronized, so this is not transport
            # latency. It is the observable upper bound on how stale a held
            # packet can be when the 15 ms frame/state gate is evaluated.
            "robot_state_gap_ms": float(bundle.get("robot_state_gap_ms") or 0.0),
            "robot_state_gap_max_ms": float(
                bundle.get("robot_state_gap_max_ms") or 0.0
            ),
            "track_count": {
                camera_id: len(self.tracks[camera_id]) for camera_id in CAMERA_IDS
            },
        }

    def _assign_track(self, camera_id: str, obj: dict[str, Any]) -> str:
        """Associate only within one camera namespace; never cross wrist seam."""
        center = obj.get("center_goal_m") or obj.get("center_camera_m")
        if center is None:
            track_id = self.next_track_id[camera_id]
            self.next_track_id[camera_id] += 1
            return f"{camera_id}:{obj['label']}:track{track_id:04d}"
        point = np.asarray(center, dtype=np.float64).reshape(3)
        label = str(obj["label"])
        candidates = [
            (float(np.linalg.norm(point - old_center)), key)
            for key, (old_center, _last_sequence) in self.tracks[camera_id].items()
            if key.startswith(f"{label}:track")
        ]
        if candidates and min(candidates)[0] <= self.track_association_radius_m:
            local_id = min(candidates)[1]
        else:
            local_id = f"{label}:track{self.next_track_id[camera_id]:04d}"
            self.next_track_id[camera_id] += 1
        self.tracks[camera_id][local_id] = (point, self.sequence)
        for key, (_center, last_sequence) in list(self.tracks[camera_id].items()):
            if self.sequence - last_sequence > 5:
                del self.tracks[camera_id][key]
        return f"{camera_id}:{local_id}"

    def run(self) -> None:
        self.running = True
        self._publish_heartbeat()
        while self.running:
            self.bus.handle_timeout(20)
            # SAM3 is slower than capture. Drain queued UDP packets and retain
            # only the most recent atomic pair, never an inference backlog.
            while self.bus.handle_timeout(0) > 0:
                pass
            self.process_latest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--concepts", default="object")
    parser.add_argument("--backend", choices=("mock", "pytorch", "tensorrt"), default="pytorch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trt-engine", type=Path)
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=Path("config/wrist_d405_extrinsics_v11_translation_calibrated.yaml"),
    )
    parser.add_argument("--belief-config", type=Path, default=Path("config/pipeline.yaml"),
                        help="existing TASC filter/cylinder/RANSAC configuration")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-px", type=int, default=100)
    parser.add_argument(
        "--seam-guard-px",
        type=int,
        default=0,
        help="deprecated compatibility option; seam attribution uses area ratio",
    )
    parser.add_argument(
        "--seam-minority-ratio",
        type=float,
        default=0.05,
        help="discard only masks with more than this area in both wrist images",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.07)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-depth-points", type=int, default=100)
    parser.add_argument("--robot-state-max-delta-ms", type=float, default=15.0)
    parser.add_argument(
        "--track-association-radius-m",
        type=float,
        default=0.06,
        help="same-camera 3-D association radius for the D405 near workspace",
    )
    parser.add_argument("--geometry-fallback-ttl-frames", type=int, default=3)
    parser.add_argument("--geometry-fallback-score-decay", type=float, default=0.97)
    args = parser.parse_args()
    if args.backend == "mock":
        backend: Any = MockSam3Backend()
    elif args.backend == "pytorch":
        backend = TorchSam3Backend(checkpoint=args.checkpoint, threshold=args.threshold, min_area_px=args.min_area_px)
    else:
        backend = TensorRTSam3Backend(engine_path=args.trt_engine, model_id="facebook/sam3", threshold=args.threshold, min_area_px=args.min_area_px)
    point_filter, contact_config, pseudo_config = load_surface_configs(args.belief_config)
    WristD405Sam3Worker(wifi_lcm_url=args.wifi_lcm_url, concepts=parse_concepts(args.concepts), backend=backend,
                        extrinsics=load_wrist_extrinsics(args.extrinsics), seam_guard_px=args.seam_guard_px,
                        seam_minority_ratio=args.seam_minority_ratio,
                        min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m, min_depth_points=args.min_depth_points,
                        robot_state_max_delta_ms=args.robot_state_max_delta_ms, point_filter=point_filter,
                        contact_config=contact_config, pseudo_config=pseudo_config,
                        track_association_radius_m=args.track_association_radius_m,
                        geometry_fallback_ttl_frames=args.geometry_fallback_ttl_frames,
                        geometry_fallback_score_decay=args.geometry_fallback_score_decay).run()


if __name__ == "__main__":
    main()
