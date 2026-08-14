"""Backend-agnostic RGB-D geometry for remote segmentation workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from seg_web_node.geometry import CameraIntrinsics, transform_points
from seg_web_node.geometry_fallback import LastValidGeometryFallback
from seg_web_node.grip_candidates import (
    GripCandidateConfig,
    PersistentCylinderSurfaceTracker,
    PseudoContactConfig,
    extract_surface_primitives,
    generate_grip_candidates,
)
from seg_web_node.object_point_filter import (
    ObjectPointFilterConfig,
    extract_filtered_object_cloud,
)

from .protocol import make_object


GOAL_FRAME = "pelvis_identity_fk"


class Stable3DTracker:
    """Associate current detections to same-label tracks by 3-D distance."""

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


def camera_geometry_metadata(t_goal_cam: np.ndarray | None) -> dict[str, Any]:
    if t_goal_cam is None:
        return {}
    transform = np.asarray(t_goal_cam, dtype=np.float64).reshape(4, 4)
    return {
        "camera_goal_origin_m": transform[:3, 3].tolist(),
        "camera_goal_forward_m": (
            transform[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ).tolist(),
    }


class RemoteRgbdGeometryPipeline:
    """Create protocol-ready objects from detector masks and synchronized RGB-D."""

    def __init__(
        self,
        *,
        object_point_filter: ObjectPointFilterConfig,
        contact_config: GripCandidateConfig,
        pseudo_config: PseudoContactConfig | None = None,
        contact_candidates: bool = True,
        min_depth_m: float = 0.15,
        max_depth_m: float = 3.0,
        min_depth_points: int = 100,
        tracker: Stable3DTracker | None = None,
        fallback_ttl_frames: int = 0,
        fallback_score_decay: float = 0.97,
        fallback_reason: str = "segmentation_missing",
    ) -> None:
        self.object_point_filter = object_point_filter
        self.contact_config = contact_config
        self.pseudo_config = pseudo_config or PseudoContactConfig()
        self.contact_candidates_enabled = bool(contact_candidates)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_depth_points = max(1, int(min_depth_points))
        self.tracker = tracker or Stable3DTracker()
        self.tracker.missing_ttl_frames = max(
            int(self.tracker.missing_ttl_frames), int(fallback_ttl_frames)
        )
        self.surface_tracker = PersistentCylinderSurfaceTracker(contact_config)
        self.geometry_fallback = LastValidGeometryFallback(
            fallback_ttl_frames,
            fallback_score_decay,
            fallback_reason=fallback_reason,
        )

    def reset_tracking(self) -> None:
        """Drop geometry state when the semantic vocabulary changes."""
        self.tracker.tracks.clear()
        self.surface_tracker.models.clear()
        self.geometry_fallback.entries.clear()

    def build_object_geometry(
        self,
        *,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[str],
        depth_raw: np.ndarray,
        depth_scale_um: int,
        intrinsics: CameraIntrinsics | None,
        t_goal_cam: np.ndarray | None,
        up_goal: np.ndarray | None,
        source_robot_state_delta_ms: float | None,
        source_timestamp_us: int,
        image_width: int,
        image_height: int,
        allow_geometry_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        depth_m = np.asarray(depth_raw, dtype=np.float32) * (
            float(depth_scale_um) / 1_000_000.0
        )
        detections: list[dict[str, Any]] = []
        for index, (mask, label, score) in enumerate(zip(masks, labels, scores)):
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
                "surface_primitives": [],
                "surface_model": None,
                "point_filter": None,
                "_points_goal_m": None,
            }
            cloud = None
            if intrinsics is not None and t_goal_cam is not None:
                try:
                    cloud = extract_filtered_object_cloud(
                        depth_m,
                        np.asarray(mask, dtype=bool),
                        intrinsics,
                        self.object_point_filter,
                        min_depth_m=self.min_depth_m,
                        max_depth_m=self.max_depth_m,
                        min_valid_points=self.min_depth_points,
                        max_points=self.contact_config.max_points,
                    )
                except (TypeError, ValueError) as error:
                    detection["point_filter"] = {
                        "valid": False,
                        "reason": f"{type(error).__name__}: {error}",
                    }
            if cloud is not None:
                detection["median_depth_m"] = cloud.median_depth_m
                detection["center_camera_m"] = cloud.center_camera_m.tolist()
                detection["center_pixel_uv"] = [
                    int(cloud.center_pixel_uv[0]),
                    int(cloud.center_pixel_uv[1]),
                ]
                detection["valid_depth_points"] = cloud.valid_depth_points
                detection["point_filter"] = {"valid": True, **cloud.stats}
                detection["center_goal_m"] = transform_points(
                    cloud.center_camera_m, t_goal_cam
                ).tolist()
                detection["_points_goal_m"] = transform_points(
                    cloud.points_camera_m, t_goal_cam
                )
            detections.append(detection)

        track_ids = self.tracker.assign(detections)
        active_track_ids: set[str] = set()
        wire_objects: list[dict[str, Any]] = []
        for index, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
            detection = detections[index]
            track_id = track_ids.get(index, f"{label}:untracked{index:04d}")
            points_goal = detection.pop("_points_goal_m", None)
            if points_goal is not None:
                active_track_ids.add(track_id)
                cylinder_candidates = self.surface_tracker.update(
                    track_id, str(label), points_goal
                )
                if self.contact_candidates_enabled:
                    fitted_candidates = cylinder_candidates or generate_grip_candidates(
                        points_goal,
                        self.contact_config,
                        up_goal=up_goal,
                    )
                    candidates = [candidate.to_dict() for candidate in fitted_candidates]
                    detection["contact_candidates"] = candidates
                    if candidates:
                        detection["target_contact_goal_m"] = candidates[0][
                            "center_goal_m"
                        ]
                model = self.surface_tracker.models.get(track_id)
                detection["surface_model"] = self.surface_tracker.status(track_id)
                detection["surface_primitives"] = [
                    primitive.to_dict()
                    for primitive in extract_surface_primitives(
                        points_goal,
                        self.pseudo_config,
                        cylinder_model=model,
                        label=str(label),
                        up_goal=up_goal,
                    )
                ]
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
            wire_object.update(
                {
                    "surface_primitives": list(detection.get("surface_primitives", [])),
                    "point_filter": detection.get("point_filter"),
                    "geometry_fallback": False,
                    "missing_frames": 0,
                    "source_robot_state_delta_ms": (
                        None
                        if source_robot_state_delta_ms is None
                        else float(source_robot_state_delta_ms)
                    ),
                    "geometry_source_timestamp_us": int(source_timestamp_us),
                    "goal_frame": GOAL_FRAME,
                }
            )
            wire_objects.append(wire_object)

        self.surface_tracker.finish_frame(active_track_ids)
        completed = self.geometry_fallback.complete_frame(
            wire_objects,
            self.surface_tracker,
            image_width=image_width,
            image_height=image_height,
        )
        if allow_geometry_fallback:
            return completed
        # Advance bounded caches, but never turn a frame with missing
        # calibration/state/transform into a valid Thor belief input.
        return [obj for obj in completed if not bool(obj.get("geometry_fallback", False))]
