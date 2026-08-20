"""Native-grid D405 mask geometry for the wrist-only v11 worker.

Unlike the head-D435i path this module never rescales a mask or performs a
depth-to-RGB alignment.  UYVY direct-left-imager colour and Z16 depth must
arrive on the identical factory grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from seg_web_node.geometry import CameraIntrinsics, transform_points
from seg_web_node.grip_candidates import (
    GripCandidateConfig,
    PersistentCylinderSurfaceTracker,
    PseudoContactConfig,
    extract_surface_primitives,
    generate_grip_candidates,
)
from seg_web_node.lcm_protocol import RobotState
from seg_web_node.object_point_filter import (
    ObjectPointFilterConfig,
    compute_depth_edge_mask,
    extract_filtered_object_cloud,
)
from seg_web_node.robot_geometry import g1_link_transforms, gravity_up_in_pelvis


CAMERA_TO_WRIST_LINK = {
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}


def load_wrist_extrinsics(path: Path) -> dict[str, np.ndarray]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    transforms: dict[str, np.ndarray] = {}
    for camera_id in CAMERA_TO_WRIST_LINK:
        matrix = np.asarray(data.get(f"T_{camera_id}_camera"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"extrinsics missing finite T_{camera_id}_camera")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
            raise ValueError(f"T_{camera_id}_camera has invalid homogeneous row")
        transforms[camera_id] = matrix
    return transforms


def t_goal_camera(camera_id: str, state: RobotState, wrist_extrinsics: dict[str, np.ndarray]) -> np.ndarray:
    if camera_id not in CAMERA_TO_WRIST_LINK:
        raise ValueError(f"unknown D405 camera_id {camera_id!r}")
    wrist = g1_link_transforms(np.asarray(state.q, dtype=np.float64)).get(CAMERA_TO_WRIST_LINK[camera_id])
    if wrist is None:
        raise ValueError(f"missing FK link for {camera_id}")
    return wrist @ wrist_extrinsics[camera_id]


def deproject_mask(*, mask: np.ndarray, depth_z16: np.ndarray, depth_scale_m: float,
                   intrinsics: dict[str, float | int], min_depth_m: float, max_depth_m: float,
                   min_depth_points: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_z16, dtype=np.uint16)
    height, width = mask.shape
    if depth.shape != (height, width):
        raise ValueError("D405 mask/depth grids differ; refusing geometry")
    if (int(intrinsics["width"]), int(intrinsics["height"])) != (width, height):
        raise ValueError("D405 intrinsics/grid differ; refusing geometry")
    z = depth.astype(np.float32) * float(depth_scale_m)
    valid = mask & np.isfinite(z) & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    rows, cols = np.nonzero(valid)
    if len(rows) < int(min_depth_points):
        return None, {"valid": False, "reason": "too_few_depth_points", "count": int(len(rows))}
    values = z[rows, cols]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    ppx, ppy = float(intrinsics["ppx"]), float(intrinsics["ppy"])
    points = np.column_stack(((cols - ppx) * values / fx, (rows - ppy) * values / fy, values))
    # Reject the outer 10% of depth samples before the robust centre. This is a
    # bounded near-contact outlier gate, not a cross-camera fusion operation.
    lo, hi = np.quantile(values, [0.05, 0.95])
    points = points[(values >= lo) & (values <= hi)]
    if len(points) < int(min_depth_points):
        return None, {"valid": False, "reason": "too_few_filtered_points", "count": int(len(points))}
    return points.astype(np.float64), {"valid": True, "count": int(len(points)), "depth_lo_m": float(lo), "depth_hi_m": float(hi)}


def object_geometry(*, camera_id: str, serial: str, mask: np.ndarray, box_xyxy: list[float],
                    label: str, score: float, depth_z16: np.ndarray, depth_scale_m: float,
                    intrinsics: dict[str, float | int], source_timestamp_us: int,
                    calibration_id: str, state: RobotState | None, state_age_ms: float | None,
                    wrist_extrinsics: dict[str, np.ndarray], min_depth_m: float,
                    max_depth_m: float, min_depth_points: int,
                    point_filter: ObjectPointFilterConfig | None = None) -> dict[str, Any]:
    """Create geometry from the directly aligned D405 left-imager RGB-D grid.

    This deliberately uses the legacy robust object-cloud extractor: mask
    erosion, depth-edge rejection, MAD depth rejection, voxelization and
    radius-outlier filtering all operate on the native D405 grid.  There is no
    resize or RGB/depth alignment stage because this colour image is the left
    depth imager itself.
    """
    pinhole = CameraIntrinsics(
        width=int(intrinsics["width"]), height=int(intrinsics["height"]),
        fx=float(intrinsics["fx"]), fy=float(intrinsics["fy"]),
        cx=float(intrinsics["ppx"]), cy=float(intrinsics["ppy"]),
    )
    depth_m = np.asarray(depth_z16, dtype=np.float32) * float(depth_scale_m)
    diagnostics: dict[str, Any] = {}
    cloud = None
    try:
        cloud = extract_filtered_object_cloud(
            depth_m,
            np.asarray(mask, dtype=bool),
            pinhole,
            point_filter or ObjectPointFilterConfig(),
            min_depth_m=float(min_depth_m), max_depth_m=float(max_depth_m),
            min_valid_points=int(min_depth_points), max_points=8000,
            diagnostics=diagnostics,
            depth_edge_mask=compute_depth_edge_mask(
                depth_m, min_depth_m=float(min_depth_m), max_depth_m=float(max_depth_m),
                threshold_m=(point_filter or ObjectPointFilterConfig()).depth_edge_threshold_m,
            ),
        )
    except (TypeError, ValueError) as error:
        diagnostics = {"valid": False, "reason": f"{type(error).__name__}: {error}"}
    result: dict[str, Any] = {
        "camera_id": camera_id, "serial": serial, "label": str(label), "score": float(score),
        "box_xyxy": [float(v) for v in box_xyxy], "source_timestamp_us": int(source_timestamp_us),
        "calibration_id": str(calibration_id), "state_age_ms": state_age_ms,
        "intrinsics": {key: float(value) if key not in {"width", "height"} else int(value) for key, value in intrinsics.items()},
        "mask_height": int(mask.shape[0]), "mask_width": int(mask.shape[1]),
        "valid_depth_points": 0, "geometry": diagnostics,
        "center_camera_m": None, "center_goal_m": None, "t_goal_cam": None,
        "gravity_up_goal": None, "contact_candidates": [], "surface_primitives": [],
        "surface_model": None, "point_filter": diagnostics,
    }
    if cloud is None or state is None:
        if state is None:
            result["geometry"] = {"valid": False, "reason": "robot_state_unavailable"}
        return result
    transform = t_goal_camera(camera_id, state, wrist_extrinsics)
    points_goal = transform_points(cloud.points_camera_m, transform)
    result.update(
        valid_depth_points=int(cloud.valid_depth_points),
        center_camera_m=cloud.center_camera_m.tolist(), center_goal_m=np.median(points_goal, axis=0).tolist(),
        t_goal_cam=transform.reshape(-1).astype(float).tolist(),
        point_filter={"valid": True, **cloud.stats},
        _points_goal_m=points_goal,
    )
    up_goal = gravity_up_in_pelvis(state.base_quat_wxyz)
    if up_goal is not None:
        result["gravity_up_goal"] = np.asarray(up_goal, dtype=float).reshape(3).tolist()
    return result


def attach_surface_geometry(
    result: dict[str, Any],
    *,
    track_id: str,
    label: str,
    source_timestamp_us: int,
    surface_tracker: PersistentCylinderSurfaceTracker,
    contact_config: GripCandidateConfig,
    pseudo_config: PseudoContactConfig,
) -> bool:
    """Attach legacy cylinder/RANSAC geometry and remove non-wire point data."""
    points_goal = result.pop("_points_goal_m", None)
    if points_goal is None:
        return False
    points = np.asarray(points_goal, dtype=np.float64).reshape(-1, 3)
    if points.size == 0 or not np.isfinite(points).all():
        return False
    fitted_candidates = surface_tracker.update(track_id, str(label), points)
    candidates = fitted_candidates or generate_grip_candidates(points, contact_config)
    result["contact_candidates"] = [candidate.to_dict() for candidate in candidates]
    if candidates:
        result["target_contact_goal_m"] = candidates[0].center.tolist()
    model = surface_tracker.models.get(track_id)
    result["surface_model"] = surface_tracker.status(track_id)
    view_origin = None
    try:
        view_origin = np.asarray(result["t_goal_cam"], dtype=np.float64).reshape(4, 4)[:3, 3]
    except (KeyError, TypeError, ValueError):
        pass
    primitives = extract_surface_primitives(
        points, pseudo_config, cylinder_model=model, view_origin=view_origin, label=str(label),
        up_goal=np.asarray(result["gravity_up_goal"], dtype=np.float64)
        if result.get("gravity_up_goal") is not None else None,
    )
    result["surface_primitives"] = [
        {
            **primitive.to_dict(),
            "observation_timestamp_us": int(source_timestamp_us),
            "observation_track_id": str(track_id),
            "observation_state": "active",
        }
        for primitive in primitives
    ]
    return True
