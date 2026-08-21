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
        rotation = matrix[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6):
            raise ValueError(f"T_{camera_id}_camera rotation is not orthonormal")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-6:
            raise ValueError(f"T_{camera_id}_camera is not a proper rotation")
        transforms[camera_id] = matrix
    return transforms


def t_goal_camera(camera_id: str, state: RobotState, wrist_extrinsics: dict[str, np.ndarray]) -> np.ndarray:
    if camera_id not in CAMERA_TO_WRIST_LINK:
        raise ValueError(f"unknown D405 camera_id {camera_id!r}")
    wrist = g1_link_transforms(np.asarray(state.q, dtype=np.float64)).get(CAMERA_TO_WRIST_LINK[camera_id])
    if wrist is None:
        raise ValueError(f"missing FK link for {camera_id}")
    return wrist @ wrist_extrinsics[camera_id]


def camera_transform_diagnostics(
    camera_id: str,
    state: RobotState,
    wrist_extrinsics: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Expose the exact D405-optical -> wrist -> pelvis transform chain.

    D405 points are always RealSense optical coordinates: +X right, +Y down,
    +Z forward.  Publishing the three resulting pelvis-frame basis vectors
    makes an axis/sign issue observable in a captured result instead of
    silently folding it into a belief or IPOPT target.
    """
    wrist_link = CAMERA_TO_WRIST_LINK[camera_id]
    goal_wrist = g1_link_transforms(np.asarray(state.q, dtype=np.float64))[wrist_link]
    wrist_camera = wrist_extrinsics[camera_id]
    goal_camera = goal_wrist @ wrist_camera
    return {
        "convention": (
            "p_goal=T_goal_wrist@T_wrist_camera@p_d405_optical; "
            "D405 optical axes: +x right, +y down, +z forward"
        ),
        "wrist_link": wrist_link,
        "t_goal_wrist": goal_wrist.reshape(-1).astype(float).tolist(),
        "t_wrist_camera": wrist_camera.reshape(-1).astype(float).tolist(),
        "t_goal_camera": goal_camera.reshape(-1).astype(float).tolist(),
        "camera_basis_goal": {
            "x_right": goal_camera[:3, 0].astype(float).tolist(),
            "y_down": goal_camera[:3, 1].astype(float).tolist(),
            "z_forward": goal_camera[:3, 2].astype(float).tolist(),
        },
    }


def object_geometry(*, camera_id: str, serial: str, mask: np.ndarray, box_xyxy: list[float],
                    label: str, score: float, depth_z16: np.ndarray, depth_scale_m: float,
                    intrinsics: dict[str, float | int], source_timestamp_us: int,
                    calibration_id: str, state: RobotState | None, state_age_ms: float | None,
                    wrist_extrinsics: dict[str, np.ndarray], min_depth_m: float,
                    max_depth_m: float, min_depth_points: int,
                    point_filter: ObjectPointFilterConfig | None = None,
                    depth_edge_mask: np.ndarray | None = None,
                    t_goal_cam: np.ndarray | None = None,
                    gravity_up_goal: np.ndarray | None = None,
                    max_points: int = 8000) -> dict[str, Any]:
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
    filter_config = point_filter or ObjectPointFilterConfig()
    diagnostics: dict[str, Any] = {}
    cloud = None
    try:
        cloud = extract_filtered_object_cloud(
            depth_m,
            np.asarray(mask, dtype=bool),
            pinhole,
            filter_config,
            min_depth_m=float(min_depth_m), max_depth_m=float(max_depth_m),
            min_valid_points=int(min_depth_points), max_points=int(max_points),
            diagnostics=diagnostics,
            depth_edge_mask=(
                compute_depth_edge_mask(
                    depth_m,
                    min_depth_m=float(min_depth_m),
                    max_depth_m=float(max_depth_m),
                    threshold_m=filter_config.depth_edge_threshold_m,
                )
                if depth_edge_mask is None
                else depth_edge_mask
            ),
        )
    except (TypeError, ValueError) as error:
        diagnostics = {"valid": False, "reason": f"{type(error).__name__}: {error}"}
    if cloud is None:
        # A None return without a populated diagnostic must never look valid in
        # the worker log, which uses ``reason`` as its success discriminator.
        diagnostics.setdefault("valid", False)
        diagnostics.setdefault("reason", "extractor_returned_none")
    result: dict[str, Any] = {
        "camera_id": camera_id, "serial": serial, "label": str(label), "score": float(score),
        "box_xyxy": [float(v) for v in box_xyxy], "source_timestamp_us": int(source_timestamp_us),
        "calibration_id": str(calibration_id), "state_age_ms": state_age_ms,
        "intrinsics": {key: float(value) if key not in {"width", "height"} else int(value) for key, value in intrinsics.items()},
        "mask_height": int(mask.shape[0]), "mask_width": int(mask.shape[1]),
        "valid_depth_points": 0, "geometry": dict(diagnostics),
        "center_camera_m": None, "center_goal_m": None, "t_goal_cam": None,
        "transform_diagnostics": None,
        "center_pixel_uv": None,
        "gravity_up_goal": None, "contact_candidates": [], "surface_primitives": [],
        "surface_model": None, "point_filter": dict(diagnostics),
        # The solver consumes pelvis-local targets.  Keep this explicit on the
        # wire; calling it a generic "goal" frame made it too easy to feed a
        # wrist-local point into the PATH/IPOPT path without noticing.
        "target_position_frame": "pelvis_identity_fk",
        "target_orientation_frame": "pelvis_identity_fk",
        "wrist_link": CAMERA_TO_WRIST_LINK.get(camera_id),
    }
    if cloud is None or state is None:
        if state is None:
            unavailable = {"valid": False, "reason": "robot_state_unavailable"}
            result["geometry"] = dict(unavailable)
            result["point_filter"] = dict(unavailable)
        return result
    transform = (
        np.asarray(t_goal_cam, dtype=np.float64).reshape(4, 4)
        if t_goal_cam is not None
        else t_goal_camera(camera_id, state, wrist_extrinsics)
    )
    transform_diagnostics = camera_transform_diagnostics(
        camera_id, state, wrist_extrinsics
    )
    points_goal = transform_points(cloud.points_camera_m, transform)
    result.update(
        valid_depth_points=int(cloud.valid_depth_points),
        center_camera_m=cloud.center_camera_m.tolist(),
        center_goal_m=transform_points(cloud.center_camera_m, transform).tolist(),
        center_pixel_uv=list(cloud.center_pixel_uv),
        t_goal_cam=transform.reshape(-1).astype(float).tolist(),
        transform_diagnostics=transform_diagnostics,
        geometry={"valid": True, **cloud.stats},
        point_filter={"valid": True, **cloud.stats},
        _points_goal_m=points_goal,
    )
    up_goal = (
        gravity_up_in_pelvis(state.base_quat_wxyz)
        if gravity_up_goal is None
        else gravity_up_goal
    )
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
    up_goal: np.ndarray | None = None,
) -> bool:
    """Attach legacy cylinder/RANSAC geometry and remove non-wire point data."""
    points_goal = result.pop("_points_goal_m", None)
    if points_goal is None:
        return False
    points = np.asarray(points_goal, dtype=np.float64).reshape(-1, 3)
    if points.size == 0 or not np.isfinite(points).all():
        return False
    fitted_candidates = surface_tracker.update(track_id, str(label), points)
    candidates = fitted_candidates or generate_grip_candidates(
        points,
        contact_config,
        up_goal=up_goal,
    )
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
