"""Vectorized pinhole deprojection and frame transforms."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("invalid camera intrinsics")


def deproject(
    depth_raw: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale_um: int,
    min_depth_m: float = 0.15,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    intrinsics.validate()
    if depth_raw.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth shape does not match color intrinsics")
    if depth_scale_um <= 0:
        raise ValueError("invalid depth scale")
    z = depth_raw.astype(np.float32) * (depth_scale_um / 1_000_000.0)
    valid = (depth_raw != 0) & (z >= min_depth_m) & (z <= max_depth_m)
    u = np.arange(intrinsics.width, dtype=np.float32)[None, :]
    v = np.arange(intrinsics.height, dtype=np.float32)[:, None]
    xyz = np.empty((*z.shape, 3), dtype=np.float32)
    xyz[..., 0] = (u - intrinsics.cx) * z / intrinsics.fx
    xyz[..., 1] = (v - intrinsics.cy) * z / intrinsics.fy
    xyz[..., 2] = z
    xyz[~valid] = np.nan
    return xyz


def transform_points(points_camera: np.ndarray, t_torso_cam: np.ndarray) -> np.ndarray:
    """Apply p_torso = T_torso_cam @ p_camera to (...,3) points."""
    transform = np.asarray(t_torso_cam, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_torso_cam must be 4x4")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.asarray(points_camera) @ rotation.T + translation


def robust_mask_center(
    xyz_camera: np.ndarray,
    mask: np.ndarray,
    *,
    min_valid_points: int = 100,
) -> tuple[np.ndarray, tuple[int, int], int] | None:
    """Return a robust 3-D mask center, representative pixel, and sample count.

    The component-wise median is deliberately used instead of the bounding-box
    center: it is much less sensitive to missing depth and mask-edge outliers.
    """
    if xyz_camera.ndim != 3 or xyz_camera.shape[2] != 3:
        raise ValueError("xyz_camera must be HxWx3")
    if mask.shape != xyz_camera.shape[:2]:
        raise ValueError("mask shape does not match xyz_camera")
    valid = mask.astype(bool) & np.isfinite(xyz_camera).all(axis=2)
    rows, cols = np.nonzero(valid)
    count = int(rows.size)
    if count < max(1, int(min_valid_points)):
        return None
    center = np.median(xyz_camera[valid], axis=0).astype(np.float64)
    pixel = (int(np.median(cols)), int(np.median(rows)))
    return center, pixel, count


def central_mask_geometry_filter(
    masks: np.ndarray,
    xyz_camera: np.ndarray,
    *,
    max_area_ratio: float,
    max_center_distance_m: float,
    plane_min_points: int,
    plane_max_rms_m: float,
    support_normal_up_dot_min: float,
    t_goal_cam: np.ndarray | None,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Return central-mode mask indices surviving geometry-only heuristics.

    The area and range gates reject broad distant scene surfaces. For a
    depth-valid, low-residual planar mask, its PCA plane normal is transformed
    into the pelvis/goal frame. A normal near +Z/-Z identifies a horizontal
    support surface (floor or desk), unlike a vertical bottle or box side.
    """
    if masks.ndim != 3 or masks.shape[1:] != xyz_camera.shape[:2]:
        raise ValueError("mask and xyz shapes do not match")
    height, width = masks.shape[1:]
    total_pixels = height * width
    keep: list[int] = []
    rejected: list[dict[str, float | int | str]] = []
    rotation = None
    if t_goal_cam is not None:
        transform = np.asarray(t_goal_cam, dtype=np.float64)
        if transform.shape == (4, 4):
            rotation = transform[:3, :3]

    for index, raw_mask in enumerate(masks):
        mask = raw_mask.astype(bool)
        area_ratio = float(mask.sum()) / max(1, total_pixels)
        if area_ratio > max_area_ratio:
            rejected.append({"index": index, "reason": "area", "area_ratio": area_ratio})
            continue
        points = xyz_camera[mask]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            # No depth is not evidence that this is a floor/desk. Preserve the
            # RGB mask; downstream geometry already marks its center invalid.
            keep.append(index)
            continue
        center = np.median(points, axis=0)
        distance_m = float(np.linalg.norm(center))
        if distance_m > max_center_distance_m:
            rejected.append(
                {"index": index, "reason": "distance", "distance_m": distance_m}
            )
            continue
        if len(points) < plane_min_points or rotation is None:
            keep.append(index)
            continue
        # Fixed bounded sample makes PCA cost predictable even for a large mask.
        if len(points) > 2000:
            points = points[np.linspace(0, len(points) - 1, 2000, dtype=np.int64)]
        mean = np.mean(points, axis=0)
        centered = points - mean
        covariance = centered.T @ centered / max(1, len(centered))
        values, vectors = np.linalg.eigh(covariance)
        normal_camera = vectors[:, int(np.argmin(values))]
        residual_rms_m = float(np.sqrt(max(0.0, float(np.min(values)))))
        normal_goal = rotation @ normal_camera
        normal_goal /= max(1.0e-9, float(np.linalg.norm(normal_goal)))
        up_dot = float(abs(np.dot(normal_goal, np.array([0.0, 0.0, 1.0]))))
        if residual_rms_m <= plane_max_rms_m and up_dot >= support_normal_up_dot_min:
            rejected.append(
                {
                    "index": index,
                    "reason": "horizontal_flat_plane",
                    "distance_m": distance_m,
                    "plane_rms_m": residual_rms_m,
                    "normal_up_dot": up_dot,
                }
            )
            continue
        keep.append(index)
    return np.asarray(keep, dtype=np.int64), rejected
