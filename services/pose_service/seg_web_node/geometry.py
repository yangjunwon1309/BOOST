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
