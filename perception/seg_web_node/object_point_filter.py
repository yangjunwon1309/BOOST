"""Robust object point-cloud extraction from aligned RGB-D masks."""

from __future__ import annotations

import dataclasses
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import CameraIntrinsics


@dataclasses.dataclass
class ObjectPointFilterConfig:
    enabled: bool = True
    erosion_px: int = 2
    depth_edge_threshold_m: float = 0.030
    mad_scale: float = 3.5
    min_depth_band_m: float = 0.020
    voxel_size_m: float = 0.005
    radius_outlier_m: float = 0.018
    min_radius_neighbors: int = 3

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ObjectPointFilterConfig":
        data = dict(payload or {})
        config = cls()
        for field in dataclasses.fields(cls):
            if field.name in data:
                setattr(config, field.name, data[field.name])
        config.enabled = bool(config.enabled)
        config.erosion_px = max(0, int(config.erosion_px))
        config.depth_edge_threshold_m = max(float(config.depth_edge_threshold_m), 0.0)
        config.mad_scale = max(float(config.mad_scale), 0.0)
        config.min_depth_band_m = max(float(config.min_depth_band_m), 0.0)
        config.voxel_size_m = max(float(config.voxel_size_m), 0.0)
        config.radius_outlier_m = max(float(config.radius_outlier_m), 0.0)
        config.min_radius_neighbors = max(0, int(config.min_radius_neighbors))
        return config


@dataclasses.dataclass(frozen=True)
class FilteredObjectCloud:
    points_camera_m: np.ndarray
    center_camera_m: np.ndarray
    center_pixel_uv: tuple[int, int]
    median_depth_m: float
    valid_depth_points: int
    clean_mask: np.ndarray
    stats: dict[str, Any]


def _deproject_selected(
    rows: np.ndarray,
    cols: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    z = depth_m[rows, cols].astype(np.float64)
    return np.column_stack(
        (
            (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx,
            (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy,
            z,
        )
    )


def _voxel_centroids(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if voxel_size_m <= 0.0 or points.shape[0] < 2:
        return points
    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centroids = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)]
    )
    return centroids / counts[:, None]


def _radius_filter(
    points: np.ndarray,
    radius_m: float,
    min_neighbors: int,
) -> np.ndarray:
    if radius_m <= 0.0 or min_neighbors <= 0 or points.shape[0] <= min_neighbors:
        return points
    distances, _ = cKDTree(points).query(
        points,
        k=min(points.shape[0], min_neighbors + 1),
        distance_upper_bound=radius_m,
    )
    if distances.ndim == 1:
        return points
    keep = np.isfinite(distances[:, 1:]).sum(axis=1) >= min_neighbors
    return points[keep]


def extract_filtered_object_cloud(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: ObjectPointFilterConfig,
    *,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_points: int,
    max_points: int,
) -> FilteredObjectCloud | None:
    """Extract one robust cloud used by both center and surface estimation."""
    intrinsics.validate()
    depth = np.asarray(depth_m, dtype=np.float32)
    clean = np.asarray(mask, dtype=bool).copy()
    if depth.shape != clean.shape or depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth, mask, and intrinsics dimensions must match")

    input_pixels = int(clean.sum())
    if config.enabled and config.erosion_px > 0 and input_pixels:
        size = 2 * config.erosion_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        clean = cv2.erode(clean.astype(np.uint8), kernel, iterations=1).astype(bool)
    eroded_pixels = int(clean.sum())

    valid_depth = (
        np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    clean &= valid_depth

    if config.enabled and config.depth_edge_threshold_m > 0.0 and clean.any():
        padded = np.pad(depth, 1, mode="edge")
        neighbors = (
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
        )
        edge = np.zeros(depth.shape, dtype=bool)
        for neighbor in neighbors:
            neighbor_valid = (
                np.isfinite(neighbor)
                & (neighbor >= float(min_depth_m))
                & (neighbor <= float(max_depth_m))
            )
            edge |= ~neighbor_valid
            edge |= np.abs(depth - neighbor) > config.depth_edge_threshold_m
        clean &= ~edge
    edge_pixels = int(clean.sum())

    if config.enabled and config.mad_scale > 0.0 and clean.any():
        samples = depth[clean].astype(np.float64)
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        robust_sigma = 1.4826 * mad
        half_band = max(config.min_depth_band_m, config.mad_scale * robust_sigma)
        clean &= np.abs(depth - median) <= half_band
    mad_pixels = int(clean.sum())
    if mad_pixels < max(1, int(min_valid_points)):
        return None

    rows, cols = np.nonzero(clean)
    points = _deproject_selected(rows, cols, depth, intrinsics)
    points = _voxel_centroids(points, config.voxel_size_m if config.enabled else 0.0)
    voxel_points = int(points.shape[0])
    points = _radius_filter(
        points,
        config.radius_outlier_m if config.enabled else 0.0,
        config.min_radius_neighbors if config.enabled else 0,
    )
    radius_points = int(points.shape[0])
    if radius_points < max(1, int(min_valid_points)):
        return None
    if points.shape[0] > max(1, int(max_points)):
        indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[indices]

    center = np.median(points, axis=0).astype(np.float64)
    pixel = (int(np.median(cols)), int(np.median(rows)))
    return FilteredObjectCloud(
        points_camera_m=np.asarray(points, dtype=np.float64),
        center_camera_m=center,
        center_pixel_uv=pixel,
        median_depth_m=float(np.median(depth[clean])),
        valid_depth_points=int(points.shape[0]),
        clean_mask=clean,
        stats={
            "input_mask_pixels": input_pixels,
            "after_erosion_pixels": eroded_pixels,
            "after_depth_edge_pixels": edge_pixels,
            "after_mad_pixels": mad_pixels,
            "after_voxel_points": voxel_points,
            "after_radius_points": radius_points,
            "output_points": int(points.shape[0]),
        },
    )
