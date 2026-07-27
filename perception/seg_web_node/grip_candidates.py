"""Geometry-only contact candidate generation from segmented object point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-9


@dataclass(frozen=True)
class GripCandidate:
    center: np.ndarray
    approach: np.ndarray
    width: float
    geom_score: float
    source: str = "geometry"

    def to_dict(self) -> dict:
        return {
            "center_goal_m": np.asarray(self.center, dtype=np.float64).reshape(3).tolist(),
            "approach_goal": np.asarray(self.approach, dtype=np.float64).reshape(3).tolist(),
            "width_m": float(self.width),
            "geom_score": float(self.geom_score),
            "source": self.source,
        }


@dataclass(frozen=True)
class GripCandidateConfig:
    top_k: int = 4
    surface_augment: int = 8
    h_th_m: float = 0.12
    max_points: int = 8000
    table_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict | None) -> "GripCandidateConfig":
        data = dict(data or {})
        return cls(
            top_k=int(data.get("top_k", 4)),
            surface_augment=int(data.get("surface_augment", 8)),
            h_th_m=float(data.get("h_th_m", 0.12)),
            max_points=int(data.get("max_points", data.get("object_max_points", 8000))),
            table_normal=tuple(float(v) for v in data.get("table_normal", [0, 0, 1])),
        )


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= EPS:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return arr / norm


def _finite_points(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_points:
        step = int(np.ceil(pts.shape[0] / max_points))
        pts = pts[::step][:max_points]
    return pts


def _nearest_point(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points - np.asarray(target, dtype=np.float64).reshape(3), axis=1)
    return points[int(np.argmin(distances))].copy()


def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    demeaned = points - center
    cov = np.cov(demeaned.T)
    if not np.isfinite(cov).all():
        return center, np.eye(3), np.zeros(3)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order].T
    projections = demeaned @ axes.T
    extents = np.ptp(projections, axis=0)
    for i in range(3):
        axes[i] = _unit(axes[i], np.eye(3)[i])
    return center, axes, extents


def _farthest_points(points: np.ndarray, seed: np.ndarray, count: int) -> list[np.ndarray]:
    if count <= 0 or points.size == 0:
        return []
    selected = [np.asarray(seed, dtype=np.float64).reshape(3)]
    min_dist = np.linalg.norm(points - selected[0], axis=1)
    output: list[np.ndarray] = []
    for _ in range(min(count, points.shape[0])):
        index = int(np.argmax(min_dist))
        candidate = points[index].copy()
        output.append(candidate)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - candidate, axis=1))
    return output


def generate_grip_candidates(
    points_goal: np.ndarray,
    config: GripCandidateConfig | None = None,
) -> list[GripCandidate]:
    cfg = config or GripCandidateConfig()
    points = _finite_points(points_goal, cfg.max_points)
    if points.shape[0] == 0:
        return []

    table_normal = _unit(np.asarray(cfg.table_normal), np.array([0.0, 0.0, 1.0]))
    center, axes, extents = _principal_axes(points)
    height = float(np.ptp(points @ table_normal))
    top_down = height < cfg.h_th_m
    approach = -table_normal if top_down else -_unit(center * np.array([1.0, 1.0, 0.0]), table_normal)

    candidates: list[GripCandidate] = []
    horizontal_axes = []
    for axis in axes:
        horizontal = axis - table_normal * float(np.dot(axis, table_normal))
        if np.linalg.norm(horizontal) > 1.0e-4:
            horizontal_axes.append(_unit(horizontal, np.array([1.0, 0.0, 0.0])))
    if not horizontal_axes:
        horizontal_axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]

    spread = max(float(np.max(extents)) if extents.size else 0.0, 1.0e-3)
    for axis_index, axis in enumerate(horizontal_axes[:2]):
        projections = (points - center) @ axis
        extent = max(float(np.ptp(projections)), 1.0e-3)
        for sign in (-1.0, 1.0):
            target = center + sign * axis * 0.45 * extent
            surface_point = _nearest_point(points, target)
            side_score = 1.0 - min(1.0, abs(float(np.dot(surface_point - center, table_normal))) / (height + EPS))
            width_margin = 1.0 - min(1.0, abs(extent - 0.06) / 0.12)
            pinch = 1.0 - min(1.0, extent / (spread + EPS))
            geom_score = 0.4 * pinch + 0.3 * width_margin + 0.3 * side_score
            if axis_index == 0:
                geom_score += 0.05
            candidates.append(
                GripCandidate(
                    center=surface_point,
                    approach=approach,
                    width=extent,
                    geom_score=float(geom_score),
                    source="pca_surface",
                )
            )

    for point in _farthest_points(points, center, cfg.surface_augment):
        support = 1.0 - min(1.0, float(np.linalg.norm(point - center)) / (spread + EPS))
        candidates.append(
            GripCandidate(
                center=point,
                approach=-table_normal,
                width=0.0,
                geom_score=0.20 + 0.20 * support,
                source="surface_fps",
            )
        )

    unique: list[GripCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.geom_score, reverse=True):
        if all(float(np.linalg.norm(candidate.center - kept.center)) > 0.015 for kept in unique):
            unique.append(candidate)
        if len(unique) >= max(1, cfg.top_k):
            break
    if not unique:
        unique.append(
            GripCandidate(
                center=_nearest_point(points, center),
                approach=approach,
                width=0.0,
                geom_score=0.0,
                source="median_fallback",
            )
        )
    return unique
