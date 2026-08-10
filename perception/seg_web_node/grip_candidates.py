"""Geometry-only contact candidate generation from segmented object point clouds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


EPS = 1.0e-9


@dataclass(frozen=True)
class CylinderSurfaceConfig:
    enabled: bool = True
    labels: tuple[str, ...] = ("bottle", "can", "cup")
    min_points: int = 120
    min_radius_m: float = 0.015
    max_radius_m: float = 0.12
    min_height_m: float = 0.04
    max_height_m: float = 0.50
    inlier_threshold_m: float = 0.015
    min_inlier_ratio: float = 0.45
    min_angular_coverage: float = 0.20
    max_fit_rmse_m: float = 0.015
    max_center_shift_m: float = 0.15
    update_alpha: float = 0.35
    stale_ttl_frames: int = 15
    sample_spacing_m: float = 0.025
    max_candidates: int = 24
    max_nfev: int = 80
    prior_weight: float = 2.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "CylinderSurfaceConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            labels=tuple(str(v).strip().lower() for v in data.get("labels", cls.labels)),
            min_points=int(data.get("min_points", 120)),
            min_radius_m=float(data.get("min_radius_m", 0.015)),
            max_radius_m=float(data.get("max_radius_m", 0.12)),
            min_height_m=float(data.get("min_height_m", 0.04)),
            max_height_m=float(data.get("max_height_m", 0.50)),
            inlier_threshold_m=float(data.get("inlier_threshold_m", 0.015)),
            min_inlier_ratio=float(data.get("min_inlier_ratio", 0.45)),
            min_angular_coverage=float(data.get("min_angular_coverage", 0.20)),
            max_fit_rmse_m=float(data.get("max_fit_rmse_m", 0.015)),
            max_center_shift_m=float(data.get("max_center_shift_m", 0.15)),
            update_alpha=float(data.get("update_alpha", 0.35)),
            stale_ttl_frames=int(data.get("stale_ttl_frames", 15)),
            sample_spacing_m=float(data.get("sample_spacing_m", 0.025)),
            max_candidates=int(data.get("max_candidates", 24)),
            max_nfev=int(data.get("max_nfev", 80)),
            prior_weight=float(data.get("prior_weight", 2.0)),
        )


@dataclass
class CylinderSurfaceModel:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    height: float
    fit_rmse: float
    inlier_ratio: float
    angular_coverage: float
    updates: int = 1
    stale_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "type": "cylinder",
            "center_goal_m": np.asarray(self.center, dtype=np.float64).tolist(),
            "axis_goal": np.asarray(self.axis, dtype=np.float64).tolist(),
            "radius_m": float(self.radius),
            "height_m": float(self.height),
            "fit_rmse_m": float(self.fit_rmse),
            "inlier_ratio": float(self.inlier_ratio),
            "angular_coverage": float(self.angular_coverage),
            "updates": int(self.updates),
            "stale_frames": int(self.stale_frames),
        }


@dataclass(frozen=True)
class GripCandidate:
    center: np.ndarray
    approach: np.ndarray
    width: float
    geom_score: float
    source: str = "geometry"
    surface_normal: np.ndarray | None = None
    surface_tangent: np.ndarray | None = None
    surface_axis: np.ndarray | None = None

    def to_dict(self) -> dict:
        payload = {
            "center_goal_m": np.asarray(self.center, dtype=np.float64).reshape(3).tolist(),
            "approach_goal": np.asarray(self.approach, dtype=np.float64).reshape(3).tolist(),
            "width_m": float(self.width),
            "geom_score": float(self.geom_score),
            "source": self.source,
        }
        for key, vector in (
            ("surface_normal", self.surface_normal),
            ("surface_tangent", self.surface_tangent),
            ("surface_axis", self.surface_axis),
        ):
            payload[key] = (
                None
                if vector is None
                else np.asarray(vector, dtype=np.float64).reshape(3).tolist()
            )
        return payload


@dataclass(frozen=True)
class PseudoContactPoint:
    """Surface sample used as an orientation reference, not a physical contact."""

    center: np.ndarray
    normal: np.ndarray
    fit_quality: float
    source: str

    def to_dict(self) -> dict:
        return {
            "center_goal_m": np.asarray(self.center, dtype=np.float64).reshape(3).tolist(),
            "normal_goal": np.asarray(self.normal, dtype=np.float64).reshape(3).tolist(),
            "fit_quality": float(self.fit_quality),
            "source": str(self.source),
        }


@dataclass(frozen=True)
class PseudoContactConfig:
    enabled: bool = True
    max_points: int = 6
    plane_ransac_iters: int = 60
    plane_inlier_thresh_m: float = 0.006
    min_inlier_ratio: float = 0.35
    normal_sign_reference: str = "camera"

    @classmethod
    def from_dict(cls, data: dict | None) -> "PseudoContactConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_points=int(data.get("max_points", 6)),
            plane_ransac_iters=int(data.get("plane_ransac_iters", 60)),
            plane_inlier_thresh_m=float(data.get("plane_inlier_thresh_m", 0.006)),
            min_inlier_ratio=float(data.get("min_inlier_ratio", 0.35)),
            normal_sign_reference=str(data.get("normal_sign_reference", "camera")),
        )


@dataclass(frozen=True)
class GripCandidateConfig:
    top_k: int = 4
    surface_augment: int = 8
    h_th_m: float = 0.12
    max_points: int = 8000
    table_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    merge_distance_m: float = 0.03
    surface: CylinderSurfaceConfig = field(default_factory=CylinderSurfaceConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "GripCandidateConfig":
        data = dict(data or {})
        return cls(
            top_k=int(data.get("top_k", 4)),
            surface_augment=int(data.get("surface_augment", 8)),
            h_th_m=float(data.get("h_th_m", 0.12)),
            max_points=int(data.get("max_points", data.get("object_max_points", 8000))),
            table_normal=tuple(float(v) for v in data.get("table_normal", [0, 0, 1])),
            merge_distance_m=float(data.get("merge_distance_m", 0.03)),
            surface=CylinderSurfaceConfig.from_dict(data.get("surface_model")),
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


def _axis_from_angles(azimuth: float, elevation: float) -> np.ndarray:
    cos_elevation = np.cos(elevation)
    return np.asarray(
        [
            cos_elevation * np.cos(azimuth),
            cos_elevation * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )


def _axis_angles(axis: np.ndarray) -> tuple[float, float]:
    unit_axis = _unit(axis, np.array([0.0, 0.0, 1.0]))
    return (
        float(np.arctan2(unit_axis[1], unit_axis[0])),
        float(np.arctan2(unit_axis[2], np.linalg.norm(unit_axis[:2]))),
    )


def _radial_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit_axis = _unit(axis, np.array([0.0, 0.0, 1.0]))
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(unit_axis, reference))) > 0.90:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    first = _unit(np.cross(unit_axis, reference), np.array([0.0, 1.0, 0.0]))
    second = _unit(np.cross(unit_axis, first), np.array([1.0, 0.0, 0.0]))
    return first, second


def _cylinder_geometry(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = points - np.asarray(center, dtype=np.float64).reshape(3)
    axial = delta @ np.asarray(axis, dtype=np.float64).reshape(3)
    radial_vectors = delta - axial[:, None] * axis[None, :]
    radial_distance = np.linalg.norm(radial_vectors, axis=1)
    return axial, radial_vectors, radial_distance


def fit_cylinder_surface(
    points_goal: np.ndarray,
    config: CylinderSurfaceConfig,
    previous: CylinderSurfaceModel | None = None,
) -> CylinderSurfaceModel | None:
    """Fit a finite cylinder with robust nonlinear least squares.

    The residual is the radial point-to-cylinder distance used by PCL's
    cylinder model.  The previous fit is a weak temporal prior and the fitted
    finite extent is recovered robustly from axial inlier quantiles.
    """
    points = _finite_points(points_goal, max(4 * config.min_points, 2000))
    if points.shape[0] < config.min_points:
        return None

    median = np.median(points, axis=0)
    demeaned = points - median
    covariance = np.cov(demeaned.T)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return None
    values, vectors = np.linalg.eigh(covariance)
    pca_axis = _unit(vectors[:, int(np.argmax(values))], np.array([0.0, 0.0, 1.0]))
    reference_axis = previous.axis if previous is not None else np.array([0.0, 0.0, 1.0])
    if float(np.dot(pca_axis, reference_axis)) < 0.0:
        pca_axis = -pca_axis
    initial_axis = previous.axis.copy() if previous is not None else pca_axis
    initial_center = previous.center.copy() if previous is not None else median

    basis_u, basis_v = _radial_basis(initial_axis)
    projections_u = demeaned @ basis_u
    projections_v = demeaned @ basis_v
    extent_radius = 0.25 * (
        np.ptp(np.quantile(projections_u, [0.02, 0.98]))
        + np.ptp(np.quantile(projections_v, [0.02, 0.98]))
    )
    initial_radius = previous.radius if previous is not None else extent_radius
    initial_radius = float(
        np.clip(initial_radius, config.min_radius_m, config.max_radius_m)
    )
    initial_azimuth, initial_elevation = _axis_angles(initial_axis)
    x0 = np.asarray(
        [*initial_center, initial_azimuth, initial_elevation, initial_radius],
        dtype=np.float64,
    )
    center_bound = max(config.max_center_shift_m, 1.0e-3)
    lower = np.asarray(
        [
            *(median - center_bound),
            -np.pi,
            -0.5 * np.pi + 1.0e-4,
            config.min_radius_m,
        ]
    )
    upper = np.asarray(
        [
            *(median + center_bound),
            np.pi,
            0.5 * np.pi - 1.0e-4,
            config.max_radius_m,
        ]
    )
    x0 = np.clip(x0, lower + 1.0e-8, upper - 1.0e-8)
    prior_scale = np.sqrt(max(0.0, config.prior_weight))

    def residual(parameters: np.ndarray) -> np.ndarray:
        center = parameters[:3]
        axis = _axis_from_angles(parameters[3], parameters[4])
        _axial, _radial_vectors, radial_distance = _cylinder_geometry(
            points, center, axis
        )
        values_out = [radial_distance - parameters[5]]
        if previous is not None and prior_scale > 0.0:
            values_out.extend(
                [
                    prior_scale * (center - previous.center),
                    0.05 * prior_scale * (axis - previous.axis),
                    prior_scale * np.asarray([parameters[5] - previous.radius]),
                ]
            )
        return np.concatenate(values_out)

    try:
        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=max(1.0e-4, config.inlier_threshold_m),
            max_nfev=max(10, config.max_nfev),
        )
    except (ValueError, FloatingPointError):
        return None
    if not result.success or not np.isfinite(result.x).all():
        return None

    axis_point = result.x[:3]
    axis = _axis_from_angles(result.x[3], result.x[4])
    if float(np.dot(axis, reference_axis)) < 0.0:
        axis = -axis
    radius = float(result.x[5])
    axial, radial_vectors, radial_distance = _cylinder_geometry(points, axis_point, axis)
    radial_residual = radial_distance - radius
    inlier = np.abs(radial_residual) <= config.inlier_threshold_m
    if int(inlier.sum()) < config.min_points:
        return None
    inlier_ratio = float(np.mean(inlier))
    fit_rmse = float(np.sqrt(np.mean(np.square(radial_residual[inlier]))))

    axial_low, axial_high = np.quantile(axial[inlier], [0.02, 0.98])
    height = float(axial_high - axial_low)
    center = axis_point + 0.5 * float(axial_low + axial_high) * axis
    basis_u, basis_v = _radial_basis(axis)
    inlier_radial = radial_vectors[inlier]
    angles = np.mod(
        np.arctan2(inlier_radial @ basis_v, inlier_radial @ basis_u),
        2.0 * np.pi,
    )
    occupied, _ = np.histogram(angles, bins=24, range=(0.0, 2.0 * np.pi))
    angular_coverage = float(np.mean(occupied > 0))

    if not (
        config.min_radius_m <= radius <= config.max_radius_m
        and config.min_height_m <= height <= config.max_height_m
        and fit_rmse <= config.max_fit_rmse_m
        and inlier_ratio >= config.min_inlier_ratio
        and angular_coverage >= config.min_angular_coverage
    ):
        return None
    return CylinderSurfaceModel(
        center=center,
        axis=axis,
        radius=radius,
        height=height,
        fit_rmse=fit_rmse,
        inlier_ratio=inlier_ratio,
        angular_coverage=angular_coverage,
    )


def sample_cylinder_candidates(
    model: CylinderSurfaceModel,
    config: GripCandidateConfig,
) -> list[GripCandidate]:
    surface = config.surface
    circumference_samples = max(
        4,
        int(np.ceil(2.0 * np.pi * model.radius / max(0.005, surface.sample_spacing_m))),
    )
    axial_count = max(
        1,
        int(
            np.ceil(
                0.5 * model.height / max(0.005, surface.sample_spacing_m)
            )
        ),
    )
    axial_positions = (
        np.asarray([0.0])
        if axial_count == 1
        else np.linspace(-0.25 * model.height, 0.25 * model.height, axial_count)
    )
    basis_u, basis_v = _radial_basis(model.axis)
    confidence = float(
        np.clip(
            model.inlier_ratio
            * np.exp(-model.fit_rmse / max(EPS, surface.max_fit_rmse_m)),
            0.0,
            1.0,
        )
    )
    candidates: list[GripCandidate] = []
    for axial in axial_positions:
        for angle_index in range(circumference_samples):
            angle = 2.0 * np.pi * angle_index / circumference_samples
            normal = np.cos(angle) * basis_u + np.sin(angle) * basis_v
            tangent = -np.sin(angle) * basis_u + np.cos(angle) * basis_v
            normal = _unit(normal, basis_u)
            tangent = _unit(tangent, basis_v)
            axis = _unit(model.axis, np.array([0.0, 0.0, 1.0]))
            # These are analytical cylinder-frame vectors. Keep the invariant
            # explicit because orientation belief consumes all three.
            assert abs(float(np.dot(normal, tangent))) < 1.0e-6
            assert abs(float(np.dot(normal, axis))) < 1.0e-6
            assert abs(float(np.dot(tangent, axis))) < 1.0e-6
            assert abs(float(np.linalg.norm(normal)) - 1.0) < 1.0e-6
            assert abs(float(np.linalg.norm(tangent)) - 1.0) < 1.0e-6
            assert abs(float(np.linalg.norm(axis)) - 1.0) < 1.0e-6
            center = model.center + float(axial) * model.axis + model.radius * normal
            candidates.append(
                GripCandidate(
                    center=center,
                    approach=-normal,
                    width=2.0 * model.radius,
                    geom_score=confidence * (1.0 - 0.25 * abs(float(axial)) / max(EPS, model.height)),
                    source="cylinder_surface",
                    surface_normal=normal,
                    surface_tangent=tangent,
                    surface_axis=axis,
                )
            )
    max_candidates = max(int(config.top_k), int(surface.max_candidates), 1)
    if len(candidates) <= max_candidates:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, max_candidates, dtype=np.int64)
    return [candidates[int(index)] for index in indices]


def _orient_normal(
    normal: np.ndarray,
    point: np.ndarray,
    view_origin: np.ndarray | None,
) -> np.ndarray:
    oriented = _unit(normal, np.array([0.0, 0.0, 1.0]))
    if view_origin is not None:
        to_view = _unit(np.asarray(view_origin, dtype=np.float64).reshape(3) - point, oriented)
        if float(np.dot(oriented, to_view)) < 0.0:
            oriented = -oriented
    elif oriented[2] < 0.0:
        oriented = -oriented
    return oriented


def _fit_plane_ransac(
    points: np.ndarray,
    config: PseudoContactConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Return refitted plane point, normal, inlier points, and inlier ratio."""
    if points.shape[0] < 3:
        return None
    rng = np.random.default_rng(0)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(max(1, int(config.plane_ransac_iters))):
        indices = rng.choice(points.shape[0], size=3, replace=False)
        a, b, c = points[indices]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            continue
        normal /= norm
        mask = np.abs((points - a) @ normal) <= max(EPS, config.plane_inlier_thresh_m)
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
    if best_mask is None:
        return None
    ratio = float(best_count / points.shape[0])
    if ratio < float(config.min_inlier_ratio):
        return None
    inliers = points[best_mask]
    point = np.mean(inliers, axis=0)
    covariance = np.cov((inliers - point).T)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return None
    values, vectors = np.linalg.eigh(covariance)
    normal = _unit(vectors[:, int(np.argmin(values))], np.array([0.0, 0.0, 1.0]))
    return point, normal, inliers, ratio


def fit_circle_in_plane(
    points: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Fit a 3-D circle after projection into a plane-local 2-D basis."""
    pts = _finite_points(points, 8000)
    if pts.shape[0] < 3:
        return None
    normal = _unit(plane_normal, np.array([0.0, 0.0, 1.0]))
    basis_u, basis_v = _radial_basis(normal)
    delta = pts - np.asarray(plane_point, dtype=np.float64).reshape(3)
    x, y = delta @ basis_u, delta @ basis_v
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    rhs = x * x + y * y
    try:
        cx, cy, constant = np.linalg.lstsq(design, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = float(constant + cx * cx + cy * cy)
    if not np.isfinite(radius_sq) or radius_sq <= EPS:
        return None
    radius = float(np.sqrt(radius_sq))
    residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    rmse = float(np.sqrt(np.mean(residual * residual)))
    center = np.asarray(plane_point, dtype=np.float64).reshape(3) + cx * basis_u + cy * basis_v
    if not np.isfinite(center).all() or not np.isfinite(rmse):
        return None
    return center, radius, rmse


def extract_pseudo_contacts(
    points_goal: np.ndarray,
    config: PseudoContactConfig,
    *,
    cylinder_model: CylinderSurfaceModel | None = None,
    view_origin: np.ndarray | None = None,
    label: str = "",
) -> list[PseudoContactPoint]:
    """Extract orientation-reference surface samples from one object cloud."""
    if not config.enabled:
        return []
    points = _finite_points(points_goal, 8000)
    if points.shape[0] == 0:
        return []
    count = max(1, int(config.max_points))
    if points.shape[0] < 3:
        center = np.mean(points, axis=0)
        return [PseudoContactPoint(
            center=center,
            normal=_orient_normal(np.array([0.0, 0.0, 1.0]), center, view_origin),
            fit_quality=0.0,
            source="center_fallback",
        )]
    if cylinder_model is not None and cylinder_model.inlier_ratio >= 0.35:
        quality = float(np.clip(
            cylinder_model.inlier_ratio * np.exp(-cylinder_model.fit_rmse / 0.015),
            0.0,
            1.0,
        ))
        basis_u, basis_v = _radial_basis(cylinder_model.axis)
        contacts: list[PseudoContactPoint] = []
        for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
            normal = np.cos(angle) * basis_u + np.sin(angle) * basis_v
            center = cylinder_model.center + cylinder_model.radius * normal
            contacts.append(PseudoContactPoint(
                center=center,
                normal=_orient_normal(normal, center, view_origin),
                fit_quality=quality,
                source="cylinder",
            ))
        return contacts

    plane = _fit_plane_ransac(points, config)
    if plane is not None:
        plane_point, normal, inliers, ratio = plane
        normal = _orient_normal(normal, plane_point, view_origin)
        lower_label = str(label).lower()
        if "hole" in lower_label or "circle" in lower_label:
            circle = fit_circle_in_plane(inliers, plane_point, normal)
            if circle is not None:
                center, radius, rmse = circle
                if 0.002 <= radius <= 0.100 and rmse <= max(EPS, config.plane_inlier_thresh_m):
                    return [PseudoContactPoint(
                        center=center,
                        normal=normal,
                        fit_quality=float(np.clip(ratio * np.exp(-rmse / config.plane_inlier_thresh_m), 0.0, 1.0)),
                        source="hole_circle",
                    )]
        indices = np.linspace(0, inliers.shape[0] - 1, min(count, inliers.shape[0]), dtype=np.int64)
        return [PseudoContactPoint(
            center=inliers[int(index)].copy(), normal=normal, fit_quality=ratio, source="plane"
        ) for index in indices]

    center, axes, _extents = _principal_axes(points)
    normal = _orient_normal(axes[-1], center, view_origin)
    indices = np.linspace(0, points.shape[0] - 1, min(count, points.shape[0]), dtype=np.int64)
    return [PseudoContactPoint(
        center=points[int(index)].copy(), normal=normal, fit_quality=0.3, source="pca"
    ) for index in indices]


class PersistentCylinderSurfaceTracker:
    """Maintain one temporally filtered finite-cylinder model per object ID."""

    def __init__(self, config: GripCandidateConfig | None = None) -> None:
        self.config = config or GripCandidateConfig()
        self.models: dict[str, CylinderSurfaceModel] = {}

    def _enabled_for_label(self, label: str) -> bool:
        surface = self.config.surface
        labels = set(surface.labels)
        return surface.enabled and (not labels or str(label).strip().lower() in labels)

    def update(
        self,
        object_id: str,
        label: str,
        points_goal: np.ndarray,
    ) -> list[GripCandidate]:
        if not self._enabled_for_label(label):
            return []
        previous = self.models.get(object_id)
        fitted = fit_cylinder_surface(points_goal, self.config.surface, previous)
        if fitted is not None:
            if previous is not None:
                alpha = float(np.clip(self.config.surface.update_alpha, 0.0, 1.0))
                axis = fitted.axis.copy()
                if float(np.dot(axis, previous.axis)) < 0.0:
                    axis = -axis
                filtered_axis = _unit(
                    (1.0 - alpha) * previous.axis + alpha * axis,
                    previous.axis,
                )
                fitted = CylinderSurfaceModel(
                    center=(1.0 - alpha) * previous.center + alpha * fitted.center,
                    axis=filtered_axis,
                    radius=(1.0 - alpha) * previous.radius + alpha * fitted.radius,
                    height=(1.0 - alpha) * previous.height + alpha * fitted.height,
                    fit_rmse=fitted.fit_rmse,
                    inlier_ratio=fitted.inlier_ratio,
                    angular_coverage=fitted.angular_coverage,
                    updates=previous.updates + 1,
                    stale_frames=0,
                )
            self.models[object_id] = fitted
        elif previous is not None:
            previous.stale_frames += 1
            if previous.stale_frames > self.config.surface.stale_ttl_frames:
                del self.models[object_id]
                return []
        model = self.models.get(object_id)
        return sample_cylinder_candidates(model, self.config) if model is not None else []

    def finish_frame(self, active_object_ids: set[str]) -> None:
        for object_id in list(self.models):
            if object_id in active_object_ids:
                continue
            self.models[object_id].stale_frames += 1
            if self.models[object_id].stale_frames > self.config.surface.stale_ttl_frames:
                del self.models[object_id]

    def status(self, object_id: str) -> dict | None:
        model = self.models.get(object_id)
        return None if model is None else model.to_dict()


def _merge_candidate_cluster(cluster: list[GripCandidate]) -> GripCandidate:
    if len(cluster) == 1:
        return cluster[0]
    scores = np.asarray([max(EPS, float(item.geom_score)) for item in cluster], dtype=np.float64)
    weights = scores / max(EPS, float(np.sum(scores)))
    centers = np.stack([np.asarray(item.center, dtype=np.float64).reshape(3) for item in cluster])
    approaches = np.stack([np.asarray(item.approach, dtype=np.float64).reshape(3) for item in cluster])
    center = np.sum(centers * weights[:, None], axis=0)
    approach = _unit(np.sum(approaches * weights[:, None], axis=0), cluster[0].approach)
    width = float(np.sum(np.asarray([float(item.width) for item in cluster]) * weights))
    geom_score = float(np.max([float(item.geom_score) for item in cluster]))
    sources = sorted({str(item.source) for item in cluster})
    source = sources[0] if len(sources) == 1 else "merged(" + "+".join(sources) + ")"

    def merge_vector(name: str) -> np.ndarray | None:
        vectors = [getattr(item, name) for item in cluster]
        if any(vector is None for vector in vectors):
            return None
        stacked = np.stack(
            [np.asarray(vector, dtype=np.float64).reshape(3) for vector in vectors]
        )
        return _unit(np.sum(stacked * weights[:, None], axis=0), stacked[0])

    return GripCandidate(
        center=center,
        approach=approach,
        width=width,
        geom_score=geom_score,
        source=source,
        surface_normal=merge_vector("surface_normal"),
        surface_tangent=merge_vector("surface_tangent"),
        surface_axis=merge_vector("surface_axis"),
    )


def _merge_nearby_candidates(
    candidates: list[GripCandidate], merge_distance_m: float
) -> list[GripCandidate]:
    distance_threshold = max(0.0, float(merge_distance_m))
    if distance_threshold <= 0.0 or len(candidates) <= 1:
        return list(candidates)
    clusters: list[list[GripCandidate]] = []
    cluster_centers: list[np.ndarray] = []
    for candidate in sorted(candidates, key=lambda item: item.geom_score, reverse=True):
        center = np.asarray(candidate.center, dtype=np.float64).reshape(3)
        best_index = -1
        best_distance = float("inf")
        for index, cluster_center in enumerate(cluster_centers):
            distance = float(np.linalg.norm(center - cluster_center))
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index >= 0 and best_distance <= distance_threshold:
            clusters[best_index].append(candidate)
            cluster_centers[best_index] = _merge_candidate_cluster(clusters[best_index]).center
        else:
            clusters.append([candidate])
            cluster_centers.append(center.copy())
    merged = [_merge_candidate_cluster(cluster) for cluster in clusters]
    merged.sort(key=lambda item: item.geom_score, reverse=True)
    return merged


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
                    surface_normal=_unit(surface_point - center, -approach),
                    surface_tangent=_unit(
                        np.cross(axis, surface_point - center), table_normal
                    ),
                    surface_axis=_unit(axis, np.array([1.0, 0.0, 0.0])),
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
                surface_normal=-table_normal,
                surface_axis=None,
            )
        )

    merged = _merge_nearby_candidates(candidates, cfg.merge_distance_m)
    selected = merged[: max(1, cfg.top_k)]
    if not selected:
        selected.append(
            GripCandidate(
                center=_nearest_point(points, center),
                approach=approach,
                width=0.0,
                geom_score=0.0,
                source="median_fallback",
            )
        )
    return selected
