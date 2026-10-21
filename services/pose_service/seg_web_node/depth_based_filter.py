"""Depth-buffer robot self-filter for segmentation masks.

The filter follows the RGB-D self-filter recipe: render padded robot collision
primitives into a depth buffer, then veto measured pixels whose depth is at or
behind that robot depth with hysteresis.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import cv2
import numpy as np

from .geometry import CameraIntrinsics
from .robot_geometry import g1_link_transforms


DEFAULT_LINK_CHAINS = (
    (
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ),
    (
        "torso_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ),
    ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"),
)


@dataclasses.dataclass(frozen=True)
class CapsulePrimitive:
    start_goal_m: np.ndarray
    end_goal_m: np.ndarray
    radius_m: float


@dataclasses.dataclass
class DepthSelfFilterConfig:
    enabled: bool = True
    margin_high_m: float = 0.055
    margin_low_m: float = 0.030
    temporal_alpha: float = 0.35
    temporal_reset_threshold_m: float = 0.080
    bilateral_diameter: int = 5
    bilateral_sigma_depth_m: float = 0.040
    bilateral_sigma_space_px: float = 4.0
    padding_scale: float = 1.05
    arm_radius_m: float = 0.055
    wrist_radius_m: float = 0.075
    torso_radius_m: float = 0.090
    sample_step_m: float = 0.020

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DepthSelfFilterConfig":
        data = dict(payload or {})
        config = cls()
        for field in dataclasses.fields(cls):
            if field.name in data:
                setattr(config, field.name, data[field.name])
        config.enabled = bool(config.enabled)
        config.margin_high_m = max(float(config.margin_high_m), 0.0)
        config.margin_low_m = max(float(config.margin_low_m), 0.0)
        if config.margin_low_m > config.margin_high_m:
            config.margin_low_m, config.margin_high_m = (
                config.margin_high_m,
                config.margin_low_m,
            )
        config.temporal_alpha = float(np.clip(config.temporal_alpha, 0.0, 1.0))
        config.temporal_reset_threshold_m = max(float(config.temporal_reset_threshold_m), 0.0)
        config.bilateral_diameter = max(0, int(config.bilateral_diameter))
        if config.bilateral_diameter and config.bilateral_diameter % 2 == 0:
            config.bilateral_diameter += 1
        config.bilateral_sigma_depth_m = max(float(config.bilateral_sigma_depth_m), 1.0e-6)
        config.bilateral_sigma_space_px = max(float(config.bilateral_sigma_space_px), 1.0e-6)
        config.padding_scale = max(float(config.padding_scale), 1.0)
        config.arm_radius_m = max(float(config.arm_radius_m), 1.0e-4)
        config.wrist_radius_m = max(float(config.wrist_radius_m), 1.0e-4)
        config.torso_radius_m = max(float(config.torso_radius_m), 1.0e-4)
        config.sample_step_m = max(float(config.sample_step_m), 1.0e-3)
        return config


class DepthBasedRobotFilter:
    """Apply padded robot-depth veto with spatial/temporal depth filtering."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.config = DepthSelfFilterConfig.from_dict(options)
        self._temporal_depth_m: np.ndarray | None = None
        self._hysteresis_mask: np.ndarray | None = None
        self._robot_mask: np.ndarray | None = None
        self.last_status: dict[str, Any] = self._empty_status("uninitialized")

    def metadata(self) -> dict[str, Any]:
        return dict(self.last_status)

    def _empty_status(self, reason: str | None = None) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "ready": False,
            "reason": reason,
            "robot_pixels": 0,
            "trimmed_masks": 0,
            "removed_masks": 0,
            "margin_high_m": float(self.config.margin_high_m),
            "margin_low_m": float(self.config.margin_low_m),
        }

    def clear_context(self, reason: str | None = None) -> None:
        self._robot_mask = None
        self.last_status = self._empty_status(reason)

    def set_context(
        self,
        *,
        image_bgr: np.ndarray | None,
        depth_raw: np.ndarray | None,
        depth_scale_um: int | None,
        intrinsics: CameraIntrinsics | None,
        t_goal_cam: np.ndarray | None,
        q: np.ndarray | None,
    ) -> None:
        if not self.config.enabled:
            self.clear_context("disabled")
            return
        if depth_raw is None or depth_scale_um is None or intrinsics is None:
            self.clear_context("missing_depth_or_intrinsics")
            return
        if t_goal_cam is None or q is None:
            self.clear_context("missing_robot_pose")
            return
        depth = np.asarray(depth_raw)
        if depth.ndim != 2 or depth.shape != (intrinsics.height, intrinsics.width):
            self.clear_context("depth_shape_mismatch")
            return
        try:
            measured_depth = self._prefilter_depth(
                depth.astype(np.float32) * (float(depth_scale_um) / 1_000_000.0),
                image_bgr,
            )
            robot_depth = render_robot_link_depth(
                q=np.asarray(q, dtype=np.float64),
                t_goal_cam=np.asarray(t_goal_cam, dtype=np.float64),
                intrinsics=intrinsics,
                config=self.config,
            )
            robot_mask = self._depth_veto(measured_depth, robot_depth)
        except Exception as error:
            self.clear_context(f"{type(error).__name__}: {error}")
            return
        self._robot_mask = robot_mask
        self.last_status = {
            **self._empty_status(None),
            "ready": True,
            "robot_pixels": int(robot_mask.sum()),
        }

    def filter_mask(self, raw_mask: np.ndarray | None, *, min_area_px: int) -> np.ndarray | None:
        if raw_mask is None:
            return None
        mask = np.asarray(raw_mask, dtype=bool)
        if int(mask.sum()) < int(min_area_px):
            return None
        robot_mask = self._robot_mask
        if robot_mask is None:
            return mask
        if robot_mask.shape != mask.shape:
            robot_mask = cv2.resize(
                robot_mask.astype(np.uint8),
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        trimmed = mask & ~robot_mask
        if int(trimmed.sum()) != int(mask.sum()):
            self.last_status["trimmed_masks"] = int(self.last_status.get("trimmed_masks", 0)) + 1
        if int(trimmed.sum()) < int(min_area_px):
            self.last_status["removed_masks"] = int(self.last_status.get("removed_masks", 0)) + 1
            return None
        return trimmed

    def _prefilter_depth(self, depth_m: np.ndarray, image_bgr: np.ndarray | None) -> np.ndarray:
        del image_bgr  # The deployed OpenCV path uses depth bilateral filtering.
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        spatial = depth_m.astype(np.float32, copy=True)
        if self.config.bilateral_diameter > 0 and valid.any():
            filtered = cv2.bilateralFilter(
                spatial,
                self.config.bilateral_diameter,
                self.config.bilateral_sigma_depth_m,
                self.config.bilateral_sigma_space_px,
            )
            spatial[valid] = filtered[valid]
        previous = self._temporal_depth_m
        if previous is None or previous.shape != spatial.shape:
            combined = spatial
        else:
            temporal = (
                self.config.temporal_alpha * spatial
                + (1.0 - self.config.temporal_alpha) * previous
            )
            reset = np.abs(spatial - temporal) > self.config.temporal_reset_threshold_m
            combined = np.where(reset, spatial, temporal).astype(np.float32)
            combined[~valid] = spatial[~valid]
        self._temporal_depth_m = combined.copy()
        return combined

    def _depth_veto(self, measured_depth_m: np.ndarray, robot_depth_m: np.ndarray) -> np.ndarray:
        valid = (
            np.isfinite(measured_depth_m)
            & (measured_depth_m > 0.0)
            & np.isfinite(robot_depth_m)
        )
        enter = valid & (measured_depth_m >= robot_depth_m - self.config.margin_low_m)
        hold = valid & (measured_depth_m >= robot_depth_m - self.config.margin_high_m)
        previous = self._hysteresis_mask
        if previous is None or previous.shape != enter.shape:
            robot = enter
        else:
            robot = enter | (previous & hold)
        self._hysteresis_mask = robot.copy()
        return robot


class DepthSelfFilterBackendMixin:
    """Backend helper exposing the shared depth self-filter API."""

    def _init_depth_self_filter(self, options: dict[str, Any] | None = None) -> None:
        self.depth_self_filter = DepthBasedRobotFilter(options)

    def set_depth_filter_context(
        self,
        *,
        image_bgr: np.ndarray | None,
        depth_raw: np.ndarray | None,
        depth_scale_um: int | None,
        intrinsics: CameraIntrinsics | None,
        t_goal_cam: np.ndarray | None,
        q: np.ndarray | None,
    ) -> None:
        self.depth_self_filter.set_context(
            image_bgr=image_bgr,
            depth_raw=depth_raw,
            depth_scale_um=depth_scale_um,
            intrinsics=intrinsics,
            t_goal_cam=t_goal_cam,
            q=q,
        )

    def _apply_depth_self_filter(self, mask: np.ndarray | None) -> np.ndarray | None:
        return self.depth_self_filter.filter_mask(mask, min_area_px=int(self.min_area_px))

    def _depth_self_filter_metadata(self) -> dict[str, Any]:
        return self.depth_self_filter.metadata()


def robot_capsule_primitives(q: np.ndarray, config: DepthSelfFilterConfig) -> list[CapsulePrimitive]:
    transforms = g1_link_transforms(q)
    primitives: list[CapsulePrimitive] = []
    for chain in DEFAULT_LINK_CHAINS:
        for first, second in zip(chain[:-1], chain[1:]):
            if first not in transforms or second not in transforms:
                continue
            radius = config.arm_radius_m
            if first == "pelvis" or second == "torso_link":
                radius = config.torso_radius_m
            if "wrist" in first or "wrist" in second:
                radius = config.wrist_radius_m
            primitives.append(
                CapsulePrimitive(
                    start_goal_m=transforms[first][:3, 3].astype(np.float64),
                    end_goal_m=transforms[second][:3, 3].astype(np.float64),
                    radius_m=float(radius) * config.padding_scale,
                )
            )
    return primitives


def render_robot_link_depth(
    *,
    q: np.ndarray,
    t_goal_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: DepthSelfFilterConfig,
) -> np.ndarray:
    """Render sampled padded link capsules to a camera-frame depth buffer."""
    intrinsics.validate()
    transform = np.asarray(t_goal_cam, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    depth = np.full((intrinsics.height, intrinsics.width), np.inf, dtype=np.float32)

    for primitive in robot_capsule_primitives(q, config):
        start = np.asarray(primitive.start_goal_m, dtype=np.float64)
        end = np.asarray(primitive.end_goal_m, dtype=np.float64)
        radius = float(primitive.radius_m)
        length = float(np.linalg.norm(end - start))
        samples = max(2, int(np.ceil(length / config.sample_step_m)) + 1)
        for alpha in np.linspace(0.0, 1.0, samples):
            center_goal = start * (1.0 - alpha) + end * alpha
            center_camera = rotation.T @ (center_goal - translation)
            _render_sphere_depth(depth, center_camera, radius, intrinsics)
    return depth


def _render_sphere_depth(
    depth: np.ndarray,
    center_camera_m: np.ndarray,
    radius_m: float,
    intrinsics: CameraIntrinsics,
) -> None:
    center = np.asarray(center_camera_m, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all() or center[2] <= radius_m:
        return
    u_center = intrinsics.fx * center[0] / center[2] + intrinsics.cx
    v_center = intrinsics.fy * center[1] / center[2] + intrinsics.cy
    radius_px = int(np.ceil(max(intrinsics.fx, intrinsics.fy) * radius_m / center[2])) + 1
    if radius_px <= 0:
        return
    x0 = max(0, int(np.floor(u_center - radius_px)))
    x1 = min(intrinsics.width, int(np.ceil(u_center + radius_px + 1)))
    y0 = max(0, int(np.floor(v_center - radius_px)))
    y1 = min(intrinsics.height, int(np.ceil(v_center + radius_px + 1)))
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float64)
    ys = np.arange(y0, y1, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)
    dx = (uu - intrinsics.cx) / intrinsics.fx
    dy = (vv - intrinsics.cy) / intrinsics.fy
    dz = np.ones_like(dx)
    a = dx * dx + dy * dy + dz * dz
    b = -2.0 * (center[0] * dx + center[1] * dy + center[2] * dz)
    c = float(center @ center - radius_m * radius_m)
    discriminant = b * b - 4.0 * a * c
    hit = discriminant >= 0.0
    if not hit.any():
        return
    t = (-b - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * a)
    hit &= t > 0.0
    if not hit.any():
        return
    patch = depth[y0:y1, x0:x1]
    candidate = t.astype(np.float32)
    patch[hit] = np.minimum(patch[hit], candidate[hit])
