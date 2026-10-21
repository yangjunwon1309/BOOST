"""RGB-D calibration adaptation and raw depth-to-color reprojection."""

from __future__ import annotations

import dataclasses

import numpy as np

from .geometry import CameraIntrinsics
from .lcm_protocol import CameraCalibration, DepthFrame


def _proportional_scale(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    name: str,
) -> tuple[float, float]:
    """Return a safe resize scale, rejecting crops and aspect changes."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    if not np.isclose(scale_x, scale_y, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError(
            f"{name} aspect ratio changed from {source_width}x{source_height} "
            f"to {target_width}x{target_height}; crop/letterbox calibration is unsupported"
        )
    return scale_x, scale_y


def scale_camera_intrinsics(
    intrinsics: CameraIntrinsics,
    *,
    width: int,
    height: int,
) -> CameraIntrinsics:
    """Scale pinhole intrinsics for a proportional image resize."""
    scale_x, scale_y = _proportional_scale(
        source_width=intrinsics.width,
        source_height=intrinsics.height,
        target_width=width,
        target_height=height,
        name="camera image",
    )
    scaled = CameraIntrinsics(
        int(width),
        int(height),
        float(intrinsics.fx) * scale_x,
        float(intrinsics.fy) * scale_y,
        float(intrinsics.cx) * scale_x,
        float(intrinsics.cy) * scale_y,
    )
    scaled.validate()
    return scaled


def calibration_for_frame(
    calibration: CameraCalibration,
    *,
    color_width: int,
    color_height: int,
    depth_width: int | None = None,
    depth_height: int | None = None,
) -> CameraCalibration:
    """Adapt an LCM calibration to proportionally resized RGB and depth frames.

    This supports, for example, a calibrated 640x360 stream subsequently sent
    as 1280x720.  It deliberately rejects crop or letterbox transformations:
    those require new calibrated principal points rather than simple scaling.
    """
    color_scale_x, color_scale_y = _proportional_scale(
        source_width=calibration.color_width,
        source_height=calibration.color_height,
        target_width=color_width,
        target_height=color_height,
        name="color calibration",
    )
    depth_intrinsics = np.asarray(calibration.depth_intrinsics, dtype=np.float64).copy()
    color_intrinsics = np.asarray(calibration.color_intrinsics, dtype=np.float64).copy()
    effective_depth_width = calibration.depth_width
    effective_depth_height = calibration.depth_height
    if (depth_width is None) != (depth_height is None):
        raise ValueError("depth_width and depth_height must be provided together")
    if depth_width is not None and depth_height is not None:
        depth_scale_x, depth_scale_y = _proportional_scale(
            source_width=calibration.depth_width,
            source_height=calibration.depth_height,
            target_width=depth_width,
            target_height=depth_height,
            name="depth calibration",
        )
        depth_intrinsics *= np.array(
            [depth_scale_x, depth_scale_y, depth_scale_x, depth_scale_y],
            dtype=np.float64,
        )
        effective_depth_width = int(depth_width)
        effective_depth_height = int(depth_height)
    color_intrinsics *= np.array(
        [color_scale_x, color_scale_y, color_scale_x, color_scale_y], dtype=np.float64
    )
    return dataclasses.replace(
        calibration,
        depth_width=effective_depth_width,
        depth_height=effective_depth_height,
        color_width=int(color_width),
        color_height=int(color_height),
        depth_intrinsics=depth_intrinsics,
        color_intrinsics=color_intrinsics,
    )


def align_depth_to_color(frame: DepthFrame, calibration: CameraCalibration) -> np.ndarray:
    if (frame.width, frame.height) != (calibration.depth_width, calibration.depth_height):
        raise ValueError("raw depth dimensions do not match transmitted calibration")
    if frame.depth_scale_um != calibration.depth_scale_um:
        raise ValueError("depth scale does not match transmitted calibration")
    if not calibration.rectified:
        raise ValueError("pinhole alignment requires rectified depth and RGB streams")

    depth = frame.depth
    v, u = np.nonzero(depth)
    output = np.zeros((calibration.color_height, calibration.color_width), dtype=np.uint16)
    if not len(u):
        return output
    raw = depth[v, u]
    z = raw.astype(np.float64) * calibration.depth_scale_um / 1_000_000.0
    fx, fy, cx, cy = calibration.depth_intrinsics
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=0)
    points_color = calibration.rotation_color_depth @ points
    points_color += calibration.translation_color_depth_m[:, None]
    valid = points_color[2] > 0
    points_color = points_color[:, valid]
    if not points_color.shape[1]:
        return output
    cfx, cfy, ccx, ccy = calibration.color_intrinsics
    cu = np.rint(cfx * points_color[0] / points_color[2] + ccx).astype(np.int64)
    cv = np.rint(cfy * points_color[1] / points_color[2] + ccy).astype(np.int64)
    valid = (cu >= 0) & (cu < calibration.color_width) & (cv >= 0) & (cv < calibration.color_height)
    cu, cv, cz = cu[valid], cv[valid], points_color[2, valid]
    aligned_raw = np.clip(np.rint(cz * 1_000_000.0 / calibration.depth_scale_um), 1, 65535).astype(np.uint16)
    flat = cv * calibration.color_width + cu
    z_buffer = np.full(output.size, 65536, dtype=np.uint32)
    np.minimum.at(z_buffer, flat, aligned_raw.astype(np.uint32))
    z_buffer[z_buffer == 65536] = 0
    return z_buffer.astype(np.uint16).reshape(output.shape)
