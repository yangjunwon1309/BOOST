"""Vectorized raw D435 depth-to-VideoHub-color reprojection."""

from __future__ import annotations

import numpy as np

from .lcm_protocol import CameraCalibration, DepthFrame


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
