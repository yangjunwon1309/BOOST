"""Deterministic synthetic depth generation for Thor-only validation."""

from __future__ import annotations

import cv2
import numpy as np


def make_mock_depth(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image must be HxWx3 BGR")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 9.0)
    depth = 700.0 + 1100.0 * (smooth / 255.0)
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    ellipse = ((xx - width * 0.5) / (width * 0.32)) ** 2 + (
        (yy - height * 0.55) / (height * 0.28)
    ) ** 2 <= 1.0
    depth[ellipse] -= 180.0
    depth[:2, :] = 0
    depth[-2:, :] = 0
    depth[:, :2] = 0
    depth[:, -2:] = 0
    return np.clip(depth, 0, 65535).astype(np.uint16)
