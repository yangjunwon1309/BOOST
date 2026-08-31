"""Numpy rotation helpers retained for the legacy PICO pose stream."""

from __future__ import annotations

import numpy as np


def decompose_rotation_aa(rotation_aa: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split axis-angle rotations into twist about ``axis`` and residual swing."""
    rotation = np.asarray(rotation_aa, dtype=np.float64).reshape(-1, 3)
    twist_axis = np.asarray(axis, dtype=np.float64).reshape(3)
    twist_axis /= max(1.0e-12, float(np.linalg.norm(twist_axis)))
    angle = np.linalg.norm(rotation, axis=1, keepdims=True)
    half = 0.5 * angle
    scale = np.divide(np.sin(half), angle, out=np.full_like(angle, 0.5), where=angle > 1.0e-12)
    quaternion = np.concatenate([np.cos(half), scale * rotation], axis=1)
    vector = quaternion[:, 1:]
    twist_vector = (vector @ twist_axis)[:, None] * twist_axis
    twist = np.concatenate([quaternion[:, :1], twist_vector], axis=1)
    twist /= np.maximum(np.linalg.norm(twist, axis=1, keepdims=True), 1.0e-12)
    twist_inverse = twist * np.array([1.0, -1.0, -1.0, -1.0])
    w1, x1, y1, z1 = np.moveaxis(twist_inverse, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(quaternion, -1, 0)
    swing = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )
    return twist, swing
