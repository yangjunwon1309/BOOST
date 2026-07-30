"""One-shot alignment from native XR controller poses to robot EE poses."""

from __future__ import annotations

import numpy as np


def align_controller_pose(
    controller_transform: np.ndarray,
    controller_anchor: np.ndarray,
    robot_ee_anchor: np.ndarray,
) -> np.ndarray:
    """Apply native controller motion relative to a calibrated robot EE anchor.

    Both translations already use the robot/head-yaw basis, so position deltas
    transfer directly. Relative controller rotation is applied on top of the
    robot EE orientation captured at calibration.
    """

    current = np.asarray(controller_transform, dtype=np.float64).reshape(4, 4)
    source = np.asarray(controller_anchor, dtype=np.float64).reshape(4, 4)
    target = np.asarray(robot_ee_anchor, dtype=np.float64).reshape(4, 4)
    if not all(np.isfinite(value).all() for value in (current, source, target)):
        raise ValueError("non-finite XR controller alignment input")

    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, 3] = target[:3, 3] + current[:3, 3] - source[:3, 3]
    aligned[:3, :3] = (
        target[:3, :3] @ source[:3, :3].T @ current[:3, :3]
    )
    return aligned
