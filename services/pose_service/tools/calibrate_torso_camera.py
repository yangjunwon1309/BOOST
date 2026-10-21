#!/usr/bin/env python3
"""Fit T_torso_cam from corresponding 3-D calibration target points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def fit_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("camera_points_m and torso_points_m must be matching Nx3 arrays")
    if len(source) < 6:
        raise ValueError("use at least six well-spread point correspondences")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    if np.linalg.matrix_rank(source - source_center, tol=1e-5) < 2 or np.linalg.matrix_rank(target - target_center, tol=1e-5) < 2:
        raise ValueError("calibration points must span at least two spatial axes")
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correspondences", type=Path)
    parser.add_argument("--output", type=Path, default=Path("config/extrinsics.yaml"))
    parser.add_argument("--max-rms-m", type=float, default=0.01)
    args = parser.parse_args()
    data = yaml.safe_load(args.correspondences.read_text())
    camera = np.asarray(data["camera_points_m"], dtype=float)
    torso = np.asarray(data["torso_points_m"], dtype=float)
    transform = fit_rigid(camera, torso)
    predicted = camera @ transform[:3, :3].T + transform[:3, 3]
    errors = np.linalg.norm(predicted - torso, axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    print(f"points={len(errors)} rms={rms * 1000:.1f}mm max={errors.max() * 1000:.1f}mm")
    if rms > args.max_rms_m:
        raise SystemExit(f"refusing output: RMS exceeds {args.max_rms_m * 1000:.1f}mm")
    output = {
        "T_torso_cam": transform.tolist(),
        "T_goal_cam": None,
        "calibration_rms_m": rms,
        "calibration_max_error_m": float(errors.max()),
    }
    args.output.write_text(yaml.safe_dump(output, sort_keys=False))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
