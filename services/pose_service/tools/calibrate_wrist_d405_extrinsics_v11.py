#!/usr/bin/env python3
"""Fit one D405 optical->wrist transform from measured 3-D correspondences.

Each observation must contain a D405 point and the same physical point. A
known ``wrist_point_m`` may be used for an individual correspondence, but
repeating one wrist-mounted point cannot identify a full transform. For a
target fixed in the pelvis frame, provide ``goal_point_m`` plus either the
robot's 29-element joint vector or the timestamp-matched ``t_goal_wrist``
matrix emitted in result ``transform_diagnostics``::

  observations:
    - camera_point_m: [0.012, -0.004, 0.310]
      goal_point_m: [0.420, 0.180, 0.260]
      q: [ ... 29 G1 joint values ... ]
    - camera_point_m: [0.018, 0.006, 0.294]
      wrist_point_m: [0.200, 0.000, 0.000]

``goal_point_m`` must come from an independent measurement (for example a
surveyed fixture/marker or a repeatable robot-contact point).  Do not derive
it from the current D405 transform: that would only reproduce the existing
axis error.  Capture at least 12 observations spanning several wrist poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from remote_offload.wrist_d405_geometry_pipeline_v11 import CAMERA_TO_WRIST_LINK
from seg_web_node.robot_geometry import g1_link_transforms


def fit_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper rigid transform mapping Nx3 ``source`` to ``target``."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching Nx3 arrays")
    if len(source) < 6:
        raise ValueError("use at least six well-spread correspondences")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    if np.linalg.matrix_rank(source - source_center, tol=1e-5) < 2:
        raise ValueError("camera points must span at least two spatial axes")
    if np.linalg.matrix_rank(target - target_center, tol=1e-5) < 2:
        raise ValueError("wrist-frame points must span at least two spatial axes")
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def observations_to_wrist_points(
    observations: list[dict[str, Any]], side: str
) -> tuple[np.ndarray, np.ndarray]:
    """Convert independently measured pelvis points to the active wrist frame."""
    if side not in CAMERA_TO_WRIST_LINK:
        raise ValueError(f"side must be one of {sorted(CAMERA_TO_WRIST_LINK)}")
    camera_points: list[np.ndarray] = []
    wrist_points: list[np.ndarray] = []
    wrist_link = CAMERA_TO_WRIST_LINK[side]
    for index, observation in enumerate(observations):
        try:
            camera = np.asarray(observation["camera_point_m"], dtype=np.float64).reshape(3)
            raw_wrist_point = observation.get("wrist_point_m")
            if raw_wrist_point is not None:
                wrist = np.asarray(
                    raw_wrist_point, dtype=np.float64
                ).reshape(3)
            else:
                goal = np.asarray(
                    observation["goal_point_m"], dtype=np.float64
                ).reshape(3)
                raw_goal_wrist = observation.get("t_goal_wrist")
                if raw_goal_wrist is not None:
                    goal_wrist = np.asarray(
                        raw_goal_wrist, dtype=np.float64
                    ).reshape(4, 4)
                    if (
                        not np.isfinite(goal_wrist).all()
                        or not np.allclose(
                            goal_wrist[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6
                        )
                    ):
                        raise ValueError(
                            "t_goal_wrist is not a finite homogeneous transform"
                        )
                else:
                    q = np.asarray(observation["q"], dtype=np.float64).reshape(-1)
                    goal_wrist = g1_link_transforms(q)[wrist_link]
                wrist = (np.linalg.inv(goal_wrist) @ np.append(goal, 1.0))[:3]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid observation {index}: {error}") from error
        if not np.isfinite(camera).all():
            raise ValueError(f"observation {index} contains a non-finite camera point")
        if not np.isfinite(wrist).all():
            raise ValueError(f"observation {index} contains non-finite wrist point")
        camera_points.append(camera)
        wrist_points.append(wrist)
    return np.asarray(camera_points), np.asarray(wrist_points)


def robust_fit(
    camera_points: np.ndarray,
    wrist_points: np.ndarray,
    outlier_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively reject gross correspondence mistakes, then refit."""
    active = np.ones(len(camera_points), dtype=bool)
    for _ in range(3):
        transform = fit_rigid(camera_points[active], wrist_points[active])
        predicted = camera_points @ transform[:3, :3].T + transform[:3, 3]
        errors = np.linalg.norm(predicted - wrist_points, axis=1)
        proposed = errors <= float(outlier_threshold_m)
        if proposed.sum() < 6 or np.array_equal(proposed, active):
            break
        active = proposed
    transform = fit_rigid(camera_points[active], wrist_points[active])
    predicted = camera_points @ transform[:3, :3].T + transform[:3, 3]
    errors = np.linalg.norm(predicted - wrist_points, axis=1)
    return transform, errors, active


def fit_translation_with_fixed_rotation(
    camera_points: np.ndarray,
    wrist_points: np.ndarray,
    rotation_wrist_camera: np.ndarray,
    outlier_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit translation only, retaining the supplied wrist<-camera rotation."""
    rotation = np.asarray(rotation_wrist_camera, dtype=np.float64).reshape(3, 3)
    translations = wrist_points - camera_points @ rotation.T
    estimate = np.median(translations, axis=0)
    errors = np.linalg.norm(translations - estimate, axis=1)
    active = errors <= float(outlier_threshold_m)
    if not active.any():
        active = np.ones(len(translations), dtype=bool)
    estimate = np.median(translations[active], axis=0)
    errors = np.linalg.norm(translations - estimate, axis=1)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = estimate
    return transform, errors, active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correspondences", type=Path, help="YAML observations file")
    parser.add_argument("--side", choices=sorted(CAMERA_TO_WRIST_LINK), required=True)
    parser.add_argument(
        "--base-extrinsics", type=Path,
        default=Path("config/wrist_d405_extrinsics_v11.yaml"),
        help="Existing wrist-extrinsics YAML whose selected matrix is replaced.",
    )
    parser.add_argument("--output", type=Path, help="Write calibrated YAML here; omitted prints YAML only.")
    parser.add_argument("--outlier-threshold-mm", type=float, default=15.0)
    parser.add_argument("--max-rms-mm", type=float, default=10.0)
    parser.add_argument(
        "--translation-only", action="store_true",
        help="Keep the base YAML rotation and fit only T_wrist_camera translation; one known point is sufficient.",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.correspondences.read_text(encoding="utf-8")) or {}
    observations = raw.get("observations") if isinstance(raw, dict) else raw
    if not isinstance(observations, list):
        raise SystemExit("correspondences must be a YAML list or contain observations: [...]")
    camera, wrist = observations_to_wrist_points(observations, args.side)
    output = yaml.safe_load(args.base_extrinsics.read_text(encoding="utf-8")) or {}
    threshold_m = max(0.001, float(args.outlier_threshold_mm) / 1000.0)
    if args.translation_only:
        base = np.asarray(output.get(f"T_{args.side}_camera"), dtype=np.float64)
        if base.shape != (4, 4) or not np.isfinite(base).all():
            raise SystemExit(f"base YAML has no valid T_{args.side}_camera")
        transform, errors, active = fit_translation_with_fixed_rotation(
            camera, wrist, base[:3, :3], threshold_m
        )
    else:
        transform, errors, active = robust_fit(camera, wrist, threshold_m)
    kept = errors[active]
    rms_m = float(np.sqrt(np.mean(kept ** 2)))
    max_m = float(kept.max())
    print(
        f"side={args.side} observations={len(errors)} kept={int(active.sum())} "
        f"rms_mm={rms_m * 1000.0:.2f} max_mm={max_m * 1000.0:.2f}"
    )
    if rms_m > float(args.max_rms_mm) / 1000.0:
        raise SystemExit(
            f"refusing calibrated output: RMS {rms_m * 1000.0:.2f}mm exceeds "
            f"--max-rms-mm {args.max_rms_mm:.2f}"
        )
    output[f"T_{args.side}_camera"] = transform.astype(float).tolist()
    output["extrinsics_source"] = (
        "translation_fit_fixed_rotation_wrist_d405_v11"
        if args.translation_only
        else "correspondence_fit_wrist_d405_v11"
    )
    output["calibration_status"] = (
        "translation_fit_fixed_rotation"
        if args.translation_only
        else "correspondence_fit"
    )
    output[f"{args.side}_calibration_rms_m"] = rms_m
    output[f"{args.side}_calibration_max_error_m"] = max_m
    output[f"{args.side}_calibration_observations"] = int(active.sum())
    serialized = yaml.safe_dump(output, sort_keys=False)
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
