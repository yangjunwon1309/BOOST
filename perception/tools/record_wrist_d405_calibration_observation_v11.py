#!/usr/bin/env python3
"""Append one timestamp-matched wrist-D405 calibration observation from LCM.

The selected segmented object must be a calibration target with a known point
either fixed in the pelvis frame (``--goal-point-m``) or, for one reference
capture, known relative to the chosen wrist (``--wrist-point-m``). A single
wrist-mounted point is only one correspondence; it cannot calibrate all 6
extrinsic degrees of freedom when repeated. The script records the worker's
native-grid D405 point and paired wrist FK transform; it never derives a
reference point from the camera extrinsic being calibrated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import lcm
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.wrist_d405_protocol_v11 import (  # noqa: E402
    WRIST_RESULT_CHANNEL,
    unpack_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", required=True)
    parser.add_argument("--side", choices=("left_wrist", "right_wrist"), required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--goal-point-m", nargs=3, type=float,
        metavar=("X", "Y", "Z"),
        help="Independently measured fixed-target point in pelvis metres.",
    )
    reference.add_argument(
        "--wrist-point-m", nargs=3, type=float,
        metavar=("X", "Y", "Z"),
        help="Known target point in the selected wrist-yaw-link frame, metres.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label",
        help="Require this exact SAM3 label; recommended when several objects are visible.",
    )
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    return parser.parse_args()


def load_observations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    observations = content.get("observations") if isinstance(content, dict) else None
    if not isinstance(observations, list):
        raise ValueError(f"{path} must contain 'observations: [...]'")
    return observations


def object_to_observation(
    obj: dict[str, Any], reference_point_m: np.ndarray, reference_key: str
) -> dict[str, Any] | None:
    try:
        camera_point = np.asarray(obj["center_camera_m"], dtype=np.float64).reshape(3)
        diagnostic = obj["transform_diagnostics"]
        goal_wrist = np.asarray(
            diagnostic["t_goal_wrist"], dtype=np.float64
        ).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not np.isfinite(camera_point).all()
        or not np.isfinite(goal_wrist).all()
        or not np.allclose(goal_wrist[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6)
    ):
        return None
    output = {
        "camera_point_m": camera_point.astype(float).tolist(),
        "t_goal_wrist": goal_wrist.astype(float).tolist(),
        "camera_id": str(obj.get("camera_id", "")),
        "track_id": str(obj.get("track_id", "")),
        "label": str(obj.get("label", "")),
        "source_timestamp_us": int(obj.get("source_timestamp_us", 0) or 0),
    }
    output[reference_key] = reference_point_m.astype(float).tolist()
    return output


def main() -> int:
    args = parse_args()
    reference_key = (
        "goal_point_m" if args.goal_point_m is not None else "wrist_point_m"
    )
    reference_point = np.asarray(
        args.goal_point_m if args.goal_point_m is not None else args.wrist_point_m,
        dtype=np.float64,
    ).reshape(3)
    if not np.isfinite(reference_point).all():
        raise SystemExit(f"--{reference_key.replace('_', '-')} must be finite")
    bus = lcm.LCM(args.lcm_url)
    captured: dict[str, Any] | None = None
    status: dict[str, Any] = {
        "results": 0,
        "valid_results": 0,
        "left_objects": 0,
        "valid_center_objects": 0,
        "diagnostic_objects": 0,
        "labels": set(),
        "last_error": None,
    }

    def on_result(_channel: str, data: bytes) -> None:
        nonlocal captured
        if captured is not None:
            return
        try:
            result = unpack_result(data)
        except (TypeError, ValueError):
            return
        status["results"] += 1
        status["last_error"] = result.get("error")
        if bool(result.get("valid", False)):
            status["valid_results"] += 1
        left_objects = [
            obj
            for obj in result.get("objects", [])
            if isinstance(obj, dict) and str(obj.get("camera_id", "")) == args.side
        ]
        status["left_objects"] += len(left_objects)
        status["valid_center_objects"] += sum(
            obj.get("center_camera_m") is not None for obj in left_objects
        )
        status["diagnostic_objects"] += sum(
            isinstance(obj.get("transform_diagnostics"), dict)
            for obj in left_objects
        )
        status["labels"].update(
            str(obj.get("label", "")) for obj in left_objects
        )
        candidates = [
            obj
            for obj in left_objects
            if obj.get("center_camera_m") is not None
            and (args.label is None or str(obj.get("label", "")) == args.label)
        ]
        if not candidates:
            return
        # For an explicitly isolated calibration target, highest SAM score is
        # the least ambiguous deterministic selection.
        candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        captured = object_to_observation(
            candidates[0], reference_point, reference_key
        )

    bus.subscribe(WRIST_RESULT_CHANNEL, on_result)
    deadline = time.monotonic() + max(0.1, float(args.timeout_sec))
    while captured is None and time.monotonic() < deadline:
        bus.handle_timeout(100)
    if captured is None:
        raise SystemExit(
            "no usable calibration object; "
            f"results={status['results']} valid_results={status['valid_results']} "
            f"{args.side}_objects={status['left_objects']} "
            f"valid_center_objects={status['valid_center_objects']} "
            f"transform_diagnostics={status['diagnostic_objects']} "
            f"labels={sorted(status['labels'])} last_error={status['last_error']!r}. "
            "The target must be a left-wrist SAM3 object with valid depth; "
            "restart the updated remote wrist worker if transform_diagnostics=0."
        )
    observations = load_observations(args.output)
    observations.append(captured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump({"observations": observations}, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"appended observation={len(observations)} side={args.side} "
        f"track={captured['track_id']!r} to {args.output}"
    )
    if reference_key == "wrist_point_m":
        goal_wrist = np.asarray(captured["t_goal_wrist"], dtype=np.float64)
        fixed_goal = (
            goal_wrist @ np.append(reference_point, 1.0)
        )[:3]
        print(
            "fixed_goal_point_m="
            + " ".join(f"{value:.9f}" for value in fixed_goal)
            + " (use this for later --goal-point-m captures only if the dot "
              "remains fixed in the environment)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
