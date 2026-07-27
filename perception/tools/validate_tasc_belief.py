#!/usr/bin/env python3
"""Validate TASC hierarchical belief with a synthetic RGB-D contact scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

try:
    import yaml
except ModuleNotFoundError:  # Host Python may be slimmer than the Thor container.
    yaml = None

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from seg_web_node.tasc_belief import BeliefInputObject, HierarchicalTascBelief


def box_points(center: np.ndarray, size: np.ndarray, n: int = 15) -> np.ndarray:
    axes = [np.linspace(-0.5 * size[i], 0.5 * size[i], n) for i in range(3)]
    surfaces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            grids = np.meshgrid(*axes, indexing="ij")
            grids[axis][...] = sign * 0.5 * size[axis]
            surface = np.stack(grids, axis=-1).reshape(-1, 3) + center
            surfaces.append(surface)
    return np.concatenate(surfaces, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Perception pipeline YAML. Uses tasc_belief section.",
    )
    parser.add_argument("--ticks", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = (
        yaml.safe_load(args.config.read_text())
        if yaml is not None and args.config.is_file()
        else {}
    )
    belief = HierarchicalTascBelief(config.get("tasc_belief", {}))

    bottle_center = np.array([0.42, 0.10, 0.18], dtype=np.float64)
    box_center = np.array([0.42, -0.12, 0.18], dtype=np.float64)
    objects = [
        BeliefInputObject(
            object_id="bottle:0",
            label="bottle",
            score=0.95,
            center_goal_m=bottle_center,
            points_goal_m=box_points(bottle_center, np.array([0.07, 0.07, 0.22])),
        ),
        BeliefInputObject(
            object_id="box:1",
            label="box",
            score=0.94,
            center_goal_m=box_center,
            points_goal_m=box_points(box_center, np.array([0.16, 0.12, 0.08])),
        ),
    ]
    ray_origin = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    ray_direction = bottle_center + np.array([0.0, 0.04, 0.0]) - ray_origin
    user_translation = np.array([0.45, 0.10, 0.05], dtype=np.float64)

    telemetry = {}
    now = time.monotonic()
    for tick in range(args.ticks):
        telemetry = belief.update(
            objects,
            eef_position_m=np.zeros(3, dtype=np.float64),
            user_translation_mps=user_translation,
            ray_origin_m=ray_origin,
            ray_direction=ray_direction,
            controller_orientation_xyzw=np.array([0, 0, 0, 1], dtype=np.float64),
            now_monotonic=now + tick * 0.12,
        )
        print(belief.format_log_line(), flush=True)

    target = telemetry.get("target") or {}
    if target.get("object_id") != "bottle:0":
        raise SystemExit(f"expected bottle target, got {target}")
    contact = np.asarray(target.get("position_goal_m"), dtype=np.float64)
    center_distance = float(np.linalg.norm(contact - bottle_center))
    if center_distance < 0.015:
        raise SystemExit(
            "selected grip contact stayed at the object center; expected surface contact"
        )
    print(
        "[TASC-belief] validation PASS "
        f"target={target['object_id']} grip={target['grip_index']} "
        f"contact={np.round(contact, 4).tolist()} "
        f"center_offset_m={center_distance:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
