#!/usr/bin/env python3
"""Write a repeated synthetic fist/open hand-landmark stream for MuJoCo tests."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POINT_COUNT = 25
SIDES = ("left", "right")


def _read_pose(path: Path) -> dict[str, list[list[float]]]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read hand pose {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object with left and right arrays")

    result: dict[str, list[list[float]]] = {}
    for side in SIDES:
        points = payload.get(side)
        if not isinstance(points, list) or len(points) != POINT_COUNT:
            raise ValueError(
                f"{path}: {side} must contain exactly {POINT_COUNT} points"
            )
        checked_points: list[list[float]] = []
        for point_index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(
                    f"{path}: {side}[{point_index}] must contain three coordinates"
                )
            checked: list[float] = []
            for coordinate_index, coordinate in enumerate(point):
                if isinstance(coordinate, bool) or not isinstance(
                    coordinate, (int, float)
                ):
                    raise ValueError(
                        f"{path}: {side}[{point_index}][{coordinate_index}] "
                        "must be numeric"
                    )
                value = float(coordinate)
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}: {side}[{point_index}][{coordinate_index}] "
                        "must be finite"
                    )
                checked.append(value)
            checked_points.append(checked)
        result[side] = checked_points
    return result


def _interpolate_pose(
    start: dict[str, list[list[float]]],
    end: dict[str, list[list[float]]],
    alpha: float,
) -> dict[str, list[list[float]]]:
    return {
        side: [
            [a + (b - a) * alpha for a, b in zip(start[side][i], end[side][i])]
            for i in range(POINT_COUNT)
        ]
        for side in SIDES
    }


def _atomic_write(path: Path, pose: dict[str, list[list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_name = stream.name
            json.dump(pose, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)


def _pose_at(
    elapsed: float,
    *,
    fist: dict[str, list[list[float]]],
    opened: dict[str, list[list[float]]],
    fist_hold_s: float,
    transition_s: float,
    open_hold_s: float,
    cycles: int,
) -> tuple[dict[str, list[list[float]]], str, int, bool]:
    cycle_s = 2.0 * fist_hold_s + 2.0 * transition_s + open_hold_s
    if cycles and elapsed >= cycles * cycle_s:
        return fist, "complete", cycles, True

    cycle_index = int(elapsed // cycle_s)
    phase_time = elapsed - cycle_index * cycle_s
    if phase_time < fist_hold_s:
        return fist, "fist_hold", cycle_index + 1, False
    phase_time -= fist_hold_s
    if phase_time < transition_s:
        alpha = phase_time / transition_s
        return _interpolate_pose(fist, opened, alpha), "opening", cycle_index + 1, False
    phase_time -= transition_s
    if phase_time < open_hold_s:
        return opened, "open_hold", cycle_index + 1, False
    phase_time -= open_hold_s
    alpha = min(1.0, phase_time / transition_s)
    return _interpolate_pose(opened, fist, alpha), "closing", cycle_index + 1, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fist-file",
        type=Path,
        default=ROOT / "configs/hands/revo2_hand_override_fist.json",
    )
    parser.add_argument(
        "--open-file",
        type=Path,
        default=ROOT / "configs/hands/revo2_hand_override.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--fist-hold-s", type=float, default=1.0)
    parser.add_argument("--transition-s", type=float, default=1.5)
    parser.add_argument("--open-hold-s", type=float, default=1.0)
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="number of complete fist/open cycles; 0 repeats forever (default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive and finite")
    if args.cycles < 0:
        raise ValueError("--cycles cannot be negative")
    for name in ("fist_hold_s", "transition_s", "open_hold_s"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")

    fist_path = args.fist_file.expanduser().resolve()
    open_path = args.open_file.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path in (fist_path, open_path):
        raise ValueError("output path must not overwrite either input template")
    fist = _read_pose(fist_path)
    opened = _read_pose(open_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[hand-motion] fist={fist_path} open={open_path} "
        f"output={output_path} rate={args.rate_hz:g}Hz "
        f"timing={args.fist_hold_s:g}/{args.transition_s:g}/"
        f"{args.open_hold_s:g}/{args.transition_s:g}s "
        f"cycles={'infinite' if args.cycles == 0 else args.cycles}",
        flush=True,
    )
    start = time.monotonic()
    period = 1.0 / args.rate_hz
    tick = 0
    last_phase: str | None = None
    while True:
        elapsed = time.monotonic() - start
        pose, phase, cycle_number, done = _pose_at(
            elapsed,
            fist=fist,
            opened=opened,
            fist_hold_s=args.fist_hold_s,
            transition_s=args.transition_s,
            open_hold_s=args.open_hold_s,
            cycles=args.cycles,
        )
        _atomic_write(output_path, pose)
        if phase != last_phase:
            if phase == "complete":
                print(f"[hand-motion] completed {cycle_number} cycle(s)", flush=True)
            else:
                print(f"[hand-motion] cycle={cycle_number} phase={phase}", flush=True)
            last_phase = phase
        if done:
            return 0
        tick += 1
        deadline = start + tick * period
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[hand-motion] error: {exc}") from exc
