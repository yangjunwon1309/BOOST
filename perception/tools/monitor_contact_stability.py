#!/usr/bin/env python3
"""Monitor filtered object/cylinder/contact stability on remote perception LCM."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import lcm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.protocol import REMOTE_RESULT_CHANNEL, unpack_result  # noqa: E402


def _vector(payload: Any, length: int = 3) -> np.ndarray | None:
    if payload is None:
        return None
    try:
        value = np.asarray(payload, dtype=np.float64).reshape(length)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value).all() else None


def _fmt_vector(value: np.ndarray | None) -> str:
    if value is None:
        return "none"
    return "[" + ",".join(f"{item:+.4f}" for item in value) + "]"


def _point_statistics(samples: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    points = [sample[key] for sample in samples if sample.get(key) is not None]
    if not points:
        return None
    array = np.asarray(points, dtype=np.float64)
    mean = np.mean(array, axis=0)
    deviations = np.linalg.norm(array - mean, axis=1)
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    return {
        "count": int(array.shape[0]),
        "mean": mean,
        "rms_cm": 100.0 * float(np.sqrt(np.mean(np.square(deviations)))),
        "p95_cm": 100.0 * float(np.percentile(deviations, 95)),
        "max_cm": 100.0 * float(np.max(deviations)),
        "step_p50_cm": 100.0 * float(np.percentile(steps, 50)) if steps.size else 0.0,
        "step_p95_cm": 100.0 * float(np.percentile(steps, 95)) if steps.size else 0.0,
        "step_max_cm": 100.0 * float(np.max(steps)) if steps.size else 0.0,
    }


def _axis_jitter_deg(samples: list[dict[str, Any]]) -> float | None:
    axes = [sample["axis"] for sample in samples if sample.get("axis") is not None]
    if not axes:
        return None
    reference = np.asarray(axes[0], dtype=np.float64)
    aligned = []
    for axis in axes:
        value = np.asarray(axis, dtype=np.float64)
        aligned.append(-value if float(np.dot(value, reference)) < 0.0 else value)
    mean = np.mean(np.asarray(aligned), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1.0e-9:
        return None
    mean /= norm
    angles = [
        math.degrees(math.acos(float(np.clip(np.dot(axis, mean), -1.0, 1.0))))
        for axis in aligned
    ]
    return float(np.sqrt(np.mean(np.square(angles))))


def _median(samples: list[dict[str, Any]], key: str, scale: float = 1.0) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return scale * float(np.median(values)) if values else None


def _format_metric(value: float | None, suffix: str) -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def _csv_row(sample: dict[str, Any]) -> dict[str, Any]:
    row = {
        "receive_time_s": sample["receive_time_s"],
        "sequence": sample["sequence"],
        "source_rgb_timestamp_us": sample["source_rgb_timestamp_us"],
        "track_id": sample["track_id"],
        "label": sample["label"],
        "score": sample["score"],
        "median_depth_m": sample["median_depth_m"],
        "fit_rmse_m": sample["fit_rmse_m"],
        "inlier_ratio": sample["inlier_ratio"],
        "angular_coverage": sample["angular_coverage"],
        "radius_m": sample["radius_m"],
        "height_m": sample["height_m"],
        "surface_updates": sample["surface_updates"],
        "surface_stale_frames": sample["surface_stale_frames"],
    }
    for prefix in ("center", "contact", "axis"):
        value = sample.get(prefix)
        for index, axis in enumerate("xyz"):
            row[f"{prefix}_{axis}"] = None if value is None else float(value[index])
    for key, value in sample["point_filter"].items():
        row[f"filter_{key}"] = value
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lcm-url",
        default="udpm://239.255.77.67:7667?ttl=1",
        help="Wi-Fi LCM URL on the remote worker network.",
    )
    parser.add_argument("--result-channel", default=REMOTE_RESULT_CHANNEL)
    parser.add_argument("--label", default="bottle")
    parser.add_argument("--track-id", default=None)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--report-every-sec", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-rms-jitter-cm", type=float, default=1.0)
    parser.add_argument("--reference-contact-goal-m", type=float, nargs=3, default=None)
    parser.add_argument("--max-mean-error-cm", type=float, default=1.0)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--print-every-frame", action="store_true")
    args = parser.parse_args()

    if args.duration_sec <= 0.0 or args.window_sec <= 0.0:
        raise SystemExit("duration/window must be positive")
    reference = _vector(args.reference_contact_goal_m)
    bus = lcm.LCM(args.lcm_url)
    samples: list[dict[str, Any]] = []
    locked_track_id = args.track_id
    received_results = 0
    missing_object = 0
    missing_geometry = 0
    warned_old_worker = False
    started = time.monotonic()
    next_report = started + max(0.1, args.report_every_sec)

    print(
        f"[contact-stability] listening {args.lcm_url}/{args.result_channel} "
        f"label={args.label!r} duration={args.duration_sec:.1f}s",
        flush=True,
    )
    if reference is None:
        print(
            "[contact-stability] no ground truth: reports temporal precision only; "
            "use --reference-contact-goal-m X Y Z for absolute error.",
            flush=True,
        )

    def report(now: float, *, final: bool = False) -> None:
        window = [sample for sample in samples if now - sample["receive_time_s"] <= args.window_sec]
        if final:
            window = samples
        if not window:
            print(
                f"[contact-stability] {'final' if final else 'window'} no selected samples "
                f"results={received_results} missing_object={missing_object}",
                flush=True,
            )
            return
        latest = window[-1]
        center_stats = _point_statistics(window, "center")
        contact_stats = _point_statistics(window, "contact")
        point_filter = latest["point_filter"]
        input_count = int(point_filter.get("input_mask_pixels", 0) or 0)
        output_count = int(point_filter.get("output_points", 0) or 0)
        retention = 100.0 * output_count / input_count if input_count else float("nan")
        absolute_mean_cm = None
        if reference is not None:
            contacts = [sample["contact"] for sample in window if sample.get("contact") is not None]
            if contacts:
                errors = np.linalg.norm(np.asarray(contacts) - reference, axis=1)
                absolute_mean_cm = 100.0 * float(np.mean(errors))
        prefix = "final" if final else f"{args.window_sec:.1f}s"
        print(
            f"[contact-stability/{prefix}] n={len(window)} track={latest['track_id']} "
            f"center={_fmt_vector(latest['center'])} contact={_fmt_vector(latest['contact'])} "
            f"center_rms={_format_metric(None if center_stats is None else center_stats['rms_cm'], 'cm')} "
            f"contact_rms={_format_metric(None if contact_stats is None else contact_stats['rms_cm'], 'cm')} "
            f"contact_step_p95={_format_metric(None if contact_stats is None else contact_stats['step_p95_cm'], 'cm')} "
            f"absolute_mean={_format_metric(absolute_mean_cm, 'cm')} "
            f"cloud={input_count}->{output_count}({retention:.1f}%) "
            f"fit_rmse={_format_metric(_median(window, 'fit_rmse_m', 1000.0), 'mm')} "
            f"axis_rms={_format_metric(_axis_jitter_deg(window), 'deg')} "
            f"radius={_format_metric(_median(window, 'radius_m', 1000.0), 'mm')} "
            f"stale={latest['surface_stale_frames']}",
            flush=True,
        )

    def on_result(_channel: str, data: bytes) -> None:
        nonlocal locked_track_id, received_results, missing_object, missing_geometry
        nonlocal warned_old_worker
        received_results += 1
        try:
            payload = unpack_result(data)
        except Exception as error:
            print(f"[contact-stability] decode error: {error}", flush=True)
            return
        objects = payload.get("objects")
        if not isinstance(objects, list):
            missing_object += 1
            return
        candidates = [
            obj
            for obj in objects
            if isinstance(obj, dict)
            and (not args.label or str(obj.get("label", "")).lower() == args.label.lower())
        ]
        if locked_track_id is not None:
            candidates = [obj for obj in candidates if obj.get("track_id") == locked_track_id]
        if not candidates:
            missing_object += 1
            return
        selected = max(candidates, key=lambda obj: float(obj.get("score", 0.0)))
        if locked_track_id is None:
            locked_track_id = str(selected.get("track_id"))
            print(f"[contact-stability] locked track_id={locked_track_id}", flush=True)

        center = _vector(selected.get("center_goal_m"))
        contact = _vector(selected.get("target_contact_goal_m"))
        surface = selected.get("surface_model") or {}
        axis = _vector(surface.get("axis_goal"))
        point_filter = selected.get("point_filter")
        if not isinstance(point_filter, dict):
            point_filter = {}
            if not warned_old_worker:
                warned_old_worker = True
                print(
                    "[contact-stability] WARNING: point_filter telemetry is missing; "
                    "restart the updated remote YOLO11 worker.",
                    flush=True,
                )
        if center is None or contact is None:
            missing_geometry += 1
        sample = {
            "receive_time_s": time.monotonic(),
            "sequence": int(payload.get("sequence", -1)),
            "source_rgb_timestamp_us": int(payload.get("source_rgb_timestamp_us", 0)),
            "track_id": str(selected.get("track_id")),
            "label": str(selected.get("label")),
            "score": float(selected.get("score", 0.0)),
            "median_depth_m": selected.get("median_depth_m"),
            "center": center,
            "contact": contact,
            "axis": axis,
            "fit_rmse_m": surface.get("fit_rmse_m"),
            "inlier_ratio": surface.get("inlier_ratio"),
            "angular_coverage": surface.get("angular_coverage"),
            "radius_m": surface.get("radius_m"),
            "height_m": surface.get("height_m"),
            "surface_updates": surface.get("updates"),
            "surface_stale_frames": int(surface.get("stale_frames", -1)),
            "point_filter": point_filter,
        }
        samples.append(sample)
        if args.print_every_frame:
            print(
                f"[contact-stability/frame] seq={sample['sequence']} "
                f"center={_fmt_vector(center)} contact={_fmt_vector(contact)} "
                f"filter={point_filter} surface={surface}",
                flush=True,
            )

    bus.subscribe(args.result_channel, on_result)
    deadline = started + args.duration_sec
    while time.monotonic() < deadline:
        bus.handle_timeout(100)
        now = time.monotonic()
        if now >= next_report:
            report(now)
            next_report = now + max(0.1, args.report_every_sec)
    report(time.monotonic(), final=True)

    if args.csv_out is not None and samples:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        rows = [_csv_row(sample) for sample in samples]
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with args.csv_out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[contact-stability] CSV saved: {args.csv_out}", flush=True)

    contact_stats = _point_statistics(samples, "contact")
    enough = len(samples) >= max(1, args.min_samples)
    stable = (
        contact_stats is not None
        and contact_stats["rms_cm"] <= args.max_rms_jitter_cm
    )
    accurate = True
    absolute_mean_cm = None
    if reference is not None:
        contacts = [sample["contact"] for sample in samples if sample.get("contact") is not None]
        if contacts:
            absolute_mean_cm = 100.0 * float(
                np.mean(np.linalg.norm(np.asarray(contacts) - reference, axis=1))
            )
            accurate = absolute_mean_cm <= args.max_mean_error_cm
        else:
            accurate = False
    passed = enough and stable and accurate
    print(
        f"[contact-stability] {'PASS' if passed else 'FAIL'} samples={len(samples)} "
        f"contact_rms={_format_metric(None if contact_stats is None else contact_stats['rms_cm'], 'cm')} "
        f"absolute_mean={_format_metric(absolute_mean_cm, 'cm')} "
        f"missing_object={missing_object} missing_geometry={missing_geometry}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
