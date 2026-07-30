#!/usr/bin/env python3
"""Measure the PICO 3-point pose stream before using it as an EE target.

The stream rows are left wrist, right wrist, and neck. ``vr_*`` is calibrated
to the G1 reference pose; ``vr_*_raw`` is root-relative body tracking before
that calibration. Neither representation contains SMPL root translation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import zmq


HEADER_SIZE = 1280
POINT_NAMES = ("left_wrist", "right_wrist", "neck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--session-dir",
        type=Path,
        help="Recorded session containing session_selected.npz.",
    )
    source.add_argument(
        "--npz",
        type=Path,
        help="Direct path to a recorder session_selected.npz.",
    )
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5559")
    parser.add_argument("--topic", default="pose")
    parser.add_argument(
        "--debug-topic",
        default="vr_pose_debug",
        help="Mode-independent diagnostic topic used when the normal pose topic is idle.",
    )
    parser.add_argument("--duration-sec", type=float, default=15.0)
    parser.add_argument("--print-every-sec", type=float, default=1.0)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args()


def unpack_pose(raw: bytes, topic: str) -> dict[str, np.ndarray]:
    prefix = topic.encode()
    if not raw.startswith(prefix):
        raise ValueError(f"message does not start with {topic!r}")
    offset = len(prefix)
    header_raw = raw[offset : offset + HEADER_SIZE].split(b"\0", 1)[0]
    header = json.loads(header_raw.decode("utf-8"))
    dtype_map = {
        "u8": np.uint8,
        "i32": np.int32,
        "i64": np.int64,
        "f32": np.float32,
        "f64": np.float64,
        "bool": np.bool_,
    }
    cursor = offset + HEADER_SIZE
    fields: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        dtype = np.dtype(dtype_map[field["dtype"]])
        shape = tuple(field["shape"])
        size = int(np.prod(shape)) * dtype.itemsize
        fields[field["name"]] = np.frombuffer(
            raw[cursor : cursor + size], dtype=dtype
        ).reshape(shape).copy()
        cursor += size
    return fields


def _latest_frame(fields: dict[str, np.ndarray]) -> int:
    frame = np.asarray(fields.get("frame_index", [-1]), dtype=np.int64).reshape(-1)
    return int(frame[-1])


def _field_pose(
    fields: dict[str, np.ndarray], position_key: str, orientation_key: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if position_key not in fields or orientation_key not in fields:
        return None
    position = np.asarray(fields[position_key], dtype=np.float64).reshape(3, 3)
    orientation = np.asarray(fields[orientation_key], dtype=np.float64).reshape(3, 4)
    if not np.isfinite(position).all() or not np.isfinite(orientation).all():
        return None
    return position, orientation


def collect_live(args: argparse.Namespace) -> dict[str, np.ndarray]:
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    topics = tuple(dict.fromkeys((args.topic, args.debug_topic)))
    for topic in topics:
        socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
    socket.setsockopt(zmq.RCVHWM, 32)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(
        f"[PICO-VR] listening {args.endpoint}/{{{','.join(topics)}}}", flush=True
    )
    started = time.monotonic()
    last_print = started
    samples: list[tuple] = []
    last_frame = None
    selected_topic: str | None = None
    while time.monotonic() - started < args.duration_sec:
        if socket not in dict(poller.poll(100)):
            continue
        message = socket.recv()
        message_topic = next(
            (topic for topic in topics if message.startswith(topic.encode())), None
        )
        if message_topic is None:
            continue
        if selected_topic is not None and message_topic != selected_topic:
            continue
        fields = unpack_pose(message, message_topic)
        calibrated = _field_pose(fields, "vr_position", "vr_orientation")
        if calibrated is None:
            continue
        if selected_topic is None:
            selected_topic = message_topic
            print(f"[PICO-VR] selected source topic={selected_topic}", flush=True)
        frame = _latest_frame(fields)
        if frame == last_frame:
            continue
        last_frame = frame
        timestamp = float(
            np.asarray(
                fields.get("timestamp_monotonic", [time.monotonic()]),
                dtype=np.float64,
            ).reshape(-1)[-1]
        )
        if not np.isfinite(timestamp) or timestamp <= 0.0:
            timestamp = time.monotonic()
        raw_pose = _field_pose(
            fields, "vr_position_raw", "vr_orientation_raw"
        )
        raw_position = raw_pose[0] if raw_pose else np.full((3, 3), np.nan)
        raw_orientation = raw_pose[1] if raw_pose else np.full((3, 4), np.nan)
        samples.append(
            (
                timestamp,
                frame,
                calibrated[0],
                calibrated[1],
                raw_position,
                raw_orientation,
            )
        )
        now = time.monotonic()
        if now - last_print >= args.print_every_sec:
            left, right, neck = calibrated[0]
            print(
                f"[PICO-VR] topic={message_topic} frame={frame} "
                f"L={left.round(3).tolist()} "
                f"R={right.round(3).tolist()} H={neck.round(3).tolist()}",
                flush=True,
            )
            last_print = now
    socket.close()
    if len(samples) < 2:
        raise RuntimeError(
            "fewer than two unique PICO pose frames received; restart the updated "
            "PICO manager for vr_pose_debug, or switch the legacy manager to POSE mode"
        )
    return _samples_to_arrays(samples)


def _samples_to_arrays(samples: list[tuple]) -> dict[str, np.ndarray]:
    return {
        "time": np.asarray([sample[0] for sample in samples], dtype=np.float64),
        "frame": np.asarray([sample[1] for sample in samples], dtype=np.int64),
        "position": np.stack([sample[2] for sample in samples]),
        "orientation": np.stack([sample[3] for sample in samples]),
        "position_raw": np.stack([sample[4] for sample in samples]),
        "orientation_raw": np.stack([sample[5] for sample in samples]),
    }


def load_recorded(args: argparse.Namespace) -> dict[str, np.ndarray]:
    npz_path = args.npz
    if npz_path is None:
        npz_path = args.session_dir / "session_selected.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        position = np.asarray(data["pose_vr_position"], dtype=np.float64).reshape(-1, 3, 3)
        orientation = np.asarray(data["pose_vr_orientation"], dtype=np.float64).reshape(-1, 3, 4)
        frame_window = np.asarray(data["pose_frame_index"], dtype=np.int64)
        frame = frame_window.reshape(frame_window.shape[0], -1)[:, -1]
        timestamp = np.asarray(data["record_monotonic_time"], dtype=np.float64).reshape(-1)
        if "pose_vr_position_raw" in data:
            raw_position = np.asarray(
                data["pose_vr_position_raw"], dtype=np.float64
            ).reshape(-1, 3, 3)
            raw_orientation = np.asarray(
                data["pose_vr_orientation_raw"], dtype=np.float64
            ).reshape(-1, 3, 4)
        else:
            raw_position = np.full_like(position, np.nan)
            raw_orientation = np.full_like(orientation, np.nan)

    # The recorder runs faster than PICO. Analyze each source frame once.
    unique = np.r_[True, frame[1:] != frame[:-1]]
    arrays = {
        "time": timestamp[unique],
        "frame": frame[unique],
        "position": position[unique],
        "orientation": orientation[unique],
        "position_raw": raw_position[unique],
        "orientation_raw": raw_orientation[unique],
    }
    print(
        f"[PICO-VR] loaded {npz_path}: records={len(frame)}, "
        f"unique_pose_frames={int(np.count_nonzero(unique))}"
    )
    return arrays


def quaternion_step_angle(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1.0e-12)
    dot = np.sum(q[1:] * q[:-1], axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def quantiles(values: np.ndarray, scale: float = 1.0) -> str:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)] * scale
    if not finite.size:
        return "n/a"
    q50, q95, q99, maximum = np.percentile(finite, [50, 95, 99, 100])
    return f"{q50:.3f}/{q95:.3f}/{q99:.3f}/{maximum:.3f}"


def best_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    x = source.reshape(-1, 3)
    y = target.reshape(-1, 3)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[valid]
    y = y[valid]
    if len(x) < 6:
        return float("nan"), float("nan"), float("nan")
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(singular.sum() / max(np.sum(x * x), 1.0e-12))
    residual = scale * x @ rotation - y
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return scale, rmse, float(np.linalg.det(rotation))


def write_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["time", "frame"]
    for suffix in ("", "_raw"):
        for point in POINT_NAMES:
            headers.extend(f"{point}_{axis}{suffix}" for axis in "xyz")
            headers.extend(f"{point}_q{axis}{suffix}" for axis in "wxyz")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(len(arrays["time"])):
            row: list[float | int] = [arrays["time"][index], int(arrays["frame"][index])]
            for position_key, orientation_key in (
                ("position", "orientation"),
                ("position_raw", "orientation_raw"),
            ):
                for point_index in range(3):
                    row.extend(arrays[position_key][index, point_index].tolist())
                    row.extend(arrays[orientation_key][index, point_index].tolist())
            writer.writerow(row)
    print(f"[PICO-VR] CSV saved: {path}")


def report(arrays: dict[str, np.ndarray]) -> None:
    timestamp = arrays["time"]
    dt = np.diff(timestamp)
    valid_dt = np.isfinite(dt) & (dt > 1.0e-4) & (dt < 1.0)
    if not np.any(valid_dt):
        raise RuntimeError("timestamps do not provide a valid sampling interval")
    duration = float(timestamp[-1] - timestamp[0])
    hz = (len(timestamp) - 1) / max(duration, 1.0e-9)
    print(
        f"[PICO-VR] samples={len(timestamp)} duration={duration:.2f}s "
        f"source_rate={hz:.2f}Hz "
        f"dt_ms(p50/p95/p99/max)={quantiles(dt[valid_dt], 1000.0)}"
    )
    position = arrays["position"]
    orientation = arrays["orientation"]
    for point_index, name in enumerate(POINT_NAMES):
        step = np.linalg.norm(np.diff(position[:, point_index], axis=0), axis=1)
        speed = step[valid_dt] / dt[valid_dt]
        angle = quaternion_step_angle(orientation[:, point_index])
        angular_speed = angle[valid_dt] / dt[valid_dt]
        qnorm_error = np.abs(np.linalg.norm(orientation[:, point_index], axis=1) - 1.0)
        print(
            f"[PICO-VR] {name:11s} step_cm(p50/p95/p99/max)={quantiles(step, 100.0)} "
            f"speed_m_s={quantiles(speed)} angle_deg={quantiles(angle, 180.0 / np.pi)} "
            f"omega_rad_s={quantiles(angular_speed)} qnorm_max={np.nanmax(qnorm_error):.2e}"
        )

    relative = position[:, :2] - position[:, 2:3]
    for side, name in enumerate(("left", "right")):
        step = np.linalg.norm(np.diff(relative[:, side], axis=0), axis=1)
        jumps = int(np.count_nonzero(step > 0.05))
        print(
            f"[PICO-VR] {name} wrist-minus-neck step_cm={quantiles(step, 100.0)} "
            f">5cm={jumps}/{len(step)}"
        )

    raw_position = arrays["position_raw"]
    if np.isfinite(raw_position).all():
        raw_relative = raw_position[:, :2] - raw_position[:, 2:3]
        scale, rmse, determinant = best_similarity(raw_relative, relative)
        print(
            "[PICO-VR] raw->calibrated wrist-minus-neck best rigid map: "
            f"scale={scale:.4f} rmse={100.0 * rmse:.2f}cm det={determinant:.3f}"
        )
    else:
        print(
            "[PICO-VR] raw 3-point fields unavailable in this recording. "
            "Restart the updated pico manager to compare pre/post calibration values."
        )
    print(
        "[PICO-VR] Interpretation: use wrist-minus-neck displacement for body-relative "
        "position targets; use vr_orientation only after checking quaternion steps and "
        "the raw-to-calibrated map. These fields are body-tracking-derived, not a direct "
        "PICO controller API stream."
    )


def main() -> int:
    args = parse_args()
    arrays = (
        load_recorded(args)
        if args.session_dir is not None or args.npz is not None
        else collect_live(args)
    )
    report(arrays)
    if args.csv_out is not None:
        write_csv(args.csv_out, arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
