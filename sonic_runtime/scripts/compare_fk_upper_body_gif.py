#!/usr/bin/env python3
"""Create upper-body FK comparison GIFs from fk_error_analysis outputs.

This script reads segment_XX_fk_positions_{mode}.csv files produced by
fk_error_analysis.py and overlays:

  1. GMR reference
  2. SONIC measured FK
  3. SONIC+QP measured FK

By default, all segment CSVs in each directory are concatenated first, then
resampled onto one shared time grid. This avoids comparing segment_00 from one
run against segment_00 from another run when their segmentation boundaries are
different.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


UPPER_BODIES = [
    "pelvis",
    "torso_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

UPPER_BONES = [
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]

WRISTS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


def load_positions_csv(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return times[F], body_names[B], gmr[F,B,3], q[F,B,3]."""
    times: list[float] = []
    body_names: list[str] = []
    gmr_frames: list[list[np.ndarray]] = []
    q_frames: list[list[np.ndarray]] = []
    current_time: float | None = None

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_sec"])
            if current_time is None or t != current_time:
                times.append(t)
                gmr_frames.append([])
                q_frames.append([])
                current_time = t
            if len(times) == 1:
                body_names.append(row["body"])
            gmr_frames[-1].append(np.array(
                [float(row["gmr_x"]), float(row["gmr_y"]), float(row["gmr_z"])],
                dtype=np.float64,
            ))
            q_frames[-1].append(np.array(
                [float(row["q_x"]), float(row["q_y"]), float(row["q_z"])],
                dtype=np.float64,
            ))

    if not times:
        return np.empty(0), [], np.empty((0, 0, 3)), np.empty((0, 0, 3))
    return np.asarray(times), body_names, np.asarray(gmr_frames), np.asarray(q_frames)


def select_bodies(
    body_names: list[str],
    *arrays: np.ndarray,
) -> tuple[list[str], list[np.ndarray]]:
    body_to_idx = {name: i for i, name in enumerate(body_names)}
    selected = [name for name in UPPER_BODIES if name in body_to_idx]
    indices = [body_to_idx[name] for name in selected]
    return selected, [arr[:, indices, :] for arr in arrays]


def common_segment_ids(*dirs: Path, mode: str) -> list[int]:
    segment_sets: list[set[int]] = []
    for directory in dirs:
        ids = set()
        for path in directory.glob(f"segment_*_fk_positions_{mode}.csv"):
            parts = path.name.split("_")
            if len(parts) >= 2 and parts[0] == "segment":
                try:
                    ids.add(int(parts[1]))
                except ValueError:
                    pass
        segment_sets.append(ids)
    if not segment_sets:
        return []
    return sorted(set.intersection(*segment_sets))


def segment_csvs(directory: Path, mode: str) -> list[Path]:
    def segment_id(path: Path) -> int:
        try:
            return int(path.name.split("_")[1])
        except (IndexError, ValueError):
            return 10**9

    return sorted(
        directory.glob(f"segment_*_fk_positions_{mode}.csv"),
        key=segment_id,
    )


def infer_dt(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.02
    diffs = np.diff(times)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return 0.02
    return float(np.median(diffs))


def load_merged_sequence(directory: Path, mode: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Concatenate all segment CSVs and reset time to a continuous 0-based axis."""
    paths = segment_csvs(directory, mode)
    if not paths:
        raise FileNotFoundError(f"no segment_*_fk_positions_{mode}.csv in {directory}")

    merged_times: list[np.ndarray] = []
    merged_gmr: list[np.ndarray] = []
    merged_q: list[np.ndarray] = []
    body_names: list[str] | None = None
    next_start = 0.0

    for path in paths:
        times, bodies, gmr, q = load_positions_csv(path)
        if len(times) == 0:
            continue
        if body_names is None:
            body_names = bodies
        elif bodies != body_names:
            raise ValueError(f"body order/name mismatch in {path}")

        rel_times = times - times[0]
        dt = infer_dt(times)
        merged_times.append(next_start + rel_times)
        merged_gmr.append(gmr)
        merged_q.append(q)
        next_start += float(rel_times[-1]) + dt

    if body_names is None or not merged_times:
        return np.empty(0), [], np.empty((0, 0, 3)), np.empty((0, 0, 3))
    return (
        np.concatenate(merged_times, axis=0),
        body_names,
        np.concatenate(merged_gmr, axis=0),
        np.concatenate(merged_q, axis=0),
    )


def resample_positions(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate [F,B,3] positions onto grid times."""
    out = np.empty((len(grid), values.shape[1], 3), dtype=np.float64)
    for b in range(values.shape[1]):
        for c in range(3):
            out[:, b, c] = np.interp(grid, times, values[:, b, c])
    return out


def shift_overlap_duration(
    a_times: np.ndarray,
    b_times: np.ndarray,
    shift: float,
) -> tuple[float, float]:
    """Return t range where a(t) and b(t + shift) are both valid."""
    start = max(float(a_times[0]), float(b_times[0]) - shift)
    end = min(float(a_times[-1]), float(b_times[-1]) - shift)
    return start, end


def find_best_gmr_time_shift(
    sonic_times: np.ndarray,
    sonic_gmr: np.ndarray,
    qp_times: np.ndarray,
    qp_gmr: np.ndarray,
    body_names: list[str],
    max_shift_sec: float,
    step_sec: float,
    sample_fps: float,
) -> tuple[float, float]:
    """Find shift minimizing RMS between sonic_gmr(t) and qp_gmr(t + shift)."""
    selected_names, (sonic_upper, qp_upper) = select_bodies(
        body_names, sonic_gmr, qp_gmr
    )
    if not selected_names:
        raise ValueError("no upper-body bodies available for alignment")

    step = max(1e-6, float(step_sec))
    shifts = np.arange(-abs(max_shift_sec), abs(max_shift_sec) + step * 0.5, step)
    dt = 1.0 / max(1.0, float(sample_fps))
    best_shift = 0.0
    best_rms = float("inf")

    for shift in shifts:
        start, end = shift_overlap_duration(sonic_times, qp_times, float(shift))
        if end <= start + dt:
            continue
        grid = np.arange(start, end, dt, dtype=np.float64)
        if len(grid) < 2:
            continue
        sonic_sample = resample_positions(sonic_times, sonic_upper, grid)
        qp_sample = resample_positions(qp_times, qp_upper, grid + shift)
        rms = float(np.sqrt(np.mean((sonic_sample - qp_sample) ** 2)))
        if rms < best_rms:
            best_rms = rms
            best_shift = float(shift)

    if not np.isfinite(best_rms):
        raise ValueError("failed to find valid overlap for GMR time alignment")
    return best_shift, best_rms


def axis_limits(*arrays: np.ndarray, margin: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([arr.reshape(-1, 3) for arr in arrays if arr.size], axis=0)
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    center = (lo + hi) * 0.5
    half = max((hi - lo).max() * 0.5, 0.05) + margin
    return center - half, center + half


def set_axes_equal(ax, lo: np.ndarray, hi: np.ndarray) -> None:
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


def make_comparison_gif(
    sonic_csv: Path,
    qp_csv: Path,
    gmr_csv: Path,
    output_path: Path,
    fps: int,
    duration_sec: float | None,
    stride: int,
    elev: float,
    azim: float,
    max_frames: int | None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sonic_times, sonic_bodies, sonic_gmr, sonic_q = load_positions_csv(sonic_csv)
    qp_times, qp_bodies, _qp_gmr, qp_q = load_positions_csv(qp_csv)
    gmr_times, gmr_bodies, gmr_ref, _gmr_q = load_positions_csv(gmr_csv)

    if len(sonic_times) == 0 or len(qp_times) == 0 or len(gmr_times) == 0:
        print(f"[skip] empty csv: {sonic_csv.name}, {qp_csv.name}, {gmr_csv.name}")
        return
    if sonic_bodies != qp_bodies or sonic_bodies != gmr_bodies:
        raise ValueError("body order/name mismatch between FK CSV files")

    body_names, (gmr_ref, sonic_q, qp_q) = select_bodies(
        sonic_bodies, gmr_ref, sonic_q, qp_q
    )
    n = min(len(gmr_ref), len(sonic_q), len(qp_q))
    frame_indices = np.arange(0, n, max(1, stride), dtype=int)
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if len(frame_indices) == 0:
        print(f"[skip] no frames after stride: {sonic_csv.name}")
        return

    gmr_ref = gmr_ref[frame_indices]
    sonic_q = sonic_q[frame_indices]
    qp_q = qp_q[frame_indices]
    sonic_times = sonic_times[:n][frame_indices]

    if duration_sec is not None and duration_sec > 0:
        save_fps = max(1, int(round(len(frame_indices) / duration_sec)))
    else:
        save_fps = max(1, fps)

    lo, hi = axis_limits(gmr_ref, sonic_q, qp_q)
    body_to_idx = {name: i for i, name in enumerate(body_names)}
    bones = [
        (body_to_idx[parent], body_to_idx[child])
        for parent, child in UPPER_BONES
        if parent in body_to_idx and child in body_to_idx
    ]
    wrist_indices = [body_to_idx[name] for name in WRISTS if name in body_to_idx]

    colors = {
        "gmr": "#111111",
        "sonic": "#ff7f0e",
        "qp": "#1f77b4",
        "sonic_err": "#ffbb78",
        "qp_err": "#6baed6",
    }

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, lo, hi)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)

    scatters = {
        "gmr": ax.scatter([], [], [], c=colors["gmr"], s=14, marker="o",
                          depthshade=False, label="GMR ref"),
        "sonic": ax.scatter([], [], [], c=colors["sonic"], s=12, marker="^",
                            depthshade=False, label="SONIC"),
        "qp": ax.scatter([], [], [], c=colors["qp"], s=12, marker="s",
                         depthshade=False, label="SONIC+QP"),
    }
    lines = {
        key: [
            ax.plot([], [], [], color=colors[key], linewidth=0.9, alpha=0.9)[0]
            for _ in bones
        ]
        for key in ("gmr", "sonic", "qp")
    }
    error_lines = {
        "sonic": [
            ax.plot([], [], [], color=colors["sonic_err"], linewidth=0.7,
                    linestyle=":", alpha=0.85)[0]
            for _ in wrist_indices
        ],
        "qp": [
            ax.plot([], [], [], color=colors["qp_err"], linewidth=0.7,
                    linestyle=":", alpha=0.85)[0]
            for _ in wrist_indices
        ],
    }

    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")
    err_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=9)

    def set_points(scatter, points: np.ndarray) -> None:
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    def set_bones(line_list, points: np.ndarray) -> None:
        for line, (parent, child) in zip(line_list, bones):
            segment = points[[parent, child]]
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])

    def set_error_lines(line_list, ref: np.ndarray, meas: np.ndarray) -> None:
        for line, idx in zip(line_list, wrist_indices):
            segment = np.vstack([ref[idx], meas[idx]])
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])

    seg_id = sonic_csv.name.split("_")[1]
    mode = sonic_csv.stem.split("_fk_positions_")[-1]

    def update(i: int):
        points = {"gmr": gmr_ref[i], "sonic": sonic_q[i], "qp": qp_q[i]}
        for key, arr in points.items():
            set_points(scatters[key], arr)
            set_bones(lines[key], arr)
        set_error_lines(error_lines["sonic"], gmr_ref[i], sonic_q[i])
        set_error_lines(error_lines["qp"], gmr_ref[i], qp_q[i])

        sonic_wrist_err = np.mean([
            np.linalg.norm(sonic_q[i, idx] - gmr_ref[i, idx])
            for idx in wrist_indices
        ]) if wrist_indices else 0.0
        qp_wrist_err = np.mean([
            np.linalg.norm(qp_q[i, idx] - gmr_ref[i, idx])
            for idx in wrist_indices
        ]) if wrist_indices else 0.0
        title.set_text(
            f"Upper-body FK comparison | seg {seg_id} [{mode}] | "
            f"frame {i + 1}/{len(frame_indices)}"
        )
        err_text.set_text(
            f"mean wrist error: SONIC {sonic_wrist_err * 100:.1f} cm | "
            f"SONIC+QP {qp_wrist_err * 100:.1f} cm"
        )
        return (
            *scatters.values(),
            title,
            err_text,
            *lines["gmr"],
            *lines["sonic"],
            *lines["qp"],
            *error_lines["sonic"],
            *error_lines["qp"],
        )

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / save_fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=PillowWriter(fps=save_fps))
    plt.close(fig)
    print(
        f"[gif] saved {output_path} "
        f"frames={len(frame_indices)} fps={save_fps} "
        f"source_frames={n} stride={stride} "
        f"sonic_time={sonic_times[0]:.2f}~{sonic_times[-1]:.2f}s"
    )


def make_array_comparison_gif(
    body_names: list[str],
    common_times: np.ndarray,
    gmr_ref: np.ndarray,
    sonic_q: np.ndarray,
    qp_q: np.ndarray,
    output_path: Path,
    fps: int,
    title_prefix: str,
    elev: float,
    azim: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if len(common_times) == 0:
        print(f"[skip] no frames for {output_path.name}")
        return

    body_names, (gmr_ref, sonic_q, qp_q) = select_bodies(
        body_names, gmr_ref, sonic_q, qp_q
    )
    lo, hi = axis_limits(gmr_ref, sonic_q, qp_q)
    body_to_idx = {name: i for i, name in enumerate(body_names)}
    bones = [
        (body_to_idx[parent], body_to_idx[child])
        for parent, child in UPPER_BONES
        if parent in body_to_idx and child in body_to_idx
    ]
    wrist_indices = [body_to_idx[name] for name in WRISTS if name in body_to_idx]

    colors = {
        "gmr": "#111111",
        "sonic": "#ff7f0e",
        "qp": "#1f77b4",
        "sonic_err": "#ffbb78",
        "qp_err": "#6baed6",
    }

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, lo, hi)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)

    scatters = {
        "gmr": ax.scatter([], [], [], c=colors["gmr"], s=14, marker="o",
                          depthshade=False, label="GMR ref"),
        "sonic": ax.scatter([], [], [], c=colors["sonic"], s=12, marker="^",
                            depthshade=False, label="SONIC"),
        "qp": ax.scatter([], [], [], c=colors["qp"], s=12, marker="s",
                         depthshade=False, label="SONIC+QP"),
    }
    lines = {
        key: [
            ax.plot([], [], [], color=colors[key], linewidth=0.85, alpha=0.9)[0]
            for _ in bones
        ]
        for key in ("gmr", "sonic", "qp")
    }
    error_lines = {
        "sonic": [
            ax.plot([], [], [], color=colors["sonic_err"], linewidth=0.65,
                    linestyle=":", alpha=0.85)[0]
            for _ in wrist_indices
        ],
        "qp": [
            ax.plot([], [], [], color=colors["qp_err"], linewidth=0.65,
                    linestyle=":", alpha=0.85)[0]
            for _ in wrist_indices
        ],
    }

    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")
    err_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=9)

    def set_points(scatter, points: np.ndarray) -> None:
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    def set_bones(line_list, points: np.ndarray) -> None:
        for line, (parent, child) in zip(line_list, bones):
            segment = points[[parent, child]]
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])

    def set_error_lines(line_list, ref: np.ndarray, meas: np.ndarray) -> None:
        for line, idx in zip(line_list, wrist_indices):
            segment = np.vstack([ref[idx], meas[idx]])
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])

    def update(i: int):
        points = {"gmr": gmr_ref[i], "sonic": sonic_q[i], "qp": qp_q[i]}
        for key, arr in points.items():
            set_points(scatters[key], arr)
            set_bones(lines[key], arr)
        set_error_lines(error_lines["sonic"], gmr_ref[i], sonic_q[i])
        set_error_lines(error_lines["qp"], gmr_ref[i], qp_q[i])

        sonic_wrist_err = np.mean([
            np.linalg.norm(sonic_q[i, idx] - gmr_ref[i, idx])
            for idx in wrist_indices
        ]) if wrist_indices else 0.0
        qp_wrist_err = np.mean([
            np.linalg.norm(qp_q[i, idx] - gmr_ref[i, idx])
            for idx in wrist_indices
        ]) if wrist_indices else 0.0
        title.set_text(
            f"{title_prefix} | t={common_times[i]:.2f}s | frame {i + 1}/{len(common_times)}"
        )
        err_text.set_text(
            f"mean wrist error: SONIC {sonic_wrist_err * 100:.1f} cm | "
            f"SONIC+QP {qp_wrist_err * 100:.1f} cm"
        )
        return (
            *scatters.values(),
            title,
            err_text,
            *lines["gmr"],
            *lines["sonic"],
            *lines["qp"],
            *error_lines["sonic"],
            *error_lines["qp"],
        )

    anim = FuncAnimation(fig, update, frames=len(common_times), interval=1000 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[gif] saved {output_path} frames={len(common_times)} fps={fps}")


def make_merged_chunk_gifs(args: argparse.Namespace, gmr_dir: Path) -> int:
    sonic_times, sonic_bodies, sonic_gmr, sonic_q = load_merged_sequence(args.sonic_dir, args.mode)
    qp_times, qp_bodies, qp_gmr, qp_q = load_merged_sequence(args.qp_dir, args.mode)
    gmr_times, gmr_bodies, gmr_ref, _gmr_q = load_merged_sequence(gmr_dir, args.mode)
    if sonic_bodies != qp_bodies or sonic_bodies != gmr_bodies:
        raise ValueError("body order/name mismatch between merged FK sequences")

    if args.auto_align_gmr:
        shift, rms = find_best_gmr_time_shift(
            sonic_times=sonic_times,
            sonic_gmr=sonic_gmr,
            qp_times=qp_times,
            qp_gmr=qp_gmr,
            body_names=sonic_bodies,
            max_shift_sec=args.align_max_shift_sec,
            step_sec=args.align_step_sec,
            sample_fps=args.align_sample_fps,
        )
        args.time_shift_qp_sec = shift
        print(
            f"[align] best qp time shift={shift:+.3f}s "
            f"(min upper-body GMR RMS={rms * 100:.2f} cm)"
        )
    else:
        print(f"[align] using manual qp time shift={args.time_shift_qp_sec:+.3f}s")

    # Definition: compare SONIC/GMR at t against QP at t + shift.
    shift = float(args.time_shift_qp_sec)
    start_sonic_qp, end_sonic_qp = shift_overlap_duration(sonic_times, qp_times, shift)
    start_gmr_qp, end_gmr_qp = shift_overlap_duration(gmr_times, qp_times, shift)
    common_start = max(start_sonic_qp, start_gmr_qp)
    common_end = min(end_sonic_qp, end_gmr_qp)
    common_duration = common_end - common_start
    if common_duration <= 0:
        raise ValueError("common merged duration is empty")

    sample_fps = max(1, int(round(args.sample_fps)))
    chunk_sec = max(1e-6, float(args.chunk_sec))
    made = 0
    chunk_start = common_start
    chunk_idx = 0
    while chunk_start < common_end - 1e-9:
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        chunk_end = min(common_end, chunk_start + chunk_sec)
        grid = np.arange(chunk_start, chunk_end, 1.0 / sample_fps, dtype=np.float64)
        if len(grid) == 0:
            break
        if args.max_frames is not None:
            grid = grid[:args.max_frames]
        if len(grid) == 0:
            break

        sonic_chunk = resample_positions(sonic_times, sonic_q, grid)
        qp_chunk = resample_positions(qp_times, qp_q, grid + shift)
        gmr_chunk = resample_positions(gmr_times, gmr_ref, grid)

        playback_sec = args.duration_sec if args.duration_sec and args.duration_sec > 0 else (grid[-1] - grid[0] + 1.0 / sample_fps)
        gif_fps = max(1, int(round(len(grid) / playback_sec)))
        out = args.output_dir / (
            f"chunk_{chunk_idx:03d}_{int(round(chunk_start)):04d}-"
            f"{int(round(chunk_end)):04d}s_{args.mode}_upper_compare.gif"
        )
        make_array_comparison_gif(
            body_names=sonic_bodies,
            common_times=grid,
            gmr_ref=gmr_chunk,
            sonic_q=sonic_chunk,
            qp_q=qp_chunk,
            output_path=out,
            fps=gif_fps,
            title_prefix=f"Upper-body FK comparison [{args.mode}] chunk {chunk_idx:03d}",
            elev=args.elev,
            azim=args.azim,
        )
        made += 1
        chunk_idx += 1
        chunk_start += chunk_sec

    print(
        f"[merged] overlap={common_start:.2f}~{common_end:.2f}s "
        f"duration={common_duration:.2f}s sample_fps={sample_fps} "
        f"chunk_sec={chunk_sec:.2f} shift={shift:+.3f}s generated={made}"
    )
    return made


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay upper-body FK GIFs for GMR, SONIC, and SONIC+QP."
    )
    parser.add_argument("--sonic-dir", type=Path, required=True)
    parser.add_argument("--qp-dir", type=Path, required=True)
    parser.add_argument(
        "--gmr-dir", type=Path, default=None,
        help="Directory to read GMR reference from. Defaults to --sonic-dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", default="base_local",
                        choices=["base_local", "root_oriented"])
    parser.add_argument("--by-segment", action="store_true",
                        help="Use old segment-id matching instead of merged time chunks.")
    parser.add_argument("--segments", nargs="+", type=int, default=None,
                        help="Segment ids for --by-segment mode.")
    parser.add_argument("--chunk-sec", type=float, default=10.0,
                        help="Merged-mode chunk duration in seconds. Default: 10.")
    parser.add_argument("--sample-fps", type=float, default=50.0,
                        help="Merged-mode common resampling FPS. Default: 50.")
    parser.add_argument("--time-shift-qp-sec", type=float, default=0.0,
                        help="Manual QP time shift for merged mode. QP is sampled at t + shift.")
    parser.add_argument("--auto-align-gmr", dest="auto_align_gmr",
                        action="store_true", default=True,
                        help="Find QP time shift by minimizing upper-body GMR RMS. Default: on.")
    parser.add_argument("--no-auto-align-gmr", dest="auto_align_gmr",
                        action="store_false",
                        help="Disable automatic GMR-based time alignment.")
    parser.add_argument("--align-max-shift-sec", type=float, default=20.0,
                        help="Search range for auto GMR alignment. Default: +/-20s.")
    parser.add_argument("--align-step-sec", type=float, default=0.02,
                        help="Search step for auto GMR alignment. Default: 0.02s.")
    parser.add_argument("--align-sample-fps", type=float, default=25.0,
                        help="Sampling FPS used for shift search. Default: 25.")
    parser.add_argument("--fps", type=int, default=15,
                        help="GIF fps for --by-segment mode when --duration-sec is not set.")
    parser.add_argument("--duration-sec", type=float, default=None,
                        help="Playback duration per GIF. In merged mode default is real-time chunk duration.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride. Default 1 keeps every frame.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional debug cap after stride.")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Optional debug cap for merged-mode chunks.")
    parser.add_argument("--elev", type=float, default=15.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gmr_dir = args.gmr_dir or args.sonic_dir

    if not args.by_segment:
        print(f"[compare] sonic={args.sonic_dir}")
        print(f"[compare] qp={args.qp_dir}")
        print(f"[compare] gmr={gmr_dir}")
        print(
            f"[compare] merged mode={args.mode} chunk_sec={args.chunk_sec} "
            f"sample_fps={args.sample_fps}"
        )
        made = make_merged_chunk_gifs(args, gmr_dir)
        if made == 0:
            raise SystemExit("[error] no GIFs generated")
        return 0

    segments = args.segments or common_segment_ids(args.sonic_dir, args.qp_dir, gmr_dir, mode=args.mode)
    if not segments:
        raise SystemExit("[error] no common segment CSV files found")

    print(f"[compare] sonic={args.sonic_dir}")
    print(f"[compare] qp={args.qp_dir}")
    print(f"[compare] gmr={gmr_dir}")
    print(f"[compare] mode={args.mode} segments={segments}")

    made = 0
    for seg_id in segments:
        name = f"segment_{seg_id:02d}_fk_positions_{args.mode}.csv"
        sonic_csv = args.sonic_dir / name
        qp_csv = args.qp_dir / name
        gmr_csv = gmr_dir / name
        missing = [p for p in (sonic_csv, qp_csv, gmr_csv) if not p.exists()]
        if missing:
            print("[skip] missing " + ", ".join(str(p) for p in missing))
            continue
        output = args.output_dir / f"segment_{seg_id:02d}_{args.mode}_upper_compare.gif"
        make_comparison_gif(
            sonic_csv=sonic_csv,
            qp_csv=qp_csv,
            gmr_csv=gmr_csv,
            output_path=output,
            fps=args.fps,
            duration_sec=args.duration_sec,
            stride=args.stride,
            elev=args.elev,
            azim=args.azim,
            max_frames=args.max_frames,
        )
        made += 1
    if made == 0:
        raise SystemExit("[error] no GIFs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
