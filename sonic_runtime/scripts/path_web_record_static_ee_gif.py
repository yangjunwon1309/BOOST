#!/usr/bin/env python3
"""Create static-EE target GIFs from PATH web-view recorder outputs.

The recorder in scripts/g1_mujoco_gmr_web_view.py stores q_measured,
pd_q_corrected, QP-internal wrist positions, and terminal EE targets.  This
visualizer keeps the upper-body FK style of compare_fk_upper_body_gif.py, but
replaces the GMR overlay with a red static EE target and target-error text.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

LEFT_CHAIN = {"torso_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link"}
RIGHT_CHAIN = {"torso_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link"}


@dataclass
class RecordData:
    path: Path
    name: str
    elapsed: np.ndarray
    q_measured: np.ndarray
    q_corrected: np.ndarray
    target: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    active_wrist: np.ndarray
    active_error: np.ndarray
    servo_after: np.ndarray
    target_side: np.ndarray
    target_link: list[str]


class UpperBodyFk:
    def __init__(self, model_xml: Path, body_names: Iterable[str] = UPPER_BODIES) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model_xml = model_xml.expanduser().resolve()
        old_cwd = Path.cwd()
        try:
            os.chdir(self.model_xml.parent)
            self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        finally:
            os.chdir(old_cwd)
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_names = list(body_names)
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.body_names
        ]
        missing = [name for name, body_id in zip(self.body_names, self.body_ids) if body_id < 0]
        if missing:
            raise ValueError(f"missing bodies in {self.model_xml}: {missing}")

    def positions(self, q29: np.ndarray) -> np.ndarray:
        q = np.asarray(q29, dtype=np.float64).reshape(-1)
        if q.shape[0] < 29:
            raise ValueError(f"expected q29, got shape {q.shape}")
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[7:36] = q[:29]
        self.mujoco.mj_forward(self.model, self.data)
        pelvis = self.data.xpos[self.pelvis_id].copy() if self.pelvis_id >= 0 else np.zeros(3)
        return np.stack([self.data.xpos[body_id].copy() - pelvis for body_id in self.body_ids], axis=0)


def _stack_columns(data: np.lib.npyio.NpzFile, prefix: str, n: int) -> np.ndarray:
    columns = []
    for i in range(n):
        key = f"{prefix}_{i:02d}"
        if key not in data:
            raise KeyError(f"{key} missing from {data.filename}")
        columns.append(np.asarray(data[key], dtype=np.float64))
    return np.stack(columns, axis=1)


def _object_strings(values: np.ndarray) -> list[str]:
    return [str(v) for v in values.tolist()]


def load_record(path: Path) -> RecordData:
    if path.is_dir():
        npz = path / "path_static_ee_tracking.npz"
    else:
        npz = path
    if not npz.exists():
        raise FileNotFoundError(npz)
    z = np.load(npz, allow_pickle=True)
    elapsed = np.asarray(z["elapsed_sec"], dtype=np.float64)
    elapsed = elapsed - elapsed[0]
    q_measured = _stack_columns(z, "body_q_measured", 29)
    if "pd_q_corrected_00" in z:
        q_corrected = _stack_columns(z, "pd_q_corrected", 29)
    else:
        q_corrected = q_measured.copy()
    target = _stack_columns(z, "qp_input_target_position", 3)
    left_wrist = _stack_columns(z, "ee_qp_left_wrist_position", 3)
    right_wrist = _stack_columns(z, "ee_qp_right_wrist_position", 3)
    if "ee_qp_active_wrist_position_00" in z:
        active_wrist = _stack_columns(z, "ee_qp_active_wrist_position", 3)
    else:
        side = np.asarray(z["ee_qp_terminal_target_side"], dtype=np.float64)
        active_wrist = np.where((side[:, None] == 2.0), right_wrist, left_wrist)
    active_error = np.linalg.norm(active_wrist - target, axis=1)
    if "ee_qp_servo_error_after_m" in z:
        servo_after = np.asarray(z["ee_qp_servo_error_after_m"], dtype=np.float64)
    else:
        servo_after = np.full_like(active_error, np.nan)
    return RecordData(
        path=npz,
        name=npz.parent.name,
        elapsed=elapsed,
        q_measured=q_measured,
        q_corrected=q_corrected,
        target=target,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        active_wrist=active_wrist,
        active_error=active_error,
        servo_after=servo_after,
        target_side=np.asarray(z["ee_qp_terminal_target_side"], dtype=np.float64),
        target_link=_object_strings(z["ee_qp_terminal_target_link"]),
    )


def infer_fps(times: np.ndarray) -> float:
    diffs = np.diff(times)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return 25.0
    return float(1.0 / np.median(diffs))


def resample_array(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(grid, times, values)
    out = np.empty((len(grid), *values.shape[1:]), dtype=np.float64)
    flat = values.reshape(values.shape[0], -1)
    out_flat = out.reshape(len(grid), -1)
    for i in range(flat.shape[1]):
        out_flat[:, i] = np.interp(grid, times, flat[:, i])
    return out


def resample_record(record: RecordData, sample_fps: float) -> RecordData:
    if sample_fps <= 0:
        return record
    end = float(record.elapsed[-1])
    grid = np.arange(0.0, end + 0.5 / sample_fps, 1.0 / sample_fps, dtype=np.float64)
    side = np.rint(resample_array(record.elapsed, record.target_side, grid))
    link = []
    for t in grid:
        idx = int(np.clip(np.searchsorted(record.elapsed, t), 0, len(record.target_link) - 1))
        link.append(record.target_link[idx])
    return RecordData(
        path=record.path,
        name=record.name,
        elapsed=grid,
        q_measured=resample_array(record.elapsed, record.q_measured, grid),
        q_corrected=resample_array(record.elapsed, record.q_corrected, grid),
        target=resample_array(record.elapsed, record.target, grid),
        left_wrist=resample_array(record.elapsed, record.left_wrist, grid),
        right_wrist=resample_array(record.elapsed, record.right_wrist, grid),
        active_wrist=resample_array(record.elapsed, record.active_wrist, grid),
        active_error=resample_array(record.elapsed, record.active_error, grid),
        servo_after=resample_array(record.elapsed, record.servo_after, grid),
        target_side=side,
        target_link=link,
    )


def axis_limits(*arrays: np.ndarray, margin: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    valid = [arr.reshape(-1, 3) for arr in arrays if arr.size]
    flat = np.concatenate(valid, axis=0)
    flat = flat[np.isfinite(flat).all(axis=1)]
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    center = 0.5 * (lo + hi)
    half = max(0.5 * float((hi - lo).max()), 0.06) + margin
    return center - half, center + half


def set_axes_equal(ax, lo: np.ndarray, hi: np.ndarray) -> None:
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


def compute_fk_sequence(fk: UpperBodyFk, q_seq: np.ndarray) -> np.ndarray:
    return np.stack([fk.positions(q) for q in q_seq], axis=0)


def override_wrist_positions(points: np.ndarray, record: RecordData) -> np.ndarray:
    out = points.copy()
    body_to_idx = {name: i for i, name in enumerate(UPPER_BODIES)}
    if "left_wrist_yaw_link" in body_to_idx:
        out[:, body_to_idx["left_wrist_yaw_link"], :] = record.left_wrist
    if "right_wrist_yaw_link" in body_to_idx:
        out[:, body_to_idx["right_wrist_yaw_link"], :] = record.right_wrist
    return out


def summarize_record(record: RecordData) -> dict[str, float | str]:
    err_cm = record.active_error * 100.0
    after = record.servo_after * 100.0
    side_mode = int(round(float(np.nanmedian(record.target_side)))) if len(record.target_side) else 0
    return {
        "name": record.name,
        "frames": float(len(record.elapsed)),
        "duration_sec": float(record.elapsed[-1] - record.elapsed[0]) if len(record.elapsed) else 0.0,
        "target_side": {1: "left", 2: "right", 3: "both"}.get(side_mode, "none"),
        "target_link": record.target_link[min(len(record.target_link) - 1, len(record.target_link) // 2)] if record.target_link else "",
        "mean_error_cm": float(np.nanmean(err_cm)),
        "median_error_cm": float(np.nanmedian(err_cm)),
        "p95_error_cm": float(np.nanpercentile(err_cm, 95)),
        "max_error_cm": float(np.nanmax(err_cm)),
        "mean_servo_after_cm": float(np.nanmean(after)),
        "p95_servo_after_cm": float(np.nanpercentile(after, 95)),
    }


def write_summary(rows: list[dict[str, float | str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "frames", "duration_sec", "target_side", "target_link",
        "mean_error_cm", "median_error_cm", "p95_error_cm", "max_error_cm",
        "mean_servo_after_cm", "p95_servo_after_cm",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] saved {output_csv}")


def make_gif(
    record: RecordData,
    measured: np.ndarray,
    corrected: np.ndarray,
    output_path: Path,
    fps: int,
    elev: float,
    azim: float,
    focus_margin: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    body_to_idx = {name: i for i, name in enumerate(UPPER_BODIES)}
    bones = [(body_to_idx[a], body_to_idx[b], a, b) for a, b in UPPER_BONES]
    side_mode = int(round(float(np.nanmedian(record.target_side)))) if len(record.target_side) else 0
    active_chain = LEFT_CHAIN if side_mode == 1 else RIGHT_CHAIN if side_mode == 2 else LEFT_CHAIN | RIGHT_CHAIN
    active_wrist_name = "left_wrist_yaw_link" if side_mode == 1 else "right_wrist_yaw_link"

    active_indices = [body_to_idx[name] for name in active_chain if name in body_to_idx]
    focus_points = [
        measured[:, active_indices, :] if active_indices else measured,
        corrected[:, active_indices, :] if active_indices else corrected,
        record.target[:, None, :],
        record.active_wrist[:, None, :],
    ]
    lo, hi = axis_limits(*focus_points, margin=focus_margin)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, lo, hi)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)

    colors = {
        "measured": "#2563eb",
        "corrected": "#7c3aed",
        "target": "#dc2626",
        "waist": "#111827",
        "error": "#ef4444",
    }
    measured_scatter = ax.scatter([], [], [], c=colors["measured"], s=16, marker="o",
                                  depthshade=False, label="measured FK")
    corrected_scatter = ax.scatter([], [], [], c=colors["corrected"], s=12, marker=".",
                                   depthshade=False, alpha=0.65, label="PD/QP target")
    target_scatter = ax.scatter([], [], [], c=colors["target"], s=60, marker="*",
                                depthshade=False, label="EE target")
    measured_lines = []
    corrected_lines = []
    for parent, child, parent_name, child_name in bones:
        active = parent_name in active_chain and child_name in active_chain
        width = 1.8 if active else 0.8
        alpha = 0.95 if active else 0.35
        measured_lines.append(
            ax.plot([], [], [], color=colors["measured"], linewidth=width, alpha=alpha)[0]
        )
        corrected_lines.append(
            ax.plot([], [], [], color=colors["corrected"], linewidth=max(0.7, width - 0.5),
                    linestyle="--", alpha=0.55 if active else 0.22)[0]
        )
    error_line = ax.plot([], [], [], color=colors["error"], linewidth=1.4,
                         linestyle=":", alpha=0.95)[0]
    waist_marker = ax.scatter([], [], [], c=colors["waist"], s=46, marker="s",
                              depthshade=False, label="waist/torso")

    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")
    err_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=9)

    def set_points(scatter, points: np.ndarray) -> None:
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    def set_line(line, points: np.ndarray) -> None:
        line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])

    active_wrist_idx = body_to_idx.get(active_wrist_name, body_to_idx["left_wrist_yaw_link"])
    torso_idx = body_to_idx.get("torso_link", 0)
    pelvis_idx = body_to_idx.get("pelvis", 0)

    def update(i: int):
        measured_i = measured[i]
        corrected_i = corrected[i]
        set_points(measured_scatter, measured_i)
        set_points(corrected_scatter, corrected_i)
        set_points(target_scatter, record.target[i:i + 1])
        set_points(waist_marker, measured_i[[pelvis_idx, torso_idx]])
        for line, (parent, child, _pn, _cn) in zip(measured_lines, bones):
            set_line(line, measured_i[[parent, child]])
        for line, (parent, child, _pn, _cn) in zip(corrected_lines, bones):
            set_line(line, corrected_i[[parent, child]])
        set_line(error_line, np.vstack([record.active_wrist[i], record.target[i]]))

        err_cm = record.active_error[i] * 100.0
        after_cm = record.servo_after[i] * 100.0
        title.set_text(
            f"{record.name} | {record.target_link[i]} | "
            f"t={record.elapsed[i]:.2f}s | frame {i + 1}/{len(record.elapsed)}"
        )
        err_text.set_text(
            f"target error: {err_cm:.2f} cm | QP 1-step after: {after_cm:.2f} cm\n"
            f"median/p95/max: {np.nanmedian(record.active_error)*100:.2f} / "
            f"{np.nanpercentile(record.active_error, 95)*100:.2f} / "
            f"{np.nanmax(record.active_error)*100:.2f} cm"
        )
        return (
            measured_scatter, corrected_scatter, target_scatter, waist_marker,
            title, err_text, error_line, *measured_lines, *corrected_lines,
        )

    anim = FuncAnimation(fig, update, frames=len(record.elapsed), interval=1000.0 / max(1, fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    print(f"[gif] saved {output_path} frames={len(record.elapsed)} fps={fps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PATH web-recorded static EE tracking as upper-body GIFs."
    )
    parser.add_argument(
        "records", nargs="+", type=Path,
        help="Record directories or path_static_ee_tracking.npz files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-xml", type=Path,
        default=Path("/home/url/GMR/assets/unitree_g1/g1_mocap_29dof.xml"),
    )
    parser.add_argument("--sample-fps", type=float, default=15.0,
                        help="Resample records before GIF generation. Use <=0 to keep all frames.")
    parser.add_argument("--playback-sec", type=float, default=8.0,
                        help="Playback duration for each full-record GIF.")
    parser.add_argument("--chunk-sec", type=float, default=0.0,
                        help="Optional chunk length. 0 creates one GIF per record.")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--elev", type=float, default=12.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--focus-margin", type=float, default=0.08)
    return parser.parse_args()


def record_chunk(record: RecordData, start: float, end: float, suffix: str) -> RecordData:
    mask = (record.elapsed >= start) & (record.elapsed < end)
    if not np.any(mask):
        raise ValueError("empty chunk")
    idx = np.flatnonzero(mask)
    return RecordData(
        path=record.path,
        name=f"{record.name}_{suffix}",
        elapsed=record.elapsed[idx] - record.elapsed[idx[0]],
        q_measured=record.q_measured[idx],
        q_corrected=record.q_corrected[idx],
        target=record.target[idx],
        left_wrist=record.left_wrist[idx],
        right_wrist=record.right_wrist[idx],
        active_wrist=record.active_wrist[idx],
        active_error=record.active_error[idx],
        servo_after=record.servo_after[idx],
        target_side=record.target_side[idx],
        target_link=[record.target_link[i] for i in idx],
    )


def main() -> int:
    args = parse_args()
    fk = UpperBodyFk(args.model_xml)
    summaries = []
    for input_path in args.records:
        record = resample_record(load_record(input_path), args.sample_fps)
        summaries.append(summarize_record(record))
        print(
            f"[record] {record.name}: frames={len(record.elapsed)} "
            f"duration={record.elapsed[-1]:.2f}s "
            f"median/p95/max={np.nanmedian(record.active_error)*100:.2f}/"
            f"{np.nanpercentile(record.active_error, 95)*100:.2f}/"
            f"{np.nanmax(record.active_error)*100:.2f}cm"
        )
        chunks: list[RecordData] = []
        if args.chunk_sec and args.chunk_sec > 0.0:
            chunk_idx = 0
            start = 0.0
            while start < record.elapsed[-1] - 1e-9:
                if args.max_chunks is not None and chunk_idx >= args.max_chunks:
                    break
                end = min(record.elapsed[-1] + 1e-9, start + args.chunk_sec)
                try:
                    chunks.append(record_chunk(record, start, end, f"chunk_{chunk_idx:03d}"))
                except ValueError:
                    pass
                chunk_idx += 1
                start += args.chunk_sec
        else:
            chunks = [record]
        for chunk in chunks:
            measured = override_wrist_positions(compute_fk_sequence(fk, chunk.q_measured), chunk)
            corrected = override_wrist_positions(compute_fk_sequence(fk, chunk.q_corrected), chunk)
            playback_sec = max(1e-6, args.playback_sec)
            fps = max(1, int(round(len(chunk.elapsed) / playback_sec)))
            output = args.output_dir / f"{chunk.name}_static_ee_upper.gif"
            make_gif(
                record=chunk,
                measured=measured,
                corrected=corrected,
                output_path=output,
                fps=fps,
                elev=args.elev,
                azim=args.azim,
                focus_margin=args.focus_margin,
            )
    write_summary(summaries, args.output_dir / "path_static_ee_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
