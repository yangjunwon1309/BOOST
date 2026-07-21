#!/usr/bin/env python3
"""Create static-EE target or reference-tracking GIFs from PATH web records.

The recorder in scripts/g1_mujoco_gmr_web_view.py stores q_measured,
pd_q_corrected, QP-internal wrist positions, and terminal EE targets.  This
visualizer keeps the upper-body FK style of compare_fk_upper_body_gif.py, but
can either show a red static EE target or a green FK reference from q_ref.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, replace
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
    q_ref: np.ndarray
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


def _stack_columns(data: np.lib.npyio.NpzFile, prefix: str, n: int, source: Path) -> np.ndarray:
    columns = []
    for i in range(n):
        key = f"{prefix}_{i:02d}"
        if key not in data:
            raise KeyError(f"{key} missing from {source}")
        columns.append(np.asarray(data[key], dtype=np.float64))
    return np.stack(columns, axis=1)


def _object_strings(values: np.ndarray) -> list[str]:
    return [str(v) for v in values.tolist()]


def _zeros_like_rows(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.float64)


def load_record(path: Path, mode: str, ref_q_key: str = "auto") -> RecordData:
    if path.is_dir():
        npz = path / "path_static_ee_tracking.npz"
    else:
        npz = path
    if not npz.exists():
        raise FileNotFoundError(npz)
    z = np.load(npz, allow_pickle=True)
    elapsed = np.asarray(z["elapsed_sec"], dtype=np.float64)
    elapsed = elapsed - elapsed[0]
    q_measured = _stack_columns(z, "body_q_measured", 29, npz)
    if mode == "ref" and ref_q_key in {"auto", "gmr"} and "gmr_q_ref_00" in z:
        q_ref = _stack_columns(z, "gmr_q_ref", 29, npz)
    elif ref_q_key in {"auto", "bspline"} and "bspline_q_ref_00" in z:
        q_ref = _stack_columns(z, "bspline_q_ref", 29, npz)
    elif ref_q_key == "gmr":
        raise KeyError(f"{npz} has no gmr_q_ref_* columns")
    elif ref_q_key == "bspline":
        raise KeyError(f"{npz} has no bspline_q_ref_* columns")
    else:
        q_ref = q_measured.copy()
    if "pd_q_corrected_00" in z:
        q_corrected = _stack_columns(z, "pd_q_corrected", 29, npz)
    else:
        q_corrected = q_measured.copy()
    rows = q_measured.shape[0]
    side = np.asarray(z["ee_qp_terminal_target_side"], dtype=np.float64) \
        if "ee_qp_terminal_target_side" in z else np.full(rows, 3.0)
    if mode == "static":
        if "qp_input_target_position_00" in z:
            target = _stack_columns(z, "qp_input_target_position", 3, npz)
        elif "terminal_target_position_00" in z:
            target = _stack_columns(z, "terminal_target_position", 3, npz)
        else:
            raise KeyError(
                f"{npz} has no recorded target position. Use --mode ref for "
                "GMR/reference tracking records, or re-record static target data."
            )
        left_wrist = _stack_columns(z, "ee_qp_left_wrist_position", 3, npz)
        right_wrist = _stack_columns(z, "ee_qp_right_wrist_position", 3, npz)
        if "ee_qp_active_wrist_position_00" in z:
            active_wrist = _stack_columns(z, "ee_qp_active_wrist_position", 3, npz)
        else:
            active_wrist = np.where((side[:, None] == 2.0), right_wrist, left_wrist)
        active_error = np.linalg.norm(active_wrist - target, axis=1)
    elif mode == "ref":
        if ref_q_key == "gmr" and "gmr_q_ref_00" not in z:
            raise KeyError(f"{npz} has no gmr_q_ref_* columns for --mode ref")
        if ref_q_key == "bspline" and "bspline_q_ref_00" not in z:
            raise KeyError(f"{npz} has no bspline_q_ref_* columns for --mode ref")
        if ref_q_key == "auto" and "gmr_q_ref_00" not in z and "bspline_q_ref_00" not in z:
            raise KeyError(f"{npz} has no gmr_q_ref_* or bspline_q_ref_* columns for --mode ref")
        target = _zeros_like_rows(rows, 3)
        left_wrist = _zeros_like_rows(rows, 3)
        right_wrist = _zeros_like_rows(rows, 3)
        active_wrist = _zeros_like_rows(rows, 3)
        active_error = np.zeros(rows, dtype=np.float64)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if "ee_qp_servo_error_after_m" in z:
        servo_after = np.asarray(z["ee_qp_servo_error_after_m"], dtype=np.float64)
    else:
        servo_after = np.full_like(active_error, np.nan)
    return RecordData(
        path=npz,
        name=npz.parent.name,
        elapsed=elapsed,
        q_measured=q_measured,
        q_ref=q_ref,
        q_corrected=q_corrected,
        target=target,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        active_wrist=active_wrist,
        active_error=active_error,
        servo_after=servo_after,
        target_side=side,
        target_link=_object_strings(z["ee_qp_terminal_target_link"])
        if "ee_qp_terminal_target_link" in z else ["gmr_ref"] * rows,
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


def resample_positions(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return resample_array(times, values, grid)


def shift_overlap_duration(
    target_times: np.ndarray,
    source_times: np.ndarray,
    shift: float,
) -> tuple[float, float]:
    """Compare target(t) against source(t + shift)."""
    start = max(float(target_times[0]), float(source_times[0]) - shift)
    end = min(float(target_times[-1]), float(source_times[-1]) - shift)
    return start, end


def find_best_ref_source_shift(
    target_times: np.ndarray,
    target_positions: np.ndarray,
    source_times: np.ndarray,
    source_positions: np.ndarray,
    max_shift_sec: float,
    step_sec: float,
    sample_fps: float,
) -> tuple[float, float]:
    shifts = np.arange(-abs(max_shift_sec), abs(max_shift_sec) + step_sec * 0.5, step_sec)
    best_shift = 0.0
    best_rms = math.inf
    for shift in shifts:
        start, end = shift_overlap_duration(target_times, source_times, float(shift))
        if end <= start:
            continue
        grid = np.arange(start, end, 1.0 / max(sample_fps, 1e-6), dtype=np.float64)
        if len(grid) < 10:
            continue
        target_sample = resample_positions(target_times, target_positions, grid)
        source_sample = resample_positions(source_times, source_positions, grid + shift)
        diff = target_sample - source_sample
        rms = float(np.sqrt(np.nanmean(diff * diff)))
        if rms < best_rms:
            best_rms = rms
            best_shift = float(shift)
    if not math.isfinite(best_rms):
        raise ValueError("failed to find valid overlap for external reference alignment")
    return best_shift, best_rms


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
        q_ref=resample_array(record.elapsed, record.q_ref, grid),
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
    reference: np.ndarray | None,
    corrected: np.ndarray,
    output_path: Path,
    mode: str,
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
    if mode == "ref":
        side_mode = 3
    active_chain = LEFT_CHAIN if side_mode == 1 else RIGHT_CHAIN if side_mode == 2 else LEFT_CHAIN | RIGHT_CHAIN
    active_wrist_name = "left_wrist_yaw_link" if side_mode == 1 else "right_wrist_yaw_link"

    active_indices = [body_to_idx[name] for name in active_chain if name in body_to_idx]
    focus_points = [
        measured[:, active_indices, :] if active_indices else measured,
        corrected[:, active_indices, :] if active_indices else corrected,
        record.active_wrist[:, None, :],
    ]
    if mode == "static":
        focus_points.append(record.target[:, None, :])
    if reference is not None:
        focus_points.append(reference[:, active_indices, :] if active_indices else reference)
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
        "reference": "#16a34a",
        "corrected": "#7c3aed",
        "target": "#dc2626",
        "waist": "#111827",
        "error": "#ef4444",
    }
    measured_scatter = ax.scatter([], [], [], c=colors["measured"], s=16, marker="o",
                                  depthshade=False, label="measured FK")
    reference_scatter = ax.scatter([], [], [], c=colors["reference"], s=12, marker=".",
                                   depthshade=False, alpha=0.70, label="ref FK")
    corrected_scatter = ax.scatter([], [], [], c=colors["corrected"], s=12, marker=".",
                                   depthshade=False, alpha=0.65, label="PD/QP target")
    target_scatter = ax.scatter([], [], [], c=colors["target"], s=60, marker="*",
                                depthshade=False, label="EE target")
    measured_lines = []
    reference_lines = []
    corrected_lines = []
    for parent, child, parent_name, child_name in bones:
        active = parent_name in active_chain and child_name in active_chain
        width = 1.8 if active else 0.8
        alpha = 0.95 if active else 0.35
        measured_lines.append(
            ax.plot([], [], [], color=colors["measured"], linewidth=width, alpha=alpha)[0]
        )
        reference_lines.append(
            ax.plot([], [], [], color=colors["reference"], linewidth=max(0.7, width - 0.4),
                    linestyle="-.", alpha=0.60 if active else 0.22)[0]
        )
        corrected_lines.append(
            ax.plot([], [], [], color=colors["corrected"], linewidth=max(0.7, width - 0.5),
                    linestyle="--", alpha=0.55 if active else 0.22)[0]
        )
    error_line = ax.plot([], [], [], color=colors["error"], linewidth=1.4,
                         linestyle=":", alpha=0.95)[0]
    error_line_right = ax.plot([], [], [], color=colors["error"], linewidth=1.2,
                               linestyle=":", alpha=0.80)[0]
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
    left_wrist_idx = body_to_idx["left_wrist_yaw_link"]
    right_wrist_idx = body_to_idx["right_wrist_yaw_link"]
    torso_idx = body_to_idx.get("torso_link", 0)
    pelvis_idx = body_to_idx.get("pelvis", 0)

    def update(i: int):
        measured_i = measured[i]
        reference_i = reference[i] if reference is not None else None
        corrected_i = corrected[i]
        set_points(measured_scatter, measured_i)
        if reference_i is not None:
            set_points(reference_scatter, reference_i)
        else:
            set_points(reference_scatter, np.empty((0, 3)))
        set_points(corrected_scatter, corrected_i)
        if mode == "static":
            set_points(target_scatter, record.target[i:i + 1])
        else:
            set_points(target_scatter, np.empty((0, 3)))
        set_points(waist_marker, measured_i[[pelvis_idx, torso_idx]])
        for line, (parent, child, _pn, _cn) in zip(measured_lines, bones):
            set_line(line, measured_i[[parent, child]])
        for line, (parent, child, _pn, _cn) in zip(reference_lines, bones):
            if reference_i is not None:
                set_line(line, reference_i[[parent, child]])
            else:
                set_line(line, np.empty((0, 3)))
        for line, (parent, child, _pn, _cn) in zip(corrected_lines, bones):
            set_line(line, corrected_i[[parent, child]])
        if mode == "ref" and reference_i is not None:
            set_line(error_line, np.vstack([measured_i[left_wrist_idx], reference_i[left_wrist_idx]]))
            set_line(error_line_right, np.vstack([measured_i[right_wrist_idx], reference_i[right_wrist_idx]]))
        else:
            set_line(error_line, np.vstack([record.active_wrist[i], record.target[i]]))
            set_line(error_line_right, np.empty((0, 3)))

        err_cm = record.active_error[i] * 100.0
        after_cm = record.servo_after[i] * 100.0
        error_label = "ref wrist mean error" if mode == "ref" else "target error"
        after_label = "corrected ref mean" if mode == "ref" else "QP 1-step after"
        title.set_text(
            f"{record.name} | {mode} | {record.target_link[i]} | "
            f"t={record.elapsed[i]:.2f}s | frame {i + 1}/{len(record.elapsed)}"
        )
        err_text.set_text(
            f"{error_label}: {err_cm:.2f} cm | {after_label}: {after_cm:.2f} cm\n"
            f"median/p95/max: {np.nanmedian(record.active_error)*100:.2f} / "
            f"{np.nanpercentile(record.active_error, 95)*100:.2f} / "
            f"{np.nanmax(record.active_error)*100:.2f} cm"
        )
        return (
            measured_scatter, reference_scatter, corrected_scatter, target_scatter, waist_marker,
            title, err_text, error_line, error_line_right,
            *measured_lines, *reference_lines, *corrected_lines,
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
    parser.add_argument("--mode", choices=("static", "ref"), default="static",
                        help="'static': compare wrist to static EE target. "
                             "'ref': compare measured wrist FK to GMR/reference FK.")
    parser.add_argument("--ref-q-key", choices=("auto", "gmr", "bspline"), default="auto",
                        help="Reference q columns for --mode ref. auto prefers gmr_q_ref "
                             "and falls back to bspline_q_ref.")
    parser.add_argument("--ref-source-record", type=Path, default=None,
                        help="For --mode ref, use this record's reference q as the reference.")
    parser.add_argument("--align-ref-source", action="store_true",
                        help="Auto-align --ref-source-record before sampling its q_ref.")
    parser.add_argument("--align-by", choices=("measured", "ref"), default="measured",
                        help="Alignment signal for --ref-source-record. 'measured' matches "
                             "target measured FK to source measured FK; 'ref' matches target "
                             "measured FK to source q_ref FK.")
    parser.add_argument("--align-max-shift-sec", type=float, default=20.0)
    parser.add_argument("--align-step-sec", type=float, default=0.05)
    parser.add_argument("--align-sample-fps", type=float, default=10.0)
    parser.add_argument(
        "--model-xml", type=Path,
        default=Path("/home/url/GMR/assets/unitree_g1/g1_mocap_29dof.xml"),
    )
    parser.add_argument("--sample-fps", type=float, default=15.0,
                        help="Resample records before GIF generation. Use <=0 to keep all frames.")
    parser.add_argument("--playback-sec", type=float, default=8.0,
                        help="Playback duration for each full-record GIF.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier. 2.0 makes the GIF twice as fast.")
    parser.add_argument("--max-gif-fps", type=int, default=60,
                        help="Cap GIF FPS to avoid huge/unsupported animation delays.")
    parser.add_argument("--skip-invalid", action="store_true",
                        help="Skip records missing required target/wrist fields.")
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
        q_ref=record.q_ref[idx],
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
    ref_source_record = None
    ref_source_measured = None
    ref_source_reference = None
    if args.mode == "ref" and args.ref_source_record is not None:
        ref_source_record = resample_record(
            load_record(args.ref_source_record, "ref", args.ref_q_key),
            args.sample_fps,
        )
        ref_source_measured = compute_fk_sequence(fk, ref_source_record.q_measured)
        ref_source_reference = compute_fk_sequence(fk, ref_source_record.q_ref)
        print(
            f"[ref-source] {ref_source_record.name}: frames={len(ref_source_record.elapsed)} "
            f"duration={ref_source_record.elapsed[-1]:.2f}s"
        )
    for input_path in args.records:
        try:
            record = resample_record(
                load_record(input_path, args.mode, args.ref_q_key),
                args.sample_fps,
            )
        except (KeyError, FileNotFoundError, ValueError) as exc:
            if args.skip_invalid:
                print(f"[skip] {input_path}: {exc}")
                continue
            raise
        if ref_source_record is not None:
            shift = 0.0
            if args.align_ref_source:
                target_measured = compute_fk_sequence(fk, record.q_measured)
                source_signal = (
                    ref_source_reference if args.align_by == "ref" else ref_source_measured
                )
                assert source_signal is not None
                shift, rms = find_best_ref_source_shift(
                    record.elapsed,
                    target_measured,
                    ref_source_record.elapsed,
                    source_signal,
                    max_shift_sec=args.align_max_shift_sec,
                    step_sec=args.align_step_sec,
                    sample_fps=args.align_sample_fps,
                )
                print(
                    f"[align] {record.name}: source={ref_source_record.name} "
                    f"shift={shift:+.3f}s rms={rms*100:.2f}cm by={args.align_by}"
                )
            record = replace(
                record,
                q_ref=resample_array(
                    ref_source_record.elapsed,
                    ref_source_record.q_ref,
                    record.elapsed + shift,
                ),
                target_link=[f"{ref_source_record.name}/q_ref_shift_{shift:+.3f}s"]
                * len(record.elapsed),
                target_side=np.full(len(record.elapsed), 3.0),
            )
        if args.mode == "ref":
            measured_all = compute_fk_sequence(fk, record.q_measured)
            reference_all = compute_fk_sequence(fk, record.q_ref)
            corrected_all = compute_fk_sequence(fk, record.q_corrected)
            body_to_idx = {name: i for i, name in enumerate(UPPER_BODIES)}
            left_idx = body_to_idx["left_wrist_yaw_link"]
            right_idx = body_to_idx["right_wrist_yaw_link"]
            left_err = np.linalg.norm(
                measured_all[:, left_idx, :] - reference_all[:, left_idx, :], axis=1)
            right_err = np.linalg.norm(
                measured_all[:, right_idx, :] - reference_all[:, right_idx, :], axis=1)
            left_after = np.linalg.norm(
                corrected_all[:, left_idx, :] - reference_all[:, left_idx, :], axis=1)
            right_after = np.linalg.norm(
                corrected_all[:, right_idx, :] - reference_all[:, right_idx, :], axis=1)
            record.active_error = 0.5 * (left_err + right_err)
            record.servo_after = 0.5 * (left_after + right_after)
            record.target_link = ["left/right_wrist_yaw_link_ref"] * len(record.elapsed)
            record.target_side = np.full(len(record.elapsed), 3.0)
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
            measured = compute_fk_sequence(fk, chunk.q_measured)
            reference = compute_fk_sequence(fk, chunk.q_ref) if args.mode == "ref" else None
            corrected = compute_fk_sequence(fk, chunk.q_corrected)
            if args.mode == "static":
                measured = override_wrist_positions(measured, chunk)
                corrected = override_wrist_positions(corrected, chunk)
            elif reference is not None:
                body_to_idx = {name: i for i, name in enumerate(UPPER_BODIES)}
                left_idx = body_to_idx["left_wrist_yaw_link"]
                right_idx = body_to_idx["right_wrist_yaw_link"]
                left_err = np.linalg.norm(measured[:, left_idx, :] - reference[:, left_idx, :], axis=1)
                right_err = np.linalg.norm(measured[:, right_idx, :] - reference[:, right_idx, :], axis=1)
                left_after = np.linalg.norm(corrected[:, left_idx, :] - reference[:, left_idx, :], axis=1)
                right_after = np.linalg.norm(corrected[:, right_idx, :] - reference[:, right_idx, :], axis=1)
                chunk.active_error = 0.5 * (left_err + right_err)
                chunk.servo_after = 0.5 * (left_after + right_after)
                chunk.target_link = ["left/right_wrist_yaw_link_ref"] * len(chunk.elapsed)
            playback_sec = max(1e-6, args.playback_sec / max(1e-6, args.speed))
            fps = max(1, int(round(len(chunk.elapsed) / playback_sec)))
            fps = min(fps, max(1, args.max_gif_fps))
            output = args.output_dir / f"{chunk.name}_{args.mode}_upper.gif"
            make_gif(
                record=chunk,
                measured=measured,
                reference=reference,
                corrected=corrected,
                output_path=output,
                mode=args.mode,
                fps=fps,
                elev=args.elev,
                azim=args.azim,
                focus_margin=args.focus_margin,
            )
    summary_name = "path_static_ee_summary.csv" if args.mode == "static" else "path_ref_ee_summary.csv"
    write_summary(summaries, args.output_dir / summary_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
