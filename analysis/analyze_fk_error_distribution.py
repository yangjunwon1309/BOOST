#!/usr/bin/env python3
"""Plot frame-level FK errors without reducing motion to one RMSE value.

The input is an output directory produced by fk_error_analysis.py. The script
reads segment_*_fk_positions_{mode}.csv files and produces time histories,
sample distributions, motion-versus-error scatter plots, signed XYZ error
plots, and robust distribution statistics.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fk-error-analysis")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POSITION_FILE_RE = re.compile(
    r"segment_(?P<segment>\d+)_fk_positions_(?P<mode>base_local|root_oriented)\.csv"
)
DEFAULT_COMPARISON_BODIES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_toe_link",
    "right_toe_link",
)


@dataclass
class BodySeries:
    dataset: str
    segment: int
    mode: str
    body: str
    time_sec: np.ndarray
    gmr_position: np.ndarray
    error_xyz: np.ndarray
    error_norm: np.ndarray
    gmr_speed: np.ndarray


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        if array.ndim == 1:
            mask &= np.isfinite(array)
        else:
            mask &= np.isfinite(array).all(axis=1)
    return mask


def load_position_file(path: Path, dataset: str, segment: int, mode: str) -> list[BodySeries]:
    rows: dict[str, list[tuple[float, np.ndarray, np.ndarray, float]]] = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            body = row["body"]
            rows[body].append((
                float(row["time_sec"]),
                np.array([row["gmr_x"], row["gmr_y"], row["gmr_z"]], dtype=float),
                np.array([row["err_x"], row["err_y"], row["err_z"]], dtype=float),
                float(row["err_norm_m"]),
            ))

    output: list[BodySeries] = []
    for body, values in rows.items():
        values.sort(key=lambda item: item[0])
        time_sec = np.asarray([item[0] for item in values])
        gmr_position = np.asarray([item[1] for item in values])
        error_xyz = np.asarray([item[2] for item in values])
        error_norm = np.asarray([item[3] for item in values])

        speed = np.full(time_sec.shape, np.nan)
        if len(time_sec) > 1:
            dt = np.diff(time_sec)
            valid_dt = dt > 1e-6
            step_speed = np.full(dt.shape, np.nan)
            step_speed[valid_dt] = (
                np.linalg.norm(np.diff(gmr_position, axis=0)[valid_dt], axis=1)
                / dt[valid_dt]
            )
            speed[1:] = step_speed

        valid = _finite_rows(time_sec, gmr_position, error_xyz, error_norm)
        output.append(BodySeries(
            dataset=dataset,
            segment=segment,
            mode=mode,
            body=body,
            time_sec=time_sec[valid],
            gmr_position=gmr_position[valid],
            error_xyz=error_xyz[valid],
            error_norm=error_norm[valid],
            gmr_speed=speed[valid],
        ))
    return output


def load_analysis_dir(path: Path, modes: set[str]) -> list[BodySeries]:
    series: list[BodySeries] = []
    for csv_path in sorted(path.glob("segment_*_fk_positions_*.csv")):
        match = POSITION_FILE_RE.fullmatch(csv_path.name)
        if not match or match.group("mode") not in modes:
            continue
        series.extend(load_position_file(
            csv_path,
            dataset=path.name,
            segment=int(match.group("segment")),
            mode=match.group("mode"),
        ))
    if not series:
        raise FileNotFoundError(
            f"No segment FK position CSV files for {sorted(modes)} in {path}"
        )
    return series


def select_points(count: int, limit: int, seed: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))


def body_order(series: list[BodySeries]) -> list[str]:
    first_seen: dict[str, int] = {}
    for item in series:
        first_seen.setdefault(item.body, len(first_seen))
    return list(first_seen)


def concatenate_body(series: list[BodySeries], body: str, field: str) -> np.ndarray:
    chunks = [getattr(item, field) for item in series if item.body == body]
    if not chunks:
        return np.empty((0, 3)) if field in {"error_xyz", "gmr_position"} else np.empty(0)
    return np.concatenate(chunks, axis=0)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def write_distribution_summary(series: list[BodySeries], output_path: Path) -> None:
    fields = [
        "dataset", "mode", "body", "sample_count", "median_cm", "q25_cm",
        "q75_cm", "iqr_cm", "p90_cm", "p95_cm", "max_cm", "bias_x_cm",
        "bias_y_cm", "bias_z_cm", "error_over_3cm_pct", "error_over_5cm_pct",
        "motion_error_pearson", "low_motion_median_cm", "high_motion_median_cm",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode in sorted({item.mode for item in series}):
            mode_series = [item for item in series if item.mode == mode]
            for body in body_order(mode_series):
                error_cm = concatenate_body(mode_series, body, "error_norm") * 100.0
                error_xyz_cm = concatenate_body(mode_series, body, "error_xyz") * 100.0
                speed = concatenate_body(mode_series, body, "gmr_speed")
                valid_speed = np.isfinite(speed)
                low_median = high_median = float("nan")
                if valid_speed.sum() >= 4:
                    low, high = np.percentile(speed[valid_speed], [25, 75])
                    low_median = float(np.median(error_cm[valid_speed & (speed <= low)]))
                    high_median = float(np.median(error_cm[valid_speed & (speed >= high)]))
                q25, median, q75, p90, p95 = np.percentile(
                    error_cm, [25, 50, 75, 90, 95]
                )
                writer.writerow({
                    "dataset": mode_series[0].dataset,
                    "mode": mode,
                    "body": body,
                    "sample_count": len(error_cm),
                    "median_cm": median,
                    "q25_cm": q25,
                    "q75_cm": q75,
                    "iqr_cm": q75 - q25,
                    "p90_cm": p90,
                    "p95_cm": p95,
                    "max_cm": np.max(error_cm),
                    "bias_x_cm": np.mean(error_xyz_cm[:, 0]),
                    "bias_y_cm": np.mean(error_xyz_cm[:, 1]),
                    "bias_z_cm": np.mean(error_xyz_cm[:, 2]),
                    "error_over_3cm_pct": 100.0 * np.mean(error_cm > 3.0),
                    "error_over_5cm_pct": 100.0 * np.mean(error_cm > 5.0),
                    "motion_error_pearson": correlation(speed, error_cm),
                    "low_motion_median_cm": low_median,
                    "high_motion_median_cm": high_median,
                })


def plot_error_distribution(
    series: list[BodySeries], mode: str, output_path: Path, max_points: int
) -> None:
    mode_series = [item for item in series if item.mode == mode]
    bodies = body_order(mode_series)
    fig, ax = plt.subplots(figsize=(max(11, len(bodies) * 1.05), 6.5))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(bodies)))
    rng = np.random.default_rng(2026)
    for index, (body, color) in enumerate(zip(bodies, colors)):
        values = concatenate_body(mode_series, body, "error_norm") * 100.0
        chosen = select_points(len(values), max_points, index)
        jitter = rng.uniform(-0.28, 0.28, len(chosen))
        ax.scatter(index + jitter, values[chosen], s=7, alpha=0.18, color=color,
                   edgecolors="none", rasterized=True)
        q25, median, q75, p95 = np.percentile(values, [25, 50, 75, 95])
        ax.vlines(index, q25, q75, color="black", linewidth=4, zorder=4)
        ax.scatter(index, median, marker="_", s=180, linewidths=2.5,
                   color="black", zorder=5)
        ax.scatter(index, p95, marker="D", s=20, color="#d1495b", zorder=5)
    ax.set_xticks(range(len(bodies)), bodies, rotation=35, ha="right")
    ax.set_ylabel("FK position error (cm)")
    ax.set_title(f"Frame-level FK error distribution: {mode}\n"
                 "points=frames, black=median/IQR, red diamond=P95")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_timeline(series: list[BodySeries], mode: str, output_path: Path) -> None:
    mode_series = [item for item in series if item.mode == mode]
    bodies = body_order(mode_series)
    groups = {
        "Upper body": [b for b in bodies if any(k in b for k in ("shoulder", "elbow", "wrist"))],
        "Lower body": [b for b in bodies if any(k in b for k in ("knee", "toe"))],
        "Pelvis / torso": [b for b in bodies if b not in {
            x for x in bodies if any(k in x for k in ("shoulder", "elbow", "wrist", "knee", "toe"))
        }],
    }
    groups = {name: values for name, values in groups.items() if values}
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 3.6 * len(groups)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (group_name, group_bodies) in zip(axes, groups.items()):
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(group_bodies)))
        for body, color in zip(group_bodies, colors):
            labeled = False
            for item in mode_series:
                if item.body != body:
                    continue
                ax.plot(item.time_sec, item.error_norm * 100.0, color=color,
                        alpha=0.8, linewidth=0.9, label=body if not labeled else None)
                labeled = True
        ax.set_ylabel("Error (cm)")
        ax.set_title(group_name, loc="left")
        ax.grid(alpha=0.22)
        ax.legend(ncol=min(4, len(group_bodies)), fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Recorded time (s)")
    fig.suptitle(f"FK error over time: {mode}", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_motion_error_scatter(
    series: list[BodySeries], mode: str, output_path: Path, max_points: int
) -> None:
    mode_series = [item for item in series if item.mode == mode]
    bodies = body_order(mode_series)
    columns = 3
    rows = int(np.ceil(len(bodies) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows * 3.5), squeeze=False)
    for index, (ax, body) in enumerate(zip(axes.flat, bodies)):
        speed = concatenate_body(mode_series, body, "gmr_speed")
        error_cm = concatenate_body(mode_series, body, "error_norm") * 100.0
        valid = np.isfinite(speed) & np.isfinite(error_cm)
        speed, error_cm = speed[valid], error_cm[valid]
        chosen = select_points(len(speed), max_points, index + 100)
        ax.scatter(speed[chosen], error_cm[chosen], s=7, alpha=0.22,
                   color="#2a6f97", edgecolors="none", rasterized=True)
        ax.set_title(f"{body}\nr={correlation(speed, error_cm):.2f}", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(bodies):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("GMR link speed (m/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("FK error (cm)")
    fig.suptitle(f"Reference motion magnitude vs FK error: {mode}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_signed_axis_error(
    series: list[BodySeries], mode: str, output_path: Path, max_points: int
) -> None:
    mode_series = [item for item in series if item.mode == mode]
    bodies = body_order(mode_series)
    fig, axes = plt.subplots(3, 1, figsize=(max(11, len(bodies) * 1.05), 11), sharex=True)
    axis_names = ("X", "Y", "Z")
    axis_colors = ("#0077b6", "#d1495b", "#2a9d8f")
    rng = np.random.default_rng(707)
    for component, ax in enumerate(axes):
        for index, body in enumerate(bodies):
            values = concatenate_body(mode_series, body, "error_xyz")[:, component] * 100.0
            chosen = select_points(len(values), max_points, 300 + component * 50 + index)
            jitter = rng.uniform(-0.25, 0.25, len(chosen))
            ax.scatter(index + jitter, values[chosen], s=6, alpha=0.16,
                       color=axis_colors[component], edgecolors="none", rasterized=True)
            ax.scatter(index, np.median(values), marker="_", s=150,
                       linewidths=2.2, color="black", zorder=4)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{axis_names[component]} error (cm)")
        ax.grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(range(len(bodies)), bodies, rotation=35, ha="right")
    fig.suptitle(f"Signed FK error by world/base axis: {mode}\nblack marker=median")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_dataset(
    path: Path, series: list[BodySeries], output_dir: Path, max_points: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_distribution_summary(series, output_dir / "fk_error_distribution_summary.csv")
    for mode in sorted({item.mode for item in series}):
        plot_error_distribution(
            series, mode, output_dir / f"error_distribution_{mode}.png", max_points
        )
        plot_error_timeline(series, mode, output_dir / f"error_timeline_{mode}.png")
        plot_motion_error_scatter(
            series, mode, output_dir / f"motion_vs_error_{mode}.png", max_points
        )
        plot_signed_axis_error(
            series, mode, output_dir / f"signed_axis_error_{mode}.png", max_points
        )
    print(f"[done] {path.name}: {output_dir}")


def plot_dataset_comparison(
    all_series: dict[str, list[BodySeries]], modes: set[str], output_dir: Path,
    max_points: int, requested_bodies: list[str] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = list(all_series)
    for mode in sorted(modes):
        available = {
            item.body for series in all_series.values() for item in series if item.mode == mode
        }
        bodies = requested_bodies or [b for b in DEFAULT_COMPARISON_BODIES if b in available]
        if not bodies:
            continue
        fig, axes = plt.subplots(
            len(bodies), 1, figsize=(max(10, len(dataset_names) * 1.35), 3.6 * len(bodies)),
            sharex=True, squeeze=False,
        )
        rng = np.random.default_rng(404)
        for body_index, body in enumerate(bodies):
            ax = axes[body_index, 0]
            for dataset_index, dataset in enumerate(dataset_names):
                mode_series = [item for item in all_series[dataset] if item.mode == mode]
                values = concatenate_body(mode_series, body, "error_norm") * 100.0
                if not len(values):
                    continue
                chosen = select_points(len(values), max_points, body_index * 100 + dataset_index)
                jitter = rng.uniform(-0.27, 0.27, len(chosen))
                ax.scatter(dataset_index + jitter, values[chosen], s=6, alpha=0.13,
                           color="#457b9d", edgecolors="none", rasterized=True)
                q25, median, q75, p95 = np.percentile(values, [25, 50, 75, 95])
                ax.vlines(dataset_index, q25, q75, color="black", linewidth=4)
                ax.scatter(dataset_index, median, marker="_", s=170,
                           linewidths=2.3, color="black", zorder=4)
                ax.scatter(dataset_index, p95, marker="D", s=18,
                           color="#d1495b", zorder=4)
            ax.set_ylabel("Error (cm)")
            ax.set_title(body, loc="left")
            ax.grid(axis="y", alpha=0.22)
        axes[-1, 0].set_xticks(range(len(dataset_names)), dataset_names,
                               rotation=30, ha="right")
        fig.suptitle(f"Frame-level experiment comparison: {mode}\n"
                     "points=frames, black=median/IQR, red diamond=P95")
        fig.tight_layout()
        fig.savefig(output_dir / f"experiment_comparison_{mode}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis_dirs", nargs="*", type=Path,
        help="fk_error_analysis output directories (absolute or relative to --root-dir)",
    )
    parser.add_argument(
        "--root-dir", type=Path, default=Path("~/gmr_ref_from_session").expanduser(),
        help="root used for relative directory names and --all discovery",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="analyze every directory under --root-dir containing FK position CSV files",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("base_local", "root_oriented"),
        default=("base_local", "root_oriented"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="write each result to OUTPUT_ROOT/DATASET instead of inside the input directory",
    )
    parser.add_argument("--output-name", default="fk_error_distribution")
    parser.add_argument("--max-scatter-points", type=int, default=2500)
    parser.add_argument(
        "--comparison-bodies", nargs="*",
        help="body names for multi-dataset plots; defaults to wrists and toes",
    )
    return parser.parse_args()


def resolve_analysis_dirs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw_path in args.analysis_dirs:
        path = raw_path.expanduser()
        if not path.is_absolute():
            path = args.root_dir / path
        paths.append(path.resolve())
    if args.all:
        discovered = sorted({path.parent for path in args.root_dir.glob(
            "*/segment_*_fk_positions_*.csv"
        )})
        paths.extend(discovered)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("Provide at least one analysis directory or use --all")
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise SystemExit("Not directories: " + ", ".join(missing))
    return paths


def main() -> None:
    args = parse_args()
    modes = set(args.modes)
    paths = resolve_analysis_dirs(args)
    all_series: dict[str, list[BodySeries]] = {}
    for path in paths:
        series = load_analysis_dir(path, modes)
        all_series[path.name] = series
        output_dir = (
            args.output_root.expanduser() / path.name
            if args.output_root else path / args.output_name
        )
        analyze_dataset(path, series, output_dir, args.max_scatter_points)

    if len(all_series) > 1:
        comparison_dir = (
            args.output_root.expanduser() / "comparison"
            if args.output_root else args.root_dir / "fk_error_distribution_comparison"
        )
        plot_dataset_comparison(
            all_series, modes, comparison_dir, args.max_scatter_points,
            args.comparison_bodies,
        )
        print(f"[done] comparison: {comparison_dir}")


if __name__ == "__main__":
    main()
