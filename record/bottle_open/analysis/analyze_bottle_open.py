#!/usr/bin/env python3
"""Analyze bottle-opening session CSV/JSON logs and produce tables/figures.

Primary method comparisons use trials for which both ``trial_*.csv`` and the
matching ``trial_*.summary.json`` are present.  Orphaned files are still parsed
where possible and are listed in the data-quality audit, but are excluded from
primary comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["PATH", "No jump diffusion", "No orientation", "Position Only", "Alpha blend"]
HANDS = ["left", "right"]
COLORS = {
    "PATH": "#0072B2",
    "No jump diffusion": "#009E73",
    "No orientation": "#E69F00",
    "Position Only": "#CC79A7",
    "Alpha blend": "#D55E00",
}


def safe_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def malformed_json(value: str | None) -> bool:
    if not value:
        return False
    try:
        json.loads(value)
        return False
    except (json.JSONDecodeError, TypeError):
        return True


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def std(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else math.nan)


def rmse(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.sqrt(np.mean(np.square(values)))) if values else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else math.nan


def norm3(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)))


def matrix_angle_deg(previous: list[float], current: list[float]) -> float:
    a = np.asarray(previous, dtype=float).reshape(3, 3)
    b = np.asarray(current, dtype=float).reshape(3, 3)
    cosine = float(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def unsigned_axis_angle_deg(previous: list[float], current: list[float]) -> float:
    a = np.asarray(previous, dtype=float)
    b = np.asarray(current, dtype=float)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return math.nan
    cosine = float(np.clip(abs(float(np.dot(a, b))) / (a_norm * b_norm), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def as_sequence_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, value)


def core_trial_id(method: str, trial: str) -> str:
    return f"{method}/{trial}"


def read_csv_log(path: Path, method: str, trial: str) -> dict[str, Any]:
    rows = 0
    malformed_json_cells = 0
    last_elapsed = math.nan
    all_sequences: set[str] = set()
    seen_belief_keys: set[tuple[str, str]] = set()
    belief_updates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    applied_records: list[dict[str, Any]] = []
    distance_increment = {hand: 0.0 for hand in HANDS}
    angular_path = {hand: 0.0 for hand in HANDS}
    angular_steps = {hand: 0 for hand in HANDS}
    valid_ee_samples = {hand: 0 for hand in HANDS}
    previous_orientation: dict[str, list[float] | None] = {hand: None for hand in HANDS}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            elapsed = finite(row.get("elapsed_s"))
            if elapsed is not None:
                last_elapsed = elapsed
            sequence = row.get("belief_result_sequence", "")
            if sequence:
                all_sequences.add(sequence)
            belief_age = finite(row.get("belief_age_ms"))

            for hand in HANDS:
                ee_raw = row.get(f"{hand}_ee_json")
                ipopt_raw = row.get(f"{hand}_ipopt_target_json")
                belief_raw = row.get(f"{hand}_belief_target_json")
                ee = safe_json(ee_raw)
                ipopt = safe_json(ipopt_raw)
                belief = safe_json(belief_raw)
                malformed_json_cells += int(malformed_json(ee_raw))
                malformed_json_cells += int(malformed_json(ipopt_raw))
                malformed_json_cells += int(malformed_json(belief_raw))

                if isinstance(ee, dict):
                    increment = finite(ee.get("distance_increment_m"))
                    if increment is not None:
                        distance_increment[hand] += increment
                    position = ee.get("position_m")
                    orientation = ee.get("orientation")
                    if isinstance(position, list) and len(position) == 3:
                        valid_ee_samples[hand] += 1
                    if isinstance(orientation, list) and len(orientation) == 9:
                        previous = previous_orientation[hand]
                        if previous is not None:
                            try:
                                angular_path[hand] += matrix_angle_deg(previous, orientation)
                                angular_steps[hand] += 1
                            except (ValueError, TypeError):
                                pass
                        previous_orientation[hand] = orientation
                    else:
                        previous_orientation[hand] = None

                if sequence and isinstance(belief, dict):
                    key = (sequence, hand)
                    position = belief.get("position_m")
                    if key not in seen_belief_keys and isinstance(position, list) and len(position) == 3:
                        seen_belief_keys.add(key)
                        belief_updates[hand].append(
                            {
                                "method": method,
                                "trial": trial,
                                "trial_id": core_trial_id(method, trial),
                                "hand": hand,
                                "sequence": sequence,
                                "elapsed_s": elapsed,
                                "belief_age_ms": belief_age,
                                "object_id": belief.get("object_id"),
                                "target_x_m": finite(position[0]),
                                "target_y_m": finite(position[1]),
                                "target_z_m": finite(position[2]),
                                "axis": belief.get("orientation_axis"),
                                "relation": belief.get("relation"),
                                "joint_posterior": finite(belief.get("joint_posterior")),
                                "object_posterior": finite(belief.get("object_posterior")),
                                "orientation_posterior": finite(belief.get("orientation_posterior")),
                            }
                        )

                if isinstance(ipopt, dict) and isinstance(ee, dict):
                    position_applied = ipopt.get("position_source") == "belief_contact" or bool(
                        ipopt.get("belief_position_applied")
                    )
                    orientation_applied = ipopt.get("orientation_source") == "belief_minimal_rotation"
                    if position_applied or orientation_applied:
                        position_error = finite(ee.get("tracking_position_error_m")) if position_applied else None
                        orientation_error = (
                            finite(ee.get("tracking_orientation_error_deg")) if orientation_applied else None
                        )
                        discrepancy = None
                        if position_applied and isinstance(belief, dict):
                            belief_position = belief.get("position_m")
                            command_position = ipopt.get("position_m")
                            if (
                                isinstance(belief_position, list)
                                and len(belief_position) == 3
                                and isinstance(command_position, list)
                                and len(command_position) == 3
                            ):
                                discrepancy = 1000.0 * norm3(belief_position, command_position)
                        applied_records.append(
                            {
                                "method": method,
                                "trial": trial,
                                "trial_id": core_trial_id(method, trial),
                                "hand": hand,
                                "elapsed_s": elapsed,
                                "position_applied": position_applied,
                                "orientation_applied": orientation_applied,
                                "ee_position_error_mm": (
                                    1000.0 * position_error if position_error is not None else math.nan
                                ),
                                "ee_orientation_error_deg": (
                                    orientation_error if orientation_error is not None else math.nan
                                ),
                                "belief_to_command_error_mm": (
                                    discrepancy if discrepancy is not None else math.nan
                                ),
                            }
                        )

    for hand in HANDS:
        belief_updates[hand].sort(
            key=lambda item: (
                item["elapsed_s"] if item["elapsed_s"] is not None else math.inf,
                as_sequence_key(item["sequence"]),
            )
        )

    return {
        "rows": rows,
        "observed_elapsed_s": last_elapsed,
        "unique_belief_sequences": len(all_sequences),
        "malformed_json_cells": malformed_json_cells,
        "distance_increment": distance_increment,
        "angular_path_deg": angular_path,
        "angular_steps": angular_steps,
        "valid_ee_samples": valid_ee_samples,
        "belief_updates": belief_updates,
        "applied_records": applied_records,
    }


def build_transitions(
    updates: list[dict[str, Any]], summary_elapsed: float | None, observed_elapsed: float | None
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    duration = summary_elapsed if summary_elapsed and summary_elapsed > 0 else observed_elapsed
    for previous, current in zip(updates, updates[1:]):
        previous_position = [previous[f"target_{axis}_m"] for axis in "xyz"]
        current_position = [current[f"target_{axis}_m"] for axis in "xyz"]
        if any(value is None for value in previous_position + current_position):
            continue
        same_id = bool(previous.get("object_id")) and previous.get("object_id") == current.get("object_id")
        step_mm = 1000.0 * norm3(previous_position, current_position)
        dt = None
        if previous.get("elapsed_s") is not None and current.get("elapsed_s") is not None:
            candidate = current["elapsed_s"] - previous["elapsed_s"]
            dt = candidate if candidate > 0 else None
        axis_step = math.nan
        if same_id:
            previous_axis = previous.get("axis")
            current_axis = current.get("axis")
            if (
                isinstance(previous_axis, list)
                and len(previous_axis) == 3
                and isinstance(current_axis, list)
                and len(current_axis) == 3
            ):
                axis_step = unsigned_axis_angle_deg(previous_axis, current_axis)
        elapsed = current.get("elapsed_s")
        transitions.append(
            {
                "method": current["method"],
                "trial": current["trial"],
                "trial_id": current["trial_id"],
                "hand": current["hand"],
                "from_sequence": previous["sequence"],
                "to_sequence": current["sequence"],
                "elapsed_s": elapsed,
                "normalized_time_pct": (
                    100.0 * elapsed / duration if elapsed is not None and duration and duration > 0 else math.nan
                ),
                "same_object_id": same_id,
                "target_step_mm": step_mm,
                "target_speed_mm_s": step_mm / dt if dt else math.nan,
                "same_id_position_step_mm": step_mm if same_id else math.nan,
                "same_id_axis_step_deg": axis_step,
                "same_relation": (
                    previous.get("relation") == current.get("relation")
                    if previous.get("relation") is not None and current.get("relation") is not None
                    else None
                ),
            }
        )
    return transitions


def rows_for(df: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    result = df
    for column, value in filters.items():
        if column in result.columns:
            result = result[result[column] == value]
    return result


def numeric_list(df: pd.DataFrame, column: str) -> list[float]:
    if df.empty or column not in df.columns:
        return []
    return pd.to_numeric(df[column], errors="coerce").dropna().astype(float).tolist()


def aggregate_belief(
    update_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    applied_df: pd.DataFrame,
    denominator_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    transitions_with_ids = transition_df.dropna(subset=["same_object_id"]) if not transition_df.empty else transition_df
    same_id = (
        transitions_with_ids[transitions_with_ids["same_object_id"].astype(bool)]
        if not transitions_with_ids.empty
        else transitions_with_ids
    )
    id_retained = int(same_id.shape[0])
    id_total = int(transitions_with_ids.shape[0])
    denominator = sum(int(row.get("unique_belief_sequences") or 0) for row in denominator_rows)
    valid_ee = sum(int(row.get("valid_ee_samples") or 0) for row in denominator_rows)
    dominant_shares: list[float] = []
    if not update_df.empty:
        for _, group in update_df.groupby(["trial_id", "hand"], sort=False):
            ids = group["object_id"].dropna().astype(str)
            if not ids.empty:
                dominant_shares.append(100.0 * ids.value_counts().iloc[0] / len(ids))
    position_errors = numeric_list(
        applied_df[applied_df["position_applied"].astype(bool)] if not applied_df.empty else applied_df,
        "ee_position_error_mm",
    )
    orientation_errors = numeric_list(
        applied_df[applied_df["orientation_applied"].astype(bool)] if not applied_df.empty else applied_df,
        "ee_orientation_error_deg",
    )
    command_errors = numeric_list(
        applied_df[applied_df["position_applied"].astype(bool)] if not applied_df.empty else applied_df,
        "belief_to_command_error_mm",
    )
    same_id_steps = numeric_list(same_id, "same_id_position_step_mm")
    axis_steps = numeric_list(same_id, "same_id_axis_step_deg")
    position_applied_samples = len(position_errors)
    return {
        "trials_with_belief": int(update_df["trial_id"].nunique()) if not update_df.empty else 0,
        "belief_updates": int(update_df.shape[0]),
        "belief_availability_pct": 100.0 * update_df.shape[0] / denominator if denominator else math.nan,
        "id_transition_count": id_total,
        "target_id_retention_pct": 100.0 * id_retained / id_total if id_total else math.nan,
        "target_id_switches_per_100_updates": 100.0 * (id_total - id_retained) / id_total if id_total else math.nan,
        "dominant_target_share_mean_pct": mean(dominant_shares),
        "same_id_position_step_count": len(same_id_steps),
        "same_id_position_step_rmse_mm": rmse(same_id_steps),
        "same_id_position_step_median_mm": percentile(same_id_steps, 50),
        "same_id_position_step_p95_mm": percentile(same_id_steps, 95),
        "same_id_axis_step_count": len(axis_steps),
        "same_id_axis_step_median_deg": percentile(axis_steps, 50),
        "same_id_axis_step_p95_deg": percentile(axis_steps, 95),
        "joint_posterior_median": percentile(numeric_list(update_df, "joint_posterior"), 50),
        "joint_posterior_p10": percentile(numeric_list(update_df, "joint_posterior"), 10),
        "belief_age_median_ms": percentile(numeric_list(update_df, "belief_age_ms"), 50),
        "belief_age_p95_ms": percentile(numeric_list(update_df, "belief_age_ms"), 95),
        "belief_position_applied_samples": position_applied_samples,
        "belief_position_guidance_sample_pct": (
            100.0 * position_applied_samples / valid_ee if valid_ee else math.nan
        ),
        "ee_to_belief_command_position_rmse_mm": rmse(position_errors),
        "ee_to_belief_command_position_median_mm": percentile(position_errors, 50),
        "ee_to_belief_command_position_p95_mm": percentile(position_errors, 95),
        "belief_orientation_applied_samples": len(orientation_errors),
        "ee_to_belief_command_orientation_rmse_deg": rmse(orientation_errors),
        "ee_to_belief_command_orientation_p95_deg": percentile(orientation_errors, 95),
        "belief_to_command_position_rmse_mm": rmse(command_errors),
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str, int | None]]) -> str:
    headers = [label for _, label, _ in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for column, _, digits in columns:
            value = row.get(column)
            cells.append(fmt(value, digits if digits is not None else 2) if digits is not None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_method_performance(trial_df: pd.DataFrame, output: Path) -> None:
    included = trial_df[trial_df["primary_included"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    specs = [
        ("elapsed_s", "Completion time", "seconds"),
        ("ee_distance_total_m", "Cumulative EE translation", "m (left + right)"),
        ("ee_angular_total_deg", "Cumulative EE angular displacement", "degrees (left + right)"),
    ]
    x = np.arange(len(METHOD_ORDER))
    for ax, (column, title, ylabel) in zip(axes.flat[:3], specs):
        means = []
        errors = []
        for method in METHOD_ORDER:
            values = numeric_list(included[included["method"] == method], column)
            means.append(mean(values))
            errors.append(std(values))
        ax.bar(x, means, yerr=errors, capsize=4, color=[COLORS[m] for m in METHOD_ORDER], alpha=0.78)
        for index, method in enumerate(METHOD_ORDER):
            values = numeric_list(included[included["method"] == method], column)
            offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
            ax.scatter(index + np.asarray(offsets), values, s=32, color="#222222", zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, METHOD_ORDER, rotation=18, ha="right")
        style_axes(ax)
    success_ax = axes.flat[3]
    success_ax.bar(x, [100.0] * len(x), color=[COLORS[m] for m in METHOD_ORDER], alpha=0.78)
    success_ax.set_ylim(0, 108)
    success_ax.set_title("Success rate (user-provided label)")
    success_ax.set_ylabel("percent")
    success_ax.set_xticks(x, METHOD_ORDER, rotation=18, ha="right")
    for index in x:
        success_ax.text(index, 101, "100%", ha="center", fontsize=9)
    style_axes(success_ax)
    fig.suptitle("Bottle-opening method performance (complete CSV-summary pairs)", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def grouped_metric(
    ax: plt.Axes,
    belief_method_df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    lower_is_better: bool = False,
) -> None:
    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    for offset, hand, color in [(-width / 2, "left", "#56B4E9"), (width / 2, "right", "#D55E00")]:
        values = []
        for method in METHOD_ORDER:
            selected = belief_method_df[
                (belief_method_df["method"] == method) & (belief_method_df["hand"] == hand)
            ]
            values.append(float(selected.iloc[0][column]) if len(selected) else math.nan)
        ax.bar(x + offset, np.nan_to_num(values, nan=0.0), width, label=hand, color=color, alpha=0.82)
        for index, value in enumerate(values):
            if not math.isfinite(value):
                ax.text(index + offset, 0, "N/A", rotation=90, va="bottom", ha="center", fontsize=7)
    ax.set_title(title + (" (lower is better)" if lower_is_better else ""))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, METHOD_ORDER, rotation=18, ha="right")
    style_axes(ax)


def plot_belief_consistency(belief_method_df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    grouped_metric(
        axes[0, 0], belief_method_df, "target_id_retention_pct", "Target ID retention", "percent"
    )
    grouped_metric(
        axes[0, 1],
        belief_method_df,
        "same_id_position_step_rmse_mm",
        "Same-ID target position jitter (RMSE)",
        "mm per belief update",
        True,
    )
    grouped_metric(
        axes[1, 0],
        belief_method_df,
        "same_id_axis_step_median_deg",
        "Same-ID orientation-axis jitter (median)",
        "degrees per belief update",
        True,
    )
    grouped_metric(
        axes[1, 1], belief_method_df, "joint_posterior_median", "Joint posterior confidence", "probability"
    )
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Belief target consistency and precision proxies", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_belief_tracking(applied_df: pd.DataFrame, belief_method_df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specs = [
        (axes[0, 0], "ee_position_error_mm", "position_applied", "EE tracking error to belief-derived command", "mm"),
        (
            axes[0, 1],
            "ee_orientation_error_deg",
            "orientation_applied",
            "EE orientation tracking error (belief orientation active)",
            "degrees",
        ),
    ]
    for ax, column, flag, title, ylabel in specs:
        datasets = []
        positions = []
        for index, method in enumerate(METHOD_ORDER):
            selected = applied_df[
                (applied_df["method"] == method)
                & (applied_df["primary_included"])
                & (applied_df[flag].astype(bool))
            ]
            values = numeric_list(selected, column)
            if values:
                datasets.append(values)
                positions.append(index)
            else:
                ax.text(index, 0, "N/A", rotation=90, va="bottom", ha="center", fontsize=8)
        if datasets:
            boxes = ax.boxplot(datasets, positions=positions, widths=0.58, patch_artist=True, showfliers=False)
            for patch, position in zip(boxes["boxes"], positions):
                patch.set_facecolor(COLORS[METHOD_ORDER[position]])
                patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER, rotation=18, ha="right")
        style_axes(ax)

    combined = belief_method_df[belief_method_df["hand"] == "combined"].set_index("method")
    x = np.arange(len(METHOD_ORDER))
    coverage = [combined.loc[m, "belief_position_guidance_sample_pct"] if m in combined.index else math.nan for m in METHOD_ORDER]
    axes[1, 0].bar(x, np.nan_to_num(coverage, nan=0.0), color=[COLORS[m] for m in METHOD_ORDER], alpha=0.78)
    for index, value in enumerate(coverage):
        if not math.isfinite(value):
            axes[1, 0].text(index, 0, "N/A", rotation=90, va="bottom", ha="center", fontsize=8)
    axes[1, 0].set_title("Belief position-guidance coverage")
    axes[1, 0].set_ylabel("percent of valid EE samples")
    axes[1, 0].set_xticks(x, METHOD_ORDER, rotation=18, ha="right")
    style_axes(axes[1, 0])

    fidelity = [combined.loc[m, "belief_to_command_position_rmse_mm"] if m in combined.index else math.nan for m in METHOD_ORDER]
    axes[1, 1].bar(x, np.nan_to_num(fidelity, nan=0.0), color=[COLORS[m] for m in METHOD_ORDER], alpha=0.78)
    for index, value in enumerate(fidelity):
        if not math.isfinite(value):
            axes[1, 1].text(index, 0, "N/A", rotation=90, va="bottom", ha="center", fontsize=8)
    axes[1, 1].set_title("Belief target-to-command discrepancy (RMSE)")
    axes[1, 1].set_ylabel("mm (lower is better)")
    axes[1, 1].set_xticks(x, METHOD_ORDER, rotation=18, ha="right")
    style_axes(axes[1, 1])
    fig.suptitle("Execution tracking while belief guidance is active", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_diagnostics(transition_df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(len(METHOD_ORDER), 2, figsize=(15, 16), sharex=True, constrained_layout=True)
    for row_index, method in enumerate(METHOD_ORDER):
        for column_index, hand in enumerate(HANDS):
            ax = axes[row_index, column_index]
            selected = transition_df[
                (transition_df["method"] == method)
                & (transition_df["hand"] == hand)
                & (transition_df["primary_included"])
            ]
            same = selected[selected["same_object_id"].astype(bool)] if not selected.empty else selected
            switched = selected[~selected["same_object_id"].astype(bool)] if not selected.empty else selected
            if not same.empty:
                ax.scatter(
                    same["normalized_time_pct"],
                    np.maximum(same["target_step_mm"], 0.01),
                    s=12,
                    alpha=0.38,
                    color="#0072B2",
                    label="same target ID",
                )
            if not switched.empty:
                ax.scatter(
                    switched["normalized_time_pct"],
                    np.maximum(switched["target_step_mm"], 0.01),
                    s=22,
                    alpha=0.6,
                    color="#D55E00",
                    marker="x",
                    label="ID switch",
                )
            ax.set_yscale("log")
            ax.set_xlim(0, 100)
            ax.set_title(f"{method} — {hand}")
            ax.set_ylabel("target step (mm, log)")
            style_axes(ax)
    axes[-1, 0].set_xlabel("normalized session time (%)")
    axes[-1, 1].set_xlabel("normalized session time (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Belief target motion over time: stable-ID jitter vs ID switches", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_target_xy(update_df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(len(METHOD_ORDER), 2, figsize=(14, 18), constrained_layout=True)
    trial_colors = plt.get_cmap("tab10")
    for row_index, method in enumerate(METHOD_ORDER):
        for column_index, hand in enumerate(HANDS):
            ax = axes[row_index, column_index]
            selected = update_df[
                (update_df["method"] == method)
                & (update_df["hand"] == hand)
                & (update_df["primary_included"])
            ]
            for trial_index, (trial, group) in enumerate(selected.groupby("trial", sort=True)):
                group = group.sort_values("elapsed_s")
                color = trial_colors(trial_index % 10)
                ax.scatter(group["target_x_m"], group["target_y_m"], s=7, alpha=0.35, color=color, label=trial)
                previous = None
                for _, current in group.iterrows():
                    if previous is not None:
                        if previous["object_id"] == current["object_id"]:
                            ax.plot(
                                [previous["target_x_m"], current["target_x_m"]],
                                [previous["target_y_m"], current["target_y_m"]],
                                color=color,
                                alpha=0.23,
                                linewidth=0.7,
                            )
                        else:
                            ax.scatter(current["target_x_m"], current["target_y_m"], marker="x", s=18, color="#D55E00")
                    previous = current
            ax.set_title(f"{method} — {hand}")
            ax.set_xlabel("target x (m)")
            ax.set_ylabel("target y (m)")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if not selected.empty:
                ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Belief target XY trajectories (red x = target ID switch)", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(
    trial_df: pd.DataFrame,
    method_df: pd.DataFrame,
    belief_method_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    generated_files: list[str],
) -> str:
    method_table = markdown_table(
        method_df,
        [
            ("method", "방법론", None),
            ("n_analyzed_trials", "완전 trial 수", 0),
            ("elapsed_mean_s", "완료 시간 평균 (s)", 2),
            ("ee_distance_total_mean_m", "EE 이동 거리 평균 (m)", 3),
            ("ee_angular_total_mean_deg", "EE 각변위 평균 (deg)", 1),
            ("success_rate_pct", "성공률 (%)", 1),
            ("target_id_retention_pct", "Belief ID 유지율 (%)", 1),
            ("same_id_position_step_rmse_mm", "같은-ID 위치 jitter RMSE (mm)", 2),
            ("ee_to_belief_command_position_rmse_mm", "Belief 적용 시 EE RMSE (mm)", 2),
        ],
    )
    trial_table = markdown_table(
        trial_df,
        [
            ("method", "방법론", None),
            ("trial", "trial", None),
            ("status", "상태", None),
            ("elapsed_s", "시간 (s)", 2),
            ("ee_distance_total_m", "EE 거리 (m)", 3),
            ("ee_angular_total_deg", "EE 각변위 (deg)", 1),
            ("success_pct", "성공 (%)", 1),
        ],
    )
    quality_issues = quality_df[quality_df["notes"].astype(str) != ""]
    quality_table = markdown_table(
        quality_issues,
        [
            ("method", "방법론", None),
            ("trial", "trial", None),
            ("status", "상태", None),
            ("notes", "검사 결과", None),
        ],
    )

    fastest = method_df.loc[method_df["elapsed_mean_s"].idxmin(), "method"]
    shortest = method_df.loc[method_df["ee_distance_total_mean_m"].idxmin(), "method"]
    least_rotation = method_df.loc[method_df["ee_angular_total_mean_deg"].idxmin(), "method"]
    combined = belief_method_df[belief_method_df["hand"] == "combined"].copy()
    best_retention = combined.loc[combined["target_id_retention_pct"].idxmax(), "method"]
    jitter_valid = combined.dropna(subset=["same_id_position_step_rmse_mm"])
    best_jitter = jitter_valid.loc[jitter_valid["same_id_position_step_rmse_mm"].idxmin(), "method"]
    tracking_valid = combined.dropna(subset=["ee_to_belief_command_position_rmse_mm"])
    best_tracking = (
        tracking_valid.loc[tracking_valid["ee_to_belief_command_position_rmse_mm"].idxmin(), "method"]
        if not tracking_valid.empty
        else "해당 없음"
    )

    files_text = "\n".join(f"- `{name}`" for name in generated_files)
    return f"""# Bottle-open 로그 종합 분석

## 요약

완전한 CSV-summary 쌍만 방법론 간 1차 비교에 사용했다. 이 기준에서 완료 시간이 가장 짧은 방법은 **{fastest}**, 누적 EE 이동 거리가 가장 짧은 방법은 **{shortest}**, 누적 EE 각변위가 가장 작은 방법은 **{least_rotation}**이다. Belief 타겟 ID 유지율이 가장 높은 방법은 **{best_retention}**, 동일 ID 내 위치 jitter RMSE가 가장 낮은 방법은 **{best_jitter}**이다. 실제 belief 유도 명령이 적용된 방법들 중 EE 위치 추종 RMSE가 가장 낮은 방법은 **{best_tracking}**이다.

성공률은 로그에서 추론한 값이 아니라 사용자 조건(모든 완료 trial 성공)을 적용해 **모두 100%**로 기록했다.

## 방법론별 핵심 지표

평균은 완전한 CSV-summary 쌍에 대해 trial을 동일 가중치로 계산했다. 상세 평균·표준편차와 좌/우 분해 값은 `method_metrics.csv`에 있다.

{method_table}

![Method performance](method_performance_overview.png)

## Trial별 실행 지표

`complete_pair`만 방법론 평균에 포함했다. `csv_only_incomplete`와 `summary_only`는 읽을 수 있는 값은 보존하되 비교에서는 제외했다.

{trial_table}

## Belief 지표 정의와 해석

- **타겟 일관성**: 새 belief result sequence만 1회 샘플링한 뒤, 인접 업데이트의 `object_id`가 같은 비율(ID 유지율)로 정의했다. ID switch율과 dominant-target 점유율도 CSV에 함께 제공한다.
- **위치 정밀도 proxy**: 같은 `object_id`가 유지된 인접 belief 업데이트 사이의 3D 타겟 위치 변화량을 사용한다. RMSE/median/p95가 작을수록 target jitter가 작다.
- **방향 정밀도 proxy**: 같은 ID의 `orientation_axis` 변화각을 계산한다. 축의 부호 모호성을 고려해 ±axis를 동일하게 보고 0–90° unsigned angle을 사용한다.
- **confidence/latency**: joint posterior의 median/p10과 belief age의 median/p95를 제공한다.
- **실제 추종 정밀도**: `position_source=belief_contact`인 제어 샘플에서 EE와 belief-derived IPOpt command의 위치 오차 RMSE/median/p95를 계산했다. 방향은 `orientation_source=belief_minimal_rotation`일 때만 계산했다.
- 로그에는 독립적인 bottle ground truth가 없다. 따라서 jitter와 posterior는 belief의 **안정성/정밀도 proxy**이지 실제 물체 pose에 대한 절대 정확도는 아니다.
- `Alpha blend`와 `Position Only`는 belief target 레코드는 존재하지만 로그상 belief 위치가 제어 명령에 적용된 샘플이 없어 실제 EE 추종 오차는 N/A다.

![Belief consistency](belief_consistency_overview.png)

![Belief execution tracking](belief_execution_tracking.png)

![Belief temporal diagnostics](belief_temporal_diagnostics.png)

![Belief target XY trajectories](belief_target_xy_trajectories.png)

## 데이터 품질과 제외 기준

{quality_table}

추가 원칙:

- 세션 길이와 EE 이동 거리는 summary 값을 우선 사용하고 CSV 누적값과 교차검증했다.
- EE 각변위는 유효한 연속 EE rotation matrix 사이 SO(3) geodesic angle을 누적한 뒤 좌·우 합으로 정의했다.
- summary가 없는 CSV는 완료 여부를 확인할 수 없어 1차 비교에서 제외했다.
- CSV가 없는 summary는 각변위와 belief 지표를 재구성할 수 없어 1차 비교에서 제외했다.

## 생성 파일

{files_text}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Folder containing one subfolder per method")
    parser.add_argument("--output", type=Path, default=None, help="Output folder (default: ROOT/analysis)")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    csv_files = {path.relative_to(root).with_suffix("").as_posix(): path for path in root.glob("*/trial_*.csv")}
    summary_files = {
        path.relative_to(root).as_posix().removesuffix(".summary.json"): path
        for path in root.glob("*/trial_*.summary.json")
    }
    trial_keys = sorted(set(csv_files) | set(summary_files), key=lambda key: (METHOD_ORDER.index(key.split("/")[0]) if key.split("/")[0] in METHOD_ORDER else 999, key))

    trial_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    all_applied: list[dict[str, Any]] = []
    denominator_by_trial_hand: list[dict[str, Any]] = []

    for key in trial_keys:
        method, trial = key.split("/", 1)
        csv_path = csv_files.get(key)
        summary_path = summary_files.get(key)
        summary = safe_json(summary_path.read_text(encoding="utf-8")) if summary_path else None
        csv_data = read_csv_log(csv_path, method, trial) if csv_path else None
        status = "complete_pair" if csv_path and summary_path else ("csv_only_incomplete" if csv_path else "summary_only")
        primary_included = status == "complete_pair"
        summary_elapsed = finite(summary.get("elapsed_s")) if isinstance(summary, dict) else None
        observed_elapsed = finite(csv_data.get("observed_elapsed_s")) if csv_data else None
        elapsed = summary_elapsed if summary_elapsed is not None else observed_elapsed
        summary_distance = finite(summary.get("ee_distance_total_m")) if isinstance(summary, dict) else None
        csv_distance = (
            sum(csv_data["distance_increment"].values()) if csv_data is not None else None
        )
        distance = summary_distance if summary_distance is not None else csv_distance
        angular_left = csv_data["angular_path_deg"]["left"] if csv_data else math.nan
        angular_right = csv_data["angular_path_deg"]["right"] if csv_data else math.nan
        angular_total = angular_left + angular_right if csv_data else math.nan
        trial_rows.append(
            {
                "method": method,
                "trial": trial,
                "trial_id": core_trial_id(method, trial),
                "status": status,
                "primary_included": primary_included,
                "success_pct": 100.0,
                "samples_csv": csv_data["rows"] if csv_data else math.nan,
                "samples_summary": summary.get("samples") if isinstance(summary, dict) else math.nan,
                "elapsed_s": elapsed,
                "observed_csv_elapsed_s": observed_elapsed,
                "ee_distance_left_m": (
                    finite(summary.get("ee_distance_left_m"))
                    if isinstance(summary, dict)
                    else (csv_data["distance_increment"]["left"] if csv_data else math.nan)
                ),
                "ee_distance_right_m": (
                    finite(summary.get("ee_distance_right_m"))
                    if isinstance(summary, dict)
                    else (csv_data["distance_increment"]["right"] if csv_data else math.nan)
                ),
                "ee_distance_total_m": distance,
                "ee_angular_left_deg": angular_left,
                "ee_angular_right_deg": angular_right,
                "ee_angular_total_deg": angular_total,
                "valid_ee_samples_left": csv_data["valid_ee_samples"]["left"] if csv_data else math.nan,
                "valid_ee_samples_right": csv_data["valid_ee_samples"]["right"] if csv_data else math.nan,
                "unique_belief_sequences": csv_data["unique_belief_sequences"] if csv_data else math.nan,
            }
        )

        notes: list[str] = []
        declared_csv = summary.get("csv", "") if isinstance(summary, dict) else ""
        basename_match = bool(summary_path and csv_path and Path(declared_csv).name == csv_path.name)
        if status == "csv_only_incomplete":
            notes.append("summary 없음; primary 비교 제외")
        if status == "summary_only":
            notes.append("CSV 없음; 각변위/belief 분석 불가, primary 비교 제외")
        if summary_path and csv_path and not basename_match:
            notes.append(f"summary csv 필드({declared_csv})가 실제 basename과 불일치")
        sample_match = None
        if csv_data and isinstance(summary, dict):
            sample_match = int(summary.get("samples", -1)) == csv_data["rows"]
            if not sample_match:
                notes.append("summary sample 수와 CSV row 수 불일치")
        distance_difference = (
            abs(summary_distance - csv_distance)
            if summary_distance is not None and csv_distance is not None
            else math.nan
        )
        if math.isfinite(distance_difference) and distance_difference > 1e-6:
            notes.append(f"summary/CSV 거리 차이 {distance_difference:.6g} m")
        if csv_data and csv_data["rows"]:
            coverage = min(csv_data["valid_ee_samples"].values()) / csv_data["rows"]
            if coverage < 0.9:
                notes.append(f"유효 EE pose coverage {coverage * 100:.1f}%")
        if csv_data and csv_data["malformed_json_cells"]:
            notes.append(f"파싱 실패 JSON cell {csv_data['malformed_json_cells']}개")
        quality_rows.append(
            {
                "method": method,
                "trial": trial,
                "status": status,
                "csv_present": bool(csv_path),
                "summary_present": bool(summary_path),
                "primary_included": primary_included,
                "summary_declared_csv": declared_csv,
                "summary_basename_matches": basename_match if summary_path and csv_path else math.nan,
                "sample_count_matches": sample_match,
                "summary_csv_distance_abs_difference_m": distance_difference,
                "notes": "; ".join(notes),
            }
        )

        if csv_data:
            duration_for_norm = summary_elapsed if summary_elapsed is not None else observed_elapsed
            for hand in HANDS:
                denominator_by_trial_hand.append(
                    {
                        "method": method,
                        "trial": trial,
                        "trial_id": core_trial_id(method, trial),
                        "hand": hand,
                        "primary_included": primary_included,
                        "unique_belief_sequences": csv_data["unique_belief_sequences"],
                        "valid_ee_samples": csv_data["valid_ee_samples"][hand],
                    }
                )
                for update in csv_data["belief_updates"][hand]:
                    update["primary_included"] = primary_included
                    update["normalized_time_pct"] = (
                        100.0 * update["elapsed_s"] / duration_for_norm
                        if update["elapsed_s"] is not None and duration_for_norm and duration_for_norm > 0
                        else math.nan
                    )
                    all_updates.append(update)
                transitions = build_transitions(
                    csv_data["belief_updates"][hand], summary_elapsed, observed_elapsed
                )
                for transition in transitions:
                    transition["primary_included"] = primary_included
                    all_transitions.append(transition)
            for record in csv_data["applied_records"]:
                record["primary_included"] = primary_included
                all_applied.append(record)

    trial_df = pd.DataFrame(trial_rows)
    quality_df = pd.DataFrame(quality_rows)
    update_df = pd.DataFrame(all_updates)
    transition_df = pd.DataFrame(all_transitions)
    applied_df = pd.DataFrame(all_applied)
    denominator_df = pd.DataFrame(denominator_by_trial_hand)

    if update_df.empty:
        update_df = pd.DataFrame(columns=["method", "trial", "trial_id", "hand", "primary_included"])
    if transition_df.empty:
        transition_df = pd.DataFrame(columns=["method", "trial", "trial_id", "hand", "primary_included", "same_object_id"])
    if applied_df.empty:
        applied_df = pd.DataFrame(columns=["method", "trial", "trial_id", "hand", "primary_included", "position_applied", "orientation_applied"])

    trial_belief_rows: list[dict[str, Any]] = []
    for _, trial_row in trial_df.iterrows():
        trial_id = trial_row["trial_id"]
        for hand in HANDS + ["combined"]:
            hands = HANDS if hand == "combined" else [hand]
            updates = update_df[(update_df["trial_id"] == trial_id) & (update_df["hand"].isin(hands))]
            transitions = transition_df[
                (transition_df["trial_id"] == trial_id) & (transition_df["hand"].isin(hands))
            ]
            applied = applied_df[(applied_df["trial_id"] == trial_id) & (applied_df["hand"].isin(hands))]
            denominators = denominator_df[
                (denominator_df["trial_id"] == trial_id) & (denominator_df["hand"].isin(hands))
            ].to_dict("records")
            trial_belief_rows.append(
                {
                    "method": trial_row["method"],
                    "trial": trial_row["trial"],
                    "trial_id": trial_id,
                    "hand": hand,
                    "primary_included": bool(trial_row["primary_included"]),
                    **aggregate_belief(updates, transitions, applied, denominators),
                }
            )
    trial_belief_df = pd.DataFrame(trial_belief_rows)

    belief_method_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for hand in HANDS + ["combined"]:
            hands = HANDS if hand == "combined" else [hand]
            updates = update_df[
                (update_df["method"] == method)
                & (update_df["hand"].isin(hands))
                & (update_df["primary_included"])
            ]
            transitions = transition_df[
                (transition_df["method"] == method)
                & (transition_df["hand"].isin(hands))
                & (transition_df["primary_included"])
            ]
            applied = applied_df[
                (applied_df["method"] == method)
                & (applied_df["hand"].isin(hands))
                & (applied_df["primary_included"])
            ]
            denominators = denominator_df[
                (denominator_df["method"] == method)
                & (denominator_df["hand"].isin(hands))
                & (denominator_df["primary_included"])
            ].to_dict("records")
            belief_method_rows.append(
                {"method": method, "hand": hand, **aggregate_belief(updates, transitions, applied, denominators)}
            )
    belief_method_df = pd.DataFrame(belief_method_rows)

    method_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        trials = trial_df[(trial_df["method"] == method) & (trial_df["primary_included"])]
        belief = belief_method_df[
            (belief_method_df["method"] == method) & (belief_method_df["hand"] == "combined")
        ].iloc[0]
        row: dict[str, Any] = {
            "method": method,
            "n_analyzed_trials": int(len(trials)),
            "n_csv_only": int(((trial_df["method"] == method) & (trial_df["status"] == "csv_only_incomplete")).sum()),
            "n_summary_only": int(((trial_df["method"] == method) & (trial_df["status"] == "summary_only")).sum()),
            "success_rate_pct": 100.0,
        }
        for source, label in [
            ("elapsed_s", "elapsed"),
            ("ee_distance_left_m", "ee_distance_left"),
            ("ee_distance_right_m", "ee_distance_right"),
            ("ee_distance_total_m", "ee_distance_total"),
            ("ee_angular_left_deg", "ee_angular_left"),
            ("ee_angular_right_deg", "ee_angular_right"),
            ("ee_angular_total_deg", "ee_angular_total"),
        ]:
            values = numeric_list(trials, source)
            row[f"{label}_mean_{'s' if label == 'elapsed' else ('m' if 'distance' in label else 'deg')}"] = mean(values)
            row[f"{label}_sd_{'s' if label == 'elapsed' else ('m' if 'distance' in label else 'deg')}"] = std(values)
        for column in [
            "belief_updates",
            "belief_availability_pct",
            "target_id_retention_pct",
            "target_id_switches_per_100_updates",
            "dominant_target_share_mean_pct",
            "same_id_position_step_rmse_mm",
            "same_id_position_step_median_mm",
            "same_id_position_step_p95_mm",
            "same_id_axis_step_median_deg",
            "same_id_axis_step_p95_deg",
            "joint_posterior_median",
            "joint_posterior_p10",
            "belief_age_median_ms",
            "belief_age_p95_ms",
            "belief_position_guidance_sample_pct",
            "ee_to_belief_command_position_rmse_mm",
            "ee_to_belief_command_position_p95_mm",
            "ee_to_belief_command_orientation_rmse_deg",
            "ee_to_belief_command_orientation_p95_deg",
            "belief_to_command_position_rmse_mm",
        ]:
            row[column] = belief[column]
        method_rows.append(row)
    method_df = pd.DataFrame(method_rows)

    # Stable display/output ordering.
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    trial_df["_rank"] = trial_df["method"].map(method_rank)
    trial_df = trial_df.sort_values(["_rank", "trial"]).drop(columns="_rank")
    quality_df["_rank"] = quality_df["method"].map(method_rank)
    quality_df = quality_df.sort_values(["_rank", "trial"]).drop(columns="_rank")

    csv_outputs = {
        "trial_metrics.csv": trial_df,
        "method_metrics.csv": method_df,
        "belief_metrics_by_trial_hand.csv": trial_belief_df,
        "belief_metrics_by_method_hand.csv": belief_method_df,
        "belief_updates.csv": update_df.drop(columns=["axis"], errors="ignore"),
        "belief_transitions.csv": transition_df,
        "belief_applied_tracking_samples.csv": applied_df,
        "data_quality_audit.csv": quality_df,
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output / filename, index=False, float_format="%.6f")

    plot_method_performance(trial_df, output / "method_performance_overview.png")
    plot_belief_consistency(belief_method_df, output / "belief_consistency_overview.png")
    plot_belief_tracking(applied_df, belief_method_df, output / "belief_execution_tracking.png")
    plot_temporal_diagnostics(transition_df, output / "belief_temporal_diagnostics.png")
    plot_target_xy(update_df, output / "belief_target_xy_trajectories.png")

    script_target = output / "analyze_bottle_open.py"
    source_script = Path(__file__).resolve()
    if source_script != script_target.resolve():
        shutil.copy2(source_script, script_target)

    generated_files = list(csv_outputs) + [
        "method_performance_overview.png",
        "belief_consistency_overview.png",
        "belief_execution_tracking.png",
        "belief_temporal_diagnostics.png",
        "belief_target_xy_trajectories.png",
        "analyze_bottle_open.py",
        "analysis_report.md",
    ]
    report = build_report(trial_df, method_df, belief_method_df, quality_df, generated_files)
    (output / "analysis_report.md").write_text(report, encoding="utf-8")
    print(f"Analyzed {len(trial_df)} trial records ({int(trial_df['primary_included'].sum())} complete pairs).")
    print(f"Wrote {len(generated_files)} files to {output}")


if __name__ == "__main__":
    main()
