"""Append-only CSV recording for remote two-hand TASC orientation telemetry."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


CSV_FIELDS = (
    "published_timestamp_us",
    "source_rgb_timestamp_us",
    "remote_sequence",
    "result_valid",
    "selected_hand",
    "selection_active",
    "side",
    "hand_armed",
    "hand_alpha",
    "object_id",
    "label",
    "relation",
    "track_id",
    "candidate_source",
    "candidate_selected",
    "candidate_probability",
    "candidate_advantage_raw",
    "candidate_advantage_ema",
    "candidate_advantage_standardized",
    "candidate_label_prior",
    "candidate_fit_quality",
    "candidate_position_x",
    "candidate_position_y",
    "candidate_position_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "normal_dot_hand_axis",
    "normal_alignment_error",
    "normal_align_advantage",
    "normal_hold_advantage",
    "normal_converge_advantage",
    "normal_omega_norm",
    "normal_weighted_total",
    "hand_axis_x",
    "hand_axis_y",
    "hand_axis_z",
    "hand_omega_x",
    "hand_omega_y",
    "hand_omega_z",
    "orientation_selected_index",
    "orientation_selected_track_id",
    "orientation_held_probability",
    "orientation_leader_probability",
    "orientation_switch_dwell_count",
    "target_object_id",
    "target_grip_index",
    "target_object_confidence",
    "target_joint_confidence",
    "target_position_x",
    "target_position_y",
    "target_position_z",
    "target_axis_x",
    "target_axis_y",
    "target_axis_z",
    "target_relation",
    "target_orientation_confidence",
)


def _vector(value: Any, length: int) -> list[float | None]:
    if value is None:
        return [None] * length
    try:
        array = np.asarray(value, dtype=np.float64).reshape(length)
    except (TypeError, ValueError):
        return [None] * length
    if not np.isfinite(array).all():
        return [None] * length
    return [float(item) for item in array]


def _number(value: Any) -> float | int | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    return None


class RemoteBeliefCsvRecorder:
    """Write one row per hand/orientation candidate for every telemetry tick."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        if not exists:
            self._writer.writeheader()
            self._handle.flush()
        self.rows_written = 0

    def close(self) -> None:
        self._handle.close()

    def record(self, telemetry: dict[str, Any]) -> int:
        """Append candidate rows; no-candidate hand states still get one row."""
        base = {
            "published_timestamp_us": telemetry.get("published_timestamp_us"),
            "source_rgb_timestamp_us": telemetry.get("source_rgb_timestamp_us"),
            "remote_sequence": telemetry.get("remote_sequence"),
            "result_valid": telemetry.get("result_valid"),
            "selected_hand": telemetry.get("selected_hand", "none"),
            "selection_active": telemetry.get("selection_active", False),
        }
        rows = 0
        hands = telemetry.get("hands") if isinstance(telemetry.get("hands"), dict) else {}
        for side in ("left", "right"):
            hand = hands.get(side) if isinstance(hands.get(side), dict) else {}
            hand_state = hand.get("hand") if isinstance(hand.get("hand"), dict) else {}
            orientation = hand.get("orientation") if isinstance(hand.get("orientation"), dict) else {}
            target = hand.get("target") if isinstance(hand.get("target"), dict) else {}
            target_position = _vector(target.get("position_goal_m"), 3)
            target_axis = _vector(target.get("orientation_axis_world"), 3)
            common = {
                **base,
                "side": side,
                "hand_armed": hand_state.get("armed", False),
                "hand_alpha": _number(hand_state.get("alpha")),
                "hand_axis_x": _vector(orientation.get("hand_axis_world"), 3)[0],
                "hand_axis_y": _vector(orientation.get("hand_axis_world"), 3)[1],
                "hand_axis_z": _vector(orientation.get("hand_axis_world"), 3)[2],
                "hand_omega_x": _vector(orientation.get("hand_omega_rps"), 3)[0],
                "hand_omega_y": _vector(orientation.get("hand_omega_rps"), 3)[1],
                "hand_omega_z": _vector(orientation.get("hand_omega_rps"), 3)[2],
                "orientation_selected_index": orientation.get("selected_index"),
                "orientation_selected_track_id": orientation.get("selected_track_id"),
                "orientation_held_probability": _number(orientation.get("held_probability")),
                "orientation_leader_probability": _number(orientation.get("leader_probability")),
                "orientation_switch_dwell_count": orientation.get("switch_dwell_count"),
                "target_object_id": target.get("object_id"),
                "target_grip_index": target.get("grip_index"),
                "target_object_confidence": _number(target.get("object_confidence")),
                "target_joint_confidence": _number(target.get("joint_confidence")),
                "target_position_x": target_position[0],
                "target_position_y": target_position[1],
                "target_position_z": target_position[2],
                "target_axis_x": target_axis[0],
                "target_axis_y": target_axis[1],
                "target_axis_z": target_axis[2],
                "target_relation": target.get("orientation_relation"),
                "target_orientation_confidence": _number(target.get("orientation_confidence")),
            }
            candidates = orientation.get("candidates") if isinstance(orientation.get("candidates"), list) else []
            if not candidates:
                self._writer.writerow(common)
                rows += 1
                continue
            selected_index = orientation.get("selected_index")
            for index, candidate in enumerate(candidates):
                candidate = candidate if isinstance(candidate, dict) else {}
                normal = _vector(candidate.get("normal"), 3)
                position = _vector(candidate.get("position_goal_m"), 3)
                breakdown = candidate.get("breakdown") if isinstance(candidate.get("breakdown"), dict) else {}
                row = {
                    **common,
                    "object_id": candidate.get("object_id"),
                    "label": candidate.get("label"),
                    "relation": candidate.get("relation"),
                    "track_id": candidate.get("track_id"),
                    "candidate_source": candidate.get("source"),
                    "candidate_selected": bool(index == selected_index),
                    "candidate_probability": _number(candidate.get("probability")),
                    "candidate_advantage_raw": _number(candidate.get("advantage_raw")),
                    "candidate_advantage_ema": _number(candidate.get("advantage_ema")),
                    "candidate_advantage_standardized": _number(candidate.get("advantage_standardized")),
                    "candidate_label_prior": _number(candidate.get("label_prior")),
                    "candidate_fit_quality": _number(candidate.get("fit_quality")),
                    "candidate_position_x": position[0],
                    "candidate_position_y": position[1],
                    "candidate_position_z": position[2],
                    "normal_x": normal[0],
                    "normal_y": normal[1],
                    "normal_z": normal[2],
                    "normal_dot_hand_axis": _number(breakdown.get("cos_c")),
                    "normal_alignment_error": _number(breakdown.get("alignment_error")),
                    "normal_align_advantage": _number(breakdown.get("align_advantage")),
                    "normal_hold_advantage": _number(breakdown.get("hold_advantage")),
                    "normal_converge_advantage": _number(breakdown.get("converge_advantage")),
                    "normal_omega_norm": _number(breakdown.get("omega_norm")),
                    "normal_weighted_total": _number(breakdown.get("weighted_total")),
                }
                self._writer.writerow(row)
                rows += 1
        self._handle.flush()
        self.rows_written += rows
        return rows
