"""Append-only, candidate-level CSV recording for remote two-hand TASC belief."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


ROBOT_Q_FIELDS = tuple(f"robot_q_{index}" for index in range(29))
CSV_SCHEMA_VERSION = 2

FEATURE_FIELDS = (
    "feature_target_x",
    "feature_target_y",
    "feature_target_z",
    "feature_eef_after_x",
    "feature_eef_after_y",
    "feature_eef_after_z",
    "feature_distance_m",
    "feature_distance_after_action_m",
    "feature_distance_value",
    "feature_distance_value_after_action",
    "feature_distance_advantage",
    "feature_progress_advantage",
    "feature_agreement_cost",
    "feature_agreement_advantage",
    "feature_agreement_direction_x",
    "feature_agreement_direction_y",
    "feature_agreement_direction_z",
    "feature_agreement_u_opt_x",
    "feature_agreement_u_opt_y",
    "feature_agreement_u_opt_z",
    "feature_shoulder_position_x",
    "feature_shoulder_position_y",
    "feature_shoulder_position_z",
    "feature_shoulder_target_distance_m",
    "feature_reach_advantage",
    "feature_delta_advantage",
    "feature_alignment_advantage",
    "feature_contact_fit_quality",
    "feature_contact_normal_x",
    "feature_contact_normal_y",
    "feature_contact_normal_z",
    "feature_hand_approach_axis_x",
    "feature_hand_approach_axis_y",
    "feature_hand_approach_axis_z",
    "feature_weight_distance",
    "feature_weight_ray",
    "feature_weight_progress",
    "feature_weight_agreement",
    "feature_weight_reach",
    "feature_weight_delta",
    "feature_weight_align",
    "feature_contribution_distance",
    "feature_contribution_ray",
    "feature_contribution_progress",
    "feature_contribution_agreement",
    "feature_contribution_reach",
    "feature_contribution_delta",
    "feature_contribution_align",
    "feature_weighted_total",
)


CSV_FIELDS = (
    "schema_version",
    "record_type",
    "published_timestamp_us",
    "source_rgb_timestamp_us",
    "source_depth_timestamp_us",
    "remote_published_timestamp_us",
    "remote_to_gateway_latency_ms",
    "remote_sequence",
    "result_valid",
    "backend",
    "inference_ms",
    "image_width",
    "image_height",
    "selected_hand",
    "selection_active",
    "belief_reason",
    "belief_robot_state_timestamp_us",
    "source_to_belief_state_delta_ms",
    "source_to_belief_state_abs_delta_ms",
    "belief_hand_model",
    "grasp_center_local_x",
    "grasp_center_local_y",
    "grasp_center_local_z",
    "base_quat_w",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    *ROBOT_Q_FIELDS,
    "remote_object_count",
    "side",
    "hand_armed",
    "hand_alpha",
    "hand_beta",
    "hand_lambda0",
    "belief_update_dt_s",
    "belief_w_sc",
    "belief_max_prob_any_goal",
    "grip_update_applied",
    "grip_update_hz",
    "feature_ema_alpha",
    "orientation_beta",
    "orientation_w_sc",
    "orientation_max_prob",
    "orientation_advantage_scale",
    "eef_position_x",
    "eef_position_y",
    "eef_position_z",
    "eef_velocity_x",
    "eef_velocity_y",
    "eef_velocity_z",
    "hand_approach_axis_x",
    "hand_approach_axis_y",
    "hand_approach_axis_z",
    "hand_rotation_r00",
    "hand_rotation_r01",
    "hand_rotation_r02",
    "hand_rotation_r10",
    "hand_rotation_r11",
    "hand_rotation_r12",
    "hand_rotation_r20",
    "hand_rotation_r21",
    "hand_rotation_r22",
    "belief_object_count",
    "orientation_candidate_count",
    "object_lambda_dt",
    "object_advantage_variance",
    "orientation_lambda_dt",
    "orientation_advantage_variance",
    # Orientation-candidate record fields (kept for compatibility).
    "object_id",
    "label",
    "relation",
    "track_id",
    "candidate_source",
    "candidate_role",
    "candidate_primitive_kind",
    "candidate_matched_previous",
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
    # Selected target fields live on the hand record.
    "target_object_id",
    "target_grip_index",
    "target_object_confidence",
    "target_grip_confidence",
    "target_joint_confidence",
    "target_position_x",
    "target_position_y",
    "target_position_z",
    "target_axis_x",
    "target_axis_y",
    "target_axis_z",
    "target_relation",
    "target_orientation_confidence",
    # Object-candidate record fields.
    "belief_object_index",
    "belief_object_id",
    "belief_object_label",
    "belief_object_score",
    "belief_object_selected",
    "belief_object_probability",
    "belief_object_advantage_raw",
    "belief_object_advantage_ema",
    "belief_object_advantage_standardized",
    "generated_grip_candidate_count",
    "persistent_grip_candidate_count",
    # Source geometry associated with an object/grip record.
    "source_robot_state_delta_ms",
    "geometry_source_timestamp_us",
    "geometry_age_at_source_ms",
    "geometry_fallback",
    "geometry_missing_frames",
    "remote_valid_depth_points",
    "remote_center_camera_x",
    "remote_center_camera_y",
    "remote_center_camera_z",
    "remote_center_goal_x",
    "remote_center_goal_y",
    "remote_center_goal_z",
    "remote_target_contact_x",
    "remote_target_contact_y",
    "remote_target_contact_z",
    # Grip-candidate record fields.
    "grip_candidate_index",
    "grip_track_id",
    "grip_source",
    "grip_selected",
    "grip_probability",
    "grip_advantage_raw",
    "grip_advantage_ema",
    "grip_advantage_standardized",
    "grip_center_x",
    "grip_center_y",
    "grip_center_z",
    "grip_approach_x",
    "grip_approach_y",
    "grip_approach_z",
    "grip_surface_normal_x",
    "grip_surface_normal_y",
    "grip_surface_normal_z",
    "grip_surface_tangent_x",
    "grip_surface_tangent_y",
    "grip_surface_tangent_z",
    "grip_surface_axis_x",
    "grip_surface_axis_y",
    "grip_surface_axis_z",
    "grip_width_m",
    "grip_geom_score",
    *FEATURE_FIELDS,
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


def _xyz(prefix: str, value: Any) -> dict[str, float | None]:
    vector = _vector(value, 3)
    return {
        f"{prefix}_x": vector[0],
        f"{prefix}_y": vector[1],
        f"{prefix}_z": vector[2],
    }


def _feature_columns(value: Any) -> dict[str, Any]:
    breakdown = value if isinstance(value, dict) else {}
    weights = breakdown.get("weights") if isinstance(breakdown.get("weights"), dict) else {}
    contributions = (
        breakdown.get("weighted_contributions")
        if isinstance(breakdown.get("weighted_contributions"), dict)
        else {}
    )
    result: dict[str, Any] = {
        **_xyz("feature_target", breakdown.get("target_m")),
        **_xyz("feature_eef_after", breakdown.get("eef_after_action_m")),
        **_xyz("feature_contact_normal", breakdown.get("contact_normal")),
        **_xyz(
            "feature_hand_approach_axis", breakdown.get("hand_approach_axis")
        ),
        **_xyz(
            "feature_agreement_direction", breakdown.get("agreement_direction")
        ),
        **_xyz(
            "feature_agreement_u_opt", breakdown.get("agreement_u_opt_mps")
        ),
        **_xyz("feature_shoulder_position", breakdown.get("shoulder_position_m")),
        "feature_distance_m": _number(breakdown.get("distance_m")),
        "feature_distance_after_action_m": _number(
            breakdown.get("distance_after_action_m")
        ),
        "feature_distance_value": _number(breakdown.get("distance_value")),
        "feature_distance_value_after_action": _number(
            breakdown.get("distance_value_after_action")
        ),
        "feature_distance_advantage": _number(
            breakdown.get("distance_advantage")
        ),
        "feature_progress_advantage": _number(
            breakdown.get("progress_advantage")
        ),
        "feature_agreement_cost": _number(breakdown.get("agreement_cost")),
        "feature_agreement_advantage": _number(
            breakdown.get("agreement_advantage")
        ),
        "feature_shoulder_target_distance_m": _number(
            breakdown.get("shoulder_target_distance_m")
        ),
        "feature_reach_advantage": _number(breakdown.get("reach_advantage")),
        "feature_delta_advantage": _number(breakdown.get("delta_advantage")),
        "feature_alignment_advantage": _number(
            breakdown.get("alignment_advantage")
        ),
        "feature_contact_fit_quality": _number(
            breakdown.get("contact_fit_quality")
        ),
        "feature_weighted_total": _number(breakdown.get("weighted_total")),
    }
    for name in ("distance", "ray", "progress", "agreement", "reach", "delta", "align"):
        result[f"feature_weight_{name}"] = _number(weights.get(name))
        result[f"feature_contribution_{name}"] = _number(
            contributions.get(name)
        )
    return result


class RemoteBeliefCsvRecorder:
    """Write normalized hand and candidate records for every telemetry tick."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                existing_fields = tuple(next(csv.reader(handle), []))
            if existing_fields != CSV_FIELDS:
                raise ValueError(
                    "belief CSV schema mismatch; choose a new output path "
                    f"for schema v{CSV_SCHEMA_VERSION}: {self.path}"
                )
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        if not exists:
            self._writer.writeheader()
            self._handle.flush()
        self.rows_written = 0

    def close(self) -> None:
        self._handle.close()

    def record(self, telemetry: dict[str, Any]) -> int:
        """Append normalized hand, orientation, object, and grip records."""
        published_timestamp_us = _number(telemetry.get("published_timestamp_us"))
        remote_published_timestamp_us = _number(
            telemetry.get("remote_published_timestamp_us")
        )
        remote_to_gateway_latency_ms = None
        if published_timestamp_us and remote_published_timestamp_us:
            remote_to_gateway_latency_ms = (
                float(published_timestamp_us) - float(remote_published_timestamp_us)
            ) / 1000.0

        source_rgb_timestamp_us = _number(telemetry.get("source_rgb_timestamp_us"))
        belief_state_timestamp_us = _number(
            telemetry.get("belief_robot_state_timestamp_us")
        )
        source_to_state_delta_ms = None
        if source_rgb_timestamp_us and belief_state_timestamp_us:
            source_to_state_delta_ms = (
                float(belief_state_timestamp_us) - float(source_rgb_timestamp_us)
            ) / 1000.0

        remote_objects = (
            telemetry.get("remote_objects")
            if isinstance(telemetry.get("remote_objects"), list)
            else []
        )
        remote_by_id = {
            str(obj.get("track_id")): obj
            for obj in remote_objects
            if isinstance(obj, dict) and obj.get("track_id") is not None
        }
        grasp_center = _vector(telemetry.get("belief_grasp_center_local_m"), 3)
        base_quat = _vector(telemetry.get("belief_base_quat_wxyz"), 4)
        base = {
            "schema_version": CSV_SCHEMA_VERSION,
            "published_timestamp_us": published_timestamp_us,
            "source_rgb_timestamp_us": source_rgb_timestamp_us,
            "source_depth_timestamp_us": telemetry.get("source_depth_timestamp_us"),
            "remote_published_timestamp_us": remote_published_timestamp_us,
            "remote_to_gateway_latency_ms": remote_to_gateway_latency_ms,
            "remote_sequence": telemetry.get("remote_sequence"),
            "result_valid": telemetry.get("result_valid"),
            "backend": telemetry.get("backend"),
            "inference_ms": _number(telemetry.get("inference_ms")),
            "image_width": telemetry.get("image_width"),
            "image_height": telemetry.get("image_height"),
            "selected_hand": telemetry.get("selected_hand", "none"),
            "selection_active": telemetry.get("selection_active", False),
            "belief_reason": telemetry.get("reason"),
            "belief_robot_state_timestamp_us": belief_state_timestamp_us,
            "source_to_belief_state_delta_ms": source_to_state_delta_ms,
            "source_to_belief_state_abs_delta_ms": (
                None
                if source_to_state_delta_ms is None
                else abs(source_to_state_delta_ms)
            ),
            "belief_hand_model": telemetry.get("belief_hand_model"),
            "grasp_center_local_x": grasp_center[0],
            "grasp_center_local_y": grasp_center[1],
            "grasp_center_local_z": grasp_center[2],
            "base_quat_w": base_quat[0],
            "base_quat_x": base_quat[1],
            "base_quat_y": base_quat[2],
            "base_quat_z": base_quat[3],
            "remote_object_count": len(remote_objects),
        }
        robot_q = telemetry.get("belief_robot_q")
        if isinstance(robot_q, (list, tuple, np.ndarray)):
            robot_q_values = np.asarray(robot_q, dtype=np.float64).reshape(-1)
        else:
            robot_q_values = np.empty(0, dtype=np.float64)
        for index, field in enumerate(ROBOT_Q_FIELDS):
            base[field] = (
                float(robot_q_values[index])
                if index < robot_q_values.size
                and np.isfinite(robot_q_values[index])
                else None
            )

        rows = 0
        hands = telemetry.get("hands") if isinstance(telemetry.get("hands"), dict) else {}
        for side in ("left", "right"):
            hand = hands.get(side) if isinstance(hands.get(side), dict) else {}
            hand_state = hand.get("hand") if isinstance(hand.get("hand"), dict) else {}
            orientation = hand.get("orientation") if isinstance(hand.get("orientation"), dict) else {}
            target = hand.get("target") if isinstance(hand.get("target"), dict) else {}
            objects = hand.get("objects") if isinstance(hand.get("objects"), list) else []
            update = hand.get("update") if isinstance(hand.get("update"), dict) else {}
            jump = hand.get("jump") if isinstance(hand.get("jump"), dict) else {}
            eef_position = _vector(hand.get("eef_position_m"), 3)
            eef_velocity = _vector(hand.get("eef_velocity_mps"), 3)
            hand_approach = _vector(hand.get("hand_approach_axis"), 3)
            hand_rotation = _vector(hand.get("hand_rotation_matrix"), 9)
            orientation_candidates = (
                orientation.get("candidates")
                if isinstance(orientation.get("candidates"), list)
                else []
            )
            target_position = _vector(target.get("position_goal_m"), 3)
            target_axis = _vector(target.get("orientation_axis_world"), 3)
            target_grip_confidence = None
            target_object = next(
                (
                    obj
                    for obj in objects
                    if isinstance(obj, dict)
                    and str(obj.get("object_id")) == str(target.get("object_id"))
                ),
                None,
            )
            if isinstance(target_object, dict):
                probabilities = target_object.get("grip_probabilities")
                grip_index = target.get("grip_index")
                if isinstance(probabilities, list) and isinstance(grip_index, int):
                    if 0 <= grip_index < len(probabilities):
                        target_grip_confidence = _number(probabilities[grip_index])
            common = {
                **base,
                "side": side,
                "hand_armed": hand_state.get("armed", False),
                "hand_alpha": _number(hand_state.get("alpha")),
                "hand_beta": _number(hand_state.get("beta")),
                "hand_lambda0": _number(hand_state.get("lambda0")),
                "belief_update_dt_s": _number(update.get("dt_s")),
                "belief_w_sc": _number(update.get("w_sc")),
                "belief_max_prob_any_goal": _number(
                    update.get("max_prob_any_goal")
                ),
                "grip_update_applied": update.get("grip_update_applied"),
                "grip_update_hz": _number(update.get("grip_update_hz")),
                "feature_ema_alpha": _number(update.get("feature_ema_alpha")),
                "orientation_beta": _number(update.get("orientation_beta")),
                "orientation_w_sc": _number(update.get("orientation_w_sc")),
                "orientation_max_prob": _number(update.get("orientation_max_prob")),
                "orientation_advantage_scale": _number(
                    update.get("orientation_advantage_scale")
                ),
                "eef_position_x": eef_position[0],
                "eef_position_y": eef_position[1],
                "eef_position_z": eef_position[2],
                "eef_velocity_x": eef_velocity[0],
                "eef_velocity_y": eef_velocity[1],
                "eef_velocity_z": eef_velocity[2],
                "hand_approach_axis_x": hand_approach[0],
                "hand_approach_axis_y": hand_approach[1],
                "hand_approach_axis_z": hand_approach[2],
                **{
                    field: hand_rotation[index]
                    for index, field in enumerate(
                        (
                            "hand_rotation_r00", "hand_rotation_r01", "hand_rotation_r02",
                            "hand_rotation_r10", "hand_rotation_r11", "hand_rotation_r12",
                            "hand_rotation_r20", "hand_rotation_r21", "hand_rotation_r22",
                        )
                    )
                },
                "belief_object_count": len(objects),
                "orientation_candidate_count": len(orientation_candidates),
                "object_lambda_dt": _number(jump.get("object_lambda_dt")),
                "object_advantage_variance": _number(
                    jump.get("object_advantage_variance")
                ),
                "orientation_lambda_dt": _number(
                    jump.get("orientation_lambda_dt")
                ),
                "orientation_advantage_variance": _number(
                    jump.get("orientation_advantage_variance")
                ),
            }
            # Joint angles are sufficient to reproduce FK and are intentionally
            # stored once per hand tick instead of repeated on every candidate.
            candidate_common = {
                **common,
                **{field: None for field in ROBOT_Q_FIELDS},
            }

            hand_axis = _vector(orientation.get("hand_axis_world"), 3)
            hand_omega = _vector(orientation.get("hand_omega_rps"), 3)
            hand_row = {
                **common,
                "record_type": "hand",
                "hand_axis_x": hand_axis[0],
                "hand_axis_y": hand_axis[1],
                "hand_axis_z": hand_axis[2],
                "hand_omega_x": hand_omega[0],
                "hand_omega_y": hand_omega[1],
                "hand_omega_z": hand_omega[2],
                "orientation_selected_index": orientation.get("selected_index"),
                "orientation_selected_track_id": orientation.get("selected_track_id"),
                "orientation_held_probability": _number(orientation.get("held_probability")),
                "orientation_leader_probability": _number(orientation.get("leader_probability")),
                "orientation_switch_dwell_count": orientation.get("switch_dwell_count"),
                "target_object_id": target.get("object_id"),
                "target_grip_index": target.get("grip_index"),
                "target_object_confidence": _number(target.get("object_confidence")),
                "target_grip_confidence": target_grip_confidence,
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
            self._writer.writerow(hand_row)
            rows += 1

            selected_index = orientation.get("selected_index")
            for index, candidate in enumerate(orientation_candidates):
                candidate = candidate if isinstance(candidate, dict) else {}
                normal = _vector(candidate.get("normal"), 3)
                position = _vector(candidate.get("position_goal_m"), 3)
                breakdown = candidate.get("breakdown") if isinstance(candidate.get("breakdown"), dict) else {}
                row = {
                    **candidate_common,
                    "record_type": "orientation",
                    "object_id": candidate.get("object_id"),
                    "label": candidate.get("label"),
                    "relation": candidate.get("relation"),
                    "track_id": candidate.get("track_id"),
                    "candidate_source": candidate.get("source"),
                    "candidate_role": candidate.get("role"),
                    "candidate_primitive_kind": candidate.get("primitive_kind"),
                    "candidate_matched_previous": candidate.get(
                        "matched_previous"
                    ),
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

            for object_index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue
                object_id = str(obj.get("object_id", ""))
                remote = remote_by_id.get(object_id, {})
                geometry_timestamp_us = _number(
                    remote.get("geometry_source_timestamp_us")
                )
                geometry_age_ms = None
                if source_rgb_timestamp_us and geometry_timestamp_us:
                    geometry_age_ms = (
                        float(source_rgb_timestamp_us) - float(geometry_timestamp_us)
                    ) / 1000.0
                remote_fields = {
                    "source_robot_state_delta_ms": _number(
                        remote.get("source_robot_state_delta_ms")
                    ),
                    "geometry_source_timestamp_us": geometry_timestamp_us,
                    "geometry_age_at_source_ms": geometry_age_ms,
                    "geometry_fallback": remote.get("geometry_fallback"),
                    "geometry_missing_frames": remote.get("missing_frames"),
                    "remote_valid_depth_points": remote.get("valid_depth_points"),
                    **_xyz("remote_center_camera", remote.get("center_camera_m")),
                    **_xyz("remote_center_goal", remote.get("center_goal_m")),
                    **_xyz(
                        "remote_target_contact",
                        remote.get("target_contact_goal_m"),
                    ),
                }
                object_fields = {
                    "belief_object_index": object_index,
                    "belief_object_id": object_id,
                    "belief_object_label": obj.get("label"),
                    "belief_object_score": _number(obj.get("score")),
                    "belief_object_selected": object_id
                    == str(target.get("object_id", "")),
                    "belief_object_probability": _number(
                        obj.get("object_probability")
                    ),
                    "belief_object_advantage_raw": _number(
                        obj.get("object_advantage_raw")
                    ),
                    "belief_object_advantage_ema": _number(
                        obj.get("object_advantage_ema")
                    ),
                    "belief_object_advantage_standardized": _number(
                        obj.get("object_advantage_standardized")
                    ),
                    "generated_grip_candidate_count": obj.get(
                        "generated_grip_candidate_count"
                    ),
                    "persistent_grip_candidate_count": obj.get(
                        "persistent_grip_candidate_count"
                    ),
                }
                self._writer.writerow(
                    {
                        **candidate_common,
                        "record_type": "object",
                        **object_fields,
                        **remote_fields,
                        **_feature_columns(obj.get("object_feature_breakdown")),
                    }
                )
                rows += 1

                grip_candidates = (
                    obj.get("grip_candidates")
                    if isinstance(obj.get("grip_candidates"), list)
                    else []
                )
                selected_grip_index = obj.get("selected_grip_index")
                for grip_index, grip in enumerate(grip_candidates):
                    if not isinstance(grip, dict):
                        continue
                    self._writer.writerow(
                        {
                            **candidate_common,
                            "record_type": "grip",
                            **object_fields,
                            **remote_fields,
                            "grip_candidate_index": grip_index,
                            "grip_track_id": grip.get("track_id"),
                            "grip_source": grip.get("source"),
                            "grip_selected": grip_index == selected_grip_index,
                            "grip_probability": _number(grip.get("probability")),
                            "grip_advantage_raw": _number(
                                grip.get("advantage_raw")
                            ),
                            "grip_advantage_ema": _number(
                                grip.get("advantage_ema")
                            ),
                            "grip_advantage_standardized": _number(
                                grip.get("advantage_standardized")
                            ),
                            **_xyz("grip_center", grip.get("center_goal_m")),
                            **_xyz("grip_approach", grip.get("approach_goal")),
                            **_xyz(
                                "grip_surface_normal", grip.get("surface_normal")
                            ),
                            **_xyz(
                                "grip_surface_tangent", grip.get("surface_tangent")
                            ),
                            **_xyz(
                                "grip_surface_axis", grip.get("surface_axis")
                            ),
                            "grip_width_m": _number(grip.get("width_m")),
                            "grip_geom_score": _number(grip.get("geom_score")),
                            **_feature_columns(grip.get("feature_breakdown")),
                        }
                    )
                    rows += 1
        self._handle.flush()
        self.rows_written += rows
        return rows
