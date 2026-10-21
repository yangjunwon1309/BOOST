import csv

import pytest

from remote_offload.belief_csv import CSV_FIELDS, RemoteBeliefCsvRecorder


def test_belief_csv_rejects_appending_to_an_old_schema(tmp_path):
    output = tmp_path / "old.csv"
    output.write_text("published_timestamp_us,side\n1,left\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema mismatch"):
        RemoteBeliefCsvRecorder(output)


def test_belief_csv_records_both_hands_and_orientation_terms(tmp_path):
    output = tmp_path / "belief.csv"
    recorder = RemoteBeliefCsvRecorder(output)
    rows = recorder.record(
        {
            "published_timestamp_us": 101,
            "source_rgb_timestamp_us": 99,
            "remote_sequence": 7,
            "result_valid": True,
            "selected_hand": "left",
            "selection_active": True,
            "hands": {
                "left": {
                    "hand": {"armed": True, "alpha": 0.7},
                    "orientation": {
                        "hand_axis_world": [1.0, 0.0, 0.0],
                        "hand_omega_rps": [0.1, 0.2, 0.3],
                        "selected_index": 0,
                        "selected_track_id": "bottle:parallel",
                        "held_probability": 0.8,
                        "leader_probability": 0.8,
                        "switch_dwell_count": 2,
                        "candidates": [
                            {
                                "object_id": "bottle:track0001",
                                "label": "bottle",
                                "relation": "parallel",
                                "track_id": "bottle:parallel",
                                "source": "cylinder",
                                "position_goal_m": [0.3, 0.1, 0.2],
                                "normal": [0.0, 1.0, 0.0],
                                "fit_quality": 0.9,
                                "probability": 0.8,
                                "advantage_raw": 0.2,
                                "advantage_ema": 0.15,
                                "advantage_standardized": 0.3,
                                "label_prior": 0.1,
                                "breakdown": {
                                    "cos_c": 0.0,
                                    "alignment_error": 1.0,
                                    "align_advantage": -0.2,
                                    "hold_advantage": 0.0,
                                    "converge_advantage": 0.05,
                                    "omega_norm": 0.37,
                                    "weighted_total": -0.12,
                                },
                            }
                        ],
                    },
                    "target": {
                        "object_id": "bottle:track0001",
                        "position_goal_m": [0.3, 0.1, 0.2],
                        "orientation_axis_world": [0.0, 1.0, 0.0],
                    },
                },
                "right": {"hand": {"armed": False, "alpha": 0.0}},
            },
        }
    )
    recorder.close()

    assert rows == 3
    with output.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    assert tuple(values[0].keys()) == CSV_FIELDS
    assert values[0]["side"] == "left"
    assert values[0]["record_type"] == "hand"
    assert values[1]["record_type"] == "orientation"
    assert values[1]["normal_y"] == "1.0"
    assert values[1]["normal_weighted_total"] == "-0.12"
    assert values[2]["side"] == "right"
    assert values[2]["record_type"] == "hand"
    assert values[2]["normal_x"] == ""


def test_belief_csv_records_candidates_while_both_hands_are_unarmed(tmp_path):
    output = tmp_path / "unarmed.csv"
    recorder = RemoteBeliefCsvRecorder(output)
    candidate = {
        "object_id": "bottle:track0001",
        "label": "bottle",
        "relation": "parallel",
        "track_id": "bottle:parallel",
        "source": "bank_active",
        "position_goal_m": [0.3, 0.1, 0.2],
        "normal": [0.0, 1.0, 0.0],
        "probability": 0.75,
    }
    rows = recorder.record(
        {
            "published_timestamp_us": 201,
            "source_rgb_timestamp_us": 199,
            "remote_sequence": 8,
            "result_valid": True,
            "selected_hand": "none",
            "selection_active": False,
            "hands": {
                side: {
                    "hand": {"armed": False, "alpha": 0.0},
                    "orientation": {
                        "selected_index": 0,
                        "selected_track_id": "bottle:parallel",
                        "candidates": [candidate],
                    },
                }
                for side in ("left", "right")
            },
        }
    )
    recorder.close()

    assert rows == 4
    with output.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    orientation_rows = [row for row in values if row["record_type"] == "orientation"]
    assert [row["side"] for row in orientation_rows] == ["left", "right"]
    assert all(row["hand_armed"] == "False" for row in values)
    assert all(row["selection_active"] == "False" for row in values)
    assert all(row["object_id"] == "bottle:track0001" for row in orientation_rows)
    assert all(row["candidate_probability"] == "0.75" for row in orientation_rows)


def test_belief_csv_records_fk_timing_and_all_object_grip_features(tmp_path):
    output = tmp_path / "full.csv"
    recorder = RemoteBeliefCsvRecorder(output)
    feature = {
        "target_m": [0.4, 0.2, 0.3],
        "eef_position_m": [0.1, 0.2, 0.3],
        "eef_after_action_m": [0.11, 0.2, 0.3],
        "distance_m": 0.3,
        "distance_after_action_m": 0.29,
        "distance_value": -0.2,
        "distance_value_after_action": -0.19,
        "distance_advantage": -0.01,
        "progress_advantage": 1.0,
        "agreement_cost": 0.2,
        "agreement_advantage": -0.2,
        "shoulder_target_distance_m": None,
        "reach_advantage": 0.0,
        "delta_advantage": 0.0,
        "alignment_advantage": 0.4,
        "contact_fit_quality": 0.8,
        "contact_normal": [0.0, 0.0, 1.0],
        "hand_approach_axis": [0.0, 0.0, -1.0],
        "weights": {
            "distance": 1.4,
            "ray": 0.0,
            "progress": 1.0,
            "agreement": 0.4,
            "reach": 1.0,
            "delta": 0.0,
            "align": 0.9,
        },
        "weighted_contributions": {
            "distance": -0.014,
            "ray": 0.0,
            "progress": 1.0,
            "agreement": -0.08,
            "reach": 0.0,
            "delta": 0.0,
            "align": 0.36,
        },
        "weighted_total": 1.266,
    }
    object_telemetry = {
        "object_id": "stick:track0001",
        "label": "white stick",
        "score": 0.9,
        "object_probability": 0.8,
        "object_advantage_raw": 1.2,
        "object_advantage_ema": 1.1,
        "object_advantage_standardized": 1.5,
        "object_feature_breakdown": feature,
        "generated_grip_candidate_count": 2,
        "persistent_grip_candidate_count": 2,
        "grip_probabilities": [0.3, 0.7],
        "selected_grip_index": 1,
        "grip_candidates": [
            {
                "track_id": f"grip:{index}",
                "source": "surface",
                "center_goal_m": [0.4, 0.2 + 0.01 * index, 0.3],
                "approach_goal": [0.0, 0.0, -1.0],
                "surface_normal": [0.0, 0.0, 1.0],
                "probability": probability,
                "advantage_raw": 0.1 + index,
                "advantage_ema": 0.2 + index,
                "advantage_standardized": -1.0 + 2.0 * index,
                "feature_breakdown": feature,
            }
            for index, probability in enumerate((0.3, 0.7))
        ],
    }
    rows = recorder.record(
        {
            "published_timestamp_us": 2_010_000,
            "remote_published_timestamp_us": 2_000_000,
            "source_rgb_timestamp_us": 1_000_000,
            "source_depth_timestamp_us": 1_000_000,
            "belief_robot_state_timestamp_us": 1_100_000,
            "belief_robot_q": list(range(29)),
            "belief_base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "belief_hand_model": "dex1",
            "belief_grasp_center_local_m": [0.0, 0.1, 0.0],
            "image_width": 640,
            "image_height": 360,
            "remote_objects": [
                {
                    "track_id": "stick:track0001",
                    "source_robot_state_delta_ms": 75.0,
                    "geometry_source_timestamp_us": 990_000,
                    "center_camera_m": [0.1, 0.2, 0.3],
                    "center_goal_m": [0.4, 0.2, 0.3],
                    "target_contact_goal_m": [0.4, 0.21, 0.3],
                    "valid_depth_points": 500,
                    "geometry_fallback": False,
                    "missing_frames": 0,
                }
            ],
            "hands": {
                "left": {
                    "hand": {
                        "armed": True,
                        "alpha": 0.5,
                        "beta": 1.0,
                        "lambda0": 0.05,
                    },
                    "eef_position_m": [0.1, 0.2, 0.3],
                    "eef_velocity_mps": [1.0, 0.0, 0.0],
                    "objects": [object_telemetry],
                    "target": {
                        "object_id": "stick:track0001",
                        "grip_index": 1,
                        "object_confidence": 0.8,
                        "joint_confidence": 0.56,
                        "position_goal_m": [0.4, 0.21, 0.3],
                    },
                    "update": {"dt_s": 0.033, "grip_update_applied": True},
                },
                "right": {"hand": {"armed": False}},
            },
        }
    )
    recorder.close()

    assert rows == 5
    with output.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    assert [row["record_type"] for row in values] == [
        "hand", "object", "grip", "grip", "hand"
    ]
    hand = values[0]
    assert hand["source_to_belief_state_delta_ms"] == "100.0"
    assert hand["remote_to_gateway_latency_ms"] == "10.0"
    assert hand["robot_q_28"] == "28.0"
    assert hand["target_grip_confidence"] == "0.7"
    obj = values[1]
    assert obj["source_robot_state_delta_ms"] == "75.0"
    assert obj["geometry_age_at_source_ms"] == "10.0"
    assert obj["feature_contribution_progress"] == "1.0"
    selected_grip = values[3]
    assert selected_grip["grip_selected"] == "True"
    assert selected_grip["grip_advantage_standardized"] == "1.0"
    assert selected_grip["feature_distance_m"] == "0.3"
