import csv

from remote_offload.belief_csv import CSV_FIELDS, RemoteBeliefCsvRecorder


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

    assert rows == 2
    with output.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    assert tuple(values[0].keys()) == CSV_FIELDS
    assert values[0]["side"] == "left"
    assert values[0]["normal_y"] == "1.0"
    assert values[0]["normal_weighted_total"] == "-0.12"
    assert values[1]["side"] == "right"
    assert values[1]["normal_x"] == ""
