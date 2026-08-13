import pytest

from remote_offload.protocol import (
    MASK_ENCODING,
    decode_perception_result,
    encode_mask,
    encode_perception_result,
)


def _payload():
    return {
        "version": 1,
        "sequence": 5,
        "published_timestamp_us": 100,
        "source_rgb_timestamp_us": 90,
        "source_depth_timestamp_us": 90,
        "calibration_id": 7,
        "image_width": 640,
        "image_height": 360,
        "backend": "yolo11_pytorch",
        "inference_ms": 130.0,
        "valid": True,
        "error": None,
        "objects": [
            {
                "track_id": "object:track0001",
                "label": "object",
                "score": 0.9,
                "box_xyxy": [1.0, 2.0, 3.0, 4.0],
                "mask_encoding": MASK_ENCODING,
                "mask_data": encode_mask(
                    [
                        [(100 <= x <= 140 and 120 <= y <= 160) for x in range(640)]
                        for y in range(360)
                    ]
                ),
                "mask_height": 360,
                "mask_width": 640,
                "median_depth_m": 0.45,
                "center_pixel_uv": [100, 120],
                "center_camera_m": [0.1, 0.0, 0.45],
                "center_goal_m": [0.3, 0.1, 0.2],
                "valid_depth_points": 1000,
                "target_contact_goal_m": [0.31, 0.11, 0.2],
                "contact_candidates": [
                    {
                        "center_goal_m": [0.31, 0.11, 0.2],
                        "approach_goal": [0.0, 0.0, -1.0],
                        "width_m": 0.06,
                        "geom_score": 0.7,
                        "source": "pca_surface",
                    }
                ],
                "surface_model": {
                    "type": "cylinder",
                    "center_goal_m": [0.3, 0.1, 0.2],
                    "axis_goal": [0.0, 0.0, 1.0],
                    "radius_m": 0.04,
                    "height_m": 0.20,
                    "fit_rmse_m": 0.002,
                    "inlier_ratio": 0.91,
                    "angular_coverage": 0.50,
                    "updates": 4,
                    "stale_frames": 0,
                },
                "surface_primitives": [
                    {
                        "kind": "cylinder",
                        "origin_goal_m": [0.3, 0.1, 0.2],
                        "axis_goal": [0.0, 0.0, 1.0],
                        "fit_quality": 0.81,
                        "source": "cylinder",
                        "radius": 0.04,
                        "half_height": 0.10,
                        "extent": None,
                        "snapped_to_gravity": True,
                    }
                ],
                "source_robot_state_delta_ms": 4.2,
                "geometry_source_timestamp_us": 90,
                "goal_frame": "pelvis_identity_fk",
                "geometry_fallback": False,
                "missing_frames": 0,
            }
        ],
    }


def test_remote_perception_protocol_round_trip():
    assert decode_perception_result(encode_perception_result(_payload())) == _payload()


def test_remote_perception_rejects_mask_dimension_mismatch():
    payload = _payload()
    payload["objects"][0]["mask_width"] = 320
    with pytest.raises(ValueError, match="mask dimensions"):
        encode_perception_result(payload)


def test_remote_perception_rejects_invalid_cylinder_surface():
    payload = _payload()
    payload["objects"][0]["surface_model"]["radius_m"] = float("nan")
    with pytest.raises(ValueError, match="surface_model.radius_m"):
        encode_perception_result(payload)


def test_remote_perception_rejects_invalid_surface_primitive():
    payload = _payload()
    payload["objects"][0]["surface_primitives"][0]["axis_goal"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="surface_primitives.*axis_goal"):
        encode_perception_result(payload)
