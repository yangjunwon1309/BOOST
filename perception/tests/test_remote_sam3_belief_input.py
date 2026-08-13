import numpy as np

from remote_offload.protocol import ValidationContext, validate_result
from remote_offload.remote_geometry_pipeline import RemoteRgbdGeometryPipeline
from remote_offload.remote_sam3_belief_input_worker import (
    RemoteSam3BeliefInputWorker,
)
from remote_offload.remote_sam3_worker import MatchedFrameGeometry, RemoteSam3Worker
from remote_offload.remote_web_server import _belief_for_object, _draw_remote_belief_overlay
from remote_offload.remote_yolo11_worker import RemoteYolo11Worker
from remote_offload.thor_gateway import ThorRemoteGateway, _surface_primitive_from_dict
from seg_web_node.geometry import CameraIntrinsics
from seg_web_node.grip_candidates import (
    GripCandidateConfig,
    PseudoContactConfig,
    generate_grip_candidates,
)
from seg_web_node.object_point_filter import ObjectPointFilterConfig
from seg_web_node.lcm_protocol import DEPTH_SOURCE_RAW_DEPTH_OPTICAL


def _pipeline(fallback_ttl=2):
    return RemoteRgbdGeometryPipeline(
        object_point_filter=ObjectPointFilterConfig(enabled=False),
        contact_config=GripCandidateConfig.from_dict(
            {
                "top_k": 2,
                "surface_augment": 0,
                "max_points": 1000,
                "surface_model": {"enabled": False},
            }
        ),
        pseudo_config=PseudoContactConfig(min_inlier_ratio=0.2),
        min_depth_m=0.15,
        max_depth_m=3.0,
        min_depth_points=20,
        fallback_ttl_frames=fallback_ttl,
    )


def _frame_inputs():
    height, width = 48, 64
    mask = np.zeros((1, height, width), dtype=bool)
    mask[0, 12:38, 18:48] = True
    return {
        "masks": mask,
        "boxes": np.asarray([[18, 12, 47, 37]], dtype=np.float32),
        "scores": np.asarray([0.93], dtype=np.float32),
        "labels": ["bottle"],
        "depth_raw": np.full((height, width), 1000, dtype=np.uint16),
        "depth_scale_um": 1000,
        "intrinsics": CameraIntrinsics(width, height, 100.0, 100.0, 32.0, 24.0),
        "t_goal_cam": np.eye(4),
        "up_goal": np.array([0.0, 0.0, 1.0]),
        "source_robot_state_delta_ms": 4.2,
        "source_timestamp_us": 1_000_000,
        "image_width": width,
        "image_height": height,
    }


def test_shared_geometry_pipeline_is_backend_agnostic_and_protocol_valid():
    first = _pipeline().build_object_geometry(**_frame_inputs())
    second = _pipeline().build_object_geometry(**_frame_inputs())
    assert first == second
    obj = first[0]
    assert obj["track_id"] == "bottle:track0000"
    assert obj["center_goal_m"] is not None
    assert obj["contact_candidates"]
    assert len(obj["surface_primitives"]) == 1
    assert obj["source_robot_state_delta_ms"] == 4.2
    assert obj["goal_frame"] == "pelvis_identity_fk"

    payload = {
        "version": 1,
        "sequence": 1,
        "published_timestamp_us": 1_001_000,
        "source_rgb_timestamp_us": 1_000_000,
        "source_depth_timestamp_us": 1_000_000,
        "calibration_id": 1,
        "image_width": 64,
        "image_height": 48,
        "backend": "sam3_pytorch",
        "inference_ms": 10.0,
        "valid": True,
        "error": None,
        "objects": first,
    }
    assert validate_result(
        payload,
        ValidationContext(
            result_max_age_ms=800.0,
            now_timestamp_us=1_001_000,
        ),
    ) == []


def test_sam3_and_yolo_workers_emit_identical_geometry_for_fixed_masks():
    frame = _frame_inputs()
    pair = type(
        "Pair",
        (),
        {
            "rgb": type(
                "Rgb",
                (),
                {"timestamp_us": 1_000_000, "width": 64, "height": 48},
            )(),
            "depth": type(
                "Depth",
                (),
                {"timestamp_us": 1_000_000, "depth_scale_um": 1000},
            )(),
        },
    )()

    def worker(worker_type):
        value = worker_type.__new__(worker_type)
        value.sequence = 0
        value.calibration = type("Calibration", (), {"calibration_id": 1})()
        value.geometry_pipeline = _pipeline()
        value.frame_geometry = MatchedFrameGeometry(
            timestamp_us=1_000_000,
            state=object(),
            state_delta_ms=4.2,
            t_goal_cam=np.eye(4),
            up_goal=np.array([0.0, 0.0, 1.0]),
        )
        return value

    arguments = (
        pair,
        "sam3_pytorch",
        10.0,
        frame["masks"],
        frame["boxes"],
        frame["scores"],
        frame["labels"],
        frame["depth_raw"],
        frame["intrinsics"],
    )
    sam_objects = worker(RemoteSam3BeliefInputWorker)._build_result(*arguments)[
        "objects"
    ]
    yolo_objects = worker(RemoteYolo11Worker)._build_result(*arguments)["objects"]
    assert sam_objects == yolo_objects


def test_missing_geometry_context_keeps_mask_but_never_emits_cached_belief_geometry():
    pipeline = _pipeline()
    current = pipeline.build_object_geometry(**_frame_inputs())
    assert current[0]["center_goal_m"] is not None

    invalid = _frame_inputs()
    invalid.update(
        intrinsics=None,
        t_goal_cam=None,
        up_goal=None,
        source_robot_state_delta_ms=None,
        source_timestamp_us=1_100_000,
        allow_geometry_fallback=False,
    )
    diagnostic = pipeline.build_object_geometry(**invalid)
    assert len(diagnostic) == 1
    assert diagnostic[0]["center_camera_m"] is None
    assert diagnostic[0]["center_goal_m"] is None
    assert diagnostic[0]["contact_candidates"] == []
    assert diagnostic[0]["surface_primitives"] == []
    assert diagnostic[0]["geometry_fallback"] is False


def test_uncalibrated_raw_depth_still_allows_rgb_mask_diagnostics():
    worker = RemoteSam3Worker.__new__(RemoteSam3Worker)
    pair = type(
        "Pair",
        (),
        {
            "rgb": type("Rgb", (), {"width": 64, "height": 48})(),
            "depth": type(
                "Depth",
                (),
                {
                    "source": DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
                    "depth": np.full((24, 32), 1000, dtype=np.uint16),
                },
            )(),
        },
    )()
    depth, intrinsics = worker._prepare_depth_and_intrinsics(pair, None)
    assert depth.shape == (48, 64)
    assert not np.any(depth)
    assert intrinsics is None


def test_detection_fallback_retains_track_and_original_geometry_timestamp():
    pipeline = _pipeline()
    current = pipeline.build_object_geometry(**_frame_inputs())
    missing = _frame_inputs()
    missing.update(
        masks=np.empty((0, 48, 64), dtype=bool),
        boxes=np.empty((0, 4), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        labels=[],
        source_timestamp_us=1_200_000,
    )
    fallback = pipeline.build_object_geometry(**missing)
    assert fallback[0]["track_id"] == current[0]["track_id"]
    assert fallback[0]["geometry_fallback"] is True
    assert fallback[0]["geometry_source_timestamp_us"] == 1_000_000


def test_fifteen_degree_gravity_tilt_controls_non_cylinder_grip_approach():
    angle = np.deg2rad(15.0)
    up = np.array([0.0, np.sin(angle), np.cos(angle)])
    points = np.array(
        [[x, y, 0.2] for x in np.linspace(0.3, 0.4, 12) for y in np.linspace(-0.04, 0.04, 10)]
    )
    candidates = generate_grip_candidates(
        points,
        GripCandidateConfig.from_dict(
            {"top_k": 1, "surface_augment": 0, "surface_model": {"enabled": False}}
        ),
        up_goal=up,
    )
    np.testing.assert_allclose(candidates[0].approach, -up, atol=1.0e-6)


def test_thor_reconstructs_remote_surface_primitive():
    primitive = _surface_primitive_from_dict(
        {
            "kind": "cylinder",
            "origin_goal_m": [0.3, 0.1, 0.2],
            "axis_goal": [0.0, 0.0, 2.0],
            "fit_quality": 0.8,
            "source": "cylinder",
            "radius": 0.04,
            "half_height": 0.1,
            "snapped_to_gravity": True,
        }
    )
    assert primitive is not None
    np.testing.assert_allclose(primitive.axis, [0.0, 0.0, 1.0])
    assert primitive.snapped_to_gravity is True


def test_thor_fallback_age_uses_source_frame_clock_domain():
    gateway = ThorRemoteGateway.__new__(ThorRemoteGateway)
    gateway.ee_target_label = ""
    gateway.ee_target_min_score = 0.0
    gateway.result_max_age_ms = 800.0
    obj = {
        "track_id": "bottle:track0000",
        "label": "bottle",
        "score": 0.9,
        "center_goal_m": [0.3, 0.1, 0.2],
        "contact_candidates": [],
        "surface_primitives": [],
        "geometry_fallback": True,
        "geometry_source_timestamp_us": 9_500_000,
    }
    payload = {"source_rgb_timestamp_us": 10_000_000, "objects": [obj]}
    assert len(gateway._belief_inputs_from_result(payload)) == 1
    obj["geometry_source_timestamp_us"] = 9_000_000
    assert gateway._belief_inputs_from_result(payload) == []


def test_remote_web_reads_both_hands_and_draws_dim_or_armed_overlays():
    telemetry = {
        "selected_hand": "none",
        "hands": {
            "left": {
                "armed": False,
                "object_probabilities": {"bottle:track0000": 0.7},
                "objects": [
                    {
                        "object_id": "bottle:track0000",
                        "object_probability": 0.7,
                        "grip_probabilities": [0.6],
                        "selected_grip_index": 0,
                    }
                ],
                "target": {"object_id": "bottle:track0000"},
            },
            "right": {
                "armed": False,
                "object_probabilities": {"bottle:track0000": 0.3},
                "objects": [],
                "target": {"object_id": "bottle:track0000"},
            },
        },
    }
    left = _belief_for_object(telemetry, "left", "bottle:track0000")
    right = _belief_for_object(telemetry, "right", "bottle:track0000")
    assert np.isclose(left["joint_probability"], 0.42)
    assert np.isclose(right["object_probability"], 0.3)

    objects = [
        {
            "track_id": "bottle:track0000",
            "box_xyxy": [10, 10, 40, 40],
            "center_pixel_uv": [25, 25],
        }
    ]
    dim = np.zeros((64, 64, 3), dtype=np.uint8)
    _draw_remote_belief_overlay(dim, objects, telemetry, stale=False)
    telemetry["selected_hand"] = "left"
    telemetry["hands"]["left"]["armed"] = True
    armed = np.zeros_like(dim)
    _draw_remote_belief_overlay(armed, objects, telemetry, stale=False)
    assert np.any(dim)
    assert np.any(armed)
    assert not np.array_equal(dim, armed)
