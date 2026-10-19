import collections
import threading

import numpy as np
import pytest
from remote_offload.protocol import (
    REMOTE_DEPTH_CHANNEL,
    REMOTE_RGB_CHANNEL,
    ValidationContext,
    make_empty_result,
    make_object,
    now_us,
    pack_result,
    strip_invalid_optional_fields,
    unpack_belief_telemetry,
    validate_result,
    validate_result_soft,
)
from remote_offload.remote_geometry_pipeline_head_d435i_legacy import (
    GEOMETRY_COLOR_SIZE,
    RemoteRgbdGeometryPipeline,
    camera_geometry_metadata,
    geometry_intrinsics_from_color,
    masks_to_geometry_grid,
)
from remote_offload.remote_sam3_belief_input_worker_head_d435i_legacy import (
    RemoteSam3BeliefInputWorker,
)
from remote_offload.remote_sam3_worker_head_d435i_legacy import (
    MatchedFrameGeometry,
    RemoteSam3Worker,
)
from remote_offload.remote_sam3_worker_head_d435i_legacy import (
    build_arg_parser as build_base_worker_arg_parser,
)
from remote_offload.remote_web_server_head_d435i_legacy import (
    RemoteWebRuntime,
    _belief_for_object,
    _draw_remote_belief_overlay,
    _goal_to_pixel,
    _project_axis_arrow,
)
from remote_offload.thor_gateway_head_d435i_legacy import (
    COMMAND_HISTORY_SPAN_S,
    SHARED_HAND_PUBLISH_HZ,
    ForwardedPairRecord,
    RawFrame,
    RawPair,
    ThorRemoteGateway,
    _surface_primitive_from_dict,
)
from remote_offload.thor_gateway_head_d435i_legacy import (
    build_arg_parser as build_gateway_arg_parser,
)
from seg_web_node.feature_cost import FeatureConfig
from seg_web_node.geometry import CameraIntrinsics
from seg_web_node.grip_candidates import (
    GripCandidateConfig,
    PseudoContactConfig,
    SurfacePrimitive,
    generate_grip_candidates,
)
from seg_web_node.lcm_protocol import (
    DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    CameraCalibration,
)
from seg_web_node.object_point_filter import ObjectPointFilterConfig
from seg_web_node.robot_geometry import (
    g1_grasp_center_position,
    g1_hand_base_rotation,
    g1_link_transforms,
    grasp_center_from_wrist,
    hand_frame_profile,
    hand_rotation_from_wrist_rotation,
)


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


def test_v91_revised_gateway_and_worker_defaults():
    gateway = build_gateway_arg_parser().parse_args(
        ["--g1-lcm-url", "udpm://g1", "--wifi-lcm-url", "udpm://wifi"]
    )
    worker = build_base_worker_arg_parser().parse_args(
        ["--wifi-lcm-url", "udpm://wifi"]
    )
    assert gateway.forward_hz == 15.0
    assert gateway.sync_tolerance_ms == 20.0
    assert worker.inference_hz == 8.0
    assert worker.sync_tolerance_ms == 20.0
    assert worker.min_depth_points == 80


def test_v101_command_belief_defaults():
    gateway = build_gateway_arg_parser().parse_args(
        ["--g1-lcm-url", "udpm://g1", "--wifi-lcm-url", "udpm://wifi"]
    )
    assert gateway.belief_input_source == "operator_command"
    assert gateway.belief_command_divergence_max_m == 0.20
    assert gateway.belief_command_max_delta_ms == 25.0
    assert FeatureConfig().min_hand_omega_rps == 0.08


def test_command_path_matches_fk_path_for_both_hands():
    q = np.linspace(-0.15, 0.15, 29, dtype=np.float64)
    transforms = g1_link_transforms(q)
    for side in ("left", "right"):
        link = f"{side}_wrist_yaw_link"
        wrist = transforms[link]
        np.testing.assert_allclose(
            grasp_center_from_wrist(
                wrist[:3, 3], wrist[:3, :3], side, "dex1"
            ),
            g1_grasp_center_position(q, link, None, "dex1"),
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            hand_rotation_from_wrist_rotation(
                wrist[:3, :3], side, "dex1"
            ),
            g1_hand_base_rotation(q, link, "dex1"),
            atol=1.0e-9,
        )


def _command_gateway() -> ThorRemoteGateway:
    gateway = ThorRemoteGateway.__new__(ThorRemoteGateway)
    gateway.command_history = collections.deque(
        maxlen=int(COMMAND_HISTORY_SPAN_S * SHARED_HAND_PUBLISH_HZ)
    )
    gateway.belief_command_max_delta_ms = 25.0
    gateway.stats = {
        "command_lookup_failures": 0,
        "command_lookup_age_ms": {},
    }
    return gateway


def test_command_lookup_empty_and_outside_25_ms_fail_cleanly():
    gateway = _command_gateway()
    assert gateway._command_features_at(1_000_000, "left") is None
    assert gateway.stats["command_lookup_failures"] == 1
    gateway.command_history.append(
        (
            1_000_000,
            {
                "left": {
                    "command_wrist_position_m": [0.1, 0.2, 0.3],
                    "command_wrist_rotation": np.eye(3).reshape(9).tolist(),
                }
            },
        )
    )
    assert gateway._command_features_at(1_025_001, "left") is None
    assert gateway.stats["command_lookup_failures"] == 2
    assert gateway.stats["command_lookup_age_ms"]["left"] == pytest.approx(
        25.001
    )


def test_command_history_span_and_rotation_round_trip():
    gateway = _command_gateway()
    start_us = 2_000_000
    interval_us = int(1.0e6 / SHARED_HAND_PUBLISH_HZ)
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    for index in range(100):
        timestamp_us = start_us + index * interval_us
        gateway.command_history.append(
            (
                timestamp_us,
                {
                    "left": {
                        "command_wrist_position_m": [0.1, 0.2, 0.3],
                        "command_wrist_rotation": rotation.reshape(9).tolist(),
                        "command_velocity_mps": [0.01, 0.0, 0.0],
                        "command_omega_rps": [0.0, 0.0, 0.2],
                    }
                },
            )
        )
    assert len(gateway.command_history) == 75
    assert (
        gateway.command_history[-1][0] - gateway.command_history[0][0]
    ) / 1.0e6 <= COMMAND_HISTORY_SPAN_S
    features = gateway._command_features_at(
        gateway.command_history[-1][0], "left"
    )
    assert features is not None
    recovered_rotation = features[1]
    assert np.linalg.norm(
        recovered_rotation @ recovered_rotation.T - np.eye(3)
    ) < 1.0e-9


def test_operator_command_divergence_uses_fk_fallback():
    class FakeParallelBelief:
        class Config:
            enabled = True

        config = Config()

        def __init__(self):
            self.trackers = {"left": object(), "right": object()}
            self.hand_inputs = None

        def update_hands(self, _geometry, hand_inputs):
            self.hand_inputs = hand_inputs
            return {
                side: {"hand": {"armed": False}, "target": None}
                for side in hand_inputs
            }

    gateway = _command_gateway()
    gateway.belief_enabled = True
    gateway.belief_use_operator_command = True
    gateway.belief_command_divergence_max_m = 0.20
    gateway.belief_hand_model = "dex1"
    gateway.belief_hand_profile = hand_frame_profile("dex1")
    gateway.belief_grasp_center_local_m = np.asarray(
        gateway.belief_hand_profile["grasp_center_local"], dtype=np.float64
    )
    gateway.parallel_tasc_belief = FakeParallelBelief()
    gateway.belief_selected_hand = "none"
    gateway.belief_selection_active = False
    gateway.ee_target_link = "left_wrist_yaw_link"
    gateway.lock = threading.Lock()
    gateway.stats.update(
        belief_command_divergence_m={},
        belief_input={},
        last_belief={},
    )
    timestamp_us = 3_000_000
    command_entry = {
        side: {
            "command_wrist_position_m": [1.0, 0.0, 0.0],
            "command_wrist_rotation": np.eye(3).reshape(9).tolist(),
            "command_velocity_mps": [0.4, 0.0, 0.0],
            "command_omega_rps": [0.0, 0.0, 0.3],
        }
        for side in ("left", "right")
    }
    gateway.command_history.append((timestamp_us, command_entry))
    gateway._drain_shared_hand_selection = lambda: None
    gateway._belief_inputs_from_result = lambda _payload: []
    gateway._belief_state_features = lambda _packet, _side: (
        np.zeros(3),
        np.array([0.01, 0.0, 0.0]),
        np.eye(3),
        np.zeros(3),
    )
    gateway._publish_belief_telemetry = lambda _payload, _hands: None
    gateway._record_belief_csv = lambda _payload, _telemetry: None

    selection = gateway._belief_selection_from_result(
        {"sequence": 1, "source_rgb_timestamp_us": timestamp_us},
        None,
        object(),
    )

    assert selection is None
    assert gateway.parallel_tasc_belief.hand_inputs is not None
    for side, hand in gateway.parallel_tasc_belief.hand_inputs.items():
        assert hand.input_source == "fk_divergence_fallback"
        np.testing.assert_allclose(hand.eef_position, np.zeros(3))
        np.testing.assert_allclose(hand.user_translation, [0.01, 0.0, 0.0])
        assert gateway.stats["belief_input"]["source"][side] == (
            "fk_divergence_fallback"
        )
        assert gateway.stats["belief_input"]["divergence_m"][side] > 0.20


def test_v91_geometry_grid_scales_intrinsics_and_masks_with_nearest_neighbor():
    color = CameraIntrinsics(1280, 720, 908.216, 908.216, 640.0, 360.0)
    geometry = geometry_intrinsics_from_color(color)
    assert (geometry.width, geometry.height) == GEOMETRY_COLOR_SIZE
    np.testing.assert_allclose(
        [geometry.fx, geometry.fy, geometry.cx, geometry.cy],
        [454.108, 454.108, 320.0, 180.0],
    )
    masks = np.zeros((1, 720, 1280), dtype=bool)
    masks[0, 200:520, 400:880] = True
    reduced = masks_to_geometry_grid(masks, GEOMETRY_COLOR_SIZE)
    assert reduced.shape == (1, 360, 640)
    assert reduced[0, 100:260, 200:440].all()
    assert not reduced[0, :99].any()
    with pytest.raises(ValueError, match="aspect ratio changed"):
        geometry_intrinsics_from_color(
            CameraIntrinsics(640, 480, 386.3, 386.3, 320.0, 240.0)
        )


def test_v91_geometry_outputs_full_resolution_mask_and_center_coordinates():
    masks = np.zeros((1, 720, 1280), dtype=bool)
    masks[0, 200:520, 400:880] = True
    arguments = {
        "masks": masks,
        "boxes": np.asarray([[400, 200, 879, 519]], dtype=np.float32),
        "scores": np.asarray([0.95], dtype=np.float32),
        "labels": ["bottle"],
        "depth_raw": np.full((360, 640), 1000, dtype=np.uint16),
        "depth_scale_um": 1000,
        "intrinsics": CameraIntrinsics(
            640, 360, 454.108, 454.108, 320.0, 180.0
        ),
        "color_intrinsics": CameraIntrinsics(
            1280, 720, 908.216, 908.216, 640.0, 360.0
        ),
        "t_goal_cam": np.eye(4),
        "up_goal": np.array([0.0, 0.0, 1.0]),
        "source_robot_state_delta_ms": 8.0,
        "source_timestamp_us": 1_000_000,
        "image_width": 1280,
        "image_height": 720,
    }
    obj = _pipeline().build_object_geometry(**arguments)[0]
    assert obj["mask_width"] == 1280
    assert obj["mask_height"] == 720
    assert 630 <= obj["center_pixel_uv"][0] <= 650
    assert 350 <= obj["center_pixel_uv"][1] <= 370
    assert obj["valid_depth_points"] >= 80


def test_v91_worker_reprojects_raw_depth_directly_to_geometry_grid():
    worker = RemoteSam3Worker.__new__(RemoteSam3Worker)
    depth = np.full((360, 640), 1000, dtype=np.uint16)
    pair = type(
        "Pair",
        (),
        {
            "rgb": type("Rgb", (), {"width": 1280, "height": 720})(),
            "depth": type(
                "Depth",
                (),
                {
                    "source": DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
                    "width": 640,
                    "height": 360,
                    "depth_scale_um": 1000,
                    "depth": depth,
                },
            )(),
        },
    )()
    calibration = CameraCalibration(
        calibration_id=1,
        timestamp_us=1,
        depth_width=640,
        depth_height=360,
        color_width=640,
        color_height=360,
        depth_scale_um=1000,
        depth_intrinsics=np.asarray([454.108, 454.108, 320.0, 180.0]),
        color_intrinsics=np.asarray([454.108, 454.108, 320.0, 180.0]),
        rotation_color_depth=np.eye(3),
        translation_color_depth_m=np.zeros(3),
        rectified=True,
    )
    aligned, intrinsics = worker._prepare_depth_and_intrinsics(
        pair,
        calibration,
        target_size=GEOMETRY_COLOR_SIZE,
    )
    assert aligned.shape == (360, 640)
    assert np.count_nonzero(aligned) / aligned.size > 0.99
    assert intrinsics is not None
    assert (intrinsics.width, intrinsics.height) == (640, 360)


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
    assert sam_objects


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


def test_worker_warns_once_and_reports_geometry_status_when_context_is_missing(caplog):
    frame = _frame_inputs()
    pair = type(
        "Pair",
        (),
        {
            "rgb": type("Rgb", (), {"timestamp_us": 1_000_000, "width": 64, "height": 48})(),
            "depth": type("Depth", (), {"timestamp_us": 1_000_000, "depth_scale_um": 1000})(),
        },
    )()
    worker = RemoteSam3BeliefInputWorker.__new__(RemoteSam3BeliefInputWorker)
    worker.sequence = 0
    worker.calibration = type("Calibration", (), {"calibration_id": 1})()
    worker.geometry_pipeline = _pipeline()
    worker.frame_geometry = MatchedFrameGeometry(
        timestamp_us=1_000_000,
        state=None,
        state_delta_ms=None,
        t_goal_cam=None,
        up_goal=None,
    )
    arguments = (
        pair,
        "sam3_tensorrt",
        10.0,
        frame["masks"],
        frame["boxes"],
        frame["scores"],
        frame["labels"],
        frame["depth_raw"],
        frame["intrinsics"],
    )
    with caplog.at_level("WARNING"):
        first = worker._build_result(*arguments)
        worker._build_result(*arguments)
    warnings = [
        record
        for record in caplog.records
        if "publishing masks only" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert first["geometry_status"] == {
        "state_available": False,
        "t_goal_cam_available": False,
        "calibration_available": True,
        "intrinsics_available": True,
        "image_size": [64, 48],
        "depth_size": [64, 48],
        "intrinsics_size": [64, 48],
        "geometry_intrinsics_size": [64, 48],
        "geometry_valid_depth_ratio": 1.0,
        "up_goal_available": False,
        "state_delta_ms": None,
        "objects_with_valid_depth": 0,
        "contact_candidate_count": 0,
        "primitive_count": 0,
        "fallback_count": 0,
    }


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
    assert "surface_primitives" not in fallback[0]


def _result_with_primitive(primitive: dict):
    timestamp = now_us()
    mask = np.ones((4, 6), dtype=bool)
    payload = make_empty_result(
        sequence=1,
        published_timestamp_us=timestamp,
        source_rgb_timestamp_us=100,
        source_depth_timestamp_us=101,
        calibration_id=7,
        image_width=6,
        image_height=4,
        backend="sam3_tensorrt",
        inference_ms=10.0,
    )
    payload["objects"] = [
        make_object(
            track_id="object:track0000",
            label="object",
            score=0.9,
            box_xyxy=[0, 0, 5, 3],
            mask=mask,
            median_depth_m=1.0,
            center_pixel_uv=[3, 2],
            center_camera_m=[0.0, 0.0, 1.0],
            center_goal_m=[0.0, 0.0, 1.0],
            valid_depth_points=24,
            surface_primitives=[primitive],
            geometry_source_timestamp_us=100,
        )
    ]
    return payload


def test_surface_primitive_satisfies_wire_contract_for_all_kinds():
    primitives = (
        SurfacePrimitive("plane", np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.8, "plane"),
        SurfacePrimitive("cylinder", np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.7, "cylinder_axis", radius=0.03, half_height=0.1),
        SurfacePrimitive("circle", np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.6, "hole_circle", radius=0.02),
        SurfacePrimitive("pca", np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.3, "pca"),
        SurfacePrimitive("point", np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0, "center_fallback"),
    )
    for primitive in primitives:
        wire = primitive.to_dict()
        payload = _result_with_primitive(wire)
        assert validate_result(
            payload,
            ValidationContext(
                result_max_age_ms=float("inf"),
                now_timestamp_us=payload["published_timestamp_us"],
            ),
        ) == []
        restored = _surface_primitive_from_dict(wire)
        assert restored is not None
        assert restored.kind == primitive.kind


def test_invalid_primitive_is_soft_and_does_not_discard_mask_frame():
    payload = _result_with_primitive(
        SurfacePrimitive(
            "plane", np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.8, "plane"
        ).to_dict()
    )
    payload["objects"][0]["surface_primitives"][0]["kind"] = "bogus"
    context = ValidationContext(
        result_max_age_ms=float("inf"),
        now_timestamp_us=payload["published_timestamp_us"],
    )
    assert validate_result(payload, context, include_soft=False) == []
    assert any("kind is invalid" in error for error in validate_result_soft(payload))
    mask_data = payload["objects"][0]["mask_data"]
    assert strip_invalid_optional_fields(payload) == 1
    assert payload["objects"][0]["surface_primitives"] == []
    assert payload["objects"][0]["mask_data"] == mask_data
    assert validate_result(payload, context) == []


def test_thor_gateway_accepts_frame_after_stripping_invalid_primitive():
    payload = _result_with_primitive(
        SurfacePrimitive(
            "plane", np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.8, "plane"
        ).to_dict()
    )
    payload["objects"][0]["surface_primitives"][0]["kind"] = "bogus"
    payload["stage_us"] = {
        "source_rgb": 100,
        "worker_dequeue": 200,
        "decode_done": 300,
        "infer_done": 500,
        "geometry_done": 800,
        "packed": 900,
    }
    pair = RawPair(
        rgb=RawFrame(100, 6, 4, b"rgb"),
        depth=RawFrame(101, 6, 4, b"depth"),
        delta_us=1,
    )
    gateway = ThorRemoteGateway.__new__(ThorRemoteGateway)
    gateway.lock = threading.Lock()
    gateway.forwarded_pairs = collections.deque(
        [ForwardedPairRecord(pair=pair, calibration_id=7, robot_state=None)]
    )
    gateway.last_result_sequence = None
    gateway.result_max_age_ms = float("inf")
    gateway.stats = {
        "invalid_results": 0,
        "valid_results": 0,
        "soft_result_warnings": [],
        "soft_optional_fields_dropped": 0,
        "decode_errors": 0,
    }
    accepted = []
    gateway._publish_ee_target_from_result = lambda value, record: accepted.append(value)
    gateway._record_error = lambda error: (_ for _ in ()).throw(error)
    gateway._on_remote_result("unused", pack_result(payload))
    assert gateway.stats["invalid_results"] == 0
    assert gateway.stats["valid_results"] == 1
    assert gateway.stats["soft_optional_fields_dropped"] == 1
    assert gateway.stats["last_stage_latency_ms"]["inference"] == 0.2
    assert accepted[0]["objects"][0]["surface_primitives"] == []
    assert accepted[0]["objects"][0]["mask_data"]
    assert accepted[0]["stage_us"]["gateway_receive"] > 0


def test_thor_gateway_forwards_immediately_when_pair_is_submitted():
    published = []
    depth_published = threading.Event()

    class FakeBus:
        def publish(self, channel, data):
            published.append((channel, data))
            if channel == REMOTE_DEPTH_CHANNEL:
                depth_published.set()

    gateway = ThorRemoteGateway.__new__(ThorRemoteGateway)
    gateway.lock = threading.Lock()
    gateway.pair_ready = threading.Condition(gateway.lock)
    gateway.running = True
    gateway.pending_pair = None
    gateway.latest_pair = None
    gateway.latest_calibration = None
    gateway.latest_robot_state = None
    gateway.forward_period = 0.0
    gateway.last_forward_monotonic = 0.0
    gateway.calibration_resend_sec = 2.0
    gateway.robot_state_period = 0.05
    gateway.forwarded_pairs = collections.deque(maxlen=60)
    gateway.wifi_bus = FakeBus()
    gateway.stats = {
        "dropped_before_forward": 0,
        "forwarded_pairs": 0,
        "forwarded_calibration": 0,
        "forwarded_robot_state": 0,
        "forward_publish_bytes": 0,
        "forward_publish_errors": 0,
    }
    thread = threading.Thread(target=gateway._forward_loop, daemon=True)
    thread.start()
    pair = RawPair(
        rgb=RawFrame(10, 6, 4, b"rgb"),
        depth=RawFrame(11, 6, 4, b"depth"),
        delta_us=1,
    )
    gateway._submit_pair(pair)

    assert depth_published.wait(timeout=1.0)
    with gateway.pair_ready:
        gateway.running = False
        gateway.pair_ready.notify_all()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert published == [
        (REMOTE_RGB_CHANNEL, b"rgb"),
        (REMOTE_DEPTH_CHANNEL, b"depth"),
    ]
    assert gateway.stats["forwarded_pairs"] == 1
    assert gateway.stats["forward_publish_bytes"] == len(b"depth")
    assert gateway.stats["forward_publish_errors"] == 0


def test_camera_metadata_supports_goal_reprojection_and_axis_foreshortening():
    intrinsics = CameraIntrinsics(100, 80, 100.0, 100.0, 50.0, 40.0)
    geometry_intrinsics = CameraIntrinsics(50, 40, 50.0, 50.0, 25.0, 20.0)
    metadata = camera_geometry_metadata(
        np.eye(4),
        intrinsics,
        geometry_intrinsics,
    )
    transform = np.asarray(metadata["t_goal_cam"]).reshape(4, 4)
    assert metadata["intrinsics"]["width"] == 100
    assert metadata["geometry_intrinsics"]["width"] == 50
    assert _goal_to_pixel([0.0, 0.0, 1.0], transform, metadata["intrinsics"]) == (50, 40)
    parallel = _project_axis_arrow(
        [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], transform, metadata["intrinsics"]
    )
    toward_camera = _project_axis_arrow(
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], transform, metadata["intrinsics"]
    )
    assert parallel is not None and parallel[0] != parallel[1]
    assert toward_camera is not None and toward_camera[0] == toward_camera[1]


def test_ipopt_display_camera_z_offset_can_project_wrist_target_behind_camera():
    intrinsics = {
        "width": 100,
        "height": 80,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 40.0,
    }
    # The wrist-yaw origin is 4 cm behind a forward-mounted D405 at Q_READY.
    assert _goal_to_pixel([0.0, 0.0, -0.04], np.eye(4), intrinsics) is None
    assert _goal_to_pixel(
        [0.0, 0.0, -0.04],
        np.eye(4),
        intrinsics,
        camera_z_offset_m=0.25,
    ) == (50, 40)


def test_web_pairs_belief_telemetry_to_result_sequence_or_timestamp():
    runtime = RemoteWebRuntime.__new__(RemoteWebRuntime)
    runtime.lock = threading.Lock()
    first = {"sequence": 10, "source_rgb_timestamp_us": 100}
    second = {"sequence": 11, "source_rgb_timestamp_us": 200}
    runtime.result = second
    runtime.result_history = collections.OrderedDict(((10, first), (11, second)))
    runtime.belief_telemetry = {"result_sequence": 10, "source_rgb_timestamp_us": 100}
    assert runtime._select_belief_pair() == (second, first, "matched")
    runtime.belief_telemetry = {"result_sequence": 9, "source_rgb_timestamp_us": 100}
    assert runtime._select_belief_pair() == (second, first, "matched_ts")
    runtime.belief_telemetry = {"result_sequence": 9, "source_rgb_timestamp_us": 99}
    assert runtime._select_belief_pair() == (second, None, "unmatched")


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


def test_gateway_keeps_unarmed_belief_and_csv_active_during_empty_detection():
    class FakeParallelBelief:
        class Config:
            enabled = True

        config = Config()

        def __init__(self):
            self.trackers = {"left": object(), "right": object()}
            self.calls = []

        def update_hands(self, shared_geometry, hand_inputs):
            self.calls.append((shared_geometry, hand_inputs))
            return {
                side: {
                    "hand": {
                        "armed": hand.button_pressed,
                        "alpha": 0.0,
                    },
                    "orientation": {
                        "selected_index": 0,
                        "selected_track_id": "bank:coasting",
                        "candidates": [
                            {
                                "track_id": "bank:coasting",
                                "probability": 1.0,
                            }
                        ],
                    },
                    "target": None,
                }
                for side, hand in hand_inputs.items()
            }

    gateway = ThorRemoteGateway.__new__(ThorRemoteGateway)
    gateway.belief_enabled = True
    gateway.parallel_tasc_belief = FakeParallelBelief()
    gateway.belief_selected_hand = "none"
    gateway.belief_selection_active = False
    gateway.belief_hand_profile = {"approach_axis_local": [0.0, 1.0, 0.0]}
    gateway.ee_target_link = "left_wrist_yaw_link"
    gateway.lock = threading.Lock()
    gateway.stats = {"last_belief": {}}
    gateway._drain_shared_hand_selection = lambda: None
    gateway._belief_inputs_from_result = lambda _payload: []
    gateway._belief_state_features = lambda _packet, _side: (
        np.zeros(3),
        None,
        np.eye(3),
        np.zeros(3),
    )
    published = []
    recorded = []
    gateway._publish_belief_telemetry = (
        lambda payload, hands: published.append((payload, hands))
    )
    gateway._record_belief_csv = (
        lambda payload, telemetry: recorded.append((payload, telemetry))
    )

    shared_geometry = object()
    selection = gateway._belief_selection_from_result(
        {"sequence": 9, "valid": True},
        None,
        shared_geometry,
    )

    assert selection is None
    assert len(gateway.parallel_tasc_belief.calls) == 1
    assert gateway.parallel_tasc_belief.calls[0][0] is shared_geometry
    assert set(gateway.parallel_tasc_belief.calls[0][1]) == {"left", "right"}
    assert all(
        not hand.button_pressed
        for hand in gateway.parallel_tasc_belief.calls[0][1].values()
    )
    assert len(published) == 1
    assert len(recorded) == 1
    assert set(recorded[0][1]["hands"]) == {"left", "right"}
    assert recorded[0][1]["selection_active"] is False
    assert recorded[0][1]["reason"] == "no_valid_remote_geometry_coasting"


def test_bounded_telemetry_preserves_contact_and_orientation_origins():
    gateway = ThorRemoteGateway.__new__(ThorRemoteGateway)
    gateway.belief_telemetry_max_objects = 4
    gateway.belief_telemetry_max_contacts = 4
    gateway.belief_telemetry_max_orientations = 4
    gateway.belief_orientation_min_confidence = 0.35
    bounded = gateway._bounded_hand_telemetry(
        {
            "hand": {"armed": True, "alpha": 1.0},
            "objects": [],
            "target": {
                "object_id": "object:1",
                "position_goal_m": [0.1, 0.2, 0.3],
                "orientation_axis_world": [0.0, 0.0, 1.0],
                "orientation_relation": "parallel",
                "orientation_confidence": 0.8,
                "orientation_source_position_m": [0.1, 0.2, 0.31],
            },
        }
    )
    assert bounded["target"]["position_goal_m"] == [0.1, 0.2, 0.3]
    assert bounded["target"]["orientation_source_position_m"] == [0.1, 0.2, 0.31]
    assert bounded["target"]["orientation_valid"] is True


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


def test_remote_belief_cross_projects_published_contact_not_object_center():
    overlay = np.zeros((80, 100, 3), dtype=np.uint8)
    telemetry = {
        "selected_hand": "left",
        "hands": {
            "left": {
                "armed": True,
                "object_probabilities": {"object:1": 0.8},
                "objects": [
                    {
                        "object_id": "object:1",
                        "object_probability": 0.8,
                        "grip_probabilities": [0.5],
                        "selected_grip_index": 0,
                        "selected_contact_goal_m": [0.1, 0.0, 1.0],
                    }
                ],
                "target": {
                    "object_id": "object:1",
                    "position_goal_m": [0.1, 0.0, 1.0],
                },
            },
            "right": {"armed": False, "target": None, "objects": []},
        },
    }
    intrinsics = {
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 40.0,
        "width": 100,
        "height": 80,
    }
    _draw_remote_belief_overlay(
        overlay,
        [{"track_id": "object:1", "box_xyxy": [2, 60, 20, 76]}],
        telemetry,
        stale=False,
        t_goal_cam=np.eye(4),
        intrinsics=intrinsics,
    )
    assert np.any(overlay[38:43, 58:63])

    no_box = np.zeros_like(overlay)
    _draw_remote_belief_overlay(
        no_box,
        [],
        telemetry,
        stale=False,
        t_goal_cam=np.eye(4),
        intrinsics=intrinsics,
    )
    # The target point survives a temporarily unavailable segmentation box.
    assert np.any(no_box[38:43, 58:63])


def test_remote_web_draws_fresh_control_markers_and_hides_stale_ones():
    telemetry = {
        "selected_hand": "left",
        "hands": {
            "left": {
                "armed": True,
                "objects": [{"object_id": "object:1", "grip_probabilities": [0.7], "selected_grip_index": 0}],
                "target": {"object_id": "object:1", "position_goal_m": [0.2, 0.0, 1.0]},
            },
            "right": {"armed": False, "objects": [], "target": {}},
        },
    }
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0, "width": 100, "height": 80}
    objects = [{"track_id": "object:1", "box_xyxy": [60, 40, 80, 65]}]
    fresh = np.zeros((80, 100, 3), dtype=np.uint8)
    control = {
        "age_ms": 10.0,
        "hands": {
            "left": {
                "human_grasp_goal_m": [-0.2, 0.25, 1.0],
                "reference_grasp_goal_m": [-0.1, 0.25, 1.0],
            }
        },
    }
    _draw_remote_belief_overlay(
        fresh, objects, telemetry, stale=False, t_goal_cam=np.eye(4),
        intrinsics=intrinsics, control=control,
    )
    assert np.any(fresh[55:77, 20:41])
    assert np.any(fresh[55:77, 30:51])

    stale = np.zeros_like(fresh)
    _draw_remote_belief_overlay(
        stale, objects, telemetry, stale=False, t_goal_cam=np.eye(4),
        intrinsics=intrinsics, control={**control, "age_ms": 501.0},
    )
    # The belief target remains, but the operator marker at u=30 is absent.
    assert not np.any(stale[55:77, 22:39])


def test_remote_web_draws_control_reference_without_belief_geometry():
    overlay = np.zeros((80, 100, 3), dtype=np.uint8)
    _draw_remote_belief_overlay(
        overlay,
        [],
        {"selected_hand": "left", "hands": {"left": {}, "right": {}}},
        stale=False,
        t_goal_cam=np.eye(4),
        intrinsics={
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "width": 100,
            "height": 80,
        },
        control={
            "age_ms": 10.0,
            "hands": {
                "left": {
                    "armed": True,
                    "reference_grasp_goal_m": [-0.1, 0.25, 1.0],
                }
            },
        },
    )
    # u=40, v=65: control reference remains visible without a depth-derived
    # belief object or hand telemetry.
    assert np.any(overlay[55:76, 30:51])


def test_gateway_forwards_shared_control_v2_inside_belief_telemetry():
    class FakeBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        def publish(self, channel: str, payload: bytes) -> None:
            self.published.append((channel, payload))

    gateway = ThorRemoteGateway.__new__(ThorRemoteGateway)
    gateway.last_belief_telemetry_monotonic = 0.0
    gateway.belief_telemetry_period_sec = 0.0
    gateway.belief_selected_hand = "left"
    gateway.belief_telemetry_max_objects = 4
    gateway.belief_telemetry_max_contacts = 4
    gateway.belief_telemetry_max_orientations = 4
    gateway.belief_orientation_min_confidence = 0.35
    gateway.latest_hand_selection = {
        "schema": "path.shared_control_hand.v2",
        "timestamp_us": now_us(),
        "state": "teleop",
        "selected_hand": "left",
        "active": True,
        "revision": 3,
        "grasp_center_local_m": [0.0, 0.0, 0.1],
        "hands": {"left": {"human_grasp_goal_m": [0.1, 0.0, 1.0]}},
    }
    gateway.wifi_bus = FakeBus()
    gateway.lock = threading.Lock()
    gateway.stats = {"belief_telemetry_published": 0}
    gateway._publish_belief_telemetry(
        {"sequence": 7, "valid": True, "published_timestamp_us": now_us()},
        {"left": {}, "right": {}},
    )
    packet = unpack_belief_telemetry(gateway.wifi_bus.published[0][1])
    assert packet["control"]["schema"] == "path.shared_control_hand.v2"
    assert packet["control"]["selected_hand"] == "left"
    assert packet["control"]["active"] is True
    assert packet["control"]["hands"]["left"]["human_grasp_goal_m"] == [0.1, 0.0, 1.0]
    assert packet["control"]["age_ms"] is not None


def test_dim_hands_share_one_alpha_blend_without_overlap_accumulation():
    overlay = np.zeros((100, 120, 3), dtype=np.uint8)
    hand = {
        "armed": False,
        "object_probabilities": {"object:1": 0.5},
        "objects": [{"object_id": "object:1", "object_probability": 0.5}],
        "target": {"object_id": "object:1"},
    }
    telemetry = {
        "selected_hand": "none",
        "hands": {"left": dict(hand), "right": dict(hand)},
    }
    _draw_remote_belief_overlay(
        overlay,
        [{"track_id": "object:1", "box_xyxy": [40, 50, 90, 85]}],
        telemetry,
        stale=False,
    )
    # A single 0.75 blend caps a fully saturated dim element near 191. Two
    # element-wise blends would compound well above that at the shared box.
    assert int(np.max(overlay[50, 52:59])) <= 192
