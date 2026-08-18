import numpy as np
import msgpack
import pytest
import struct
import threading
import zmq

from seg_web_node.ee_target import GoalPublisher, Workspace, make_goal_payload
from seg_web_node.geometry import (
    CameraIntrinsics,
    deproject,
    robust_mask_center,
    transform_points,
)
from seg_web_node.lcm_protocol import (
    CALIBRATION_PACKET,
    CameraCalibration,
    CameraFrame,
    DepthFrame,
    DEPTH_ENCODING_RAW_ZLIB,
    DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    decode_calibration,
    decode_camera,
    decode_depth,
    encode_camera,
    encode_depth,
)
from seg_web_node.depth_alignment import (
    align_depth_to_color,
    calibration_for_frame,
    scale_camera_intrinsics,
)
from seg_web_node.robot_geometry import pelvis_to_torso, waist_fk
from seg_web_node.lcm_protocol import RobotState
from seg_web_node.rgbd_sync import RgbdSynchronizer
from seg_web_node.visualization import render_segmented_rgbd
from seg_web_node.web_server import (
    SegmentationRuntime,
    parse_concepts,
    select_tracking_candidate,
    suppress_duplicate_masks,
)


def test_protocol_and_sync_round_trip():
    timestamp = 77
    rgb = decode_camera(encode_camera(CameraFrame(2, 2, timestamp, b"jpeg")))
    depth_array = np.array([[1000, 0], [1500, 2000]], dtype=np.uint16)
    depth = decode_depth(encode_depth(depth_array, timestamp))
    sync = RgbdSynchronizer(tolerance_ms=1)
    assert sync.add_rgb(rgb) is None
    pair = sync.add_depth(depth)
    assert pair is not None
    assert pair.delta_us == 0
    np.testing.assert_array_equal(pair.depth.depth, depth_array)


def test_deprojection_and_transform_direction():
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    k = CameraIntrinsics(2, 2, 1.0, 1.0, 0.0, 0.0)
    xyz = deproject(depth, k, 1000)
    np.testing.assert_allclose(xyz[0, 0], [0, 0, 1])
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(transform_points(xyz[0, 0], transform), [1, 2, 4])


def test_robust_segment_center_ignores_invalid_depth():
    depth = np.full((4, 4), 1000, dtype=np.uint16)
    depth[0, 0] = 0
    xyz = deproject(depth, CameraIntrinsics(4, 4, 2, 2, 1.5, 1.5), 1000)
    mask = np.ones((4, 4), dtype=bool)
    result = robust_mask_center(xyz, mask, min_valid_points=10)
    assert result is not None
    center, pixel, count = result
    np.testing.assert_allclose(center[2], 1.0)
    assert pixel == (2, 2)
    assert count == 15


def test_path_goal_payload_and_reject_workspace():
    workspace = Workspace()
    info = workspace.validate(np.array([0.25, 0.12, 0.18]), "left")
    payload = make_goal_payload(
        np.array([0.25, 0.12, 0.18]),
        duration_sec=2.0,
        final_hold_sec=1.0,
        workspace_info=info,
    )
    assert payload["target_link"] == "left_wrist_yaw_link"
    assert payload["coordinate_frame"] == "pico_head_position_waist"
    assert payload["target_position_frame"] == "pico_head_position_waist"
    assert payload["human_wrist_frame"] == "pico_head_position_waist"
    assert len(payload["human_left_wrist_transform"]) == 16
    assert len(payload["human_right_wrist_transform"]) == 16
    assert payload["target_valid"] is True
    assert payload["position"] == [0.25, 0.12, 0.18]
    assert payload["duration_sec"] == 2.0
    assert payload["final_hold_sec"] == 1.0
    try:
        workspace.validate(np.array([1.0, 0.12, 0.18]), "left")
    except ValueError as error:
        assert "outside conservative workspace" in str(error)
    else:
        raise AssertionError("unsafe target was accepted")


def test_goal_publisher_uses_reference_topic_msgpack_wire_format():
    endpoint = "inproc://sam3-ee-target-test"
    context = zmq.Context.instance()
    receiver = context.socket(zmq.PAIR)
    receiver.bind(endpoint)
    sender = context.socket(zmq.PAIR)
    sender.connect(endpoint)
    try:
        payload = make_goal_payload(np.array([0.25, 0.12, 0.18]))
        GoalPublisher._send(sender, "path_static_ee_target", payload)
        raw = receiver.recv()
        prefix = b"path_static_ee_target"
        assert raw.startswith(prefix)
        decoded = msgpack.unpackb(raw[len(prefix):], raw=False)
        assert decoded["goal_uid"] == payload["goal_uid"]
        assert decoded["position"] == payload["position"]
    finally:
        sender.close()
        receiver.close()


def test_unaligned_source_rejected_by_default():
    packet = encode_depth(np.ones((2, 2), dtype=np.uint16), 1, source=1)
    try:
        decode_depth(packet)
    except ValueError as error:
        assert "not calibrated" in str(error)
    else:
        raise AssertionError("unaligned source was accepted")
    assert decode_depth(packet, allow_unaligned=True).source == 1


def test_segmented_rgbd_renderer():
    image = np.full((120, 160, 3), 80, dtype=np.uint8)
    depth = np.full((120, 160), 1250, dtype=np.uint16)
    masks = np.zeros((1, 120, 160), dtype=bool)
    masks[0, 30:90, 40:120] = True
    boxes = np.array([[40, 30, 119, 89]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    output, objects = render_segmented_rgbd(
        image, depth, masks, boxes, scores, ["object"], 1000
    )
    assert output.shape == image.shape
    assert objects[0]["mask_pixels"] == 60 * 80
    assert objects[0]["label"] == "object"
    assert abs(objects[0]["median_depth_m"] - 1.25) < 1e-6


def test_segmented_rgbd_reviewer_mode_has_no_depth_panel_and_returns_mask_tint():
    image = np.full((60, 80, 3), 80, dtype=np.uint8)
    depth = np.full((60, 80), 1250, dtype=np.uint16)
    masks = np.zeros((1, 60, 80), dtype=bool)
    masks[0, 10:50, 20:60] = True
    boxes = np.array([[20, 10, 59, 49]], dtype=np.float32)
    output, _, colors = render_segmented_rgbd(
        image,
        depth,
        masks,
        boxes,
        np.array([0.9], dtype=np.float32),
        ["object"],
        1000,
        include_depth_panel=False,
        draw_score=False,
        draw_depth_text=False,
        return_instance_colors=True,
    )
    assert output.shape == image.shape
    assert colors == [(64, 64, 255)]
    # The former depth inset occupies the bottom-right corner; in reviewer
    # mode it remains the unmodified background.
    np.testing.assert_array_equal(output[-1, -1], image[-1, -1])


def test_multi_concept_parsing_and_duplicate_suppression():
    assert parse_concepts("person, box, person") == ["person", "box"]
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[:, 2:6, 2:6] = True
    boxes = np.array([[2, 2, 5, 5], [2, 2, 5, 5]], dtype=np.float32)
    scores = np.array([0.7, 0.9], dtype=np.float32)
    kept_masks, _, kept_scores, kept_labels = suppress_duplicate_masks(
        masks, boxes, scores, ["object", "box"]
    )
    assert len(kept_masks) == 1
    assert kept_scores.tolist() == [np.float32(0.9)]
    assert kept_labels == ["box"]


def test_continuous_tracking_keeps_nearest_same_label_instance():
    instances = [
        {"index": 0, "label": "bottle", "score": 0.99,
         "center_goal_m": [0.65, 0.10, 0.20]},
        {"index": 1, "label": "bottle", "score": 0.80,
         "center_goal_m": [0.31, 0.10, 0.20]},
        {"index": 2, "label": "box", "score": 0.95,
         "center_goal_m": [0.30, 0.10, 0.20]},
    ]
    selected, state = select_tracking_candidate(
        instances, "bottle", np.array([0.30, 0.10, 0.20]), 0.25
    )
    assert state == "ok"
    assert selected is not None
    assert selected["index"] == 1

    selected, state = select_tracking_candidate(
        instances[:1], "bottle", np.array([0.30, 0.10, 0.20]), 0.25
    )
    assert selected is None
    assert state == "jump"

    selected, state = select_tracking_candidate(
        [{"index": 0, "label": "bottle", "center_goal_m": None}],
        "bottle", np.array([0.30, 0.10, 0.20]), 0.25,
    )
    assert selected is None
    assert state == "depth_unavailable"

    selected, state = select_tracking_candidate(
        instances, "cup", np.array([0.30, 0.10, 0.20]), 0.25
    )
    assert selected is None
    assert state == "missing"


def test_continuous_target_starts_updates_and_stops_after_object_loss():
    class FakePublisher:
        def __init__(self):
            self.payloads = []

        def publish(self, payload, **_kwargs):
            self.payloads.append(payload)

    runtime = SegmentationRuntime.__new__(SegmentationRuntime)
    runtime.target_lock = threading.Lock()
    runtime.lock = threading.Lock()
    runtime.publisher = FakePublisher()
    runtime.target_bypass_checks = True
    runtime.target_config = {"duration_sec": 2.0, "final_hold_sec": 1.0}
    runtime.target_side = "left"
    runtime.target_link = "left_wrist_yaw_link"
    runtime.target_quaternion = (1.0, 0.0, 0.0, 0.0)
    runtime.target_offset = np.zeros(3)
    runtime.workspace = Workspace()
    runtime.target_auto_publish_hz = 5.0
    runtime.target_tracking_max_jump_m = 0.25
    runtime.target_tracking_lost_grace_frames = 3
    runtime.target_last_stream_publish_monotonic = 0.0
    runtime.target_tracking = None
    runtime.target_tracking_missing_frames = 0
    runtime.status = {
        "last_frame_ts": 100,
        "instances": [{
            "index": 0,
            "label": "bottle",
            "score": 0.9,
            "center_camera_m": [0.0, 0.0, 0.4],
            "center_goal_m": [0.30, 0.10, 0.20],
        }],
        "ee_target_last_result": None,
    }

    result = runtime.send_object_target(
        0, 100, "bottle", [0.30, 0.10, 0.20]
    )
    assert result["tracking"] is True
    assert runtime.target_tracking is not None
    goal_uid = result["goal_uid"]

    runtime.status["instances"][0]["center_goal_m"] = [0.32, 0.10, 0.21]
    runtime.target_last_stream_publish_monotonic = 0.0
    with runtime.lock:
        publisher, payload, _, _ = runtime._auto_publish_target_locked()
    assert publisher is runtime.publisher
    assert payload is not None
    assert payload["goal_uid"] == goal_uid
    assert payload["position"] == [0.32, 0.10, 0.21]

    runtime.status["instances"] = []
    for _ in range(3):
        with runtime.lock:
            runtime._auto_publish_target_locked()
    assert runtime.target_tracking is None
    assert "disappeared" in runtime.status["ee_target_last_result"]


def test_calibration_packet_and_thor_side_alignment():
    values = [2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0]
    values += np.eye(3).reshape(-1).tolist() + [0.0, 0.0, 0.0]
    packet = CALIBRATION_PACKET.pack(
        0x4C414343, 1, 1, 9, 100, 2, 2, 2, 2, 1000, *values
    )
    calibration = decode_calibration(packet)
    frame = decode_depth(encode_depth(
        np.array([[1000, 0], [0, 1500]], dtype=np.uint16), 100,
        source=DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    ))
    aligned = align_depth_to_color(frame, calibration)
    np.testing.assert_array_equal(aligned, frame.depth)


def test_calibration_scales_for_proportional_rgbd_resize():
    values = [2.0, 3.0, 0.5, 0.75, 4.0, 5.0, 1.0, 1.25]
    values += np.eye(3).reshape(-1).tolist() + [0.0, 0.0, 0.0]
    calibration = decode_calibration(
        CALIBRATION_PACKET.pack(
            0x4C414343, 1, 1, 9, 100, 640, 360, 640, 360, 1000, *values
        )
    )

    scaled = calibration_for_frame(
        calibration,
        color_width=1280,
        color_height=720,
        depth_width=1280,
        depth_height=720,
    )

    assert (scaled.color_width, scaled.color_height) == (1280, 720)
    assert (scaled.depth_width, scaled.depth_height) == (1280, 720)
    np.testing.assert_allclose(scaled.depth_intrinsics, [4.0, 6.0, 1.0, 1.5])
    np.testing.assert_allclose(scaled.color_intrinsics, [8.0, 10.0, 2.0, 2.5])

    intrinsics = scale_camera_intrinsics(
        CameraIntrinsics(640, 360, 400.0, 401.0, 320.0, 180.0),
        width=1280,
        height=720,
    )
    assert (intrinsics.width, intrinsics.height) == (1280, 720)
    np.testing.assert_allclose(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
        [800.0, 802.0, 640.0, 360.0],
    )


def test_calibration_rejects_non_proportional_resize():
    values = [2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0]
    values += np.eye(3).reshape(-1).tolist() + [0.0, 0.0, 0.0]
    calibration = decode_calibration(
        CALIBRATION_PACKET.pack(
            0x4C414343, 1, 1, 9, 100, 640, 360, 640, 360, 1000, *values
        )
    )
    with pytest.raises(ValueError, match="aspect ratio changed"):
        calibration_for_frame(
            calibration,
            color_width=1280,
            color_height=720,
            depth_width=640,
            depth_height=480,
        )


def test_aligned_depth_ignores_raw_depth_camera_aspect_ratio():
    values = [380.0, 380.0, 320.0, 240.0, 400.0, 401.0, 320.0, 180.0]
    values += np.eye(3).reshape(-1).tolist() + [0.0, 0.0, 0.0]
    calibration = decode_calibration(
        CALIBRATION_PACKET.pack(
            0x4C414343, 1, 1, 9, 100, 640, 480, 640, 360, 1000, *values
        )
    )

    scaled = calibration_for_frame(
        calibration,
        color_width=1280,
        color_height=720,
    )

    # The aligned 1280x720 depth packet uses the color pixel grid. The raw
    # depth-camera calibration remains 640x480 and is not applied to it.
    assert (scaled.depth_width, scaled.depth_height) == (640, 480)
    np.testing.assert_allclose(
        scaled.depth_intrinsics, [380.0, 380.0, 320.0, 240.0]
    )
    assert (scaled.color_width, scaled.color_height) == (1280, 720)
    np.testing.assert_allclose(
        scaled.color_intrinsics, [800.0, 802.0, 640.0, 360.0]
    )


def test_g1_waist_fk_and_dual_imu_estimator():
    q = np.zeros(29)
    fk = waist_fk(q)
    np.testing.assert_allclose(fk[:3, 3], [-0.0039635, 0.0, 0.054])
    state = RobotState(
        1, 1, 123, 0, True,
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]), q, np.zeros(29),
    )
    transform, metadata = pelvis_to_torso(state)
    np.testing.assert_allclose(transform, fk)
    assert metadata["mode"] == "dual_imu"
