from __future__ import annotations

import threading
import time

import numpy as np

from remote_offload.remote_sam3_wrist_d405_worker_v11 import (
    WristD405Sam3Worker,
    attribute_stacked_masks,
    suppress_camera_duplicate_masks,
    stack_wrist_images,
)
from remote_offload.thor_gateway_head_d435i_legacy import ThorRemoteGateway
from remote_offload.wrist_belief_gateway_v11 import (
    WristBeliefGateway,
    wrist_object_ids_for_hand,
)
from remote_offload.wrist_d405_protocol_v11 import (
    CAMERA_IDS,
    make_result,
    pack_ipopt_visual,
    pack_mask,
    pack_result,
    pack_depth_z16,
    pack_bundle,
    unpack_bundle,
    unpack_depth_z16,
    unpack_ipopt_visual,
    unpack_result,
)
from remote_offload.thor_wrist_d405_gateway_v11 import (
    RobotStateHistory,
    nearest_state_sample,
)
from seg_web_node.lcm_protocol import RobotState
from remote_offload.remote_web_wrist_d405_v11 import WristV11WebViewer
from remote_offload.wrist_d405_geometry_pipeline_v11 import (
    camera_transform_diagnostics,
)


def test_d405_depth_round_trip_preserves_native_grid() -> None:
    depth = np.arange(640 * 360, dtype=np.uint16).reshape(360, 640)
    restored = unpack_depth_z16(pack_depth_z16(depth), width=640, height=360)
    np.testing.assert_array_equal(restored, depth)


def test_v14_bundle_preserves_camera_specific_state_pairing() -> None:
    depth = np.zeros((1, 1), dtype=np.uint16)
    cameras = {}
    for index, camera_id in enumerate(CAMERA_IDS):
        cameras[camera_id] = {
            "camera_id": camera_id,
            "serial": f"d405-{index}",
            "host_timestamp_us": 10_000 + index,
            "color_timestamp_us": 20_000 + index,
            "depth_timestamp_us": 20_000 + index,
            "width": 1,
            "height": 1,
            "color_jpeg": b"jpeg",
            "depth_z16_zlib": pack_depth_z16(depth),
            "depth_scale_m": 0.001,
            "intrinsics": {"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0},
            "q_at_exposure": [float(index)] * 29,
            "base_quat_at_exposure_wxyz": [1.0, 0.0, 0.0, 0.0],
            "state_pairing": {
                "lookup_us": 30_000 + index,
                "state_arrival_us": 30_001 + index,
                "residual_ms": 0.001,
                "bias_us": -2_000,
                "method": "nearest",
                "wrist_speed_mps": 0.2,
                "wrist_omega_rps": 1.0,
                "estimated_error_m": 0.0005,
            },
        }
    payload = {
        "version": 1,
        "sequence": 1,
        "bundle_timestamp_us": 40_000,
        "split_y": 1,
        "inter_wrist_skew_us": 1,
        "cameras": cameras,
    }

    restored = unpack_bundle(pack_bundle(payload))

    assert restored["cameras"]["left_wrist"]["q_at_exposure"] != restored["cameras"]["right_wrist"]["q_at_exposure"]
    assert restored["cameras"]["right_wrist"]["state_pairing"]["bias_us"] == -2_000


def test_state_history_samples_nearest_arrival_without_hold() -> None:
    history = RobotStateHistory(span_s=1.0, publish_hz=10.0)
    for received_us in (1_000, 2_000, 3_000):
        state = RobotState(
            sequence=received_us,
            lowstate_sequence=received_us,
            timestamp_us=1,
            mode_machine=0,
            has_torso_imu=False,
            base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            torso_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            q=np.zeros(29),
            dq=np.zeros(29),
        )
        history.add(received_us, bytes([received_us // 1_000]), state)

    matched = nearest_state_sample(history.snapshot(), 2_600)

    assert matched is not None
    index, sample, residual_ms = matched
    assert index == 2
    assert sample.received_us == 3_000
    assert residual_ms == 0.4


def test_worker_uses_camera_paired_q_instead_of_legacy_snapshot() -> None:
    fallback = RobotState(
        sequence=1,
        lowstate_sequence=1,
        timestamp_us=1,
        mode_machine=0,
        has_torso_imu=False,
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        torso_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        q=np.zeros(29),
        dq=np.zeros(29),
    )
    frame = {
        "q_at_exposure": [0.25] * 29,
        "base_quat_at_exposure_wxyz": [0.0, 1.0, 0.0, 0.0],
        "state_pairing": {
            "lookup_us": 100,
            "state_arrival_us": 101,
            "residual_ms": 0.001,
            "bias_us": 0,
            "method": "nearest",
            "wrist_speed_mps": 0.0,
            "wrist_omega_rps": 0.0,
            "estimated_error_m": 0.0,
        },
    }

    state, pairing = WristD405Sam3Worker._frame_paired_state(frame, fallback, 10.0)

    assert state is not None and pairing is not None
    np.testing.assert_allclose(state.q, 0.25)
    np.testing.assert_allclose(state.base_quat_wxyz, [0.0, 1.0, 0.0, 0.0])


def test_wrist_ipopt_visual_round_trip() -> None:
    payload = {
        "schema": "path.wrist_ipopt_visual.v1",
        "published_timestamp_us": 123,
        "state": "teleop",
        "hands": {
            "left": {"ipopt_wrist_target_m": [0.1, 0.2, 0.3]},
        },
    }
    assert unpack_ipopt_visual(pack_ipopt_visual(payload)) == {
        "version": 1,
        **payload,
    }


def test_wrist_ipopt_freshness_uses_local_receipt_clock() -> None:
    viewer = WristV11WebViewer.__new__(WristV11WebViewer)
    viewer.ipopt_visual = {
        "schema": "path.wrist_ipopt_visual.v1",
        # Deliberately belongs to a different monotonic clock domain.
        "published_timestamp_us": 1,
        "state": "teleop",
        "hands": {"left": {"ipopt_wrist_target_m": [0.1, 0.2, 0.3]}},
    }
    viewer.ipopt_visual_received_monotonic = time.monotonic() - 0.020

    control = viewer._ipopt_control_locked()

    assert control is not None
    assert 10.0 <= control["age_ms"] < 200.0


def test_vertical_masks_map_to_their_local_wrist_grid() -> None:
    masks = np.zeros((2, 720, 640), dtype=bool)
    masks[0, 20:40, 10:30] = True
    masks[1, 500:520, 10:30] = True
    attributed, discarded = attribute_stacked_masks(
        masks, np.array([[10, 20, 30, 40], [10, 500, 30, 520]], dtype=float),
        np.array([0.9, 0.8]), ["top", "bottom"], split_y=360, seam_guard_px=8,
    )
    assert discarded == 0
    assert len(attributed["left_wrist"]) == 1
    assert len(attributed["right_wrist"]) == 1
    assert attributed["right_wrist"][0]["box"][1] == 140
    assert attributed["left_wrist"][0]["mask"].shape == (360, 640)


def test_seam_touching_mask_is_discarded_not_split() -> None:
    masks = np.zeros((1, 720, 640), dtype=bool)
    masks[0, 351:370, 100:120] = True
    attributed, discarded = attribute_stacked_masks(
        masks, np.array([[100, 351, 120, 370]], dtype=float), np.array([0.9]), ["unsafe"],
        split_y=360, seam_guard_px=8,
    )
    assert discarded == 1
    assert not attributed["left_wrist"] and not attributed["right_wrist"]


def test_wrist_bottom_edge_mask_is_kept_when_it_is_only_in_one_camera() -> None:
    masks = np.zeros((1, 720, 640), dtype=bool)
    # A grasp object commonly occupies the bottom edge of the left wrist view.
    # It touches the stack seam but is not a cross-camera SAM3 merge.
    masks[0, 350:360, 100:120] = True
    attributed, discarded = attribute_stacked_masks(
        masks,
        np.array([[100, 350, 120, 360]], dtype=float),
        np.array([0.9]),
        ["object"],
        split_y=360,
        seam_guard_px=8,
    )
    assert discarded == 0
    assert len(attributed["left_wrist"]) == 1


def test_duplicate_masks_are_suppressed_only_within_one_camera() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:15, 4:16] = True
    kept, suppressed = suppress_camera_duplicate_masks(
        [
            {"mask": mask, "box": np.array([4, 3, 16, 15]), "score": 0.9, "label": "cup"},
            {"mask": mask.copy(), "box": np.array([4, 3, 16, 15]), "score": 0.8, "label": "container"},
        ]
    )
    assert len(kept) == 1
    assert suppressed == 1


def test_result_carries_sequence_match_and_geometry_diagnostics() -> None:
    result = make_result(
        sequence=11,
        source_bundle_timestamp_us=123,
        source_bundle_sequence=7,
        objects=[],
        discarded_seam_masks=0,
        geometry_status={"state_age_rejects": 2, "up_goal_available": True},
    )
    assert result["source_bundle_sequence"] == 7
    assert result["geometry_status"]["state_age_rejects"] == 2


def test_frame_level_camera_transform_is_validated_and_not_object_payload() -> None:
    transforms = {
        camera_id: np.eye(4, dtype=float).reshape(-1).tolist()
        for camera_id in CAMERA_IDS
    }
    diagnostics = {
        camera_id: {"t_goal_camera": transforms[camera_id]}
        for camera_id in CAMERA_IDS
    }
    result = make_result(
        sequence=12,
        source_bundle_timestamp_us=456,
        source_bundle_sequence=8,
        objects=[{"camera_id": "left_wrist", "track_id": "left_wrist:cup:0"}],
        discarded_seam_masks=0,
        t_goal_cam=transforms,
        transform_diagnostics=diagnostics,
    )
    restored = unpack_result(pack_result(result))
    assert restored["t_goal_cam"] == transforms
    assert restored["transform_diagnostics"]["right_wrist"]["t_goal_camera"] == transforms["right_wrist"]
    assert "t_goal_cam" not in restored["objects"][0]
    assert "transform_diagnostics" not in restored["objects"][0]


def test_frame_transform_diagnostics_use_supplied_fk_without_recomputation() -> None:
    goal_wrist = np.eye(4, dtype=float)
    goal_wrist[:3, 3] = [0.2, -0.1, 0.3]
    wrist_camera = np.eye(4, dtype=float)
    wrist_camera[:3, 3] = [0.05, 0.0, -0.02]
    diagnostics = camera_transform_diagnostics(
        "left_wrist", {"left_wrist": wrist_camera}, goal_wrist=goal_wrist
    )
    expected = goal_wrist @ wrist_camera
    np.testing.assert_array_equal(
        diagnostics["t_goal_camera"], expected.reshape(-1).tolist()
    )
    assert diagnostics["t_goal_wrist"] == goal_wrist.reshape(-1).tolist()


def test_wrist_viewer_keeps_latest_frame_transform_without_object_pose_copy() -> None:
    viewer = WristV11WebViewer.__new__(WristV11WebViewer)
    viewer.lock = threading.Condition()
    viewer.latest_images = None
    viewer.objects = []
    viewer.latest_transforms = {}
    viewer.belief = None
    viewer.result_timestamp_us = None
    viewer.rendered_result_sequence = None
    viewer.result_source_bundle_sequence = None
    viewer.rendered_belief = None

    first_transform = np.eye(4, dtype=float).reshape(-1).tolist()
    intrinsics = {
        "width": 640,
        "height": 360,
        "fx": 400.0,
        "fy": 400.0,
        "ppx": 320.0,
        "ppy": 180.0,
    }
    first = make_result(
        sequence=1,
        source_bundle_timestamp_us=1,
        objects=[{"camera_id": "left_wrist", "intrinsics": intrinsics}],
        discarded_seam_masks=0,
        t_goal_cam={"left_wrist": first_transform},
        transform_diagnostics={"left_wrist": {"t_goal_camera": first_transform}},
    )
    viewer._on_result("result", pack_result(first))

    second_transform = np.eye(4, dtype=float)
    second_transform[0, 3] = 0.15
    second_wire = second_transform.reshape(-1).tolist()
    second = make_result(
        sequence=2,
        source_bundle_timestamp_us=2,
        objects=[],
        discarded_seam_masks=0,
        t_goal_cam={"left_wrist": second_wire},
        transform_diagnostics={"left_wrist": {"t_goal_camera": second_wire}},
    )
    viewer._on_result("result", pack_result(second))

    assert viewer.latest_transforms["left_wrist"]["t_goal_cam"] == second_wire
    assert viewer.latest_transforms["left_wrist"]["intrinsics"] == intrinsics


def test_wrist_viewer_alpha_blends_sam3_masks() -> None:
    viewer = WristV11WebViewer.__new__(WristV11WebViewer)
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:9, 4:12] = True
    viewer.objects = [
        {
            "camera_id": "left_wrist",
            "mask_encoding": "bitpack-zlib-row-major",
            "mask_data": pack_mask(mask),
        }
    ]
    images = {
        "left_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
        "right_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
    }

    viewer._draw_sam3_masks_locked(images)

    assert np.any(images["left_wrist"][mask])
    assert not np.any(images["left_wrist"][~mask])
    assert not np.any(images["right_wrist"])


def test_wrist_viewer_marks_stale_belief_after_half_second() -> None:
    viewer = WristV11WebViewer.__new__(WristV11WebViewer)
    viewer.belief = {"published_timestamp_us": 1}
    assert viewer._belief_stale_locked() is True


def test_wrist_gateway_executes_preserved_thor_belief_methods_directly() -> None:
    """Prevent a simplified wrist-only belief implementation from drifting."""
    for method in (
        "_belief_loop",
        "_belief_inputs_from_result",
        "_drain_shared_hand_selection",
        "_command_features_at",
        "_belief_state_features",
        "_belief_selection_from_result",
        "_publish_belief_target",
    ):
        assert getattr(WristBeliefGateway, method) is getattr(ThorRemoteGateway, method)


def test_wrist_belief_candidates_are_owned_by_their_camera() -> None:
    payload = {
        "objects": [
            {"track_id": "left_wrist:cup:1", "camera_id": "left_wrist"},
            {"track_id": "right_wrist:cup:2", "camera_id": "right_wrist"},
            {"track_id": "unknown:cup:3", "camera_id": "other"},
        ]
    }
    assert wrist_object_ids_for_hand(payload, "left") == {"left_wrist:cup:1"}
    assert wrist_object_ids_for_hand(payload, "right") == {"right_wrist:cup:2"}
