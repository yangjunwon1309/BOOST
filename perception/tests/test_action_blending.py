from __future__ import annotations

import numpy as np

from path_core.intention.action_blending import (
    EETarget,
    SharedControlConfig,
    TaskSpaceBlender,
    apply_explicit_orientation_to_wrist_targets,
    confidence_to_alpha,
    explicit_orientation_wxyz,
    quaternion_wxyz_to_rotation_matrix,
)


def _target(
    position=(0.4, 0.1, 0.2),
    confidence=0.9,
    uid="a",
    grip=0,
    ts=0,
    object_id="bottle",
):
    return EETarget(
        position=np.asarray(position, dtype=np.float64),
        confidence=float(confidence),
        object_id=object_id,
        grip_index=grip,
        goal_uid=uid,
        timestamp_us=ts,
    )


def test_confidence_piecewise_alpha_mapping():
    assert confidence_to_alpha(0.54, 0.55, 0.85) == 0.0
    assert confidence_to_alpha(0.85, 0.55, 0.85) == 1.0
    assert np.isclose(confidence_to_alpha(0.70, 0.55, 0.85), 0.5)


def test_explicit_orientation_requires_valid_flag_and_normalizes_wxyz():
    quaternion = [2.0, 0.0, 0.0, 0.0]
    assert explicit_orientation_wxyz({"quaternion": quaternion}) is None
    parsed = explicit_orientation_wxyz(
        {"orientation_valid": True, "quaternion": quaternion}
    )
    assert np.allclose(parsed, [1.0, 0.0, 0.0, 0.0])
    assert explicit_orientation_wxyz(
        {"orientation_valid": True, "quaternion": [0.0, 0.0, 0.0, 0.0]}
    ) is None


def test_explicit_orientation_replaces_only_selected_ipopt_wrist_rotation():
    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    right[:3, :3] = quaternion_wxyz_to_rotation_matrix(
        [np.cos(np.pi / 8.0), 0.0, np.sin(np.pi / 8.0), 0.0]
    )
    target_quaternion = np.array(
        [np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)],
        dtype=np.float64,
    )
    updated_left, updated_right, source = (
        apply_explicit_orientation_to_wrist_targets(
            left,
            right,
            selected_hand="left",
            orientation_wxyz=target_quaternion,
        )
    )
    assert source == "explicit_static_target"
    assert np.allclose(
        updated_left[:3, :3],
        quaternion_wxyz_to_rotation_matrix(target_quaternion),
    )
    assert np.allclose(updated_right, right)


def test_missing_explicit_orientation_preserves_both_human_wrist_rotations():
    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    left[:3, :3] = quaternion_wxyz_to_rotation_matrix(
        [np.cos(np.pi / 12.0), np.sin(np.pi / 12.0), 0.0, 0.0]
    )
    updated_left, updated_right, source = (
        apply_explicit_orientation_to_wrist_targets(
            left,
            right,
            selected_hand="left",
            orientation_wxyz=None,
        )
    )
    assert source == "human_wrist_prior"
    assert np.allclose(updated_left, left)
    assert np.allclose(updated_right, right)


def test_adaptive_alpha_rate_bounds_transfer_velocity():
    cfg = SharedControlConfig(
        startup_hold_s=0.0,
        v_transfer_max=0.25,
        alpha_rate_up=10.0,
        d_max_m=2.0,
    )
    blender = TaskSpaceBlender(cfg)
    result = blender.update(
        target=_target(position=(1.0, 0.0, 0.0), confidence=1.0, ts=0),
        p_human_left=np.zeros(3),
        p_human_right=np.array([0.0, -1.0, 0.0]),
        p_measured_left=np.zeros(3),
        p_measured_right=np.array([0.0, -1.0, 0.0]),
        now_us=1_000_000,
        dt=0.1,
    )
    assert result.gap_m == 1.0
    assert np.isclose(result.alpha, 0.025)
    assert result.alpha * result.gap_m <= cfg.v_transfer_max * 0.1 + 1e-12
    assert np.allclose(result.p_ref_left, [0.025, 0.0, 0.0])


def test_alpha_zero_returns_gmr_wrist_reference():
    cfg = SharedControlConfig(startup_hold_s=0.0)
    blender = TaskSpaceBlender(cfg)
    left = np.array([0.2, 0.2, 0.1])
    right = np.array([0.2, -0.2, 0.1])
    result = blender.update(
        target=_target(position=(0.4, 0.2, 0.1), confidence=0.1, ts=0),
        p_human_left=left,
        p_human_right=right,
        p_measured_left=left,
        p_measured_right=right,
        now_us=1_000_000,
        dt=0.02,
    )
    assert result.alpha == 0.0
    assert np.allclose(result.p_ref_left, left)
    assert np.allclose(result.p_ref_right, right)


def test_hand_hysteresis_and_lock_while_alpha_active():
    cfg = SharedControlConfig(
        startup_hold_s=0.0,
        hand_h_margin_m=0.05,
        hand_switch_confirm_n=2,
        lock_hand_while_active=True,
        alpha_rate_up=10.0,
        ref_lpf_alpha=1.0,
    )
    blender = TaskSpaceBlender(cfg)
    left = np.array([0.0, 0.2, 0.0])
    right = np.array([0.0, -0.2, 0.0])
    first = blender.update(
        target=_target(position=(0.0, 0.19, 0.0), confidence=1.0, uid="same"),
        p_human_left=left,
        p_human_right=right,
        p_measured_left=left,
        p_measured_right=right,
        now_us=1_000_000,
        dt=0.02,
    )
    assert first.hand == "left"
    assert first.alpha > 0.0
    locked = blender.update(
        target=_target(position=(0.0, -0.21, 0.0), confidence=1.0, uid="same"),
        p_human_left=left,
        p_human_right=right,
        p_measured_left=left,
        p_measured_right=right,
        now_us=1_020_000,
        dt=0.02,
    )
    assert locked.hand == "left"

    blender.alpha = 0.0
    maybe_switch = blender.update(
        target=_target(position=(0.0, -0.22, 0.0), confidence=0.0, uid="same"),
        p_human_left=left,
        p_human_right=right,
        p_measured_left=left,
        p_measured_right=right,
        now_us=1_040_000,
        dt=0.02,
    )
    assert maybe_switch.hand == "left"
    switched = blender.update(
        target=_target(position=(0.0, -0.23, 0.0), confidence=0.0, uid="same"),
        p_human_left=left,
        p_human_right=right,
        p_measured_left=left,
        p_measured_right=right,
        now_us=1_060_000,
        dt=0.02,
    )
    assert switched.hand == "right"


def test_target_switch_resets_alpha_and_lpf():
    cfg = SharedControlConfig(startup_hold_s=0.0, alpha_rate_up=10.0)
    blender = TaskSpaceBlender(cfg)
    left = np.zeros(3)
    right = np.array([0.0, -1.0, 0.0])
    blender.update(
        target=_target(position=(0.2, 0.0, 0.0), confidence=1.0, uid="old"),
        p_human_left=left,
        p_human_right=right,
        p_measured_left=left,
        p_measured_right=right,
        now_us=1_000_000,
        dt=0.1,
    )
    assert blender.alpha > 0.0
    result = blender.update(
        target=_target(
            position=(0.8, 0.0, 0.0),
            confidence=1.0,
            uid="new",
            object_id="box",
        ),
        p_human_left=left,
        p_human_right=right,
        p_measured_left=left,
        p_measured_right=right,
        now_us=1_100_000,
        dt=0.1,
    )
    assert result.target_switched
    assert np.allclose(result.p_contact, [0.8, 0.0, 0.0])
    assert result.alpha <= cfg.v_transfer_max * 0.1 / result.gap_m + 1e-12
