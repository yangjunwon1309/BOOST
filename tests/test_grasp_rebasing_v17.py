import numpy as np

from path_core.intention.reference_dynamics import (
    FirstOrderPositionTargetFilter,
    GraspTransitionDetector,
    OperatorFrameOffset,
    OperatorRotationOffset,
)


def test_alpha_position_lpf_seeds_at_operator_then_filters_blended_target():
    lpf = FirstOrderPositionTargetFilter(alpha=0.30)
    operator = np.array([0.10, 0.0, 0.0])
    blended = np.array([0.30, 0.0, 0.0])

    # Enabling alpha assistance begins continuously at the operator command.
    assert np.allclose(lpf.step(blended, seed=operator), operator)
    # Each later target update applies the ordinary first-order EWMA.
    assert np.allclose(lpf.step(blended), [0.16, 0.0, 0.0])


def test_operator_frame_offset_is_stationary_then_washes_out_by_path_length():
    offset = OperatorFrameOffset(washout_per_m=4.6, max_offset_m=0.30)
    offset.track(np.array([0.15, 0.0, 0.0]), np.zeros(3))

    # Authority release at a stationary controller has no target jump.
    assert np.allclose(offset.apply(np.zeros(3)), [0.15, 0.0, 0.0])
    assert np.allclose(offset.apply(np.zeros(3)), [0.15, 0.0, 0.0])

    # 50 cm of deliberate VR motion leaves exp(-4.6 * .5) of the offset.
    released = offset.apply(np.array([0.50, 0.0, 0.0]))
    assert np.isclose(released[0] - 0.50, 0.15 * np.exp(-4.6 * 0.50))


def test_rotation_rebase_captures_without_a_grasp_handoff_jump():
    rotation = OperatorRotationOffset(washout_per_rad=2.0)
    reference = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    command = np.eye(3)
    rotation.capture(reference, command)
    assert np.allclose(rotation.apply(command), reference)


def test_grasp_timeout_only_leaves_closing_and_manual_override_remains_available():
    detector = GraspTransitionDetector(closing_timeout_s=0.60)
    for tick in range(8):
        state = detector.update(
            commanded_closed=True,
            jaw_gap_m=None,
            p_camera=None,
            p_goal=None,
            now_s=0.1 * tick,
            dt=0.1,
        )
    assert state == "reaching"
    assert not detector.channels["jaw"]
    assert not detector.channels["comotion"]

    state = detector.update(
        commanded_closed=True,
        jaw_gap_m=None,
        p_camera=None,
        p_goal=None,
        now_s=0.9,
        dt=0.1,
        manual_override=True,
    )
    assert state == "holding"
    assert detector.override_count == 1


def test_comotion_channel_enters_holding_with_camera_stationary_goal_motion():
    detector = GraspTransitionDetector(comotion_min_excitation_m=0.02)
    state = "reaching"
    for tick in range(7):
        state = detector.update(
            commanded_closed=True,
            jaw_gap_m=None,
            p_camera=np.array([0.0, 0.0, 0.40]),
            p_goal=np.array([0.02 * tick, 0.0, 0.40]),
            now_s=0.1 * tick,
            dt=0.1,
        )
    assert detector.channels["comotion"]
    assert detector.comotion_log_ratio is not None
    assert state == "holding"
