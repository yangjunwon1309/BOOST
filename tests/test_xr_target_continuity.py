from __future__ import annotations

import numpy as np

from boost.teleop.path_core.references.damped_se3 import DampedSE3Reference, so3_exp
from boost.teleop.path_core.references.xr_target_continuity import XRTargetContinuity


def test_single_position_and_half_turn_outliers_are_rejected() -> None:
    continuity = XRTargetContinuity()
    initial = np.eye(4)
    assert continuity.accept("left", initial, 1_000_000, 1.0) == "accepted"

    jumped = initial.copy()
    jumped[0, 3] = 0.5
    assert continuity.accept("left", jumped, 1_010_000, 1.01) == "rejected"

    turned = initial.copy()
    turned[:3, :3] = so3_exp([0.0, 0.0, np.pi])
    assert continuity.accept("right", initial, 1_000_000, 1.0) == "accepted"
    assert continuity.accept("right", turned, 1_010_000, 1.01) == "rejected"


def test_continuous_outlier_resynchronizes_after_timeout() -> None:
    continuity = XRTargetContinuity(
        max_linear_speed_mps=0.01,
        max_angular_speed_rps=0.1,
        resync_after_sec=0.3,
    )
    initial = np.eye(4)
    jumped = initial.copy()
    jumped[0, 3] = 0.2
    continuity.accept("left", initial, 1_000_000, 1.0)
    assert continuity.accept("left", jumped, 1_050_000, 1.05) == "rejected"
    assert continuity.accept("left", jumped, 1_150_000, 1.15) == "rejected"
    assert continuity.accept("left", jumped, 1_250_000, 1.25) == "rejected"
    assert continuity.accept("left", jumped, 1_350_000, 1.35) == "resync"
    assert continuity.resync_count["left"] == 1


def test_tracking_gap_recovers_even_when_the_jump_has_plausible_average_speed() -> None:
    continuity = XRTargetContinuity()
    initial = np.eye(4)
    continuity.accept("left", initial, 1_000_000, 1.0)
    jumped = initial.copy()
    jumped[0, 3] = 0.30
    assert continuity.accept("left", jumped, 2_000_000, 2.0) == "recovered"
    assert continuity.recovery_count["left"] == 1
    # Acceptance does not reset the filtered reference onto the new raw pose.
    continuity.filtered["left"] = initial.copy()
    filtered = continuity.tick(0.02)["left"]
    assert 0.0 < filtered[0, 3] < jumped[0, 3]


def test_explicit_loss_requires_handoff_and_duplicate_packets_cannot_clear_it() -> None:
    continuity = XRTargetContinuity()
    for side in ("left", "right"):
        continuity.accept(side, np.eye(4), 1_000_000, 1.0)
    continuity.mark_lost()
    assert continuity.accept("left", np.eye(4), 1_000_000, 1.01) == "rejected"
    assert continuity.tracking_lost["left"]
    assert continuity.accept("left", np.eye(4), 1_020_000, 1.02) == "recovered"
    assert not continuity.tracking_lost["left"]
    assert continuity.tracking_lost["right"]


def test_recovered_wrist_uses_existing_bounded_damping_from_held_pose() -> None:
    continuity = XRTargetContinuity()
    held = np.eye(4)
    continuity.accept("left", held, 1_000_000, 1.0)
    recovered = held.copy()
    recovered[0, 3] = 0.4
    recovered[:3, :3] = so3_exp([0.0, 0.0, 2.0])
    assert continuity.accept("left", recovered, 2_000_000, 2.0) == "recovered"
    continuity.reset_filter_to_accepted("left")
    reference = DampedSE3Reference(v_max=0.06, omega_max=0.6)
    reference.reset(held)
    output = reference.advance(continuity.tick(0.02)["left"], 0.02)
    assert np.linalg.norm(output[:3, 3] - held[:3, 3]) <= 0.06 * 0.02 + 1e-12
    assert not np.allclose(output, recovered)


def test_five_hz_lpf_has_the_expected_se3_step_response() -> None:
    continuity = XRTargetContinuity(
        max_linear_speed_mps=1000.0,
        max_angular_speed_rps=1000.0,
        cutoff_hz=5.0,
    )
    goal = np.eye(4)
    goal[:3, 3] = [1.0, 0.0, 0.0]
    goal[:3, :3] = so3_exp([0.0, 0.0, 0.8])
    continuity.accept("left", goal, 1_000_000, 1.0)
    continuity.reset_filter_to_accepted("left")
    continuity.accept("left", np.eye(4), 1_010_000, 1.01)
    for _ in range(4):
        result = continuity.tick(0.01)["left"]
    expected_beta = 1.0 - np.exp(-2.0 * np.pi * 5.0 * 0.04)
    assert result is not None
    assert np.isclose(result[0, 3], 1.0 - expected_beta, atol=1.0e-12)
    expected_rotation = so3_exp([0.0, 0.0, 0.8 * (1.0 - expected_beta)])
    np.testing.assert_allclose(result[:3, :3], expected_rotation, atol=1.0e-10)
