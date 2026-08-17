from __future__ import annotations

import numpy as np

from path_core.intention.action_blending import ContactComplianceConfig
from path_core.intention.reference_dynamics import (
    BimanualCouplingCorrector,
    ContactResistanceEstimator,
    SecondOrderPositionReference,
)


def test_kinetic_resistance_enters_blocks_and_releases_with_hysteresis():
    estimator = ContactResistanceEstimator(
        enter_ticks=2,
        block_ticks=4,
        release_ticks=5,
        ema_alpha=1.0,
        min_commanded_mps=0.01,
    )
    target = np.array([1.0, 0.0, 0.0])
    measured = np.zeros(3)

    assert estimator.update(np.zeros(3), measured, target, 0.02) == "free"
    assert estimator.update(np.array([0.01, 0.0, 0.0]), measured, target, 0.02) == "free"
    assert estimator.update(np.array([0.02, 0.0, 0.0]), measured, target, 0.02) == "resisted"
    assert estimator.update(np.array([0.03, 0.0, 0.0]), measured, target, 0.02) == "resisted"
    assert estimator.update(np.array([0.04, 0.0, 0.0]), measured, target, 0.02) == "blocked"

    for tick in range(4):
        position = np.array([0.05 + 0.01 * tick, 0.0, 0.0])
        assert estimator.update(position, position, target, 0.02) == "blocked"
    position = np.array([0.09, 0.0, 0.0])
    assert estimator.update(position, position, target, 0.02) == "free"


def test_static_resistance_detects_persistent_holding_error():
    estimator = ContactResistanceEstimator(
        enter_ticks=2,
        block_ticks=3,
        static_error_m=0.015,
        min_commanded_mps=0.02,
    )
    reference = np.array([0.02, 0.0, 0.0])
    measured = np.zeros(3)

    assert estimator.update(reference, measured, reference, 0.02) == "free"
    assert estimator.update(reference, measured, reference, 0.02) == "free"
    assert estimator.update(reference, measured, reference, 0.02) == "resisted"
    assert estimator.update(reference, measured, reference, 0.02) == "blocked"


def test_blocked_reference_freezes_and_does_not_resync_to_measurement():
    reference = SecondOrderPositionReference(
        omega_n=10.0,
        max_speed_mps=10.0,
        resync_distance_m=0.06,
    )
    reference.reset(np.array([0.10, 0.0, 0.0]))
    blocked = reference.step(
        np.array([0.20, 0.0, 0.0]),
        0.02,
        measured=np.zeros(3),
        resistance_state="blocked",
    )
    np.testing.assert_allclose(blocked, [0.10, 0.0, 0.0], atol=1.0e-12)

    free = reference.step(
        np.array([0.20, 0.0, 0.0]),
        0.02,
        measured=np.zeros(3),
        resistance_state="free",
    )
    assert 0.0 < free[0] < 0.10


def test_bimanual_coupling_latches_once_and_preserves_center():
    corrector = BimanualCouplingCorrector(gain=0.3)
    left = np.array([0.20, 0.0, 0.0])
    right = np.array([-0.20, 0.0, 0.0])

    initial_left, initial_right = corrector.update(
        left,
        right,
        both_blocked=True,
        same_object=True,
        measured_distance_m=0.40,
        dt=0.02,
    )
    assert corrector.latched_distance_m == 0.40
    assert not corrector.engaged
    np.testing.assert_allclose(initial_left, left)
    np.testing.assert_allclose(initial_right, right)

    stretched_left = np.array([0.22, 0.0, 0.0])
    stretched_right = np.array([-0.22, 0.0, 0.0])
    corrected_left, corrected_right = corrector.update(
        stretched_left,
        stretched_right,
        both_blocked=True,
        same_object=True,
        measured_distance_m=0.45,
        dt=0.02,
    )
    assert corrector.engaged
    assert abs(corrector.correction_m - 0.012) < 1.0e-12
    np.testing.assert_allclose(
        0.5 * (corrected_left + corrected_right),
        0.5 * (stretched_left + stretched_right),
    )
    assert np.linalg.norm(corrected_left - corrected_right) < 0.44
    assert corrector.latched_distance_m == 0.40


def test_bimanual_coupling_stays_off_without_strict_gate():
    corrector = BimanualCouplingCorrector()
    left = np.array([0.25, 0.0, 0.0])
    right = np.array([-0.25, 0.0, 0.0])
    out_left, out_right = corrector.update(
        left,
        right,
        both_blocked=True,
        same_object=False,
        measured_distance_m=0.40,
        dt=0.02,
    )
    assert not corrector.engaged
    assert corrector.latched_distance_m is None
    np.testing.assert_allclose(out_left, left)
    np.testing.assert_allclose(out_right, right)


def test_compliance_defaults_keep_release_longer_than_entry():
    config = ContactComplianceConfig()
    assert config.release_ticks > config.enter_ticks
    assert config.blocked_position_scale < config.resisted_position_scale < 1.0
    assert config.blocked_smooth_scale > config.resisted_smooth_scale > 1.0
    assert config.engage_error_m > config.release_error_m
