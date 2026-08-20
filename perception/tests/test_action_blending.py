from __future__ import annotations

import numpy as np

from path_core.intention.action_blending import (
    explicit_orientation_wxyz,
    rotation_target_from_relation,
)
from path_core.intention.reference_dynamics import (
    OrientationAuthority,
    OrientationTargetGate,
    SecondOrderOrientationReference,
    SecondOrderPositionReference,
    _second_order_transition,
    rotation_from_vector,
    rotation_vector,
    slerp_so3,
)


def test_explicit_orientation_remains_legacy_diagnostic_parser():
    assert explicit_orientation_wxyz({"quaternion": [1, 0, 0, 0]}) is None
    np.testing.assert_allclose(
        explicit_orientation_wxyz({"orientation_valid": True, "quaternion": [2, 0, 0, 0]}),
        [1, 0, 0, 0],
    )


def test_perpendicular_relation_refuses_singular_axis():
    assert rotation_target_from_relation(
        np.eye(3), np.array([0.0, 1.0, 0.0]), "perpendicular", np.array([0.0, 1.0, 0.0])
    ) is None


def test_overdamped_transition_is_stable_and_step_has_no_overshoot():
    for omega_n in (3.0, 12.0, 20.0):
        for dt in (0.005, 0.02, 0.05):
            phi = np.asarray(_second_order_transition(omega_n, 2.0, dt)).reshape(2, 2)
            assert np.max(np.abs(np.linalg.eigvals(phi))) < 1.0
    reference = SecondOrderPositionReference(omega_n=12.0)
    values = [reference.step(np.ones(3), 0.02)[0] for _ in range(200)]
    assert max(values) <= 1.0 + 1.0e-9
    assert values[-1] > 0.99


def test_orientation_reference_dt_zero_is_identity():
    reference = SecondOrderOrientationReference(zeta=2.0, omega_n=4.0)
    reference.reset(np.eye(3))
    out = reference.step(rotation_from_vector(np.array([0.0, 0.0, 0.5])), 0.0)
    assert np.linalg.norm(rotation_vector(out)) < 1.0e-9


def test_orientation_reference_converges_monotonically():
    reference = SecondOrderOrientationReference(zeta=2.0, omega_n=4.0)
    reference.reset(np.eye(3))
    target = rotation_from_vector(np.array([0.0, 0.0, 0.5]))
    previous = float("inf")
    for _ in range(250):
        out = reference.step(target, 0.02)
        error = float(np.linalg.norm(rotation_vector(target @ out.T)))
        assert error <= previous + 1.0e-9
        previous = error
    assert previous < np.deg2rad(0.5)


def test_position_ramp_lag_matches_overdamped_formula():
    zeta, omega_n, velocity, dt = 1.2, 10.0, 0.2, 0.002
    reference = SecondOrderPositionReference(zeta=zeta, omega_n=omega_n, max_speed_mps=10.0)
    reference.reset(np.zeros(3))
    for tick in range(8000):
        target = np.array([velocity * tick * dt, 0.0, 0.0])
        out = reference.step(target, dt)
    lag = float(target[0] - out[0])
    expected = 2.0 * zeta / omega_n * velocity
    assert abs(lag - expected) / expected < 0.05


def test_orientation_gate_rejects_large_target_and_accepts_dwell():
    gate = OrientationTargetGate(max_angle_rad=np.deg2rad(60.0), accept_dwell=3)
    identity = np.eye(3)
    assert not gate.accept(rotation_from_vector(np.array([0.0, 0.0, np.deg2rad(120.0)])), identity, 0.02)
    assert gate.last_reason.startswith("angle_")
    target = rotation_from_vector(np.array([0.0, 0.0, np.deg2rad(25.0)]))
    assert not gate.accept(target, identity, 0.02)
    assert not gate.accept(target, identity, 0.02)
    assert gate.accept(target, identity, 0.02)


def test_slerp_so3_endpoints_and_geodesic_midpoint():
    source = np.eye(3)
    target = rotation_from_vector(np.array([0.0, 0.0, np.pi / 2.0]))
    np.testing.assert_allclose(slerp_so3(source, target, 0.0), source)
    np.testing.assert_allclose(slerp_so3(source, target, 1.0), target)
    midpoint = slerp_so3(source, target, 0.5)
    assert abs(np.linalg.norm(rotation_vector(midpoint)) - np.pi / 4.0) < 1.0e-9


def test_orientation_authority_is_continuous_and_rate_limited():
    authority = OrientationAuthority(ema_alpha=1.0, max_authority_rate_per_s=1.2)
    assert authority.update(0.44, gate_ok=True, dt=0.1) == 0.0
    # It engages above enter confidence but can only ramp 0.12 per tick.
    assert authority.update(1.0, gate_ok=True, dt=0.1) == 0.12
    assert authority.update(1.0, gate_ok=True, dt=0.1) == 0.24
    # Leaving the Schmitt band releases continuously at the same rate.
    assert authority.update(0.0, gate_ok=True, dt=0.1) == 0.12


def test_orientation_gate_tapers_before_max_angle():
    gate = OrientationTargetGate(max_angle_rad=1.0, taper_start_fraction=0.5, accept_dwell=1)
    identity = np.eye(3)
    taper = gate.authority(rotation_from_vector(np.array([0.0, 0.0, 0.75])), identity, 1.0)
    assert 0.0 < taper < 1.0
    assert gate.authority(rotation_from_vector(np.array([0.0, 0.0, 1.01])), identity, 1.0) == 0.0
