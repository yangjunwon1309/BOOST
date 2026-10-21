from __future__ import annotations

import numpy as np

from boost.teleop.legacy.intention.reference_dynamics import (
    SecondOrderOrientationReference,
    SecondOrderPositionReference,
    rotation_from_vector,
    rotation_vector,
)


def test_position_reference_keeps_advancing_during_large_measured_lag() -> None:
    reference = SecondOrderPositionReference(
        omega_n=10.0,
        max_speed_mps=10.0,
    )
    reference.reset(np.zeros(3))
    target = np.array([0.30, 0.0, 0.0])
    measured = np.zeros(3)

    outputs = [
        reference.step(target, 0.02, measured=measured)[0]
        for _ in range(50)
    ]

    assert np.all(np.diff(outputs) > 0.0)
    assert outputs[-1] > 0.25


def test_orientation_reference_keeps_advancing_during_measured_lag() -> None:
    reference = SecondOrderOrientationReference(
        omega_n=8.0,
        max_rate_rps=10.0,
    )
    reference.reset(np.eye(3))
    target = rotation_from_vector(np.array([0.0, 0.0, 1.0]))
    measured = np.eye(3)

    errors = []
    for _ in range(80):
        output = reference.step(target, 0.02, measured=measured)
        errors.append(float(np.linalg.norm(rotation_vector(target @ output.T))))

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 0.05
