from __future__ import annotations

import numpy as np

from path_core.intention.action_blending import (
    explicit_orientation_wxyz,
    rotation_target_from_relation,
)
from path_core.intention.reference_dynamics import (
    SecondOrderPositionReference,
    _second_order_transition,
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
