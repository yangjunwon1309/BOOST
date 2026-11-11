from __future__ import annotations

import numpy as np

from boost.teleop.path_core.references.damped_se3 import (
    DampedSE3Reference,
    so3_exp,
    so3_log,
)


def _reach_time(distance: float, threshold: float = 0.10) -> float:
    reference = DampedSE3Reference()
    reference.reset(np.eye(4))
    target = np.eye(4)
    target[0, 3] = distance
    dt = 0.02
    elapsed = 0.0
    while elapsed < 10.0 and abs(target[0, 3] - reference.position[0]) >= threshold:
        reference.advance(target, dt)
        elapsed += dt
    return elapsed


def test_damped_translation_respects_norm_speed_limits() -> None:
    reference = DampedSE3Reference(v_max=0.06, omega_max=0.6)
    reference.reset(np.eye(4))
    target = np.eye(4)
    target[:3, 3] = [0.5, -0.3, 0.2]
    target[:3, :3] = so3_exp([1.2, -0.7, 0.4])

    for _ in range(500):
        reference.advance(target, 0.02)
        assert np.linalg.norm(reference.velocity) <= 0.06 + 1.0e-12
        assert np.linalg.norm(reference.angular_velocity) <= 0.6 + 1.0e-12


def test_damped_translation_matches_the_specified_release_distance_table() -> None:
    assert abs(_reach_time(0.20) - 1.7) <= 0.1
    assert abs(_reach_time(0.30) - 3.3) <= 0.1
    assert abs(_reach_time(0.40) - 5.0) <= 0.1


def test_exact_half_turn_uses_shortest_rotation_and_converges() -> None:
    target_rotation = so3_exp([0.0, 0.0, np.pi])
    shortest_error = so3_log(np.eye(3).T @ target_rotation)
    assert np.isclose(np.linalg.norm(shortest_error), np.pi, atol=1.0e-7)

    reference = DampedSE3Reference()
    reference.reset(np.eye(4))
    target = np.eye(4)
    target[:3, :3] = target_rotation
    for _ in range(1000):
        reference.advance(target, 0.01)
        assert np.linalg.norm(reference.angular_velocity) <= 0.6 + 1.0e-12
    assert np.linalg.norm(so3_log(reference.rotation.T @ target_rotation)) < 0.02
