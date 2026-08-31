import numpy as np
from scipy.spatial.transform import Rotation

from path_core.teleop.g1_gripper_ik import G1GripperInverseKinematicsSolver
from path_core.teleop.rotation_decomposition import decompose_rotation_aa


def _finger_payload(active_tip: int) -> dict:
    fingers = np.zeros((25, 4, 4))
    fingers[4, 0, 3] = 1.0
    fingers[active_tip, 0, 3] = 1.0
    return {"position": fingers}


def test_gripper_solver_returns_7d_targets_for_each_selectable_finger():
    for side in ("left", "right"):
        solver = G1GripperInverseKinematicsSolver(side)
        for fingertip in (9, 14, 19, 24):
            target = solver(_finger_payload(fingertip))
            assert target.shape == (7,)
            assert np.isfinite(target).all()


def test_axis_angle_twist_swing_reconstructs_original_rotation():
    axis_angle = np.array([[0.2, -0.3, 0.4], [-0.7, 0.1, 0.2]])
    twist, swing = decompose_rotation_aa(axis_angle, np.array([0.0, 1.0, 0.0]))
    reconstructed = Rotation.from_quat(twist[:, [1, 2, 3, 0]]) * Rotation.from_quat(
        swing[:, [1, 2, 3, 0]]
    )
    assert np.allclose(
        reconstructed.as_matrix(), Rotation.from_rotvec(axis_angle).as_matrix()
    )
