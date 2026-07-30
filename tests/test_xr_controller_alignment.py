import unittest

import numpy as np

from path_core.teleop.xr_controller_alignment import align_controller_pose


def z_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class XrControllerAlignmentTest(unittest.TestCase):
    def test_anchor_maps_exactly_to_robot_ee(self) -> None:
        source = np.eye(4)
        source[:3, :3] = z_rotation(30.0)
        source[:3, 3] = [0.3, -0.2, -0.4]
        target = np.eye(4)
        target[:3, :3] = z_rotation(-20.0)
        target[:3, 3] = [0.25, 0.20, 0.10]

        result = align_controller_pose(source, source, target)
        np.testing.assert_allclose(result, target, atol=1.0e-10)

    def test_metric_motion_and_relative_rotation_are_preserved(self) -> None:
        source = np.eye(4)
        source[:3, :3] = z_rotation(30.0)
        source[:3, 3] = [0.3, -0.2, -0.4]
        target = np.eye(4)
        target[:3, :3] = z_rotation(-20.0)
        target[:3, 3] = [0.25, 0.20, 0.10]
        current = source.copy()
        current[:3, 3] += [0.10, -0.03, 0.04]
        current[:3, :3] = source[:3, :3] @ z_rotation(12.0)

        result = align_controller_pose(current, source, target)
        np.testing.assert_allclose(
            result[:3, 3], target[:3, 3] + [0.10, -0.03, 0.04], atol=1.0e-10
        )
        np.testing.assert_allclose(
            result[:3, :3], target[:3, :3] @ z_rotation(12.0), atol=1.0e-10
        )


if __name__ == "__main__":
    unittest.main()
