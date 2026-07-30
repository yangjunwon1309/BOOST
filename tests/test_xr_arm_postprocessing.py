import unittest

import numpy as np

from path_core.teleop.xr_arm_postprocessing import XrArmCommandPostprocessor


class XrArmCommandPostprocessorTest(unittest.TestCase):
    def test_matches_xr_weighted_moving_filter(self) -> None:
        post = XrArmCommandPostprocessor(
            velocity_limit_rad_s=1000.0,
            control_dt_s=0.02,
            data_size=2,
        )
        outputs = []
        for value in (1.0, 2.0, 3.0, 4.0):
            outputs.append(
                post.apply(
                    np.array([value, -value]),
                    np.zeros(2),
                ).filtered_q
            )
        np.testing.assert_allclose(outputs[0], [1.0, -1.0])
        np.testing.assert_allclose(outputs[1], [2.0, -2.0])
        np.testing.assert_allclose(outputs[2], [3.0, -3.0])
        np.testing.assert_allclose(outputs[3], [3.0, -3.0])

    def test_velocity_limit_uses_one_common_scale(self) -> None:
        post = XrArmCommandPostprocessor(
            velocity_limit_rad_s=20.0,
            control_dt_s=1.0 / 250.0,
            data_size=3,
        )
        result = post.apply(
            np.array([0.8, -0.4, 0.2]),
            np.zeros(3),
        )
        self.assertTrue(result.rate_limited)
        self.assertAlmostEqual(result.motion_scale, 10.0)
        np.testing.assert_allclose(result.command_q, [0.08, -0.04, 0.02])
        self.assertAlmostEqual(result.max_command_step_rad, 0.08)

    def test_duplicate_solution_does_not_advance_filter_queue(self) -> None:
        post = XrArmCommandPostprocessor(
            velocity_limit_rad_s=1000.0,
            control_dt_s=0.02,
            data_size=1,
        )
        post.apply(np.array([1.0]), np.zeros(1))
        post.apply(np.array([1.0]), np.zeros(1))
        post.apply(np.array([2.0]), np.zeros(1))
        result = post.apply(np.array([3.0]), np.zeros(1))
        np.testing.assert_allclose(result.filtered_q, [3.0])


if __name__ == "__main__":
    unittest.main()
