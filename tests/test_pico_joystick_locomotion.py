import math
import unittest

import numpy as np

from path_core.teleop.pico_joystick_locomotion import (
    LOCOMOTION_MODE_IDLE,
    LOCOMOTION_MODE_SLOW_WALK,
    PicoButtonGestureInterpreter,
    PicoJoystickLocomotionMapper,
    yaw_from_quaternion_wxyz,
)


class PicoButtonGestureInterpreterTest(unittest.TestCase):
    def test_x_release_without_a_selects_q(self) -> None:
        gestures = PicoButtonGestureInterpreter()
        self.assertNotIn("mode_q", gestures.update((False, False, True, False)))
        events = gestures.update((False, False, False, False))
        self.assertIn("mode_q", events)

    def test_y_release_without_b_selects_e(self) -> None:
        gestures = PicoButtonGestureInterpreter()
        gestures.update((False, False, False, True))
        events = gestures.update((False, False, False, False))
        self.assertIn("mode_e", events)

    def test_ax_cancels_x_release(self) -> None:
        gestures = PicoButtonGestureInterpreter()
        gestures.update((False, False, True, False))
        self.assertIn("ax", gestures.update((True, False, True, False)))
        events = gestures.update((False, False, False, False))
        self.assertNotIn("mode_q", events)

    def test_by_cancels_y_release(self) -> None:
        gestures = PicoButtonGestureInterpreter()
        gestures.update((False, False, False, True))
        self.assertIn("by", gestures.update((False, True, False, True)))
        events = gestures.update((False, False, False, False))
        self.assertNotIn("mode_e", events)

    def test_four_button_combo_cancels_both_individual_modes(self) -> None:
        gestures = PicoButtonGestureInterpreter()
        gestures.update((False, False, True, True))
        self.assertIn("start_combo", gestures.update((True, True, True, True)))
        events = gestures.update((False, False, False, False))
        self.assertNotIn("mode_q", events)
        self.assertNotIn("mode_e", events)

    def test_y_and_b_select_shared_control_hands_only_when_unchorded(self) -> None:
        gestures = PicoButtonGestureInterpreter()
        self.assertIn("select_left", gestures.update((False, False, False, True)))
        gestures.update((False, False, False, False))
        self.assertIn("select_right", gestures.update((False, True, False, False)))

        gestures = PicoButtonGestureInterpreter()
        events = gestures.update((False, True, False, True))
        self.assertIn("by", events)
        self.assertNotIn("select_left", events)


class PicoJoystickLocomotionMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = PicoJoystickLocomotionMapper()
        self.mapper.reset(0.0)

    def settle(self, left, right=(0.0, 0.0), steps=100):
        command = None
        for _ in range(steps):
            command = self.mapper.update(
                left, right, 0.02, current_yaw_rad=0.0
            )
        return command

    def test_deadzone_is_idle(self) -> None:
        command = self.mapper.update(
            (0.17, -0.17), (0.17, 0.0), 0.02, current_yaw_rad=0.0
        )
        self.assertEqual(command.mode, LOCOMOTION_MODE_IDLE)
        self.assertEqual(command.speed, -1.0)
        self.assertAlmostEqual(command.yaw_rate_rad_s, 0.0)

    def test_forward_reverse_and_lateral_caps(self) -> None:
        forward = self.settle((0.0, 1.0))
        self.assertEqual(forward.mode, LOCOMOTION_MODE_SLOW_WALK)
        self.assertAlmostEqual(forward.speed, 0.33, places=6)
        np.testing.assert_allclose(forward.movement, [1.0, 0.0, 0.0], atol=1e-6)

        self.mapper.reset(0.0)
        reverse = self.settle((0.0, -1.0))
        self.assertAlmostEqual(reverse.speed, 0.165, places=6)
        np.testing.assert_allclose(reverse.movement, [-1.0, 0.0, 0.0], atol=1e-6)

        self.mapper.reset(0.0)
        right = self.settle((1.0, 0.0))
        self.assertAlmostEqual(right.speed, 0.198, places=6)
        np.testing.assert_allclose(right.movement, [0.0, -1.0, 0.0], atol=1e-6)

    def test_linear_acceleration_limit(self) -> None:
        command = self.mapper.update(
            (0.0, 1.0), (0.0, 0.0), 0.02, current_yaw_rad=0.0
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(command.body_velocity_mps)), 0.35 * 0.02, places=6
        )

    def test_right_x_controls_rate_limited_yaw(self) -> None:
        command = self.mapper.update(
            (0.0, 0.0), (1.0, 0.8), 0.02, current_yaw_rad=0.0
        )
        self.assertAlmostEqual(command.yaw_rate_rad_s, -0.60 * 0.02, places=6)
        for _ in range(40):
            command = self.mapper.update(
                (0.0, 0.0), (1.0, -1.0), 0.02, current_yaw_rad=0.0
            )
        self.assertAlmostEqual(command.yaw_rate_rad_s, -0.30, places=6)
        self.assertLess(math.atan2(command.facing[1], command.facing[0]), 0.0)

    def test_hard_idle_clears_motion_and_preserves_measured_heading(self) -> None:
        self.settle((0.0, 1.0), steps=10)
        command = self.mapper.idle(yaw_rad=0.4, reason="state_auto_planning")
        self.assertEqual(command.mode, LOCOMOTION_MODE_IDLE)
        self.assertEqual(command.speed, -1.0)
        self.assertEqual(command.reason, "state_auto_planning")
        self.assertAlmostEqual(math.atan2(command.facing[1], command.facing[0]), 0.4)

    def test_quaternion_heading(self) -> None:
        yaw = 0.7
        q = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        self.assertAlmostEqual(yaw_from_quaternion_wxyz(q), yaw)


if __name__ == "__main__":
    unittest.main()
