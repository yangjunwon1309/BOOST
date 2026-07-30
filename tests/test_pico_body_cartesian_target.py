import unittest

import numpy as np

from path_core.teleop.pico_body_cartesian_target import (
    PicoBodyCartesianTargetTracker,
    PicoBodyTrackingSample,
    PicoCartesianTargetConfig,
)


def z_quaternion(degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def sample(
    timestamp_sec: float,
    frame: int,
    *,
    left_dx: float = 0.0,
    left_yaw_deg: float = 0.0,
) -> PicoBodyTrackingSample:
    return PicoBodyTrackingSample(
        left_position=np.array([0.20 + left_dx, 0.25, 0.0]),
        right_position=np.array([0.20, -0.25, 0.0]),
        neck_position=np.zeros(3),
        left_orientation_wxyz=z_quaternion(left_yaw_deg),
        right_orientation_wxyz=z_quaternion(0.0),
        neck_orientation_wxyz=z_quaternion(0.0),
        source_timestamp_us=int(timestamp_sec * 1.0e6),
        frame_index=frame,
    )


class PicoBodyCartesianTargetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = PicoBodyCartesianTargetTracker(
            PicoCartesianTargetConfig(
                calibration_sec=0.5,
                min_calibration_frames=10,
                gap_hold_sec=0.1,
                position_jump_m=0.05,
                orientation_jump_rad=np.deg2rad(12.0),
                recovery_frames=3,
            )
        )
        self.left_anchor = np.eye(4)
        self.left_anchor[:3, 3] = [0.25, 0.25, 0.10]
        self.right_anchor = np.eye(4)
        self.right_anchor[:3, 3] = [0.25, -0.25, 0.10]
        for frame in range(41):
            self.output = self.update(sample(frame / 80.0, frame))

    def update(self, value: PicoBodyTrackingSample):
        return self.tracker.update(sample=value)

    def test_position_and_orientation_are_direct(self) -> None:
        output = self.update(
            sample(0.5125, 41, left_dx=0.02, left_yaw_deg=5.0)
        )
        self.assertTrue(output.valid)
        self.assertFalse(output.held)
        np.testing.assert_allclose(
            output.left_transform[:3, 3], [0.22, 0.25, 0.0], atol=1.0e-8
        )
        angle = np.deg2rad(5.0)
        expected_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(
            output.left_transform[:3, :3], expected_rotation, atol=1.0e-8
        )

    def test_jump_holds_until_three_frames_return_near_last_accepted(self) -> None:
        self.update(sample(0.5125, 41, left_dx=0.02))
        output = self.update(sample(0.525, 42, left_dx=0.08))
        self.assertTrue(output.held)
        self.assertTrue(output.reason.startswith("position_jump_"))
        for frame, dx in zip((43, 44), (0.081, 0.082)):
            output = self.update(
                sample(0.525 + (frame - 42) / 80.0, frame, left_dx=dx)
            )
            self.assertTrue(output.held)
            self.assertTrue(output.reason.startswith("position_jump_"))

        for frame, dx in zip((45, 46), (0.021, 0.022)):
            output = self.update(
                sample(0.525 + (frame - 42) / 80.0, frame, left_dx=dx)
            )
            self.assertTrue(output.held)
            self.assertTrue(output.reason.startswith("recovering_"))
        output = self.update(sample(0.5875, 47, left_dx=0.023))
        self.assertFalse(output.held)
        self.assertEqual(output.reason, "tracking")

    def test_timestamp_gap_holds(self) -> None:
        self.update(sample(0.5125, 41))
        output = self.update(sample(0.8, 42))
        self.assertTrue(output.held)
        self.assertTrue(output.reason.startswith("timestamp_gap_"))

    def test_orientation_jump_holds(self) -> None:
        self.update(sample(0.5125, 41, left_yaw_deg=2.0))
        output = self.update(sample(0.525, 42, left_yaw_deg=20.0))
        self.assertTrue(output.held)
        self.assertTrue(output.reason.startswith("orientation_jump_"))

    def test_position_gain_scales_displacement_about_calibration_anchor(self) -> None:
        tracker = PicoBodyCartesianTargetTracker(
            PicoCartesianTargetConfig(
                calibration_sec=0.5,
                min_calibration_frames=10,
                scale_xyz=(1.5, 2.0, 0.5),
            )
        )
        for frame in range(41):
            output = tracker.update(sample=sample(frame / 80.0, frame))
        anchor = output.left_transform[:3, 3].copy()
        output = tracker.update(
            sample=PicoBodyTrackingSample(
                left_position=np.array([0.22, 0.26, 0.02]),
                right_position=np.array([0.20, -0.25, 0.0]),
                neck_position=np.zeros(3),
                left_orientation_wxyz=z_quaternion(0.0),
                right_orientation_wxyz=z_quaternion(0.0),
                neck_orientation_wxyz=z_quaternion(0.0),
                source_timestamp_us=512500,
                frame_index=41,
            )
        )
        np.testing.assert_allclose(
            output.left_transform[:3, 3] - anchor,
            [0.03, 0.02, 0.01],
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            output.left_raw_position, [0.22, 0.26, 0.02], atol=1.0e-8
        )


if __name__ == "__main__":
    unittest.main()
