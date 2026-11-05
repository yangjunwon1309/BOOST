import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from teleimager.camera_preview import FPSMeter, main, parse_args


class FPSMeterTest(unittest.TestCase):
    def test_reports_average_fps_over_recent_time_window(self):
        meter = FPSMeter(window_seconds=1.0)

        for timestamp_ns in (0, 250_000_000, 500_000_000, 750_000_000):
            meter.tick(timestamp_ns)

        self.assertEqual(meter.fps, 4.0)

        meter.tick(1_500_000_000)

        self.assertEqual(meter.fps, 2.0)

    def test_requires_two_frames_before_reporting_fps(self):
        meter = FPSMeter()

        meter.tick(123)

        self.assertEqual(meter.fps, 0.0)

    def test_rejects_non_monotonic_timestamps(self):
        meter = FPSMeter()
        meter.tick(10)

        with self.assertRaisesRegex(ValueError, "monotonic"):
            meter.tick(9)


class PreviewArgumentsTest(unittest.TestCase):
    def test_hbvcam_defaults(self):
        args = parse_args([])

        self.assertEqual(args.device, "/dev/video0")
        self.assertEqual((args.width, args.height), (1280, 480))
        self.assertEqual(args.fps, 30)

    def test_headless_cleanup_preserves_gui_error_message(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30
        capture.read.return_value = (True, np.zeros((480, 1280, 3), dtype=np.uint8))

        with patch(
            "teleimager.camera_preview.cv2.VideoCapture", return_value=capture
        ), patch(
            "teleimager.camera_preview.cv2.imshow", side_effect=cv2.error("no GUI")
        ), patch(
            "teleimager.camera_preview.cv2.destroyAllWindows",
            side_effect=cv2.error("no GUI"),
        ), patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "GUI is unavailable"):
                main([])

        capture.release.assert_called_once()
        capture.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 2)


if __name__ == "__main__":
    unittest.main()
