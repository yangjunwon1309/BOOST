import unittest
from unittest.mock import Mock, patch

from teleimager import image_server
from teleimager.image_server import CameraFinder, RealSenseCamera, reset_realsense_devices


class BrokenDevices:
    def __iter__(self):
        raise RuntimeError("UVCIOC_CTRL_QUERY failed")


class RealSenseCameraTest(unittest.TestCase):
    def test_discovery_retries_iterator_failure(self):
        finder = CameraFinder.__new__(CameraFinder)
        rs = Mock()
        device = Mock()
        device.get_info.return_value = "serial"
        rs.context.return_value.query_devices.side_effect = [BrokenDevices(), [device]]
        with patch.object(finder, "get_realsense_module", return_value=rs), patch(
            "teleimager.image_server.time.sleep"
        ) as sleep:
            self.assertEqual(finder._list_realsense_serial_numbers(), ["serial"])
            sleep.assert_called_once_with(1)

    def test_discovery_persistent_failure_is_fatal(self):
        finder = CameraFinder.__new__(CameraFinder)
        rs = Mock()
        rs.context.return_value.query_devices.return_value = BrokenDevices()
        with patch.object(finder, "get_realsense_module", return_value=rs), patch(
            "teleimager.image_server.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                finder._list_realsense_serial_numbers()
            self.assertEqual(sleep.call_count, 2)

    def test_release_stops_native_pipeline_without_private_running_attribute(self):
        pipeline = Mock(spec=["stop"])
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.pipeline = pipeline
        camera._pipeline_started = True
        camera._cam_topic = "test"
        camera.release()
        camera.release()
        pipeline.stop.assert_called_once_with()


    def test_update_frame_uses_short_frame_timeout(self):
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.pipeline = Mock()
        camera._enable_depth = False
        camera.pipeline.wait_for_frames.return_value.get_color_frame.return_value = None
        camera._update_frame()
        camera.pipeline.wait_for_frames.assert_called_once_with(2000)

    def test_reset_waits_for_drop_and_reenumeration(self):
        rs = Mock()
        devices = {}
        for serial in ("left", "right", "other"):
            devices[serial] = Mock()
            devices[serial].get_info.return_value = serial
        everyone = list(devices.values())
        # initial scan, then: still present, dropped off, one back, both back
        rs.context.return_value.query_devices.side_effect = [
            everyone, everyone, [devices["other"]], [devices["left"], devices["other"]], everyone,
        ]
        with patch("teleimager.image_server.time.sleep"):
            back = reset_realsense_devices(rs, {"left", "right"})
        self.assertEqual(back, {"left", "right"})
        devices["left"].hardware_reset.assert_called_once_with()
        devices["right"].hardware_reset.assert_called_once_with()
        devices["other"].hardware_reset.assert_not_called()

    def test_reset_reports_device_that_never_returns(self):
        rs = Mock()
        device = Mock()
        device.get_info.return_value = "left"
        rs.context.return_value.query_devices.side_effect = [[device]] + [[]] * 1000
        clock = iter(range(1000))
        with patch("teleimager.image_server.time.sleep"), patch(
            "teleimager.image_server.time.monotonic", side_effect=lambda: next(clock)
        ):
            self.assertEqual(reset_realsense_devices(rs, {"left"}, timeout=5), set())

    def test_ready_timeout_is_two_seconds(self):
        self.assertEqual(image_server.CAMERA_READY_TIMEOUT_S, 2.0)


if __name__ == "__main__":
    unittest.main()
