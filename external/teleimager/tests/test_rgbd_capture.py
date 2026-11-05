"""RGB capture must stay independent from native depth alignment."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from teleimager.image_server import RealSenseCamera


def camera_with_frames():
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.pipeline = Mock()
    camera.align = Mock()
    camera._enable_depth = True
    camera._rgbd_port = 55561
    camera._rgbd_worker = None
    camera._rgbd_calibration = {"factory": "test"}
    camera._pipeline_started = False
    camera._serial_number = "test-d435i"
    camera._enable_webrtc = False
    camera._enable_zmq = True
    camera._enable_recording_zmq = True
    camera._ready = threading.Event()
    camera._apply_rotation = lambda value: value
    camera._store_captured_jpeg = Mock()
    camera.g_depth_scale = .001
    camera._cam_topic = "test"
    camera.intrinsics = SimpleNamespace(width=16, height=12, fx=10, fy=11, ppx=8, ppy=6, coeffs=[0] * 5)
    frames = camera.pipeline.wait_for_frames.return_value
    frames.get_color_frame.return_value.get_data.return_value = np.full((12, 16, 3), 120, np.uint8)
    frames.get_color_frame.return_value.get_frame_number.return_value = 17
    frames.get_depth_frame.return_value.get_data.return_value = np.full((12, 16), 700, np.uint16)
    return camera, frames


def test_rgbd_and_existing_rgb_ports_share_the_same_encoded_color_frame():
    camera, frames = camera_with_frames()
    worker = Mock()
    with patch("teleimager.rgbd_stream.RgbdAlignmentProcess", return_value=worker):
        try:
            camera._update_frame()
            camera.align.process.assert_not_called()  # SDK alignment cannot hold the capture GIL.
            depth, jpeg, metadata = worker.submit.call_args.args
            assert depth.shape == (12, 16)
            assert (depth == 700).all()
            assert metadata["sequence"] == 17
            assert metadata["depth_scale_m"] == .001
            assert metadata["fx"] == 10
            assert not np.shares_memory(depth, frames.get_depth_frame.return_value.get_data.return_value)
            camera._store_captured_jpeg.assert_called_once_with(jpeg, 16, 12)
            assert camera._ready.is_set()
        finally:
            camera.release()
    worker.close.assert_called_once()


def test_missing_depth_never_reuses_a_previous_frameset_depth():
    camera, frames = camera_with_frames()
    worker = Mock()
    with patch("teleimager.rgbd_stream.RgbdAlignmentProcess", return_value=worker):
        try:
            camera._update_frame()
            worker.reset_mock()
            frames.get_depth_frame.return_value = None
            camera._update_frame()
            worker.submit.assert_not_called()
            assert camera._store_captured_jpeg.call_count == 2
        finally:
            camera.release()
