import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from teleimager.image_server import OpenCVCamera
from teleimager.recording_protocol import decode_recording_frame


class OpenCVCameraTest(unittest.TestCase):
    def test_uses_two_v4l2_buffers_to_sustain_camera_fps(self):
        capture = Mock()
        capture.set.return_value = True
        capture.read.return_value = (True, np.zeros((480, 1280, 3), dtype=np.uint8))

        with patch(
            "teleimager.image_server.cv2.VideoCapture", return_value=capture
        ):
            camera = OpenCVCamera(
                "head_camera",
                "/dev/video0",
                (480, 1280),
                30,
                enable_zmq=False,
                enable_webrtc=False,
            )
            camera.release()

        capture.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 2)

    def test_180_degree_rotation_reaches_zmq_recording_and_webrtc(self):
        image = np.zeros((48, 128, 3), dtype=np.uint8)
        image[:24, :64] = (0, 0, 230)
        image[:24, 64:] = (0, 230, 0)
        image[24:, :64] = (230, 0, 0)
        image[24:, 64:] = (230, 230, 230)
        capture = Mock()
        capture.set.return_value = True
        capture.read.return_value = (True, image)
        with patch('teleimager.image_server.cv2.VideoCapture', return_value=capture):
            camera = OpenCVCamera(
                'head_camera', '/dev/video0', (48, 128), 30,
                enable_zmq=True, enable_recording_zmq=True, enable_webrtc=True,
            )
        self.addCleanup(camera.release)
        camera.set_rotation(180)
        camera._update_frame()

        expected = image[::-1, ::-1]
        np.testing.assert_array_equal(camera.get_bgr_frame(), expected)
        jpeg = camera.get_jpeg_bytes()
        recorded = decode_recording_frame(camera.get_recording_bytes())
        self.assertEqual(recorded.jpeg, jpeg)
        self.assertEqual((recorded.height, recorded.width), (48, 128))
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        for y, x in ((12, 32), (12, 96), (36, 32), (36, 96)):
            np.testing.assert_allclose(decoded[y, x], expected[y, x], atol=2)


if __name__ == "__main__":
    unittest.main()
