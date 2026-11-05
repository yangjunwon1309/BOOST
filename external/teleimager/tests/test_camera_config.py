import unittest

from teleimager.camera_config import video_path_for_id


class CameraConfigTest(unittest.TestCase):
    def test_video_id_zero_maps_to_video_zero(self):
        self.assertEqual(video_path_for_id(0), "/dev/video0")

    def test_none_video_id_disables_device_path(self):
        self.assertIsNone(video_path_for_id(None))


if __name__ == "__main__":
    unittest.main()
