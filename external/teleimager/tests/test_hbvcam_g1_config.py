import unittest
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[1] / "config" / "hbvcam_4m2214hd_g1.yaml"


class HbvcamG1ConfigTest(unittest.TestCase):
    def test_two_d405_wrist_cameras_use_independent_recording_streams(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        wrists = [config["left_wrist_camera"], config["right_wrist_camera"]]

        self.assertEqual({camera["type"] for camera in wrists}, {"realsense"})
        self.assertTrue(all(camera["enable_recording_zmq"] for camera in wrists))
        self.assertTrue(all(not camera["enable_zmq"] for camera in wrists))
        self.assertTrue(all(not camera["enable_webrtc"] for camera in wrists))
        self.assertEqual(
            {camera["recording_zmq_port"] for camera in wrists}, {55559, 55560}
        )
        self.assertEqual(
            len({camera["serial_number"] for camera in wrists}), 2
        )


if __name__ == "__main__":
    unittest.main()
