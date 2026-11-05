from pathlib import Path
import unittest


class PackagingTest(unittest.TestCase):
    def test_server_extra_declares_psutil_used_by_cli(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

        self.assertIn('"psutil"', pyproject)

    def test_declares_camera_preview_command(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

        self.assertIn(
            'teleimager-preview = "teleimager.camera_preview:main"', pyproject
        )

    def test_declares_zmq_fps_monitor_command(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

        self.assertIn(
            'teleimager-fps-monitor = "teleimager.zmq_fps_monitor:main"', pyproject
        )


if __name__ == "__main__":
    unittest.main()
