import unittest

from teleimager.zmq_fps_monitor import StreamStats, recording_streams


class RecordingStreamsTest(unittest.TestCase):
    def test_selects_enabled_recording_ports(self):
        config = {
            "head_camera": {
                "enable_recording_zmq": True,
                "recording_zmq_port": 55558,
            },
            "disabled_camera": {
                "enable_recording_zmq": False,
                "recording_zmq_port": 55559,
            },
        }

        self.assertEqual(recording_streams(config), {"head_camera": 55558})

    def test_rejects_duplicate_ports(self):
        config = {
            name: {
                "enable_recording_zmq": True,
                "recording_zmq_port": 55558,
            }
            for name in ("left", "right")
        }

        with self.assertRaisesRegex(ValueError, "unique"):
            recording_streams(config)


class StreamStatsTest(unittest.TestCase):
    def test_reports_fps_bandwidth_and_sequence_gaps(self):
        stats = StreamStats()
        stats.observe(sequence=10, packet_size=100_000, width=640, height=480)
        stats.observe(sequence=12, packet_size=100_000, width=640, height=480)

        report = stats.report(elapsed_seconds=1.0)

        self.assertEqual(report, (2.0, 1.6, 1, 12, 640, 480))


if __name__ == "__main__":
    unittest.main()
