import unittest

from teleimager.capture_fanout import CaptureFanout
from teleimager.recording_protocol import decode_recording_frame


JPEG = b"\xff\xd8payload\xff\xd9"


class CaptureFanoutTest(unittest.TestCase):
    def test_reuses_original_jpeg_for_legacy_and_adds_recording_metadata(self):
        fanout = CaptureFanout(legacy_enabled=True, recording_enabled=True)

        fanout.store(
            JPEG,
            width=1280,
            height=480,
            capture_wall_time_us=123,
            capture_monotonic_time_ns=456,
        )

        self.assertIs(fanout.get_legacy(), JPEG)
        decoded = decode_recording_frame(fanout.get_recording())
        self.assertEqual(decoded.jpeg, JPEG)
        self.assertEqual(decoded.sequence, 0)
        self.assertEqual(decoded.capture_wall_time_us, 123)
        self.assertEqual(decoded.capture_monotonic_time_ns, 456)

    def test_sequence_increments_once_per_stored_capture(self):
        fanout = CaptureFanout(legacy_enabled=False, recording_enabled=True)

        fanout.store(JPEG, width=1280, height=480)
        first = decode_recording_frame(fanout.get_recording())
        fanout.store(JPEG, width=1280, height=480)
        second = decode_recording_frame(fanout.get_recording())

        self.assertEqual((first.sequence, second.sequence), (0, 1))

    def test_legacy_only_mode_does_not_build_recording_packet(self):
        fanout = CaptureFanout(legacy_enabled=True, recording_enabled=False)

        fanout.store(JPEG, width=1280, height=480)

        self.assertIs(fanout.get_legacy(), JPEG)
        self.assertIsNone(fanout.get_recording())

    def test_recording_only_mode_does_not_populate_legacy_slot(self):
        fanout = CaptureFanout(legacy_enabled=False, recording_enabled=True)

        fanout.store(JPEG, width=1280, height=480)

        self.assertIsNone(fanout.get_legacy())
        self.assertEqual(decode_recording_frame(fanout.get_recording()).jpeg, JPEG)


if __name__ == "__main__":
    unittest.main()
