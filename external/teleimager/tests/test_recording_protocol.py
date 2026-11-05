import struct
import unittest

from teleimager.recording_protocol import (
    HEADER_SIZE,
    RecordingFrame,
    decode_recording_frame,
    encode_recording_frame,
)


JPEG = b"\xff\xd8payload\xff\xd9"


def make_frame(**overrides):
    values = {
        "sequence": 7,
        "capture_wall_time_us": 123_456,
        "capture_monotonic_time_ns": 987_654,
        "width": 1280,
        "height": 480,
        "jpeg": JPEG,
    }
    values.update(overrides)
    return RecordingFrame(**values)


class RecordingProtocolTest(unittest.TestCase):
    def test_recording_frame_round_trip(self):
        source = make_frame()

        packet = encode_recording_frame(source)

        self.assertEqual(len(packet), HEADER_SIZE + len(JPEG))
        self.assertEqual(decode_recording_frame(packet), source)

    def test_rejects_malformed_headers(self):
        cases = [
            (0, b"BAD!", "magic"),
            (4, struct.pack("<H", 99), "version"),
            (32, struct.pack("<I", 0), "dimensions"),
            (36, struct.pack("<I", 0), "dimensions"),
            (40, struct.pack("<I", 999), "JPEG length"),
        ]
        for offset, replacement, message in cases:
            with self.subTest(offset=offset, message=message):
                packet = bytearray(encode_recording_frame(make_frame()))
                packet[offset : offset + len(replacement)] = replacement
                with self.assertRaisesRegex(ValueError, message):
                    decode_recording_frame(packet)

    def test_rejects_invalid_jpeg(self):
        for jpeg in [b"payload\xff\xd9", b"\xff\xd8payload", b"\xff\xd8\xff\xd9trailing"]:
            with self.subTest(jpeg=jpeg):
                with self.assertRaisesRegex(ValueError, "JPEG"):
                    encode_recording_frame(make_frame(jpeg=jpeg))

    def test_rejects_trailing_packet_bytes(self):
        packet = encode_recording_frame(make_frame()) + b"extra"

        with self.assertRaisesRegex(ValueError, "JPEG length"):
            decode_recording_frame(packet)


if __name__ == "__main__":
    unittest.main()
