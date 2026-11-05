"""Fan one captured JPEG out without re-encoding it."""

from __future__ import annotations

import time

from .image_client import TripleRingBuffer
from .recording_protocol import RecordingFrame, encode_recording_frame


class CaptureFanout:
    def __init__(self, *, legacy_enabled: bool, recording_enabled: bool):
        self._legacy = TripleRingBuffer() if legacy_enabled else None
        self._recording = TripleRingBuffer() if recording_enabled else None
        self._sequence = 0

    def store(
        self,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        capture_wall_time_us: int | None = None,
        capture_monotonic_time_ns: int | None = None,
    ) -> None:
        if self._legacy is not None:
            self._legacy.write(jpeg)
        if self._recording is None:
            return

        wall_us = time.time_ns() // 1_000 if capture_wall_time_us is None else capture_wall_time_us
        monotonic_ns = time.monotonic_ns() if capture_monotonic_time_ns is None else capture_monotonic_time_ns
        packet = encode_recording_frame(
            RecordingFrame(
                sequence=self._sequence,
                capture_wall_time_us=wall_us,
                capture_monotonic_time_ns=monotonic_ns,
                width=width,
                height=height,
                jpeg=jpeg,
            )
        )
        self._recording.write(packet)
        self._sequence += 1

    def get_legacy(self) -> bytes | None:
        return self._legacy.read() if self._legacy is not None else None

    def get_recording(self) -> bytes | None:
        return self._recording.read() if self._recording is not None else None
