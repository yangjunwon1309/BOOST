"""Latest-frame subscriber for the teleimager TIM2 recording streams.

`teleimager.image_client.ImageClient` speaks the legacy raw-JPEG streams, which
carry no capture time. The recorder needs to know *when* each image was taken so
it can be lined up with the action computed in the same control cycle, so this
client speaks the timestamped TIM2 protocol instead.

Two properties matter for recording quality:

Latency -- every socket is CONFLATE + RCVHWM 1, so ZMQ keeps only the newest
message. A control loop that falls behind therefore skips stale frames instead
of draining a queue that grows further behind real time with every cycle.

CPU -- the original JPEG bytes are carried through untouched. Only cameras
asked to `decode` pay for `cv2.imdecode`, and that happens on the subscriber
thread rather than the control loop. The wrist images are never decoded on this
machine at all: they are written to disk as the exact bytes the robot encoded.
"""

from __future__ import annotations

import threading
import time
import math
from typing import Optional

import cv2
import numpy as np
import zmq

from teleimager.recording_protocol import decode_recording_frame

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


class RecordingFrame:
    """One captured image plus the metadata needed to align it with an action."""

    __slots__ = ("jpg", "sequence", "capture_wall_time_us", "capture_monotonic_time_ns",
                 "width", "height", "receive_monotonic_ns", "_bgr", "_decoded",
                 "source_capture_wall_time_us", "clock_uncertainty_us")

    def __init__(self, jpg, sequence, capture_wall_time_us, capture_monotonic_time_ns,
                 width, height, receive_monotonic_ns, bgr=None, clock_calibration=None):
        self.jpg = jpg
        self.sequence = sequence
        self.source_capture_wall_time_us = capture_wall_time_us if clock_calibration else None
        self.clock_uncertainty_us = clock_calibration.uncertainty_us if clock_calibration else None
        self.capture_wall_time_us = (
            clock_calibration.capture_wall_time_us(capture_monotonic_time_ns)
            if clock_calibration else capture_wall_time_us
        )
        self.capture_monotonic_time_ns = capture_monotonic_time_ns
        self.width = width
        self.height = height
        self.receive_monotonic_ns = receive_monotonic_ns
        self._bgr = bgr
        self._decoded = bgr is not None

    @property
    def bgr(self) -> Optional[np.ndarray]:
        """Decoded image. Decoded on first access if the stream was not decoding."""
        if not self._decoded:
            self._bgr = cv2.imdecode(np.frombuffer(self.jpg, np.uint8), cv2.IMREAD_COLOR)
            self._decoded = True
        return self._bgr

    def age_ms(self) -> float:
        """Milliseconds between capture on the robot and now, on this machine's clock."""
        return (time.time() * 1e6 - self.capture_wall_time_us) / 1000.0


class _StreamThread(threading.Thread):
    def __init__(self, host, port, name, decode, context, clock_calibration=None):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._name = name
        self._decode = decode
        self._context = context
        self._clock_calibration = clock_calibration
        self._socket = None
        self._running = True
        self._lock = threading.Lock()
        self._latest: Optional[RecordingFrame] = None
        self._received = 0
        self._malformed = 0
        self._last_sequence: Optional[int] = None
        self._sequence_gaps = 0
        self._last_packet_monotonic_ns = None
        self._last_packet_error = None
        self._error = None

    def run(self):
        try:
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            # Keep only the newest frame: a slow consumer must skip, never queue.
            self._socket.setsockopt(zmq.RCVHWM, 1)
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.setsockopt(zmq.RCVTIMEO, 200)
            self._socket.connect(f"tcp://{self._host}:{self._port}")

            while self._running:
                try:
                    packet = self._socket.recv()
                except zmq.Again:
                    continue
                with self._lock:
                    self._last_packet_monotonic_ns = time.monotonic_ns()
                try:
                    decoded = decode_recording_frame(packet)
                    bgr = None
                    if self._decode:
                        # Decoding stays off the control loop and outside its lock.
                        bgr = cv2.imdecode(np.frombuffer(decoded.jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise ValueError("JPEG decoder returned no image")
                    frame = RecordingFrame(
                        jpg=decoded.jpeg,
                        sequence=decoded.sequence,
                        capture_wall_time_us=decoded.capture_wall_time_us,
                        capture_monotonic_time_ns=decoded.capture_monotonic_time_ns,
                        width=decoded.width,
                        height=decoded.height,
                        receive_monotonic_ns=time.monotonic_ns(),
                        bgr=bgr,
                        clock_calibration=self._clock_calibration,
                    )
                except (ValueError, TypeError) as error:
                    with self._lock:
                        self._malformed += 1
                        self._last_packet_error = str(error)
                        malformed = self._malformed
                    if malformed <= 3:
                        logger_mp.warning(f"[RecordingImageClient] {self._name}: bad packet ({error})")
                    continue

                with self._lock:
                    if self._last_sequence is not None and decoded.sequence > self._last_sequence + 1:
                        self._sequence_gaps += decoded.sequence - self._last_sequence - 1
                    self._last_sequence = decoded.sequence
                    self._last_packet_error = None
                    self._latest = frame
                    self._received += 1
        except Exception as error:
            with self._lock:
                self._error = str(error) or type(error).__name__
            logger_mp.error(f"[RecordingImageClient] {self._name}: stream stopped ({error})")
        finally:
            self._running = False
            if self._socket is not None:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass

    def latest(self) -> Optional[RecordingFrame]:
        with self._lock:
            return self._latest

    def stats(self):
        with self._lock:
            return {
                "received": self._received,
                "malformed": self._malformed,
                "sequence_gaps": self._sequence_gaps,
                "last_sequence": self._last_sequence,
            }

    def health(self, now_ns=None, max_age_ms=None):
        """Nonblocking snapshot; frame age uses local valid-receive time only."""
        if max_age_ms is not None and (not math.isfinite(max_age_ms) or max_age_ms < 0):
            raise ValueError("max_age_ms must be finite and nonnegative")
        with self._lock:
            frame = self._latest
            received_ns = frame.receive_monotonic_ns if frame is not None else None
            result = dict(received=self._received, malformed=self._malformed,
                          last_sequence=self._last_sequence, error=self._error,
                          last_packet_error=self._last_packet_error,
                          last_packet_monotonic_ns=self._last_packet_monotonic_ns,
                          last_frame_receive_monotonic_ns=received_ns)
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        age_ms = (now_ns - received_ns) / 1_000_000 if received_ns is not None else None
        result.update(alive=self.is_alive(), has_frame=frame is not None,
                      last_frame_age_ms=age_ms,
                      stale=frame is None or (max_age_ms is not None and age_ms > max_age_ms))
        return result

    def stop(self):
        self._running = False


class RecordingImageClient:
    """Subscribes to every camera in `cam_config` that publishes a TIM2 stream."""

    #: Only the head image is needed as pixels (XR display + binocular split).
    DECODED_CAMERAS = ("head_camera",)

    def __init__(self, host, cam_config, decode_cameras=None, cameras=None,
                 clock_sync=False, request_port=60000):
        """`cameras` limits the subscription; None subscribes to every TIM2 stream.

        `host` is either one address for every camera or a mapping from camera
        name to the board that camera is wired into (see multi_host_image_client).

        Restricting it matters when teleoperating without --record: subscribing to
        the wrist streams would pull their bandwidth across the link for frames
        nothing reads.
        """
        self._hosts = host if isinstance(host, dict) else None
        self._host = host
        self._threads = {}
        self._closed = False
        self.clock_calibrations = {}

        # Validate all streams before starting any of them. A bad later camera
        # must not leak the already-started subscribers of an unreturned client.
        decode_cameras = self.DECODED_CAMERAS if decode_cameras is None else decode_cameras
        streams = []
        for name, camera in cam_config.items():
            if not isinstance(camera, dict) or not camera.get("enable_recording_zmq", False):
                continue
            if cameras is not None and name not in cameras:
                continue
            port = camera.get("recording_zmq_port")
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise ValueError(f"[RecordingImageClient] {name} has no valid recording_zmq_port")
            camera_host = self._hosts[name] if self._hosts is not None else host
            streams.append((name, camera_host, port, name in decode_cameras))
        if not streams:
            raise RuntimeError(
                "No camera in the server config has enable_recording_zmq set; "
                "recording has no image source."
            )

        if clock_sync:
            from teleimager.clock_sync import calibrate_camera_clock

            enabled_hosts = {camera_host for _, camera_host, _, _ in streams}
            for camera_host in sorted(enabled_hosts):
                calibration = calibrate_camera_clock(camera_host, request_port)
                if calibration.uncertainty_us > 20_000:
                    raise RuntimeError(f"camera clock uncertainty on {camera_host} exceeds 20 ms")
                self.clock_calibrations[camera_host] = calibration
                logger_mp.info(f"[RecordingImageClient] clock {camera_host}: {calibration.metadata()}")

        self._context = zmq.Context.instance()
        try:
            for name, camera_host, port, decode in streams:
                clock_args = ({"clock_calibration": self.clock_calibrations[camera_host]} if clock_sync else {})
                thread = _StreamThread(camera_host, port, name, decode, self._context, **clock_args)
                self._threads[name] = thread
                thread.start()
                logger_mp.info(f"[RecordingImageClient] {name}: tcp://{camera_host}:{port}"
                               f"{' (decoded)' if decode else ' (jpeg passthrough)'}")
        except BaseException:
            self.close()
            raise

    def has(self, name) -> bool:
        return name in self._threads

    def get_frame(self, name) -> Optional[RecordingFrame]:
        thread = self._threads.get(name)
        return thread.latest() if thread else None

    def get_head_frame(self):
        return self.get_frame("head_camera")

    def get_left_wrist_frame(self):
        return self.get_frame("left_wrist_camera")

    def get_right_wrist_frame(self):
        return self.get_frame("right_wrist_camera")

    def wait_until_ready(self, timeout=10.0) -> bool:
        """Wait for frames, failing immediately if a subscriber has stopped."""
        if self._closed:
            raise RuntimeError("RecordingImageClient is closed")
        deadline = time.monotonic() + timeout
        while True:
            health = self.health()
            failed = [f"{name}: {state['error'] or 'subscriber stopped'}"
                      for name, state in health.items() if state['error'] or not state['alive']]
            if failed:
                raise RuntimeError("Recording camera stream failed: " + "; ".join(failed))
            missing = [name for name, state in health.items() if not state['has_frame']]
            if not missing:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger_mp.warning(f"[RecordingImageClient] no frames yet from: {missing}")
                return False
            time.sleep(min(0.05, remaining))

    def health(self, now_ns=None, max_age_ms=None):
        """Per-camera health; local receive age is separate from capture lag.

        Cached frames remain accessible. The caller decides how stale images
        affect recording, without this API waiting for a camera or retrying it.
        """
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        return {name: thread.health(now_ns=now_ns, max_age_ms=max_age_ms)
                for name, thread in self._threads.items()}

    def stats(self):
        return {name: thread.stats() for name, thread in self._threads.items()}

    def close(self):
        if self._closed:
            return
        self._closed = True
        for thread in self._threads.values():
            thread.stop()
        for thread in self._threads.values():
            # Thread.start may have failed before the native thread existed.
            if thread.is_alive():
                thread.join(timeout=1.0)
                if thread.is_alive():
                    logger_mp.warning(f"[RecordingImageClient] {thread.name}: subscriber did not stop within 1s")
        logger_mp.info("[RecordingImageClient] closed.")
