#!/usr/bin/env python3
"""Validate external Wi-Fi perception results and republish them on Thor local LCM.

This process never publishes an EE target or any robot command.
"""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass

import lcm

from seg_web_node.lcm_protocol import decode_calibration, decode_camera
from .protocol import decode_perception_result


@dataclass
class ReceiverStats:
    accepted: int = 0
    rejected: int = 0
    heartbeat: int = 0
    latest_result_age_ms: float | None = None


class ThorRemoteResultReceiver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.g1_bus = lcm.LCM(args.g1_lcm_url)
        self.wifi_bus = lcm.LCM(args.wifi_lcm_url)
        self.stats = ReceiverStats()
        self.running = True
        self.latest_rgb_timestamp_us: int | None = None
        self.latest_rgb_size: tuple[int, int] | None = None
        self.latest_calibration_id: int | None = None
        self.last_sequence = -1
        self.last_result_received = 0.0
        self.g1_bus.subscribe(args.rgb_channel, self._on_rgb)
        self.g1_bus.subscribe(args.calibration_channel, self._on_calibration)
        self.wifi_bus.subscribe(args.result_channel, self._on_result)
        self.wifi_bus.subscribe(args.heartbeat_channel, self._on_heartbeat)

    def _on_rgb(self, _channel: str, payload: bytes) -> None:
        try:
            frame = decode_camera(payload)
            self.latest_rgb_timestamp_us = frame.timestamp_us
            self.latest_rgb_size = (frame.width, frame.height)
        except Exception:
            pass

    def _on_calibration(self, _channel: str, payload: bytes) -> None:
        try:
            self.latest_calibration_id = decode_calibration(payload).calibration_id
        except Exception:
            pass

    def _on_result(self, _channel: str, payload: bytes) -> None:
        try:
            result = decode_perception_result(payload)
            sequence = int(result["sequence"])
            if sequence <= self.last_sequence:
                raise ValueError("non-monotonic sequence")
            if self.latest_rgb_size is None or self.latest_rgb_timestamp_us is None:
                raise ValueError("local G1 RGB stream is unavailable")
            if self.latest_calibration_id is None:
                raise ValueError("local G1 calibration is unavailable")
            if self.latest_rgb_size != (result["image_width"], result["image_height"]):
                raise ValueError("result image dimensions do not match current G1 stream")
            if result["calibration_id"] != self.latest_calibration_id:
                raise ValueError("result calibration_id does not match current G1 calibration")
            lag_us = self.latest_rgb_timestamp_us - int(result["source_rgb_timestamp_us"])
            if lag_us > self.args.max_source_lag_ms * 1000:
                raise ValueError(f"source frame is stale by {lag_us / 1000:.1f}ms")
            self.g1_bus.publish(self.args.local_result_channel, payload)
            self.last_sequence = sequence
            self.last_result_received = time.monotonic()
            self.stats.accepted += 1
        except Exception as error:
            self.stats.rejected += 1
            print(f"[remote-result] rejected: {error}", flush=True)

    def _on_heartbeat(self, _channel: str, payload: bytes) -> None:
        if len(payload) <= 64 * 1024:
            self.g1_bus.publish(self.args.local_heartbeat_channel, payload)
            self.stats.heartbeat += 1

    def run(self) -> None:
        last_report = 0.0
        while self.running:
            self.g1_bus.handle_timeout(20)
            self.wifi_bus.handle_timeout(20)
            now = time.monotonic()
            if self.last_result_received:
                self.stats.latest_result_age_ms = (now - self.last_result_received) * 1000.0
            if now - last_report >= 5.0:
                print(f"[remote-result] accepted={self.stats.accepted} rejected={self.stats.rejected} heartbeat={self.stats.heartbeat} age_ms={self.stats.latest_result_age_ms}", flush=True)
                last_report = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-lcm-url", required=True)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--rgb-channel", default="G1_CAMERA_FRAME")
    parser.add_argument("--calibration-channel", default="G1_CAMERA_CALIBRATION")
    parser.add_argument("--result-channel", default="PATH_REMOTE_PERCEPTION_RESULT")
    parser.add_argument("--heartbeat-channel", default="PATH_REMOTE_PERCEPTION_HEARTBEAT")
    parser.add_argument("--local-result-channel", default="PATH_REMOTE_PERCEPTION_RESULT_LOCAL")
    parser.add_argument("--local-heartbeat-channel", default="PATH_REMOTE_PERCEPTION_HEARTBEAT_LOCAL")
    parser.add_argument("--max-source-lag-ms", type=float, default=1500.0)
    return parser.parse_args()


def main() -> int:
    receiver = ThorRemoteResultReceiver(parse_args())
    signal.signal(signal.SIGINT, lambda *_: setattr(receiver, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(receiver, "running", False))
    receiver.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
