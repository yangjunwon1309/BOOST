#!/usr/bin/env python3
"""Forward timestamp-matched G1 RGB-D packets from Thor Ethernet to Wi-Fi LCM."""

from __future__ import annotations

import argparse
import collections
import signal
import time
from dataclasses import dataclass
from typing import Any

import lcm

from seg_web_node.lcm_protocol import decode_calibration, decode_camera, decode_depth, decode_robot_state
from seg_web_node.rgbd_sync import RgbdSynchronizer


@dataclass
class GatewayStats:
    rgb_received: int = 0
    depth_received: int = 0
    pairs_forwarded: int = 0
    pairs_rate_dropped: int = 0
    calibration_forwarded: int = 0
    robot_state_forwarded: int = 0
    decode_errors: int = 0


class ThorRemoteRgbdGateway:
    """Pair locally, then forward raw G1 packets over an independent Wi-Fi LCM URL."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.g1_bus = lcm.LCM(args.g1_lcm_url)
        self.wifi_bus = lcm.LCM(args.wifi_lcm_url)
        self.sync = RgbdSynchronizer(args.sync_tolerance_ms, capacity=30)
        self.stats = GatewayStats()
        self.running = True
        self.last_pair_forwarded = 0.0
        self.last_state_forwarded = 0.0
        self.last_calibration_forwarded = 0.0
        self.latest_calibration: bytes | None = None
        self.rgb_payloads: dict[int, bytes] = {}
        self.depth_payloads: dict[int, bytes] = {}
        self.g1_bus.subscribe(args.rgb_channel, self._on_rgb)
        self.g1_bus.subscribe(args.depth_channel, self._on_depth)
        self.g1_bus.subscribe(args.calibration_channel, self._on_calibration)
        self.g1_bus.subscribe(args.robot_state_channel, self._on_robot_state)

    @staticmethod
    def _remember(cache: dict[int, bytes], timestamp_us: int, payload: bytes) -> None:
        cache[timestamp_us] = payload
        if len(cache) > 64:
            for key in sorted(cache)[: len(cache) - 64]:
                del cache[key]

    def _on_rgb(self, _channel: str, payload: bytes) -> None:
        try:
            frame = decode_camera(payload)
            self.stats.rgb_received += 1
            self._remember(self.rgb_payloads, frame.timestamp_us, payload)
            pair = self.sync.add_rgb(frame)
            if pair is not None:
                self._forward_pair(pair)
        except Exception:
            self.stats.decode_errors += 1

    def _on_depth(self, _channel: str, payload: bytes) -> None:
        try:
            frame = decode_depth(payload, allow_unaligned=True)
            self.stats.depth_received += 1
            self._remember(self.depth_payloads, frame.timestamp_us, payload)
            pair = self.sync.add_depth(frame)
            if pair is not None:
                self._forward_pair(pair)
        except Exception:
            self.stats.decode_errors += 1

    def _forward_pair(self, pair: Any) -> None:
        now = time.monotonic()
        if now - self.last_pair_forwarded < 1.0 / max(0.1, self.args.forward_hz):
            self.stats.pairs_rate_dropped += 1
            return
        rgb_payload = self.rgb_payloads.pop(pair.rgb.timestamp_us, None)
        depth_payload = self.depth_payloads.pop(pair.depth.timestamp_us, None)
        if rgb_payload is None or depth_payload is None:
            self.stats.decode_errors += 1
            return
        self.wifi_bus.publish(self.args.wifi_rgb_channel, rgb_payload)
        self.wifi_bus.publish(self.args.wifi_depth_channel, depth_payload)
        self.last_pair_forwarded = now
        self.stats.pairs_forwarded += 1

    def _on_calibration(self, _channel: str, payload: bytes) -> None:
        try:
            decode_calibration(payload)
            self.latest_calibration = payload
            self.wifi_bus.publish(self.args.wifi_calibration_channel, payload)
            self.last_calibration_forwarded = time.monotonic()
            self.stats.calibration_forwarded += 1
        except Exception:
            self.stats.decode_errors += 1

    def _on_robot_state(self, _channel: str, payload: bytes) -> None:
        try:
            decode_robot_state(payload)
            now = time.monotonic()
            if now - self.last_state_forwarded < 1.0 / max(0.1, self.args.robot_state_hz):
                return
            self.wifi_bus.publish(self.args.wifi_robot_state_channel, payload)
            self.last_state_forwarded = now
            self.stats.robot_state_forwarded += 1
        except Exception:
            self.stats.decode_errors += 1

    def run(self) -> None:
        last_report = 0.0
        while self.running:
            self.g1_bus.handle_timeout(100)
            now = time.monotonic()
            if self.latest_calibration is not None and now - self.last_calibration_forwarded >= self.args.calibration_repeat_sec:
                self.wifi_bus.publish(self.args.wifi_calibration_channel, self.latest_calibration)
                self.last_calibration_forwarded = now
                self.stats.calibration_forwarded += 1
            if now - last_report >= 5.0:
                print(f"[remote-gateway] rgb={self.stats.rgb_received} depth={self.stats.depth_received} pairs={self.stats.pairs_forwarded} rate_drop={self.stats.pairs_rate_dropped} state={self.stats.robot_state_forwarded} errors={self.stats.decode_errors}", flush=True)
                last_report = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-lcm-url", required=True)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--rgb-channel", default="G1_CAMERA_FRAME")
    parser.add_argument("--depth-channel", default="G1_DEPTH_FRAME")
    parser.add_argument("--calibration-channel", default="G1_CAMERA_CALIBRATION")
    parser.add_argument("--robot-state-channel", default="G1_ROBOT_STATE")
    parser.add_argument("--wifi-rgb-channel", default="PATH_REMOTE_RGB_FRAME")
    parser.add_argument("--wifi-depth-channel", default="PATH_REMOTE_DEPTH_FRAME")
    parser.add_argument("--wifi-calibration-channel", default="PATH_REMOTE_CAMERA_CALIBRATION")
    parser.add_argument("--wifi-robot-state-channel", default="PATH_REMOTE_ROBOT_STATE")
    parser.add_argument("--forward-hz", type=float, default=5.0)
    parser.add_argument("--robot-state-hz", type=float, default=50.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--calibration-repeat-sec", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    gateway = ThorRemoteRgbdGateway(parse_args())
    signal.signal(signal.SIGINT, lambda *_: setattr(gateway, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(gateway, "running", False))
    gateway.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
