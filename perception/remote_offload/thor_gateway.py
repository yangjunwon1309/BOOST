"""Thor-side RGB-D Wi-Fi gateway and remote-result validator."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import threading
import time
from typing import Any

import lcm

from seg_web_node.lcm_protocol import (
    decode_calibration,
    decode_camera,
    decode_depth,
    decode_robot_state,
)

from .protocol import (
    LOCAL_CALIBRATION_CHANNEL,
    LOCAL_DEPTH_CHANNEL,
    LOCAL_RGB_CHANNEL,
    LOCAL_ROBOT_STATE_CHANNEL,
    REMOTE_CALIBRATION_CHANNEL,
    REMOTE_DEPTH_CHANNEL,
    REMOTE_HEARTBEAT_CHANNEL,
    REMOTE_OVERLAY_CHANNEL,
    REMOTE_RESULT_CHANNEL,
    REMOTE_RGB_CHANNEL,
    REMOTE_ROBOT_STATE_CHANNEL,
    ValidationContext,
    unpack_heartbeat,
    unpack_result,
    validate_result,
)


LOG = logging.getLogger("remote_offload.thor_gateway")


@dataclasses.dataclass(frozen=True)
class RawFrame:
    timestamp_us: int
    width: int
    height: int
    data: bytes


@dataclasses.dataclass(frozen=True)
class RawPair:
    rgb: RawFrame
    depth: RawFrame
    delta_us: int


@dataclasses.dataclass(frozen=True)
class ForwardedPairRecord:
    pair: RawPair
    calibration_id: int | None


class RawRgbdSynchronizer:
    def __init__(self, tolerance_ms: float = 50.0, capacity: int = 12) -> None:
        self.tolerance_us = int(round(float(tolerance_ms) * 1000.0))
        self.rgb: collections.deque[RawFrame] = collections.deque(maxlen=capacity)
        self.depth: collections.deque[RawFrame] = collections.deque(maxlen=capacity)
        self.dropped_rgb = 0
        self.dropped_depth = 0
        self.matched = 0

    def add_rgb(self, frame: RawFrame) -> RawPair | None:
        if len(self.rgb) == self.rgb.maxlen:
            self.dropped_rgb += 1
        self.rgb.append(frame)
        return self._match()

    def add_depth(self, frame: RawFrame) -> RawPair | None:
        if len(self.depth) == self.depth.maxlen:
            self.dropped_depth += 1
        self.depth.append(frame)
        return self._match()

    def _match(self) -> RawPair | None:
        if not self.rgb or not self.depth:
            return None
        best: tuple[int, int, int, RawFrame, RawFrame] | None = None
        for rgb_index, rgb in enumerate(self.rgb):
            for depth_index, depth in enumerate(self.depth):
                delta = abs(rgb.timestamp_us - depth.timestamp_us)
                candidate = (delta, rgb_index, depth_index, rgb, depth)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        assert best is not None
        delta, rgb_index, depth_index, rgb, depth = best
        if delta > self.tolerance_us:
            return None
        del self.rgb[rgb_index]
        del self.depth[depth_index]
        self.matched += 1
        return RawPair(rgb, depth, delta)


class ThorRemoteGateway:
    def __init__(
        self,
        *,
        g1_lcm_url: str,
        wifi_lcm_url: str,
        forward_hz: float,
        result_max_age_ms: float,
        sync_tolerance_ms: float,
        calibration_resend_sec: float,
        robot_state_hz: float,
        rgb_channel: str,
        depth_channel: str,
        calibration_channel: str,
        robot_state_channel: str,
    ) -> None:
        if g1_lcm_url == wifi_lcm_url:
            raise ValueError("G1 and Wi-Fi LCM URLs must be distinct")
        self.g1_bus = lcm.LCM(g1_lcm_url)
        self.wifi_bus = lcm.LCM(wifi_lcm_url)
        self.forward_period = 1.0 / max(0.1, float(forward_hz))
        self.calibration_resend_sec = max(0.1, float(calibration_resend_sec))
        self.robot_state_period = 1.0 / max(0.1, float(robot_state_hz))
        self.result_max_age_ms = float(result_max_age_ms)
        self.sync = RawRgbdSynchronizer(sync_tolerance_ms)
        self.lock = threading.Lock()
        self.running = False
        self.pending_pair: RawPair | None = None
        self.latest_pair: RawPair | None = None
        self.forwarded_pairs: collections.deque[ForwardedPairRecord] = (
            collections.deque(maxlen=60)
        )
        self.latest_calibration: tuple[int | None, bytes] | None = None
        self.latest_robot_state: tuple[int | None, bytes] | None = None
        self.last_result_sequence: int | None = None
        self.stats: dict[str, Any] = {
            "local_rgb": 0,
            "local_depth": 0,
            "local_calibration": 0,
            "local_robot_state": 0,
            "forwarded_pairs": 0,
            "forwarded_calibration": 0,
            "forwarded_robot_state": 0,
            "valid_results": 0,
            "invalid_results": 0,
            "heartbeats": 0,
            "overlays": 0,
            "decode_errors": 0,
            "last_error": None,
            "last_valid_result": None,
            "last_invalid_result": None,
            "last_heartbeat": None,
        }
        self.threads: list[threading.Thread] = []
        self.g1_bus.subscribe(rgb_channel, self._on_local_rgb)
        self.g1_bus.subscribe(depth_channel, self._on_local_depth)
        self.g1_bus.subscribe(calibration_channel, self._on_local_calibration)
        self.g1_bus.subscribe(robot_state_channel, self._on_local_robot_state)
        self.wifi_bus.subscribe(REMOTE_RESULT_CHANNEL, self._on_remote_result)
        self.wifi_bus.subscribe(REMOTE_HEARTBEAT_CHANNEL, self._on_remote_heartbeat)
        self.wifi_bus.subscribe(REMOTE_OVERLAY_CHANNEL, self._on_remote_overlay)

    def start(self) -> None:
        self.running = True
        self.threads = [
            threading.Thread(target=self._g1_loop, name="g1-lcm", daemon=True),
            threading.Thread(target=self._wifi_loop, name="wifi-lcm", daemon=True),
            threading.Thread(target=self._forward_loop, name="wifi-forward", daemon=True),
            threading.Thread(target=self._log_loop, name="gateway-log", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2.0)

    def _record_error(self, error: Exception | str) -> None:
        with self.lock:
            self.stats["decode_errors"] += 1
            self.stats["last_error"] = str(error)

    def _submit_pair(self, pair: RawPair | None) -> None:
        if pair is None:
            return
        with self.lock:
            if self.pending_pair is not None:
                self.stats["dropped_before_forward"] = (
                    int(self.stats.get("dropped_before_forward", 0)) + 1
                )
            self.pending_pair = pair
            self.latest_pair = pair

    def _on_local_rgb(self, _channel: str, data: bytes) -> None:
        try:
            frame = decode_camera(data)
            raw = RawFrame(frame.timestamp_us, frame.width, frame.height, bytes(data))
            with self.lock:
                self.stats["local_rgb"] += 1
            self._submit_pair(self.sync.add_rgb(raw))
        except Exception as error:
            self._record_error(error)

    def _on_local_depth(self, _channel: str, data: bytes) -> None:
        try:
            frame = decode_depth(data, allow_unaligned=True)
            raw = RawFrame(frame.timestamp_us, frame.width, frame.height, bytes(data))
            with self.lock:
                self.stats["local_depth"] += 1
            self._submit_pair(self.sync.add_depth(raw))
        except Exception as error:
            self._record_error(error)

    def _on_local_calibration(self, _channel: str, data: bytes) -> None:
        try:
            calibration = decode_calibration(data)
            with self.lock:
                self.latest_calibration = (calibration.calibration_id, bytes(data))
                self.stats["local_calibration"] += 1
        except Exception as error:
            self._record_error(error)

    def _on_local_robot_state(self, _channel: str, data: bytes) -> None:
        try:
            state = decode_robot_state(data)
            with self.lock:
                self.latest_robot_state = (state.timestamp_us, bytes(data))
                self.stats["local_robot_state"] += 1
        except Exception as error:
            self._record_error(error)

    def _on_remote_result(self, _channel: str, data: bytes) -> None:
        try:
            payload = unpack_result(data)
            with self.lock:
                source_rgb = payload.get("source_rgb_timestamp_us")
                source_depth = payload.get("source_depth_timestamp_us")
                record = next(
                    (
                        item
                        for item in self.forwarded_pairs
                        if item.pair.rgb.timestamp_us == source_rgb
                        and item.pair.depth.timestamp_us == source_depth
                    ),
                    None,
                )
                previous_sequence = self.last_result_sequence
            result_errors = []
            if record is None:
                result_errors.append("source RGB-D pair was not forwarded recently")
            context = ValidationContext(
                result_max_age_ms=self.result_max_age_ms,
                expected_rgb_timestamp_us=record.pair.rgb.timestamp_us if record else None,
                expected_depth_timestamp_us=record.pair.depth.timestamp_us if record else None,
                expected_calibration_id=record.calibration_id if record else None,
                expected_image_width=record.pair.rgb.width if record else None,
                expected_image_height=record.pair.rgb.height if record else None,
                previous_sequence=previous_sequence,
            )
            errors = result_errors + validate_result(payload, context)
            with self.lock:
                if errors:
                    self.stats["invalid_results"] += 1
                    self.stats["last_invalid_result"] = {
                        "sequence": payload.get("sequence"),
                        "errors": errors,
                    }
                    LOG.warning(
                        "rejected remote result seq=%s: %s",
                        payload.get("sequence"),
                        "; ".join(errors),
                    )
                else:
                    self.last_result_sequence = int(payload["sequence"])
                    self.stats["valid_results"] += 1
                    self.stats["last_valid_result"] = {
                        "sequence": payload["sequence"],
                        "objects": len(payload.get("objects", [])),
                        "source_rgb_timestamp_us": payload.get("source_rgb_timestamp_us"),
                        "valid": payload.get("valid"),
                        "error": payload.get("error"),
                    }
                    LOG.info(
                        "valid remote result seq=%s objects=%s valid=%s",
                        payload["sequence"],
                        len(payload.get("objects", [])),
                        payload.get("valid"),
                    )
        except Exception as error:
            self._record_error(error)

    def _on_remote_heartbeat(self, _channel: str, data: bytes) -> None:
        try:
            heartbeat = unpack_heartbeat(data)
            with self.lock:
                self.stats["heartbeats"] += 1
                self.stats["last_heartbeat"] = heartbeat
        except Exception as error:
            self._record_error(error)

    def _on_remote_overlay(self, _channel: str, data: bytes) -> None:
        with self.lock:
            self.stats["overlays"] += 1
            self.stats["last_overlay_bytes"] = len(data)

    def _g1_loop(self) -> None:
        while self.running:
            try:
                self.g1_bus.handle_timeout(100)
            except Exception as error:
                self._record_error(error)
                time.sleep(0.1)

    def _wifi_loop(self) -> None:
        while self.running:
            try:
                self.wifi_bus.handle_timeout(100)
            except Exception as error:
                self._record_error(error)
                time.sleep(0.1)

    def _forward_loop(self) -> None:
        last_calibration = 0.0
        last_robot_state = 0.0
        while self.running:
            started = time.monotonic()
            pair = None
            calibration = None
            robot_state = None
            with self.lock:
                pair = self.pending_pair
                self.pending_pair = None
                now = time.monotonic()
                if (
                    self.latest_calibration is not None
                    and now - last_calibration >= self.calibration_resend_sec
                ):
                    calibration = self.latest_calibration[1]
                    last_calibration = now
                if (
                    self.latest_robot_state is not None
                    and now - last_robot_state >= self.robot_state_period
                ):
                    robot_state = self.latest_robot_state[1]
                    last_robot_state = now
            if calibration is not None:
                self.wifi_bus.publish(REMOTE_CALIBRATION_CHANNEL, calibration)
                with self.lock:
                    self.stats["forwarded_calibration"] += 1
            if robot_state is not None:
                self.wifi_bus.publish(REMOTE_ROBOT_STATE_CHANNEL, robot_state)
                with self.lock:
                    self.stats["forwarded_robot_state"] += 1
            if pair is not None:
                self.wifi_bus.publish(REMOTE_RGB_CHANNEL, pair.rgb.data)
                self.wifi_bus.publish(REMOTE_DEPTH_CHANNEL, pair.depth.data)
                with self.lock:
                    calibration_id = (
                        self.latest_calibration[0]
                        if self.latest_calibration is not None
                        else None
                    )
                    self.forwarded_pairs.append(
                        ForwardedPairRecord(pair=pair, calibration_id=calibration_id)
                    )
                    self.stats["forwarded_pairs"] += 1
                    self.stats["last_forwarded"] = {
                        "rgb_timestamp_us": pair.rgb.timestamp_us,
                        "depth_timestamp_us": pair.depth.timestamp_us,
                        "delta_ms": pair.delta_us / 1000.0,
                        "width": pair.rgb.width,
                        "height": pair.rgb.height,
                    }
            time.sleep(max(0.0, self.forward_period - (time.monotonic() - started)))

    def _log_loop(self) -> None:
        while self.running:
            time.sleep(1.0)
            with self.lock:
                snapshot = dict(self.stats)
                snapshot.update(
                    sync_matched=self.sync.matched,
                    sync_dropped_rgb=self.sync.dropped_rgb,
                    sync_dropped_depth=self.sync.dropped_depth,
                )
            LOG.info("gateway status %s", snapshot)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-lcm-url", required=True)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--forward-hz", type=float, default=5.0)
    parser.add_argument("--result-max-age-ms", type=float, default=800.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--calibration-resend-sec", type=float, default=2.0)
    parser.add_argument("--robot-state-hz", type=float, default=20.0)
    parser.add_argument("--rgb-channel", default=LOCAL_RGB_CHANNEL)
    parser.add_argument("--depth-channel", default=LOCAL_DEPTH_CHANNEL)
    parser.add_argument("--calibration-channel", default=LOCAL_CALIBRATION_CHANNEL)
    parser.add_argument("--robot-state-channel", default=LOCAL_ROBOT_STATE_CHANNEL)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    gateway = ThorRemoteGateway(
        g1_lcm_url=args.g1_lcm_url,
        wifi_lcm_url=args.wifi_lcm_url,
        forward_hz=args.forward_hz,
        result_max_age_ms=args.result_max_age_ms,
        sync_tolerance_ms=args.sync_tolerance_ms,
        calibration_resend_sec=args.calibration_resend_sec,
        robot_state_hz=args.robot_state_hz,
        rgb_channel=args.rgb_channel,
        depth_channel=args.depth_channel,
        calibration_channel=args.calibration_channel,
        robot_state_channel=args.robot_state_channel,
    )
    gateway.start()
    LOG.info("Thor remote RGB-D gateway started; no EE target/control publishing is enabled")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOG.info("stopping gateway")
    finally:
        gateway.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
