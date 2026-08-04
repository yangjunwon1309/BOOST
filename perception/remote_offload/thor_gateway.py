"""Thor-side RGB-D Wi-Fi gateway and remote-result validator."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import math
import os
import threading
import time
from typing import Any

import lcm
import msgpack
import numpy as np
import zmq

from seg_web_node.ee_target import make_goal_payload
from seg_web_node.grip_candidates import GripCandidate
from seg_web_node.lcm_protocol import (
    decode_calibration,
    decode_camera,
    decode_depth,
    decode_robot_state,
    encode_robot_state,
)
from seg_web_node.robot_geometry import g1_wrist_ee_position
from seg_web_node.tasc_belief import BeliefInputObject, HierarchicalTascBelief

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


DEFAULT_EE_TARGET_ENDPOINT = os.environ.get(
    "PATH_EE_TARGET_ENDPOINT",
    os.environ.get(
        "PATH_STATIC_TARGET_ENDPOINT",
        f"tcp://127.0.0.1:{os.environ.get('PATH_STATIC_TARGET_PORT', '5564')}",
    ),
)
DEFAULT_EE_TARGET_TOPIC = os.environ.get(
    "PATH_EE_TARGET_TOPIC",
    os.environ.get("PATH_STATIC_TARGET_TOPIC", "path_static_ee_target"),
)


def _topic_msgpack_unpack(raw: bytes, topic: str) -> dict[str, Any]:
    prefix = str(topic).encode("utf-8")
    if not raw.startswith(prefix):
        raise ValueError(f"message does not start with topic {topic!r}")
    payload = msgpack.unpackb(raw[len(prefix) :], raw=False, strict_map_key=False)
    if not isinstance(payload, dict):
        raise ValueError("topic payload is not a map")
    return payload


def _vector_from_payload(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    length: int,
    default: tuple[float, ...] | None = None,
) -> np.ndarray | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if array.shape[0] >= length and np.isfinite(array[:length]).all():
            return array[:length].copy()
    if default is None:
        return None
    return np.asarray(default, dtype=np.float64).reshape(length)


def _int_from_payload(payload: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _retimestamp_robot_state_packet(data: bytes, timestamp_us: int) -> bytes:
    state = decode_robot_state(data)
    return encode_robot_state(
        sequence=state.sequence,
        lowstate_sequence=state.lowstate_sequence,
        timestamp_us=int(timestamp_us),
        mode_machine=state.mode_machine,
        base_quat_wxyz=state.base_quat_wxyz,
        torso_quat_wxyz=state.torso_quat_wxyz,
        q=state.q,
        dq=state.dq,
        tau=None,
        has_torso_imu=state.has_torso_imu,
    )


def _load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        LOG.warning("PyYAML is unavailable; continuing without %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file must contain a map: {path}")
    return data


def _grip_candidate_from_dict(data: dict[str, Any]) -> GripCandidate | None:
    try:
        center = np.asarray(data.get("center_goal_m"), dtype=np.float64).reshape(3)
        approach = np.asarray(
            data.get("approach_goal", [0.0, 0.0, -1.0]), dtype=np.float64
        ).reshape(3)
        width = float(data.get("width_m", 0.0))
        geom_score = float(data.get("geom_score", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(center).all() or not np.isfinite(approach).all():
        return None
    return GripCandidate(
        center=center,
        approach=approach,
        width=width,
        geom_score=geom_score,
        source=str(data.get("source", "remote")),
    )


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
    robot_state: bytes | None


@dataclasses.dataclass(frozen=True)
class EeTargetSelection:
    obj: dict[str, Any]
    object_index: int
    position: np.ndarray
    position_field: str


class ZmqEeTargetPublisher:
    def __init__(self, *, endpoint: str, topic: str) -> None:
        self.endpoint = str(endpoint)
        self.topic = str(topic)
        self.socket = zmq.Context.instance().socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)

    def publish(self, payload: dict[str, Any]) -> None:
        self.socket.send(
            self.topic.encode("utf-8") + msgpack.packb(payload, use_bin_type=True)
        )

    def close(self) -> None:
        self.socket.close(0)


def _finite_vec3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(array).all():
        return None
    return array


def _side_from_link(target_link: str, fallback: str = "left") -> str:
    name = str(target_link).lower()
    if "right" in name:
        return "right"
    if "left" in name:
        return "left"
    return fallback


def _best_remote_object(
    payload: dict[str, Any],
    *,
    label_filter: str,
    min_score: float,
    position_field: str,
) -> EeTargetSelection | None:
    label_filter = str(label_filter).strip().lower()
    candidates: list[EeTargetSelection] = []
    for index, obj in enumerate(payload.get("objects", [])):
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("label", ""))
        if label_filter and label.lower() != label_filter:
            continue
        try:
            score = float(obj.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score) or score < float(min_score):
            continue
        fields: tuple[str, ...]
        if position_field == "contact":
            fields = ("target_contact_goal_m",)
        elif position_field == "center":
            fields = ("center_goal_m",)
        else:
            fields = ("target_contact_goal_m", "center_goal_m")
        for field in fields:
            position = _finite_vec3(obj.get(field))
            if position is not None:
                candidates.append(
                    EeTargetSelection(
                        obj=obj,
                        object_index=index,
                        position=position,
                        position_field=field,
                    )
                )
                break
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.obj.get("score", 0.0)))


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
        ee_target_enabled: bool,
        ee_target_endpoint: str,
        ee_target_topic: str,
        ee_target_link: str,
        ee_target_label: str,
        ee_target_min_score: float,
        ee_target_position_field: str,
        ee_target_duration_sec: float,
        ee_target_final_hold_sec: float,
        ee_target_quaternion_wxyz: tuple[float, float, float, float],
        sim_state_zmq_endpoint: str,
        sim_state_zmq_topic: str,
        belief_enabled: bool,
        belief_config: dict[str, Any],
        belief_ee_offset_m: float,
    ) -> None:
        if g1_lcm_url == wifi_lcm_url:
            raise ValueError("G1 and Wi-Fi LCM URLs must be distinct")
        self.g1_bus = lcm.LCM(g1_lcm_url)
        self.wifi_bus = lcm.LCM(wifi_lcm_url)
        self.forward_period = 1.0 / max(0.1, float(forward_hz))
        self.calibration_resend_sec = max(0.1, float(calibration_resend_sec))
        self.robot_state_period = 1.0 / max(0.1, float(robot_state_hz))
        self.result_max_age_ms = float(result_max_age_ms)
        self.ee_target_enabled = bool(ee_target_enabled)
        self.ee_target_link = str(ee_target_link)
        self.ee_target_side = _side_from_link(self.ee_target_link)
        self.ee_target_label = str(ee_target_label)
        self.ee_target_min_score = float(ee_target_min_score)
        self.ee_target_position_field = str(ee_target_position_field)
        self.ee_target_duration_sec = float(ee_target_duration_sec)
        self.ee_target_final_hold_sec = float(ee_target_final_hold_sec)
        self.ee_target_quaternion_wxyz = tuple(
            float(value) for value in ee_target_quaternion_wxyz
        )
        self.belief_enabled = bool(belief_enabled)
        self.tasc_belief = HierarchicalTascBelief(belief_config)
        self.belief_ee_offset_m = float(belief_ee_offset_m)
        self.belief_last_eef_position: np.ndarray | None = None
        self.belief_last_eef_timestamp_us: int | None = None
        self.ee_target_pub = (
            ZmqEeTargetPublisher(endpoint=ee_target_endpoint, topic=ee_target_topic)
            if self.ee_target_enabled
            else None
        )
        self.sim_state_zmq_endpoint = str(sim_state_zmq_endpoint)
        self.sim_state_zmq_topic = str(sim_state_zmq_topic)
        self.sim_state_socket = None
        if self.sim_state_zmq_endpoint:
            self.sim_state_socket = zmq.Context.instance().socket(zmq.SUB)
            self.sim_state_socket.setsockopt(zmq.LINGER, 0)
            self.sim_state_socket.setsockopt_string(zmq.SUBSCRIBE, self.sim_state_zmq_topic)
            self.sim_state_socket.connect(self.sim_state_zmq_endpoint)
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
            "ee_targets_published": 0,
            "ee_targets_skipped": 0,
            "last_ee_target": None,
            "last_ee_target_skip": None,
            "belief_enabled": bool(belief_enabled),
            "belief_targets": 0,
            "belief_fallback_targets": 0,
            "last_belief": self.tasc_belief.last_telemetry,
            "sim_robot_state": 0,
            "sim_robot_state_errors": 0,
            "last_sim_robot_state": None,
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
        if self.sim_state_socket is not None:
            self.threads.append(
                threading.Thread(
                    target=self._sim_state_loop,
                    name="sim-state-zmq",
                    daemon=True,
                )
            )
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2.0)
        if self.ee_target_pub is not None:
            self.ee_target_pub.close()
        if self.sim_state_socket is not None:
            self.sim_state_socket.close(0)

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

    def _on_sim_state_payload(self, payload: dict[str, Any]) -> None:
        q = _vector_from_payload(
            payload,
            ("body_q_measured", "body_q", "q_measured", "state_body_q_measured", "state_body_q"),
            29,
        )
        if q is None:
            raise ValueError("sim state payload has no 29-DoF measured q")
        dq = _vector_from_payload(
            payload,
            ("body_dq_measured", "body_dq", "dq_measured", "state_body_dq_measured"),
            29,
            default=tuple([0.0] * 29),
        )
        tau = _vector_from_payload(
            payload,
            ("tau_est", "body_tau", "tau", "state_tau_est"),
            29,
            default=tuple([0.0] * 29),
        )
        base_quat = _vector_from_payload(
            payload,
            ("base_quat_measured", "base_quat", "state_base_quat_measured", "state_base_quat"),
            4,
            default=(1.0, 0.0, 0.0, 0.0),
        )
        torso_quat = _vector_from_payload(
            payload,
            (
                "torso_quat_measured",
                "torso_quat",
                "state_torso_quat_measured",
                "state_torso_quat",
            ),
            4,
            default=(1.0, 0.0, 0.0, 0.0),
        )
        base_gyro = _vector_from_payload(
            payload,
            ("base_gyro_measured", "base_gyro", "state_base_gyro_measured"),
            3,
            default=(0.0, 0.0, 0.0),
        )
        base_accel = _vector_from_payload(
            payload,
            ("base_accel_measured", "base_accel", "state_base_accel_measured"),
            3,
            default=(0.0, 0.0, 0.0),
        )
        torso_gyro = _vector_from_payload(
            payload,
            ("torso_gyro_measured", "torso_gyro", "state_torso_gyro_measured"),
            3,
            default=(0.0, 0.0, 0.0),
        )
        torso_accel = _vector_from_payload(
            payload,
            ("torso_accel_measured", "torso_accel", "state_torso_accel_measured"),
            3,
            default=(0.0, 0.0, 0.0),
        )
        timestamp_us = _int_from_payload(
            payload,
            ("timestamp_us", "state_timestamp_us", "send_time_us", "time_us"),
            int(time.time_ns() // 1000),
        )
        sequence = _int_from_payload(
            payload,
            ("sequence", "state_sequence", "frame_index", "source_frame_index"),
            self.stats.get("sim_robot_state", 0),
        )
        lowstate_sequence = _int_from_payload(
            payload,
            ("lowstate_sequence", "lowstate_seq", "state_lowstate_sequence"),
            sequence,
        )
        mode_machine = _int_from_payload(payload, ("mode_machine", "mode"), 0)
        packet = encode_robot_state(
            sequence=sequence,
            lowstate_sequence=lowstate_sequence,
            timestamp_us=timestamp_us,
            mode_machine=mode_machine,
            base_quat_wxyz=base_quat,
            base_gyro=base_gyro,
            base_accel=base_accel,
            torso_quat_wxyz=torso_quat,
            torso_gyro=torso_gyro,
            torso_accel=torso_accel,
            q=q,
            dq=dq,
            tau=tau,
            has_torso_imu=True,
        )
        with self.lock:
            self.latest_robot_state = (timestamp_us, packet)
            self.stats["sim_robot_state"] += 1
            self.stats["last_sim_robot_state"] = {
                "timestamp_us": timestamp_us,
                "sequence": sequence,
                "q0": float(q[0]),
            }

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
            if not errors:
                self._publish_ee_target_from_result(payload, record)
        except Exception as error:
            self._record_error(error)

    def _belief_inputs_from_result(
        self,
        payload: dict[str, Any],
    ) -> list[BeliefInputObject]:
        inputs: list[BeliefInputObject] = []
        for index, obj in enumerate(payload.get("objects", [])):
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label", "object"))
            if self.ee_target_label and label.lower() != self.ee_target_label.strip().lower():
                continue
            try:
                score = float(obj.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or score < self.ee_target_min_score:
                continue
            center = _finite_vec3(obj.get("center_goal_m"))
            if center is None:
                continue
            candidates: list[GripCandidate] = []
            raw_candidates = obj.get("contact_candidates")
            if isinstance(raw_candidates, list):
                for raw_candidate in raw_candidates:
                    if not isinstance(raw_candidate, dict):
                        continue
                    candidate = _grip_candidate_from_dict(raw_candidate)
                    if candidate is not None:
                        candidates.append(candidate)
            object_id = str(obj.get("track_id") or f"{obj.get('label', 'object')}:{index}")
            inputs.append(
                BeliefInputObject(
                    object_id=object_id,
                    label=label,
                    score=score,
                    center_goal_m=center,
                    points_goal_m=None,
                    grip_candidates=candidates or None,
                )
            )
        return inputs

    def _belief_state_features(
        self,
        record: ForwardedPairRecord | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if record is None or record.robot_state is None:
            return None, None
        try:
            state = decode_robot_state(record.robot_state)
            position = g1_wrist_ee_position(
                state.q,
                self.ee_target_link,
                self.belief_ee_offset_m,
            )
            velocity = None
            if (
                self.belief_last_eef_position is not None
                and self.belief_last_eef_timestamp_us is not None
            ):
                dt = (state.timestamp_us - self.belief_last_eef_timestamp_us) / 1.0e6
                if 1.0e-4 <= dt <= 0.5:
                    velocity = (position - self.belief_last_eef_position) / dt
            self.belief_last_eef_position = position.copy()
            self.belief_last_eef_timestamp_us = int(state.timestamp_us)
            return position, velocity
        except Exception as error:
            with self.lock:
                self.stats["last_belief_state_error"] = str(error)
            return None, None

    def _belief_selection_from_result(
        self,
        payload: dict[str, Any],
        record: ForwardedPairRecord | None,
    ) -> tuple[np.ndarray, dict[str, Any], str] | None:
        if not self.belief_enabled or not self.tasc_belief.config.enabled:
            return None
        inputs = self._belief_inputs_from_result(payload)
        if not inputs:
            with self.lock:
                self.stats["last_belief"] = {"enabled": True, "objects": [], "target": None}
            return None
        eef_position, eef_velocity = self._belief_state_features(record)
        controller_xyzw = np.asarray(
            [
                self.ee_target_quaternion_wxyz[1],
                self.ee_target_quaternion_wxyz[2],
                self.ee_target_quaternion_wxyz[3],
                self.ee_target_quaternion_wxyz[0],
            ],
            dtype=np.float64,
        )
        telemetry = self.tasc_belief.update(
            inputs,
            eef_position_m=eef_position,
            user_translation_mps=eef_velocity,
            ray_origin_m=None,
            ray_direction=None,
            controller_orientation_xyzw=controller_xyzw,
        )
        if self.tasc_belief.should_log():
            LOG.info(self.tasc_belief.format_log_line())
            self.tasc_belief.mark_logged()
        with self.lock:
            self.stats["last_belief"] = telemetry
        target = telemetry.get("target")
        if not isinstance(target, dict):
            return None
        position = _finite_vec3(target.get("position_goal_m"))
        if position is None:
            return None
        object_id = str(target.get("object_id", ""))
        selected = next(
            (
                obj for obj in payload.get("objects", [])
                if str(obj.get("track_id", "")) == object_id
            ),
            {},
        )
        label = str(selected.get("label") or object_id or "belief_target")
        metadata = {
            "remote_sequence": int(payload.get("sequence", -1)),
            "frame_timestamp_us": int(payload.get("source_rgb_timestamp_us", 0) or 0),
            "source_rgb_timestamp_us": payload.get("source_rgb_timestamp_us"),
            "source_depth_timestamp_us": payload.get("source_depth_timestamp_us"),
            "remote_published_timestamp_us": payload.get("published_timestamp_us"),
            "gateway_published_timestamp_us": int(time.time_ns() // 1000),
            "backend": payload.get("backend"),
            "inference_ms": payload.get("inference_ms"),
            "object_index": int(selected.get("index", -1)),
            "object_id": object_id,
            "track_id": selected.get("track_id", object_id),
            "label": label,
            "score": float(selected.get("score", 0.0)),
            "center_camera_m": selected.get("center_camera_m"),
            "center_goal_m": selected.get("center_goal_m"),
            "target_contact_goal_m": position.tolist(),
            "tasc_object_id": object_id,
            "tasc_selected_grip_index": target.get("grip_index"),
            "tasc_object_probability": target.get("object_confidence"),
            "tasc_joint_confidence": target.get("joint_confidence"),
            "streaming_goal": True,
            "target_valid": True,
            "belief_selected": True,
        }
        output = {
            "position": position,
            "metadata": metadata,
            "source": "remote_tasc_belief",
            "goal_uid": f"remote_belief:{self.ee_target_link}:{object_id or label}",
            "confidence": float(target.get("object_confidence", 1.0)),
            "object_confidence": float(target.get("object_confidence", 1.0)),
            "joint_confidence": float(target.get("joint_confidence", 1.0)),
            "object_id": object_id,
            "label": label,
            "grip_index": int(target.get("grip_index", -1)),
        }
        return position, output, "belief"

    def _publish_ee_target_from_result(
        self,
        payload: dict[str, Any],
        record: ForwardedPairRecord | None,
    ) -> None:
        if self.ee_target_pub is None:
            return
        if not bool(payload.get("valid", False)):
            with self.lock:
                self.stats["ee_targets_skipped"] += 1
                self.stats["last_ee_target_skip"] = {
                    "sequence": payload.get("sequence"),
                    "reason": "remote_result_invalid",
                    "error": payload.get("error"),
                }
            return
        belief_selection = self._belief_selection_from_result(payload, record)
        if belief_selection is not None:
            position, selected_output, selected_position_field = belief_selection
            payload_out = make_goal_payload(
                position,
                side=self.ee_target_side,
                target_link=self.ee_target_link,
                quaternion=self.ee_target_quaternion_wxyz,
                duration_sec=self.ee_target_duration_sec,
                final_hold_sec=self.ee_target_final_hold_sec,
                workspace_info={
                    "workspace_mode": "remote_gateway_belief",
                    "workspace_reason": "remote_tasc_belief_result",
                    "workspace_clamped": False,
                    "requested_position": position.tolist(),
                },
                metadata=selected_output["metadata"],
            )
            payload_out.update(
                {
                    "goal_uid": selected_output["goal_uid"],
                    "force_replan": False,
                    "streaming_goal": True,
                    "target_valid": True,
                    "source": selected_output["source"],
                    "confidence": selected_output["confidence"],
                    "object_confidence": selected_output["object_confidence"],
                    "joint_confidence": selected_output["joint_confidence"],
                    "object_id": selected_output["object_id"],
                    "label": selected_output["label"],
                    "grip_index": selected_output["grip_index"],
                }
            )
            self.ee_target_pub.publish(payload_out)
            with self.lock:
                self.stats["ee_targets_published"] += 1
                self.stats["belief_targets"] += 1
                self.stats["last_ee_target"] = {
                    "sequence": payload.get("sequence"),
                    "label": selected_output["label"],
                    "score": selected_output["confidence"],
                    "joint_confidence": selected_output["joint_confidence"],
                    "target_link": self.ee_target_link,
                    "position": np.round(position, 4).tolist(),
                    "position_field": selected_position_field,
                }
            return
        selection = _best_remote_object(
            payload,
            label_filter=self.ee_target_label,
            min_score=self.ee_target_min_score,
            position_field=self.ee_target_position_field,
        )
        if selection is None:
            with self.lock:
                self.stats["ee_targets_skipped"] += 1
                self.stats["last_ee_target_skip"] = {
                    "sequence": payload.get("sequence"),
                    "reason": "no_object_with_goal_position",
                    "label_filter": self.ee_target_label,
                    "min_score": self.ee_target_min_score,
                    "position_field": self.ee_target_position_field,
                }
            return
        obj = selection.obj
        score = float(obj.get("score", 0.0))
        label = str(obj.get("label", "object"))
        object_id = str(obj.get("track_id") or f"{label}:{selection.object_index}")
        metadata = {
            "remote_sequence": int(payload.get("sequence", -1)),
            "frame_timestamp_us": int(payload.get("source_rgb_timestamp_us", 0) or 0),
            "source_rgb_timestamp_us": payload.get("source_rgb_timestamp_us"),
            "source_depth_timestamp_us": payload.get("source_depth_timestamp_us"),
            "remote_published_timestamp_us": payload.get("published_timestamp_us"),
            "gateway_published_timestamp_us": int(time.time_ns() // 1000),
            "backend": payload.get("backend"),
            "inference_ms": payload.get("inference_ms"),
            "object_index": int(selection.object_index),
            "object_id": object_id,
            "track_id": obj.get("track_id"),
            "label": label,
            "score": score,
            "confidence": score,
            "center_camera_m": obj.get("center_camera_m"),
            "center_goal_m": obj.get("center_goal_m"),
            "target_contact_goal_m": obj.get("target_contact_goal_m"),
            "selected_position_field": selection.position_field,
            "valid_depth_points": obj.get("valid_depth_points"),
            "streaming_goal": True,
        }
        payload_out = make_goal_payload(
            selection.position,
            side=self.ee_target_side,
            target_link=self.ee_target_link,
            quaternion=self.ee_target_quaternion_wxyz,
            duration_sec=self.ee_target_duration_sec,
            final_hold_sec=self.ee_target_final_hold_sec,
            workspace_info={
                "workspace_mode": "remote_gateway",
                "workspace_reason": "validated_remote_perception_result",
                "workspace_clamped": False,
                "requested_position": selection.position.tolist(),
            },
            metadata=metadata,
        )
        payload_out.update(
            {
                "goal_uid": f"remote:{object_id}",
                "force_replan": False,
                "streaming_goal": True,
                "source": (
                    "remote_yolo11_contact"
                    if selection.position_field == "target_contact_goal_m"
                    else "remote_yolo11_center"
                ),
                "confidence": score,
                "object_confidence": score,
                "object_id": object_id,
                "label": label,
                "object_index": int(selection.object_index),
            }
        )
        self.ee_target_pub.publish(payload_out)
        with self.lock:
            self.stats["ee_targets_published"] += 1
            if self.belief_enabled:
                self.stats["belief_fallback_targets"] += 1
            self.stats["last_ee_target"] = {
                "sequence": payload.get("sequence"),
                "label": label,
                "score": score,
                "target_link": self.ee_target_link,
                "position": np.round(selection.position, 4).tolist(),
                "position_field": selection.position_field,
            }

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

    def _sim_state_loop(self) -> None:
        assert self.sim_state_socket is not None
        poller = zmq.Poller()
        poller.register(self.sim_state_socket, zmq.POLLIN)
        while self.running:
            try:
                events = dict(poller.poll(100))
                if self.sim_state_socket not in events:
                    continue
                raw = self.sim_state_socket.recv()
                payload = _topic_msgpack_unpack(raw, self.sim_state_zmq_topic)
                self._on_sim_state_payload(payload)
            except Exception as error:
                with self.lock:
                    self.stats["sim_robot_state_errors"] += 1
                    self.stats["last_sim_robot_state_error"] = str(error)
                time.sleep(0.02)

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
                    and self.sim_state_socket is None
                    and now - last_robot_state >= self.robot_state_period
                ):
                    robot_state = self.latest_robot_state[1]
                    last_robot_state = now
                if (
                    pair is not None
                    and self.sim_state_socket is not None
                    and self.latest_robot_state is not None
                ):
                    robot_state = _retimestamp_robot_state_packet(
                        self.latest_robot_state[1],
                        pair.rgb.timestamp_us,
                    )
                    last_robot_state = now
                    self.stats["sim_robot_state_timestamp_aligned"] = (
                        int(self.stats.get("sim_robot_state_timestamp_aligned", 0)) + 1
                    )
                    self.stats["last_sim_robot_state_aligned"] = {
                        "state_timestamp_us": pair.rgb.timestamp_us,
                        "rgb_timestamp_us": pair.rgb.timestamp_us,
                    }
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
                        ForwardedPairRecord(
                            pair=pair,
                            calibration_id=calibration_id,
                            robot_state=robot_state,
                        )
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
    parser.add_argument(
        "--config",
        default=os.environ.get("PATH_REMOTE_GATEWAY_CONFIG", "config/pipeline.yaml"),
        help="Pipeline config used for tasc_belief settings.",
    )
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
    parser.add_argument(
        "--ee-target-publishing",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PATH_REMOTE_EE_TARGET_ENABLED", "1").lower()
        not in {"0", "false", "no", "off"},
        help=(
            "Publish validated remote perception results as PATH static EE targets "
            "(default: enabled; use --no-ee-target-publishing to disable)."
        ),
    )
    parser.add_argument(
        "--ee-target-endpoint",
        default=DEFAULT_EE_TARGET_ENDPOINT,
        help="ZMQ endpoint consumed by the PATH state machine EE_TARGET input.",
    )
    parser.add_argument(
        "--ee-target-topic",
        default=DEFAULT_EE_TARGET_TOPIC,
        help="ZMQ topic consumed by the PATH state machine EE_TARGET input.",
    )
    parser.add_argument(
        "--ee-target-link",
        default=os.environ.get("PATH_EE_TARGET_LINK", "left_wrist_yaw_link"),
    )
    parser.add_argument(
        "--ee-target-label",
        default=os.environ.get("PATH_EE_TARGET_LABEL", ""),
        help="Optional object label filter, e.g. bottle. Empty means highest-score object.",
    )
    parser.add_argument(
        "--ee-target-min-score",
        type=float,
        default=float(os.environ.get("PATH_EE_TARGET_MIN_SCORE", "0.0")),
    )
    parser.add_argument(
        "--ee-target-position-field",
        choices=("auto", "contact", "center"),
        default=os.environ.get("PATH_EE_TARGET_POSITION_FIELD", "auto"),
        help="auto prefers target_contact_goal_m and falls back to center_goal_m.",
    )
    parser.add_argument(
        "--ee-target-duration-sec",
        type=float,
        default=float(os.environ.get("PATH_EE_TARGET_DURATION_SEC", "2.0")),
    )
    parser.add_argument(
        "--ee-target-final-hold-sec",
        type=float,
        default=float(os.environ.get("PATH_EE_TARGET_FINAL_HOLD_SEC", "1.0")),
    )
    parser.add_argument(
        "--ee-target-quaternion-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        default=tuple(
            float(value)
            for value in os.environ.get(
                "PATH_EE_TARGET_QUATERNION_WXYZ", "1.0 0.0 0.0 0.0"
            ).split()
        ),
    )
    parser.add_argument(
        "--sim-state-zmq-endpoint",
        default=os.environ.get("PATH_SIM_STATE_ZMQ_ENDPOINT", ""),
        help=(
            "Optional MuJoCo/SONIC g1_debug ZMQ endpoint to repack as "
            "REMOTE_ROBOT_STATE for remote perception sim tests."
        ),
    )
    parser.add_argument(
        "--sim-state-zmq-topic",
        default=os.environ.get("PATH_SIM_STATE_ZMQ_TOPIC", "g1_debug"),
    )
    parser.add_argument(
        "--belief-inference",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PATH_REMOTE_BELIEF_ENABLED", "1").lower()
        not in {"0", "false", "no", "off"},
        help=(
            "Run the same hierarchical TASC belief selector used by the Thor web "
            "server before publishing EE targets."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _load_config(args.config)
    target_config = dict(config.get("ee_target", {}))
    belief_config = dict(config.get("tasc_belief", {}))
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
        ee_target_enabled=args.ee_target_publishing,
        ee_target_endpoint=args.ee_target_endpoint,
        ee_target_topic=args.ee_target_topic,
        ee_target_link=args.ee_target_link,
        ee_target_label=args.ee_target_label,
        ee_target_min_score=args.ee_target_min_score,
        ee_target_position_field=args.ee_target_position_field,
        ee_target_duration_sec=args.ee_target_duration_sec,
        ee_target_final_hold_sec=args.ee_target_final_hold_sec,
        ee_target_quaternion_wxyz=tuple(args.ee_target_quaternion_wxyz),
        sim_state_zmq_endpoint=args.sim_state_zmq_endpoint,
        sim_state_zmq_topic=args.sim_state_zmq_topic,
        belief_enabled=args.belief_inference,
        belief_config=belief_config,
        belief_ee_offset_m=float(belief_config.get("ee_offset_m", target_config.get("ee_offset_m", 0.05))),
    )
    gateway.start()
    if args.ee_target_publishing:
        LOG.info(
            "Thor remote RGB-D gateway started; publishing EE targets to %s/%s link=%s",
            args.ee_target_endpoint,
            args.ee_target_topic,
            args.ee_target_link,
        )
    else:
        LOG.info("Thor remote RGB-D gateway started; EE target publishing disabled")
    LOG.info(
        "Remote TASC belief inference %s using config=%s",
        "enabled" if args.belief_inference else "disabled",
        args.config,
    )
    if args.sim_state_zmq_endpoint:
        LOG.info(
            "MuJoCo sim robot-state forwarding enabled from %s/%s to %s",
            args.sim_state_zmq_endpoint,
            args.sim_state_zmq_topic,
            REMOTE_ROBOT_STATE_CHANNEL,
        )
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
