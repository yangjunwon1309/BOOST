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
from seg_web_node.grip_candidates import GripCandidate, SurfacePrimitive
from seg_web_node.lcm_protocol import (
    decode_calibration,
    decode_camera,
    decode_depth,
    decode_robot_state,
)
from seg_web_node.robot_geometry import (
    angular_velocity_from_rotations,
    g1_grasp_center_position,
    g1_hand_base_rotation,
    grasp_center_from_wrist,
    gravity_up_in_pelvis,
    hand_frame_profile,
    hand_rotation_from_wrist_rotation,
)
from seg_web_node.tasc_belief import (
    BeliefInputObject,
    HandInput,
    ParallelTascBelief,
)

from .belief_csv import RemoteBeliefCsvRecorder
from .protocol import (
    LOCAL_CALIBRATION_CHANNEL,
    LOCAL_DEPTH_CHANNEL,
    LOCAL_RGB_CHANNEL,
    LOCAL_ROBOT_STATE_CHANNEL,
    REMOTE_BELIEF_TELEMETRY_CHANNEL,
    REMOTE_CALIBRATION_CHANNEL,
    REMOTE_DEPTH_CHANNEL,
    REMOTE_HEARTBEAT_CHANNEL,
    REMOTE_OVERLAY_CHANNEL,
    REMOTE_RESULT_CHANNEL,
    REMOTE_RGB_CHANNEL,
    REMOTE_ROBOT_STATE_CHANNEL,
    ValidationContext,
    now_us,
    pack_belief_telemetry,
    strip_invalid_optional_fields,
    unpack_heartbeat,
    unpack_result,
    validate_result,
    validate_result_soft,
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
DEFAULT_SHARED_HAND_SELECTION_ENDPOINT = os.environ.get(
    "PATH_SHARED_HAND_SELECTION_ENDPOINT", "tcp://127.0.0.1:5567"
)
DEFAULT_SHARED_HAND_SELECTION_TOPIC = os.environ.get(
    "PATH_SHARED_HAND_SELECTION_TOPIC", "path_shared_control_hand"
)
SHARED_HAND_PUBLISH_HZ = 50.0
# Camera-frame lookup needs the complete pipeline history.  1.5 s covers the
# pre-v9.1 639 ms path with more than 2x margin, while retaining 50 Hz samples.
COMMAND_HISTORY_SPAN_S = 1.5


def _topic_msgpack_unpack(raw: bytes, topic: str) -> dict[str, Any]:
    prefix = str(topic).encode("utf-8")
    if not raw.startswith(prefix):
        raise ValueError(f"message does not start with topic {topic!r}")
    payload = msgpack.unpackb(raw[len(prefix) :], raw=False, strict_map_key=False)
    if not isinstance(payload, dict):
        raise ValueError("topic payload is not a map")
    return payload


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


def _surface_primitive_from_dict(data: dict[str, Any]) -> SurfacePrimitive | None:
    kind = data.get("kind")
    if kind not in {"cylinder", "plane", "circle", "pca", "point"}:
        return None
    try:
        origin = np.asarray(data.get("origin_goal_m"), dtype=np.float64).reshape(3)
        axis = np.asarray(data.get("axis_goal"), dtype=np.float64).reshape(3)
        fit_quality = float(data.get("fit_quality", 0.0))
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(axis))
    if (
        not np.isfinite(origin).all()
        or not np.isfinite(axis).all()
        or not math.isfinite(fit_quality)
        or norm <= 1.0e-9
    ):
        return None
    radius = data.get("radius", 0.0)
    half_height = data.get("half_height", 0.0)
    extent = data.get("extent")
    try:
        radius = float(radius)
        half_height = float(half_height)
        extent = None if extent is None else np.asarray(extent, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(radius)
        or radius < 0.0
        or not math.isfinite(half_height)
        or half_height < 0.0
        or (extent is not None and not np.isfinite(extent).all())
    ):
        return None
    return SurfacePrimitive(
        kind=str(kind),
        origin=origin,
        axis=axis / norm,
        fit_quality=float(np.clip(fit_quality, 0.0, 1.0)),
        source=str(data.get("source", "remote")),
        radius=radius,
        half_height=half_height,
        extent=extent,
        snapped_to_gravity=bool(data.get("snapped_to_gravity", False)),
        observation_timestamp_us=int(
            data.get("observation_timestamp_us", 0) or 0
        ),
        observation_track_id=str(data.get("observation_track_id", "")),
        observation_state=str(data.get("observation_state", "active")),
    )


def _msgpack_safe(value: Any) -> Any:
    """Convert bounded telemetry values to plain MessagePack-safe Python."""
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _msgpack_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _msgpack_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_msgpack_safe(item) for item in value]
    return str(value)


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


def _finite_vec4(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(array).all() or float(np.linalg.norm(array)) <= 1.0e-9:
        return None
    return array / float(np.linalg.norm(array))


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
        self.tolerance_us = round(float(tolerance_ms) * 1000.0)
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
        belief_enabled: bool,
        belief_config: dict[str, Any],
        belief_ee_offset_m: float,
        belief_csv_path: str,
        belief_log_hz: float,
        shared_hand_selection_endpoint: str,
        shared_hand_selection_topic: str,
        belief_use_operator_command: bool,
        belief_command_divergence_max_m: float,
        belief_command_max_delta_ms: float,
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
        self.belief_update_hz = max(
            0.1, float(belief_config.get("belief_update_hz", 30.0))
        )
        self.geometry_update_hz = max(
            0.1, float(belief_config.get("geometry_update_hz", 8.0))
        )
        self.belief_log_period = 1.0 / max(0.1, float(belief_log_hz))
        self.last_belief_log_monotonic = 0.0
        self.belief_csv = (
            RemoteBeliefCsvRecorder(belief_csv_path) if str(belief_csv_path) else None
        )
        self.parallel_tasc_belief = ParallelTascBelief(belief_config)
        # Compatibility handle for callers that inspect a single selected hand.
        self.tasc_belief = self.parallel_tasc_belief.trackers["left"]
        self.belief_hand_model = str(
            belief_config.get("hand_model", "dex1")
        ).strip().lower()
        self.belief_hand_profile = hand_frame_profile(self.belief_hand_model)
        self.belief_grasp_center_local_m = np.asarray(
            belief_config.get(
                "grasp_center_local_m",
                self.belief_hand_profile["grasp_center_local"],
            ),
            dtype=np.float64,
        ).reshape(3)
        self.belief_orientation_min_confidence = float(
            belief_config.get("orientation_min_confidence", 0.35)
        )
        self.belief_telemetry_period_sec = 1.0 / max(
            15.0, float(belief_config.get("belief_telemetry_hz", 15.0))
        )
        self.last_belief_telemetry_monotonic = 0.0
        self.belief_telemetry_max_objects = max(
            1, int(belief_config.get("max_objects", 10))
        )
        self.belief_telemetry_max_contacts = max(
            1, int(belief_config.get("max_contacts_per_object", 6))
        )
        self.belief_telemetry_max_orientations = max(
            1, int(belief_config.get("max_orientation_candidates", 96))
        )
        # Compatibility only. Canonical EE geometry is now supplied by the
        # selected hand profile instead of a wrist +x scalar offset.
        self.belief_ee_offset_m = float(belief_ee_offset_m)
        self.belief_last_eef_by_hand: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, int]] = {}
        self.shared_hand_selection_endpoint = str(shared_hand_selection_endpoint)
        self.shared_hand_selection_topic = str(shared_hand_selection_topic)
        self.shared_hand_selection_socket = zmq.Context.instance().socket(zmq.SUB)
        self.shared_hand_selection_socket.setsockopt(zmq.LINGER, 0)
        # Do not conflate this 50 Hz stream: every timestamped sample is needed
        # to reconstruct the operator command at the camera-frame epoch.
        self.shared_hand_selection_socket.setsockopt(zmq.RCVHWM, 100)
        self.shared_hand_selection_socket.setsockopt_string(
            zmq.SUBSCRIBE, self.shared_hand_selection_topic
        )
        self.shared_hand_selection_socket.connect(self.shared_hand_selection_endpoint)
        self.belief_selected_hand = "none"
        self.belief_selection_active = False
        self.belief_selection_timestamp_us = 0
        self.belief_use_operator_command = bool(belief_use_operator_command)
        self.belief_command_divergence_max_m = max(
            0.0, float(belief_command_divergence_max_m)
        )
        self.belief_command_max_delta_ms = max(
            0.0, float(belief_command_max_delta_ms)
        )
        self.command_history: collections.deque[
            tuple[int, dict[str, Any]]
        ] = collections.deque(
            maxlen=int(COMMAND_HISTORY_SPAN_S * SHARED_HAND_PUBLISH_HZ)
        )
        # v2 of the existing shared-hand heartbeat carries the timestamped raw
        # operator command plus visualization/reference data.  Keeping the
        # latest copy mirrors both to Wi-Fi telemetry without a new channel.
        self.latest_hand_selection: dict[str, Any] = {}
        self.ee_target_pub = (
            ZmqEeTargetPublisher(endpoint=ee_target_endpoint, topic=ee_target_topic)
            if self.ee_target_enabled
            else None
        )
        self.sync = RawRgbdSynchronizer(sync_tolerance_ms)
        self.lock = threading.Lock()
        self.pair_ready = threading.Condition(self.lock)
        self.last_forward_monotonic = 0.0
        self.running = False
        self.pending_pair: RawPair | None = None
        self.latest_pair: RawPair | None = None
        self.forwarded_pairs: collections.deque[ForwardedPairRecord] = (
            collections.deque(maxlen=60)
        )
        self.latest_calibration: tuple[int | None, bytes] | None = None
        self.latest_robot_state: tuple[int | None, bytes] | None = None
        # Remote SAM3 results supply geometry at the worker rate.  The local
        # TASC loop below combines that cached geometry with the newest G1
        # state at its own control rate, exactly as the local SAM3 server does.
        self.latest_belief_result: dict[str, Any] | None = None
        self.latest_belief_geometry: Any | None = None
        self.latest_belief_geometry_sequence = -1
        self.last_result_sequence: int | None = None
        self._zero_primitives_warned = False
        self.stats: dict[str, Any] = {
            "local_rgb": 0,
            "local_depth": 0,
            "local_calibration": 0,
            "local_robot_state": 0,
            "forwarded_pairs": 0,
            "forward_publish_bytes": 0,
            "forward_publish_errors": 0,
            "forwarded_calibration": 0,
            "forwarded_robot_state": 0,
            "valid_results": 0,
            "invalid_results": 0,
            "soft_result_warnings": [],
            "soft_optional_fields_dropped": 0,
            "objects_with_primitives": 0,
            "objects_total": 0,
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
            "last_belief": {"enabled": self.parallel_tasc_belief.config.enabled, "hands": {}},
            "belief_selected_hand": "none",
            "belief_selection_active": False,
            "belief_selection_messages": 0,
            "belief_log_ticks": 0,
            "belief_log_rows": 0,
            "belief_csv_path": str(self.belief_csv.path) if self.belief_csv else None,
            "belief_telemetry_published": 0,
            "belief_update_hz": self.belief_update_hz,
            "belief_geometry_update_hz": self.geometry_update_hz,
            "belief_updates": 0,
            "belief_input": {
                "source": {},
                "divergence_m": {},
                "command_lookup_age_ms": {},
                "command_lookup_failures": 0,
            },
            "belief_command_divergence_m": {},
            "command_lookup_age_ms": {},
            "command_lookup_failures": 0,
            "last_stage_latency_ms": None,
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
        if self.belief_enabled and self.parallel_tasc_belief.config.enabled:
            self.threads.append(
                threading.Thread(
                    target=self._belief_loop,
                    name="remote-tasc-belief",
                    daemon=True,
                )
            )
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        with self.pair_ready:
            self.running = False
            self.pair_ready.notify_all()
        for thread in self.threads:
            thread.join(timeout=2.0)
        if self.ee_target_pub is not None:
            self.ee_target_pub.close()
        self.shared_hand_selection_socket.close(0)
        if self.belief_csv is not None:
            self.belief_csv.close()

    def _record_error(self, error: Exception | str) -> None:
        with self.lock:
            self.stats["decode_errors"] += 1
            self.stats["last_error"] = str(error)

    def _submit_pair(self, pair: RawPair | None) -> None:
        if pair is None:
            return
        with self.pair_ready:
            if self.pending_pair is not None:
                self.stats["dropped_before_forward"] = (
                    int(self.stats.get("dropped_before_forward", 0)) + 1
                )
            self.pending_pair = pair
            self.latest_pair = pair
            self.pair_ready.notify()

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
        gateway_receive_us = time.time_ns() // 1000
        try:
            payload = unpack_result(data)
            stages = dict(payload.get("stage_us") or {})
            stages["gateway_receive"] = int(gateway_receive_us)
            payload["stage_us"] = stages
            stage_keys = (
                "source_rgb",
                "worker_dequeue",
                "decode_done",
                "infer_done",
                "geometry_done",
                "gateway_receive",
            )
            if all(isinstance(stages.get(key), int) for key in stage_keys):
                with self.lock:
                    self.stats["last_stage_latency_ms"] = {
                        "transport_in": (
                            stages["worker_dequeue"] - stages["source_rgb"]
                        )
                        / 1000.0,
                        "decode": (
                            stages["decode_done"] - stages["worker_dequeue"]
                        )
                        / 1000.0,
                        "inference": (
                            stages["infer_done"] - stages["decode_done"]
                        )
                        / 1000.0,
                        "geometry": (
                            stages["geometry_done"] - stages["infer_done"]
                        )
                        / 1000.0,
                        "pack_transport": (
                            stages["gateway_receive"] - stages["geometry_done"]
                        )
                        / 1000.0,
                        "total": (
                            stages["gateway_receive"] - stages["source_rgb"]
                        )
                        / 1000.0,
                    }
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
            soft_errors = validate_result_soft(payload)
            dropped_optional = 0
            if soft_errors:
                dropped_optional = strip_invalid_optional_fields(payload)
                LOG.warning(
                    "remote result seq=%s: dropping %d invalid optional field(s): %s",
                    payload.get("sequence"),
                    dropped_optional,
                    "; ".join(soft_errors[:3]),
                )
            fatal_errors = result_errors + validate_result(
                payload, context, include_soft=False
            )
            with self.lock:
                if soft_errors:
                    self.stats["soft_result_warnings"] = soft_errors[:8]
                    self.stats["soft_optional_fields_dropped"] += dropped_optional
                if fatal_errors:
                    self.stats["invalid_results"] += 1
                    self.stats["last_invalid_result"] = {
                        "sequence": payload.get("sequence"),
                        "errors": fatal_errors,
                    }
                    LOG.warning(
                        "rejected remote result seq=%s: %s",
                        payload.get("sequence"),
                        "; ".join(fatal_errors),
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
            if not fatal_errors:
                self._publish_ee_target_from_result(payload, record)
        except Exception as error:
            self._record_error(error)

    def _belief_inputs_from_result(
        self,
        payload: dict[str, Any],
    ) -> list[BeliefInputObject]:
        inputs: list[BeliefInputObject] = []
        # Geometry timestamps share the original camera packet clock domain,
        # which is not required to be Unix wall time.  Compare fallback age to
        # the current result's source frame, never to this process's clock.
        current_timestamp_us = int(payload.get("source_rgb_timestamp_us", 0) or 0)
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
            geometry_timestamp_us = obj.get(
                "geometry_source_timestamp_us",
                payload.get("source_rgb_timestamp_us"),
            )
            if bool(obj.get("geometry_fallback", False)):
                try:
                    geometry_age_ms = (
                        current_timestamp_us - int(geometry_timestamp_us)
                    ) / 1000.0
                except (TypeError, ValueError):
                    continue
                if geometry_age_ms < -100.0 or geometry_age_ms > self.result_max_age_ms:
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
            surface_primitives: list[SurfacePrimitive] = []
            raw_surface_primitives = obj.get("surface_primitives")
            if isinstance(raw_surface_primitives, list):
                for raw_primitive in raw_surface_primitives[:1]:
                    if not isinstance(raw_primitive, dict):
                        continue
                    primitive = _surface_primitive_from_dict(raw_primitive)
                    if primitive is not None:
                        surface_primitives.append(primitive)
            object_id = str(obj.get("track_id") or f"{obj.get('label', 'object')}:{index}")
            inputs.append(
                BeliefInputObject(
                    object_id=object_id,
                    label=label,
                    score=score,
                    center_goal_m=center,
                    points_goal_m=None,
                    grip_candidates=candidates or None,
                    surface_primitives=(
                        surface_primitives
                        if isinstance(raw_surface_primitives, list)
                        else None
                    ),
                )
            )
        raw_objects = payload.get("objects", [])
        objects_total = len(raw_objects) if isinstance(raw_objects, list) else 0
        objects_with_primitives = sum(
            1
            for obj in raw_objects
            if isinstance(obj, dict)
            and isinstance(obj.get("surface_primitives"), list)
            and bool(obj["surface_primitives"])
        )
        if hasattr(self, "stats") and hasattr(self, "lock"):
            with self.lock:
                self.stats["objects_with_primitives"] = objects_with_primitives
                self.stats["objects_total"] = objects_total
        if objects_total and not objects_with_primitives:
            if not getattr(self, "_zero_primitives_warned", False):
                LOG.warning(
                    "remote result carries %d objects but zero surface primitives; "
                    "orientation belief cannot run",
                    objects_total,
                )
                self._zero_primitives_warned = True
        else:
            self._zero_primitives_warned = False
        return inputs

    def _bounded_hand_telemetry(self, value: Any) -> dict[str, Any]:
        data = _msgpack_safe(value if isinstance(value, dict) else {})
        if not isinstance(data, dict):
            data = {}
        objects = data.get("objects")
        if not isinstance(objects, list):
            objects = []
        objects = objects[: self.belief_telemetry_max_objects]
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            for key in (
                "grip_candidates",
                "grip_probabilities",
                "grip_advantages",
                "grip_track_ids",
            ):
                if isinstance(obj.get(key), list):
                    obj[key] = obj[key][: self.belief_telemetry_max_contacts]
        orientation = data.get("orientation")
        if isinstance(orientation, dict) and isinstance(
            orientation.get("candidates"), list
        ):
            orientation["candidates"] = orientation["candidates"][
                : self.belief_telemetry_max_orientations
            ]
        hand_state = data.get("hand") if isinstance(data.get("hand"), dict) else {}
        target = data.get("target") if isinstance(data.get("target"), dict) else None
        if target is not None:
            frame = target.get("orientation_frame_world")
            frame_ok = False
            try:
                matrix = np.asarray(frame, dtype=np.float64).reshape(3, 3)
                frame_ok = bool(np.isfinite(matrix).all() and abs(float(np.linalg.det(matrix))) > 0.5)
            except (TypeError, ValueError, np.linalg.LinAlgError):
                pass
            target["orientation_valid"] = bool(
                ((frame_ok and target.get("orientation_policy") == "grasp_frame")
                or (_finite_vec3(target.get("orientation_axis_world")) is not None
                and target.get("orientation_relation") in {"parallel", "perpendicular"}))
                and float(target.get("orientation_confidence", 0.0) or 0.0)
                >= self.belief_orientation_min_confidence
            )
        return {
            **data,
            "armed": bool(hand_state.get("armed", False)),
            "alpha": float(hand_state.get("alpha", 0.0) or 0.0),
            "target": target,
            "objects": objects,
            "orientation": orientation if isinstance(orientation, dict) else {},
        }

    def _publish_belief_telemetry(
        self,
        payload: dict[str, Any],
        per_hand: dict[str, Any],
    ) -> None:
        now_monotonic = time.monotonic()
        if (
            now_monotonic - self.last_belief_telemetry_monotonic
            < self.belief_telemetry_period_sec
        ):
            return
        self.last_belief_telemetry_monotonic = now_monotonic
        published_timestamp_us = now_us()
        source_rgb_timestamp_us = int(
            payload.get("source_rgb_timestamp_us", 0) or 0
        )
        control = dict(getattr(self, "latest_hand_selection", {}) or {})
        try:
            control_timestamp_us = int(control.get("timestamp_us", 0) or 0)
        except (TypeError, ValueError):
            control_timestamp_us = 0
        control_age_ms = (
            max(0.0, (published_timestamp_us - control_timestamp_us) / 1000.0)
            if control_timestamp_us > 0
            else None
        )
        packet = {
            "version": 1,
            "published_timestamp_us": published_timestamp_us,
            "source_rgb_timestamp_us": source_rgb_timestamp_us,
            "result_sequence": int(payload.get("sequence", -1)),
            "selected_hand": self.belief_selected_hand,
            "hands": {
                side: self._bounded_hand_telemetry(per_hand.get(side, {}))
                for side in ("left", "right")
            },
            "result_valid": bool(payload.get("valid", False)),
            "geometry_status": payload.get("geometry_status", {}),
            "result_age_ms": max(
                0.0,
                (published_timestamp_us - int(payload.get("published_timestamp_us", 0)))
                / 1000.0,
            ),
            "control": {
                "schema": str(control.get("schema", "")),
                "state": str(control.get("state", "")),
                "selected_hand": str(control.get("selected_hand", "none")),
                "active": bool(control.get("active", False)),
                "revision": int(control.get("revision", 0) or 0),
                "grasp_center_local_m": control.get("grasp_center_local_m"),
                "end_effector_model": control.get("end_effector_model"),
                "hands": control.get("hands", {})
                if isinstance(control.get("hands"), dict)
                else {},
                "age_ms": control_age_ms,
            },
            "belief_input": dict(
                getattr(self, "stats", {}).get("belief_input", {}) or {}
            ),
        }
        self.wifi_bus.publish(
            REMOTE_BELIEF_TELEMETRY_CHANNEL,
            pack_belief_telemetry(_msgpack_safe(packet)),
        )
        with self.lock:
            self.stats["belief_telemetry_published"] += 1

    def _drain_shared_hand_selection(self) -> None:
        while True:
            try:
                raw = self.shared_hand_selection_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                payload = _topic_msgpack_unpack(raw, self.shared_hand_selection_topic)
                hand = str(payload.get("selected_hand", "none")).lower()
                if hand not in {"left", "right", "none"}:
                    continue
                active = bool(payload.get("active", False)) and hand != "none"
                timestamp_us = int(payload.get("timestamp_us", 0) or 0)
                next_hand = hand if active else "none"
                if next_hand != self.belief_selected_hand:
                    # A_delta histories are isolated by hand tracker.
                    for tracker in self.parallel_tasc_belief.trackers.values():
                        tracker.feature_cost.previous_direction = None
                self.belief_selected_hand = next_hand
                self.belief_selection_active = active
                self.belief_selection_timestamp_us = timestamp_us
                self.latest_hand_selection = {
                    "schema": str(payload.get("schema", "")),
                    "timestamp_us": timestamp_us,
                    "state": str(payload.get("state", "")),
                    "selected_hand": hand,
                    "active": active,
                    "revision": int(payload.get("revision", 0) or 0),
                    "grasp_center_local_m": payload.get("grasp_center_local_m"),
                    "end_effector_model": payload.get("end_effector_model"),
                    "hands": payload.get("hands", {})
                    if isinstance(payload.get("hands"), dict)
                    else {},
                }
                if isinstance(payload.get("hands"), dict) and timestamp_us > 0:
                    self.command_history.append((timestamp_us, payload["hands"]))
                with self.lock:
                    self.stats["belief_selected_hand"] = self.belief_selected_hand
                    self.stats["belief_selection_active"] = active
                    self.stats["belief_selection_messages"] = int(
                        self.stats.get("belief_selection_messages", 0)
                    ) + 1
            except Exception as error:
                with self.lock:
                    self.stats["last_belief_selection_error"] = str(error)

    def _command_features_at(
        self,
        timestamp_us: int,
        hand: str,
        max_delta_ms: float | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ] | None:
        """Return the operator command nearest the geometry frame epoch.

        CONVENTION: ``source_rgb_timestamp_us`` feeds two independent matchers:
        robot-state matching supplies geometry/T_goal_cam, and this command
        lookup supplies the belief hand state.  Changing the timestamp meaning
        shifts both lookup targets and therefore requires validating both
        ``geometry_status.state_delta_ms`` and ``command_lookup_age_ms``.
        """
        history = getattr(self, "command_history", ())
        if not history:
            self.stats["command_lookup_failures"] = int(
                self.stats.get("command_lookup_failures", 0)
            ) + 1
            return None
        limit_ms = (
            float(getattr(self, "belief_command_max_delta_ms", 25.0))
            if max_delta_ms is None
            else float(max_delta_ms)
        )
        target_timestamp_us = int(timestamp_us)
        best = min(
            history,
            key=lambda item: abs(int(item[0]) - target_timestamp_us),
        )
        age_ms = abs(int(best[0]) - target_timestamp_us) / 1000.0
        self.stats.setdefault("command_lookup_age_ms", {})[hand] = age_ms
        if age_ms > limit_ms:
            self.stats["command_lookup_failures"] = int(
                self.stats.get("command_lookup_failures", 0)
            ) + 1
            return None
        entry = (best[1] or {}).get(hand) or {}
        position = _finite_vec3(entry.get("command_wrist_position_m"))
        raw_rotation = entry.get("command_wrist_rotation")
        if (
            position is None
            or not isinstance(raw_rotation, list)
            or len(raw_rotation) != 9
        ):
            self.stats["command_lookup_failures"] = int(
                self.stats.get("command_lookup_failures", 0)
            ) + 1
            return None
        rotation = np.asarray(raw_rotation, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(rotation).all():
            self.stats["command_lookup_failures"] = int(
                self.stats.get("command_lookup_failures", 0)
            ) + 1
            return None
        return (
            position,
            rotation,
            _finite_vec3(entry.get("command_velocity_mps")),
            _finite_vec3(entry.get("command_omega_rps")),
        )

    def _belief_target_link(self) -> str:
        if self.belief_selection_active and self.belief_selected_hand in {"left", "right"}:
            return f"{self.belief_selected_hand}_wrist_yaw_link"
        return self.ee_target_link

    def _belief_state_features(
        self,
        robot_state_packet: bytes | None,
        hand: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if robot_state_packet is None or hand not in {"left", "right"}:
            return None, None, None, None
        try:
            state = decode_robot_state(robot_state_packet)
            position = g1_grasp_center_position(
                state.q,
                f"{hand}_wrist_yaw_link",
                self.belief_grasp_center_local_m,
                self.belief_hand_model,
            )
            hand_rotation = g1_hand_base_rotation(
                state.q, f"{hand}_wrist_yaw_link", self.belief_hand_model
            )
            velocity = None
            angular_velocity = None
            previous = self.belief_last_eef_by_hand.get(hand)
            if previous is not None:
                previous_position, previous_rotation, previous_omega, previous_timestamp_us = previous
                dt = (state.timestamp_us - previous_timestamp_us) / 1.0e6
                if 1.0e-4 <= dt <= 0.5:
                    velocity = (position - previous_position) / dt
                    raw_omega = angular_velocity_from_rotations(previous_rotation, hand_rotation, dt)
                    if raw_omega is not None:
                        angular_velocity = raw_omega if previous_omega is None else 0.25 * raw_omega + 0.75 * previous_omega
            self.belief_last_eef_by_hand[hand] = (position.copy(), hand_rotation.copy(), angular_velocity, int(state.timestamp_us))
            return position, velocity, hand_rotation, angular_velocity
        except Exception as error:
            with self.lock:
                self.stats["last_belief_state_error"] = str(error)
            return None, None, None, None

    def _belief_update_hands(
        self,
        shared_geometry: Any,
        hand_inputs: dict[str, HandInput],
        payload: dict[str, Any],
    ) -> dict[str, dict]:
        """Update belief with the legacy, globally shared geometry.

        Camera-specific gateways may override this narrow hook to partition
        geometry without duplicating the established head-camera belief,
        selection, logging, or target-publication pipeline.
        """
        del payload
        return self.parallel_tasc_belief.update_hands(
            shared_geometry, hand_inputs
        )

    def _belief_selection_from_result(
        self,
        payload: dict[str, Any],
        robot_state_packet: bytes | None,
        shared_geometry: Any | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], str] | None:
        if not self.belief_enabled or not self.parallel_tasc_belief.config.enabled:
            return None
        self._drain_shared_hand_selection()
        inputs = self._belief_inputs_from_result(payload)
        # An empty detector frame must not stop the continuous belief update.
        # Shared geometry can still contain coasting orientation-bank tracks (or
        # the gravity candidate), and both unarmed hand posteriors remain useful
        # diagnostics.  A/B only gates target publication below; it never gates
        # inference or CSV recording.
        source_timestamp_us = int(
            payload.get("source_rgb_timestamp_us", 0) or 0
        )
        hand_inputs: dict[str, HandInput] = {}
        input_sources: dict[str, str] = {}
        for side in ("left", "right"):
            command = (
                self._command_features_at(source_timestamp_us, side)
                if getattr(self, "belief_use_operator_command", False)
                else None
            )
            (
                fk_position,
                fk_velocity,
                fk_rotation,
                fk_omega,
            ) = self._belief_state_features(robot_state_packet, side)
            source = "robot_fk"
            position = None
            velocity = None
            hand_rotation = None
            hand_omega = None
            if command is not None:
                wrist_position, wrist_rotation, velocity, hand_omega = command
                position = grasp_center_from_wrist(
                    wrist_position,
                    wrist_rotation,
                    side,
                    self.belief_hand_model,
                    self.belief_grasp_center_local_m,
                )
                hand_rotation = hand_rotation_from_wrist_rotation(
                    wrist_rotation,
                    side,
                    self.belief_hand_model,
                )
                source = "operator_command"
                if fk_position is not None:
                    divergence = float(np.linalg.norm(position - fk_position))
                    self.stats.setdefault(
                        "belief_command_divergence_m", {}
                    )[side] = divergence
                    if divergence > float(
                        getattr(
                            self,
                            "belief_command_divergence_max_m",
                            0.20,
                        )
                    ):
                        command = None
                        source = "fk_divergence_fallback"
            if command is None:
                if fk_position is None:
                    continue
                position, velocity = fk_position, fk_velocity
                hand_rotation, hand_omega = fk_rotation, fk_omega
            input_sources[side] = source
            hand_inputs[side] = HandInput(
                eef_position=position,
                user_translation=velocity,
                hand_rotation=hand_rotation,
                hand_axis=None,
                approach_axis=(hand_rotation @ np.asarray(self.belief_hand_profile["approach_axis_local"], dtype=np.float64) if hand_rotation is not None else None),
                angular_velocity=hand_omega,
                button_pressed=(self.belief_selection_active and self.belief_selected_hand == side),
                input_source=source,
            )
        belief_input_telemetry = {
            "source": dict(input_sources),
            "divergence_m": dict(
                self.stats.get("belief_command_divergence_m", {})
            ),
            "command_lookup_age_ms": dict(
                self.stats.get("command_lookup_age_ms", {})
            ),
            "command_lookup_failures": int(
                self.stats.get("command_lookup_failures", 0)
            ),
        }
        self.stats["belief_input"] = belief_input_telemetry
        if not hand_inputs:
            telemetry = {
                "enabled": True,
                "objects": [],
                "target": None,
                "hands": {},
                "selected_hand": self.belief_selected_hand,
                "selection_active": self.belief_selection_active,
                "reason": "robot_state_unavailable_for_both_hands",
                "belief_input": belief_input_telemetry,
            }
            with self.lock:
                self.stats["last_belief"] = telemetry
            self._record_belief_csv(payload, telemetry)
            return None
        ray_origin = _finite_vec3(payload.get("camera_goal_origin_m"))
        ray_direction = _finite_vec3(payload.get("camera_goal_forward_m"))
        up_goal = None
        base_quat_wxyz = None
        belief_robot_state_timestamp_us = None
        belief_robot_q = None
        if robot_state_packet is not None:
            try:
                local_state = decode_robot_state(robot_state_packet)
                belief_robot_state_timestamp_us = int(local_state.timestamp_us)
                belief_robot_q = np.asarray(
                    local_state.q, dtype=np.float64
                ).reshape(-1).tolist()
                base_quat_wxyz = local_state.base_quat_wxyz
                # Remote geometry can be 100-200 ms old. Belief orientation
                # uses Thor's current IMU; remote gravity is diagnostic only.
                up_goal = gravity_up_in_pelvis(base_quat_wxyz)
            except Exception as error:
                with self.lock:
                    self.stats["last_belief_gravity_error"] = str(error)
        if shared_geometry is None:
            shared_geometry = self.parallel_tasc_belief.build_shared_geometry(
                inputs, ray_origin=ray_origin, ray_direction=ray_direction,
                up_goal=up_goal,
                base_quat_wxyz=base_quat_wxyz,
                stamp_us=int(payload.get("source_rgb_timestamp_us", 0) or 0),
            )
        per_hand = self._belief_update_hands(
            shared_geometry, hand_inputs, payload
        )
        for side, hand_telemetry in per_hand.items():
            hand_input = hand_inputs.get(side)
            if hand_input is None or not isinstance(hand_telemetry, dict):
                continue
            control_hand = (
                (self.latest_hand_selection.get("hands") or {}).get(side) or {}
                if isinstance(self.latest_hand_selection, dict)
                else {}
            )
            grasp_commanded = bool(control_hand.get("grasp_commanded", False))
            grasp_engaged = bool(control_hand.get("grasp_engaged", False))
            jaw_gap_m = control_hand.get("jaw_gap_m")
            grasp_state = str(control_hand.get("grasp_state", "reaching"))
            hand_telemetry["grasp_commanded"] = grasp_commanded
            hand_telemetry["grasp_engaged"] = grasp_engaged
            hand_telemetry["jaw_gap_m"] = jaw_gap_m
            hand_telemetry["grasp_state"] = grasp_state
            hand_telemetry["grasp"] = control_hand.get("grasp")
            hand_telemetry["frame_offset"] = control_hand.get("frame_offset")
            hand_telemetry["grasp_state_source"] = control_hand.get(
                "grasp_state_source", "unavailable"
            )
            hand_target = hand_telemetry.get("target")
            if isinstance(hand_target, dict):
                hand_target["position_valid"] = not grasp_engaged
                if grasp_engaged:
                    # Do not change the posterior or orientation fields. The
                    # held object no longer supplies a reach/contact target.
                    hand_target["position_suppressed_reason"] = "grasp_engaged"
            # Tracker telemetry can return early when there are no valid
            # objects.  Preserve the actual gateway FK input on those ticks so
            # timing/FK gaps remain diagnosable instead of appearing blank.
            hand_telemetry["eef_position_m"] = np.asarray(
                hand_input.eef_position, dtype=np.float64
            ).reshape(3).tolist()
            hand_telemetry["eef_velocity_mps"] = (
                None
                if hand_input.user_translation is None
                else np.asarray(
                    hand_input.user_translation, dtype=np.float64
                ).reshape(3).tolist()
            )
            hand_telemetry["hand_rotation_matrix"] = (
                None
                if hand_input.hand_rotation is None
                else np.asarray(
                    hand_input.hand_rotation, dtype=np.float64
                ).reshape(3, 3).tolist()
            )
            hand_telemetry["hand_approach_axis"] = (
                None
                if hand_input.approach_axis is None
                else np.asarray(
                    hand_input.approach_axis, dtype=np.float64
                ).reshape(3).tolist()
            )
            hand_telemetry["input_source"] = input_sources.get(side, "robot_fk")
        self._publish_belief_telemetry(payload, per_hand)
        selected = per_hand.get(self.belief_selected_hand)
        telemetry = (
            dict(selected)
            if selected is not None
            else {"enabled": True, "objects": [], "target": None}
        )
        telemetry["hands"] = per_hand
        telemetry["selected_hand"] = self.belief_selected_hand
        telemetry["selection_active"] = self.belief_selection_active
        telemetry["belief_input"] = belief_input_telemetry
        telemetry["eef_target_link"] = self._belief_target_link()
        telemetry["belief_robot_state_timestamp_us"] = (
            belief_robot_state_timestamp_us
        )
        telemetry["belief_robot_q"] = belief_robot_q
        telemetry["belief_base_quat_wxyz"] = (
            None
            if base_quat_wxyz is None
            else np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4).tolist()
        )
        telemetry["belief_hand_model"] = getattr(
            self, "belief_hand_model", None
        )
        grasp_center_local_m = getattr(
            self, "belief_grasp_center_local_m", None
        )
        telemetry["belief_grasp_center_local_m"] = (
            None
            if grasp_center_local_m is None
            else np.asarray(
                grasp_center_local_m, dtype=np.float64
            ).reshape(3).tolist()
        )
        if not inputs:
            telemetry["reason"] = "no_valid_remote_geometry_coasting"
        telemetry["ray_source"] = (
            "camera_optical_center" if ray_origin is not None and ray_direction is not None else "unavailable"
        )
        selected_tracker = self.parallel_tasc_belief.trackers.get(self.belief_selected_hand)
        if selected_tracker is not None and selected_tracker.should_log():
            LOG.info(selected_tracker.format_log_line())
            selected_tracker.mark_logged()
        with self.lock:
            self.stats["last_belief"] = telemetry
        self._record_belief_csv(payload, telemetry)
        # A/B only gates command injection. Both posteriors above were updated
        # regardless of button state; with no hold we deliberately publish none.
        if not self.belief_selection_active:
            return None
        target = telemetry.get("target")
        if not isinstance(target, dict):
            return None
        position = _finite_vec3(target.get("position_goal_m"))
        if position is None:
            return None
        control_hand = (
            (self.latest_hand_selection.get("hands") or {}).get(
                self.belief_selected_hand
            )
            or {}
            if isinstance(self.latest_hand_selection, dict)
            else {}
        )
        grasp_commanded = bool(control_hand.get("grasp_commanded", False))
        grasp_engaged = bool(control_hand.get("grasp_engaged", False))
        grasp_state = str(control_hand.get("grasp_state", "reaching"))
        position_valid = bool(target.get("position_valid", True)) and not grasp_engaged
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
            "geometry_fallback": bool(selected.get("geometry_fallback", False)),
            "missing_frames": int(selected.get("missing_frames", 0)),
            "center_camera_m": selected.get("center_camera_m"),
            "center_goal_m": selected.get("center_goal_m"),
            "target_position_frame": selected.get("target_position_frame", ""),
            "target_orientation_frame": selected.get("target_orientation_frame", ""),
            "wrist_link": selected.get("wrist_link"),
            "transform_diagnostics": selected.get("transform_diagnostics"),
            "target_contact_goal_m": position.tolist(),
            "tasc_object_id": object_id,
            "tasc_selected_grip_index": target.get("grip_index"),
            "tasc_object_probability": target.get("object_confidence"),
            "tasc_joint_confidence": target.get("joint_confidence"),
            "streaming_goal": True,
            "target_valid": True,
            "belief_selected": True,
            "belief_selected_hand": self.belief_selected_hand,
            "belief_ee_target_link": self._belief_target_link(),
            "grasp_commanded": grasp_commanded,
            "grasp_engaged": grasp_engaged,
            "grasp_state": grasp_state,
            "grasp": control_hand.get("grasp"),
            "frame_offset": control_hand.get("frame_offset"),
            "jaw_gap_m": control_hand.get("jaw_gap_m"),
            "grasp_state_source": control_hand.get(
                "grasp_state_source", "unavailable"
            ),
            "position_valid": position_valid,
            "position_suppressed_reason": (
                "grasp_engaged" if not position_valid else None
            ),
            "belief_ray_origin_m": ray_origin.tolist() if ray_origin is not None else None,
            "belief_ray_direction": ray_direction.tolist() if ray_direction is not None else None,
            "orientation_object_id": target.get("orientation_object_id"),
            "orientation_relation": target.get("orientation_relation"),
            "orientation_axis_world": target.get("orientation_axis_world"),
            "orientation_confidence": target.get("orientation_confidence"),
            "orientation_source_position_m": target.get("orientation_source_position_m"),
            "orientation_policy": target.get("orientation_policy", "minimal_rotation"),
            "orientation_frame_world": target.get("orientation_frame_world"),
        }
        output = {
            "position": position,
            "metadata": metadata,
            "source": "remote_tasc_belief",
            "goal_uid": f"remote_belief:{self._belief_target_link()}:{object_id or label}",
            "confidence": float(target.get("object_confidence", 1.0)),
            "object_confidence": float(target.get("object_confidence", 1.0)),
            "joint_confidence": float(target.get("joint_confidence", 1.0)),
            "object_id": object_id,
            "label": label,
            "grip_index": int(target.get("grip_index", -1)),
            "orientation_xyzw": target.get("orientation_xyzw"),
            "orientation_axis_world": target.get("orientation_axis_world"),
            "orientation_relation": target.get("orientation_relation"),
            "orientation_confidence": target.get("orientation_confidence"),
            "orientation_policy": target.get("orientation_policy", "minimal_rotation"),
            "orientation_frame_world": target.get("orientation_frame_world"),
            "position_point": f"{self.belief_hand_model}_grasp_center",
            "position_valid": position_valid,
            "position_suppressed_reason": (
                "grasp_engaged" if not position_valid else None
            ),
            "grasp_engaged": grasp_engaged,
            "grasp_state": grasp_state,
        }
        return position, output, "belief"

    def _record_belief_csv(
        self,
        result: dict[str, Any],
        telemetry: dict[str, Any],
    ) -> None:
        if self.belief_csv is None:
            return
        now = time.monotonic()
        if now - self.last_belief_log_monotonic < self.belief_log_period:
            return
        self.last_belief_log_monotonic = now
        payload = {
            "published_timestamp_us": int(time.time_ns() // 1000),
            "source_rgb_timestamp_us": int(result.get("source_rgb_timestamp_us", 0) or 0),
            "source_depth_timestamp_us": int(result.get("source_depth_timestamp_us", 0) or 0),
            "remote_published_timestamp_us": int(result.get("published_timestamp_us", 0) or 0),
            "remote_sequence": int(result.get("sequence", -1)),
            "result_valid": bool(result.get("valid", False)),
            "backend": result.get("backend"),
            "inference_ms": result.get("inference_ms"),
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "selected_hand": self.belief_selected_hand,
            "selection_active": bool(self.belief_selection_active),
            "hands": telemetry.get("hands", {}),
            "reason": telemetry.get("reason"),
            "belief_robot_state_timestamp_us": telemetry.get(
                "belief_robot_state_timestamp_us"
            ),
            "belief_robot_q": telemetry.get("belief_robot_q"),
            "belief_base_quat_wxyz": telemetry.get("belief_base_quat_wxyz"),
            "belief_hand_model": telemetry.get("belief_hand_model"),
            "belief_grasp_center_local_m": telemetry.get(
                "belief_grasp_center_local_m"
            ),
            "remote_objects": result.get("objects", []),
        }
        try:
            rows = self.belief_csv.record(payload)
            with self.lock:
                self.stats["belief_log_ticks"] = int(self.stats.get("belief_log_ticks", 0)) + 1
                self.stats["belief_log_rows"] = int(self.stats.get("belief_log_rows", 0)) + int(rows)
        except Exception as error:
            self._record_error(error)

    def _publish_belief_target(
        self,
        payload: dict[str, Any],
        belief_selection: tuple[np.ndarray, dict[str, Any], str],
    ) -> None:
        """Publish the selected hand target from the continuous belief loop."""
        if self.ee_target_pub is None:
            return
        position, selected_output, selected_position_field = belief_selection
        target_link = self._belief_target_link()
        target_side = _side_from_link(target_link)
        orientation_xyzw = _finite_vec4(selected_output.get("orientation_xyzw"))
        orientation_wxyz = (
            self.ee_target_quaternion_wxyz
            if orientation_xyzw is None
            else (
                float(orientation_xyzw[3]),
                float(orientation_xyzw[0]),
                float(orientation_xyzw[1]),
                float(orientation_xyzw[2]),
            )
        )
        payload_out = make_goal_payload(
            position,
            side=target_side,
            target_link=target_link,
            quaternion=orientation_wxyz,
            duration_sec=self.ee_target_duration_sec,
            final_hold_sec=self.ee_target_final_hold_sec,
            workspace_info={
                "workspace_mode": "remote_gateway_belief",
                "workspace_reason": "continuous_remote_tasc_belief",
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
                "position_point": selected_output["position_point"],
                "position_valid": bool(selected_output.get("position_valid", True)),
                "position_suppressed_reason": selected_output.get(
                    "position_suppressed_reason"
                ),
                "grasp_engaged": bool(selected_output.get("grasp_engaged", False)),
                "grasp_state": selected_output.get("grasp_state", "reaching"),
                "orientation_axis_world": selected_output["orientation_axis_world"],
                "orientation_relation": selected_output["orientation_relation"],
                "orientation_confidence": selected_output["orientation_confidence"],
                "orientation_policy": selected_output.get("orientation_policy", "minimal_rotation"),
                "orientation_frame_world": selected_output.get("orientation_frame_world"),
            }
        )
        try:
            matrix = np.asarray(
                payload_out.get("orientation_frame_world"), dtype=np.float64
            ).reshape(3, 3)
            frame_ok = bool(
                np.isfinite(matrix).all() and abs(float(np.linalg.det(matrix))) > 0.5
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            frame_ok = False
        relation_ok = bool(
            _finite_vec3(payload_out.get("orientation_axis_world")) is not None
            and payload_out.get("orientation_relation") in {"parallel", "perpendicular"}
        )
        policy_ok = bool(
            frame_ok and payload_out.get("orientation_policy") == "grasp_frame"
        )
        payload_out["orientation_valid"] = bool(
            (policy_ok or relation_ok)
            and float(payload_out.get("orientation_confidence", 0.0) or 0.0)
            >= self.belief_orientation_min_confidence
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
                "target_link": target_link,
                "position": np.round(position, 4).tolist(),
                "position_field": selected_position_field,
            }

    def _belief_loop(self) -> None:
        """Run belief from cached remote geometry and the latest G1 state."""
        belief_period = 1.0 / self.belief_update_hz
        geometry_period = 1.0 / self.geometry_update_hz
        next_belief = 0.0
        next_geometry = 0.0
        while self.running:
            now = time.monotonic()
            if now < next_belief:
                time.sleep(min(0.005, next_belief - now))
                continue
            next_belief = max(next_belief + belief_period, now + belief_period)
            try:
                self._drain_shared_hand_selection()
                with self.lock:
                    result = self.latest_belief_result
                    robot_state = (
                        None
                        if self.latest_robot_state is None
                        else self.latest_robot_state[1]
                    )
                if result is None or robot_state is None:
                    continue
                inputs = self._belief_inputs_from_result(result)
                if (
                    self.latest_belief_geometry is None
                    or now >= next_geometry
                    or int(result.get("sequence", -1))
                    != self.latest_belief_geometry_sequence
                ):
                    state = decode_robot_state(robot_state)
                    self.latest_belief_geometry = self.parallel_tasc_belief.build_shared_geometry(
                        inputs,
                        ray_origin=_finite_vec3(result.get("camera_goal_origin_m")),
                        ray_direction=_finite_vec3(result.get("camera_goal_forward_m")),
                        up_goal=gravity_up_in_pelvis(state.base_quat_wxyz),
                        base_quat_wxyz=state.base_quat_wxyz,
                        stamp_us=int(result.get("source_rgb_timestamp_us", 0) or 0),
                    )
                    self.latest_belief_geometry_sequence = int(result.get("sequence", -1))
                    next_geometry = now + geometry_period
                selection = self._belief_selection_from_result(
                    result, robot_state, self.latest_belief_geometry
                )
                with self.lock:
                    self.stats["belief_updates"] += 1
                if selection is not None:
                    self._publish_belief_target(result, selection)
            except Exception as error:
                self._record_error(error)
                time.sleep(0.01)

    def _publish_ee_target_from_result(
        self,
        payload: dict[str, Any],
        record: ForwardedPairRecord | None,
    ) -> None:
        if not bool(payload.get("valid", False)):
            with self.lock:
                self.stats["ee_targets_skipped"] += 1
                self.stats["last_ee_target_skip"] = {
                    "sequence": payload.get("sequence"),
                    "reason": "remote_result_invalid",
                    "error": payload.get("error"),
                }
            return
        if self.belief_enabled and self.parallel_tasc_belief.config.enabled:
            # Result callbacks only refresh remote geometry.  The dedicated
            # belief loop combines it with current G1 FK at 30 Hz.
            with self.lock:
                self.latest_belief_result = dict(payload)
                self.latest_belief_geometry = None
                self.latest_belief_geometry_sequence = -1
            return
        belief_selection = None
        if self.ee_target_pub is None:
            return
        target_link = self._belief_target_link()
        target_side = _side_from_link(target_link)
        if belief_selection is not None:
            position, selected_output, selected_position_field = belief_selection
            orientation_xyzw = _finite_vec4(selected_output.get("orientation_xyzw"))
            orientation_wxyz = (
                self.ee_target_quaternion_wxyz
                if orientation_xyzw is None
                else (float(orientation_xyzw[3]), float(orientation_xyzw[0]), float(orientation_xyzw[1]), float(orientation_xyzw[2]))
            )
            payload_out = make_goal_payload(
                position,
                side=target_side,
                target_link=target_link,
                quaternion=orientation_wxyz,
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
                    "position_point": selected_output["position_point"],
                    "orientation_axis_world": selected_output[
                        "orientation_axis_world"
                    ],
                    "orientation_relation": selected_output[
                        "orientation_relation"
                    ],
                    "orientation_confidence": selected_output[
                        "orientation_confidence"
                    ],
                    "orientation_policy": selected_output.get("orientation_policy", "minimal_rotation"),
                    "orientation_frame_world": selected_output.get("orientation_frame_world"),
                }
            )
            frame_ok = False
            try:
                matrix = np.asarray(payload_out.get("orientation_frame_world"), dtype=np.float64).reshape(3, 3)
                frame_ok = bool(np.isfinite(matrix).all() and abs(float(np.linalg.det(matrix))) > 0.5)
            except (TypeError, ValueError, np.linalg.LinAlgError):
                pass
            relation_ok = bool(
                _finite_vec3(payload_out.get("orientation_axis_world")) is not None
                and payload_out.get("orientation_relation") in {"parallel", "perpendicular"}
            )
            policy_ok = bool(frame_ok and payload_out.get("orientation_policy") == "grasp_frame")
            payload_out["orientation_valid"] = bool(
                (policy_ok or relation_ok)
                and float(payload_out.get("orientation_confidence", 0.0) or 0.0)
                >= self.belief_orientation_min_confidence
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
                    "target_link": target_link,
                    "position": np.round(position, 4).tolist(),
                    "position_field": selected_position_field,
                }
            return
        if self.belief_enabled and self.parallel_tasc_belief.config.enabled:
            # Do not fall through to the legacy one-hand/raw object publisher:
            # A/B is the sole authority gate for injecting a belief reference.
            with self.lock:
                self.stats["ee_targets_skipped"] += 1
                self.stats["last_ee_target_skip"] = {
                    "sequence": payload.get("sequence"),
                    "reason": (
                        "waiting_for_A_or_B_hand_selection"
                        if not self.belief_selection_active
                        else "selected_hand_has_no_valid_belief_target"
                    ),
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
            "geometry_fallback": bool(obj.get("geometry_fallback", False)),
            "missing_frames": int(obj.get("missing_frames", 0)),
            "streaming_goal": True,
        }
        payload_out = make_goal_payload(
            selection.position,
            side=target_side,
            target_link=target_link,
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
                    "remote_sam3_contact"
                    if selection.position_field == "target_contact_goal_m"
                    else "remote_sam3_center"
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
                "target_link": target_link,
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

    def _forward_loop(self) -> None:
        last_calibration = 0.0
        last_robot_state = 0.0
        while self.running:
            pair = None
            calibration = None
            robot_state = None
            with self.pair_ready:
                # Frames wake the forwarding loop immediately.  The timeout
                # keeps calibration/state resend and shutdown responsive when
                # the camera is idle.
                self.pair_ready.wait_for(
                    lambda: self.pending_pair is not None or not self.running,
                    timeout=0.2,
                )
                if not self.running:
                    break
                pair = self.pending_pair
                self.pending_pair = None
                now = time.monotonic()
                if (
                    self.latest_calibration is not None
                    and now - last_calibration >= self.calibration_resend_sec
                ):
                    calibration = self.latest_calibration[1]
                    last_calibration = now
                if (\r\n                    self.latest_robot_state is not None\r\n                    and now - last_robot_state >= self.robot_state_period\r\n                ):\r\n                    robot_state = self.latest_robot_state[1]\r\n                    last_robot_state = now\r\n            # This period is a Wi-Fi bandwidth cap.  It throttles bursts but
            # no longer makes each frame wait for an unrelated polling tick.
            if pair is not None:
                elapsed = time.monotonic() - self.last_forward_monotonic
                if elapsed < self.forward_period:
                    time.sleep(self.forward_period - elapsed)
                self.last_forward_monotonic = time.monotonic()
            if calibration is not None:
                self.wifi_bus.publish(REMOTE_CALIBRATION_CHANNEL, calibration)
                with self.lock:
                    self.stats["forwarded_calibration"] += 1
            if robot_state is not None:
                self.wifi_bus.publish(REMOTE_ROBOT_STATE_CHANNEL, robot_state)
                with self.lock:
                    self.stats["forwarded_robot_state"] += 1
            if pair is not None:
                try:
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
                        self.stats["forward_publish_bytes"] = len(pair.depth.data)
                        self.stats["forwarded_pairs"] += 1
                        self.stats["last_forwarded"] = {
                            "rgb_timestamp_us": pair.rgb.timestamp_us,
                            "depth_timestamp_us": pair.depth.timestamp_us,
                            "delta_ms": pair.delta_us / 1000.0,
                            "width": pair.rgb.width,
                            "height": pair.rgb.height,
                        }
                except Exception as error:
                    with self.lock:
                        self.stats["forward_publish_errors"] += 1
                        self.stats["last_error"] = str(error)

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
    parser.add_argument(
        "--forward-hz",
        type=float,
        default=15.0,
        help=(
            "Wi-Fi link bandwidth cap, not a target rate. Frames are published "
            "on arrival; this only throttles bursts."
        ),
    )
    parser.add_argument("--result-max-age-ms", type=float, default=800.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=20.0)
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
        "--belief-inference",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PATH_REMOTE_BELIEF_ENABLED", "1").lower()
        not in {"0", "false", "no", "off"},
        help=(
            "Run the same hierarchical TASC belief selector used by the Thor web "
            "server before publishing EE targets."
        ),
    )
    parser.add_argument(
        "--belief-csv",
        default=os.environ.get("PATH_REMOTE_BELIEF_CSV", ""),
        help=(
            "Append two-hand belief/orientation diagnostics on Thor to this CSV. "
            "Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--belief-log-hz",
        type=float,
        default=float(os.environ.get("PATH_REMOTE_BELIEF_LOG_HZ", "10.0")),
        help="Maximum Thor-side belief CSV logging rate.",
    )
    parser.add_argument(
        "--belief-input-source",
        choices=("operator_command", "robot_fk"),
        default=os.environ.get(
            "PATH_REMOTE_BELIEF_INPUT_SOURCE", "operator_command"
        ),
        help=(
            "Hand state used by TASC belief. Operator command is aligned to "
            "the RGB frame timestamp and falls back to robot FK on lookup or "
            "divergence failure."
        ),
    )
    parser.add_argument(
        "--belief-command-divergence-max-m",
        type=float,
        default=float(
            os.environ.get(
                "PATH_REMOTE_BELIEF_COMMAND_DIVERGENCE_MAX_M", "0.20"
            )
        ),
        help="Maximum command/FK grasp-centre separation before FK fallback.",
    )
    parser.add_argument(
        "--belief-command-max-delta-ms",
        type=float,
        default=float(
            os.environ.get("PATH_REMOTE_BELIEF_COMMAND_MAX_DELTA_MS", "25.0")
        ),
        help="Maximum command sample distance from source_rgb_timestamp_us.",
    )
    parser.add_argument(
        "--shared-hand-selection-endpoint",
        default=DEFAULT_SHARED_HAND_SELECTION_ENDPOINT,
        help="ZMQ state-machine A/B hand-selection heartbeat endpoint.",
    )
    parser.add_argument(
        "--shared-hand-selection-topic",
        default=DEFAULT_SHARED_HAND_SELECTION_TOPIC,
        help="ZMQ state-machine A/B hand-selection heartbeat topic.",
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
        belief_enabled=args.belief_inference,
        belief_config=belief_config,
        belief_ee_offset_m=float(belief_config.get("ee_offset_m", target_config.get("ee_offset_m", 0.05))),
        belief_csv_path=args.belief_csv,
        belief_log_hz=args.belief_log_hz,
        shared_hand_selection_endpoint=args.shared_hand_selection_endpoint,
        shared_hand_selection_topic=args.shared_hand_selection_topic,
        belief_use_operator_command=(
            args.belief_input_source == "operator_command"
        ),
        belief_command_divergence_max_m=(
            args.belief_command_divergence_max_m
        ),
        belief_command_max_delta_ms=args.belief_command_max_delta_ms,
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
        "Remote TASC belief inference %s using config=%s input=%s "
        "command_delta_ms=%.1f divergence_m=%.3f",
        "enabled" if args.belief_inference else "disabled",
        args.config,
        args.belief_input_source,
        args.belief_command_max_delta_ms,
        args.belief_command_divergence_max_m,
    )
    if args.belief_csv:
        LOG.info("Thor belief CSV logging enabled: %s at %.1f Hz", args.belief_csv, args.belief_log_hz)
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
