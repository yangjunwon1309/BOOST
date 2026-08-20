"""Exact legacy Thor belief execution adapted to wrist-v11 wire channels.

This class intentionally inherits the belief implementation from the preserved
head-D435i Thor gateway. The adapter only initializes its required state,
normalizes wrist-result timestamps, and changes the telemetry channel. Belief
input construction, command-history matching, FK features, the continuous
geometry/belief loop, ParallelTascBelief updates, and target selection execute
the legacy methods directly.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

import lcm
import numpy as np
import zmq

from seg_web_node.lcm_protocol import decode_robot_state
from seg_web_node.robot_geometry import hand_frame_profile
from seg_web_node.tasc_belief import ParallelTascBelief

from .thor_gateway_head_d435i_legacy import (
    COMMAND_HISTORY_SPAN_S,
    DEFAULT_EE_TARGET_ENDPOINT,
    DEFAULT_EE_TARGET_TOPIC,
    DEFAULT_SHARED_HAND_SELECTION_ENDPOINT,
    DEFAULT_SHARED_HAND_SELECTION_TOPIC,
    SHARED_HAND_PUBLISH_HZ,
    ThorRemoteGateway,
    ZmqEeTargetPublisher,
    _msgpack_safe,
)
from .wrist_d405_protocol_v11 import (
    WRIST_BELIEF_TELEMETRY_CHANNEL,
    now_us,
    pack_belief_telemetry,
)


LOG = logging.getLogger(__name__)


class WristBeliefGateway(ThorRemoteGateway):
    """Run the preserved ThorRemoteGateway belief methods on wrist-v11 results."""

    def __init__(
        self,
        *,
        wifi_bus: lcm.LCM,
        belief_config: dict[str, Any],
        shared_hand_selection_endpoint: str = DEFAULT_SHARED_HAND_SELECTION_ENDPOINT,
        shared_hand_selection_topic: str = DEFAULT_SHARED_HAND_SELECTION_TOPIC,
        belief_use_operator_command: bool = True,
        belief_command_divergence_max_m: float = 0.20,
        belief_command_max_delta_ms: float = 25.0,
        ee_target_enabled: bool = True,
        ee_target_endpoint: str = DEFAULT_EE_TARGET_ENDPOINT,
        ee_target_topic: str = DEFAULT_EE_TARGET_TOPIC,
        ee_target_link: str = "left_wrist_yaw_link",
        ee_target_label: str = "",
        ee_target_min_score: float = 0.0,
        ee_target_position_field: str = "auto",
        ee_target_duration_sec: float = 2.0,
        ee_target_final_hold_sec: float = 1.0,
        ee_target_quaternion_wxyz: tuple[float, float, float, float] = (
            1.0,
            0.0,
            0.0,
            0.0,
        ),
    ) -> None:
        # Calling ThorRemoteGateway.__init__ would subscribe the legacy head-camera
        # path. These are the exact fields consumed by its belief methods,
        # initialized with the same expressions/defaults.
        self.wifi_bus = wifi_bus
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.belief_enabled = True
        self.parallel_tasc_belief = ParallelTascBelief(belief_config)
        self.tasc_belief = self.parallel_tasc_belief.trackers["left"]
        self.belief_update_hz = max(0.1, float(belief_config.get("belief_update_hz", 30.0)))
        self.geometry_update_hz = max(0.1, float(belief_config.get("geometry_update_hz", 8.0)))
        self.belief_hand_model = str(belief_config.get("hand_model", "dex1")).strip().lower()
        self.belief_hand_profile = hand_frame_profile(self.belief_hand_model)
        self.belief_grasp_center_local_m = np.asarray(
            belief_config.get("grasp_center_local_m", self.belief_hand_profile["grasp_center_local"]),
            dtype=np.float64,
        ).reshape(3)
        self.belief_orientation_min_confidence = float(
            belief_config.get("orientation_min_confidence", 0.35)
        )
        self.belief_telemetry_period_sec = 1.0 / max(
            15.0, float(belief_config.get("belief_telemetry_hz", 15.0))
        )
        self.last_belief_telemetry_monotonic = 0.0
        self.last_belief_publish_log_monotonic = 0.0
        self.belief_telemetry_max_objects = max(1, int(belief_config.get("max_objects", 10)))
        self.belief_telemetry_max_contacts = max(1, int(belief_config.get("max_contacts_per_object", 6)))
        self.belief_telemetry_max_orientations = max(1, int(belief_config.get("max_orientation_candidates", 96)))
        self.belief_last_eef_by_hand: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray | None, int]
        ] = {}

        self.shared_hand_selection_endpoint = str(shared_hand_selection_endpoint)
        self.shared_hand_selection_topic = str(shared_hand_selection_topic)
        self.shared_hand_selection_socket = zmq.Context.instance().socket(zmq.SUB)
        self.shared_hand_selection_socket.setsockopt(zmq.LINGER, 0)
        self.shared_hand_selection_socket.setsockopt(zmq.RCVHWM, 100)
        self.shared_hand_selection_socket.setsockopt_string(
            zmq.SUBSCRIBE, self.shared_hand_selection_topic
        )
        self.shared_hand_selection_socket.connect(self.shared_hand_selection_endpoint)
        self.belief_selected_hand = "none"
        self.belief_selection_active = False
        self.belief_selection_timestamp_us = 0
        self.latest_hand_selection: dict[str, Any] = {}
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

        self.latest_robot_state: tuple[int | None, bytes] | None = None
        self.latest_belief_result: dict[str, Any] | None = None
        self.latest_belief_geometry: Any | None = None
        self.latest_belief_geometry_sequence = -1
        self._zero_primitives_warned = False
        self.result_max_age_ms = 800.0
        # Keep the exact legacy head-gateway output contract. The PATH state
        # machine consumes this topic, applies the common position/orientation
        # reference dynamics, and owns the subsequent IPOPT solve.
        self.ee_target_enabled = bool(ee_target_enabled)
        self.ee_target_label = str(ee_target_label)
        self.ee_target_min_score = float(ee_target_min_score)
        self.ee_target_position_field = str(ee_target_position_field)
        self.ee_target_link = str(ee_target_link)
        self.ee_target_endpoint = str(ee_target_endpoint)
        self.ee_target_topic = str(ee_target_topic)
        self.ee_target_pub = (
            ZmqEeTargetPublisher(
                endpoint=self.ee_target_endpoint,
                topic=self.ee_target_topic,
            )
            if self.ee_target_enabled
            else None
        )
        self.ee_target_duration_sec = float(ee_target_duration_sec)
        self.ee_target_final_hold_sec = float(ee_target_final_hold_sec)
        self.ee_target_quaternion_wxyz = tuple(
            float(value) for value in ee_target_quaternion_wxyz
        )
        self.belief_csv = None
        self.belief_log_period = 1.0
        self.last_belief_log_monotonic = 0.0
        self.stats: dict[str, Any] = {
            "decode_errors": 0,
            "last_error": None,
            "belief_enabled": True,
            "belief_targets": 0,
            "ee_targets_published": 0,
            "ee_targets_skipped": 0,
            "last_ee_target": None,
            "last_ee_target_skip": None,
            "last_belief": {
                "enabled": self.parallel_tasc_belief.config.enabled,
                "hands": {},
            },
            "belief_selected_hand": "none",
            "belief_selection_active": False,
            "belief_selection_messages": 0,
            "belief_telemetry_published": 0,
            "belief_updates": 0,
            "objects_with_primitives": 0,
            "objects_total": 0,
            "belief_command_divergence_m": {},
            "command_lookup_age_ms": {},
            "command_lookup_failures": 0,
            "belief_input": {
                "source": {},
                "divergence_m": {},
                "command_lookup_age_ms": {},
                "command_lookup_failures": 0,
            },
        }

    def start(self) -> None:
        self.running = True
        # This is ThorRemoteGateway._belief_loop itself, not a reimplementation.
        self.thread = threading.Thread(
            target=self._belief_loop,
            name="wrist-legacy-tasc-belief",
            daemon=True,
        )
        self.thread.start()
        LOG.info(
            "legacy Thor belief loop started update_hz=%.1f geometry_hz=%.1f input=%s",
            self.belief_update_hz,
            self.geometry_update_hz,
            "operator_command" if self.belief_use_operator_command else "robot_fk",
        )
        if self.ee_target_pub is not None:
            LOG.info(
                "legacy IPOPT target publisher enabled endpoint=%s topic=%s "
                "selected-hand-link=A:left/B:right",
                self.ee_target_endpoint,
                self.ee_target_topic,
            )
        else:
            LOG.info("legacy IPOPT target publisher disabled")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.ee_target_pub is not None:
            self.ee_target_pub.close()
        self.shared_hand_selection_socket.close(0)

    def update_robot_state(self, data: bytes) -> None:
        state = decode_robot_state(data)
        with self.lock:
            self.latest_robot_state = (int(state.timestamp_us), bytes(data))

    def update_result(self, result: dict[str, Any]) -> None:
        """Adapt v11 timestamps/metadata, then feed the exact legacy cache."""
        payload = dict(result)
        source_us = int(payload.get("source_bundle_timestamp_us", 0) or 0)
        payload.setdefault("source_rgb_timestamp_us", source_us)
        payload.setdefault("source_depth_timestamp_us", source_us)
        payload.setdefault("backend", "sam3_wrist_d405_v11")
        payload.setdefault("geometry_status", {})
        objects = [
            obj for obj in payload.get("objects", []) if isinstance(obj, dict)
        ]
        # Wrist geometry attaches gravity to each object because it is produced
        # beside the object cloud. The inherited belief loop reads the frame
        # value from the result top level; hoist the first finite producer so
        # its persistent __gravity__ orientation candidates are available.
        for obj in objects:
            up_goal = obj.get("gravity_up_goal")
            if up_goal is not None:
                try:
                    candidate = np.asarray(up_goal, dtype=np.float64).reshape(3)
                    if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 1.0e-9:
                        payload.setdefault("gravity_up_goal", candidate.tolist())
                        break
                except (TypeError, ValueError):
                    continue
        if objects:
            first = objects[0]
            payload.setdefault("image_width", first.get("mask_width"))
            payload.setdefault("image_height", first.get("mask_height"))
            try:
                transform = np.asarray(
                    first.get("t_goal_cam"), dtype=np.float64
                ).reshape(4, 4)
                payload.setdefault(
                    "camera_goal_origin_m", transform[:3, 3].tolist()
                )
                payload.setdefault(
                    "camera_goal_forward_m",
                    (
                        transform[:3, :3]
                        @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    ).tolist(),
                )
            except (TypeError, ValueError):
                pass
        with self.lock:
            self.latest_belief_result = payload
            self.latest_belief_geometry = None
            self.latest_belief_geometry_sequence = -1

    def _publish_belief_telemetry(
        self,
        payload: dict[str, Any],
        per_hand: dict[str, Any],
    ) -> None:
        """Use legacy telemetry semantics on the wrist-specific channel."""
        now_monotonic = time.monotonic()
        if (
            now_monotonic - self.last_belief_telemetry_monotonic
            < self.belief_telemetry_period_sec
        ):
            return
        self.last_belief_telemetry_monotonic = now_monotonic
        published_timestamp_us = now_us()
        control = dict(self.latest_hand_selection or {})
        try:
            control_timestamp_us = int(control.get("timestamp_us", 0) or 0)
        except (TypeError, ValueError):
            control_timestamp_us = 0
        objects_by_track = {
            str(obj.get("track_id", "")): obj
            for obj in payload.get("objects", [])
            if isinstance(obj, dict)
        }
        hands: dict[str, Any] = {}
        target_presence: dict[str, bool] = {}
        for side in ("left", "right"):
            hand = self._bounded_hand_telemetry(per_hand.get(side, {}))
            target = hand.get("target")
            if isinstance(target, dict):
                source = objects_by_track.get(str(target.get("object_id", "")))
                if source is not None:
                    target.update(
                        camera_id=source.get("camera_id"),
                        box_xyxy=source.get("box_xyxy"),
                        t_goal_cam=source.get("t_goal_cam"),
                        intrinsics=source.get("intrinsics"),
                    )
            hands[side] = hand
            target_presence[side] = isinstance(hand.get("target"), dict)
        control_age_ms = (
            max(0.0, (published_timestamp_us - control_timestamp_us) / 1000.0)
            if control_timestamp_us > 0
            else None
        )
        packet = {
            "version": 1,
            "published_timestamp_us": published_timestamp_us,
            "source_bundle_timestamp_us": int(
                payload.get("source_bundle_timestamp_us", 0) or 0
            ),
            "source_rgb_timestamp_us": int(
                payload.get("source_rgb_timestamp_us", 0) or 0
            ),
            "result_sequence": int(payload.get("sequence", -1)),
            "selected_hand": self.belief_selected_hand,
            "hands": hands,
            "result_valid": bool(payload.get("valid", False)),
            "geometry_status": payload.get("geometry_status", {}),
            "result_age_ms": max(
                0.0,
                (
                    published_timestamp_us
                    - int(payload.get("published_timestamp_us", 0) or 0)
                )
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
                "hands": (
                    control.get("hands", {})
                    if isinstance(control.get("hands"), dict)
                    else {}
                ),
                "age_ms": control_age_ms,
            },
            "belief_input": dict(self.stats.get("belief_input", {}) or {}),
        }
        encoded = pack_belief_telemetry(_msgpack_safe(packet))
        self.wifi_bus.publish(WRIST_BELIEF_TELEMETRY_CHANNEL, encoded)
        with self.lock:
            self.stats["belief_telemetry_published"] += 1
        if (
            now_monotonic - self.last_belief_publish_log_monotonic
            >= 5.0
        ):
            LOG.info(
                "BELIEF_TELEMETRY sequence=%s targets=%s objects=%d bytes=%d channel=%s",
                payload.get("sequence"),
                target_presence,
                len(objects_by_track),
                len(encoded),
                WRIST_BELIEF_TELEMETRY_CHANNEL,
            )
            self.last_belief_publish_log_monotonic = now_monotonic
