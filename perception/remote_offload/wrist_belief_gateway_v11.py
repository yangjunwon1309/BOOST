"""Thor-side, two-hand TASC belief for v11 wrist D405 remote geometry.

The SAM3 laptop owns segmentation and aligned RGB-D surface reconstruction.
Thor owns the unchanged :class:`ParallelTascBelief` state: it evaluates both
hands against the latest remote geometry with the newest local robot FK, then
publishes visualization telemetry back to the laptop.  It never publishes an
autonomous end-effector target; the CAD wrist-to-camera transform remains
provisional.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Any

import lcm
import numpy as np

from seg_web_node.grip_candidates import GripCandidate, SurfacePrimitive
from seg_web_node.lcm_protocol import decode_robot_state
from seg_web_node.robot_geometry import (
    angular_velocity_from_rotations,
    g1_grasp_center_position,
    g1_hand_base_rotation,
    gravity_up_in_pelvis,
    hand_frame_profile,
)
from seg_web_node.tasc_belief import BeliefInputObject, HandInput, ParallelTascBelief

from .wrist_d405_protocol_v11 import (
    WRIST_BELIEF_TELEMETRY_CHANNEL,
    now_us,
    pack_belief_telemetry,
)


LOG = logging.getLogger(__name__)


def _vec3(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    return vector if np.isfinite(vector).all() else None


def _wire_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _wire_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _wire_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_safe(item) for item in value]
    return str(value)


def _candidate(value: Any) -> GripCandidate | None:
    if not isinstance(value, dict):
        return None
    center, approach = _vec3(value.get("center_goal_m")), _vec3(value.get("approach_goal"))
    if center is None or approach is None:
        return None
    try:
        return GripCandidate(
            center=center, approach=approach, width=float(value.get("width_m", 0.0)),
            geom_score=float(value.get("geom_score", 0.0)), source=str(value.get("source", "remote")),
            surface_normal=_vec3(value.get("surface_normal")),
            surface_tangent=_vec3(value.get("surface_tangent")),
            surface_axis=_vec3(value.get("surface_axis")),
        )
    except (TypeError, ValueError):
        return None


def _primitive(value: Any) -> SurfacePrimitive | None:
    if not isinstance(value, dict) or value.get("kind") not in {"cylinder", "plane", "circle", "pca", "point"}:
        return None
    origin, axis = _vec3(value.get("origin_goal_m")), _vec3(value.get("axis_goal"))
    if origin is None or axis is None or np.linalg.norm(axis) <= 1.0e-9:
        return None
    try:
        return SurfacePrimitive(
            kind=str(value["kind"]), origin=origin, axis=axis / np.linalg.norm(axis),
            fit_quality=float(np.clip(float(value.get("fit_quality", 0.0)), 0.0, 1.0)),
            source=str(value.get("source", "remote")), radius=float(value.get("radius", 0.0)),
            half_height=float(value.get("half_height", 0.0)), extent=_vec3(value.get("extent")),
            snapped_to_gravity=bool(value.get("snapped_to_gravity", False)),
            observation_timestamp_us=int(value.get("observation_timestamp_us", 0) or 0),
            observation_track_id=str(value.get("observation_track_id", "")),
            observation_state=str(value.get("observation_state", "active")),
        )
    except (TypeError, ValueError):
        return None


class WristBeliefGateway:
    """Runs the legacy ParallelTascBelief algorithm for both wrists on Thor."""

    def __init__(self, *, wifi_bus: lcm.LCM, belief_config: dict[str, Any]) -> None:
        self.wifi_bus = wifi_bus
        self.parallel = ParallelTascBelief(belief_config)
        self.profile = hand_frame_profile(str(belief_config.get("hand_model", "dex1")).lower())
        self.hand_model = str(belief_config.get("hand_model", "dex1")).lower()
        self.grasp_center_local_m = np.asarray(
            belief_config.get("grasp_center_local_m", self.profile["grasp_center_local"]), dtype=np.float64
        ).reshape(3)
        self.belief_hz = max(0.1, float(belief_config.get("belief_update_hz", 30.0)))
        self.geometry_hz = max(0.1, float(belief_config.get("geometry_update_hz", 8.0)))
        self.telemetry_hz = max(1.0, float(belief_config.get("belief_telemetry_hz", 15.0)))
        self.lock = threading.Lock()
        self.latest_result: dict[str, Any] | None = None
        self.latest_robot_packet: bytes | None = None
        self.latest_geometry: Any | None = None
        self.latest_geometry_sequence = -1
        self.last_eef: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, int]] = {}
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_publish_log_monotonic = 0.0

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="wrist-tasc-belief", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def update_robot_state(self, data: bytes) -> None:
        with self.lock:
            self.latest_robot_packet = bytes(data)

    def update_result(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.latest_result = dict(result)

    def _objects(self, result: dict[str, Any]) -> list[BeliefInputObject]:
        output: list[BeliefInputObject] = []
        for index, obj in enumerate(result.get("objects", [])):
            if not isinstance(obj, dict):
                continue
            center = _vec3(obj.get("center_goal_m"))
            if center is None:
                continue
            candidates = [item for item in (_candidate(value) for value in obj.get("contact_candidates", [])) if item is not None]
            primitives = [item for item in (_primitive(value) for value in obj.get("surface_primitives", [])) if item is not None]
            output.append(BeliefInputObject(
                object_id=str(obj.get("track_id") or f"{obj.get('label', 'object')}:{index}"),
                label=str(obj.get("label", "object")), score=float(obj.get("score", 0.0)), center_goal_m=center,
                grip_candidates=candidates or None, surface_primitives=primitives or None,
            ))
        return output

    def _hand_inputs(self, packet: bytes) -> tuple[dict[str, HandInput], Any]:
        state = decode_robot_state(packet)
        output: dict[str, HandInput] = {}
        for side in ("left", "right"):
            position = g1_grasp_center_position(state.q, f"{side}_wrist_yaw_link", self.grasp_center_local_m, self.hand_model)
            rotation = g1_hand_base_rotation(state.q, f"{side}_wrist_yaw_link", self.hand_model)
            velocity = omega = None
            previous = self.last_eef.get(side)
            if previous is not None:
                old_position, old_rotation, old_omega, old_stamp = previous
                dt = (int(state.timestamp_us) - old_stamp) / 1.0e6
                if 1.0e-4 <= dt <= 0.5:
                    velocity = (position - old_position) / dt
                    raw_omega = angular_velocity_from_rotations(old_rotation, rotation, dt)
                    omega = raw_omega if old_omega is None or raw_omega is None else 0.25 * raw_omega + 0.75 * old_omega
            self.last_eef[side] = (position.copy(), rotation.copy(), omega, int(state.timestamp_us))
            output[side] = HandInput(
                eef_position=position, user_translation=velocity, hand_rotation=rotation, hand_axis=None,
                approach_axis=rotation @ np.asarray(self.profile["approach_axis_local"], dtype=np.float64),
                angular_velocity=omega, button_pressed=False, input_source="robot_fk",
            )
        return output, state

    def _publish(self, result: dict[str, Any], per_hand: dict[str, dict[str, Any]], state: Any) -> None:
        hands: dict[str, dict[str, Any]] = {}
        objects_by_track = {
            str(obj.get("track_id", "")): obj
            for obj in result.get("objects", [])
            if isinstance(obj, dict)
        }
        for side in ("left", "right"):
            hand = dict(per_hand.get(side, {}))
            hand["armed"] = False
            target = hand.get("target")
            if isinstance(target, dict):
                source = objects_by_track.get(str(target.get("object_id", "")))
                if source is not None:
                    # Keep all projection data in the telemetry packet.  The
                    # viewer intentionally renders this latest target even
                    # after its image/result has advanced to another frame.
                    hand["target"] = {
                        **target,
                        "camera_id": source.get("camera_id"),
                        "box_xyxy": source.get("box_xyxy"),
                        "t_goal_cam": source.get("t_goal_cam"),
                        "intrinsics": source.get("intrinsics"),
                    }
            hands[side] = hand
        targets = {
            side: bool((hands.get(side) or {}).get("target"))
            for side in ("left", "right")
        }
        self.wifi_bus.publish(
            WRIST_BELIEF_TELEMETRY_CHANNEL,
            pack_belief_telemetry(_wire_safe({
                "version": 1, "published_timestamp_us": now_us(),
                "source_bundle_timestamp_us": int(result.get("source_bundle_timestamp_us", 0)),
                "result_sequence": int(result.get("sequence", -1)), "selected_hand": "none",
                "hands": hands, "result_valid": bool(result.get("valid", False)),
                "belief_robot_state_timestamp_us": int(state.timestamp_us),
                "belief_hand_model": self.hand_model,
                "belief_grasp_center_local_m": self.grasp_center_local_m.tolist(),
            })),
        )
        now = time.monotonic()
        if now - self.last_publish_log_monotonic >= 5.0:
            LOG.info("BELIEF_TELEMETRY result_sequence=%s targets=%s channel=%s", result.get("sequence"), targets, WRIST_BELIEF_TELEMETRY_CHANNEL)
            self.last_publish_log_monotonic = now

    def _loop(self) -> None:
        belief_period, geometry_period, telemetry_period = 1.0 / self.belief_hz, 1.0 / self.geometry_hz, 1.0 / self.telemetry_hz
        next_belief = next_geometry = next_telemetry = 0.0
        while self.running:
            now = time.monotonic()
            if now < next_belief:
                time.sleep(min(0.005, next_belief - now))
                continue
            next_belief = now + belief_period
            with self.lock:
                result, packet = self.latest_result, self.latest_robot_packet
            if result is None or packet is None:
                continue
            try:
                hand_inputs, state = self._hand_inputs(packet)
                objects = self._objects(result)
                sequence = int(result.get("sequence", -1))
                if self.latest_geometry is None or sequence != self.latest_geometry_sequence or now >= next_geometry:
                    self.latest_geometry = self.parallel.build_shared_geometry(
                        objects, up_goal=gravity_up_in_pelvis(state.base_quat_wxyz),
                        base_quat_wxyz=state.base_quat_wxyz,
                        stamp_us=int(result.get("source_bundle_timestamp_us", 0)),
                    )
                    self.latest_geometry_sequence = sequence
                    next_geometry = now + geometry_period
                per_hand = self.parallel.update_hands(self.latest_geometry, hand_inputs)
                if now >= next_telemetry:
                    self._publish(result, per_hand, state)
                    next_telemetry = now + telemetry_period
            except Exception:
                # Capture must continue if a malformed remote result is received.
                continue
