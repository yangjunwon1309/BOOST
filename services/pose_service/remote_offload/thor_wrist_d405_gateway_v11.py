"""Thor-owned direct D405 capture gateway for wrist-only v11 perception."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import lcm
import numpy as np

from seg_web_node.lcm_protocol import RobotState, decode_robot_state
from seg_web_node.robot_geometry import g1_link_transforms

from .wrist_d405_capture_v11 import WristD405Capture, discover_d405_serials, load_camera_specs
from .wrist_belief_gateway_v11 import WristBeliefGateway
from .thor_gateway_head_d435i_legacy import (
    DEFAULT_EE_TARGET_ENDPOINT,
    DEFAULT_EE_TARGET_TOPIC,
)
from .wrist_d405_protocol_v11 import (
    CAMERA_IDS,
    WRIST_BUNDLE_CHANNEL,
    WRIST_RESULT_CHANNEL,
    WRIST_ROBOT_STATE_CHANNEL,
    now_us,
    pack_depth_z16,
    pack_bundle,
    unpack_result,
)


LOG = logging.getLogger(__name__)


_D405_TRANSPORT_ERROR_MARKERS = (
    "device disconnected",
    "no device connected",
    "wait_for_frames cannot be called before start()",
    "frame didn't arrive within",
    "capture has not started",
)


def is_d405_transport_error(error: BaseException) -> bool:
    """Whether retrying the RealSense pipeline can recover this failure.

    Geometry and state-pairing errors deliberately do not match here: restarting
    a camera cannot correct those and would hide useful diagnostics.
    """

    message = str(error).lower()
    return any(marker in message for marker in _D405_TRANSPORT_ERROR_MARKERS)


def monotonic_us() -> int:
    """Thor-local clock shared by D405 arrivals and G1 LCM arrivals."""
    return time.monotonic_ns() // 1_000


@dataclasses.dataclass(frozen=True)
class RobotStateSample:
    received_us: int
    wall_received_us: int
    packet: bytes
    state: RobotState


class RobotStateHistory:
    """Bounded Thor-arrival history used to sample state at camera exposure.

    The robot packet timestamp belongs to the G1 clock and is deliberately not
    used here.  Both arrival stamps are made on Thor's monotonic clock, so a
    constant lookup bias is the only clock/transport calibration required.
    """

    def __init__(self, *, span_s: float = 2.0, publish_hz: float = 500.0) -> None:
        capacity = max(2, int(np.ceil(float(span_s) * float(publish_hz))))
        self.entries: collections.deque[RobotStateSample] = collections.deque(maxlen=capacity)

    def add(
        self, received_us: int, packet: bytes, state: RobotState, *, wall_received_us: int | None = None
    ) -> None:
        self.entries.append(
            RobotStateSample(
                int(received_us),
                int(received_us if wall_received_us is None else wall_received_us),
                bytes(packet),
                state,
            )
        )

    def snapshot(self) -> list[RobotStateSample]:
        return list(self.entries)


def nearest_state_sample(
    samples: list[RobotStateSample], lookup_us: int
) -> tuple[int, RobotStateSample, float] | None:
    """Return nearest arrival-clock sample without extrapolating."""
    if not samples:
        return None
    index = min(range(len(samples)), key=lambda i: abs(samples[i].received_us - int(lookup_us)))
    sample = samples[index]
    return index, sample, abs(sample.received_us - int(lookup_us)) / 1000.0


def _rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def wrist_motion_from_history(
    samples: list[RobotStateSample], sample_index: int, wrist_link: str
) -> tuple[float, float] | None:
    """Finite-difference wrist linear/angular speed around a matched state."""
    if len(samples) < 2:
        return None
    candidate_indices = [index for index in (sample_index - 1, sample_index + 1) if 0 <= index < len(samples)]
    if not candidate_indices:
        return None
    other_index = min(
        candidate_indices,
        key=lambda index: abs(samples[index].received_us - samples[sample_index].received_us),
    )
    first, second = sorted((samples[sample_index], samples[other_index]), key=lambda item: item.received_us)
    dt_s = (second.received_us - first.received_us) / 1_000_000.0
    if dt_s <= 0.0:
        return None
    first_wrist = g1_link_transforms(np.asarray(first.state.q, dtype=np.float64))[wrist_link]
    second_wrist = g1_link_transforms(np.asarray(second.state.q, dtype=np.float64))[wrist_link]
    linear_mps = float(np.linalg.norm(second_wrist[:3, 3] - first_wrist[:3, 3]) / dt_s)
    angular_rps = _rotation_angle_rad(first_wrist[:3, :3].T @ second_wrist[:3, :3]) / dt_s
    return linear_mps, float(angular_rps)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load the wrist D405 configuration") from error
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"D405 config must be a YAML mapping: {path}")
    return data


def _import_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError("pyrealsense2 is required inside the Thor D405 capture runtime") from error
    return rs


def diagnose_runtime(rs: Any) -> None:
    """Report the container's view of exactly the devices given to Docker."""
    LOG.info("pyrealsense2 version=%s", getattr(rs, "__version__", "unknown"))
    for variable in ("D405_VIDEO_DEVICES", "D405_USB_DEVICES"):
        values = [item for item in os.environ.get(variable, "").split(",") if item]
        LOG.info(
            "%s=%s", variable,
            [(item, Path(item).exists(), os.access(item, os.R_OK | os.W_OK)) for item in values],
        )
    video_sysfs = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            video_sysfs.append((node.name, (node / "name").read_text(encoding="utf-8").strip()))
        except OSError:
            continue
    LOG.info("visible V4L2 sysfs nodes=%s", video_sysfs)
    LOG.info("librealsense devices=%s", [
        {"name": _safe_info(device, rs.camera_info.name), "serial": _safe_info(device, rs.camera_info.serial_number)}
        for device in rs.context().query_devices()
    ])


def _safe_info(device: Any, info: Any) -> str:
    return device.get_info(info) if device.supports(info) else ""


def assert_video_nodes_unowned() -> None:
    """Fail clearly before opening a camera; never terminate another service."""
    try:
        nodes = sorted(Path("/dev").glob("video*"))
        result = subprocess.run(["fuser", "-v", *map(str, nodes)], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return
    # The exact video-node assignment is machine dependent. Any active owner is
    # an operator-visible conflict; the D405 launcher must be the sole owner.
    owner_output = (result.stdout + result.stderr).strip()
    if result.returncode == 0 and owner_output:
        raise RuntimeError("a process already owns a V4L2 node; stop it manually before v11 capture:\n" + owner_output)


class ThorWristD405Gateway:
    def __init__(self, *, g1_lcm_url: str, wifi_lcm_url: str, config: Path, jpeg_quality: int, one_camera_diagnostic: bool,
                 robot_state_channel: str, belief_inference: bool, belief_config: Path,
                 belief_input_source: str, belief_command_max_delta_ms: float,
                 belief_command_divergence_max_m: float,
                 shared_hand_selection_endpoint: str, shared_hand_selection_topic: str,
                 ee_target_enabled: bool, ee_target_endpoint: str,
                 ee_target_topic: str, ee_target_link: str,
                  ee_target_label: str, ee_target_min_score: float,
                  ee_target_position_field: str, ee_target_duration_sec: float,
                  ee_target_final_hold_sec: float,
                 ee_target_quaternion_wxyz: tuple[float, float, float, float],
                 state_history_span_s: float = 2.0,
                 state_history_publish_hz: float = 500.0,
                 left_state_lookup_bias_ms: float = 0.0,
                 right_state_lookup_bias_ms: float = 0.0,
                 state_pairing_reference_range_m: float = 0.30,
                 camera_recovery_initial_delay_ms: float = 300.0,
                 camera_recovery_max_delay_ms: float = 2000.0) -> None:
        self.g1_bus = lcm.LCM(g1_lcm_url)
        self.wifi_bus = lcm.LCM(wifi_lcm_url)
        self.config = config
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.one_camera_diagnostic = bool(one_camera_diagnostic)
        self.robot_state_channel = str(robot_state_channel)
        self.sequence = 0
        self.last_publish_log_monotonic = 0.0
        self.last_skip_log_monotonic = 0.0
        self.stats: dict[str, Any] = {
            "skipped_frames": 0,
            "last_skip_reason": None,
            "robot_state_gap_ms": 0.0,
            "robot_state_gap_max_ms": 0.0,
            "pairing_residual_ms": {},
            "pairing_estimated_error_m": {},
            "camera_recoveries": 0,
            "camera_recovery_attempts": 0,
        }
        self.capture: WristD405Capture | None = None
        self.belief: WristBeliefGateway | None = None
        self.state_lock = threading.Lock()
        self.state_history = RobotStateHistory(
            span_s=state_history_span_s, publish_hz=state_history_publish_hz
        )
        self.latest_robot_state_packet: tuple[int, bytes] | None = None
        self.last_robot_state_received_us: int | None = None
        self.robot_state_arrival_interval_us: int | None = None
        self.robot_state_max_arrival_interval_us = 0
        self.state_lookup_bias_us = {
            "left_wrist": int(round(float(left_state_lookup_bias_ms) * 1_000.0)),
            "right_wrist": int(round(float(right_state_lookup_bias_ms) * 1_000.0)),
        }
        self.state_pairing_reference_range_m = max(0.0, float(state_pairing_reference_range_m))
        self.camera_recovery_initial_delay_s = max(
            0.05, float(camera_recovery_initial_delay_ms) / 1000.0
        )
        self.camera_recovery_max_delay_s = max(
            self.camera_recovery_initial_delay_s,
            float(camera_recovery_max_delay_ms) / 1000.0,
        )
        self._state_pump_stop = threading.Event()
        self._state_pump_thread: threading.Thread | None = None
        capture_config = _load_yaml(config).get("capture", {})
        self.max_inter_wrist_skew_us = int(float(capture_config.get("max_inter_wrist_skew_ms", 80.0)) * 1000.0)
        self.wifi_bus.subscribe(WRIST_RESULT_CHANNEL, self._on_result)
        self.g1_bus.subscribe(self.robot_state_channel, self._on_robot_state)
        if belief_inference:
            belief_data = _load_yaml(belief_config)
            belief_settings = belief_data.get("tasc_belief", belief_data.get("belief", {}))
            if not isinstance(belief_settings, dict):
                raise ValueError(f"belief config must contain a tasc_belief mapping: {belief_config}")
            self.belief = WristBeliefGateway(
                wifi_bus=self.wifi_bus,
                belief_config=belief_settings,
                shared_hand_selection_endpoint=shared_hand_selection_endpoint,
                shared_hand_selection_topic=shared_hand_selection_topic,
                belief_use_operator_command=(belief_input_source == "operator_command"),
                belief_command_max_delta_ms=belief_command_max_delta_ms,
                belief_command_divergence_max_m=belief_command_divergence_max_m,
                ee_target_enabled=ee_target_enabled,
                ee_target_endpoint=ee_target_endpoint,
                ee_target_topic=ee_target_topic,
                ee_target_link=ee_target_link,
                ee_target_label=ee_target_label,
                ee_target_min_score=ee_target_min_score,
                ee_target_position_field=ee_target_position_field,
                ee_target_duration_sec=ee_target_duration_sec,
                ee_target_final_hold_sec=ee_target_final_hold_sec,
                ee_target_quaternion_wxyz=ee_target_quaternion_wxyz,
            )

    def _on_robot_state(self, _channel: str, data: bytes) -> None:
        # Robot state is the only legacy LCM input retained in v11. It is
        # forwarded unchanged for remote timestamp matching; no head image,
        # depth, or head calibration channel is subscribed.
        try:
            state = decode_robot_state(data)
        except Exception as error:
            LOG.warning("rejected G1 robot state for D405 pairing: %s", error)
            return
        received_us = monotonic_us()
        wall_received_us = now_us()
        with self.state_lock:
            if self.last_robot_state_received_us is not None:
                self.robot_state_arrival_interval_us = max(
                    0, received_us - self.last_robot_state_received_us
                )
                self.robot_state_max_arrival_interval_us = max(
                    self.robot_state_max_arrival_interval_us,
                    self.robot_state_arrival_interval_us,
                )
                gap_ms = self.robot_state_arrival_interval_us / 1000.0
                self.stats["robot_state_gap_ms"] = gap_ms
                self.stats["robot_state_gap_max_ms"] = max(
                    float(self.stats.get("robot_state_gap_max_ms", 0.0)), gap_ms
                )
            self.last_robot_state_received_us = received_us
            self.latest_robot_state_packet = (received_us, bytes(data))
            self.state_history.add(
                received_us, bytes(data), state, wall_received_us=wall_received_us
            )
        self.wifi_bus.publish(WRIST_ROBOT_STATE_CHANNEL, data)
        if self.belief is not None:
            self.belief.update_robot_state(data)

    def _state_pump(self) -> None:
        """Keep receiving q while D405 waits for its next frame."""
        while not self._state_pump_stop.is_set():
            try:
                self.g1_bus.handle_timeout(25)
            except Exception as error:
                if not self._state_pump_stop.is_set():
                    LOG.warning("G1 robot-state receive loop stopped: %s", error)
                return

    def _start_state_pump(self) -> None:
        if self._state_pump_thread is not None:
            return
        self._state_pump_stop.clear()
        self._state_pump_thread = threading.Thread(
            target=self._state_pump, name="wrist-d405-g1-state", daemon=True
        )
        self._state_pump_thread.start()

    def _stop_state_pump(self) -> None:
        self._state_pump_stop.set()
        thread, self._state_pump_thread = self._state_pump_thread, None
        if thread is not None:
            thread.join(timeout=0.2)

    def preflight(self) -> None:
        rs = _import_realsense()
        specs = load_camera_specs(self.config)
        capture_config = _load_yaml(self.config).get("capture", {})
        visible = discover_d405_serials(rs)
        missing = [spec.serial for spec in specs.values() if spec.serial not in visible]
        unexpected = set(visible) - {spec.serial for spec in specs.values()}
        if missing or unexpected or len(visible) != 2:
            raise RuntimeError(f"D405 preflight requires exactly configured pair; missing={missing}, unexpected={sorted(unexpected)}")
        for side, spec in specs.items():
            usb_type = str(visible[spec.serial].get("usb_type", ""))
            try:
                usb_major = int(usb_type.split(".", 1)[0])
            except ValueError:
                usb_major = 0
            reduced_usb2_profile = (
                spec.color.width <= 480 and spec.color.height <= 270 and spec.color.fps <= 15
                and spec.depth.width <= 480 and spec.depth.height <= 270 and spec.depth.fps <= 15
            )
            if usb_major < 3 and not (
                bool(capture_config.get("allow_usb2_reduced_profile", False)) and reduced_usb2_profile
            ):
                raise RuntimeError(
                    f"{side} serial={spec.serial} is negotiated at USB {usb_type or 'unknown'}; "
                    "v11 requires a SuperSpeed (USB 3.x) path for two 640x360@30 UYVY+Z16 D405 streams"
                )
            if usb_major < 3:
                LOG.warning(
                    "%s uses approved reduced USB-2 profile UYVY/Z16 %dx%d@%d",
                    side, spec.color.width, spec.color.height, spec.color.fps,
                )
            LOG.info("%s serial=%s port=%s profile=UYVY/Z16 %dx%d@%d", side, spec.serial,
                     visible[spec.serial].get("physical_port", ""), spec.color.width, spec.color.height, spec.color.fps)

    def _start_capture(self) -> None:
        assert_video_nodes_unowned()
        rs = _import_realsense()
        specs = load_camera_specs(self.config)
        capture_config = _load_yaml(self.config).get("capture", {})
        capture = WristD405Capture(
            rs, specs,
            max_rgb_depth_delta_ms=float(capture_config.get("max_rgb_depth_delta_ms", 20.0)),
        )
        try:
            capture.start()
        except Exception:
            capture.stop()
            raise
        self.capture = capture

    def _stop_capture(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

    def start(self, *, state_pairing: bool = True) -> None:
        self._start_capture()
        if state_pairing:
            self._start_state_pump()
        if self.belief is not None:
            self.belief.start()
        LOG.info("direct D405 capture started for %s", sorted(self.capture.specs))

    def stop(self) -> None:
        self._stop_state_pump()
        if self.belief is not None:
            self.belief.stop()
        self._stop_capture()

    def _recover_capture(self, error: RuntimeError) -> None:
        """Recreate both RealSense pipelines after a USB/UVC transport fault.

        The first reset starts after 300 ms by default. If the USB device is
        still re-enumerating, retries back off only to a short bounded delay;
        robot-state reception and belief telemetry remain alive meanwhile.
        """

        reason = str(error)
        started = time.monotonic()
        delay_s = self.camera_recovery_initial_delay_s
        attempt = 0
        self._stop_capture()
        while True:
            attempt += 1
            self.stats["camera_recovery_attempts"] = int(
                self.stats.get("camera_recovery_attempts", 0)
            ) + 1
            LOG.warning(
                "D405 transport fault; resetting both capture pipelines in %.0f ms "
                "(attempt=%d): %s",
                delay_s * 1000.0, attempt, reason,
            )
            time.sleep(delay_s)
            try:
                self._start_capture()
            except Exception as recovery_error:
                reason = str(recovery_error)
                self._stop_capture()
                delay_s = min(self.camera_recovery_max_delay_s, delay_s * 2.0)
                continue
            self.stats["camera_recoveries"] = int(
                self.stats.get("camera_recoveries", 0)
            ) + 1
            self.stats["last_camera_recovery_ms"] = round(
                (time.monotonic() - started) * 1000.0, 1
            )
            LOG.info(
                "D405 capture recovered after %.1f ms (%d reset attempt%s)",
                self.stats["last_camera_recovery_ms"], attempt,
                "" if attempt == 1 else "s",
            )
            return

    def _on_result(self, _channel: str, data: bytes) -> None:
        try:
            result = unpack_result(data)
            objects = [obj for obj in result.get("objects", []) if isinstance(obj, dict)]
            valid_centers = sum(obj.get("center_goal_m") is not None for obj in objects)
            primitives = sum(bool(obj.get("surface_primitives")) for obj in objects)
            geometry_status = dict(result.get("geometry_status") or {})
            LOG.info(
                "v11 result sequence=%s objects=%s center_goal=%s primitives=%s "
                "seam_discarded=%s geometry_status=%s",
                result.get("sequence"), len(objects), valid_centers, primitives,
                result.get("discarded_seam_masks"), geometry_status,
            )
            if self.belief is not None:
                self.belief.update_result(result)
        except Exception as error:
            LOG.warning("rejected v11 worker result: %s", error)

    def publish_once(self) -> None:
        if self.capture is None:
            raise RuntimeError("D405 gateway is not started")
        with self.state_lock:
            have_robot_state = bool(self.state_history.entries)
        if not have_robot_state:
            raise RuntimeError(
                f"no robot state received on {self.robot_state_channel}; "
                "refusing RGB-D without the FK snapshot required for belief geometry"
            )
        captured = self.capture.wait_for_bundles()
        # Take this snapshot *after* the blocking camera waits. The independent
        # G1 receive thread has continued filling it during that interval.
        with self.state_lock:
            history = self.state_history.snapshot()
        if not history:
            raise RuntimeError("robot-state history became empty during D405 capture")
        matches: dict[str, tuple[int, RobotStateSample, float]] = {}
        motions: dict[str, tuple[float, float] | None] = {}
        for camera_id in CAMERA_IDS:
            frame_arrival_us = int(captured[camera_id].host_timestamp_us)
            lookup_us = frame_arrival_us + self.state_lookup_bias_us[camera_id]
            matched = nearest_state_sample(history, lookup_us)
            if matched is None:
                raise RuntimeError("robot-state history became empty during D405 capture")
            matches[camera_id] = matched
            motions[camera_id] = wrist_motion_from_history(
                history, matched[0],
                "left_wrist_yaw_link" if camera_id == "left_wrist" else "right_wrist_yaw_link",
            )
        frames: dict[str, dict[str, Any]] = {}
        for camera_id in CAMERA_IDS:
            bundle = captured[camera_id]
            _sample_index, sample, residual_ms = matches[camera_id]
            wrist_motion = motions[camera_id]
            wrist_speed_mps = None if wrist_motion is None else wrist_motion[0]
            wrist_omega_rps = None if wrist_motion is None else wrist_motion[1]
            estimated_error_m = (
                None
                if wrist_motion is None
                else (wrist_speed_mps + self.state_pairing_reference_range_m * wrist_omega_rps)
                * (residual_ms / 1000.0)
            )
            import cv2
            ok, encoded = cv2.imencode(".jpg", bundle.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError(f"JPEG encode failed for {camera_id}")
            frames[camera_id] = {
                "camera_id": camera_id, "serial": bundle.serial,
                # Wall time matches the forwarded G1 robot-state clock; retain
                # the monotonic capture stamp independently for diagnostics.
                "host_timestamp_us": now_us(),
                "host_monotonic_timestamp_us": int(bundle.host_timestamp_us),
                "color_timestamp_us": int(round(bundle.color_timestamp_ms * 1000.0)),
                "depth_timestamp_us": int(round(bundle.depth_timestamp_ms * 1000.0)),
                "width": int(bundle.intrinsics["width"]), "height": int(bundle.intrinsics["height"]),
                "color_jpeg": encoded.tobytes(), "depth_z16_zlib": pack_depth_z16(bundle.depth_z16),
                "depth_scale_m": float(bundle.depth_scale_m), "intrinsics": bundle.intrinsics,
                "calibration_id": "d405_factory_2026-08-20",
                "diagnostics": {
                    "rgb_depth_delta_ms": abs(bundle.color_timestamp_ms - bundle.depth_timestamp_ms),
                    "color_timestamp_domain": bundle.color_timestamp_domain,
                    "depth_timestamp_domain": bundle.depth_timestamp_domain,
                },
                # These values are selected on Thor, where camera and robot
                # arrivals share a clock.  The worker must not re-pair them.
                "q_at_exposure": np.asarray(sample.state.q, dtype=np.float64).tolist(),
                "base_quat_at_exposure_wxyz": np.asarray(
                    sample.state.base_quat_wxyz, dtype=np.float64
                ).tolist(),
                "state_pairing": {
                    "lookup_us": int(bundle.host_timestamp_us) + self.state_lookup_bias_us[camera_id],
                    "state_arrival_us": int(sample.received_us),
                    "residual_ms": float(residual_ms),
                    "bias_us": int(self.state_lookup_bias_us[camera_id]),
                    "method": "nearest",
                    "wrist_speed_mps": wrist_speed_mps,
                    "wrist_omega_rps": wrist_omega_rps,
                    "estimated_error_m": estimated_error_m,
                },
            }
        # Use the capture timestamps, not sequential JPEG/packet construction,
        # to measure actual inter-camera timing after concurrent waits.
        skew = abs(
            int(captured["left_wrist"].host_timestamp_us)
            - int(captured["right_wrist"].host_timestamp_us)
        )
        if skew > self.max_inter_wrist_skew_us:
            raise RuntimeError(
                f"inter-wrist capture skew {skew / 1000.0:.1f} ms exceeds "
                f"{self.max_inter_wrist_skew_us / 1000.0:.1f} ms"
            )
        self.sequence += 1
        representative = min(matches.values(), key=lambda value: value[2])[1]
        with self.state_lock:
            interval_us = self.robot_state_arrival_interval_us
            max_interval_us = self.robot_state_max_arrival_interval_us
            gap_ms = float(self.stats.get("robot_state_gap_ms", 0.0))
            gap_max_ms = float(self.stats.get("robot_state_gap_max_ms", 0.0))
            self.stats["pairing_residual_ms"] = {
                camera_id: float(matches[camera_id][2]) for camera_id in CAMERA_IDS
            }
            self.stats["pairing_estimated_error_m"] = {
                camera_id: frames[camera_id]["state_pairing"]["estimated_error_m"]
                for camera_id in CAMERA_IDS
            }
        payload = {
            "version": 1,
            "sequence": self.sequence,
            "bundle_timestamp_us": now_us(),
            "split_y": frames["left_wrist"]["height"],
            "inter_wrist_skew_us": skew,
            # Compatibility snapshot for rolling worker upgrades. v14 workers
            # use each camera's q/base quaternion above instead.
            "robot_state_packet": representative.packet,
            # Preserve the legacy wall-clock field for an old remote worker
            # during a rolling deployment. v14 fields use monotonic arrival.
            "robot_state_host_received_us": int(representative.wall_received_us),
            "robot_state_arrival_interval_us": interval_us,
            "robot_state_max_arrival_interval_us": max_interval_us,
            "robot_state_gap_ms": gap_ms,
            "robot_state_gap_max_ms": gap_max_ms,
            "cameras": frames,
        }
        encoded = pack_bundle(payload)
        self.wifi_bus.publish(WRIST_BUNDLE_CHANNEL, encoded)
        if time.monotonic() - self.last_publish_log_monotonic >= 5.0:
            belief_status = {}
            if self.belief is not None:
                with self.belief.lock:
                    belief_status = {
                        "selected_hand": self.belief.belief_selected_hand,
                        "selection_active": self.belief.belief_selection_active,
                        "belief_targets": int(self.belief.stats.get("belief_targets", 0)),
                        "ee_targets_published": int(
                            self.belief.stats.get("ee_targets_published", 0)
                        ),
                    }
            LOG.info(
                "Wrist bundle published sequence=%d bytes=%d skew_ms=%.2f "
                "channel=%s control=%s capture=%s",
                self.sequence, len(encoded), skew / 1000.0,
                WRIST_BUNDLE_CHANNEL, belief_status,
                {
                    "skipped_frames": self.stats["skipped_frames"],
                    "robot_state_gap_ms": round(
                        float(self.stats.get("robot_state_gap_ms", 0.0)), 2
                    ),
                    "robot_state_gap_max_ms": round(
                        float(self.stats.get("robot_state_gap_max_ms", 0.0)), 2
                    ),
                    "pairing_residual_ms": {
                        camera_id: round(float(matches[camera_id][2]), 2)
                        for camera_id in CAMERA_IDS
                    },
                },
            )
            self.last_publish_log_monotonic = time.monotonic()

    def capture_smoke(self) -> None:
        """Verify both direct camera streams without sending any LCM traffic."""
        self.start(state_pairing=False)
        try:
            assert self.capture is not None
            for camera_id, bundle in self.capture.wait_for_bundles(
                timeout_ms=5000
            ).items():
                LOG.info(
                    "CAPTURE_OK camera=%s serial=%s grid=%dx%d rgb_ts_ms=%.3f "
                    "depth_ts_ms=%.3f delta_ms=%.3f rgb_domain=%s depth_domain=%s depth_scale_m=%g",
                    camera_id, bundle.serial, bundle.intrinsics["width"], bundle.intrinsics["height"],
                    bundle.color_timestamp_ms, bundle.depth_timestamp_ms,
                    abs(bundle.color_timestamp_ms - bundle.depth_timestamp_ms),
                    bundle.color_timestamp_domain, bundle.depth_timestamp_domain, bundle.depth_scale_m,
                )
        finally:
            self.stop()

    def run(self, *, max_hz: float) -> None:
        self.start()
        period = 1.0 / max(0.1, float(max_hz))
        try:
            while True:
                started = time.monotonic()
                try:
                    self.publish_once()
                except RuntimeError as error:
                    self.stats["skipped_frames"] = int(
                        self.stats.get("skipped_frames", 0)
                    ) + 1
                    self.stats["last_skip_reason"] = str(error)
                    if time.monotonic() - self.last_skip_log_monotonic >= 5.0:
                        LOG.warning(
                            "frame skipped (%d total): %s",
                            self.stats["skipped_frames"],
                            error,
                        )
                        self.last_skip_log_monotonic = time.monotonic()
                    if is_d405_transport_error(error):
                        self._recover_capture(error)
                while self.wifi_bus.handle_timeout(0) > 0:
                    pass
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        finally:
            self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g1-lcm-url",
        default=os.environ.get("G1_LCM_URL", "udpm://239.255.76.67:7667?ttl=1"),
        help="G1-side LCM input carrying timestamped robot state.",
    )
    parser.add_argument(
        "--wifi-lcm-url",
        default=os.environ.get("WIFI_LCM_URL", "udpm://239.255.77.67:7667?ttl=1"),
        help="Wi-Fi LCM transport for PATH_V11_WRIST_* bundles and results.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/wrist_d405_v11.yaml"))
    parser.add_argument("--preflight", action="store_true", help="enumerate both configured D405s without streaming")
    parser.add_argument("--capture-smoke", action="store_true", help="capture one RGB-D frame from each D405, then exit without LCM publishing")
    parser.add_argument("--diagnose", action="store_true", help="log container-visible D405 devices, V4L2 sysfs, and USB node mounts")
    parser.add_argument("--max-hz", type=float, default=8.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--camera-recovery-initial-delay-ms",
        type=float,
        default=300.0,
        help="delay before the first D405 stop/start reset after USB/UVC loss",
    )
    parser.add_argument(
        "--camera-recovery-max-delay-ms",
        type=float,
        default=2000.0,
        help="maximum retry delay while a disconnected D405 is re-enumerating",
    )
    parser.add_argument("--robot-state-channel", default="G1_ROBOT_STATE")
    parser.add_argument("--state-history-span-sec", type=float, default=2.0)
    parser.add_argument("--state-history-publish-hz", type=float, default=500.0)
    parser.add_argument("--left-state-lookup-bias-ms", type=float, default=0.0,
                        help="Thor-arrival lookup offset for left D405; tune from static-object sweep")
    parser.add_argument("--right-state-lookup-bias-ms", type=float, default=0.0,
                        help="Thor-arrival lookup offset for right D405; tune independently")
    parser.add_argument("--state-pairing-reference-range-m", type=float, default=0.30,
                        help="diagnostic range used before worker replaces it with object distance")
    parser.add_argument("--belief-config", type=Path, default=Path("config/pipeline.yaml"),
                        help="existing TASC belief settings reused for both wrists")
    parser.add_argument("--belief-inference", action=argparse.BooleanOptionalAction, default=True,
                        help="run two-hand TASC belief on Thor and publish telemetry to Wi-Fi")
    parser.add_argument("--belief-input-source", choices=("operator_command", "robot_fk"),
                        default="operator_command",
                        help="match the legacy gateway's belief hand-input selection")
    parser.add_argument("--belief-command-max-delta-ms", type=float, default=25.0)
    parser.add_argument("--belief-command-divergence-max-m", type=float, default=0.20)
    parser.add_argument("--shared-hand-selection-endpoint",
                        default=os.environ.get("PATH_SHARED_HAND_SELECTION_ENDPOINT", "tcp://127.0.0.1:5567"))
    parser.add_argument("--shared-hand-selection-topic",
                        default=os.environ.get("PATH_SHARED_HAND_SELECTION_TOPIC", "path_shared_control_hand"))
    parser.add_argument(
        "--ee-target-publishing",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PATH_REMOTE_EE_TARGET_ENABLED", "1").lower()
        not in {"0", "false", "no", "off"},
        help=(
            "publish selected wrist belief targets to the existing PATH "
            "state-machine/IPOPT input (default: enabled)"
        ),
    )
    parser.add_argument(
        "--ee-target-endpoint",
        default=DEFAULT_EE_TARGET_ENDPOINT,
        help="existing PATH state-machine static-target endpoint",
    )
    parser.add_argument(
        "--ee-target-topic",
        default=DEFAULT_EE_TARGET_TOPIC,
        help="existing PATH state-machine static-target topic",
    )
    parser.add_argument(
        "--ee-target-link",
        default=os.environ.get("PATH_EE_TARGET_LINK", "left_wrist_yaw_link"),
        help="fallback link before A/B selects left/right",
    )
    parser.add_argument(
        "--ee-target-label",
        default=os.environ.get("PATH_EE_TARGET_LABEL", ""),
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
    parser.add_argument("--one-camera-diagnostic", action="store_true", help="reserved explicit diagnostic mode; normal v11 requires both cameras")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    gateway = ThorWristD405Gateway(
        g1_lcm_url=args.g1_lcm_url, wifi_lcm_url=args.wifi_lcm_url,
        config=args.config, jpeg_quality=args.jpeg_quality,
        one_camera_diagnostic=args.one_camera_diagnostic,
        robot_state_channel=args.robot_state_channel,
        belief_inference=args.belief_inference, belief_config=args.belief_config,
        belief_input_source=args.belief_input_source,
        belief_command_max_delta_ms=args.belief_command_max_delta_ms,
        belief_command_divergence_max_m=args.belief_command_divergence_max_m,
        shared_hand_selection_endpoint=args.shared_hand_selection_endpoint,
        shared_hand_selection_topic=args.shared_hand_selection_topic,
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
        state_history_span_s=args.state_history_span_sec,
        state_history_publish_hz=args.state_history_publish_hz,
        left_state_lookup_bias_ms=args.left_state_lookup_bias_ms,
        right_state_lookup_bias_ms=args.right_state_lookup_bias_ms,
        state_pairing_reference_range_m=args.state_pairing_reference_range_m,
        camera_recovery_initial_delay_ms=args.camera_recovery_initial_delay_ms,
        camera_recovery_max_delay_ms=args.camera_recovery_max_delay_ms,
    )
    if args.diagnose:
        diagnose_runtime(_import_realsense())
    if args.preflight:
        gateway.preflight()
    elif args.capture_smoke:
        gateway.capture_smoke()
    else:
        gateway.run(max_hz=args.max_hz)


if __name__ == "__main__":
    main()
