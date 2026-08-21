"""Thor-owned direct D405 capture gateway for wrist-only v11 perception."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import lcm
import numpy as np

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
                 ee_target_quaternion_wxyz: tuple[float, float, float, float]) -> None:
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
        }
        self.capture: WristD405Capture | None = None
        self.belief: WristBeliefGateway | None = None
        self.latest_robot_state_packet: tuple[int, bytes] | None = None
        self.last_robot_state_received_us: int | None = None
        self.robot_state_arrival_interval_us: int | None = None
        self.robot_state_max_arrival_interval_us = 0
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
        self.wifi_bus.publish(WRIST_ROBOT_STATE_CHANNEL, data)
        received_us = now_us()
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
        if self.belief is not None:
            self.belief.update_robot_state(data)

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

    def start(self) -> None:
        assert_video_nodes_unowned()
        rs = _import_realsense()
        specs = load_camera_specs(self.config)
        capture_config = _load_yaml(self.config).get("capture", {})
        self.capture = WristD405Capture(
            rs, specs,
            max_rgb_depth_delta_ms=float(capture_config.get("max_rgb_depth_delta_ms", 20.0)),
        )
        self.capture.start()
        if self.belief is not None:
            self.belief.start()
        LOG.info("direct D405 capture started for %s", sorted(self.capture.specs))

    def stop(self) -> None:
        if self.belief is not None:
            self.belief.stop()
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

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
        if self.latest_robot_state_packet is None:
            raise RuntimeError(
                f"no robot state received on {self.robot_state_channel}; "
                "refusing RGB-D without the FK snapshot required for belief geometry"
            )
        captured = self.capture.wait_for_bundles()
        frames: dict[str, dict[str, Any]] = {}
        for camera_id in CAMERA_IDS:
            bundle = captured[camera_id]
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
                "diagnostics": {"rgb_depth_delta_ms": abs(bundle.color_timestamp_ms - bundle.depth_timestamp_ms)},
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
        state_received_us, state_packet = self.latest_robot_state_packet
        payload = {
            "version": 1,
            "sequence": self.sequence,
            "bundle_timestamp_us": now_us(),
            "split_y": frames["left_wrist"]["height"],
            "inter_wrist_skew_us": skew,
            "robot_state_packet": state_packet,
            "robot_state_host_received_us": int(state_received_us),
            "robot_state_arrival_interval_us": self.robot_state_arrival_interval_us,
            "robot_state_max_arrival_interval_us": self.robot_state_max_arrival_interval_us,
            "robot_state_gap_ms": float(self.stats.get("robot_state_gap_ms", 0.0)),
            "robot_state_gap_max_ms": float(
                self.stats.get("robot_state_gap_max_ms", 0.0)
            ),
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
                },
            )
            self.last_publish_log_monotonic = time.monotonic()

    def capture_smoke(self) -> None:
        """Verify both direct camera streams without sending any LCM traffic."""
        self.start()
        try:
            assert self.capture is not None
            for camera_id, bundle in self.capture.wait_for_bundles(
                timeout_ms=5000
            ).items():
                LOG.info(
                    "CAPTURE_OK camera=%s serial=%s grid=%dx%d rgb_ts_ms=%.3f "
                    "depth_ts_ms=%.3f delta_ms=%.3f depth_scale_m=%g",
                    camera_id, bundle.serial, bundle.intrinsics["width"], bundle.intrinsics["height"],
                    bundle.color_timestamp_ms, bundle.depth_timestamp_ms,
                    abs(bundle.color_timestamp_ms - bundle.depth_timestamp_ms), bundle.depth_scale_m,
                )
        finally:
            self.stop()

    def run(self, *, max_hz: float) -> None:
        self.start()
        period = 1.0 / max(0.1, float(max_hz))
        try:
            while True:
                started = time.monotonic()
                # Keep the newest robot state: at 50 Hz a single callback per
                # 15 Hz capture tick would otherwise violate the 15 ms v11 FK
                # matching budget before the remote worker even receives it.
                while self.g1_bus.handle_timeout(0) > 0:
                    pass
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
    parser.add_argument("--robot-state-channel", default="G1_ROBOT_STATE")
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
