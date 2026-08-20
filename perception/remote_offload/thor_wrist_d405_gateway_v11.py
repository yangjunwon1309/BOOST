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
    def __init__(self, *, lcm_url: str, config: Path, jpeg_quality: int, one_camera_diagnostic: bool,
                 robot_state_channel: str) -> None:
        self.bus = lcm.LCM(lcm_url)
        self.config = config
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.one_camera_diagnostic = bool(one_camera_diagnostic)
        self.robot_state_channel = str(robot_state_channel)
        self.sequence = 0
        self.capture: WristD405Capture | None = None
        capture_config = _load_yaml(config).get("capture", {})
        self.max_inter_wrist_skew_us = int(float(capture_config.get("max_inter_wrist_skew_ms", 80.0)) * 1000.0)
        self.bus.subscribe(WRIST_RESULT_CHANNEL, self._on_result)
        self.bus.subscribe(self.robot_state_channel, self._on_robot_state)

    def _on_robot_state(self, _channel: str, data: bytes) -> None:
        # Robot state is the only legacy LCM input retained in v11. It is
        # forwarded unchanged for remote timestamp matching; no head image,
        # depth, or head calibration channel is subscribed.
        self.bus.publish(WRIST_ROBOT_STATE_CHANNEL, data)

    def preflight(self) -> None:
        rs = _import_realsense()
        specs = load_camera_specs(self.config)
        visible = discover_d405_serials(rs)
        missing = [spec.serial for spec in specs.values() if spec.serial not in visible]
        unexpected = set(visible) - {spec.serial for spec in specs.values()}
        if missing or unexpected or len(visible) != 2:
            raise RuntimeError(f"D405 preflight requires exactly configured pair; missing={missing}, unexpected={sorted(unexpected)}")
        for side, spec in specs.items():
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

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

    def _on_result(self, _channel: str, data: bytes) -> None:
        try:
            result = unpack_result(data)
            LOG.info("v11 result sequence=%s objects=%s seam_discarded=%s", result.get("sequence"), len(result.get("objects", [])), result.get("discarded_seam_masks"))
        except Exception as error:
            LOG.warning("rejected v11 worker result: %s", error)

    def publish_once(self) -> None:
        if self.capture is None:
            raise RuntimeError("D405 gateway is not started")
        frames: dict[str, dict[str, Any]] = {}
        for camera_id in CAMERA_IDS:
            bundle = self.capture.wait_for_bundle(camera_id)
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
        skew = abs(frames["left_wrist"]["host_timestamp_us"] - frames["right_wrist"]["host_timestamp_us"])
        if skew > self.max_inter_wrist_skew_us:
            raise RuntimeError(
                f"inter-wrist capture skew {skew / 1000.0:.1f} ms exceeds "
                f"{self.max_inter_wrist_skew_us / 1000.0:.1f} ms"
            )
        self.sequence += 1
        payload = {"version": 1, "sequence": self.sequence, "bundle_timestamp_us": now_us(), "split_y": frames["left_wrist"]["height"], "inter_wrist_skew_us": skew, "cameras": frames}
        self.bus.publish(WRIST_BUNDLE_CHANNEL, pack_bundle(payload))

    def run(self, *, max_hz: float) -> None:
        self.start()
        period = 1.0 / max(0.1, float(max_hz))
        try:
            while True:
                started = time.monotonic()
                self.publish_once()
                self.bus.handle_timeout(0)
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        finally:
            self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", default=os.environ.get("PATH_LCM_URL", "udpm://239.255.76.67:7667?ttl=1"))
    parser.add_argument("--config", type=Path, default=Path("config/wrist_d405_v11.yaml"))
    parser.add_argument("--preflight", action="store_true", help="enumerate both configured D405s without streaming")
    parser.add_argument("--max-hz", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--robot-state-channel", default="G1_ROBOT_STATE")
    parser.add_argument("--one-camera-diagnostic", action="store_true", help="reserved explicit diagnostic mode; normal v11 requires both cameras")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    gateway = ThorWristD405Gateway(lcm_url=args.lcm_url, config=args.config, jpeg_quality=args.jpeg_quality, one_camera_diagnostic=args.one_camera_diagnostic, robot_state_channel=args.robot_state_channel)
    if args.preflight:
        gateway.preflight()
    else:
        gateway.run(max_hz=args.max_hz)


if __name__ == "__main__":
    main()
