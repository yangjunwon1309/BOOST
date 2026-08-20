"""Direct, serial-addressed D405 capture primitives for the wrist-only v11 path.

The capture intentionally requests the D405 direct UYVY left-imager color
stream and same-size Z16 depth. It does not call rs.align(): this v11 mode uses
the direct left-imager basis, not the D405 RGB+ISP endpoint.
"""

from __future__ import annotations

import dataclasses
import concurrent.futures
import time
from pathlib import Path
from typing import Any


D405_PRODUCT_ID = "0B5B"
SIDES = ("left_wrist", "right_wrist")


@dataclasses.dataclass(frozen=True)
class StreamSpec:
    width: int
    height: int
    fps: int
    format: str


@dataclasses.dataclass(frozen=True)
class WristCameraSpec:
    side: str
    serial: str
    color: StreamSpec
    depth: StreamSpec


@dataclasses.dataclass(frozen=True)
class FrameBundle:
    side: str
    serial: str
    host_timestamp_us: int
    color_timestamp_ms: float
    depth_timestamp_ms: float
    color_bgr: Any
    depth_z16: Any
    intrinsics: dict[str, float | int]
    depth_scale_m: float


def _stream_spec(data: dict[str, Any], *, name: str) -> StreamSpec:
    try:
        spec = StreamSpec(
            width=int(data["width"]),
            height=int(data["height"]),
            fps=int(data["fps"]),
            format=str(data["format"]).lower(),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must define width, height, fps, and format") from error
    if min(spec.width, spec.height, spec.fps) <= 0:
        raise ValueError(f"{name} dimensions/fps must be positive")
    return spec


def load_camera_specs(path: Path) -> dict[str, WristCameraSpec]:
    """Load and strictly validate the two serial-addressed D405 configurations."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load the v11 D405 config") from error
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cameras = raw.get("cameras") if isinstance(raw, dict) else None
    if not isinstance(cameras, dict):
        raise ValueError("v11 config must contain a cameras mapping")
    result: dict[str, WristCameraSpec] = {}
    serials: set[str] = set()
    for side in SIDES:
        entry = cameras.get(side)
        if not isinstance(entry, dict):
            raise ValueError(f"v11 config is missing cameras.{side}")
        serial = str(entry.get("serial", "")).strip()
        if not serial or serial.startswith("REQUIRED_"):
            raise ValueError(f"cameras.{side}.serial must be a real D405 serial")
        if serial in serials:
            raise ValueError("left and right wrist D405 serials must differ")
        serials.add(serial)
        color = _stream_spec(dict(entry.get("color") or {}), name=f"cameras.{side}.color")
        depth = _stream_spec(dict(entry.get("depth") or {}), name=f"cameras.{side}.depth")
        if color.format != "uyvy":
            raise ValueError(f"cameras.{side}.color.format must be uyvy (direct left-imager stream)")
        if depth.format != "z16":
            raise ValueError(f"cameras.{side}.depth.format must be z16")
        if (color.width, color.height, color.fps) != (depth.width, depth.height, depth.fps):
            raise ValueError(f"cameras.{side} color/depth profiles must match in v11")
        result[side] = WristCameraSpec(side=side, serial=serial, color=color, depth=depth)
    return result


def _device_info(device: Any, rs: Any, info: Any) -> str:
    return device.get_info(info) if device.supports(info) else ""


def discover_d405_serials(rs: Any) -> dict[str, dict[str, str]]:
    """Return visible D405 factory identity without starting a stream."""

    found: dict[str, dict[str, str]] = {}
    for device in rs.context().query_devices():
        product_id = _device_info(device, rs, rs.camera_info.product_id).upper()
        if product_id != D405_PRODUCT_ID:
            continue
        serial = _device_info(device, rs, rs.camera_info.serial_number)
        found[serial] = {
            "name": _device_info(device, rs, rs.camera_info.name),
            "physical_port": _device_info(device, rs, rs.camera_info.physical_port),
            "usb_type": _device_info(device, rs, rs.camera_info.usb_type_descriptor),
        }
    return found


class WristD405Capture:
    """Own two direct D405 pipelines; one process must own the UVC nodes."""

    def __init__(self, rs: Any, specs: dict[str, WristCameraSpec], *, max_rgb_depth_delta_ms: float = 20.0) -> None:
        self.rs = rs
        self.specs = specs
        self.pipelines: dict[str, Any] = {}
        self.depth_scales: dict[str, float] = {}
        self.max_rgb_depth_delta_ms = float(max_rgb_depth_delta_ms)
        self._bundle_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def start(self) -> None:
        visible = discover_d405_serials(self.rs)
        expected = {spec.serial for spec in self.specs.values()}
        if set(visible) != expected:
            raise RuntimeError(
                f"expected exactly configured D405 serials {sorted(expected)}, found {sorted(visible)}"
            )
        for side, spec in self.specs.items():
            pipeline = self.rs.pipeline()
            config = self.rs.config()
            config.enable_device(spec.serial)
            config.enable_stream(
                # D405 exposes its direct UYVY left-depth-imager colour as
                # the Infrared stream. Do not request Color/YUYV+ISP here.
                self.rs.stream.infrared,
                spec.color.width,
                spec.color.height,
                self.rs.format.uyvy,
                spec.color.fps,
            )
            config.enable_stream(
                self.rs.stream.depth,
                spec.depth.width,
                spec.depth.height,
                self.rs.format.z16,
                spec.depth.fps,
            )
            profile = pipeline.start(config)
            self.pipelines[side] = pipeline
            self.depth_scales[side] = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self._bundle_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.pipelines),
            thread_name_prefix="d405-rgbd",
        )

    def stop(self) -> None:
        if self._bundle_executor is not None:
            self._bundle_executor.shutdown(wait=True, cancel_futures=True)
            self._bundle_executor = None
        for pipeline in self.pipelines.values():
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        self.pipelines.clear()

    def wait_for_bundle(self, side: str, timeout_ms: int = 2000) -> FrameBundle:
        """Return native same-grid UYVY/Z16 data without depth-to-color alignment."""

        if side not in self.pipelines:
            raise RuntimeError(f"capture has not started for {side}")
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError("OpenCV and NumPy are required for D405 frame conversion") from error
        frames = self.pipelines[side].wait_for_frames(timeout_ms=int(timeout_ms))
        color = frames.get_infrared_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError(f"{side} did not return both color and depth")
        timestamp_delta_ms = abs(float(color.get_timestamp()) - float(depth.get_timestamp()))
        if timestamp_delta_ms > self.max_rgb_depth_delta_ms:
            raise RuntimeError(
                f"{side} RGB/depth timestamp delta {timestamp_delta_ms:.1f} ms exceeds "
                f"{self.max_rgb_depth_delta_ms:.1f} ms"
            )
        color_profile = color.profile.as_video_stream_profile()
        if (color.get_width(), color.get_height()) != (depth.get_width(), depth.get_height()):
            raise RuntimeError(f"{side} color/depth grids differ; refusing unaligned v11 capture")
        intrinsics = color_profile.get_intrinsics()
        width, height = int(color.get_width()), int(color.get_height())
        # pyrealsense exposes packed UYVY as a flat/one-channel byte buffer.
        # OpenCV's UYVY conversion requires the explicit two bytes per pixel.
        uyvy = np.frombuffer(color.get_data(), dtype=np.uint8)
        expected_uyvy_bytes = width * height * 2
        if uyvy.size != expected_uyvy_bytes:
            raise RuntimeError(
                f"{side} did not deliver packed UYVY: got {uyvy.size} bytes, "
                f"expected {expected_uyvy_bytes} for {width}x{height}"
            )
        uyvy = uyvy.reshape(height, width, 2)
        return FrameBundle(
            side=side,
            serial=self.specs[side].serial,
            host_timestamp_us=time.monotonic_ns() // 1000,
            color_timestamp_ms=float(color.get_timestamp()),
            depth_timestamp_ms=float(depth.get_timestamp()),
            color_bgr=cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY).copy(),
            depth_z16=np.asanyarray(depth.get_data()).copy(),
            intrinsics={
                "width": int(intrinsics.width), "height": int(intrinsics.height),
                "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx), "ppy": float(intrinsics.ppy),
            },
            depth_scale_m=self.depth_scales[side],
        )

    def wait_for_bundles(self, timeout_ms: int = 2000) -> dict[str, FrameBundle]:
        """Wait for both independent D405 pipelines concurrently.

        Sequential waits impose a frame-period skew even when both cameras are
        healthy. Each RealSense pipeline is independent, so request the newest
        native RGB-D pair in parallel and let the gateway enforce the remaining
        measured skew budget.
        """
        if self._bundle_executor is None:
            raise RuntimeError("capture has not started")
        futures = {
            side: self._bundle_executor.submit(self.wait_for_bundle, side, timeout_ms)
            for side in self.specs
        }
        try:
            return {side: future.result() for side, future in futures.items()}
        except Exception:
            for future in futures.values():
                future.cancel()
            raise
