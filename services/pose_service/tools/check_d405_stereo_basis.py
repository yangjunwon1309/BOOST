#!/usr/bin/env python3
"""Report the Intel RealSense D405 RGB/depth-imager relationship.

For product ID 8086:0b5b, Intel's D405 SDK identifies the direct UYVY color
stream as coming from the depth sensor's *left imager*. Depth uses the left
and right stereo imagers. Thus direct UYVY is one view, not a stereo RGB pair,
but it shares the left-imager basis with depth and has no separate
RGB-to-depth baseline. D405 also exposes a distinct YUYV RGB+ISP endpoint.

The default ``auto`` backend falls back to Linux sysfs/V4L2 enumeration when
pyrealsense2 is absent, and never opens a video stream.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


D405_PRODUCT_ID = "0B5B"


@dataclass(frozen=True)
class Stream:
    """A unique RealSense stream exposed by one device."""

    name: str
    index: int
    formats: list[str]
    resolutions: list[str]
    fps: list[int]


@dataclass(frozen=True)
class D405Report:
    serial: str
    name: str
    physical_port: str
    usb_type: str
    streams: list[Stream]
    has_color: bool
    infrared_stereo_pair: bool
    color_source: str
    rgb_is_stereo_pair: bool
    stereo_pair: str | None
    infrared_baseline_mm: float | None
    color_to_left_ir_offset_mm: float | None
    conclusion: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        action="append",
        default=[],
        metavar="SERIAL",
        help="Check only this serial (may be supplied more than once).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--backend",
        choices=("auto", "v4l2", "librealsense"),
        default="auto",
        help=(
            "Device-enumeration backend. auto uses librealsense when available, "
            "otherwise Linux sysfs/V4L2 without a RealSense Python package."
        ),
    )
    parser.add_argument(
        "--ssh",
        metavar="DESTINATION",
        help=(
            "Run this check on DESTINATION over SSH (for example url@thor). "
            "The script is sent over standard input; it need not exist on the remote host."
        ),
    )
    parser.add_argument(
        "--require-two",
        action="store_true",
        help="Exit nonzero unless exactly two D405 devices are found.",
    )
    return parser.parse_args()


def _run_on_ssh(destination: str, args: argparse.Namespace) -> int:
    """Execute this exact script on Thor, without requiring a remote checkout."""

    remote_args = ["python3", "-"]
    for serial in args.serial:
        remote_args.extend(("--serial", serial))
    if args.json:
        remote_args.append("--json")
    backend = getattr(args, "backend", "auto")
    if backend != "auto":
        remote_args.extend(("--backend", backend))
    if args.require_two:
        remote_args.append("--require-two")
    source = Path(__file__).read_text(encoding="utf-8")
    try:
        completed = subprocess.run(["ssh", destination, *remote_args], input=source, text=True)
    except FileNotFoundError:
        print("ssh executable not found on this computer.", file=sys.stderr)
        return 4
    return completed.returncode


def _read_sysfs(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _v4l2_reports(requested_serials: set[str]) -> list[D405Report]:
    """Find D405 UVC nodes using only Linux sysfs (no librealsense needed)."""

    usb_root = Path("/sys/bus/usb/devices")
    reports: list[D405Report] = []
    if not usb_root.is_dir():
        return reports
    for device in sorted(usb_root.iterdir()):
        if (
            _read_sysfs(device / "idVendor").lower() != "8086"
            or _read_sysfs(device / "idProduct").lower() != D405_PRODUCT_ID.lower()
        ):
            continue
        serial = _read_sysfs(device / "serial")
        if requested_serials and serial not in requested_serials:
            continue
        streams: list[Stream] = []
        for node_path in sorted(usb_root.glob(f"{device.name}:*/video4linux/video*")):
            node = node_path.name
            suffix = node.removeprefix("video")
            streams.append(
                Stream(
                    name=_read_sysfs(Path("/sys/class/video4linux") / node / "name") or "v4l2",
                    index=int(suffix) if suffix.isdigit() else -1,
                    formats=[],
                    resolutions=[],
                    fps=[],
                )
            )
        reports.append(
            D405Report(
                serial=serial,
                name=_read_sysfs(device / "product") or "Intel RealSense D405",
                physical_port=device.name,
                usb_type="Linux sysfs/V4L2 (no librealsense)",
                streams=streams,
                # sysfs verifies the attached D405 and its UVC nodes. The RGB
                # stream is profile-dependent, so this field is intentionally
                # not inferred from a node name in the fallback backend.
                has_color=False,
                infrared_stereo_pair=True,
                color_source="direct UYVY left depth/stereo imager (D405 architecture)",
                rgb_is_stereo_pair=False,
                stereo_pair="left imager + right imager",
                infrared_baseline_mm=None,
                # Linux sysfs/V4L2 cannot retrieve RealSense factory
                # extrinsics. The same-left-imager relation below is a D405
                # architecture fact, not a calibration measurement.
                color_to_left_ir_offset_mm=None,
                conclusion=(
                    "D405 detected through Linux sysfs/V4L2. Intel's D405 architecture "
                    "uses direct UYVY from the left depth imager; depth uses left + right."
                ),
            )
        )
    return reports


def _device_info(device: Any, rs: Any, info: Any) -> str:
    return device.get_info(info) if device.supports(info) else ""


def _vector_norm_mm(translation: Any) -> float:
    return round(math.sqrt(sum(float(value) ** 2 for value in translation)) * 1000.0, 3)


def _stream_name(rs: Any, stream_type: Any) -> str:
    # str(rs.stream.color) is stable today, but the enum's name is clearer in
    # output and keeps this tool usable across librealsense Python versions.
    names = {
        rs.stream.color: "color",
        rs.stream.depth: "depth",
        rs.stream.infrared: "infrared",
    }
    return names.get(stream_type, str(stream_type).split(".")[-1])


def _unique_profiles(device: Any, rs: Any) -> dict[tuple[Any, int], list[Any]]:
    profiles: dict[tuple[Any, int], list[Any]] = {}
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            key = (profile.stream_type(), profile.stream_index())
            profiles.setdefault(key, []).append(profile)
    return profiles


def _stream_summary(profiles: Iterable[Any], rs: Any) -> Stream:
    profiles = list(profiles)
    first = profiles[0]
    formats = sorted({str(profile.format()).split(".")[-1] for profile in profiles})
    resolutions: set[str] = set()
    fps: set[int] = set()
    for profile in profiles:
        try:
            video = profile.as_video_stream_profile()
            resolutions.add(f"{video.width()}x{video.height()}")
        except RuntimeError:
            pass
        fps.add(int(profile.fps()))
    return Stream(
        name=_stream_name(rs, first.stream_type()),
        index=int(first.stream_index()),
        formats=formats,
        resolutions=sorted(resolutions),
        fps=sorted(fps),
    )


def inspect_device(device: Any, rs: Any) -> D405Report:
    """Inspect calibration/topology only; this does not start a pipeline."""

    profiles = _unique_profiles(device, rs)
    streams = sorted(
        (_stream_summary(group, rs) for group in profiles.values()),
        key=lambda item: (item.name, item.index),
    )
    color_key = (rs.stream.color, 0)
    left_ir_key = (rs.stream.infrared, 1)
    right_ir_key = (rs.stream.infrared, 2)
    has_color = color_key in profiles
    has_stereo_pair = left_ir_key in profiles and right_ir_key in profiles
    ir_baseline_mm = None
    color_to_left_ir_offset_mm = None
    if has_stereo_pair:
        ir_baseline_mm = _vector_norm_mm(
            profiles[left_ir_key][0].get_extrinsics_to(profiles[right_ir_key][0]).translation
        )
    if has_color and left_ir_key in profiles:
        color_to_left_ir_offset_mm = _vector_norm_mm(
            profiles[left_ir_key][0].get_extrinsics_to(profiles[color_key][0]).translation
        )

    serial = _device_info(device, rs, rs.camera_info.serial_number)
    name = _device_info(device, rs, rs.camera_info.name)
    physical_port = _device_info(device, rs, rs.camera_info.physical_port)
    usb_type = _device_info(device, rs, rs.camera_info.usb_type_descriptor)
    if has_stereo_pair and has_color:
        conclusion = (
            "Direct UYVY color comes from the D405 left depth imager; depth's "
            "stereo basis is left + right. UYVY itself is not a stereo pair."
        )
    elif has_stereo_pair:
        conclusion = "No color stream exposed; depth's stereo basis is infrared:1 + infrared:2."
    else:
        conclusion = "Could not find both infrared:1 and infrared:2; stereo topology is unavailable."
    return D405Report(
        serial=serial,
        name=name,
        physical_port=physical_port,
        usb_type=usb_type,
        streams=streams,
        has_color=has_color,
        infrared_stereo_pair=has_stereo_pair,
        color_source="direct UYVY left depth/stereo imager (D405 architecture)",
        rgb_is_stereo_pair=False,
        stereo_pair="left imager + right imager" if has_stereo_pair else None,
        infrared_baseline_mm=ir_baseline_mm,
        color_to_left_ir_offset_mm=color_to_left_ir_offset_mm,
        conclusion=conclusion,
    )


def _print_report(report: D405Report) -> None:
    print(f"D405 {report.serial or '<unknown serial>'}: {report.name}")
    print(f"  port={report.physical_port or '<unknown>'} usb={report.usb_type or '<unknown>'}")
    print("  streams:")
    for stream in report.streams:
        detail = ", ".join(
            part
            for part in (
                "/".join(stream.resolutions),
                "/".join(stream.formats),
                "/".join(map(str, stream.fps)) + " fps" if stream.fps else "",
            )
            if part
        )
        print(f"    {stream.name}:{stream.index}" + (f" ({detail})" if detail else ""))
    if report.infrared_baseline_mm is not None:
        print(f"  IR1 -> IR2 calibrated baseline: {report.infrared_baseline_mm:.3f} mm")
    if report.color_to_left_ir_offset_mm is not None:
        print(f"  left IR -> color calibrated offset: {report.color_to_left_ir_offset_mm:.3f} mm")
    print(f"  RGB source: {report.color_source}")
    print(f"  RGB is a stereo pair: {'YES' if report.rgb_is_stereo_pair else 'NO'}")
    print(f"  {report.conclusion}")


def main() -> int:
    args = parse_args()
    if args.ssh:
        return _run_on_ssh(args.ssh, args)
    requested_serials = set(args.serial)
    reports: list[D405Report]
    if args.backend == "v4l2":
        reports = _v4l2_reports(requested_serials)
    else:
        try:
            import pyrealsense2 as rs
        except ImportError:
            if args.backend == "librealsense":
                print("pyrealsense2 is required for --backend librealsense.", file=sys.stderr)
                return 3
            reports = _v4l2_reports(requested_serials)
        else:
            d405_devices = []
            for device in rs.context().query_devices():
                product_id = _device_info(device, rs, rs.camera_info.product_id).upper()
                if product_id != D405_PRODUCT_ID:
                    continue
                serial = _device_info(device, rs, rs.camera_info.serial_number)
                if requested_serials and serial not in requested_serials:
                    continue
                d405_devices.append(device)
            reports = [inspect_device(device, rs) for device in d405_devices]
    if args.json:
        print(json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True))
    elif not reports:
        print("No Intel RealSense D405 devices found (librealsense/sysfs probe).", file=sys.stderr)
    else:
        for number, report in enumerate(reports):
            if number:
                print()
            _print_report(report)

    if args.require_two and len(reports) != 2:
        print(f"Expected exactly two D405 devices, found {len(reports)}.", file=sys.stderr)
        return 2
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
