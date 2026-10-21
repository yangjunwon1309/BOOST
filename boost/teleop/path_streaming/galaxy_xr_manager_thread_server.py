#!/usr/bin/env python3
"""Galaxy XR/WebXR controller server for PATH v15.1.

This is deliberately separate from ``pico_manager_thread_server.py``. It
publishes controller, button, and absolute waist-wrist topics consumed by
PATH's state-machine bridge, but never imports or starts
the Pico/XRoboToolkit runtime.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

from boost.io.path_transport.zmq_wire import pack_pose_message
from boost.teleop.xr.pico_button_gestures import PicoButtonGestureInterpreter
from boost.teleop.path_streaming.xr_source_v15 import (
    TeleVuerSource,
    XrStreamReader,
    controller_waist_targets_se3,
    parse_basis_correction,
)


STREAM_MODE_OFF = 0
STREAM_MODE_TRACKING = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream-mode",
        choices=("controller", "body", "vr3pt"),
        default="controller",
        help="Only controller is supported by WebXR; body modes fail before startup.",
    )
    parser.add_argument("--button-event-port", type=int, default=5565)
    parser.add_argument("--vr-pose-debug-port", type=int, default=5566)
    parser.add_argument("--pose-hz", type=float, default=50.0)
    parser.add_argument("--controller-hz", type=float, default=100.0)
    parser.add_argument("--source-poll-hz", type=float, default=120.0)
    parser.add_argument("--auto-tracking", action="store_true")
    parser.add_argument(
        "--televuer-root",
        type=Path,
        default=None,
        help="TeleVuer checkout root; omit after `pip install -e /path/to/televuer`.",
    )
    parser.add_argument(
        "--televuer-cert",
        type=Path,
        default=Path("~/.config/xr_teleoperate/cert.pem").expanduser(),
    )
    parser.add_argument(
        "--televuer-key",
        type=Path,
        default=Path("~/.config/xr_teleoperate/key.pem").expanduser(),
    )
    parser.add_argument("--televuer-hand-tracking", action="store_true")
    parser.add_argument(
        "--basis-correction",
        type=float,
        nargs=16,
        default=None,
        metavar="M",
        help="Optional raw-WebXR basis correction, row-major 4x4 SE(3).",
    )
    return parser.parse_args()


def _send_controller(
    socket: zmq.Socket,
    sample,
    *,
    sequence: int,
    timestamp_us: int,
) -> None:
    buttons = sample.buttons if sample is not None else {}
    payload = {
        "schema": "path.pico_controller_state.v1",
        "sequence": sequence,
        "controller_read_timestamp_us": timestamp_us,
        "publish_timestamp_us": int(time.monotonic_ns() // 1000),
        "left_axis": [0.0, 0.0] if sample is None else list(sample.left_axis),
        "right_axis": [0.0, 0.0] if sample is None else list(sample.right_axis),
        "left_axis_click": False,
        "right_axis_click": False,
        "left_trigger": 0.0 if sample is None else float(sample.left_trigger),
        "right_trigger": 0.0 if sample is None else float(sample.right_trigger),
        "left_grip": 0.0 if sample is None else float(sample.left_grip),
        "right_grip": 0.0 if sample is None else float(sample.right_grip),
        "a": bool(buttons.get("A", False)),
        "b": bool(buttons.get("B", False)),
        "x": bool(buttons.get("X", False)),
        "y": bool(buttons.get("Y", False)),
    }
    socket.send(b"pico_controller_state" + msgpack.packb(payload, use_bin_type=True))


def _send_events(
    socket: zmq.Socket,
    gestures: PicoButtonGestureInterpreter,
    sample,
    *,
    sequence: int,
    timestamp_us: int,
) -> int:
    buttons = sample.buttons if sample is not None else {}
    state = (
        bool(buttons.get("A", False)),
        bool(buttons.get("B", False)),
        bool(buttons.get("X", False)),
        bool(buttons.get("Y", False)),
    )
    for event in gestures.update(state):
        sequence += 1
        socket.send(
            b"pico_button_event"
            + msgpack.packb(
                {
                    "schema": "path.pico_button_event.v1",
                    "event": event,
                    "sequence": sequence,
                    "button_read_timestamp_us": timestamp_us,
                    "publish_timestamp_us": int(time.monotonic_ns() // 1000),
                    "a": state[0], "b": state[1], "x": state[2], "y": state[3],
                },
                use_bin_type=True,
            )
        )
    return sequence


def _send_wrist_targets(
    socket: zmq.Socket,
    sample,
    *,
    tracking: bool,
    session_id: int,
    frame_index: int,
    monitor: dict,
) -> None:
    valid = bool(
        tracking
        and sample is not None
        and sample.head_valid
        and sample.left_valid
        and sample.right_valid
    )
    if valid:
        left, right, valid = controller_waist_targets_se3(
            sample.left, sample.right, sample.head,
            left_ok=sample.left_valid,
            right_ok=sample.right_valid,
            head_ok=sample.head_valid,
        )
    else:
        left = np.full((4, 4), np.nan, dtype=np.float64)
        right = np.full((4, 4), np.nan, dtype=np.float64)
    values = {
        "wrist_left_waist": left.reshape(-1).astype(np.float32),
        "wrist_right_waist": right.reshape(-1).astype(np.float32),
        "targets_valid": np.array([valid], dtype=bool),
        "frame_index": np.array([frame_index], dtype=np.int64),
        "timestamp_monotonic": np.array(
            [0.0 if sample is None else sample.arrival_monotonic], dtype=np.float64
        ),
        "ee_frame_offset_m": np.array([0.05], dtype=np.float32),
        "stream_mode": np.array(
            [STREAM_MODE_TRACKING if tracking else STREAM_MODE_OFF], dtype=np.int32
        ),
        "session_id": np.array([session_id], dtype=np.int32),
        "arm_reference_mode": np.array([0], dtype=np.int32),
        "left_trigger": np.array([0.0 if sample is None else sample.left_trigger], dtype=np.float32),
        "right_trigger": np.array([0.0 if sample is None else sample.right_trigger], dtype=np.float32),
        "left_grip": np.array([0.0 if sample is None else sample.left_grip], dtype=np.float32),
        "right_grip": np.array([0.0 if sample is None else sample.right_grip], dtype=np.float32),
    }
    for key in ("source_gap_median_ms", "source_gap_p99_ms", "source_gap_max_ms", "source_dropouts_over_150ms"):
        value = monitor.get(key)
        if value is not None and math.isfinite(float(value)):
            values[key] = np.array([float(value)], dtype=np.float32)
    socket.send(pack_pose_message(values, topic="vr_pose_debug"))


def main() -> int:
    args = parse_args()
    if args.stream_mode != "controller":
        raise SystemExit(
            "Galaxy/WebXR provides head and controller poses only; "
            f"--stream-mode {args.stream_mode} requires body joints."
        )
    if args.televuer_hand_tracking and not args.auto_tracking:
        raise SystemExit(
            "--televuer-hand-tracking has no A+B+X+Y start chord; "
            "pass --auto-tracking after validating its Y/B pinch dead-man."
        )
    basis_correction = parse_basis_correction(args.basis_correction)
    source = TeleVuerSource(
        cert_file=args.televuer_cert,
        key_file=args.televuer_key,
        televuer_root=args.televuer_root,
        use_hand_tracking=bool(args.televuer_hand_tracking),
        basis_correction=basis_correction,
    )
    reader = XrStreamReader(source, poll_hz=args.source_poll_hz)
    context = zmq.Context()
    button_socket = context.socket(zmq.PUB)
    wrist_socket = context.socket(zmq.PUB)
    for socket in (button_socket, wrist_socket):
        socket.setsockopt(zmq.SNDHWM, 4)
        socket.setsockopt(zmq.LINGER, 0)
    button_socket.bind(f"tcp://*:{args.button_event_port}")
    wrist_socket.bind(f"tcp://*:{args.vr_pose_debug_port}")

    print(
        "[GalaxyXR] raw WebXR -> OpenXR SE(3) -> PATH waist server; "
        "TeleVuer image transport disabled (pass-through, no ZMQ/WebRTC)",
        flush=True,
    )
    print(
        f"[GalaxyXR] PUB buttons={args.button_event_port} "
        f"wrist={args.vr_pose_debug_port}; pose={args.pose_hz:.0f}Hz controller={args.controller_hz:.0f}Hz",
        flush=True,
    )

    reader.start()
    tracking = bool(args.auto_tracking)
    session_id = 1 if tracking else 0
    controller_sequence = 0
    event_sequence = 0
    frame_index = 0
    gestures = PicoButtonGestureInterpreter()
    pose_period = 1.0 / max(1.0, float(args.pose_hz))
    controller_period = 1.0 / max(1.0, float(args.controller_hz))
    next_pose = next_control = next_report = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            sample = reader.latest()
            if now >= next_control:
                timestamp_us = time.monotonic_ns() // 1000
                controller_sequence += 1
                _send_controller(button_socket, sample, sequence=controller_sequence, timestamp_us=timestamp_us)
                old_event_sequence = event_sequence
                event_sequence = _send_events(
                    button_socket, gestures, sample, sequence=event_sequence, timestamp_us=timestamp_us
                )
                # Match the PICO manager: the all-four-button chord gates
                # whether valid controller poses are supplied to PATH.
                if event_sequence != old_event_sequence:
                    buttons = sample.buttons if sample is not None else {}
                    if all(bool(buttons.get(key, False)) for key in ("A", "B", "X", "Y")):
                        tracking = not tracking
                        if tracking:
                            session_id += 1
                        print(f"[GalaxyXR] tracking={'on' if tracking else 'off'} session={session_id}", flush=True)
                next_control += controller_period
                if next_control < now - controller_period:
                    next_control = now + controller_period
            if now >= next_pose:
                frame_index += 1
                _send_wrist_targets(
                    wrist_socket,
                    sample,
                    tracking=tracking,
                    session_id=session_id,
                    frame_index=frame_index,
                    monitor=reader.monitor.report(),
                )
                next_pose += pose_period
                if next_pose < now - pose_period:
                    next_pose = now + pose_period
            if now >= next_report:
                report = reader.monitor.report()
                if report:
                    print(f"[GalaxyXR] connected={source.connected} tracking={tracking} {report}", flush=True)
                next_report = now + 5.0
            time.sleep(0.001)
    except KeyboardInterrupt:
        return 0
    finally:
        reader.stop()
        for socket in (wrist_socket, button_socket):
            socket.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
