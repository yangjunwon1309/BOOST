#!/usr/bin/env python3
"""Galaxy XR/WebXR hand-tracking source for PATH.

It publishes hand-tracked absolute waist-wrist targets consumed by PATH's
state-machine bridge, with no headset-controller input runtime.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import zmq

from boost.io.path_transport.zmq_wire import pack_pose_message
from boost.teleop.path_streaming.xr_source_v15 import (
    TeleVuerSource,
    XrStreamReader,
    xr_waist_targets_se3,
    parse_basis_correction,
)


STREAM_MODE_TRACKING = 7
HAND_LANDMARKS_SCHEMA_VERSION = 1
HAND_LANDMARKS_FRAME_OPENXR_WORLD = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vr-pose-debug-port", type=int, default=5566)
    parser.add_argument("--pose-hz", type=float, default=50.0)
    parser.add_argument("--source-poll-hz", type=float, default=120.0)
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
    parser.add_argument(
        "--basis-correction",
        type=float,
        nargs=16,
        default=None,
        metavar="M",
        help="Optional raw-WebXR basis correction, row-major 4x4 SE(3).",
    )
    return parser.parse_args()


def _send_wrist_targets(
    socket: zmq.Socket,
    sample,
    *,
    tracking: bool,
    session_id: int,
    frame_index: int,
) -> None:
    valid = bool(
        tracking
        and sample is not None
        and sample.head_valid
        and sample.left_valid
        and sample.right_valid
    )
    if valid:
        left, right, valid = xr_waist_targets_se3(
            sample.left, sample.right, sample.head,
            left_ok=sample.left_valid,
            right_ok=sample.right_valid,
            head_ok=sample.head_valid,
        )
    else:
        left = np.full((4, 4), np.nan, dtype=np.float64)
        right = np.full((4, 4), np.nan, dtype=np.float64)
    hand_landmarks_valid = np.array(
        [
            bool(tracking and sample is not None and sample.left_hand_landmarks_valid),
            bool(tracking and sample is not None and sample.right_hand_landmarks_valid),
        ],
        dtype=bool,
    )
    hand_landmarks_left = (
        np.asarray(sample.left_hand_landmarks, dtype=np.float32).reshape(25, 3)
        if hand_landmarks_valid[0]
        and sample is not None
        and sample.left_hand_landmarks is not None
        else np.full((25, 3), np.nan, dtype=np.float32)
    )
    hand_landmarks_right = (
        np.asarray(sample.right_hand_landmarks, dtype=np.float32).reshape(25, 3)
        if hand_landmarks_valid[1]
        and sample is not None
        and sample.right_hand_landmarks is not None
        else np.full((25, 3), np.nan, dtype=np.float32)
    )
    values = {
        "wrist_left_waist": left.reshape(-1).astype(np.float32),
        "wrist_right_waist": right.reshape(-1).astype(np.float32),
        "targets_valid": np.array([valid], dtype=bool),
        "frame_index": np.array([frame_index], dtype=np.int64),
        "timestamp_monotonic": np.array(
            [0.0 if sample is None else sample.arrival_monotonic], dtype=np.float64
        ),
        "ee_frame_offset_m": np.array([0.05], dtype=np.float32),
        "stream_mode": np.array([STREAM_MODE_TRACKING], dtype=np.int32),
        "session_id": np.array([session_id], dtype=np.int32),
        "arm_reference_mode": np.array([0], dtype=np.int32),
        "left_trigger": np.array([0.0 if sample is None else sample.left_trigger], dtype=np.float32),
        "right_trigger": np.array([0.0 if sample is None else sample.right_trigger], dtype=np.float32),
        "left_grip": np.array([0.0 if sample is None else sample.left_grip], dtype=np.float32),
        "right_grip": np.array([0.0 if sample is None else sample.right_grip], dtype=np.float32),
        "hand_landmarks_schema_version": np.array(
            [HAND_LANDMARKS_SCHEMA_VERSION], dtype=np.int32
        ),
        "hand_landmarks_coordinate_frame": np.array(
            [HAND_LANDMARKS_FRAME_OPENXR_WORLD], dtype=np.int32
        ),
        "hand_landmarks_sequence": np.array([frame_index], dtype=np.int64),
        "hand_landmarks_timestamp_monotonic": np.array(
            [0.0 if sample is None else sample.arrival_monotonic], dtype=np.float64
        ),
        "hand_landmarks_valid": hand_landmarks_valid,
        "hand_landmarks_left": hand_landmarks_left,
        "hand_landmarks_right": hand_landmarks_right,
    }
    # Source-gap diagnostics are printed separately. Keep them out of this
    # fixed-header packet: consumers do not use them, and they can overflow the
    # legacy 1280-byte metadata header.
    socket.send(pack_pose_message(values, topic="vr_pose_debug"))


def main() -> int:
    args = parse_args()
    basis_correction = parse_basis_correction(args.basis_correction)
    source = TeleVuerSource(
        cert_file=args.televuer_cert,
        key_file=args.televuer_key,
        televuer_root=args.televuer_root,
        use_hand_tracking=True,
        basis_correction=basis_correction,
    )
    reader = XrStreamReader(source, poll_hz=args.source_poll_hz)
    context = zmq.Context()
    wrist_socket = context.socket(zmq.PUB)
    wrist_socket.setsockopt(zmq.SNDHWM, 4)
    wrist_socket.setsockopt(zmq.LINGER, 0)
    wrist_socket.bind(f"tcp://*:{args.vr_pose_debug_port}")

    print(
        "[GalaxyXR] raw WebXR hand tracking -> OpenXR SE(3) -> PATH; "
        "terminal [r/s/q] controls motion and recording",
        flush=True,
    )
    print(
        f"[GalaxyXR] PUB tcp://*:{args.vr_pose_debug_port}/vr_pose_debug "
        f"pose={args.pose_hz:.0f}Hz hand_tracking=on",
        flush=True,
    )

    reader.start()
    session_id = 1
    frame_index = 0
    pose_period = 1.0 / max(1.0, float(args.pose_hz))
    next_pose = next_report = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            sample = reader.latest()
            if now >= next_pose:
                frame_index += 1
                _send_wrist_targets(
                    wrist_socket,
                    sample,
                    tracking=source.connected,
                    session_id=session_id,
                    frame_index=frame_index,
                )
                next_pose += pose_period
                if next_pose < now - pose_period:
                    next_pose = now + pose_period
            if now >= next_report:
                report = reader.monitor.report()
                status = report if report else {"waiting_for_webxr_frame": True}
                print(
                    f"[GalaxyXR] connected={source.connected} "
                    f"hand_tracking={source.connected} {status}",
                    flush=True,
                )
                next_report = now + 2.0
            time.sleep(0.001)
    except KeyboardInterrupt:
        return 0
    finally:
        reader.stop()
        wrist_socket.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
