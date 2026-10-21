#!/usr/bin/env python3
"""Stream manual PATH shared-control EE targets with an explicit belief value.

This tool bypasses perception and publishes directly to the state machine's
static EE target input.  It is intended for validating e-mode shared control:
the state machine receives a target position plus a confidence/belief value,
and this message also carries neutral Q_READY left/right wrist poses.  A fresh
live PICO stream takes priority; otherwise these wrist poses make the sender a
self-contained shared-control input.

Examples:
  # Stream a left-wrist target with belief 0.8.
  python boost/teleop/legacy/tools/path_shared_control_goal_sender.py --side left --position 0.30 0.18 -0.05 --belief 0.8

  # Let the state machine choose the nearest hand instead of forcing left/right.
  python boost/teleop/legacy/tools/path_shared_control_goal_sender.py --side auto --position 0.30 0.00 -0.05 --belief 0.8

Keys in --interactive mode:
  w/s: x +/-   a/d: y +/-   e/q: z +/-   ]/[: belief +/-
  u/i: local EE roll CCW/CW   o/p: local EE yaw CCW/CW (5 deg default)
  l/r: force left/right hand   n: nearest-hand auto   0/1: belief 0/1
  c: publish invalid target once   Ctrl+C: stop
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import select
import sys
import termios
import time
import tty
import uuid

import msgpack
import numpy as np
import zmq


PICO_WAIST_FRAME = "pico_head_position_waist"

# G1 29-DoF Q_READY FK at the PATH wrist-yaw + 5 cm EE frames.  These are the
# neutral human wrist targets used when this standalone sender is run without
# a live PICO stream.
Q_READY_LEFT_EE_POSITION = [
    0.24977428385080602,
    0.1486521246660517,
    0.09523008429534556,
]
Q_READY_RIGHT_EE_POSITION = [
    0.24977428385080602,
    -0.1486421246660517,
    0.09523008429534556,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal-endpoint",
        default=f"tcp://127.0.0.1:{os.environ.get('PATH_STATIC_TARGET_PORT', '5564')}",
        help="State-machine static EE target endpoint.",
    )
    parser.add_argument(
        "--goal-topic",
        default=os.environ.get("PATH_STATIC_TARGET_TOPIC", "path_static_ee_target"),
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "auto"),
        default="auto",
        help=(
            "Target hand. 'auto' omits a left/right target_link so the shared-control "
            "blender selects the nearest measured hand."
        ),
    )
    parser.add_argument(
        "--target-link",
        default="",
        help="Explicit target link. Overrides --side when set.",
    )
    parser.add_argument(
        "--position",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help=(
            "Initial object/contact target in the canonical PICO head-position "
            "waist frame, in meters."
        ),
    )
    parser.add_argument(
        "--quaternion",
        nargs=4,
        type=float,
        default=[1.0, 0.0, 0.0, 0.0],
        metavar=("W", "X", "Y", "Z"),
        help="Explicit IPOPT EE target quaternion in the canonical waist frame (wxyz).",
    )
    parser.add_argument(
        "--belief",
        type=float,
        default=0.8,
        help="Shared-control belief/confidence value. alpha ramps from rho1 to rho2.",
    )
    parser.add_argument("--object-id", default="manual_shared_control_target")
    parser.add_argument("--label", default="manual")
    parser.add_argument("--grip-index", type=int, default=0)
    parser.add_argument("--stream-hz", type=float, default=20.0)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat-period-sec", type=float, default=0.05)
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously stream and accept keyboard edits.",
    )
    parser.add_argument("--step-cm", type=float, default=1.0)
    parser.add_argument(
        "--orientation-step-deg",
        type=float,
        default=5.0,
        help="Interactive U/I roll and O/P yaw increment in degrees (default: 5).",
    )
    parser.add_argument("--belief-step", type=float, default=0.05)
    parser.add_argument(
        "--left-wrist-position",
        nargs=3,
        type=float,
        default=Q_READY_LEFT_EE_POSITION,
        metavar=("X", "Y", "Z"),
        help="Fallback left human wrist target in the canonical PICO waist frame.",
    )
    parser.add_argument(
        "--right-wrist-position",
        nargs=3,
        type=float,
        default=Q_READY_RIGHT_EE_POSITION,
        metavar=("X", "Y", "Z"),
        help="Fallback right human wrist target in the canonical PICO waist frame.",
    )
    parser.add_argument(
        "--left-wrist-quaternion",
        nargs=4,
        type=float,
        default=[1.0, 0.0, 0.0, 0.0],
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument(
        "--right-wrist-quaternion",
        nargs=4,
        type=float,
        default=[1.0, 0.0, 0.0, 0.0],
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument("--print-period-sec", type=float, default=0.5)
    parser.add_argument("--workspace-mode", choices=("clamp", "reject", "none"), default="clamp")
    parser.add_argument("--workspace-x", nargs=2, type=float, default=[-0.15, 0.50])
    parser.add_argument("--workspace-left-y", nargs=2, type=float, default=[0.02, 0.52])
    parser.add_argument("--workspace-right-y", nargs=2, type=float, default=[-0.52, -0.02])
    parser.add_argument("--workspace-auto-y", nargs=2, type=float, default=[-0.52, 0.52])
    parser.add_argument("--workspace-z", nargs=2, type=float, default=[-0.20, 0.35])
    parser.add_argument("--workspace-radius-m", type=float, default=0.70)
    parser.add_argument(
        "--clear-on-exit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Publish one invalid target when the tool exits.",
    )
    return parser.parse_args()


class TerminalKeys:
    DELTAS = {
        "w": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "s": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "a": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "d": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "e": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "q": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("--interactive requires a TTY stdin")
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        atexit.register(self.close)

    def close(self) -> None:
        if self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.old_settings = None

    def poll(self) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        return key.lower()


def normalize_quat_wxyz(values: list[float] | np.ndarray) -> list[float]:
    quat = np.asarray(values, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12 or not np.isfinite(quat).all():
        raise ValueError(f"invalid quaternion: {quat.tolist()}")
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat.tolist()


def quat_multiply_wxyz(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
) -> np.ndarray:
    """Hamilton product for wxyz quaternions, returned normalized."""
    aw, ax, ay, az = normalize_quat_wxyz(left)
    bw, bx, by, bz = normalize_quat_wxyz(right)
    return np.asarray(
        normalize_quat_wxyz(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ]
        ),
        dtype=np.float64,
    )


def local_x_roll_quaternion(delta_rad: float) -> np.ndarray:
    half = 0.5 * float(delta_rad)
    return np.array([math.cos(half), math.sin(half), 0.0, 0.0], dtype=np.float64)


def local_z_yaw_quaternion(delta_rad: float) -> np.ndarray:
    half = 0.5 * float(delta_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def transform_from_position_quaternion(
    position: list[float] | np.ndarray,
    quaternion_wxyz: list[float] | np.ndarray,
) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quaternion_wxyz)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def target_link_for_side(side: str, explicit: str = "") -> str:
    if explicit:
        return str(explicit)
    if side == "left":
        return "left_wrist_yaw_link"
    if side == "right":
        return "right_wrist_yaw_link"
    return ""


def side_from_target_link(target_link: str, fallback: str) -> str:
    name = str(target_link).lower()
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    return fallback


def workspace_limits(args: argparse.Namespace, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "left":
        y_limits = args.workspace_left_y
    elif side == "right":
        y_limits = args.workspace_right_y
    else:
        y_limits = args.workspace_auto_y
    lower = np.array([args.workspace_x[0], y_limits[0], args.workspace_z[0]], dtype=np.float64)
    upper = np.array([args.workspace_x[1], y_limits[1], args.workspace_z[1]], dtype=np.float64)
    return lower, upper


def apply_workspace_guard(
    args: argparse.Namespace,
    target: np.ndarray,
    side: str,
) -> tuple[np.ndarray, dict]:
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if args.workspace_mode == "none":
        return target.copy(), {
            "workspace_mode": "none",
            "kinematic_feasible": True,
            "workspace_clamped": False,
            "workspace_reason": "disabled",
            "workspace_side": side,
            "requested_position": target.tolist(),
        }
    lower, upper = workspace_limits(args, side)
    clamped = np.minimum(np.maximum(target, lower), upper)
    reasons: list[str] = []
    if not np.allclose(clamped, target, atol=1.0e-12):
        reasons.append("box")
    radius = max(1.0e-6, float(args.workspace_radius_m))
    norm = float(np.linalg.norm(clamped))
    if norm > radius:
        clamped = clamped * (radius / norm)
        clamped = np.minimum(np.maximum(clamped, lower), upper)
        reasons.append("radius")
    feasible = not reasons
    info = {
        "workspace_mode": args.workspace_mode,
        "kinematic_feasible": feasible,
        "workspace_clamped": not feasible,
        "workspace_reason": ",".join(reasons) if reasons else "ok",
        "workspace_side": side,
        "workspace_lower": lower.tolist(),
        "workspace_upper": upper.tolist(),
        "workspace_radius_m": radius,
        "requested_position": target.tolist(),
    }
    if args.workspace_mode == "reject" and not feasible:
        raise ValueError(
            "target outside workspace "
            f"side={side} requested={np.round(target, 4).tolist()} "
            f"clamped={np.round(clamped, 4).tolist()} reason={info['workspace_reason']}"
        )
    return clamped, info


def make_payload(
    args: argparse.Namespace,
    *,
    position: np.ndarray,
    quaternion: list[float] | np.ndarray,
    side: str,
    target_link: str,
    belief: float,
    goal_uid: str,
    target_valid: bool = True,
    workspace_info: dict | None = None,
) -> dict:
    belief = float(np.clip(float(belief), 0.0, 1.0))
    now_us = int(time.time_ns() // 1000)
    left_wrist_transform = transform_from_position_quaternion(
        args.left_wrist_position,
        args.left_wrist_quaternion,
    )
    right_wrist_transform = transform_from_position_quaternion(
        args.right_wrist_position,
        args.right_wrist_quaternion,
    )
    payload = {
        "goal_uid": goal_uid,
        "force_replan": False,
        "streaming_goal": True,
        "target_valid": bool(target_valid),
        "target_link": str(target_link),
        "side": str(side),
        "position": np.asarray(position, dtype=np.float64).reshape(3).tolist(),
        "quaternion": normalize_quat_wxyz(quaternion),
        "orientation_valid": True,
        "source": "manual_shared_control_belief",
        "coordinate_frame": PICO_WAIST_FRAME,
        "target_position_frame": PICO_WAIST_FRAME,
        "human_wrist_frame": PICO_WAIST_FRAME,
        "human_left_wrist_transform": left_wrist_transform.reshape(-1).tolist(),
        "human_right_wrist_transform": right_wrist_transform.reshape(-1).tolist(),
        "human_left_wrist_position": left_wrist_transform[:3, 3].tolist(),
        "human_right_wrist_position": right_wrist_transform[:3, 3].tolist(),
        "confidence": belief,
        "object_confidence": belief,
        "joint_confidence": belief,
        "tasc_joint_confidence": belief,
        "object_id": str(args.object_id),
        "label": str(args.label),
        "grip_index": int(args.grip_index),
        "published_timestamp_us": now_us,
        "metadata": {
            "source": "path_shared_control_goal_sender",
            "manual_belief": belief,
            "manual_side": side,
            "manual_target_link": target_link,
            "orientation_valid": True,
            "streaming_goal": True,
            "coordinate_frame": PICO_WAIST_FRAME,
            "human_wrist_source": "g1_q_ready_fk",
        },
    }
    if workspace_info:
        payload.update(workspace_info)
    return payload


def send_goal(socket: zmq.Socket, topic: str, payload: dict) -> None:
    socket.send(topic.encode("utf-8") + msgpack.packb(payload, use_bin_type=True))


def print_status(
    *,
    args: argparse.Namespace,
    position: np.ndarray,
    quaternion: list[float] | np.ndarray,
    belief: float,
    side: str,
    target_link: str,
    workspace_reason: str,
) -> None:
    hand = side if side in {"left", "right"} else "nearest"
    print(
        "[PATH-shared-goal] "
        f"target={np.round(position, 4).tolist()} "
        f"quat={np.round(normalize_quat_wxyz(quaternion), 4).tolist()} "
        f"belief={belief:.3f} hand={hand} "
        f"link={target_link or '<auto-nearest>'} "
        f"frame={PICO_WAIST_FRAME} "
        f"workspace={workspace_reason} "
        f"topic={args.goal_topic} hz={args.stream_hz:.1f}",
        flush=True,
    )


def interactive_loop(args: argparse.Namespace, socket: zmq.Socket) -> None:
    keys = TerminalKeys()
    step = max(0.0, float(args.step_cm)) * 0.01
    belief_step = max(0.0, float(args.belief_step))
    period = 1.0 / max(1.0e-6, float(args.stream_hz))
    position = np.asarray(args.position, dtype=np.float64).reshape(3).copy()
    quaternion = np.asarray(normalize_quat_wxyz(args.quaternion), dtype=np.float64)
    orientation_step = math.radians(max(0.0, float(args.orientation_step_deg)))
    side = str(args.side)
    belief = float(np.clip(float(args.belief), 0.0, 1.0))
    goal_uid = f"manual_shared:{uuid.uuid4().hex}"
    next_print = 0.0
    print(
        "[PATH-shared-goal] interactive: "
        "w/s=x +/- a/d=y +/- e/q=z +/- ]/[=belief +/- "
        "u/i=roll CCW/CW o/p=yaw CCW/CW "
        "l=left r=right n=nearest 0/1=belief c=clear Ctrl+C=stop"
    )
    try:
        while True:
            key = keys.poll()
            publish_valid = True
            if key in TerminalKeys.DELTAS:
                position = position + step * TerminalKeys.DELTAS[key]
            elif key == "u":
                quaternion = quat_multiply_wxyz(
                    quaternion, local_x_roll_quaternion(+orientation_step)
                )
            elif key == "i":
                quaternion = quat_multiply_wxyz(
                    quaternion, local_x_roll_quaternion(-orientation_step)
                )
            elif key == "o":
                quaternion = quat_multiply_wxyz(
                    quaternion, local_z_yaw_quaternion(+orientation_step)
                )
            elif key == "p":
                quaternion = quat_multiply_wxyz(
                    quaternion, local_z_yaw_quaternion(-orientation_step)
                )
            elif key == "]":
                belief = min(1.0, belief + belief_step)
            elif key == "[":
                belief = max(0.0, belief - belief_step)
            elif key == "0":
                belief = 0.0
            elif key == "1":
                belief = 1.0
            elif key == "l":
                side = "left"
            elif key == "r":
                side = "right"
            elif key == "n":
                side = "auto"
            elif key == "c":
                publish_valid = False

            target_link = target_link_for_side(side, args.target_link)
            guard_side = side_from_target_link(target_link, side)
            if publish_valid:
                try:
                    position, workspace_info = apply_workspace_guard(args, position, guard_side)
                except ValueError as exc:
                    print(f"[PATH-shared-goal] rejected target: {exc}", flush=True)
                    workspace_info = {
                        "workspace_mode": args.workspace_mode,
                        "workspace_reason": "rejected",
                    }
                    publish_valid = False
            else:
                workspace_info = {
                    "workspace_mode": args.workspace_mode,
                    "workspace_reason": "manual_clear",
                }
            payload = make_payload(
                args,
                position=position,
                quaternion=quaternion,
                side=side,
                target_link=target_link,
                belief=belief,
                goal_uid=goal_uid,
                target_valid=publish_valid,
                workspace_info=workspace_info,
            )
            send_goal(socket, args.goal_topic, payload)
            now = time.monotonic()
            if now >= next_print or key is not None:
                print_status(
                    args=args,
                    position=position,
                    quaternion=quaternion,
                    belief=belief,
                    side=side,
                    target_link=target_link,
                    workspace_reason=str(payload.get("workspace_reason", "n/a")),
                )
                next_print = now + max(0.1, float(args.print_period_sec))
            time.sleep(period)
    finally:
        keys.close()
        if args.clear_on_exit:
            payload = make_payload(
                args,
                position=position,
                quaternion=quaternion,
                side=side,
                target_link=target_link_for_side(side, args.target_link),
                belief=0.0,
                goal_uid=goal_uid,
                target_valid=False,
                workspace_info={"workspace_mode": "none", "workspace_reason": "exit_clear"},
            )
            send_goal(socket, args.goal_topic, payload)


def main() -> int:
    args = parse_args()
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.goal_endpoint)
    time.sleep(max(0.0, float(args.settle_sec)))

    if args.interactive:
        interactive_loop(args, socket)
        socket.close()
        return 0

    position = np.asarray(args.position, dtype=np.float64).reshape(3)
    target_link = target_link_for_side(args.side, args.target_link)
    guard_side = side_from_target_link(target_link, args.side)
    position, workspace_info = apply_workspace_guard(args, position, guard_side)
    payload = make_payload(
        args,
        position=position,
        quaternion=args.quaternion,
        side=args.side,
        target_link=target_link,
        belief=args.belief,
        goal_uid=f"manual_shared:{uuid.uuid4().hex}",
        target_valid=True,
        workspace_info=workspace_info,
    )
    for _ in range(max(1, int(args.repeat))):
        send_goal(socket, args.goal_topic, payload)
        time.sleep(max(0.0, float(args.repeat_period_sec)))
    print_status(
        args=args,
        position=position,
        quaternion=args.quaternion,
        belief=float(args.belief),
        side=args.side,
        target_link=target_link,
        workspace_reason=str(payload.get("workspace_reason", "n/a")),
    )
    socket.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
