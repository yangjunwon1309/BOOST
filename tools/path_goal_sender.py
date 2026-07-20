#!/usr/bin/env python3
"""Send a PATH static EE target or planner goal.

Examples:
  # Absolute base-local static target for Cartesian EE servo
  python tools/path_goal_sender.py --position 0.25 0.12 0.18

  # Target a few cm from the current GMR-retargeted left wrist
  python tools/path_goal_sender.py --from-gmr --dx 0.03 --dy 0.00 --dz 0.00

  # Target a few cm from the current/default right wrist through GMR, with state fallback
  python tools/path_goal_sender.py --from-gmr --side right --offset 0.00 -0.04 0.02

  # Target a few cm from the current measured MuJoCo/G1 wrist directly
  python tools/path_goal_sender.py --from-state --side right --offset 0.00 -0.04 0.02

  # Same, but fall back to the current MuJoCo/G1 state if GMR is not running
  python tools/path_goal_sender.py --from-gmr --allow-state-fallback --dx 0.03
"""

from __future__ import annotations

import argparse
import atexit
import os
import select
import sys
import termios
import time
import tty
import uuid
from pathlib import Path

import msgpack
import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal-endpoint",
        default=f"tcp://127.0.0.1:{os.environ.get('PATH_STATIC_TARGET_PORT', '5564')}",
    )
    parser.add_argument(
        "--goal-topic",
        default=os.environ.get("PATH_STATIC_TARGET_TOPIC", "path_static_ee_target"),
    )
    parser.add_argument(
        "--planner-goal-endpoint",
        default=os.environ.get("PATH_PLANNER_GOAL_ENDPOINT", "tcp://127.0.0.1:5561"),
        help="Optional cuRobo planner goal endpoint used to compute q_ik prior.",
    )
    parser.add_argument(
        "--planner-goal-topic",
        default=os.environ.get("PATH_PLANNER_GOAL_TOPIC", "path_goal"),
    )
    parser.add_argument(
        "--send-planner-goal",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PATH_SEND_PLANNER_GOAL", "1") != "0",
        help=(
            "Also send the target to the cuRobo planner. The state-machine uses "
            "the returned q_goal as a QP joint prior when goal_uid matches."
        ),
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Convenience selector for left/right wrist target link.",
    )
    parser.add_argument(
        "--target-link",
        default=None,
        help="Explicit target body/link. Overrides --side when provided.",
    )
    parser.add_argument("--position", nargs=3, type=float, default=None)
    parser.add_argument(
        "--quaternion",
        nargs=4,
        type=float,
        default=[1.0, 0.0, 0.0, 0.0],
        help="Target EE quaternion in wxyz.",
    )
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--final-hold-sec", type=float, default=None)
    parser.add_argument(
        "--from-gmr",
        action="store_true",
        help="Use current GMR q_ref FK as current_position and add dx/dy/dz.",
    )
    parser.add_argument(
        "--from-state",
        action="store_true",
        help="Use current g1_debug measured q FK as current_position and add dx/dy/dz.",
    )
    parser.add_argument("--gmr-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--gmr-topic", default="gmr_ref")
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--state-topic", default="g1_debug")
    parser.add_argument(
        "--allow-state-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --from-gmr times out, use current g1_debug q as the seed.",
    )
    parser.add_argument(
        "--allow-default-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If neither GMR nor g1_debug is available, use a zero/default q seed.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("sonic_runtime/gear_sonic_deploy/g1/g1_29dof.xml"),
    )
    parser.add_argument(
        "--offset",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Base-local offset added to the current/default EE position.",
    )
    parser.add_argument("--dx", type=float, default=0.03)
    parser.add_argument("--dy", type=float, default=0.00)
    parser.add_argument("--dz", type=float, default=0.00)
    parser.add_argument("--root-z", type=float, default=0.0)
    parser.add_argument("--current-position", nargs=3, type=float, default=None)
    parser.add_argument("--q-seed", nargs=29, type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--repeat-period-sec", type=float, default=0.05)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Continuously stream the target pose and let the terminal keys "
            "w/a/s/d/q/e adjust it in robot-local xyz."
        ),
    )
    parser.add_argument("--stream-hz", type=float, default=50.0)
    parser.add_argument(
        "--step-cm",
        type=float,
        default=1.0,
        help="Interactive key step size in centimeters.",
    )
    parser.add_argument(
        "--stream-include-q-seed",
        action="store_true",
        help=(
            "Include the initial q_seed in every streaming goal. By default "
            "streaming goals let the planner use its latest robot state."
        ),
    )
    parser.add_argument(
        "--planner-seed-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For offset goals, ask the planner server to compute the fixed "
            "target as cuRobo_FK(initial_seed_q) + offset. This avoids mixing "
            "MuJoCo FK coordinates with cuRobo IK coordinates."
        ),
    )
    parser.add_argument("--print-period-sec", type=float, default=0.5)
    parser.add_argument(
        "--force-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the planner to accept this user goal even inside the normal replan debounce window.",
    )
    parser.add_argument(
        "--workspace-mode",
        choices=("clamp", "reject", "none"),
        default="clamp",
        help=(
            "Conservative pelvis-identity FK workspace guard for manual targets. "
            "This is a first-pass filter; cuRobo IK remains the exact feasibility check."
        ),
    )
    parser.add_argument("--workspace-x", nargs=2, type=float, default=[-0.15, 0.45])
    parser.add_argument("--workspace-left-y", nargs=2, type=float, default=[0.03, 0.50])
    parser.add_argument("--workspace-right-y", nargs=2, type=float, default=[-0.50, -0.03])
    parser.add_argument("--workspace-z", nargs=2, type=float, default=[-0.30, 0.35])
    parser.add_argument("--workspace-radius-m", type=float, default=0.62)
    return parser.parse_args()


class TerminalKeys:
    KEY_TO_DELTA = {
        "w": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "s": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "a": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "d": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "q": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "e": np.array([0.0, 0.0, 1.0], dtype=np.float64),
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

    def poll_delta(self) -> np.ndarray | None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        return self.KEY_TO_DELTA.get(key.lower())


def target_link_from_args(args: argparse.Namespace) -> str:
    if args.target_link:
        return str(args.target_link)
    return f"{args.side}_wrist_yaw_link"


def offset_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.offset is not None:
        return np.asarray(args.offset, dtype=np.float64).reshape(3)
    return np.array([args.dx, args.dy, args.dz], dtype=np.float64)


def unpack_prefixed(raw: bytes, topic: str) -> dict:
    prefix = topic.encode()
    if not raw.startswith(prefix):
        raise ValueError(f"unexpected topic; expected {topic}")
    return msgpack.unpackb(raw[len(prefix):], raw=False)


def wait_for_gmr_q(endpoint: str, topic: str, timeout_sec: float) -> np.ndarray:
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 100)
    socket.connect(endpoint)
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            try:
                payload = unpack_prefixed(socket.recv(), topic)
            except zmq.Again:
                continue
            q = payload.get("q_ref", payload.get("q_mujoco"))
            if q is None:
                continue
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            if q.shape[0] == 29 and np.isfinite(q).all():
                return q
    finally:
        socket.close()
    raise TimeoutError(f"no valid {topic} q_ref on {endpoint} within {timeout_sec:.1f}s")


def wait_for_state_q(endpoint: str, topic: str, timeout_sec: float) -> np.ndarray:
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 100)
    socket.connect(endpoint)
    q_keys = (
        "body_q",
        "q_measured",
        "q",
        "ee_qp_q_corrected",
        "body_q_measured",
    )
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            try:
                payload = unpack_prefixed(socket.recv(), topic)
            except zmq.Again:
                continue
            for key in q_keys:
                q = payload.get(key)
                if q is None:
                    continue
                q = np.asarray(q, dtype=np.float64).reshape(-1)
                if q.shape[0] == 29 and np.isfinite(q).all():
                    print(f"[PATH-goal] seed from {topic}/{key}")
                    return q
    finally:
        socket.close()
    raise TimeoutError(f"no valid {topic} q on {endpoint} within {timeout_sec:.1f}s")


def default_q_seed() -> np.ndarray:
    return np.zeros(29, dtype=np.float64)


def side_from_target_link(target_link: str, fallback_side: str) -> str:
    if "right" in target_link:
        return "right"
    if "left" in target_link:
        return "left"
    return fallback_side


def workspace_limits(args: argparse.Namespace, side: str) -> tuple[np.ndarray, np.ndarray]:
    y_limits = args.workspace_right_y if side == "right" else args.workspace_left_y
    lower = np.array([args.workspace_x[0], y_limits[0], args.workspace_z[0]], dtype=np.float64)
    upper = np.array([args.workspace_x[1], y_limits[1], args.workspace_z[1]], dtype=np.float64)
    return lower, upper


def apply_workspace_guard(
    args: argparse.Namespace,
    target: np.ndarray,
    target_link: str,
) -> tuple[np.ndarray, dict]:
    target = np.asarray(target, dtype=np.float64).reshape(3)
    side = side_from_target_link(target_link, args.side)
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
            "target outside conservative workspace "
            f"side={side} requested={np.round(target, 4).tolist()} "
            f"clamped={np.round(clamped, 4).tolist()} reason={info['workspace_reason']}"
        )
    return clamped, info


def resolve_seed_q(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    try:
        return wait_for_gmr_q(args.gmr_endpoint, args.gmr_topic, args.timeout_sec), "gmr"
    except TimeoutError as gmr_error:
        print(f"[PATH-goal] WARNING: {gmr_error}")
    if args.allow_state_fallback:
        try:
            return wait_for_state_q(args.state_endpoint, args.state_topic, args.timeout_sec), "state"
        except TimeoutError as state_error:
            print(f"[PATH-goal] WARNING: {state_error}")
    if args.allow_default_fallback:
        print("[PATH-goal] WARNING: using zero/default q seed")
        return default_q_seed(), "default"
    raise TimeoutError(
        "no seed q available; start GMR, start g1_debug, provide --q-seed, "
        "or allow default fallback"
    )


def wrist_position_from_q(model_xml: Path, target_link: str, q29: np.ndarray, root_z: float) -> np.ndarray:
    import mujoco

    model_path = model_xml.expanduser().resolve()
    old_cwd = os.getcwd()
    try:
        os.chdir(model_path.parent)
        model = mujoco.MjModel.from_xml_path(str(model_path))
    finally:
        os.chdir(old_cwd)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nq < 36:
        raise ValueError(f"expected free-root G1 model with nq>=36, got nq={model.nq}")
    data.qpos[:3] = np.array([0.0, 0.0, root_z], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[7:36] = np.asarray(q29, dtype=np.float64).reshape(29)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_link)
    if body_id < 0:
        raise ValueError(f"body not found in MuJoCo model: {target_link}")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis = data.xpos[pelvis_id].copy() if pelvis_id >= 0 else np.zeros(3)
    return data.xpos[body_id].copy() - pelvis


def make_goal_payload(
    args: argparse.Namespace,
    target_link: str,
    target: np.ndarray,
    current_position: np.ndarray | None,
    q_seed: np.ndarray | None,
    offset: np.ndarray,
    seed_source: str | None,
    streaming_goal: bool,
    workspace_info: dict | None = None,
    goal_uid: str | None = None,
) -> dict:
    payload = {
        "goal_uid": goal_uid or uuid.uuid4().hex,
        "force_replan": bool(args.force_replan),
        "streaming_goal": bool(streaming_goal),
        "target_link": target_link,
        "side": args.side,
        "position": np.asarray(target, dtype=np.float64).reshape(3).tolist(),
        "quaternion": [float(x) for x in args.quaternion],
        "source": (
            "interactive_state_offset"
            if streaming_goal and args.from_state
            else "interactive_gmr_offset"
            if streaming_goal and args.from_gmr
            else "interactive_absolute"
            if streaming_goal
            else "manual_state_offset"
            if args.from_state
            else "manual_gmr_offset"
            if args.from_gmr
            else "manual_absolute"
        ),
        "coordinate_frame": "pelvis_identity_fk",
    }
    if workspace_info is not None:
        payload.update(workspace_info)
    if current_position is not None:
        payload["current_position"] = np.asarray(current_position, dtype=np.float64).reshape(3).tolist()
    if q_seed is not None and ((not streaming_goal) or args.stream_include_q_seed):
        payload["q_seed"] = np.asarray(q_seed, dtype=np.float64).reshape(29).tolist()
    if q_seed is not None and args.planner_seed_offset and (args.from_gmr or args.from_state):
        payload["position_mode"] = "planner_seed_offset"
        payload["target_q_seed"] = np.asarray(q_seed, dtype=np.float64).reshape(29).tolist()
        payload["target_offset"] = np.asarray(offset, dtype=np.float64).reshape(3).tolist()
    if args.duration_sec is not None:
        payload["duration_sec"] = float(args.duration_sec)
    if args.final_hold_sec is not None:
        payload["final_hold_sec"] = float(args.final_hold_sec)
    if args.from_gmr or args.from_state:
        payload["offset"] = np.asarray(offset, dtype=np.float64).reshape(3).tolist()
        payload["seed_source"] = seed_source
    return payload


def send_goal(socket: zmq.Socket, topic: str, payload: dict) -> None:
    socket.send(topic.encode() + msgpack.packb(payload, use_bin_type=True))


def interactive_stream(
    args: argparse.Namespace,
    socket: zmq.Socket,
    planner_socket: zmq.Socket | None,
    target_link: str,
    target: np.ndarray,
    current_position: np.ndarray | None,
    q_seed: np.ndarray | None,
    offset: np.ndarray,
    seed_source: str | None,
) -> None:
    keys = TerminalKeys()
    step = max(0.0, float(args.step_cm)) * 0.01
    period = 1.0 / max(float(args.stream_hz), 1.0e-6)
    next_print = 0.0
    target = np.asarray(target, dtype=np.float64).reshape(3).copy()
    offset = np.asarray(offset, dtype=np.float64).reshape(3).copy()
    stream_goal_uid = uuid.uuid4().hex
    print(
        "[PATH-goal] interactive streaming started: "
        "w=+x, s=-x, a=+y, d=-y, q=-z, e=+z, Ctrl+C=stop"
    )
    try:
        while True:
            delta = keys.poll_delta()
            workspace_info = None
            if delta is not None:
                candidate_target = target + step * delta
                try:
                    target, workspace_info = apply_workspace_guard(
                        args, candidate_target, target_link)
                except ValueError as exc:
                    print(f"[PATH-goal] rejected target: {exc}", flush=True)
                    target = candidate_target - step * delta
                    target, workspace_info = apply_workspace_guard(
                        args, target, target_link)
                if current_position is not None:
                    offset = target - current_position
                else:
                    offset += step * delta
            else:
                target, workspace_info = apply_workspace_guard(args, target, target_link)
            payload = make_goal_payload(
                args,
                target_link,
                target,
                current_position,
                q_seed,
                offset,
                seed_source,
                streaming_goal=True,
                workspace_info=workspace_info,
                goal_uid=stream_goal_uid,
            )
            send_goal(socket, args.goal_topic, payload)
            if planner_socket is not None:
                send_goal(planner_socket, args.planner_goal_topic, payload)
            now = time.monotonic()
            if now >= next_print:
                print(
                    f"[PATH-goal] streaming target~={np.round(target, 4).tolist()} "
                    f"offset={np.round(offset, 4).tolist()} "
                    f"workspace={payload.get('workspace_reason', 'n/a')} "
                    f"link={target_link} step={args.step_cm:.2f}cm hz={args.stream_hz:.1f}",
                    flush=True,
                )
                next_print = now + max(0.1, float(args.print_period_sec))
            time.sleep(period)
    finally:
        keys.close()


def main() -> int:
    args = parse_args()
    if args.from_gmr and args.from_state:
        raise SystemExit("Use only one of --from-gmr or --from-state.")
    target_link = target_link_from_args(args)
    offset = offset_from_args(args)
    q_seed = np.asarray(args.q_seed, dtype=np.float64).reshape(29) if args.q_seed is not None else None
    current_position = (
        np.asarray(args.current_position, dtype=np.float64).reshape(3)
        if args.current_position is not None
        else None
    )

    if args.from_gmr or args.from_state:
        if q_seed is None:
            if args.from_state:
                q_seed = wait_for_state_q(args.state_endpoint, args.state_topic, args.timeout_sec)
                seed_source = "state"
            else:
                q_seed, seed_source = resolve_seed_q(args)
        else:
            seed_source = "cli_q_seed"
        current_position = wrist_position_from_q(
            args.model_xml, target_link, q_seed, args.root_z
        )
        target = current_position + offset
    elif args.position is not None:
        target = np.asarray(args.position, dtype=np.float64).reshape(3)
    else:
        raise SystemExit(
            "Provide --position X Y Z, --from-state, or --from-gmr with "
            "--offset/--dx/--dy/--dz."
        )
    try:
        target, workspace_info = apply_workspace_guard(args, target, target_link)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if current_position is not None:
        offset = target - current_position

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.goal_endpoint)
    planner_socket = None
    if args.send_planner_goal:
        planner_socket = ctx.socket(zmq.PUB)
        planner_socket.setsockopt(zmq.LINGER, 0)
        planner_socket.connect(args.planner_goal_endpoint)
    time.sleep(max(0.0, args.settle_sec))

    if args.interactive:
        interactive_stream(
            args,
            socket,
            planner_socket,
            target_link,
            target,
            current_position,
            q_seed,
            offset,
            seed_source if (args.from_gmr or args.from_state) else None,
        )
        socket.close()
        if planner_socket is not None:
            planner_socket.close()
        return 0

    payload = make_goal_payload(
        args,
        target_link,
        target,
        current_position,
        q_seed,
        offset,
        seed_source if (args.from_gmr or args.from_state) else None,
        streaming_goal=False,
        workspace_info=workspace_info,
        goal_uid=uuid.uuid4().hex,
    )
    for _ in range(max(1, args.repeat)):
        send_goal(socket, args.goal_topic, payload)
        if planner_socket is not None:
            send_goal(planner_socket, args.planner_goal_topic, payload)
        time.sleep(max(0.0, args.repeat_period_sec))
    socket.close()
    if planner_socket is not None:
        planner_socket.close()
    print(
        f"[PATH-goal] sent {args.goal_topic} -> {args.goal_endpoint} "
        f"target={np.round(target, 4).tolist()} link={target_link}"
    )
    if args.send_planner_goal:
        print(
            f"[PATH-goal] sent {args.planner_goal_topic} -> {args.planner_goal_endpoint} "
            "for cuRobo IK prior"
        )
    print(
        "[PATH-goal] workspace "
        f"mode={workspace_info.get('workspace_mode')} "
        f"feasible={workspace_info.get('kinematic_feasible')} "
        f"reason={workspace_info.get('workspace_reason')}"
    )
    if current_position is not None:
        print(f"[PATH-goal] current={np.round(current_position, 4).tolist()}")
        print(f"[PATH-goal] offset={np.round(offset, 4).tolist()}")
    if args.from_gmr or args.from_state:
        print(f"[PATH-goal] seed_source={seed_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
