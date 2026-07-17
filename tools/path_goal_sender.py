#!/usr/bin/env python3
"""Send a PATH EE goal to the cuRobo planning server.

Examples:
  # Absolute base-local target
  python tools/path_goal_sender.py --position 0.25 0.12 0.18

  # Target a few cm from the current GMR-retargeted left wrist
  python tools/path_goal_sender.py --from-gmr --dx 0.03 --dy 0.00 --dz 0.00

  # Target a few cm from the current/default right wrist
  python tools/path_goal_sender.py --from-gmr --side right --offset 0.00 -0.04 0.02

  # Same, but fall back to the current MuJoCo/G1 state if GMR is not running
  python tools/path_goal_sender.py --from-gmr --allow-state-fallback --dx 0.03
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import msgpack
import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--goal-topic", default="path_goal")
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
        "--force-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the planner to accept this user goal even inside the normal replan debounce window.",
    )
    return parser.parse_args()


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
        "body_q_measured",
        "body_q",
        "q_measured",
        "q",
        "ee_qp_q_corrected",
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
                    print(f"[PATH-goal] fallback seed from {topic}/{key}")
                    return q
    finally:
        socket.close()
    raise TimeoutError(f"no valid {topic} q on {endpoint} within {timeout_sec:.1f}s")


def default_q_seed() -> np.ndarray:
    return np.zeros(29, dtype=np.float64)


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


def main() -> int:
    args = parse_args()
    target_link = target_link_from_args(args)
    offset = offset_from_args(args)
    q_seed = np.asarray(args.q_seed, dtype=np.float64).reshape(29) if args.q_seed is not None else None
    current_position = (
        np.asarray(args.current_position, dtype=np.float64).reshape(3)
        if args.current_position is not None
        else None
    )

    if args.from_gmr:
        if q_seed is None:
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
        raise SystemExit("Provide --position X Y Z or use --from-gmr with --offset/--dx/--dy/--dz.")

    payload = {
        "goal_uid": uuid.uuid4().hex,
        "force_replan": bool(args.force_replan),
        "target_link": target_link,
        "side": args.side,
        "position": target.tolist(),
        "quaternion": [float(x) for x in args.quaternion],
        "source": "manual_gmr_offset" if args.from_gmr else "manual_absolute",
    }
    if current_position is not None:
        payload["current_position"] = current_position.tolist()
    if q_seed is not None:
        payload["q_seed"] = q_seed.tolist()
    if args.duration_sec is not None:
        payload["duration_sec"] = float(args.duration_sec)
    if args.final_hold_sec is not None:
        payload["final_hold_sec"] = float(args.final_hold_sec)
    if args.from_gmr:
        payload["offset"] = offset.astype(float).tolist()
        payload["seed_source"] = seed_source

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.goal_endpoint)
    time.sleep(max(0.0, args.settle_sec))
    packed = args.goal_topic.encode() + msgpack.packb(payload, use_bin_type=True)
    for _ in range(max(1, args.repeat)):
        socket.send(packed)
        time.sleep(max(0.0, args.repeat_period_sec))
    socket.close()
    print(
        f"[PATH-goal] sent {args.goal_topic} -> {args.goal_endpoint} "
        f"target={np.round(target, 4).tolist()} link={target_link}"
    )
    if current_position is not None:
        print(f"[PATH-goal] current={np.round(current_position, 4).tolist()}")
        print(f"[PATH-goal] offset={np.round(offset, 4).tolist()}")
    if args.from_gmr:
        print(f"[PATH-goal] seed_source={seed_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
