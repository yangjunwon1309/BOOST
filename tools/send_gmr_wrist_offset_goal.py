#!/usr/bin/env python3
"""Send a cuRobo path goal near the current GMR-retargeted wrist pose.

This is a demo trigger: it listens to the current GMR q_ref stream, computes the
target wrist position with MuJoCo FK in the base-local frame, adds a small
offset, and publishes a PATH ``path_goal`` message for the cuRobo planner.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmr-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--gmr-topic", default="gmr_ref")
    parser.add_argument("--goal-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--goal-topic", default="path_goal")
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("sonic_runtime/gear_sonic_deploy/g1/g1_29dof.xml"),
    )
    parser.add_argument("--target-link", default="left_wrist_yaw_link")
    parser.add_argument("--dx", type=float, default=0.03)
    parser.add_argument("--dy", type=float, default=0.00)
    parser.add_argument("--dz", type=float, default=0.00)
    parser.add_argument("--root-z", type=float, default=0.0)
    parser.add_argument(
        "--quaternion",
        nargs=4,
        type=float,
        default=[1.0, 0.0, 0.0, 0.0],
        help="Target wrist quaternion in wxyz. The demo keeps identity by default.",
    )
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--repeat-period-sec", type=float, default=0.05)
    return parser.parse_args()


def unpack_prefixed_msgpack(raw: bytes, topic: str) -> dict:
    topic_bytes = topic.encode()
    if not raw.startswith(topic_bytes):
        raise ValueError(f"unexpected topic; expected {topic}")
    return msgpack.unpackb(raw[len(topic_bytes):], raw=False)


def wait_for_gmr_q(socket: zmq.Socket, topic: str, timeout_sec: float) -> np.ndarray:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            payload = unpack_prefixed_msgpack(socket.recv(), topic)
        except zmq.Again:
            continue
        q = payload.get("q_ref", payload.get("q_mujoco"))
        if q is None:
            continue
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_arr.shape[0] == 29 and np.isfinite(q_arr).all():
            return q_arr
    raise TimeoutError(f"no valid {topic} q_ref received within {timeout_sec:.1f}s")


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
    wrist_world = data.xpos[body_id].copy()
    pelvis_world = data.xpos[pelvis_id].copy() if pelvis_id >= 0 else np.zeros(3)
    # The cuRobo G1 retarget model uses the base-local kinematic frame. Removing
    # pelvis translation keeps the goal in the same frame used by the QP analysis.
    return wrist_world - pelvis_world


def main() -> int:
    args = parse_args()
    ctx = zmq.Context.instance()

    gmr = ctx.socket(zmq.SUB)
    gmr.setsockopt(zmq.SUBSCRIBE, args.gmr_topic.encode())
    gmr.setsockopt(zmq.CONFLATE, 1)
    gmr.setsockopt(zmq.RCVTIMEO, 100)
    gmr.connect(args.gmr_endpoint)

    goal = ctx.socket(zmq.PUB)
    goal.setsockopt(zmq.LINGER, 0)
    goal.connect(args.goal_endpoint)
    time.sleep(args.settle_sec)

    q_ref = wait_for_gmr_q(gmr, args.gmr_topic, args.timeout_sec)
    current = wrist_position_from_q(args.model_xml, args.target_link, q_ref, args.root_z)
    target = current + np.array([args.dx, args.dy, args.dz], dtype=np.float64)
    payload = {
        "target_link": args.target_link,
        "position": target.tolist(),
        "quaternion": [float(x) for x in args.quaternion],
        "source": "gmr_fk_offset",
        "current_position": current.tolist(),
        "q_seed": q_ref.tolist(),
        "offset": [args.dx, args.dy, args.dz],
    }
    packed = args.goal_topic.encode() + msgpack.packb(payload, use_bin_type=True)
    for _ in range(max(1, args.repeat)):
        goal.send(packed)
        time.sleep(max(0.0, args.repeat_period_sec))
    print(
        f"[PATH-goal] {args.target_link} current={current.round(4).tolist()} "
        f"target={target.round(4).tolist()} offset={[args.dx, args.dy, args.dz]}"
    )
    print(f"[PATH-goal] sent {args.goal_topic} -> {args.goal_endpoint}")
    gmr.close()
    goal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
