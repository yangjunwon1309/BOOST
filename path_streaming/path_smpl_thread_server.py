#!/usr/bin/env python3
"""Replay recorded SMPL through GMR and drive SONIC planner upper-body commands.

This is the PATH input unification server: SONIC receives only the existing
``planner`` high-level command stream. The upper body is filled from GMR-retargeted
recorded SMPL, while root navigation is held fixed (zero movement, idle mode).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)
from gear_sonic.scripts.gmr_realtime_ee_reference import (
    ROOT_OFFSET_WXYZ,
    build_human_frame,
)


UPPER_BODY_PLANNER_INDICES = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--gmr-root", type=Path, default=Path.home() / "GMR")
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--height", type=float, default=0.74)
    parser.add_argument("--human-height", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-planner", action="store_true")
    parser.add_argument("--command-period-sec", type=float, default=0.5)
    parser.add_argument("--warmup-sec", type=float, default=0.5)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[PATH-SMPL] {message}", flush=True)


def load_recording(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "session_selected.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    required = ["pose_smpl_pose", "pose_body_quat_w"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"recording missing required keys: {missing}")
    return {key: data[key] for key in data.files}


def extract_pose_window_last(array: np.ndarray, frame: int) -> np.ndarray:
    value = np.asarray(array[frame])
    if value.ndim >= 2:
        return np.asarray(value[-1])
    return value


def retarget_recording(data: dict[str, np.ndarray], gmr_root: Path, human_height: float | None) -> np.ndarray:
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting
    from smplx.joint_names import JOINT_NAMES
    import smplx

    torch.set_num_threads(1)
    body_model = smplx.create(
        gmr_root / "assets" / "body_models",
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    body_model.eval()
    with torch.inference_mode():
        rest_output = body_model(
            betas=torch.zeros((1, 10), dtype=torch.float32),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            body_pose=torch.zeros((1, 63), dtype=torch.float32),
            transl=torch.zeros((1, 3), dtype=torch.float32),
            left_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
            right_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32),
            expression=torch.zeros((1, 10), dtype=torch.float32),
            return_full_pose=True,
        )
    rest_joints = rest_output.joints[0].detach().cpu().numpy()
    parents = np.asarray(body_model.parents, dtype=np.int64)
    joint_names = list(JOINT_NAMES)
    retargeter = GeneralMotionRetargeting(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )

    smpl_pose = data["pose_smpl_pose"]
    body_quat = data["pose_body_quat_w"]
    n = min(len(smpl_pose), len(body_quat))
    q_refs = np.zeros((n, 29), dtype=np.float64)
    offset = R.from_quat(ROOT_OFFSET_WXYZ, scalar_first=True)
    for i in range(n):
        body_pose = extract_pose_window_last(smpl_pose, i).reshape(-1, 3)[:21]
        root_wxyz = extract_pose_window_last(body_quat, i).reshape(-1, 4)[-1].astype(np.float64)
        root_wxyz /= max(np.linalg.norm(root_wxyz), 1e-12)
        root_rotation = R.from_quat(root_wxyz, scalar_first=True) * offset.inv()
        human_frame = build_human_frame(body_pose, root_rotation, rest_joints, parents, joint_names)
        qpos = retargeter.retarget(human_frame, offset_to_ground=False)
        q_refs[i] = np.asarray(qpos[7:36], dtype=np.float64)
        if i > 0 and i % 250 == 0:
            log(f"retargeted {i}/{n}")
    return q_refs


def main() -> int:
    args = parse_args()
    recording = load_recording(args.recording_dir)
    gmr_root = args.gmr_root.expanduser().resolve()
    log(f"loading and retargeting {args.recording_dir}")
    q_refs = retarget_recording(recording, gmr_root, args.human_height)
    upper_refs = q_refs[:, UPPER_BODY_PLANNER_INDICES].astype(np.float32)
    log(f"ready frames={len(upper_refs)}, fps={args.fps:g}")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{args.bind_host}:{args.port}")
    log(f"publishing SONIC planner input tcp://{args.bind_host}:{args.port}/planner")
    time.sleep(max(0.0, args.warmup_sec))

    period = 1.0 / max(args.fps, 1e-6)
    next_command = 0.0
    frame = 0
    try:
      while True:
        now = time.monotonic()
        if args.start_planner and now >= next_command:
            socket.send(build_command_message(start=True, stop=False, planner=True))
            next_command = now + max(0.05, args.command_period_sec)

        upper = upper_refs[frame]
        msg = build_planner_message(
            0,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            speed=-1.0,
            height=args.height,
            upper_body_position=upper.tolist(),
            upper_body_velocity=np.zeros(17, dtype=np.float32).tolist(),
            left_hand_position=np.zeros(7, dtype=np.float32).tolist(),
            right_hand_position=np.zeros(7, dtype=np.float32).tolist(),
        )
        socket.send(msg)
        if frame % 250 == 0:
            log(f"sent frame={frame}/{len(upper_refs)}")
        frame += 1
        if frame >= len(upper_refs):
            if not args.loop:
                log("recording finished")
                return 0
            frame = 0
        time.sleep(period)
    except KeyboardInterrupt:
        return 0
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
