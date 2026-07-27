#!/usr/bin/env python3
"""Publish one fixed SMPL-derived GMR q_ref for real shared-control tests.

This is a deliberate stand-in for ``gear_sonic/scripts/gmr_realtime_ee_reference.py``.
It loads an early SMPL frame from an exp_qp recording, retargets it once to G1,
then republishes the same 29-DoF q_ref at a fixed rate on the normal
``gmr_ref`` ZMQ topic.  The PATH state-machine bridge therefore sees a fresh,
steady "human is holding this upper-body posture" input even when no Pico/VR
stream is present.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R

from gear_sonic.scripts.gmr_realtime_ee_reference import (
    ROOT_OFFSET_WXYZ,
    build_human_frame,
)


DEFAULT_RECORDING_DIR = Path(
    "/home/url/GR00T-WholeBodyControl/gear_sonic/scripts/exp_qp_7"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=DEFAULT_RECORDING_DIR,
        help="Directory containing frame_XXXXXX.npz files.",
    )
    parser.add_argument(
        "--recording-frame",
        type=int,
        default=0,
        help="Which frame_XXXXXX.npz file to use. Default: 0.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help=(
            "Index inside pose_smpl_pose/body_quat_w window. "
            "0 is the earliest sample; -1 matches the usual realtime worker."
        ),
    )
    parser.add_argument(
        "--gmr-root",
        type=Path,
        default=Path(os.environ.get("PATH_GMR_ROOT", str(Path.home() / "GMR"))),
    )
    parser.add_argument("--human-height", type=float, default=None)
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--output-port", type=int, default=5558)
    parser.add_argument("--output-topic", default="gmr_ref")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--warmup-sec", type=float, default=0.5)
    parser.add_argument(
        "--zero-lower-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Set joints 0:12 to zero in the published q_ref. The state bridge "
            "uses this stream for waist/arms posture; lower body remains still."
        ),
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Retarget and print the selected q_ref, then exit without binding ZMQ.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[FixedGMR] {message}", flush=True)


def load_fixed_smpl_sample(
    recording_dir: Path,
    recording_frame: int,
    window_index: int,
) -> tuple[np.ndarray, np.ndarray, Path, int]:
    recording_dir = recording_dir.expanduser().resolve()
    path = recording_dir / f"frame_{int(recording_frame):06d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    for key in ("pose_smpl_pose", "pose_body_quat_w"):
        if key not in data:
            raise KeyError(f"{path} missing required key: {key}")
    smpl_window = np.asarray(data["pose_smpl_pose"], dtype=np.float32)
    quat_window = np.asarray(data["pose_body_quat_w"], dtype=np.float64)
    if smpl_window.ndim != 3 or smpl_window.shape[1:] != (21, 3):
        raise ValueError(f"unexpected pose_smpl_pose shape: {smpl_window.shape}")
    if quat_window.ndim != 2 or quat_window.shape[1] != 4:
        raise ValueError(f"unexpected pose_body_quat_w shape: {quat_window.shape}")
    count = min(len(smpl_window), len(quat_window))
    if count <= 0:
        raise ValueError(f"{path} contains no SMPL samples")
    selected = int(window_index)
    if selected < 0:
        selected = count + selected
    if selected < 0 or selected >= count:
        raise IndexError(
            f"window-index {window_index} out of range for {count} samples"
        )
    return smpl_window[selected], quat_window[selected], path, selected


def retarget_static_smpl(
    *,
    body_pose: np.ndarray,
    root_wxyz: np.ndarray,
    gmr_root: Path,
    human_height: float | None,
    zero_lower_body: bool,
) -> tuple[np.ndarray, float]:
    gmr_root = gmr_root.expanduser().resolve()
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

    root_wxyz = np.asarray(root_wxyz, dtype=np.float64).reshape(4)
    root_wxyz /= max(float(np.linalg.norm(root_wxyz)), 1.0e-12)
    offset = R.from_quat(ROOT_OFFSET_WXYZ, scalar_first=True)
    root_rotation = R.from_quat(root_wxyz, scalar_first=True) * offset.inv()
    human_frame = build_human_frame(
        np.asarray(body_pose, dtype=np.float32).reshape(21, 3),
        root_rotation,
        rest_joints,
        parents,
        joint_names,
    )

    started = time.perf_counter()
    qpos = retargeter.retarget(human_frame, offset_to_ground=False)
    solve_ms = (time.perf_counter() - started) * 1000.0
    q_ref = np.asarray(qpos[7:36], dtype=np.float64).reshape(29)
    if not np.isfinite(q_ref).all():
        raise ValueError("retargeted q_ref contains non-finite values")
    if zero_lower_body:
        q_ref[:12] = 0.0
    return q_ref, solve_ms


def pack_reference(topic: str, payload: dict) -> bytes:
    return topic.encode() + msgpack.packb(payload, use_bin_type=True)


def main() -> int:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")

    body_pose, root_wxyz, source_path, selected_window = load_fixed_smpl_sample(
        args.recording_dir,
        args.recording_frame,
        args.window_index,
    )
    q_ref, solve_ms = retarget_static_smpl(
        body_pose=body_pose,
        root_wxyz=root_wxyz,
        gmr_root=args.gmr_root,
        human_height=args.human_height,
        zero_lower_body=bool(args.zero_lower_body),
    )
    log(
        f"selected {source_path.name} window={selected_window} "
        f"retarget={solve_ms:.2f}ms zero_lower={args.zero_lower_body}"
    )
    log(f"q_ref[12:29]={np.round(q_ref[12:29], 4).tolist()}")

    if args.no_publish:
        return 0

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{args.bind_host}:{args.output_port}")
    log(
        f"publishing tcp://{args.bind_host}:{args.output_port}/"
        f"{args.output_topic} at {args.fps:g}Hz"
    )
    time.sleep(max(0.0, float(args.warmup_sec)))

    period = 1.0 / float(args.fps)
    frame_index = 0
    stream_started = time.monotonic()
    next_send = stream_started
    try:
        while True:
            now = time.monotonic()
            if now < next_send:
                time.sleep(next_send - now)
            timestamp_us = time.monotonic_ns() // 1000
            payload = {
                "source": "gmr",
                "control_mode": "teleop",
                "feasible": True,
                "plan_id": 0,
                "phase": 0.0,
                "frame_index": int(frame_index),
                "timestamp_us": int(timestamp_us),
                "solve_time_ms": float(solve_ms),
                "prediction_horizon_ms": 0.0,
                "q_ref": q_ref.tolist(),
                "q_mujoco": q_ref.tolist(),
                "static_reference": True,
                "source_recording": str(source_path),
                "source_window_index": int(selected_window),
            }
            socket.send(pack_reference(args.output_topic, payload))
            frame_index += 1
            if frame_index % max(1, int(args.fps * 5.0)) == 0:
                elapsed = max(time.monotonic() - stream_started, 1.0e-9)
                log(f"sent frame={frame_index} rate={frame_index / elapsed:.1f}Hz")
            next_send += period
            lag = time.monotonic() - next_send
            if lag > period:
                next_send = time.monotonic() + period
    except KeyboardInterrupt:
        return 0
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
