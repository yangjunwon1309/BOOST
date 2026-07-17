#!/usr/bin/env python3
"""Warm-started real-time SMPL-X to G1 reference publisher for EE QP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R


POSE_HEADER_SIZE = 1280
ROOT_OFFSET_WXYZ = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)


def unpack_pose(raw: bytes, topic: str) -> dict[str, np.ndarray]:
    topic_bytes = topic.encode()
    if not raw.startswith(topic_bytes):
        raise ValueError("unexpected pose topic")
    offset = len(topic_bytes)
    header_bytes = raw[offset : offset + POSE_HEADER_SIZE]
    header_bytes = header_bytes.split(b"\0", 1)[0]
    header = json.loads(header_bytes.decode())
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": np.bool_,
    }
    result: dict[str, np.ndarray] = {}
    cursor = offset + POSE_HEADER_SIZE
    for field in header.get("fields", []):
        dtype = np.dtype(dtype_map[field["dtype"]])
        shape = tuple(field["shape"])
        size = int(np.prod(shape)) * dtype.itemsize
        result[field["name"]] = np.frombuffer(
            raw[cursor : cursor + size], dtype=dtype
        ).reshape(shape).copy()
        cursor += size
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmr-root", type=Path, default=Path.home() / "GMR")
    parser.add_argument("--pose-host", default="127.0.0.1")
    parser.add_argument("--pose-port", type=int, default=5556)
    parser.add_argument("--pose-topic", default="pose")
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--output-port", type=int, default=5558)
    parser.add_argument("--output-topic", default="gmr_ref")
    parser.add_argument("--human-height", type=float, default=None)
    parser.add_argument("--source-rate-hz", type=float, default=50.0)
    parser.add_argument("--output-rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--pose-timeout-sec",
        type=float,
        default=1.0,
        help="Reset warm-start state after this long without pose messages.",
    )
    parser.add_argument(
        "--frame-rewind-reset-threshold",
        type=int,
        default=5,
        help="Treat a frame-index rewind of at least this many frames as a new stream.",
    )
    return parser.parse_args()


def extrapolate_human_frame(
    previous: dict[str, tuple[np.ndarray, np.ndarray]],
    current: dict[str, tuple[np.ndarray, np.ndarray]],
    ratio: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Constant-velocity extrapolation of GMR link targets."""
    predicted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (current_position, current_wxyz) in current.items():
        previous_position, previous_wxyz = previous[name]
        position = current_position + ratio * (
            current_position - previous_position
        )
        previous_rotation = R.from_quat(previous_wxyz, scalar_first=True)
        current_rotation = R.from_quat(current_wxyz, scalar_first=True)
        delta = current_rotation * previous_rotation.inv()
        predicted_rotation = (
            R.from_rotvec(ratio * delta.as_rotvec()) * current_rotation
        )
        predicted[name] = (
            np.asarray(position, dtype=np.float64),
            predicted_rotation.as_quat(scalar_first=True),
        )
    return predicted


def build_human_frame(
    body_pose: np.ndarray,
    root_rotation: R,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    joint_names: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the SMPL-X link targets with exact rigid-skeleton FK."""
    joint_count = min(24, len(rest_joints), len(parents), len(joint_names))
    rotations = [root_rotation]
    positions = [np.asarray(rest_joints[0], dtype=np.float64)]
    for index in range(1, joint_count):
        parent = int(parents[index])
        local_rotvec = body_pose[index - 1] if index <= 21 else np.zeros(3)
        rotations.append(rotations[parent] * R.from_rotvec(local_rotvec))
        rest_offset = rest_joints[index] - rest_joints[parent]
        positions.append(
            positions[parent] + rotations[parent].apply(rest_offset)
        )
    return {
        joint_names[index]: (
            positions[index], rotations[index].as_quat(scalar_first=True)
        )
        for index in range(joint_count)
    }


def main() -> None:
    args = parse_args()
    if args.source_rate_hz <= 0.0 or args.output_rate_hz <= 0.0:
        raise ValueError("source and output rates must be positive")
    if args.output_rate_hz < args.source_rate_hz:
        raise ValueError("output rate must be at least the source rate")
    gmr_root = args.gmr_root.expanduser().resolve()
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
    def make_retargeter() -> GeneralMotionRetargeting:
        return GeneralMotionRetargeting(
            actual_human_height=args.human_height,
            src_human="smplx",
            tgt_robot="unitree_g1",
            verbose=False,
        )

    retargeter = make_retargeter()

    context = zmq.Context()
    pose_socket = context.socket(zmq.SUB)
    pose_socket.setsockopt(zmq.CONFLATE, 1)
    pose_socket.setsockopt(zmq.SUBSCRIBE, args.pose_topic.encode())
    pose_socket.setsockopt(zmq.RCVTIMEO, max(1, int(args.pose_timeout_sec * 1000.0)))
    pose_socket.connect(f"tcp://{args.pose_host}:{args.pose_port}")

    output_socket = context.socket(zmq.PUB)
    output_socket.setsockopt(zmq.SNDHWM, 1)
    output_socket.setsockopt(zmq.LINGER, 0)
    output_socket.bind(f"tcp://{args.bind_host}:{args.output_port}")

    print(
        f"[GMR-EE] pose tcp://{args.pose_host}:{args.pose_port}/{args.pose_topic}"
    )
    print(
        f"[GMR-EE] publishing tcp://{args.bind_host}:{args.output_port}/"
        f"{args.output_topic} at {args.output_rate_hz:g}Hz "
        f"(source={args.source_rate_hz:g}Hz, predicted subframes enabled)"
    )

    offset = R.from_quat(ROOT_OFFSET_WXYZ, scalar_first=True)
    last_frame = -1
    count = 0
    predicted_count = 0
    stream_started = None
    previous_human_frame = None
    previous_frame_index = -1
    output_period = 1.0 / args.output_rate_hz
    last_pose_wall = time.monotonic()
    stream_active = False
    reset_count = 0

    def reset_stream_state(reason: str, *, recreate_retargeter: bool) -> None:
        nonlocal retargeter
        nonlocal last_frame, previous_human_frame, previous_frame_index
        nonlocal stream_started, stream_active, reset_count
        last_frame = -1
        previous_human_frame = None
        previous_frame_index = -1
        stream_started = None
        stream_active = False
        reset_count += 1
        if recreate_retargeter:
            retargeter = make_retargeter()
        print(
            f"[GMR-EE] stream reset #{reset_count}: {reason}"
            f"{' (retargeter recreated)' if recreate_retargeter else ''}"
        )

    try:
        while True:
            try:
                raw = pose_socket.recv()
            except zmq.Again:
                silence_sec = time.monotonic() - last_pose_wall
                if stream_active and silence_sec >= args.pose_timeout_sec:
                    reset_stream_state(
                        f"pose timeout after {silence_sec:.2f}s",
                        recreate_retargeter=True,
                    )
                continue
            try:
                pose = unpack_pose(raw, args.pose_topic)
                last_pose_wall = time.monotonic()
                frame_index = int(np.asarray(pose["frame_index"]).reshape(-1)[-1])
                if frame_index <= last_frame:
                    rewind = last_frame - frame_index
                    if rewind >= args.frame_rewind_reset_threshold:
                        reset_stream_state(
                            f"frame index rewind {last_frame}->{frame_index}",
                            recreate_retargeter=True,
                        )
                    else:
                        continue

                if not stream_active:
                    print(f"[GMR-EE] stream resumed at frame={frame_index}")
                    stream_active = True

                if frame_index <= last_frame:
                    continue

                body_pose = np.asarray(pose["smpl_pose"], dtype=np.float32)[-1]
                root_wxyz = np.asarray(pose["body_quat_w"], dtype=np.float64)[-1]
                root_wxyz /= max(np.linalg.norm(root_wxyz), 1e-12)
                root_rotation = R.from_quat(root_wxyz, scalar_first=True) * offset.inv()
                human_frame = build_human_frame(
                    body_pose, root_rotation, rest_joints, parents, joint_names
                )

                observed_source_interval = (
                    (frame_index - previous_frame_index) / args.source_rate_hz
                    if previous_frame_index >= 0
                    else 1.0 / args.source_rate_hz
                )
                prediction_window = 1.0 / args.source_rate_hz
                references = [(0.0, human_frame)]
                if (previous_human_frame is not None and
                        observed_source_interval > 0.0):
                    horizon = output_period
                    while horizon < prediction_window - 1e-9:
                        references.append(
                            (
                                horizon,
                                extrapolate_human_frame(
                                    previous_human_frame,
                                    human_frame,
                                    horizon / observed_source_interval,
                                ),
                            )
                        )
                        horizon += output_period

                cycle_started = time.monotonic()
                if stream_started is None:
                    stream_started = cycle_started
                for prediction_horizon, reference_frame in references:
                    solve_started = time.perf_counter()
                    qpos = retargeter.retarget(
                        reference_frame, offset_to_ground=False
                    )
                    solve_ms = (time.perf_counter() - solve_started) * 1000.0
                    q_mujoco = np.asarray(qpos[7:36], dtype=np.float64)
                    if q_mujoco.shape != (29,) or not np.isfinite(q_mujoco).all():
                        continue

                    send_at = cycle_started + prediction_horizon
                    remaining = send_at - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    payload = msgpack.packb(
                        {
                            "source": "gmr",
                            "control_mode": "teleop",
                            "feasible": True,
                            "plan_id": 0,
                            "phase": 0.0,
                            "frame_index": frame_index,
                            "timestamp_us": time.monotonic_ns() // 1000,
                            "solve_time_ms": solve_ms,
                            "prediction_horizon_ms": prediction_horizon * 1000.0,
                            "q_ref": q_mujoco.tolist(),
                            "q_mujoco": q_mujoco.tolist(),
                        },
                        use_bin_type=True,
                    )
                    output_socket.send(args.output_topic.encode() + payload)
                    count += 1
                    if prediction_horizon > 0.0:
                        predicted_count += 1

                previous_human_frame = human_frame
                previous_frame_index = frame_index
                last_frame = frame_index
                if count > 0 and count % 100 < len(references):
                    elapsed = max(time.monotonic() - stream_started, 1e-6)
                    print(
                        f"[GMR-EE] frame={frame_index} solve={solve_ms:.2f}ms "
                        f"output={count / elapsed:.1f}Hz "
                        f"predicted={predicted_count}/{count}"
                    )
            except Exception as exc:
                print(f"[GMR-EE] skipped frame: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        pose_socket.close()
        output_socket.close()
        context.term()


if __name__ == "__main__":
    main()
