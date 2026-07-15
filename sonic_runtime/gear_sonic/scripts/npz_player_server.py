"""
npz_player_server.py
Kimodo _amass.npz -> SONIC WBC ZMQ player

Coordinate transforms use SONIC's own functions (same as pico_manager):
  1. smpl_root_ytoz_up     (Y-up -> Z-up)
  2. compute_human_joints  (SMPL FK, 24 joints)
  3. remove_smpl_base_rot  (SMPL T-pose base removal)
  4. smpl_joints_local     = quat_inv(q) @ joints

Must be run inside SONIC venv:
    source ~/GR00T-WholeBodyControl/.venv_sim/bin/activate

Usage:
    python3 npz_player_server.py --npz /path/to/test_amass.npz --port 5556 --loop

] key: rewind motion to frame 0 while frame_index keeps incrementing
"""

import argparse
import json
import struct
import time
import threading
from pathlib import Path

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa

# SONIC's own transform functions (same as pico_manager)
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)
from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up

HEADER_SIZE = 1280


# ─── Wire format ────────────────────────────────────────────────────────────

def _build_header(fields, version=3, count=1):
    h = {"v": version, "endian": "le", "count": count, "fields": fields}
    hj = json.dumps(h, separators=(",", ":")).encode("utf-8")
    if len(hj) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(hj)}")
    return hj.ljust(HEADER_SIZE, b"\x00")


def pack_pose_message(pose_data, topic="pose", version=3):
    fields, binary_data = [], []
    dtype_map = {
        np.float32: "f32", np.float64: "f64",
        np.int32: "i32", np.int64: "i64",
        bool: "bool", np.bool_: "bool",
    }
    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            dt = dtype_map.get(value.dtype.type, "f32")
            if value.dtype.type not in dtype_map:
                value = value.astype(np.float32)
            fields.append({"name": key, "dtype": dt, "shape": list(value.shape)})
            if not value.flags["C_CONTIGUOUS"]:
                value = np.ascontiguousarray(value)
            binary_data.append(value.tobytes())
    return topic.encode() + _build_header(fields, version=version) + b"".join(binary_data)


def build_command_message(start, stop, planner):
    fields = [
        {"name": "start",   "dtype": "u8", "shape": [1]},
        {"name": "stop",    "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = struct.pack("BBB",
        1 if start else 0,
        1 if stop else 0,
        1 if planner else 0,
    )
    return b"command" + _build_header(fields, version=1) + payload


def build_planner_message(mode=0, movement=None, facing=None, speed=-1.0, height=-1.0):
    if movement is None:
        movement = [0.0, 0.0, 0.0]
    if facing is None:
        facing = [1.0, 0.0, 0.0]
    fields = [
        {"name": "mode",     "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing",   "dtype": "f32", "shape": [3]},
        {"name": "speed",    "dtype": "f32", "shape": [1]},
        {"name": "height",   "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join([
        struct.pack("<i", int(mode)),
        struct.pack("<fff", *movement),
        struct.pack("<fff", *facing),
        struct.pack("<f", float(speed)),
        struct.pack("<f", float(height)),
    ])
    return b"planner" + _build_header(fields, version=1) + payload


# ─── NPZ loading ────────────────────────────────────────────────────────────

def load_amass_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    pose_body   = data["pose_body"].astype(np.float32)   # (T, 63)
    root_orient = data["root_orient"].astype(np.float32) # (T, 3)
    fps         = float(data["mocap_frame_rate"])
    T = pose_body.shape[0]
    print(f"[npz_player] {T} frames @ {fps:.1f} fps ({T/fps:.2f}s)")
    return pose_body, root_orient, fps


# ─── Pre-compute transforms (pico_manager pipeline) ─────────────────────────

def precompute_transforms(pose_body_np, root_orient_np):
    """
    Exact same pipeline as pico_manager process_smpl_joints():
      1. aa -> quat
      2. smpl_root_ytoz_up  (Y-up -> Z-up)
      3. compute_human_joints (SMPL FK -> 24 joints)
      4. remove_smpl_base_rot
      5. smpl_joints_local = quat_inv(q) @ joints
    Returns numpy arrays ready to send to SONIC.
    """
    T = pose_body_np.shape[0]
    BATCH = 64  # process in batches to avoid OOM

    all_body_quat   = []
    all_smpl_joints = []
    all_smpl_pose   = []

    for i in range(0, T, BATCH):
        body_pose   = torch.tensor(pose_body_np[i:i+BATCH])
        root_orient = torch.tensor(root_orient_np[i:i+BATCH])

        # Step 1+2: aa -> quat, Y-up -> Z-up
        q = angle_axis_to_quaternion(root_orient)
        q = smpl_root_ytoz_up(q)

        # Step 3: SMPL FK (24 joints, Z-up)
        global_orient_new = quaternion_to_angle_axis(q)
        joints = compute_human_joints(
            body_pose=body_pose[..., :63],
            global_orient=global_orient_new,
        )  # (B, 24, 3)

        # Step 4: remove SMPL T-pose base rotation
        q = remove_smpl_base_rot(q, w_last=False)

        # Step 5: local joints = quat_inv(q) @ joints
        q_inv = quat_inv(q).unsqueeze(1).repeat(1, joints.shape[1], 1)
        smpl_joints_local = quat_apply(q_inv, joints)  # (B, 24, 3)

        all_body_quat.append(q.detach().numpy().astype(np.float32))
        all_smpl_joints.append(smpl_joints_local.detach().numpy().astype(np.float32))
        # smpl_pose: just body pose (21 joints), reshape to (B, 21, 3)
        all_smpl_pose.append(body_pose[..., :63].detach().numpy().reshape(-1, 21, 3).astype(np.float32))

    body_quat_seq   = np.concatenate(all_body_quat,   axis=0)  # (T, 4)  [x,y,z,w] w_last=False -> actually xyzw
    smpl_joints_seq = np.concatenate(all_smpl_joints, axis=0)  # (T, 24, 3)
    smpl_pose_seq   = np.concatenate(all_smpl_pose,   axis=0)  # (T, 21, 3)

    # Note: gear_sonic quat convention is w_last=False -> [w,x,y,z]
    print(f"[npz_player] body_quat[0]: {body_quat_seq[0]}")
    print(f"[npz_player] smpl_joints_local[0,0]: {smpl_joints_seq[0,0]}")
    return body_quat_seq, smpl_joints_seq, smpl_pose_seq


# ─── Frame dict builder ──────────────────────────────────────────────────────

def compute_joint_pos_from_smpl(pose_win):
    """
    Compute G1 wrist joint angles from SMPL body pose.
    Same logic as pico_manager_thread_server.py (@Jiefeng).
    pose_win: (N, 21, 3) SMPL body pose axis-angle
    Returns: (29,) joint_pos for first frame
    """
    body_pose = pose_win.reshape(-1, 21, 3)

    SMPL_L_ELBOW_IDX = 17
    SMPL_L_WRIST_IDX = 19
    SMPL_R_ELBOW_IDX = 18
    SMPL_R_WRIST_IDX = 20

    G1_L_WRIST_ROLL_IDX  = 23
    G1_L_WRIST_PITCH_IDX = 25
    G1_L_WRIST_YAW_IDX   = 27
    G1_R_WRIST_ROLL_IDX  = 24
    G1_R_WRIST_PITCH_IDX = 26
    G1_R_WRIST_YAW_IDX   = 28

    smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
    smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
    smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
    smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

    g1_l_elbow_axis = np.array([0, 1, 0])
    g1_r_elbow_axis = np.array([0, 1, 0])

    g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, g1_l_elbow_axis)
    g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, g1_r_elbow_axis)

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1,2,3,0]]).as_euler("XYZ", degrees=False)
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1,2,3,0]]).as_euler("XYZ", degrees=False)

    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll  = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw   = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    g1_r_wrist_roll  = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw   = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    joint_pos_seq = np.zeros((len(body_pose), 29), dtype=np.float32)
    joint_pos_seq[:, G1_L_WRIST_ROLL_IDX]  = g1_l_wrist_roll
    joint_pos_seq[:, G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch
    joint_pos_seq[:, G1_L_WRIST_YAW_IDX]   = g1_l_wrist_yaw
    joint_pos_seq[:, G1_R_WRIST_ROLL_IDX]  = g1_r_wrist_roll
    joint_pos_seq[:, G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch
    joint_pos_seq[:, G1_R_WRIST_YAW_IDX]   = g1_r_wrist_yaw

    return joint_pos_seq


def make_frame_dict(pose_win, bquat_win, joints_win, frame_indices, fps):
    N = len(frame_indices)
    dt = 1.0 / fps
    # compute wrist joint_pos from SMPL pose
    joint_pos_seq = compute_joint_pos_from_smpl(pose_win.reshape(N, 21, 3))
    return {
        "smpl_pose":   pose_win.reshape(N, 21, 3).astype(np.float32),
        "smpl_joints": joints_win.reshape(N, 24, 3).astype(np.float32),
        "body_quat_w": bquat_win.astype(np.float32),
        "joint_pos":   joint_pos_seq.astype(np.float32),
        "joint_vel":   np.zeros((N, 29), dtype=np.float32),
        "frame_index": np.array(frame_indices, dtype=np.int64),
        "vr_position":    np.zeros(9,  dtype=np.float32),
        "vr_orientation": np.zeros(12, dtype=np.float32),
        "left_trigger":  np.array([0.0], dtype=np.float32),
        "right_trigger": np.array([0.0], dtype=np.float32),
        "left_grip":     np.array([0.0], dtype=np.float32),
        "right_grip":    np.array([0.0], dtype=np.float32),
        "pico_dt":  np.array([dt],  dtype=np.float32),
        "pico_fps": np.array([fps], dtype=np.float32),
        "timestamp_realtime":  np.array([time.time()],      dtype=np.float64),
        "timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
        "left_hand_joints":  np.zeros(7, dtype=np.float32),
        "right_hand_joints": np.zeros(7, dtype=np.float32),
        "toggle_data_collection": np.array([False], dtype=bool),
        "toggle_data_abort":      np.array([False], dtype=bool),
        "heading_increment":      np.array([0.0], dtype=np.float32),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kimodo npz -> SONIC ZMQ player")
    parser.add_argument("--npz",   required=True)
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=5556)
    parser.add_argument("--fps",   type=float, default=None)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--loop",  action="store_true")
    parser.add_argument("--start-delay", type=float, default=3.0)
    args = parser.parse_args()

    print(f"[npz_player] Loading {args.npz} ...")
    pose_body_np, root_orient_np, npz_fps = load_amass_npz(args.npz)
    fps = args.fps if args.fps is not None else npz_fps
    dt  = 1.0 / fps

    print("[npz_player] Pre-computing SMPL FK + coordinate transforms ...")
    body_quat_seq, smpl_joints_seq, smpl_pose_seq = precompute_transforms(
        pose_body_np, root_orient_np
    )
    T = smpl_pose_seq.shape[0]

    # Pre-tile for seamless loop
    REPEAT = 100 if args.loop else 1
    smpl_pose_full   = np.tile(smpl_pose_seq,   (REPEAT, 1, 1))
    body_quat_full   = np.tile(body_quat_seq,   (REPEAT, 1))
    smpl_joints_full = np.tile(smpl_joints_seq, (REPEAT, 1, 1))
    T_full = smpl_pose_full.shape[0]
    print(f"[npz_player] Tiled: {T_full} frames ({T_full/fps:.1f}s)")

    # ZMQ
    context = zmq.Context()
    socket  = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 20)
    socket.setsockopt(zmq.LINGER, 0)
    addr = f"tcp://{args.host}:{args.port}"
    socket.bind(addr)
    print(f"[npz_player] ZMQ PUB bound to {addr}")

    N = args.window

    def send_start_sequence():
        # planner=False -> STREAMED_MOTION mode (not PLANNER mode)
        print("[npz_player] Sending START (STREAMED_MOTION mode) ...")
        for _ in range(10):
            socket.send(build_command_message(start=True, stop=False, planner=False))
            time.sleep(0.1)
        print("[npz_player] START sent. Waiting 2s ...")
        time.sleep(2.0)

    # Keyboard listener
    restart_flag = threading.Event()

    def keyboard_listener():
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == ']':
                    restart_flag.set()
                    sys.stdout.write("\r[npz_player] ] pressed - rewinding ...\n")
                    sys.stdout.flush()
                elif ch in ('\x03', '\x04'):
                    raise KeyboardInterrupt
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    print(f"[npz_player] Waiting {args.start_delay}s for SONIC to connect ...")
    time.sleep(args.start_delay)
    send_start_sequence()

    step     = N
    motion_t = N

    buf_pose   = list(smpl_pose_full[:N])
    buf_bquat  = list(body_quat_full[:N])
    buf_joints = list(smpl_joints_full[:N])
    buf_idx    = list(range(N))

    print("[npz_player] Starting playback. Press ] to rewind motion.")
    try:
        while True:
            t_start = time.monotonic()

            if restart_flag.is_set():
                restart_flag.clear()
                motion_t = N
                buf_pose   = list(smpl_pose_full[:N])
                buf_bquat  = list(body_quat_full[:N])
                buf_joints = list(smpl_joints_full[:N])
                buf_idx    = list(range(step - N, step))
                print(f"[npz_player] Rewound. frame_index continues from {step}.")
                continue

            if motion_t >= T_full:
                if args.loop:
                    motion_t = N
                    buf_pose   = list(smpl_pose_full[:N])
                    buf_bquat  = list(body_quat_full[:N])
                    buf_joints = list(smpl_joints_full[:N])
                    buf_idx    = list(range(step - N, step))
                    continue
                else:
                    break


            buf_pose   = buf_pose[1:]   + [smpl_pose_full[motion_t]]
            buf_bquat  = buf_bquat[1:]  + [body_quat_full[motion_t]]
            buf_joints = buf_joints[1:] + [smpl_joints_full[motion_t]]
            buf_idx    = buf_idx[1:]    + [step]

            packed = pack_pose_message(make_frame_dict(
                np.stack(buf_pose),
                np.stack(buf_bquat),
                np.stack(buf_joints),
                list(buf_idx),
                fps,
            ))
            try:
                socket.send(packed, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            step     += 1
            motion_t += 1

            elapsed = time.monotonic() - t_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        print("[npz_player] Playback complete.")

    except KeyboardInterrupt:
        print("\n[npz_player] Interrupted.")

    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()