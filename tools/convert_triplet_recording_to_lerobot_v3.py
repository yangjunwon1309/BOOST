"""Convert record_sonic_motion_triplet.py output to LeRobotDataset v3.0.

The recorder writes one ``frame_XXXXXX.npz`` per synchronized sample, plus
optional JPEGs under ``frames/``.  This script converts one or more recorder
directories into a file-sharded LeRobot v3-style dataset:

  data/chunk-000/file-000000.parquet
  videos/chunk-000/observation.images.ego_view/file-000000.mp4
  meta/info.json
  meta/stats.json
  meta/tasks.jsonl
  meta/episodes/chunk-000/file-000000.parquet

The schema intentionally mirrors the Sonic VLA exporter where possible:
``observation.state`` and ``action`` are 43D vectors formed as
body(29) + left_hand(7) + right_hand(7).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "This converter requires pandas and pyarrow.\n"
        "Install in your data environment, for example:\n"
        "  python -m pip install pandas pyarrow imageio pillow\n"
    ) from exc

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None


BODY_DOF = 29
HAND_DOF = 7
WHOLE_DOF = BODY_DOF + 2 * HAND_DOF
DEFAULT_VIDEO_KEY = "observation.images.ego_view"

JOINT_NAMES_43 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_0",
    "left_hand_1",
    "left_hand_2",
    "left_hand_3",
    "left_hand_4",
    "left_hand_5",
    "left_hand_6",
    "right_hand_0",
    "right_hand_1",
    "right_hand_2",
    "right_hand_3",
    "right_hand_4",
    "right_hand_5",
    "right_hand_6",
]


@dataclass
class EpisodeRecord:
    path: Path
    rows: list[dict[str, Any]]
    length: int
    task_index: int
    task: str
    video_frames: list[np.ndarray] | None
    video_size: tuple[int, int] | None


def _arr(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> np.ndarray | None:
    if key not in data:
        return None if default is None else np.asarray(default)
    value = np.asarray(data[key])
    if value.shape == ():
        value = value.reshape(1)
    return value


def _vec(
    data: np.lib.npyio.NpzFile,
    keys: list[str],
    size: int,
    dtype: np.dtype = np.float64,
    default: float = 0.0,
) -> np.ndarray:
    for key in keys:
        value = _arr(data, key)
        if value is None:
            continue
        value = np.asarray(value, dtype=dtype).reshape(-1)
        if value.size == size:
            return value
    return np.full(size, default, dtype=dtype)


def _scalar(data: np.lib.npyio.NpzFile, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        value = _arr(data, key)
        if value is not None and value.size:
            return float(np.asarray(value).reshape(-1)[0])
    return float(default)


def _int_scalar(data: np.lib.npyio.NpzFile, keys: list[str], default: int = 0) -> int:
    return int(round(_scalar(data, keys, float(default))))


def _int_last(data: np.lib.npyio.NpzFile, keys: list[str], default: int = 0) -> int:
    for key in keys:
        value = _arr(data, key)
        if value is not None and value.size:
            return int(round(float(np.asarray(value).reshape(-1)[-1])))
    return int(default)


def _string_scalar(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    value = _arr(data, key)
    if value is None or value.size == 0:
        return default
    return str(value.reshape(-1)[0])


def quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat = quat / norm

    if R is not None:
        mat = R.from_quat(quat, scalar_first=True).as_matrix()
    else:
        w, x, y, z = quat
        mat = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    return mat[:, :2].reshape(-1, order="F").astype(np.float32)


def projected_gravity_from_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size != 4 or np.linalg.norm(quat) < 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if R is not None:
        mat = R.from_quat(quat / np.linalg.norm(quat), scalar_first=True).as_matrix()
    else:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return (mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float64)


def choose_action(data: np.lib.npyio.NpzFile, action_source: str) -> np.ndarray:
    candidates: list[str]
    if action_source == "qp":
        candidates = ["state_ee_qp_q_corrected", "sonic_last_action", "state_last_action"]
    elif action_source == "state":
        candidates = ["state_last_action", "sonic_last_action"]
    else:
        candidates = ["sonic_last_action", "state_last_action"]

    body = _vec(data, candidates, BODY_DOF)
    left = _vec(data, ["state_last_left_hand_action", "sonic_last_left_hand_action"], HAND_DOF)
    right = _vec(data, ["state_last_right_hand_action", "sonic_last_right_hand_action"], HAND_DOF)
    return np.concatenate([body, left, right]).astype(np.float64)


def choose_state(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    body_q = _vec(data, ["state_body_q_measured", "state_body_q"], BODY_DOF)
    body_dq = _vec(data, ["state_body_dq_measured", "state_body_dq"], BODY_DOF)
    left_q = _vec(data, ["state_left_hand_q_measured", "state_left_hand_q"], HAND_DOF)
    right_q = _vec(data, ["state_right_hand_q_measured", "state_right_hand_q"], HAND_DOF)
    left_dq = _vec(data, ["state_left_hand_dq"], HAND_DOF)
    right_dq = _vec(data, ["state_right_hand_dq"], HAND_DOF)
    return (
        np.concatenate([body_q, left_q, right_q]).astype(np.float64),
        np.concatenate([body_dq, left_dq, right_dq]).astype(np.float64),
    )


def smpl_current(data: np.lib.npyio.NpzFile, key: str, out_size: int, dtype: np.dtype) -> np.ndarray:
    value = _arr(data, key)
    if value is None:
        return np.zeros(out_size, dtype=dtype)
    value = np.asarray(value, dtype=dtype)
    if value.ndim >= 2 and value.shape[0] > 0:
        value = value[-1]
    value = value.reshape(-1)
    if value.size != out_size:
        out = np.zeros(out_size, dtype=dtype)
        n = min(out_size, value.size)
        out[:n] = value[:n]
        return out
    return value


def load_image(record_dir: Path, rel_path: str) -> np.ndarray | None:
    if not rel_path:
        return None
    path = record_dir / rel_path
    if not path.exists() or iio is None:
        return None
    image = iio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return np.asarray(image, dtype=np.uint8)


def iter_frame_files(record_dir: Path) -> list[Path]:
    files = sorted(record_dir.glob("frame_*.npz"))
    if not files:
        raise FileNotFoundError(f"No frame_*.npz files found in {record_dir}")
    return files


def build_episode(
    record_dir: Path,
    *,
    episode_index: int,
    task_index: int,
    task: str,
    fps: float,
    action_source: str,
    include_raw_windows: bool,
    encode_video: bool,
) -> EpisodeRecord:
    rows: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []
    video_size: tuple[int, int] | None = None

    initial_yaw: float | None = None
    for local_idx, frame_path in enumerate(iter_frame_files(record_dir)):
        with np.load(frame_path, allow_pickle=True) as data:
            state, velocity = choose_state(data)
            action = choose_action(data, action_source)
            root_quat = _vec(data, ["state_base_quat_measured", "state_base_quat"], 4)
            init_quat = _vec(data, ["state_init_base_quat"], 4, default=0.0)
            if not np.any(init_quat):
                init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

            smpl_quat = smpl_current(data, "pose_body_quat_w", 4, np.float32)
            delta_heading = np.array([
                _scalar(data, ["pose_heading_increment", "state_delta_heading"], 0.0)
            ], dtype=np.float64)
            target_rot6d = quat_wxyz_to_rot6d(smpl_quat)
            if R is not None and smpl_quat.size == 4:
                try:
                    rot = R.from_quat(smpl_quat, scalar_first=True)
                    euler = rot.as_euler("ZYX", degrees=False)
                    if initial_yaw is None:
                        initial_yaw = float(euler[0])
                    norm_euler = np.array([euler[0] - initial_yaw + delta_heading[0], euler[1], euler[2]])
                    target_rot6d = quat_wxyz_to_rot6d(
                        R.from_euler("ZYX", norm_euler).as_quat(scalar_first=True)
                    )
                except Exception:
                    pass

            stream_mode = np.array([
                _int_scalar(data, ["manager_stream_mode"], -1)
            ], dtype=np.int32)
            smpl_frame_idx = np.array([
                _int_last(data, ["pose_frame_index"], local_idx)
            ], dtype=np.int64)

            row: dict[str, Any] = {
                "episode_index": episode_index,
                "frame_index": local_idx,
                "timestamp": float(local_idx) / float(fps),
                "task_index": task_index,
                "index": -1,
                "observation.state": state,
                "observation.velocity": velocity,
                "observation.root_orientation": root_quat.astype(np.float64),
                "observation.projected_gravity": projected_gravity_from_quat(root_quat),
                "observation.init_base_quat": init_quat.astype(np.float64),
                "action": action,
                "action.wbc": action,
                "action.motion_token": _vec(data, ["state_token_state"], 64),
                "teleop.delta_heading": delta_heading,
                "teleop.smpl_joints": smpl_current(data, "pose_smpl_joints", 72, np.float32),
                "teleop.smpl_pose": smpl_current(data, "pose_smpl_pose", 63, np.float32),
                "teleop.body_quat_w": smpl_quat.astype(np.float32),
                "teleop.target_body_orientation": target_rot6d.astype(np.float32),
                "teleop.left_hand_joints": _vec(data, ["pose_left_hand_joints"], HAND_DOF, np.float32),
                "teleop.right_hand_joints": _vec(data, ["pose_right_hand_joints"], HAND_DOF, np.float32),
                "teleop.smpl_frame_index": smpl_frame_idx,
                "teleop.stream_mode": stream_mode,
                "teleop.vr_3pt_position": _vec(
                    data, ["pose_vr_position", "state_vr_3point_position"], 9, np.float32
                ),
                "teleop.vr_3pt_orientation_quat": _vec(
                    data, ["pose_vr_orientation", "state_vr_3point_orientation"], 12, np.float32
                ),
                "recorder.record_wall_time": _vec(data, ["record_wall_time"], 1),
                "recorder.pose_age_sec": _vec(data, ["pose_age_sec"], 1),
                "recorder.state_age_sec": _vec(data, ["state_age_sec"], 1),
                "recorder.sonic_age_sec": _vec(data, ["sonic_age_sec"], 1),
            }

            image_rel = _string_scalar(data, "image_path", "")
            if encode_video:
                image = load_image(record_dir, image_rel)
                if image is not None:
                    video_frames.append(image)
                    video_size = (int(image.shape[1]), int(image.shape[0]))
            if image_rel:
                row[DEFAULT_VIDEO_KEY] = ""

            if include_raw_windows:
                smpl_joints_window = _arr(data, "pose_smpl_joints")
                smpl_pose_window = _arr(data, "pose_smpl_pose")
                if smpl_joints_window is not None:
                    row["teleop.raw_smpl_joints_window"] = np.asarray(
                        smpl_joints_window, dtype=np.float32
                    )
                if smpl_pose_window is not None:
                    row["teleop.raw_smpl_pose_window"] = np.asarray(
                        smpl_pose_window, dtype=np.float32
                    )
            rows.append(row)

    return EpisodeRecord(
        path=record_dir,
        rows=rows,
        length=len(rows),
        task_index=task_index,
        task=task,
        video_frames=video_frames if video_frames else None,
        video_size=video_size,
    )


def feature(dtype: str, shape: list[int] | tuple[int, ...], names: Any = None) -> dict[str, Any]:
    out = {"dtype": dtype, "shape": list(shape)}
    if names is not None:
        out["names"] = names
    return out


def build_features(include_video: bool, include_raw_windows: bool) -> dict[str, Any]:
    features = {
        "observation.state": feature("float64", [WHOLE_DOF], JOINT_NAMES_43),
        "observation.velocity": feature("float64", [WHOLE_DOF], JOINT_NAMES_43),
        "observation.root_orientation": feature("float64", [4], ["base_qw", "base_qx", "base_qy", "base_qz"]),
        "observation.projected_gravity": feature("float64", [3], ["gravity_x", "gravity_y", "gravity_z"]),
        "observation.init_base_quat": feature("float64", [4], ["init_qw", "init_qx", "init_qy", "init_qz"]),
        "action": feature("float64", [WHOLE_DOF], JOINT_NAMES_43),
        "action.wbc": feature("float64", [WHOLE_DOF], JOINT_NAMES_43),
        "action.motion_token": feature("float64", [64], "motion_token"),
        "teleop.delta_heading": feature("float64", [1], ["delta_heading"]),
        "teleop.smpl_joints": feature("float32", [72], "smpl_joints"),
        "teleop.smpl_pose": feature("float32", [63], "smpl_pose"),
        "teleop.body_quat_w": feature("float32", [4], "body_quat_w"),
        "teleop.target_body_orientation": feature("float32", [6], "target_body_orientation"),
        "teleop.left_hand_joints": feature("float32", [7], "left_hand_joints"),
        "teleop.right_hand_joints": feature("float32", [7], "right_hand_joints"),
        "teleop.smpl_frame_index": feature("int64", [1], ["smpl_frame_index"]),
        "teleop.stream_mode": feature("int32", [1], ["stream_mode"]),
        "teleop.vr_3pt_position": feature("float32", [9], "vr_3pt_position"),
        "teleop.vr_3pt_orientation_quat": feature("float32", [12], "vr_3pt_orientation_quat"),
        "recorder.record_wall_time": feature("float64", [1], ["wall_time"]),
        "recorder.pose_age_sec": feature("float64", [1], ["pose_age_sec"]),
        "recorder.state_age_sec": feature("float64", [1], ["state_age_sec"]),
        "recorder.sonic_age_sec": feature("float64", [1], ["sonic_age_sec"]),
        "timestamp": {"dtype": "float64", "shape": [1], "names": ["timestamp"]},
        "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
        "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
        "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
        "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
    }
    if include_video:
        features[DEFAULT_VIDEO_KEY] = {
            "dtype": "video",
            "shape": [0, 0, 3],
            "names": ["height", "width", "channel"],
        }
    if include_raw_windows:
        features["teleop.raw_smpl_joints_window"] = feature("float32", [5, 24, 3], "raw_smpl_joints_window")
        features["teleop.raw_smpl_pose_window"] = feature("float32", [5, 21, 3], "raw_smpl_pose_window")
    return features


def build_modality(include_video: bool) -> dict[str, Any]:
    modality = {
        "state": {
            "left_leg": {"start": 0, "end": 6, "original_key": "observation.state"},
            "right_leg": {"start": 6, "end": 12, "original_key": "observation.state"},
            "waist": {"start": 12, "end": 15, "original_key": "observation.state"},
            "left_arm": {"start": 15, "end": 22, "original_key": "observation.state"},
            "right_arm": {"start": 22, "end": 29, "original_key": "observation.state"},
            "left_hand": {"start": 29, "end": 36, "original_key": "observation.state"},
            "right_hand": {"start": 36, "end": 43, "original_key": "observation.state"},
            "root_orientation": {
                "start": 0,
                "end": 4,
                "original_key": "observation.root_orientation",
                "rotation_type": "quaternion",
            },
        },
        "action": {
            "left_leg": {"start": 0, "end": 6, "original_key": "action"},
            "right_leg": {"start": 6, "end": 12, "original_key": "action"},
            "waist": {"start": 12, "end": 15, "original_key": "action"},
            "left_arm": {"start": 15, "end": 22, "original_key": "action"},
            "right_arm": {"start": 22, "end": 29, "original_key": "action"},
            "left_hand": {"start": 29, "end": 36, "original_key": "action"},
            "right_hand": {"start": 36, "end": 43, "original_key": "action"},
            "smpl_joints": {"start": 0, "end": 72, "original_key": "teleop.smpl_joints"},
            "smpl_pose": {"start": 0, "end": 63, "original_key": "teleop.smpl_pose"},
        },
        "video": {},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    if include_video:
        modality["video"]["ego_view"] = {"original_key": DEFAULT_VIDEO_KEY}
    return modality


def normalize_cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    if iio is None:
        raise RuntimeError("imageio is required to encode mp4 videos")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(path, frames, fps=float(fps), codec="libx264", macro_block_size=1)
    except Exception:
        iio.imwrite(path, frames, fps=float(fps), macro_block_size=1)


def compute_stats(df: pd.DataFrame, features: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    skip = {"index", "episode_index", "frame_index", "timestamp", "task_index"}
    for key, spec in features.items():
        if key in skip or spec.get("dtype") == "video" or key not in df.columns:
            continue
        values = []
        for item in df[key]:
            arr = np.asarray(item, dtype=np.float64).reshape(-1)
            if arr.size:
                values.append(arr)
        if not values:
            continue
        stacked = np.stack(values, axis=0)
        stats[key] = {
            "mean": np.mean(stacked, axis=0).tolist(),
            "std": np.std(stacked, axis=0).tolist(),
            "min": np.min(stacked, axis=0).tolist(),
            "max": np.max(stacked, axis=0).tolist(),
        }
    return stats


def write_dataset(
    episodes: list[EpisodeRecord],
    output_dir: Path,
    *,
    fps: float,
    task: str,
    include_video: bool,
    include_raw_windows: bool,
    overwrite: bool,
) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "data" / "chunk-000" / "file-000000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    global_index = 0
    for ep in episodes:
        start = global_index
        for row in ep.rows:
            row["index"] = global_index
            if include_video and DEFAULT_VIDEO_KEY in row:
                row[DEFAULT_VIDEO_KEY] = (
                    "videos/chunk-000/"
                    f"{DEFAULT_VIDEO_KEY}/file-000000.mp4"
                )
            all_rows.append({k: normalize_cell(v) for k, v in row.items()})
            global_index += 1
        end = global_index
        episode_rows.append(
            {
                "episode_index": len(episode_rows),
                "tasks": [ep.task],
                "task_index": ep.task_index,
                "length": ep.length,
                "data_file_index": 0,
                "data_start_index": start,
                "data_end_index": end,
                "video_file_index": 0 if include_video else -1,
                "video_start_frame": start if include_video else -1,
                "video_end_frame": end if include_video else -1,
            }
        )

    df = pd.DataFrame(all_rows)
    df.to_parquet(data_path, index=False)

    if include_video:
        video_path = output_dir / "videos" / "chunk-000" / DEFAULT_VIDEO_KEY / "file-000000.mp4"
        merged_frames: list[np.ndarray] = []
        for ep in episodes:
            if ep.video_frames is not None:
                merged_frames.extend(ep.video_frames)
        if merged_frames:
            write_video(video_path, merged_frames, fps)
            h, w = int(merged_frames[0].shape[0]), int(merged_frames[0].shape[1])
        else:
            h, w = 0, 0
    else:
        h, w = 0, 0

    features = build_features(include_video=include_video, include_raw_windows=include_raw_windows)
    if include_video:
        features[DEFAULT_VIDEO_KEY]["shape"] = [h, w, 3]

    episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1",
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": len(all_rows),
        "total_tasks": 1,
        "total_videos": 1 if include_video else 0,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/file-{file_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/file-{file_index:06d}.mp4",
        "episodes_path": "meta/episodes/chunk-{episode_chunk:03d}/file-{file_index:06d}.parquet",
        "features": features,
        "video_keys": [DEFAULT_VIDEO_KEY] if include_video else [],
        "splits": {"train": f"0:{len(all_rows)}"},
        "source": {
            "converter": "gear_sonic/scripts/convert_triplet_recording_to_lerobot_v3.py",
            "recording_dirs": [str(ep.path) for ep in episodes],
        },
    }
    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with (meta_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(compute_stats(df, features), f, indent=2)
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    with (meta_dir / "modality.json").open("w", encoding="utf-8") as f:
        json.dump(build_modality(include_video), f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording_dirs",
        nargs="+",
        type=Path,
        help="One or more directories produced by record_sonic_motion_triplet.py.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--task", default="demo")
    parser.add_argument(
        "--action-source",
        choices=["sonic", "qp", "state"],
        default="sonic",
        help="Which recorded PD target to expose as LeRobot action.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 creation even if JPEGs exist.")
    parser.add_argument("--include-raw-windows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir is not empty: {args.output_dir}. Use --overwrite.")

    include_video = not args.no_video
    episodes = []
    for idx, record_dir in enumerate(args.recording_dirs):
        record_dir = record_dir.expanduser().resolve()
        print(f"[convert] loading episode {idx}: {record_dir}")
        ep = build_episode(
            record_dir,
            episode_index=idx,
            task_index=0,
            task=args.task,
            fps=args.fps,
            action_source=args.action_source,
            include_raw_windows=args.include_raw_windows,
            encode_video=include_video,
        )
        if include_video and ep.video_frames is None:
            print(f"[convert] warning: no usable images found in {record_dir}; dataset will be tabular-only")
            include_video = False
        episodes.append(ep)

    write_dataset(
        episodes,
        args.output_dir.expanduser().resolve(),
        fps=args.fps,
        task=args.task,
        include_video=include_video,
        include_raw_windows=args.include_raw_windows,
        overwrite=args.overwrite,
    )
    print(f"[convert] wrote LeRobot v3 dataset: {args.output_dir}")
    print(f"[convert] episodes={len(episodes)} frames={sum(ep.length for ep in episodes)} video={include_video}")


if __name__ == "__main__":
    main()
