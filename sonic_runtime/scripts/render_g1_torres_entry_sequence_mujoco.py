#!/usr/bin/env python3
"""Render a scripted G1 + Torres door-entry-and-sit sequence in MuJoCo.

This is a video-generation helper, not a learned walking policy.  It keeps the
Torres collision scene active, directly animates the G1 free base and joints,
and writes an MP4 showing: approach handle, unlatch/open door, enter cabin, sit.

Recommended:

  source .venv_sim/bin/activate
  python scripts/render_g1_torres_entry_sequence_mujoco.py \
    --prepare-scene --detail-mode \
    --output outputs/torres_entry_sequence.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# Offscreen rendering needs a GL backend before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


DOOR_OPEN_ANGLE = 1.507964474
HANDLE_OPEN_ANGLE = 0.145


@dataclass(frozen=True)
class Keyframe:
    t: float
    root_pos: tuple[float, float, float]
    yaw_deg: float
    door: float
    handle: float
    joints: Mapping[str, float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_scene(root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        root / "gear_sonic_deploy" / "gear_sonic_deploy" / "g1" / "scene_29dof_torres_mesh_hinge.xml",
        root / "gear_sonic_deploy" / "g1" / "scene_29dof_torres_mesh_hinge.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find scene_29dof_torres_mesh_hinge.xml")


def prepare_scene(args: argparse.Namespace, root: Path) -> None:
    converter = root / "scripts" / "convert_torres_dae_to_obj_and_scene.py"
    cmd = [
        sys.executable,
        converter.as_posix(),
        "--root",
        root.as_posix(),
        "--car-collision-mode",
        "hulls",
    ]
    if args.detail_mode:
        cmd.append("--detail-mode")
    if args.skip_collision:
        cmd.append("--skip-collision")
    if args.keep_entry_collision_hulls:
        cmd.append("--keep-entry-collision-hulls")
    if args.show_collision_hulls:
        cmd.extend(["--collision-alpha", str(args.collision_alpha), "--show-visual-hulls"])

    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def quat_from_yaw(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, u: float) -> float:
    return float((1.0 - u) * a + u * b)


def lerp_vec(a: tuple[float, float, float], b: tuple[float, float, float], u: float) -> np.ndarray:
    return (1.0 - u) * np.asarray(a, dtype=float) + u * np.asarray(b, dtype=float)


def lerp_yaw(a: float, b: float, u: float) -> float:
    delta = ((b - a + 180.0) % 360.0) - 180.0
    return a + u * delta


def full_pose(*overrides: Mapping[str, float]) -> dict[str, float]:
    pose = {
        "left_hip_pitch_joint": -0.18,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.42,
        "left_ankle_pitch_joint": -0.23,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.18,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.42,
        "right_ankle_pitch_joint": -0.23,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.30,
        "left_shoulder_roll_joint": 0.20,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.80,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.30,
        "right_shoulder_roll_joint": -0.20,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.80,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    for side in ("left", "right"):
        pose.update(
            {
                f"{side}_hand_thumb_0_joint": 0.0,
                f"{side}_hand_thumb_1_joint": 0.0,
                f"{side}_hand_thumb_2_joint": 0.0,
                f"{side}_hand_middle_0_joint": 0.0,
                f"{side}_hand_middle_1_joint": 0.0,
                f"{side}_hand_index_0_joint": 0.0,
                f"{side}_hand_index_1_joint": 0.0,
            }
        )

    for override in overrides:
        pose.update(override)
    return pose


def hand_closed(side: str) -> dict[str, float]:
    return {
        f"{side}_hand_thumb_0_joint": 0.45,
        f"{side}_hand_thumb_1_joint": 0.35,
        f"{side}_hand_thumb_2_joint": 0.95,
        f"{side}_hand_middle_0_joint": -0.95,
        f"{side}_hand_middle_1_joint": -1.15,
        f"{side}_hand_index_0_joint": -0.95,
        f"{side}_hand_index_1_joint": -1.15,
    }


REACH_HANDLE = {
    "left_shoulder_pitch_joint": 0.55,
    "left_shoulder_roll_joint": 0.58,
    "left_shoulder_yaw_joint": 0.18,
    "left_elbow_joint": 0.42,
    "left_wrist_pitch_joint": 0.22,
    "left_wrist_yaw_joint": 0.20,
    "right_shoulder_pitch_joint": 0.25,
    "right_shoulder_roll_joint": -0.22,
    "right_elbow_joint": 0.95,
}

PULL_DOOR = {
    "left_shoulder_pitch_joint": 0.42,
    "left_shoulder_roll_joint": 0.72,
    "left_shoulder_yaw_joint": 0.35,
    "left_elbow_joint": 0.75,
    "left_wrist_pitch_joint": 0.18,
    "left_wrist_yaw_joint": 0.35,
    "waist_yaw_joint": -0.12,
}

COMPACT_ARMS = {
    "left_shoulder_pitch_joint": 0.34,
    "left_shoulder_roll_joint": 0.12,
    "left_elbow_joint": 1.10,
    "right_shoulder_pitch_joint": 0.34,
    "right_shoulder_roll_joint": -0.12,
    "right_elbow_joint": 1.10,
    "right_wrist_yaw_joint": 0.0,
}

PRE_SIT = {
    "left_hip_pitch_joint": -0.48,
    "left_knee_joint": 0.92,
    "left_ankle_pitch_joint": -0.42,
    "right_hip_pitch_joint": -0.48,
    "right_knee_joint": 0.92,
    "right_ankle_pitch_joint": -0.42,
    "waist_pitch_joint": 0.08,
}

SIT = {
    "left_hip_pitch_joint": -1.05,
    "left_knee_joint": 1.72,
    "left_ankle_pitch_joint": -0.66,
    "right_hip_pitch_joint": -1.05,
    "right_knee_joint": 1.72,
    "right_ankle_pitch_joint": -0.66,
    "waist_pitch_joint": 0.16,
    "left_shoulder_pitch_joint": 0.50,
    "left_shoulder_roll_joint": 0.20,
    "left_elbow_joint": 1.25,
    "right_shoulder_pitch_joint": 0.50,
    "right_shoulder_roll_joint": -0.20,
    "right_elbow_joint": 1.25,
}


def build_keyframes() -> list[Keyframe]:
    return [
        Keyframe(0.0, (0.80, -1.05, 0.793), 90.0, 0.0, 0.0, full_pose()),
        Keyframe(1.3, (0.82, -0.86, 0.793), 90.0, 0.0, 0.0, full_pose(REACH_HANDLE)),
        Keyframe(
            2.2,
            (0.82, -0.86, 0.793),
            90.0,
            0.0,
            HANDLE_OPEN_ANGLE,
            full_pose(REACH_HANDLE, hand_closed("left")),
        ),
        Keyframe(
            4.2,
            (0.91, -0.92, 0.793),
            82.0,
            1.42,
            HANDLE_OPEN_ANGLE,
            full_pose(PULL_DOOR, hand_closed("left")),
        ),
        Keyframe(5.4, (1.05, -0.82, 0.793), 120.0, DOOR_OPEN_ANGLE, 0.0, full_pose(COMPACT_ARMS)),
        Keyframe(6.9, (1.02, -0.30, 0.793), 180.0, DOOR_OPEN_ANGLE, 0.0, full_pose(COMPACT_ARMS)),
        Keyframe(
            8.4,
            (0.98, 0.20, 0.745),
            180.0,
            DOOR_OPEN_ANGLE,
            0.0,
            full_pose(COMPACT_ARMS, PRE_SIT),
        ),
        Keyframe(
            10.4,
            (0.92, 0.58, 0.540),
            180.0,
            DOOR_OPEN_ANGLE,
            0.0,
            full_pose(COMPACT_ARMS, SIT),
        ),
        Keyframe(
            11.6,
            (0.92, 0.61, 0.525),
            180.0,
            DOOR_OPEN_ANGLE,
            0.0,
            full_pose(COMPACT_ARMS, SIT),
        ),
    ]


def sample_keyframes(keyframes: list[Keyframe], t: float) -> Keyframe:
    if t <= keyframes[0].t:
        return keyframes[0]
    if t >= keyframes[-1].t:
        return keyframes[-1]

    lo = keyframes[0]
    hi = keyframes[-1]
    for idx in range(len(keyframes) - 1):
        if keyframes[idx].t <= t <= keyframes[idx + 1].t:
            lo, hi = keyframes[idx], keyframes[idx + 1]
            break

    u = smoothstep((t - lo.t) / max(hi.t - lo.t, 1e-6))
    keys = set(lo.joints) | set(hi.joints)
    joints = {name: lerp(lo.joints.get(name, 0.0), hi.joints.get(name, 0.0), u) for name in keys}
    return Keyframe(
        t=t,
        root_pos=tuple(lerp_vec(lo.root_pos, hi.root_pos, u)),
        yaw_deg=lerp_yaw(lo.yaw_deg, hi.yaw_deg, u),
        door=lerp(lo.door, hi.door, u),
        handle=lerp(lo.handle, hi.handle, u),
        joints=joints,
    )


def joint_qadr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return -1
    return int(model.jnt_qposadr[joint_id])


def clamp_joint(model: mujoco.MjModel, name: str, value: float) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0 or not model.jnt_limited[joint_id]:
        return value
    lo, hi = model.jnt_range[joint_id]
    return float(np.clip(value, lo, hi))


def set_hinge(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    qadr = joint_qadr(model, name)
    if qadr >= 0:
        data.qpos[qadr] = clamp_joint(model, name, value)


def apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_qpos: np.ndarray,
    frame: Keyframe,
) -> None:
    data.qpos[:] = base_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0

    free_qadr = joint_qadr(model, "floating_base_joint")
    if free_qadr < 0:
        raise RuntimeError("floating_base_joint was not found in the scene")
    data.qpos[free_qadr : free_qadr + 3] = np.asarray(frame.root_pos, dtype=float)
    data.qpos[free_qadr + 3 : free_qadr + 7] = quat_from_yaw(frame.yaw_deg)

    set_hinge(model, data, "torres_driver_door_hinge", frame.door)
    set_hinge(model, data, "torres_driver_handle_hinge", frame.handle)

    for name, value in frame.joints.items():
        qadr = joint_qadr(model, name)
        if qadr >= 0:
            data.qpos[qadr] = clamp_joint(model, name, value)

    data.time = frame.t
    mujoco.mj_forward(model, data)


def collision_inventory(model: mujoco.MjModel) -> dict[str, int]:
    counts = {"body": 0, "door": 0, "handle": 0, "seat": 0}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        active = bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        if not active:
            continue
        if name.startswith("torres_body_collision_"):
            counts["body"] += 1
        elif name.startswith("torres_door_collision_"):
            counts["door"] += 1
        elif name.startswith(("torres_handle_collision_", "torres_handle_lock_collision_")):
            counts["handle"] += 1
        elif name.startswith("driver_seat_"):
            counts["seat"] += 1
    return counts


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom_names = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            geom_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        a, b = geom_names
        a_is_torres = a.startswith(("torres_", "driver_seat_"))
        b_is_torres = b.startswith(("torres_", "driver_seat_"))
        if a == "floor" or b == "floor":
            continue
        if a_is_torres == b_is_torres:
            continue
        pairs[tuple(sorted((a or "<unnamed>", b or "<unnamed>")))] += 1
    return pairs


def configure_camera(camera: mujoco.MjvCamera, preset: str, progress: float) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if preset == "wide":
        camera.lookat[:] = np.array([0.70, -0.05, 0.75])
        camera.distance = 3.35
        camera.azimuth = 135.0
        camera.elevation = -20.0
    elif preset == "side":
        camera.lookat[:] = np.array([0.85, -0.10, 0.70])
        camera.distance = 2.85
        camera.azimuth = 90.0
        camera.elevation = -15.0
    else:
        u = smoothstep(progress)
        camera.lookat[:] = (1.0 - u) * np.array([0.86, -0.58, 0.82]) + u * np.array([0.92, 0.35, 0.64])
        camera.distance = lerp(2.65, 2.35, u)
        camera.azimuth = lerp(125.0, 155.0, u)
        camera.elevation = lerp(-18.0, -24.0, u)


def render_video(args: argparse.Namespace, scene_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos.copy()

    inventory = collision_inventory(model)
    print(
        "Active Torres collision geoms: "
        f"body={inventory['body']}, door={inventory['door']}, "
        f"handle={inventory['handle']}, seat={inventory['seat']}"
    )
    if inventory["body"] == 0:
        print("WARNING: no active torres_body_collision_* geoms found; this scene looks bodyless/easy.")

    keyframes = build_keyframes()
    source_end = keyframes[-1].t
    duration = float(args.duration) if args.duration else source_end
    frame_count = max(2, int(round(duration * args.fps)))
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = repo_root() / output
    output.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    writer = cv2.VideoWriter(
        output.as_posix(),
        cv2.VideoWriter_fourcc(*args.codec),
        float(args.fps),
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    max_contacts = 0
    pair_counts: Counter[tuple[str, str]] = Counter()
    preview_written = False

    try:
        for frame_idx in range(frame_count):
            video_progress = frame_idx / max(frame_count - 1, 1)
            source_t = video_progress * source_end
            frame = sample_keyframes(keyframes, source_t)
            apply_frame(model, data, base_qpos, frame)
            max_contacts = max(max_contacts, int(data.ncon))
            pair_counts.update(contact_summary(model, data))

            configure_camera(camera, args.camera, video_progress)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()

            if args.preview and not preview_written and frame_idx >= frame_count // 2:
                preview_path = output.with_suffix(".preview.png")
                cv2.imwrite(preview_path.as_posix(), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                preview_written = True

            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()

    print(f"Wrote video: {output}")
    print(f"Frames: {frame_count} at {args.fps} fps, duration: {frame_count / args.fps:.2f}s")
    print(f"Max MuJoCo contacts in a scripted frame: {max_contacts}")
    if pair_counts:
        print("Most frequent robot-vs-Torres contact pairs:")
        for (a, b), count in pair_counts.most_common(8):
            print(f"  {count:5d}  {a} <-> {b}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None, help="Explicit scene_29dof_torres_mesh_hinge.xml path")
    parser.add_argument("--output", default="outputs/torres_entry_sequence.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--duration", type=float, default=None, help="Override output duration in seconds")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc, e.g. mp4v or avc1")
    parser.add_argument("--camera", choices=("side", "director", "wide"), default="side")
    parser.add_argument("--preview", action="store_true", help="Also write a mid-sequence preview PNG")

    parser.add_argument("--prepare-scene", action="store_true", help="Regenerate the Torres scene before rendering")
    parser.add_argument("--detail-mode", action="store_true", help="Use detail OBJ/MTL assets when preparing the scene")
    parser.add_argument("--skip-collision", action="store_true", help="Reuse existing convex collision OBJ files")
    parser.add_argument(
        "--keep-entry-collision-hulls",
        action="store_true",
        help="Do not filter body hulls that block the driver-entry clearance volume",
    )
    parser.add_argument("--show-collision-hulls", action="store_true", help="Render collision hulls visibly")
    parser.add_argument("--collision-alpha", type=float, default=0.22, help="Visible collision alpha when showing hulls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    if args.prepare_scene:
        prepare_scene(args, root)
    scene_path = resolve_scene(root, args.scene)
    render_video(args, scene_path)


if __name__ == "__main__":
    main()
