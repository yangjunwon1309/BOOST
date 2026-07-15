#!/usr/bin/env python3
"""Open the Torres mesh-hinge scene with visible collision hulls.

By default this shows the normal, non-detail-mode car.  Pass --detail-mode to
inspect the older detail asset path with the same collision debug settings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import mujoco.viewer


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_scene(root: Path) -> Path:
    candidates = [
        root / "gear_sonic_deploy" / "gear_sonic_deploy" / "g1" / "scene_29dof_torres_mesh_hinge.xml",
        root / "gear_sonic_deploy" / "g1" / "scene_29dof_torres_mesh_hinge.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def regenerate_scene(args: argparse.Namespace, root: Path) -> None:
    converter = root / "scripts" / "convert_torres_dae_to_obj_and_scene.py"
    cmd = [
        sys.executable,
        converter.as_posix(),
        "--root",
        root.as_posix(),
        "--collision-alpha",
        f"{args.collision_alpha:g}",
        "--show-visual-hulls",
        "--max-hull-points",
        str(args.max_hull_points),
    ]
    if args.detail_mode:
        cmd.append("--detail-mode")
    if args.reuse_collision:
        cmd.append("--skip-collision")
    if args.keep_entry_collision_hulls:
        cmd.append("--keep-entry-collision-hulls")

    mode = "detail-mode" if args.detail_mode else "normal non-detail-mode"
    print(f"Preparing Torres collision debug scene ({mode})...", flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def open_viewer(args: argparse.Namespace, scene_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
    data = mujoco.MjData(model)

    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "torres_driver_door_hinge"
    )
    if joint_id >= 0 and not args.closed_door:
        qadr = model.jnt_qposadr[joint_id]
        if args.door_angle is None:
            data.qpos[qadr] = model.jnt_range[joint_id, 1]
        else:
            data.qpos[qadr] = args.door_angle

    mujoco.mj_forward(model, data)
    print("Viewer tip: toggle Geom Group 3 to compare collision hulls.", flush=True)
    mujoco.viewer.launch(model, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="GR00T-WholeBodyControl root")
    parser.add_argument(
        "--detail-mode",
        "--detail_mode",
        action="store_true",
        dest="detail_mode",
        help="inspect the older detail asset path instead of the normal car",
    )
    parser.add_argument(
        "--reuse-collision",
        action="store_true",
        help="reuse existing collision OBJ files instead of regenerating them",
    )
    parser.add_argument(
        "--keep-entry-collision-hulls",
        action="store_true",
        help="show the unfiltered body collision hulls that can block driver entry",
    )
    parser.add_argument(
        "--collision-alpha",
        type=float,
        default=0.25,
        help="alpha used for collision hull debug geoms",
    )
    parser.add_argument(
        "--max-hull-points",
        type=int,
        default=2500,
        help="max sampled points per generated convex hull",
    )
    parser.add_argument(
        "--closed-door",
        action="store_true",
        help="leave the driver door closed in the debug viewer",
    )
    parser.add_argument(
        "--door-angle",
        type=float,
        default=None,
        help="explicit driver door hinge angle; default opens to joint limit",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else repo_root()
    regenerate_scene(args, root)
    open_viewer(args, resolve_scene(root))


if __name__ == "__main__":
    main()
