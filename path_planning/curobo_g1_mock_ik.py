#!/usr/bin/env python3
"""Minimal cuRoboV2 G1 29DoF IK smoke test.

The test loads cuRobo's Unitree G1 retarget config, computes the current FK
poses at zero joint state, offsets one wrist target, and asks cuRobo IK to
solve for a joint reference. It is intentionally small so it can be used as
the first check before wiring a cuRobo reference provider into SONIC+QP.
"""

from __future__ import annotations

import argparse
import copy
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="unitree_g1_29dof_retarget.yml",
        help="cuRobo robot yaml name",
    )
    parser.add_argument(
        "--target-link",
        default="left_wrist_yaw_link",
        help="Tool frame to offset and solve for",
    )
    parser.add_argument("--dx", type=float, default=0.03, help="Target x offset in meters")
    parser.add_argument("--dy", type=float, default=0.0, help="Target y offset in meters")
    parser.add_argument("--dz", type=float, default=0.0, help="Target z offset in meters")
    parser.add_argument("--num-seeds", type=int, default=32, help="IK seed count")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed IK solves after solver construction",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of untimed warm-up IK solves before measurement",
    )
    parser.add_argument(
        "--all-tool-frames",
        action="store_true",
        help=(
            "Track every tool frame in the robot config. The default smoke test "
            "tracks only --target-link to avoid over-constraining IK."
        ),
    )
    parser.add_argument(
        "--particle-iters",
        type=int,
        default=40,
        help="Override particle optimizer iterations for the smoke test",
    )
    parser.add_argument(
        "--lbfgs-iters",
        type=int,
        default=40,
        help="Override LBFGS optimizer iterations for the smoke test",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Enable cuRobo CUDA graph capture/replay for runtime-like timing",
    )
    parser.add_argument(
        "--self-collision",
        action="store_true",
        help="Enable self-collision checking for the mock solve",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[curobo-mock] {message}", flush=True)


def load_robot_config(robot: str, target_link: str, all_tool_frames: bool):
    if all_tool_frames:
        return robot

    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

    robot_data = copy.deepcopy(load_yaml(join_path(get_robot_configs_path(), robot)))
    if "robot_cfg" in robot_data:
        kinematics = robot_data["robot_cfg"]["kinematics"]
    elif "kinematics" in robot_data:
        kinematics = robot_data["kinematics"]
    else:
        raise KeyError(f"Cannot find kinematics config in {robot}")
    kinematics["tool_frames"] = [target_link]
    kinematics["load_tool_frames_with_mesh"] = False
    cspace_joint_names = kinematics["cspace"]["joint_names"]
    active_joint_names = set(j for j in cspace_joint_names if j.startswith("base_j_"))
    active_joint_names.update(
        ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    )
    if target_link.startswith("left_"):
        active_joint_names.update(
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
        )
    elif target_link.startswith("right_"):
        active_joint_names.update(
            [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
        )
    else:
        raise ValueError(
            "single-tool mock only knows how to lock joints for left/right arm targets; "
            f"got {target_link}"
        )
    kinematics["lock_joints"] = {
        joint_name: 0.0
        for joint_name in cspace_joint_names
        if joint_name not in active_joint_names
    }
    return robot_data


def main() -> int:
    args = parse_args()

    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        log(
            "CUDA is not available in this Python environment; "
            "cuRoboV2 IK requires CUDA for this smoke test.",
        )
        return 2
    log(f"device={torch.cuda.get_device_name(0)}")

    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import GoalToolPose, JointState, Pose

    robot_config = load_robot_config(args.robot, args.target_link, args.all_tool_frames)
    log(
        "creating IK config "
        f"robot={args.robot} target={args.target_link} "
        f"tool_mode={'all' if args.all_tool_frames else 'single'} "
        f"seeds={args.num_seeds}"
    )
    t0 = time.perf_counter()
    cfg = InverseKinematicsCfg.create(
        robot=robot_config,
        num_seeds=args.num_seeds,
        self_collision_check=args.self_collision,
        load_collision_spheres=args.self_collision,
        use_cuda_graph=args.use_cuda_graph,
        override_optimizer_num_iters={
            "particle": args.particle_iters,
            "lbfgs": args.lbfgs_iters,
        },
    )
    log(f"config created in {(time.perf_counter() - t0) * 1000.0:.1f} ms")

    log("constructing IK solver")
    t0 = time.perf_counter()
    ik = InverseKinematics(cfg)
    log(f"IK solver constructed in {(time.perf_counter() - t0) * 1000.0:.1f} ms")

    if args.target_link not in ik.tool_frames:
        log(f"target link {args.target_link!r} is not in tool frames:")
        for name in ik.tool_frames:
            print(f"  - {name}")
        return 1

    log("computing FK at zero joint state")
    q0 = torch.zeros(1, ik.kinematics.get_dof(), device="cuda", dtype=torch.float32)
    current_state = JointState.from_position(q0, joint_names=ik.joint_names)
    fk = ik.kinematics.compute_kinematics(current_state)

    target_poses = {}
    for link_name in ik.tool_frames:
        pose = fk.tool_poses.get_link_pose(link_name, make_contiguous=True)
        position = pose.position.clone()
        quaternion = pose.quaternion.clone()
        if link_name == args.target_link:
            position = position + torch.tensor(
                [[args.dx, args.dy, args.dz]], device="cuda", dtype=torch.float32
            )
        target_poses[link_name] = Pose(
            position=position,
            quaternion=quaternion,
            name=link_name,
            normalize_rotation=False,
        )

    log("building goal")
    goal = GoalToolPose.from_poses(
        target_poses,
        ordered_tool_frames=ik.tool_frames,
        num_goalset=1,
    )
    for i in range(args.warmup):
        if i == 0:
            log(f"running {args.warmup} warm-up IK solve(s)")
        ik.solve_pose(goal, current_state=current_state)
    torch.cuda.synchronize()

    repeat = max(1, args.repeat)
    result = None
    solve_times_ms = []
    log(f"running {repeat} timed IK solve(s)")
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = ik.solve_pose(goal, current_state=current_state)
        torch.cuda.synchronize()
        solve_times_ms.append((time.perf_counter() - t0) * 1000.0)
    assert result is not None

    target_pose = target_poses[args.target_link]
    log(f"robot={args.robot}")
    log(f"dof={ik.kinematics.get_dof()} target_link={args.target_link}")
    log(f"tool_frames={ik.tool_frames}")
    log(f"target_position={target_pose.position.detach().cpu().numpy()[0]}")
    log(f"success={bool(result.success.item())}")
    log(
        "solve_time_ms="
        f"{solve_times_ms[-1]:.3f} last, "
        f"{sum(solve_times_ms) / len(solve_times_ms):.3f} avg, "
        f"{min(solve_times_ms):.3f} min, "
        f"{max(solve_times_ms):.3f} max"
    )
    log(f"position_error_mm={float(result.position_error.item()) * 1000.0:.3f}")
    log(f"js_solution_shape={tuple(result.js_solution.position.shape)}")
    if bool(result.success.item()):
        q_ref = result.js_solution.position[0, 0].detach().cpu().numpy()
        log(f"first_solution_q_ref_rad={q_ref.tolist()}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
