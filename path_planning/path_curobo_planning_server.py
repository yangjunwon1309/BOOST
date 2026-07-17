#!/usr/bin/env python3
"""PATH cuRobo B-spline reference server for SONIC+QP.

Input:
  topic ``path_goal`` on ``--goal-port`` as a msgpack map. The minimal goal is:
    {"target_link": "left_wrist_yaw_link", "position": [x,y,z], "quaternion": [w,x,y,z]}

Output:
  topic ``curobo_plan`` on ``--output-port`` as a msgpack map. The PATH
  state-machine bridge consumes it and republishes the active QP reference schema.

The server keeps publishing the latest cuRobo trajectory at a low heartbeat
rate. The QP client evaluates the B-spline analytically at its own control tick.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq


UPPER_HW_INDICES = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

JOINT_NAME_TO_HW_INDEX = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curobo-root", type=Path, default=Path("external_dependencies/curobo_v2"))
    parser.add_argument("--robot", default="unitree_g1_29dof_retarget.yml")
    parser.add_argument("--goal-bind-host", default="*")
    parser.add_argument("--goal-port", type=int, default=5561)
    parser.add_argument("--goal-topic", default="path_goal")
    parser.add_argument("--output-bind-host", default="*")
    parser.add_argument("--output-port", type=int, default=5562)
    parser.add_argument("--output-topic", default="curobo_plan")
    parser.add_argument("--state-zmq-host", default="127.0.0.1")
    parser.add_argument("--state-zmq-port", type=int, default=5557)
    parser.add_argument("--state-zmq-topic", default="g1_debug")
    parser.add_argument("--publish-hz", type=float, default=5.0)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--min-duration-sec", type=float, default=0.75)
    parser.add_argument("--max-duration-sec", type=float, default=5.0)
    parser.add_argument(
        "--final-hold-sec",
        type=float,
        default=0.75,
        help="Append repeated final control points so the active plan holds its endpoint.",
    )
    parser.add_argument(
        "--plan-clear-grace-sec",
        type=float,
        default=0.25,
        help="Clear the active plan this long after duration_sec + final_hold_sec.",
    )
    parser.add_argument("--control-dt-sec", type=float, default=0.05)
    parser.add_argument(
        "--min-replan-interval-sec",
        type=float,
        default=2.0,
        help="Ignore new non-duplicate goals while an accepted plan is still this fresh.",
    )
    parser.add_argument(
        "--allow-replan-during-execution",
        action="store_true",
        help="Allow immediate replanning before min-replan-interval-sec elapses.",
    )
    parser.add_argument("--duplicate-goal-window-sec", type=float, default=1.0)
    parser.add_argument("--duplicate-goal-eps-m", type=float, default=0.002)
    parser.add_argument(
        "--workspace-min",
        nargs=3,
        type=float,
        default=[0.05, -0.35, -0.05],
        metavar=("X", "Y", "Z"),
        help="Reject EE targets below this base-local xyz bound.",
    )
    parser.add_argument(
        "--workspace-max",
        nargs=3,
        type=float,
        default=[0.45, 0.35, 0.30],
        metavar=("X", "Y", "Z"),
        help="Clamp/reject EE targets above this base-local xyz bound.",
    )
    parser.add_argument(
        "--reject-outside-workspace",
        action="store_true",
        help="Reject workspace violations instead of clamping the target.",
    )
    parser.add_argument(
        "--max-goal-offset-m",
        type=float,
        default=0.10,
        help="If the goal includes current_position, clamp/reject offsets larger than this.",
    )
    parser.add_argument(
        "--reject-large-goal-offset",
        action="store_true",
        help="Reject target-current offsets larger than --max-goal-offset-m instead of clamping.",
    )
    parser.add_argument(
        "--no-retime-plan",
        action="store_true",
        help="Publish cuRobo's native trajectory timing instead of retiming to duration/control-dt.",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--infeasible-cooldown-sec", type=float, default=1.0)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--particle-iters", type=int, default=40)
    parser.add_argument("--lbfgs-iters", type=int, default=40)
    parser.add_argument("--interpolation-dt", type=float, default=0.025)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--self-collision", action="store_true")
    parser.add_argument(
        "--solver",
        choices=["trajopt", "ik"],
        default="trajopt",
        help="Use cuRobo trajectory optimizer or endpoint IK. IK is kept only as a fallback.",
    )
    parser.add_argument(
        "--mock-without-curobo",
        action="store_true",
        help="Publish a small deterministic upper-body spline without constructing cuRobo.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[PATH-cuRobo] {message}", flush=True)


def unpack_topic_message(raw: bytes, topic: str) -> dict:
    topic_bytes = topic.encode()
    if not raw.startswith(topic_bytes):
        raise ValueError("unexpected topic")
    return msgpack.unpackb(raw[len(topic_bytes):], raw=False)


def maybe_read_q_current(socket, topic: str, q_current: np.ndarray) -> np.ndarray:
    try:
        while True:
            raw = socket.recv(zmq.NOBLOCK)
            if not raw.startswith(topic.encode()):
                continue
            payload = msgpack.unpackb(raw[len(topic):], raw=False)
            for key in ("body_q_measured", "body_q", "q_measured"):
                if key in payload:
                    q = np.asarray(payload[key], dtype=np.float64).reshape(-1)
                    if q.shape[0] >= 29 and np.isfinite(q[:29]).all():
                        q_current = q[:29].copy()
                        break
    except zmq.Again:
        pass
    return q_current


def normalize_quaternion_wxyz(values: list[float]) -> list[float]:
    quat = np.asarray(values, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid goal quaternion")
    return (quat / norm).tolist()


def validate_goal(goal: dict, args: argparse.Namespace) -> tuple[bool, str, np.ndarray, list[float], dict]:
    position = np.asarray(goal.get("position", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
    meta: dict = {
        "requested_target_position": position.tolist(),
        "target_clamped": False,
        "target_clamp_reasons": [],
    }
    if not np.isfinite(position).all():
        return False, "nonfinite_target_position", position, [1.0, 0.0, 0.0, 0.0], meta
    lower = np.asarray(args.workspace_min, dtype=np.float64)
    upper = np.asarray(args.workspace_max, dtype=np.float64)
    if np.any(position < lower) or np.any(position > upper):
        if args.reject_outside_workspace:
            return (
                False,
                f"target_outside_workspace min={lower.tolist()} max={upper.tolist()}",
                position,
                [1.0, 0.0, 0.0, 0.0],
                meta,
            )
        position = np.minimum(np.maximum(position, lower), upper)
        meta["target_clamped"] = True
        meta["target_clamp_reasons"].append("workspace")
    if "current_position" in goal:
        current = np.asarray(goal["current_position"], dtype=np.float64).reshape(3)
        if np.isfinite(current).all():
            offset = float(np.linalg.norm(position - current))
            if offset > args.max_goal_offset_m:
                if args.reject_large_goal_offset:
                    return (
                        False,
                        f"target_offset_too_large {offset:.3f}m>{args.max_goal_offset_m:.3f}m",
                        position,
                        [1.0, 0.0, 0.0, 0.0],
                        meta,
                    )
                direction = (position - current) / max(offset, 1.0e-9)
                position = current + direction * args.max_goal_offset_m
                meta["target_clamped"] = True
                meta["target_clamp_reasons"].append("offset")
    try:
        quaternion = normalize_quaternion_wxyz(list(goal.get("quaternion", [1.0, 0.0, 0.0, 0.0])))
    except Exception as exc:
        return False, str(exc), position, [1.0, 0.0, 0.0, 0.0], meta
    meta["target_position"] = position.tolist()
    reason = "ok"
    if meta["target_clamped"]:
        reason = "clamped:" + ",".join(meta["target_clamp_reasons"])
    return True, reason, position, quaternion, meta


def goal_duration(goal: dict, args: argparse.Namespace) -> tuple[float, dict]:
    requested = float(goal.get("duration_sec", args.duration_sec))
    if not np.isfinite(requested) or requested <= 0.0:
        requested = float(args.duration_sec)
    lo = max(1.0e-3, float(args.min_duration_sec))
    hi = max(lo, float(args.max_duration_sec))
    duration = float(np.clip(requested, lo, hi))
    return duration, {
        "requested_duration_sec": requested,
        "duration_clamped": bool(abs(duration - requested) > 1.0e-9),
    }


def q_seed_from_goal(goal: dict, fallback: np.ndarray) -> tuple[np.ndarray, str]:
    for key in ("q_seed", "q_current", "q_ref"):
        if key not in goal:
            continue
        q = np.asarray(goal[key], dtype=np.float64).reshape(-1)
        if q.shape[0] >= 29 and np.isfinite(q[:29]).all():
            return q[:29].copy(), key
    return np.asarray(fallback, dtype=np.float64).reshape(29).copy(), "state_stream"


def is_duplicate_goal(
    goal: dict,
    last_goal: dict | None,
    now: float,
    last_goal_time: float,
    window_sec: float,
    eps_m: float,
) -> bool:
    if last_goal is None or now - last_goal_time > window_sec:
        return False
    if str(goal.get("target_link", "left_wrist_yaw_link")) != str(last_goal.get("target_link", "")):
        return False
    p = np.asarray(goal.get("position", [np.inf, np.inf, np.inf]), dtype=np.float64)
    q = np.asarray(last_goal.get("position", [-np.inf, -np.inf, -np.inf]), dtype=np.float64)
    return bool(np.linalg.norm(p - q) <= eps_m)


def load_robot_config(robot: str, target_link: str):
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

    robot_data = copy.deepcopy(load_yaml(join_path(get_robot_configs_path(), robot)))
    kinematics = robot_data.get("robot_cfg", robot_data).get("kinematics")
    if kinematics is None:
        raise KeyError(f"Cannot find kinematics config in {robot}")
    kinematics["tool_frames"] = [target_link]
    kinematics["load_tool_frames_with_mesh"] = False

    cspace_joint_names = kinematics["cspace"]["joint_names"]
    active = {j for j in cspace_joint_names if j.startswith("base_j_")}
    active.update(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
    side = "left" if target_link.startswith("left_") else "right"
    active.update(
        [
            f"{side}_shoulder_pitch_joint",
            f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint",
            f"{side}_elbow_joint",
            f"{side}_wrist_roll_joint",
            f"{side}_wrist_pitch_joint",
            f"{side}_wrist_yaw_joint",
        ]
    )
    kinematics["lock_joints"] = {name: 0.0 for name in cspace_joint_names if name not in active}
    return robot_data


class CuroboIkPlanner:
    def __init__(self, args: argparse.Namespace, target_link: str):
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for cuRobo IK")
        self.torch = torch
        self.target_link = target_link
        robot_config = load_robot_config(args.robot, target_link)
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
        self.ik = InverseKinematics(cfg)
        self.current_state = None
        log(
            f"IK ready: target={target_link}, dof={self.ik.kinematics.get_dof()}, "
            f"seeds={args.num_seeds}"
        )

    def solve(self, position_xyz: list[float], quaternion_wxyz: list[float]) -> tuple[bool, np.ndarray]:
        from curobo.types import GoalToolPose, JointState, Pose

        q0 = self.torch.zeros(1, self.ik.kinematics.get_dof(), device="cuda", dtype=self.torch.float32)
        self.current_state = JointState.from_position(q0, joint_names=self.ik.joint_names)
        position = self.torch.tensor([position_xyz], device="cuda", dtype=self.torch.float32)
        quaternion = self.torch.tensor([quaternion_wxyz], device="cuda", dtype=self.torch.float32)
        pose = Pose(
            position=position,
            quaternion=quaternion,
            name=self.target_link,
            normalize_rotation=True,
        )
        goal = GoalToolPose.from_poses(
            {self.target_link: pose},
            ordered_tool_frames=self.ik.tool_frames,
            num_goalset=1,
        )
        self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self.ik.solve_pose(goal, current_state=self.current_state)
        self.torch.cuda.synchronize()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        success = bool(result.success.item())
        if not success:
            log(f"IK infeasible/failed solve_ms={solve_ms:.2f}")
            return False, np.zeros(29, dtype=np.float64)
        q_full = result.js_solution.position[0, 0].detach().cpu().numpy().astype(np.float64)
        q_29 = q_full[-29:] if q_full.shape[0] >= 29 else np.pad(q_full, (0, 29 - q_full.shape[0]))
        log(
            f"IK success solve_ms={solve_ms:.2f}, "
            f"pos_err_mm={float(result.position_error.item()) * 1000.0:.3f}"
        )
        return True, q_29


class CuroboTrajectoryPlanner:
    def __init__(self, args: argparse.Namespace, target_link: str):
        import torch
        from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for cuRobo trajectory optimization")
        self.torch = torch
        self.target_link = target_link
        robot_config = load_robot_config(args.robot, target_link)
        lbfgs_iters = int(args.lbfgs_iters)
        if lbfgs_iters % 25 != 0:
            adjusted = ((lbfgs_iters + 24) // 25) * 25
            log(
                f"TrajOpt lbfgs_iters={lbfgs_iters} is not a multiple of 25; "
                f"using {adjusted}"
            )
            lbfgs_iters = adjusted
        cfg = TrajectoryOptimizerCfg.create(
            robot=robot_config,
            num_seeds=args.num_seeds,
            self_collision_check=args.self_collision,
            load_collision_spheres=args.self_collision,
            use_cuda_graph=args.use_cuda_graph,
            interpolation_dt=args.interpolation_dt,
            override_optimizer_num_iters={"lbfgs": lbfgs_iters},
        )
        self.trajopt = TrajectoryOptimizer(cfg)
        self.joint_names = list(self.trajopt.joint_names)
        log(
            f"TrajOpt ready: target={target_link}, dof={self.trajopt.kinematics.get_dof()}, "
            f"seeds={args.num_seeds}, interpolation_dt={args.interpolation_dt:g}"
        )

    def _active_q_from_29(self, q_29: np.ndarray) -> np.ndarray:
        """Map G1 29DoF hardware q into cuRobo active cspace order."""
        # cuRobo's G1 retarget config has six virtual base joints followed by
        # the 29 hardware joints in the same order used by SONIC/G1.
        full35 = np.zeros(35, dtype=np.float32)
        full35[-29:] = np.asarray(q_29, dtype=np.float32).reshape(29)
        active = []
        for name in self.joint_names:
            if name.startswith("base_j_"):
                active.append(0.0)
            else:
                hw_index = JOINT_NAME_TO_HW_INDEX.get(name)
                active.append(q_29[hw_index] if hw_index is not None else 0.0)
        return np.asarray(active, dtype=np.float32)

    @staticmethod
    def _to_q29_from_full_or_active(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] >= 35:
            return q[-29:].copy()
        if q.shape[0] == 29:
            return q.copy()
        out = np.zeros(29, dtype=np.float64)
        # Active set is usually base + waist + one arm. Fill from the tail as a
        # conservative fallback; the dense full trajectory path should normally
        # be available and preferred when knot is not full DoF.
        n = min(q.shape[0], 29)
        out[:n] = q[:n]
        return out

    def _active_points_to_q29(
        self,
        points: np.ndarray,
        q_reference: np.ndarray,
        joint_names: list[str] | None = None,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        q_reference = np.asarray(q_reference, dtype=np.float64).reshape(29)
        if points.shape[-1] in (29, 35):
            return np.stack([self._to_q29_from_full_or_active(row) for row in points], axis=0)
        names = list(joint_names or self.joint_names)
        rows = []
        for point in points:
            q29 = q_reference.copy()
            for value, name in zip(point, names):
                hw_index = JOINT_NAME_TO_HW_INDEX.get(name)
                if hw_index is not None:
                    q29[hw_index] = float(value)
            rows.append(q29)
        return np.stack(rows, axis=0)

    def solve(self, q_current: np.ndarray, position_xyz: list[float], quaternion_wxyz: list[float]):
        from curobo.types import GoalToolPose, JointState, Pose

        q_active = self._active_q_from_29(q_current)
        current_state = JointState.from_position(
            self.torch.as_tensor(q_active.reshape(1, -1), device="cuda", dtype=self.torch.float32),
            joint_names=self.joint_names,
        )
        position = self.torch.as_tensor(
            np.asarray(position_xyz, dtype=np.float32).reshape(1, 1, 1, 1, 3),
            device="cuda",
        )
        quaternion = self.torch.as_tensor(
            np.asarray(quaternion_wxyz, dtype=np.float32).reshape(1, 1, 1, 1, 4),
            device="cuda",
        )
        goal = GoalToolPose(
            tool_frames=self.trajopt.tool_frames,
            position=position,
            quaternion=quaternion,
        )
        self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self.trajopt.solve_pose(
            goal_tool_poses=goal,
            current_state=current_state,
            return_seeds=1,
        )
        self.torch.cuda.synchronize()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        success = bool(result is not None and result.success is not None and result.success.any().item())
        if not success:
            log(f"TrajOpt infeasible/failed solve_ms={solve_ms:.2f}")
            return False, [], 0.0, {"solve_time_ms": solve_ms}

        # Prefer the actual B-spline knots optimized by cuRobo. If the knots are
        # in active DoF only, use cuRobo's dense full-DoF interpolated trajectory
        # as the published points so QP still follows the optimized cuRobo path.
        source = "curobo_trajopt_knot"
        knot = getattr(result.js_solution, "knot", None)
        knot_dt = getattr(result.js_solution, "knot_dt", None)
        control_dt = float(knot_dt.reshape(-1)[0].detach().cpu().item()) if knot_dt is not None else 0.05
        points: np.ndarray | None = None
        if knot is not None:
            knot_np = knot.detach().cpu().numpy()
            # Shape is normally [batch, seed, horizon, dof].
            candidate = np.asarray(knot_np[0, 0] if knot_np.ndim == 4 else knot_np.reshape(-1, knot_np.shape[-1]))
            points = self._active_points_to_q29(candidate, q_current, self.joint_names)

        if points is None:
            source = "curobo_trajopt_interpolated"
            dense = result.get_interpolated_plan()
            dense_np = dense.position.detach().cpu().numpy()
            candidate = np.asarray(dense_np[0, 0] if dense_np.ndim == 4 else dense_np.reshape(-1, dense_np.shape[-1]))
            dense_names = list(getattr(dense, "joint_names", None) or self.joint_names)
            points = self._active_points_to_q29(candidate, q_current, dense_names)
            control_dt = float(getattr(self.trajopt.config, "interpolation_dt", 0.025))

        pos_error_mm = (
            float(result.position_error.reshape(-1)[0].detach().cpu().item()) * 1000.0
            if result.position_error is not None
            else -1.0
        )
        rot_error = (
            float(result.rotation_error.reshape(-1)[0].detach().cpu().item())
            if result.rotation_error is not None
            else -1.0
        )
        log(
            f"TrajOpt success solve_ms={solve_ms:.2f}, source={source}, "
            f"points={points.shape}, dt={control_dt:.4f}, pos_err_mm={pos_error_mm:.3f}"
        )
        meta = {
            "solve_time_ms": solve_ms,
            "position_error_mm": pos_error_mm,
            "rotation_error": rot_error,
            "trajectory_source": source,
        }
        return True, points.tolist(), control_dt, meta


def make_clamped_bspline(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration_sec: float,
    dt_sec: float,
    final_hold_sec: float = 0.0,
) -> list[list[float]]:
    q_start = np.asarray(q_start, dtype=np.float64).reshape(29)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(29)
    segments = max(4, int(round(duration_sec / max(dt_sec, 1.0e-6))))
    n_internal = max(0, segments - 4)
    rows = [q_start.copy() for _ in range(4)]
    for j in range(n_internal):
        alpha = float((j + 1) / (n_internal + 1))
        rows.append((1.0 - alpha) * q_start + alpha * q_goal)
    rows.extend([q_goal.copy() for _ in range(4)])
    hold_steps = max(0, int(np.ceil(max(0.0, final_hold_sec) / max(dt_sec, 1.0e-6))))
    rows.extend([q_goal.copy() for _ in range(hold_steps)])
    return [row.tolist() for row in rows]


def retime_points_as_clamped_bspline(
    points: list[list[float]] | np.ndarray,
    duration_sec: float,
    dt_sec: float,
    final_hold_sec: float = 0.0,
) -> list[list[float]]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 29 or pts.shape[0] < 1:
        raise ValueError(f"cannot retime invalid plan points shape={pts.shape}")
    if pts.shape[0] == 1:
        return make_clamped_bspline(pts[0], pts[0], duration_sec, dt_sec, final_hold_sec)
    segments = max(4, int(round(duration_sec / max(dt_sec, 1.0e-6))))
    n_internal = max(0, segments - 4)
    rows = [pts[0].copy() for _ in range(4)]
    for j in range(n_internal):
        alpha = float((j + 1) / (n_internal + 1))
        source = alpha * (pts.shape[0] - 1)
        lo = int(np.floor(source))
        hi = min(lo + 1, pts.shape[0] - 1)
        frac = source - lo
        rows.append((1.0 - frac) * pts[lo] + frac * pts[hi])
    rows.extend([pts[-1].copy() for _ in range(4)])
    hold_steps = max(0, int(np.ceil(max(0.0, final_hold_sec) / max(dt_sec, 1.0e-6))))
    rows.extend([pts[-1].copy() for _ in range(hold_steps)])
    return [row.tolist() for row in rows]


def publish_plan(socket, topic: str, payload: dict) -> None:
    socket.send(topic.encode() + msgpack.packb(payload, use_bin_type=True))


def main() -> int:
    args = parse_args()
    curobo_root = args.curobo_root.expanduser().resolve()
    if not args.mock_without_curobo:
        sys.path.insert(0, str(curobo_root))

    context = zmq.Context()
    goal_socket = context.socket(zmq.SUB)
    goal_socket.setsockopt(zmq.SUBSCRIBE, args.goal_topic.encode())
    goal_socket.setsockopt(zmq.RCVTIMEO, 1)
    goal_socket.bind(f"tcp://{args.goal_bind_host}:{args.goal_port}")

    output_socket = context.socket(zmq.PUB)
    output_socket.setsockopt(zmq.SNDHWM, 1)
    output_socket.setsockopt(zmq.LINGER, 0)
    output_socket.bind(f"tcp://{args.output_bind_host}:{args.output_port}")

    state_socket = context.socket(zmq.SUB)
    state_socket.setsockopt(zmq.SUBSCRIBE, args.state_zmq_topic.encode())
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.setsockopt(zmq.RCVTIMEO, 1)
    state_socket.connect(f"tcp://{args.state_zmq_host}:{args.state_zmq_port}")

    log(f"goal tcp://{args.goal_bind_host}:{args.goal_port}/{args.goal_topic}")
    log(f"publishing tcp://{args.output_bind_host}:{args.output_port}/{args.output_topic}")
    log(
        "state tcp://"
        f"{args.state_zmq_host}:{args.state_zmq_port}/{args.state_zmq_topic}"
    )

    planner: CuroboIkPlanner | CuroboTrajectoryPlanner | None = None
    q_current = np.zeros(29, dtype=np.float64)
    latest_payload: dict | None = None
    latest_payload_expires_at = -1.0
    plan_id = 0
    accepted_plan_id = 0
    last_goal: dict | None = None
    last_goal_time = -1.0e9
    last_accepted_time = -1.0e9
    last_failure_time = -1.0e9
    consecutive_failures = 0
    publish_period = 1.0 / max(args.publish_hz, 1e-6)
    next_pub = 0.0

    try:
        while True:
            now = time.monotonic()
            q_current = maybe_read_q_current(state_socket, args.state_zmq_topic, q_current)
            try:
                raw = goal_socket.recv(zmq.NOBLOCK)
                goal = unpack_topic_message(raw, args.goal_topic)
                plan_id += 1
                target_link = str(goal.get("target_link", "left_wrist_yaw_link"))
                force_replan = bool(goal.get("force_replan", False))
                goal_uid = str(goal.get("goal_uid", ""))
                valid_goal, reason, position_arr, quaternion, goal_meta = validate_goal(goal, args)
                plan_duration_sec, duration_meta = goal_duration(goal, args)
                final_hold_sec = max(0.0, float(goal.get("final_hold_sec", args.final_hold_sec)))
                now_us = time.monotonic_ns() // 1000
                last_goal_uid = str(last_goal.get("goal_uid", "")) if last_goal is not None else ""
                if goal_uid and goal_uid == last_goal_uid:
                    log(f"ignored repeated goal message plan_id={plan_id} uid={goal_uid[:8]}")
                    continue
                if (not force_replan) and is_duplicate_goal(
                    goal,
                    last_goal,
                    now,
                    last_goal_time,
                    args.duplicate_goal_window_sec,
                    args.duplicate_goal_eps_m,
                ):
                    log(f"ignored duplicate goal plan_id={plan_id} target={target_link}")
                    continue
                last_goal = copy.deepcopy(goal)
                last_goal_time = now

                if not valid_goal:
                    log(f"rejected goal plan_id={plan_id} target={target_link}: {reason}")
                    latest_payload = {
                        "source": "curobo",
                        "control_mode": "plan_pending",
                        "feasible": False,
                        "failure_reason": reason,
                        "plan_id": accepted_plan_id,
                        "timestamp_us": now_us,
                        "t_start_us": now_us,
                        "control_dt_sec": float(args.control_dt_sec),
                        "duration_sec": float(plan_duration_sec),
                        "final_hold_sec": float(final_hold_sec),
                        "target_link": target_link,
                        "target_position": position_arr.tolist(),
                        "control_points": [],
                        **goal_meta,
                        **duration_meta,
                    }
                    latest_payload_expires_at = now + float(args.plan_clear_grace_sec)
                    continue

                if (
                    latest_payload is not None
                    and latest_payload.get("feasible", False)
                    and not args.allow_replan_during_execution
                    and not force_replan
                    and now - last_accepted_time < args.min_replan_interval_sec
                ):
                    log(
                        f"ignored goal plan_id={plan_id}: active plan {accepted_plan_id} "
                        f"is still inside min_replan_interval={args.min_replan_interval_sec:.2f}s"
                    )
                    continue

                if (not force_replan) and now - last_failure_time < args.infeasible_cooldown_sec:
                    log(
                        f"ignored goal plan_id={plan_id}: infeasible cooldown "
                        f"{args.infeasible_cooldown_sec:.2f}s"
                    )
                    continue

                position = position_arr.tolist()
                q_plan_start, q_plan_start_source = q_seed_from_goal(goal, q_current)

                meta = {}
                if args.mock_without_curobo:
                    q_goal = q_plan_start.copy()
                    q_goal[15] += 0.10
                    q_goal[18] += 0.20
                    feasible = True
                    control_points = make_clamped_bspline(
                        q_plan_start,
                        q_goal,
                        plan_duration_sec,
                        args.control_dt_sec,
                        final_hold_sec,
                    )
                    control_dt_sec = args.control_dt_sec
                else:
                    if planner is None or planner.target_link != target_link:
                        planner = (
                            CuroboTrajectoryPlanner(args, target_link)
                            if args.solver == "trajopt"
                            else CuroboIkPlanner(args, target_link)
                    )
                    if isinstance(planner, CuroboTrajectoryPlanner):
                        feasible, control_points, control_dt_sec, meta = planner.solve(
                            q_plan_start, position, quaternion
                        )
                    else:
                        feasible, q_goal = planner.solve(position, quaternion)
                        control_points = make_clamped_bspline(
                            q_plan_start,
                            q_goal,
                            plan_duration_sec,
                            args.control_dt_sec,
                            final_hold_sec,
                        )
                        control_dt_sec = args.control_dt_sec
                        meta = {"trajectory_source": "ik_endpoint_clamped_bspline"}
                meta = {
                    **meta,
                    **goal_meta,
                    **duration_meta,
                    "q_start_source": q_plan_start_source,
                    "final_hold_sec": float(final_hold_sec),
                }

                if feasible:
                    consecutive_failures = 0
                    accepted_plan_id = plan_id
                    last_accepted_time = now
                    native_points = len(control_points)
                    native_dt_sec = float(control_dt_sec)
                    if not args.no_retime_plan:
                        control_points = retime_points_as_clamped_bspline(
                            control_points,
                            plan_duration_sec,
                            args.control_dt_sec,
                            final_hold_sec,
                        )
                        control_dt_sec = args.control_dt_sec
                        meta = {
                            **meta,
                            "retimed": True,
                            "native_control_points": native_points,
                            "native_control_dt_sec": native_dt_sec,
                        }
                    else:
                        meta = {**meta, "retimed": False}
                else:
                    consecutive_failures += 1
                    last_failure_time = now
                    if consecutive_failures >= args.max_consecutive_failures:
                        log(
                            f"resetting planner after {consecutive_failures} consecutive failures"
                        )
                        planner = None
                        consecutive_failures = 0

                if not feasible:
                    latest_payload = {
                        "source": "curobo",
                        "control_mode": "plan_pending",
                        "feasible": False,
                        "failure_reason": "planner_infeasible",
                        "plan_id": accepted_plan_id,
                        "timestamp_us": now_us,
                        "t_start_us": now_us,
                        "control_dt_sec": float(args.control_dt_sec),
                        "duration_sec": float(plan_duration_sec),
                        "final_hold_sec": float(final_hold_sec),
                        "target_link": target_link,
                        "target_position": position,
                        "control_points": [],
                        **meta,
                    }
                    latest_payload_expires_at = now + float(args.plan_clear_grace_sec)
                    log(
                        f"plan_id={plan_id} feasible=False reason=planner_infeasible "
                        f"consecutive_failures={consecutive_failures}"
                    )
                    continue

                latest_payload = {
                    "source": "curobo",
                    "control_mode": "plan_exec",
                    "feasible": True,
                    "plan_id": accepted_plan_id,
                    "timestamp_us": now_us,
                    "t_start_us": now_us,
                    "control_dt_sec": float(control_dt_sec),
                    "duration_sec": float(plan_duration_sec),
                    "final_hold_sec": float(final_hold_sec),
                    "total_duration_sec": float(plan_duration_sec + final_hold_sec),
                    "target_link": target_link,
                    "target_position": position,
                    "target_quaternion": quaternion,
                    "q_start_source": q_plan_start_source,
                    "planning_frames": max(1, int(round(plan_duration_sec / max(control_dt_sec, 1.0e-6)))),
                    "total_frames": max(1, int(round((plan_duration_sec + final_hold_sec) / max(control_dt_sec, 1.0e-6)))),
                    "control_points": control_points,
                    **meta,
                }
                latest_payload_expires_at = (
                    now
                    + float(plan_duration_sec)
                    + float(final_hold_sec)
                    + float(args.plan_clear_grace_sec)
                )
                log(
                    f"plan_id={accepted_plan_id} feasible=True target={np.round(position_arr, 4).tolist()} "
                    f"cps={len(control_points)} dt={control_dt_sec:.3f}s "
                    f"duration={plan_duration_sec:.2f}s hold={final_hold_sec:.2f}s "
                    f"retimed={not args.no_retime_plan} q_start={q_plan_start_source} "
                    f"goal={reason}"
                )
            except zmq.Again:
                pass

            if (
                latest_payload is not None
                and latest_payload.get("feasible", False)
                and latest_payload_expires_at > 0.0
                and now >= latest_payload_expires_at
            ):
                expired_plan_id = int(latest_payload.get("plan_id", accepted_plan_id))
                log(f"clearing expired plan id={expired_plan_id}")
                now_us = time.monotonic_ns() // 1000
                latest_payload = {
                    "source": "curobo",
                    "control_mode": "plan_pending",
                    "feasible": False,
                    "failure_reason": "plan_expired",
                    "plan_id": expired_plan_id,
                    "timestamp_us": now_us,
                    "t_start_us": now_us,
                    "control_dt_sec": float(args.control_dt_sec),
                    "duration_sec": 0.0,
                    "final_hold_sec": 0.0,
                    "total_duration_sec": 0.0,
                    "target_link": "",
                    "target_position": [0.0, 0.0, 0.0],
                    "control_points": [],
                }
                latest_payload_expires_at = -1.0

            if latest_payload is not None and now >= next_pub:
                latest_payload["timestamp_us"] = time.monotonic_ns() // 1000
                publish_plan(output_socket, args.output_topic, latest_payload)
                next_pub = now + publish_period
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        return 0
    finally:
        goal_socket.close()
        output_socket.close()
        state_socket.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
