#!/usr/bin/env python3
"""Run the G1+Revo2 MuJoCo model as a Unitree DDS robot endpoint.

HOMIE owns rt/lowcmd. This process consumes those full-body commands, applies
the transmitted per-joint PD gains as MuJoCo torques, and publishes simulated
rt/lowstate plus Revo2 hand state. PATH's existing G1 bridge can therefore
close its measured-state loop against the simulator.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from boost.sim.mujoco.hand_backends import REVO2_JOINT_SUFFIXES
from boost.sim.mujoco.surface_query_service import (
    MuJoCoSurfaceQueryService,
    load_model_with_surface_probes,
)
from boost.sim.mujoco.unitree_sdk2py_bridge import REVO2_BACKEND, UnitreeSdk2Bridge


ROOT = Path(__file__).resolve().parents[3]
BODY_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {kind.name} '{name}'")
    return int(object_id)


def _joint_addresses(model: mujoco.MjModel, names: tuple[str, ...]):
    ids = np.asarray([_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    return ids, model.jnt_qposadr[ids].astype(int), model.jnt_dofadr[ids].astype(int)


def _actuator_by_joint(model: mujoco.MjModel) -> dict[int, int]:
    result: dict[int, int] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            result[joint_id] = actuator_id
    return result


def _torque_limit(model: mujoco.MjModel, actuator_id: int, torque: float) -> float:
    if bool(model.actuator_ctrllimited[actuator_id]):
        low, high = model.actuator_ctrlrange[actuator_id]
        return float(np.clip(torque, low, high))
    return float(torque)


def _format_command_status(status: dict) -> str:
    count = int(status["count"])
    age = status["age_sec"]
    if count == 0 or age is None:
        return "waiting"
    return f"{count}@{age * 1000:.0f}ms"


class KeyboardLocomotion:
    """Latched MuJoCo viewer keys translated to HOMIE body-frame cmd_vel."""

    SPEED_STEP = 0.1
    INITIAL_SPEED_SCALE = 0.5
    MIN_SPEED_SCALE = 0.2

    def __init__(self, max_cmd: np.ndarray):
        self.max_cmd = np.asarray(max_cmd, dtype=np.float32).reshape(3)
        self._lock = threading.Lock()
        self._direction = np.zeros(3, dtype=np.float32)
        self._speed_scale = self.INITIAL_SPEED_SCALE

    def on_key(self, keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except (TypeError, ValueError):
            return

        with self._lock:
            if key == "w":
                self._direction[0] = 1.0
            elif key == "s":
                self._direction[0] = -1.0
            elif key == "a":
                self._direction[1] = 1.0
            elif key == "d":
                self._direction[1] = -1.0
            elif key == "q":
                self._direction[2] = 1.0
            elif key == "r":
                self._direction[2] = -1.0
            elif key == "o":
                self._speed_scale = max(
                    self.MIN_SPEED_SCALE, self._speed_scale - self.SPEED_STEP
                )
            elif key == "p":
                self._speed_scale = min(1.0, self._speed_scale + self.SPEED_STEP)
            elif key == " ":
                self._direction[:] = 0.0

    def command(self) -> tuple[np.ndarray, float]:
        with self._lock:
            cmd = self._direction * self.max_cmd * self._speed_scale
            return cmd, self._speed_scale


def _state_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_qpos: np.ndarray,
    body_dof: np.ndarray,
    body_actuators: np.ndarray,
    left_hand_qpos: np.ndarray,
    left_hand_dof: np.ndarray,
    left_hand_actuators: np.ndarray,
    right_hand_qpos: np.ndarray,
    right_hand_dof: np.ndarray,
    right_hand_actuators: np.ndarray,
    torso_id: int,
) -> dict:
    torso_velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, torso_velocity, 1
    )
    # MuJoCo returns angular then linear velocity; the Sonic bridge contract
    # stores linear then angular velocity in this observation field.
    secondary_velocity = np.concatenate((torso_velocity[3:6], torso_velocity[0:3]))
    return {
        "time": float(data.time),
        "body_q": data.qpos[body_qpos].copy(),
        "body_dq": data.qvel[body_dof].copy(),
        "body_ddq": data.qacc[body_dof].copy(),
        "body_tau_est": data.actuator_force[body_actuators].copy(),
        "floating_base_pose": data.qpos[:7].copy(),
        "floating_base_vel": data.qvel[:6].copy(),
        "floating_base_acc": data.qacc[:6].copy(),
        "secondary_imu_quat": data.xquat[torso_id].copy(),
        "secondary_imu_vel": secondary_velocity,
        "left_hand_q": data.qpos[left_hand_qpos].copy(),
        "left_hand_dq": data.qvel[left_hand_dof].copy(),
        "left_hand_tau_est": data.actuator_force[left_hand_actuators].copy(),
        "right_hand_q": data.qpos[right_hand_qpos].copy(),
        "right_hand_dq": data.qvel[right_hand_dof].copy(),
        "right_hand_tau_est": data.actuator_force[right_hand_actuators].copy(),
    }


def _make_bridge_config(sim_config: dict, hand_config: dict) -> dict:
    return {
        "ROBOT_TYPE": "g1",
        "NUM_MOTORS": int(sim_config["num_body_motors"]),
        "NUM_HAND_MOTORS": int(sim_config["num_hand_motors"]),
        "NUM_HAND_JOINTS": int(sim_config["num_hand_joints"]),
        "HAND_BACKEND": str(sim_config["hand_backend"]),
        "USE_SENSOR": False,
        "BRAINCO_Q_MIN": hand_config["q_min"],
        "BRAINCO_Q_MAX": hand_config["q_max"],
        "BRAINCO_BINARY_OPEN": hand_config["binary_open"],
        "BRAINCO_BINARY_CLOSE": hand_config["binary_close"],
        "BRAINCO_LEGACY_CLOSE_THRESHOLD": hand_config["legacy_close_threshold"],
        "BRAINCO_ACCEPT_LEGACY_DEX3": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sim/mujoco_revo2.yaml")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--homie-config", type=Path, default=None,
                        help="HOMIE gains/default pose used only during startup before rt/lowcmd arrives.")
    parser.add_argument("--interface", default=None, help="DDS interface shared with HOMIE (normally lo).")
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--initial-height", type=float, default=None)
    parser.add_argument("--hand-kp", type=float, default=None)
    parser.add_argument("--hand-kd", type=float, default=None)
    parser.add_argument(
        "--surface-query-endpoint",
        default=None,
        help="bind a local ZMQ endpoint for contact-guided hand retargeting",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        sim_config = yaml.safe_load(stream)
    hand_config_path = ROOT / sim_config["hand_config"]
    with hand_config_path.open("r", encoding="utf-8") as stream:
        hand_config = yaml.safe_load(stream)
    args.scene = args.scene or ROOT / sim_config["scene"]
    args.homie_config = args.homie_config or ROOT / sim_config["homie_config"]
    args.interface = args.interface or sim_config.get("interface", "lo")
    args.domain_id = args.domain_id if args.domain_id is not None else int(sim_config.get("domain_id", 0))
    args.sim_dt = args.sim_dt if args.sim_dt is not None else float(sim_config.get("sim_dt", 0.005))
    args.initial_height = (
        args.initial_height
        if args.initial_height is not None
        else float(sim_config.get("initial_height", 0.793))
    )
    args.hand_kp = args.hand_kp if args.hand_kp is not None else float(sim_config["hand_pd"]["kp"])
    args.hand_kd = args.hand_kd if args.hand_kd is not None else float(sim_config["hand_pd"]["kd"])
    with args.homie_config.open("r", encoding="utf-8") as stream:
        homie = yaml.safe_load(stream)
    default_q = np.asarray(homie["default_angles"], dtype=np.float64).reshape(29)
    startup_kp = np.asarray(homie["kps"], dtype=np.float64).reshape(29)
    startup_kd = np.asarray(homie["kds"], dtype=np.float64).reshape(29)

    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = UnitreeSdk2Bridge(_make_bridge_config(sim_config, hand_config))
    keyboard = KeyboardLocomotion(np.asarray(homie["max_cmd"], dtype=np.float32))
    nav_cmd_publisher = ChannelPublisher("rt/nav_cmd", String_)
    nav_cmd_publisher.Init()
    surface_config = sim_config.get("surface_query", {})
    probe_radius_m = float(surface_config.get("probe_radius_m", 0.0005))
    if args.surface_query_endpoint:
        model = load_model_with_surface_probes(
            args.scene.resolve(), probe_radius_m=probe_radius_m
        )
    else:
        model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    model.opt.timestep = float(args.sim_dt)
    data = mujoco.MjData(model)
    if model.nu != 41:
        raise RuntimeError(f"Expected 29 body + 12 Revo2 actuators, found {model.nu}")

    data.qpos[:7] = [0.0, 0.0, args.initial_height, 1.0, 0.0, 0.0, 0.0]
    body_ids, body_qpos, body_dof = _joint_addresses(model, BODY_JOINTS)
    data.qpos[body_qpos] = default_q
    mujoco.mj_forward(model, data)

    left_names = tuple(f"left_{suffix}" for suffix in REVO2_JOINT_SUFFIXES)
    right_names = tuple(f"right_{suffix}" for suffix in REVO2_JOINT_SUFFIXES)
    left_ids, left_qpos, left_dof = _joint_addresses(model, left_names)
    right_ids, right_qpos, right_dof = _joint_addresses(model, right_names)
    actuator_map = _actuator_by_joint(model)
    try:
        body_actuators = np.asarray([actuator_map[int(j)] for j in body_ids], dtype=int)
        left_hand_actuators = np.asarray([actuator_map[int(j)] for j in left_ids[:6]], dtype=int)
        right_hand_actuators = np.asarray([actuator_map[int(j)] for j in right_ids[:6]], dtype=int)
    except KeyError as exc:
        raise RuntimeError(f"Model has no torque actuator for joint id {exc.args[0]}") from exc
    torso_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    revo2 = bridge.revo2_adapter
    surface_service = None
    if args.surface_query_endpoint:
        object_geom_names = [
            str(name) for name in surface_config.get("object_geom_names", [])
        ]
        surface_service = MuJoCoSurfaceQueryService(
            model,
            data,
            endpoint=args.surface_query_endpoint,
            object_geom_names=object_geom_names,
            distmax_m=float(surface_config.get("distmax_m", 0.03)),
            probe_radius_m=probe_radius_m,
        )
        print(
            f"[BOOST MuJoCo surface] bound {args.surface_query_endpoint}; "
            f"object collision geoms={surface_service.object_geom_count}",
            flush=True,
        )
        if surface_service.object_geom_count == 0:
            print(
                "[BOOST MuJoCo surface] no object geoms configured; contact guidance "
                "will fall back to the original SBR targets",
                flush=True,
            )
    viewer = (
        None
        if args.headless
        else mujoco.viewer.launch_passive(
            model, data, key_callback=keyboard.on_key
        )
    )
    sim_period = float(model.opt.timestep)
    next_tick = time.monotonic()
    next_report = next_tick
    next_nav_publish = next_tick
    nav_cmd_sequence = 0
    tick = 0
    print(
        f"[BOOST MuJoCo] scene={args.scene} interface={args.interface} "
        f"DDS domain={args.domain_id} dt={sim_period:g}s nq={model.nq} nu={model.nu}; "
        "publishing rt/lowstate, rt/nav_cmd, and Revo2 state; waiting for "
        "HOMIE rt/lowcmd and Revo2 DDS commands",
        flush=True,
    )
    if viewer is not None:
        print(
            "[MuJoCo keyboard] W/S forward/back, A/D left/right, "
            "Q/R yaw left/right; O/P change linear and angular speed "
            "(20-100%); Space stops. Direction keys latch until reversed or stopped.",
            flush=True,
        )

    try:
        while viewer is None or viewer.is_running():
            now = time.monotonic()
            if now >= next_nav_publish:
                cmd, _speed_scale = keyboard.command()
                nav_payload = json.dumps(
                    {
                        "vx": float(cmd[0]),
                        "vy": float(cmd[1]),
                        "wz": float(cmd[2]),
                        "seq": nav_cmd_sequence,
                        "state": "keyboard",
                    },
                    separators=(",", ":"),
                )
                nav_message = String_(nav_payload)
                nav_cmd_publisher.Write(nav_message)
                nav_cmd_sequence += 1
                next_nav_publish = now + 0.02

            if surface_service is not None:
                surface_service.poll_once()
            obs = _state_observation(
                model, data, body_qpos, body_dof, body_actuators,
                left_qpos, left_dof, left_hand_actuators,
                right_qpos, right_dof, right_hand_actuators, torso_id,
            )
            bridge.PublishLowState(obs)

            with bridge.low_cmd_lock:
                command = bridge.low_cmd
                have_body_command = bool(bridge.low_cmd_received)
            for i, (qpos_adr, dof_adr, actuator_id) in enumerate(
                zip(body_qpos, body_dof, body_actuators)
            ):
                if have_body_command:
                    motor = command.motor_cmd[i]
                    torque = (
                        float(motor.tau)
                        + float(motor.kp) * (float(motor.q) - data.qpos[qpos_adr])
                        + float(motor.kd) * (float(motor.dq) - data.qvel[dof_adr])
                    )
                else:
                    torque = (
                        startup_kp[i] * (default_q[i] - data.qpos[qpos_adr])
                        - startup_kd[i] * data.qvel[dof_adr]
                    )
                data.ctrl[actuator_id] = _torque_limit(model, actuator_id, torque)

            left_target = revo2.normalized_to_actuated(bridge.get_hand_command_q(True))
            right_target = revo2.normalized_to_actuated(bridge.get_hand_command_q(False))
            for target, qpos_adrs, dof_adrs, actuator_ids in (
                (left_target, left_qpos[:6], left_dof[:6], left_hand_actuators),
                (right_target, right_qpos[:6], right_dof[:6], right_hand_actuators),
            ):
                for i, (q_des, qpos_adr, dof_adr, actuator_id) in enumerate(
                    zip(target, qpos_adrs, dof_adrs, actuator_ids)
                ):
                    torque = args.hand_kp * (q_des - data.qpos[qpos_adr]) - args.hand_kd * data.qvel[dof_adr]
                    data.ctrl[actuator_id] = _torque_limit(model, actuator_id, torque)

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            tick += 1
            if viewer is not None and tick % 20 == 0:
                cmd, speed_scale = keyboard.command()
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "W/S forward/back  A/D left/right  Q/R yaw  O/P speed  Space stop",
                        f"cmd_vel=[{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}]  speed={speed_scale:.1f}x",
                    )
                )
            if tick % 200 == 0:
                command_status = bridge.command_status()
                keyboard_cmd, speed_scale = keyboard.command()
                w, x, y, z = data.qpos[3:7]
                roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
                print(
                    f"[BOOST MuJoCo] t={data.time:.1f}s "
                    f"lowstate_pub={bridge.low_state_count} "
                    f"rt/lowcmd={_format_command_status(command_status['lowcmd'])} "
                    f"Revo2_cmd(L/R)="
                    f"{_format_command_status(command_status['left_revo2'])}/"
                    f"{_format_command_status(command_status['right_revo2'])} "
                    f"cmd_vel=[{keyboard_cmd[0]:+.2f},{keyboard_cmd[1]:+.2f},"
                    f"{keyboard_cmd[2]:+.2f}] speed={speed_scale:.1f}x "
                    f"base_z={data.qpos[2]:.3f} "
                    f"tilt_deg=({np.degrees(roll):+.1f},{np.degrees(pitch):+.1f})",
                    flush=True,
                )

            next_tick += sim_period
            if not args.no_realtime:
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -5.0 * sim_period:
                    next_tick = time.monotonic()
            if now >= next_report:
                next_report = now + 1.0
    except KeyboardInterrupt:
        return 0
    finally:
        if surface_service is not None:
            surface_service.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
