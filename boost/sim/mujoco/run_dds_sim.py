#!/usr/bin/env python3
"""Run the BOOST G1+Revo2 MuJoCo model as a Unitree DDS robot endpoint.

The selected body controller either consumes external ``rt/lowcmd`` commands
or evaluates BOOST's vendored DreamWaQ policy directly against MuJoCo state.
Both modes publish simulated ``rt/lowstate`` and Revo2 hand state.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _sensor_address(model: mujoco.MjModel, name: str) -> int:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"MuJoCo model is missing sensor '{name}'")
    return int(model.sensor_adr[sensor_id])


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
    """Latched viewer inputs for body-frame velocity and DreamWaQ mode."""

    SPEED_STEP = 0.1
    INITIAL_SPEED_SCALE = 0.5
    MIN_SPEED_SCALE = 0.2

    def __init__(self, max_cmd: np.ndarray):
        self.max_cmd = np.asarray(max_cmd, dtype=np.float32).reshape(3)
        self._lock = threading.Lock()
        self._direction = np.zeros(3, dtype=np.float32)
        self._speed_scale = self.INITIAL_SPEED_SCALE
        self._dreamwaq_mode = 0

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
            elif key == "x":
                self._dreamwaq_mode = 1 - self._dreamwaq_mode
            elif key == " ":
                self._direction[:] = 0.0

    def command(self) -> tuple[np.ndarray, float]:
        with self._lock:
            cmd = self._direction * self.max_cmd * self._speed_scale
            return cmd, self._speed_scale

    def dreamwaq_mode(self) -> int:
        with self._lock:
            return self._dreamwaq_mode


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


def _make_bridge_config(
    sim_config: dict, hand_config: dict, *, accept_low_cmd: bool = True
) -> dict:
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
        "ACCEPT_LOW_CMD": bool(accept_low_cmd),
    }


def _load_dreamwaq_components():
    """Load the vendored upstream policy and arm-target source."""

    dreamwaq_root = ROOT / "external" / "dreamwaq_g1_29dof"
    deploy_root = dreamwaq_root / "deploy"
    if not (deploy_root / "dreamwaq_policy.py").is_file():
        raise FileNotFoundError(
            f"DreamWaQ runtime snapshot not found under {dreamwaq_root}"
        )
    for source_path in (str(dreamwaq_root), str(deploy_root)):
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    from arm_target_source import make_arm_source
    from dreamwaq_policy import DreamWaQPolicy
    from robot_control.config import Config as DreamWaQConfig
    from dreamwaq_interface.g1_29dof.contract import JOINT_LOWER_SDK, JOINT_UPPER_SDK

    return (
        DreamWaQPolicy,
        DreamWaQConfig,
        make_arm_source,
        np.asarray(JOINT_LOWER_SDK, dtype=np.float64),
        np.asarray(JOINT_UPPER_SDK, dtype=np.float64),
    )


def _dreamwaq_command(
    target_q: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_ff: np.ndarray,
    measured_q: np.ndarray,
    measured_dq: np.ndarray,
    torque_limit: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the target G1 wrapper's joint and total-PD-effort limits."""

    q_cmd = np.clip(target_q, joint_lower, joint_upper).astype(np.float64, copy=True)
    kp_cmd = np.maximum(np.asarray(kp, dtype=np.float64), 0.0)
    kd_cmd = np.maximum(np.asarray(kd, dtype=np.float64), 0.0)
    tau_cmd = np.clip(tau_ff, -torque_limit, torque_limit).astype(np.float64, copy=True)
    active = kp_cmd > 1.0e-6
    positive_error = (
        torque_limit - tau_cmd + kd_cmd * measured_dq
    ) / np.maximum(kp_cmd, 1.0e-6)
    negative_error = (
        -torque_limit - tau_cmd + kd_cmd * measured_dq
    ) / np.maximum(kp_cmd, 1.0e-6)
    q_cmd[active] = np.clip(
        q_cmd[active],
        measured_q[active] + negative_error[active],
        measured_q[active] + positive_error[active],
    )
    q_cmd = np.clip(q_cmd, joint_lower, joint_upper)
    return q_cmd, tau_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sim/mujoco_revo2.yaml")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--body-config", type=Path, default=None,
                        help="29-DoF gains/default pose for the selected sim controller.")
    parser.add_argument("--controller", choices=("external-lowcmd", "dreamwaq"), default=None,
                        help="Consume DDS lowcmd or run DreamWaQ directly in the sim loop.")
    parser.add_argument("--dreamwaq-export", type=Path, default=None,
                        help="DreamWaQ TorchScript export directory; defaults to the sim config.")
    parser.add_argument("--dreamwaq-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--interface", default=None, help="DDS interface for the isolated simulator (normally lo).")
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
    parser.add_argument(
        "--hold-start-delay-s",
        type=float,
        default=0.0,
        help="hold initial physics for a delay, or until external rt/lowcmd is fresh",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.hold_start_delay_s) or args.hold_start_delay_s < 0.0:
        raise ValueError("--hold-start-delay-s must be finite and non-negative")
    with args.config.open("r", encoding="utf-8") as stream:
        sim_config = yaml.safe_load(stream)
    hand_config_path = ROOT / sim_config["hand_config"]
    with hand_config_path.open("r", encoding="utf-8") as stream:
        hand_config = yaml.safe_load(stream)
    args.scene = args.scene or ROOT / sim_config["scene"]
    args.controller = args.controller or str(sim_config.get("controller", "external-lowcmd"))
    body_config_value = sim_config.get("body_config")
    if args.body_config is None:
        if body_config_value is None:
            raise ValueError("sim config must define body_config")
        args.body_config = ROOT / body_config_value
    elif not args.body_config.is_absolute():
        args.body_config = ROOT / args.body_config
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
    with args.body_config.open("r", encoding="utf-8") as stream:
        body_config = yaml.safe_load(stream)
    default_q = np.asarray(body_config["default_angles"], dtype=np.float64).reshape(29)
    startup_kp = np.asarray(body_config["kps"], dtype=np.float64).reshape(29)
    startup_kd = np.asarray(body_config["kds"], dtype=np.float64).reshape(29)
    if not all(np.all(np.isfinite(value)) for value in (default_q, startup_kp, startup_kd)):
        raise ValueError("body config vectors must contain finite values")
    if np.any(startup_kp < 0.0) or np.any(startup_kd < 0.0):
        raise ValueError("body config gains must be non-negative")

    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = UnitreeSdk2Bridge(
        _make_bridge_config(
            sim_config,
            hand_config,
            accept_low_cmd=args.controller == "external-lowcmd",
        )
    )
    keyboard = KeyboardLocomotion(np.asarray(body_config["max_cmd"], dtype=np.float32))
    dreamwaq_policy = None
    dreamwaq_arm_source = None
    dreamwaq_target_q15 = default_q[:15].astype(np.float32).copy()
    dreamwaq_control_dt = 0.0
    dreamwaq_next_time = 0.0
    dreamwaq_policy_ms = 0.0
    dreamwaq_mode = 0
    dreamwaq_imu_quat_adr = None
    dreamwaq_imu_gyro_adr = None
    dreamwaq_torque_limit = None
    dreamwaq_joint_lower = None
    dreamwaq_joint_upper = None
    if args.controller == "dreamwaq":
        (
            DreamWaQPolicy,
            DreamWaQConfig,
            make_arm_source,
            dreamwaq_joint_lower,
            dreamwaq_joint_upper,
        ) = _load_dreamwaq_components()
        export_value = args.dreamwaq_export or sim_config.get("dreamwaq_export")
        if export_value is None:
            raise ValueError("DreamWaQ mode requires dreamwaq_export in the sim config")
        export_path = Path(export_value)
        if not export_path.is_absolute():
            export_path = ROOT / export_path
        payload_path = export_path / "backpack_payload.json"
        if not payload_path.is_file():
            raise FileNotFoundError(
                f"selected DreamWaQ export has no backpack payload receipt: {payload_path}"
            )
        with payload_path.open("r", encoding="utf-8") as stream:
            payload_receipt = json.load(stream)
        if payload_receipt.get("backpack_enabled") is not True:
            raise ValueError("selected DreamWaQ export is not marked as backpack-trained")
        dreamwaq_config = DreamWaQConfig(args.body_config)
        dreamwaq_control_dt = float(dreamwaq_config.control_dt)
        if not np.isfinite(dreamwaq_control_dt) or dreamwaq_control_dt <= 0.0:
            raise ValueError("DreamWaQ control_dt must be finite and positive")
        dreamwaq_policy = DreamWaQPolicy(export_path, device=args.dreamwaq_device)
        if dreamwaq_policy.contract.action_dim != 12:
            dreamwaq_policy.close()
            raise ValueError(
                "BOOST sim expects the backpack NEW3-4 12-action export; "
                f"got action_dim={dreamwaq_policy.contract.action_dim}"
            )
        dreamwaq_torque_limit = np.asarray(
            dreamwaq_policy.contract.torque_limit_sdk, dtype=np.float64
        ).reshape(-1)
        if (
            dreamwaq_torque_limit.shape != (29,)
            or not np.all(np.isfinite(dreamwaq_torque_limit))
            or np.any(dreamwaq_torque_limit <= 0.0)
            or dreamwaq_joint_lower.shape != (29,)
            or dreamwaq_joint_upper.shape != (29,)
        ):
            dreamwaq_policy.close()
            raise ValueError("DreamWaQ G1 joint/torque limit contract is malformed")
        dreamwaq_arm_source = make_arm_source(dreamwaq_config, net=args.interface)
        dreamwaq_policy.reset(default_q.astype(np.float32))
        print(
            f"[BOOST DreamWaQ] export={export_path} action_dim=12 "
            f"policy_hz={1.0 / dreamwaq_control_dt:.1f} "
            f"arm_source={dreamwaq_config.arm_source} "
            f"arm_smoothing_tau={dreamwaq_config.arm_smoothing_tau:g}s",
            flush=True,
        )
    elif args.controller != "external-lowcmd":
        raise ValueError(f"unsupported controller: {args.controller}")
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
    if dreamwaq_policy is not None:
        dreamwaq_imu_quat_adr = _sensor_address(model, "imu_quat")
        dreamwaq_imu_gyro_adr = _sensor_address(model, "imu_gyro")
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
    if dreamwaq_policy is not None:
        if not np.all(model.actuator_ctrllimited[body_actuators]):
            dreamwaq_policy.close()
            raise RuntimeError("DreamWaQ MuJoCo actuators must have explicit torque limits")
        model_torque_limit = model.actuator_ctrlrange[body_actuators]
        expected_torque_limit = np.column_stack(
            (-dreamwaq_torque_limit, dreamwaq_torque_limit)
        )
        if not np.allclose(model_torque_limit, expected_torque_limit, atol=1.0e-5, rtol=0.0):
            dreamwaq_policy.close()
            raise RuntimeError(
                "MuJoCo body actuator limits differ from the selected DreamWaQ export; "
                "update the model instead of silently changing the deployment contract"
            )
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
    physics_start_wall = next_tick
    next_report = next_tick
    next_nav_publish = next_tick
    nav_cmd_sequence = 0
    tick = 0
    physics_started = args.hold_start_delay_s <= 0.0
    print(
        f"[BOOST MuJoCo] scene={args.scene} interface={args.interface} "
        f"DDS domain={args.domain_id} dt={sim_period:g}s nq={model.nq} nu={model.nu}; "
        "publishing rt/lowstate, rt/nav_cmd, and Revo2 state; "
        f"body controller={args.controller}; "
        f"lowcmd DDS {'disabled' if args.controller == 'dreamwaq' else 'subscribed'}; "
        "waiting for Revo2 DDS commands",
        flush=True,
    )
    if args.hold_start_delay_s > 0.0:
        print(
            "[BOOST MuJoCo] holding initial physics until "
            f"{args.hold_start_delay_s:.1f}s"
            + (" and a fresh rt/lowcmd" if args.controller == "external-lowcmd" else ""),
            flush=True,
        )
    if viewer is not None:
        print(
            "[MuJoCo keyboard] W/S forward/back, A/D left/right, "
            "Q/R yaw left/right; O/P change linear and angular speed "
            "(20-100%); Space stops; X toggles DreamWaQ standing/walking. "
            "Direction keys latch until reversed or stopped.",
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

            have_body_command = args.controller == "dreamwaq"
            if dreamwaq_policy is not None:
                if data.time + 1.0e-9 >= dreamwaq_next_time:
                    command, _speed_scale = keyboard.command()
                    dreamwaq_mode = keyboard.dreamwaq_mode()
                    dreamwaq_policy.set_command(command)
                    dreamwaq_policy.set_locomotion_mode(dreamwaq_mode)
                    q29 = data.qpos[body_qpos].astype(np.float32, copy=True)
                    dq29 = data.qvel[body_dof].astype(np.float32, copy=True)
                    quat = data.sensordata[
                        dreamwaq_imu_quat_adr:dreamwaq_imu_quat_adr + 4
                    ].astype(np.float32, copy=True)
                    gyro = data.sensordata[
                        dreamwaq_imu_gyro_adr:dreamwaq_imu_gyro_adr + 3
                    ].astype(np.float32, copy=True)
                    dreamwaq_target_q15, _mode_name, inference_ns = dreamwaq_policy.update(
                        q29, dq29, quat, gyro
                    )
                    dreamwaq_policy_ms = float(inference_ns) / 1.0e6
                    dreamwaq_next_time += dreamwaq_control_dt
                    if dreamwaq_next_time <= data.time:
                        dreamwaq_next_time = data.time + dreamwaq_control_dt

                # Read and smooth arm targets at the physics rate while the
                # lower-body policy target is held at its 50-Hz output rate.
                arm_q, arm_kp, arm_kd, arm_tau = dreamwaq_arm_source.get_arm_command()
                body_target = np.concatenate(
                    (dreamwaq_target_q15, np.asarray(arm_q, dtype=np.float32))
                )
                body_kp = startup_kp.copy()
                body_kd = startup_kd.copy()
                body_tau = np.zeros(29, dtype=np.float64)
                body_kp[15:29] = np.asarray(arm_kp, dtype=np.float64)
                body_kd[15:29] = np.asarray(arm_kd, dtype=np.float64)
                body_tau[15:29] = np.asarray(arm_tau, dtype=np.float64)
                measured_q29 = data.qpos[body_qpos].astype(np.float64, copy=True)
                measured_dq29 = data.qvel[body_dof].astype(np.float64, copy=True)
                bounded_target_q, bounded_tau_ff = _dreamwaq_command(
                    body_target.astype(np.float64),
                    body_kp,
                    body_kd,
                    body_tau,
                    measured_q29,
                    measured_dq29,
                    dreamwaq_torque_limit,
                    dreamwaq_joint_lower,
                    dreamwaq_joint_upper,
                )
                for i, (qpos_adr, dof_adr, actuator_id) in enumerate(
                    zip(body_qpos, body_dof, body_actuators)
                ):
                    torque = (
                        bounded_tau_ff[i]
                        + body_kp[i] * (float(bounded_target_q[i]) - data.qpos[qpos_adr])
                        - body_kd[i] * data.qvel[dof_adr]
                    )
                    data.ctrl[actuator_id] = _torque_limit(model, actuator_id, torque)
            else:
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

            delay_elapsed = now - physics_start_wall >= args.hold_start_delay_s
            physics_ready = args.hold_start_delay_s <= 0.0 or (
                delay_elapsed and have_body_command
            )
            if physics_ready:
                if not physics_started:
                    print(
                        "[BOOST MuJoCo] startup condition met; starting physics",
                        flush=True,
                    )
                    physics_started = True
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
                        "W/S forward/back  A/D left/right  Q/R yaw  O/P speed  Space stop  X stand/walk",
                        f"cmd_vel=[{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}]  "
                        f"speed={speed_scale:.1f}x  "
                        f"mode={'walking' if keyboard.dreamwaq_mode() else 'standing'}",
                    )
                )
            if tick % 200 == 0:
                command_status = bridge.command_status()
                keyboard_cmd, speed_scale = keyboard.command()
                w, x, y, z = data.qpos[3:7]
                roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
                controller_status = f"controller={args.controller} "
                lowcmd_status = (
                    "not subscribed"
                    if args.controller == "dreamwaq"
                    else _format_command_status(command_status["lowcmd"])
                )
                if dreamwaq_policy is not None:
                    controller_status += (
                        f"dreamwaq_mode={'walking' if dreamwaq_mode else 'standing'} "
                        f"policy_ms={dreamwaq_policy_ms:.2f} "
                    )
                print(
                    f"[BOOST MuJoCo] t={data.time:.1f}s "
                    f"lowstate_pub={bridge.low_state_count} "
                    f"rt/lowcmd={lowcmd_status} "
                    f"{controller_status}"
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
        if dreamwaq_arm_source is not None:
            close = getattr(dreamwaq_arm_source, "close", None)
            if callable(close):
                close()
        if dreamwaq_policy is not None:
            dreamwaq_policy.close()
        if surface_service is not None:
            surface_service.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
