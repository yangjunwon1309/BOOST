import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.mujoco_sim.hand_backends import (
    DEX3_BACKEND,
    REVO2_BACKEND,
    REVO2_COMMAND_SUFFIXES,
    REVO2_JOINT_SUFFIXES,
    Revo2CommandAdapter,
)

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        # [추가] GR00T 모델용 특정 Scene 카메라 설정 통합
        if config.get("ROBOT_SCENE", "") == "decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml":
            camera_configs["global_view"] = {
                "height": 400,
                "width": 400,
            }

        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.num_hand_motor = self.robot.NUM_HAND_MOTORS
        self.hand_backend = self.robot.HAND_BACKEND
        if self.hand_backend not in (DEX3_BACKEND, REVO2_BACKEND):
            raise ValueError(f"Unsupported HAND_BACKEND: {self.hand_backend}")
        self.revo2_adapter = Revo2CommandAdapter(
            binary_open=tuple(self.config.get("BRAINCO_BINARY_OPEN", (0.0,) * 6)),
            binary_close=tuple(
                self.config.get("BRAINCO_BINARY_CLOSE", (0.70, 1.0, 1.0, 1.0, 1.0, 1.0))
            ),
            legacy_close_threshold=float(
                self.config.get("BRAINCO_LEGACY_CLOSE_THRESHOLD", 0.10)
            ),
        )
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def _get_ordered_body_joint_indices(self):
        motor2joint = self.config.get("MOTOR2JOINT", {})
        if isinstance(motor2joint, dict) and motor2joint:
            ordered_names = [
                joint_name
                for joint_name, _ in sorted(motor2joint.items(), key=lambda item: int(item[1]))
            ]
            joint_indices, missing = [], []
            for joint_name in ordered_names:
                joint_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id < 0:
                    missing.append(joint_name)
                else:
                    joint_indices.append(joint_id)
            if missing or len(joint_indices) != self.num_body_dof:
                raise ValueError(
                    "Configured body joints do not match the MuJoCo model. "
                    f"missing={missing}, found={len(joint_indices)}, expected={self.num_body_dof}"
                )
            return np.array(joint_indices, dtype=int)

        body_joint_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(p in name for p in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]):
                body_joint_index.append(i)
        return np.array(body_joint_index, dtype=int)

    def _get_ordered_hand_joint_indices(self, side: str):
        if self.hand_backend == REVO2_BACKEND:
            hand_order = REVO2_JOINT_SUFFIXES
            prefix = f"{side}_"
        else:
            hand_order = (
                "thumb_0_joint",
                "thumb_1_joint",
                "thumb_2_joint",
                "middle_0_joint",
                "middle_1_joint",
                "index_0_joint",
                "index_1_joint",
            )
            prefix = f"{side}_hand_"
        available = {
            self.mj_model.joint(i).name: i
            for i in range(self.mj_model.njnt)
            if self.mj_model.joint(i).name.startswith(prefix)
        }
        if not available:
            if self.num_hand_dof > 0:
                raise ValueError(f"Configured {side} hand joints were not found in the MuJoCo model.")
            return np.array([], dtype=int)

        joint_indices, missing = [], []
        for suffix in hand_order:
            joint_name = f"{prefix}{suffix}"
            joint_id = available.get(joint_name)
            if joint_id is None:
                missing.append(joint_name)
            else:
                joint_indices.append(joint_id)
        if missing or len(joint_indices) != self.num_hand_dof:
            raise ValueError(
                f"Configured {side} hand joints do not match the MuJoCo model. "
                f"missing={missing}, found={len(joint_indices)}, expected={self.num_hand_dof}"
            )
        return np.array(joint_indices, dtype=int)

    def _get_ordered_hand_command_joint_indices(self, side: str, hand_joint_indices):
        if self.hand_backend == DEX3_BACKEND:
            return hand_joint_indices

        available = {
            self.mj_model.joint(i).name: i
            for i in hand_joint_indices
        }
        names = [f"{side}_{suffix}" for suffix in REVO2_COMMAND_SUFFIXES]
        missing = [name for name in names if name not in available]
        if missing or len(names) != self.num_hand_motor:
            raise ValueError(
                f"Configured {side} REVO2 command joints do not match the MuJoCo model. "
                f"missing={missing}, commands={len(names)}, expected={self.num_hand_motor}"
            )
        return np.asarray([available[name] for name in names], dtype=int)

    def _get_actuator_indices_for_joints(self, joint_indices):
        joint_to_actuator = {}
        for actuator_id in range(self.mj_model.nu):
            joint_id = int(self.mj_model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                joint_to_actuator[joint_id] = actuator_id

        actuator_indices, missing = [], []
        for joint_id in joint_indices:
            actuator_id = joint_to_actuator.get(int(joint_id))
            if actuator_id is None:
                missing.append(self.mj_model.joint(int(joint_id)).name)
            else:
                actuator_indices.append(actuator_id)
        if missing:
            raise ValueError(f"Missing actuators for joints: {missing}")
        return np.array(actuator_indices, dtype=int)

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()
        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        elif self.use_constrained_root_link:
            self.qpos_offset = 1
            self.qvel_offset = 1
        else:
            raise ValueError("No root link found -- The absolute static root will make the simulation unstable.")

        # Elastic Band 및 Viewer 설정
        self.viewer = None
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            # 로봇 타입에 따른 부착 링크 설정
            if "g1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("pelvis" if self.config["enable_waist"] else "torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False, show_right_ui=False
                )
        elif self.onscreen:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False)
        
        if self.viewer is None:
            mujoco.mj_forward(self.mj_model, self.mj_data)
        else:
            self.viewer.cam.azimuth, self.viewer.cam.elevation = 120, -30
            self.viewer.cam.distance, self.viewer.cam.lookat = 2.0, np.array([0, 0, 0.5])
            self.viewer.cam.type, self.viewer.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, self.mj_model.body("pelvis").id

        # Keep control/observation order tied to the WBC config, not XML traversal.
        self.body_joint_index = self._get_ordered_body_joint_indices()
        self.left_hand_index = self._get_ordered_hand_joint_indices("left")
        self.right_hand_index = self._get_ordered_hand_joint_indices("right")
        self.left_hand_command_joint_index = self._get_ordered_hand_command_joint_indices(
            "left", self.left_hand_index
        )
        self.right_hand_command_joint_index = self._get_ordered_hand_command_joint_indices(
            "right", self.right_hand_index
        )
        self.body_actuator_index = self._get_actuator_indices_for_joints(self.body_joint_index)
        self.left_hand_actuator_index = self._get_actuator_indices_for_joints(
            self.left_hand_command_joint_index
        )
        self.right_hand_actuator_index = self._get_actuator_indices_for_joints(
            self.right_hand_command_joint_index
        )
        self.torques = np.zeros(self.mj_model.nu)
        self.actuator_ctrllimited = self.mj_model.actuator_ctrllimited.astype(bool)
        self.actuator_ctrlrange = self.mj_model.actuator_ctrlrange.copy()

    # [추가] GR00T의 디버깅용 Kinematics Step 통합
    def kinematics_step(self):
        """Run kinematics only: compute the qpos and directly set it for debugging."""
        if self.unitree_bridge is not None:
            self.unitree_bridge.PublishLowState(self.prepare_obs())
            if self.unitree_bridge.joystick:
                self.unitree_bridge.PublishWirelessController()

        body_qpos = self.compute_body_qpos()
        hand_qpos = self.compute_hand_qpos()

        # base_sim의 동적 오프셋 적용
        self.mj_data.qpos[self.body_joint_index + self.qpos_offset - 1] = body_qpos
        if self.num_hand_dof > 0:
            self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1] = hand_qpos[:self.num_hand_dof]
            self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1] = hand_qpos[self.num_hand_dof:]

        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_motor)
        right_hand_torques = np.zeros(self.num_hand_motor)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            if self.hand_backend == REVO2_BACKEND:
                left_target = self.revo2_adapter.normalized_to_actuated(
                    self.unitree_bridge.get_hand_command_q(True)
                )
                right_target = self.revo2_adapter.normalized_to_actuated(
                    self.unitree_bridge.get_hand_command_q(False)
                )
                kp = float(self.robot.HAND_MOTOR_KP)
                kd = float(self.robot.HAND_MOTOR_KD)
                left_q = self.mj_data.qpos[
                    self.left_hand_command_joint_index + self.qpos_offset - 1
                ]
                left_dq = self.mj_data.qvel[
                    self.left_hand_command_joint_index + self.qvel_offset - 1
                ]
                right_q = self.mj_data.qpos[
                    self.right_hand_command_joint_index + self.qpos_offset - 1
                ]
                right_dq = self.mj_data.qvel[
                    self.right_hand_command_joint_index + self.qvel_offset - 1
                ]
                left_hand_torques[:] = kp * (left_target - left_q) - kd * left_dq
                right_hand_torques[:] = kp * (right_target - right_q) - kd * right_dq
            else:
                for i in range(self.unitree_bridge.num_hand_motor):
                    left_command = self.unitree_bridge.left_hand_cmd.motor_cmd[i]
                    right_command = self.unitree_bridge.right_hand_cmd.motor_cmd[i]
                    left_hand_torques[i] = (
                        left_command.tau
                        + left_command.kp
                        * (
                            left_command.q
                            - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                        )
                        + left_command.kd
                        * (
                            left_command.dq
                            - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                        )
                    )
                    right_hand_torques[i] = (
                        right_command.tau
                        + right_command.kp
                        * (
                            right_command.q
                            - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                        )
                        + right_command.kd
                        * (
                            right_command.dq
                            - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                        )
                    )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            if self.hand_backend == REVO2_BACKEND:
                hand_qpos[: self.num_hand_dof] = self.revo2_adapter.normalized_to_all_joints(
                    self.unitree_bridge.get_hand_command_q(True)
                )
                hand_qpos[self.num_hand_dof :] = self.revo2_adapter.normalized_to_all_joints(
                    self.unitree_bridge.get_hand_command_q(False)
                )
            else:
                for i in range(self.unitree_bridge.num_hand_motor):
                    hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                    hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_actuator_index]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_actuator_index]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_actuator_index]
        obs["time"] = self.mj_data.time
        return obs

    def _apply_env_step_assists(self):
        pass

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        self.torques.fill(0.0)
        self.torques[self.body_actuator_index] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_actuator_index] = hand_torques[: self.num_hand_motor]
            self.torques[self.right_hand_actuator_index] = hand_torques[self.num_hand_motor :]

        limited = self.actuator_ctrllimited
        self.torques[limited] = np.clip(
            self.torques[limited],
            self.actuator_ctrlrange[limited, 0],
            self.actuator_ctrlrange[limited, 1],
        )

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        self._apply_env_step_assists()
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)


class G1TorresOpenEnv(DefaultEnv):
    """MuJoCo scene for PICO teleop around an already-open Torres driver door.

    This keeps the normal Unitree SDK2 DDS sim path intact. The Torres mesh is
    visual-only and lightweight proxy boxes provide stable collision/seat targets.
    """

    DEFAULT_START_JOINT_POS = {
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.18,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.18,
        "right_shoulder_pitch_joint": 0.35,
    }

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        config = config.copy()
        config["ROBOT_SCENE"] = config.get(
            "G1_TORRES_OPEN_SCENE",
            "gear_sonic_deploy/gear_sonic_deploy/g1/scene_29dof_torres_open.xml",
        )
        super().__init__(config, "g1_torres_open", {}, onscreen, offscreen, enable_image_publish)
        self.robot_start_pos = np.array(
            config.get("G1_TORRES_OPEN_ROBOT_START_POS", [-1.20, -0.07, 0.793]),
            dtype=float,
        )
        self.robot_start_quat = np.array(
            config.get("G1_TORRES_OPEN_ROBOT_START_QUAT", [1.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.seat_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "driver_seat_target"
        )
        self.doorway_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "driver_doorway_target"
        )
        self.reset()

    def _apply_pattern_joint_defaults(self) -> None:
        import re

        for pattern, value in self.DEFAULT_START_JOINT_POS.items():
            regex = re.compile(pattern)
            for joint_id in range(self.mj_model.njnt):
                joint_name = self.mj_model.joint(joint_id).name
                if regex.fullmatch(joint_name):
                    qadr = self.mj_model.jnt_qposadr[joint_id]
                    self.mj_data.qpos[qadr] = value

    def _set_start_pose(self) -> None:
        if self.use_floating_root_link:
            self.mj_data.qpos[:3] = self.robot_start_pos
            self.mj_data.qpos[3:7] = self.robot_start_quat / np.linalg.norm(self.robot_start_quat)
            self.mj_data.qvel[:6] = 0.0
        self._apply_pattern_joint_defaults()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._set_start_pose()

    def update_reward(self):
        if self.seat_site_id < 0:
            with self.reward_lock:
                self.last_reward = 0.0
            return
        pelvis_pos = self.mj_data.body("pelvis").xpos
        seat_pos = self.mj_data.site_xpos[self.seat_site_id]
        dist = np.linalg.norm(pelvis_pos[:2] - seat_pos[:2])
        height_ok = 0.35 < pelvis_pos[2] < 0.90
        with self.reward_lock:
            self.last_reward = float(max(0.0, 1.0 - dist) * (1.0 if height_ok else 0.25))


class G1TorresHingeEnv(DefaultEnv):
    """MuJoCo scene with G1 and a native articulated Torres driver door.

    The car is proxy geometry for stability. The door is a real MuJoCo hinge
    joint initialized open at reset. This avoids DAE decoding and does not use
    the real-G1 LCM bridge.
    """

    DEFAULT_START_JOINT_POS = {
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.18,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.18,
        "right_shoulder_pitch_joint": 0.35,
    }

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        config = config.copy()
        config["ROBOT_SCENE"] = config.get(
            "G1_TORRES_HINGE_SCENE",
            "gear_sonic_deploy/gear_sonic_deploy/g1/scene_29dof_torres_hinge.xml",
        )
        super().__init__(config, "g1_torres_hinge", {}, onscreen, offscreen, enable_image_publish)
        self.robot_start_pos = np.array(
            config.get("G1_TORRES_HINGE_ROBOT_START_POS", [-1.20, -0.07, 0.793]),
            dtype=float,
        )
        self.robot_start_quat = np.array(
            config.get("G1_TORRES_HINGE_ROBOT_START_QUAT", [1.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.door_open_angle = float(config.get("G1_TORRES_DOOR_OPEN_ANGLE", 1.15))
        self.door_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "torres_driver_door_hinge"
        )
        self.handle_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "torres_driver_handle_hinge"
        )
        self.door_open_angle = self._clamp_joint_target(self.door_joint_id, self.door_open_angle)
        self.door_dof_id = (
            self.mj_model.jnt_dofadr[self.door_joint_id] if self.door_joint_id >= 0 else -1
        )
        self.handle_dof_id = (
            self.mj_model.jnt_dofadr[self.handle_joint_id] if self.handle_joint_id >= 0 else -1
        )
        self.entry_disable_body_collisions = bool(
            config.get("G1_TORRES_ENTRY_DISABLE_BODY_COLLISIONS", False)
        )
        self.seat_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "driver_seat_target"
        )
        self.door_hold_open = bool(config.get("G1_TORRES_DOOR_HOLD_OPEN", False))
        self.door_hold_kp = float(config.get("G1_TORRES_DOOR_HOLD_KP", 90.0))
        self.door_hold_kd = float(config.get("G1_TORRES_DOOR_HOLD_KD", 10.0))
        self.door_hold_max_torque = float(config.get("G1_TORRES_DOOR_HOLD_MAX_TORQUE", 120.0))
        self.door_assist_enabled = bool(config.get("G1_TORRES_DOOR_ASSIST_ENABLED", True))
        self.door_assist_grasp_threshold = float(
            config.get("G1_TORRES_DOOR_ASSIST_GRASP_THRESHOLD", 0.12)
        )
        self.door_assist_open_angle = self._clamp_joint_target(
            self.door_joint_id, float(config.get("G1_TORRES_DOOR_ASSIST_OPEN_ANGLE", 1.15))
        )
        self.door_assist_kp = float(config.get("G1_TORRES_DOOR_ASSIST_KP", 75.0))
        self.door_assist_kd = float(config.get("G1_TORRES_DOOR_ASSIST_KD", 9.0))
        self.door_assist_max_torque = float(config.get("G1_TORRES_DOOR_ASSIST_MAX_TORQUE", 90.0))
        self.handle_assist_open_angle = self._clamp_joint_target(
            self.handle_joint_id, float(config.get("G1_TORRES_HANDLE_ASSIST_OPEN_ANGLE", 0.14))
        )
        self.handle_assist_unlock_angle = float(config.get("G1_TORRES_HANDLE_UNLOCK_ANGLE", 0.05))
        self.handle_assist_kp = float(config.get("G1_TORRES_HANDLE_ASSIST_KP", 25.0))
        self.handle_assist_kd = float(config.get("G1_TORRES_HANDLE_ASSIST_KD", 2.5))
        self.handle_assist_max_torque = float(config.get("G1_TORRES_HANDLE_ASSIST_MAX_TORQUE", 4.0))
        self.handle_geom_ids = self._collect_geom_ids(
            ("torres_handle_collision_", "torres_handle_lock_collision_")
        )
        self.handle_grasp_region_geom_ids = self._collect_geom_ids(("torres_handle_collision_",))
        self.handle_hand_body_ids = self._collect_body_ids(("left_hand_", "right_hand_"))
        self.handle_hand_geom_ids = self._collect_geom_ids_for_bodies(self.handle_hand_body_ids)
        self.handle_grasp_region_enabled = bool(
            config.get("G1_TORRES_HANDLE_GRASP_REGION_ENABLED", True)
        )
        self.handle_grasp_region_radius = float(
            config.get("G1_TORRES_HANDLE_GRASP_REGION_RADIUS", 0.01)
        )
        self.handle_grasp_region_local_axis = np.array(
            config.get("G1_TORRES_HANDLE_GRASP_REGION_LOCAL_AXIS", [0.0, -1.0, 0.0]),
            dtype=float,
        )
        self.handle_grasp_region_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torres_handle_lock_body"
        )
        self.handle_grasp_region_marker_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "torres_handle_grasp_region_marker"
        )
        self.entry_disabled_geom_names = self._disable_entry_body_collision_geoms()
        self.reset()

    def _clamp_joint_target(self, joint_id: int, target: float) -> float:
        if joint_id < 0 or not self.mj_model.jnt_limited[joint_id]:
            return target
        lo, hi = self.mj_model.jnt_range[joint_id]
        return float(np.clip(target, lo, hi))

    def _collect_geom_ids(self, prefixes) -> set[int]:
        geom_ids = set()
        for geom_id in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name.startswith(prefixes):
                geom_ids.add(geom_id)
        return geom_ids

    def _collect_body_ids(self, prefixes) -> set[int]:
        body_ids = set()
        for body_id in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name.startswith(prefixes):
                body_ids.add(body_id)
        return body_ids

    def _collect_geom_ids_for_bodies(self, body_ids: set[int]) -> set[int]:
        return {
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if int(self.mj_model.geom_bodyid[geom_id]) in body_ids
        }

    def _disable_entry_body_collision_geoms(self) -> list[str]:
        if not self.entry_disable_body_collisions:
            return []

        disabled = []
        for geom_id in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not geom_name.startswith("torres_body_collision_"):
                continue

            self.mj_model.geom_contype[geom_id] = 0
            self.mj_model.geom_conaffinity[geom_id] = 0
            disabled.append(geom_name)
        return disabled

    def _apply_pattern_joint_defaults(self) -> None:
        import re

        for pattern, value in self.DEFAULT_START_JOINT_POS.items():
            regex = re.compile(pattern)
            for joint_id in range(self.mj_model.njnt):
                joint_name = self.mj_model.joint(joint_id).name
                if regex.fullmatch(joint_name):
                    qadr = self.mj_model.jnt_qposadr[joint_id]
                    self.mj_data.qpos[qadr] = value

    def _set_door_open(self) -> None:
        if self.door_joint_id < 0:
            return
        qadr = self.mj_model.jnt_qposadr[self.door_joint_id]
        dadr = self.mj_model.jnt_dofadr[self.door_joint_id]
        self.mj_data.qpos[qadr] = self.door_open_angle
        self.mj_data.qvel[dadr] = 0.0

    def _set_start_pose(self) -> None:
        if self.use_floating_root_link:
            self.mj_data.qpos[:3] = self.robot_start_pos
            self.mj_data.qpos[3:7] = self.robot_start_quat / np.linalg.norm(self.robot_start_quat)
            self.mj_data.qvel[:6] = 0.0
        self._apply_pattern_joint_defaults()
        self._set_door_open()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._set_start_pose()

    def _right_hand_touching_handle(self) -> bool:
        return self._hand_contacting_handle() or self._hand_in_handle_grasp_region()

    def _hand_contacting_handle(self) -> bool:
        if not self.handle_geom_ids or not self.handle_hand_body_ids:
            return False

        for contact_id in range(self.mj_data.ncon):
            contact = self.mj_data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.mj_model.geom_bodyid[geom1])
            body2 = int(self.mj_model.geom_bodyid[geom2])
            if geom1 in self.handle_geom_ids and body2 in self.handle_hand_body_ids:
                return True
            if geom2 in self.handle_geom_ids and body1 in self.handle_hand_body_ids:
                return True
        return False

    def _hand_in_handle_grasp_region(self) -> bool:
        if (
            not self.handle_grasp_region_enabled
            or self.handle_grasp_region_radius <= 0.0
        ):
            return False

        axis_norm = np.linalg.norm(self.handle_grasp_region_local_axis)
        if axis_norm <= 1e-8:
            return False

        center = self._handle_grasp_region_center()
        axis_body_id = self.handle_grasp_region_body_id
        if center is None or axis_body_id < 0:
            return False

        body_xmat = self.mj_data.xmat[axis_body_id].reshape(3, 3)
        outside_axis = body_xmat @ (self.handle_grasp_region_local_axis / axis_norm)
        radius = self.handle_grasp_region_radius

        if self.handle_grasp_region_marker_geom_id >= 0:
            fromto = np.zeros(6, dtype=float)
            for geom_id in self.handle_hand_geom_ids:
                try:
                    dist = mujoco.mj_geomDistance(
                        self.mj_model,
                        self.mj_data,
                        self.handle_grasp_region_marker_geom_id,
                        geom_id,
                        radius * 2.0,
                        fromto,
                    )
                except Exception:
                    continue
                hand_point = fromto[3:6]
                if dist <= 0.0 and np.dot(hand_point - center, outside_axis) >= -1e-6:
                    return True

        for geom_id in self.handle_hand_geom_ids:
            delta = self.mj_data.geom_xpos[geom_id] - center
            if np.linalg.norm(delta) <= radius and np.dot(delta, outside_axis) >= -1e-6:
                return True

        for body_id in self.handle_hand_body_ids:
            delta = self.mj_data.xpos[body_id] - center
            if np.linalg.norm(delta) <= radius and np.dot(delta, outside_axis) >= -1e-6:
                return True

        return False

    def _handle_grasp_region_center(self) -> np.ndarray | None:
        if self.handle_grasp_region_marker_geom_id >= 0:
            return self.mj_data.geom_xpos[self.handle_grasp_region_marker_geom_id]

        if self.handle_grasp_region_geom_ids:
            mins, maxs = [], []
            for geom_id in self.handle_grasp_region_geom_ids:
                mesh_id = int(self.mj_model.geom_dataid[geom_id])
                if mesh_id < 0:
                    continue
                vert_adr = int(self.mj_model.mesh_vertadr[mesh_id])
                vert_num = int(self.mj_model.mesh_vertnum[mesh_id])
                if vert_num <= 0:
                    continue
                verts = self.mj_model.mesh_vert[vert_adr : vert_adr + vert_num]
                geom_xmat = self.mj_data.geom_xmat[geom_id].reshape(3, 3)
                world_verts = self.mj_data.geom_xpos[geom_id] + verts @ geom_xmat.T
                mins.append(world_verts.min(axis=0))
                maxs.append(world_verts.max(axis=0))
            if mins:
                return (np.min(np.vstack(mins), axis=0) + np.max(np.vstack(maxs), axis=0)) * 0.5

        if self.handle_grasp_region_body_id >= 0:
            return self.mj_data.xpos[self.handle_grasp_region_body_id]
        return None

    def _right_hand_grasp_requested(self) -> bool:
        if self.unitree_bridge is None:
            return False
        for side in ("left", "right"):
            if not getattr(self.unitree_bridge, f"{side}_hand_cmd_received", False):
                continue
            hand_cmd = getattr(self.unitree_bridge, f"{side}_hand_cmd", None)
            if hand_cmd is None:
                continue

            try:
                q = np.array(
                    [
                        hand_cmd.motor_cmd[i].q
                        for i in range(self.unitree_bridge.num_hand_motor)
                    ],
                    dtype=float,
                )
            except Exception:
                continue

            if q.size > 0 and np.all(np.isfinite(q)):
                if np.max(np.abs(q)) > self.door_assist_grasp_threshold:
                    return True
        return False

    def _hinge_pd_torque(
        self,
        joint_id: int,
        dof_id: int,
        target: float,
        kp: float,
        kd: float,
        max_abs_torque: float,
    ) -> float:
        if joint_id < 0 or dof_id < 0:
            return 0.0
        qadr = self.mj_model.jnt_qposadr[joint_id]
        qpos = float(self.mj_data.qpos[qadr])
        qvel = float(self.mj_data.qvel[dof_id])
        torque = kp * (target - qpos) - kd * qvel
        return float(np.clip(torque, -max_abs_torque, max_abs_torque))

    def _apply_env_step_assists(self):
        if self.door_dof_id >= 0:
            self.mj_data.qfrc_applied[self.door_dof_id] = 0.0
        if self.handle_dof_id >= 0:
            self.mj_data.qfrc_applied[self.handle_dof_id] = 0.0

        if not self.door_assist_enabled:
            if self.door_hold_open and self.door_dof_id >= 0:
                self.mj_data.qfrc_applied[self.door_dof_id] = self._hinge_pd_torque(
                    self.door_joint_id,
                    self.door_dof_id,
                    self.door_open_angle,
                    self.door_hold_kp,
                    self.door_hold_kd,
                    self.door_hold_max_torque,
                )
            return

        if self.door_hold_open and self.door_dof_id >= 0:
            self.mj_data.qfrc_applied[self.door_dof_id] = self._hinge_pd_torque(
                self.door_joint_id,
                self.door_dof_id,
                self.door_open_angle,
                self.door_hold_kp,
                self.door_hold_kd,
                self.door_hold_max_torque,
            )

        touching_handle = self._right_hand_touching_handle()
        grasping = self._right_hand_grasp_requested()
        pulling_handle = touching_handle and grasping

        handle_target = self.handle_assist_open_angle if pulling_handle else 0.0
        if self.handle_dof_id >= 0:
            self.mj_data.qfrc_applied[self.handle_dof_id] = self._hinge_pd_torque(
                self.handle_joint_id,
                self.handle_dof_id,
                handle_target,
                self.handle_assist_kp,
                self.handle_assist_kd,
                self.handle_assist_max_torque,
            )

        handle_unlocked = self.handle_joint_id < 0
        if self.handle_joint_id >= 0:
            handle_qadr = self.mj_model.jnt_qposadr[self.handle_joint_id]
            handle_unlocked = self.mj_data.qpos[handle_qadr] >= self.handle_assist_unlock_angle

        if pulling_handle and handle_unlocked and self.door_dof_id >= 0:
            door_torque = self._hinge_pd_torque(
                self.door_joint_id,
                self.door_dof_id,
                self.door_assist_open_angle,
                self.door_assist_kp,
                self.door_assist_kd,
                self.door_assist_max_torque,
            )
            self.mj_data.qfrc_applied[self.door_dof_id] = door_torque

    def update_reward(self):
        if self.seat_site_id < 0:
            with self.reward_lock:
                self.last_reward = 0.0
            return
        pelvis_pos = self.mj_data.body("pelvis").xpos
        seat_pos = self.mj_data.site_xpos[self.seat_site_id]
        dist = np.linalg.norm(pelvis_pos[:2] - seat_pos[:2])
        height_ok = 0.35 < pelvis_pos[2] < 0.90
        with self.reward_lock:
            self.last_reward = float(max(0.0, 1.0 - dist) * (1.0 if height_ok else 0.25))


class G1TorresMeshHingeEnv(DefaultEnv):
    """MuJoCo scene with G1 and IsaacLab-like Torres visual meshes.

    DAE source meshes are converted to OBJ by
    scripts/convert_torres_dae_to_obj_and_scene.py.  OBJ meshes are visual-only;
    stable proxy geoms remain responsible for collision.  The driver door has a
    MuJoCo hinge joint initialized open at reset.
    """

    DEFAULT_START_JOINT_POS = {
        "left_shoulder_roll_joint": 0.4,
    }

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        config = config.copy()
        config["ROBOT_SCENE"] = config.get(
            "G1_TORRES_MESH_HINGE_SCENE",
            "gear_sonic_deploy/gear_sonic_deploy/g1/scene_29dof_torres_mesh_hinge.xml",
        )
        super().__init__(config, "g1_torres_mesh_hinge", {}, onscreen, offscreen, enable_image_publish)
        self.robot_start_pos = np.array(
            config.get("G1_TORRES_MESH_HINGE_ROBOT_START_POS", [-1.20, -0.07, 0.793]),
            dtype=float,
        )
        self.robot_start_quat = np.array(
            config.get("G1_TORRES_MESH_HINGE_ROBOT_START_QUAT", [1.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        if hasattr(self, "elastic_band"):
            self.elastic_band.point = np.array(
                config.get(
                    "G1_TORRES_ELASTIC_BAND_ANCHOR",
                    [self.robot_start_pos[0], self.robot_start_pos[1], 1.0],
                ),
                dtype=float,
            )
        self.door_open_angle = float(config.get("G1_TORRES_DOOR_OPEN_ANGLE", 1.15))
        self.door_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "torres_driver_door_hinge"
        )
        self.handle_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "torres_driver_handle_hinge"
        )
        self.door_open_angle = self._clamp_joint_target(self.door_joint_id, self.door_open_angle)
        self.door_dof_id = (
            self.mj_model.jnt_dofadr[self.door_joint_id] if self.door_joint_id >= 0 else -1
        )
        self.handle_dof_id = (
            self.mj_model.jnt_dofadr[self.handle_joint_id] if self.handle_joint_id >= 0 else -1
        )
        self.entry_disable_body_collisions = bool(
            config.get("G1_TORRES_ENTRY_DISABLE_BODY_COLLISIONS", False)
        )
        self.seat_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "driver_seat_target"
        )
        self.door_hold_open = bool(config.get("G1_TORRES_DOOR_HOLD_OPEN", False))
        self.door_hold_kp = float(config.get("G1_TORRES_DOOR_HOLD_KP", 90.0))
        self.door_hold_kd = float(config.get("G1_TORRES_DOOR_HOLD_KD", 10.0))
        self.door_hold_max_torque = float(config.get("G1_TORRES_DOOR_HOLD_MAX_TORQUE", 120.0))
        self.door_assist_enabled = bool(config.get("G1_TORRES_DOOR_ASSIST_ENABLED", True))
        self.door_assist_grasp_threshold = float(
            config.get("G1_TORRES_DOOR_ASSIST_GRASP_THRESHOLD", 0.12)
        )
        self.door_assist_open_angle = self._clamp_joint_target(
            self.door_joint_id, float(config.get("G1_TORRES_DOOR_ASSIST_OPEN_ANGLE", 1.15))
        )
        self.door_assist_kp = float(config.get("G1_TORRES_DOOR_ASSIST_KP", 75.0))
        self.door_assist_kd = float(config.get("G1_TORRES_DOOR_ASSIST_KD", 9.0))
        self.door_assist_max_torque = float(config.get("G1_TORRES_DOOR_ASSIST_MAX_TORQUE", 90.0))
        self.handle_assist_open_angle = self._clamp_joint_target(
            self.handle_joint_id, float(config.get("G1_TORRES_HANDLE_ASSIST_OPEN_ANGLE", 0.14))
        )
        self.handle_assist_unlock_angle = float(config.get("G1_TORRES_HANDLE_UNLOCK_ANGLE", 0.05))
        self.handle_assist_kp = float(config.get("G1_TORRES_HANDLE_ASSIST_KP", 25.0))
        self.handle_assist_kd = float(config.get("G1_TORRES_HANDLE_ASSIST_KD", 2.5))
        self.handle_assist_max_torque = float(config.get("G1_TORRES_HANDLE_ASSIST_MAX_TORQUE", 4.0))
        self.handle_geom_ids = self._collect_geom_ids(
            ("torres_handle_collision_", "torres_handle_lock_collision_")
        )
        self.handle_grasp_region_geom_ids = self._collect_geom_ids(("torres_handle_collision_",))
        self.handle_hand_body_ids = self._collect_body_ids(("left_hand_", "right_hand_"))
        self.handle_hand_geom_ids = self._collect_geom_ids_for_bodies(self.handle_hand_body_ids)
        self.handle_grasp_region_enabled = bool(
            config.get("G1_TORRES_HANDLE_GRASP_REGION_ENABLED", True)
        )
        self.handle_grasp_region_radius = float(
            config.get("G1_TORRES_HANDLE_GRASP_REGION_RADIUS", 0.03)
        )
        self.handle_grasp_region_local_axis = np.array(
            config.get("G1_TORRES_HANDLE_GRASP_REGION_LOCAL_AXIS", [0.0, -1.0, 0.0]),
            dtype=float,
        )
        self.handle_grasp_region_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torres_handle_lock_body"
        )
        self.handle_grasp_region_marker_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "torres_handle_grasp_region_marker"
        )
        self.entry_disabled_geom_names = self._disable_entry_body_collision_geoms()
        self.reset()

    def _clamp_joint_target(self, joint_id: int, target: float) -> float:
        if joint_id < 0 or not self.mj_model.jnt_limited[joint_id]:
            return target
        lo, hi = self.mj_model.jnt_range[joint_id]
        return float(np.clip(target, lo, hi))

    def _collect_geom_ids(self, prefixes) -> set[int]:
        geom_ids = set()
        for geom_id in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name.startswith(prefixes):
                geom_ids.add(geom_id)
        return geom_ids

    def _collect_body_ids(self, prefixes) -> set[int]:
        body_ids = set()
        for body_id in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name.startswith(prefixes):
                body_ids.add(body_id)
        return body_ids

    def _collect_geom_ids_for_bodies(self, body_ids: set[int]) -> set[int]:
        return {
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if int(self.mj_model.geom_bodyid[geom_id]) in body_ids
        }

    def _disable_entry_body_collision_geoms(self) -> list[str]:
        if not self.entry_disable_body_collisions:
            return []

        disabled = []
        for geom_id in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not geom_name.startswith("torres_body_collision_"):
                continue

            self.mj_model.geom_contype[geom_id] = 0
            self.mj_model.geom_conaffinity[geom_id] = 0
            disabled.append(geom_name)
        return disabled

    def _apply_pattern_joint_defaults(self) -> None:
        import re

        for pattern, value in self.DEFAULT_START_JOINT_POS.items():
            regex = re.compile(pattern)
            for joint_id in range(self.mj_model.njnt):
                joint_name = self.mj_model.joint(joint_id).name
                if regex.fullmatch(joint_name):
                    qadr = self.mj_model.jnt_qposadr[joint_id]
                    self.mj_data.qpos[qadr] = value

    def _set_door_open(self) -> None:
        if self.door_joint_id < 0:
            return
        qadr = self.mj_model.jnt_qposadr[self.door_joint_id]
        dadr = self.mj_model.jnt_dofadr[self.door_joint_id]
        self.mj_data.qpos[qadr] = self.door_open_angle
        self.mj_data.qvel[dadr] = 0.0

    def _set_start_pose(self) -> None:
        if self.use_floating_root_link:
            self.mj_data.qpos[:3] = self.robot_start_pos
            self.mj_data.qpos[3:7] = self.robot_start_quat / np.linalg.norm(self.robot_start_quat)
            self.mj_data.qvel[:6] = 0.0
        self._apply_pattern_joint_defaults()
        self._set_door_open()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._set_start_pose()

    def _right_hand_touching_handle(self) -> bool:
        return self._hand_contacting_handle() or self._hand_in_handle_grasp_region()

    def _hand_contacting_handle(self) -> bool:
        if not self.handle_geom_ids or not self.handle_hand_body_ids:
            return False

        for contact_id in range(self.mj_data.ncon):
            contact = self.mj_data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.mj_model.geom_bodyid[geom1])
            body2 = int(self.mj_model.geom_bodyid[geom2])
            if geom1 in self.handle_geom_ids and body2 in self.handle_hand_body_ids:
                return True
            if geom2 in self.handle_geom_ids and body1 in self.handle_hand_body_ids:
                return True
        return False

    def _hand_in_handle_grasp_region(self) -> bool:
        if (
            not self.handle_grasp_region_enabled
            or self.handle_grasp_region_radius <= 0.0
        ):
            return False

        axis_norm = np.linalg.norm(self.handle_grasp_region_local_axis)
        if axis_norm <= 1e-8:
            return False

        center = self._handle_grasp_region_center()
        axis_body_id = self.handle_grasp_region_body_id
        if center is None or axis_body_id < 0:
            return False

        body_xmat = self.mj_data.xmat[axis_body_id].reshape(3, 3)
        outside_axis = body_xmat @ (self.handle_grasp_region_local_axis / axis_norm)
        radius = self.handle_grasp_region_radius

        if self.handle_grasp_region_marker_geom_id >= 0:
            fromto = np.zeros(6, dtype=float)
            for geom_id in self.handle_hand_geom_ids:
                try:
                    dist = mujoco.mj_geomDistance(
                        self.mj_model,
                        self.mj_data,
                        self.handle_grasp_region_marker_geom_id,
                        geom_id,
                        radius * 2.0,
                        fromto,
                    )
                except Exception:
                    continue
                hand_point = fromto[3:6]
                if dist <= 0.0 and np.dot(hand_point - center, outside_axis) >= -1e-6:
                    return True

        for geom_id in self.handle_hand_geom_ids:
            delta = self.mj_data.geom_xpos[geom_id] - center
            if np.linalg.norm(delta) <= radius and np.dot(delta, outside_axis) >= -1e-6:
                return True

        for body_id in self.handle_hand_body_ids:
            delta = self.mj_data.xpos[body_id] - center
            if np.linalg.norm(delta) <= radius and np.dot(delta, outside_axis) >= -1e-6:
                return True

        return False

    def _handle_grasp_region_center(self) -> np.ndarray | None:
        if self.handle_grasp_region_marker_geom_id >= 0:
            return self.mj_data.geom_xpos[self.handle_grasp_region_marker_geom_id]

        if self.handle_grasp_region_geom_ids:
            mins, maxs = [], []
            for geom_id in self.handle_grasp_region_geom_ids:
                mesh_id = int(self.mj_model.geom_dataid[geom_id])
                if mesh_id < 0:
                    continue
                vert_adr = int(self.mj_model.mesh_vertadr[mesh_id])
                vert_num = int(self.mj_model.mesh_vertnum[mesh_id])
                if vert_num <= 0:
                    continue
                verts = self.mj_model.mesh_vert[vert_adr : vert_adr + vert_num]
                geom_xmat = self.mj_data.geom_xmat[geom_id].reshape(3, 3)
                world_verts = self.mj_data.geom_xpos[geom_id] + verts @ geom_xmat.T
                mins.append(world_verts.min(axis=0))
                maxs.append(world_verts.max(axis=0))
            if mins:
                return (np.min(np.vstack(mins), axis=0) + np.max(np.vstack(maxs), axis=0)) * 0.5

        if self.handle_grasp_region_body_id >= 0:
            return self.mj_data.xpos[self.handle_grasp_region_body_id]
        return None

    def _right_hand_grasp_requested(self) -> bool:
        if self.unitree_bridge is None:
            return False
        for side in ("left", "right"):
            if not getattr(self.unitree_bridge, f"{side}_hand_cmd_received", False):
                continue
            hand_cmd = getattr(self.unitree_bridge, f"{side}_hand_cmd", None)
            if hand_cmd is None:
                continue

            try:
                q = np.array(
                    [
                        hand_cmd.motor_cmd[i].q
                        for i in range(self.unitree_bridge.num_hand_motor)
                    ],
                    dtype=float,
                )
            except Exception:
                continue

            if q.size > 0 and np.all(np.isfinite(q)):
                if np.max(np.abs(q)) > self.door_assist_grasp_threshold:
                    return True
        return False

    def _hinge_pd_torque(
        self,
        joint_id: int,
        dof_id: int,
        target: float,
        kp: float,
        kd: float,
        max_abs_torque: float,
    ) -> float:
        if joint_id < 0 or dof_id < 0:
            return 0.0
        qadr = self.mj_model.jnt_qposadr[joint_id]
        qpos = float(self.mj_data.qpos[qadr])
        qvel = float(self.mj_data.qvel[dof_id])
        torque = kp * (target - qpos) - kd * qvel
        return float(np.clip(torque, -max_abs_torque, max_abs_torque))

    def _apply_env_step_assists(self):
        if self.door_dof_id >= 0:
            self.mj_data.qfrc_applied[self.door_dof_id] = 0.0
        if self.handle_dof_id >= 0:
            self.mj_data.qfrc_applied[self.handle_dof_id] = 0.0

        if not self.door_assist_enabled:
            if self.door_hold_open and self.door_dof_id >= 0:
                self.mj_data.qfrc_applied[self.door_dof_id] = self._hinge_pd_torque(
                    self.door_joint_id,
                    self.door_dof_id,
                    self.door_open_angle,
                    self.door_hold_kp,
                    self.door_hold_kd,
                    self.door_hold_max_torque,
                )
            return

        if self.door_hold_open and self.door_dof_id >= 0:
            self.mj_data.qfrc_applied[self.door_dof_id] = self._hinge_pd_torque(
                self.door_joint_id,
                self.door_dof_id,
                self.door_open_angle,
                self.door_hold_kp,
                self.door_hold_kd,
                self.door_hold_max_torque,
            )

        touching_handle = self._right_hand_touching_handle()
        grasping = self._right_hand_grasp_requested()
        pulling_handle = touching_handle and grasping

        handle_target = self.handle_assist_open_angle if pulling_handle else 0.0
        if self.handle_dof_id >= 0:
            self.mj_data.qfrc_applied[self.handle_dof_id] = self._hinge_pd_torque(
                self.handle_joint_id,
                self.handle_dof_id,
                handle_target,
                self.handle_assist_kp,
                self.handle_assist_kd,
                self.handle_assist_max_torque,
            )

        handle_unlocked = self.handle_joint_id < 0
        if self.handle_joint_id >= 0:
            handle_qadr = self.mj_model.jnt_qposadr[self.handle_joint_id]
            handle_unlocked = self.mj_data.qpos[handle_qadr] >= self.handle_assist_unlock_angle

        if pulling_handle and handle_unlocked and self.door_dof_id >= 0:
            door_torque = self._hinge_pd_torque(
                self.door_joint_id,
                self.door_dof_id,
                self.door_assist_open_angle,
                self.door_assist_kp,
                self.door_assist_kd,
                self.door_assist_max_torque,
            )
            self.mj_data.qfrc_applied[self.door_dof_id] = door_torque

    def update_reward(self):
        if self.seat_site_id < 0:
            with self.reward_lock:
                self.last_reward = 0.0
            return
        pelvis_pos = self.mj_data.body("pelvis").xpos
        seat_pos = self.mj_data.site_xpos[self.seat_site_id]
        dist = np.linalg.norm(pelvis_pos[:2] - seat_pos[:2])
        height_ok = 0.35 < pelvis_pos[2] < 0.90
        with self.reward_lock:
            self.last_reward = float(max(0.0, 1.0 - dist) * (1.0 if height_ok else 0.25))

class CubeEnv(DefaultEnv):
    def __init__(self, config: Dict[str, any], onscreen: bool = False, offscreen: bool = False, enable_image_publish: bool = False):
        config = config.copy()
        # ⭕ 올바른 경로 적용
        config["ROBOT_SCENE"] = "decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml"
        super().__init__(config, "cube", {}, onscreen, offscreen, enable_image_publish)

    def update_reward(self):
        right_hand_body = ["right_hand_thumb_2_link", "right_hand_middle_1_link", "right_hand_index_1_link"]
        gripper_cube_contact = check_contact(self.mj_model, self.mj_data, right_hand_body, "cube_body")
        cube_lifted = check_height(self.mj_model, self.mj_data, "cube", 0.85, 2.0)
        with self.reward_lock:
            self.last_reward = gripper_cube_contact & cube_lifted

class BoxEnv(DefaultEnv):
    def __init__(self, config: Dict[str, any], onscreen: bool = False, offscreen: bool = False, enable_image_publish: bool = False):
        config = config.copy()
        # ⭕ 올바른 경로 적용
        config["ROBOT_SCENE"] = "decoupled_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml"
        super().__init__(config, "box", {}, onscreen, offscreen, enable_image_publish)

    def update_reward(self):
        left_hand_body = ["left_hand_thumb_2_link", "left_hand_middle_1_link", "left_hand_index_1_link"]
        right_hand_body = ["right_hand_thumb_2_link", "right_hand_middle_1_link", "right_hand_index_1_link"]
        gripper_box_contact = check_contact(self.mj_model, self.mj_data, left_hand_body, "box_body")
        gripper_box_contact &= check_contact(self.mj_model, self.mj_data, right_hand_body, "box_body")
        box_lifted = check_height(self.mj_model, self.mj_data, "box", 0.92, 2.0)
        with self.reward_lock:
            self.last_reward = gripper_box_contact & box_lifted

class BottleEnv(DefaultEnv):
    def __init__(self, config: Dict[str, any], onscreen: bool = False, offscreen: bool = False, enable_image_publish: bool = False):
        config = config.copy()
        # ⭕ 올바른 경로 적용
        config["ROBOT_SCENE"] = "decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml"
        camera_configs = {"egoview": {"height": 400, "width": 400}}
        super().__init__(config, "cylinder", camera_configs, onscreen, offscreen, enable_image_publish)
        self.bottle_body = self.mj_model.body("bottle_body")
        if self.viewer is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.mj_model.camera("egoview").id

    def get_privileged_obs(self):
        return {"bottle_pos": self.mj_data.xpos[self.bottle_body.id], "bottle_quat": self.mj_data.xquat[self.bottle_body.id]}


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        elif env_name == "g1_torres_open":
            self.sim_env = G1TorresOpenEnv(config, **kwargs)
        elif env_name == "g1_torres_hinge":
            self.sim_env = G1TorresHingeEnv(config, **kwargs)
        elif env_name == "g1_torres_mesh_hinge":
            self.sim_env = G1TorresMeshHingeEnv(config, **kwargs)
        elif env_name == "pnp_cube":
            self.sim_env = CubeEnv(config, **kwargs)
        elif env_name == "lift_box":
            self.sim_env = BoxEnv(config, **kwargs)
        elif env_name == "pnp_bottle":
            self.sim_env = BottleEnv(config, **kwargs)
        else:
            raise ValueError(f"Invalid environment name: {env_name}")

        # Unitree Bridge 설정
        #self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        #self.sim_env.set_unitree_bridge(self.unitree_bridge)

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
