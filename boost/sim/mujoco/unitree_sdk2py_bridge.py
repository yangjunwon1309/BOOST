# Ported from PATH-G1 sonic_runtime/gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py.
# SPDX-License-Identifier: Apache-2.0
"""Bridge between Unitree SDK2 DDS topics and the MuJoCo simulation.

Subscribes to low-level motor commands (body + hands) and publishes
simulated sensor state (joint pos/vel, IMU, odometry) back over DDS,
so the WBC policy sees the sim as a real robot.
"""

import sys
import threading
from typing import Dict, Tuple

import numpy as np
import scipy.spatial.transform
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__MotorCmd_,
    unitree_go_msg_dds__MotorState_,
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    MotorCmds_,
    MotorStates_,
    WirelessController_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, OdoState_

from boost.sim.mujoco.hand_backends import (
    DEX3_BACKEND,
    REVO2_BACKEND,
    Revo2CommandAdapter,
)


def _new_brainco_command(count: int) -> MotorCmds_:
    return MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(count)])


def _new_brainco_state(count: int) -> MotorStates_:
    return MotorStates_([unitree_go_msg_dds__MotorState_() for _ in range(count)])


class UnitreeSdk2Bridge:
    """
    This class is responsible for bridging the Unitree SDK2 with the Groot environment.
    It is responsible for sending and receiving messages to and from the Unitree SDK2.
    Both the body and hand are supported.
    """

    def __init__(self, config):
        # Note that we do not give the mjdata and mjmodel to the UnitreeSdk2Bridge.
        # It is unsafe and would be unflexible if we use a hand-plugged robot model

        robot_type = config["ROBOT_TYPE"]
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
                unitree_hg_msg_dds__LowCmd_,
                unitree_hg_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__OdoState_ as OdoState_default,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" == robot_type or "go2" == robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_, LowCmd_, LowState_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")

        self.num_body_motor = config["NUM_MOTORS"]
        self.num_hand_motor = config.get("NUM_HAND_MOTORS", 0)
        self.hand_backend = config.get("HAND_BACKEND", DEX3_BACKEND)
        if self.hand_backend not in (DEX3_BACKEND, REVO2_BACKEND):
            raise ValueError(f"Unsupported HAND_BACKEND: {self.hand_backend}")
        self.revo2_adapter = Revo2CommandAdapter(
            binary_open=tuple(config.get("BRAINCO_BINARY_OPEN", (0.0,) * 6)),
            binary_close=tuple(
                config.get("BRAINCO_BINARY_CLOSE", (0.70, 1.0, 1.0, 1.0, 1.0, 1.0))
            ),
            legacy_close_threshold=float(config.get("BRAINCO_LEGACY_CLOSE_THRESHOLD", 0.10)),
        )
        self.accept_legacy_dex3 = bool(config.get("BRAINCO_ACCEPT_LEGACY_DEX3", False))
        self.use_sensor = config["USE_SENSOR"]

        # Subscribers can invoke callbacks as soon as Init returns, so locks
        # must exist before any subscriber is initialized.
        self.low_cmd_lock = threading.Lock()
        self.left_hand_cmd_lock = threading.Lock()
        self.right_hand_cmd_lock = threading.Lock()

        self.have_imu_ = False
        self.have_frame_sensor_ = False

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        # Only create odo_state for supported robot types
        if "g1" in robot_type or "h1-2" in robot_type:
            self.odo_state = OdoState_default()
            self.odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
            self.odo_state_puber.Init()
        else:
            self.odo_state = None
            self.odo_state_puber = None
        self.torso_imu_state = IMUState_default()
        self.torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_puber.Init()

        if self.hand_backend == REVO2_BACKEND:
            self.left_hand_state = _new_brainco_state(self.num_hand_motor)
            self.left_hand_state_puber = ChannelPublisher(
                "rt/brainco/left/state", MotorStates_
            )
            self.left_hand_state_puber.Init()
            self.right_hand_state = _new_brainco_state(self.num_hand_motor)
            self.right_hand_state_puber = ChannelPublisher(
                "rt/brainco/right/state", MotorStates_
            )
            self.right_hand_state_puber.Init()
        else:
            self.left_hand_state = HandState_default()
            self.left_hand_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
            self.left_hand_state_puber.Init()
            self.right_hand_state = HandState_default()
            self.right_hand_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
            self.right_hand_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.legacy_left_hand_cmd_suber = None
        self.legacy_right_hand_cmd_suber = None
        if self.hand_backend == REVO2_BACKEND:
            self.left_hand_cmd = _new_brainco_command(self.num_hand_motor)
            self.left_hand_cmd_suber = ChannelSubscriber(
                "rt/brainco/left/cmd", MotorCmds_
            )
            self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 1)
            self.right_hand_cmd = _new_brainco_command(self.num_hand_motor)
            self.right_hand_cmd_suber = ChannelSubscriber(
                "rt/brainco/right/cmd", MotorCmds_
            )
            self.right_hand_cmd_suber.Init(self.RightHandCmdHandler, 1)
            if self.accept_legacy_dex3:
                self.legacy_left_hand_cmd_suber = ChannelSubscriber(
                    "rt/dex3/left/cmd", HandCmd_
                )
                self.legacy_left_hand_cmd_suber.Init(self.LegacyLeftHandCmdHandler, 1)
                self.legacy_right_hand_cmd_suber = ChannelSubscriber(
                    "rt/dex3/right/cmd", HandCmd_
                )
                self.legacy_right_hand_cmd_suber.Init(self.LegacyRightHandCmdHandler, 1)
        else:
            self.left_hand_cmd = HandCmd_default()
            self.left_hand_cmd_suber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
            self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 1)
            self.right_hand_cmd = HandCmd_default()
            self.right_hand_cmd_suber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
            self.right_hand_cmd_suber.Init(self.RightHandCmdHandler, 1)

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

        self.reset()

    def reset(self):
        with self.low_cmd_lock:
            self.low_cmd_received = False
            self.new_low_cmd = False
        with self.left_hand_cmd_lock:
            self.left_hand_cmd_received = False
            self.new_left_hand_cmd = False
        with self.right_hand_cmd_lock:
            self.right_hand_cmd_received = False
            self.new_right_hand_cmd = False

    def LowCmdHandler(self, msg):
        with self.low_cmd_lock:
            self.low_cmd = msg
            self.low_cmd_received = True
            self.new_low_cmd = True

    def LeftHandCmdHandler(self, msg):
        with self.left_hand_cmd_lock:
            if self.hand_backend == REVO2_BACKEND:
                if len(msg.cmds) != self.num_hand_motor:
                    return
                command = _new_brainco_command(self.num_hand_motor)
                for index in range(self.num_hand_motor):
                    command.cmds[index].q = float(np.clip(msg.cmds[index].q, 0.0, 1.0))
                    command.cmds[index].dq = float(np.clip(msg.cmds[index].dq, 0.0, 1.0))
                self.left_hand_cmd = command
            else:
                self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True

    def RightHandCmdHandler(self, msg):
        with self.right_hand_cmd_lock:
            if self.hand_backend == REVO2_BACKEND:
                if len(msg.cmds) != self.num_hand_motor:
                    return
                command = _new_brainco_command(self.num_hand_motor)
                for index in range(self.num_hand_motor):
                    command.cmds[index].q = float(np.clip(msg.cmds[index].q, 0.0, 1.0))
                    command.cmds[index].dq = float(np.clip(msg.cmds[index].dq, 0.0, 1.0))
                self.right_hand_cmd = command
            else:
                self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True

    def _set_revo2_binary_command(self, is_left: bool, closed: bool):
        if self.hand_backend != REVO2_BACKEND:
            raise RuntimeError("binary REVO2 command requested while DEX3 backend is active")
        target = self.revo2_adapter.binary_target(closed)
        lock = self.left_hand_cmd_lock if is_left else self.right_hand_cmd_lock
        with lock:
            command = self.left_hand_cmd if is_left else self.right_hand_cmd
            for index, value in enumerate(target):
                command.cmds[index].q = float(value)
                command.cmds[index].dq = 1.0
            if is_left:
                self.left_hand_cmd_received = True
                self.new_left_hand_cmd = True
            else:
                self.right_hand_cmd_received = True
                self.new_right_hand_cmd = True

    def SetBinaryGrip(self, is_left: bool, closed: bool):
        """Public binary-grip adapter used by tests and local sim controllers."""
        self._set_revo2_binary_command(is_left, closed)

    def _legacy_dex3_handler(self, msg, is_left: bool):
        values = [msg.motor_cmd[i].q for i in range(7)]
        self._set_revo2_binary_command(
            is_left,
            self.revo2_adapter.legacy_dex3_is_closed(values),
        )

    def LegacyLeftHandCmdHandler(self, msg):
        self._legacy_dex3_handler(msg, True)

    def LegacyRightHandCmdHandler(self, msg):
        self._legacy_dex3_handler(msg, False)

    def get_hand_command_q(self, is_left: bool) -> np.ndarray:
        lock = self.left_hand_cmd_lock if is_left else self.right_hand_cmd_lock
        with lock:
            command = self.left_hand_cmd if is_left else self.right_hand_cmd
            if self.hand_backend == REVO2_BACKEND:
                values = [command.cmds[i].q for i in range(self.num_hand_motor)]
                return self.revo2_adapter.clamp_normalized(values)
            return np.asarray(
                [command.motor_cmd[i].q for i in range(self.num_hand_motor)],
                dtype=np.float64,
            )

    def cmd_received(self):
        with self.low_cmd_lock:
            low_cmd_received = self.low_cmd_received
        with self.left_hand_cmd_lock:
            left_hand_cmd_received = self.left_hand_cmd_received
        with self.right_hand_cmd_lock:
            right_hand_cmd_received = self.right_hand_cmd_received
        return low_cmd_received or left_hand_cmd_received or right_hand_cmd_received

    def PublishLowState(self, obs: Dict[str, any]):
        # publish body state
        if self.use_sensor:
            raise NotImplementedError("Sensor data is not implemented yet.")
        else:
            for i in range(self.num_body_motor):
                self.low_state.motor_state[i].q = obs["body_q"][i]
                self.low_state.motor_state[i].dq = obs["body_dq"][i]
                self.low_state.motor_state[i].ddq = obs["body_ddq"][i]
                self.low_state.motor_state[i].tau_est = obs["body_tau_est"][i]

        if self.use_sensor and self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        else:
            # Get data from ground truth
            self.odo_state.position[:] = obs["floating_base_pose"][:3]
            self.odo_state.linear_velocity[:] = obs["floating_base_vel"][:3]
            self.odo_state.orientation[:] = obs["floating_base_pose"][3:7]
            self.odo_state.angular_velocity[:] = obs["floating_base_vel"][3:6]
            # quaternion: w, x, y, z
            self.low_state.imu_state.quaternion[:] = obs["floating_base_pose"][3:7]
            # angular velocity
            self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
            # linear acceleration
            self.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]

            self.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
            self.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]

        # acceleration: x, y, z (only available when frame sensor is enabled)
        if self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        self.low_state.tick = int(obs["time"] * 1e3)
        self.low_state_puber.Write(self.low_state)

        self.odo_state.tick = int(obs["time"] * 1e3)
        self.odo_state_puber.Write(self.odo_state)

        self.torso_imu_puber.Write(self.torso_imu_state)

        # Publish the active hand protocol.  BrainCo q/dq are normalized,
        # whereas legacy DEX3 state remains in joint radians.
        if self.num_hand_motor > 0:
            if self.hand_backend == REVO2_BACKEND:
                left_q = self.revo2_adapter.actuated_to_normalized(
                    obs["left_hand_q"][: self.num_hand_motor]
                )
                right_q = self.revo2_adapter.actuated_to_normalized(
                    obs["right_hand_q"][: self.num_hand_motor]
                )
                left_dq = self.revo2_adapter.normalized_velocity(
                    obs["left_hand_dq"][: self.num_hand_motor]
                )
                right_dq = self.revo2_adapter.normalized_velocity(
                    obs["right_hand_dq"][: self.num_hand_motor]
                )
                for i in range(self.num_hand_motor):
                    self.left_hand_state.states[i].q = float(left_q[i])
                    self.left_hand_state.states[i].dq = float(left_dq[i])
                    self.left_hand_state.states[i].tau_est = float(obs["left_hand_tau_est"][i])
                    self.right_hand_state.states[i].q = float(right_q[i])
                    self.right_hand_state.states[i].dq = float(right_dq[i])
                    self.right_hand_state.states[i].tau_est = float(obs["right_hand_tau_est"][i])
            else:
                for i in range(self.num_hand_motor):
                    self.left_hand_state.motor_state[i].q = obs["left_hand_q"][i]
                    self.left_hand_state.motor_state[i].dq = obs["left_hand_dq"][i]
                    self.right_hand_state.motor_state[i].q = obs["right_hand_q"][i]
                    self.right_hand_state.motor_state[i].dq = obs["right_hand_dq"][i]
            self.left_hand_state_puber.Write(self.left_hand_state)
            self.right_hand_state_puber.Write(self.right_hand_state)

    def GetAction(self) -> Tuple[np.ndarray, bool, bool]:
        with self.low_cmd_lock:
            body_q = [self.low_cmd.motor_cmd[i].q for i in range(self.num_body_motor)]
        left_hand_q = self.get_hand_command_q(True).tolist()
        right_hand_q = self.get_hand_command_q(False).tolist()
        with self.low_cmd_lock, self.left_hand_cmd_lock, self.right_hand_cmd_lock:
            is_new_action = self.new_low_cmd and self.new_left_hand_cmd and self.new_right_hand_cmd
            if is_new_action:
                self.new_low_cmd = False
                self.new_left_hand_cmd = False
                self.new_right_hand_cmd = False

        return (
            np.concatenate([body_q[:-7], left_hand_q, body_q[-7:], right_hand_q]),
            self.cmd_received(),
            is_new_action,
        )

    def PublishWirelessController(self):
        import pygame

        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            # Pack 16 button states into a single integer via bit-shifting
            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        import pygame

        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 3, "RY": 4,
                    "LT": 2, "RT": 5, "DX": 6, "DY": 7,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
                    "XBOX": 8, "LSB": 9, "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                    "LT": 4, "RT": 5,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 9, "RB": 10, "SELECT": 4, "START": 6,
                    "XBOX": 5, "LSB": 7, "RSB": 8,
                    "DYU": 11, "DYD": 12, "DXL": 13, "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0,
                "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        import mujoco
        from loguru import logger
        from termcolor import colored

        print(" ")
        logger.info(colored("<<------------- Link ------------->>", "green"))
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                logger.info(f"link_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Joint ------------->>", "green"))
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                logger.info(f"joint_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Actuator ------------->>", "green"))
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                logger.info(f"actuator_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Sensor ------------->>", "green"))
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                logger.info(
                    f"sensor_index: {index}, name: {name}, dim: {self.mj_model.sensor_dim[i]}"
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(self):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        self.point = np.array([0, 0, 1])
        self.length = 0
        self.enable = True

    def Advance(self, pose):
        pos = pose[0:3]
        quat = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        δx = self.point - pos
        f = self.kp_pos * (δx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)

        # Convert quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot = scipy.spatial.transform.Rotation.from_quat(quat)
        rotvec = rot.as_rotvec()
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        import glfw

        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def handle_keyboard_button(self, key):
        if key == "9":
            self.enable = not self.enable
            print(f"ElasticBand enable: {self.enable}")
