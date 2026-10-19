"""MuJoCo sim backend for the G1 29DoF, drop-in for G1RealWorldEnv (sim2sim).

Same interface the real DDS wrapper exposes (get_robot_state / send_robot_action /
move_to_default_pos / close) so the policy loop is identical in sim and on hardware.
Physics uses the standard Unitree `g1_29dof` model whose joint/actuator order is
identity to the hardware (== our 0..28), with torque actuators driven by the same HOMIE
PD gains as the real robot.

Keyboard (in the viewer window) drives cmd_vel:
    W / S : vx +/-        A / D : vy +/-        Q / E : wz +/-
    space : zero cmd (force stand)             close window / Esc : quit
"""
import numpy as np

import mujoco
import mujoco.viewer as mjv


class SimController:
    """Mimics the wireless-controller object fields the loop reads (unused in sim)."""
    def __init__(self):
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.keys = 0


class G1MujocoEnv:
    N_DOF = 29

    def __init__(self, xml_path, config, sim_dt=0.005, init_height=0.793,
                 cmd_step=0.1, settle_time=0.5, realtime=True):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)

        self.sim_dt = sim_dt
        self.control_dt = config.control_dt
        self.decimation = max(1, int(round(self.control_dt / sim_dt)))
        self.realtime = realtime

        self.kps = np.asarray(config.kps, dtype=np.float64)
        self.kds = np.asarray(config.kds, dtype=np.float64)
        self.default_angles = np.asarray(config.default_angles, dtype=np.float64)
        self.ctrl_range = self.model.actuator_ctrlrange.copy()  # (29, 2) torque limits

        # base pose + default joints -> full qpos [pos(3), quat(4), joints(29)]
        self.init_qpos = np.zeros(self.model.nq)
        self.init_qpos[0:3] = [0.0, 0.0, init_height]
        self.init_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.init_qpos[7:7 + self.N_DOF] = self.default_angles

        # IMU sensor addresses (framequat + gyro at the pelvis "imu" site)
        self._quat_adr = self._sensor_adr("imu_quat")
        self._gyro_adr = self._sensor_adr("imu_gyro")

        self.cmd = np.zeros(3, dtype=np.float32)
        self.cmd_step = float(cmd_step)
        self.max_cmd = np.asarray(config.max_cmd, dtype=np.float32)
        self.settle_time = float(settle_time)
        self._target = self.default_angles.copy()
        self.controller_mapping = {"select": 0x0008}

        self.viewer = mjv.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False,
            key_callback=self._on_key,
        )
        self.viewer.cam.distance = 3.0

    # --- sensors -----------------------------------------------------------
    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            return None
        return self.model.sensor_adr[sid]

    def _quat(self):
        if self._quat_adr is not None:
            return self.data.sensordata[self._quat_adr:self._quat_adr + 4].copy()
        return self.data.qpos[3:7].copy()

    def _gyro(self):
        if self._gyro_adr is not None:
            return self.data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        return self.data.qvel[3:6].copy()

    # --- keyboard cmd_vel --------------------------------------------------
    def _on_key(self, keycode):
        try:
            c = chr(keycode).upper()
        except ValueError:
            return
        s = self.cmd_step
        if c == "W":   self.cmd[0] += s
        elif c == "S": self.cmd[0] -= s
        elif c == "A": self.cmd[1] += s
        elif c == "D": self.cmd[1] -= s
        elif c == "Q": self.cmd[2] += s
        elif c == "E": self.cmd[2] -= s
        elif c == " ": self.cmd[:] = 0.0
        self.cmd[:] = np.clip(self.cmd, -self.max_cmd, self.max_cmd)

    def get_command(self):
        return self.cmd.copy()

    def read_controller_input(self):
        return SimController()

    def is_running(self):
        return self.viewer.is_running()

    # --- state / action (same signatures as the real env) -----------------
    def get_robot_state(self):
        dof_pos = self.data.qpos[7:7 + self.N_DOF].astype(np.float32).copy()
        dof_vel = self.data.qvel[6:6 + self.N_DOF].astype(np.float32).copy()
        quat = self._quat().astype(np.float32)       # [w, x, y, z]
        ang_vel = self._gyro().astype(np.float32)     # base-frame gyro
        return dof_pos, dof_vel, quat, ang_vel

    def _pd_torque(self, target, kp_scale, kd_scale):
        q = self.data.qpos[7:7 + self.N_DOF]
        dq = self.data.qvel[6:6 + self.N_DOF]
        tau = self.kps * kp_scale * (target - q) - self.kds * kd_scale * dq
        return np.clip(tau, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0):
        """Advance the sim one control step (decimation substeps of torque PD)."""
        self._target = np.asarray(target_dof_pos, dtype=np.float64).copy()
        for _ in range(self.decimation):
            self.data.ctrl[:] = self._pd_torque(self._target, kp_scale, kd_scale)
            mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def send_damping(self, kd=8.0):
        for _ in range(self.decimation):
            dq = self.data.qvel[6:6 + self.N_DOF]
            self.data.ctrl[:] = np.clip(-kd * dq, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    # --- bring-up ----------------------------------------------------------
    def move_to_default_pos(self):
        """Reset to the standing default pose and settle under PD (no waiting)."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(self.settle_time / self.control_dt)):
            self.send_robot_action(self.default_angles)

    def default_pos_state(self):
        return  # nothing to wait for in sim

    def close(self):
        try:
            self.viewer.close()
        except Exception:
            pass
