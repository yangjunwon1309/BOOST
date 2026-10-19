"""Self-contained HOMIE G1 29DoF policy (numpy + onnxruntime only).

Faithful port of robofinals `G1HomiePolicyV2` (wbc_policy/policy/g1_homie_policy.py),
stripped of Isaac / robot_model / torch dependencies. Because the HOMIE "body" joint
group is sorted and identical to the Unitree G1JointIndex ordering, no joint
permutation is needed: the 29-dim hardware q/dq feed straight in.

Observation (single frame = 86, history = 6, total = 516):
    [0:3]   cmd[vx,vy,wz] * cmd_scale
    [3:4]   height_cmd
    [4:7]   torso rpy cmd
    [7:10]  base angular velocity * ang_vel_scale
    [10:13] projected gravity (from base quat, w-x-y-z)
    [13:42] (q - default) * dof_pos_scale        (29; arm defaults padded to 0)
    [42:71] dq * dof_vel_scale                    (29)
    [71:86] previous 15-dim policy action

Action -> target: q_target[0:15] = action * action_scale + default_angles(15)
Policy selection: ||cmd[vx,vy,wz]|| < stand_cmd_threshold -> stand.onnx, else walk.onnx
"""
import collections

import numpy as np
import onnxruntime as ort

from homie_utils import get_gravity_orientation


class HomiePolicyV2:
    N_JOINTS = 29
    SINGLE_OBS_DIM = 86

    def __init__(self, config, stand_onnx_path, walk_onnx_path, providers=None, onnx_threads=1):
        self.config = config
        self.num_actions = int(config["num_actions"])            # 15 (legs + waist)
        self.obs_history_len = int(config["obs_history_len"])    # 6
        self.num_obs = int(config["num_obs"])                    # 516

        self.default_angles = np.asarray(config["default_angles"], dtype=np.float32)  # 15
        self.action_scale = float(config["action_scale"])
        self.dof_pos_scale = float(config["dof_pos_scale"])
        self.dof_vel_scale = float(config["dof_vel_scale"])
        self.ang_vel_scale = float(config["ang_vel_scale"])
        self.cmd_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
        self.height_cmd = float(config["height_cmd"])
        self.rpy_cmd = np.asarray(config["rpy_cmd"], dtype=np.float32)
        self.stand_cmd_threshold = float(config.get("stand_cmd_threshold", 0.2))

        # 29-dim defaults with arm entries padded to 0 so arm obs stays absolute.
        self.padded_defaults = np.zeros(self.N_JOINTS, dtype=np.float32)
        self.padded_defaults[: self.num_actions] = self.default_angles

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.stand_sess, self.stand_input = self._load(stand_onnx_path, providers, onnx_threads)
        self.walk_sess, self.walk_input = self._load(walk_onnx_path, providers, onnx_threads)

        self.cmd = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        self.last_policy = "stand"

    @staticmethod
    def _load(path, providers, onnx_threads=1):
        """onnx_threads caps onnxruntime's own intra/inter-op thread pools. Left at its
        default, onnxruntime spawns a thread per CPU core for EVERY inference call, even
        for a tiny 516->15 MLP -- fine in isolation, but on a control loop that shares
        the process with DDS reader/writer threads, that thread pool's context-switching
        competes for CPU/GIL time and was observed (empirically, in this repo) to spike
        rt/lowcmd write latency into the tens of ms exactly when inference ran. Pin it to
        1 unless you've profiled otherwise -- a model this small gains nothing from
        multithreading anyway."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(onnx_threads)
        opts.inter_op_num_threads = int(onnx_threads)
        sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
        name = sess.get_inputs()[0].name
        out_shape = sess.get_outputs()[0].shape
        print(f"[homie] loaded {path}  in={sess.get_inputs()[0].shape} out={out_shape} "
              f"providers={sess.get_providers()}  onnx_threads={onnx_threads}")
        return sess, name

    def reset(self):
        """Clear history so the next update warm-starts from the current pose."""
        self.obs_history.clear()

    def set_command(self, cmd_xy_yaw):
        self.cmd = np.asarray(cmd_xy_yaw, dtype=np.float32).reshape(3)

    def _single_obs(self, q29, dq29, base_quat, base_ang_vel):
        obs = np.zeros(self.SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:3] = self.cmd * self.cmd_scale
        obs[3] = self.height_cmd
        obs[4:7] = self.rpy_cmd
        obs[7:10] = base_ang_vel * self.ang_vel_scale
        obs[10:13] = get_gravity_orientation(base_quat)
        obs[13:13 + self.N_JOINTS] = (q29 - self.padded_defaults) * self.dof_pos_scale
        obs[13 + self.N_JOINTS:13 + 2 * self.N_JOINTS] = dq29 * self.dof_vel_scale
        obs[13 + 2 * self.N_JOINTS:13 + 2 * self.N_JOINTS + self.num_actions] = self.action
        return obs

    def update(self, q29, dq29, base_quat, base_ang_vel):
        """Run one inference step.

        Args:
            q29, dq29:      29-dim joint pos/vel in hardware order.
            base_quat:      base IMU quaternion [w, x, y, z].
            base_ang_vel:   base IMU angular velocity [wx, wy, wz] (rad/s).

        Returns:
            (q_target15, policy_name) -- absolute targets for joints 0..14.
        """
        q29 = np.asarray(q29, dtype=np.float32)
        dq29 = np.asarray(dq29, dtype=np.float32)
        base_quat = np.asarray(base_quat, dtype=np.float32).reshape(4)
        base_ang_vel = np.asarray(base_ang_vel, dtype=np.float32).reshape(3)

        # Warm start: on reset, seed the previous action so the first obs (and the 5
        # duplicated history frames) correspond to holding the current pose. Zeros
        # are out-of-distribution for the standing policy and cause a startup jump.
        if len(self.obs_history) == 0:
            self.action = (
                (q29[: self.num_actions] - self.default_angles) / self.action_scale
            ).astype(np.float32)

        single = self._single_obs(q29, dq29, base_quat, base_ang_vel)
        self.obs_history.append(single)
        while len(self.obs_history) < self.obs_history_len:
            self.obs_history.appendleft(single.copy())

        obs_buf = np.concatenate(list(self.obs_history)).reshape(1, -1)
        assert obs_buf.shape[1] == self.num_obs, (
            f"obs size {obs_buf.shape[1]} != {self.num_obs}"
        )

        if np.linalg.norm(self.cmd) < self.stand_cmd_threshold:
            sess, name, self.last_policy = self.stand_sess, self.stand_input, "stand"
        else:
            sess, name, self.last_policy = self.walk_sess, self.walk_input, "walk"

        out = sess.run(None, {name: obs_buf})[0]
        self.action = np.asarray(out[0], dtype=np.float32)

        q_target15 = self.action * self.action_scale + self.default_angles
        return q_target15, self.last_policy
