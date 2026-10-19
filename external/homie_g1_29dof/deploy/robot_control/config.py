import numpy as np
import yaml


class Config:
    """Hardware/deploy config loader (mirrors TWIST2 deploy_real/robot_control/config.py)."""

    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.control_dt = config["control_dt"]

        self.msg_type = config["msg_type"]
        self.imu_type = config["imu_type"]

        self.weak_motor = config.get("weak_motor", [])

        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]

        self.joint2motor_idx = config["joint2motor_idx"]
        self.kps = config["kps"]
        self.kds = config["kds"]
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)

        self.num_actions = config.get("num_actions")

        # cmd_vel from the wireless controller
        self.max_cmd = np.array(config.get("max_cmd", [0.6, 0.4, 0.6]), dtype=np.float32)
        self.cmd_deadzone = float(config.get("cmd_deadzone", 0.1))

        # cmd_vel source: "wireless", "udp" (Thor->G1 production path) or
        # "dds" (legacy rt/nav_cmd compatibility). Physical sticks always win.
        self.cmd_source = config.get("cmd_source", "wireless")
        self.cmd_dds_timeout = float(config.get("cmd_dds_timeout", 0.3))
        self.cmd_udp_bind_host = config.get("cmd_udp_bind_host", "0.0.0.0")
        self.cmd_udp_port = int(config.get("cmd_udp_port", 15001))
        self.cmd_udp_allowed_host = config.get(
            "cmd_udp_allowed_host", "192.168.123.2"
        )
        self.cmd_udp_timeout = float(
            config.get("cmd_udp_timeout", self.cmd_dds_timeout)
        )

        # arm target source
        self.arm_source = config.get("arm_source", "hold")
        self.arm_redis_key = config.get("arm_redis_key", "action_arm_unitree_g1_homie")
        self.arm_redis_timeout = float(config.get("arm_redis_timeout", 0.2))
        self.arm_smoothing_tau = float(config.get("arm_smoothing_tau", 0.03))

        # arm feedforward torque (tau) safety: conservative defaults if unset, never
        # allow an arbitrarily large torque just because the config omitted these.
        # arm_tau_clamp may be a scalar (broadcast) or a length-14 per-joint list (e.g.
        # G1 URDF effort limits) -- kept as-is here, normalized/validated to a (14,)
        # array by arm_target_source.py's _normalize_tau_clamp().
        self.arm_tau_scale = float(config.get("arm_tau_scale", 1.0))
        self.arm_tau_clamp = config.get("arm_tau_clamp", 5.0)  # N*m, scalar or length-14
