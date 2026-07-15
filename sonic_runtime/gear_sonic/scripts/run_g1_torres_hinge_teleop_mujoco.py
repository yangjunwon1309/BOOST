"""Run PICO/SONIC teleop in MuJoCo with G1 + articulated Torres open-door scene.

This is the 3rd-stage scene: the car door is a real MuJoCo hinge joint, while
vehicle/body contacts remain simple proxy geoms for stability.

Run from GR00T root:
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_g1_torres_hinge_teleop_mujoco.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import tyro

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel


ArgsConfig = SimLoopConfig


def _resolve_scene_path() -> str:
    candidates = [
        Path("gear_sonic_deploy/gear_sonic_deploy/g1/scene_29dof_torres_hinge.xml"),
        Path("gear_sonic_deploy/g1/scene_29dof_torres_hinge.xml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    return candidates[0].as_posix()


class TorresHingeTeleopSimWrapper:
    def __init__(self, robot_model: RobotModel, config: Dict[str, Any], **kwargs):
        self.robot_model = robot_model
        self.config = config
        try:
            init_channel(config=self.config)
        except Exception as exc:
            # Match the original run_sim_loop behavior: ChannelFactory can already
            # be initialized by another process; this warning is non-fatal.
            print(f"Note: Channel factory initialization attempt: {exc}")
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name="g1_torres_hinge",
            **kwargs,
        )


def main(config: ArgsConfig):
    # In the user's working sim2sim setup the Unitree DDS interface is loopback.
    config.interface = "lo"
    config.env_name = "g1_torres_hinge"
    config.enable_onscreen = True

    scene_path = _resolve_scene_path()
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = "g1_torres_hinge"
    wbc_config["ENV_TYPE"] = "sim"
    wbc_config["INTERFACE"] = "lo"
    wbc_config["ROBOT_SCENE"] = scene_path
    wbc_config["G1_TORRES_HINGE_SCENE"] = scene_path
    wbc_config["G1_TORRES_DOOR_OPEN_ANGLE"] = 1.15
    wbc_config["ENABLE_ONSCREEN"] = True
    wbc_config["ENABLE_OFFSCREEN"] = bool(config.enable_offscreen)

    if config.enable_image_publish:
        assert config.enable_offscreen, "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()
    sim_wrapper = TorresHingeTeleopSimWrapper(
        robot_model=robot_model,
        config=wbc_config,
        onscreen=True,
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
