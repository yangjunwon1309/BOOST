"""Run PICO/SONIC teleop in MuJoCo with G1 + Torres visual-mesh hinge scene.

This uses converted Torres OBJ visual meshes for an IsaacLab-like car shape,
with decomposed MuJoCo mesh collisions and a real hinge joint for the front-left
driver door.

Before running this script, generate the OBJ assets and scene:
  python scripts/convert_torres_dae_to_obj_and_scene.py --yaw-deg 90

Run from GR00T root:
  source .venv_sim/bin/activate
  python gear_sonic/scripts/run_g1_torres_mesh_hinge_teleop_mujoco.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import tyro

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel


@dataclass
class TorresMeshHingeTeleopConfig(SimLoopConfig):
    start_mode: Literal["handle-test", "entry-test"] = "handle-test"
    """Initial setup: handle-test starts outside the closed door; entry-test starts in the doorway with the door open."""

    entry_door_open_angle: float = 1.507964474
    """Driver door hinge angle used by entry-test mode."""

    entry_disable_body_collisions: bool = False
    """Disable static Torres body collision hulls in entry-test mode when debugging cabin entry clearance."""

    easy_mode: bool = False
    """Make only the Torres car base/body visual-only; keep door, handle, and driver-seat collisions."""

    detail_mode: bool = False
    """Use the older deploy OBJ/MTL asset set with denser visual materials and collision pieces."""


ArgsConfig = TorresMeshHingeTeleopConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _regenerate_torres_scene(detail_mode: bool, easy_mode: bool) -> None:
    root = _repo_root()
    converter = root / "scripts" / "convert_torres_dae_to_obj_and_scene.py"
    cmd = [sys.executable, converter.as_posix(), "--root", root.as_posix()]
    if detail_mode:
        cmd.append("--detail_mode")
    if easy_mode:
        cmd.extend(["--car-collision-mode", "bodyless"])

    mode = "detail deploy OBJ" if detail_mode else "default IsaacLab OBJ"
    if easy_mode:
        mode += ", easy bodyless collision"
    print(f"Preparing Torres scene assets ({mode})...", flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def _resolve_scene_path() -> str:
    candidates = [
        Path("gear_sonic_deploy/gear_sonic_deploy/g1/scene_29dof_torres_mesh_hinge.xml"),
        Path("gear_sonic_deploy/g1/scene_29dof_torres_mesh_hinge.xml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    return candidates[0].as_posix()


class TorresMeshHingeTeleopSimWrapper:
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
            env_name="g1_torres_mesh_hinge",
            **kwargs,
        )


def main(config: ArgsConfig):
    # In the working sim2sim setup the Unitree DDS interface is loopback.
    config.interface = "lo"
    config.env_name = "g1_torres_mesh_hinge"
    config.enable_onscreen = True

    _regenerate_torres_scene(config.detail_mode, config.easy_mode)
    scene_path = _resolve_scene_path()
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = "g1_torres_mesh_hinge"
    wbc_config["ENV_TYPE"] = "sim"
    wbc_config["INTERFACE"] = "lo"
    wbc_config["ROBOT_SCENE"] = scene_path
    wbc_config["G1_TORRES_MESH_HINGE_SCENE"] = scene_path
    if config.start_mode == "entry-test":
        # Start in the open doorway, facing the same direction as the car front.
        # The 180-degree yaw keeps the test pose mirrored from the previous
        # rear-facing entry setup.
        # robot_start_pos = [1.0, 0.25, 0.793]
        robot_start_pos = [1.0, -0.25, 0.793]
        door_open_angle = config.entry_door_open_angle
        robot_start_quat = [0.0, 0.0, 0.0, 1.0]
        wbc_config["G1_TORRES_ENTRY_DISABLE_BODY_COLLISIONS"] = bool(
            config.entry_disable_body_collisions
        )
        wbc_config["G1_TORRES_DOOR_HOLD_OPEN"] = True
        wbc_config["G1_TORRES_DOOR_ASSIST_ENABLED"] = False
    else:
        # Start on the MuJoCo driver side, facing the closed handle.  This keeps
        # the robot clear of the door sweep and hinge-side collision hulls when
        # the elastic band is released.
        robot_start_pos = [0.80, -0.85, 0.793]
        door_open_angle = 0.0
        robot_start_quat = [0.70710678, 0.0, 0.0, 0.70710678]
        wbc_config["G1_TORRES_ENTRY_DISABLE_BODY_COLLISIONS"] = False
        wbc_config["G1_TORRES_DOOR_HOLD_OPEN"] = False
        wbc_config["G1_TORRES_DOOR_ASSIST_ENABLED"] = True

    # This is the open angle for the MuJoCo hinge/proxy door.  It does not affect
    # the static car-board mesh; it rotates the door mesh body and collision proxy.
    wbc_config["G1_TORRES_DOOR_OPEN_ANGLE"] = door_open_angle
    wbc_config["G1_TORRES_DOOR_ASSIST_OPEN_ANGLE"] = max(1.35, door_open_angle)
    wbc_config["G1_TORRES_MESH_HINGE_ROBOT_START_POS"] = robot_start_pos
    wbc_config["G1_TORRES_MESH_HINGE_ROBOT_START_QUAT"] = robot_start_quat
    wbc_config["G1_TORRES_ELASTIC_BAND_ANCHOR"] = [robot_start_pos[0], robot_start_pos[1], 1.0]
    wbc_config["ENABLE_ONSCREEN"] = True
    wbc_config["ENABLE_OFFSCREEN"] = bool(config.enable_offscreen)

    if config.enable_image_publish:
        assert config.enable_offscreen, "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()
    sim_wrapper = TorresMeshHingeTeleopSimWrapper(
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
