#!/usr/bin/env python3
"""Sim2sim: run the HOMIE stand/walk policy in MuJoCo (mirrors run_homie.py).

Uses the SAME HomiePolicyV2, config, gains, and stand/walk switch as the real robot;
only the backend env changes (MuJoCo instead of DDS). Drive cmd_vel from the viewer
keyboard: W/S=vx, A/D=vy, Q/E=wz, space=stand.

Example:
  python run_homie_sim.py \
    --xml ~/01_NRF/url_sim2real/third_party/LEB_unitree_mujoco/unitree_robots/g1/scene_29dof.xml
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from homie_utils import load_config              # noqa: E402
from homie_policy import HomiePolicyV2           # noqa: E402
from arm_target_source import make_arm_source    # noqa: E402
from robot_control.config import Config          # noqa: E402

DEFAULT_XML = os.path.expanduser(
    "~/01_NRF/url_sim2real/third_party/LEB_unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
)


def parse_args():
    p = argparse.ArgumentParser(description="Run HOMIE stand/walk policy in MuJoCo (sim2sim)")
    p.add_argument("--xml", default=DEFAULT_XML, help="MuJoCo scene xml (g1_29dof)")
    p.add_argument("--config", default=os.path.join(_HERE, "robot_control/configs/g1_homie.yaml"))
    p.add_argument("--policy-config", default=os.path.join(_HERE, "config/g1_homie_v2.yaml"))
    p.add_argument("--stand", default=os.path.join(_HERE, "../ckpts/stand.onnx"))
    p.add_argument("--walk", default=os.path.join(_HERE, "../ckpts/walk.onnx"))
    p.add_argument("--sim-dt", type=float, default=0.005, help="MuJoCo timestep")
    p.add_argument("--init-height", type=float, default=0.793, help="initial pelvis height")
    p.add_argument("--ramp-time", type=float, default=1.0, help="ease-in from default to policy")
    p.add_argument("--no-realtime", action="store_true", help="run as fast as possible")
    return p.parse_args()


def main():
    args = parse_args()
    for path in (args.xml, args.config, args.policy_config, args.stand, args.walk):
        if not os.path.exists(os.path.expanduser(path)):
            sys.exit(f"Error: file not found: {path}")

    # Imported here so the rest of the package works without mujoco installed.
    from robot_control.g1_mujoco_wrapper import G1MujocoEnv

    config = Config(args.config)
    policy_config = load_config(args.policy_config)
    policy = HomiePolicyV2(policy_config, args.stand, args.walk)
    arm_source = make_arm_source(config)   # "hold" by default in sim

    env = G1MujocoEnv(os.path.expanduser(args.xml), config,
                      sim_dt=args.sim_dt, init_height=args.init_height,
                      realtime=not args.no_realtime)

    print("\n" + "=" * 60)
    print("MuJoCo sim2sim. Focus the viewer, then:")
    print("  W/S=vx  A/D=vy  Q/E=wz  space=stand   (close window to quit)")
    print(f"  stand/walk threshold: ||cmd|| = {policy.stand_cmd_threshold} m/s")
    print("=" * 60 + "\n")

    env.move_to_default_pos()
    policy.reset()

    control_dt = config.control_dt
    ramp_steps = max(1, int(args.ramp_time / control_dt))
    default_full = config.default_angles.copy()
    step = 0
    last_print = 0.0

    try:
        while env.is_running():
            t0 = time.time()

            cmd = env.get_command()
            policy.set_command(cmd)

            q29, dq29, base_quat, base_ang_vel = env.get_robot_state()
            q_target15, which = policy.update(q29, dq29, base_quat, base_ang_vel)
            arm_target = arm_source.get_arm_target()
            q_target = np.concatenate([q_target15, arm_target]).astype(np.float32)

            if step < ramp_steps:
                alpha = (step + 1) / ramp_steps
                q_target = (1.0 - alpha) * default_full + alpha * q_target
            step += 1

            env.send_robot_action(q_target)  # steps physics + viewer sync

            if t0 - last_print > 0.5:
                print(f"[{which:>5}] |cmd|={np.linalg.norm(cmd):.2f} "
                      f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}]  "
                      f"base_z={env.data.qpos[2]:+.2f}", end="\r")
                last_print = t0

            if env.realtime:
                elapsed = time.time() - t0
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt -> exiting.")
    finally:
        env.close()
        print("\nSim finished.")


if __name__ == "__main__":
    main()
