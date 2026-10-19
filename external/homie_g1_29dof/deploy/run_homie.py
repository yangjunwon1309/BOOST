#!/usr/bin/env python3
"""Run the HOMIE stand/walk policy on the G1 29DoF, onboard, straight over DDS.

Two OS processes (see policy_ipc.py for why -- short version: CPython's GIL means a
slow onnxruntime inference call in the SAME process as the rt/lowcmd writer can stall
that writer for tens of ms, even with CPU clocks pinned; a separate process has its own
GIL, so inference latency can never block DDS I/O):

I/O process (this one, 50 Hz policy tick):
  1. read wireless controller -> cmd_vel [vx, vy, wz]   (Select = exit -> damping)
  2. read robot state (q, dq, base quat, base gyro)
  3. hand state + cmd_vel to the policy process over shared memory (policy_ipc.py)
  4. read back whatever 15-dim leg/waist target the policy process last produced and
     stash it for the writer thread below

Policy process (homie_policy_process.py, decoupled, its own GIL):
  HOMIE policy -> 15 leg+waist targets, picking stand.onnx / walk.onnx on ||cmd||

`rt/lowcmd` writer (decoupled, default 250 Hz -- see G1RealWorldEnv.start_writer):
  5. re-read the arm source fresh every write (xr_teleoperate publishes rt/arm_sdk at
     ~250 Hz; sampling it once per 50 Hz policy tick would throttle teleop arm tracking
     to 1/5 of its native rate) and combine with the latest stashed leg/waist target
  6. send the absolute 29-dim q_target with the combined PD gains (HOMIE for legs/
     waist, forwarded-from-teleop for arms when arm_source is "arm_sdk")

Usage:
  python run_homie.py --net eth0 \
      --stand ../ckpts/stand.onnx --walk ../ckpts/walk.onnx
"""
import argparse
import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from homie_utils import load_config, get_gravity_orientation  # noqa: E402
from command_source import make_command_source   # noqa: E402
from arm_target_source import make_arm_source    # noqa: E402
from robot_control.config import Config          # noqa: E402
from robot_control.g1_wrapper import G1RealWorldEnv  # noqa: E402
from policy_ipc import PolicyIPC                 # noqa: E402
import homie_policy_process                      # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Run HOMIE stand/walk policy on real G1 29DoF")
    p.add_argument("--net", required=True, help="network interface to the robot (e.g. eth0)")
    p.add_argument("--config", default=os.path.join(_HERE, "robot_control/configs/g1_homie.yaml"),
                   help="hardware/deploy config (gains, joint map, cmd limits)")
    p.add_argument("--policy-config", default=os.path.join(_HERE, "config/g1_homie_v2.yaml"),
                   help="HOMIE policy spec (obs/action scales, defaults)")
    p.add_argument("--stand", default=os.path.join(_HERE, "../ckpts/stand.onnx"),
                   help="stand policy ONNX (||cmd|| < threshold)")
    p.add_argument("--walk", default=os.path.join(_HERE, "../ckpts/walk.onnx"),
                   help="walk policy ONNX (||cmd|| >= threshold)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="onnxruntime execution device")
    p.add_argument("--onnx-threads", type=int, default=1,
                   help="onnxruntime intra/inter-op thread pool size. Keep at 1 for this "
                        "small MLP -- letting it spawn a thread per core adds scheduling "
                        "overhead for no benefit on a model this size")
    p.add_argument("--ramp-time", type=float, default=1.0,
                   help="seconds to blend from default pose into policy output at startup")
    p.add_argument("--writer-rate-hz", type=float, default=250.0,
                   help="rt/lowcmd publish rate (Hz), decoupled from the 50 Hz policy tick "
                        "so the arm-teleop portion of the command resamples close to "
                        "xr_teleoperate's own ~250 Hz rt/arm_sdk publish rate")
    p.add_argument("--dry-run", action="store_true",
                   help="connect + read state + run policy at 50 Hz, but NEVER command "
                        "motion (robot held in zero-stiffness damping). Use this first.")
    return p.parse_args()


def start_policy_process(policy_config, stand_path, walk_path, onnx_threads, control_dt):
    """Spawn (not fork) the policy process -- 'spawn' starts a clean interpreter with no
    inherited threads/file-descriptors, so it's safe regardless of whether DDS/threads in
    this process have already been set up. Blocks until the child confirms its ONNX
    models are loaded, mirroring the old synchronous HomiePolicyV2(...) construction."""
    ctx = mp.get_context("spawn")
    ipc = PolicyIPC(ctx=ctx)
    proc = ctx.Process(
        target=homie_policy_process.run,
        args=(ipc, policy_config, stand_path, walk_path, onnx_threads, control_dt),
        daemon=True,
    )
    proc.start()
    if not ipc.ready_event.wait(timeout=30.0):
        sys.exit("Error: policy process did not become ready within 30s")
    return ipc, proc


def stop_policy_process(ipc, proc):
    ipc.stop_event.set()
    proc.join(timeout=2.0)
    if proc.is_alive():
        proc.terminate()


def dry_run(env, ipc, commands, arm_source, config, stand_cmd_threshold):
    """Read state + run the policy at the control rate, but command ONLY damping.

    Verifies, safely, everything except motor motion: DDS connectivity, state/IMU
    reads, wireless-controller cmd_vel, stand/walk switching, arm-target plumbing, and
    that the computed 29-dim targets look sane next to the measured pose. The robot is
    held limp (zero kp, small kd) the whole time -- it never tracks the policy output.
    """
    print("\n" + "=" * 60)
    print("DRY RUN: robot held in DAMPING. No policy motion is commanded.")
    print(f"  stand/walk threshold: ||cmd|| = {stand_cmd_threshold} m/s")
    print("  Move the sticks to watch cmd_vel + stand/walk switch.")
    print("  Ctrl+C or Select to stop.")
    print("=" * 60 + "\n")

    control_dt = config.control_dt
    step = 0
    try:
        while True:
            t0 = time.time()
            controller = env.read_controller_input()
            if controller.keys == env.controller_mapping["select"]:
                print("\nSelect pressed -> stopping dry run.")
                break

            cmd = commands.get_command(controller)
            q29, dq29, base_quat, base_ang_vel = env.get_robot_state()
            ipc.write_state(q29, dq29, base_quat, base_ang_vel, cmd)
            q_target15, which = ipc.read_action()
            arm_target = arm_source.get_arm_target()
            arm_kp, arm_kd = arm_source.get_arm_gains()

            # SAFETY: keep the robot limp -- never send policy targets in dry run.
            env.send_damping()

            if step % 25 == 0:  # ~2 Hz console update
                leg_dev = float(np.max(np.abs(q_target15 - config.default_angles[:15])))
                arm_dev = float(np.max(np.abs(arm_target - config.default_angles[15:29])))
                grav = get_gravity_orientation(base_quat)
                kp_str = ",".join(f"{v:.0f}" for v in arm_kp)
                kd_str = ",".join(f"{v:.1f}" for v in arm_kd)
                # Periodic dry-run status logging disabled to keep the console quiet.
                # print(f"[{which:>5}] |cmd|={np.linalg.norm(cmd):.2f} "
                #       f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}]  "
                #       f"grav_z={grav[2]:+.2f}  legΔ={leg_dev:.3f} armΔ={arm_dev:.3f}  "
                #       f"q0={q29[0]:+.2f}  arm_kp=[{kp_str}] arm_kd=[{kd_str}]")
            step += 1

            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt -> stopping dry run.")
    finally:
        env.close()
        close_commands = getattr(commands, "close", None)
        if callable(close_commands):
            close_commands()
        print("Dry run finished (robot in damping).")


def main():
    args = parse_args()
    for path in (args.config, args.policy_config, args.stand, args.walk):
        if not os.path.exists(path):
            sys.exit(f"Error: file not found: {path}")

    config = Config(args.config)
    policy_config = load_config(args.policy_config)
    stand_cmd_threshold = float(policy_config.get("stand_cmd_threshold", 0.2))

    if args.device == "cuda":
        print("[homie] --device cuda is not used by the out-of-process policy yet; "
              "running the policy process on CPU.")

    control_dt = config.control_dt
    ipc, policy_proc = start_policy_process(
        policy_config, args.stand, args.walk, args.onnx_threads, control_dt
    )

    commands = make_command_source(config, net=args.net)
    env = G1RealWorldEnv(net=args.net, config=config)
    arm_source = make_arm_source(config, net=args.net)

    if args.dry_run:
        try:
            return dry_run(env, ipc, commands, arm_source, config, stand_cmd_threshold)
        finally:
            stop_policy_process(ipc, policy_proc)

    print("\n" + "=" * 60)
    print("SAFETY: policy runs on a REAL robot. Keep the e-stop in hand.")
    print(f"  stand/walk threshold: ||cmd|| = {stand_cmd_threshold} m/s")
    print("  START = ramp to default, A = start policy, Select = exit to damping.")
    print("=" * 60 + "\n")

    # Bring-up: ramp to the default pose, then hold until the operator presses A.
    env.move_to_default_pos()
    env.default_pos_state()

    ramp_steps = max(1, int(args.ramp_time / control_dt))
    default_full = config.default_angles.copy()

    # Leg/waist target from the 50 Hz policy tick, resampled by the writer thread below
    # alongside a FRESH arm read every write -- this is what keeps teleop arm tracking
    # close to xr_teleoperate's own ~250 Hz rt/arm_sdk rate instead of being throttled
    # down to the policy's 50 Hz. See G1RealWorldEnv.start_writer().
    leg_lock = threading.Lock()
    leg_state = {
        "q": default_full[:15].copy(),
        "kp": np.asarray(config.kps[:15], dtype=np.float32),
        "kd": np.asarray(config.kds[:15], dtype=np.float32),
        "alpha": 1.0,
    }

    leg_tau = np.zeros(15, dtype=np.float32)  # legs/waist never get feedforward torque

    def get_command():
        with leg_lock:
            leg_q, leg_kp, leg_kd, alpha = (
                leg_state["q"], leg_state["kp"], leg_state["kd"], leg_state["alpha"],
            )
        # Single atomic read so q/kp/kd/tau all come from the same underlying rt/arm_sdk
        # sample -- see arm_target_source.py's get_arm_command() docstring for why that
        # matters more for tau than it does for q/kp/kd.
        arm_target, arm_kp, arm_kd, arm_tau = arm_source.get_arm_command()
        q_target = np.concatenate([leg_q, arm_target]).astype(np.float32)
        kp = np.concatenate([leg_kp, arm_kp]).astype(np.float32)
        kd = np.concatenate([leg_kd, arm_kd]).astype(np.float32)
        if alpha < 1.0:
            q_target = (1.0 - alpha) * default_full + alpha * q_target
            arm_tau = alpha * arm_tau  # ease the feedforward torque in too, not just position
        tau_ff = np.concatenate([leg_tau, arm_tau]).astype(np.float32)
        return q_target, kp, kd, tau_ff

    env.start_writer(get_command, rate_hz=args.writer_rate_hz)

    step = 0
    last_print = 0.0
    tick_ms_sum = tick_ms_max = 0.0
    tick_n = 0

    print(f"Starting HOMIE policy loop ({os.cpu_count()} CPUs visible). Press [Select] to exit.")
    print(f"  rt/lowcmd writer: {args.writer_rate_hz:.0f} Hz (decoupled from the 50 Hz policy tick)")
    print("  policy inference: separate process (see [policy-proc] log lines)")
    try:
        while True:
            t0 = time.time()

            controller = env.read_controller_input()
            if controller.keys == env.controller_mapping["select"]:
                print("Select pressed -> exiting to damping.")
                break

            cmd = commands.get_command(controller)
            q29, dq29, base_quat, base_ang_vel = env.get_robot_state()
            ipc.write_state(q29, dq29, base_quat, base_ang_vel, cmd)
            q_target15, which = ipc.read_action()

            # Ease in: blend default -> policy target over ramp_steps to avoid a jump.
            alpha = min(1.0, (step + 1) / ramp_steps)
            step += 1
            with leg_lock:
                leg_state["q"] = q_target15
                leg_state["alpha"] = alpha

            tick_ms = (time.time() - t0) * 1000.0
            tick_ms_sum += tick_ms
            tick_ms_max = max(tick_ms_max, tick_ms)
            tick_n += 1

            if t0 - last_print > 0.5:
                arm_kp, arm_kd = arm_source.get_arm_gains()
                kp_str = ",".join(f"{v:.0f}" for v in arm_kp)
                kd_str = ",".join(f"{v:.1f}" for v in arm_kd)
                # Periodic control-loop status logging disabled to keep the console quiet.
                # print(f"[{which:>5}] |cmd|={np.linalg.norm(cmd):.2f} "
                #       f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}]  "
                #       f"arm_kp=[{kp_str}] arm_kd=[{kd_str}]  "
                #       f"tick_ms avg={tick_ms_sum / tick_n:.2f} max={tick_ms_max:.2f}")
                last_print = t0
                tick_ms_sum = tick_ms_max = 0.0
                tick_n = 0

            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt -> exiting to damping.")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nError in control loop: {exc}")
    finally:
        env.close()  # stops the writer thread, then sends a final damping command
        stop_policy_process(ipc, policy_proc)
        close_commands = getattr(commands, "close", None)
        if callable(close_commands):
            close_commands()
        print("\nHOMIE controller stopped (robot in damping).")


if __name__ == "__main__":
    main()
