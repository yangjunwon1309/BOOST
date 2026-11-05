#!/usr/bin/env python3
"""HOMIE-base DreamWaQ deployment for the Unitree G1 29-DoF.

The DDS-owning parent and 250-Hz LowCmd writer remain isolated from inference,
but the policy IPC is sequence-coupled: the parent accepts only the action
computed from its current state request.  ``previous_action`` remains the
last parent-accepted actor output, exactly as in DreamWaQ training/sim2sim.
The writer's published target is retained only as transport/safety telemetry.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from homie_utils import get_gravity_orientation  # noqa: E402
from command_source import (  # noqa: E402
    check_command_damping, configure_command_limits, make_command_source, set_command_limits,
)
from locomotion_mode import LocomotionModeSwitch, TOGGLE_BUTTON_MASKS  # noqa: E402
from controller_handoff import ControllerHandoff, UnitreeMotionService  # noqa: E402
from robot_control.config import Config  # noqa: E402
from dreamwaq_actor_history import AcceptedActorActionHistory  # noqa: E402
from dreamwaq_action_feedback import PublishedActionTracker  # noqa: E402
from dreamwaq_policy_ipc import DreamWaQPolicyIPC, IPCLockTimeout  # noqa: E402
from runtime_reporter import RuntimeReporter  # noqa: E402
import dreamwaq_policy_process  # noqa: E402
from cpu_affinity import format_cpu_set  # noqa: E402
from dreamwaq_interface.g1_29dof.contract import DreamWaQ29Contract, JointOrderMap  # noqa: E402
from waist_override import WAIST_YAW_SDK_INDEX, WaistYawOverride  # noqa: E402
from policy_switch import SoloButtonRelease, Stillness, incompatible_fields  # noqa: E402
from remote_mode import DEFAULT_PORT as REMOTE_DEFAULT_PORT, RemoteModeServer  # noqa: E402


def _gain_scale(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gain scale must be in [0, 1]") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("gain scale must be in [0, 1]")
    return result


def _nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative finite number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative finite number")
    return result


def _positive_float(value: str) -> float:
    result = _nonnegative_float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _cpu_set(value: str) -> tuple[int, ...]:
    """Parse CPU lists such as ``3-5`` or ``3,4,5`` for argparse."""

    cpus: set[int] = set()
    try:
        for part in value.split(","):
            part = part.strip()
            if not part:
                raise ValueError
            if "-" in part:
                start, end = (int(item) for item in part.split("-", 1))
                if start < 0 or end < start:
                    raise ValueError
                cpus.update(range(start, end + 1))
            else:
                cpu = int(part)
                if cpu < 0:
                    raise ValueError
                cpus.add(cpu)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "CPU set must look like '3-5' or '3,4,5'"
        ) from exc
    if not cpus:
        raise argparse.ArgumentTypeError("CPU set cannot be empty")
    return tuple(sorted(cpus))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DreamWaQ on the HOMIE G1 29-DoF deployment path"
    )
    parser.add_argument("--net", required=True, help="G1 network interface, e.g. eth0")
    parser.add_argument("--expected-mode-machine", type=int, default=5)
    parser.add_argument(
        "--config",
        default=os.path.join(_HERE, "robot_control/configs/g1_homie.yaml"),
        help="HOMIE hardware config containing joint map, gains, and default pose",
    )
    parser.add_argument(
        "--export",
        default=os.path.join(_HERE, "../checkpoints/new3_4_widerange_quality_390000/exported"),
        help="DreamWaQ exported manifest/policy/CENet directory",
    )
    parser.add_argument(
        "--locomotion-mode", choices=("standing", "walking"), default="standing",
        help="initial NEW3 mode; standing gates every velocity source to zero",
    )
    parser.add_argument(
        "--mode-toggle-button", choices=tuple(TOGGLE_BUTTON_MASKS), default="X",
        help="wireless button that toggles NEW3 standing/walking once per press",
    )
    parser.add_argument(
        "--controller-switch", action="store_true",
        help="L1+Y hands control between DreamWaQ Develop mode and Unitree regular walking",
    )
    parser.add_argument(
        "--initial-controller", choices=("dreamwaq", "unitree"), default="dreamwaq",
        help="controller to start with when --controller-switch is enabled",
    )
    parser.add_argument(
        "--unitree-mode", default="ai",
        help="MotionSwitcher native controller name/alias (default: ai)",
    )
    parser.add_argument(
        "--controller-timeout", type=_positive_float, default=8.0,
        help="seconds to wait for each controller/FSM transition",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--python-switch-interval-ms", type=_positive_float, default=None,
        help="Python thread handoff interval in ms (autonomy launcher uses 1 ms)",
    )
    parser.add_argument(
        "--disable-cyclic-gc", action="store_true",
        help="disable cyclic GC in this deployment process to avoid collection pauses",
    )
    parser.add_argument(
        "--gain-scale", "--motor-gain-scale", dest="gain_scale",
        type=_gain_scale, default=1.0,
        help="scale HOMIE Kp/Kd and arm feed-forward torque in [0, 1]",
    )
    parser.add_argument(
        "--policy-cpu",
        type=int,
        default=None,
        help=(
            "policy-process CPU pin; by default use the second-highest available "
            "CPU and keep the DDS-owning parent off that CPU"
        ),
    )
    parser.add_argument(
        "--policy-prewake-us",
        type=_nonnegative_float,
        default=8000.0,
        help=(
            "busy-poll the dedicated policy CPU for this long before each "
            "expected 50-Hz request to avoid schedutil/DVFS wake-up delay"
        ),
    )
    parser.add_argument(
        "--control-cpu",
        type=int,
        default=None,
        help=(
            "50-Hz parent control-thread CPU; defaults to the highest available "
            "CPU other than the policy CPU (CPU 7 on the 8-core G1)"
        ),
    )
    parser.add_argument(
        "--camera-cpus",
        type=_cpu_set,
        default=(0, 1, 2),
        help="reserved onboard teleimager CPUs; defaults to the server's 0-2 mask",
    )
    parser.add_argument(
        "--io-cpus",
        type=_cpu_set,
        default=(3, 4, 5),
        help="allowed CPU mask for DDS-owning process threads (default: 3-5)",
    )
    parser.add_argument(
        "--lowstate-reader-cpu",
        type=int,
        default=3,
        help="pin the LowState sampling thread here (default: CPU 3)",
    )
    parser.add_argument(
        "--arm-reader-cpu",
        type=int,
        default=4,
        help="pin the rt/arm_sdk reader thread here (default: CPU 4)",
    )
    parser.add_argument(
        "--writer-cpu",
        type=int,
        default=5,
        help="pin the 250-Hz rt/lowcmd writer here (default: CPU 5)",
    )
    parser.add_argument("--ramp-time", type=_positive_float, default=1.0)
    parser.add_argument("--writer-rate-hz", type=float, default=250.0)
    parser.add_argument(
        "--arm-tau-mode",
        choices=("off", "forward"),
        default="forward",
        help="forward fresh PATH arm tau_ff, or force arm tau_ff to zero",
    )
    parser.add_argument(
        "--arm-smoothing-tau",
        type=_nonnegative_float,
        default=None,
        metavar="SECONDS",
        help=(
            "override the arm_sdk position-target smoothing time constant for this "
            "run; 0 disables smoothing, default uses the hardware config"
        ),
    )
    parser.add_argument(
        "--max-vx",
        type=_nonnegative_float,
        default=None,
        help="maximum forward speed from the wireless controller, m/s (default: config)",
    )
    parser.add_argument(
        "--max-vy",
        type=_nonnegative_float,
        default=None,
        help="maximum lateral speed from the wireless controller, m/s (default: config)",
    )
    parser.add_argument(
        "--max-yaw",
        type=_nonnegative_float,
        default=None,
        help="maximum yaw speed from the wireless controller, rad/s (default: config)",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="read state and run the production-load probe without LowCmd publishing",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--status-every", type=float, default=1.0)
    parser.add_argument(
        "--runtime-log",
        default="/tmp/dreamwaq-g1-29dof-last.log",
        help=(
            "persistent real-motion diagnostics; control threads never wait "
            "for terminal output"
        ),
    )
    parser.add_argument(
        "--shadow-reader-hz",
        type=_positive_float,
        default=250.0,
        help="read-only LowState cache rate used by --shadow",
    )
    parser.add_argument(
        "--shadow-expected-lowstate-hz",
        type=_positive_float,
        default=1000.0,
        help="expected Unitree hardware tick rate used to detect DDS backlog",
    )
    parser.add_argument(
        "--shadow-warmup-s",
        type=_nonnegative_float,
        default=2.0,
        help="unscored warmup before shadow timing collection",
    )
    parser.add_argument(
        "--shadow-report",
        default=None,
        help="shadow summary JSON path; raw samples use the same path with .csv",
    )
    parser.add_argument(
        "--shadow-max-state-age-ms",
        type=_positive_float,
        default=10.0,
        help="maximum allowed p99 LowState age for a no-delay shadow verdict",
    )
    parser.add_argument(
        "--shadow-max-tick-jitter-ms",
        type=_positive_float,
        default=5.0,
        help="p99 policy tick lateness threshold reported as a warning",
    )
    parser.add_argument(
        "--shadow-max-writer-handoff-ms",
        type=_positive_float,
        default=8.0,
        help="maximum p99 response-to-next-250Hz-slot delay in shadow",
    )
    parser.add_argument(
        "--shadow-production-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reproduce the read-only arm subscription and 250-Hz command-composition "
            "callback during shadow (default: enabled; never creates LowCmd)"
        ),
    )
    parser.add_argument(
        "--policy-timeout-ms",
        type=_positive_float,
        default=15.0,
        help="deadline for the action matching the current state (must be below control_dt)",
    )
    parser.add_argument(
        "--max-policy-timeouts",
        type=_positive_int,
        default=3,
        help=(
            "consecutive missed action deadlines used only for the stale-action "
            "diagnostic window; deadline misses no longer terminate the policy"
        ),
    )
    parser.add_argument(
        "--disable-lower-joint-target-clip",
        action="store_true",
        help=(
            "EXPERIMENTAL: allow the 12 lower-body policy position targets outside "
            "the canonical joint limits; the per-joint torque envelope remains active"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read state and run DreamWaQ while sending damping only",
    )
    switch = parser.add_argument_group(
        "policy switch",
        "load a second export at startup and hand the standing robot to it and "
        "back, without damping; options not listed here are shared by both exports",
    )
    switch.add_argument(
        "--switch-export", default=None,
        help="DreamWaQ export to switch to; enables the switch",
    )
    switch.add_argument(
        "--switch-button", choices=tuple(TOGGLE_BUTTON_MASKS), default="Y",
        help="wireless button that switches either way when pressed and released on its own",
    )
    for axis, unit in (("vx", "m/s"), ("vy", "m/s"), ("yaw", "rad/s")):
        switch.add_argument(
            "--switch-max-%s" % axis, type=_nonnegative_float, default=None,
            help="--max-%s after the switch, %s (default: config)" % (axis, unit),
        )
    switch.add_argument(
        "--switch-arm-smoothing-tau", type=_nonnegative_float, default=None, metavar="SECONDS",
        help="--arm-smoothing-tau after the switch (default: config)",
    )
    switch.add_argument(
        "--switch-disable-lower-joint-target-clip", action="store_true",
        help="--disable-lower-joint-target-clip after the switch",
    )
    # The way back starts from an export with no standing mode, so it checks
    # the measured legs and base rather than the command (policy_switch.Stillness).
    switch.add_argument(
        "--switch-back-max-leg-dq", type=_nonnegative_float, default=0.5, metavar="RAD_S",
        help="switch back only while every leg joint is slower than this (default: 0.5)",
    )
    switch.add_argument(
        "--switch-back-max-gyro", type=_nonnegative_float, default=0.3, metavar="RAD_S",
        help="... and the base turns slower than this about every axis (default: 0.3)",
    )
    switch.add_argument(
        "--switch-back-still-time", type=_nonnegative_float, default=0.5, metavar="SECONDS",
        help="... and both have held for this long (default: 0.5)",
    )
    switch.add_argument(
        "--switch-back-arm-time", type=_nonnegative_float, default=3.0, metavar="SECONDS",
        help="after switching back, ramp the arms to the default pose over this (default: 3.0)",
    )
    remote = parser.add_argument_group(
        "remote requests",
        "let one host on the robot LAN ask for walking, standing, hold_external, "
        "release_external and the switch over UDP, and receive the runtime state "
        "(remote_mode.py); the wireless remote keeps every button",
    )
    remote.add_argument(
        "--remote-bind", default=None, metavar="HOST",
        help="G1 address to listen on, e.g. 192.168.123.164; enables remote requests",
    )
    remote.add_argument("--remote-port", type=int, default=REMOTE_DEFAULT_PORT)
    remote.add_argument(
        "--remote-allowed-host", default=None, metavar="HOST",
        help="the only sender accepted, e.g. Thor at 192.168.123.2",
    )
    args = parser.parse_args(argv)
    if args.remote_bind is None:
        if args.remote_allowed_host is not None:
            parser.error("--remote-allowed-host needs --remote-bind")
    else:
        if args.shadow or args.dry_run:
            parser.error("--remote-bind requires real control; remove --shadow/--dry-run")
        if not (args.remote_allowed_host or "").strip():
            parser.error("--remote-bind needs --remote-allowed-host")
        if not 1 <= args.remote_port <= 65535:
            parser.error("--remote-port must be in [1, 65535]")
    if args.controller_switch and (args.shadow or args.dry_run):
        parser.error("--controller-switch requires real control; remove --shadow/--dry-run")
    if args.switch_export is None:
        if any(value is not None for value in (
            args.switch_max_vx, args.switch_max_vy, args.switch_max_yaw,
            args.switch_arm_smoothing_tau,
        )) or args.switch_disable_lower_joint_target_clip:
            parser.error("--switch-* options need --switch-export")
    else:
        if args.shadow or args.dry_run:
            parser.error("--switch-export requires real control; remove --shadow/--dry-run")
        if args.switch_button == args.mode_toggle_button:
            parser.error("--switch-button must differ from --mode-toggle-button")
    if args.initial_controller == "unitree" and not args.controller_switch:
        parser.error("--initial-controller unitree requires --controller-switch")
    if not args.unitree_mode.strip():
        parser.error("--unitree-mode cannot be empty")
    return args


def resolve_policy_cpu(policy_cpu=None, *, device="cpu", available_cpus=None):
    """Select the G1 policy core while respecting the launch cpuset."""

    if str(device) != "cpu":
        return None if policy_cpu is None else int(policy_cpu)
    available = sorted(
        os.sched_getaffinity(0) if available_cpus is None else available_cpus
    )
    if not available:
        return None
    if policy_cpu is None:
        # On the 8-core G1 this selects CPU 6 and leaves CPU 7 to the parent.
        return available[-2] if len(available) >= 2 else None
    selected = int(policy_cpu)
    if selected not in available:
        raise ValueError(
            "policy CPU %d is outside the available affinity %s"
            % (selected, available)
        )
    return selected


def isolate_parent_from_policy_cpu(policy_cpu, io_cpus=None):
    """Constrain future DDS/middleware threads before their channels are created."""

    available = set(os.sched_getaffinity(0))
    parent_affinity = set(io_cpus) if io_cpus is not None else available
    if policy_cpu is not None:
        parent_affinity.discard(int(policy_cpu))
    if not parent_affinity or not parent_affinity <= available:
        raise ValueError(
            "DDS I/O CPU mask %s is not available from launch affinity %s"
            % (format_cpu_set(parent_affinity), format_cpu_set(available))
        )
    os.sched_setaffinity(0, parent_affinity)
    return sorted(os.sched_getaffinity(0))


def resolve_control_cpu(control_cpu=None, *, policy_cpu=None, available_cpus=None):
    """Select the G1's final big core for the 50-Hz parent loop."""

    available = sorted(
        os.sched_getaffinity(0) if available_cpus is None else available_cpus
    )
    candidates = [cpu for cpu in available if cpu != policy_cpu]
    if control_cpu is None:
        return candidates[-1] if candidates else None
    selected = int(control_cpu)
    if selected not in candidates:
        raise ValueError(
            "control CPU %d is outside the available non-policy affinity %s"
            % (selected, candidates)
        )
    return selected


def resolve_cpu_topology(args, available_cpus):
    """Validate that cameras, DDS I/O, policy, and control use separate CPUs."""

    available = set(available_cpus)
    camera_cpus = set(args.camera_cpus)
    io_cpus = set(args.io_cpus)
    io_threads = {
        "LowState reader": int(args.lowstate_reader_cpu),
        "arm_sdk reader": int(args.arm_reader_cpu),
        "LowCmd writer": int(args.writer_cpu),
    }
    fixed_roles = {"image server": camera_cpus}
    if args.policy_cpu is not None:
        fixed_roles["policy"] = {int(args.policy_cpu)}
    if args.control_cpu is not None:
        fixed_roles["control loop"] = {int(args.control_cpu)}
    role_names = list(fixed_roles)
    for index, role in enumerate(role_names):
        for other_role in role_names[index + 1 :]:
            overlap = fixed_roles[role] & fixed_roles[other_role]
            if overlap:
                raise ValueError(
                    "%s CPUs %s overlap %s CPUs"
                    % (role, format_cpu_set(overlap), other_role)
                )
    missing_io = io_cpus - available
    if missing_io:
        raise ValueError(
            "DDS I/O CPUs %s are outside the launch affinity %s"
            % (format_cpu_set(missing_io), format_cpu_set(available))
        )
    overlaps = io_cpus & camera_cpus
    if overlaps:
        raise ValueError(
            "DDS I/O CPUs %s overlap the reserved image-server CPUs %s"
            % (format_cpu_set(overlaps), format_cpu_set(camera_cpus))
        )
    reserved = set().union(*fixed_roles.values()) if fixed_roles else set()
    overlaps = io_cpus & reserved
    if overlaps:
        raise ValueError(
            "DDS I/O CPUs %s overlap camera, policy, or control CPUs"
            % format_cpu_set(overlaps)
        )
    thread_cpus = list(io_threads.values())
    if len(set(thread_cpus)) != len(thread_cpus):
        raise ValueError("LowState reader, arm_sdk reader, and LowCmd writer need distinct CPUs")
    for role, cpu in io_threads.items():
        if cpu not in io_cpus:
            raise ValueError(
                "%s CPU %d must be in the DDS I/O mask %s"
                % (role, cpu, format_cpu_set(io_cpus))
            )
    args.io_cpus = tuple(sorted(io_cpus))


def pin_control_thread(control_cpu):
    """Pin only the calling thread; existing I/O threads keep their affinity."""

    if control_cpu is not None:
        os.sched_setaffinity(0, {int(control_cpu)})
    return sorted(os.sched_getaffinity(0))


def start_policy_process(
    export_path,
    device,
    control_dt,
    policy_cpu=None,
    policy_prewake_us=8000.0,
    clip_lower_joint_targets=True,
    action_dim=12,
    io_cpus=None,
):
    """Spawn a clean policy process; it imports no DDS or robot controller code."""

    effective_prewake_us = (
        float(policy_prewake_us) if policy_cpu is not None else 0.0
    )
    context = mp.get_context("spawn")
    ipc = DreamWaQPolicyIPC(ctx=context, action_dim=action_dim)
    process = context.Process(
        target=dreamwaq_policy_process.run,
        args=(
            ipc,
            export_path,
            device,
            control_dt,
            policy_cpu,
            effective_prewake_us,
            clip_lower_joint_targets,
        ),
        daemon=True,
    )
    process.start()
    if not ipc.ready_event.wait(timeout=30.0):
        sys.exit("Error: DreamWaQ policy process did not become ready within 30s")
    parent_affinity = isolate_parent_from_policy_cpu(policy_cpu, io_cpus)
    print(
        "[policy-proc] affinity policy=%s dds_io=%s prewake_us=%.0f"
        % (
            "unbound" if policy_cpu is None else policy_cpu,
            format_cpu_set(parent_affinity),
            effective_prewake_us,
        ),
        flush=True,
    )
    return ipc, process


def stop_policy_process(ipc, process) -> None:
    ipc.request_stop()
    process.join(timeout=2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)


def dry_run(env, ipc, commands, arm_source, config, args) -> None:
    """Exercise state, joystick, arm, and policy paths while keeping damping."""

    print("\n" + "=" * 60)
    print("DRY RUN: robot held in DAMPING. No policy motion is commanded.")
    print("Move the sticks to exercise DreamWaQ command input.")
    print("Ctrl+C or Select to stop.")
    print("=" * 60 + "\n")

    control_affinity = pin_control_thread(getattr(args, "control_cpu", None))
    print("[G1-29][runtime] control affinity=%s" % control_affinity, flush=True)
    control_dt = float(config.control_dt)
    timeout_s = float(args.policy_timeout_ms) / 1000.0
    sequence = 0
    consecutive_timeouts = 0
    actor_history = AcceptedActorActionHistory(action_dim=ipc.action_dim)
    mode_switch = LocomotionModeSwitch(
        enabled=args.explicit_locomotion_mode, initial_mode=args.locomotion_mode,
        button=args.mode_toggle_button, initial_keys=env.read_controller_input().keys,
    )
    try:
        while True:
            started = time.monotonic()
            if env.reader_fault is not None:
                raise RuntimeError("LowState reader faulted: %s" % env.reader_fault)
            controller = env.read_controller_input()
            if controller.keys & env.controller_mapping["select"]:
                break
            command, mode_value, changed = mode_switch.update(
                controller.keys, commands.get_command(controller)
            )
            if changed:
                print("[G1-29][dry-run] locomotion_mode=%s" % mode_switch.name, flush=True)
            q29, dq29, quat, gyro = env.get_robot_state()
            state_ns = time.monotonic_ns()
            previous_actor = actor_history.snapshot()
            try:
                request_published_ns = ipc.publish_state(
                    q29,
                    dq29,
                    quat,
                    gyro,
                    command,
                    sequence=sequence,
                    monotonic_ns=state_ns,
                    previous_lower_action=previous_actor.normalized,
                    previous_action_sequence=previous_actor.sequence,
                    locomotion_mode=mode_value,
                )
            except IPCLockTimeout:
                # The policy process held the state lock past the deadline
                # (scheduling delay). Count this tick as a miss and keep going.
                request_published_ns = None
            response = (
                None
                if request_published_ns is None
                else ipc.wait_for_action(sequence, timeout_s)
            )
            current_sequence = sequence
            sequence += 1
            if response is None:
                consecutive_timeouts += 1
                # Communication delay is tolerated: keep running instead of exiting.
                q_target15 = np.asarray(config.default_angles[:15], dtype=np.float32)
            else:
                consecutive_timeouts = 0
                if (
                    response.source_state_monotonic_ns != state_ns
                    or response.source_state_published_ns != request_published_ns
                ):
                    raise RuntimeError("dry-run policy response does not match its request")
                q_target15 = response.q_target15
                actor_history.accept(current_sequence, response.normalized_lower_action)
            arm_source.get_arm_command()
            env.send_damping()
            _ = float(np.max(np.abs(q_target15 - config.default_angles[:15])))
            _ = get_gravity_orientation(quat)
            remaining = control_dt - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    if args.python_switch_interval_ms is not None:
        sys.setswitchinterval(float(args.python_switch_interval_ms) / 1000.0)
        print("[threads] Python switch interval=%.3f ms" %
              (sys.getswitchinterval() * 1000.0), flush=True)
    if args.disable_cyclic_gc:
        gc.disable()
        print("[threads] cyclic GC disabled; reference counting remains active", flush=True)
    # Keep offline export validation/import usable on a workstation without
    # unitree_sdk2py. The DDS bindings are loaded only when real/dry-run I/O
    # is actually requested on the G1.
    for path in (args.config, args.export, args.switch_export):
        if path is not None and not os.path.exists(path):
            raise SystemExit("Error: file not found: %s" % path)

    config = Config(args.config)
    # The switch target starts from the hardware config, exactly as a separate
    # run with only its own options would, not from this run's overrides.
    config_max_cmd = np.asarray(config.max_cmd, dtype=np.float32).copy()
    config_arm_smoothing_tau = float(config.arm_smoothing_tau)
    if args.arm_smoothing_tau is not None:
        # This is deliberately an in-memory override.  Experiments can choose
        # smoothing without mutating the shared hardware YAML.
        config.arm_smoothing_tau = float(args.arm_smoothing_tau)
    control_dt = float(config.control_dt)
    policy_timeout_s = float(args.policy_timeout_ms) / 1000.0
    if policy_timeout_s >= control_dt:
        raise SystemExit(
            "Error: --policy-timeout-ms must be below the %.3f ms policy period"
            % (control_dt * 1000.0)
        )
    contract = DreamWaQ29Contract.load(args.export)
    args.explicit_locomotion_mode = contract.explicit_locomotion_mode
    if contract.explicit_locomotion_mode:
        print(
            "[G1-29][mode] initial=%s; %s toggles standing/walking; standing command=(0,0,0)"
            % (args.locomotion_mode, args.mode_toggle_button), flush=True,
        )
    switch_contract = None
    if args.switch_export is not None:
        switch_contract = DreamWaQ29Contract.load(args.switch_export)
        mismatched = incompatible_fields(contract, switch_contract)
        if mismatched:
            raise SystemExit(
                "Error: --switch-export differs from --export in %s; the writer, ramps "
                "and gains are fixed at startup, so these exports cannot be switched live"
                % ", ".join(mismatched)
            )
    launch_cpus = sorted(os.sched_getaffinity(0))
    args.policy_cpu = resolve_policy_cpu(
        args.policy_cpu, device=args.device, available_cpus=launch_cpus
    )
    args.control_cpu = resolve_control_cpu(
        args.control_cpu,
        policy_cpu=args.policy_cpu,
        available_cpus=launch_cpus,
    )
    resolve_cpu_topology(args, launch_cpus)
    print(
        "[cpu] image=%s dds_io=%s lowstate_reader=%d arm_sdk_reader=%d "
        "lowcmd_writer=%d policy=%s control=%s"
        % (
            format_cpu_set(args.camera_cpus),
            format_cpu_set(args.io_cpus),
            args.lowstate_reader_cpu,
            args.arm_reader_cpu,
            args.writer_cpu,
            "unbound" if args.policy_cpu is None else args.policy_cpu,
            "unbound" if args.control_cpu is None else args.control_cpu,
        ),
        flush=True,
    )
    command_overrides = (args.max_vx, args.max_vy, args.max_yaw)
    configure_command_limits(config, contract.command_lower, contract.command_upper,
                             command_overrides)
    gain_scale = float(args.gain_scale)
    clip_lower_joint_targets = not bool(args.disable_lower_joint_target_clip)
    # Keep the arm source's nominal fallback gains separate: the writer applies
    # gain_scale exactly once to both fallback and live rt/arm_sdk gains.
    config._arm_source_kps = np.asarray(config.kps[15:29], dtype=np.float32).copy()
    config._arm_source_kds = np.asarray(config.kds[15:29], dtype=np.float32).copy()
    config.kps = (np.asarray(config.kps, dtype=np.float32) * gain_scale).tolist()
    config.kds = (np.asarray(config.kds, dtype=np.float32) * gain_scale).tolist()
    print(
        "[G1-29][real] gain_scale=%.3f lower_kp_max=%.3f lower_kd_max=%.3f "
        "max_cmd=(%.3f, %.3f, %.3f) min_cmd=(%.3f, %.3f, %.3f) "
        "lower_joint_target_clip=%s arm_smoothing_s=%.3f"
        % (
            gain_scale,
            max(config.kps[:15]),
            max(config.kds[:15]),
            *config.max_cmd,
            *config.min_cmd,
            "enabled" if clip_lower_joint_targets else "DISABLED",
            config.arm_smoothing_tau,
        ),
        flush=True,
    )
    if not clip_lower_joint_targets:
        print(
            "[G1-29][real] WARNING: lower-body joint target hard-limit clipping is "
            "DISABLED; torque limiting does not prevent sustained force against a joint stop",
            flush=True,
        )
    # Everything a switch changes, per export: [--export, --switch-export].
    exports = [argparse.Namespace(
        path=args.export,
        contract=contract,
        min_cmd=np.asarray(config.min_cmd, dtype=np.float32).copy(),
        max_cmd=np.asarray(config.max_cmd, dtype=np.float32).copy(),
        clip_lower_joint_targets=clip_lower_joint_targets,
        arm_smoothing_tau=float(config.arm_smoothing_tau),
    )]
    if switch_contract is not None:
        switch_limits = argparse.Namespace(max_cmd=config_max_cmd.copy())
        configure_command_limits(
            switch_limits, switch_contract.command_lower, switch_contract.command_upper,
            (args.switch_max_vx, args.switch_max_vy, args.switch_max_yaw),
        )
        switch_clip_lower_joint_targets = not bool(args.switch_disable_lower_joint_target_clip)
        switch_arm_smoothing_tau = (
            config_arm_smoothing_tau if args.switch_arm_smoothing_tau is None
            else float(args.switch_arm_smoothing_tau)
        )
        exports.append(argparse.Namespace(
            path=args.switch_export,
            contract=switch_contract,
            min_cmd=switch_limits.min_cmd,
            max_cmd=switch_limits.max_cmd,
            clip_lower_joint_targets=switch_clip_lower_joint_targets,
            arm_smoothing_tau=switch_arm_smoothing_tau,
        ))
        print(
            "[G1-29][switch] %s alone -> %s and back (stand still first); "
            "max_cmd=(%.3f, %.3f, %.3f) min_cmd=(%.3f, %.3f, %.3f) lower_joint_target_clip=%s "
            "arm_smoothing_s=%.3f locomotion_mode=%s; back only with leg |dq|<=%.2f rad/s and "
            "|gyro|<=%.2f rad/s for %.2f s, then the arms to default over %.1f s"
            % (
                args.switch_button,
                args.switch_export,
                *switch_limits.max_cmd,
                *switch_limits.min_cmd,
                "enabled" if switch_clip_lower_joint_targets else "DISABLED",
                switch_arm_smoothing_tau,
                "standing/walking" if switch_contract.explicit_locomotion_mode else "none",
                args.switch_back_max_leg_dq,
                args.switch_back_max_gyro,
                args.switch_back_still_time,
                args.switch_back_arm_time,
            ),
            flush=True,
        )
    ipc, policy_process = start_policy_process(
        args.export,
        args.device,
        control_dt,
        args.policy_cpu,
        args.policy_prewake_us,
        clip_lower_joint_targets,
        contract.action_dim,
        args.io_cpus,
    )
    policy_workers = [(ipc, policy_process)]
    if switch_contract is not None:
        # Started now so the switch never waits for TorchScript loading, and
        # kept for the way back.  An idle worker sleeps on the policy CPU until
        # its next request.
        switch_worker = start_policy_process(
            args.switch_export,
            args.device,
            control_dt,
            args.policy_cpu,
            args.policy_prewake_us,
            switch_clip_lower_joint_targets,
            switch_contract.action_dim,
            args.io_cpus,
        )
        policy_workers.append(switch_worker)

    if args.shadow:
        from run_shadow import run_shadow

        try:
            return run_shadow(
                args,
                config,
                ipc,
                policy_process,
                torque_limit_sdk=contract.torque_limit_sdk,
            )
        finally:
            stop_policy_process(ipc, policy_process)

    from arm_target_source import make_arm_source
    from robot_control.g1_wrapper import G1RealWorldEnv

    commands = make_command_source(config, net=args.net)
    env = G1RealWorldEnv(
        net=args.net,
        config=config,
        expected_mode_machine=int(args.expected_mode_machine),
        clip_lower_joint_targets=clip_lower_joint_targets,
        torque_limit_sdk=contract.torque_limit_sdk,
        reader_cpu=args.lowstate_reader_cpu,
        writer_cpu=args.writer_cpu,
        lowcmd_enabled=not args.controller_switch,
    )
    arm_source = make_arm_source(
        config, net=args.net, reader_cpu=args.arm_reader_cpu
    )

    if args.dry_run:
        try:
            dry_run(env, ipc, commands, arm_source, config, args)
            return 0
        finally:
            close_commands = getattr(commands, "close", None)
            if callable(close_commands):
                close_commands()
            stop_policy_process(ipc, policy_process)

    published_actions = PublishedActionTracker(contract)

    def record_published(q_target, metadata, monotonic_ns):
        # Looked up per write: a switch or a controller handoff replaces the tracker.
        published_actions.record(q_target, metadata, monotonic_ns)

    reporter = RuntimeReporter(args.runtime_log)
    reporter.emit(
        "real runtime diagnostics started; persistent_log=%s" % reporter.path
    )
    set_arm_reporter = getattr(arm_source, "set_diagnostic_callback", None)
    if callable(set_arm_reporter):
        set_arm_reporter(
            lambda message: reporter.emit(message, level="WARNING")
        )
    runtime_cleanup_done = False
    handoff = None

    def shutdown_real_runtime():
        nonlocal runtime_cleanup_done
        if runtime_cleanup_done:
            return None
        runtime_cleanup_done = True
        close_status = None
        try:
            close_commands = getattr(commands, "close", None)
            if callable(close_commands):
                close_commands()
        except BaseException:
            reporter.emit(
                "velocity command source close failed: %s" % traceback.format_exc(),
                level="ERROR",
            )
        try:
            if handoff is not None:
                handoff.prepare_shutdown()
        except BaseException:
            reporter.emit(
                "controller cleanup failed; LowCmd ownership was not overridden: %s"
                % traceback.format_exc(), level="ERROR",
            )
        try:
            close_status = env.close()
            reporter.emit("G1 environment close status: %s" % close_status)
            if not close_status.get("damping_sent"):
                reporter.emit("final damping was not sent; inspect controller cleanup diagnostics", level="WARNING")
        except BaseException:
            reporter.emit(
                "G1 environment close failed: %s" % traceback.format_exc(),
                level="ERROR",
            )
        for worker_ipc, worker_process in policy_workers:
            try:
                stop_policy_process(worker_ipc, worker_process)
            except BaseException:
                reporter.emit(
                    "policy process shutdown failed: %s" % traceback.format_exc(),
                    level="ERROR",
                )
        return close_status

    print("\n" + "=" * 60)
    print("SAFETY: DreamWaQ policy runs on the real robot. Keep the E-stop in hand.")
    print("START = HOMIE default-pose ramp, A = start DreamWaQ, Select = damping")
    if contract.explicit_locomotion_mode:
        print("%s = toggle standing/walking (initial: %s)" % (args.mode_toggle_button, args.locomotion_mode))
    if args.controller_switch:
        print("L1+Y = DreamWaQ <-> Unitree regular walking (release before pressing again)")
        print("Controller switching uses damping and native restart; posture support drops during transition.")
    if switch_contract is not None:
        print("%s alone (press and release) = switch to %s and back; stand still first"
              % (args.switch_button, args.switch_export))
        print("Switching there holds the bridge at zero; switching back returns the arms "
              "to default once rt/arm_sdk is quiet, and the bridge stays held until Thor "
              "releases it")
    print("=" * 60 + "\n")

    # Use the same operator-selected duration for the physical default-pose
    # ramp and the subsequent policy ease-in. A short hard-coded ramp can leave
    # the policy starting from a large measured-pose error.
    try:
        def cancel_handoff():
            check_command_damping(commands)
            if env.read_controller_input().keys & env.controller_mapping["select"]:
                raise KeyboardInterrupt("Select pressed during controller handoff")
            if env.reader_fault is not None:
                raise RuntimeError("LowState reader faulted: %s" % env.reader_fault)

        if args.controller_switch:
            handoff = ControllerHandoff(
                env,
                UnitreeMotionService.connect(
                    timeout_s=args.controller_timeout, cancel=cancel_handoff,
                    notify=reporter.emit,
                ),
                native_mode=args.unitree_mode, notify=reporter.emit,
                initial_keys=env.read_controller_input().keys,
            )
            if args.initial_controller == "unitree":
                handoff.enter_unitree()
            else:
                handoff.enter_dreamwaq()
        if handoff is None or handoff.controller == "dreamwaq":
            bringup_options = {"cancel": cancel_handoff}
            env.move_to_default_pos(
                total_time=float(args.ramp_time), notify=reporter.emit, **bringup_options
            )
            env.default_pos_state(notify=reporter.emit, **bringup_options)
    except KeyboardInterrupt:
        reporter.emit(
            "KeyboardInterrupt during bring-up; damping shutdown requested",
            level="WARNING",
        )
        shutdown_real_runtime()
        reporter.emit("DreamWaQ bring-up stopped; see G1 environment close status.")
        reporter.close()
        return 130
    except BaseException:
        reporter.emit(
            "DreamWaQ bring-up failed: %s" % traceback.format_exc(),
            level="ERROR",
        )
        shutdown_real_runtime()
        reporter.emit("DreamWaQ bring-up stopped; see G1 environment close status.")
        reporter.close()
        return 1

    ramp_steps = max(1, int(float(args.ramp_time) / float(config.control_dt)))
    default_full = config.default_angles.copy()
    # Thor may turn the waist yaw over rt/arm_sdk (waist_override.py).  Only a
    # policy that leaves the spine to the writer can take that: a 15-action
    # export commands the waist itself and would see its own action history
    # disagree with the joint.
    waist_owned_by_policy = WAIST_YAW_SDK_INDEX in list(
        JointOrderMap().sdk_indices(contract.action_joint_names)
    )
    waist = WaistYawOverride(
        enabled=not waist_owned_by_policy,
        default_rad=float(default_full[WAIST_YAW_SDK_INDEX]),
    )
    get_waist_request = getattr(arm_source, "get_waist_yaw_request", lambda: None)
    waist_ignored_reported = False
    action_stale_timeout_s = (
        float(args.max_policy_timeouts) * control_dt + policy_timeout_s
    )
    leg_lock = threading.Lock()
    leg_state = {
        "q": default_full[:15].copy(),
        "kp": np.asarray(config.kps[:15], dtype=np.float32),
        "kd": np.asarray(config.kds[:15], dtype=np.float32),
        "alpha": 1.0,
        "action_sequence": -1,
        "accepted_monotonic_ns": time.monotonic_ns(),
    }
    leg_tau = np.zeros(15, dtype=np.float32)

    def get_command():
        nonlocal waist_ignored_reported
        check_command_damping(commands, writer=True)
        if env.reader_fault is not None:
            raise RuntimeError("LowState reader faulted: %s" % env.reader_fault)
        with leg_lock:
            leg_q = leg_state["q"].copy()
            leg_kp = leg_state["kp"].copy()
            leg_kd = leg_state["kd"].copy()
            alpha = leg_state["alpha"]
            action_sequence = leg_state["action_sequence"]
        # A stale policy action (communication delay) is tolerated: the writer
        # keeps holding the last accepted leg target instead of faulting.
        arm_q, arm_kp, arm_kd, arm_tau = arm_source.get_arm_command()
        arm_kp = np.asarray(arm_kp, dtype=np.float32) * gain_scale
        arm_kd = np.asarray(arm_kd, dtype=np.float32) * gain_scale
        arm_tau = np.asarray(arm_tau, dtype=np.float32) * gain_scale
        if args.arm_tau_mode == "off":
            arm_tau.fill(0.0)
        q_target = np.concatenate([leg_q, arm_q]).astype(np.float32)
        kp = np.concatenate([leg_kp, arm_kp]).astype(np.float32)
        kd = np.concatenate([leg_kd, arm_kd]).astype(np.float32)
        if alpha < 1.0:
            q_target = (1.0 - alpha) * default_full + alpha * q_target
            arm_tau = alpha * arm_tau
        tau_ff = np.concatenate([leg_tau, arm_tau]).astype(np.float32)
        # After the ease-in blend: the turn has its own rate limit, and the
        # policy observation takes the full offset back out.
        waist_event = waist.step(get_waist_request(), time.monotonic_ns())
        q_target = waist.command(q_target)
        if waist_event is not None:
            reporter.emit("[waist] yaw override %s: target=%.3f rad" % (waist_event, waist.target))
        if waist.ignored_requests and not waist_ignored_reported:
            waist_ignored_reported = True
            reporter.emit(
                "[waist] rt/arm_sdk requests a waist yaw, but this export's policy "
                "commands the waist itself; the request is ignored",
                level="WARNING",
            )
        return q_target, kp, kd, tau_ff, {"action_sequence": action_sequence}

    remote = None
    try:
        if handoff is None or handoff.controller == "dreamwaq":
            env.start_writer(
                get_command,
                rate_hz=float(args.writer_rate_hz),
                on_published=record_published,
                on_fault=lambda message: reporter.emit(message, level="ERROR"),
            )
        control_affinity = pin_control_thread(args.control_cpu)
        if args.remote_bind is not None:
            # Only now: a request taken during START/A bring-up would have
            # nothing to act on, so Thor sees no status until the loop runs.
            remote = RemoteModeServer(
                args.remote_bind, args.remote_port, args.remote_allowed_host,
                notify=reporter.emit,
            )
    except BaseException:
        reporter.emit(
            "DreamWaQ writer startup failed: %s" % traceback.format_exc(),
            level="ERROR",
        )
        shutdown_real_runtime()
        reporter.emit("DreamWaQ startup stopped; see G1 environment close status.")
        reporter.close()
        return 1
    sequence = 0
    policy_epoch = 0
    accepted_steps = 0
    consecutive_timeouts = 0
    total_timeouts = 0
    actor_history = AcceptedActorActionHistory(action_dim=contract.action_dim)
    mode_switch = LocomotionModeSwitch(
        enabled=contract.explicit_locomotion_mode, initial_mode=args.locomotion_mode,
        button=args.mode_toggle_button, initial_keys=env.read_controller_input().keys,
    )
    switch_button = None
    if switch_contract is not None:
        switch_button = SoloButtonRelease(
            TOGGLE_BUTTON_MASKS[args.switch_button], env.read_controller_input().keys
        )
    # Leg target held when the switch fired; blended out over ramp_steps.
    switch_from_q15 = None
    switch_steps = 0
    # Index into exports and policy_workers; 1 means switched.
    active = 0
    stillness = Stillness(
        args.switch_back_max_leg_dq, args.switch_back_max_gyro,
        ticks=round(float(args.switch_back_still_time) / control_dt),
    )
    hold_external = getattr(commands, "hold_external", None)
    release_external = getattr(commands, "release_external", None)
    external_held = getattr(commands, "external_held", lambda: False)
    external_status = getattr(commands, "external_status", None)
    arm_sender_active = getattr(arm_source, "sender_active", lambda: False)

    def export_name(path):
        return os.path.basename(os.path.dirname(os.path.normpath(path)))

    def remote_status(controller_name, requested, effective):
        """What Thor's Nature demo waits on: mode, export, bridge, commands."""
        return {
            "controller": controller_name,
            "export": export_name(exports[active].path),
            "switched": active == 1,
            "blending": switch_from_q15 is not None,
            "mode": mode_switch.name if mode_switch.enabled else None,
            "external_hold": bool(external_held()),
            "bridge": external_status() if callable(external_status) else None,
            "requested": [float(value) for value in requested],
            "command": [float(value) for value in effective],
        }

    def answer(requests, result, detail):
        for request in requests:
            remote.resolve(request, result, detail)
    started = time.monotonic()
    next_deadline = started
    next_status = started + max(0.1, float(args.status_every))
    inference_samples = []
    state_to_action_samples = []
    policy_loop_samples = []
    reporter.emit(
        "Starting HOMIE-base DreamWaQ control; Select exits to damping; "
        "policy_deadline_ms=%.1f max_consecutive_misses=%d "
        "writer_stale_ms=%.1f lowcmd_hz=%.0f arm_smoothing_s=%.3f "
        "arm_tau_mode=%s control_affinity=%s"
        % (
            args.policy_timeout_ms,
            args.max_policy_timeouts,
            action_stale_timeout_s * 1000.0,
            args.writer_rate_hz,
            config.arm_smoothing_tau,
            args.arm_tau_mode,
            control_affinity,
        )
    )
    runtime_exit_code = 0
    try:
        while True:
            check_command_damping(commands)
            started = time.monotonic()
            if env.writer_fault is not None:
                raise RuntimeError("LowCmd writer faulted: %s" % env.writer_fault)
            if env.reader_fault is not None:
                raise RuntimeError("LowState reader faulted: %s" % env.reader_fault)
            if not policy_process.is_alive():
                raise RuntimeError("DreamWaQ policy process exited while LowCmd was active")
            controller = env.read_controller_input()
            if controller.keys & env.controller_mapping["select"]:
                reporter.emit("Select pressed; normal damping shutdown requested")
                break
            remote_requests = remote.pending() if remote is not None else []
            if handoff is not None:
                if handoff.toggle_requested(controller.keys):
                    answer(remote_requests, "unavailable", "the controller is changing hands (L1+Y)")
                    if handoff.controller == "dreamwaq":
                        handoff.enter_unitree()
                    else:
                        handoff.enter_dreamwaq()
                        env.ramp_to_default_pos(
                            total_time=float(args.ramp_time), notify=reporter.emit,
                            cancel=cancel_handoff,
                        )
                        # The ramp put the waist at its default; a turn still
                        # requested over rt/arm_sdk ramps back in from there.
                        waist.reset()
                        # Restart the policy timeline without restarting the
                        # process. The worker resets CENet on the next epoch.
                        policy_epoch += 1
                        actor_history = AcceptedActorActionHistory(contract.action_dim)
                        published_actions = PublishedActionTracker(contract)
                        accepted_steps = 0
                        consecutive_timeouts = 0
                        mode_switch = LocomotionModeSwitch(
                            enabled=contract.explicit_locomotion_mode,
                            initial_mode="standing", button=args.mode_toggle_button,
                            initial_keys=env.read_controller_input().keys,
                        )
                        # The default-pose ramp replaces any switch blend, and a
                        # switch press is only taken from presses seen from here on.
                        switch_from_q15 = None
                        if switch_button is not None:
                            switch_button = SoloButtonRelease(
                                switch_button.mask, env.read_controller_input().keys
                            )
                        stillness.reset()
                        with leg_lock:
                            leg_state["q"] = default_full[:15].copy()
                            leg_state["alpha"] = 0.0
                            leg_state["action_sequence"] = -1
                            leg_state["accepted_monotonic_ns"] = time.monotonic_ns()
                        env.start_writer(
                            get_command, rate_hz=float(args.writer_rate_hz),
                            on_published=record_published,
                            on_fault=lambda message: reporter.emit(message, level="ERROR"),
                        )
                        reporter.emit("[controller] DreamWaQ resumed in standing with fresh policy history")
                    next_deadline = time.monotonic()
                    next_status = next_deadline
                    continue
                if handoff.controller == "unitree":
                    # DDS readers and the policy worker remain alive. Only the
                    # native controller may drive the motors in this state.
                    answer(remote_requests, "unavailable", "Unitree regular walking owns the robot (L1+Y)")
                    if remote is not None:
                        remote.publish_status(remote_status("unitree", np.zeros(3), np.zeros(3)))
                    time.sleep(control_dt)
                    continue
            # Native R1+X and other modified button shortcuts must not also
            # toggle DreamWaQ's standing/walking mode during a handoff session.
            locomotion_keys = controller.keys
            if handoff is not None and controller.keys & 0x0033:
                locomotion_keys = 0
            requested_command = commands.get_command(controller)
            remote_switches = []
            for request in remote_requests:
                if request.action in ("walking", "standing"):
                    if not mode_switch.enabled:
                        remote.resolve(request, "refused", "%s has no standing/walking mode"
                                       % export_name(exports[active].path))
                    elif mode_switch.set_mode(request.action):
                        reporter.emit("locomotion_mode=%s requested by Thor" % mode_switch.name)
                        remote.resolve(request, "applied", "")
                    else:
                        remote.resolve(request, "already", "")
                elif request.action == "hold_external":
                    if not callable(hold_external):
                        remote.resolve(request, "unavailable", "this command source takes no external input")
                        continue
                    latched = hold_external()
                    if latched:
                        reporter.emit(
                            "[remote] external velocity held at zero until Thor releases it; "
                            "the sticks still drive", level="WARNING",
                        )
                    remote.resolve(request, "applied" if latched else "already", "")
                    requested_command = commands.get_command(controller)
                elif request.action == "release_external":
                    if not callable(release_external):
                        remote.resolve(request, "unavailable", "this command source takes no external input")
                        continue
                    # Released, the bridge is gated only by standing, so only
                    # an export that has it, and only while it is in it.
                    if not mode_switch.enabled:
                        remote.resolve(request, "refused", "%s has no standing mode"
                                       % export_name(exports[active].path))
                        continue
                    if mode_switch.mode:
                        remote.resolve(request, "refused", "standing first")
                        continue
                    released = release_external()
                    if released:
                        reporter.emit(
                            "[remote] external velocity released; standing keeps it at zero "
                            "until walking", level="WARNING",
                        )
                    remote.resolve(request, "applied" if released else "already", "")
                    requested_command = commands.get_command(controller)
                elif request.action == "switch":
                    remote_switches.append(request)
            command, mode_value, changed = mode_switch.update(
                locomotion_keys, requested_command
            )
            if changed:
                reporter.emit("locomotion_mode=%s requested by %s" % (mode_switch.name, args.mode_toggle_button))
            switch_by = None
            if switch_button is not None and switch_button.update(controller.keys):
                switch_by = args.switch_button
            thor_switches = []
            for request in remote_switches:
                # Thor only ever asks for the way there; the way back is Y's.
                if switch_contract is None:
                    remote.resolve(request, "unavailable", "no --switch-export")
                elif active == 1:
                    remote.resolve(request, "already", "already running %s" % export_name(args.switch_export))
                elif not external_held():
                    # The second export has no standing mode: without the
                    # hold, a bridge out of HOLD walks the robot after it.
                    remote.resolve(request, "refused", "hold_external first")
                else:
                    switch_by = switch_by or "Thor"
                    thor_switches.append(request)
            if switch_by is not None:
                target = 1 - active
                # The held target below is a standing pose only if the robot
                # is standing: a fresh policy must not take over mid-step.
                # Check the command before NEW3's standing gate too: the
                # second export may have no gate and would act on it at once.
                if np.any(requested_command) or (contract.explicit_locomotion_mode and mode_value):
                    refusal = (
                        "stand still first: %s to standing, sticks centered, no external "
                        "velocity command such as a bridge out of HOLD" % args.mode_toggle_button
                    )
                elif switch_from_q15 is not None:
                    refusal = "the last switch is still blending the legs"
                elif target == 0 and not stillness.still:
                    # No standing mode on this side to go by: a zero command
                    # can still be stepping in place.
                    refusal = (
                        "legs not still: leg |dq| %.2f rad/s, gyro %.2f rad/s; needs <= %.2f "
                        "and <= %.2f for %.2f s"
                        % (stillness.leg_dq, stillness.gyro, stillness.max_leg_dq,
                           stillness.max_gyro, args.switch_back_still_time)
                    )
                elif target == 0 and arm_sender_active():
                    refusal = "rt/arm_sdk still drives the arms; let the mission client exit first"
                elif not policy_workers[target][1].is_alive():
                    refusal = "the %s policy process has exited" % export_name(exports[target].path)
                else:
                    refusal = None
                if refusal is not None:
                    reporter.emit("[switch] %s ignored: %s" % (switch_by, refusal), level="WARNING")
                    answer(thor_switches, "refused", refusal)
                else:
                    settings = exports[target]
                    with leg_lock:
                        switch_from_q15 = leg_state["q"].copy()
                    switch_steps = 0
                    # The worker switched away from stays loaded for the way
                    # back.  A new epoch makes the next one reset CENet and its
                    # history from the measured pose, as at bring-up; its first
                    # previous_action is zero.
                    policy_epoch += 1
                    ipc, policy_process = policy_workers[target]
                    contract = settings.contract
                    active = target
                    switch_button = SoloButtonRelease(
                        TOGGLE_BUTTON_MASKS[args.switch_button], controller.keys
                    )
                    set_command_limits(commands, settings.min_cmd, settings.max_cmd)
                    env.clip_lower_joint_targets = settings.clip_lower_joint_targets
                    if hasattr(arm_source, "smoothing_tau"):
                        arm_source.smoothing_tau = settings.arm_smoothing_tau
                    actor_history = AcceptedActorActionHistory(contract.action_dim)
                    published_actions = PublishedActionTracker(contract)
                    consecutive_timeouts = 0
                    mode_switch = LocomotionModeSwitch(
                        enabled=contract.explicit_locomotion_mode,
                        initial_mode="standing", button=args.mode_toggle_button,
                        initial_keys=controller.keys,
                    )
                    if target == 1:
                        # No standing mode there: hold the bridge whoever
                        # switched, so only the sticks move the robot.  It
                        # stays held on the way back, until Thor releases it.
                        arms = "untouched"
                        if callable(hold_external) and hold_external():
                            reporter.emit(
                                "[switch] external velocity held at zero until Thor releases "
                                "it; the sticks still drive", level="WARNING",
                            )
                    else:
                        arms = "to default over %.1f s" % args.switch_back_arm_time
                        return_arms = getattr(arm_source, "return_to_default", None)
                        if callable(return_arms):
                            return_arms(float(args.switch_back_arm_time))
                    reporter.emit(
                        "[switch] DreamWaQ now runs %s; legs blend from the held target "
                        "over %.2f s, arms %s, max_cmd=(%.3f, %.3f, %.3f) "
                        "min_cmd=(%.3f, %.3f, %.3f)"
                        % (settings.path, ramp_steps * control_dt, arms,
                           *settings.max_cmd, *settings.min_cmd)
                    )
                    answer(thor_switches, "applied", "now running %s" % export_name(settings.path))
                    next_deadline = time.monotonic()
                    next_status = next_deadline
                    continue
            if remote is not None:
                remote.publish_status(remote_status("dreamwaq", requested_command, command))
            q29, dq29, quat, gyro, state_ns = env.get_robot_state_timed()
            stillness.update(dq29[:12], gyro)
            # The policy reads the waist as if it were at its default; see
            # waist_override.py.  A no-op unless Thor has turned it.
            q29, dq29 = waist.observe(q29, dq29)
            previous_actor = actor_history.snapshot()
            published = published_actions.at_or_before(state_ns)
            previous_to_published_linf = float(
                np.max(np.abs(previous_actor.normalized - published.normalized))
            )
            try:
                request_published_ns = ipc.publish_state(
                    q29,
                    dq29,
                    quat,
                    gyro,
                    command,
                    sequence=sequence,
                    monotonic_ns=state_ns,
                    previous_lower_action=previous_actor.normalized,
                    previous_action_sequence=previous_actor.sequence,
                    locomotion_mode=mode_value,
                    policy_epoch=policy_epoch,
                )
            except IPCLockTimeout:
                # The policy process held the state lock past the deadline
                # (scheduling delay). Count this tick as a miss and keep going.
                request_published_ns = None
            response = (
                None
                if request_published_ns is None
                else ipc.wait_for_action(sequence, policy_timeout_s)
            )
            current_sequence = sequence
            sequence += 1
            if response is None:
                consecutive_timeouts += 1
                total_timeouts += 1
                reporter.emit(
                    "policy deadline miss: state_seq=%d deadline_ms=%.1f "
                    "consecutive=%d total=%d"
                    % (
                        current_sequence,
                        args.policy_timeout_ms,
                        consecutive_timeouts,
                        total_timeouts,
                    ),
                    level="WARNING",
                )
                which = "hold"
                inference_ms = float("nan")
                max_action = float("nan")
                state_to_action_ms = float("nan")
                skipped_states = 0
                # Communication delay is tolerated: hold the last accepted
                # action and keep running instead of exiting to damping.
            else:
                if (
                    response.source_state_monotonic_ns != state_ns
                    or response.source_state_published_ns != request_published_ns
                ):
                    raise RuntimeError("policy action does not match its source state request")
                consecutive_timeouts = 0
                inference_ms = response.inference_ms
                max_action = response.max_action
                which = response.mode
                state_to_action_ms = response.state_to_action_ms
                skipped_states = response.skipped_state_count
                if inference_ms > 0.0 and np.isfinite(inference_ms):
                    inference_samples.append(float(inference_ms))
                state_to_action_samples.append(state_to_action_ms)
                accepted_steps += 1
                actor_history.accept(current_sequence, response.normalized_lower_action)
                alpha = min(1.0, float(accepted_steps) / ramp_steps)
                q_target15 = response.q_target15.copy()
                if switch_from_q15 is not None:
                    # Legs only: alpha would also pull the arms to default.
                    switch_steps += 1
                    beta = min(1.0, float(switch_steps) / ramp_steps)
                    q_target15 = ((1.0 - beta) * switch_from_q15 + beta * q_target15).astype(np.float32)
                    if beta >= 1.0:
                        switch_from_q15 = None
                with leg_lock:
                    leg_state["q"] = q_target15
                    leg_state["alpha"] = alpha
                    leg_state["action_sequence"] = current_sequence
                    leg_state["accepted_monotonic_ns"] = time.monotonic_ns()
            work_ms = (time.monotonic() - started) * 1000.0
            policy_loop_samples.append(work_ms)
            now = time.monotonic()
            if now >= next_status:
                state_diag = env.state_reader_diagnostics()
                writer_diag = env.writer_diagnostics()
                command_diagnostics = getattr(commands, "diagnostics", None)
                if callable(command_diagnostics):
                    command_diag = command_diagnostics(controller)
                    age_ms = command_diag["age_ms"]
                    age_text = "never" if age_ms is None else "%.0fms" % age_ms
                    udp_command = command_diag["udp_command"]
                    gate = ("standing" if contract.explicit_locomotion_mode and not mode_value
                            else command_diag["selection"])
                    mode_name = (mode_switch.name if contract.explicit_locomotion_mode
                                 else "n/a")
                    published_lower = published_actions.latest().normalized
                    lowcmd_delta_max = float(np.max(
                        np.abs(published_lower * contract.physical_action_scale)
                    ))
                    lower_dq_max = float(np.max(np.abs(dq29[:15])))
                    reporter.emit(
                        "[dwcontrol] UDP %s:%d latest=(vx=%+.3f vy=%+.3f wz=%+.3f) "
                        "state=%s age=%s | selected=(vx=%+.3f vy=%+.3f wz=%+.3f) "
                        "policy_in=(vx=%+.3f vy=%+.3f wz=%+.3f) mode=%s gate=%s "
                        "request=%s lowcmd_published=%d lowcmd_q_delta_max=%.3frad "
                        "measured_lower_dq_max=%.3frad/s measured_gyro_z=%+.3frad/s"
                        % (commands.bind_host, commands.port,
                           udp_command[0], udp_command[1], udp_command[2],
                           command_diag["state"], age_text,
                           requested_command[0], requested_command[1], requested_command[2],
                           command[0], command[1], command[2], mode_name, gate,
                           "queued" if request_published_ns is not None else "missed",
                           writer_diag["publish_count"], lowcmd_delta_max,
                           lower_dq_max, float(gyro[2]))
                    )
                reporter.emit(
                    "[G1-29][runtime] state_seq=%d policy_prev_seq=%d published_seq=%d "
                    "prev_vs_published_linf=%.5f command=(%.3f, %.3f, %.3f) "
                    "inference_ms=%.3f state_to_action_ms=%.3f max_action=%.3f mode=%s "
                    "policy_work_ms=%.3f misses=%d/%d skipped_states=%d gain_scale=%.3f "
                    "lowstate_age_ms=%.3f lowstate_tick=%d mode_machine=%d "
                    "motor_faults=%s max_temp=%s writer_hz=%.1f writer_age_ms=%.3f "
                    "writer_overruns=%d writer_get_max_ms=%.3f writer_send_max_ms=%.3f "
                    "lowstate_cpu=%s arm_reader_cpu=%s writer_cpu=%s"
                    % (
                        current_sequence,
                        previous_actor.sequence,
                        published.sequence,
                        previous_to_published_linf,
                        command[0], command[1], command[2],
                        inference_ms, state_to_action_ms, max_action, which,
                        work_ms, consecutive_timeouts, total_timeouts,
                        skipped_states, gain_scale,
                        state_diag["state_age_ms"],
                        state_diag["source_tick"],
                        state_diag["mode_machine"],
                        state_diag["nonzero_motorstate"],
                        state_diag["max_temperature"],
                        writer_diag["achieved_hz"],
                        writer_diag["last_publish_age_ms"],
                        writer_diag["overruns"],
                        writer_diag["get_ms_max"],
                        writer_diag["send_ms_max"],
                        format_cpu_set(state_diag["reader_cpu_affinity"] or []),
                        format_cpu_set(getattr(arm_source, "_reader_cpu_affinity", None) or []),
                        format_cpu_set(writer_diag["cpu_affinity"] or []),
                    )
                )
                next_status = now + max(0.1, float(args.status_every))
            next_deadline += control_dt
            remaining = next_deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining <= -control_dt:
                # Do not issue a catch-up burst after a scheduler stall.
                next_deadline = time.monotonic()
    except KeyboardInterrupt:
        reporter.emit("KeyboardInterrupt; damping shutdown requested", level="WARNING")
        runtime_exit_code = 130
    except BaseException as exc:
        reporter.emit(
            "real runtime failed: %s" % traceback.format_exc(),
            level="ERROR",
        )
        runtime_exit_code = 1
    finally:
        if remote is not None:
            remote.close()
        shutdown_real_runtime()
        if inference_samples:
            reporter.emit(
                "[G1-29][runtime] inference_ms median=%.3f p95=%.3f max=%.3f "
                "state_to_action_ms median=%.3f p95=%.3f max=%.3f "
                "policy_work_ms median=%.3f p95=%.3f max=%.3f timeouts=%d"
                % (
                    float(np.median(inference_samples)),
                    float(np.percentile(inference_samples, 95)),
                    max(inference_samples),
                    float(np.median(state_to_action_samples)),
                    float(np.percentile(state_to_action_samples, 95)),
                    max(state_to_action_samples),
                    float(np.median(policy_loop_samples)),
                    float(np.percentile(policy_loop_samples, 95)),
                    max(policy_loop_samples),
                    total_timeouts,
                )
            )
        reporter.emit("DreamWaQ stopped; see G1 environment close status.")
        reporter.close()
    return runtime_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
