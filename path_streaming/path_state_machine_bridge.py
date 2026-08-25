#!/usr/bin/env python3
"""PATH state-machine bridge between PICO body targets and SONIC.

This bridge is PATH's direct upper-body controller. It never publishes to a
SONIC ZMQ manager. Other producers are subordinate inputs:

  Pico manager     -> tcp://127.0.0.1:5559/{pose,command,planner,manager_state}
  Pico buttons     -> tcp://127.0.0.1:5565/pico_button_event
  State command    -> tcp://*:5563/path_state_command
  G1 direct mux    <- LCM G1_PATH_UPPER_COMMAND
  G1 state         <- LCM G1_ROBOT_STATE
  Telemetry        <- tcp://*:5560/path_qp_reference

State behavior:
  teleop/intention:
    Pass absolute PICO head-position-to-waist wrist poses directly to
    CasADi/IPOPT while teleop_tracking is armed. A+X enters
    teleop_wait_ready: IPOPT receives fixed FK(Q_READY) wrist SE(3) at 50 Hz,
    PICO arm targets are ignored, and the last teleop root command is latched.
    The next A+X requires measured Q_READY convergence before it resets the
    joystick mapper and resumes live PICO targets.
    Holding Y selects the left arm and holding B selects the right arm for
    belief/contact blending.  Without Y/B, both arms follow PICO directly.
  damped_return:
    Return the arm target toward the current PICO body target while
    The G1-local direct mux keeps lower-body and waist in damping mode.

Runtime hotkeys:
  q/w/e/r switch PATH state.
  p arm-disable requests are rejected because IPOPT owns both arms.
  m SONIC-mode requests are ignored; q/e keep identical planner conditioning
    and differ only in Cartesian target blending.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import select
import struct
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import lcm
import msgpack
import numpy as np
import zmq
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_command_message,
    build_planner_message,
)

from path_core.intention.action_blending import (
    LEFT_WRIST_LINK,
    RIGHT_WRIST_LINK,
    EETarget,
    SharedControlConfig,
    rotation_target_from_relation,
)
from path_core.intention.grasp_engagement import GraspEngagementDetector
from path_core.intention.reference_dynamics import (
    OrientationAuthority,
    OrientationTargetGate,
    SecondOrderOrientationReference,
    SecondOrderPositionReference,
    rotation_vector,
    slerp_so3,
)
from path_core.messaging.g1_direct_protocol import (
    ARM_START,
    MOTOR_COMMAND_CHANNEL,
    ROBOT_STATE_CHANNEL,
    UPPER_COMMAND_CHANNEL,
    pack_upper_command,
    unpack_robot_state,
)
from path_core.messaging.path_schema import (
    QP_REFERENCE_TOPIC,
    STATIC_EE_TARGET_TOPIC,
    ipopt_arm_sample_reference,
    now_us,
    qp_cartesian_target_reference,
    qp_none_reference,
    qp_sample_reference,
)
from path_core.messaging.zmq_bus import pack_topic, unpack_topic
from path_core.teleop.pico_joystick_locomotion import (
    PicoJoystickLocomotionConfig,
    PicoJoystickLocomotionMapper,
    SonicLocomotionCommand,
    yaw_from_quaternion_wxyz,
)
from path_planning.unitree_arm_nonlinear_ik import UnitreeG1ArmNonlinearIK
from path_streaming.path_smpl_thread_server import UPPER_BODY_PLANNER_INDICES
from perception.seg_web_node.robot_geometry import (
    HAND_AXIS_LOCAL,
    angular_velocity_from_rotations,
    hand_frame_profile,
)
from perception.remote_offload.wrist_d405_protocol_v11 import (
    WRIST_IPOPT_VISUAL_CHANNEL,
    pack_ipopt_visual,
)

# ``auto_planning`` remains an accepted external compatibility value, but is
# canonicalized to teleop.  Y/B is now the sole shared-control gate.
VALID_STATES = {"teleop", "intention", "auto_planning", "damped_return"}
SHARED_HAND_PUBLISH_HZ = 50.0

# Retained only for legacy compatibility helpers. The direct backend never
# publishes a SONIC planner message.
SONIC_DEFAULT_STANDING_HEIGHT = -1.0

# Canonical task-space contract used by PICO wrists, manual shared-control
# targets, and the arm-only IPOPT FK.  PICO world poses are first expressed as
# controller - headset + WAIST_OFFSET_XYZ by pico_manager_thread_server.py.
PICO_WAIST_FRAME = "pico_head_position_waist"
STATIC_TARGET_FRAME_ALIASES = {
    PICO_WAIST_FRAME,
    # Existing perception publishers already produce pelvis-identity robot
    # coordinates.  They are numerically equivalent to the canonical waist
    # frame and remain accepted while producers migrate their schema label.
    "pelvis_identity_fk",
}

STREAM_MODE_OFF = 0
STREAM_MODE_ALIGNING = 6
STREAM_MODE_TRACKING = 7

# G1 URDF zero pose: upper arms hang vertically from the shoulders while the
# forearms point forward (+X), giving the requested geometric 90-degree elbow.
Q_READY = np.zeros(29, dtype=np.float64)


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation to the scalar-first quaternion wire form."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        ])
    else:
        diagonal = np.diag(r)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        s = 2.0 * np.sqrt(max(1.0e-12, 1.0 + diagonal[index] - diagonal[next_index] - diagonal[last_index]))
        q = np.zeros(4, dtype=np.float64)
        q[index + 1] = 0.25 * s
        q[0] = (r[last_index, next_index] - r[next_index, last_index]) / s
        q[next_index + 1] = (r[next_index, index] + r[index, next_index]) / s
        q[last_index + 1] = (r[last_index, index] + r[index, last_index]) / s
    q /= max(1.0e-12, float(np.linalg.norm(q)))
    return -q if q[0] < 0.0 else q


def copy_locomotion_command(
    command: SonicLocomotionCommand,
    *,
    reason: str | None = None,
) -> SonicLocomotionCommand:
    """Copy a planner command before latching it across a control-mode change."""

    return SonicLocomotionCommand(
        mode=int(command.mode),
        movement=np.asarray(command.movement, dtype=np.float64).reshape(3).copy(),
        facing=np.asarray(command.facing, dtype=np.float64).reshape(3).copy(),
        speed=float(command.speed),
        height=float(command.height),
        body_velocity_mps=(
            None
            if command.body_velocity_mps is None
            else np.asarray(command.body_velocity_mps, dtype=np.float64)
            .reshape(-1)
            .copy()
        ),
        yaw_rate_rad_s=float(command.yaw_rate_rad_s),
        active=bool(command.active),
        reason=str(command.reason if reason is None else reason),
    )


def fixed_idle_locomotion(reason: str) -> SonicLocomotionCommand:
    """Return PATH's invariant no-locomotion planner command.

    ``facing`` is a SONIC command, not telemetry. Feeding measured base yaw
    back into it while idle creates a self-reinforcing turning command.
    """

    return SonicLocomotionCommand(
        mode=0,
        movement=np.zeros(3, dtype=np.float32),
        facing=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        speed=-1.0,
        height=SONIC_DEFAULT_STANDING_HEIGHT,
        body_velocity_mps=np.zeros(2, dtype=np.float32),
        yaw_rate_rad_s=0.0,
        active=False,
        reason=str(reason),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=sorted(VALID_STATES), default="teleop")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--height",
        type=float,
        default=SONIC_DEFAULT_STANDING_HEIGHT,
        help=(
            "Deprecated compatibility option. PATH always sends -1 so SONIC uses "
            "its configured default standing height; PATH does not command root height."
        ),
    )
    parser.add_argument(
        "--planner-lower-body-idle-default",
        "--teleop-gmr-arms-only-default",
        dest="planner_lower_body_idle_default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Condition SONIC with zero root/lower-body planner input instead of "
            "raw SMPL. The old option name remains as a CLI compatibility alias."
        ),
    )
    parser.add_argument(
        "--freeze-planner-lower-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force planner q_ref lower-body joints to zero before publishing to SONIC/QP.",
    )
    parser.add_argument(
        "--freeze-planner-waist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force planner q_ref waist yaw/roll/pitch joints to zero before publishing to SONIC/QP.",
    )
    parser.add_argument("--pico-hand-timeout-sec", type=float, default=0.35)
    parser.add_argument("--pico-body-gap-hold-ms", type=float, default=100.0)
    parser.add_argument(
        "--arm-reference-mode",
        choices=["head_position", "head_yaw"],
        default="head_position",
    )

    parser.add_argument("--pico-endpoint", default="tcp://127.0.0.1:5559")
    parser.add_argument(
        "--pico-vr-endpoint",
        default="tcp://127.0.0.1:5566",
        help="Independent absolute PICO waist-frame wrist stream.",
    )
    parser.add_argument("--button-endpoint", default="tcp://127.0.0.1:5565")
    parser.add_argument("--button-topic", default="pico_button_event")
    parser.add_argument(
        "--controller-state-topic", default="pico_controller_state"
    )
    parser.add_argument(
        "--joystick-locomotion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow safe PICO locomotion conditioning in q/teleop state only.",
    )
    parser.add_argument("--joystick-deadzone", type=float, default=0.18)
    parser.add_argument("--joystick-forward-max-mps", type=float, default=0.33)
    parser.add_argument("--joystick-backward-max-mps", type=float, default=0.165)
    parser.add_argument("--joystick-lateral-max-mps", type=float, default=0.198)
    parser.add_argument("--joystick-min-speed-mps", type=float, default=0.08)
    parser.add_argument("--joystick-max-yaw-rate-rad-s", type=float, default=0.30)
    parser.add_argument("--joystick-linear-accel-mps2", type=float, default=0.35)
    parser.add_argument("--joystick-linear-decel-mps2", type=float, default=0.70)
    parser.add_argument("--joystick-yaw-accel-rad-s2", type=float, default=0.60)
    parser.add_argument("--joystick-stale-timeout-sec", type=float, default=0.15)
    parser.add_argument("--static-target-bind", default="tcp://*:5564")
    parser.add_argument("--static-target-topic", default=STATIC_EE_TARGET_TOPIC)
    parser.add_argument("--static-target-timeout-sec", type=float, default=0.25)
    parser.add_argument(
        "--static-target-auto-enter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Automatically enter auto_planning when a new static EE target arrives. "
            "Disabled by default so perception gateways can buffer targets without "
            "moving the robot until the operator explicitly presses e."
        ),
    )
    parser.add_argument(
        "--static-target-hold-sec",
        type=float,
        default=0.0,
        help=(
            "Keep tracking the last valid static EE target for this long after "
            "messages stop. Values <= 0 hold forever until another state/target "
            "command replaces it; this is intended for 3-5Hz perception streams."
        ),
    )
    parser.add_argument("--robot-state-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--robot-state-topic", default="g1_debug")
    parser.add_argument(
        "--direct-lcm-url",
        default=os.environ.get("PATH_LCM_URL", "udpm://239.255.76.67:7667?ttl=1"),
        help="LCM transport carrying PATH direct upper commands and G1 state.",
    )
    parser.add_argument(
        "--wrist-visual-lcm-url",
        default=os.environ.get(
            "PATH_WRIST_VISUAL_LCM_URL",
            os.environ.get("WIFI_LCM_URL", "udpm://239.255.77.67:7667?ttl=1"),
        ),
        help="Wi-Fi LCM transport for direct Thor-to-laptop IPOPT visualization.",
    )
    parser.add_argument(
        "--wrist-visual-channel",
        default=WRIST_IPOPT_VISUAL_CHANNEL,
        help="Wi-Fi LCM channel carrying current IPOPT wrist targets.",
    )
    parser.add_argument(
        "--direct-upper-command-channel",
        default=UPPER_COMMAND_CHANNEL,
        help="Thor-to-G1 IPOPT arm command channel; never a SONIC action channel.",
    )
    parser.add_argument(
        "--direct-state-channel",
        default=ROBOT_STATE_CHANNEL,
        help="G1 low-level state channel consumed directly by PATH.",
    )
    parser.add_argument(
        "--direct-lcm-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use G1_ROBOT_STATE directly for IPOPT seeds and gravity on real hardware.",
    )
    parser.add_argument("--gravity-filter-cutoff-hz", type=float, default=8.0)
    parser.add_argument(
        "--gravity-accel-bias-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Slowly align quaternion gravity to accelerometer gravity only while "
            "the base is quasi-static. Requires base_accel_measured in g1_debug."
        ),
    )
    parser.add_argument("--gravity-bias-time-constant-sec", type=float, default=10.0)
    parser.add_argument("--gravity-bias-max-deg", type=float, default=3.0)
    parser.add_argument("--gravity-accel-norm-tolerance", type=float, default=0.50)
    parser.add_argument("--gravity-bias-gyro-gate-rad-s", type=float, default=0.05)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("sonic_runtime/gear_sonic_deploy/g1/g1_29dof.xml"),
        help="MuJoCo model used to detect when a planned EE target has been reached.",
    )

    parser.add_argument("--sonic-bind-host", default="*")
    parser.add_argument("--sonic-port", type=int, default=5556)
    parser.add_argument("--qp-bind-host", default="*")
    parser.add_argument("--qp-port", type=int, default=5560)
    parser.add_argument("--qp-topic", default=QP_REFERENCE_TOPIC)
    parser.add_argument("--state-command-bind", default="tcp://*:5563")
    parser.add_argument("--state-command-topic", default="path_state_command")
    parser.add_argument(
        "--shared-hand-selection-bind",
        default="tcp://*:5567",
        help="PUB endpoint that reports the Y/B-selected shared-control hand.",
    )
    parser.add_argument(
        "--shared-hand-selection-topic",
        default="path_shared_control_hand",
        help="Topic carrying the Y/B-selected shared-control hand heartbeat.",
    )

    parser.add_argument("--command-period-sec", type=float, default=0.25)
    parser.add_argument("--damped-return-rate", type=float, default=3.0)
    parser.add_argument(
        "--terminal-capture-radius-m",
        type=float,
        default=0.07,
        help="QP switches to the static terminal EE target once measured wrist is within this radius.",
    )
    parser.add_argument("--shared-rho1", type=float, default=0.60)
    parser.add_argument("--shared-rho2", type=float, default=0.90)
    parser.add_argument("--shared-alpha-rate-up", type=float, default=1.0)
    parser.add_argument("--shared-alpha-rate-down", type=float, default=3.0)
    parser.add_argument("--shared-v-transfer-max", type=float, default=0.25)
    parser.add_argument(
        "--shared-startup-hold-sec",
        type=float,
        default=0.0,
        help="Delay shared-control alpha after a Y/B hold (0 injects immediately).",
    )
    parser.add_argument("--shared-stale-after-sec", type=float, default=0.3)
    parser.add_argument("--shared-d-max-m", type=float, default=0.9)
    parser.add_argument(
        "--grasp-engagement-delay-sec",
        type=float,
        default=1.5,
        help=(
            "Continuous trigger/grip hold before the no-jaw-state fallback "
            "marks a grasp engaged and returns position control to VR."
        ),
    )
    parser.add_argument(
        "--end-effector-model",
        choices=("dex1", "brainco"),
        default="dex1",
        help="Canonical hand-frame profile used by belief orientation/EE targets.",
    )
    parser.add_argument(
        "--hand-axis-local",
        type=float,
        nargs=3,
        default=None,
        help="Override the active hand model's local orientation-belief axis.",
    )
    parser.add_argument(
        "--grasp-center-local",
        type=float,
        nargs=3,
        default=None,
        help="Override the active hand model's canonical grasp-center offset.",
    )
    parser.add_argument(
        "--grasp-center-wrist-x-offset-m",
        type=float,
        default=0.045,
        help=(
            "Wrist-yaw +X offset to the Dex1 hand-base/palm centre; 0.045 m "
            "matches the calibrated Dex1 attachment profile."
        ),
    )
    parser.add_argument("--shared-ref-lpf-alpha", type=float, default=0.3)
    parser.add_argument(
        "--shared-reset-alpha-on-target-switch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shared-hand-h-margin-m", type=float, default=0.05)
    parser.add_argument("--shared-hand-switch-confirm-n", type=int, default=10)
    parser.add_argument(
        "--shared-lock-hand-while-active",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shared-orientation-mode",
        choices=("gmr_prior", "normal_assist"),
        default="gmr_prior",
    )
    parser.add_argument("--orientation-enter-confidence", type=float, default=0.45)
    parser.add_argument("--orientation-exit-confidence", type=float, default=0.25)
    parser.add_argument("--orientation-ema-alpha", type=float, default=0.15)
    parser.add_argument("--orientation-max-authority-rate-per-s", type=float, default=1.2)
    parser.add_argument("--orientation-gate-max-angle-rad", type=float, default=1.047)
    parser.add_argument("--orientation-gate-taper-start-fraction", type=float, default=0.667)
    parser.add_argument(
        "--shared-waist-mode",
        choices=("sonic_passthrough", "hold_identity"),
        default="sonic_passthrough",
    )
    parser.add_argument(
        "--shared-lock-inactive-arm-q-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In e/shared-control Cartesian target mode, keep the non-target arm "
            "out of the GMR posture prior so the opposite operator hand does not "
            "jitter the active EE-QP solve."
        ),
    )
    parser.add_argument("--outer-waist-enabled", action="store_true",
                        help="Enable slow hierarchical waist yaw setpoint for static Cartesian EE servo.")
    parser.add_argument("--outer-waist-update-hz", type=float, default=1.0)
    parser.add_argument("--outer-waist-enable-error-m", type=float, default=0.045)
    parser.add_argument("--outer-waist-enable-dwell-sec", type=float, default=0.50)
    parser.add_argument("--outer-waist-freeze-error-m", type=float, default=0.025)
    parser.add_argument("--outer-waist-yaw-bound-rad", type=float, default=0.15)
    parser.add_argument("--outer-waist-roll-bound-rad", type=float, default=0.02)
    parser.add_argument("--outer-waist-pitch-bound-rad", type=float, default=0.03)
    parser.add_argument("--outer-waist-yaw-velocity-rad-s", type=float, default=0.06)
    parser.add_argument("--outer-waist-roll-velocity-rad-s", type=float, default=0.01)
    parser.add_argument("--outer-waist-pitch-velocity-rad-s", type=float, default=0.015)
    parser.add_argument("--outer-waist-yaw-samples", type=int, default=31)
    parser.add_argument("--outer-waist-current-weight", type=float, default=0.08)
    parser.add_argument("--outer-waist-neutral-weight", type=float, default=0.18)
    parser.add_argument(
        "--arm-nonlinear-ik-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Required PATH arm owner: Unitree-style CasADi Opti + IPOPT arm-only "
            "nonlinear IK for both q and e modes."
        ),
    )
    parser.add_argument(
        "--arm-ik-urdf",
        type=Path,
        default=Path("/home/url/00_workspace/xr_teleop/assets/g1/g1_body29_hand14.urdf"),
    )
    parser.add_argument(
        "--arm-ik-model-dir",
        type=Path,
        default=Path("/home/url/00_workspace/xr_teleop/assets/g1"),
    )
    parser.add_argument("--arm-ik-max-iter", type=int, default=30)
    # The IPOPT frame is wrist_yaw_link origin. BrainCo grasp-centre conversion
    # is performed exactly once by the second-order reference path.
    parser.add_argument("--arm-ik-ee-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--arm-ik-orientation-full-error-m", type=float, default=0.01
    )
    parser.add_argument(
        "--arm-ik-orientation-zero-error-m", type=float, default=0.03
    )
    parser.add_argument("--arm-ik-orientation-min-scale", type=float, default=1.0)
    parser.add_argument(
        "--arm-ik-residual-projection-threshold-m",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--arm-ik-residual-success-threshold-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--arm-ik-projection-step-m", type=float, default=0.05
    )
    parser.add_argument(
        "--arm-ik-output-velocity-limit-rad-s",
        type=float,
        default=10.0,
        help="XR G1-29 arm command velocity limit applied after the weighted filter.",
    )
    parser.add_argument(
        "--arm-ik-output-control-hz",
        type=float,
        default=50.0,
        help="PATH state-machine rate used to derive the per-command arm step.",
    )
    parser.add_argument(
        "--arm-ik-hard-lock-inactive-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set dynamic lower=upper bounds for the non-target arm.",
    )
    parser.add_argument(
        "--plan-complete-radius-m",
        type=float,
        default=0.07,
        help="Target distance used by optional auto-clear logging/behavior.",
    )
    parser.add_argument(
        "--auto-clear-on-target-reached",
        action="store_true",
        help="Clear active auto plan after duration_sec once measured EE is within --plan-complete-radius-m.",
    )
    parser.add_argument(
        "--clear-plan-after-total-duration",
        action="store_true",
        help="Clear active auto plan after duration_sec + final_hold_sec. By default the final knot is held.",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable terminal hotkeys q/w/e/r/p.",
    )
    parser.add_argument(
        "--qp-reference-disabled",
        action="store_true",
        help="Start with QP reference gated off. Press p to toggle at runtime.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[PATH-state] {message}", flush=True)


class KeyboardController:
    """Tiny non-blocking stdin controller for live demo state switching."""

    TOGGLE_QP = "__toggle_qp__"
    TOGGLE_TELEOP_SOURCE = "__toggle_teleop_source__"
    KEY_TO_STATE = {
        "w": "intention",
        "r": "damped_return",
        "p": TOGGLE_QP,
        "m": TOGGLE_TELEOP_SOURCE,
    }

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and sys.stdin.isatty())
        self._old_settings = None
        if not self.enabled:
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        atexit.register(self.close)

    def close(self) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def poll_command(self) -> str | None:
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        return self.KEY_TO_STATE.get(key.lower())


class MujocoWristFk:
    def __init__(self, model_xml: Path):
        import mujoco

        self.mujoco = mujoco
        model_path = model_xml.expanduser().resolve()
        old_cwd = os.getcwd()
        try:
            os.chdir(model_path.parent)
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        finally:
            os.chdir(old_cwd)
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_ids: dict[str, int] = {}

    def wrist_position(self, q29: np.ndarray, target_link: str) -> np.ndarray:
        q = np.asarray(q29, dtype=np.float64).reshape(29)
        body_id = self.body_ids.get(target_link)
        if body_id is None:
            body_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, target_link
            )
            if body_id < 0:
                raise ValueError(f"body not found in model: {target_link}")
            self.body_ids[target_link] = body_id
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[7:36] = q
        self.mujoco.mj_forward(self.model, self.data)
        pelvis = self.data.xpos[self.pelvis_id].copy() if self.pelvis_id >= 0 else np.zeros(3)
        return self.data.xpos[body_id].copy() - pelvis


class DirectLcmRobotStateSubscriber:
    """Keep the real G1 state independent of the retired SONIC debug stream."""

    def __init__(self, url: str, channel: str):
        self._bus = lcm.LCM(url)
        self._lock = threading.Lock()
        self._latest = None
        self._received_us = 0
        self._running = True
        self._bus.subscribe(channel, self._on_packet)
        self._thread = threading.Thread(
            target=self._run, name="path-g1-lcm-state", daemon=True
        )
        self._thread.start()

    def _on_packet(self, _channel: str, data: bytes) -> None:
        try:
            state = unpack_robot_state(data)
        except ValueError:
            return
        with self._lock:
            self._latest = state
            self._received_us = now_us()

    def _run(self) -> None:
        while self._running:
            try:
                self._bus.handle_timeout(20)
            except Exception:
                time.sleep(0.02)

    def latest(self):
        with self._lock:
            return self._latest, self._received_us

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=0.2)


def starts(raw: bytes, topic: str) -> bool:
    return raw.startswith(topic.encode())


def unpack_prefixed_msgpack(raw: bytes, topic: str) -> dict:
    return msgpack.unpackb(raw[len(topic):], raw=False)


def parse_packed_fields(raw: bytes, topic: str) -> dict[str, np.ndarray]:
    topic_b = topic.encode()
    if not raw.startswith(topic_b):
        raise ValueError(f"unexpected topic for {topic}")
    offset = len(topic_b)
    header_bytes = raw[offset : offset + HEADER_SIZE].split(b"\0", 1)[0]
    header = json.loads(header_bytes.decode("utf-8"))
    dtype_map = {
        "u8": np.uint8,
        "i32": np.int32,
        "i64": np.int64,
        "f32": np.float32,
        "f64": np.float64,
        "bool": np.bool_,
    }
    cursor = offset + HEADER_SIZE
    out: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        dtype = np.dtype(dtype_map[field["dtype"]])
        shape = tuple(field["shape"])
        size = int(np.prod(shape)) * dtype.itemsize
        out[field["name"]] = np.frombuffer(raw[cursor : cursor + size], dtype=dtype).reshape(shape).copy()
        cursor += size
    return out


def command_flags(raw: bytes) -> tuple[bool, bool, bool] | None:
    try:
        fields = parse_packed_fields(raw, "command")
        start = bool(np.asarray(fields.get("start", [0])).reshape(-1)[0])
        stop = bool(np.asarray(fields.get("stop", [0])).reshape(-1)[0])
        planner = bool(np.asarray(fields.get("planner", [0])).reshape(-1)[0])
        return start, stop, planner
    except Exception:
        return None


def pose_frame_index(raw: bytes) -> int | None:
    try:
        fields = parse_packed_fields(raw, "pose")
        indices = np.asarray(fields.get("frame_index", []), dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return None
        return int(indices[-1])
    except Exception:
        return None


def smpl_sample_from_pose(
    raw: bytes, fallback_timestamp_us: int
) -> tuple[np.ndarray, int, int] | None:
    """Extract the newest 24-joint SMPL sample without using root translation."""

    try:
        fields = parse_packed_fields(raw, "pose")
        joints = np.asarray(fields.get("smpl_joints"), dtype=np.float64)
        if joints.ndim == 2:
            joints = joints[None, ...]
        if joints.ndim != 3 or joints.shape[1:] != (24, 3) or not np.isfinite(joints[-1]).all():
            return None
        indices = np.asarray(fields.get("frame_index", []), dtype=np.int64).reshape(-1)
        frame_index = int(indices[-1]) if indices.size else -1
        source_timestamp_us = int(fallback_timestamp_us)
        for name in ("timestamp_monotonic_us", "timestamp_us", "source_timestamp_us"):
            value = fields.get(name)
            if value is None:
                continue
            values = np.asarray(value, dtype=np.int64).reshape(-1)
            if values.size and int(values[-1]) > 0:
                source_timestamp_us = int(values[-1])
                break
        return joints[-1].copy(), source_timestamp_us, frame_index
    except Exception:
        return None


@dataclass(frozen=True)
class WaistWristTargets:
    left_waist: np.ndarray
    right_waist: np.ndarray
    session_id: int
    frame_index: int
    source_timestamp_us: int
    valid: bool


def _valid_wrist_transform(value) -> np.ndarray | None:
    try:
        transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(transform).all():
        return None
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
        return None
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
        return None
    if abs(float(np.linalg.det(rotation)) - 1.0) > 2.0e-3:
        return None
    return transform.copy()


def static_target_frame_is_supported(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    frame = str(
        payload.get("target_position_frame", payload.get("coordinate_frame", ""))
    ).strip()
    return frame in STATIC_TARGET_FRAME_ALIASES


def manual_wrist_targets_from_static_payload(
    payload: dict | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(payload, dict) or not bool(payload.get("target_valid", True)):
        return None
    if not static_target_frame_is_supported(payload):
        return None
    wrist_frame = str(payload.get("human_wrist_frame", "")).strip()
    if wrist_frame != PICO_WAIST_FRAME:
        return None
    left = _valid_wrist_transform(payload.get("human_left_wrist_transform"))
    right = _valid_wrist_transform(payload.get("human_right_wrist_transform"))
    if left is None or right is None:
        return None
    return left, right


def static_target_payload_for_hand(payload: dict | None, hand: str) -> dict | None:
    """Select a per-hand belief target while retaining legacy single-target input."""
    if not isinstance(payload, dict):
        return None
    side = str(hand).lower()
    nested = payload.get("hand_targets")
    if isinstance(nested, dict) and side in {"left", "right"}:
        candidate = nested.get(side)
        return candidate if isinstance(candidate, dict) else None
    return payload


def wrist_targets_from_message(
    raw: bytes,
    topic: str,
    fallback_timestamp_us: int,
) -> WaistWristTargets | None:
    """Read absolute waist-frame wrist transforms without anchor or gain."""

    try:
        fields = parse_packed_fields(raw, topic)
        valid_values = np.asarray(
            fields.get("targets_valid", [False]), dtype=bool
        ).reshape(-1)
        if not valid_values.size or not bool(valid_values[-1]):
            return None
        offset = np.asarray(
            fields.get("ee_frame_offset_m", [0.05]), dtype=np.float64
        ).reshape(-1)
        if offset.size and abs(float(offset[-1]) - 0.05) > 1.0e-4:
            return None
        reference_modes = np.asarray(
            fields.get("arm_reference_mode", [-1]), dtype=np.int32
        ).reshape(-1)
        if not reference_modes.size or int(reference_modes[-1]) != 0:
            return None
        left_waist = np.asarray(
            fields["wrist_left_waist"], dtype=np.float64
        ).reshape(4, 4)
        right_waist = np.asarray(
            fields["wrist_right_waist"], dtype=np.float64
        ).reshape(4, 4)
        if not (
            np.isfinite(left_waist).all()
            and np.isfinite(right_waist).all()
        ):
            return None

        indices = np.asarray(fields.get("frame_index", []), dtype=np.int64).reshape(-1)
        frame_index = int(indices[-1]) if indices.size else -1
        sessions = np.asarray(
            fields.get("session_id", [-1]), dtype=np.int32
        ).reshape(-1)
        session_id = int(sessions[-1]) if sessions.size else -1
        source_timestamp_us = int(fallback_timestamp_us)
        monotonic_value = fields.get("timestamp_monotonic")
        if monotonic_value is not None:
            values = np.asarray(monotonic_value, dtype=np.float64).reshape(-1)
            if values.size and np.isfinite(values[-1]) and values[-1] > 0.0:
                source_timestamp_us = int(round(float(values[-1]) * 1.0e6))
        else:
            for name in ("timestamp_monotonic_us", "timestamp_us", "source_timestamp_us"):
                value = fields.get(name)
                if value is None:
                    continue
                values = np.asarray(value, dtype=np.int64).reshape(-1)
                if values.size and int(values[-1]) > 0:
                    source_timestamp_us = int(values[-1])
                    break

        return WaistWristTargets(
            left_waist=left_waist.copy(),
            right_waist=right_waist.copy(),
            session_id=session_id,
            source_timestamp_us=source_timestamp_us,
            frame_index=frame_index,
            valid=True,
        )
    except Exception:
        return None


def upper_from_q29(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64).reshape(29)[UPPER_BODY_PLANNER_INDICES].astype(np.float32)


def frozen_planner_q29(
    q29: np.ndarray,
    *,
    freeze_lower_body: bool = True,
    freeze_waist: bool = True,
) -> np.ndarray:
    """Return a q29 safe for SONIC planner/QP posture refs.

    G1 29-DoF order used here:
      0:12   lower body
      12:15  waist yaw/roll/pitch
      15:22  left arm
      22:29  right arm

    The planner wire message separately fixes locomotion/root via
    mode=IDLE, movement=[0,0,0], facing=[1,0,0], speed=-1, and height=-1
    (the SONIC default-standing-height sentinel).
    """

    q = np.asarray(q29, dtype=np.float64).reshape(29).copy()
    if freeze_lower_body:
        q[:12] = 0.0
    if freeze_waist:
        q[12:15] = 0.0
    return q


def send_planner_command(
    socket: zmq.Socket | None,
    q29: np.ndarray,
    neutralize_waist: bool = False,
    freeze_lower_body: bool = True,
    left_hand_position: np.ndarray | list[float] | None = None,
    right_hand_position: np.ndarray | list[float] | None = None,
    locomotion: SonicLocomotionCommand | None = None,
) -> None:
    # The direct backend deliberately has no SONIC publisher.  Keeping this
    # compatibility helper as an explicit no-op avoids feeding a stale planner
    # command into a policy process that happens to still be running.
    if socket is None:
        return
    q_for_command = np.asarray(q29, dtype=np.float64).reshape(29).copy()
    if freeze_lower_body:
        q_for_command[:12] = 0.0
    if neutralize_waist:
        q_for_command[12:15] = 0.0
    upper = upper_from_q29(q_for_command)
    left_hand = (
        np.asarray(left_hand_position, dtype=np.float32).reshape(7)
        if left_hand_position is not None
        else np.zeros(7, dtype=np.float32)
    )
    right_hand = (
        np.asarray(right_hand_position, dtype=np.float32).reshape(7)
        if right_hand_position is not None
        else np.zeros(7, dtype=np.float32)
    )
    locomotion_mode = 0 if locomotion is None else int(locomotion.mode)
    movement = (
        [0.0, 0.0, 0.0]
        if locomotion is None
        else np.asarray(locomotion.movement, dtype=np.float32).reshape(3).tolist()
    )
    facing = (
        [1.0, 0.0, 0.0]
        if locomotion is None
        else np.asarray(locomotion.facing, dtype=np.float32).reshape(3).tolist()
    )
    speed = -1.0 if locomotion is None else float(locomotion.speed)
    height = (
        SONIC_DEFAULT_STANDING_HEIGHT
        if locomotion is None
        else float(locomotion.height)
    )
    socket.send(
        build_planner_message(
            locomotion_mode,
            movement,
            facing,
            speed=speed,
            # Do not inject a root/base-height target from PATH.  -1 is the
            # protocol's use-trained-default sentinel, not a zero-height
            # command.  The remaining locomotion command is intentionally
            # static as well.
            height=height,
            upper_body_position=upper.tolist(),
            upper_body_velocity=np.zeros(17, dtype=np.float32).tolist(),
            left_hand_position=left_hand.tolist(),
            right_hand_position=right_hand.tolist(),
        )
    )


def send_mode_command(socket: zmq.Socket | None, planner: bool) -> None:
    if socket is None:
        return
    socket.send(build_command_message(start=False, stop=False, planner=planner))


def hand_joints_from_fields(fields: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    def _hand(name: str) -> np.ndarray | None:
        value = fields.get(name)
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.shape[0] != 7 or not np.isfinite(arr).all():
            return None
        return arr.copy()

    meta: dict[str, float] = {}
    for key in ("left_trigger", "right_trigger", "left_grip", "right_grip"):
        value = fields.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size and np.isfinite(arr[-1]):
                meta[key] = float(arr[-1])
        except (TypeError, ValueError):
            continue
    return _hand("left_hand_joints"), _hand("right_hand_joints"), meta


def latest_q_from_robot_state(payload: dict | None) -> np.ndarray | None:
    if not payload:
        return None
    for key in ("body_q", "q_measured", "state_body_q", "body_q_measured"):
        q = payload.get(key)
        if q is None:
            continue
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape[0] >= 29 and np.isfinite(arr[:29]).all():
            return arr[:29].copy()
    return None


def latest_base_quat_from_robot_state(
    payload: dict | None,
) -> np.ndarray | None:
    if not payload:
        return None
    for key in (
        "base_quat_measured",
        "base_quat",
        "state_base_quat_measured",
        "state_base_quat",
    ):
        value = payload.get(key)
        if value is None:
            continue
        quat = np.asarray(value, dtype=np.float64).reshape(-1)
        if quat.shape[0] < 4 or not np.isfinite(quat[:4]).all():
            continue
        norm = float(np.linalg.norm(quat[:4]))
        if norm > 1.0e-8:
            return quat[:4].copy() / norm
    return None


def latest_vector_from_robot_state(
    payload: dict | None,
    keys: tuple[str, ...],
    size: int,
) -> np.ndarray | None:
    if not payload:
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape[0] >= size and np.isfinite(vector[:size]).all():
            return vector[:size].copy()
    return None


def gravity_direction_from_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    w, x, y, z = quat
    rotation_world_pelvis = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    gravity = rotation_world_pelvis.T @ np.array([0.0, 0.0, -1.0])
    return gravity / max(float(np.linalg.norm(gravity)), 1.0e-12)


def rotate_by_rotation_vector(vector: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 1.0e-12:
        return vector.copy()
    axis = rotation_vector / angle
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * float(np.dot(axis, vector)) * (1.0 - np.cos(angle))
    )


class PelvisGravityEstimator:
    """Filtered pelvis-frame gravity with gated accelerometer bias correction."""

    def __init__(self, args: argparse.Namespace):
        self.cutoff_hz = max(float(args.gravity_filter_cutoff_hz), 0.0)
        self.accel_bias_enabled = bool(args.gravity_accel_bias_enabled)
        self.bias_time_constant_sec = max(
            float(args.gravity_bias_time_constant_sec), 1.0e-3
        )
        self.bias_max_rad = np.deg2rad(max(float(args.gravity_bias_max_deg), 0.0))
        self.accel_norm_tolerance = max(
            float(args.gravity_accel_norm_tolerance), 0.0
        )
        self.gyro_gate = max(float(args.gravity_bias_gyro_gate_rad_s), 0.0)
        self.filtered_gravity: np.ndarray | None = None
        self.bias_rotation_vector = np.zeros(3, dtype=np.float64)
        self.accel_correction_active = False
        self.accel_samples = 0

    def update(
        self,
        quat_wxyz: np.ndarray,
        accel_pelvis: np.ndarray | None,
        gyro_pelvis: np.ndarray | None,
        dt: float,
    ) -> np.ndarray:
        gravity_quat = gravity_direction_from_quaternion(quat_wxyz)
        self.accel_correction_active = False
        if self.accel_bias_enabled and accel_pelvis is not None:
            accel = np.asarray(accel_pelvis, dtype=np.float64).reshape(3)
            accel_norm = float(np.linalg.norm(accel))
            gyro_norm = (
                float(np.linalg.norm(gyro_pelvis))
                if gyro_pelvis is not None
                else float("inf")
            )
            if (
                np.isfinite(accel_norm)
                and abs(accel_norm - 9.81) <= self.accel_norm_tolerance
                and gyro_norm <= self.gyro_gate
            ):
                gravity_accel = -accel / max(accel_norm, 1.0e-12)
                cross = np.cross(gravity_quat, gravity_accel)
                cross_norm = float(np.linalg.norm(cross))
                dot = float(np.clip(np.dot(gravity_quat, gravity_accel), -1.0, 1.0))
                angle = float(np.arctan2(cross_norm, dot))
                observed_rotation = (
                    cross * (angle / cross_norm)
                    if cross_norm > 1.0e-12
                    else np.zeros(3, dtype=np.float64)
                )
                bias_alpha = 1.0 - np.exp(
                    -max(float(dt), 0.0) / self.bias_time_constant_sec
                )
                self.bias_rotation_vector += bias_alpha * (
                    observed_rotation - self.bias_rotation_vector
                )
                bias_norm = float(np.linalg.norm(self.bias_rotation_vector))
                if bias_norm > self.bias_max_rad > 0.0:
                    self.bias_rotation_vector *= self.bias_max_rad / bias_norm
                self.accel_correction_active = True
                self.accel_samples += 1

        corrected = rotate_by_rotation_vector(
            gravity_quat, self.bias_rotation_vector
        )
        corrected /= max(float(np.linalg.norm(corrected)), 1.0e-12)
        if self.filtered_gravity is None:
            self.filtered_gravity = corrected
        else:
            alpha = (
                1.0
                if self.cutoff_hz <= 0.0
                else 1.0
                - np.exp(-2.0 * np.pi * self.cutoff_hz * max(float(dt), 0.0))
            )
            self.filtered_gravity += alpha * (
                corrected - self.filtered_gravity
            )
            self.filtered_gravity /= max(
                float(np.linalg.norm(self.filtered_gravity)), 1.0e-12
            )
        return self.filtered_gravity.copy()

    def status(self) -> str:
        return (
            f"gravity_bias={np.rad2deg(np.linalg.norm(self.bias_rotation_vector)):.2f}deg "
            f"accel_corr={'on' if self.accel_correction_active else 'off'}"
            f"/{self.accel_samples}"
        )


def _nested(payload: dict | None, *keys: str):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _float_field(payload: dict | None, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = payload.get(key) if isinstance(payload, dict) else None
        if value is None:
            value = _nested(payload, "perception", key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return float(default)


def _str_field(payload: dict | None, *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key) if isinstance(payload, dict) else None
        if value is None:
            value = _nested(payload, "perception", key)
        if value is not None:
            return str(value)
    return default


def _int_field(payload: dict | None, *keys: str, default: int = -1) -> int:
    for key in keys:
        value = payload.get(key) if isinstance(payload, dict) else None
        if value is None:
            value = _nested(payload, "perception", key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def ee_target_from_static_payload(
    payload: dict | None,
    *,
    received_us: int,
) -> EETarget | None:
    if (
        not payload
        or not bool(payload.get("target_valid", True))
        or not static_target_frame_is_supported(payload)
    ):
        return None
    try:
        position = np.asarray(
            payload.get(
                "terminal_target_position",
                payload.get("position", payload.get("target_position", [])),
            ),
            dtype=np.float64,
        ).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(position).all():
        return None
    confidence = _float_field(
        payload,
        "confidence",
        "joint_confidence",
        "tasc_joint_confidence",
        default=1.0,
    )
    object_id = _str_field(
        payload,
        "object_id",
        "tasc_object_id",
        "label",
        default=str(payload.get("goal_uid", "")),
    )
    grip_index = _int_field(
        payload,
        "grip_index",
        "selected_grip_index",
        "tasc_selected_grip_index",
        "object_index",
        default=-1,
    )
    axis_vector = None
    try:
        candidate = np.asarray(payload.get("orientation_axis_world"), dtype=np.float64).reshape(3)
        if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 1.0e-9:
            axis_vector = candidate
    except (TypeError, ValueError):
        pass
    frame_world = None
    try:
        candidate = np.asarray(payload.get("orientation_frame_world"), dtype=np.float64).reshape(3, 3)
        if np.isfinite(candidate).all() and abs(float(np.linalg.det(candidate))) > 0.5:
            u, _singular, vt = np.linalg.svd(candidate)
            frame_world = u @ vt
            if float(np.linalg.det(frame_world)) < 0.0:
                u[:, -1] *= -1.0
                frame_world = u @ vt
    except (TypeError, ValueError, np.linalg.LinAlgError):
        pass
    return EETarget(
        position=position,
        confidence=confidence,
        position_valid=bool(payload.get("position_valid", True)),
        # Dynamic orientation is controlled exclusively by the surface-normal
        # relation below; legacy payload quaternions can be arbitrary at the
        # perpendicular singularity and must never reach the IK objective.
        orientation_wxyz=None,
        object_id=object_id,
        grip_index=grip_index,
        goal_uid=str(payload.get("goal_uid", "")),
        timestamp_us=int(received_us),
        # ``orientation_valid`` is retained as producer telemetry only.  A
        # finite axis/frame plus confidence is the continuous-authority input;
        # an old binary field must not abruptly remove a valid constraint.
        orientation_axis_world=axis_vector,
        orientation_relation=str(payload.get("orientation_relation", "") or ""),
        orientation_confidence=_float_field(payload, "orientation_confidence", default=0.0),
        orientation_policy=str(payload.get("orientation_policy", "minimal_rotation") or "minimal_rotation"),
        orientation_frame_world=frame_world,
        position_point=str(payload.get("position_point", "wrist_origin")),
    )


def target_side_from_link(target_link: str) -> str:
    if target_link.startswith("right_") or "right" in target_link:
        return "right"
    if target_link.startswith("left_") or "left" in target_link:
        return "left"
    return ""


def hold_inactive_arm_for_single_target(
    q_command: np.ndarray,
    q_hold: np.ndarray | None,
    target_side: str,
) -> np.ndarray:
    output = np.asarray(q_command, dtype=np.float64).reshape(29).copy()
    if q_hold is None:
        return output
    hold = np.asarray(q_hold, dtype=np.float64).reshape(29)
    if target_side == "left":
        output[22:29] = hold[22:29]
    elif target_side == "right":
        output[15:22] = hold[15:22]
    return output


def target_arm_prior_from_ik(
    q_base: np.ndarray,
    q_ik: np.ndarray | None,
    target_side: str,
) -> np.ndarray:
    output = np.asarray(q_base, dtype=np.float64).reshape(29).copy()
    if q_ik is None:
        return output
    ik = np.asarray(q_ik, dtype=np.float64).reshape(29)
    if target_side == "left":
        output[15:22] = ik[15:22]
    elif target_side == "right":
        output[22:29] = ik[22:29]
    return output


class OuterWaistPlanner:
    """Slow yaw-only waist posture selector for static Cartesian servo."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.enabled = bool(args.outer_waist_enabled)
        self.start = np.zeros(3, dtype=np.float64)
        self.ref = np.zeros(3, dtype=np.float64)
        self.desired = np.zeros(3, dtype=np.float64)
        self.hold_plan_id = -1
        self.hold_side = ""
        self.high_error_sec = 0.0
        self.last_select_us = 0
        self.active = False
        self.last_reason = "disabled"
        self.last_score = float("nan")

    def reset(self, q_seed: np.ndarray, plan_id: int, target_side: str) -> None:
        waist = np.asarray(q_seed, dtype=np.float64).reshape(29)[12:15].copy()
        self.start = waist
        self.ref = waist
        self.desired = waist
        self.hold_plan_id = int(plan_id)
        self.hold_side = target_side
        self.high_error_sec = 0.0
        self.last_select_us = 0
        self.active = False
        self.last_reason = "reset"
        self.last_score = float("nan")

    def maybe_reset(self, q_seed: np.ndarray, plan_id: int, target_side: str) -> None:
        if self.hold_plan_id != int(plan_id) or self.hold_side != target_side:
            self.reset(q_seed, plan_id, target_side)

    def _bounded_target(self, candidate_yaw: float) -> np.ndarray:
        target = self.start.copy()
        target[0] = float(np.clip(
            candidate_yaw,
            self.start[0] - self.args.outer_waist_yaw_bound_rad,
            self.start[0] + self.args.outer_waist_yaw_bound_rad,
        ))
        target[1] = float(np.clip(
            self.start[1],
            self.start[1] - self.args.outer_waist_roll_bound_rad,
            self.start[1] + self.args.outer_waist_roll_bound_rad,
        ))
        target[2] = float(np.clip(
            self.start[2],
            self.start[2] - self.args.outer_waist_pitch_bound_rad,
            self.start[2] + self.args.outer_waist_pitch_bound_rad,
        ))
        return target

    def _select_desired_yaw(
        self,
        wrist_fk: MujocoWristFk,
        q_base: np.ndarray,
        q_posture: np.ndarray,
        target_link: str,
        target_position: np.ndarray,
    ) -> None:
        n = max(3, int(self.args.outer_waist_yaw_samples))
        lo = self.start[0] - self.args.outer_waist_yaw_bound_rad
        hi = self.start[0] + self.args.outer_waist_yaw_bound_rad
        candidates = np.linspace(lo, hi, n, dtype=np.float64)
        best_score = float("inf")
        best = self.desired.copy()
        current_yaw = float(self.ref[0])
        neutral_yaw = float(self.start[0])
        for yaw in candidates:
            q = np.asarray(q_posture, dtype=np.float64).reshape(29).copy()
            q[12:15] = self._bounded_target(float(yaw))
            try:
                wrist = wrist_fk.wrist_position(q, target_link)
            except Exception:
                continue
            err = float(np.linalg.norm(wrist - target_position))
            score = (
                err * err
                + self.args.outer_waist_current_weight * (float(yaw) - current_yaw) ** 2
                + self.args.outer_waist_neutral_weight * (float(yaw) - neutral_yaw) ** 2
            )
            if score < best_score:
                best_score = score
                best = self._bounded_target(float(yaw))
        if np.isfinite(best_score):
            self.desired = best
            self.last_score = best_score
            self.last_reason = "selected"
        else:
            self.last_reason = "select_failed"

    def update(
        self,
        dt: float,
        tick_us: int,
        terminal_error_m: float,
        wrist_fk: MujocoWristFk | None,
        q_command: np.ndarray,
        q_posture: np.ndarray,
        target_link: str,
        target_position: np.ndarray,
    ) -> np.ndarray:
        if not self.enabled:
            self.last_reason = "disabled"
            return np.asarray(q_command, dtype=np.float64).reshape(29)
        output = np.asarray(q_command, dtype=np.float64).reshape(29).copy()
        if not np.isfinite(terminal_error_m):
            self.last_reason = "no_error"
            output[12:15] = self.ref
            return output
        if terminal_error_m > self.args.outer_waist_enable_error_m:
            self.high_error_sec += max(0.0, float(dt))
        else:
            self.high_error_sec = 0.0
        if terminal_error_m < self.args.outer_waist_freeze_error_m:
            self.active = False
            self.last_reason = "freeze_near_target"
        elif self.high_error_sec >= self.args.outer_waist_enable_dwell_sec:
            self.active = True

        update_interval_us = int(1.0e6 / max(self.args.outer_waist_update_hz, 1.0e-6))
        if (
            self.active
            and wrist_fk is not None
            and (tick_us - self.last_select_us) >= update_interval_us
        ):
            self._select_desired_yaw(
                wrist_fk=wrist_fk,
                q_base=output,
                q_posture=q_posture,
                target_link=target_link,
                target_position=target_position,
            )
            self.last_select_us = tick_us

        velocity = np.array([
            self.args.outer_waist_yaw_velocity_rad_s,
            self.args.outer_waist_roll_velocity_rad_s,
            self.args.outer_waist_pitch_velocity_rad_s,
        ], dtype=np.float64)
        step = np.clip(self.desired - self.ref, -velocity * dt, velocity * dt)
        self.ref = self._bounded_target(self.ref[0] + step[0])
        # Keep roll/pitch effectively fixed at the latched start posture for
        # now. Their bounds/velocities are present for later expansion.
        self.ref[1:] = self.start[1:]
        output[12:15] = self.ref
        return output

    def status(self) -> str:
        return (
            f"outer_waist={'on' if self.enabled else 'off'}"
            f"/{'active' if self.active else 'frozen'} "
            f"yaw={self.ref[0]:+.3f}->{self.desired[0]:+.3f} "
            f"err_hold={self.high_error_sec:.2f}s reason={self.last_reason}"
        )


def main() -> int:
    args = parse_args()
    hand_axis_local_override = args.hand_axis_local is not None
    if args.hand_axis_local is None:
        args.hand_axis_local = tuple(HAND_AXIS_LOCAL[args.end_effector_model].tolist())
    else:
        print(
            "[PATH-state] WARNING: --hand-axis-local overrides the canonical "
            "physical hand-axis convention.",
            flush=True,
        )
    if args.grasp_center_local is None:
        args.grasp_center_local = tuple(
            np.asarray(hand_frame_profile(args.end_effector_model)["grasp_center_local"], dtype=np.float64)
        )
    if not args.arm_nonlinear_ik_enabled:
        raise RuntimeError(
            "PATH requires CasADi/IPOPT arm ownership; "
            "--no-arm-nonlinear-ik-enabled is not supported."
        )
    if args.qp_reference_disabled:
        raise RuntimeError(
            "PATH requires the CasADi/IPOPT arm reference stream in q/e modes; "
            "--qp-reference-disabled is not supported."
        )
    # PATH's arm ownership is invariant. The planner input keeps a neutral
    # waist reference while the deploy forwards SONIC's resulting waist
    # q/dq/tau command without PATH filtering.
    args.freeze_planner_lower_body = True
    args.freeze_planner_waist = True
    args.shared_waist_mode = "hold_identity"
    args.outer_waist_enabled = False
    ctx = zmq.Context.instance()

    def make_pico_sub(
        topic: str,
        *,
        conflate: bool,
        hwm: int = 8,
        endpoint: str | None = None,
    ) -> zmq.Socket:
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        if conflate:
            socket.setsockopt(zmq.CONFLATE, 1)
        else:
            socket.setsockopt(zmq.RCVHWM, int(hwm))
        socket.setsockopt(zmq.RCVTIMEO, 1)
        socket.connect(endpoint or args.pico_endpoint)
        return socket

    # Keep independent queues so high-rate manager/planner traffic cannot push
    # control or controller-pose frames behind a backlog. vr_pose_debug is
    # independent of PICO's SONIC stream mode and is the authoritative arm
    # target source.
    pico_vr_body = make_pico_sub(
        "vr_pose_debug", conflate=True, endpoint=args.pico_vr_endpoint
    )
    pico_pose = make_pico_sub("pose", conflate=True)
    pico_command = make_pico_sub("command", conflate=False, hwm=32)
    pico_planner = make_pico_sub("planner", conflate=True)
    pico_manager = make_pico_sub("manager_state", conflate=True)

    button_sub = ctx.socket(zmq.SUB)
    button_sub.setsockopt(zmq.SUBSCRIBE, args.button_topic.encode())
    button_sub.setsockopt(zmq.RCVHWM, 64)
    button_sub.setsockopt(zmq.RCVTIMEO, 1)
    button_sub.connect(args.button_endpoint)

    controller_sub = ctx.socket(zmq.SUB)
    controller_sub.setsockopt(
        zmq.SUBSCRIBE, args.controller_state_topic.encode()
    )
    controller_sub.setsockopt(zmq.CONFLATE, 1)
    controller_sub.setsockopt(zmq.RCVTIMEO, 1)
    controller_sub.connect(args.button_endpoint)

    static_target_sub = ctx.socket(zmq.SUB)
    static_target_sub.setsockopt(zmq.SUBSCRIBE, args.static_target_topic.encode())
    static_target_sub.setsockopt(zmq.CONFLATE, 1)
    static_target_sub.setsockopt(zmq.RCVTIMEO, 1)
    static_target_sub.bind(args.static_target_bind)

    robot_state = ctx.socket(zmq.SUB)
    robot_state.setsockopt(zmq.SUBSCRIBE, args.robot_state_topic.encode())
    robot_state.setsockopt(zmq.CONFLATE, 1)
    robot_state.setsockopt(zmq.RCVTIMEO, 1)
    robot_state.connect(args.robot_state_endpoint)

    state_sub = ctx.socket(zmq.SUB)
    state_sub.setsockopt(zmq.SUBSCRIBE, args.state_command_topic.encode())
    state_sub.setsockopt(zmq.CONFLATE, 1)
    state_sub.setsockopt(zmq.RCVTIMEO, 1)
    state_sub.bind(args.state_command_bind)

    # SONIC is intentionally absent from the direct-control topology.  All
    # arm commands leave this process only as G1_PATH_UPPER_COMMAND over LCM.
    sonic: zmq.Socket | None = None
    direct_lcm = lcm.LCM(args.direct_lcm_url)
    wrist_visual_lcm = lcm.LCM(args.wrist_visual_lcm_url)
    direct_lcm_sequence = 0
    direct_state = (
        DirectLcmRobotStateSubscriber(
            args.direct_lcm_url, args.direct_state_channel
        )
        if args.direct_lcm_state
        else None
    )

    qp = ctx.socket(zmq.PUB)
    qp.setsockopt(zmq.SNDHWM, 2)
    qp.setsockopt(zmq.LINGER, 0)
    qp.bind(f"tcp://{args.qp_bind_host}:{args.qp_port}")

    shared_hand_pub = ctx.socket(zmq.PUB)
    shared_hand_pub.setsockopt(zmq.SNDHWM, 2)
    shared_hand_pub.setsockopt(zmq.LINGER, 0)
    shared_hand_pub.bind(args.shared_hand_selection_bind)

    log(f"Pico input SUB {args.pico_endpoint}/pose,command,planner,manager_state")
    log(
        f"Pico controller SUB {args.button_endpoint}/"
        f"{args.button_topic},{args.controller_state_topic}"
    )
    log(
        "GMR input disabled: wrist targets use manager-calibrated PICO "
        "pelvis-frame SE(3) directly"
    )
    log(f"static EE target SUB {args.static_target_bind}/{args.static_target_topic}")
    log(f"robot state SUB {args.robot_state_endpoint}/{args.robot_state_topic}")
    log(f"state command SUB {args.state_command_bind}/{args.state_command_topic}")
    log(
        "SONIC output disabled; direct IPOPT arm PUB "
        f"{args.direct_lcm_url}/{args.direct_upper_command_channel}"
    )
    log(
        "wrist IPOPT visual PUB "
        f"{args.wrist_visual_lcm_url}/{args.wrist_visual_channel}"
    )
    if direct_state is not None:
        log(
            "direct G1 state SUB "
            f"{args.direct_lcm_url}/{args.direct_state_channel}"
        )
    log(f"QP reference PUB tcp://{args.qp_bind_host}:{args.qp_port}/{args.qp_topic}")
    log(
        "shared-control hand PUB "
        f"{args.shared_hand_selection_bind}/{args.shared_hand_selection_topic}"
    )
    arm_nonlinear_ik: UnitreeG1ArmNonlinearIK | None = None
    if args.arm_nonlinear_ik_enabled:
        arm_nonlinear_ik = UnitreeG1ArmNonlinearIK(
            urdf_path=args.arm_ik_urdf,
            model_dir=args.arm_ik_model_dir,
            max_iter=args.arm_ik_max_iter,
            ee_offset_m=args.arm_ik_ee_offset_m,
            orientation_full_error_m=args.arm_ik_orientation_full_error_m,
            orientation_zero_error_m=args.arm_ik_orientation_zero_error_m,
            orientation_min_scale=args.arm_ik_orientation_min_scale,
            residual_projection_threshold_m=(
                args.arm_ik_residual_projection_threshold_m
            ),
            residual_success_threshold_m=(
                args.arm_ik_residual_success_threshold_m
            ),
            projection_step_m=args.arm_ik_projection_step_m,
            output_velocity_limit_rad_s=(
                args.arm_ik_output_velocity_limit_rad_s
            ),
            output_control_dt_s=(
                1.0 / max(float(args.arm_ik_output_control_hz), 1.0)
            ),
        )
        log(
            "Unitree-style arm nonlinear IK enabled: "
            f"urdf={args.arm_ik_urdf}, max_iter={args.arm_ik_max_iter}, "
            f"hard_lock_inactive={args.arm_ik_hard_lock_inactive_arm}, "
            "orientation_weight_scale=fixed_1.0, "
            f"residual_project/success="
            f"{100.0 * args.arm_ik_residual_projection_threshold_m:.1f}/"
            f"{100.0 * args.arm_ik_residual_success_threshold_m:.1f}cm, "
            "q_output=xr_wma[0.4,0.3,0.2,0.1]+"
            f"{args.arm_ik_output_velocity_limit_rad_s:.1f}rad/s@"
            f"{args.arm_ik_output_control_hz:.0f}Hz"
        )
    if arm_nonlinear_ik is None:
        raise RuntimeError(
            "teleop_wait_ready requires the CasADi/IPOPT arm solver; "
            "--no-arm-nonlinear-ik-enabled is not supported"
        )
    q_ready_left_wrist_tf, q_ready_right_wrist_tf = (
        arm_nonlinear_ik.wrist_transforms_from_q29(Q_READY)
    )
    log(
        "teleop_wait_ready fixed IPOPT target prepared from FK(Q_READY): "
        f"L={q_ready_left_wrist_tf[:3, 3].round(4).tolist()} "
        f"R={q_ready_right_wrist_tf[:3, 3].round(4).tolist()}"
    )
    gravity_estimator = PelvisGravityEstimator(args)
    log(
        "50Hz arm-only RNEA enabled (waist feedforward zero): "
        f"cutoff={args.gravity_filter_cutoff_hz:.1f}Hz "
        f"accel_bias={bool(args.gravity_accel_bias_enabled)} "
        f"bias_tau={args.gravity_bias_time_constant_sec:.1f}s"
    )

    current_state = "teleop" if args.state == "auto_planning" else args.state
    if args.state == "auto_planning":
        log("initial auto_planning alias -> teleop (Y/B gates belief)")
    # q/e share one SONIC planner-conditioning mode. State changes affect only
    # the Cartesian target blend sent to IPOPT.
    planner_lower_body_idle = True
    released_pose_frames = 0
    latest_static_target: dict | None = None
    latest_static_target_us = 0
    static_target_revision = 0
    static_target_plan_id = 0
    latest_static_target_uid = ""
    static_target_hold_q: np.ndarray | None = None
    static_target_hold_plan_id = -1
    static_target_hold_side = ""
    latest_robot_q: np.ndarray | None = None
    latest_robot_dq: np.ndarray | None = None
    latest_robot_base_quat = np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    latest_robot_base_accel: np.ndarray | None = None
    latest_robot_base_gyro: np.ndarray | None = None
    latest_gravity_direction = np.array(
        [0.0, 0.0, -1.0], dtype=np.float64
    )
    latest_robot_state_us = 0
    latest_pose_raw: bytes | None = None
    latest_pose_us = 0
    latest_smpl_joints: np.ndarray | None = None
    latest_smpl_source_us = 0
    latest_smpl_frame = -1
    latest_wrist_targets: WaistWristTargets | None = None
    latest_pico_body_receive_us = 0
    latest_pico_session_id = -1
    pico_tracking_armed = False
    pico_arm_pending = False
    last_forwarded_pose_frame = -1
    latest_manager_mode = 0
    policy_started = False
    last_button_sequence = 0
    dedicated_button_events_seen = False
    latest_left_hand_joints: np.ndarray | None = None
    latest_right_hand_joints: np.ndarray | None = None
    latest_hand_us = 0
    latest_hand_control_us = 0
    latest_hand_meta: dict[str, float] = {}
    grasp_engagement = {
        side: GraspEngagementDetector(
            command_delay_sec=max(0.0, float(args.grasp_engagement_delay_sec))
        )
        for side in ("left", "right")
    }
    latest_controller_left_axis = np.zeros(2, dtype=np.float64)
    latest_controller_right_axis = np.zeros(2, dtype=np.float64)
    latest_controller_y = False
    latest_controller_b = False
    latest_controller_state_us = 0
    latest_controller_sequence = 0
    joystick_mapper = PicoJoystickLocomotionMapper(
        PicoJoystickLocomotionConfig(
            deadzone=float(args.joystick_deadzone),
            forward_max_mps=float(args.joystick_forward_max_mps),
            backward_max_mps=float(args.joystick_backward_max_mps),
            lateral_max_mps=float(args.joystick_lateral_max_mps),
            minimum_nonzero_speed_mps=float(args.joystick_min_speed_mps),
            maximum_yaw_rate_rad_s=float(args.joystick_max_yaw_rate_rad_s),
            linear_acceleration_limit_mps2=float(args.joystick_linear_accel_mps2),
            linear_deceleration_limit_mps2=float(args.joystick_linear_decel_mps2),
            yaw_acceleration_limit_rad_s2=float(args.joystick_yaw_accel_rad_s2),
            input_stale_timeout_sec=float(args.joystick_stale_timeout_sec),
        )
    )
    joystick_mapper.reset(yaw_from_quaternion_wxyz(latest_robot_base_quat))
    # Keep the most recently dispatched tracking command so A+X can freeze
    # the root planner input while the arms transition to Q_READY.
    last_tracking_locomotion = joystick_mapper.idle(
        yaw_rad=yaw_from_quaternion_wxyz(latest_robot_base_quat),
        reason="startup_idle",
    )
    wait_locomotion: SonicLocomotionCommand | None = None
    phase_mode = "none"
    phase_sync_error_rad = float("nan")
    last_arm_ik_solve_ms = float("nan")
    last_arm_ik_error_m = float("nan")
    last_arm_ik_rot_error_rad = float("nan")
    last_arm_ik_status = "disabled"
    arm_ik_cached_revision = -1
    arm_ik_cached_q_ref: np.ndarray | None = None
    arm_ik_cached_tau_ff: np.ndarray | None = None
    arm_ik_cached_feasible = False
    arm_ik_cached_source = "none"
    last_ipopt_q_ref: np.ndarray | None = None
    last_ipopt_tau_ff: np.ndarray | None = None
    blended_q: np.ndarray | None = None
    last_plan_q: np.ndarray | None = None
    qp_reference_enabled = not args.qp_reference_disabled
    next_mode_command = float("inf")
    last_tick = time.monotonic()
    last_log = 0.0
    period = 1.0 / max(args.fps, 1.0e-6)
    keyboard = KeyboardController(enabled=not args.no_keyboard)
    outer_waist = OuterWaistPlanner(args)
    shared_config = SharedControlConfig(
        stale_after_s=float(args.shared_stale_after_sec),
        d_max_m=float(args.shared_d_max_m),
        waist_mode=str(args.shared_waist_mode),
        orientation_enter_confidence=float(args.orientation_enter_confidence),
        orientation_exit_confidence=float(args.orientation_exit_confidence),
        orientation_ema_alpha=float(args.orientation_ema_alpha),
        orientation_max_authority_rate_per_s=float(args.orientation_max_authority_rate_per_s),
        orientation_max_angle_rad=float(args.orientation_gate_max_angle_rad),
        orientation_gate_taper_start_fraction=float(args.orientation_gate_taper_start_fraction),
    )
    position_reference = {side: SecondOrderPositionReference(
        zeta=shared_config.zeta_released, omega_n=shared_config.omega_n_position_released,
        max_speed_mps=shared_config.max_speed_mps,
    ) for side in ("left", "right")}
    orientation_reference = {side: SecondOrderOrientationReference(
        zeta=shared_config.zeta_released, omega_n=shared_config.omega_n_orientation_released,
        max_rate_rps=shared_config.max_rate_rps,
    ) for side in ("left", "right")}
    hand_axis_local = np.asarray(args.hand_axis_local, dtype=np.float64).reshape(3)
    grasp_center_local = np.asarray(args.grasp_center_local, dtype=np.float64).reshape(3)
    hand_profile = hand_frame_profile(args.end_effector_model)
    grasp_center_wrist_offset = {
        side: np.asarray(hand_profile["translation"][side], dtype=np.float64)
        + np.asarray(hand_profile["rotation"][side], dtype=np.float64)
        @ grasp_center_local
        for side in ("left", "right")
    }
    # Use the calibrated wrist-yaw -> Dex1 hand-base +X translation for both
    # hands.  Any explicit CLI value remains a deliberate hardware override.
    for side in ("left", "right"):
        grasp_center_wrist_offset[side][0] = float(
            args.grasp_center_wrist_x_offset_m
        )
    latest_shared_result = None
    orientation_gate = {
        side: OrientationTargetGate(
            max_angle_rad=shared_config.orientation_max_angle_rad,
            max_step_rate_rps=shared_config.orientation_max_step_rate_rps,
            accept_dwell=shared_config.orientation_accept_dwell,
            taper_start_fraction=shared_config.orientation_gate_taper_start_fraction,
        )
        for side in ("left", "right")
    }
    orientation_authority = {
        side: OrientationAuthority(
            enter_confidence=shared_config.orientation_enter_confidence,
            exit_confidence=shared_config.orientation_exit_confidence,
            ema_alpha=shared_config.orientation_ema_alpha,
            max_authority_rate_per_s=shared_config.orientation_max_authority_rate_per_s,
        )
        for side in ("left", "right")
    }
    # Y/B is a dead-man switch for the left/right belief blend.
    shared_selected_hand = "none"
    shared_hand_selection_revision = 0
    next_shared_hand_publish = 0.0
    # Per-hand raw operator command history: (position, rotation, tick_us,
    # filtered angular velocity).  This is intentionally separate from the
    # assisted second-order reference history.
    command_history = {side: None for side in ("left", "right")}
    log(
        "PICO absolute waist-frame controller targets enabled: "
        f"topic={args.pico_vr_endpoint}/vr_pose_debug "
        "mode=head_position, anchor=none, gain=none; waiting for Q_READY/A+X"
    )
    log(
        "teleop second_order_reference="
        f"zeta=armed/released={shared_config.zeta_armed:.2f}/{shared_config.zeta_released:.2f} armed_pos/orient="
        f"{shared_config.omega_n_position_armed:.1f}/"
        f"{shared_config.omega_n_orientation_armed:.1f} "
        f"hand_model={args.end_effector_model} "
        f"axis={np.round(hand_axis_local, 4).tolist()} "
        f"grasp_local={np.round(grasp_center_local, 4).tolist()}"
    )

    def resync_references(reason: str) -> None:
        """Re-seed only on session/state discontinuities, never Y/B edges."""
        q = latest_robot_q if latest_robot_q is not None else last_ipopt_or_measured_hold()
        left_tf, right_tf = arm_nonlinear_ik.wrist_transforms_from_q29(q)
        for side, transform in (("left", left_tf), ("right", right_tf)):
            position_reference[side].reset(transform[:3, 3])
            orientation_reference[side].reset(transform[:3, :3])
            orientation_gate[side].reset()
            orientation_authority[side].reset()
        log(f"reference resync ({reason})")
    wrist_fk: MujocoWristFk | None = None
    try:
        wrist_fk = MujocoWristFk(args.model_xml)
        log(f"loaded MuJoCo FK model for plan completion: {args.model_xml}")
    except Exception as exc:
        log(f"WARNING: plan completion FK disabled: {exc}")

    def solve_authoritative_ipopt_transforms(
        left_target_tf: np.ndarray,
        right_target_tf: np.ndarray,
    ):
        """Solve authoritative arm joints from absolute waist-frame PICO targets."""

        nonlocal last_ipopt_q_ref, last_ipopt_tau_ff
        nonlocal last_arm_ik_solve_ms, last_arm_ik_error_m
        nonlocal last_arm_ik_rot_error_rad, last_arm_ik_status

        if arm_nonlinear_ik is None:
            raise RuntimeError("CasADi/IPOPT arm solver is not initialized")
        if latest_robot_q is not None:
            q_seed = latest_robot_q.copy()
        elif last_ipopt_q_ref is not None:
            q_seed = last_ipopt_q_ref.copy()
        else:
            q_seed = np.zeros(29, dtype=np.float64)
        q_seed[:15] = 0.0
        q_smooth = (
            last_ipopt_q_ref.copy()
            if last_ipopt_q_ref is not None
            else q_seed.copy()
        )
        q_smooth[:15] = 0.0
        result = arm_nonlinear_ik.solve_bimanual_targets(
            left_target_tf=np.asarray(left_target_tf, dtype=np.float64),
            right_target_tf=np.asarray(right_target_tf, dtype=np.float64),
            q_seed29=q_seed,
            q_smooth29=q_smooth,
            base_quaternion_wxyz=latest_robot_base_quat,
        )
        last_arm_ik_solve_ms = float(result.solve_time_ms)
        last_arm_ik_error_m = float(result.position_error_m)
        last_arm_ik_rot_error_rad = float(result.rotation_error_rad)
        last_arm_ik_status = (
            f"{result.message} "
            f"L/R={100.0 * result.left_position_error_m:.1f}/"
            f"{100.0 * result.right_position_error_m:.1f}cm "
            f"ori_w={result.left_orientation_weight_scale:.3f}/"
            f"{result.right_orientation_weight_scale:.3f} "
            f"projected={int(result.target_projected)} "
            f"post_rate={int(result.postprocess_rate_limited)} "
            f"post_scale={result.postprocess_motion_scale:.2f} "
            f"post_step={result.postprocess_max_step_rad:.3f}rad"
        )
        if result.success:
            result.q_ref[:15] = 0.0
            # Lower body and waist remain SONIC-owned. Only arm RNEA torque is
            # carried with the authoritative IPOPT arm target.
            result.tau_ff_ref[:12] = 0.0
            result.tau_ff_ref[12:15] = 0.0
            last_ipopt_q_ref = result.q_ref.copy()
            last_ipopt_tau_ff = result.tau_ff_ref.copy()
        return result

    def last_ipopt_or_measured_hold() -> np.ndarray:
        if last_ipopt_q_ref is not None:
            hold = last_ipopt_q_ref.copy()
        elif latest_robot_q is not None:
            hold = latest_robot_q.copy()
        else:
            hold = np.zeros(29, dtype=np.float64)
        hold[:15] = 0.0
        return hold

    def ipopt_visual_hands_from_transforms(
        left_wrist_tf: np.ndarray,
        right_wrist_tf: np.ndarray,
        *,
        target_source: str,
    ) -> dict[str, dict]:
        """Encode the pelvis-frame wrist poses currently held by IPOPT.

        The viewer must retain an explicit target even while PICO is waiting
        for its first fresh body frame.  In those states the direct controller
        holds Q_READY or the last IPOPT solution, both of which still have a
        well-defined wrist FK pose.
        """
        return {
            side: {
                "ipopt_wrist_target_m": transform[:3, 3].astype(float).tolist(),
                "ipopt_wrist_rotation": transform[:3, :3].reshape(9).astype(float).tolist(),
                "target_source": target_source,
            }
            for side, transform in (
                ("left", np.asarray(left_wrist_tf, dtype=np.float64).reshape(4, 4)),
                ("right", np.asarray(right_wrist_tf, dtype=np.float64).reshape(4, 4)),
            )
        }

    def remember_pico_hand_fields(raw: bytes, topic: str, rx_us: int) -> None:
        nonlocal latest_left_hand_joints, latest_right_hand_joints
        nonlocal latest_hand_us, latest_hand_meta
        try:
            fields = parse_packed_fields(raw, topic)
        except Exception:
            return
        left, right, meta = hand_joints_from_fields(fields)
        if left is None and right is None:
            return
        if left is not None:
            latest_left_hand_joints = left
        if right is not None:
            latest_right_hand_joints = right
        latest_hand_meta.update(meta)
        latest_hand_us = int(rx_us)

    def current_pico_hand_joints(tick_us: int) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        fresh = (
            latest_hand_us > 0
            and (tick_us - latest_hand_us) <= max(0.0, args.pico_hand_timeout_sec) * 1.0e6
        )
        # Preserve the last explicit open/close command over a transient gap.
        # Returning None is serialized as all-zero/open and loses a binary grip
        # command exactly when PICO changes stream mode.
        return latest_left_hand_joints, latest_right_hand_joints, fresh

    def pico_grip_active(side: str, tick_us: int) -> bool:
        """Return the latest valid binary close request from Pico controls."""
        control_age_us = tick_us - latest_hand_control_us
        pose_age_us = tick_us - latest_hand_us
        timeout_us = max(0.0, args.pico_hand_timeout_sec) * 1.0e6
        if latest_hand_control_us <= 0 or control_age_us > timeout_us:
            # The pose stream remains a compatibility fallback only.  The
            # normal route is pico_controller_state at 100 Hz.
            if latest_hand_us <= 0 or pose_age_us > timeout_us:
                return False
        return bool(max(
            latest_hand_meta.get(f"{side}_trigger", 0.0),
            latest_hand_meta.get(f"{side}_grip", 0.0),
        ) >= 0.5)

    def shared_target_available(tick_us: int, hand: str | None = None) -> bool:
        payload = static_target_payload_for_hand(latest_static_target, hand or "")
        if payload is None:
            return False
        if not bool(payload.get("target_valid", True)):
            return False
        try:
            position = np.asarray(
                payload.get(
                    "position", payload.get("target_position", [])
                ),
                dtype=np.float64,
            ).reshape(-1)
        except Exception:
            return False
        if position.shape[0] != 3 or not np.isfinite(position).all():
            return False
        if args.static_target_hold_sec <= 0.0:
            return True
        return (tick_us - latest_static_target_us) <= args.static_target_hold_sec * 1.0e6

    def control_q_from_gmr(q: np.ndarray) -> np.ndarray:
        return frozen_planner_q29(
            q,
            freeze_lower_body=bool(args.freeze_planner_lower_body),
            freeze_waist=bool(args.freeze_planner_waist),
        )

    def inactive_hold_q() -> np.ndarray | None:
        if latest_robot_q is None:
            return None
        return frozen_planner_q29(
            latest_robot_q,
            freeze_lower_body=bool(args.freeze_planner_lower_body),
            freeze_waist=bool(args.freeze_planner_waist),
        )

    def planner_mode_for_current_state() -> bool:
        return True

    def apply_state_request(requested: str, source: str, tick_us: int | None = None) -> None:
        nonlocal current_state, next_mode_command, blended_q
        nonlocal phase_mode, phase_sync_error_rad
        nonlocal shared_selected_hand, shared_hand_selection_revision
        if tick_us is None:
            tick_us = now_us()
        if requested == "auto_planning":
            log(f"{source} auto_planning alias -> teleop (Y/B gates belief)")
            requested = "teleop"
        if requested not in VALID_STATES:
            log(f"ignored invalid {source} state command: {requested}")
            return
        previous_state = current_state
        if requested != current_state:
            log(f"{source} state command: {current_state} -> {requested}")
            joystick_mapper.reset(
                yaw_from_quaternion_wxyz(latest_robot_base_quat)
            )
        else:
            log(f"{source} state command: staying in {requested}")
        current_state = requested
        if requested in {"teleop", "intention"}:
            resync_references("state_change")
            if shared_selected_hand != "none":
                shared_selected_hand = "none"
                shared_hand_selection_revision += 1
            phase_mode = "none"
            phase_sync_error_rad = float("nan")
            # q/e are PATH reference states, not SONIC planner-mode changes.
            # Re-sending planner=true makes ZMQManager reinitialize its
            # heading/reference state and can destabilize an otherwise idle
            # robot during a pure shared-control transition.
            next_mode_command = float("inf")
        else:
            next_mode_command = float("inf")
            if requested == "damped_return":
                resync_references("damped_return")

    def set_pico_tracking_armed(enabled: bool, source: str) -> None:
        """Switch between live PICO tracking and fixed-FK(Q_READY) IPOPT."""

        nonlocal pico_tracking_armed
        nonlocal pico_arm_pending
        nonlocal blended_q, wait_locomotion
        enabled = bool(enabled)
        if enabled:
            ready_error_rad = (
                float(
                    np.max(
                        np.abs(
                            np.asarray(latest_robot_q, dtype=np.float64)[15:29]
                            - Q_READY[15:29]
                        )
                    )
                )
                if latest_robot_q is not None
                else float("inf")
            )
            robot_ready = (
                latest_robot_q is not None
                and ready_error_rad < 0.10
            )
            if not robot_ready:
                pico_arm_pending = True
                log(
                    f"{source}: A+X arm pending until Q_READY "
                    f"(max_arm_error={ready_error_rad:.3f}rad, threshold=0.100rad)"
                )
                return
            if latest_wrist_targets is None:
                pico_arm_pending = True
                log(f"{source}: A+X arm pending until a valid Pico wrist frame arrives")
                return
            pico_arm_pending = False
        if enabled == pico_tracking_armed:
            log(
                f"{source}: PICO tracking already "
                f"{'armed' if enabled else 'disarmed'}"
            )
            return
        blended_q = None
        pico_tracking_armed = enabled
        pico_arm_pending = False
        resync_references("pico_tracking_armed" if enabled else "pico_tracking_disarmed")
        if enabled:
            # Entering tracking must not integrate the stick displacement that
            # occurred during the fixed-arm transition. Start its rate limiter
            # from the actual base yaw, then react to only subsequent samples.
            joystick_mapper.reset(
                yaw_from_quaternion_wxyz(latest_robot_base_quat)
            )
            wait_locomotion = None
            log(
                f"{source}: teleop_tracking armed after Q_READY confirmation; "
                "PICO wrist targets re-enabled and joystick reset"
            )
        else:
            # Preserve the last planner command exactly while moving both arms
            # to Q_READY. Do not clear IPOPT's previous solution: it is the
            # next solve's warm start and prevents a numerical discontinuity.
            wait_locomotion = copy_locomotion_command(
                last_tracking_locomotion,
                reason="wait_ready_latched",
            )
            log(
                f"{source}: teleop_wait_ready active; fixed FK(Q_READY) "
                "wrist SE(3), PICO arm targets ignored, root command latched"
            )

    def try_complete_pico_arm_request() -> None:
        if not pico_arm_pending or pico_tracking_armed:
            return
        if latest_robot_q is None or latest_wrist_targets is None:
            return
        ready_error_rad = float(
            np.max(np.abs(np.asarray(latest_robot_q, dtype=np.float64)[15:29] - Q_READY[15:29]))
        )
        if ready_error_rad < 0.10:
            set_pico_tracking_armed(True, "Pico A+X pending")

    def set_qp_reference_enabled(enabled: bool, source: str) -> None:
        nonlocal qp_reference_enabled
        qp_reference_enabled = True
        if not enabled:
            log(
                f"{source} arm solver disable ignored: "
                "CasADi/IPOPT owns both arms in every PATH mode"
            )
        else:
            log(f"{source} CasADi/IPOPT arm reference: staying ON")

    def set_planner_lower_body_idle(enabled: bool, source: str) -> None:
        nonlocal planner_lower_body_idle, next_mode_command
        planner_lower_body_idle = True
        if not enabled:
            log(
                f"{source} raw-SMPL SONIC mode request ignored: "
                "q/e keep one planner-conditioning mode"
            )
        else:
            log(f"{source} SONIC planner conditioning: staying ON")
        send_mode_command(sonic, planner=True)
        next_mode_command = float("inf")

    if keyboard.enabled:
        log(
            "keyboard hotkeys enabled: w=intention, r=damped_return, "
            "p=IPOPT arm disable rejected, "
            "m=SONIC mode toggle ignored (planner conditioning is invariant)"
        )
    else:
        log("keyboard hotkeys disabled; use tcp://*:5563/path_state_command")
    log(f"QP reference gate initial state: {'ON' if qp_reference_enabled else 'OFF'}")
    log(
        "SONIC lower-body source initial state: "
        f"{'zero-root planner' if planner_lower_body_idle else 'raw SMPL'}; "
        f"freeze_lower={bool(args.freeze_planner_lower_body)} "
        f"freeze_waist={bool(args.freeze_planner_waist)} "
        "root=idle/movement=0/speed=default/height=default"
    )
    log(
        "PICO joystick locomotion: "
        f"{'q-only enabled' if args.joystick_locomotion else 'disabled'}; "
        f"deadzone={args.joystick_deadzone:.2f} "
        f"v_fwd/rev/lat={args.joystick_forward_max_mps:.2f}/"
        f"{args.joystick_backward_max_mps:.2f}/"
        f"{args.joystick_lateral_max_mps:.2f}m/s "
        f"v_min={args.joystick_min_speed_mps:.2f}m/s "
        f"yaw_max={args.joystick_max_yaw_rate_rad_s:.2f}rad/s "
        f"stale={args.joystick_stale_timeout_sec:.2f}s "
        "mode=SLOW_WALK height=-1"
    )

    next_deadline = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            tick_us = now_us()
            dt = max(1.0e-4, now - last_tick)
            last_tick = now

            # Raw controller state is conflated independently of discrete
            # button events.  This prevents an IPOPT-heavy tick from replaying
            # stale locomotion samples out of a shared queue.
            try:
                controller_raw = controller_sub.recv(zmq.NOBLOCK)
                controller = unpack_prefixed_msgpack(
                    controller_raw, args.controller_state_topic
                )
                controller_sequence = int(controller.get("sequence", 0))
                left_axis = np.asarray(
                    controller.get("left_axis", []), dtype=np.float64
                ).reshape(-1)
                right_axis = np.asarray(
                    controller.get("right_axis", []), dtype=np.float64
                ).reshape(-1)
                controller_hand_meta: dict[str, float] = {}
                for key in ("left_trigger", "right_trigger", "left_grip", "right_grip"):
                    try:
                        value = float(controller.get(key, float("nan")))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        controller_hand_meta[key] = value
                if controller_hand_meta:
                    latest_hand_meta.update(controller_hand_meta)
                    latest_hand_control_us = tick_us
                if (
                    controller_sequence > latest_controller_sequence
                    and left_axis.size >= 2
                    and right_axis.size >= 2
                    and np.isfinite(left_axis[:2]).all()
                    and np.isfinite(right_axis[:2]).all()
                ):
                    latest_controller_left_axis = left_axis[:2].copy()
                    latest_controller_right_axis = right_axis[:2].copy()
                    latest_controller_y = bool(controller.get("y", False))
                    latest_controller_b = bool(controller.get("b", False))
                    latest_controller_state_us = tick_us
                    latest_controller_sequence = controller_sequence
            except zmq.Again:
                pass

            # ABXY has a dedicated 100Hz event queue so pose/FK/IK work cannot
            # delay mode commands. The timestamps reach g1_debug unchanged.
            try:
                while True:
                    event_raw = button_sub.recv(zmq.NOBLOCK)
                    if not starts(event_raw, args.button_topic):
                        continue
                    event = unpack_prefixed_msgpack(event_raw, args.button_topic)
                    sequence = int(event.get("sequence", 0))
                    if sequence <= last_button_sequence:
                        continue
                    last_button_sequence = sequence
                    dedicated_button_events_seen = True
                    event_name = str(event.get("event", ""))
                    if event_name not in {
                        "start_combo", "ax", "by", "select_left", "select_right",
                    }:
                        continue
                    bridge_receive_us = now_us()
                    start = False
                    stop = False
                    planner = planner_mode_for_current_state()
                    if event_name == "start_combo":
                        start = not policy_started
                        stop = policy_started
                        planner = True
                        policy_started = start
                        set_pico_tracking_armed(False, "Pico start/stop")
                        if stop:
                            current_state = "teleop"
                            joystick_mapper.reset(
                                yaw_from_quaternion_wxyz(latest_robot_base_quat)
                            )
                    elif event_name == "ax":
                        set_pico_tracking_armed(
                            not (pico_tracking_armed or pico_arm_pending), "Pico A+X"
                        )
                        planner = True
                    elif event_name == "by":
                        planner = True
                    elif event_name == "select_left":
                        log("Pico Y: hold to enable left shared-control")
                    elif event_name == "select_right":
                        log("Pico B: hold to enable right shared-control")
                    # Button edges now drive PATH state only. They must not
                    # reach a concurrently running SONIC policy.
            except zmq.Again:
                pass

            try:
                while True:
                    raw = pico_command.recv(zmq.NOBLOCK)
                    # The Pico manager emits a legacy command after handling
                    # the same physical edge. Once the dedicated button stream
                    # is alive, forwarding that command would apply A+X twice.
                    if dedicated_button_events_seen:
                        continue
                    flags = command_flags(raw)
                    if flags is not None:
                        start, stop, planner_requested = flags
                        if start:
                            policy_started = True
                        if stop:
                            policy_started = False
                            current_state = "teleop"
                            resync_references("pico_stop")
                            set_pico_tracking_armed(False, "Pico stop")
                            continue

                        if planner_requested:
                            if not start:
                                set_pico_tracking_armed(
                                    not (pico_tracking_armed or pico_arm_pending),
                                    "Pico A+X legacy",
                                )
                            continue

                    # Non-button Pico command packets belong to the retired
                    # SONIC ZMQ protocol and are intentionally discarded.
            except zmq.Again:
                pass

            # Native PICO controller tracking is published independently of
            # SONIC's stream mode and remains available in q/e planner modes.
            try:
                raw = pico_vr_body.recv(zmq.NOBLOCK)
                remember_pico_hand_fields(raw, "vr_pose_debug", tick_us)
                targets = wrist_targets_from_message(
                    raw, "vr_pose_debug", tick_us
                )
                if targets is not None:
                    if targets.session_id != latest_pico_session_id:
                        latest_pico_session_id = targets.session_id
                        last_ipopt_q_ref = None
                        last_ipopt_tau_ff = None
                        blended_q = None
                        log(
                            "PICO absolute-target session="
                            f"{latest_pico_session_id}; reset IPOPT warm start"
                        )
                    latest_wrist_targets = targets
                    latest_pico_body_receive_us = tick_us
            except zmq.Again:
                pass

            # Pose remains subscribed only for hand fields and legacy recording.
            # Arm targets never fall back from the independent calibrated
            # vr_pose_debug stream to this mode-dependent channel.
            try:
                raw = pico_pose.recv(zmq.NOBLOCK)
                remember_pico_hand_fields(raw, "pose", tick_us)
                sample = smpl_sample_from_pose(raw, tick_us)
                if sample is not None:
                    latest_smpl_joints, latest_smpl_source_us, latest_smpl_frame = sample
                    latest_pose_raw = raw
                    latest_pose_us = tick_us
            except zmq.Again:
                pass

            try:
                raw = pico_planner.recv(zmq.NOBLOCK)
                remember_pico_hand_fields(raw, "planner", tick_us)
            except zmq.Again:
                pass

            try:
                raw = pico_manager.recv(zmq.NOBLOCK)
                try:
                    manager_fields = parse_packed_fields(raw, "manager_state")
                    manager_modes = np.asarray(
                        manager_fields.get("stream_mode", []), dtype=np.int32
                    ).reshape(-1)
                    if manager_modes.size:
                        latest_manager_mode = int(manager_modes[-1])
                        policy_started = latest_manager_mode != 0
                except Exception:
                    pass
            except zmq.Again:
                pass

            try:
                while True:
                    incoming_target = unpack_topic(
                        static_target_sub.recv(zmq.NOBLOCK), args.static_target_topic)
                    incoming_uid = str(incoming_target.get("goal_uid", ""))
                    target_link = str(incoming_target.get("target_link", "left_wrist_yaw_link"))
                    target_side = target_side_from_link(target_link)
                    target_valid = bool(incoming_target.get("target_valid", True))
                    if target_valid and not static_target_frame_is_supported(
                        incoming_target
                    ):
                        incoming_target = dict(incoming_target)
                        incoming_target["target_valid"] = False
                        incoming_target["failure_reason"] = (
                            "unsupported_coordinate_frame:"
                            + str(
                                incoming_target.get(
                                    "target_position_frame",
                                    incoming_target.get("coordinate_frame", "missing"),
                                )
                            )
                        )
                        target_valid = False
                    position_value = incoming_target.get(
                        "position", incoming_target.get("target_position", None)
                    )
                    if (not target_valid) or position_value is None:
                        was_valid_target = (
                            latest_static_target is not None
                            and bool(latest_static_target.get("target_valid", True))
                        )
                        latest_static_target = incoming_target
                        latest_static_target_us = tick_us
                        static_target_revision += 1
                        if was_valid_target or (
                            incoming_uid and incoming_uid != latest_static_target_uid
                        ):
                            latest_static_target_uid = incoming_uid
                            resync_references("static_target_update")
                        continue
                    position = np.asarray(position_value, dtype=np.float64).reshape(-1)
                    if position.shape[0] == 3 and np.isfinite(position).all():
                        previous_object_id = str(
                            latest_static_target.get("object_id", "")
                            if latest_static_target is not None
                            else ""
                        )
                        incoming_object_id = str(
                            incoming_target.get("object_id", "")
                        )
                        explicit_reset = bool(
                            incoming_target.get("new_goal", False)
                            or incoming_target.get("reset_hold", False)
                        )
                        is_new_goal = (
                            latest_static_target is None
                            or explicit_reset
                            or (
                                incoming_object_id
                                and previous_object_id
                                and incoming_object_id != previous_object_id
                            )
                        )
                        latest_static_target = incoming_target
                        latest_static_target_us = tick_us
                        static_target_revision += 1
                        if is_new_goal:
                            latest_static_target_uid = incoming_uid
                            static_target_plan_id += 1
                            resync_references("static_target_replace")
                            if (
                                args.static_target_auto_enter
                                and current_state != "auto_planning"
                            ):
                                apply_state_request(
                                    "auto_planning", "static EE target", tick_us
                                )
            except zmq.Again:
                pass

            try:
                while True:
                    latest_robot_state = unpack_prefixed_msgpack(
                        robot_state.recv(zmq.NOBLOCK), args.robot_state_topic)
                    q_robot = latest_q_from_robot_state(latest_robot_state)
                    if q_robot is not None:
                        latest_robot_q = q_robot
                        latest_robot_state_us = tick_us
                    base_quat = latest_base_quat_from_robot_state(
                        latest_robot_state
                    )
                    if base_quat is not None:
                        latest_robot_base_quat = base_quat
                    body_dq = latest_vector_from_robot_state(
                        latest_robot_state,
                        ("body_dq", "body_dq_measured", "state_body_dq"),
                        29,
                    )
                    if body_dq is not None:
                        latest_robot_dq = body_dq
                    base_accel = latest_vector_from_robot_state(
                        latest_robot_state,
                        (
                            "base_accel_measured",
                            "base_accel",
                            "state_base_accel",
                        ),
                        3,
                    )
                    if base_accel is not None:
                        latest_robot_base_accel = base_accel
                    base_gyro = latest_vector_from_robot_state(
                        latest_robot_state,
                        (
                            "base_ang_vel",
                            "base_gyro",
                            "state_base_gyro",
                        ),
                        3,
                    )
                    if base_gyro is not None:
                        latest_robot_base_gyro = base_gyro
            except zmq.Again:
                pass

            # On real hardware this is authoritative and overrides the legacy
            # ZMQ debug sample when both happen to be present.
            if direct_state is not None:
                lcm_state, lcm_state_received_us = direct_state.latest()
                if (
                    lcm_state is not None
                    and tick_us - lcm_state_received_us <= 200_000
                ):
                    latest_robot_q = lcm_state.q.copy()
                    latest_robot_dq = lcm_state.dq.copy()
                    latest_robot_base_quat = lcm_state.base_quat_wxyz.copy()
                    latest_robot_state_us = int(lcm_state_received_us)

            latest_gravity_direction = gravity_estimator.update(
                latest_robot_base_quat,
                latest_robot_base_accel,
                latest_robot_base_gyro,
                dt,
            )

            try:
                while True:
                    command = unpack_topic(state_sub.recv(zmq.NOBLOCK), args.state_command_topic)
                    if "qp_reference_enabled" in command:
                        set_qp_reference_enabled(
                            bool(command.get("qp_reference_enabled")),
                            "external",
                        )
                    if "toggle_qp_reference" in command and command.get("toggle_qp_reference"):
                        set_qp_reference_enabled(not qp_reference_enabled, "external")
                    if "teleop_gmr_arms_only" in command:
                        set_planner_lower_body_idle(
                            bool(command.get("teleop_gmr_arms_only")),
                            "external",
                        )
                    if "planner_lower_body_idle" in command:
                        set_planner_lower_body_idle(
                            bool(command.get("planner_lower_body_idle")),
                            "external",
                        )
                    if (
                        "toggle_teleop_gmr_arms_only" in command
                        and command.get("toggle_teleop_gmr_arms_only")
                    ):
                        set_planner_lower_body_idle(
                            not planner_lower_body_idle, "external"
                        )
                    if "state" in command:
                        requested = str(command.get("state", current_state))
                        apply_state_request(requested, "external", tick_us)
            except zmq.Again:
                pass

            keyboard_command = keyboard.poll_command()
            if keyboard_command == KeyboardController.TOGGLE_QP:
                set_qp_reference_enabled(not qp_reference_enabled, "keyboard")
            elif keyboard_command == KeyboardController.TOGGLE_TELEOP_SOURCE:
                set_planner_lower_body_idle(
                    not planner_lower_body_idle, "keyboard"
                )
            elif keyboard_command is not None:
                apply_state_request(keyboard_command, "keyboard", tick_us)

            released_pose_frame = -1
            pose_to_send: bytes | None = None
            smpl_age_sec = (
                (tick_us - latest_pose_us) / 1.0e6
                if latest_pose_us > 0
                else float("inf")
            )
            smpl_fresh = (
                latest_smpl_joints is not None
                and smpl_age_sec <= 0.20
            )
            pico_body_age_sec = (
                (tick_us - latest_pico_body_receive_us) / 1.0e6
                if latest_pico_body_receive_us > 0
                else float("inf")
            )
            pico_body_fresh = (
                pico_tracking_armed
                and latest_wrist_targets is not None
                and pico_body_age_sec
                <= max(0.001, float(args.pico_body_gap_hold_ms) / 1000.0)
            )
            try_complete_pico_arm_request()

            left_hand_cmd, right_hand_cmd, hand_fresh = current_pico_hand_joints(tick_us)
            grasp_commanded = {
                side: pico_grip_active(side, tick_us) for side in ("left", "right")
            }
            # No measured Dex1 jaw gap reaches this bridge yet.  ``None``
            # selects the documented close-command delay fallback.
            jaw_gap_m = {side: None for side in ("left", "right")}
            grasp_engaged = {
                side: grasp_engagement[side].update(
                    commanded_closed=grasp_commanded[side],
                    jaw_gap_m=jaw_gap_m[side],
                    dt=dt,
                )
                for side in ("left", "right")
            }
            controller_age_sec = (
                (tick_us - latest_controller_state_us) / 1.0e6
                if latest_controller_state_us > 0
                else float("inf")
            )
            controller_fresh = (
                latest_controller_state_us > 0
                and controller_age_sec
                <= float(args.joystick_stale_timeout_sec)
            )
            # Y/B are a dead-man arm selector in teleop.  Never retain the
            # selected arm after a release or controller-state timeout.
            requested_shared_hand = "none"
            if current_state == "teleop" and controller_fresh:
                if latest_controller_y and not latest_controller_b:
                    requested_shared_hand = "left"
                elif latest_controller_b and not latest_controller_y:
                    requested_shared_hand = "right"
            if requested_shared_hand != shared_selected_hand:
                shared_selected_hand = requested_shared_hand
                shared_hand_selection_revision += 1
                # Y/B only selects which target is tracked. Reference state
                # remains continuous across the button edge.
                log(
                    "shared-control hold "
                    + (
                        f"enabled for {shared_selected_hand} arm"
                        if shared_selected_hand != "none"
                        else "released"
                    )
                )
            current_base_yaw = yaw_from_quaternion_wxyz(
                latest_robot_base_quat
            )
            joystick_enabled = bool(
                args.joystick_locomotion
                and current_state == "teleop"
                and pico_tracking_armed
                and policy_started
                and controller_fresh
            )
            if current_state == "teleop" and not pico_tracking_armed:
                # teleop_wait_ready: freeze exactly the command that was last
                # accepted in teleop_tracking. Do not consume PICO stick
                # samples or advance the mapper while arm Q_READY is solved.
                if wait_locomotion is not None:
                    joystick_command = copy_locomotion_command(wait_locomotion)
                else:
                    joystick_command = fixed_idle_locomotion(
                        "wait_ready_fixed_heading"
                    )
            elif joystick_enabled:
                joystick_command = joystick_mapper.update(
                    latest_controller_left_axis,
                    latest_controller_right_axis,
                    dt,
                    current_yaw_rad=current_base_yaw,
                )
            else:
                if not args.joystick_locomotion:
                    joystick_reason = "disabled"
                elif current_state != "teleop":
                    joystick_reason = f"state_{current_state}"
                elif not policy_started:
                    joystick_reason = "policy_stopped"
                else:
                    joystick_reason = "stale"
                # State exit, policy stop, and stale input are hard safety
                # gates: zero immediately rather than applying the normal
                # deceleration ramp to an uncommanded motion.
                joystick_command = fixed_idle_locomotion(joystick_reason)
            if current_state == "teleop" and pico_tracking_armed:
                last_tracking_locomotion = copy_locomotion_command(
                    joystick_command,
                    reason="tracking_last",
                )
            static_target_age_sec = (
                (tick_us - latest_static_target_us) / 1.0e6
                if latest_static_target_us > 0
                else float("inf")
            )
            static_target_fresh = bool(
                latest_static_target is not None
                and static_target_age_sec <= max(0.0, args.static_target_timeout_sec)
            )
            static_target_usable = shared_target_available(tick_us)
            qp_payload: dict | None = None
            # Published only for the tick that produced these transforms.  Do
            # not retain an old command: a stale visual marker looks like a
            # tracking error to the operator.
            control_visual_hands: dict[str, dict] = {}
            if current_state in {"teleop", "intention"}:
                if planner_lower_body_idle:
                    pose_to_send = None
                if current_state == "teleop" and not pico_tracking_armed:
                    # A+X enters teleop_wait_ready.  It is deliberately a
                    # Cartesian IPOPT task, not a direct q=0 command: FK of
                    # Q_READY provides both fixed wrist positions and fixed
                    # orientations every 50 Hz.
                    ready_result = solve_authoritative_ipopt_transforms(
                        q_ready_left_wrist_tf,
                        q_ready_right_wrist_tf,
                    )
                    q_for_sonic = (
                        ready_result.q_ref.copy()
                        if ready_result.success
                        else last_ipopt_or_measured_hold()
                    )
                    q_for_sonic[:15] = 0.0
                    if ready_result.success:
                        control_visual_hands = ipopt_visual_hands_from_transforms(
                            q_ready_left_wrist_tf,
                            q_ready_right_wrist_tf,
                            target_source="teleop_wait_ready_q_ready",
                        )
                    else:
                        hold_left_tf, hold_right_tf = (
                            arm_nonlinear_ik.wrist_transforms_from_q29(q_for_sonic)
                        )
                        control_visual_hands = ipopt_visual_hands_from_transforms(
                            hold_left_tf,
                            hold_right_tf,
                            target_source="teleop_wait_ready_ipopt_hold",
                        )
                    if policy_started:
                        send_planner_command(
                            sonic,
                            q_for_sonic,
                            neutralize_waist=True,
                            freeze_lower_body=True,
                            left_hand_position=left_hand_cmd,
                            right_hand_position=right_hand_cmd,
                            locomotion=joystick_command,
                        )
                    qp_payload = ipopt_arm_sample_reference(
                        q_for_sonic,
                        state="teleop",
                        source="pico_wait_ready_fk",
                        control_mode="teleop_wait_ready",
                        feasible=bool(ready_result.success),
                        plan_id=0,
                        timestamp_us=tick_us,
                        tau_ff_ref=(
                            ready_result.tau_ff_ref
                            if ready_result.success
                            else (
                                last_ipopt_tau_ff
                                if last_ipopt_tau_ff is not None
                                else np.zeros(29, dtype=np.float64)
                            )
                        ),
                        bridge_timestamp_us=tick_us,
                        smpl_target_reason="fixed_fk_q_ready",
                        smpl_target_held=True,
                        smpl_left_target_position=q_ready_left_wrist_tf[:3, 3].tolist(),
                        smpl_right_target_position=q_ready_right_wrist_tf[:3, 3].tolist(),
                        q_prior_source="ipopt_fixed_q_ready_fk",
                        belief_enabled=False,
                        shared_control=False,
                        teleop_qp_bimanual=True,
                        teleop_waist_locked=True,
                        arm_ik_solve_ms=float(ready_result.solve_time_ms),
                        arm_ik_position_error_m=float(ready_result.position_error_m),
                        arm_ik_rotation_error_rad=float(
                            ready_result.rotation_error_rad
                        ),
                        arm_ik_status=str(last_arm_ik_status),
                    )
                elif (
                    (pico_body_fresh and latest_wrist_targets is not None)
                    or (
                        shared_selected_hand in {"left", "right"}
                        and shared_target_available(tick_us, shared_selected_hand)
                    )
                ):
                    requested_hand = shared_selected_hand
                    shared_selection_armed = requested_hand in {"left", "right"}
                    selected_target_payload = static_target_payload_for_hand(
                        latest_static_target, requested_hand
                    )
                    shared_target = (
                        ee_target_from_static_payload(
                            selected_target_payload,
                            received_us=latest_static_target_us,
                        )
                        if (
                            shared_selection_armed
                            and shared_target_available(tick_us, requested_hand)
                        )
                        else None
                    )
                    shared_orientation_source = "pico_operator"
                    if pico_body_fresh and latest_wrist_targets is not None:
                        left_target_tf = latest_wrist_targets.left_waist.copy()
                        right_target_tf = latest_wrist_targets.right_waist.copy()
                        human_source_timestamp_us = int(
                            latest_wrist_targets.source_timestamp_us
                        )
                        human_frame_index = int(latest_wrist_targets.frame_index)
                        human_wrist_source = "pico_live"
                    else:
                        fallback_q = (
                            latest_robot_q
                            if latest_robot_q is not None
                            else last_ipopt_or_measured_hold()
                        )
                        left_target_tf, right_target_tf = (
                            arm_nonlinear_ik.wrist_transforms_from_q29(fallback_q)
                        )
                        human_source_timestamp_us = int(tick_us)
                        human_frame_index = -1
                        human_wrist_source = "robot_fk_hold"
                    q_measured_for_hand = latest_robot_q if latest_robot_q is not None else last_ipopt_or_measured_hold()
                    measured_left_tf, measured_right_tf = arm_nonlinear_ik.wrist_transforms_from_q29(q_measured_for_hand)
                    measured_tf = {"left": measured_left_tf, "right": measured_right_tf}
                    human_tf = {"left": left_target_tf, "right": right_target_tf}
                    # The reference loop below overwrites human_tf in place.
                    # Belief must see the operator's raw command, not the
                    # assisted reference, or it recreates a self-confirming
                    # intent loop through the controlled robot motion.
                    command_tf = {
                        side: human_tf[side].copy() for side in ("left", "right")
                    }
                    command_features: dict[
                        str,
                        tuple[
                            np.ndarray,
                            np.ndarray,
                            np.ndarray | None,
                            np.ndarray | None,
                        ],
                    ] = {}
                    for side in ("left", "right"):
                        p_cmd = command_tf[side][:3, 3]
                        r_cmd = command_tf[side][:3, :3]
                        velocity = None
                        omega = None
                        previous = command_history[side]
                        if previous is not None:
                            p_prev, r_prev, t_prev, omega_prev = previous
                            delta = (tick_us - t_prev) / 1.0e6
                            if 1.0e-4 <= delta <= 0.5:
                                velocity = (p_cmd - p_prev) / delta
                                raw_omega = angular_velocity_from_rotations(
                                    r_prev, r_cmd, delta
                                )
                                if raw_omega is not None:
                                    omega = (
                                        raw_omega
                                        if omega_prev is None
                                        else 0.25 * raw_omega
                                        + 0.75 * omega_prev
                                    )
                        command_history[side] = (
                            p_cmd.copy(),
                            r_cmd.copy(),
                            int(tick_us),
                            None if omega is None else omega.copy(),
                        )
                        command_features[side] = (
                            p_cmd,
                            r_cmd,
                            velocity,
                            omega,
                        )
                    reference_diagnostics: dict[str, dict] = {}
                    for side in ("left", "right"):
                        belief_target = shared_target if shared_selection_armed and requested_hand == side else None
                        rotation_command = human_tf[side][:3, :3]
                        rotation_target = rotation_command
                        orientation_source = "operator"
                        zeta_ori = shared_config.zeta_released
                        omega_n_ori = shared_config.omega_n_orientation_released
                        gate_reason = "no_constraint"
                        gate_weight = 0.0
                        belief_weight = 0.0
                        final_orientation_weight = 0.0
                        current_rotation = orientation_reference[side].rotation if orientation_reference[side].rotation is not None else measured_tf[side][:3, :3]
                        if belief_target is not None and belief_target.has_orientation_constraint:
                            target_model = str(args.end_effector_model).lower()
                            position_point = str(belief_target.position_point or "")
                            if position_point.endswith("_grasp_center"):
                                candidate_model = position_point[: -len("_grasp_center")]
                                if candidate_model in HAND_AXIS_LOCAL:
                                    target_model = candidate_model
                            target_hand_axis = (
                                hand_axis_local
                                if hand_axis_local_override
                                else HAND_AXIS_LOCAL[target_model]
                            )
                            constrained = (
                                belief_target.orientation_frame_world
                                if belief_target.orientation_policy == "grasp_frame"
                                and belief_target.orientation_frame_world is not None
                                else rotation_target_from_relation(
                                    current_rotation,
                                    belief_target.orientation_axis_world,
                                    belief_target.orientation_relation,
                                    target_hand_axis,
                                )
                            )
                            gate = orientation_gate[side]
                            gate_weight = orientation_gate[side].authority(
                                constrained, current_rotation, dt
                            )
                            belief_weight = orientation_authority[side].update(
                                belief_target.orientation_confidence,
                                gate_ok=gate_weight > 0.0,
                                dt=dt,
                            )
                            final_orientation_weight = belief_weight * gate_weight
                            if constrained is not None and final_orientation_weight > 0.0:
                                rotation_target = slerp_so3(
                                    rotation_command, constrained, final_orientation_weight
                                )
                                orientation_source = f"belief_{belief_target.orientation_policy}"
                            gate_reason = gate.last_reason
                        else:
                            orientation_gate[side].authority(None, current_rotation, dt)
                            orientation_authority[side].update(0.0, gate_ok=False, dt=dt)
                        # Modulate reference dynamics continuously. The state
                        # itself is never blended or reset on authority edges.
                        zeta_ori = (
                            (1.0 - final_orientation_weight) * shared_config.zeta_released
                            + final_orientation_weight * shared_config.zeta_armed
                        )
                        omega_n_ori = (
                            (1.0 - final_orientation_weight) * shared_config.omega_n_orientation_released
                            + final_orientation_weight * shared_config.omega_n_orientation_armed
                        )
                        if grasp_engaged[side]:
                            # With the contact target suppressed, orientation
                            # is the sole assisted degree of freedom.
                            omega_n_ori = shared_config.omega_n_orientation_armed
                        r_ref = orientation_reference[side].step(
                            rotation_target, dt, measured=measured_tf[side][:3, :3],
                            omega_n=omega_n_ori, zeta=zeta_ori,
                        )
                        position_target = human_tf[side][:3, 3]
                        position_source = "operator"
                        zeta_pos = shared_config.zeta_released
                        omega_n_pos = shared_config.omega_n_position_released
                        if (
                            belief_target is not None
                            and belief_target.position_valid
                            and not grasp_engaged[side]
                        ):
                            goal = np.asarray(belief_target.position, dtype=np.float64).reshape(3)
                            is_canonical_grasp_center = (
                                belief_target.position_point == "canonical_grasp_center"
                                or belief_target.position_point.endswith("_grasp_center")
                            )
                            goal_wrist = (
                                goal - r_ref @ grasp_center_wrist_offset[side]
                                if is_canonical_grasp_center
                                else goal
                            )
                            if np.linalg.norm(goal_wrist - human_tf[side][:3, 3]) <= shared_config.d_max_m:
                                position_target = goal_wrist
                                position_source = "belief_contact"
                                zeta_pos = shared_config.zeta_armed
                                omega_n_pos = shared_config.omega_n_position_armed
                        elif grasp_engaged[side]:
                            position_source = "operator_grasp_engaged"
                        # Post-grasp position is operator-owned, while the
                        # orientation belief above remains active.  Do not
                        # reset the second-order position state here.
                        p_ref = position_reference[side].step(
                            position_target, dt, measured=measured_tf[side][:3, 3],
                            omega_n=omega_n_pos, zeta=zeta_pos,
                        )
                        # Measured lag is telemetry only. It must never freeze,
                        # resync, or down-weight a valid operator/belief target.
                        measured_lag_m = float(
                            np.linalg.norm(p_ref - measured_tf[side][:3, 3])
                        )
                        human_tf[side][:3, :3] = r_ref
                        human_tf[side][:3, 3] = p_ref
                        reference_diagnostics[side] = {
                            "armed": bool(belief_target is not None), "position_source": position_source,
                            "orientation_source": orientation_source, "omega_n_position": float(omega_n_pos),
                            "omega_n_orientation": float(omega_n_ori), "position_error_m": float(np.linalg.norm(p_ref-position_target)),
                            "orientation_error_rad": float(np.linalg.norm(rotation_vector(rotation_target @ r_ref.T))),
                            "reference_speed_mps": float(np.linalg.norm(position_reference[side].velocity)),
                            "reference_rate_rps": float(np.linalg.norm(orientation_reference[side].rate)),
                            "measured_lag_m": measured_lag_m,
                            "measured_lag_policy": "diagnostic_only",
                            "object_id": "" if belief_target is None else belief_target.object_id,
                            "relation": "" if belief_target is None else belief_target.orientation_relation,
                            "orientation_gate_reason": gate_reason,
                            "orientation_gate_rejects": int(orientation_gate[side].reject_count),
                            "orientation_gate_weight": float(gate_weight),
                            "orientation_belief_weight": float(belief_weight),
                            "orientation_authority_weight": float(final_orientation_weight),
                            "orientation_smoothed_confidence": float(orientation_authority[side].smoothed_confidence),
                            "zeta_position": float(zeta_pos),
                            "zeta_orientation": float(zeta_ori),
                            "grasp_commanded": bool(grasp_commanded[side]),
                            "grasp_engaged": bool(grasp_engaged[side]),
                            "jaw_gap_m": jaw_gap_m[side],
                            "grasp_state_source": "command_delay_fallback",
                        }
                    control_visual_hands = {
                        side: {
                            # Exact wrist-yaw Cartesian reference handed to
                            # solve_authoritative_ipopt_transforms below.
                            "ipopt_wrist_target_m": human_tf[side][:3, 3].tolist(),
                            "ipopt_wrist_rotation": human_tf[side][:3, :3].reshape(9).tolist(),
                            "command_wrist_position_m": p_cmd.tolist(),
                            "command_wrist_rotation": r_cmd.reshape(9).tolist(),
                            "command_velocity_mps": (
                                None if velocity is None else velocity.tolist()
                            ),
                            "command_omega_rps": (
                                None if omega is None else omega.tolist()
                            ),
                            # Compatibility marker retained for the v8 web
                            # viewer; it is derived from the same raw command.
                            "human_grasp_goal_m": (
                                command_tf[side][:3, 3]
                                + command_tf[side][:3, :3] @ grasp_center_local
                            ).tolist(),
                            "reference_grasp_goal_m": (
                                human_tf[side][:3, 3]
                                + human_tf[side][:3, :3] @ grasp_center_local
                            ).tolist(),
                            "measured_wrist_position_m": measured_tf[side][
                                :3, 3
                            ].tolist(),
                            "measured_wrist_rotation": measured_tf[side][
                                :3, :3
                            ].reshape(9).tolist(),
                            "armed": bool(shared_selected_hand == side),
                            "position_source": reference_diagnostics[side][
                                "position_source"
                            ],
                            "belief_position_applied": reference_diagnostics[side][
                                "position_source"
                            ].startswith("belief"),
                            "orientation_source": reference_diagnostics[side][
                                "orientation_source"
                            ],
                            "orientation_authority_weight": reference_diagnostics[
                                side
                            ]["orientation_authority_weight"],
                            "reference_speed_mps": reference_diagnostics[side][
                                "reference_speed_mps"
                            ],
                            "grasp_commanded": bool(grasp_commanded[side]),
                            "grasp_engaged": bool(grasp_engaged[side]),
                            "jaw_gap_m": jaw_gap_m[side],
                            "grasp_state_source": "command_delay_fallback",
                        }
                        for side, (
                            p_cmd,
                            r_cmd,
                            velocity,
                            omega,
                        ) in command_features.items()
                    }
                    left_target_tf, right_target_tf = human_tf["left"], human_tf["right"]
                    latest_shared_result = reference_diagnostics
                    ik_result = solve_authoritative_ipopt_transforms(
                        left_target_tf,
                        right_target_tf,
                    )
                    q_for_sonic = (
                        ik_result.q_ref.copy()
                        if ik_result.success
                        else last_ipopt_or_measured_hold()
                    )
                    q_for_sonic[:15] = 0.0
                    if planner_lower_body_idle:
                        pose_to_send = None
                        send_planner_command(
                            sonic,
                            q_for_sonic,
                            neutralize_waist=True,
                            freeze_lower_body=True,
                            left_hand_position=left_hand_cmd,
                            right_hand_position=right_hand_cmd,
                            locomotion=joystick_command,
                        )
                    source_timestamp_us = human_source_timestamp_us
                    if ik_result.success:
                        qp_payload = ipopt_arm_sample_reference(
                            ik_result.q_ref,
                            state=current_state,
                            source="pico_body_cartesian",
                            control_mode="teleop",
                            feasible=True,
                            plan_id=0,
                            timestamp_us=tick_us,
                            tau_ff_ref=ik_result.tau_ff_ref,
                            bridge_timestamp_us=tick_us,
                            smpl_source_timestamp_us=source_timestamp_us,
                            smpl_frame_index=human_frame_index,
                            smpl_target_reason="absolute_head_position",
                            smpl_target_held=False,
                            smpl_left_target_position=left_target_tf[:3, 3].tolist(),
                            smpl_right_target_position=right_target_tf[:3, 3].tolist(),
                            q_prior_source="ipopt_pico_controller_ee",
                            belief_enabled=bool(shared_selection_armed),
                            shared_control=bool(shared_selection_armed),
                            teleop_qp_bimanual=True,
                            teleop_waist_locked=False,
                            arm_ik_solve_ms=float(ik_result.solve_time_ms),
                            arm_ik_position_error_m=float(ik_result.position_error_m),
                            arm_ik_rotation_error_rad=float(ik_result.rotation_error_rad),
                            arm_ik_status=str(last_arm_ik_status),
                        )
                        if latest_shared_result is not None:
                            qp_payload.update(
                                # Deprecated constant retained for old record readers.
                                shared_alpha=0.0,
                                shared_selected_hand=str(shared_selected_hand),
                                shared_selection_armed=bool(shared_selection_armed),
                                shared_requested_hand=str(requested_hand),
                                shared_requested_target_link=str(
                                    selected_target_payload.get("target_link", "")
                                    if selected_target_payload else ""
                                ),
                                reference_left=latest_shared_result["left"],
                                reference_right=latest_shared_result["right"],
                                orientation_gate={
                                    side: {
                                        "accepted": latest_shared_result[side]["orientation_gate_reason"] == "ok",
                                        "reason": latest_shared_result[side]["orientation_gate_reason"],
                                        "reject_count": latest_shared_result[side]["orientation_gate_rejects"],
                                    }
                                    for side in ("left", "right")
                                },
                                reference_zeta=float(shared_config.zeta_armed if shared_selection_armed else shared_config.zeta_released),
                                shared_human_wrist_source=human_wrist_source,
                                shared_coordinate_frame=PICO_WAIST_FRAME,
                                shared_orientation_source="second_order_reference",
                                shared_target_quaternion_wxyz=(
                                    None
                                    if shared_target is None
                                    or shared_target.orientation_wxyz is None
                                    else shared_target.orientation_wxyz.tolist()
                                ),
                            )
                            if shared_target is not None:
                                target_tf = left_target_tf if requested_hand == "left" else right_target_tf
                                qp_payload.update(
                                    q_prior_source="ipopt_second_order_ee_pose",
                                     terminal_target_link=str(selected_target_payload.get("target_link", "") if selected_target_payload else ""),
                                     terminal_target_position=target_tf[:3, 3].tolist(),
                                     # Preserve the exact pose that was sent to
                                     # IPOPT.  Previously only position was
                                     # emitted, so an orientation/frame mismatch
                                     # could not be diagnosed from the solver
                                     # stream and consumers fell back to a
                                     # legacy quaternion.
                                     terminal_target_orientation_wxyz=rotation_matrix_to_quaternion_wxyz(
                                         target_tf[:3, :3]
                                     ).tolist(),
                                     terminal_target_transform=target_tf.reshape(-1).tolist(),
                                     terminal_target_frame="pelvis_identity_fk",
                                     target_source=str(
                                        latest_static_target.get("source", "sam3_tasc_belief")
                                        if latest_static_target else "none"
                                     ),
                                 )
                    else:
                        hold_q = last_ipopt_or_measured_hold()
                        hold_left_tf, hold_right_tf = (
                            arm_nonlinear_ik.wrist_transforms_from_q29(hold_q)
                        )
                        control_visual_hands = ipopt_visual_hands_from_transforms(
                            hold_left_tf,
                            hold_right_tf,
                            target_source="ipopt_failed_hold",
                        )
                        qp_payload = ipopt_arm_sample_reference(
                            hold_q,
                            state=current_state,
                            source="pico_body_cartesian",
                            control_mode="teleop",
                            feasible=True,
                            plan_id=0,
                            timestamp_us=tick_us,
                            tau_ff_ref=(
                                last_ipopt_tau_ff
                                if last_ipopt_tau_ff is not None
                                else np.zeros(29, dtype=np.float64)
                            ),
                            bridge_timestamp_us=tick_us,
                            smpl_source_timestamp_us=source_timestamp_us,
                            smpl_frame_index=human_frame_index,
                            smpl_target_reason="ipopt_failed_hold",
                            smpl_target_held=True,
                            arm_target_owner="casadi_ipopt_hold",
                            solver_backend="casadi_opti_ipopt_hold",
                            arm_target_valid=True,
                            arm_ik_solve_ms=float(ik_result.solve_time_ms),
                            arm_ik_status=str(ik_result.message),
                        )
                        qp_payload.update(
                            reference_left=latest_shared_result["left"],
                            reference_right=latest_shared_result["right"],
                        )
                else:
                    missing_pico_source_us = (
                        int(latest_wrist_targets.source_timestamp_us)
                        if latest_wrist_targets is not None
                        else 0
                    )
                    missing_pico_frame = (
                        int(latest_wrist_targets.frame_index)
                        if latest_wrist_targets is not None
                        else -1
                    )
                    hold_tracking_target = bool(pico_tracking_armed)
                    hold_q = (
                        last_ipopt_or_measured_hold()
                        if hold_tracking_target
                        else Q_READY.copy()
                    )
                    hold_left_tf, hold_right_tf = (
                        arm_nonlinear_ik.wrist_transforms_from_q29(hold_q)
                    )
                    control_visual_hands = ipopt_visual_hands_from_transforms(
                        hold_left_tf,
                        hold_right_tf,
                        target_source=(
                            "stale_pico_ipopt_hold"
                            if hold_tracking_target
                            else "teleop_wait_ready_q_ready"
                        ),
                    )
                    if policy_started:
                        send_planner_command(
                            sonic,
                            hold_q,
                            neutralize_waist=True,
                            freeze_lower_body=True,
                            left_hand_position=left_hand_cmd,
                            right_hand_position=right_hand_cmd,
                            locomotion=joystick_command,
                        )
                    # Q_READY is used only before tracking is armed. Once A+X
                    # completes the one-shot alignment, transient source gaps
                    # hold the last successful IPOPT solution.
                    qp_payload = ipopt_arm_sample_reference(
                        hold_q,
                        state=current_state,
                        source="pico_body_cartesian",
                        control_mode="teleop",
                        feasible=True,
                        timestamp_us=tick_us,
                        tau_ff_ref=(
                            last_ipopt_tau_ff
                            if hold_tracking_target
                            and last_ipopt_tau_ff is not None
                            else np.zeros(29, dtype=np.float64)
                        ),
                        bridge_timestamp_us=tick_us,
                        smpl_source_timestamp_us=missing_pico_source_us,
                        smpl_frame_index=missing_pico_frame,
                        smpl_target_reason=(
                            "stale_absolute_target_hold"
                            if hold_tracking_target
                            else "q_ready_waiting_pico_ax"
                        ),
                        smpl_target_held=hold_tracking_target,
                        arm_target_owner=(
                            "casadi_ipopt_hold"
                            if hold_tracking_target
                            else "q_ready_direct"
                        ),
                        solver_backend=(
                            "casadi_opti_ipopt_hold"
                            if hold_tracking_target
                            else "q_ready_direct"
                        ),
                        arm_target_valid=True,
                    )
            else:  # damped_return
                if pico_body_fresh and latest_wrist_targets is not None:
                    return_result = solve_authoritative_ipopt_transforms(
                        latest_wrist_targets.left_waist,
                        latest_wrist_targets.right_waist,
                    )
                    return_target = (
                        return_result.q_ref.copy()
                        if return_result.success
                        else last_ipopt_or_measured_hold()
                    )
                    if blended_q is None:
                        blended_q = (
                            last_plan_q.copy()
                            if last_plan_q is not None
                            else return_target.copy()
                        )
                    blended_q = frozen_planner_q29(
                        blended_q,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    alpha = 1.0 - np.exp(-args.damped_return_rate * dt)
                    blended_q = (1.0 - alpha) * blended_q + alpha * return_target
                    blended_q = frozen_planner_q29(
                        blended_q,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    blended_left_tf, blended_right_tf = (
                        arm_nonlinear_ik.wrist_transforms_from_q29(blended_q)
                    )
                    control_visual_hands = ipopt_visual_hands_from_transforms(
                        blended_left_tf,
                        blended_right_tf,
                        target_source="damped_return_ipopt_reference",
                    )
                    send_planner_command(
                        sonic,
                        blended_q,
                        neutralize_waist=bool(args.freeze_planner_waist),
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        left_hand_position=left_hand_cmd,
                        right_hand_position=right_hand_cmd,
                        locomotion=joystick_command,
                    )
                    qp_payload = ipopt_arm_sample_reference(
                        blended_q,
                        tau_ff_ref=return_result.tau_ff_ref,
                        state="damped_return",
                        source="pico_body_damped_return",
                        control_mode="blend_back",
                        feasible=bool(return_result.success),
                        plan_id=0,
                        timestamp_us=tick_us,
                        smpl_source_timestamp_us=int(
                            latest_wrist_targets.source_timestamp_us
                        ),
                        smpl_frame_index=int(latest_wrist_targets.frame_index),
                    )
                else:
                    qp_payload = qp_none_reference(
                        state="damped_return",
                        source="damped_return",
                        control_mode="blend_back",
                        feasible=False,
                        timestamp_us=tick_us,
                    )

            # IK owns q, while arm gravity compensation is evaluated independently
            # at the 50 Hz bridge rate. This keeps arm-load feedforward
            # synchronized with the latest pelvis gravity even when a static IK
            # solution is cached for many seconds.
            if (
                qp_payload is not None
                and bool(qp_payload.get("feasible", False))
                and arm_nonlinear_ik is not None
                and qp_payload.get("q_ref") is not None
            ):
                torque_q = np.asarray(
                    qp_payload["q_ref"], dtype=np.float64
                ).reshape(29).copy()
                torque_q[12:15] = 0.0
                model_tau = arm_nonlinear_ik.gravity_torque_from_q29(
                    torque_q,
                    gravity_direction_pelvis=latest_gravity_direction,
                )
                tau_ff = model_tau.copy()
                tau_ff[:15] = 0.0
                qp_payload["tau_ff_ref"] = tau_ff.tolist()
                qp_payload["gravity_direction_pelvis"] = (
                    latest_gravity_direction.tolist()
                )
                qp_payload["gravity_bias_angle_deg"] = float(
                    np.rad2deg(
                        np.linalg.norm(
                            gravity_estimator.bias_rotation_vector
                        )
                    )
                )
                qp_payload["gravity_accel_correction_active"] = bool(
                    gravity_estimator.accel_correction_active
                )
                qp_payload["waist_gravity_model_tau_nm"] = (
                    model_tau[12:15].tolist()
                )
                qp_payload["waist_adaptive_bias_tau_nm"] = (
                    [0.0, 0.0, 0.0]
                )
                qp_payload["waist_command_tau_nm"] = tau_ff[12:15].tolist()

            # Direct path: Thor owns *only* IPOPT arm q/dq/tau. The G1-local
            # mux fills waist/lower joints from measured state as damping
            # commands, so no SONIC action can enter this packet.
            direct_command_sent = False
            direct_left_grip = grasp_commanded["left"]
            direct_right_grip = grasp_commanded["right"]
            if (
                qp_payload is not None
                and qp_payload.get("schema") == "path.ipopt_arm_reference.v1"
                and bool(qp_payload.get("feasible", False))
                and qp_payload.get("q_ref") is not None
            ):
                q_direct = np.asarray(qp_payload["q_ref"], dtype=np.float64).reshape(29)
                tau_direct = np.asarray(
                    qp_payload.get("tau_ff_ref", np.zeros(29, dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(29)
                try:
                    direct_lcm_sequence += 1
                    direct_lcm.publish(
                        args.direct_upper_command_channel,
                        pack_upper_command(
                            sequence=direct_lcm_sequence,
                            timestamp_us=tick_us,
                            q_arm=q_direct[ARM_START:ARM_START + 14],
                            dq_arm=np.zeros(14, dtype=np.float64),
                            tau_arm=tau_direct[ARM_START:ARM_START + 14],
                            # Trigger/grip values arrive independently at
                            # 100 Hz through pico_controller_state.  They are
                            # not coupled to the conflated VR pose frame.
                            left_grip=direct_left_grip,
                            right_grip=direct_right_grip,
                        ),
                    )
                    direct_command_sent = True
                except Exception as exc:
                    log(f"direct upper command publish failed: {exc}")
            if qp_payload is not None:
                qp_payload["direct_command_sent"] = direct_command_sent
                qp_payload["direct_command_channel"] = args.direct_upper_command_channel
                qp_payload["direct_command_sequence"] = direct_lcm_sequence
                qp_payload["direct_left_grip"] = direct_left_grip
                qp_payload["direct_right_grip"] = direct_right_grip

            if qp_payload is not None and not qp_reference_enabled:
                qp_payload = qp_none_reference(
                    state=current_state,
                    source="manual_qp_gate",
                    control_mode="qp_disabled",
                    feasible=False,
                    plan_id=int(qp_payload.get("plan_id", 0)),
                    timestamp_us=tick_us,
                )

            if qp_payload is not None:
                pico_frame = (
                    int(latest_wrist_targets.frame_index)
                    if latest_wrist_targets is not None
                    else -1
                )
                pico_source_us = (
                    int(latest_wrist_targets.source_timestamp_us)
                    if latest_wrist_targets is not None
                    else 0
                )
                pico_target_valid = bool(pico_body_fresh)
                pico_target_reason = (
                    "absolute_head_position"
                    if pico_body_fresh
                    else (
                        "waiting_fresh_absolute_target"
                        if pico_tracking_armed
                        else "teleop_wait_ready_fixed_fk"
                    )
                )
                qp_payload["pico_body_frame_index"] = pico_frame
                qp_payload["pico_body_source_timestamp_us"] = pico_source_us
                qp_payload["pico_body_tracking_armed"] = bool(
                    pico_tracking_armed
                )
                qp_payload["pico_body_calibration_session"] = int(
                    latest_pico_session_id
                )
                qp_payload["pico_body_input_age_ms"] = float(
                    1000.0 * pico_body_age_sec
                    if np.isfinite(pico_body_age_sec)
                    else -1.0
                )
                qp_payload["pico_body_target_valid"] = pico_target_valid
                qp_payload["pico_body_target_reason"] = pico_target_reason
                qp_payload["pico_body_target_held"] = bool(
                    qp_payload.get("smpl_target_held", False)
                )
                qp_payload["pico_pose_source"] = 1
                qp_payload["pico_controller_alignment_revision"] = int(
                    latest_pico_session_id
                )
                if latest_wrist_targets is not None:
                    qp_payload["pico_left_target_position"] = (
                        latest_wrist_targets.left_waist[:3, 3].tolist()
                    )
                    qp_payload["pico_right_target_position"] = (
                        latest_wrist_targets.right_waist[:3, 3].tolist()
                    )
                    qp_payload["pico_left_target_rotation"] = (
                        latest_wrist_targets.left_waist[:3, :3].reshape(-1).tolist()
                    )
                    qp_payload["pico_right_target_rotation"] = (
                        latest_wrist_targets.right_waist[:3, :3].reshape(-1).tolist()
                    )
                    qp_payload["pico_body_source"] = "absolute_head_position"
                    qp_payload["pico_body_calibration_frames"] = 0
                    qp_payload["pico_body_recovery_count"] = 0

                # Compatibility fields consumed by existing viewers/recorders.
                # Their values now identify the PICO body target, not GMR/SMPL.
                qp_payload["smpl_frame_index"] = pico_frame
                qp_payload["smpl_source_timestamp_us"] = pico_source_us
                qp_payload["smpl_input_age_ms"] = qp_payload[
                    "pico_body_input_age_ms"
                ]
                qp_payload["smpl_target_valid"] = pico_target_valid
                qp_payload["smpl_target_reason"] = pico_target_reason

            if now >= next_mode_command:
                send_mode_command(sonic, planner=planner_mode_for_current_state())
                next_mode_command = now + max(0.05, args.command_period_sec)

            if qp_payload is not None:
                static_hand_payloads = {
                    side: static_target_payload_for_hand(latest_static_target, side)
                    for side in ("left", "right")
                }
                static_hand_valid = {
                    side: shared_target_available(tick_us, side)
                    for side in ("left", "right")
                }
                qp_payload.update(
                    static_target_received=latest_static_target is not None,
                    static_target_revision=int(static_target_revision),
                    static_target_age_ms=(
                        float(static_target_age_sec * 1000.0)
                        if np.isfinite(static_target_age_sec)
                        else None
                    ),
                    static_target_source=(
                        str(latest_static_target.get("source", ""))
                        if latest_static_target is not None
                        else None
                    ),
                    static_target_goal_uid=(
                        str(latest_static_target.get("goal_uid", ""))
                        if latest_static_target is not None
                        else None
                    ),
                    static_target_hand_valid=static_hand_valid,
                    static_target_hand_position={
                        side: (
                            payload.get("position", payload.get("target_position"))
                            if payload is not None
                            else None
                        )
                        for side, payload in static_hand_payloads.items()
                    },
                )
                qp.send(pack_topic(args.qp_topic, qp_payload))
            if sonic is not None and pose_to_send is not None:
                sonic.send(pose_to_send)

            # This direct Wi-Fi diagnostic path deliberately bypasses the
            # Thor-local belief ZMQ subscriber. Perception/belief may be late,
            # but the laptop must always see the current IPOPT wrist target.
            wrist_visual_lcm.publish(
                args.wrist_visual_channel,
                pack_ipopt_visual(
                    {
                        "schema": "path.wrist_ipopt_visual.v1",
                        "published_timestamp_us": int(now_us()),
                        "source_timestamp_us": int(tick_us),
                        "state": current_state,
                        "hands": control_visual_hands,
                    }
                ),
            )

            # PUB/SUB has no retained state.  A short heartbeat makes the
            # Y/B selection available to a gateway that starts later too.
            if now >= next_shared_hand_publish:
                shared_hand_pub.send(
                    pack_topic(
                        args.shared_hand_selection_topic,
                        {
                            "schema": "path.shared_control_hand.v2",
                            "timestamp_us": int(tick_us),
                            "state": current_state,
                            "selected_hand": shared_selected_hand,
                            "active": bool(
                                current_state == "teleop"
                                and shared_selected_hand in {"left", "right"}
                            ),
                            "revision": int(shared_hand_selection_revision),
                            "grasp_center_local_m": [
                                float(value) for value in grasp_center_local
                            ],
                            "end_effector_model": str(
                                args.end_effector_model
                            ).lower(),
                            "hands": control_visual_hands,
                        },
                    )
                )
                next_shared_hand_publish = (
                    now + 1.0 / SHARED_HAND_PUBLISH_HZ
                )

            if now - last_log > 1.0:
                shared_log = "shared=none"
                pico_position_log = "pico_L/R=missing"
                if latest_wrist_targets is not None:
                    left_position = latest_wrist_targets.left_waist[:3, 3]
                    right_position = latest_wrist_targets.right_waist[:3, 3]
                    pico_position_log = (
                        "pico_L=("
                        f"{left_position[0]:+.3f},"
                        f"{left_position[1]:+.3f},"
                        f"{left_position[2]:+.3f}) "
                        "pico_R=("
                        f"{right_position[0]:+.3f},"
                        f"{right_position[1]:+.3f},"
                        f"{right_position[2]:+.3f})"
                    )
                if latest_shared_result is not None:
                    left_ref = latest_shared_result["left"]
                    right_ref = latest_shared_result["right"]
                    shared_log = (
                        f"shared_select={shared_selected_hand} "
                        f"ref_L[{left_ref['position_source'][:3]}/{left_ref['orientation_source'][:3]} "
                        f"e={left_ref['position_error_m'] * 100:.1f}cm/{np.rad2deg(left_ref['orientation_error_rad']):.1f}deg "
                        f"gate={left_ref['orientation_gate_reason']}] "
                        f"ref_R[{right_ref['position_source'][:3]}/{right_ref['orientation_source'][:3]} "
                        f"e={right_ref['position_error_m'] * 100:.1f}cm/{np.rad2deg(right_ref['orientation_error_rad']):.1f}deg "
                        f"gate={right_ref['orientation_gate_reason']}]"
                    )
                log(
                    f"state={current_state} pico_fresh={pico_body_fresh} "
                    f"pico_session={latest_pico_session_id}/"
                    f"{'teleop_tracking' if pico_tracking_armed else 'teleop_wait_ready'} "
                    f"smpl_lower_fresh={smpl_fresh} "
                    f"static_target={'fresh' if static_target_fresh else 'held' if static_target_usable else 'missing'} "
                    f"static_age={static_target_age_sec if np.isfinite(static_target_age_sec) else float('nan'):.2f}s "
                    f"static_rev={static_target_revision} "
                    f"qp_gate={'on' if qp_reference_enabled else 'off'} "
                    f"direct={'sent' if direct_command_sent else 'hold'}"
                    f"#{direct_lcm_sequence} "
                    f"g1_state_age={1000.0 * (tick_us - latest_robot_state_us) / 1.0e6 if latest_robot_state_us else -1.0:.1f}ms "
                    f"lower_src={'idle_planner' if planner_lower_body_idle else 'raw_smpl'} "
                    f"freeze_lower={bool(args.freeze_planner_lower_body)} "
                    f"freeze_waist={bool(args.freeze_planner_waist)} "
                    f"hand_fresh={hand_fresh} "
                    f"trig=({latest_hand_meta.get('left_trigger', float('nan')):.2f},"
                    f"{latest_hand_meta.get('right_trigger', float('nan')):.2f}) "
                    f"grip=({latest_hand_meta.get('left_grip', float('nan')):.2f},"
                    f"{latest_hand_meta.get('right_grip', float('nan')):.2f}) "
                    f"hand_ctrl_age={(tick_us - latest_hand_control_us) / 1000.0 if latest_hand_control_us else -1.0:.1f}ms "
                    f"direct_grip=({int(direct_left_grip)},{int(direct_right_grip)}) "
                    f"qp_type={qp_payload.get('reference_type') if qp_payload else 'none'} "
                    f"source={qp_payload.get('source', 'none') if qp_payload else 'none'} "
                    f"q_prior={qp_payload.get('q_prior_source', 'n/a') if qp_payload else 'n/a'} "
                    f"pico_target={'absolute_head_position' if latest_wrist_targets else 'waiting'} "
                    f"{pico_position_log} "
                    f"pico_age={1000.0 * pico_body_age_sec if np.isfinite(pico_body_age_sec) else -1.0:.1f}ms "
                    f"released={released_pose_frame}/{released_pose_frames} "
                    f"pose_frame={latest_smpl_frame} "
                    f"button_seq={last_button_sequence} "
                    f"joy_seq={latest_controller_sequence} "
                    f"joy_age={1000.0 * controller_age_sec if np.isfinite(controller_age_sec) else -1.0:.1f}ms "
                    f"joy_L=({latest_controller_left_axis[0]:+.2f},"
                    f"{latest_controller_left_axis[1]:+.2f}) "
                    f"joy_Rx={latest_controller_right_axis[0]:+.2f} "
                    f"loco={joystick_command.reason}/m{joystick_command.mode} "
                    f"v={joystick_command.speed if joystick_command.speed >= 0.0 else 0.0:.2f}m/s "
                    f"yaw_rate={joystick_command.yaw_rate_rad_s:+.2f}rad/s "
                    f"facing=({joystick_command.facing[0]:+.2f},"
                    f"{joystick_command.facing[1]:+.2f},"
                    f"{joystick_command.facing[2]:+.2f}) "
                    f"control={qp_payload.get('control_mode', 'none') if qp_payload else 'none'} "
                    f"plan={qp_payload.get('plan_id', 0) if qp_payload else 0} "
                    f"phase={float(qp_payload.get('phase', 0.0)) if qp_payload else 0.0:.2f} "
                    f"phase_mode={phase_mode} "
                    f"phase_err={phase_sync_error_rad if np.isfinite(phase_sync_error_rad) else float('nan'):.3f}rad "
                    f"arm_ik={last_arm_ik_status} "
                    f"ik_err={100.0 * last_arm_ik_error_m if np.isfinite(last_arm_ik_error_m) else float('nan'):.2f}cm "
                    f"ik_solve={last_arm_ik_solve_ms if np.isfinite(last_arm_ik_solve_ms) else float('nan'):.2f}ms "
                    f"{gravity_estimator.status()} "
                    f"waist_ff=off "
                    f"{shared_log} "
                    f"{outer_waist.status()} "
                    f"feasible={qp_payload.get('feasible') if qp_payload else False}"
                )
                last_log = now

            next_deadline += period
            sleep_sec = next_deadline - time.monotonic()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            elif sleep_sec < -period:
                # Drop missed deadlines instead of accumulating drift.
                next_deadline = time.monotonic()
    except KeyboardInterrupt:
        return 0
    finally:
        pico_vr_body.close()
        pico_pose.close()
        pico_command.close()
        pico_planner.close()
        pico_manager.close()
        button_sub.close()
        controller_sub.close()
        robot_state.close()
        state_sub.close()
        if sonic is not None:
            sonic.close()
        qp.close()
        shared_hand_pub.close()
        if direct_state is not None:
            direct_state.close()
        keyboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
