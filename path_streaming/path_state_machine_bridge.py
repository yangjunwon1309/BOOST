#!/usr/bin/env python3
"""PATH state-machine bridge between Pico SMPL targets and SONIC.

This bridge is the only process that publishes to SONIC's ZMQ manager port.
Other producers are subordinate inputs:

  Pico manager     -> tcp://127.0.0.1:5559/{pose,command,planner,manager_state}
  Pico buttons     -> tcp://127.0.0.1:5565/pico_button_event
  cuRobo planner   -> tcp://127.0.0.1:5562/curobo_plan
  State command    -> tcp://*:5563/path_state_command
  SONIC policy     <- tcp://*:5556/{pose,command,planner,manager_state}
  EE-QP reference  <- tcp://*:5560/path_qp_reference

State behavior:
  teleop/intention:
    Convert torso-relative SMPL wrist displacement into pelvis-local Cartesian
    targets anchored at the robot's initial wrist FK, then solve both arms with
    CasADi/IPOPT. SMPL root/global translation is never used.
  auto_planning:
    Blend the belief/object contact point with the same torso-relative SMPL
    wrist target. CasADi/IPOPT remains the sole owner of both arm commands.
  damped_return:
    Return the arm target toward the current torso-relative SMPL target while
    SONIC continues to own only the lower body.

Runtime hotkeys:
  q/w/e/r switch PATH state.
  p toggles whether this bridge publishes a valid QP reference. When disabled,
    the bridge still forwards the proper SONIC input, but QP receives an
    infeasible none-reference and falls back to the uncorrected SONIC target.
  m toggles SONIC lower-body conditioning between the safe zero-root planner
    input and raw SMPL pose input. Arm and waist ownership never changes.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import select
import struct
import sys
import termios
import time
import tty
from pathlib import Path

import msgpack
import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_command_message,
    build_planner_message,
)
from path_core.intention.action_blending import (
    EETarget,
    LEFT_WRIST_LINK,
    RIGHT_WRIST_LINK,
    SharedControlConfig,
    TaskSpaceBlender,
)
from path_core.messaging.path_schema import (
    CUROBO_PLAN_TOPIC,
    QP_REFERENCE_TOPIC,
    STATIC_EE_TARGET_TOPIC,
    ipopt_arm_cartesian_target_reference,
    ipopt_arm_sample_reference,
    now_us,
    qp_bspline_reference,
    qp_cartesian_target_reference,
    qp_none_reference,
    qp_sample_reference,
)
from path_core.messaging.zmq_bus import pack_topic, unpack_topic
from path_planning.unitree_arm_nonlinear_ik import UnitreeG1ArmNonlinearIK
from path_streaming.path_smpl_thread_server import UPPER_BODY_PLANNER_INDICES
from path_core.teleop.smpl_cartesian_target import (
    SmplCartesianTargetConfig,
    SmplCartesianTargetTracker,
    SmplCartesianTargets,
)

VALID_STATES = {"teleop", "intention", "auto_planning", "damped_return"}

# A planner height of -1 is the SONIC protocol sentinel for its trained default
# standing height.  PATH never owns the floating-base height: the lower body
# remains under SONIC's locomotion policy in every state.
SONIC_DEFAULT_STANDING_HEIGHT = -1.0


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
    parser.add_argument("--smpl-timeout-sec", type=float, default=0.20)
    parser.add_argument("--smpl-scale-x", type=float, default=0.85)
    parser.add_argument("--smpl-scale-y", type=float, default=0.85)
    parser.add_argument("--smpl-scale-z", type=float, default=0.85)
    parser.add_argument("--smpl-displacement-sign", type=float, default=-1.0)
    parser.add_argument("--smpl-target-ema-alpha", type=float, default=0.35)
    parser.add_argument("--smpl-target-max-velocity-m-s", type=float, default=0.25)
    parser.add_argument("--smpl-target-max-acceleration-m-s2", type=float, default=1.5)
    parser.add_argument("--smpl-jump-reject-m", type=float, default=0.07)
    parser.add_argument("--smpl-max-anchor-offset-m", type=float, default=0.45)

    parser.add_argument("--pico-endpoint", default="tcp://127.0.0.1:5559")
    parser.add_argument("--button-endpoint", default="tcp://127.0.0.1:5565")
    parser.add_argument("--button-topic", default="pico_button_event")
    parser.add_argument("--plan-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--plan-topic", default=CUROBO_PLAN_TOPIC)
    parser.add_argument("--static-target-bind", default="tcp://*:5564")
    parser.add_argument("--static-target-topic", default=STATIC_EE_TARGET_TOPIC)
    parser.add_argument("--static-target-timeout-sec", type=float, default=0.25)
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

    parser.add_argument("--command-period-sec", type=float, default=0.25)
    parser.add_argument("--damped-return-rate", type=float, default=3.0)
    parser.add_argument("--plan-timeout-sec", type=float, default=1.00)
    parser.add_argument(
        "--phase-search-window-sec",
        type=float,
        default=0.50,
        help="Search this much B-spline time around the previous phase for measured-state alignment.",
    )
    parser.add_argument("--phase-search-samples", type=int, default=41)
    parser.add_argument(
        "--phase-lookahead-sec",
        type=float,
        default=0.02,
        help="Advance the measured-state-aligned B-spline phase by this prediction horizon.",
    )
    parser.add_argument(
        "--phase-tracking-mode",
        choices=("fixed_rate", "state_aligned", "wall_time"),
        default="fixed_rate",
        help=(
            "How auto_planning advances along the cuRobo B-spline. "
            "fixed_rate streams the spline at the bridge FPS independent of tracking error; "
            "state_aligned projects measured q onto the spline; wall_time uses message t_start_us."
        ),
    )
    parser.add_argument(
        "--terminal-capture-radius-m",
        type=float,
        default=0.07,
        help="QP switches to the static terminal EE target once measured wrist is within this radius.",
    )
    parser.add_argument(
        "--auto-planning-mode",
        choices=("shared_control", "legacy"),
        default="shared_control",
        help=(
            "Behavior of PATH auto_planning/e mode. shared_control blends SAM3/TASC "
            "contact targets with GMR wrist FK and lets EE-QP solve the arms; legacy "
            "keeps the older static-target/cuRobo path."
        ),
    )
    parser.add_argument("--shared-rho1", type=float, default=0.60)
    parser.add_argument("--shared-rho2", type=float, default=0.90)
    parser.add_argument("--shared-alpha-rate-up", type=float, default=1.0)
    parser.add_argument("--shared-alpha-rate-down", type=float, default=3.0)
    parser.add_argument("--shared-v-transfer-max", type=float, default=0.25)
    parser.add_argument("--shared-startup-hold-sec", type=float, default=2.0)
    parser.add_argument("--shared-stale-after-sec", type=float, default=0.3)
    parser.add_argument("--shared-d-max-m", type=float, default=0.9)
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
    parser.add_argument("--arm-ik-ee-offset-m", type=float, default=0.05)
    parser.add_argument(
        "--arm-ik-smooth-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Unitree's short weighted moving average on the IK q solution.",
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
        "q": "teleop",
        "w": "intention",
        "e": "auto_planning",
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


def bspline_eval(control_points: np.ndarray, dt: float, t_start_us: int, query_us: int) -> tuple[np.ndarray, float]:
    cps = np.asarray(control_points, dtype=np.float64)
    if cps.ndim != 2 or cps.shape[0] < 4 or cps.shape[1] != 29 or dt <= 0.0:
        raise ValueError("invalid B-spline control points")
    max_segment = float(cps.shape[0] - 4)
    s = np.clip((query_us - int(t_start_us)) / (dt * 1.0e6), 0.0, max_segment)
    return bspline_eval_s(cps, s)


def bspline_eval_s(control_points: np.ndarray, s: float) -> tuple[np.ndarray, float]:
    cps = np.asarray(control_points, dtype=np.float64)
    if cps.ndim != 2 or cps.shape[0] < 4 or cps.shape[1] != 29:
        raise ValueError("invalid B-spline control points")
    max_segment = float(cps.shape[0] - 4)
    s = float(np.clip(s, 0.0, max_segment))
    i = int(np.clip(np.floor(s), 0, cps.shape[0] - 4))
    u = float(np.clip(s - i, 0.0, 1.0))
    omu = 1.0 - u
    basis = np.array([
        omu * omu * omu / 6.0,
        (3.0 * u**3 - 6.0 * u**2 + 4.0) / 6.0,
        (-3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0) / 6.0,
        u**3 / 6.0,
    ])
    q = basis @ cps[i : i + 4]
    phase = float(np.clip(s / max_segment, 0.0, 1.0)) if max_segment > 0.0 else 1.0
    return q, phase


def nearest_bspline_s(
    control_points: np.ndarray,
    q_current: np.ndarray,
    previous_s: float | None,
    dt_plan: float,
    search_window_sec: float,
    samples: int,
) -> tuple[float, float]:
    cps = np.asarray(control_points, dtype=np.float64)
    q_current = np.asarray(q_current, dtype=np.float64).reshape(29)
    max_segment = float(cps.shape[0] - 4)
    if max_segment <= 0.0:
        return 0.0, 0.0
    if previous_s is None:
        lo, hi = 0.0, max_segment
    else:
        half_window_s = max(0.0, search_window_sec) / max(dt_plan, 1.0e-6)
        lo = max(0.0, previous_s - half_window_s)
        hi = min(max_segment, previous_s + half_window_s)
        if hi <= lo + 1.0e-9:
            hi = min(max_segment, lo + 1.0)
    grid = np.linspace(lo, hi, max(3, int(samples)))
    current_upper = q_current[UPPER_BODY_PLANNER_INDICES]
    best_s = float(grid[0])
    best_error = float("inf")
    for candidate_s in grid:
        q_candidate, _ = bspline_eval_s(cps, float(candidate_s))
        diff = q_candidate[UPPER_BODY_PLANNER_INDICES] - current_upper
        error = float(np.sqrt(np.mean(diff * diff)))
        if error < best_error:
            best_error = error
            best_s = float(candidate_s)
    return best_s, best_error


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
    socket: zmq.Socket,
    q29: np.ndarray,
    neutralize_waist: bool = False,
    freeze_lower_body: bool = True,
    left_hand_position: np.ndarray | list[float] | None = None,
    right_hand_position: np.ndarray | list[float] | None = None,
) -> None:
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
    socket.send(
        build_planner_message(
            0,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            speed=-1.0,
            # Do not inject a root/base-height target from PATH.  -1 is the
            # protocol's use-trained-default sentinel, not a zero-height
            # command.  The remaining locomotion command is intentionally
            # static as well.
            height=SONIC_DEFAULT_STANDING_HEIGHT,
            upper_body_position=upper.tolist(),
            upper_body_velocity=np.zeros(17, dtype=np.float32).tolist(),
            left_hand_position=left_hand.tolist(),
            right_hand_position=right_hand.tolist(),
        )
    )


def send_mode_command(socket: zmq.Socket, planner: bool) -> None:
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
    if not payload:
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
    return EETarget(
        position=position,
        confidence=confidence,
        object_id=object_id,
        grip_index=grip_index,
        goal_uid=str(payload.get("goal_uid", "")),
        timestamp_us=int(received_us),
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


def q_ik_prior_from_plan(
    plan: dict | None,
    target: dict | None,
    target_link: str,
    tick_us: int,
    plan_timeout_sec: float,
) -> tuple[np.ndarray | None, str]:
    if not plan or not target or not bool(plan.get("feasible", False)):
        return None, "none"
    if (tick_us - int(plan.get("timestamp_us", 0))) > plan_timeout_sec * 1.0e6:
        return None, "stale"
    if str(plan.get("target_link", "")) != str(target_link):
        return None, "link_mismatch"
    target_uid = str(target.get("goal_uid", ""))
    plan_uid = str(plan.get("goal_uid", ""))
    if target_uid and plan_uid and target_uid != plan_uid:
        return None, "uid_mismatch"
    q_goal = plan.get("q_goal", plan.get("q_ik_ref", plan.get("ik_q_goal")))
    if q_goal is None:
        return None, "missing_q_goal"
    q = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    if q.shape[0] < 29 or not np.isfinite(q[:29]).all():
        return None, "invalid_q_goal"
    return q[:29].copy(), f"curobo_ik_plan_{int(plan.get('plan_id', 0))}"


def main() -> int:
    args = parse_args()
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
    if args.auto_planning_mode != "shared_control":
        raise RuntimeError(
            "PATH e-mode requires shared_control position blending followed by "
            "CasADi/IPOPT arm IK; legacy planner mode is disabled."
        )
    # PATH's command ownership is invariant, not a tunable mode:
    # SONIC lower body only, identity waist, CasADi/IPOPT arms.
    args.freeze_planner_lower_body = True
    args.freeze_planner_waist = True
    args.shared_waist_mode = "hold_identity"
    args.outer_waist_enabled = False
    ctx = zmq.Context.instance()

    def make_pico_sub(topic: str, *, conflate: bool, hwm: int = 8) -> zmq.Socket:
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        if conflate:
            socket.setsockopt(zmq.CONFLATE, 1)
        else:
            socket.setsockopt(zmq.RCVHWM, int(hwm))
        socket.setsockopt(zmq.RCVTIMEO, 1)
        socket.connect(args.pico_endpoint)
        return socket

    # Keep independent queues so high-rate manager/planner traffic cannot push
    # SMPL pose frames behind a backlog. Pose/planner/manager use latest-only
    # semantics; command keeps a small edge queue for replay compatibility.
    pico_pose = make_pico_sub("pose", conflate=True)
    pico_command = make_pico_sub("command", conflate=False, hwm=32)
    pico_planner = make_pico_sub("planner", conflate=True)
    pico_manager = make_pico_sub("manager_state", conflate=True)

    button_sub = ctx.socket(zmq.SUB)
    button_sub.setsockopt(zmq.SUBSCRIBE, args.button_topic.encode())
    button_sub.setsockopt(zmq.RCVHWM, 64)
    button_sub.setsockopt(zmq.RCVTIMEO, 1)
    button_sub.connect(args.button_endpoint)

    plan_sub = ctx.socket(zmq.SUB)
    plan_sub.setsockopt(zmq.SUBSCRIBE, args.plan_topic.encode())
    plan_sub.setsockopt(zmq.CONFLATE, 1)
    plan_sub.setsockopt(zmq.RCVTIMEO, 1)
    plan_sub.connect(args.plan_endpoint)

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

    sonic = ctx.socket(zmq.PUB)
    sonic.setsockopt(zmq.SNDHWM, 20)
    sonic.setsockopt(zmq.LINGER, 0)
    sonic.bind(f"tcp://{args.sonic_bind_host}:{args.sonic_port}")

    qp = ctx.socket(zmq.PUB)
    qp.setsockopt(zmq.SNDHWM, 2)
    qp.setsockopt(zmq.LINGER, 0)
    qp.bind(f"tcp://{args.qp_bind_host}:{args.qp_port}")

    log(f"Pico input SUB {args.pico_endpoint}/pose,command,planner,manager_state")
    log(f"Pico ABXY SUB {args.button_endpoint}/{args.button_topic}")
    log("GMR input disabled: wrist targets come directly from relative SMPL motion")
    log(f"cuRobo plan SUB {args.plan_endpoint}/{args.plan_topic}")
    log(f"static EE target SUB {args.static_target_bind}/{args.static_target_topic}")
    log(f"robot state SUB {args.robot_state_endpoint}/{args.robot_state_topic}")
    log(f"state command SUB {args.state_command_bind}/{args.state_command_topic}")
    log(f"SONIC output PUB tcp://{args.sonic_bind_host}:{args.sonic_port}")
    log(f"QP reference PUB tcp://{args.qp_bind_host}:{args.qp_port}/{args.qp_topic}")
    arm_nonlinear_ik: UnitreeG1ArmNonlinearIK | None = None
    if args.arm_nonlinear_ik_enabled:
        arm_nonlinear_ik = UnitreeG1ArmNonlinearIK(
            urdf_path=args.arm_ik_urdf,
            model_dir=args.arm_ik_model_dir,
            max_iter=args.arm_ik_max_iter,
            ee_offset_m=args.arm_ik_ee_offset_m,
            use_smooth_filter=args.arm_ik_smooth_filter,
        )
        log(
            "Unitree-style arm nonlinear IK enabled: "
            f"urdf={args.arm_ik_urdf}, max_iter={args.arm_ik_max_iter}, "
            f"hard_lock_inactive={args.arm_ik_hard_lock_inactive_arm}"
        )

    current_state = args.state
    planner_lower_body_idle = bool(args.planner_lower_body_idle_default)
    released_pose_frames = 0
    latest_plan: dict | None = None
    latest_plan_us = 0
    latest_static_target: dict | None = None
    latest_static_target_us = 0
    static_target_revision = 0
    static_target_plan_id = 0
    latest_static_target_uid = ""
    static_target_hold_q: np.ndarray | None = None
    static_target_hold_plan_id = -1
    static_target_hold_side = ""
    active_plan: dict | None = None
    active_plan_us = 0
    active_plan_s: float | None = None
    latest_robot_q: np.ndarray | None = None
    latest_robot_state_us = 0
    latest_pose_raw: bytes | None = None
    latest_pose_us = 0
    latest_smpl_joints: np.ndarray | None = None
    latest_smpl_source_us = 0
    latest_smpl_frame = -1
    last_forwarded_pose_frame = -1
    latest_manager_mode = 0
    policy_started = False
    last_button_sequence = 0
    dedicated_button_events_seen = False
    latest_left_hand_joints: np.ndarray | None = None
    latest_right_hand_joints: np.ndarray | None = None
    latest_hand_us = 0
    latest_hand_meta: dict[str, float] = {}
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
    next_mode_command = 0.0
    last_tick = time.monotonic()
    last_log = 0.0
    completed_plan_ids: set[int] = set()
    period = 1.0 / max(args.fps, 1.0e-6)
    keyboard = KeyboardController(enabled=not args.no_keyboard)
    outer_waist = OuterWaistPlanner(args)
    shared_config = SharedControlConfig(
        rho1=float(args.shared_rho1),
        rho2=float(args.shared_rho2),
        alpha_rate_up=float(args.shared_alpha_rate_up),
        alpha_rate_down=float(args.shared_alpha_rate_down),
        v_transfer_max=float(args.shared_v_transfer_max),
        startup_hold_s=float(args.shared_startup_hold_sec),
        stale_after_s=float(args.shared_stale_after_sec),
        d_max_m=float(args.shared_d_max_m),
        reset_alpha_on_target_switch=bool(args.shared_reset_alpha_on_target_switch),
        ref_lpf_alpha=float(args.shared_ref_lpf_alpha),
        hand_h_margin_m=float(args.shared_hand_h_margin_m),
        hand_switch_confirm_n=int(args.shared_hand_switch_confirm_n),
        lock_hand_while_active=bool(args.shared_lock_hand_while_active),
        orientation_mode=str(args.shared_orientation_mode),
        waist_mode=str(args.shared_waist_mode),
    )
    shared_control = TaskSpaceBlender(shared_config)
    latest_shared_result = None
    smpl_targets = SmplCartesianTargetTracker(
        SmplCartesianTargetConfig(
            scale_xyz=(
                float(args.smpl_scale_x),
                float(args.smpl_scale_y),
                float(args.smpl_scale_z),
            ),
            displacement_sign=float(args.smpl_displacement_sign),
            ema_alpha=float(args.smpl_target_ema_alpha),
            max_velocity_m_s=float(args.smpl_target_max_velocity_m_s),
            max_acceleration_m_s2=float(args.smpl_target_max_acceleration_m_s2),
            jump_reject_m=float(args.smpl_jump_reject_m),
            max_anchor_offset_m=float(args.smpl_max_anchor_offset_m),
            stale_after_sec=float(args.smpl_timeout_sec),
        )
    )
    latest_smpl_targets: SmplCartesianTargets | None = None
    log(
        "auto_planning mode="
        f"{args.auto_planning_mode} shared_control="
        f"rho=[{args.shared_rho1:.2f},{args.shared_rho2:.2f}] "
        f"v_transfer={args.shared_v_transfer_max:.2f}m/s "
        f"startup_hold={args.shared_startup_hold_sec:.2f}s"
    )
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
        """Solve the authoritative arm joints from pelvis-local SMPL targets."""

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
        result = arm_nonlinear_ik.solve_bimanual_targets(
            left_target_tf=np.asarray(left_target_tf, dtype=np.float64),
            right_target_tf=np.asarray(right_target_tf, dtype=np.float64),
            q_seed29=q_seed,
        )
        last_arm_ik_solve_ms = float(result.solve_time_ms)
        last_arm_ik_error_m = float(result.position_error_m)
        last_arm_ik_rot_error_rad = float(result.rotation_error_rad)
        last_arm_ik_status = "ok" if result.success else str(result.message)
        if result.success:
            result.q_ref[:15] = 0.0
            # Lower body remains SONIC-owned. Preserve waist and arm gravity
            # compensation generated from the full Pinocchio model.
            result.tau_ff_ref[:12] = 0.0
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

    def latest_plan_ready(tick_us: int) -> bool:
        if not latest_plan or not latest_plan.get("feasible", False):
            return False
        if int(latest_plan.get("plan_id", 0)) in completed_plan_ids:
            return False
        total_duration_sec = float(latest_plan.get("total_duration_sec", 0.0))
        t_start_us = int(latest_plan.get("t_start_us", latest_plan.get("timestamp_us", tick_us)))
        if total_duration_sec > 0.0 and (tick_us - t_start_us) > (total_duration_sec + 0.25) * 1.0e6:
            return False
        return bool(
            (tick_us - latest_plan_us) <= args.plan_timeout_sec * 1.0e6
        )

    def active_plan_target_distance() -> float:
        if active_plan is None or latest_robot_q is None or wrist_fk is None:
            return float("nan")
        target_link = str(active_plan.get("target_link", "left_wrist_yaw_link"))
        target_position = np.asarray(
            active_plan.get("target_position", [np.nan, np.nan, np.nan]),
            dtype=np.float64,
        ).reshape(3)
        if not np.isfinite(target_position).all():
            return float("nan")
        current = wrist_fk.wrist_position(latest_robot_q, target_link)
        return float(np.linalg.norm(current - target_position))

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
        if not fresh:
            return None, None, False
        return latest_left_hand_joints, latest_right_hand_joints, True

    def clear_active_plan(reason: str) -> None:
        nonlocal active_plan, active_plan_us, active_plan_s, phase_mode, phase_sync_error_rad
        plan_id = int(active_plan.get("plan_id", 0)) if active_plan else 0
        log(f"clearing active plan id={plan_id}: {reason}")
        if plan_id:
            completed_plan_ids.add(plan_id)
        active_plan = None
        active_plan_us = 0
        active_plan_s = None
        phase_mode = "none"
        phase_sync_error_rad = float("nan")

    def latch_latest_plan(source: str, tick_us: int) -> bool:
        nonlocal active_plan, active_plan_us, active_plan_s, phase_mode, phase_sync_error_rad
        if not latest_plan_ready(tick_us):
            active_plan = None
            active_plan_us = 0
            active_plan_s = None
            phase_mode = "none"
            phase_sync_error_rad = float("nan")
            log(f"{source}: no fresh feasible plan to latch")
            return False
        active_plan = copy.deepcopy(latest_plan)
        active_plan_us = tick_us
        active_plan_s = 0.0
        phase_mode = "latched"
        phase_sync_error_rad = float("nan")
        log(
            f"{source}: latched active plan id={active_plan.get('plan_id', 0)} "
            f"cps={len(active_plan.get('control_points', []))} "
            f"phase_mode={args.phase_tracking_mode}"
        )
        return True

    def shared_target_available(tick_us: int) -> bool:
        if args.auto_planning_mode != "shared_control":
            return False
        if latest_static_target is None:
            return False
        if not bool(latest_static_target.get("target_valid", True)):
            return False
        try:
            position = np.asarray(
                latest_static_target.get(
                    "position", latest_static_target.get("target_position", [])
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
        if current_state in {"auto_planning", "damped_return"}:
            return True
        return current_state in {"teleop", "intention"} and planner_lower_body_idle

    def apply_state_request(requested: str, source: str, tick_us: int | None = None) -> None:
        nonlocal current_state, next_mode_command, blended_q, active_plan, active_plan_us
        nonlocal active_plan_s, phase_mode, phase_sync_error_rad
        if tick_us is None:
            tick_us = now_us()
        if requested not in VALID_STATES:
            log(f"ignored invalid {source} state command: {requested}")
            return
        previous_state = current_state
        if requested != current_state:
            log(f"{source} state command: {current_state} -> {requested}")
        else:
            log(f"{source} state command: staying in {requested}")
        current_state = requested
        if requested in {"teleop", "intention"}:
            shared_control.reset()
            active_plan = None
            active_plan_us = 0
            active_plan_s = None
            phase_mode = "none"
            phase_sync_error_rad = float("nan")
            send_mode_command(sonic, planner=planner_lower_body_idle)
            next_mode_command = time.monotonic() + max(0.05, args.command_period_sec)
        else:
            send_mode_command(sonic, planner=True)
            next_mode_command = 0.0
            if requested == "auto_planning":
                if previous_state != "auto_planning":
                    shared_control.reset()
                if args.auto_planning_mode == "shared_control":
                    active_plan = None
                    active_plan_us = 0
                    active_plan_s = None
                    phase_mode = "shared_control_wait"
                    phase_sync_error_rad = float("nan")
                    log(
                        f"{source}: shared_control auto_planning entered; "
                        f"target={'available' if shared_target_available(tick_us) else 'waiting'}"
                    )
                else:
                    latch_latest_plan(source, tick_us)
                blended_q = None
            elif requested == "damped_return":
                shared_control.reset()

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
        enabled = bool(enabled)
        if enabled == planner_lower_body_idle:
            log(
                f"{source} SONIC lower-body source: staying "
                f"{'zero-root planner' if enabled else 'raw SMPL'}"
            )
            return
        planner_lower_body_idle = enabled
        mode = "zero-root planner" if enabled else "raw SMPL"
        log(f"{source} SONIC lower-body source: {mode}")
        if current_state in {"teleop", "intention"}:
            send_mode_command(sonic, planner=planner_lower_body_idle)
            next_mode_command = time.monotonic() + max(0.05, args.command_period_sec)

    if keyboard.enabled:
        log(
            "keyboard hotkeys enabled: q=teleop, w=intention, e=auto_planning, "
            "r=damped_return, p=IPOPT arm disable rejected, "
            "m=SONIC lower-body conditioning planner/raw-SMPL toggle"
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

    next_deadline = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            tick_us = now_us()
            dt = max(1.0e-4, now - last_tick)
            last_tick = now

            # ABXY has a dedicated 100Hz source so pose/FK/IK work cannot delay
            # mode commands. The timestamps are carried all the way to g1_debug.
            try:
                while True:
                    event_raw = button_sub.recv(zmq.NOBLOCK)
                    event = unpack_prefixed_msgpack(event_raw, args.button_topic)
                    sequence = int(event.get("sequence", 0))
                    if sequence <= last_button_sequence:
                        continue
                    last_button_sequence = sequence
                    dedicated_button_events_seen = True
                    event_name = str(event.get("event", ""))
                    if event_name not in {"start_combo", "ax", "by"}:
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
                        if stop:
                            current_state = "teleop"
                            smpl_targets.reset()
                    elif event_name == "ax":
                        requested = (
                            "teleop"
                            if current_state in {"auto_planning", "damped_return"}
                            else "auto_planning"
                        )
                        apply_state_request(requested, "Pico A+X", tick_us)
                        planner = requested == "auto_planning" or planner_lower_body_idle
                    elif event_name == "by":
                        planner = True
                    bridge_forward_us = now_us()
                    sonic.send(
                        build_command_message(
                            start=start,
                            stop=stop,
                            planner=planner,
                            button_sequence=sequence,
                            button_read_timestamp_us=int(
                                event.get("button_read_timestamp_us", 0)
                            ),
                            bridge_receive_timestamp_us=bridge_receive_us,
                            bridge_forward_timestamp_us=bridge_forward_us,
                        )
                    )
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
                        start, stop, planner = flags
                        if start:
                            policy_started = True
                        if stop:
                            sonic.send(raw)
                            policy_started = False
                            current_state = "teleop"
                            shared_control.reset()
                            smpl_targets.reset()
                            continue

                        if planner and current_state in {"teleop", "intention"}:
                            if start:
                                # A+B+X+Y startup/alignment uses start=True, planner=True.
                                sonic.send(raw)
                            else:
                                apply_state_request(
                                    "auto_planning", "Pico command", tick_us
                                )
                                sonic.send(raw)
                            continue

                        if (not planner) and current_state in {
                            "auto_planning",
                            "damped_return",
                        }:
                            log(f"A+X/pose command: {current_state} -> teleop")
                            current_state = "teleop"
                            shared_control.reset()
                            sonic.send(raw)
                            continue

                        if (
                            current_state in {"teleop", "intention"}
                            and planner_lower_body_idle
                        ):
                            # Safe q/w default: keep SONIC in planner mode and
                            # prevent raw SMPL/Pico pose-mode commands from
                            # re-enabling lower/root tracking. Press m first
                            # if full raw SMPL passthrough is intentionally needed.
                            if start:
                                sonic.send(
                                    build_command_message(
                                        start=True, stop=False, planner=True
                                    )
                                )
                            else:
                                send_mode_command(sonic, planner=True)
                            continue

                        sonic.send(raw)
                    else:
                        sonic.send(raw)
            except zmq.Again:
                pass

            # Pose has latest-only semantics. If the 50Hz loop misses a
            # deadline, ZeroMQ replaces the stale sample instead of replaying a
            # burst of old SMPL frames on the next tick.
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
                if current_state in {"teleop", "intention"} and not planner_lower_body_idle:
                    sonic.send(raw)
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
                sonic.send(raw)
            except zmq.Again:
                pass

            try:
                while True:
                    incoming_plan = unpack_topic(plan_sub.recv(zmq.NOBLOCK), args.plan_topic)
                    incoming_feasible = bool(incoming_plan.get("feasible", False))
                    incoming_plan_id = int(incoming_plan.get("plan_id", 0))
                    incoming_cps = len(incoming_plan.get("control_points", []))
                    if incoming_feasible and incoming_cps >= 4:
                        latest_plan = incoming_plan
                        latest_plan_us = tick_us
                        if current_state == "auto_planning":
                            active_plan_id = int(active_plan.get("plan_id", 0)) if active_plan else 0
                            if active_plan is None or incoming_plan_id != active_plan_id:
                                latch_latest_plan("auto_planning/plan_update", tick_us)
                    else:
                        # Infeasible/expired planner heartbeats are status
                        # messages, not executable references. Do not let them
                        # overwrite the last feasible plan before the bridge has
                        # a chance to latch it on an A+X/e transition.
                        reason = str(incoming_plan.get("failure_reason", ""))
                        if (
                            reason == "plan_expired"
                            and active_plan is not None
                            and int(active_plan.get("plan_id", 0)) == incoming_plan_id
                        ):
                            clear_active_plan("server plan_expired")
                    log(
                        "plan update "
                        f"id={incoming_plan_id} feasible={incoming_plan.get('feasible')} "
                        f"cps={incoming_cps}"
                    )
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
                            static_target_hold_q = None
                            static_target_hold_plan_id = -1
                            static_target_hold_side = ""
                            arm_ik_cached_revision = -1
                            arm_ik_cached_q_ref = None
                            arm_ik_cached_tau_ff = None
                            arm_ik_cached_feasible = False
                            arm_ik_cached_source = "null_target"
                            shared_control.reset()
                        continue
                    position = np.asarray(position_value, dtype=np.float64).reshape(-1)
                    if position.shape[0] == 3 and np.isfinite(position).all():
                        explicit_reset = bool(
                            incoming_target.get("new_goal", False)
                            or incoming_target.get("reset_hold", False)
                        )
                        is_new_goal = (
                            latest_static_target is None
                            or explicit_reset
                            or (incoming_uid and incoming_uid != latest_static_target_uid)
                            or (
                                args.auto_planning_mode != "shared_control"
                                and target_side != static_target_hold_side
                            )
                        )
                        latest_static_target = incoming_target
                        latest_static_target_us = tick_us
                        static_target_revision += 1
                        if is_new_goal:
                            latest_static_target_uid = incoming_uid
                            static_target_plan_id += 1
                            static_target_hold_q = None
                            static_target_hold_plan_id = -1
                            static_target_hold_side = (
                                target_side
                                if args.auto_planning_mode == "shared_control"
                                else ""
                            )
                            arm_ik_cached_revision = -1
                            arm_ik_cached_q_ref = None
                            arm_ik_cached_tau_ff = None
                            arm_ik_cached_feasible = False
                            arm_ik_cached_source = "new_goal"
                            shared_control.reset()
                            if current_state != "auto_planning":
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
            except zmq.Again:
                pass

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
                and smpl_age_sec <= max(0.01, args.smpl_timeout_sec)
            )
            if (
                smpl_fresh
                and latest_robot_q is not None
                and arm_nonlinear_ik is not None
            ):
                robot_left_tf, robot_right_tf = (
                    arm_nonlinear_ik.wrist_transforms_from_q29(latest_robot_q)
                )
                latest_smpl_targets = smpl_targets.update(
                    smpl_joints=latest_smpl_joints,
                    robot_left_transform=robot_left_tf,
                    robot_right_transform=robot_right_tf,
                    source_timestamp_us=latest_smpl_source_us,
                    frame_index=latest_smpl_frame,
                    dt=dt,
                )
            elif latest_smpl_targets is not None:
                latest_smpl_targets = smpl_targets.hold(
                    reason="stale_smpl" if not smpl_fresh else "missing_robot_state"
                )

            if (
                current_state in {"teleop", "intention"}
                and not planner_lower_body_idle
                and latest_pose_raw is not None
                and latest_smpl_frame != last_forwarded_pose_frame
            ):
                pose_to_send = latest_pose_raw
                released_pose_frame = latest_smpl_frame
                released_pose_frames += 1
                last_forwarded_pose_frame = latest_smpl_frame

            left_hand_cmd, right_hand_cmd, hand_fresh = current_pico_hand_joints(tick_us)
            plan_fresh = latest_plan is not None and (tick_us - latest_plan_us) <= args.plan_timeout_sec * 1.0e6
            static_target_age_sec = (
                (tick_us - latest_static_target_us) / 1.0e6
                if latest_static_target is not None
                else float("inf")
            )
            static_target_fresh = (
                latest_static_target is not None
                and static_target_age_sec <= args.static_target_timeout_sec
            )
            static_target_usable = (
                latest_static_target is not None
                and (
                    args.static_target_hold_sec <= 0.0
                    or static_target_age_sec <= args.static_target_hold_sec
                )
            )
            if active_plan is not None:
                elapsed_sec = (tick_us - active_plan_us) / 1.0e6
                move_duration_sec = float(active_plan.get("duration_sec", 0.0))
                active_total_sec = float(active_plan.get("total_duration_sec", 0.0))
                if move_duration_sec > 0.0 and elapsed_sec >= move_duration_sec:
                    target_distance_m = active_plan_target_distance()
                    if (
                        args.auto_clear_on_target_reached and
                        np.isfinite(target_distance_m)
                        and target_distance_m <= args.plan_complete_radius_m
                    ):
                        clear_active_plan(
                            f"target reached d={100.0 * target_distance_m:.1f}cm "
                            f"<= {100.0 * args.plan_complete_radius_m:.1f}cm"
                        )
                if (
                    active_plan is not None
                    and args.clear_plan_after_total_duration
                    and active_total_sec > 0.0
                    and elapsed_sec >= active_total_sec + 0.25
                ):
                    clear_active_plan("duration+hold elapsed")
            if current_state == "auto_planning" and active_plan is None and latest_plan_ready(tick_us):
                latch_latest_plan("auto_planning/pending", tick_us)

            qp_payload: dict | None = None
            if current_state in {"teleop", "intention"}:
                if planner_lower_body_idle:
                    pose_to_send = None
                if latest_smpl_targets is not None and latest_smpl_targets.valid:
                    ik_result = solve_authoritative_ipopt_transforms(
                        latest_smpl_targets.left_transform,
                        latest_smpl_targets.right_transform,
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
                        )
                    source_timestamp_us = int(latest_smpl_targets.source_timestamp_us)
                    if ik_result.success:
                        qp_payload = ipopt_arm_sample_reference(
                            ik_result.q_ref,
                            state=current_state,
                            source="smpl_relative_cartesian",
                            control_mode="teleop",
                            feasible=True,
                            plan_id=0,
                            timestamp_us=tick_us,
                            tau_ff_ref=ik_result.tau_ff_ref,
                            bridge_timestamp_us=tick_us,
                            smpl_source_timestamp_us=source_timestamp_us,
                            smpl_frame_index=int(latest_smpl_targets.frame_index),
                            smpl_target_reason=str(latest_smpl_targets.reason),
                            smpl_target_held=bool(latest_smpl_targets.held),
                            smpl_left_target_position=latest_smpl_targets.left_transform[:3, 3].tolist(),
                            smpl_right_target_position=latest_smpl_targets.right_transform[:3, 3].tolist(),
                            q_prior_source="ipopt_smpl_relative_ee",
                            belief_enabled=False,
                            shared_control=False,
                            teleop_qp_bimanual=True,
                            teleop_waist_locked=True,
                            arm_ik_solve_ms=float(ik_result.solve_time_ms),
                            arm_ik_position_error_m=float(ik_result.position_error_m),
                            arm_ik_rotation_error_rad=float(ik_result.rotation_error_rad),
                            arm_ik_status="ok",
                        )
                    else:
                        qp_payload = qp_none_reference(
                            state=current_state,
                            source="casadi_ipopt",
                            control_mode="teleop",
                            feasible=False,
                            plan_id=0,
                            timestamp_us=tick_us,
                            bridge_timestamp_us=tick_us,
                            smpl_source_timestamp_us=source_timestamp_us,
                            smpl_frame_index=int(latest_smpl_targets.frame_index),
                            arm_target_owner="casadi_ipopt",
                            solver_backend="casadi_opti_ipopt",
                            arm_target_valid=False,
                            arm_ik_solve_ms=float(ik_result.solve_time_ms),
                            arm_ik_status=str(ik_result.message),
                        )
                else:
                    qp_payload = qp_none_reference(
                        state=current_state,
                        source="casadi_ipopt",
                        control_mode="teleop",
                        feasible=False,
                        timestamp_us=tick_us,
                        bridge_timestamp_us=tick_us,
                        smpl_source_timestamp_us=int(latest_smpl_source_us),
                        smpl_frame_index=int(latest_smpl_frame),
                        smpl_target_reason=(
                            latest_smpl_targets.reason
                            if latest_smpl_targets is not None
                            else "waiting_smpl_anchor"
                        ),
                        arm_target_owner="casadi_ipopt",
                        solver_backend="casadi_opti_ipopt",
                        arm_target_valid=False,
                    )
            elif current_state == "auto_planning":
                if args.auto_planning_mode == "shared_control":
                    tau_ff_ref = np.zeros(29, dtype=np.float64)
                    target = (
                        ee_target_from_static_payload(
                            latest_static_target,
                            received_us=(
                                0
                                if args.static_target_hold_sec <= 0.0
                                else latest_static_target_us
                            ),
                        )
                        if static_target_usable
                        else None
                    )
                    requested_target_link = str(
                        latest_static_target.get("target_link", "")
                        if latest_static_target
                        else ""
                    )
                    requested_hand = target_side_from_link(requested_target_link)
                    if (
                        latest_smpl_targets is not None
                        and latest_smpl_targets.valid
                        and arm_nonlinear_ik is not None
                    ):
                        try:
                            human_left_tf = latest_smpl_targets.left_transform.copy()
                            human_right_tf = latest_smpl_targets.right_transform.copy()
                            p_human_left = human_left_tf[:3, 3].copy()
                            p_human_right = human_right_tf[:3, 3].copy()
                            q_measured_for_hand = (
                                latest_robot_q
                                if latest_robot_q is not None
                                else last_ipopt_or_measured_hold()
                            )
                            measured_left_tf, measured_right_tf = (
                                arm_nonlinear_ik.wrist_transforms_from_q29(
                                    q_measured_for_hand
                                )
                            )
                            p_measured_left = measured_left_tf[:3, 3].copy()
                            p_measured_right = measured_right_tf[:3, 3].copy()
                            result = shared_control.update(
                                target=target,
                                p_human_left=p_human_left,
                                p_human_right=p_human_right,
                                p_measured_left=p_measured_left,
                                p_measured_right=p_measured_right,
                                now_us=tick_us,
                                dt=dt,
                                blending_enabled=True,
                                preferred_hand=requested_hand,
                            )
                            latest_shared_result = result
                            active_target = result.active_p_ref
                            left_target_tf = human_left_tf.copy()
                            right_target_tf = human_right_tf.copy()
                            if active_target is not None and result.hand == "left":
                                left_target_tf[:3, 3] = active_target
                            if active_target is not None and result.hand == "right":
                                right_target_tf[:3, 3] = active_target
                            ik_result = solve_authoritative_ipopt_transforms(
                                left_target_tf,
                                right_target_tf,
                            )
                            q_for_sonic = (
                                ik_result.q_ref.copy()
                                if ik_result.success
                                else last_ipopt_or_measured_hold()
                            )
                            # The planner input may condition SONIC's lower-body
                            # policy, but PATH owns every waist/arm PD target.
                            q_for_sonic[:15] = 0.0
                            send_planner_command(
                                sonic,
                                q_for_sonic,
                                neutralize_waist=True,
                                freeze_lower_body=True,
                                left_hand_position=left_hand_cmd,
                                right_hand_position=right_hand_cmd,
                            )
                            last_plan_q = q_for_sonic.copy()
                            blended_q = q_for_sonic.copy()
                            source_timestamp_us = int(
                                latest_smpl_targets.source_timestamp_us
                            )
                            if not ik_result.success:
                                qp_payload = qp_none_reference(
                                    state="auto_planning",
                                    source="casadi_ipopt",
                                    control_mode="plan_exec",
                                    feasible=False,
                                    plan_id=static_target_plan_id,
                                    timestamp_us=tick_us,
                                    bridge_timestamp_us=tick_us,
                                    smpl_source_timestamp_us=source_timestamp_us,
                                    smpl_frame_index=int(latest_smpl_targets.frame_index),
                                    arm_target_owner="casadi_ipopt",
                                    solver_backend="casadi_opti_ipopt",
                                    arm_target_valid=False,
                                    arm_ik_solve_ms=float(ik_result.solve_time_ms),
                                    arm_ik_position_error_m=float(
                                        ik_result.position_error_m
                                    ),
                                    arm_ik_rotation_error_rad=float(
                                        ik_result.rotation_error_rad
                                    ),
                                    arm_ik_status=str(ik_result.message),
                                    shared_gate_reason=result.gate_reason,
                                )
                                phase_mode = "ipopt_failed_hold"
                            elif active_target is None:
                                # No object target yet: e mode keeps the exact
                                # same IPOPT arm ownership as q mode and uses
                                # the SMPL-relative wrist poses (zero Cartesian blend).
                                qp_payload = ipopt_arm_sample_reference(
                                    ik_result.q_ref,
                                    state="auto_planning",
                                    source="smpl_relative_cartesian",
                                    control_mode="plan_exec",
                                    feasible=True,
                                    plan_id=static_target_plan_id,
                                    timestamp_us=tick_us,
                                    tau_ff_ref=ik_result.tau_ff_ref,
                                    bridge_timestamp_us=tick_us,
                                    smpl_source_timestamp_us=source_timestamp_us,
                                    smpl_frame_index=int(latest_smpl_targets.frame_index),
                                    q_prior_source="ipopt_smpl_relative_ee",
                                    arm_ik_solve_ms=float(ik_result.solve_time_ms),
                                    arm_ik_position_error_m=float(
                                        ik_result.position_error_m
                                    ),
                                    arm_ik_rotation_error_rad=float(
                                        ik_result.rotation_error_rad
                                    ),
                                    arm_ik_status="ok",
                                    shared_control=True,
                                    shared_gate_reason=result.gate_reason,
                                )
                                phase_mode = "shared_control_wait"
                            else:
                                qp_payload = ipopt_arm_cartesian_target_reference(
                                    target_link=result.target_link,
                                    target_position=active_target,
                                    q_ref=ik_result.q_ref,
                                    state="auto_planning",
                                    source="cartesian_servo",
                                    control_mode="plan_exec",
                                    feasible=True,
                                    plan_id=static_target_plan_id,
                                    timestamp_us=tick_us,
                                    terminal_capture_radius_m=float(args.terminal_capture_radius_m),
                                    tau_ff_ref=ik_result.tau_ff_ref,
                                    bridge_timestamp_us=tick_us,
                                    smpl_source_timestamp_us=source_timestamp_us,
                                    smpl_frame_index=int(latest_smpl_targets.frame_index),
                                    target_source=str(
                                        latest_static_target.get("source", "sam3_tasc_contact")
                                        if latest_static_target
                                        else "none"
                                    ),
                                    q_prior_source="ipopt_blended_ee_pose",
                                    has_q_ik_prior=True,
                                    arm_ik_mode="casadi_opti_ipopt_bimanual",
                                    arm_ik_solve_ms=float(ik_result.solve_time_ms),
                                    arm_ik_position_error_m=float(
                                        ik_result.position_error_m
                                    ),
                                    arm_ik_rotation_error_rad=float(
                                        ik_result.rotation_error_rad
                                    ),
                                    arm_ik_status="ok",
                                    shared_control=True,
                                    shared_alpha=float(result.alpha),
                                    shared_alpha_raw=float(result.alpha_raw),
                                    shared_alpha_rate_eff=float(result.alpha_rate_eff),
                                    shared_gap_m=float(result.gap_m),
                                    shared_confidence=float(result.confidence),
                                    shared_hand=str(result.hand),
                                    shared_requested_hand=str(requested_hand),
                                    shared_requested_target_link=str(requested_target_link),
                                    shared_gate_reason=str(result.gate_reason),
                                    shared_target_switched=bool(result.target_switched),
                                    shared_object_id=str(result.object_id),
                                    shared_grip_index=int(result.grip_index),
                                    shared_lock_inactive_arm_q_prior=bool(
                                        args.shared_lock_inactive_arm_q_prior
                                    ),
                                    shared_p_contact=None
                                    if result.p_contact is None
                                    else result.p_contact.tolist(),
                                    shared_p_human_left=result.p_human_left.tolist(),
                                    shared_p_human_right=result.p_human_right.tolist(),
                                    shared_p_ref_left=result.p_ref_left.tolist(),
                                    shared_p_ref_right=result.p_ref_right.tolist(),
                                    shared_max_observed_confidence=float(
                                        shared_control.max_observed_confidence
                                    ),
                                )
                                phase_mode = "shared_control"
                            phase_sync_error_rad = float("nan")
                        except Exception as exc:
                            shared_control.reset()
                            latest_shared_result = None
                            qp_payload = qp_none_reference(
                                state="auto_planning",
                                source="shared_control",
                                control_mode="plan_exec",
                                feasible=False,
                                plan_id=static_target_plan_id,
                                timestamp_us=tick_us,
                                shared_gate_reason=f"error:{exc}",
                            )
                            phase_mode = "shared_control_error"
                            phase_sync_error_rad = float("nan")
                    else:
                        missing = []
                        if latest_smpl_targets is None or not latest_smpl_targets.valid:
                            missing.append("smpl_target")
                        if arm_nonlinear_ik is None:
                            missing.append("ipopt_fk")
                        if target is None:
                            missing.append("target")
                        latest_shared_result = None
                        qp_payload = qp_none_reference(
                            state="auto_planning",
                            source="shared_control",
                            control_mode="plan_exec",
                            feasible=False,
                            plan_id=static_target_plan_id,
                            timestamp_us=tick_us,
                            shared_gate_reason="missing_" + "_".join(missing),
                        )
                        phase_mode = "shared_control_wait"
                        phase_sync_error_rad = float("nan")
                elif static_target_usable:
                    target_link = str(latest_static_target.get("target_link", "left_wrist_yaw_link"))
                    target_position = np.asarray(
                        latest_static_target.get(
                            "terminal_target_position",
                            latest_static_target.get("position", [0.0, 0.0, 0.0]),
                        ),
                        dtype=np.float64,
                    ).reshape(3)
                    q_hold = (
                        latest_robot_q.copy()
                        if latest_robot_q is not None
                        else np.zeros(29, dtype=np.float64)
                    )
                    q_hold = frozen_planner_q29(
                        q_hold,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    target_side = target_side_from_link(target_link)
                    q_hold[12:15] = 0.0
                    if (
                        static_target_hold_q is None
                        or static_target_hold_plan_id != static_target_plan_id
                        or static_target_hold_side != target_side
                    ):
                        static_target_hold_q = q_hold.copy()
                        static_target_hold_plan_id = static_target_plan_id
                        static_target_hold_side = target_side
                        log(
                            "latched inactive-arm hold "
                            f"plan={static_target_plan_id} side={target_side or 'unknown'}"
                        )
                    outer_waist.maybe_reset(q_hold, static_target_plan_id, target_side)
                    q_command = hold_inactive_arm_for_single_target(
                        q_hold, static_target_hold_q, target_side)
                    q_command = frozen_planner_q29(
                        q_command,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    q_ik_prior, q_ik_prior_source = q_ik_prior_from_plan(
                        latest_plan,
                        latest_static_target,
                        target_link,
                        tick_us,
                        args.plan_timeout_sec,
                    )
                    q_prior = target_arm_prior_from_ik(
                        q_command,
                        q_ik_prior,
                        target_side,
                    )
                    q_prior = hold_inactive_arm_for_single_target(
                        q_prior, static_target_hold_q, target_side)
                    q_prior = frozen_planner_q29(
                        q_prior,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    tau_ff_ref = np.zeros(29, dtype=np.float64)
                    arm_ik_feasible = True
                    if arm_nonlinear_ik is not None:
                        if (
                            arm_ik_cached_revision != static_target_revision
                            or arm_ik_cached_q_ref is None
                            or arm_ik_cached_tau_ff is None
                        ):
                            target_quat = np.asarray(
                                latest_static_target.get(
                                    "quaternion",
                                    latest_static_target.get("target_quaternion", [1.0, 0.0, 0.0, 0.0]),
                                ),
                                dtype=np.float64,
                            ).reshape(4)
                            ik_seed = (
                                latest_robot_q.copy()
                                if latest_robot_q is not None
                                else q_hold.copy()
                            )
                            ik_result = arm_nonlinear_ik.solve_single_target(
                                target_link=target_link,
                                target_position=target_position,
                                target_quaternion_wxyz=target_quat,
                                q_seed29=ik_seed,
                                q_hold29=static_target_hold_q,
                                hard_lock_inactive_arm=args.arm_ik_hard_lock_inactive_arm,
                            )
                            last_arm_ik_solve_ms = ik_result.solve_time_ms
                            last_arm_ik_error_m = ik_result.position_error_m
                            last_arm_ik_rot_error_rad = ik_result.rotation_error_rad
                            last_arm_ik_status = "ok" if ik_result.success else ik_result.message
                            arm_ik_cached_revision = static_target_revision
                            arm_ik_cached_feasible = bool(ik_result.success)
                            arm_ik_cached_source = (
                                f"unitree_ipopt_arm_ik:{ik_result.solve_time_ms:.2f}ms/"
                                f"{100.0 * ik_result.position_error_m:.2f}cm/"
                                f"{ik_result.rotation_error_rad:.3f}rad"
                            )
                            if ik_result.success:
                                arm_ik_cached_q_ref = hold_inactive_arm_for_single_target(
                                    ik_result.q_ref.copy(), static_target_hold_q, target_side)
                                arm_ik_cached_q_ref = frozen_planner_q29(
                                    arm_ik_cached_q_ref,
                                    freeze_lower_body=bool(args.freeze_planner_lower_body),
                                    freeze_waist=bool(args.freeze_planner_waist),
                                )
                                arm_ik_cached_tau_ff = ik_result.tau_ff_ref.copy()
                            else:
                                arm_ik_cached_q_ref = hold_inactive_arm_for_single_target(
                                    q_command, static_target_hold_q, target_side)
                                arm_ik_cached_q_ref = frozen_planner_q29(
                                    arm_ik_cached_q_ref,
                                    freeze_lower_body=bool(args.freeze_planner_lower_body),
                                    freeze_waist=bool(args.freeze_planner_waist),
                                )
                                arm_ik_cached_tau_ff = np.zeros(29, dtype=np.float64)
                        elif arm_ik_cached_feasible:
                            last_arm_ik_status = "cached_ok"

                        arm_ik_feasible = bool(arm_ik_cached_feasible)
                        q_prior = hold_inactive_arm_for_single_target(
                            arm_ik_cached_q_ref.copy(), static_target_hold_q, target_side)
                        q_prior = frozen_planner_q29(
                            q_prior,
                            freeze_lower_body=bool(args.freeze_planner_lower_body),
                            freeze_waist=bool(args.freeze_planner_waist),
                        )
                        tau_ff_ref = arm_ik_cached_tau_ff.copy()
                        q_ik_prior_source = arm_ik_cached_source
                    terminal_error_m = float("nan")
                    if wrist_fk is not None:
                        try:
                            q_for_error = (
                                latest_robot_q
                                if latest_robot_q is not None
                                else q_hold
                            )
                            terminal_error_m = float(np.linalg.norm(
                                wrist_fk.wrist_position(q_for_error, target_link)
                                - target_position
                            ))
                        except Exception:
                            terminal_error_m = float("nan")
                    q_command = frozen_planner_q29(
                        q_command,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    q_prior = frozen_planner_q29(
                        q_prior,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    q_prior = hold_inactive_arm_for_single_target(
                        q_prior, static_target_hold_q, target_side)
                    q_prior = frozen_planner_q29(
                        q_prior,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        freeze_waist=bool(args.freeze_planner_waist),
                    )
                    send_planner_command(
                        sonic,
                        q_command,
                        neutralize_waist=True,
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        left_hand_position=left_hand_cmd,
                        right_hand_position=right_hand_cmd,
                    )
                    last_plan_q = q_command.copy()
                    blended_q = q_prior.copy()
                    qp_payload = qp_cartesian_target_reference(
                        target_link=target_link,
                        target_position=target_position,
                        q_ref=q_prior,
                        state="auto_planning",
                        source="cartesian_servo",
                        control_mode="plan_exec",
                        feasible=arm_ik_feasible,
                        plan_id=static_target_plan_id,
                        timestamp_us=tick_us,
                        terminal_capture_radius_m=float(args.terminal_capture_radius_m),
                        tau_ff_ref=tau_ff_ref,
                        target_source=str(latest_static_target.get("source", "manual_static_target")),
                        q_prior_source=q_ik_prior_source,
                        has_q_ik_prior=(q_ik_prior is not None) or (arm_nonlinear_ik is not None and arm_ik_feasible),
                        arm_ik_mode="unitree_ipopt" if arm_nonlinear_ik is not None else "external_plan_prior",
                        arm_ik_solve_ms=float(last_arm_ik_solve_ms),
                        arm_ik_position_error_m=float(last_arm_ik_error_m),
                        arm_ik_rotation_error_rad=float(last_arm_ik_rot_error_rad),
                        arm_ik_status=str(last_arm_ik_status),
                    )
                    phase_mode = "cartesian_target"
                    phase_sync_error_rad = float("nan")
                else:
                  plan = active_plan
                  feasible_plan = bool(plan and plan.get("feasible", False))
                  if feasible_plan:
                    cps = np.asarray(plan.get("control_points", []), dtype=np.float64)
                    if cps.ndim == 2 and cps.shape[1] == 29:
                        cps = np.vstack([
                            frozen_planner_q29(
                                row,
                                freeze_lower_body=bool(args.freeze_planner_lower_body),
                                freeze_waist=bool(args.freeze_planner_waist),
                            )
                            for row in cps
                        ])
                    dt_plan = float(plan.get("control_dt_sec", 0.02))
                    t0 = int(plan.get("t_start_us", plan.get("timestamp_us", tick_us)))
                    try:
                        robot_fresh = (
                            latest_robot_q is not None
                            and (tick_us - latest_robot_state_us) <= 0.25 * 1.0e6
                        )
                        max_segment = float(cps.shape[0] - 4)
                        if args.phase_tracking_mode == "fixed_rate":
                            elapsed_sec = max(0.0, (tick_us - active_plan_us) / 1.0e6)
                            query_s = (elapsed_sec + max(0.0, args.phase_lookahead_sec)) / max(
                                dt_plan, 1.0e-6
                            )
                            active_plan_s = float(np.clip(query_s, 0.0, max_segment))
                            q_plan, phase = bspline_eval_s(cps, active_plan_s)
                            phase_mode = "fixed_rate"
                            phase_sync_error_rad = float("nan")
                        elif args.phase_tracking_mode == "state_aligned" and robot_fresh:
                            best_s, phase_sync_error_rad = nearest_bspline_s(
                                cps,
                                latest_robot_q,
                                active_plan_s,
                                dt_plan,
                                args.phase_search_window_sec,
                                args.phase_search_samples,
                            )
                            advance_s = (dt + max(0.0, args.phase_lookahead_sec)) / max(dt_plan, 1.0e-6)
                            if active_plan_s is None:
                                query_s = best_s + advance_s
                            else:
                                query_s = max(active_plan_s, best_s) + advance_s
                            active_plan_s = float(np.clip(query_s, 0.0, max_segment))
                            q_plan, phase = bspline_eval_s(cps, active_plan_s)
                            phase_mode = "state_aligned"
                        else:
                            wall_t0 = active_plan_us if args.phase_tracking_mode == "wall_time" else t0
                            q_plan, phase = bspline_eval(cps, dt_plan, wall_t0, tick_us)
                            active_plan_s = phase * max_segment
                            phase_mode = "wall_time"
                            phase_sync_error_rad = float("nan")
                        q_plan = frozen_planner_q29(
                            q_plan,
                            freeze_lower_body=bool(args.freeze_planner_lower_body),
                            freeze_waist=bool(args.freeze_planner_waist),
                        )
                        last_plan_q = q_plan.copy()
                        blended_q = q_plan.copy()
                        send_planner_command(
                            sonic,
                            q_plan,
                            neutralize_waist=True,
                            freeze_lower_body=bool(args.freeze_planner_lower_body),
                            left_hand_position=left_hand_cmd,
                            right_hand_position=right_hand_cmd,
                        )
                        # Publish the same cuRobo spline that produced SONIC's
                        # q_plan. Re-anchor t_start every tick to the
                        # state-aligned phase so the deploy-side QP provider
                        # evaluates the active reference from the same plan,
                        # instead of consuming a separate stale sample stream.
                        aligned_t_start_us = int(
                            tick_us - active_plan_s * dt_plan * 1.0e6
                        )
                        qp_payload = qp_bspline_reference(
                            cps.tolist(),
                            control_dt_sec=dt_plan,
                            t_start_us=aligned_t_start_us,
                            state="auto_planning",
                            source="curobo",
                            control_mode="plan_exec",
                            feasible=True,
                            plan_id=int(plan.get("plan_id", 0)),
                            phase=phase,
                            timestamp_us=tick_us,
                            terminal_target_link=str(plan.get("target_link", "left_wrist_yaw_link")),
                            terminal_target_position=list(
                                plan.get(
                                    "terminal_target_position",
                                    plan.get("target_position", [0.0, 0.0, 0.0]),
                                )
                            ),
                            terminal_capture_radius_m=float(args.terminal_capture_radius_m),
                            phase_mode=phase_mode,
                            phase_sync_error_rad=float(phase_sync_error_rad),
                        )
                    except Exception as exc:
                        log(f"invalid active plan: {exc}")
                        qp_payload = qp_none_reference(
                            state="auto_planning",
                            source="curobo",
                            control_mode="plan_pending",
                            feasible=False,
                            plan_id=int(plan.get("plan_id", 0)) if plan else 0,
                            timestamp_us=tick_us,
                        )
                  else:
                    qp_payload = qp_none_reference(
                        state="auto_planning",
                        source="curobo",
                        control_mode="plan_pending",
                        feasible=False,
                        plan_id=int(active_plan.get("plan_id", 0)) if active_plan else 0,
                        timestamp_us=tick_us,
                    )
            else:  # damped_return
                if latest_smpl_targets is not None and latest_smpl_targets.valid:
                    return_result = solve_authoritative_ipopt_transforms(
                        latest_smpl_targets.left_transform,
                        latest_smpl_targets.right_transform,
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
                    send_planner_command(
                        sonic,
                        blended_q,
                        neutralize_waist=bool(args.freeze_planner_waist),
                        freeze_lower_body=bool(args.freeze_planner_lower_body),
                        left_hand_position=left_hand_cmd,
                        right_hand_position=right_hand_cmd,
                    )
                    qp_payload = ipopt_arm_sample_reference(
                        blended_q,
                        tau_ff_ref=return_result.tau_ff_ref,
                        state="damped_return",
                        source="smpl_relative_damped_return",
                        control_mode="blend_back",
                        feasible=bool(return_result.success),
                        plan_id=int(latest_plan.get("plan_id", 0)) if latest_plan else 0,
                        timestamp_us=tick_us,
                        smpl_source_timestamp_us=int(
                            latest_smpl_targets.source_timestamp_us
                        ),
                        smpl_frame_index=int(latest_smpl_targets.frame_index),
                    )
                else:
                    qp_payload = qp_none_reference(
                        state="damped_return",
                        source="damped_return",
                        control_mode="blend_back",
                        feasible=False,
                        timestamp_us=tick_us,
                    )

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
                qp_payload["smpl_frame_index"] = int(latest_smpl_frame)
                qp_payload["smpl_source_timestamp_us"] = int(latest_smpl_source_us)
                qp_payload["smpl_input_age_ms"] = float(
                    1000.0 * smpl_age_sec if np.isfinite(smpl_age_sec) else -1.0
                )
                qp_payload["smpl_target_valid"] = bool(
                    latest_smpl_targets is not None and latest_smpl_targets.valid
                )
                qp_payload["smpl_target_reason"] = (
                    latest_smpl_targets.reason
                    if latest_smpl_targets is not None
                    else "waiting_smpl_anchor"
                )

            if now >= next_mode_command:
                send_mode_command(sonic, planner=planner_mode_for_current_state())
                next_mode_command = now + max(0.05, args.command_period_sec)

            if qp_payload is not None:
                qp.send(pack_topic(args.qp_topic, qp_payload))
            if pose_to_send is not None:
                sonic.send(pose_to_send)

            if now - last_log > 1.0:
                shared_log = "shared=none"
                if latest_shared_result is not None:
                    shared_log = (
                        f"shared_alpha={latest_shared_result.alpha:.3f} "
                        f"shared_raw={latest_shared_result.alpha_raw:.3f} "
                        f"shared_rate={latest_shared_result.alpha_rate_eff:.3f}/s "
                        f"shared_gap={latest_shared_result.gap_m if np.isfinite(latest_shared_result.gap_m) else float('nan'):.3f}m "
                        f"shared_conf={latest_shared_result.confidence:.3f} "
                        f"shared_hand={latest_shared_result.hand} "
                        f"shared_gate={latest_shared_result.gate_reason} "
                        f"shared_obj={latest_shared_result.object_id or 'none'} "
                        f"shared_grip={latest_shared_result.grip_index}"
                    )
                log(
                    f"state={current_state} smpl_fresh={smpl_fresh} plan_fresh={plan_fresh} "
                    f"static_target={'fresh' if static_target_fresh else 'held' if static_target_usable else 'missing'} "
                    f"static_age={static_target_age_sec if np.isfinite(static_target_age_sec) else float('nan'):.2f}s "
                    f"static_rev={static_target_revision} "
                    f"active_plan={active_plan.get('plan_id', 0) if active_plan else 0} "
                    f"qp_gate={'on' if qp_reference_enabled else 'off'} "
                    f"lower_src={'idle_planner' if planner_lower_body_idle else 'raw_smpl'} "
                    f"freeze_lower={bool(args.freeze_planner_lower_body)} "
                    f"freeze_waist={bool(args.freeze_planner_waist)} "
                    f"hand_fresh={hand_fresh} "
                    f"trig=({latest_hand_meta.get('left_trigger', float('nan')):.2f},"
                    f"{latest_hand_meta.get('right_trigger', float('nan')):.2f}) "
                    f"grip=({latest_hand_meta.get('left_grip', float('nan')):.2f},"
                    f"{latest_hand_meta.get('right_grip', float('nan')):.2f}) "
                    f"qp_type={qp_payload.get('reference_type') if qp_payload else 'none'} "
                    f"source={qp_payload.get('source', 'none') if qp_payload else 'none'} "
                    f"q_prior={qp_payload.get('q_prior_source', 'n/a') if qp_payload else 'n/a'} "
                    f"smpl_target={latest_smpl_targets.reason if latest_smpl_targets else 'waiting'} "
                    f"smpl_age={1000.0 * smpl_age_sec if np.isfinite(smpl_age_sec) else -1.0:.1f}ms "
                    f"released={released_pose_frame}/{released_pose_frames} "
                    f"pose_frame={latest_smpl_frame} "
                    f"button_seq={last_button_sequence} "
                    f"control={qp_payload.get('control_mode', 'none') if qp_payload else 'none'} "
                    f"plan={qp_payload.get('plan_id', 0) if qp_payload else 0} "
                    f"phase={float(qp_payload.get('phase', 0.0)) if qp_payload else 0.0:.2f} "
                    f"phase_mode={phase_mode} "
                    f"phase_err={phase_sync_error_rad if np.isfinite(phase_sync_error_rad) else float('nan'):.3f}rad "
                    f"arm_ik={last_arm_ik_status} "
                    f"ik_err={100.0 * last_arm_ik_error_m if np.isfinite(last_arm_ik_error_m) else float('nan'):.2f}cm "
                    f"ik_solve={last_arm_ik_solve_ms if np.isfinite(last_arm_ik_solve_ms) else float('nan'):.2f}ms "
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
        pico_pose.close()
        pico_command.close()
        pico_planner.close()
        pico_manager.close()
        button_sub.close()
        plan_sub.close()
        robot_state.close()
        state_sub.close()
        sonic.close()
        qp.close()
        keyboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
