#!/usr/bin/env python3
"""PATH state-machine bridge between Pico/GMR/cuRobo and SONIC.

This bridge is the only process that publishes to SONIC's ZMQ manager port.
Other producers are subordinate inputs:

  Pico manager     -> tcp://127.0.0.1:5559/{pose,command,planner,manager_state}
  GMR reference    -> tcp://127.0.0.1:5558/gmr_ref
  cuRobo planner   -> tcp://127.0.0.1:5562/curobo_plan
  State command    -> tcp://*:5563/path_state_command
  SONIC policy     <- tcp://*:5556/{pose,command,planner,manager_state}
  EE-QP reference  <- tcp://*:5560/path_qp_reference

State behavior:
  teleop/intention:
    Forward Pico SMPL pose stream to SONIC. Forward Pico command/planner for the
    normal A+B+X+Y and A+X SONIC interaction. Publish GMR retargeted q_ref to QP.
  auto_planning:
    Suppress Pico pose. Evaluate cuRobo B-spline at the current tick and send it
    as SONIC upper-body/root planner command. Publish the same spline to QP.
  damped_return:
    Suppress Pico pose. Blend from current planned q toward GMR q_ref and send it
    as SONIC upper-body/root planner command. Publish the blend to QP.

Runtime hotkeys:
  q/w/e/r switch PATH state.
  p toggles whether this bridge publishes a valid QP reference. When disabled,
    the bridge still forwards the proper SONIC input, but QP receives an
    infeasible none-reference and falls back to the uncorrected SONIC target.
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
from path_core.messaging.path_schema import (
    CUROBO_PLAN_TOPIC,
    QP_REFERENCE_TOPIC,
    STATIC_EE_TARGET_TOPIC,
    now_us,
    qp_bspline_reference,
    qp_cartesian_target_reference,
    qp_none_reference,
    qp_sample_reference,
)
from path_core.messaging.zmq_bus import pack_topic, unpack_topic
from path_streaming.path_smpl_thread_server import UPPER_BODY_PLANNER_INDICES

VALID_STATES = {"teleop", "intention", "auto_planning", "damped_return"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=sorted(VALID_STATES), default="teleop")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--height", type=float, default=0.74)

    parser.add_argument("--pico-endpoint", default="tcp://127.0.0.1:5559")
    parser.add_argument("--gmr-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--gmr-topic", default="gmr_ref")
    parser.add_argument("--plan-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--plan-topic", default=CUROBO_PLAN_TOPIC)
    parser.add_argument("--static-target-bind", default="tcp://*:5564")
    parser.add_argument("--static-target-topic", default=STATIC_EE_TARGET_TOPIC)
    parser.add_argument("--static-target-timeout-sec", type=float, default=0.25)
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
    parser.add_argument("--gmr-timeout-sec", type=float, default=0.20)
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
    KEY_TO_STATE = {
        "q": "teleop",
        "w": "intention",
        "e": "auto_planning",
        "r": "damped_return",
        "p": TOGGLE_QP,
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


def send_planner_command(
    socket: zmq.Socket,
    q29: np.ndarray,
    height: float,
    neutralize_waist: bool = False,
) -> None:
    q_for_command = np.asarray(q29, dtype=np.float64).reshape(29).copy()
    if neutralize_waist:
        q_for_command[12:15] = 0.0
    upper = upper_from_q29(q_for_command)
    socket.send(
        build_planner_message(
            0,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            speed=-1.0,
            height=height,
            upper_body_position=upper.tolist(),
            upper_body_velocity=np.zeros(17, dtype=np.float32).tolist(),
            left_hand_position=np.zeros(7, dtype=np.float32).tolist(),
            right_hand_position=np.zeros(7, dtype=np.float32).tolist(),
        )
    )


def send_mode_command(socket: zmq.Socket, planner: bool) -> None:
    socket.send(build_command_message(start=False, stop=False, planner=planner))


def latest_q_from_gmr(payload: dict | None) -> np.ndarray | None:
    if not payload:
        return None
    q = payload.get("q_ref", payload.get("q_mujoco"))
    if q is None:
        return None
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 29 or not np.isfinite(arr).all():
        return None
    return arr


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
    ctx = zmq.Context.instance()

    pico = ctx.socket(zmq.SUB)
    pico.setsockopt(zmq.RCVTIMEO, 1)
    for topic in ("pose", "command", "planner", "manager_state"):
        pico.setsockopt(zmq.SUBSCRIBE, topic.encode())
    pico.connect(args.pico_endpoint)

    gmr = ctx.socket(zmq.SUB)
    gmr.setsockopt(zmq.SUBSCRIBE, args.gmr_topic.encode())
    gmr.setsockopt(zmq.CONFLATE, 1)
    gmr.setsockopt(zmq.RCVTIMEO, 1)
    gmr.connect(args.gmr_endpoint)

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
    log(f"GMR input SUB {args.gmr_endpoint}/{args.gmr_topic}")
    log(f"cuRobo plan SUB {args.plan_endpoint}/{args.plan_topic}")
    log(f"static EE target SUB {args.static_target_bind}/{args.static_target_topic}")
    log(f"robot state SUB {args.robot_state_endpoint}/{args.robot_state_topic}")
    log(f"state command SUB {args.state_command_bind}/{args.state_command_topic}")
    log(f"SONIC output PUB tcp://{args.sonic_bind_host}:{args.sonic_port}")
    log(f"QP reference PUB tcp://{args.qp_bind_host}:{args.qp_port}/{args.qp_topic}")

    current_state = args.state
    latest_gmr: dict | None = None
    latest_gmr_us = 0
    latest_plan: dict | None = None
    latest_plan_us = 0
    latest_static_target: dict | None = None
    latest_static_target_us = 0
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
    phase_mode = "none"
    phase_sync_error_rad = float("nan")
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
    wrist_fk: MujocoWristFk | None = None
    try:
        wrist_fk = MujocoWristFk(args.model_xml)
        log(f"loaded MuJoCo FK model for plan completion: {args.model_xml}")
    except Exception as exc:
        log(f"WARNING: plan completion FK disabled: {exc}")

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

    def apply_state_request(requested: str, source: str, tick_us: int | None = None) -> None:
        nonlocal current_state, next_mode_command, blended_q, active_plan, active_plan_us
        nonlocal active_plan_s, phase_mode, phase_sync_error_rad
        if tick_us is None:
            tick_us = now_us()
        if requested not in VALID_STATES:
            log(f"ignored invalid {source} state command: {requested}")
            return
        if requested != current_state:
            log(f"{source} state command: {current_state} -> {requested}")
        else:
            log(f"{source} state command: staying in {requested}")
        current_state = requested
        if requested in {"teleop", "intention"}:
            active_plan = None
            active_plan_us = 0
            active_plan_s = None
            phase_mode = "none"
            phase_sync_error_rad = float("nan")
            send_mode_command(sonic, planner=False)
            next_mode_command = time.monotonic() + max(0.05, args.command_period_sec)
        else:
            send_mode_command(sonic, planner=True)
            next_mode_command = 0.0
            if requested == "auto_planning":
                latch_latest_plan(source, tick_us)
                blended_q = None

    def set_qp_reference_enabled(enabled: bool, source: str) -> None:
        nonlocal qp_reference_enabled
        enabled = bool(enabled)
        if enabled == qp_reference_enabled:
            log(f"{source} QP reference: staying {'ON' if enabled else 'OFF'}")
            return
        qp_reference_enabled = enabled
        log(f"{source} QP reference: {'ON' if qp_reference_enabled else 'OFF'}")

    if keyboard.enabled:
        log("keyboard hotkeys enabled: q=teleop, w=intention, e=auto_planning, r=damped_return, p=QP on/off")
    else:
        log("keyboard hotkeys disabled; use tcp://*:5563/path_state_command")
    log(f"QP reference gate initial state: {'ON' if qp_reference_enabled else 'OFF'}")

    try:
        while True:
            now = time.monotonic()
            tick_us = now_us()
            dt = max(1.0e-4, now - last_tick)
            last_tick = now

            try:
                while True:
                    raw = pico.recv(zmq.NOBLOCK)
                    if starts(raw, "command"):
                        flags = command_flags(raw)
                        if flags is not None:
                            start, stop, planner = flags
                            if stop:
                                sonic.send(raw)
                                current_state = "teleop"
                                continue

                            if planner and current_state in {"teleop", "intention"}:
                                plan_ready = latest_plan_ready(tick_us)
                                if plan_ready:
                                    log(f"A+X/planner command: {current_state} -> auto_planning")
                                    current_state = "auto_planning"
                                    latch_latest_plan("A+X/planner", tick_us)
                                    sonic.send(raw)
                                elif start:
                                    # A+B+X+Y startup/alignment uses start=True, planner=True.
                                    sonic.send(raw)
                                else:
                                    log("A+X planner request ignored: no fresh feasible cuRobo plan")
                                continue

                            if (not planner) and current_state in {"auto_planning", "damped_return"}:
                                log(f"A+X/pose command: {current_state} -> teleop")
                                current_state = "teleop"
                                sonic.send(raw)
                                continue

                            sonic.send(raw)
                        else:
                            sonic.send(raw)
                    elif starts(raw, "pose"):
                        if current_state in {"teleop", "intention"}:
                            sonic.send(raw)
                    elif starts(raw, "planner"):
                        if current_state in {"teleop", "intention"}:
                            sonic.send(raw)
                    elif starts(raw, "manager_state"):
                        sonic.send(raw)
            except zmq.Again:
                pass

            try:
                while True:
                    latest_gmr = unpack_prefixed_msgpack(gmr.recv(zmq.NOBLOCK), args.gmr_topic)
                    latest_gmr_us = tick_us
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
                    position = np.asarray(
                        incoming_target.get("position", incoming_target.get("target_position", [])),
                        dtype=np.float64,
                    ).reshape(-1)
                    if position.shape[0] == 3 and np.isfinite(position).all():
                        incoming_uid = str(incoming_target.get("goal_uid", ""))
                        target_link = str(incoming_target.get("target_link", "left_wrist_yaw_link"))
                        target_side = target_side_from_link(target_link)
                        is_new_goal = (
                            not incoming_uid
                            or incoming_uid != latest_static_target_uid
                            or target_side != static_target_hold_side
                        )
                        latest_static_target = incoming_target
                        latest_static_target_us = tick_us
                        if is_new_goal:
                            latest_static_target_uid = incoming_uid
                            static_target_plan_id += 1
                            static_target_hold_q = None
                            static_target_hold_plan_id = -1
                            static_target_hold_side = ""
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
                    if "state" in command:
                        requested = str(command.get("state", current_state))
                        apply_state_request(requested, "external", tick_us)
            except zmq.Again:
                pass

            keyboard_command = keyboard.poll_command()
            if keyboard_command == KeyboardController.TOGGLE_QP:
                set_qp_reference_enabled(not qp_reference_enabled, "keyboard")
            elif keyboard_command is not None:
                apply_state_request(keyboard_command, "keyboard", tick_us)

            q_gmr = latest_q_from_gmr(latest_gmr)
            gmr_fresh = q_gmr is not None and (tick_us - latest_gmr_us) <= args.gmr_timeout_sec * 1.0e6
            plan_fresh = latest_plan is not None and (tick_us - latest_plan_us) <= args.plan_timeout_sec * 1.0e6
            static_target_fresh = (
                latest_static_target is not None
                and (tick_us - latest_static_target_us) <= args.static_target_timeout_sec * 1.0e6
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
                if gmr_fresh:
                    qp_payload = qp_sample_reference(
                        q_gmr,
                        state=current_state,
                        source="gmr",
                        control_mode="teleop",
                        feasible=True,
                        plan_id=0,
                        timestamp_us=tick_us,
                    )
                else:
                    qp_payload = qp_none_reference(
                        state=current_state,
                        source="gmr",
                        control_mode="teleop",
                        feasible=False,
                        timestamp_us=tick_us,
                    )
            elif current_state == "auto_planning":
                if static_target_fresh:
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
                        else q_gmr.copy()
                        if q_gmr is not None
                        else np.zeros(29, dtype=np.float64)
                    )
                    target_side = target_side_from_link(target_link)
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
                    q_command = outer_waist.update(
                        dt=dt,
                        tick_us=tick_us,
                        terminal_error_m=terminal_error_m,
                        wrist_fk=wrist_fk,
                        q_command=q_command,
                        q_posture=q_prior,
                        target_link=target_link,
                        target_position=target_position,
                    )
                    q_prior[12:15] = q_command[12:15]
                    q_prior = hold_inactive_arm_for_single_target(
                        q_prior, static_target_hold_q, target_side)
                    send_planner_command(sonic, q_command, args.height, neutralize_waist=False)
                    last_plan_q = q_command.copy()
                    blended_q = q_prior.copy()
                    qp_payload = qp_cartesian_target_reference(
                        target_link=target_link,
                        target_position=target_position,
                        q_ref=q_prior,
                        state="auto_planning",
                        source="cartesian_servo",
                        control_mode="plan_exec",
                        feasible=True,
                        plan_id=static_target_plan_id,
                        timestamp_us=tick_us,
                        terminal_capture_radius_m=float(args.terminal_capture_radius_m),
                        target_source=str(latest_static_target.get("source", "manual_static_target")),
                        q_prior_source=q_ik_prior_source,
                        has_q_ik_prior=q_ik_prior is not None,
                    )
                    phase_mode = "cartesian_target"
                    phase_sync_error_rad = float("nan")
                else:
                  plan = active_plan
                  feasible_plan = bool(plan and plan.get("feasible", False))
                  if feasible_plan:
                    cps = np.asarray(plan.get("control_points", []), dtype=np.float64)
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
                        last_plan_q = q_plan.copy()
                        blended_q = q_plan.copy()
                        send_planner_command(sonic, q_plan, args.height, neutralize_waist=True)
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
                if gmr_fresh:
                    if blended_q is None:
                        blended_q = last_plan_q.copy() if last_plan_q is not None else q_gmr.copy()
                    alpha = 1.0 - np.exp(-args.damped_return_rate * dt)
                    blended_q = (1.0 - alpha) * blended_q + alpha * q_gmr
                    send_planner_command(sonic, blended_q, args.height, neutralize_waist=False)
                    qp_payload = qp_sample_reference(
                        blended_q,
                        state="damped_return",
                        source="damped_return",
                        control_mode="blend_back",
                        feasible=True,
                        plan_id=int(latest_plan.get("plan_id", 0)) if latest_plan else 0,
                        timestamp_us=tick_us,
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

            if current_state in {"auto_planning", "damped_return"} and now >= next_mode_command:
                send_mode_command(sonic, planner=True)
                next_mode_command = now + max(0.05, args.command_period_sec)

            if qp_payload is not None:
                qp.send(pack_topic(args.qp_topic, qp_payload))

            if now - last_log > 1.0:
                log(
                    f"state={current_state} gmr_fresh={gmr_fresh} plan_fresh={plan_fresh} "
                    f"static_target_fresh={static_target_fresh} "
                    f"active_plan={active_plan.get('plan_id', 0) if active_plan else 0} "
                    f"qp_gate={'on' if qp_reference_enabled else 'off'} "
                    f"qp_type={qp_payload.get('reference_type') if qp_payload else 'none'} "
                    f"source={qp_payload.get('source', 'none') if qp_payload else 'none'} "
                    f"q_prior={qp_payload.get('q_prior_source', 'n/a') if qp_payload else 'n/a'} "
                    f"control={qp_payload.get('control_mode', 'none') if qp_payload else 'none'} "
                    f"plan={qp_payload.get('plan_id', 0) if qp_payload else 0} "
                    f"phase={float(qp_payload.get('phase', 0.0)) if qp_payload else 0.0:.2f} "
                    f"phase_mode={phase_mode} "
                    f"phase_err={phase_sync_error_rad if np.isfinite(phase_sync_error_rad) else float('nan'):.3f}rad "
                    f"{outer_waist.status()} "
                    f"feasible={qp_payload.get('feasible') if qp_payload else False}"
                )
                last_log = now

            time.sleep(period)
    except KeyboardInterrupt:
        return 0
    finally:
        pico.close()
        gmr.close()
        plan_sub.close()
        robot_state.close()
        state_sub.close()
        sonic.close()
        qp.close()
        keyboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
