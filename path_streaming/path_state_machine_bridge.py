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
"""

from __future__ import annotations

import argparse
import atexit
import json
import select
import struct
import sys
import termios
import time
import tty

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
    now_us,
    qp_bspline_reference,
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
        "--no-keyboard",
        action="store_true",
        help="Disable terminal hotkeys q/w/e/r for PATH state switching.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[PATH-state] {message}", flush=True)


class KeyboardController:
    """Tiny non-blocking stdin controller for live demo state switching."""

    KEY_TO_STATE = {
        "q": "teleop",
        "w": "intention",
        "e": "auto_planning",
        "r": "damped_return",
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

    def poll_state(self) -> str | None:
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        return self.KEY_TO_STATE.get(key.lower())


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


def upper_from_q29(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64).reshape(29)[UPPER_BODY_PLANNER_INDICES].astype(np.float32)


def send_planner_command(socket: zmq.Socket, q29: np.ndarray, height: float) -> None:
    upper = upper_from_q29(q29)
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
    log(f"state command SUB {args.state_command_bind}/{args.state_command_topic}")
    log(f"SONIC output PUB tcp://{args.sonic_bind_host}:{args.sonic_port}")
    log(f"QP reference PUB tcp://{args.qp_bind_host}:{args.qp_port}/{args.qp_topic}")

    current_state = args.state
    latest_gmr: dict | None = None
    latest_gmr_us = 0
    latest_plan: dict | None = None
    latest_plan_us = 0
    blended_q: np.ndarray | None = None
    last_plan_q: np.ndarray | None = None
    next_mode_command = 0.0
    last_tick = time.monotonic()
    last_log = 0.0
    period = 1.0 / max(args.fps, 1.0e-6)
    keyboard = KeyboardController(enabled=not args.no_keyboard)

    def apply_state_request(requested: str, source: str) -> None:
        nonlocal current_state, next_mode_command, blended_q
        if requested not in VALID_STATES:
            log(f"ignored invalid {source} state command: {requested}")
            return
        if requested != current_state:
            log(f"{source} state command: {current_state} -> {requested}")
        else:
            log(f"{source} state command: staying in {requested}")
        current_state = requested
        if requested in {"teleop", "intention"}:
            send_mode_command(sonic, planner=False)
            next_mode_command = time.monotonic() + max(0.05, args.command_period_sec)
        else:
            send_mode_command(sonic, planner=True)
            next_mode_command = 0.0
            if requested == "auto_planning":
                blended_q = None

    if keyboard.enabled:
        log("keyboard hotkeys enabled: q=teleop, w=intention, e=auto_planning, r=damped_return")
    else:
        log("keyboard hotkeys disabled; use tcp://*:5563/path_state_command")

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
                                plan_ready = bool(
                                    latest_plan
                                    and latest_plan.get("feasible", False)
                                    and (tick_us - latest_plan_us) <= args.plan_timeout_sec * 1.0e6
                                )
                                if plan_ready:
                                    log(f"A+X/planner command: {current_state} -> auto_planning")
                                    current_state = "auto_planning"
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
                    latest_plan = unpack_topic(plan_sub.recv(zmq.NOBLOCK), args.plan_topic)
                    latest_plan_us = tick_us
                    log(
                        "plan update "
                        f"id={latest_plan.get('plan_id', 0)} feasible={latest_plan.get('feasible')} "
                        f"cps={len(latest_plan.get('control_points', []))}"
                    )
            except zmq.Again:
                pass

            try:
                while True:
                    command = unpack_topic(state_sub.recv(zmq.NOBLOCK), args.state_command_topic)
                    requested = str(command.get("state", current_state))
                    apply_state_request(requested, "external")
            except zmq.Again:
                pass

            requested_from_keyboard = keyboard.poll_state()
            if requested_from_keyboard is not None:
                apply_state_request(requested_from_keyboard, "keyboard")

            q_gmr = latest_q_from_gmr(latest_gmr)
            gmr_fresh = q_gmr is not None and (tick_us - latest_gmr_us) <= args.gmr_timeout_sec * 1.0e6
            plan_fresh = latest_plan is not None and (tick_us - latest_plan_us) <= args.plan_timeout_sec * 1.0e6

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
                feasible_plan = bool(plan_fresh and latest_plan and latest_plan.get("feasible", False))
                if feasible_plan:
                    cps = np.asarray(latest_plan.get("control_points", []), dtype=np.float64)
                    dt_plan = float(latest_plan.get("control_dt_sec", 0.02))
                    t0 = int(latest_plan.get("t_start_us", latest_plan.get("timestamp_us", tick_us)))
                    try:
                        q_plan, phase = bspline_eval(cps, dt_plan, t0, tick_us)
                        last_plan_q = q_plan.copy()
                        blended_q = q_plan.copy()
                        send_planner_command(sonic, q_plan, args.height)
                        qp_payload = qp_bspline_reference(
                            cps,
                            control_dt_sec=dt_plan,
                            t_start_us=t0,
                            state="auto_planning",
                            source="curobo",
                            control_mode="plan_exec",
                            feasible=True,
                            plan_id=int(latest_plan.get("plan_id", 0)),
                            phase=phase,
                            timestamp_us=tick_us,
                        )
                    except Exception as exc:
                        log(f"invalid active plan: {exc}")
                        qp_payload = qp_none_reference(
                            state="auto_planning",
                            source="curobo",
                            control_mode="plan_pending",
                            feasible=False,
                            plan_id=int(latest_plan.get("plan_id", 0)) if latest_plan else 0,
                            timestamp_us=tick_us,
                        )
                else:
                    qp_payload = qp_none_reference(
                        state="auto_planning",
                        source="curobo",
                        control_mode="plan_pending",
                        feasible=False,
                        plan_id=int(latest_plan.get("plan_id", 0)) if latest_plan else 0,
                        timestamp_us=tick_us,
                    )
            else:  # damped_return
                if gmr_fresh:
                    if blended_q is None:
                        blended_q = last_plan_q.copy() if last_plan_q is not None else q_gmr.copy()
                    alpha = 1.0 - np.exp(-args.damped_return_rate * dt)
                    blended_q = (1.0 - alpha) * blended_q + alpha * q_gmr
                    send_planner_command(sonic, blended_q, args.height)
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

            if current_state in {"auto_planning", "damped_return"} and now >= next_mode_command:
                send_mode_command(sonic, planner=True)
                next_mode_command = now + max(0.05, args.command_period_sec)

            if qp_payload is not None:
                qp.send(pack_topic(args.qp_topic, qp_payload))

            if now - last_log > 1.0:
                log(
                    f"state={current_state} gmr_fresh={gmr_fresh} plan_fresh={plan_fresh} "
                    f"qp_type={qp_payload.get('reference_type') if qp_payload else 'none'} "
                    f"source={qp_payload.get('source', 'none') if qp_payload else 'none'} "
                    f"control={qp_payload.get('control_mode', 'none') if qp_payload else 'none'} "
                    f"plan={qp_payload.get('plan_id', 0) if qp_payload else 0} "
                    f"phase={float(qp_payload.get('phase', 0.0)) if qp_payload else 0.0:.2f} "
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
        state_sub.close()
        sonic.close()
        qp.close()
        keyboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
