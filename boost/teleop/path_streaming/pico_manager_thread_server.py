#!/usr/bin/env python3

"""PICO controller and wrist-target inputs for PATH-G1.

Publishes `pico_button_event` and `pico_controller_state` on port 5565, plus
calibrated waist-frame wrist targets on port 5566. This process publishes no
robot commands.
"""

import subprocess
import threading
import time
from enum import Enum

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import zmq

from boost.io.path_transport.zmq_wire import pack_pose_message
from boost.teleop.xr.pico_button_gestures import PicoButtonGestureInterpreter

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

class StreamMode(Enum):
    OFF = 0
    TRACKING = 7


T_ROBOT_OPENXR = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
T_OPENXR_ROBOT = np.array(
    [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
WAIST_OFFSET_XYZ = np.array([0.15, 0.0, 0.45], dtype=np.float64)
CONST_HEAD_POSE = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]],
    dtype=np.float64,
)
CONST_LEFT_ARM_POSE = np.array(
    [[1, 0, 0, -0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]],
    dtype=np.float64,
)
CONST_RIGHT_ARM_POSE = np.array(
    [[1, 0, 0, 0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]],
    dtype=np.float64,
)

# G1 URDF zero pose: upper arms hang vertically from the shoulders while the
# forearms point forward (+X), giving the requested geometric 90-degree elbow.
def _pico_pose_to_mat_openxr(
    pose7: np.ndarray | list[float],
) -> tuple[np.ndarray, bool]:
    """PICO SDK [x,y,z,qx,qy,qz,qw] to OpenXR-basis SE(3)."""

    p = np.asarray(pose7, dtype=np.float64).reshape(7)
    if not np.isfinite(p).all() or np.linalg.norm(p[3:]) < 1.0e-8:
        return np.eye(4, dtype=np.float64), False
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = sRot.from_quat(p[3:]).as_matrix()
    transform[:3, 3] = p[:3]
    return transform, True


def world_arm_to_waist_head_position(
    arm_world: np.ndarray,
    head_world: np.ndarray,
) -> np.ndarray:
    """Current head position to waist; no yaw, freeze, anchor, or gain."""

    transform = np.asarray(arm_world, dtype=np.float64).reshape(4, 4).copy()
    head = np.asarray(head_world, dtype=np.float64).reshape(4, 4)
    transform[:3, 3] = (
        transform[:3, 3] - head[:3, 3] + WAIST_OFFSET_XYZ
    )
    return transform


def controller_waist_targets(
    left_pose7: np.ndarray | list[float],
    right_pose7: np.ndarray | list[float],
    head_pose7: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """PICO raw controller poses to absolute waist-frame wrist SE(3)."""

    head_xr, head_ok = _pico_pose_to_mat_openxr(head_pose7)
    if not head_ok:
        head_xr = CONST_HEAD_POSE.copy()
    left_xr, left_ok = _pico_pose_to_mat_openxr(left_pose7)
    if not left_ok:
        left_xr = CONST_LEFT_ARM_POSE.copy()
    right_xr, right_ok = _pico_pose_to_mat_openxr(right_pose7)
    if not right_ok:
        right_xr = CONST_RIGHT_ARM_POSE.copy()

    head_robot = T_ROBOT_OPENXR @ head_xr @ T_OPENXR_ROBOT
    left_robot = T_ROBOT_OPENXR @ left_xr @ T_OPENXR_ROBOT
    right_robot = T_ROBOT_OPENXR @ right_xr @ T_OPENXR_ROBOT
    left_waist = world_arm_to_waist_head_position(left_robot, head_robot)
    right_waist = world_arm_to_waist_head_position(right_robot, head_robot)
    return left_waist, right_waist, bool(head_ok and left_ok and right_ok)


def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    return left_trigger, right_trigger, left_grip, right_grip


def get_abxy_buttons():
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False


class PicoButtonEventPublisher:
    """Publish ABXY events and trigger/grip state for PATH arm control."""

    def __init__(self, *, port: int = 5565, poll_hz: float = 100.0):
        self.port = int(port)
        self.poll_hz = max(20.0, float(poll_hz))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = (False, False, False, False)
        self._sequence = 0
        self._controller_sequence = 0
        self._thread = threading.Thread(
            target=self._run, name="pico-abxy-poller", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> tuple[bool, bool, bool, bool]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 32)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://*:{self.port}")
        print(
            f"[PicoButtons] polling ABXY at {self.poll_hz:.0f}Hz; "
            f"publishing tcp://*:{self.port}/pico_button_event + "
            "pico_controller_state",
            flush=True,
        )
        gestures = PicoButtonGestureInterpreter()
        period = 1.0 / self.poll_hz
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                buttons = get_abxy_buttons()
                # Keep analogue hand controls on this dedicated 100 Hz path.
                # vr_pose_debug is conflated downstream for Cartesian arm
                # tracking, so a short trigger press must not depend on which
                # body-pose frame happens to survive that conflation.
                left_trigger, right_trigger, left_grip, right_grip = (
                    get_controller_inputs()
                )
                read_timestamp_us = time.monotonic_ns() // 1000
                with self._lock:
                    self._latest = buttons
                a, b, x, y = buttons
                self._controller_sequence += 1
                controller_payload = {
                    "schema": "path.pico_controller_state.v1",
                    "sequence": self._controller_sequence,
                    "controller_read_timestamp_us": int(read_timestamp_us),
                    "publish_timestamp_us": int(time.monotonic_ns() // 1000),
                    "left_trigger": float(left_trigger),
                    "right_trigger": float(right_trigger),
                    "left_grip": float(left_grip),
                    "right_grip": float(right_grip),
                    "a": bool(a),
                    "b": bool(b),
                    "x": bool(x),
                    "y": bool(y),
                }
                socket.send(
                    b"pico_controller_state"
                    + msgpack.packb(controller_payload, use_bin_type=True)
                )

                events = gestures.update(buttons)
                for event in events:
                    self._sequence += 1
                    payload = {
                        "schema": "path.pico_button_event.v1",
                        "event": event,
                        "sequence": self._sequence,
                        "button_read_timestamp_us": int(read_timestamp_us),
                        "publish_timestamp_us": int(time.monotonic_ns() // 1000),
                        "a": bool(a),
                        "b": bool(b),
                        "x": bool(x),
                        "y": bool(y),
                    }
                    socket.send(
                        b"pico_button_event"
                        + msgpack.packb(payload, use_bin_type=True)
                    )
                deadline += period
                delay = deadline - time.monotonic()
                if delay > 0.0:
                    self._stop.wait(delay)
                elif delay < -period:
                    deadline = time.monotonic()
        finally:
            socket.close()
            context.term()


class PicoReader:
    """Track fresh XR timestamps used to pace wrist-target publication."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_monotonic = time.monotonic()
            try:
                sample = {
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")


class PicoVrPoseDebugPublisher:
    """Publish absolute PICO wrist targets and hand-control metadata."""

    def __init__(
        self,
        *,
        reader: PicoReader,
        session_id_getter,
        stream_mode_getter,
        port: int = 5566,
        publish_hz: float = 90.0,
    ):
        self.reader = reader
        self.session_id_getter = session_id_getter
        self.stream_mode_getter = stream_mode_getter
        self.port = int(port)
        self.publish_hz = min(90.0, max(50.0, float(publish_hz)))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="pico-vr-pose-publisher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://*:{self.port}")
        period = 1.0 / self.publish_hz
        deadline = time.monotonic()
        last_timestamp_ns = -1
        print(f"[PICO-VR] publishing tcp://*:{self.port}/vr_pose_debug", flush=True)
        try:
            while not self._stop.is_set():
                sample = self.reader.get_latest()
                timestamp_ns = int(sample.get("timestamp_ns", 0)) if sample else 0
                if sample is not None and timestamp_ns > 0 and timestamp_ns != last_timestamp_ns:
                    try:
                        mode = self.stream_mode_getter()
                        active = mode == StreamMode.TRACKING
                        left = np.full((4, 4), np.nan, dtype=np.float64)
                        right = np.full((4, 4), np.nan, dtype=np.float64)
                        targets_valid = False
                        if active:
                            left, right, targets_valid = controller_waist_targets(
                                xrt.get_left_controller_pose(),
                                xrt.get_right_controller_pose(),
                                xrt.get_headset_pose(),
                            )
                        left_trigger, right_trigger, left_grip, right_grip = (
                            get_controller_inputs()
                        )
                        socket.send(
                            pack_pose_message(
                                {
                                    "wrist_left_waist": (
                                        left.reshape(-1).astype(np.float32)
                                        if targets_valid
                                        else np.full(16, np.nan, dtype=np.float32)
                                    ),
                                    "wrist_right_waist": (
                                        right.reshape(-1).astype(np.float32)
                                        if targets_valid
                                        else np.full(16, np.nan, dtype=np.float32)
                                    ),
                                    "targets_valid": np.array([targets_valid and active], dtype=bool),
                                    "frame_index": np.array([timestamp_ns], dtype=np.int64),
                                    "timestamp_monotonic": np.array(
                                        [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                                    ),
                                    "pico_dt": np.array([sample.get("dt", 0.0)], dtype=np.float32),
                                    "pico_fps": np.array([sample.get("fps", 0.0)], dtype=np.float32),
                                    "ee_frame_offset_m": np.array([0.05], dtype=np.float32),
                                    "stream_mode": np.array([mode.value], dtype=np.int32),
                                    "session_id": np.array(
                                        [int(self.session_id_getter())], dtype=np.int32
                                    ),
                                    "arm_reference_mode": np.array([0], dtype=np.int32),
                                    "left_trigger": np.array([left_trigger], dtype=np.float32),
                                    "right_trigger": np.array([right_trigger], dtype=np.float32),
                                    "left_grip": np.array([left_grip], dtype=np.float32),
                                    "right_grip": np.array([right_grip], dtype=np.float32),
                                },
                                topic="vr_pose_debug",
                            )
                        )
                        last_timestamp_ns = timestamp_ns
                    except Exception as exc:
                        print(f"[PICO-VR] skipped frame: {exc}", flush=True)
                deadline += period
                delay = deadline - time.monotonic()
                if delay > 0.0:
                    self._stop.wait(delay)
                else:
                    deadline = time.monotonic()
        finally:
            socket.close()
            context.term()


def run_pico_manager(
    button_event_port: int = 5565,
    button_poll_hz: float = 100.0,
    vr_pose_debug_port: int = 5566,
    vr_pose_debug_hz: float = 90.0,
):
    """Publish PICO controller state and calibrated waist-frame wrist targets."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    reader = PicoReader()
    reader.start()
    button_publisher = PicoButtonEventPublisher(
        port=button_event_port, poll_hz=button_poll_hz
    )
    button_publisher.start()
    current_mode = StreamMode.OFF
    session_id = 0
    vr_pose_publisher = PicoVrPoseDebugPublisher(
        reader=reader,
        session_id_getter=lambda: session_id,
        port=vr_pose_debug_port,
        publish_hz=vr_pose_debug_hz,
        stream_mode_getter=lambda: current_mode,
    )
    vr_pose_publisher.start()
    previous_start_combo = False
    print(
        "PICO deploy inputs: A+X arms PATH tracking; A+B+X+Y starts/stops wrist targets.",
        flush=True,
    )
    try:
        while True:
            a_pressed, b_pressed, x_pressed, y_pressed = button_publisher.latest()
            start_combo = bool(a_pressed and b_pressed and x_pressed and y_pressed)
            if start_combo and not previous_start_combo:
                if current_mode == StreamMode.OFF:
                    session_id += 1
                    current_mode = StreamMode.TRACKING
                else:
                    current_mode = StreamMode.OFF
                print(
                    f"[PICO] wrist targets={'on' if current_mode == StreamMode.TRACKING else 'off'}",
                    flush=True,
                )
            previous_start_combo = start_combo
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping PICO manager...")
    finally:
        vr_pose_publisher.stop()
        button_publisher.stop()
        reader.stop()
        print("[PICO] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Publish PICO controller state and arm wrist targets for PATH-G1."
    )
    parser.add_argument("--manager", action="store_true", help="Run the PATH input publisher.")
    parser.add_argument("--button_event_port", type=int, default=5565)
    parser.add_argument("--button_poll_hz", type=float, default=100.0)
    parser.add_argument("--vr_pose_debug_port", type=int, default=5566)
    parser.add_argument("--vr_pose_debug_hz", type=float, default=90.0)
    args = parser.parse_args()
    if not args.manager:
        parser.error("--manager is required")
    run_pico_manager(
        button_event_port=args.button_event_port,
        button_poll_hz=args.button_poll_hz,
        vr_pose_debug_port=args.vr_pose_debug_port,
        vr_pose_debug_hz=args.vr_pose_debug_hz,
    )
