"""Record PICO input, SONIC policy command, and measured G1 state.

This is a motion-only recorder for MuJoCo/SONIC or real-G1 teleop sessions.
It always subscribes to:
  - PICO manager `pose` packets over ZMQ (PICO input → SONIC, port 5556)
  - SONIC policy command (`last_action` etc.) over ZMQ (`g1_debug`, port 5557)
  - Optional PATH cuRobo plan heartbeat (`curobo_plan`, port 5562)

For the *measured G1 joint state*, the source is selectable:
  --g1-source zmq  (default): same `g1_debug` ZMQ topic (MuJoCo sim or real G1
                    relayed by SONIC deploy's debug socket; state_body_q etc.)
  --g1-source lcm : G1_ROBOT_STATE LCM channel published directly by the
                    on-robot bridge (real G1, bypasses SONIC's debug socket).

Output:
  - frame_000000.npz, frame_000001.npz, ...
  - frames/frame_000000.jpg, ... (only if --record-image)
  - session_selected.npz with the most useful stacked arrays
  - manifest.json with topic/port/transport metadata

Run from repo root:
  # measured state via ZMQ g1_debug (MuJoCo sim, original behavior)
  python gear_sonic/scripts/record_sonic_motion_triplet.py --output-dir exp_output_01

  # measured state via LCM G1_ROBOT_STATE (real G1)
  python gear_sonic/scripts/record_sonic_motion_triplet.py --output-dir exp_output_01 \
      --g1-source lcm --lcm-url "udpm://239.255.76.67:7667?ttl=1"

  # also save ego-view camera frames (LCM G1_CAMERA_FRAME, real G1 only)
  python gear_sonic/scripts/record_sonic_motion_triplet.py --output-dir exp_output_01 \
      --g1-source lcm --record-image
"""

from __future__ import annotations

import argparse
import json
import queue
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import msgpack
import numpy as np
import zmq

try:
    import lcm as lcm_module
except ImportError:
    lcm_module = None


POSE_HEADER_SIZE = 1280

# ── LCM 패킷 상수 (g1_lcm_protocol.hpp / lcm_protocol.hpp 와 동일) ─────────────
LCM_ROBOT_STATE_MAGIC   = 0x31545347
LCM_CAMERA_FRAME_MAGIC  = 0x4D524143
DEFAULT_LCM_STATE_CHANNEL  = "G1_ROBOT_STATE"
DEFAULT_LCM_CAMERA_CHANNEL = "G1_CAMERA_FRAME"
DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"

# ── ZMQ 기본 포트 (파일 내 기본값, --pose-port/--state-port로 변경 가능) ──────
# Record the subordinate Pico stream before the state-machine bridge. In the
# safe default mode raw pose is intentionally not forwarded to SONIC:5556.
DEFAULT_POSE_PORT  = 5559
DEFAULT_STATE_PORT = 5557


@dataclass
class LatestMessage:
    data: dict[str, Any]
    wall_time: float
    monotonic_time: float


@dataclass
class WriteJob:
    """One frame's worth of work handed off to the background writer thread."""
    npz_path: Path
    frame: dict[str, np.ndarray]
    jpeg: Optional[bytes] = None
    jpeg_path: Optional[Path] = None


@dataclass
class LatestImage:
    jpeg: bytes
    width: int
    height: int
    device_ts_us: int
    wall_time: float
    monotonic_time: float


def unpack_pose_message(raw: bytes, topic: str) -> dict[str, Any]:
    """Unpack PICO manager binary ZMQ messages.

    Wire format: [topic_prefix][1280-byte JSON header][binary fields].
    """
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        raise ValueError(f"Message does not start with topic {topic!r}")

    offset = len(topic_bytes)
    if len(raw) < offset + POSE_HEADER_SIZE:
        raise ValueError(f"Message too small for {topic!r}: {len(raw)} bytes")

    header_bytes = raw[offset : offset + POSE_HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx >= 0:
        header_bytes = header_bytes[:null_idx]
    header = json.loads(header_bytes.decode("utf-8"))

    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": np.bool_,
    }
    result: dict[str, Any] = {
        "version": np.array([header.get("v", 0)], dtype=np.int32),
    }
    current_offset = offset + POSE_HEADER_SIZE

    for field in header.get("fields", []):
        dtype = np.dtype(dtype_map.get(field["dtype"], np.float32))
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * dtype.itemsize
        chunk = raw[current_offset : current_offset + n_bytes]
        result[field["name"]] = np.frombuffer(chunk, dtype=dtype).reshape(shape).copy()
        current_offset += n_bytes

    return result


def unpack_msgpack_topic(raw: bytes, topic: str) -> dict[str, Any]:
    """Unpack SONIC deploy topic-prefixed msgpack messages."""
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        raise ValueError(f"Message does not start with topic {topic!r}")
    payload = raw[len(topic_bytes) :]
    return msgpack.unpackb(payload, raw=False)


def unpack_lcm_robot_state(data: bytes) -> Optional[dict[str, Any]]:
    """Unpack G1_ROBOT_STATE LCM RobotStatePacket (#pragma pack 1, 461 bytes).

    Offsets match g1_lcm_protocol.hpp / lcm_protocol.hpp and the parsing
    already used by g1_web_server.py.
    """
    if len(data) < 33:
        return None
    magic, version, flags = struct.unpack_from("<IHH", data, 0)
    if magic != LCM_ROBOT_STATE_MAGIC:
        return None

    sequence, lowstate_seq, send_time_us = struct.unpack_from("<QQq", data, 8)
    mode = struct.unpack_from("B", data, 32)[0]

    off = 33
    base_quat   = list(struct.unpack_from("<4f", data, off)); off += 16
    base_gyro   = list(struct.unpack_from("<3f", data, off)); off += 12
    base_accel  = list(struct.unpack_from("<3f", data, off)); off += 12
    torso_quat  = list(struct.unpack_from("<4f", data, off)); off += 16
    torso_gyro  = list(struct.unpack_from("<3f", data, off)); off += 12
    torso_accel = list(struct.unpack_from("<3f", data, off)); off += 12
    q           = list(struct.unpack_from("<29f", data, off)); off += 116
    dq          = list(struct.unpack_from("<29f", data, off)); off += 116
    tau         = list(struct.unpack_from("<29f", data, off))

    return {
        "sequence": sequence,
        "lowstate_sequence": lowstate_seq,
        "send_time_us": send_time_us,
        "mode": mode,
        "base_quat": base_quat,
        "base_gyro": base_gyro,
        "base_accel": base_accel,
        "torso_quat": torso_quat,
        "torso_gyro": torso_gyro,
        "torso_accel": torso_accel,
        "body_q_measured": q,    # ZMQ 경로의 state_body_q_measured와 키를 맞춤
        "body_dq_measured": dq,
        "tau_est": tau,
    }


def unpack_lcm_camera_frame(data: bytes) -> Optional[tuple[bytes, int, int, int]]:
    """Unpack G1_CAMERA_FRAME LCM CameraFramePacket. Returns (jpeg, w, h, ts_us)."""
    if len(data) < 24:
        return None
    magic, w, h, jpeg_size = struct.unpack_from("<IIII", data, 0)
    if magic != LCM_CAMERA_FRAME_MAGIC:
        return None
    ts = struct.unpack_from("<Q", data, 16)[0]
    if len(data) < 24 + jpeg_size:
        return None
    jpeg = bytes(data[24: 24 + jpeg_size])
    return jpeg, w, h, ts


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if isinstance(value, (bool, int, float, np.number)):
        return np.asarray([value])
    if isinstance(value, str):
        return np.asarray(value)
    if value is None:
        return np.asarray([])
    return np.asarray(str(value))


def flatten_prefixed(prefix: str, data: dict[str, Any]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{key}": to_numpy(value) for key, value in data.items()}


def compose_frame(
    *,
    now_wall: float,
    now_mono: float,
    pose_age: float,
    state_age: float,
    sonic_age: float,
    stream_mode: np.ndarray,
    pose_data: dict[str, Any],
    state_data: dict[str, Any],
    sonic_data: dict[str, Any],
    manager_data: Optional[dict[str, Any]],
    path_plan_data: Optional[dict[str, Any]] = None,
    button_data: Optional[dict[str, Any]] = None,
    pico_target_data: Optional[dict[str, Any]] = None,
) -> dict[str, np.ndarray]:
    """Compose one raw synchronized sample without changing joint conventions."""
    frame: dict[str, np.ndarray] = {
        "record_wall_time": np.array([now_wall], dtype=np.float64),
        "record_monotonic_time": np.array([now_mono], dtype=np.float64),
        "pose_age_sec": np.array([pose_age], dtype=np.float64),
        "state_age_sec": np.array([state_age], dtype=np.float64),
        "sonic_age_sec": np.array([sonic_age], dtype=np.float64),
        "manager_stream_mode": stream_mode,
    }
    frame.update(flatten_prefixed("pose", pose_data))
    frame.update(flatten_prefixed("state", state_data))
    frame.update(flatten_prefixed("sonic", sonic_data))
    for timing_key in (
        "button_sequence",
        "button_read_timestamp_us",
        "button_bridge_receive_timestamp_us",
        "button_bridge_forward_timestamp_us",
        "button_sonic_receive_timestamp_us",
        "button_sonic_apply_timestamp_us",
    ):
        if timing_key in sonic_data:
            frame[timing_key] = np.asarray(sonic_data[timing_key]).copy()
    if manager_data is not None:
        frame.update(flatten_prefixed("manager", manager_data))
    if path_plan_data is not None:
        frame.update(flatten_prefixed("path_plan", path_plan_data))
    if pico_target_data is not None:
        frame.update(flatten_prefixed("pico_target", pico_target_data))
    if button_data is not None:
        button_fields = {
            "event": button_data.get("event", ""),
            "sequence": button_data.get("sequence", 0),
            "read_timestamp_us": button_data.get("button_read_timestamp_us", 0),
            "publish_timestamp_us": button_data.get("publish_timestamp_us", 0),
            "a": button_data.get("a", False),
            "b": button_data.get("b", False),
            "x": button_data.get("x", False),
            "y": button_data.get("y", False),
        }
        frame.update(flatten_prefixed("pico_button", button_fields))
    return frame


def missing_required_streams(
    *, pose: Any, state: Any, sonic: Any,
) -> list[str]:
    streams = {"pose": pose, "state": state, "sonic": sonic}
    return [name for name, value in streams.items() if value is None]


def format_missing_streams_message(
    missing: list[str], *, sonic_endpoint: str,
) -> str:
    parts: list[str] = []
    if "sonic" in missing:
        parts.append(
            f"WARNING: no SONIC ZMQ topic received from {sonic_endpoint}; "
            "last_action cannot be saved and no frames will be recorded "
            "until the topic is available"
        )
    other = [name for name in missing if name != "sonic"]
    if other:
        parts.append(f"waiting for streams: {', '.join(other)}")
    return "; ".join(parts)


def make_subscriber(
    ctx: zmq.Context,
    host: str,
    port: int,
    topics: list[str],
    conflate: bool,
) -> zmq.Socket:
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 100)
    sock.setsockopt(zmq.LINGER, 0)
    if conflate:
        sock.setsockopt(zmq.CONFLATE, 1)
    for topic in topics:
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.connect(f"tcp://{host}:{port}")
    return sock


# ── LCM 수신기 (G1 robot state, 옵션으로 camera) ───────────────────────────────
# --g1-source lcm 일 때만 사용. 별도 스레드에서 lcm.handle_timeout 루프를 돌며
# 최신 메시지 1개만 유지(zmq.CONFLATE=1과 동일한 효과).
class LcmReceiver:
    def __init__(self, lcm_url: str, state_channel: str, camera_channel: str,
                 want_image: bool):
        if lcm_module is None:
            raise RuntimeError(
                "lcm-python package not found. Install with `pip install lcm` "
                "to use --g1-source lcm."
            )
        self.want_image = want_image
        self._lc = lcm_module.LCM(lcm_url)
        self._lock = threading.Lock()
        self._latest_state: Optional[LatestMessage] = None
        self._latest_image: Optional[LatestImage] = None
        self._state_count = 0
        self._frame_count = 0

        self._lc.subscribe(state_channel, self._on_state)
        if want_image:
            self._lc.subscribe(camera_channel, self._on_camera)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _recv_loop(self) -> None:
        while self._running:
            self._lc.handle_timeout(10)

    def _on_state(self, channel: str, data: bytes) -> None:
        parsed = unpack_lcm_robot_state(data)
        if parsed is None:
            return
        now_wall, now_mono = time.time(), time.monotonic()
        with self._lock:
            self._latest_state = LatestMessage(parsed, now_wall, now_mono)
            self._state_count += 1

    def _on_camera(self, channel: str, data: bytes) -> None:
        parsed = unpack_lcm_camera_frame(data)
        if parsed is None:
            return
        jpeg, w, h, ts = parsed
        now_wall, now_mono = time.time(), time.monotonic()
        with self._lock:
            self._latest_image = LatestImage(jpeg, w, h, ts, now_wall, now_mono)
            self._frame_count += 1

    def get_state(self) -> Optional[LatestMessage]:
        with self._lock:
            return self._latest_state

    def get_image(self) -> Optional[LatestImage]:
        with self._lock:
            return self._latest_image

    @property
    def state_count(self) -> int:
        return self._state_count

    @property
    def frame_count(self) -> int:
        return self._frame_count


def append_selected(history: dict[str, list[np.ndarray]], frame: dict[str, np.ndarray]) -> None:
    selected_keys = [
        "record_wall_time",
        "record_monotonic_time",
        "pose_age_sec",
        "state_age_sec",
        "sonic_age_sec",
        "image_age_sec",
        "manager_stream_mode",
        "pose_frame_index",
        "pose_smpl_pose",
        "pose_smpl_joints",
        "pose_body_quat_w",
        "pose_joint_pos",
        "pose_vr_position",
        "pose_vr_orientation",
        "pose_vr_position_raw",
        "pose_vr_orientation_raw",
        "pose_left_trigger",
        "pose_right_trigger",
        "pose_left_grip",
        "pose_right_grip",
        "pose_left_hand_joints",
        "pose_right_hand_joints",
        "pico_target_frame_index",
        "pico_target_timestamp_monotonic",
        "pico_target_vr_position",
        "pico_target_vr_orientation",
        "pico_target_vr_position_raw",
        "pico_target_vr_orientation_raw",
        "pico_target_calibrated",
        "pico_target_ee_frame_offset_m",
        "pico_target_pose_source",
        "pico_target_ctrl_pos_hy",
        "pico_target_ctrl_quat_hy",
        "pico_target_ctrl_pos_aligned",
        "pico_target_ctrl_quat_aligned",
        "pico_target_ctrl_align_rev",
        "pico_target_left_trigger",
        "pico_target_right_trigger",
        "pico_target_left_grip",
        "pico_target_right_grip",
        "pico_target_left_hand_joints",
        "pico_target_right_hand_joints",
        "manager_state",
        "path_plan_source",
        "path_plan_control_mode",
        "path_plan_feasible",
        "path_plan_failure_reason",
        "path_plan_plan_id",
        "path_plan_timestamp_us",
        "path_plan_t_start_us",
        "path_plan_control_dt_sec",
        "path_plan_duration_sec",
        "path_plan_final_hold_sec",
        "path_plan_total_duration_sec",
        "path_plan_target_link",
        "path_plan_target_position",
        "path_plan_requested_target_position",
        "path_plan_target_clamped",
        "path_plan_target_clamp_reasons",
        "path_plan_requested_duration_sec",
        "path_plan_duration_clamped",
        "path_plan_planning_frames",
        "path_plan_total_frames",
        "path_plan_position_error_mm",
        "path_plan_rotation_error",
        "path_plan_solve_time_ms",
        "path_plan_trajectory_source",
        "state_last_action",
        "state_body_q",
        "state_body_q_measured",
        "state_body_q_target",
        "state_body_dq",
        "state_body_dq_measured",
        "state_base_quat",
        "state_base_quat_measured",
        "state_left_hand_q",
        "state_right_hand_q",
        "state_last_left_hand_action",
        "state_last_right_hand_action",
        "state_ee_qp_enabled",
        "state_ee_qp_valid",
        "state_ee_qp_error_before_m",
        "state_ee_qp_predicted_error_after_m",
        "state_ee_qp_nonlinear_error_after_m",
        "state_ee_qp_gmr_joint_error_before_rad",
        "state_ee_qp_predicted_gmr_joint_error_after_rad",
        "state_ee_qp_reference_joint_error_before_rad",
        "state_ee_qp_predicted_reference_joint_error_after_rad",
        "state_ee_qp_reference_source",
        "state_ee_qp_control_mode",
        "state_ee_qp_plan_id",
        "state_ee_qp_plan_phase",
        "state_ee_qp_reference_feasible",
        "state_ee_qp_terminal_target_active",
        "state_ee_qp_terminal_target_side",
        "state_ee_qp_terminal_target_distance_m",
        "state_ee_qp_terminal_capture_radius_m",
        "state_ee_qp_terminal_target_position",
        "state_ee_qp_arm_only_error_before_m",
        "state_ee_qp_waist_only_error_before_m",
        "state_ee_qp_lower_only_error_before_m",
        "state_ee_qp_upper_only_error_before_m",
        "state_ee_qp_solve_time_ms",
        "state_ee_qp_reference_age_ms",
        "state_ee_qp_correction_bound_saturation_fraction",
        "state_ee_qp_rate_bound_saturation_fraction",
        "state_ee_qp_correction_bound_saturation_mask",
        "state_ee_qp_rate_bound_saturation_mask",
        "state_ee_qp_backtracking_alpha",
        "state_ee_qp_fallback_reason",
        "state_ee_qp_fallback_active",
        "state_ee_qp_candidate_rejected",
        "state_ee_qp_joint_tracking_rmse_rad",
        "state_ee_qp_joint_tracking_max_abs_rad",
        "state_ee_qp_pd_tracking_lag_rmse_rad",
        "state_ee_qp_pd_tracking_lag_max_abs_rad",
        "state_ee_qp_q_corrected",
        "state_ee_qp_delta_q",
        "state_ee_qp_joint_tracking_error",
        "state_ee_qp_sonic_waist_raw_q_target",
        "state_ee_qp_final_waist_q_target",
        "state_ee_qp_target_position",
        "state_ee_qp_target_link",
        "state_ee_qp_left_wrist_error_before_m",
        "state_ee_qp_right_wrist_error_before_m",
        "state_ee_qp_left_wrist_error_after_m",
        "state_ee_qp_right_wrist_error_after_m",
        "sonic_last_action",
        "sonic_body_q_target",
        "sonic_body_q",
        "sonic_body_dq",
        "sonic_base_quat",
        "sonic_ee_qp_enabled",
        "sonic_ee_qp_valid",
        "sonic_ee_qp_error_before_m",
        "sonic_ee_qp_nonlinear_error_after_m",
        "sonic_ee_qp_reference_source",
        "sonic_ee_qp_control_mode",
        "sonic_ee_qp_plan_id",
        "sonic_ee_qp_plan_phase",
        "sonic_ee_qp_reference_feasible",
        "sonic_ee_qp_terminal_target_active",
        "sonic_ee_qp_terminal_target_side",
        "sonic_ee_qp_terminal_target_distance_m",
        "sonic_ee_qp_terminal_capture_radius_m",
        "sonic_ee_qp_terminal_target_position",
        "sonic_ee_qp_fallback_reason",
        "sonic_ee_qp_fallback_active",
        "sonic_ee_qp_candidate_rejected",
        "sonic_ee_qp_sonic_waist_raw_q_target",
        "sonic_ee_qp_final_waist_q_target",
        "image_path",
    ]
    for key in selected_keys:
        if key in frame:
            history.setdefault(key, []).append(np.asarray(frame[key]))


def stack_history(history: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    for key, values in history.items():
        try:
            stacked[key] = np.stack(values, axis=0)
        except ValueError:
            stacked[key] = np.asarray(values, dtype=object)
    return stacked


# ── 백그라운드 프레임 writer ───────────────────────────────────────────────────
# hot loop에서 압축/디스크 I/O를 완전히 걷어내기 위한 스레드. 메인 루프는 프레임을
# 조합해 큐에 넣기만 하고(수 μs), 이 스레드가 np.savez(무압축) + JPEG 저장 +
# session_selected용 history 누적을 담당한다. 큐가 가득 차면(디스크가 못 따라오면)
# 프레임을 drop 하고 경고만 내서 백프레셔가 수신 루프로 역류하지 않게 한다.
class FrameWriter:
    _STOP = object()

    def __init__(self, maxsize: int = 256):
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._history: dict[str, list[np.ndarray]] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._dropped = 0

    def start(self) -> None:
        self._thread.start()

    def submit(self, job: WriteJob) -> bool:
        """Non-blocking enqueue. Returns False (and drops) if the queue is full."""
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def _run(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job is FrameWriter._STOP:
                    break
                try:
                    if job.jpeg is not None and job.jpeg_path is not None:
                        with job.jpeg_path.open("wb") as f:
                            f.write(job.jpeg)
                    np.savez(job.npz_path, **job.frame)
                    append_selected(self._history, job.frame)
                except Exception as exc:  # 저장 실패해도 스레드는 계속 돈다
                    print(f"[recorder] writer error on {job.npz_path.name}: {exc}")
            finally:
                self._q.task_done()

    def close(self) -> None:
        """Flush the queue and join. Buffered frames are NOT lost."""
        self._q.put(FrameWriter._STOP)
        self._thread.join()

    @property
    def history(self) -> dict[str, list[np.ndarray]]:
        return self._history

    @property
    def dropped(self) -> int:
        return self._dropped

    def qsize(self) -> int:
        return self._q.qsize()


def write_manifest(path: Path, args: argparse.Namespace) -> None:
    state_source: dict[str, Any]
    if args.g1_source == "lcm":
        state_source = {
            "transport": "lcm",
            "lcm_url": args.lcm_url,
            "channel": args.lcm_state_channel,
        }
    else:
        state_source = {
            "transport": "zmq",
            "host": args.state_host,
            "port": args.state_port,
            "topic": args.state_topic,
        }

    manifest = {
        "created_wall_time": time.time(),
        "output_dir": str(path),
        "g1_source": args.g1_source,
        "pose_source": {
            "transport": "zmq",
            "host": args.pose_host,
            "port": args.pose_port,
            "topic": args.pose_topic,
        },
        "state_source": state_source,
        "sonic_debug_source": {
            "transport": "zmq",
            "host": args.state_host,
            "port": args.state_port,
            "topic": args.state_topic,
            "action_key": args.sonic_action_key,
        },
        "path_plan_source": None if args.disable_path_plan else {
            "transport": "zmq",
            "host": args.path_plan_host,
            "port": args.path_plan_port,
            "topic": args.path_plan_topic,
        },
        "button_source": {
            "transport": "zmq",
            "host": args.button_host,
            "port": args.button_port,
            "topic": args.button_topic,
        },
        "image_source": {
            "transport": "lcm",
            "lcm_url": args.lcm_url,
            "channel": args.lcm_camera_channel,
        } if args.record_image else None,
        "frequency_hz": args.frequency,
        "max_pose_age_sec": args.max_pose_age,
        "max_state_age_sec": args.max_state_age,
        "max_sonic_age_sec": args.max_sonic_age,
        "max_image_age_sec": args.max_image_age,
        "record_image": args.record_image,
        "allow_missing_pose": args.allow_missing_pose,
        "allow_missing_sonic": args.allow_missing_sonic,
        "notes": {
            "pose_*": "PICO manager human/SMPL input sent to SONIC (always ZMQ).",
            "state_last_action": "Compatibility copy from the ZMQ measured-state "
                                  "path. For source-independent training data use "
                                  "sonic_last_action.",
            "state_body_q": "Measured G1 joint state (ZMQ g1_debug path), MuJoCo "
                             "order, with default offsets.",
            "state_body_q_measured": "Measured G1 joint state (LCM G1_ROBOT_STATE "
                                      "path), hardware order, raw (no default "
                                      "offset subtracted).",
            "state_body_q_target": "Reference/visualisation target from the "
                                    "current motion frame, not the direct policy "
                                    "command (ZMQ path only).",
            "sonic_*": "Raw SONIC g1_debug fields. sonic_last_action is stored "
                        "without joint reordering or offset conversion.",
            "path_plan_*": "Latest PATH cuRobo plan heartbeat metadata, if the "
                            "planner server is running.",
            "pico_button_*": "Latest dedicated Pico ABXY event and read/publish timestamps.",
            "button_*_timestamp_us": "Clean end-to-end button timing aliases: Pico "
                                     "read, bridge receive/forward, and SONIC receive/apply.",
            "sonic_button_*": "Raw g1_debug copies of the same SONIC timing fields.",
            "image_path": "Relative path to saved JPEG frame under frames/ "
                           "(only if --record-image; camera transport is always LCM).",
        },
    }
    with (path / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record PICO input + SONIC PD target + G1 measured state "
                    "(+ optional ego-view camera)."
    )
    parser.add_argument("--output-dir", default="exp_output_01")
    parser.add_argument("--frequency", type=float, default=50.0)

    # PICO input — 항상 ZMQ
    parser.add_argument("--pose-host", default="localhost")
    parser.add_argument("--pose-port", type=int, default=DEFAULT_POSE_PORT)
    parser.add_argument("--pose-topic", default="pose")
    parser.add_argument("--max-pose-age", type=float, default=0.25)
    parser.add_argument("--pico-target-host", default="localhost")
    parser.add_argument("--pico-target-port", type=int, default=5566)
    parser.add_argument("--pico-target-topic", default="vr_pose_debug")
    parser.add_argument("--button-host", default="localhost")
    parser.add_argument("--button-port", type=int, default=5565)
    parser.add_argument("--button-topic", default="pico_button_event")

    # G1 measured state — zmq(g1_debug) 또는 lcm(G1_ROBOT_STATE) 선택
    parser.add_argument("--g1-source", choices=["zmq", "lcm"], default="zmq",
                         help="G1 측정 state를 어디서 읽을지: "
                              "'zmq' = SONIC deploy의 g1_debug 소켓 (MuJoCo sim "
                              "또는 SONIC이 중계하는 real G1, 기본값), "
                              "'lcm' = G1_ROBOT_STATE LCM 채널 (real G1 bridge "
                              "직접 구독, SONIC 우회).")
    parser.add_argument("--state-host", default="localhost")
    parser.add_argument("--state-port", type=int, default=DEFAULT_STATE_PORT)
    parser.add_argument("--state-topic", default="g1_debug")
    parser.add_argument("--max-state-age", type=float, default=0.25)
    parser.add_argument("--max-sonic-age", type=float, default=0.25)
    parser.add_argument(
        "--sonic-action-key",
        default="last_action",
        help="Required key in the g1_debug payload before a frame is saved.",
    )
    parser.add_argument(
        "--allow-missing-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record state/image frames even when the PICO pose ZMQ stream is absent. "
             "Enabled by default; use --no-allow-missing-pose for legacy strict sync.",
    )
    parser.add_argument(
        "--allow-missing-sonic",
        action="store_true",
        help="Record state/image frames even when SONIC g1_debug ZMQ is absent. "
             "A zero 29-DoF sonic_last_action placeholder is saved.",
    )

    parser.add_argument("--path-plan-host", default="localhost")
    parser.add_argument("--path-plan-port", type=int, default=5562)
    parser.add_argument("--path-plan-topic", default="curobo_plan")
    parser.add_argument(
        "--disable-path-plan",
        action="store_true",
        help="Do not subscribe to PATH cuRobo plan metadata.",
    )

    # LCM 공통 (state=lcm 또는 image 사용 시)
    parser.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    parser.add_argument("--lcm-state-channel", default=DEFAULT_LCM_STATE_CHANNEL)
    parser.add_argument("--lcm-camera-channel", default=DEFAULT_LCM_CAMERA_CHANNEL)

    # ego-view 이미지 — 항상 LCM (G1_CAMERA_FRAME)
    parser.add_argument("--record-image", action="store_true",
                         help="G1_CAMERA_FRAME(LCM)을 구독해 ego-view JPEG도 "
                              "함께 저장. real G1 bridge가 --publish-camera로 "
                              "실행 중이어야 한다.")
    parser.add_argument("--max-image-age", type=float, default=0.5)
    parser.add_argument("--require-image", action="store_true",
                         help="--record-image와 함께 사용. image가 없거나 너무 "
                              "오래되면 해당 프레임 자체를 skip.")

    parser.add_argument(
        "--record-only-streaming",
        action="store_true",
        help="Only save frames when manager_state stream_mode is POSE-like (1 or 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.record_image and lcm_module is None:
        raise SystemExit(
            "--record-image requires the LCM python bindings. Install with "
            "`pip install lcm`."
        )
    if args.g1_source == "lcm" and lcm_module is None:
        raise SystemExit(
            "--g1-source lcm requires the LCM python bindings. Install with "
            "`pip install lcm`."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}\n"
            f"Use --overwrite or choose a new --output-dir."
        )

    image_dir = output_dir / "frames"
    if args.record_image:
        image_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(output_dir, args)

    ctx = zmq.Context()

    # PICO pose는 항상 ZMQ
    pose_sock = make_subscriber(
        ctx, args.pose_host, args.pose_port,
        [args.pose_topic, "manager_state"], conflate=False,
    )
    pico_target_sock = make_subscriber(
        ctx,
        args.pico_target_host,
        args.pico_target_port,
        [args.pico_target_topic],
        conflate=True,
    )

    # SONIC debug/action은 measured-state source와 무관하게 항상 ZMQ로 수신한다.
    state_sock = make_subscriber(
        ctx, args.state_host, args.state_port,
        [args.state_topic], conflate=True,
    )

    path_plan_sock: zmq.Socket | None = None
    if not args.disable_path_plan:
        path_plan_sock = make_subscriber(
            ctx,
            args.path_plan_host,
            args.path_plan_port,
            [args.path_plan_topic],
            conflate=True,
        )

    button_sock = make_subscriber(
        ctx,
        args.button_host,
        args.button_port,
        [args.button_topic],
        conflate=False,
    )

    # G1 measured state: zmq debug payload 또는 LCM direct state
    lcm_receiver: Optional[LcmReceiver] = None
    if args.g1_source == "lcm":
        lcm_receiver = LcmReceiver(
            args.lcm_url, args.lcm_state_channel, args.lcm_camera_channel,
            want_image=args.record_image,
        )
        lcm_receiver.start()

    # --record-image인데 g1_source=zmq인 경우, 카메라만을 위한 별도 LCM 수신기 필요
    image_only_lcm: Optional[LcmReceiver] = None
    if args.record_image and args.g1_source == "zmq":
        image_only_lcm = LcmReceiver(
            args.lcm_url, args.lcm_state_channel, args.lcm_camera_channel,
            want_image=True,
        )
        image_only_lcm.start()

    poller = zmq.Poller()
    poller.register(pose_sock, zmq.POLLIN)
    poller.register(pico_target_sock, zmq.POLLIN)
    poller.register(state_sock, zmq.POLLIN)
    if path_plan_sock is not None:
        poller.register(path_plan_sock, zmq.POLLIN)
    poller.register(button_sock, zmq.POLLIN)

    latest_pose: LatestMessage | None = None
    latest_pico_target: LatestMessage | None = None
    latest_state: LatestMessage | None = None
    latest_sonic: LatestMessage | None = None
    latest_manager: LatestMessage | None = None
    latest_path_plan: LatestMessage | None = None
    latest_button: LatestMessage | None = None
    writer = FrameWriter()
    writer.start()
    frame_idx = 0
    next_sample = time.monotonic()
    last_status = 0.0

    print(f"[recorder] Writing to {output_dir}")
    print(f"[recorder] pose tcp://{args.pose_host}:{args.pose_port}/{args.pose_topic}")
    print(
        f"[recorder] PICO target tcp://{args.pico_target_host}:"
        f"{args.pico_target_port}/{args.pico_target_topic}"
    )
    if args.g1_source == "zmq":
        print(f"[recorder] g1 state (zmq) tcp://{args.state_host}:{args.state_port}/"
              f"{args.state_topic}")
    else:
        print(f"[recorder] g1 state (lcm) {args.lcm_url} / {args.lcm_state_channel}")
    print(f"[recorder] sonic debug/action (zmq) tcp://{args.state_host}:"
          f"{args.state_port}/{args.state_topic} key={args.sonic_action_key}")
    if path_plan_sock is not None:
        print(
            f"[recorder] path plan (zmq) tcp://{args.path_plan_host}:"
            f"{args.path_plan_port}/{args.path_plan_topic}"
        )
    print(
        f"[recorder] Pico buttons (zmq) tcp://{args.button_host}:"
        f"{args.button_port}/{args.button_topic}"
    )
    if args.record_image:
        print(f"[recorder] image (lcm) {args.lcm_url} / {args.lcm_camera_channel}")
    print("[recorder] Ctrl+C to stop and write session_selected.npz")

    try:
        while True:
            events = dict(poller.poll(timeout=5))
            now_wall = time.time()
            now_mono = time.monotonic()

            if pose_sock in events:
                for _ in range(20):
                    try:
                        raw = pose_sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        if raw.startswith(args.pose_topic.encode("utf-8")):
                            latest_pose = LatestMessage(
                                unpack_pose_message(raw, args.pose_topic),
                                now_wall, now_mono,
                            )
                        elif raw.startswith(b"manager_state"):
                            latest_manager = LatestMessage(
                                unpack_pose_message(raw, "manager_state"),
                                now_wall, now_mono,
                            )
                    except Exception as exc:
                        print(f"[recorder] skipped malformed pose-side message: {exc}")

            if pico_target_sock in events:
                try:
                    raw = pico_target_sock.recv(zmq.NOBLOCK)
                    latest_pico_target = LatestMessage(
                        unpack_pose_message(raw, args.pico_target_topic),
                        now_wall,
                        now_mono,
                    )
                except zmq.Again:
                    pass
                except Exception as exc:
                    print(f"[recorder] skipped malformed PICO target: {exc}")

            if state_sock in events:
                try:
                    raw = state_sock.recv(zmq.NOBLOCK)
                    debug_message = LatestMessage(
                        unpack_msgpack_topic(raw, args.state_topic),
                        now_wall, now_mono,
                    )
                    latest_sonic = debug_message
                    if args.g1_source == "zmq":
                        latest_state = debug_message
                except zmq.Again:
                    pass
                except Exception as exc:
                    print(f"[recorder] skipped malformed state message: {exc}")

            if lcm_receiver is not None:
                lcm_msg = lcm_receiver.get_state()
                if lcm_msg is not None:
                    latest_state = lcm_msg

            if path_plan_sock is not None and path_plan_sock in events:
                try:
                    raw = path_plan_sock.recv(zmq.NOBLOCK)
                    latest_path_plan = LatestMessage(
                        unpack_msgpack_topic(raw, args.path_plan_topic),
                        now_wall,
                        now_mono,
                    )
                except zmq.Again:
                    pass
                except Exception as exc:
                    print(f"[recorder] skipped malformed PATH plan message: {exc}")

            if button_sock in events:
                for _ in range(32):
                    try:
                        raw = button_sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        latest_button = LatestMessage(
                            unpack_msgpack_topic(raw, args.button_topic),
                            now_wall,
                            now_mono,
                        )
                    except Exception as exc:
                        print(f"[recorder] skipped malformed button event: {exc}")

            if now_mono < next_sample:
                continue
            next_sample += 1.0 / args.frequency

            pose_for_frame = latest_pose
            if args.allow_missing_pose and (
                pose_for_frame is None
                or now_mono - pose_for_frame.monotonic_time > args.max_pose_age
            ):
                pose_for_frame = LatestMessage({}, now_wall, now_mono)
            if args.allow_missing_sonic and latest_sonic is None:
                latest_sonic = LatestMessage(
                    {args.sonic_action_key: np.zeros(29, dtype=np.float64)},
                    now_wall,
                    now_mono,
                )

            missing = missing_required_streams(
                pose=pose_for_frame, state=latest_state, sonic=latest_sonic,
            )
            if missing:
                if now_mono - last_status > 1.0:
                    sonic_endpoint = (
                        f"tcp://{args.state_host}:{args.state_port}/{args.state_topic}"
                    )
                    print(
                        "[recorder] "
                        + format_missing_streams_message(
                            missing, sonic_endpoint=sonic_endpoint,
                        )
                    )
                    last_status = now_mono
                continue

            assert pose_for_frame is not None
            assert latest_state is not None
            assert latest_sonic is not None
            if args.sonic_action_key not in latest_sonic.data:
                if now_mono - last_status > 1.0:
                    keys = ", ".join(sorted(latest_sonic.data.keys()))
                    print(
                        f"[recorder] waiting for SONIC action key "
                        f"{args.sonic_action_key!r}; available keys: {keys}"
                    )
                    last_status = now_mono
                continue

            pose_age = now_mono - pose_for_frame.monotonic_time
            state_age = now_mono - latest_state.monotonic_time
            sonic_age = now_mono - latest_sonic.monotonic_time
            if (pose_age > args.max_pose_age or state_age > args.max_state_age
                    or sonic_age > args.max_sonic_age):
                if now_mono - last_status > 1.0:
                    print(
                        f"[recorder] stale message: pose_age={pose_age:.3f}s, "
                        f"state_age={state_age:.3f}s, sonic_age={sonic_age:.3f}s"
                    )
                    last_status = now_mono
                continue

            stream_mode = np.array([-1], dtype=np.int32)
            if latest_manager is not None and "stream_mode" in latest_manager.data:
                stream_mode = to_numpy(latest_manager.data["stream_mode"]).astype(np.int32).reshape(-1)
            if args.record_only_streaming and int(stream_mode[0]) not in (1, 4):
                continue

            # image: g1_source에 따라 lcm_receiver 또는 image_only_lcm에서 가져옴
            latest_image: Optional[LatestImage] = None
            image_age: Optional[float] = None
            if args.record_image:
                img_source = lcm_receiver if lcm_receiver is not None else image_only_lcm
                latest_image = img_source.get_image() if img_source is not None else None
                if latest_image is not None:
                    image_age = now_mono - latest_image.monotonic_time
                if args.require_image and (
                    latest_image is None or image_age > args.max_image_age
                ):
                    if now_mono - last_status > 1.0:
                        ia = f"{image_age:.3f}s" if image_age is not None else "N/A"
                        print(f"[recorder] stale/missing image (required): image_age={ia}")
                        last_status = now_mono
                    continue

            frame = compose_frame(
                now_wall=now_wall,
                now_mono=now_mono,
                pose_age=pose_age,
                state_age=state_age,
                sonic_age=sonic_age,
                stream_mode=stream_mode,
                pose_data=pose_for_frame.data,
                state_data=latest_state.data,
                sonic_data=latest_sonic.data,
                manager_data=latest_manager.data if latest_manager is not None else None,
                path_plan_data=latest_path_plan.data if latest_path_plan is not None else None,
                button_data=latest_button.data if latest_button is not None else None,
                pico_target_data=(
                    latest_pico_target.data
                    if latest_pico_target is not None
                    else None
                ),
            )

            jpeg_bytes: Optional[bytes] = None
            jpeg_path: Optional[Path] = None
            if args.record_image:
                image_path_rel = ""
                if latest_image is not None and (
                    image_age is None or image_age <= args.max_image_age
                ):
                    image_path_rel = f"frames/frame_{frame_idx:06d}.jpg"
                    jpeg_bytes = latest_image.jpeg
                    jpeg_path = output_dir / image_path_rel
                    frame["image_age_sec"] = np.array([image_age], dtype=np.float64)
                    frame["image_width"] = np.array([latest_image.width], dtype=np.int32)
                    frame["image_height"] = np.array([latest_image.height], dtype=np.int32)
                    frame["image_device_ts_us"] = np.array(
                        [latest_image.device_ts_us], dtype=np.int64
                    )
                else:
                    frame["image_age_sec"] = np.array([-1.0], dtype=np.float64)
                frame["image_path"] = np.asarray(image_path_rel)

            out_path = output_dir / f"frame_{frame_idx:06d}.npz"
            # 실제 저장(np.savez 무압축 + JPEG write + history 누적)은 writer 스레드로
            # 넘긴다. hot loop은 여기서 막히지 않는다.
            if not writer.submit(WriteJob(out_path, frame, jpeg_bytes, jpeg_path)):
                if now_mono - last_status > 1.0:
                    print(
                        f"[recorder] WARNING: writer queue full, dropped frame "
                        f"{frame_idx} (total dropped={writer.dropped}, "
                        f"qsize={writer.qsize()}) — disk can't keep up"
                    )
                    last_status = now_mono

            frame_idx += 1
            if frame_idx == 1 or frame_idx % int(max(args.frequency, 1.0)) == 0:
                msg = (
                    f"[recorder] saved {frame_idx} frames "
                    f"(pose_age={pose_age * 1000:.1f}ms, "
                    f"state_age={state_age * 1000:.1f}ms, "
                    f"sonic_age={sonic_age * 1000:.1f}ms"
                )
                if args.record_image and image_age is not None:
                    msg += f", image_age={image_age * 1000:.1f}ms"
                msg += ")"
                print(msg)

    except KeyboardInterrupt:
        print("\n[recorder] stopping...")
    finally:
        # 큐에 남은 프레임까지 모두 디스크에 flush 후 join (버퍼 손실 없음).
        pending = writer.qsize()
        if pending:
            print(f"[recorder] flushing {pending} buffered frame(s)...")
        writer.close()
        history = writer.history
        if history:
            selected_path = output_dir / "session_selected.npz"
            np.savez_compressed(selected_path, **stack_history(history))
            print(f"[recorder] wrote {selected_path}")
        if writer.dropped:
            print(f"[recorder] WARNING: dropped {writer.dropped} frame(s) "
                  f"due to full writer queue")
        print(f"[recorder] total frames: {frame_idx}")
        pose_sock.close()
        pico_target_sock.close()
        state_sock.close()
        if path_plan_sock is not None:
            path_plan_sock.close()
        button_sock.close()
        if lcm_receiver is not None:
            lcm_receiver.stop()
        if image_only_lcm is not None:
            image_only_lcm.stop()
        ctx.term()


if __name__ == "__main__":
    main()
