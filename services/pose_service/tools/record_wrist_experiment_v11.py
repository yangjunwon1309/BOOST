#!/usr/bin/env python3
"""Record PATH wrist, Revo2 hand, local head and G1 wrist camera data.

The recorder runs on Thor, receives PATH/G1 LCM and BrainCo DDS telemetry, and
stores each trial as a directory containing a 50 Hz CSV plus lossless TIM2 JPEGs.
The head camera may use a separate local Teleimager server from the G1 wrists.

Example:
  python services/pose_service/tools/record_wrist_experiment_v11.py \
    --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
    --output records/path_wrist_experiment.csv

Use --event-toggle to start and stop successive numbered trials when PATH's
terminal s key emits a recording edge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from boost.control.hand_mppi.contact_state import HandContactStateEstimator

try:
    import lcm
except ModuleNotFoundError:  # Keep --help usable outside the deployment image.
    lcm = None

# The recorder may be invoked by absolute path or from a deployment shell
# whose current directory is not on sys.path. It needs the pose service package
# and repository-local BOOST modules.
_SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT_PATH.parents[1]))
_BOOST_ROOT = _SCRIPT_PATH.parents[3]
sys.path.insert(0, str(_BOOST_ROOT))
sys.path.insert(0, str(_BOOST_ROOT / "external" / "g1_arm_teleop"))
sys.path.insert(0, str(_BOOST_ROOT / "external" / "teleimager" / "src"))


SCHEMA_VERSION = 3
SIDES = ("left", "right")
SPEED_MAX_GAP_SEC = 0.25
CAMERAS = {
    "head_camera": "head",
    "left_wrist_camera": "left_wrist",
    "right_wrist_camera": "right_wrist",
}
# G1 direct-state ordering: 12 lower-body joints, then waist, then both arms.
# Keep this recorder intentionally limited to the upper body; lower-body joints
# are neither teleoperation decision variables nor paper metrics for this study.
UPPER_BODY_START = 12
UPPER_BODY_COUNT = 17
UPPER_BODY_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
CSV_FIELDS = (
    "schema_version",
    "recorded_timestamp_us",
    "elapsed_s",
    "state",
    "left_ipopt_target_json",
    "right_ipopt_target_json",
    "left_ee_json",
    "right_ee_json",
    "left_hand_json",
    "right_hand_json",
    "upper_body_torque_json",
    "left_ee_distance_m",
    "right_ee_distance_m",
    "ee_distance_total_m",
    "head_seq",
    "head_capture_us",
    "head_capture_monotonic_ns",
    "left_wrist_seq",
    "left_wrist_capture_us",
    "left_wrist_capture_monotonic_ns",
    "right_wrist_seq",
    "right_wrist_capture_us",
    "right_wrist_capture_monotonic_ns",
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    result = [_number(item) for item in value]
    return None if any(item is None for item in result) else result


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _rotation_error_deg(target: list[float] | None, measured: list[float] | None) -> float | None:
    if target is None or measured is None:
        return None
    try:
        target_matrix = np.asarray(target, dtype=np.float64).reshape(3, 3)
        measured_matrix = np.asarray(measured, dtype=np.float64).reshape(3, 3)
        cosine = (float(np.trace(target_matrix @ measured_matrix.T)) - 1.0) / 2.0
    except (TypeError, ValueError):
        return None
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _next_indexed_output(template: Path) -> Path:
    """Allocate the next ``prefix_NNN/`` trial directory without overwriting."""
    match = re.fullmatch(r"(?P<prefix>.*?)(?:_(?P<index>\d+))?", template.stem)
    if match is None or not match.group("prefix"):
        raise ValueError("--output must have a non-empty CSV filename stem")
    prefix = match.group("prefix")
    requested_index = match.group("index")
    width = max(3, len(requested_index or ""))
    highest_index = int(requested_index) - 1 if requested_index else 0
    candidate_pattern = re.compile(rf"{re.escape(prefix)}_(\d+)$")
    if template.parent.exists():
        for candidate in template.parent.iterdir():
            candidate_stem = (
                candidate.name
                if candidate.is_dir()
                else candidate.stem if candidate.is_file() and candidate.suffix.lower() == ".csv" else ""
            )
            found = candidate_pattern.fullmatch(candidate_stem)
            if found is not None:
                highest_index = max(highest_index, int(found.group(1)))
                width = max(width, len(found.group(1)))
    return template.parent / f"{prefix}_{highest_index + 1:0{width}d}"


@dataclass
class UpperBodyTorqueCache:
    """Latest measured/estimated G1 upper-body torque, shared across trials."""

    max_age_sec: float
    sequence: int | None = None
    state_timestamp_us: int | None = None
    tau_est_nm: list[float] | None = None
    received_monotonic: float | None = None

    def update(self, state: Any) -> None:
        tau_est = np.asarray(state.tau_est, dtype=np.float64).reshape(-1)
        upper_body = tau_est[UPPER_BODY_START:UPPER_BODY_START + UPPER_BODY_COUNT]
        if upper_body.size != UPPER_BODY_COUNT or not np.isfinite(upper_body).all():
            raise ValueError("G1 robot-state tau_est does not contain 17 finite upper-body values")
        self.sequence = int(state.sequence)
        self.state_timestamp_us = int(state.timestamp_us)
        self.tau_est_nm = [float(value) for value in upper_body]
        self.received_monotonic = time.monotonic()

    def record(self, now: float) -> dict[str, Any]:
        age_ms = (
            None
            if self.received_monotonic is None
            else 1000.0 * (now - self.received_monotonic)
        )
        available = (
            self.tau_est_nm is not None
            and age_ms is not None
            and age_ms <= 1000.0 * self.max_age_sec
        )
        return {
            # This is the tau_est field received from G1, not a PD-gain
            # reconstruction.  Its name intentionally preserves that fact.
            "source": "g1_robot_state.tau_est",
            "available": available,
            "reason": None if available else ("unavailable" if age_ms is None else "stale"),
            "age_ms": age_ms,
            "sequence": self.sequence if available else None,
            "state_timestamp_us": self.state_timestamp_us if available else None,
            "joint_names": list(UPPER_BODY_JOINT_NAMES),
            "tau_est_nm": self.tau_est_nm if available else None,
        }


class CameraImageWriter:
    """Bounded asynchronous JPEG sink; the control/LCM thread only enqueues."""

    INDEX_FIELDS = (
        "camera", "sequence", "capture_wall_time_us", "capture_monotonic_time_ns",
        "receive_monotonic_ns", "width", "height", "file",
    )

    def __init__(self, trial_dir: Path, *, queue_size: int = 256) -> None:
        self.trial_dir = trial_dir
        self.images_dir = trial_dir / "images"
        for name in CAMERAS.values():
            (self.images_dir / name).mkdir(parents=True, exist_ok=True)
        self.index_file = (trial_dir / "images_index.csv").open(
            "x", newline="", encoding="utf-8"
        )
        self.index_writer = csv.DictWriter(self.index_file, fieldnames=self.INDEX_FIELDS)
        self.index_writer.writeheader()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._lock = threading.Lock()
        self._closed = False
        self._latest: dict[str, dict[str, int] | None] = {
            name: None for name in CAMERAS.values()
        }
        self._received = {name: 0 for name in CAMERAS.values()}
        self._saved = {name: 0 for name in CAMERAS.values()}
        self._queue_dropped = {name: 0 for name in CAMERAS.values()}
        self._write_errors = {name: 0 for name in CAMERAS.values()}
        self._sequence_gaps = {name: 0 for name in CAMERAS.values()}
        self._last_sequence: dict[str, int | None] = {name: None for name in CAMERAS.values()}
        self._saved_receive_ns: dict[str, list[int]] = {name: [] for name in CAMERAS.values()}
        self._writer = threading.Thread(target=self._run, name="path-image-writer", daemon=True)
        self._writer.start()

    def on_frame(self, camera_name: str, frame: Any) -> None:
        label = CAMERAS.get(camera_name)
        if label is None:
            return
        metadata = {
            "sequence": int(frame.sequence),
            "capture_wall_time_us": int(frame.capture_wall_time_us),
            "capture_monotonic_time_ns": int(frame.capture_monotonic_time_ns),
            "receive_monotonic_ns": int(frame.receive_monotonic_ns),
            "width": int(frame.width),
            "height": int(frame.height),
        }
        with self._lock:
            if self._closed:
                return
            previous = self._last_sequence[label]
            if previous is not None and metadata["sequence"] > previous + 1:
                self._sequence_gaps[label] += metadata["sequence"] - previous - 1
            self._last_sequence[label] = metadata["sequence"]
            self._latest[label] = metadata
            self._received[label] += 1
        item = (label, metadata, bytes(frame.jpg))
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            discarded = self._queue.get_nowait()
            if discarded is not None:
                with self._lock:
                    self._queue_dropped[discarded[0]] += 1
            self._queue.task_done()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._queue_dropped[label] += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            label, metadata, jpeg = item
            relative = Path("images") / label / f"{metadata['sequence']}.jpg"
            try:
                (self.trial_dir / relative).write_bytes(jpeg)
                self.index_writer.writerow(
                    {
                        "camera": label,
                        **metadata,
                        "file": relative.as_posix(),
                    }
                )
                with self._lock:
                    self._saved[label] += 1
                    self._saved_receive_ns[label].append(metadata["receive_monotonic_ns"])
            except Exception as error:
                with self._lock:
                    self._write_errors[label] += 1
                print(
                    f"[experiment-recorder] image write failed for {label}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                self._queue.task_done()

    def latest_columns(self) -> dict[str, int | None]:
        with self._lock:
            latest = {name: value for name, value in self._latest.items()}
        result: dict[str, int | None] = {}
        for label, prefix in (("head", "head"), ("left_wrist", "left_wrist"), ("right_wrist", "right_wrist")):
            frame = latest[label]
            result[f"{prefix}_seq"] = None if frame is None else frame["sequence"]
            result[f"{prefix}_capture_us"] = None if frame is None else frame["capture_wall_time_us"]
            result[f"{prefix}_capture_monotonic_ns"] = (
                None if frame is None else frame["capture_monotonic_time_ns"]
            )
        return result

    def metrics(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            received = dict(self._received)
            saved = dict(self._saved)
            queue_dropped = dict(self._queue_dropped)
            write_errors = dict(self._write_errors)
            gaps = dict(self._sequence_gaps)
            times = {name: list(value) for name, value in self._saved_receive_ns.items()}
        result = {}
        for camera in CAMERAS.values():
            stamps = times[camera]
            elapsed = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
            result[camera] = {
                "received_frames": received[camera],
                "saved_frames": saved[camera],
                "dropped_frames": queue_dropped[camera] + write_errors[camera] + gaps[camera],
                "queue_dropped_frames": queue_dropped[camera],
                "write_errors": write_errors[camera],
                "sequence_gaps": gaps[camera],
                "average_fps": (max(0, len(stamps) - 1) / elapsed) if elapsed > 0.0 else 0.0,
            }
        return result

    def close(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            already_closed = self._closed
            self._closed = True
        if already_closed:
            return self.metrics()
        self._queue.join()
        self._queue.put(None)
        self._writer.join()
        self.index_file.flush()
        self.index_file.close()
        return self.metrics()


class ExperimentRecorder:
    def __init__(
        self,
        trial_dir: Path,
        *,
        sample_hz: float,
        flush_every: int,
        upper_body_torque: UpperBodyTorqueCache,
        contact_config: dict[str, Any],
        teleimager_relay_config: Path,
        no_images: bool = False,
    ) -> None:
        trial_dir.parent.mkdir(parents=True, exist_ok=True)
        trial_dir.mkdir(parents=False, exist_ok=False)
        self.trial_dir = trial_dir
        self.output = trial_dir / "data.csv"
        self.summary_path = trial_dir / "summary.json"
        self.file = self.output.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.sample_period_sec = 1.0 / max(1.0, sample_hz)
        self.flush_every = max(1, flush_every)
        self.upper_body_torque = upper_body_torque
        self.last_record_monotonic: float | None = None
        self.started_monotonic = time.monotonic()
        self.last_ee: dict[str, tuple[list[float], float]] = {}
        self.distance_m = {side: 0.0 for side in SIDES}
        self.samples = 0
        self.hand_contact = {
            side: HandContactStateEstimator(
                tau_on=float(contact_config.get("command_error_on", 0.08)),
                tau_off=float(contact_config.get("command_error_off", 0.04)),
                minimum_duration_s=float(contact_config.get("minimum_duration_sec", 0.10)),
                stale_after_s=float(contact_config.get("stale_after_sec", 0.25)),
                finger_channels=contact_config.get("finger_channels"),
            )
            for side in SIDES
        }
        self._hand_lock = threading.Lock()
        self.image_writer: CameraImageWriter | None = None
        self.image_client: Any | None = None
        self.image_endpoints: dict[str, tuple[str, int]] = {}
        self.image_warning_deadline = time.monotonic() + 3.0
        self.image_warning_printed = False
        if not no_images:
            self._start_image_input(teleimager_relay_config)

    def _start_image_input(self, relay_path: Path) -> None:
        self.image_writer = CameraImageWriter(self.trial_dir)
        try:
            import yaml
            from teleop.utils.recording_image_client import RecordingImageClient

            relay_path = relay_path if relay_path.is_absolute() else _BOOST_ROOT / relay_path
            relay = yaml.safe_load(relay_path.read_text(encoding="utf-8"))
            g1_host = str(os.environ.get("G1_TELEIMAGER_HOST", relay["g1_host"]))
            camera_path = Path(os.environ.get(
                "BOOST_TELEIMAGER_G1_CONFIG", relay["teleimager_config"]
            ))
            if not camera_path.is_absolute():
                camera_path = _BOOST_ROOT / camera_path
            camera_config = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
            camera_hosts = {name: g1_host for name in CAMERAS}

            head_config_value = os.environ.get(
                "BOOST_HEAD_TELEIMAGER_CONFIG",
                relay.get("head_teleimager_config"),
            )
            head_config_path = None
            if head_config_value:
                head_config_path = Path(head_config_value)
                if not head_config_path.is_absolute():
                    head_config_path = _BOOST_ROOT / head_config_path
                head_server_config = yaml.safe_load(
                    head_config_path.read_text(encoding="utf-8")
                )
                head_camera = head_server_config.get("head_camera")
                if not isinstance(head_camera, dict):
                    raise ValueError(
                        f"head Teleimager config has no head_camera section: {head_config_path}"
                    )
                camera_config["head_camera"] = head_camera
                camera_hosts["head_camera"] = str(
                    os.environ.get("BOOST_HEAD_TELEIMAGER_HOST")
                    or relay.get("head_host", g1_host)
                )

            enabled = tuple(
                name for name, value in camera_config.items()
                if name in CAMERAS and isinstance(value, dict) and value.get("enable_recording_zmq", False)
            )
            missing = sorted(set(CAMERAS) - set(enabled))
            if missing:
                print(
                    f"[experiment-recorder] warning: no TIM2 stream configured for {', '.join(missing)}",
                    flush=True,
                )
            self.image_endpoints = {
                name: (camera_hosts[name], int(camera_config[name]["recording_zmq_port"]))
                for name in enabled
            }
            self.image_client = RecordingImageClient(
                camera_hosts,
                camera_config,
                decode_cameras=(),
                cameras=enabled,
                frame_callback=self.image_writer.on_frame,
                subscriber_hwm=256,
            )
            print(
                "[experiment-recorder] TIM2 JPEG capture subscriptions: "
                + ", ".join(
                    f"{name}={host}:{port}"
                    for name, (host, port) in self.image_endpoints.items()
                )
                + f"; G1 config={camera_path}"
                + (f"; head config={head_config_path}" if head_config_path else ""),
                flush=True,
            )
        except Exception as error:
            print(
                f"[experiment-recorder] warning: camera input unavailable; CSV and hand recording continue ({error})",
                file=sys.stderr,
                flush=True,
            )
            self.image_client = None

    def update_hand_command(self, side: str, q_cmd: list[float], received: float) -> None:
        with self._hand_lock:
            self.hand_contact[side].update_command(q_cmd, received)

    def update_hand_state(
        self, side: str, q_meas: list[float], dq_meas: list[float], tau_est: list[float], received: float
    ) -> None:
        with self._hand_lock:
            self.hand_contact[side].update_state(q_meas, dq_meas, tau_est, received)

    def _ee_record(self, hand: dict[str, Any], side: str, now: float) -> dict[str, Any]:
        position = _vector(hand.get("measured_wrist_position_m"), 3)
        rotation = _vector(hand.get("measured_wrist_rotation"), 9)
        speed_mps: float | None = None
        sample_interval_s: float | None = None
        distance_increment_m = 0.0
        previous = self.last_ee.get(side)
        if position is not None and previous is not None:
            previous_position, previous_time = previous
            sample_interval_s = now - previous_time
            distance_increment_m = math.dist(position, previous_position)
            self.distance_m[side] += distance_increment_m
            if 0.0 < sample_interval_s <= SPEED_MAX_GAP_SEC:
                speed_mps = distance_increment_m / sample_interval_s
        if position is not None:
            self.last_ee[side] = (position, now)
        target_position = _vector(hand.get("ipopt_wrist_target_m"), 3)
        tracking_error_m = (
            None if position is None or target_position is None else math.dist(position, target_position)
        )
        return {
            "position_m": position,
            "orientation": rotation,
            "speed_mps": speed_mps,
            "sample_interval_s": sample_interval_s,
            "distance_increment_m": distance_increment_m,
            "tracking_position_error_m": tracking_error_m,
            "tracking_orientation_error_deg": _rotation_error_deg(
                _vector(hand.get("ipopt_wrist_rotation"), 9), rotation
            ),
        }

    @staticmethod
    def _ipopt_record(hand: dict[str, Any]) -> dict[str, Any]:
        return {
            "position_m": _vector(hand.get("ipopt_wrist_target_m"), 3),
            "orientation": _vector(hand.get("ipopt_wrist_rotation"), 9),
        }

    def _warn_if_camera_streams_missing(self, now: float) -> None:
        if (
            self.image_writer is None
            or self.image_warning_printed
            or now < self.image_warning_deadline
        ):
            return
        metrics = self.image_writer.metrics()
        missing = [name for name, stats in metrics.items() if stats["received_frames"] == 0]
        if missing:
            health = self.image_client.health() if self.image_client is not None else {}
            details = []
            for name in missing:
                config_name = {
                    "head": "head_camera",
                    "left_wrist": "left_wrist_camera",
                    "right_wrist": "right_wrist_camera",
                }[name]
                endpoint = self.image_endpoints.get(config_name)
                endpoint_text = f"{endpoint[0]}:{endpoint[1]}" if endpoint else "not configured"
                stream_health = health.get(config_name, {})
                error = stream_health.get("error") or stream_health.get("last_packet_error")
                details.append(
                    f"{name}@{endpoint_text}"
                    + (f" ({error})" if error else " (waiting for frames)")
                )
            print(
                "[experiment-recorder] warning: no TIM2 frames received for "
                + ", ".join(details)
                + "; continuing to wait while CSV recording continues",
                flush=True,
            )
        self.image_warning_printed = True

    def on_ipopt(self, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        self._warn_if_camera_streams_missing(now)
        if self.last_record_monotonic is not None and now - self.last_record_monotonic < self.sample_period_sec:
            return
        self.last_record_monotonic = now
        hands = payload.get("hands") if isinstance(payload.get("hands"), dict) else {}
        elapsed_s = now - self.started_monotonic
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "recorded_timestamp_us": time.time_ns() // 1_000,
            "elapsed_s": elapsed_s,
            "state": payload.get("state"),
        }
        for side in SIDES:
            ipopt_hand = hands.get(side, {})
            ee = self._ee_record(ipopt_hand, side, now)
            with self._hand_lock:
                hand_json = self.hand_contact[side].snapshot(now)
            row[f"{side}_ipopt_target_json"] = _json(self._ipopt_record(ipopt_hand))
            row[f"{side}_ee_json"] = _json(ee)
            row[f"{side}_hand_json"] = _json(hand_json)
            row[f"{side}_ee_distance_m"] = self.distance_m[side]
        row["upper_body_torque_json"] = _json(self.upper_body_torque.record(now))
        row["ee_distance_total_m"] = sum(self.distance_m.values())
        if self.image_writer is not None:
            row.update(self.image_writer.latest_columns())
        else:
            row.update({field: None for field in CSV_FIELDS if field.endswith(("_seq", "_capture_us", "_capture_monotonic_ns"))})
        self.writer.writerow(row)
        self.samples += 1
        if self.samples % self.flush_every == 0:
            self.file.flush()

    def close(self) -> dict[str, Any]:
        camera_summary = {}
        if self.image_client is not None:
            self.image_client.close()
        if self.image_writer is not None:
            camera_summary = self.image_writer.close()
        self.file.flush()
        self.file.close()
        now = time.monotonic()
        summary = {
            "schema_version": SCHEMA_VERSION,
            "trial_dir": str(self.trial_dir),
            "csv": str(self.output),
            "summary_path": str(self.summary_path),
            "samples": self.samples,
            "elapsed_s": now - self.started_monotonic,
            "ee_distance_left_m": self.distance_m["left"],
            "ee_distance_right_m": self.distance_m["right"],
            "ee_distance_total_m": sum(self.distance_m.values()),
            "hand_contact": {
                side: self.hand_contact[side].summary(now) for side in SIDES
            },
            "cameras": camera_summary,
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument(
        "--g1-lcm-url",
        default=os.environ.get("G1_LCM_URL", "udpm://239.255.76.67:7667?ttl=1"),
        help="G1 direct-LCM URL used for G1_ROBOT_STATE tau_est.",
    )
    parser.add_argument("--robot-state-channel", default="G1_ROBOT_STATE")
    parser.add_argument("--robot-state-max-age-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV path template; each trial is saved in a numbered directory.")
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Deprecated and rejected: trial directories are never overwritten.")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument(
        "--event-toggle",
        action="store_true",
        help="Start/stop numbered trial directories from PATH terminal-s events.",
    )
    parser.add_argument(
        "--record-control-channel",
        default="PATH_V11_WRIST_EXPERIMENT_RECORD_CONTROL",
    )
    parser.add_argument(
        "--brainco-dds-interface",
        default=os.environ.get("BRAINCO_DDS_INTERFACE", ""),
        help="DDS NIC for Revo2 command/state telemetry; defaults to BRAINCO_DDS_INTERFACE.",
    )
    parser.add_argument(
        "--hand-config",
        type=Path,
        default=_BOOST_ROOT / "configs" / "hands" / "brainco_revo2.yaml",
    )
    parser.add_argument(
        "--teleimager-relay-config",
        type=Path,
        default=_BOOST_ROOT / "configs" / "teleop" / "teleimager_relay.yaml",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable TIM2 JPEG recording but continue the control/hand CSV.",
    )
    return parser.parse_args()


class BrainCoDDSObserver:
    """Read-only subscriptions to Revo2 command and measured-state topics."""

    def __init__(self, interface: str, on_command, on_state) -> None:
        if not interface:
            raise RuntimeError("BRAINCO_DDS_INTERFACE is empty")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from boost.control.hand_mppi.revo2 import BRAINCO_COMMAND_TOPICS, BRAINCO_STATE_TOPICS

        ChannelFactoryInitialize(0, interface)
        self.subscribers = []
        for side in SIDES:
            command_sub = ChannelSubscriber(BRAINCO_COMMAND_TOPICS[side], MotorCmds_)
            command_sub.Init(self._command_callback(side, on_command), 1)
            state_sub = ChannelSubscriber(BRAINCO_STATE_TOPICS[side], MotorStates_)
            state_sub.Init(self._state_callback(side, on_state), 1)
            self.subscribers.extend((command_sub, state_sub))

    @staticmethod
    def _command_callback(side: str, callback):
        def receive(message) -> None:
            try:
                q = [float(item.q) for item in message.cmds]
                if len(q) == 6 and np.isfinite(q).all():
                    callback(side, q, time.monotonic())
            except (AttributeError, TypeError, ValueError):
                return
        return receive

    @staticmethod
    def _state_callback(side: str, callback):
        def receive(message) -> None:
            try:
                states = message.states
                q = [float(item.q) for item in states]
                dq = [float(item.dq) for item in states]
                tau = [float(item.tau_est) for item in states]
                if len(q) == len(dq) == len(tau) == 6 and np.isfinite(q + dq + tau).all():
                    callback(side, q, dq, tau, time.monotonic())
            except (AttributeError, TypeError, ValueError):
                return
        return receive


def main() -> int:
    args = parse_args()
    if lcm is None:
        raise SystemExit("python-lcm is required; use the thor-teleop Pixi environment.")
    if args.overwrite:
        raise SystemExit("--overwrite is not supported; trial directories are never overwritten")
    if args.event_toggle and float(args.duration_sec) > 0.0:
        raise SystemExit("--duration-sec cannot be used with --event-toggle; press s again to stop")
    if float(args.robot_state_max_age_ms) <= 0.0:
        raise SystemExit("--robot-state-max-age-ms must be positive")

    try:
        import yaml
        from boost.io.path_transport.g1_direct_protocol import unpack_robot_state
        from remote_offload.wrist_d405_protocol_v11 import (
            WRIST_EXPERIMENT_RECORD_CONTROL_CHANNEL,
            WRIST_IPOPT_VISUAL_CHANNEL,
            unpack_experiment_record_control,
            unpack_ipopt_visual,
        )
        hand_config = yaml.safe_load(args.hand_config.read_text(encoding="utf-8"))
        contact_config = hand_config.get("contact_detection", {})
    except Exception as error:
        raise SystemExit(f"recorder dependency/configuration error: {error}") from error

    def stop_on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    upper_body_torque = UpperBodyTorqueCache(
        max_age_sec=float(args.robot_state_max_age_ms) / 1000.0,
    )
    bus = lcm.LCM(args.wifi_lcm_url)
    g1_bus = bus if args.g1_lcm_url == args.wifi_lcm_url else lcm.LCM(args.g1_lcm_url)
    recorder: ExperimentRecorder | None = None
    recorder_lock = threading.Lock()
    completed_trials = 0
    last_control_key: tuple[int, int] | None = None

    def print_summary(summary: dict[str, Any]) -> None:
        print(
            "[experiment-recorder] "
            f"samples={summary['samples']} elapsed={summary['elapsed_s']:.2f}s "
            f"ee_distance_total={summary['ee_distance_total_m']:.3f}m "
            f"summary={summary['summary_path']}",
            flush=True,
        )
        for camera, stats in summary.get("cameras", {}).items():
            print(
                f"[experiment-recorder] camera={camera} fps={stats['average_fps']:.1f} "
                f"saved={stats['saved_frames']} dropped={stats['dropped_frames']}",
                flush=True,
            )

    def close_trial() -> None:
        nonlocal recorder, completed_trials
        with recorder_lock:
            current = recorder
            recorder = None
        if current is None:
            return
        summary = current.close()
        completed_trials += 1
        print_summary(summary)

    def start_trial() -> Path:
        nonlocal recorder
        trial_dir = _next_indexed_output(args.output)
        current = ExperimentRecorder(
            trial_dir,
            sample_hz=float(args.sample_hz),
            flush_every=int(args.flush_every),
            upper_body_torque=upper_body_torque,
            contact_config=contact_config,
            teleimager_relay_config=args.teleimager_relay_config,
            no_images=bool(args.no_images),
        )
        with recorder_lock:
            recorder = current
        print(f"[experiment-recorder] recording started: {trial_dir}", flush=True)
        return trial_dir

    def on_ipopt(_channel: str, data: bytes) -> None:
        try:
            payload = unpack_ipopt_visual(data)
            with recorder_lock:
                if recorder is not None:
                    recorder.on_ipopt(payload)
        except Exception as error:
            print(f"[experiment-recorder] IPOPT packet rejected: {error}", file=sys.stderr)

    def on_robot_state(_channel: str, data: bytes) -> None:
        try:
            upper_body_torque.update(unpack_robot_state(data))
        except Exception as error:
            print(f"[experiment-recorder] robot-state packet rejected: {error}", file=sys.stderr)

    def on_hand_command(side: str, q_cmd: list[float], received: float) -> None:
        with recorder_lock:
            if recorder is not None:
                recorder.update_hand_command(side, q_cmd, received)

    def on_hand_state(
        side: str, q: list[float], dq: list[float], tau: list[float], received: float
    ) -> None:
        with recorder_lock:
            if recorder is not None:
                recorder.update_hand_state(side, q, dq, tau, received)

    def on_record_control(_channel: str, data: bytes) -> None:
        nonlocal last_control_key
        try:
            control = unpack_experiment_record_control(data)
            control_key = (
                int(control["sequence"]), int(control["published_timestamp_us"])
            )
            if control_key == last_control_key:
                return
            last_control_key = control_key
            with recorder_lock:
                active = recorder is not None
            if active:
                close_trial()
                print("[experiment-recorder] terminal s: stopped; waiting for next trial", flush=True)
            else:
                trial_dir = start_trial()
                print(f"[experiment-recorder] terminal s: started {trial_dir}", flush=True)
        except Exception as error:
            print(f"[experiment-recorder] record-control packet rejected: {error}", file=sys.stderr)

    bus.subscribe(WRIST_IPOPT_VISUAL_CHANNEL, on_ipopt)
    g1_bus.subscribe(args.robot_state_channel, on_robot_state)
    observer = None
    try:
        observer = BrainCoDDSObserver(
            str(args.brainco_dds_interface), on_hand_command, on_hand_state
        )
        print(
            "[experiment-recorder] BrainCo DDS read-only telemetry subscribed on "
            f"{args.brainco_dds_interface}: rt/brainco/{{left,right}}/{{cmd,state}}",
            flush=True,
        )
    except Exception as error:
        print(
            f"[experiment-recorder] warning: BrainCo hand telemetry unavailable; "
            f"hand CSV values will be null ({error})",
            file=sys.stderr,
            flush=True,
        )

    if args.event_toggle:
        if args.record_control_channel != WRIST_EXPERIMENT_RECORD_CONTROL_CHANNEL:
            print(
                "[experiment-recorder] warning: non-default record-control channel; "
                "the state machine must use the same channel",
                flush=True,
            )
        bus.subscribe(args.record_control_channel, on_record_control)
        print(
            "[experiment-recorder] waiting for PATH terminal s; "
            f"next output={_next_indexed_output(args.output)}",
            flush=True,
        )
    else:
        start_trial()

    print(
        "[experiment-recorder] upper-body tau_est listening "
        f"channel={args.robot_state_channel} url={args.g1_lcm_url} "
        f"max_age={float(args.robot_state_max_age_ms):.1f}ms",
        flush=True,
    )
    deadline = (
        None if float(args.duration_sec) <= 0.0 else time.monotonic() + float(args.duration_sec)
    )
    try:
        while deadline is None or time.monotonic() < deadline:
            bus.handle_timeout(20)
            if g1_bus is not bus:
                for _ in range(32):
                    if g1_bus.handle_timeout(0) <= 0:
                        break
    except KeyboardInterrupt:
        pass
    except OSError as error:
        if "lcm_handle_timeout() returned -1" not in str(error):
            raise
    close_trial()
    return 0 if completed_trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
