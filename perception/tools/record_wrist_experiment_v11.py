#!/usr/bin/env python3
"""Record compact paper-ready wrist-belief and IPOPT experiment data to CSV.

The recorder runs on the laptop that receives the Wi-Fi LCM streams.  It makes
one row per IPOPT visual sample (50 Hz by default) and carries forward the most
recent belief packet.  Variable-size perception results are intentionally kept
in compact JSON columns so a trial stays one row per control sample rather than
expanding into debug/candidate rows.

Example:
  python perception/tools/record_wrist_experiment_v11.py \
    --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
    --output records/trial_001.csv

Use --x-toggle to start and stop successive numbered trials with an unchorded
controller X press/release.  A+X remains reserved for PATH tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lcm
except ModuleNotFoundError:  # Keep --help usable outside the deployment image.
    lcm = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SCHEMA_VERSION = 2
SIDES = ("left", "right")
SPEED_MAX_GAP_SEC = 0.25
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
    "selected_hand",
    "belief_result_sequence",
    "belief_age_ms",
    "left_objects_json",
    "right_objects_json",
    "left_orientation_candidates_json",
    "right_orientation_candidates_json",
    "left_belief_target_json",
    "right_belief_target_json",
    "left_ipopt_target_json",
    "right_ipopt_target_json",
    "left_ee_json",
    "right_ee_json",
    "upper_body_torque_json",
    "left_ee_distance_m",
    "right_ee_distance_m",
    "ee_distance_total_m",
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


def _compact_contact(candidate: Any, index: int, selected_index: Any) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    return {
        "index": index,
        "selected": index == selected_index,
        "posterior": _number(candidate.get("probability")),
        "position_m": _vector(candidate.get("center_goal_m"), 3),
        "orientation": {
            "approach_axis": _vector(candidate.get("approach_goal"), 3),
            "surface_normal": _vector(candidate.get("surface_normal"), 3),
            "surface_tangent": _vector(candidate.get("surface_tangent"), 3),
            "surface_axis": _vector(candidate.get("surface_axis"), 3),
        },
    }


def _compact_objects(hand: Any) -> list[dict[str, Any]]:
    if not isinstance(hand, dict):
        return []
    target = hand.get("target") if isinstance(hand.get("target"), dict) else {}
    target_object_id = str(target.get("object_id", ""))
    result: list[dict[str, Any]] = []
    for object_data in hand.get("objects", []):
        if not isinstance(object_data, dict):
            continue
        selected_index = object_data.get("selected_grip_index")
        contacts = [
            compact
            for index, candidate in enumerate(object_data.get("grip_candidates", []))
            if (compact := _compact_contact(candidate, index, selected_index)) is not None
        ]
        object_id = str(object_data.get("object_id", ""))
        result.append(
            {
                "object_id": object_id,
                "label": object_data.get("label"),
                "posterior": _number(object_data.get("object_probability")),
                "selected": bool(object_id and object_id == target_object_id),
                "contacts": contacts,
            }
        )
    return result


def _compact_orientation_candidates(hand: Any) -> list[dict[str, Any]]:
    if not isinstance(hand, dict):
        return []
    orientation = hand.get("orientation") if isinstance(hand.get("orientation"), dict) else {}
    selected_index = orientation.get("selected_index")
    result: list[dict[str, Any]] = []
    for index, candidate in enumerate(orientation.get("candidates", [])):
        if not isinstance(candidate, dict):
            continue
        result.append(
            {
                "index": index,
                "selected": index == selected_index,
                "object_id": candidate.get("object_id"),
                "relation": candidate.get("relation"),
                "posterior": _number(candidate.get("probability")),
                "position_m": _vector(candidate.get("position_goal_m"), 3),
                "normal": _vector(candidate.get("normal"), 3),
            }
        )
    return result


def _compact_belief_target(hand: Any) -> dict[str, Any] | None:
    if not isinstance(hand, dict) or not isinstance(hand.get("target"), dict):
        return None
    target = hand["target"]
    return {
        "object_id": target.get("object_id"),
        "position_m": _vector(target.get("position_goal_m"), 3),
        "orientation_axis": _vector(target.get("orientation_axis_world"), 3),
        "orientation_frame": _vector(target.get("orientation_frame_world"), 9),
        "relation": target.get("orientation_relation"),
        "orientation_posterior": _number(target.get("orientation_confidence")),
        "object_posterior": _number(target.get("object_confidence")),
        "contact_posterior": _number(target.get("grip_confidence")),
        "joint_posterior": _number(target.get("joint_confidence")),
    }


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
    """Allocate the next ``prefix_NNN.csv`` without overwriting any trial."""
    match = re.fullmatch(r"(?P<prefix>.*?)(?:_(?P<index>\d+))?", template.stem)
    if match is None or not match.group("prefix"):
        raise ValueError("--output must have a non-empty CSV filename stem")
    prefix = match.group("prefix")
    requested_index = match.group("index")
    width = max(3, len(requested_index or ""))
    highest_index = int(requested_index) - 1 if requested_index else 0
    candidate_pattern = re.compile(rf"{re.escape(prefix)}_(\d+)$")
    for candidate in template.parent.glob("*.csv"):
        found = candidate_pattern.fullmatch(candidate.stem)
        if found is not None:
            highest_index = max(highest_index, int(found.group(1)))
            width = max(width, len(found.group(1)))
    return template.with_name(f"{prefix}_{highest_index + 1:0{width}d}.csv")


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


class ExperimentRecorder:
    def __init__(
        self,
        output: Path,
        *,
        sample_hz: float,
        flush_every: int,
        upper_body_torque: UpperBodyTorqueCache,
        overwrite: bool = False,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.summary_path = output.with_suffix(".summary.json")
        self.file = output.open("w" if overwrite else "x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.sample_period_sec = 1.0 / max(1.0, sample_hz)
        self.flush_every = max(1, flush_every)
        self.upper_body_torque = upper_body_torque
        self.latest_belief: dict[str, Any] | None = None
        self.latest_belief_received_monotonic: float | None = None
        self.last_record_monotonic: float | None = None
        # Trial time is defined by the controller record edge, not by the
        # first surviving IPOPT sample.
        self.started_monotonic: float | None = time.monotonic()
        self.last_ee: dict[str, tuple[list[float], float]] = {}
        self.distance_m = {side: 0.0 for side in SIDES}
        self.samples = 0

    def on_belief(self, payload: dict[str, Any]) -> None:
        self.latest_belief = payload
        self.latest_belief_received_monotonic = time.monotonic()

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
            "position_source": hand.get("position_source"),
            "belief_position_applied": bool(hand.get("belief_position_applied", False)),
            "position_blend_mode": hand.get("position_blend_mode"),
            "position_alpha_source": hand.get("position_alpha_source"),
            "position_belief_weight": _number(hand.get("position_belief_weight")),
            "position_smoothed_confidence": _number(hand.get("position_smoothed_confidence")),
            "orientation_source": hand.get("orientation_source"),
            "orientation_authority": _number(hand.get("orientation_authority_weight")),
            "reference_speed_mps": _number(hand.get("reference_speed_mps")),
            "grasp_commanded": bool(hand.get("grasp_commanded", False)),
            "grasp_engaged": bool(hand.get("grasp_engaged", False)),
            "grasp_state": hand.get("grasp_state"),
            "grasp": hand.get("grasp"),
            "frame_offset": hand.get("frame_offset"),
        }

    def on_ipopt(self, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        if self.last_record_monotonic is not None and now - self.last_record_monotonic < self.sample_period_sec:
            return
        self.last_record_monotonic = now
        hands = payload.get("hands") if isinstance(payload.get("hands"), dict) else {}
        belief_hands = (
            self.latest_belief.get("hands", {})
            if isinstance(self.latest_belief, dict) and isinstance(self.latest_belief.get("hands"), dict)
            else {}
        )
        elapsed_s = now - self.started_monotonic
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "recorded_timestamp_us": time.time_ns() // 1_000,
            "elapsed_s": elapsed_s,
            "state": payload.get("state"),
            "selected_hand": (
                self.latest_belief.get("selected_hand")
                if isinstance(self.latest_belief, dict)
                else None
            ),
            "belief_result_sequence": (
                self.latest_belief.get("result_sequence")
                if isinstance(self.latest_belief, dict)
                else None
            ),
            "belief_age_ms": (
                None
                if self.latest_belief_received_monotonic is None
                else 1000.0 * (now - self.latest_belief_received_monotonic)
            ),
        }
        for side in SIDES:
            belief_hand = belief_hands.get(side, {})
            ipopt_hand = hands.get(side, {})
            ee = self._ee_record(ipopt_hand, side, now)
            row[f"{side}_objects_json"] = _json(_compact_objects(belief_hand))
            row[f"{side}_orientation_candidates_json"] = _json(
                _compact_orientation_candidates(belief_hand)
            )
            row[f"{side}_belief_target_json"] = _json(
                _compact_belief_target(belief_hand)
            )
            row[f"{side}_ipopt_target_json"] = _json(self._ipopt_record(ipopt_hand))
            row[f"{side}_ee_json"] = _json(ee)
            row[f"{side}_ee_distance_m"] = self.distance_m[side]
        row["upper_body_torque_json"] = _json(self.upper_body_torque.record(now))
        row["ee_distance_total_m"] = sum(self.distance_m.values())
        self.writer.writerow(row)
        self.samples += 1
        if self.samples % self.flush_every == 0:
            self.file.flush()

    def close(self) -> dict[str, Any]:
        self.file.flush()
        self.file.close()
        elapsed_s = (
            0.0
            if self.started_monotonic is None
            else time.monotonic() - self.started_monotonic
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "csv": str(self.output),
            "samples": self.samples,
            "elapsed_s": elapsed_s,
            "ee_distance_left_m": self.distance_m["left"],
            "ee_distance_right_m": self.distance_m["right"],
            "ee_distance_total_m": sum(self.distance_m.values()),
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
        help=(
            "G1 direct-LCM URL used for G1_ROBOT_STATE tau_est. Defaults to "
            "$G1_LCM_URL or the standard G1 multicast URL."
        ),
    )
    parser.add_argument(
        "--robot-state-channel",
        default="G1_ROBOT_STATE",
        help="G1 direct-LCM robot-state channel carrying tau_est.",
    )
    parser.add_argument(
        "--robot-state-max-age-ms",
        type=float,
        default=100.0,
        help="Write torque as unavailable when its received age exceeds this value.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV with the same output path.",
    )
    parser.add_argument(
        "--duration-sec", type=float, default=0.0,
        help="Stop automatically after this duration; <=0 records until Ctrl-C.",
    )
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument(
        "--x-toggle",
        action="store_true",
        help=(
            "Wait for unchorded controller-X relay events: first press starts a "
            "new numbered CSV, next press closes it, then repeats."
        ),
    )
    parser.add_argument(
        "--record-control-channel",
        default="PATH_V11_WRIST_EXPERIMENT_RECORD_CONTROL",
        help="Wi-Fi LCM channel carrying state-machine-relayed controller-X edges.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if lcm is None:
        raise SystemExit("python-lcm is required; run in the laptop perception environment.")
    try:
        from path_core.messaging.g1_direct_protocol import unpack_robot_state
        from remote_offload.wrist_d405_protocol_v11 import (
            WRIST_BELIEF_TELEMETRY_CHANNEL,
            WRIST_EXPERIMENT_RECORD_CONTROL_CHANNEL,
            WRIST_IPOPT_VISUAL_CHANNEL,
            unpack_experiment_record_control,
            unpack_belief_telemetry,
            unpack_ipopt_visual,
        )
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error.name} is required in the recorder Python environment."
        ) from error

    if args.x_toggle and args.overwrite:
        raise SystemExit("--overwrite cannot be used with --x-toggle")
    if args.x_toggle and float(args.duration_sec) > 0.0:
        raise SystemExit("--duration-sec cannot be used with --x-toggle; press X again to stop")
    if not args.x_toggle and args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing trial CSV: {args.output} (pass --overwrite to replace it)"
        )

    if float(args.robot_state_max_age_ms) <= 0.0:
        raise SystemExit("--robot-state-max-age-ms must be positive")

    upper_body_torque = UpperBodyTorqueCache(
        max_age_sec=float(args.robot_state_max_age_ms) / 1000.0,
    )
    recorder: ExperimentRecorder | None = None
    if not args.x_toggle:
        recorder = ExperimentRecorder(
            args.output,
            sample_hz=float(args.sample_hz),
            flush_every=int(args.flush_every),
            upper_body_torque=upper_body_torque,
            overwrite=bool(args.overwrite),
        )
    bus = lcm.LCM(args.wifi_lcm_url)
    g1_bus = bus if args.g1_lcm_url == args.wifi_lcm_url else lcm.LCM(args.g1_lcm_url)
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

    def on_belief(_channel: str, data: bytes) -> None:
        try:
            if recorder is not None:
                recorder.on_belief(unpack_belief_telemetry(data))
        except Exception as error:
            print(f"[experiment-recorder] belief packet rejected: {error}", file=sys.stderr)

    def on_ipopt(_channel: str, data: bytes) -> None:
        try:
            if recorder is not None:
                recorder.on_ipopt(unpack_ipopt_visual(data))
        except Exception as error:
            print(f"[experiment-recorder] IPOPT packet rejected: {error}", file=sys.stderr)

    def on_robot_state(_channel: str, data: bytes) -> None:
        try:
            upper_body_torque.update(unpack_robot_state(data))
        except Exception as error:
            print(f"[experiment-recorder] robot-state packet rejected: {error}", file=sys.stderr)

    def on_record_control(_channel: str, data: bytes) -> None:
        nonlocal completed_trials, last_control_key, recorder
        try:
            control = unpack_experiment_record_control(data)
            control_key = (
                int(control["sequence"]), int(control["published_timestamp_us"])
            )
            if control_key == last_control_key:
                return
            last_control_key = control_key
            if recorder is not None:
                summary = recorder.close()
                completed_trials += 1
                print_summary({**summary, "summary_path": str(recorder.summary_path)})
                recorder = None
                print("[experiment-recorder] X toggle: stopped; waiting for next trial", flush=True)
                return
            output = _next_indexed_output(args.output)
            recorder = ExperimentRecorder(
                output,
                sample_hz=float(args.sample_hz),
                flush_every=int(args.flush_every),
                upper_body_torque=upper_body_torque,
            )
            print(f"[experiment-recorder] X toggle: started {output}", flush=True)
        except Exception as error:
            print(f"[experiment-recorder] record-control packet rejected: {error}", file=sys.stderr)

    bus.subscribe(WRIST_BELIEF_TELEMETRY_CHANNEL, on_belief)
    bus.subscribe(WRIST_IPOPT_VISUAL_CHANNEL, on_ipopt)
    g1_bus.subscribe(args.robot_state_channel, on_robot_state)
    if args.x_toggle:
        if args.record_control_channel != WRIST_EXPERIMENT_RECORD_CONTROL_CHANNEL:
            print(
                "[experiment-recorder] warning: non-default record-control channel; "
                "the state machine must use the same channel",
                flush=True,
            )
        bus.subscribe(args.record_control_channel, on_record_control)
        print(
            "[experiment-recorder] waiting for unchorded controller X; "
            f"next output={_next_indexed_output(args.output)}",
            flush=True,
        )
    else:
        print(f"[experiment-recorder] recording to {args.output}", flush=True)
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
            # Keep Wi-Fi telemetry responsive while draining a bounded number
            # of high-rate G1 state packets on the separate multicast group.
            bus.handle_timeout(20)
            if g1_bus is not bus:
                for _ in range(32):
                    if g1_bus.handle_timeout(0) <= 0:
                        break
    except KeyboardInterrupt:
        pass
    except OSError as error:
        # python-lcm returns -1 from handle_timeout when Ctrl-C interrupts
        # its poll syscall.  Treat that specific shutdown path exactly like
        # KeyboardInterrupt so an active trial is flushed and summarized.
        if "lcm_handle_timeout() returned -1" not in str(error):
            raise
    if recorder is not None:
        summary = recorder.close()
        completed_trials += 1
        print_summary({**summary, "summary_path": str(recorder.summary_path)})
    return 0 if completed_trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
