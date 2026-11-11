#!/usr/bin/env python3
"""PATH arm bridge between continuous Galaxy XR hand tracking and the G1.

The operator controls this session only from the PATH terminal: ``r`` starts
or resumes wrist following, ``s`` toggles the experiment recorder and parks
the arms on stop, and ``q`` saves any active trial and exits.

  XR hands       -> tcp://127.0.0.1:5566/vr_pose_debug
  PATH -> G1     -> LCM G1_PATH_UPPER_COMMAND
  G1 -> PATH     -> LCM G1_ROBOT_STATE
  Recording      -> LCM PATH_V11_WRIST_EXPERIMENT_RECORD_CONTROL
  Telemetry      -> tcp://*:5560/path_qp_reference
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
import numpy as np
import zmq
import yaml

from boost.io.path_transport.zmq_wire import unpack_pose_message

from boost.teleop.legacy.intention.action_blending import (
    LEFT_WRIST_LINK,
    RIGHT_WRIST_LINK,
    EETarget,
    SharedControlConfig,
    rotation_target_from_relation,
)
from boost.teleop.legacy.intention.reference_dynamics import (
    GraspTransitionDetector,
    OrientationAuthority,
    OrientationTargetGate,
    slerp_so3,
)
from boost.io.path_transport.g1_direct_protocol import (
    ARM_START,
    ROBOT_STATE_CHANNEL,
    UPPER_COMMAND_CHANNEL,
    pack_upper_command,
    unpack_robot_state,
)
from boost.io.path_transport.path_schema import (
    QP_REFERENCE_TOPIC,
    STATIC_EE_TARGET_TOPIC,
    ipopt_arm_sample_reference,
    now_us,
    qp_cartesian_target_reference,
    qp_none_reference,
    qp_sample_reference,
)
from boost.io.path_transport.zmq_bus import pack_topic, unpack_topic
from boost.control.arm.path_planning.unitree_arm_nonlinear_ik import UnitreeG1ArmNonlinearIK
from boost.teleop.path_core.references.damped_se3 import DampedSE3Reference, so3_log
from boost.teleop.path_core.references.xr_target_continuity import XRTargetContinuity
from services.pose_service.seg_web_node.robot_geometry import (
    HAND_AXIS_LOCAL,
    angular_velocity_from_rotations,
    hand_frame_profile,
)
from services.pose_service.remote_offload.wrist_d405_protocol_v11 import (
    WRIST_EXPERIMENT_RECORD_CONTROL_CHANNEL,
    WRIST_IPOPT_VISUAL_CHANNEL,
    pack_experiment_record_control,
    pack_ipopt_visual,
)

# ``auto_planning`` remains an accepted external compatibility value, but is
# canonicalized to teleop. Hand pinch can select an arm for optional assistance.
VALID_STATES = {"teleop", "intention", "auto_planning"}
SHARED_HAND_PUBLISH_HZ = 50.0

# Canonical task-space contract used by Galaxy XR wrists, shared-control
# targets, and the arm-only IPOPT FK. World poses are expressed relative to the
# headset and offset into the waist frame by xr_source_v15.py.
PICO_WAIST_FRAME = "pico_head_position_waist"
STATIC_TARGET_FRAME_ALIASES = {
    PICO_WAIST_FRAME,
    # Existing perception publishers already produce pelvis-identity robot
    # coordinates.  They are numerically equivalent to the canonical waist
    # frame and remain accepted while producers migrate their schema label.
    "pelvis_identity_fk",
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=sorted(VALID_STATES), default="teleop")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs" / "teleop" / "path_reference.yaml",
        help="YAML defaults for bounded wrist references and XR continuity.",
    )
    parser.add_argument("--damped-zeta", type=float, default=None)
    parser.add_argument("--damped-omega-n-rad-s", type=float, default=None)
    parser.add_argument("--damped-linear-speed-limit-m-s", type=float, default=None)
    parser.add_argument("--damped-angular-speed-limit-rad-s", type=float, default=None)
    parser.add_argument("--damped-minimum-duration-s", type=float, default=None)
    parser.add_argument("--damped-direct-position-error-m", type=float, default=None)
    parser.add_argument("--xr-outlier-max-linear-speed-m-s", type=float, default=None)
    parser.add_argument("--xr-outlier-max-angular-speed-rad-s", type=float, default=None)
    parser.add_argument("--xr-resync-after-s", type=float, default=None)
    parser.add_argument("--xr-lowpass-cutoff-hz", type=float, default=None)
    parser.add_argument("--pico-hand-timeout-sec", type=float, default=0.20)
    parser.add_argument("--pico-body-gap-hold-ms", type=float, default=200.0)
    parser.add_argument("--xr-future-tolerance-ms", type=float, default=100.0)
    parser.add_argument(
        "--auto-arm-xr-tracking",
        action="store_true",
        help=(
            "For simulation only, arm live wrist targets when the first fresh "
            "XR frame arrives instead of waiting for terminal r."
        ),
    )
    parser.add_argument(
        "--arm-reference-mode",
        choices=["head_position", "head_yaw"],
        default="head_position",
    )

    parser.add_argument(
        "--xr-hand-endpoint",
        default="tcp://127.0.0.1:5566",
        help="Independent absolute Galaxy XR waist-frame wrist stream.",
    )
    parser.add_argument("--static-target-bind", default="tcp://*:5564")
    parser.add_argument("--static-target-topic", default=STATIC_EE_TARGET_TOPIC)
    parser.add_argument("--static-target-timeout-sec", type=float, default=0.25)
    parser.add_argument(
        "--static-target-auto-enter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Automatically enter auto_planning when a new static EE target arrives. "
            "Disabled by default so perception gateways can buffer targets until "
            "the teleop session is started."
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
        "--wrist-record-control-channel",
        default=WRIST_EXPERIMENT_RECORD_CONTROL_CHANNEL,
        help="Wi-Fi LCM channel relaying terminal-s recording toggles.",
    )
    parser.add_argument(
        "--direct-upper-command-channel",
        default=UPPER_COMMAND_CHANNEL,
        help="Thor-to-G1 arm-only IPOPT command channel.",
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
            "the base is quasi-static. Uses base IMU fields from G1_ROBOT_STATE."
        ),
    )
    parser.add_argument("--gravity-bias-time-constant-sec", type=float, default=10.0)
    parser.add_argument("--gravity-bias-max-deg", type=float, default=3.0)
    parser.add_argument("--gravity-accel-norm-tolerance", type=float, default=0.50)
    parser.add_argument("--gravity-bias-gyro-gate-rad-s", type=float, default=0.05)
    parser.add_argument("--qp-bind-host", default="*")
    parser.add_argument("--qp-port", type=int, default=5560)
    parser.add_argument("--qp-topic", default=QP_REFERENCE_TOPIC)
    parser.add_argument("--state-command-bind", default="tcp://*:5563")
    parser.add_argument("--state-command-topic", default="path_state_command")
    parser.add_argument(
        "--shared-hand-selection-bind",
        default="tcp://*:5567",
        help="PUB endpoint that reports the pinch-selected shared-control hand.",
    )
    parser.add_argument(
        "--shared-hand-selection-topic",
        default="path_shared_control_hand",
        help="Topic carrying the pinch-selected shared-control hand heartbeat.",
    )

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
        help="Delay shared-control alpha after a hand pinch (0 injects immediately).",
    )
    parser.add_argument("--shared-stale-after-sec", type=float, default=0.3)
    parser.add_argument("--shared-d-max-m", type=float, default=0.9)
    parser.add_argument(
        "--enable-shared-target-assist",
        action="store_true",
        help=(
            "Allow a hand pinch to replace the selected Televuer arm target with a "
            "fresh perception/contact target. Disabled by default for direct XR teleop."
        ),
    )
    parser.add_argument(
        "--shared-position-mode",
        choices=("replace", "alpha"),
        default="replace",
        help=(
            "Belief-gated position policy: replace preserves the established "
            "belief-contact target; alpha linearly blends raw VR and belief targets."
        ),
    )
    parser.add_argument(
        "--shared-position-alpha",
        type=float,
        default=0.5,
        help="Belief weight in --shared-position-mode alpha, constrained to [0, 1].",
    )
    parser.add_argument(
        "--shared-position-alpha-source",
        choices=("fixed", "posterior"),
        default="fixed",
        help=(
            "fixed uses --shared-position-alpha; posterior maps the incoming "
            "b(g)*b(c|g) joint posterior through EMA, hysteresis, and rate limiting."
        ),
    )
    parser.add_argument("--shared-position-posterior-enter-confidence", type=float, default=0.45)
    parser.add_argument("--shared-position-posterior-exit-confidence", type=float, default=0.25)
    parser.add_argument("--shared-position-posterior-ema-alpha", type=float, default=0.15)
    parser.add_argument("--shared-position-posterior-max-rate-per-s", type=float, default=1.2)
    parser.add_argument(
        "--grasp-engagement-delay-sec",
        type=float,
        default=1.5,
        help=(
            "Continuous close hold used only as the manual grasp override "
            "when jaw-stall and co-motion evidence are unavailable."
        ),
    )
    parser.add_argument("--grasp-closing-timeout-sec", type=float, default=0.60)
    parser.add_argument(
        "--end-effector-model",
        choices=("brainco_revo2",),
        default="brainco_revo2",
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
        default=None,
        help="Explicit override for the Revo2 wrist-to-grasp-center X offset.",
    )
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
    parser.add_argument("--orientation-enter-confidence", type=float, default=0.45)
    parser.add_argument("--orientation-exit-confidence", type=float, default=0.25)
    parser.add_argument("--orientation-ema-alpha", type=float, default=0.15)
    parser.add_argument("--orientation-max-authority-rate-per-s", type=float, default=1.2)
    parser.add_argument("--orientation-gate-max-angle-rad", type=float, default=1.047)
    parser.add_argument("--orientation-gate-taper-start-fraction", type=float, default=0.667)
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
        default=Path(__file__).resolve().parents[3]
        / "assets"
        / "robot"
        / "g1_body29_hand14.urdf",
    )
    parser.add_argument(
        "--arm-ik-model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "assets" / "robot",
    )
    parser.add_argument("--arm-ik-max-iter", type=int, default=30)
    # The IPOPT frame is wrist_yaw_link origin. Televuer wrist targets are sent
    # directly to IPOPT; optional shared-control targets are converted here.
    parser.add_argument("--arm-ik-ee-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--arm-ik-orientation-full-error-m", type=float, default=0.01
    )
    parser.add_argument(
        "--arm-ik-orientation-zero-error-m", type=float, default=0.03
    )
    parser.add_argument("--arm-ik-orientation-min-scale", type=float, default=1.0)
    parser.add_argument(
        "--arm-ik-residual-warning-threshold-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--arm-ik-hard-lock-inactive-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set dynamic lower=upper bounds for the non-target arm.",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable terminal hotkeys r/s/q.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.reference_config.read_text(encoding="utf-8"))
        damped = config["damped_se3"]
        continuity = config["xr_continuity"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(f"cannot load reference config {args.reference_config}: {error}")
    defaults = {
        "damped_zeta": damped["zeta"],
        "damped_omega_n_rad_s": damped["omega_n_rad_s"],
        "damped_linear_speed_limit_m_s": damped["linear_speed_limit_m_s"],
        "damped_angular_speed_limit_rad_s": damped["angular_speed_limit_rad_s"],
        "damped_minimum_duration_s": damped["minimum_duration_s"],
        "damped_direct_position_error_m": damped["direct_position_error_m"],
        "xr_outlier_max_linear_speed_m_s": continuity["max_linear_speed_m_s"],
        "xr_outlier_max_angular_speed_rad_s": continuity["max_angular_speed_rad_s"],
        "xr_resync_after_s": continuity["resync_after_s"],
        "xr_lowpass_cutoff_hz": continuity["lowpass_cutoff_hz"],
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, float(default))
    return args


def log(message: str) -> None:
    print(f"[PATH-state] {message}", flush=True)


class TerminalKeyInput:
    """Non-blocking r/s/q terminal input for the PATH session."""

    START = "start"
    RECORD = "record"
    QUIT = "quit"
    KEY_TO_COMMAND = {"r": START, "s": RECORD, "q": QUIT}

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
        return self.KEY_TO_COMMAND.get(key.lower())


class DirectLcmRobotStateSubscriber:
    """Subscribe to the G1 bridge's authoritative lowstate stream."""

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


def parse_packed_fields(raw: bytes, topic: str) -> dict[str, np.ndarray]:
    return unpack_pose_message(raw, topic=topic)


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
    max_age_us: int,
    future_tolerance_us: int,
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
        source_timestamp_us: int | None = None
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
        if source_timestamp_us is None:
            return None
        source_age_us = int(fallback_timestamp_us) - source_timestamp_us
        max_age_us = max(1_000, int(max_age_us))
        future_tolerance_us = max(0, int(future_tolerance_us))
        if source_age_us > max_age_us or source_age_us < -future_tolerance_us:
            return None

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


def arm_reference_q29(q29: np.ndarray) -> np.ndarray:
    """Return a 29-joint reference with non-PATH groups locked at zero."""

    q = np.asarray(q29, dtype=np.float64).reshape(29).copy()
    q[:15] = 0.0
    return q


def hand_control_from_fields(fields: dict[str, np.ndarray]) -> dict[str, float]:
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
    return meta


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


def _vec3_field(payload: dict | None, *keys: str) -> np.ndarray | None:
    for key in keys:
        value = payload.get(key) if isinstance(payload, dict) else None
        if value is None:
            value = _nested(payload, "perception", key)
        if value is None:
            continue
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            continue
        if np.isfinite(vector).all():
            return vector
    return None


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
        center_camera_m=_vec3_field(payload, "center_camera_m"),
        center_goal_m=_vec3_field(payload, "center_goal_m"),
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


def main() -> int:
    args = parse_args()
    if not 0.0 <= float(args.shared_position_alpha) <= 1.0:
        raise ValueError("--shared-position-alpha must be within [0, 1]")
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
    ctx = zmq.Context.instance()

    def make_xr_sub(
        topic: str,
        *,
        conflate: bool,
        hwm: int = 8,
        endpoint: str,
    ) -> zmq.Socket:
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        if conflate:
            socket.setsockopt(zmq.CONFLATE, 1)
        else:
            socket.setsockopt(zmq.RCVHWM, int(hwm))
        socket.setsockopt(zmq.RCVTIMEO, 1)
        socket.connect(endpoint)
        return socket

    # Keep independent queues so pose traffic cannot delay the calibrated arm
    # target stream or controller inputs.
    xr_hand_stream = make_xr_sub(
        "vr_pose_debug", conflate=True, endpoint=args.xr_hand_endpoint
    )

    static_target_sub = ctx.socket(zmq.SUB)
    static_target_sub.setsockopt(zmq.SUBSCRIBE, args.static_target_topic.encode())
    static_target_sub.setsockopt(zmq.CONFLATE, 1)
    static_target_sub.setsockopt(zmq.RCVTIMEO, 1)
    static_target_sub.bind(args.static_target_bind)

    state_sub = ctx.socket(zmq.SUB)
    state_sub.setsockopt(zmq.SUBSCRIBE, args.state_command_topic.encode())
    state_sub.setsockopt(zmq.CONFLATE, 1)
    state_sub.setsockopt(zmq.RCVTIMEO, 1)
    state_sub.bind(args.state_command_bind)

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

    log(
        f"XR wrist target SUB {args.xr_hand_endpoint}/vr_pose_debug "
        "(legacy-compatible pose packet; hand landmarks are enabled)"
    )
    log("XR hands are tracked continuously; terminal r/s/q control the session")
    log("Galaxy XR wrist targets use manager-calibrated waist-frame SE(3)")
    log(f"static EE target SUB {args.static_target_bind}/{args.static_target_topic}")
    log(f"state command SUB {args.state_command_bind}/{args.state_command_topic}")
    log(
        "direct IPOPT arm PUB "
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
            residual_warning_threshold_m=(
                args.arm_ik_residual_warning_threshold_m
            ),
        )
        log(
            "Unitree-style arm nonlinear IK enabled: "
            f"urdf={args.arm_ik_urdf}, max_iter={args.arm_ik_max_iter}, "
            f"hard_lock_inactive={args.arm_ik_hard_lock_inactive_arm}, "
            "orientation_weight_scale=fixed_1.0, "
            f"residual_warn={100.0 * args.arm_ik_residual_warning_threshold_m:.1f}cm, "
            "q_output=direct (fresh IK result)"
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
    damped_reference = {
        side: DampedSE3Reference(
            zeta=args.damped_zeta,
            omega_n=args.damped_omega_n_rad_s,
            v_max=args.damped_linear_speed_limit_m_s,
            omega_max=args.damped_angular_speed_limit_rad_s,
        )
        for side in ("left", "right")
    }
    xr_continuity = XRTargetContinuity(
        max_linear_speed_mps=args.xr_outlier_max_linear_speed_m_s,
        max_angular_speed_rps=args.xr_outlier_max_angular_speed_rad_s,
        resync_after_sec=args.xr_resync_after_s,
        cutoff_hz=args.xr_lowpass_cutoff_hz,
    )
    log(
        "bounded SE(3) wrist reference enabled: "
        f"zeta={args.damped_zeta:.2f} omega_n={args.damped_omega_n_rad_s:.2f}rad/s "
        f"v_max={args.damped_linear_speed_limit_m_s:.3f}m/s "
        f"omega_max={args.damped_angular_speed_limit_rad_s:.3f}rad/s "
        f"direct_after={args.damped_minimum_duration_s:.2f}s/"
        f"{args.damped_direct_position_error_m:.3f}m; XR outlier="
        f"{args.xr_outlier_max_linear_speed_m_s:.1f}m/s/"
        f"{args.xr_outlier_max_angular_speed_rad_s:.1f}rad/s "
        f"resync={args.xr_resync_after_s:.2f}s LPF={args.xr_lowpass_cutoff_hz:.1f}Hz"
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
        log("initial auto_planning alias -> teleop (hand pinch may gate belief)")
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
    latest_wrist_targets: WaistWristTargets | None = None
    filtered_xr_targets: dict[str, np.ndarray | None] = {"left": None, "right": None}
    xr_reference_initialized = False
    damped_active = {"left": True, "right": True}
    resync_damping_started: dict[str, float | None] = {"left": None, "right": None}
    tracking_damping_started: float | None = None
    last_commanded_wrist_tf: dict[str, np.ndarray | None] = {
        "left": None,
        "right": None,
    }
    latest_pico_body_receive_us = 0
    latest_pico_session_id = -1
    pico_tracking_armed = False
    pico_arm_pending = bool(args.auto_arm_xr_tracking)
    last_forwarded_pose_frame = -1
    latest_hand_us = 0
    latest_hand_meta: dict[str, float] = {}
    recording_active = False
    record_control_sequence = 0
    quit_deadline: float | None = None
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
    last_plan_q: np.ndarray | None = None
    qp_reference_enabled = True
    last_tick = time.monotonic()
    last_log = 0.0
    period = 1.0 / max(args.fps, 1.0e-6)
    keyboard = TerminalKeyInput(enabled=not args.no_keyboard)
    shared_config = SharedControlConfig(
        stale_after_s=float(args.shared_stale_after_sec),
        d_max_m=float(args.shared_d_max_m),
        orientation_enter_confidence=float(args.orientation_enter_confidence),
        orientation_exit_confidence=float(args.orientation_exit_confidence),
        orientation_ema_alpha=float(args.orientation_ema_alpha),
        orientation_max_authority_rate_per_s=float(args.orientation_max_authority_rate_per_s),
        orientation_max_angle_rad=float(args.orientation_gate_max_angle_rad),
        orientation_gate_taper_start_fraction=float(args.orientation_gate_taper_start_fraction),
        grasp_closing_timeout_s=float(args.grasp_closing_timeout_sec),
        grasp_override_hold_s=float(args.grasp_engagement_delay_sec),
    )
    hand_axis_local = np.asarray(args.hand_axis_local, dtype=np.float64).reshape(3)
    grasp_center_local = np.asarray(args.grasp_center_local, dtype=np.float64).reshape(3)
    hand_profile = hand_frame_profile(args.end_effector_model)
    grasp_center_wrist_offset = {
        side: np.asarray(hand_profile["translation"][side], dtype=np.float64)
        + np.asarray(hand_profile["rotation"][side], dtype=np.float64)
        @ grasp_center_local
        for side in ("left", "right")
    }
    if args.grasp_center_wrist_x_offset_m is not None:
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
    # Position can use the same continuous authority construction as
    # orientation, but its input is the hierarchical joint posterior
    # b(object) * b(contact | object), carried in EETarget.confidence.
    position_authority = {
        side: OrientationAuthority(
            enter_confidence=float(args.shared_position_posterior_enter_confidence),
            exit_confidence=float(args.shared_position_posterior_exit_confidence),
            ema_alpha=float(args.shared_position_posterior_ema_alpha),
            max_authority_rate_per_s=float(args.shared_position_posterior_max_rate_per_s),
        )
        for side in ("left", "right")
    }
    grasp_detector = {
        side: GraspTransitionDetector(
            jaw_stall_gap_min_m=shared_config.grasp_jaw_stall_gap_min_m,
            jaw_stall_rate_max_mps=shared_config.grasp_jaw_stall_rate_max_mps,
            jaw_stall_ticks=shared_config.grasp_jaw_stall_ticks,
            comotion_window_s=shared_config.grasp_comotion_window_s,
            comotion_enter_log_ratio=shared_config.grasp_comotion_enter_log_ratio,
            comotion_exit_log_ratio=shared_config.grasp_comotion_exit_log_ratio,
            comotion_min_excitation_m=shared_config.grasp_comotion_min_excitation_m,
            closing_timeout_s=shared_config.grasp_closing_timeout_s,
        )
        for side in ("left", "right")
    }
    # Position and orientation authority have independent release edges. A
    # confirmed grasp releases the shared position target without suppressing
    # an actively held B/Y-gated orientation constraint.
    orientation_belief_active = {side: False for side in ("left", "right")}
    # Hand pinch is a dead-man switch for optional left/right belief blending.
    shared_selected_hand = "none"
    shared_hand_selection_revision = 0
    next_shared_hand_publish = 0.0
    # Per-hand raw Televuer command history used only for diagnostics.
    command_history = {side: None for side in ("left", "right")}
    log(
            "XR absolute waist-frame wrist targets enabled: "
        f"topic={args.xr_hand_endpoint}/vr_pose_debug "
        "(legacy-compatible pose packet), mode=head_position, "
        "anchor=none, gain=none"
        + (
            "; MuJoCo auto-arms on first fresh wrist frame"
            if args.auto_arm_xr_tracking
            else "; waiting for terminal r and Q_READY"
        )
    )
    if args.auto_arm_xr_tracking:
        log(
            "MuJoCo XR auto-arm enabled: live wrist targets arm on the first fresh "
            "Televuer frame; stale frames remain rejected"
        )
    log(
        "teleop Cartesian goal mode=direct Televuer wrist pose; no pose reference/filter "
        f"hand_model={args.end_effector_model} "
        f"axis={np.round(hand_axis_local, 4).tolist()} "
        f"grasp_local={np.round(grasp_center_local, 4).tolist()}"
    )
    log(
        "shared contact target assist="
        + (
            "enabled (hand pinch may select a fresh perception target)"
            if args.enable_shared_target_assist
            else "disabled (arm goals stay on Televuer)"
        )
    )

    def resync_shared_control_state(reason: str) -> None:
        """Reset shared-control authority after state/session discontinuities."""
        for side in ("left", "right"):
            orientation_gate[side].reset()
            orientation_authority[side].reset()
            position_authority[side].reset()
            grasp_detector[side].reset()
            orientation_belief_active[side] = False
        log(f"shared-control state reset ({reason})")

    def solve_authoritative_ipopt_transforms(
        left_target_tf: np.ndarray,
        right_target_tf: np.ndarray,
    ):
        """Solve authoritative arm joints from waist-frame Televuer targets."""

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
        )
        if result.success:
            result.q_ref[:15] = 0.0
            # DreamWaQ owns the lower body and waist. Only arm feed-forward
            # torque accompanies the authoritative IPOPT arm target.
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

    def apply_teleop_wrist_reference(
        targets: dict[str, np.ndarray], dt: float, now_s: float
    ) -> dict[str, np.ndarray]:
        """Apply startup/engagement/return damping and residual handoff."""
        nonlocal tracking_damping_started
        if not xr_reference_initialized:
            return {side: np.asarray(targets[side], dtype=np.float64).copy() for side in ("left", "right")}

        if not pico_tracking_armed:
            fixed_targets = {"left": q_ready_left_wrist_tf, "right": q_ready_right_wrist_tf}
            output = {
                side: damped_reference[side].advance(fixed_targets[side], dt)
                for side in ("left", "right")
            }
            for side in ("left", "right"):
                last_commanded_wrist_tf[side] = output[side].copy()
            return output

        errors = {
            side: float(
                np.linalg.norm(
                    np.asarray(targets[side], dtype=np.float64)[:3, 3]
                    - damped_reference[side].position
                )
            )
            for side in ("left", "right")
        }
        if (
            tracking_damping_started is not None
            and now_s - tracking_damping_started >= args.damped_minimum_duration_s
            and all(error < args.damped_direct_position_error_m for error in errors.values())
        ):
            for side in ("left", "right"):
                damped_reference[side].release_to_direct(targets[side])
                damped_active[side] = False
                resync_damping_started[side] = None
            log(
                "XR engagement damping released to direct tracking: "
                f"position errors L/R={errors['left'] * 100.0:.1f}/"
                f"{errors['right'] * 100.0:.1f}cm after "
                f"{now_s - tracking_damping_started:.2f}s; residual offsets decay at speed limits"
            )
            tracking_damping_started = None

        for side in ("left", "right"):
            started = resync_damping_started[side]
            if (
                started is not None
                and now_s - started >= args.damped_minimum_duration_s
                and errors[side] < args.damped_direct_position_error_m
            ):
                damped_reference[side].release_to_direct(targets[side])
                damped_active[side] = False
                resync_damping_started[side] = None
                log(
                    f"XR {side} outlier resync damping released: "
                    f"position error={errors[side] * 100.0:.1f}cm; residual offset decays"
                )

        output = {}
        for side in ("left", "right"):
            if damped_active[side]:
                output[side] = damped_reference[side].advance(targets[side], dt)
            else:
                output[side] = damped_reference[side].direct_target(targets[side], dt)
            last_commanded_wrist_tf[side] = output[side].copy()
        return output

    def ipopt_visual_hands_from_transforms(
        left_wrist_tf: np.ndarray,
        right_wrist_tf: np.ndarray,
        *,
        target_source: str,
    ) -> dict[str, dict]:
        """Encode the pelvis-frame wrist poses currently held by IPOPT.

        The viewer must retain an explicit target while hand tracking is waiting
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
        nonlocal latest_hand_us, latest_hand_meta
        try:
            fields = parse_packed_fields(raw, topic)
        except Exception:
            return
        meta = hand_control_from_fields(fields)
        if not meta:
            return
        latest_hand_meta.update(meta)
        latest_hand_us = int(rx_us)

    def pico_grip_active(side: str, tick_us: int) -> bool:
        """Return the latest valid binary pinch/grip request from hand tracking."""
        pose_age_us = tick_us - latest_hand_us
        timeout_us = max(0.0, args.pico_hand_timeout_sec) * 1.0e6
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

    def apply_state_request(requested: str, source: str, tick_us: int | None = None) -> None:
        nonlocal current_state
        nonlocal phase_mode, phase_sync_error_rad
        nonlocal shared_selected_hand, shared_hand_selection_revision
        if tick_us is None:
            tick_us = now_us()
        if requested == "auto_planning":
            log(f"{source} auto_planning alias -> teleop (hand pinch may gate belief)")
            requested = "teleop"
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
            resync_shared_control_state("state_change")
            if shared_selected_hand != "none":
                shared_selected_hand = "none"
                shared_hand_selection_revision += 1
            phase_mode = "none"
            phase_sync_error_rad = float("nan")

    def xr_targets_are_fresh(tick_us: int) -> bool:
        if latest_wrist_targets is None or latest_pico_body_receive_us <= 0:
            return False
        receive_age_us = int(tick_us) - latest_pico_body_receive_us
        source_age_us = int(tick_us) - latest_wrist_targets.source_timestamp_us
        max_age_us = max(1_000, int(args.pico_body_gap_hold_ms * 1_000.0))
        future_tolerance_us = max(0, int(args.xr_future_tolerance_ms * 1_000.0))
        return (
            receive_age_us <= max_age_us
            and -future_tolerance_us <= source_age_us <= max_age_us
        )

    def set_pico_tracking_armed(enabled: bool, source: str) -> None:
        """Switch between live Galaxy XR hand tracking and fixed-FK(Q_READY) IPOPT."""

        nonlocal pico_tracking_armed
        nonlocal pico_arm_pending
        nonlocal tracking_damping_started
        enabled = bool(enabled)
        if enabled:
            tick_us = now_us()
            robot_state_fresh = (
                latest_robot_q is not None
                and latest_robot_state_us > 0
                and tick_us - latest_robot_state_us <= 200_000
            )
            if not robot_state_fresh:
                pico_arm_pending = True
                log(f"{source}: arm pending until a fresh G1 robot state arrives")
                return
            if not args.auto_arm_xr_tracking:
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
                        f"{source}: arm pending until G1 reports Q_READY "
                        f"(max_arm_error={ready_error_rad:.3f}rad, threshold=0.100rad)"
                    )
                    return
            if not xr_targets_are_fresh(tick_us):
                pico_arm_pending = True
                log(f"{source}: arm tracking pending until a fresh valid XR wrist frame arrives")
                return
            pico_arm_pending = False
        if enabled == pico_tracking_armed:
            log(
                f"{source}: XR hand tracking already "
                f"{'armed' if enabled else 'disarmed'}"
            )
            return
        was_armed = pico_tracking_armed
        pico_tracking_armed = enabled
        pico_arm_pending = False
        resync_shared_control_state("pico_tracking_armed" if enabled else "pico_tracking_disarmed")
        if enabled:
            tracking_damping_started = time.monotonic()
            xr_continuity.reset_filter_to_accepted()
            for side in ("left", "right"):
                resync_damping_started[side] = None
                if not damped_active[side] and last_commanded_wrist_tf[side] is not None:
                    damped_reference[side].reset(last_commanded_wrist_tf[side])
                    damped_active[side] = True
            log(
                "XR engagement entered damped reference; starting from the last "
                "IPOPT wrist target and resetting XR LPF to the accepted pose"
            )
            if args.auto_arm_xr_tracking:
                log(f"{source}: teleop_tracking armed from a fresh XR wrist frame")
            else:
                log(
                    f"{source}: teleop_tracking armed after Q_READY confirmation; "
                    "XR wrist targets re-enabled"
                )
        else:
            tracking_damping_started = None
            if was_armed and xr_reference_initialized:
                for side in ("left", "right"):
                    previous_target = last_commanded_wrist_tf[side]
                    if previous_target is not None:
                        # Direct -> damped transitions restart at the exact pose
                        # sent on the preceding tick and zero their velocity.
                        damped_reference[side].reset(previous_target)
                    damped_active[side] = True
                    resync_damping_started[side] = None
                xr_continuity.reset_filter_to_accepted()
                log(
                    "XR disengagement entered damped Q_READY return from the "
                    "last IPOPT wrist targets"
                )
            # Keep IPOPT's previous solution as the warm start while both arms
            # transition to the fixed ready target.
            log(
                f"{source}: teleop_wait_ready active; fixed FK(Q_READY) "
                "wrist SE(3), XR arm targets ignored"
            )

    def try_complete_pico_arm_request(tick_us: int) -> None:
        if not pico_arm_pending or pico_tracking_armed:
            return
        if (
            latest_robot_q is None
            or latest_robot_state_us <= 0
            or tick_us - latest_robot_state_us > 200_000
            or not xr_targets_are_fresh(tick_us)
        ):
            return
        if args.auto_arm_xr_tracking:
            set_pico_tracking_armed(True, "MuJoCo XR auto-arm")
            return
        ready_error_rad = float(
            np.max(np.abs(np.asarray(latest_robot_q, dtype=np.float64)[15:29] - Q_READY[15:29]))
        )
        if ready_error_rad < 0.10:
            set_pico_tracking_armed(True, "terminal r pending")

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

    def publish_record_toggle() -> None:
        nonlocal record_control_sequence
        record_control_sequence += 1
        stamp = int(now_us())
        wrist_visual_lcm.publish(
            args.wrist_record_control_channel,
            pack_experiment_record_control(
                {
                    "event": "toggle",
                    "sequence": record_control_sequence,
                    "published_timestamp_us": stamp,
                }
            ),
        )

    if keyboard.enabled:
        log("terminal keys: r=start/resume hand tracking, s=record start/stop, q=stop")
    else:
        log("terminal keys unavailable; run PATH-state in an interactive terminal")
    log(f"QP reference gate initial state: {'ON' if qp_reference_enabled else 'OFF'}")
    log("PATH publishes arm joints only; DreamWaQ owns base, legs, and waist")

    next_deadline = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if quit_deadline is not None and now >= quit_deadline:
                break
            tick_us = now_us()
            dt = max(1.0e-4, now - last_tick)
            last_tick = now

            # Galaxy XR hand-tracked wrist targets arrive independently of
            # the PATH terminal key stream.
            try:
                raw = xr_hand_stream.recv(zmq.NOBLOCK)
                remember_pico_hand_fields(raw, "vr_pose_debug", tick_us)
                targets = wrist_targets_from_message(
                    raw,
                    "vr_pose_debug",
                    tick_us,
                    max_age_us=int(args.pico_body_gap_hold_ms * 1_000.0),
                    future_tolerance_us=int(args.xr_future_tolerance_ms * 1_000.0),
                )
                if targets is not None:
                    if targets.session_id != latest_pico_session_id:
                        latest_pico_session_id = targets.session_id
                        last_ipopt_q_ref = None
                        last_ipopt_tau_ff = None
                        xr_continuity.reset_session()
                        log(
                            "XR absolute-target session="
                            f"{latest_pico_session_id}; reset XR continuity and IPOPT warm start"
                        )
                    for side, transform in (
                        ("left", targets.left_waist),
                        ("right", targets.right_waist),
                    ):
                        result = xr_continuity.accept(
                            side,
                            transform,
                            targets.source_timestamp_us,
                            now,
                        )
                        if result == "resync":
                            xr_continuity.reset_filter_to_accepted(side)
                            if (
                                pico_tracking_armed
                                and xr_reference_initialized
                                and tracking_damping_started is None
                            ):
                                previous_target = last_commanded_wrist_tf[side]
                                if previous_target is None and latest_robot_q is not None:
                                    fk_left, fk_right = arm_nonlinear_ik.wrist_transforms_from_q29(
                                        latest_robot_q
                                    )
                                    previous_target = fk_left if side == "left" else fk_right
                                if previous_target is not None:
                                    damped_reference[side].reset(previous_target)
                                    damped_active[side] = True
                                    resync_damping_started[side] = now
                                    log(
                                        f"XR {side} outlier resync#{xr_continuity.resync_count[side]} "
                                        "accepted after timeout; entering bounded damping"
                                    )
                        elif result == "rejected":
                            pass
                    accepted_left = xr_continuity.accepted["left"]
                    accepted_right = xr_continuity.accepted["right"]
                    if accepted_left is not None and accepted_right is not None:
                        latest_wrist_targets = WaistWristTargets(
                            left_waist=accepted_left.copy(),
                            right_waist=accepted_right.copy(),
                            session_id=targets.session_id,
                            frame_index=targets.frame_index,
                            source_timestamp_us=targets.source_timestamp_us,
                            valid=True,
                        )
                        latest_pico_body_receive_us = tick_us
            except zmq.Again:
                pass

            filtered_xr_targets = xr_continuity.tick(dt)

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
                            resync_shared_control_state("static_target_update")
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
                            resync_shared_control_state("static_target_replace")
                            if (
                                args.static_target_auto_enter
                                and current_state != "auto_planning"
                            ):
                                apply_state_request(
                                    "auto_planning", "static EE target", tick_us
                                )
            except zmq.Again:
                pass

            if direct_state is not None:
                lcm_state, lcm_state_received_us = direct_state.latest()
                if (
                    lcm_state is not None
                    and tick_us - lcm_state_received_us <= 200_000
                ):
                    latest_robot_q = lcm_state.q.copy()
                    latest_robot_dq = lcm_state.dq.copy()
                    latest_robot_base_quat = lcm_state.base_quat_wxyz.copy()
                    latest_robot_base_accel = lcm_state.base_accel.copy()
                    latest_robot_base_gyro = lcm_state.base_gyro.copy()
                    latest_robot_state_us = int(lcm_state_received_us)

            if latest_robot_q is not None and not xr_reference_initialized:
                initial_left_tf, initial_right_tf = (
                    arm_nonlinear_ik.wrist_transforms_from_q29(latest_robot_q)
                )
                for side, transform in (
                    ("left", initial_left_tf),
                    ("right", initial_right_tf),
                ):
                    damped_reference[side].reset(transform)
                    last_commanded_wrist_tf[side] = transform.copy()
                    damped_active[side] = True
                xr_reference_initialized = True
                log(
                    "damped wrist references initialized from FK of the first "
                    "fresh G1 state; arm commands remain inhibited until this state"
                )
                log(
                    "initial teleop_wait_ready damping toward FK(Q_READY) started; "
                    f"v_max={args.damped_linear_speed_limit_m_s:.3f}m/s "
                    f"omega_max={args.damped_angular_speed_limit_rad_s:.3f}rad/s"
                )

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
                    if "state" in command:
                        requested = str(command.get("state", current_state))
                        apply_state_request(requested, "external", tick_us)
            except zmq.Again:
                pass

            keyboard_command = keyboard.poll_command()
            if keyboard_command == TerminalKeyInput.START:
                if current_state != "teleop":
                    apply_state_request("teleop", "terminal r", tick_us)
                if pico_tracking_armed:
                    log("terminal r: XR arm tracking is already active")
                elif pico_arm_pending:
                    log("terminal r: waiting for fresh hands and G1 Q_READY state")
                else:
                    set_pico_tracking_armed(True, "terminal r")
            elif keyboard_command == TerminalKeyInput.RECORD:
                if recording_active:
                    publish_record_toggle()
                    recording_active = False
                    set_pico_tracking_armed(False, "terminal s: recording stopped")
                    log("terminal s: save requested; arms are returning to Q_READY; press r to resume")
                elif pico_tracking_armed or pico_arm_pending:
                    publish_record_toggle()
                    recording_active = True
                    log("terminal s: PATH experiment recording started")
                else:
                    log("terminal s ignored: press r to start arm tracking first")
            elif keyboard_command == TerminalKeyInput.QUIT:
                if recording_active:
                    publish_record_toggle()
                    recording_active = False
                    log("terminal q: closing the active recording")
                if current_state != "teleop":
                    apply_state_request("teleop", "terminal q", tick_us)
                set_pico_tracking_armed(False, "terminal q")
                quit_deadline = now + 1.5
                log("terminal q: holding Q_READY briefly, then stopping PATH")

            pico_body_age_sec = (
                (tick_us - latest_pico_body_receive_us) / 1.0e6
                if latest_pico_body_receive_us > 0
                else float("inf")
            )
            pico_source_age_sec = (
                (tick_us - latest_wrist_targets.source_timestamp_us) / 1.0e6
                if latest_wrist_targets is not None
                else float("inf")
            )
            pico_body_fresh = (
                pico_tracking_armed
                and latest_wrist_targets is not None
                and pico_body_age_sec
                <= max(0.001, float(args.pico_body_gap_hold_ms) / 1000.0)
                and -max(0.0, float(args.xr_future_tolerance_ms) / 1000.0)
                <= pico_source_age_sec
                <= max(0.001, float(args.pico_body_gap_hold_ms) / 1000.0)
            )
            try_complete_pico_arm_request(tick_us)

            hand_fresh = (
                latest_hand_us > 0
                and tick_us - latest_hand_us
                <= max(0.0, args.pico_hand_timeout_sec) * 1.0e6
            )
            grasp_commanded = {
                side: pico_grip_active(side, tick_us) for side in ("left", "right")
            }
            # No measured Revo2 contact state reaches this bridge yet. ``None``
            # leaves the jaw-stall channel inactive; co-motion and the
            # explicit long-hold override remain available.
            jaw_gap_m = {side: None for side in ("left", "right")}
            grasp_engaged = {side: False for side in ("left", "right")}
            grasp_state = {
                side: grasp_detector[side].state for side in ("left", "right")
            }
            # Optional shared-target selection uses hand pinch state only.
            # It is disabled by default and never gates ordinary hand tracking.
            requested_shared_hand = "none"
            if (
                current_state == "teleop"
                and hand_fresh
                and args.enable_shared_target_assist
            ):
                left_pinch = max(
                    latest_hand_meta.get("left_trigger", 0.0),
                    latest_hand_meta.get("left_grip", 0.0),
                ) >= 0.5
                right_pinch = max(
                    latest_hand_meta.get("right_trigger", 0.0),
                    latest_hand_meta.get("right_grip", 0.0),
                ) >= 0.5
                if left_pinch and not right_pinch:
                    requested_shared_hand = "left"
                elif right_pinch and not left_pinch:
                    requested_shared_hand = "right"
            if requested_shared_hand != shared_selected_hand:
                shared_selected_hand = requested_shared_hand
                shared_hand_selection_revision += 1
                # Pinch only selects whether shared constraints are considered;
                # unselected arms receive the current Televuer goals directly.
                log(
                    "shared-control hold "
                    + (
                        f"enabled for {shared_selected_hand} arm"
                        if shared_selected_hand != "none"
                        else "released"
                    )
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
            if latest_robot_q is not None and current_state in {"teleop", "intention"}:
                if current_state == "teleop" and not pico_tracking_armed:
                    # Ease to FK(Q_READY); never send an arm command before the
                    # first fresh G1 state has initialized the SE(3) references.
                    ready_targets = apply_teleop_wrist_reference(
                        {"left": q_ready_left_wrist_tf, "right": q_ready_right_wrist_tf},
                        dt,
                        now,
                    )
                    ready_result = solve_authoritative_ipopt_transforms(
                        ready_targets["left"],
                        ready_targets["right"],
                    )
                    q_arm_reference = (
                        ready_result.q_ref.copy()
                        if ready_result.success
                        else last_ipopt_or_measured_hold()
                    )
                    if ready_result.success:
                        control_visual_hands = ipopt_visual_hands_from_transforms(
                            ready_targets["left"],
                            ready_targets["right"],
                            target_source="teleop_wait_ready_damped_q_ready",
                        )
                    else:
                        hold_left_tf, hold_right_tf = (
                            arm_nonlinear_ik.wrist_transforms_from_q29(q_arm_reference)
                        )
                        control_visual_hands = ipopt_visual_hands_from_transforms(
                            hold_left_tf,
                            hold_right_tf,
                            target_source="teleop_wait_ready_ipopt_hold",
                        )
                    qp_payload = ipopt_arm_sample_reference(
                        q_arm_reference,
                        state="teleop",
                        source="xr_wait_ready_fk",
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
                        pico_target_reason="fixed_fk_q_ready",
                        pico_target_held=True,
                        pico_left_target_position=ready_targets["left"][:3, 3].tolist(),
                        pico_right_target_position=ready_targets["right"][:3, 3].tolist(),
                        q_prior_source="ipopt_damped_q_ready_fk",
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
                    shared_orientation_source = "xr_operator"
                    if pico_body_fresh and latest_wrist_targets is not None:
                        left_target_tf = (
                            filtered_xr_targets["left"].copy()
                            if filtered_xr_targets["left"] is not None
                            else latest_wrist_targets.left_waist.copy()
                        )
                        right_target_tf = (
                            filtered_xr_targets["right"].copy()
                            if filtered_xr_targets["right"] is not None
                            else latest_wrist_targets.right_waist.copy()
                        )
                        human_source_timestamp_us = int(
                            latest_wrist_targets.source_timestamp_us
                        )
                        pico_frame_index = int(latest_wrist_targets.frame_index)
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
                        pico_frame_index = -1
                        human_wrist_source = "robot_fk_hold"
                    q_measured_for_hand = latest_robot_q if latest_robot_q is not None else last_ipopt_or_measured_hold()
                    measured_left_tf, measured_right_tf = arm_nonlinear_ik.wrist_transforms_from_q29(q_measured_for_hand)
                    measured_tf = {"left": measured_left_tf, "right": measured_right_tf}
                    human_tf = {"left": left_target_tf, "right": right_target_tf}
                    # Keep the raw command separate from any explicitly
                    # selected shared target so belief never treats a blended
                    # goal as fresh operator intent.
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
                            p_prev, r_prev, t_prev = previous
                            delta = (tick_us - t_prev) / 1.0e6
                            if 1.0e-4 <= delta <= 0.5:
                                velocity = (p_cmd - p_prev) / delta
                                raw_omega = angular_velocity_from_rotations(
                                    r_prev, r_cmd, delta
                                )
                                if raw_omega is not None:
                                    omega = raw_omega
                        command_history[side] = (
                            p_cmd.copy(),
                            r_cmd.copy(),
                            int(tick_us),
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
                        position_command = command_tf[side][:3, 3]
                        rotation_command = command_tf[side][:3, :3]
                        detector = grasp_detector[side]
                        manual_grasp_override = bool(
                            grasp_commanded[side]
                            and detector.closing_elapsed_s
                            >= shared_config.grasp_override_hold_s
                        )
                        grasp_state[side] = detector.update(
                            commanded_closed=grasp_commanded[side],
                            jaw_gap_m=jaw_gap_m[side],
                            p_camera=(
                                None
                                if belief_target is None
                                else belief_target.center_camera_m
                            ),
                            p_goal=(
                                None
                                if belief_target is None
                                else belief_target.center_goal_m
                            ),
                            now_s=now,
                            dt=dt,
                            manual_override=manual_grasp_override,
                        )
                        holding = grasp_state[side] == "holding"
                        grasp_engaged[side] = holding

                        # HOLDING disables the position belief target, while a
                        # separately gated orientation constraint may remain.
                        orientation_belief_active[side] = False
                        rotation_target = rotation_command
                        orientation_source = "operator"
                        gate_reason = "no_constraint"
                        gate_weight = 0.0
                        belief_weight = 0.0
                        final_orientation_weight = 0.0
                        current_rotation = rotation_command
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
                                orientation_belief_active[side] = True
                            gate_reason = gate.last_reason
                        else:
                            orientation_gate[side].authority(None, current_rotation, dt)
                            orientation_authority[side].update(0.0, gate_ok=False, dt=dt)
                        # Feed the current Televuer orientation directly to
                        # IK. A shared orientation constraint may explicitly
                        # alter this target, but no reference dynamics follow it.
                        # The direct Televuer position is the default. Shared
                        # target logic only changes it while a valid pinch target
                        # is actively selected.
                        position_target = position_command
                        position_source = "operator"
                        position_belief_weight = 0.0
                        position_unfiltered_target = position_command.copy()
                        if (
                            belief_target is not None
                            and belief_target.position_valid
                            and not holding
                        ):
                            goal = np.asarray(belief_target.position, dtype=np.float64).reshape(3)
                            is_canonical_grasp_center = (
                                belief_target.position_point == "canonical_grasp_center"
                                or belief_target.position_point.endswith("_grasp_center")
                            )
                            goal_wrist = (
                                goal - rotation_target @ grasp_center_wrist_offset[side]
                                if is_canonical_grasp_center
                                else goal
                            )
                            if np.linalg.norm(goal_wrist - position_command) <= shared_config.d_max_m:
                                if args.shared_position_mode == "alpha":
                                    if args.shared_position_alpha_source == "posterior":
                                        alpha = position_authority[side].update(
                                            belief_target.confidence,
                                            gate_ok=True,
                                            dt=dt,
                                        )
                                    else:
                                        position_authority[side].update(
                                            0.0, gate_ok=False, dt=dt
                                        )
                                        alpha = float(args.shared_position_alpha)
                                    position_unfiltered_target = (
                                        (1.0 - alpha) * position_command
                                        + alpha * goal_wrist
                                    )
                                    position_target = position_unfiltered_target
                                    position_source = (
                                        "belief_contact_alpha"
                                        if alpha > 0.0 else "operator"
                                    )
                                    position_belief_weight = float(alpha)
                                else:
                                    position_target = goal_wrist
                                    position_unfiltered_target = goal_wrist.copy()
                                    position_source = "belief_contact"
                                    position_authority[side].update(
                                        0.0, gate_ok=False, dt=dt
                                    )
                                    position_belief_weight = 1.0
                            else:
                                # A stale/remote target must not retain a
                                # prior posterior authority while VR remains
                                # the active position source.
                                position_authority[side].update(
                                    0.0, gate_ok=False, dt=dt
                                )
                        else:
                            position_authority[side].update(0.0, gate_ok=False, dt=dt)
                        # In ordinary XR teleoperation, this is the exact
                        # Televuer waist-frame position. The optional shared
                        # target branch above is the only Cartesian alteration.
                        position_unfiltered_target = position_target.copy()
                        measured_lag_m = float(
                            np.linalg.norm(position_target - measured_tf[side][:3, 3])
                        )
                        human_tf[side][:3, :3] = rotation_target
                        human_tf[side][:3, 3] = position_target
                        command_velocity = command_features[side][2]
                        command_omega = command_features[side][3]
                        reference_diagnostics[side] = {
                            "armed": bool(belief_target is not None), "position_source": position_source,
                            "orientation_source": orientation_source,
                            # Filled with the raw-to-IPOPT residual after the
                            # configured damped or residual-offset reference.
                            "position_error_m": 0.0,
                            "orientation_error_rad": 0.0,
                            "reference_speed_mps": (
                                0.0 if command_velocity is None
                                else float(np.linalg.norm(command_velocity))
                            ),
                            "reference_rate_rps": (
                                0.0 if command_omega is None
                                else float(np.linalg.norm(command_omega))
                            ),
                            "measured_lag_m": measured_lag_m,
                            "goal_mode": "direct_televuer_or_shared_target",
                            "object_id": "" if belief_target is None else belief_target.object_id,
                            "relation": "" if belief_target is None else belief_target.orientation_relation,
                            "orientation_gate_reason": gate_reason,
                            "orientation_gate_rejects": int(orientation_gate[side].reject_count),
                            "orientation_gate_weight": float(gate_weight),
                            "orientation_belief_weight": float(belief_weight),
                            "orientation_authority_weight": float(final_orientation_weight),
                            "orientation_smoothed_confidence": float(orientation_authority[side].smoothed_confidence),
                            "orientation_belief_active": bool(orientation_belief_active[side]),
                            "orientation_rebase_active": False,
                            "position_blend_mode": str(args.shared_position_mode),
                            "position_alpha_source": str(args.shared_position_alpha_source),
                            "position_belief_weight": float(position_belief_weight),
                            "position_alpha_lpf_enabled": False,
                            "position_alpha_lpf_applied": False,
                            "position_alpha_lpf_alpha": 1.0,
                            "position_unfiltered_target_m": position_unfiltered_target.tolist(),
                            "position_smoothed_confidence": float(position_authority[side].smoothed_confidence),
                            "frame_offset": {
                                "offset_m": [0.0, 0.0, 0.0],
                                "offset_norm_m": 0.0,
                                "peak_offset_m": 0.0,
                            },
                            "grasp_commanded": bool(grasp_commanded[side]),
                            "grasp_engaged": bool(grasp_engaged[side]),
                            "jaw_gap_m": jaw_gap_m[side],
                            "grasp_state": grasp_state[side],
                            "grasp": {
                                "state": grasp_state[side],
                                "channels": dict(detector.channels),
                                "comotion_log_ratio": detector.comotion_log_ratio,
                                "jaw_gap_m": jaw_gap_m[side],
                                "closing_elapsed_s": float(detector.closing_elapsed_s),
                                "grasp_failed": bool(detector.grasp_failed),
                                "override_active": bool(detector.override_active),
                                "override_count": int(detector.override_count),
                            },
                            "grasp_state_source": "v17_physical_state_machine",
                        }
                    if current_state == "teleop":
                        raw_goal = {side: human_tf[side].copy() for side in ("left", "right")}
                        human_tf = apply_teleop_wrist_reference(raw_goal, dt, now)
                        for side in ("left", "right"):
                            reference_diagnostics[side]["position_error_m"] = float(
                                np.linalg.norm(
                                    raw_goal[side][:3, 3] - human_tf[side][:3, 3]
                                )
                            )
                            reference_diagnostics[side]["orientation_error_rad"] = float(
                                np.linalg.norm(
                                    so3_log(
                                        human_tf[side][:3, :3].T
                                        @ raw_goal[side][:3, :3]
                                    )
                                )
                            )
                            reference_diagnostics[side]["goal_mode"] = (
                                "damped_se3"
                                if damped_active[side]
                                else "direct_with_residual_offset"
                            )
                    control_visual_hands = {
                        side: {
                            # Exact wrist-yaw Cartesian goal handed to IPOPT.
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
                            "position_blend_mode": reference_diagnostics[side]["position_blend_mode"],
                            "position_alpha_source": reference_diagnostics[side]["position_alpha_source"],
                            "position_belief_weight": reference_diagnostics[side]["position_belief_weight"],
                            "position_smoothed_confidence": reference_diagnostics[side]["position_smoothed_confidence"],
                            "reference_speed_mps": reference_diagnostics[side]["reference_speed_mps"],
                            "goal_mode": reference_diagnostics[side]["goal_mode"],
                            "grasp_commanded": bool(grasp_commanded[side]),
                            "grasp_engaged": bool(grasp_engaged[side]),
                            "jaw_gap_m": jaw_gap_m[side],
                            "grasp_state": reference_diagnostics[side]["grasp_state"],
                            "grasp": reference_diagnostics[side]["grasp"],
                            "frame_offset": reference_diagnostics[side]["frame_offset"],
                            "grasp_state_source": reference_diagnostics[side][
                                "grasp_state_source"
                            ],
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
                    q_arm_reference = (
                        ik_result.q_ref.copy()
                        if ik_result.success
                        else last_ipopt_or_measured_hold()
                    )
                    source_timestamp_us = human_source_timestamp_us
                    if ik_result.success:
                        qp_payload = ipopt_arm_sample_reference(
                            q_arm_reference,
                            state=current_state,
                            source="xr_wrist_target",
                            control_mode="teleop",
                            feasible=True,
                            plan_id=0,
                            timestamp_us=tick_us,
                            tau_ff_ref=ik_result.tau_ff_ref,
                            bridge_timestamp_us=tick_us,
                            pico_source_timestamp_us=source_timestamp_us,
                            pico_frame_index=pico_frame_index,
                            pico_target_reason="absolute_head_position",
                            pico_target_held=False,
                            pico_left_target_position=left_target_tf[:3, 3].tolist(),
                            pico_right_target_position=right_target_tf[:3, 3].tolist(),
                            q_prior_source="ipopt_xr_hand_ee",
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
                                shared_human_wrist_source=human_wrist_source,
                                shared_coordinate_frame=PICO_WAIST_FRAME,
                                shared_orientation_source="direct_xr_or_shared_constraint",
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
                                    q_prior_source="ipopt_shared_ee_pose",
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
                            source="xr_wrist_target",
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
                            pico_source_timestamp_us=source_timestamp_us,
                            pico_frame_index=pico_frame_index,
                            pico_target_reason="ipopt_failed_hold",
                            pico_target_held=True,
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
                    hold_q = last_ipopt_or_measured_hold()
                    hold_left_tf, hold_right_tf = (
                        arm_nonlinear_ik.wrist_transforms_from_q29(hold_q)
                    )
                    control_visual_hands = ipopt_visual_hands_from_transforms(
                        hold_left_tf,
                        hold_right_tf,
                        target_source="stale_xr_display_hold",
                    )
                    # Keep the last wrist marker visible for diagnostics, but
                    # release arm-SDK ownership when the live XR target ages
                    # out. A display hold is not a new robot command.
                    qp_payload = qp_none_reference(
                        state=current_state,
                        source="xr_wrist_target_stale",
                        control_mode="stale_wait",
                        feasible=False,
                        timestamp_us=tick_us,
                        bridge_timestamp_us=tick_us,
                        pico_source_timestamp_us=missing_pico_source_us,
                        pico_frame_index=missing_pico_frame,
                        pico_target_reason=(
                            "stale_absolute_target_release"
                            if hold_tracking_target
                            else "waiting_fresh_absolute_target"
                        ),
                        pico_target_held=False,
                        arm_target_valid=False,
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

            # Direct path: Thor owns *only* IPOPT arm q/dq/tau. DreamWaQ owns
            # the waist and legs; PATH sends no command for those joints.
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
                            # Hand pinch values share the fresh XR pose packet;
                            # the G1 arm bridge ignores grip flags, while the
                            # separate Revo2 MPPI process owns finger targets.
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
                    qp_payload.get("pico_target_held", False)
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
            # Pinch selection available to a gateway that starts later too.
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
                pico_position_log = "xr_L/R=missing"
                if latest_wrist_targets is not None:
                    left_position = latest_wrist_targets.left_waist[:3, 3]
                    right_position = latest_wrist_targets.right_waist[:3, 3]
                    pico_position_log = (
                        "xr_L=("
                        f"{left_position[0]:+.3f},"
                        f"{left_position[1]:+.3f},"
                        f"{left_position[2]:+.3f}) "
                        "xr_R=("
                        f"{right_position[0]:+.3f},"
                        f"{right_position[1]:+.3f},"
                        f"{right_position[2]:+.3f})"
                    )
                if latest_shared_result is not None:
                    left_ref = latest_shared_result["left"]
                    right_ref = latest_shared_result["right"]
                    shared_log = (
                        f"shared_select={shared_selected_hand} "
                        f"goal_L[{left_ref['position_source']}/{left_ref['orientation_source']} "
                        f"mode={left_ref['goal_mode']} gate={left_ref['orientation_gate_reason']}] "
                        f"goal_R[{right_ref['position_source']}/{right_ref['orientation_source']} "
                        f"mode={right_ref['goal_mode']} gate={right_ref['orientation_gate_reason']}]"
                    )
                log(
                    f"state={current_state} xr_wrist_fresh={pico_body_fresh} "
                    f"xr_session={latest_pico_session_id}/"
                    f"{'teleop_tracking' if pico_tracking_armed else 'teleop_wait_ready'} "
                    f"static_target={'fresh' if static_target_fresh else 'held' if static_target_usable else 'missing'} "
                    f"static_age={static_target_age_sec if np.isfinite(static_target_age_sec) else float('nan'):.2f}s "
                    f"static_rev={static_target_revision} "
                    f"qp_gate={'on' if qp_reference_enabled else 'off'} "
                    f"direct={'sent' if direct_command_sent else 'hold'}"
                    f"#{direct_lcm_sequence} "
                    f"g1_state_age={1000.0 * (tick_us - latest_robot_state_us) / 1.0e6 if latest_robot_state_us else -1.0:.1f}ms "
                    f"hand_fresh={hand_fresh} "
                    f"trig=({latest_hand_meta.get('left_trigger', float('nan')):.2f},"
                    f"{latest_hand_meta.get('right_trigger', float('nan')):.2f}) "
                    f"grip=({latest_hand_meta.get('left_grip', float('nan')):.2f},"
                    f"{latest_hand_meta.get('right_grip', float('nan')):.2f}) "
                    f"hand_age={(tick_us - latest_hand_us) / 1000.0 if latest_hand_us else -1.0:.1f}ms "
                    f"direct_grip=({int(direct_left_grip)},{int(direct_right_grip)}) "
                    f"qp_type={qp_payload.get('reference_type') if qp_payload else 'none'} "
                    f"source={qp_payload.get('source', 'none') if qp_payload else 'none'} "
                    f"q_prior={qp_payload.get('q_prior_source', 'n/a') if qp_payload else 'n/a'} "
                    f"xr_target={'absolute_head_position' if latest_wrist_targets else 'waiting'} "
                    f"{pico_position_log} "
                    f"xr_age={1000.0 * pico_body_age_sec if np.isfinite(pico_body_age_sec) else -1.0:.1f}ms "
                    f"xr_source_age={1000.0 * pico_source_age_sec if np.isfinite(pico_source_age_sec) else -1.0:.1f}ms "
                    f"xr_outliers=L{xr_continuity.reject_count['left']}/R{xr_continuity.reject_count['right']} "
                    f"xr_resyncs=L{xr_continuity.resync_count['left']}/R{xr_continuity.resync_count['right']} "
                    f"recording={'on' if recording_active else 'off'} "
                    f"quit_pending={quit_deadline is not None} "
                    f"control={qp_payload.get('control_mode', 'none') if qp_payload else 'none'} "
                    f"plan={qp_payload.get('plan_id', 0) if qp_payload else 0} "
                    f"phase={float(qp_payload.get('phase', 0.0)) if qp_payload else 0.0:.2f} "
                    f"phase_mode={phase_mode} "
                    f"phase_err={phase_sync_error_rad if np.isfinite(phase_sync_error_rad) else float('nan'):.3f}rad "
                    f"arm_ik={last_arm_ik_status} "
                    f"ik_err={100.0 * last_arm_ik_error_m if np.isfinite(last_arm_ik_error_m) else float('nan'):.2f}cm "
                    f"ik_solve={last_arm_ik_solve_ms if np.isfinite(last_arm_ik_solve_ms) else float('nan'):.2f}ms "
                    f"{gravity_estimator.status()} "
                    f"{shared_log} "
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
        xr_hand_stream.close()
        state_sub.close()
        qp.close()
        shared_hand_pub.close()
        if direct_state is not None:
            direct_state.close()
        keyboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
