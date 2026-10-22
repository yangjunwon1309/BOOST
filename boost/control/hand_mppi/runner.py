"""Retarget Televuer or local JSON hand landmarks into BrainCo Revo2 DDS.

Run this module from BOOST's Pixi ``hand`` environment. Publishing is disabled
unless ``--publish`` is given; the default mode prints retargeted targets only.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zmq

from boost.control.hand_mppi.brainco_dds import BrainCoDDS
from boost.control.hand_mppi.brainco_revo2_model import BrainCoRevo2HandModel
from boost.control.hand_mppi.revo2 import BrainCoRevo2Adapter
from boost.io.path_transport.zmq_wire import unpack_pose_message


ROOT = Path(__file__).resolve().parents[3]
SIDES = ("left", "right")


def _scalar(payload: dict[str, np.ndarray], name: str) -> Any:
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ValueError(f"{name} must contain one value, got shape {value.shape}")
    return value.reshape(-1)[0].item()


@dataclass(frozen=True)
class TelevuerFrame:
    sequence: int
    timestamp_monotonic: float
    hands: dict[str, np.ndarray | None]


def decode_televuer_frame(
    raw: bytes,
    *,
    topic: str,
    expected_schema_version: int,
    expected_coordinate_frame: int,
    expected_joint_count: int,
    stale_timeout_sec: float,
    future_tolerance_sec: float,
) -> TelevuerFrame:
    """Parse and validate PATH's versioned hand-landmark fields."""
    payload = unpack_pose_message(raw, topic=topic)
    required = {
        "hand_landmarks_schema_version",
        "hand_landmarks_coordinate_frame",
        "hand_landmarks_sequence",
        "hand_landmarks_timestamp_monotonic",
        "hand_landmarks_valid",
        "hand_landmarks_left",
        "hand_landmarks_right",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"PATH message has no Televuer hand fields: {sorted(missing)}")

    schema = int(_scalar(payload, "hand_landmarks_schema_version"))
    frame = int(_scalar(payload, "hand_landmarks_coordinate_frame"))
    if schema != expected_schema_version:
        raise ValueError(f"unsupported Televuer hand schema {schema}")
    if frame != expected_coordinate_frame:
        raise ValueError(f"unexpected Televuer coordinate frame {frame}")

    sequence = int(_scalar(payload, "hand_landmarks_sequence"))
    timestamp = float(_scalar(payload, "hand_landmarks_timestamp_monotonic"))
    now = time.monotonic()
    if (
        not np.isfinite(timestamp)
        or timestamp > now + future_tolerance_sec
        or now - timestamp > stale_timeout_sec
    ):
        raise ValueError("Televuer timestamp is invalid, stale, or from the future")

    valid = np.asarray(payload["hand_landmarks_valid"], dtype=bool).reshape(-1)
    if valid.shape != (2,):
        raise ValueError(f"hand validity must be [left,right], got {valid.shape}")

    hands: dict[str, np.ndarray | None] = {}
    for index, side in enumerate(SIDES):
        points = np.asarray(payload[f"hand_landmarks_{side}"], dtype=np.float32)
        if points.shape != (expected_joint_count, 3):
            raise ValueError(
                f"{side} hand must have shape ({expected_joint_count}, 3), got {points.shape}"
            )
        hands[side] = (
            points.copy() if valid[index] and np.isfinite(points).all() else None
        )

    return TelevuerFrame(sequence, timestamp, hands)


def decode_override_frame(
    path: Path, *, sequence: int, expected_joint_count: int
) -> TelevuerFrame:
    """Read an editable JSON hand pose for local simulation without XR/PATH."""
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("hand override JSON must be an object with left/right arrays")

    hands: dict[str, np.ndarray | None] = {}
    for side in SIDES:
        value = payload.get(side)
        if value is None:
            hands[side] = None
            continue
        points = np.asarray(value, dtype=np.float32)
        if points.shape != (expected_joint_count, 3):
            raise ValueError(
                f"override {side} hand must have shape "
                f"({expected_joint_count}, 3), got {points.shape}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"override {side} hand contains non-finite coordinates")
        hands[side] = points.copy()
    if not any(points is not None for points in hands.values()):
        raise ValueError("hand override JSON must provide at least one hand")
    return TelevuerFrame(sequence, time.monotonic(), hands)


@dataclass
class SideRuntime:
    side: str
    retargeter: Any
    jax_numpy: Any
    q_min: np.ndarray
    q_max: np.ndarray
    max_velocity: np.ndarray
    last_sequence: int | None = None
    last_valid_timestamp: float | None = None
    desired_rad: np.ndarray | None = None
    command_rad: np.ndarray | None = None
    output_timestamp: float | None = None
    state_seeded: bool = False


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def _create_retargeter(
    side: str,
    config: dict[str, Any],
    hand_config: dict[str, Any],
    *,
    device: str | None,
    surface_client: Any | None = None,
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    # Importing the upstream stack is isolated to this Pixi hand process. The
    # sim environment remains on its own MuJoCo 3.5 / Python 3.10 dependency set.
    import jax
    import jax.numpy as jnp
    from mimic_retargeter_lab.hand_models import ManusHandModel
    from mimic_retargeter_lab.retargeting.online import SamplingBasedRetargeter
    from mimic_retargeter_lab.types import Chirality, HandLandmark
    from mimic_retargeter_lab.utils import resolve_jax_device

    sampling_params = dict(config["retargeter"]["sampling_params"])
    if device is not None:
        sampling_params["device"] = device
    target_device = resolve_jax_device(
        str(sampling_params.get("device", "cuda"))
    )

    # Configure the persistent cache before constructing any MJX models, and
    # cache short as well as long compilation steps for subsequent launches.
    cache_dir = None
    if sampling_params.get("jax_compilation_cache_enable", True):
        cache_dir = Path(
            str(
                sampling_params.get(
                    "jax_compilation_cache_dir",
                    "~/.cache/mimic_retargeter_lab/jax_compilation",
                )
            )
        ).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    print(
        f"[hand-mppi] {side}: JAX device={target_device}; "
        f"persistent cache={cache_dir if cache_dir is not None else 'disabled'}",
        flush=True,
    )

    chirality = Chirality(side)
    hand_model = BrainCoRevo2HandModel(
        robot_hand_base_path=(ROOT / config["robot_hand"]["mjcf_directory"]).resolve(),
        chirality=chirality,
        distal_couplings=hand_config["distal_couplings"],
        mjx_device=target_device,
    )
    source_model = ManusHandModel(chirality=chirality)

    wrist_name = f"{side}_hand_base_link"
    retargeter_class = SamplingBasedRetargeter
    retargeter_kwargs: dict[str, Any] = {}
    if surface_client is not None:
        from boost.control.hand_mppi.contact_guided_retargeter import (
            ContactGuidedRetargeter,
        )

        retargeter_class = ContactGuidedRetargeter
        retargeter_kwargs = {
            "side": side,
            "surface_client": surface_client,
            "contact_config": config["retargeter"].get("contact_guidance", {}),
        }
    retargeter = retargeter_class(
        from_model=source_model,
        to_model=hand_model,
        wrist_mapping={
            "tgt_key": wrist_name,
            "root_key": wrist_name,
            "transform": np.eye(4, dtype=np.float32).tolist(),
        },
        alignment_landmarks=[
            HandLandmark(name) for name in config["retargeter"]["alignment_landmarks"]
        ],
        keyvectors_cfg=config["retargeter"]["keyvectors_cfg"],
        dexpilot_params=config["retargeter"]["dexpilot_params"],
        sampling_params=sampling_params,
        precomputed_scale=config["retargeter"].get("precomputed_scale"),
        debug_mode=bool(config["retargeter"].get("debug_mode", False)),
        debug_every_n_frames=int(config["retargeter"].get("debug_every_n_frames", 1)),
        **retargeter_kwargs,
    )

    adapter = BrainCoRevo2Adapter.from_mapping(hand_config)
    q_min = np.asarray(adapter.q_min, dtype=np.float32)
    q_max = np.asarray(adapter.q_max, dtype=np.float32)
    max_velocity = np.asarray(
        config["control"]["max_joint_velocity_rad_s"], dtype=np.float32
    )
    if q_min.shape != (6,) or q_max.shape != (6,) or max_velocity.shape != (6,):
        raise ValueError("Revo2 range and velocity settings must each have six channels")
    if not np.isfinite(max_velocity).all() or np.any(max_velocity <= 0):
        raise ValueError("max_joint_velocity_rad_s values must be positive and finite")
    if np.any(q_max <= q_min):
        raise ValueError("BrainCo q_max must exceed q_min for all six channels")
    model_limits = hand_model.get_actuated_joint_limits()
    for index, name in enumerate(hand_model.get_actuated_joint_names()):
        model_min, model_max = model_limits[name]
        if not (
            np.isclose(model_min, q_min[index], atol=1e-6)
            and np.isclose(model_max, q_max[index], atol=1e-6)
        ):
            raise ValueError(
                f"BrainCo adapter range for {name} does not match the official MJCF: "
                f"config=({q_min[index]}, {q_max[index]}), "
                f"MJCF=({model_min}, {model_max})"
            )

    # Keep Mimic's warm start aligned with measured hardware/simulator state.
    # It is filled when fresh BrainCo DDS state first arrives.
    return retargeter, jnp, q_min, q_max, max_velocity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/hands/brainco_revo2_mppi.yaml"
    )
    parser.add_argument("--input-endpoint", default=None)
    parser.add_argument(
        "--input-mode",
        choices=("televuer", "override"),
        default="televuer",
        help="read PATH's Televuer ZMQ stream or an editable local JSON pose",
    )
    parser.add_argument(
        "--override-file",
        type=Path,
        default=None,
        help="JSON hand pose file, required with --input-mode override",
    )
    parser.add_argument("--interface", default="lo", help="DDS interface")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--surface-query-endpoint",
        default=None,
        help="enable contact-guided targets from a live MuJoCo surface-query REP endpoint",
    )
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish BrainCo commands; without this flag the runner is read-only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _load_yaml(config_path.resolve())
    hand_config = _load_yaml((ROOT / config["hand_config"]).resolve())
    input_cfg = config["input"]
    endpoint = args.input_endpoint or input_cfg["endpoint"]
    override_file = args.override_file
    if args.input_mode == "override":
        if override_file is None:
            raise ValueError("--override-file is required with --input-mode override")
        if not override_file.is_absolute():
            override_file = ROOT / override_file
        override_file = override_file.resolve()
        # Fail before initializing DDS if the selected user pose is malformed.
        decode_override_frame(
            override_file, sequence=0, expected_joint_count=int(input_cfg["joint_count"])
        )
    sides = SIDES if args.side == "both" else (args.side,)
    update_hz = float(config["control"]["update_hz"])
    stale_timeout = float(input_cfg["stale_timeout_sec"])
    future_tolerance = float(input_cfg["future_tolerance_sec"])
    if args.input_mode == "televuer" and not endpoint:
        raise ValueError("Televuer input endpoint must be non-empty")
    if not np.isfinite(update_hz) or update_hz <= 0:
        raise ValueError("hand update_hz must be positive and finite")
    if not np.isfinite(stale_timeout) or stale_timeout <= 0:
        raise ValueError("Televuer stale_timeout_sec must be positive and finite")
    if not np.isfinite(future_tolerance) or future_tolerance < 0:
        raise ValueError("Televuer future_tolerance_sec must be finite and non-negative")
    if config["control"]["stale_command_pose"] not in {"open", "hold"}:
        raise ValueError("stale_command_pose must be 'open' or 'hold'")
    command_speed = float(config["control"]["command_speed"])
    if not np.isfinite(command_speed) or not 0.0 <= command_speed <= 1.0:
        raise ValueError("command_speed must be finite and within [0, 1]")

    surface_client = None
    if args.surface_query_endpoint:
        from boost.control.hand_mppi.surface.mujoco_client import MujocoSurfaceClient

        surface_cfg = config["retargeter"].get("contact_guidance", {})
        surface_timeout = float(surface_cfg.get("query_timeout_sec", 0.05))
        surface_client = MujocoSurfaceClient(
            args.surface_query_endpoint,
            timeout_sec=surface_timeout,
        )
        print(
            f"[hand-mppi] contact guidance enabled via MuJoCo surface query "
            f"{args.surface_query_endpoint} (timeout={surface_timeout * 1000:.0f}ms)",
            flush=True,
        )

    if args.input_mode == "televuer":
        print(
            f"[hand-mppi] subscribing {endpoint}/{input_cfg['topic']}", flush=True
        )
    else:
        print(f"[hand-mppi] reading editable override {override_file}", flush=True)
    print(f"[hand-mppi] DDS publish={'enabled' if args.publish else 'disabled'}", flush=True)
    print(
        "[hand-mppi] constructing Revo2 retargeter(s); "
        "the first JAX compile can take a while",
        flush=True,
    )
    retargeter_start = time.monotonic()
    runtimes: dict[str, SideRuntime] = {}
    for side in sides:
        side_start = time.monotonic()
        print(
            f"[hand-mppi] initializing {side} Revo2 retargeter (MJX/JAX warmup)",
            flush=True,
        )
        retargeter, jax_numpy, q_min, q_max, max_velocity = _create_retargeter(
            side,
            config,
            hand_config,
            device=args.device,
            surface_client=surface_client,
        )
        runtimes[side] = SideRuntime(
            side=side,
            retargeter=retargeter,
            jax_numpy=jax_numpy,
            q_min=q_min,
            q_max=q_max,
            max_velocity=max_velocity,
        )
        print(
            f"[hand-mppi] {side} retargeter ready in "
            f"{time.monotonic() - side_start:.1f}s",
            flush=True,
        )

    print(
        f"[hand-mppi] all retargeters ready in "
        f"{time.monotonic() - retargeter_start:.1f}s; initializing DDS",
        flush=True,
    )

    adapter = BrainCoRevo2Adapter.from_mapping(hand_config)
    dds = (
        BrainCoDDS(domain_id=args.domain_id, interface=args.interface)
        if args.publish
        else None
    )
    if dds is not None:
        print(
            f"[hand-mppi] publishing {args.side} targets on "
            f"{args.interface}/DDS domain {args.domain_id}; waiting for fresh Revo2 state",
            flush=True,
        )

    context = None
    subscriber = None
    poller = None
    if args.input_mode == "televuer":
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.CONFLATE, 1)
        subscriber.setsockopt(zmq.RCVHWM, 1)
        subscriber.setsockopt(zmq.SUBSCRIBE, str(input_cfg["topic"]).encode("utf-8"))
        subscriber.connect(endpoint)
        poller = zmq.Poller()
        poller.register(subscriber, zmq.POLLIN)

    update_period = 1.0 / update_hz
    joint_count = int(input_cfg["joint_count"])
    log_period = 1.0 / max(0.1, float(config["control"].get("log_hz", 2.0)))
    stop = False
    override_sequence = 0
    next_override_read = time.monotonic()

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    next_tick = time.monotonic()
    next_log = next_tick + log_period
    rate_window_start = next_tick
    control_ticks = 0
    publish_counts = {side: 0 for side in SIDES}
    retarget_ms_samples: dict[str, list[float]] = {side: [] for side in SIDES}
    surface_query_ms_samples: dict[str, list[float]] = {side: [] for side in SIDES}
    surface_query_status_samples: dict[str, list[str]] = {side: [] for side in SIDES}
    last_input_error = ""

    def timing_summary(samples: list[float]) -> str:
        if not samples:
            return "n/a"
        values = np.asarray(samples, dtype=np.float64)
        return f"{float(np.mean(values)):.1f}/{float(np.percentile(values, 95)):.1f}ms"

    def query_status_summary(samples: list[str]) -> str:
        ok = sum(status == "ok" for status in samples)
        timeout = sum(status == "timeout" for status in samples)
        other = len(samples) - ok - timeout
        return f"ok:{ok} timeout:{timeout} other:{other}"

    try:
        while not stop:
            frame = None
            if args.input_mode == "televuer":
                events = dict(poller.poll(timeout=max(0, int(update_period * 1000))))
                if subscriber in events:
                    raw = subscriber.recv()
                    try:
                        frame = decode_televuer_frame(
                            raw,
                            topic=str(input_cfg["topic"]),
                            expected_schema_version=int(input_cfg["schema_version"]),
                            expected_coordinate_frame=int(input_cfg["coordinate_frame"]),
                            expected_joint_count=joint_count,
                            stale_timeout_sec=stale_timeout,
                            future_tolerance_sec=future_tolerance,
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        message = str(exc)
                        if message != last_input_error:
                            print(
                                f"[hand-mppi] input frame rejected: {message}",
                                flush=True,
                            )
                            last_input_error = message
            else:
                now_for_override = time.monotonic()
                if now_for_override >= next_override_read:
                    next_override_read = now_for_override + update_period
                    try:
                        frame = decode_override_frame(
                            override_file,
                            sequence=override_sequence,
                            expected_joint_count=joint_count,
                        )
                        override_sequence += 1
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        message = str(exc)
                        if message != last_input_error:
                            print(f"[hand-mppi] override pose unavailable: {message}", flush=True)
                            last_input_error = message

            if frame is not None:
                try:
                    last_input_error = ""
                    for side, runtime in runtimes.items():
                        points = frame.hands[side]
                        if points is None or frame.sequence == runtime.last_sequence:
                            continue

                        set_sequence = getattr(runtime.retargeter, "set_input_sequence", None)
                        if callable(set_sequence):
                            set_sequence(frame.sequence)

                        if args.publish and not runtime.state_seeded:
                            state = dds.state(
                                side,
                                timeout_sec=float(config["control"]["state_timeout_sec"]),
                            )
                            if state is None:
                                continue
                            measured_rad = adapter.normalized_to_actuated(state.q).astype(
                                np.float32
                            )
                            # Mimic's public reset() returns to neutral. Seed its
                            # optimizer warm start from the fresh measured pose.
                            runtime.retargeter._qpos_prev = (
                                runtime.jax_numpy.asarray(measured_rad, dtype=np.float32)
                            )
                            runtime.command_rad = measured_rad
                            runtime.output_timestamp = time.monotonic()
                            runtime.state_seeded = True

                        retarget_start = time.perf_counter()
                        result, _ = runtime.retargeter.retarget(points)
                        # Materializing the JAX result synchronizes device work,
                        # so this includes actual optimizer completion latency.
                        target_rad = np.asarray(result, dtype=np.float32).reshape(-1)
                        retarget_ms_samples[side].append(
                            (time.perf_counter() - retarget_start) * 1000.0
                        )
                        if surface_client is not None:
                            surface_query_ms_samples[side].append(
                                float(surface_client.last_query_duration_ms)
                            )
                            surface_query_status_samples[side].append(
                                str(surface_client.last_query_status)
                            )
                        if target_rad.shape != (6,) or not np.isfinite(target_rad).all():
                            raise ValueError(
                                f"{side} retargeter returned an invalid six-joint target"
                            )
                        runtime.desired_rad = np.clip(
                            target_rad, runtime.q_min, runtime.q_max
                        )
                        runtime.last_valid_timestamp = frame.timestamp_monotonic
                        runtime.last_sequence = frame.sequence
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    message = str(exc)
                    if message != last_input_error:
                        print(
                            f"[hand-mppi] waiting for valid Televuer frame: {message}",
                            flush=True,
                        )
                        last_input_error = message

            now = time.monotonic()
            if now < next_tick:
                continue
            next_tick = now + update_period
            control_ticks += 1

            status_parts: list[str] = []
            for side, runtime in runtimes.items():
                if runtime.last_valid_timestamp is None:
                    status_parts.append(f"{side}:waiting-hand")
                    continue

                age = max(0.0, now - runtime.last_valid_timestamp)
                desired_rad = (
                    runtime.q_min.copy()
                    if age > stale_timeout
                    and config["control"]["stale_command_pose"] == "open"
                    else runtime.desired_rad
                )
                if desired_rad is None:
                    status_parts.append(f"{side}:waiting-target")
                    continue

                if runtime.command_rad is None:
                    runtime.command_rad = np.clip(desired_rad, runtime.q_min, runtime.q_max)
                    runtime.output_timestamp = now
                elapsed = max(0.0, now - (runtime.output_timestamp or now))
                max_step = runtime.max_velocity * elapsed
                runtime.command_rad = runtime.command_rad + np.clip(
                    desired_rad - runtime.command_rad, -max_step, max_step
                )
                runtime.output_timestamp = now
                normalized = adapter.actuated_to_normalized(runtime.command_rad)

                state = None
                if dds is not None:
                    state = dds.state(
                        side,
                        timeout_sec=float(config["control"]["state_timeout_sec"]),
                    )
                    if state is None:
                        status_parts.append(f"{side}:waiting-fresh-DDS-state")
                        continue
                    dds.publish(side, normalized.tolist(), speed=command_speed)
                    publish_counts[side] += 1

                target_text = ",".join(f"{value:.2f}" for value in normalized)
                contact_text = ""
                contact_debug = getattr(runtime.retargeter, "contact_debug", None)
                if surface_client is not None and isinstance(contact_debug, dict):
                    alpha = np.asarray(contact_debug.get("alpha", []), dtype=np.float32)
                    distances = np.asarray(
                        contact_debug.get("distance", []), dtype=np.float32
                    )
                    if alpha.shape == (5,) and distances.shape == (5,):
                        alpha_text = ",".join(f"{value:.2f}" for value in alpha)
                        distance_text = ",".join(
                            "--" if not np.isfinite(value) else f"{value * 1000:.0f}"
                            for value in distances
                        )
                        contact_text = f" contact-alpha=[{alpha_text}] surf-mm=[{distance_text}]"
                if state is None:
                    status_parts.append(
                        f"{side}:age={age:.2f}s q=[{target_text}] read-only{contact_text}"
                    )
                else:
                    measured_text = ",".join(f"{value:.2f}" for value in state.q)
                    status_parts.append(
                        f"{side}:age={age:.2f}s target=[{target_text}] "
                        f"state=[{measured_text}]{contact_text}"
                    )

            if now >= next_log:
                elapsed = max(now - rate_window_start, 1e-6)
                control_hz = control_ticks / elapsed
                left_hz = publish_counts["left"] / elapsed
                right_hz = publish_counts["right"] / elapsed
                retarget_timing = ",".join(
                    f"{side[0].upper()}={len(retarget_ms_samples[side]) / elapsed:.1f}Hz "
                    f"{timing_summary(retarget_ms_samples[side])}"
                    for side in SIDES
                )
                surface_timing = (
                    ",".join(
                        f"{side[0].upper()}={len(surface_query_ms_samples[side]) / elapsed:.1f}Hz "
                        f"{timing_summary(surface_query_ms_samples[side])}"
                        f" {query_status_summary(surface_query_status_samples[side])}"
                        for side in SIDES
                    )
                    if surface_client is not None
                    else "disabled"
                )
                print(
                    f"[hand-mppi] rate={control_hz:.1f}/{update_hz:.0f}Hz "
                    f"publish(L/R)={left_hz:.1f}/{right_hz:.1f}Hz | "
                    f"retarget(L/R Hz,avg/p95 ms)=[{retarget_timing}] "
                    f"surface-query(L/R Hz,avg/p95 ms)=[{surface_timing}] | "
                    + " | ".join(status_parts),
                    flush=True,
                )
                next_log = now + log_period
                rate_window_start = now
                control_ticks = 0
                publish_counts = {side: 0 for side in SIDES}
                retarget_ms_samples = {side: [] for side in SIDES}
                surface_query_ms_samples = {side: [] for side in SIDES}
                surface_query_status_samples = {side: [] for side in SIDES}

    finally:
        if subscriber is not None:
            subscriber.close(linger=0)
        if context is not None:
            context.term()
        if surface_client is not None:
            surface_client.close()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
