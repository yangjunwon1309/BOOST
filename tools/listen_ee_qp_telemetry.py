#!/usr/bin/env python3
"""Print short-window EE-QP telemetry summaries from g1_debug."""

from __future__ import annotations

import argparse
from collections import Counter
import math
import time
from typing import Any

import msgpack
import zmq


ACTIVE_JOINTS: list[tuple[str, int]] = [
    ("waist_yaw", 12),
    ("waist_roll", 13),
    ("waist_pitch", 14),
    ("l_sh_pitch", 15),
    ("l_sh_roll", 16),
    ("l_sh_yaw", 17),
    ("l_elbow", 18),
    ("l_wr_roll", 19),
    ("l_wr_pitch", 20),
    ("l_wr_yaw", 21),
    ("r_sh_pitch", 22),
    ("r_sh_roll", 23),
    ("r_sh_yaw", 24),
    ("r_elbow", 25),
    ("r_wr_roll", 26),
    ("r_wr_pitch", 27),
    ("r_wr_yaw", 28),
]
ARM_JOINTS: list[tuple[str, int]] = ACTIVE_JOINTS[3:]
ACTIVE_INDICES = [index for _, index in ACTIVE_JOINTS]
ARM_INDICES = [index for _, index in ARM_JOINTS]
WAIST_INDICES = [12, 13, 14]
LOWER_INDICES = list(range(12))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def numeric(samples: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def ratio(samples: list[dict[str, Any]], key: str) -> float:
    if not samples:
        return math.nan
    return sum(bool(sample.get(key, False)) for sample in samples) / len(samples)


def as_float_list(value: Any, expected: int | None = None) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if expected is not None and len(value) < expected:
        return None
    output: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            return None
        output.append(float(item))
    return output


def vector_norm(value: Any) -> float:
    values = as_float_list(value)
    if values is None:
        return math.nan
    return math.sqrt(sum(item * item for item in values))


def vector_norms(samples: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for sample in samples:
        norm = vector_norm(sample.get(key))
        if math.isfinite(norm):
            values.append(norm)
    return values


def vector_slice_stats(
    samples: list[dict[str, Any]], key: str, indices: list[int]
) -> tuple[float, float]:
    rmses: list[float] = []
    maxes: list[float] = []
    for sample in samples:
        values = as_float_list(sample.get(key))
        if values is None or len(values) <= max(indices):
            continue
        selected = [values[index] for index in indices]
        rmses.append(math.sqrt(sum(value * value for value in selected) / len(selected)))
        maxes.append(max(abs(value) for value in selected))
    return mean(rmses), max(maxes, default=math.nan)


def normalize_quat_wxyz(q: Any) -> tuple[float, float, float, float] | None:
    values = as_float_list(q, 4)
    if values is None:
        return None
    w, x, y, z = values[:4]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return None
    return w / norm, x / norm, y / norm, z / norm


def quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return w, -x, -y, -z


def quat_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_angle_rad(q: tuple[float, float, float, float]) -> float:
    w = max(-1.0, min(1.0, abs(q[0])))
    return 2.0 * math.acos(w)


def base_tilt_rad(q_value: Any) -> float:
    q = normalize_quat_wxyz(q_value)
    if q is None:
        return math.nan
    _, x, y, z = q
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    body_z_world_z = max(-1.0, min(1.0, body_z_world_z))
    return math.acos(body_z_world_z)


def base_quat_error_rad(sample: dict[str, Any]) -> float:
    target = normalize_quat_wxyz(sample.get("base_quat_target"))
    measured = normalize_quat_wxyz(
        sample.get("base_quat_measured", sample.get("base_quat"))
    )
    if target is None or measured is None:
        return math.nan
    return quat_angle_rad(quat_mul(quat_conjugate(target), measured))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def mask_top_text(
    samples: list[dict[str, Any]],
    exact_key: str,
    estimate_key: str,
    top_n: int,
) -> str:
    key = exact_key
    source = ""
    if not any(as_float_list(sample.get(key), len(ACTIVE_JOINTS)) for sample in samples):
        key = estimate_key
        source = "~"
    totals = [0.0] * len(ACTIVE_JOINTS)
    count = 0
    for sample in samples:
        values = as_float_list(sample.get(key), len(ACTIVE_JOINTS))
        if values is None:
            continue
        for i in range(len(ACTIVE_JOINTS)):
            totals[i] += 1.0 if values[i] > 0.5 else 0.0
        count += 1
    if count == 0:
        return "n/a"
    ranked = sorted(
        ((totals[i] / count, ACTIVE_JOINTS[i][0]) for i in range(len(ACTIVE_JOINTS))),
        reverse=True,
    )
    active = [(fraction, name) for fraction, name in ranked if fraction > 0.0]
    if not active:
        return "none"
    return ",".join(
        f"{source}{name}:{100.0 * fraction:.0f}%"
        for fraction, name in active[:top_n]
    )


def augment_sample(
    sample: dict[str, Any],
    previous_active_delta: list[float] | None,
    max_delta_rad: list[float],
    max_rate_step_rad: list[float],
    saturation_tolerance_rad: float,
) -> list[float] | None:
    delta_q = as_float_list(sample.get("ee_qp_delta_q"), max(ACTIVE_INDICES) + 1)
    if delta_q is None:
        return previous_active_delta
    active_delta = [delta_q[index] for index in ACTIVE_INDICES]
    if "ee_qp_correction_bound_saturation_mask" not in sample:
        sample["ee_qp_correction_bound_saturation_mask_est"] = [
            1.0
            if abs(value) >= limit - saturation_tolerance_rad
            else 0.0
            for value, limit in zip(active_delta, max_delta_rad)
        ]
    if previous_active_delta is not None and "ee_qp_rate_bound_saturation_mask" not in sample:
        sample["ee_qp_rate_bound_saturation_mask_est"] = [
            1.0
            if abs(value - previous) >= limit - saturation_tolerance_rad
            else 0.0
            for value, previous, limit in zip(
                active_delta, previous_active_delta, max_rate_step_rad
            )
        ]
    return active_delta


def summarize(samples: list[dict[str, Any]], elapsed: float, top_joints: int) -> str:
    if not samples:
        return f"[{elapsed:5.1f}s] no g1_debug messages"

    reference_age = numeric(samples, "ee_qp_reference_age_ms")
    pd_rmse = numeric(samples, "ee_qp_pd_tracking_lag_rmse_rad")
    pd_max = numeric(samples, "ee_qp_pd_tracking_lag_max_abs_rad")
    correction_sat = numeric(
        samples, "ee_qp_correction_bound_saturation_fraction"
    )
    rate_sat = numeric(samples, "ee_qp_rate_bound_saturation_fraction")
    solve = numeric(samples, "ee_qp_solve_time_ms")
    alpha = [
        value
        for value in numeric(samples, "ee_qp_backtracking_alpha")
        if value > 0.0
    ]
    fallback_reasons = Counter(
        str(sample.get("ee_qp_fallback_reason", "unknown"))
        for sample in samples
        if sample.get("ee_qp_fallback_active", False)
    )
    reason_text = ",".join(
        f"{reason}:{count}" for reason, count in fallback_reasons.most_common()
    ) or "none"
    mode_counts = Counter(
        str(sample.get("ee_qp_control_mode", "unknown")) for sample in samples
    )
    source_counts = Counter(
        str(sample.get("ee_qp_reference_source", "unknown")) for sample in samples
    )
    mode_text = mode_counts.most_common(1)[0][0] if mode_counts else "unknown"
    source_text = source_counts.most_common(1)[0][0] if source_counts else "unknown"
    plan_ids = [
        int(value)
        for value in numeric(samples, "ee_qp_plan_id")
        if math.isfinite(value)
    ]
    plan_phase = numeric(samples, "ee_qp_plan_phase")

    def active(values: list[float]) -> float:
        return sum(value > 0.0 for value in values) / len(values) \
            if values else math.nan

    waist_rmse, waist_max = vector_slice_stats(
        samples, "ee_qp_joint_tracking_error", WAIST_INDICES
    )
    lower_rmse, lower_max = vector_slice_stats(
        samples, "ee_qp_joint_tracking_error", LOWER_INDICES
    )
    base_ang = vector_norms(samples, "base_ang_vel")
    torso_ang = vector_norms(samples, "body_torso_ang_vel")
    base_tilt = [
        value for value in (base_tilt_rad(sample.get("base_quat")) for sample in samples)
        if math.isfinite(value)
    ]
    base_qerr = [
        value for value in (base_quat_error_rad(sample) for sample in samples)
        if math.isfinite(value)
    ]
    arm_err = numeric(samples, "ee_qp_arm_only_error_before_m")
    waist_err = numeric(samples, "ee_qp_waist_only_error_before_m")
    lower_err = numeric(samples, "ee_qp_lower_only_error_before_m")
    upper_err = numeric(samples, "ee_qp_upper_only_error_before_m")

    return (
        f"[{elapsed:5.1f}s] n={len(samples):3d} "
        f"mode={mode_text}/{source_text} plan={max(plan_ids, default=0)} "
        f"phase={mean(plan_phase):.2f} "
        f"ref_ok={100.0 * ratio(samples, 'ee_qp_reference_feasible'):4.1f}% | "
        f"ref_age={mean(reference_age):5.1f}/{max(reference_age, default=math.nan):5.1f}ms(avg/max) | "
        f"pd_lag={mean(pd_rmse):.4f}/{max(pd_max, default=math.nan):.4f}rad(rmse/max) | "
        f"corr_sat={100.0 * mean(correction_sat):4.1f}%j,{100.0 * active(correction_sat):4.1f}%f "
        f"rate_sat={100.0 * mean(rate_sat):4.1f}%j,{100.0 * active(rate_sat):4.1f}%f | "
        f"corr_top={mask_top_text(samples, 'ee_qp_correction_bound_saturation_mask', 'ee_qp_correction_bound_saturation_mask_est', top_joints)} "
        f"rate_top={mask_top_text(samples, 'ee_qp_rate_bound_saturation_mask', 'ee_qp_rate_bound_saturation_mask_est', top_joints)} | "
        f"waist_lag={waist_rmse:.4f}/{waist_max:.4f}rad lower_lag={lower_rmse:.4f}/{lower_max:.4f}rad | "
        f"err_split={100.0 * mean(arm_err):.1f}/{100.0 * mean(waist_err):.1f}/"
        f"{100.0 * mean(lower_err):.1f}/{100.0 * mean(upper_err):.1f}cm"
        f"(arm/waist/lower/upper) | "
        f"base=tilt {mean(base_tilt):.3f}/{max(base_tilt, default=math.nan):.3f}rad "
        f"ang {mean(base_ang):.3f}/{max(base_ang, default=math.nan):.3f}rad/s "
        f"qerr {mean(base_qerr):.3f}/{max(base_qerr, default=math.nan):.3f}rad "
        f"torso_ang {mean(torso_ang):.3f}/{max(torso_ang, default=math.nan):.3f}rad/s | "
        f"reject={100.0 * ratio(samples, 'ee_qp_candidate_rejected'):4.1f}% "
        f"fallback={100.0 * ratio(samples, 'ee_qp_fallback_active'):4.1f}% "
        f"valid={100.0 * ratio(samples, 'ee_qp_valid'):4.1f}% "
        f"reasons={reason_text} | "
        f"alpha={mean(alpha):.3f}/{min(alpha, default=math.nan):.3f}(avg/min) "
        f"solve={mean(solve):.3f}/{percentile(solve, 0.95):.3f}ms(avg/p95)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--topic", default="g1_debug")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--top-joints", type=int, default=3)
    parser.add_argument("--waist-max-delta-rad", type=float, default=0.02)
    parser.add_argument("--arm-max-delta-rad", type=float, default=0.05)
    parser.add_argument("--waist-max-rate-step-rad", type=float, default=0.004)
    parser.add_argument("--arm-max-rate-step-rad", type=float, default=0.01)
    parser.add_argument("--saturation-tolerance-rad", type=float, default=1e-4)
    args = parser.parse_args()
    if args.interval <= 0.0:
        parser.error("--interval must be positive")

    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    socket.connect(args.endpoint)

    topic = args.topic.encode()
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    next_report = started + args.interval
    previous_arm_delta: list[float] | None = None
    max_delta_rad = [args.waist_max_delta_rad] * 3 + [args.arm_max_delta_rad] * 14
    max_rate_step_rad = (
        [args.waist_max_rate_step_rad] * 3 + [args.arm_max_rate_step_rad] * 14
    )
    print(f"[EE-QP listener] {args.endpoint}/{args.topic}, interval={args.interval:.2f}s")

    try:
        while True:
            timeout_ms = max(
                0, math.ceil((next_report - time.monotonic()) * 1000.0)
            )
            if socket.poll(timeout_ms, zmq.POLLIN):
                raw = socket.recv()
                if raw.startswith(topic):
                    try:
                        message = msgpack.unpackb(raw[len(topic) :], raw=False)
                    except (ValueError, msgpack.UnpackException):
                        continue
                    if isinstance(message, dict):
                        previous_arm_delta = augment_sample(
                            message,
                            previous_arm_delta,
                            max_delta_rad,
                            max_rate_step_rad,
                            args.saturation_tolerance_rad,
                        )
                        samples.append(message)

            now = time.monotonic()
            if now >= next_report:
                print(summarize(samples, now - started, args.top_joints), flush=True)
                samples.clear()
                while next_report <= now:
                    next_report += args.interval
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()


if __name__ == "__main__":
    main()
