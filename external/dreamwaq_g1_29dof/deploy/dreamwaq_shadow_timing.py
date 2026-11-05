"""Timing analysis and durable reports for read-only G1 shadow runs."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from cpu_affinity import pin_current_thread


SAMPLE_COLUMNS = (
    "sequence",
    "source_tick",
    "source_tick_delta",
    "source_sequence",
    "state_sample_repeated",
    "locomotion_mode",
    "command_vx",
    "command_vy",
    "command_yaw",
    "accepted",
    "response_sequence",
    "response_sequence_lag",
    "skipped_state_count",
    "tick_started_ns",
    "state_received_ns",
    "request_published_ns",
    "inference_started_ns",
    "inference_finished_ns",
    "response_received_ns",
    "state_age_ms",
    "parent_prepare_ms",
    "ipc_wakeup_ms",
    "worker_compute_ms",
    "backend_inference_ms",
    "worker_overhead_ms",
    "response_transfer_ms",
    "request_roundtrip_ms",
    "state_to_response_ms",
    "loop_work_ms",
    "tick_period_ms",
    "tick_lateness_ms",
    "deadline_margin_ms",
    "shadow_writer_handoff_ms",
    "shadow_writer_callback_ms",
    "shadow_writer_ready_ms",
    "state_to_shadow_writer_ms",
    "state_to_shadow_writer_ready_ms",
    "shadow_writer_tick_lateness_ms",
    "shadow_writer_tick_ns",
    "shadow_writer_capture_ns",
    "shadow_writer_ready_ns",
    "mode",
    "max_action",
    "target_delta",
)


class ShadowWriterCadenceProbe:
    """Measure the 250-Hz parent-thread handoff without constructing LowCmd.

    This cannot measure LowCmd DDS serialization/network time. With a callback,
    it does reproduce and measure the read-only arm/command-composition work as
    well as GIL and scheduler stalls before the next writer opportunity.
    """

    def __init__(self, rate_hz: float, callback=None, cpu=None) -> None:
        rate_hz = float(rate_hz)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("shadow writer probe rate must be finite and positive")
        self._period_ns = int(round(1.0e9 / rate_hz))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._last_sequence = -1
        self._results: dict[int, dict[str, float]] = {}
        self._callback = callback
        self._fault: str | None = None
        self._cpu = None if cpu is None else int(cpu)
        self.cpu_affinity = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="dreamwaq-shadow-writer-probe",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=1.0):
            self._stop.set()
            raise RuntimeError("shadow writer probe did not initialize CPU affinity")
        if self._fault is not None:
            raise RuntimeError("shadow writer probe CPU affinity failed: %s" % self._fault)

    def submit(
        self,
        sequence: int,
        state_received_ns: int,
        response_received_ns: int,
    ) -> None:
        with self._lock:
            self._latest = (
                int(sequence),
                int(state_received_ns),
                int(response_received_ns),
            )

    def _loop(self) -> None:
        try:
            self.cpu_affinity = pin_current_thread(self._cpu)
        except Exception as exc:
            self._fault = "could not pin shadow writer probe to CPU %s: %s" % (
                self._cpu,
                exc,
            )
            self._ready.set()
            return
        self._ready.set()
        next_deadline_ns = time.monotonic_ns()
        while not self._stop.is_set():
            tick_ns = time.monotonic_ns()
            lateness_ms = max(0.0, (tick_ns - next_deadline_ns) / 1.0e6)
            with self._lock:
                latest = self._latest
                callback_started_ns = time.monotonic_ns()
            try:
                if self._callback is not None:
                    self._callback()
            except Exception as exc:
                with self._lock:
                    self._fault = "%s: %s" % (type(exc).__name__, exc)
                self._stop.set()
                return
            callback_finished_ns = time.monotonic_ns()
            if latest is not None and latest[0] != self._last_sequence:
                sequence, state_ns, response_ns = latest
                with self._lock:
                    self._results[sequence] = {
                        "shadow_writer_handoff_ms": (
                            callback_started_ns - response_ns
                        ) / 1.0e6,
                        "shadow_writer_callback_ms": (
                            callback_finished_ns - callback_started_ns
                        ) / 1.0e6,
                        "shadow_writer_ready_ms": (
                            callback_finished_ns - response_ns
                        ) / 1.0e6,
                        "state_to_shadow_writer_ms": (tick_ns - state_ns) / 1.0e6,
                        "state_to_shadow_writer_ready_ms": (
                            callback_finished_ns - state_ns
                        ) / 1.0e6,
                        "shadow_writer_tick_lateness_ms": lateness_ms,
                        "shadow_writer_tick_ns": tick_ns,
                        "shadow_writer_capture_ns": callback_started_ns,
                        "shadow_writer_ready_ns": callback_finished_ns,
                    }
                    self._last_sequence = sequence
            next_deadline_ns += self._period_ns
            remaining_ns = next_deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                self._stop.wait(remaining_ns / 1.0e9)
            else:
                next_deadline_ns = time.monotonic_ns()

    def stop(self) -> None:
        # Allow one final probe slot to observe the last submitted response.
        self._stop.wait(self._period_ns / 1.0e9)
        self._stop.set()
        self._thread.join(timeout=1.0)

    def results(self) -> dict[int, dict[str, float]]:
        with self._lock:
            return {sequence: dict(values) for sequence, values in self._results.items()}

    @property
    def fault(self) -> str | None:
        with self._lock:
            return self._fault


def _numbers(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = _numbers(rows, key)
    if values.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def build_shadow_summary(
    rows: list[dict[str, Any]],
    *,
    policy_timeout_ms: float,
    control_dt_ms: float,
    max_state_age_ms: float,
    max_tick_jitter_ms: float,
    max_writer_handoff_ms: float,
    expected_lowstate_hz: float,
    reader_diagnostics: dict[str, Any],
    reader_elapsed_s: float,
) -> dict[str, Any]:
    accepted = [row for row in rows if bool(row.get("accepted"))]
    timeouts = len(rows) - len(accepted)
    sequence_errors = sum(
        int(row.get("response_sequence") != row.get("sequence")) for row in accepted
    )
    repeated_policy_samples = sum(bool(row.get("state_sample_repeated")) for row in rows)
    skipped_states = max(
        (int(row.get("skipped_state_count") or 0) for row in accepted), default=0
    )
    loop_overruns = sum(
        float(row.get("loop_work_ms") or 0.0) >= float(control_dt_ms) for row in rows
    )
    negative_segments = 0
    for key in ("parent_prepare_ms", "ipc_wakeup_ms", "worker_compute_ms", "response_transfer_ms"):
        negative_segments += int(np.any(_numbers(accepted, key) < -1.0e-3))

    metrics = {
        key: metric_summary(rows, key)
        for key in (
            "state_age_ms",
            "parent_prepare_ms",
            "ipc_wakeup_ms",
            "worker_compute_ms",
            "backend_inference_ms",
            "worker_overhead_ms",
            "response_transfer_ms",
            "request_roundtrip_ms",
            "state_to_response_ms",
            "loop_work_ms",
            "tick_period_ms",
            "tick_lateness_ms",
            "deadline_margin_ms",
            "shadow_writer_handoff_ms",
            "shadow_writer_callback_ms",
            "shadow_writer_ready_ms",
            "state_to_shadow_writer_ms",
            "state_to_shadow_writer_ready_ms",
            "shadow_writer_tick_lateness_ms",
        )
    }
    unique_rate_hz = (
        float(reader_diagnostics.get("unique_count", 0)) / float(reader_elapsed_s)
        if reader_elapsed_s > 0.0
        else 0.0
    )
    hardware_tick_rate_hz = None
    tick_rows = [
        row
        for row in rows
        if row.get("source_tick") is not None and row.get("state_received_ns") is not None
    ]
    if len(tick_rows) >= 2:
        tick_advance = 0
        for previous, current in zip(tick_rows, tick_rows[1:]):
            tick_advance += (
                int(current["source_tick"]) - int(previous["source_tick"])
            ) & 0xFFFFFFFF
        tick_elapsed_s = (
            int(tick_rows[-1]["state_received_ns"])
            - int(tick_rows[0]["state_received_ns"])
        ) / 1.0e9
        if tick_elapsed_s > 0.0:
            hardware_tick_rate_hz = float(tick_advance) / tick_elapsed_s
    writer_probe_count = metrics["shadow_writer_handoff_ms"]["count"]

    errors = []
    warnings = []
    if not rows:
        errors.append("no measured shadow samples")
    if timeouts:
        errors.append("%d current-sequence actions missed the %.3f-ms deadline" % (timeouts, policy_timeout_ms))
    if sequence_errors:
        errors.append("%d accepted actions did not match their state sequence" % sequence_errors)
    if skipped_states:
        errors.append("policy worker skipped %d state sequences" % skipped_states)
    if negative_segments:
        errors.append("monotonic timing segments were negative")
    if int(reader_diagnostics.get("replay_count", 0)):
        errors.append("LowState tick replay was detected")
    if writer_probe_count < len(accepted):
        errors.append(
            "shadow writer probe missed %d accepted actions"
            % (len(accepted) - writer_probe_count)
        )
    state_age_p99 = metrics["state_age_ms"]["p99"]
    if state_age_p99 is not None and state_age_p99 > float(max_state_age_ms):
        errors.append(
            "LowState age p99 %.3f ms exceeds %.3f ms"
            % (state_age_p99, max_state_age_ms)
        )

    jitter_p99 = metrics["tick_lateness_ms"]["p99"]
    if jitter_p99 is not None and jitter_p99 > float(max_tick_jitter_ms):
        warnings.append(
            "policy tick lateness p99 %.3f ms exceeds %.3f ms"
            % (jitter_p99, max_tick_jitter_ms)
        )
    roundtrip_p99 = metrics["request_roundtrip_ms"]["p99"]
    if roundtrip_p99 is not None and roundtrip_p99 > 0.8 * float(policy_timeout_ms):
        warnings.append(
            "request roundtrip p99 %.3f ms consumes more than 80%% of the deadline"
            % roundtrip_p99
        )
    if repeated_policy_samples:
        warnings.append(
            "%d policy ticks reused the same LowState sample" % repeated_policy_samples
        )
    if loop_overruns:
        warnings.append("%d policy ticks consumed at least one full control period" % loop_overruns)
    configured_reader_hz = float(reader_diagnostics.get("configured_rate_hz", 0.0))
    if (
        reader_elapsed_s >= 1.0
        and configured_reader_hz > 0.0
        and unique_rate_hz < 0.8 * configured_reader_hz
    ):
        warnings.append(
            "LowState unique read rate %.1f Hz is below 80%% of configured %.1f Hz"
            % (unique_rate_hz, configured_reader_hz)
        )
    if hardware_tick_rate_hz is not None and expected_lowstate_hz > 0.0:
        ratio = hardware_tick_rate_hz / float(expected_lowstate_hz)
        if ratio < 0.5:
            errors.append(
                "LowState hardware tick advances at %.1f Hz, below 50%% of expected %.1f Hz; DDS backlog/stall is likely"
                % (hardware_tick_rate_hz, expected_lowstate_hz)
            )
        elif not 0.75 <= ratio <= 1.25:
            warnings.append(
                "LowState hardware tick advances at %.1f Hz, outside 75-125%% of expected %.1f Hz"
                % (hardware_tick_rate_hz, expected_lowstate_hz)
            )
    writer_handoff_p99 = metrics["shadow_writer_handoff_ms"]["p99"]
    if writer_handoff_p99 is not None and writer_handoff_p99 > float(max_writer_handoff_ms):
        errors.append(
            "shadow writer handoff p99 %.3f ms exceeds %.3f ms"
            % (writer_handoff_p99, max_writer_handoff_ms)
        )
    writer_ready_p99 = metrics["shadow_writer_ready_ms"]["p99"]
    if writer_ready_p99 is not None and writer_ready_p99 > float(max_writer_handoff_ms):
        errors.append(
            "shadow writer response-to-ready p99 %.3f ms exceeds %.3f ms"
            % (writer_ready_p99, max_writer_handoff_ms)
        )

    verdict = "FAIL_DELAY_DETECTED" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS_NO_DELAY")
    return {
        "verdict": verdict,
        "sample_count": len(rows),
        "accepted_count": len(accepted),
        "timeout_count": timeouts,
        "sequence_error_count": sequence_errors,
        "worker_skipped_state_count": skipped_states,
        "repeated_policy_state_count": repeated_policy_samples,
        "loop_overrun_count": loop_overruns,
        "policy_timeout_ms": float(policy_timeout_ms),
        "control_dt_ms": float(control_dt_ms),
        "max_state_age_ms": float(max_state_age_ms),
        "max_tick_jitter_ms": float(max_tick_jitter_ms),
        "max_writer_handoff_ms": float(max_writer_handoff_ms),
        "reader_unique_rate_hz": unique_rate_hz,
        "hardware_tick_rate_hz": hardware_tick_rate_hz,
        "expected_lowstate_hz": float(expected_lowstate_hz),
        "reader": dict(reader_diagnostics),
        "metrics_ms": metrics,
        "errors": errors,
        "warnings": warnings,
    }


def default_report_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("/tmp") / ("dreamwaq_shadow_timing_%s.json" % stamp)


def write_shadow_report(
    report_path: str | Path | None,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    json_path = default_report_path() if report_path is None else Path(report_path).expanduser()
    if json_path.suffix.lower() != ".json":
        json_path = json_path.with_suffix(".json")
    csv_path = json_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["report_json"] = str(json_path)
    payload["samples_csv"] = str(csv_path)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SAMPLE_COLUMNS})
    return json_path, csv_path
