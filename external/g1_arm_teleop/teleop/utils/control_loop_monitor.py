"""Control-loop observations without clocks, threads, DDS or robot imports."""

from copy import deepcopy
from collections.abc import Mapping
import math
from operator import index


def _nanoseconds(value):
    stamp = index(value)
    if isinstance(value, bool) or stamp < 0:
        raise ValueError("monotonic timestamps must be nonnegative integer nanoseconds")
    return stamp


def _received_ns(snapshot, name="received_monotonic"):
    exact = snapshot.get(f"{name}_ns")
    if exact is not None:
        try:
            return _nanoseconds(exact)
        except (TypeError, ValueError, OverflowError):
            return None
    seconds = _finite_number(snapshot.get(name))
    if seconds is None or seconds < 0:
        return None
    try:
        return int(round(seconds * 1_000_000_000))
    except (ValueError, OverflowError):
        return None


def _finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json_safe(value):
    """Copy ordinary snapshots, including optional NumPy arrays, without NumPy."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _age_ms(now_ns, stamp):
    # Preserve a signed value if a snapshot was sampled after action generation.
    return (now_ns - stamp) / 1_000_000 if stamp is not None and stamp > 0 else None


class ControlLoopMonitor:
    """Observe generated actions separately from asynchronous motor publishes.

    begin_cycle runs even when XR is unavailable. action_metadata runs only for
    an action actually generated, so action_interval_ms retains skipped-cycle
    gaps. The caller supplies all monotonic times; this class never waits or
    produces control commands. Use it from the main control-loop thread.
    """

    def __init__(self, frequency):
        frequency = float(frequency)
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be finite and positive")
        self.period_ms = 1000.0 / frequency
        self._ever_action = False
        self.resume()

    @property
    def has_generated_action(self):
        """Whether live input has produced any action in this monitor's lifetime."""
        return self._ever_action

    def resume(self):
        """Start a timing segment after an explicit operator pause.

        Call only at initial tracking start or park/resume, never for XR loss.
        Previous activity stays remembered: missing input after a later resume
        is still a tracking interruption rather than first-session setup.
        """
        self._last_cycle_ns = None
        self._main_interval_ms = None
        self._last_action_ns = None
        self._missing_xr = 0
        self._initial_wait = 0
        self._warning_times = {}

    def begin_cycle(self, now_ns):
        now_ns = _nanoseconds(now_ns)
        if self._last_cycle_ns is not None:
            if now_ns < self._last_cycle_ns:
                raise ValueError("cycle timestamps must not move backwards")
            self._main_interval_ms = (now_ns - self._last_cycle_ns) / 1_000_000
        self._last_cycle_ns = now_ns

    def missing_xr(self):
        if self._ever_action:
            self._missing_xr += 1
        else:
            self._initial_wait += 1

    def action_metadata(self, ik_start_ns, ik_end_ns, xr_received_ns,
                        state_snapshot=None, arm_command=None,
                        requested_command_sequence=None):
        ik_start_ns, ik_end_ns = _nanoseconds(ik_start_ns), _nanoseconds(ik_end_ns)
        xr_received_ns = _nanoseconds(xr_received_ns)
        if ik_end_ns < ik_start_ns:
            raise ValueError("IK end must not precede IK start")
        if self._last_action_ns is not None and ik_end_ns < self._last_action_ns:
            raise ValueError("action timestamps must not move backwards")
        state_snapshot = {} if state_snapshot is None else state_snapshot
        state_stamp = _received_ns(state_snapshot)
        state_sequence = state_snapshot.get("sequence")
        state_metadata = {
            "sequence": int(state_sequence) if state_sequence is not None else None,
            "received_monotonic_ns": state_stamp,
            "age_ms": _age_ms(ik_end_ns, state_stamp),
        }
        for field in ("motor_temperature", "motor_status", "tau_est"):
            if field in state_snapshot:
                state_metadata[field] = _json_safe(deepcopy(state_snapshot[field]))
        metadata = {
            "main_interval_ms": self._main_interval_ms,
            "action_interval_ms": ((ik_end_ns - self._last_action_ns) / 1_000_000
                                   if self._last_action_ns is not None else None),
            "ik_ms": (ik_end_ns - ik_start_ns) / 1_000_000,
            "xr_age_ms": _age_ms(ik_end_ns, xr_received_ns),
            "xr_missing_cycles": self._missing_xr,
            # The caller can announce this once at INFO when the XR session
            # first becomes ready; it is not an interruption of active input.
            "xr_initial_wait_cycles": self._initial_wait,
            "xr_received_monotonic_ns": xr_received_ns,
            "state": state_metadata,
            # This observed motor publish may precede the current action or
            # carry an older command sequence. Never relabel it as this action.
            "arm_command": _json_safe(deepcopy(arm_command)),
            "requested_command_sequence": (int(requested_command_sequence)
                                           if requested_command_sequence is not None else None),
        }
        self._ever_action = True
        self._last_action_ns = ik_end_ns
        self._missing_xr = 0
        self._initial_wait = 0
        return metadata

    def warning(self, telemetry, now_ns):
        """Return observed timing problems, rate-limited per reason to 2 seconds.

        A large interval does not identify its cause. An older motor command
        sequence is normal when sampling a separate publisher and is not itself
        a warning. Cumulative maximum intervals are not current loop intervals.
        """
        try:
            now_ns = _nanoseconds(now_ns)
        except (TypeError, ValueError, OverflowError):
            return None
        if not isinstance(telemetry, Mapping):
            return None
        observations = []

        def exceeds(reason, name, value, threshold):
            value = _finite_number(value)
            if value is not None and value > threshold:
                observations.append((reason, f"{name}={value:.1f}ms"))

        exceeds("main", "main interval", telemetry.get("main_interval_ms"), 1.5 * self.period_ms)
        exceeds("action", "action interval", telemetry.get("action_interval_ms"), 1.5 * self.period_ms)
        exceeds("ik", "IK", telemetry.get("ik_ms"), self.period_ms)
        exceeds("xr_age", "XR age", telemetry.get("xr_age_ms"), max(100.0, 3 * self.period_ms))
        missing = _finite_number(telemetry.get("xr_missing_cycles", 0))
        if missing is not None and missing > 0:
            observations.append(("xr_missing", f"XR unavailable for {missing:.0f} cycle(s) before action"))
        state = telemetry.get("state")
        state = state if isinstance(state, Mapping) else {}
        exceeds("state_age", "LowState age", state.get("age_ms"), max(20.0, self.period_ms))
        command = telemetry.get("arm_command")
        command = command if isinstance(command, Mapping) else {}
        exceeds("writer", "motor writer interval", command.get("writer_interval_ms"),
                max(12.0, self.period_ms / 2))
        publish_stamp = _received_ns(command, "publish_monotonic")
        exceeds("publish_age", "motor publish age", _age_ms(now_ns, publish_stamp),
                max(20.0, self.period_ms))
        if command.get("published") is False:
            observations.append(("publish_failed", "motor publish reported failure"))
        messages = []
        for reason, message in observations:
            previous = self._warning_times.get(reason)
            if previous is None or now_ns - previous >= 2_000_000_000:
                self._warning_times[reason] = now_ns
                messages.append(message)
        return "Control timing: " + "; ".join(messages) if messages else None
