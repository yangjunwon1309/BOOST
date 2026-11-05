"""Sequence-coupled IPC for the isolated DreamWaQ policy process.

The LowCmd writer must remain in the DDS-owning parent process so inference can
never starve its 250-Hz loop.  Unlike HOMIE's latest-value IPC, this channel
preserves the policy contract: an action is accepted only when it names the
exact state request that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Any

import numpy as np


class IPCLockTimeout(RuntimeError):
    """The peer process held an IPC lock past the deadline (scheduling delay)."""


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("%s must contain %d finite values" % (label, size))
    return result


@dataclass(frozen=True)
class PolicyStateRequest:
    sequence: int
    monotonic_ns: int
    request_published_ns: int
    previous_action_sequence: int
    q29: np.ndarray
    dq29: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity_body: np.ndarray
    command: np.ndarray
    previous_lower_action: np.ndarray
    locomotion_mode: int = 0
    policy_epoch: int = 0


@dataclass(frozen=True)
class PolicyActionResponse:
    source_state_sequence: int
    source_state_monotonic_ns: int
    source_state_published_ns: int
    inference_started_ns: int
    inference_finished_ns: int
    skipped_state_count: int
    q_target15: np.ndarray
    normalized_lower_action: np.ndarray
    mode: str
    inference_ms: float
    max_action: float

    @property
    def state_to_action_ms(self) -> float:
        return max(
            0.0,
            float(self.inference_finished_ns - self.source_state_monotonic_ns) / 1.0e6,
        )

    @property
    def request_to_action_ms(self) -> float:
        return max(
            0.0,
            float(self.inference_finished_ns - self.source_state_published_ns) / 1.0e6,
        )


class DreamWaQPolicyIPC:
    """One-outstanding-request shared-memory policy channel.

    Events are wakeups only.  Correctness comes from the integer sequences
    guarded by their corresponding state/action locks, so a stale Event cannot
    make the parent accept a stale action.
    """

    def __init__(
        self,
        ctx: mp.context.BaseContext | None = None,
        *,
        lock_timeout_s: float = 0.010,
        action_dim: int = 12,
    ) -> None:
        ctx = ctx or mp
        self.lock_timeout_s = float(lock_timeout_s)
        if not np.isfinite(self.lock_timeout_s) or self.lock_timeout_s <= 0.0:
            raise ValueError("IPC lock timeout must be finite and positive")
        self.action_dim = int(action_dim)
        if self.action_dim not in (12, 15):
            raise ValueError("DreamWaQ IPC action dimension must be 12 or 15")
        self._state_dim = 69 + self.action_dim
        self._action_dim = 18 + self.action_dim
        self.state = ctx.Array("f", self._state_dim, lock=True)
        self.action = ctx.Array("f", self._action_dim, lock=True)

        self._state_sequence = ctx.Value("q", -1, lock=False)
        self._state_monotonic_ns = ctx.Value("q", 0, lock=False)
        self._state_published_ns = ctx.Value("q", 0, lock=False)
        self._previous_action_sequence = ctx.Value("q", -1, lock=False)
        self._policy_epoch = ctx.Value("q", 0, lock=False)

        self._action_source_sequence = ctx.Value("q", -1, lock=False)
        self._action_source_monotonic_ns = ctx.Value("q", 0, lock=False)
        self._action_source_published_ns = ctx.Value("q", 0, lock=False)
        self._inference_started_ns = ctx.Value("q", 0, lock=False)
        self._inference_finished_ns = ctx.Value("q", 0, lock=False)
        self._skipped_state_count = ctx.Value("q", 0, lock=False)

        self.state_event = ctx.Event()
        self.action_event = ctx.Event()
        self.stop_event = ctx.Event()
        self.ready_event = ctx.Event()

    def _try_acquire(self, lock: Any, timeout_s: float | None = None) -> bool:
        timeout = self.lock_timeout_s if timeout_s is None else float(timeout_s)
        return timeout > 0.0 and bool(lock.acquire(True, timeout))

    def _acquire(self, lock: Any, label: str, timeout_s: float | None = None) -> None:
        if not self._try_acquire(lock, timeout_s):
            raise IPCLockTimeout("timed out acquiring %s IPC lock" % label)

    def request_stop(self) -> None:
        self.stop_event.set()
        self.state_event.set()
        self.action_event.set()

    def publish_state(
        self,
        q29: Any,
        dq29: Any,
        quaternion_wxyz: Any,
        angular_velocity_body: Any,
        command: Any,
        *,
        sequence: int,
        monotonic_ns: int,
        previous_lower_action: Any,
        previous_action_sequence: int,
        locomotion_mode: int = 0,
        policy_epoch: int = 0,
    ) -> int:
        sequence = int(sequence)
        monotonic_ns = int(monotonic_ns)
        if sequence < 0:
            raise ValueError("policy state sequence must be non-negative")
        if int(policy_epoch) < 0:
            raise ValueError("policy epoch must be non-negative")
        if monotonic_ns <= 0:
            raise ValueError("policy state timestamp must be positive")

        q = _vector(q29, 29, "q29")
        dq = _vector(dq29, 29, "dq29")
        quat = _vector(quaternion_wxyz, 4, "quaternion")
        gyro = _vector(angular_velocity_body, 3, "angular velocity")
        cmd = _vector(command, 3, "command")
        if locomotion_mode not in (0, 1):
            raise ValueError("locomotion mode must be standing=0 or walking=1")
        previous = _vector(previous_lower_action, self.action_dim, "previous policy action")
        if abs(float(np.linalg.norm(quat)) - 1.0) > 5.0e-2:
            raise ValueError("policy state quaternion is not normalized")
        if float(np.max(np.abs(previous))) > 1.0 + 1.0e-5:
            raise ValueError("previous lower action exceeds the normalized action bound")

        lock = self.state.get_lock()
        self._acquire(lock, "state-write")
        try:
            if sequence <= int(self._state_sequence.value):
                raise ValueError("policy state sequence must increase monotonically")
            buffer = np.frombuffer(self.state.get_obj(), dtype=np.float32)
            buffer[0:29] = q
            buffer[29:58] = dq
            buffer[58:62] = quat
            buffer[62:65] = gyro
            buffer[65:68] = cmd
            buffer[68 : 68 + self.action_dim] = previous
            buffer[68 + self.action_dim] = locomotion_mode
            self._previous_action_sequence.value = int(previous_action_sequence)
            self._policy_epoch.value = int(policy_epoch)
            self._state_monotonic_ns.value = monotonic_ns
            published_ns = time.monotonic_ns()
            self._state_published_ns.value = published_ns
            # Publish the sequence last while holding the same lock.
            self._state_sequence.value = sequence
        finally:
            lock.release()
        self.state_event.set()
        return published_ns

    def _read_state(
        self, lock_timeout_s: float | None = None
    ) -> PolicyStateRequest | None:
        lock = self.state.get_lock()
        # A state read happens only in the isolated policy process.  A brief
        # scheduler/DDS stall can leave the parent holding this lock longer
        # than the nominal 10 ms IPC bound.  This must not kill the worker:
        # returning None lets wait_for_state() wait for the next publication
        # (or retry the newer state) while the parent keeps its independent
        # deadline and 75 ms stale-action watchdog unchanged.
        if not self._try_acquire(lock, lock_timeout_s):
            return None
        try:
            sequence = int(self._state_sequence.value)
            if sequence < 0:
                return None
            timestamp = int(self._state_monotonic_ns.value)
            published = int(self._state_published_ns.value)
            previous_sequence = int(self._previous_action_sequence.value)
            policy_epoch = int(self._policy_epoch.value)
            buffer = np.frombuffer(self.state.get_obj(), dtype=np.float32).copy()
        finally:
            lock.release()
        return PolicyStateRequest(
            sequence=sequence,
            monotonic_ns=timestamp,
            request_published_ns=published,
            previous_action_sequence=previous_sequence,
            q29=buffer[0:29],
            dq29=buffer[29:58],
            quaternion_wxyz=buffer[58:62],
            angular_velocity_body=buffer[62:65],
            command=buffer[65:68],
            previous_lower_action=buffer[68 : 68 + self.action_dim],
            locomotion_mode=int(buffer[68 + self.action_dim]),
            policy_epoch=policy_epoch,
        )

    def wait_for_state(
        self,
        after_sequence: int,
        timeout_s: float = 0.1,
        *,
        spin_after_ns: int | None = None,
    ) -> PolicyStateRequest | None:
        after = int(after_sequence)
        deadline_ns = time.monotonic_ns() + int(
            max(0.0, float(timeout_s)) * 1.0e9
        )
        spin_after = None if spin_after_ns is None else int(spin_after_ns)
        while not self.stop_event.is_set():
            # Clear before reading the sequence. A concurrent publisher either
            # becomes visible in the read or sets the Event after this clear.
            self.state_event.clear()
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return None
            remaining_s = remaining_ns / 1.0e9
            request = self._read_state(
                lock_timeout_s=min(self.lock_timeout_s, remaining_s)
            )
            if request is not None and request.sequence > after:
                return request
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                return None
            if spin_after is not None and now_ns >= spin_after:
                # The policy runs on a parent-isolated CPU. Poll only the
                # lock-free publication sequence during this short pre-wake
                # window, then take the state lock once after publication.
                checks = 0
                while time.monotonic_ns() < deadline_ns:
                    if int(self._state_sequence.value) > after:
                        remaining_ns = deadline_ns - time.monotonic_ns()
                        if remaining_ns <= 0:
                            return None
                        remaining_s = remaining_ns / 1.0e9
                        return self._read_state(
                            lock_timeout_s=min(self.lock_timeout_s, remaining_s)
                        )
                    checks += 1
                    if (checks & 0x3FF) == 0 and self.stop_event.is_set():
                        return None
                return None
            wait_until_ns = deadline_ns
            if spin_after is not None:
                wait_until_ns = min(wait_until_ns, spin_after)
            self.state_event.wait(max(0.0, (wait_until_ns - now_ns) / 1.0e9))
        return None

    def publish_action(self, response: PolicyActionResponse) -> None:
        source_sequence = int(response.source_state_sequence)
        if source_sequence < 0:
            raise ValueError("action source sequence must be non-negative")
        if response.mode not in ("stand", "walk"):
            raise ValueError("policy action mode must be 'stand' or 'walk'")
        started_ns = int(response.inference_started_ns)
        finished_ns = int(response.inference_finished_ns)
        source_published_ns = int(response.source_state_published_ns)
        if (
            source_published_ns <= 0
            or started_ns < source_published_ns
            or finished_ns < started_ns
        ):
            raise ValueError("policy inference timestamps are invalid")
        q_target = _vector(response.q_target15, 15, "q target")
        normalized = _vector(
            response.normalized_lower_action, self.action_dim, "normalized policy action"
        )
        if float(np.max(np.abs(normalized))) > 1.0 + 1.0e-5:
            raise ValueError("normalized lower action exceeds the action bound")
        diagnostics = _vector(
            (response.inference_ms, response.max_action, float(response.mode == "walk")),
            3,
            "action diagnostics",
        )
        lock = self.action.get_lock()
        self._acquire(lock, "action-write")
        try:
            if source_sequence <= int(self._action_source_sequence.value):
                raise ValueError("policy action source sequence must increase monotonically")
            buffer = np.frombuffer(self.action.get_obj(), dtype=np.float32)
            buffer[0:15] = q_target
            action_end = 15 + self.action_dim
            buffer[15:action_end] = normalized
            buffer[action_end : action_end + 3] = diagnostics
            self._action_source_monotonic_ns.value = int(
                response.source_state_monotonic_ns
            )
            self._action_source_published_ns.value = int(
                response.source_state_published_ns
            )
            self._inference_started_ns.value = started_ns
            self._inference_finished_ns.value = finished_ns
            self._skipped_state_count.value = int(response.skipped_state_count)
            self._action_source_sequence.value = source_sequence
        finally:
            lock.release()
        self.action_event.set()

    def _read_action(
        self, lock_timeout_s: float | None = None
    ) -> PolicyActionResponse | None:
        lock = self.action.get_lock()
        try:
            self._acquire(lock, "action-read", lock_timeout_s)
        except IPCLockTimeout:
            # Treat a delayed peer as "no action yet"; the caller's deadline
            # then counts it as an ordinary miss instead of faulting.
            return None
        try:
            source_sequence = int(self._action_source_sequence.value)
            if source_sequence < 0:
                return None
            source_timestamp = int(self._action_source_monotonic_ns.value)
            source_published = int(self._action_source_published_ns.value)
            started = int(self._inference_started_ns.value)
            finished = int(self._inference_finished_ns.value)
            skipped = int(self._skipped_state_count.value)
            buffer = np.frombuffer(self.action.get_obj(), dtype=np.float32).copy()
        finally:
            lock.release()
        action_end = 15 + self.action_dim
        return PolicyActionResponse(
            source_state_sequence=source_sequence,
            source_state_monotonic_ns=source_timestamp,
            source_state_published_ns=source_published,
            inference_started_ns=started,
            inference_finished_ns=finished,
            skipped_state_count=skipped,
            q_target15=buffer[0:15],
            normalized_lower_action=buffer[15:action_end],
            mode="walk" if buffer[action_end + 2] >= 0.5 else "stand",
            inference_ms=float(buffer[action_end]),
            max_action=float(buffer[action_end + 1]),
        )

    def wait_for_action(
        self, source_state_sequence: int, timeout_s: float
    ) -> PolicyActionResponse | None:
        expected = int(source_state_sequence)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while not self.stop_event.is_set():
            self.action_event.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            response = self._read_action(
                lock_timeout_s=min(self.lock_timeout_s, remaining)
            )
            if response is not None:
                if response.source_state_sequence == expected:
                    return response
                if response.source_state_sequence > expected:
                    raise RuntimeError(
                        "policy action sequence advanced past request %d to %d"
                        % (expected, response.source_state_sequence)
                    )
                # A late response to a timed-out request is deliberately ignored.
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            self.action_event.wait(remaining)
        return None

    def latest_action(self) -> PolicyActionResponse | None:
        """Return the latest response for shadow diagnostics without accepting it."""

        return self._read_action()
