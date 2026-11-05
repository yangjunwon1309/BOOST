"""Track 250-Hz LowCmd lower targets as deployment transport telemetry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from dreamwaq_interface.g1_29dof.contract import DreamWaQ29Contract, JointOrderMap


@dataclass(frozen=True)
class PublishedLowerAction:
    sequence: int
    monotonic_ns: int
    normalized: np.ndarray


class PublishedActionTracker:
    """Timestamped history for comparing actor history with published targets.

    The writer can publish between the DDS reader receiving a sample and the
    50-Hz parent consuming it. Keeping a short history avoids incorrectly
    pairing a state with a command published after the state timestamp.  This
    data is diagnostic-only: DreamWaQ's policy observation receives the prior
    accepted actor output, matching training and sim2sim.
    """

    def __init__(self, contract: DreamWaQ29Contract, history_size: int = 512) -> None:
        if int(history_size) < 2:
            raise ValueError("published-action history must contain at least two entries")
        mapper = JointOrderMap()
        self._lower_indices = mapper.sdk_indices(contract.action_joint_names)
        self._default = contract.default_q_sdk[self._lower_indices].astype(np.float32)
        self._scale = contract.physical_action_scale.astype(np.float32)
        self._clip = float(contract.clip_actions)
        self._action_dim = int(contract.action_dim)
        self._history: deque[PublishedLowerAction] = deque(maxlen=int(history_size))
        self._lock = threading.Lock()

    def record(self, q_target_sdk: Any, metadata: Any, monotonic_ns: int) -> None:
        target = np.asarray(q_target_sdk, dtype=np.float32).reshape(-1)
        if target.shape != (29,) or not np.all(np.isfinite(target)):
            raise ValueError("published target must contain 29 finite values")
        timestamp = int(monotonic_ns)
        if timestamp <= 0:
            raise ValueError("published target timestamp must be positive")
        sequence = -1
        if metadata is not None:
            if isinstance(metadata, dict):
                sequence = int(metadata.get("action_sequence", -1))
            else:
                sequence = int(metadata)
        normalized = np.clip(
            (target[self._lower_indices] - self._default) / self._scale,
            -self._clip,
            self._clip,
        ).astype(np.float32)
        entry = PublishedLowerAction(sequence, timestamp, normalized)
        with self._lock:
            if self._history and timestamp < self._history[-1].monotonic_ns:
                raise ValueError("published target timestamps must increase monotonically")
            self._history.append(entry)

    def at_or_before(self, monotonic_ns: int) -> PublishedLowerAction:
        timestamp = int(monotonic_ns)
        with self._lock:
            for entry in reversed(self._history):
                if entry.monotonic_ns <= timestamp:
                    return PublishedLowerAction(
                        entry.sequence, entry.monotonic_ns, entry.normalized.copy()
                    )
        return PublishedLowerAction(-1, 0, np.zeros(self._action_dim, dtype=np.float32))

    def latest(self) -> PublishedLowerAction:
        with self._lock:
            if self._history:
                entry = self._history[-1]
                return PublishedLowerAction(
                    entry.sequence, entry.monotonic_ns, entry.normalized.copy()
                )
        return PublishedLowerAction(-1, 0, np.zeros(self._action_dim, dtype=np.float32))
