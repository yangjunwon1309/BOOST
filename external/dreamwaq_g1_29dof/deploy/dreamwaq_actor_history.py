"""Checkpoint-compatible previous-action state for DreamWaQ deployment.

DreamWaQ's exported policy was trained and evaluated with ``previous_action``
equal to the bounded actor output from the preceding 50-Hz policy tick.  This
is deliberately separate from the 250-Hz LowCmd writer telemetry: the latter
is valuable for diagnosing transport delay and target limiting, but it is not
the observation feature the checkpoint was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreviousActorAction:
    """The last parent-accepted, normalized lower-body actor output."""

    sequence: int
    normalized: np.ndarray


class AcceptedActorActionHistory:
    """Store only actor outputs accepted for the 50-Hz control timeline.

    A deadline miss intentionally leaves this state unchanged.  A late worker
    response is never committed by the parent, so it cannot contaminate a
    later observation's ``previous_action``.
    """

    def __init__(self, action_dim: int = 12, clip_actions: float = 1.0) -> None:
        if int(action_dim) <= 0:
            raise ValueError("actor action dimension must be positive")
        if not np.isfinite(float(clip_actions)) or float(clip_actions) <= 0.0:
            raise ValueError("actor action clip must be positive and finite")
        self._action_dim = int(action_dim)
        self._clip_actions = float(clip_actions)
        self._sequence = -1
        self._normalized = np.zeros(self._action_dim, dtype=np.float32)

    def snapshot(self) -> PreviousActorAction:
        return PreviousActorAction(self._sequence, self._normalized.copy())

    def accept(self, sequence: int, normalized_lower_action: Any) -> None:
        sequence = int(sequence)
        if sequence <= self._sequence:
            raise ValueError(
                "accepted actor-action sequence must increase monotonically"
            )
        normalized = np.asarray(normalized_lower_action, dtype=np.float32).reshape(-1)
        if normalized.shape != (self._action_dim,) or not np.all(np.isfinite(normalized)):
            raise ValueError(
                "accepted actor action must contain %d finite values" % self._action_dim
            )
        if float(np.max(np.abs(normalized))) > self._clip_actions + 1.0e-5:
            raise ValueError("accepted actor action exceeds the exported action bound")
        self._sequence = sequence
        self._normalized[:] = normalized
