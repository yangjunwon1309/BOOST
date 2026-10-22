"""Backend-neutral interface for querying object surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class SurfaceHit:
    """Signed-distance result for one query point, expressed in world frame."""

    d: float
    s: np.ndarray | None
    n: np.ndarray | None
    obj_id: int
    valid: bool


class SurfaceQuery(Protocol):
    """A stateful surface backend. ``query`` accepts world-frame XYZ points."""

    def update(self, state: Any) -> None: ...

    def query(self, points_world: np.ndarray) -> list[SurfaceHit]: ...
