from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class UpperBodyReference:
    q_upper: np.ndarray
    timestamp_us: int
    source: str
    valid: bool = True


class UpperBodyReferenceProvider(Protocol):
    def latest(self) -> UpperBodyReference:
        ...

