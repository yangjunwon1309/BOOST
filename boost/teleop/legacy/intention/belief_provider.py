from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BeliefState:
    object_id: str = ""
    probability: float = 0.0
    valid: bool = False


class BeliefProvider(Protocol):
    def latest(self) -> BeliefState:
        ...

