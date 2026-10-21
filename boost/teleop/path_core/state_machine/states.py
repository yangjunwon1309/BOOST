from __future__ import annotations

from enum import Enum


class PathState(str, Enum):
    TELEOP = "teleop"
    INTENTION = "intention"
    AUTO_PLANNING = "auto_planning"
    DAMPED_RETURN = "damped_return"

