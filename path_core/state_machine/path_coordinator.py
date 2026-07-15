from __future__ import annotations

from dataclasses import dataclass

from .states import PathState


@dataclass
class PathSignals:
    belief: float = 0.0
    belief_ready_count: int = 0
    operator_trigger: bool = False
    sonic_planner_switch: bool = False
    object_gripped: bool = False
    return_matched: bool = False


class PathStateMachine:
    def __init__(self, belief_threshold: float = 0.75, consecutive_frames: int = 5):
        self.state = PathState.TELEOP
        self.belief_threshold = belief_threshold
        self.consecutive_frames = consecutive_frames
        self._belief_count = 0

    def update(self, signals: PathSignals) -> PathState:
        if signals.belief > self.belief_threshold:
            self._belief_count += 1
        else:
            self._belief_count = 0

        if self.state == PathState.TELEOP:
            if self._belief_count >= self.consecutive_frames:
                self.state = PathState.INTENTION
        elif self.state == PathState.INTENTION:
            if (
                signals.operator_trigger
                and signals.belief > self.belief_threshold
                and signals.sonic_planner_switch
            ):
                self.state = PathState.AUTO_PLANNING
        elif self.state == PathState.AUTO_PLANNING:
            if signals.object_gripped:
                self.state = PathState.DAMPED_RETURN
        elif self.state == PathState.DAMPED_RETURN:
            if signals.return_matched and signals.sonic_planner_switch:
                self.state = PathState.TELEOP
        return self.state

