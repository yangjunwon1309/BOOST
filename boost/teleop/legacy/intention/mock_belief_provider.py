from __future__ import annotations

from .belief_provider import BeliefState


class MockBeliefProvider:
    def __init__(self, object_id: str = "mock_object", probability: float = 0.0):
        self.state = BeliefState(object_id=object_id, probability=probability, valid=True)

    def set_probability(self, probability: float) -> None:
        self.state.probability = probability

    def latest(self) -> BeliefState:
        return self.state

