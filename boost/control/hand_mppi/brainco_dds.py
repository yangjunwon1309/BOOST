"""Unitree DDS transport for the BrainCo Revo2 hand command/state contract."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from boost.control.hand_mppi.revo2 import (
    BRAINCO_CHANNELS,
    BRAINCO_COMMAND_TOPICS,
    BRAINCO_SIDES,
    BRAINCO_STATE_TOPICS,
    BrainCoRevo2Adapter,
)

Side = Literal["left", "right"]


@dataclass(frozen=True)
class BrainCoHandState:
    """One measured hand state, with q and dq in the normalized DDS units."""

    side: Side
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau_est: tuple[float, ...]
    received_monotonic: float


def _six(values, *, label: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
    if array.size != len(BRAINCO_CHANNELS):
        raise ValueError(f"{label} must have six channels, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} values must all be finite")
    return array


class BrainCoDDS:
    """Publish six-channel Revo2 targets and receive measured hand state.

    This adapter deliberately uses the same DDS topics and IDL types as the
    Revo2 MuJoCo backend and Unitree BrainCo service. It never publishes
    ``rt/lowcmd``; HOMIE remains the sole body command writer.
    """

    def __init__(self, *, domain_id: int = 0, interface: str = "lo") -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__MotorCmd_,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        except ImportError as exc:
            raise RuntimeError(
                "BrainCo DDS needs the BOOST Pixi sim environment; run `pixi run -e sim setup-sim`."
            ) from exc

        ChannelFactoryInitialize(int(domain_id), interface)
        self._MotorCmd = unitree_go_msg_dds__MotorCmd_
        self._MotorCmds = MotorCmds_
        self._adapter = BrainCoRevo2Adapter()
        self._lock = threading.Lock()
        self._state: dict[Side, BrainCoHandState | None] = {"left": None, "right": None}
        self._publishers = {}
        self._subscribers = {}

        for side in BRAINCO_SIDES:
            publisher = ChannelPublisher(BRAINCO_COMMAND_TOPICS[side], MotorCmds_)
            publisher.Init()
            self._publishers[side] = publisher

            subscriber = ChannelSubscriber(BRAINCO_STATE_TOPICS[side], MotorStates_)
            subscriber.Init(self._state_callback(side), 1)
            self._subscribers[side] = subscriber

    def _state_callback(self, side: Side):
        def callback(message) -> None:
            states = getattr(message, "states", ())
            if len(states) != len(BRAINCO_CHANNELS):
                return
            try:
                q = _six((state.q for state in states), label="BrainCo q")
                dq = _six((state.dq for state in states), label="BrainCo dq")
                tau = _six((state.tau_est for state in states), label="BrainCo tau_est")
            except (AttributeError, TypeError, ValueError):
                return
            observation = BrainCoHandState(
                side=side,
                q=tuple(float(value) for value in q),
                dq=tuple(float(value) for value in dq),
                tau_est=tuple(float(value) for value in tau),
                received_monotonic=time.monotonic(),
            )
            with self._lock:
                self._state[side] = observation

        return callback

    def publish(self, side: Side, q: tuple[float, ...] | list[float], *, speed: float = 1.0) -> None:
        """Publish normalized q and a normalized speed setting for one hand."""
        if side not in BRAINCO_SIDES:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        command_q = self._adapter.normalized_command(q)
        if not np.isfinite(speed):
            raise ValueError("speed must be finite")
        command_speed = float(np.clip(speed, 0.0, 1.0))
        message = self._MotorCmds([self._MotorCmd() for _ in BRAINCO_CHANNELS])
        for command, position in zip(message.cmds, command_q):
            command.q = float(position)
            command.dq = command_speed
        self._publishers[side].Write(message)

    def state(self, side: Side, *, timeout_sec: float = 0.25) -> BrainCoHandState | None:
        """Return the latest state only if it is fresh within ``timeout_sec``."""
        if side not in BRAINCO_SIDES:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            observation = self._state[side]
        if observation is None or time.monotonic() - observation.received_monotonic > timeout_sec:
            return None
        return observation

    def open_both(self) -> None:
        """Explicitly request the configured open pose on both sides."""
        pose = self._adapter.binary_target(False)
        self.publish("left", pose)
        self.publish("right", pose)
