"""PATH EE target publishing for segmented RGB-D objects."""

from __future__ import annotations

import time
import uuid
import math
from dataclasses import dataclass

import msgpack
import numpy as np
import zmq


PICO_WAIST_FRAME = "pico_head_position_waist"

# G1 Q_READY FK at PATH's wrist-yaw + 5 cm EE frames.  Perception targets use
# these neutral human wrist poses when no live PICO wrist stream is available.
Q_READY_LEFT_EE_POSITION = np.array(
    [0.24977428385080602, 0.1486521246660517, 0.09523008429534556],
    dtype=np.float64,
)
Q_READY_RIGHT_EE_POSITION = np.array(
    [0.24977428385080602, -0.1486421246660517, 0.09523008429534556],
    dtype=np.float64,
)


def _neutral_wrist_transform(position: np.ndarray) -> list[float]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform.reshape(-1).tolist()


@dataclass(frozen=True)
class Workspace:
    x: tuple[float, float] = (-0.15, 0.45)
    left_y: tuple[float, float] = (0.03, 0.50)
    right_y: tuple[float, float] = (-0.50, -0.03)
    z: tuple[float, float] = (-0.30, 0.35)
    radius_m: float = 0.62

    def validate(self, position: np.ndarray, side: str) -> dict:
        point = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            raise ValueError("target position contains a non-finite value")
        y_limits = self.right_y if side == "right" else self.left_y
        lower = np.array([self.x[0], y_limits[0], self.z[0]], dtype=np.float64)
        upper = np.array([self.x[1], y_limits[1], self.z[1]], dtype=np.float64)
        reasons = []
        if np.any(point < lower) or np.any(point > upper):
            reasons.append("box")
        if float(np.linalg.norm(point)) > self.radius_m:
            reasons.append("radius")
        if reasons:
            raise ValueError(
                "target outside conservative workspace "
                f"side={side} position={np.round(point, 4).tolist()} "
                f"reason={','.join(reasons)}"
            )
        return {
            "workspace_mode": "reject",
            "kinematic_feasible": True,
            "workspace_clamped": False,
            "workspace_reason": "ok",
            "workspace_side": side,
            "workspace_lower": lower.tolist(),
            "workspace_upper": upper.tolist(),
            "workspace_radius_m": float(self.radius_m),
            "requested_position": point.tolist(),
        }


def make_goal_payload(
    position: np.ndarray,
    *,
    side: str = "left",
    target_link: str = "left_wrist_yaw_link",
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    duration_sec: float = 2.0,
    final_hold_sec: float = 1.0,
    workspace_info: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    point = np.asarray(position, dtype=np.float64).reshape(3)
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(point).all() or not np.isfinite(quat).all():
        raise ValueError("goal pose contains a non-finite value")
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-9:
        raise ValueError("goal quaternion has zero norm")
    payload = {
        "goal_uid": uuid.uuid4().hex,
        "force_replan": True,
        "streaming_goal": False,
        "target_link": str(target_link),
        "side": str(side),
        "position": point.tolist(),
        "quaternion": (quat / norm).tolist(),
        "source": "sam3_segment_center",
        "coordinate_frame": PICO_WAIST_FRAME,
        "target_position_frame": PICO_WAIST_FRAME,
        "human_wrist_frame": PICO_WAIST_FRAME,
        "human_left_wrist_transform": _neutral_wrist_transform(
            Q_READY_LEFT_EE_POSITION
        ),
        "human_right_wrist_transform": _neutral_wrist_transform(
            Q_READY_RIGHT_EE_POSITION
        ),
        "human_left_wrist_position": Q_READY_LEFT_EE_POSITION.tolist(),
        "human_right_wrist_position": Q_READY_RIGHT_EE_POSITION.tolist(),
        "target_valid": True,
        "duration_sec": float(duration_sec),
        "final_hold_sec": float(final_hold_sec),
    }
    if workspace_info:
        payload.update(workspace_info)
    if metadata:
        payload["perception"] = metadata
    return payload


class GoalPublisher:
    """Persistent PUB sockets avoid the short-lived ZeroMQ slow-joiner loss."""

    def __init__(
        self,
        *,
        goal_endpoint: str,
        goal_topic: str,
        dry_run: bool = False,
    ) -> None:
        self.goal_topic = goal_topic
        self.dry_run = dry_run
        self.goal_socket = None
        if not dry_run:
            context = zmq.Context.instance()
            self.goal_socket = context.socket(zmq.PUB)
            self.goal_socket.setsockopt(zmq.LINGER, 0)
            # Belief is a latest-state stream.  Never let a disconnected or
            # slow bridge stall the perception/belief worker behind an old
            # target message.
            self.goal_socket.setsockopt(zmq.SNDHWM, 1)
            self.goal_socket.connect(goal_endpoint)

    @staticmethod
    def _send(socket: zmq.Socket, topic: str, payload: dict) -> None:
        try:
            socket.send(
                topic.encode() + msgpack.packb(payload, use_bin_type=True),
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            # Dropping an obsolete belief packet is correct: the next 5 Hz
            # publication contains the full current two-hand state.
            pass

    def publish(
        self,
        payload: dict,
        repeat: int = 3,
        period_sec: float = 0.05,
        sustain_sec: float = 0.0,
    ) -> None:
        if self.dry_run:
            return
        if self.goal_socket is None:
            raise RuntimeError("goal publisher is closed")
        period = max(0.001, float(period_sec))
        count = max(int(repeat), int(math.ceil(max(0.0, sustain_sec) / period)) + 1)
        for index in range(max(1, count)):
            self._send(self.goal_socket, self.goal_topic, payload)
            if index + 1 < count:
                time.sleep(period)

    def close(self) -> None:
        if self.goal_socket is not None:
            self.goal_socket.close()
            self.goal_socket = None
