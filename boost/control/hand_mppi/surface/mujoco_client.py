"""ZMQ client for the MuJoCo process's live collision-surface query service."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import zmq

from boost.io.path_transport.zmq_wire import pack_pose_message, unpack_pose_message

from .base import SurfaceHit
from .rpc import (
    NUM_TIPS,
    SIDE_TO_ID,
    SURFACE_PROTOCOL_VERSION,
    SURFACE_QUERY_TOPIC,
)


@dataclass(frozen=True)
class MujocoSurfaceReply:
    sequence: int
    hits: list[SurfaceHit]
    wrist_position_world: np.ndarray
    wrist_rotation_world: np.ndarray
    sim_time: float
    object_geom_count: int


class MujocoSurfaceClient:
    """Synchronous client; a timed-out request is discarded and the REQ socket reset."""

    def __init__(self, endpoint: str, timeout_sec: float = 0.05) -> None:
        if not endpoint:
            raise ValueError("MuJoCo surface-query endpoint must be non-empty")
        if not np.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("surface-query timeout must be positive and finite")
        self.endpoint = endpoint
        self.timeout_ms = max(1, int(round(timeout_sec * 1000.0)))
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self.last_query_duration_ms = float("nan")
        self.last_query_status = "never"
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket

    def query_hand_tips(
        self,
        side: str,
        points_wrist: np.ndarray,
        *,
        sequence: int,
    ) -> MujocoSurfaceReply | None:
        if side not in SIDE_TO_ID:
            raise ValueError(f"unknown hand side {side!r}")
        points = np.asarray(points_wrist, dtype=np.float32)
        if points.shape != (NUM_TIPS, 3) or not np.isfinite(points).all():
            raise ValueError("wrist-frame fingertip points must be finite with shape (5, 3)")
        request = {
            "surface_protocol_version": np.asarray(
                [SURFACE_PROTOCOL_VERSION], dtype=np.int32
            ),
            "surface_hand_side": np.asarray([SIDE_TO_ID[side]], dtype=np.int32),
            "surface_sequence": np.asarray([sequence], dtype=np.int64),
            "surface_tip_points_wrist": points,
        }
        assert self._socket is not None
        query_start = time.perf_counter()
        self.last_query_status = "error"
        try:
            self._socket.send(
                pack_pose_message(request, topic=SURFACE_QUERY_TOPIC)
            )
            if not self._socket.poll(self.timeout_ms, zmq.POLLIN):
                self.last_query_status = "timeout"
                self._connect()
                return None
            raw = self._socket.recv()
            payload = unpack_pose_message(raw, topic=SURFACE_QUERY_TOPIC)
            reply = self._decode_reply(payload, side=side, sequence=sequence)
            self.last_query_status = "ok" if reply is not None else "invalid_reply"
            return reply
        except (zmq.ZMQError, KeyError, TypeError, ValueError) as exc:
            self.last_query_status = "timeout" if isinstance(exc, zmq.Again) else "error"
            self._connect()
            return None
        finally:
            self.last_query_duration_ms = (time.perf_counter() - query_start) * 1000.0

    @staticmethod
    def _decode_reply(
        payload: dict[str, np.ndarray], *, side: str, sequence: int
    ) -> MujocoSurfaceReply | None:
        version = int(np.asarray(payload["surface_protocol_version"]).reshape(-1)[0])
        response_sequence = int(np.asarray(payload["surface_sequence"]).reshape(-1)[0])
        if version != SURFACE_PROTOCOL_VERSION or response_sequence != sequence:
            return None
        valid = np.asarray(payload["surface_valid"], dtype=bool).reshape(NUM_TIPS)
        distances = np.asarray(payload["surface_distance"], dtype=np.float64).reshape(NUM_TIPS)
        points = np.asarray(payload["surface_point_world"], dtype=np.float64).reshape(NUM_TIPS, 3)
        normals = np.asarray(payload["surface_normal_world"], dtype=np.float64).reshape(NUM_TIPS, 3)
        geom_ids = np.asarray(payload["surface_geom_id"], dtype=np.int32).reshape(NUM_TIPS)
        wrist_position = np.asarray(
            payload["surface_wrist_position_world"], dtype=np.float64
        ).reshape(3)
        wrist_rotation = np.asarray(
            payload["surface_wrist_rotation_world"], dtype=np.float64
        ).reshape(3, 3)
        if not (
            np.isfinite(distances).all()
            and np.isfinite(wrist_position).all()
            and np.isfinite(wrist_rotation).all()
            and np.isfinite(points[valid]).all()
            and np.isfinite(normals[valid]).all()
        ):
            return None
        hits = [
            SurfaceHit(
                d=float(distances[index]),
                s=points[index].copy() if valid[index] else None,
                n=normals[index].copy() if valid[index] else None,
                obj_id=int(geom_ids[index]),
                valid=bool(valid[index]),
            )
            for index in range(NUM_TIPS)
        ]
        return MujocoSurfaceReply(
            sequence=response_sequence,
            hits=hits,
            wrist_position_world=wrist_position,
            wrist_rotation_world=wrist_rotation,
            sim_time=float(np.asarray(payload["surface_sim_time"]).reshape(-1)[0]),
            object_geom_count=int(
                np.asarray(payload["surface_object_geom_count"]).reshape(-1)[0]
            ),
        )

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
