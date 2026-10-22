"""MuJoCo-process RPC endpoint for live fingertip-to-object surface queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import zmq

from boost.control.hand_mppi.surface.mj_geom import (
    FINGERS,
    MjGeomSurface,
    SIDES,
    add_surface_probes,
)
from boost.control.hand_mppi.surface.rpc import (
    ID_TO_SIDE,
    NUM_TIPS,
    SURFACE_PROTOCOL_VERSION,
    SURFACE_QUERY_TOPIC,
)
from boost.io.path_transport.zmq_wire import pack_pose_message, unpack_pose_message


def load_model_with_surface_probes(
    scene_path: Path,
    *,
    probe_radius_m: float,
) -> mujoco.MjModel:
    """Compile the selected scene after adding invisible query-only probes."""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_surface_probes(spec, radius_m=probe_radius_m)
    return spec.compile()


class MuJoCoSurfaceQueryService:
    """Serve five wrist-frame tip points at a time against the live sim state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        endpoint: str,
        object_geom_names: list[str],
        distmax_m: float = 0.03,
        probe_radius_m: float = 0.0005,
    ) -> None:
        self.model = model
        self.data = data
        self.surface = MjGeomSurface(
            model,
            object_geom_names=object_geom_names,
            distmax_m=distmax_m,
            probe_radius_m=probe_radius_m,
        )
        self.wrist_body_ids = {
            side: self._require_id(
                mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link"
            )
            for side in SIDES
        }
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(endpoint)
        self.endpoint = endpoint
        self.object_geom_count = len(self.surface.object_geom_ids)
        self._last_error = ""

    def _require_id(self, kind: mujoco.mjtObj, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, kind, name))
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing {kind.name} {name!r}")
        return object_id

    def poll_once(self) -> bool:
        """Process at most one queued query without blocking the simulation loop."""
        if not self._socket.poll(0, zmq.POLLIN):
            return False
        raw = self._socket.recv()
        sequence = -1
        try:
            request = unpack_pose_message(raw, topic=SURFACE_QUERY_TOPIC)
            version = int(np.asarray(request["surface_protocol_version"]).reshape(-1)[0])
            side_id = int(np.asarray(request["surface_hand_side"]).reshape(-1)[0])
            sequence = int(np.asarray(request["surface_sequence"]).reshape(-1)[0])
            points_wrist = np.asarray(
                request["surface_tip_points_wrist"], dtype=np.float64
            )
            if version != SURFACE_PROTOCOL_VERSION:
                raise ValueError(f"unsupported surface protocol version {version}")
            if side_id not in ID_TO_SIDE:
                raise ValueError(f"invalid surface hand-side id {side_id}")
            if points_wrist.shape != (NUM_TIPS, 3) or not np.isfinite(points_wrist).all():
                raise ValueError("request fingertip points must be finite with shape (5, 3)")
            side = ID_TO_SIDE[side_id]
            wrist_id = self.wrist_body_ids[side]
            wrist_position = self.data.xpos[wrist_id].copy()
            wrist_rotation = self.data.xmat[wrist_id].reshape(3, 3).copy()
            points_world = points_wrist @ wrist_rotation.T + wrist_position
            self.surface.update(self.data)
            hits = self.surface.query(points_world, probe_offset=side_id * NUM_TIPS)
            response = self._encode_response(
                sequence,
                hits,
                wrist_position,
                wrist_rotation,
                float(self.data.time),
            )
            self._last_error = ""
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            message = str(exc)
            if message != self._last_error:
                print(f"[BOOST MuJoCo surface] query rejected: {message}", flush=True)
                self._last_error = message
            response = self._encode_response(
                sequence,
                [self.surface._invalid_hit() for _ in range(NUM_TIPS)],
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64),
                float(self.data.time),
            )
        self._socket.send(pack_pose_message(response, topic=SURFACE_QUERY_TOPIC))
        return True

    def _encode_response(
        self,
        sequence: int,
        hits: list[Any],
        wrist_position: np.ndarray,
        wrist_rotation: np.ndarray,
        sim_time: float,
    ) -> dict[str, np.ndarray]:
        valid = np.asarray([hit.valid for hit in hits], dtype=bool)
        points = np.zeros((NUM_TIPS, 3), dtype=np.float32)
        normals = np.zeros((NUM_TIPS, 3), dtype=np.float32)
        distances = np.full(NUM_TIPS, self.surface.distmax_m, dtype=np.float32)
        geom_ids = np.full(NUM_TIPS, -1, dtype=np.int32)
        for index, hit in enumerate(hits):
            distances[index] = float(hit.d)
            if hit.valid:
                points[index] = np.asarray(hit.s, dtype=np.float32)
                normals[index] = np.asarray(hit.n, dtype=np.float32)
                geom_ids[index] = int(hit.obj_id)
        return {
            "surface_protocol_version": np.asarray(
                [SURFACE_PROTOCOL_VERSION], dtype=np.int32
            ),
            "surface_sequence": np.asarray([sequence], dtype=np.int64),
            "surface_valid": valid,
            "surface_distance": distances,
            "surface_point_world": points,
            "surface_normal_world": normals,
            "surface_geom_id": geom_ids,
            "surface_wrist_position_world": np.asarray(
                wrist_position, dtype=np.float32
            ),
            "surface_wrist_rotation_world": np.asarray(
                wrist_rotation, dtype=np.float32
            ),
            "surface_sim_time": np.asarray([sim_time], dtype=np.float64),
            "surface_object_geom_count": np.asarray(
                [self.object_geom_count], dtype=np.int32
            ),
        }

    def close(self) -> None:
        self._socket.close(linger=0)
