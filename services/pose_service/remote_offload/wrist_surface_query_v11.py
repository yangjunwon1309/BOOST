"""Thor-local, opt-in signed-distance queries over SAM3/D405 surface clouds."""

from __future__ import annotations

import logging
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import zmq
from scipy.spatial import cKDTree

from .surface_query_wire import (
    ID_TO_SIDE,
    NUM_TIPS,
    SURFACE_PROTOCOL_VERSION,
    pack_surface_query,
    unpack_surface_query,
)

from .wrist_d405_protocol_v11 import unpack_surface_points


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CloudSnapshot:
    points: np.ndarray
    tree: cKDTree | None
    object_trees: dict[int, cKDTree]
    camera_origin: np.ndarray | None
    object_ids: np.ndarray
    received_monotonic: float
    object_count: int


class WristSurfaceQueryServer:
    """Serve latest, same-side SAM3 point clouds on a Thor-local ZMQ REP socket.

    The signed distance is a local tangent-plane approximation: nearest depth
    point plus a PCA normal oriented toward that wrist camera. It is not a
    watertight/global SDF; queries are rejected when support or freshness is
    insufficient.
    """

    def __init__(
        self,
        *,
        endpoint: str = "tcp://127.0.0.1:5571",
        max_cloud_age_sec: float = 0.35,
        max_hand_age_sec: float = 0.20,
        max_query_distance_m: float = 0.05,
        neighborhood_radius_m: float = 0.025,
        min_neighbors: int = 8,
        max_pairing_error_m: float = 0.008,
        target_label: str = "",
    ) -> None:
        if max_cloud_age_sec <= 0.0 or max_hand_age_sec <= 0.0:
            raise ValueError("surface freshness limits must be positive")
        if max_query_distance_m <= 0.0 or neighborhood_radius_m <= 0.0:
            raise ValueError("surface query radii must be positive")
        self.endpoint = str(endpoint)
        self.max_cloud_age_sec = float(max_cloud_age_sec)
        self.max_hand_age_sec = float(max_hand_age_sec)
        self.max_query_distance_m = float(max_query_distance_m)
        self.neighborhood_radius_m = float(neighborhood_radius_m)
        self.min_neighbors = max(3, int(min_neighbors))
        self.max_pairing_error_m = max(0.0, float(max_pairing_error_m))
        self.target_label = str(target_label).strip().casefold()
        empty = _CloudSnapshot(
            points=np.empty((0, 3), dtype=np.float32),
            tree=None,
            object_trees={},
            camera_origin=None,
            object_ids=np.empty((0,), dtype=np.int32),
            received_monotonic=0.0,
            object_count=0,
        )
        self._snapshots = {"left": empty, "right": empty}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._serve, name="wrist-surface-query", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=1.0):
            raise RuntimeError("hand surface-query server did not finish binding")
        if self._startup_error is not None:
            raise RuntimeError(
                f"could not bind hand surface-query endpoint {self.endpoint}"
            ) from self._startup_error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def update_result(self, result: dict[str, Any]) -> None:
        """Atomically replace both sides, clearing prior clouds on every result."""
        objects = [item for item in result.get("objects", []) if isinstance(item, dict)]
        transforms = result.get("t_goal_cam", {})
        now = time.monotonic()
        updates: dict[str, _CloudSnapshot] = {}
        for side in ("left", "right"):
            camera_id = f"{side}_wrist"
            point_sets: list[np.ndarray] = []
            object_ids: list[np.ndarray] = []
            camera_origin: np.ndarray | None = None
            object_count = 0
            for obj in objects:
                if obj.get("camera_id") != camera_id:
                    continue
                if self.target_label and str(obj.get("label", "")).strip().casefold() != self.target_label:
                    continue
                cloud = obj.get("surface_cloud")
                geometry = obj.get("geometry")
                pairing = obj.get("state_pairing")
                if (
                    not isinstance(cloud, dict)
                    or cloud.get("encoding") != "zlib-f32le-xyz-v1"
                    or cloud.get("frame") != "pelvis_identity_fk"
                    or not isinstance(geometry, dict)
                    or not bool(geometry.get("valid", False))
                    or bool(obj.get("geometry_fallback", False))
                ):
                    continue
                try:
                    if not isinstance(pairing, dict) or not bool(pairing.get("accepted", False)):
                        continue
                    pairing_error = float(pairing.get("estimated_error_m"))
                    if not np.isfinite(pairing_error) or pairing_error > self.max_pairing_error_m:
                        continue
                    points = unpack_surface_points(
                        cloud["points"], count=int(cloud["count"])
                    )
                    track_id = str(cloud.get("track_id", obj.get("track_id", "")))
                    if not track_id:
                        continue
                    stable_object_id = zlib.crc32(track_id.encode("utf-8")) & 0x7FFFFFFF
                except (KeyError, TypeError, ValueError, OverflowError, zlib.error):
                    continue
                point_sets.append(points)
                object_ids.append(np.full(len(points), stable_object_id, dtype=np.int32))
                object_count += 1

                if camera_origin is None:
                    try:
                        transform = np.asarray(transforms[camera_id], dtype=np.float64).reshape(4, 4)
                        origin = transform[:3, 3]
                        if np.isfinite(origin).all():
                            camera_origin = origin.copy()
                    except (KeyError, TypeError, ValueError):
                        pass
            if point_sets and camera_origin is not None:
                points = np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
                ids = np.concatenate(object_ids)
                tree: cKDTree | None = cKDTree(points)
                object_trees = {
                    int(object_id): cKDTree(points[ids == object_id])
                    for object_id in np.unique(ids)
                }
            else:
                points = np.empty((0, 3), dtype=np.float32)
                ids = np.empty((0,), dtype=np.int32)
                tree = None
                object_trees = {}
            updates[side] = _CloudSnapshot(
                points=points,
                tree=tree,
                object_trees=object_trees,
                camera_origin=camera_origin,
                object_ids=ids,
                received_monotonic=now,
                object_count=object_count if tree is not None else 0,
            )
        with self._lock:
            self._snapshots = updates

    def _serve(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SNDHWM, 1)
        try:
            socket.bind(self.endpoint)
            self._ready.set()
            LOG.info("hand surface-query REP bound %s", self.endpoint)
            while not self._stop.is_set():
                if not socket.poll(100, zmq.POLLIN):
                    continue
                raw = socket.recv()
                try:
                    response = self._handle_request(
                        unpack_surface_query(raw)
                    )
                except (KeyError, TypeError, ValueError, OverflowError, IndexError) as error:
                    LOG.debug("rejected hand surface query: %s", error)
                    response = self._empty_response(sequence=-1)
                socket.send(pack_surface_query(response))
        except zmq.ZMQError as error:
            self._startup_error = error
            self._ready.set()
            if not self._stop.is_set():
                LOG.exception("hand surface-query server failed on %s", self.endpoint)
        finally:
            socket.close(linger=0)

    def _handle_request(self, request: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        version = int(np.asarray(request["surface_protocol_version"]).reshape(-1)[0])
        side_id = int(np.asarray(request["surface_hand_side"]).reshape(-1)[0])
        sequence = int(np.asarray(request["surface_sequence"]).reshape(-1)[0])
        if version != SURFACE_PROTOCOL_VERSION or side_id not in ID_TO_SIDE:
            return self._empty_response(sequence=sequence)
        side = ID_TO_SIDE[side_id]
        points_wrist = np.asarray(request["surface_tip_points_wrist"], dtype=np.float64)
        wrist_pose = np.asarray(request["surface_wrist_pose_waist"], dtype=np.float64).reshape(4, 4)
        hand_timestamp = float(
            np.asarray(request["surface_hand_timestamp_monotonic"]).reshape(-1)[0]
        )
        now = time.monotonic()
        if (
            points_wrist.shape != (NUM_TIPS, 3)
            or not np.isfinite(points_wrist).all()
            or not np.isfinite(wrist_pose).all()
            or not np.allclose(wrist_pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5)
            or not np.allclose(wrist_pose[:3, :3] @ wrist_pose[:3, :3].T, np.eye(3), atol=1.0e-3)
            or abs(float(np.linalg.det(wrist_pose[:3, :3])) - 1.0) > 1.0e-3
            or not np.isfinite(hand_timestamp)
            or now - hand_timestamp > self.max_hand_age_sec
            or hand_timestamp > now + 0.05
        ):
            return self._empty_response(sequence=sequence)
        with self._lock:
            snapshot = self._snapshots[side]
        if (
            snapshot.tree is None
            or not snapshot.object_trees
            or now - snapshot.received_monotonic > self.max_cloud_age_sec
            or snapshot.camera_origin is None
        ):
            return self._empty_response(sequence=sequence, wrist_pose=wrist_pose)

        points_world = (
            wrist_pose[:3, :3] @ points_wrist.T
        ).T + wrist_pose[:3, 3]
        # Select one same-side SAM3 object per hand frame: the object whose
        # observed surface is nearest this wrist target. All five fingertips
        # query that object, rather than mixing surfaces object-by-object.
        selected_object_id = min(
            snapshot.object_trees,
            key=lambda object_id: float(
                snapshot.object_trees[object_id].query(wrist_pose[:3, 3], k=1)[0]
            ),
        )
        target_tree = snapshot.object_trees[selected_object_id]
        target_points = snapshot.points[
            snapshot.object_ids == selected_object_id
        ]
        distances = np.full(NUM_TIPS, self.max_query_distance_m, dtype=np.float64)
        surface_points = np.zeros((NUM_TIPS, 3), dtype=np.float64)
        normals = np.zeros((NUM_TIPS, 3), dtype=np.float64)
        object_ids = np.full(NUM_TIPS, -1, dtype=np.int32)
        valid = np.zeros(NUM_TIPS, dtype=bool)
        for tip_index, point in enumerate(points_world):
            nearest_distance, nearest_index = target_tree.query(
                point, k=1, distance_upper_bound=self.max_query_distance_m
            )
            if not np.isfinite(nearest_distance) or nearest_index >= len(target_points):
                continue
            nearest = target_points[int(nearest_index)].astype(np.float64)
            neighbor_indices = target_tree.query_ball_point(
                nearest, r=self.neighborhood_radius_m
            )
            if len(neighbor_indices) < self.min_neighbors:
                continue
            neighborhood = target_points[np.asarray(neighbor_indices, dtype=np.int64)].astype(np.float64)
            center = neighborhood.mean(axis=0)
            centered = neighborhood - center
            covariance = centered.T @ centered / max(1, len(neighborhood) - 1)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            except np.linalg.LinAlgError:
                continue
            if eigenvalues[2] <= 1.0e-10 or eigenvalues[1] / eigenvalues[2] < 0.03:
                continue
            normal = eigenvectors[:, 0]
            if float(np.dot(snapshot.camera_origin - nearest, normal)) < 0.0:
                normal = -normal
            normal_norm = float(np.linalg.norm(normal))
            if not np.isfinite(normal_norm) or normal_norm < 1.0e-8:
                continue
            normal /= normal_norm
            signed_distance = float(np.dot(point - nearest, normal))
            if not np.isfinite(signed_distance):
                continue
            valid[tip_index] = True
            distances[tip_index] = signed_distance
            surface_points[tip_index] = nearest
            normals[tip_index] = normal
            object_ids[tip_index] = selected_object_id
        return self._response(
            sequence=sequence,
            valid=valid,
            distances=distances,
            points=surface_points,
            normals=normals,
            object_ids=object_ids,
            wrist_pose=wrist_pose,
            object_count=snapshot.object_count,
            target_id=selected_object_id,
        )

    @staticmethod
    def _empty_response(
        *, sequence: int, wrist_pose: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        pose = np.eye(4, dtype=np.float64) if wrist_pose is None else wrist_pose
        return WristSurfaceQueryServer._response(
            sequence=sequence,
            valid=np.zeros(NUM_TIPS, dtype=bool),
            distances=np.full(NUM_TIPS, 0.05, dtype=np.float64),
            points=np.zeros((NUM_TIPS, 3), dtype=np.float64),
            normals=np.zeros((NUM_TIPS, 3), dtype=np.float64),
            object_ids=np.full(NUM_TIPS, -1, dtype=np.int32),
            wrist_pose=pose,
            object_count=0,
            target_id=-1,
        )

    @staticmethod
    def _response(
        *,
        sequence: int,
        valid: np.ndarray,
        distances: np.ndarray,
        points: np.ndarray,
        normals: np.ndarray,
        object_ids: np.ndarray,
        wrist_pose: np.ndarray,
        object_count: int,
        target_id: int,
    ) -> dict[str, np.ndarray]:
        return {
            "surface_protocol_version": np.asarray([SURFACE_PROTOCOL_VERSION], dtype=np.int32),
            "surface_sequence": np.asarray([sequence], dtype=np.int64),
            "surface_valid": np.asarray(valid, dtype=bool),
            "surface_distance": np.asarray(distances, dtype=np.float64),
            "surface_point_world": np.asarray(points, dtype=np.float64).reshape(NUM_TIPS, 3),
            "surface_normal_world": np.asarray(normals, dtype=np.float64).reshape(NUM_TIPS, 3),
            "surface_geom_id": np.asarray(object_ids, dtype=np.int32),
            "surface_wrist_position_world": np.asarray(wrist_pose[:3, 3], dtype=np.float64),
            "surface_wrist_rotation_world": np.asarray(wrist_pose[:3, :3], dtype=np.float64),
            "surface_sim_time": np.asarray([time.monotonic()], dtype=np.float64),
            "surface_object_geom_count": np.asarray([object_count], dtype=np.int32),
            "surface_target_id": np.asarray([target_id], dtype=np.int32),
        }
