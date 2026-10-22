"""MuJoCo collision-geometry signed-distance queries using invisible probes."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .base import SurfaceHit


SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PROBE_RADIUS_M = 0.0005


def probe_body_name(side: str, finger: str) -> str:
    return f"surface_probe_{side}_{finger}"


def probe_geom_name(side: str, finger: str) -> str:
    return f"surface_probe_{side}_{finger}_geom"


def add_surface_probes(spec: Any, radius_m: float = PROBE_RADIUS_M) -> None:
    """Append ten non-colliding mocap probe spheres to a parsed ``MjSpec``."""
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("surface probe radius must be positive and finite")
    for side in SIDES:
        for finger in FINGERS:
            body = spec.worldbody.add_body(
                name=probe_body_name(side, finger),
                pos=[0.0, 0.0, 0.0],
                mocap=True,
            )
            body.add_geom(
                name=probe_geom_name(side, finger),
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[float(radius_m), 0.0, 0.0],
                contype=0,
                conaffinity=0,
                group=5,
                rgba=[0.0, 0.0, 0.0, 0.0],
                margin=0.0,
            )


class MjGeomSurface:
    """Query selected MuJoCo object geoms without altering simulation data.

    ``update`` copies the live state into a dedicated ``MjData``. A query then
    moves the invisible probes together, runs kinematics once, and asks
    ``mj_geomDistance`` for the nearest selected object geom per probe.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        object_geom_names: list[str],
        distmax_m: float = 0.03,
        probe_radius_m: float = PROBE_RADIUS_M,
    ) -> None:
        if not np.isfinite(distmax_m) or distmax_m <= 0.0:
            raise ValueError("surface distmax must be positive and finite")
        if not np.isfinite(probe_radius_m) or probe_radius_m <= 0.0:
            raise ValueError("surface probe radius must be positive and finite")
        if len(set(object_geom_names)) != len(object_geom_names):
            raise ValueError("surface object_geom_names contains duplicates")

        self.model = model
        self.distmax_m = float(distmax_m)
        self.probe_radius_m = float(probe_radius_m)
        self.object_geom_ids: list[int] = []
        for name in object_geom_names:
            geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            if geom_id < 0:
                raise ValueError(f"surface object geom {name!r} is missing from the MuJoCo model")
            if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
                raise ValueError(f"surface object geom {name!r} is excluded from collision")
            if abs(float(model.geom_margin[geom_id])) > 1.0e-9:
                raise ValueError(
                    f"surface object geom {name!r} must have margin=0 to match its contact surface"
                )
            self.object_geom_ids.append(geom_id)

        self.probe_geom_ids: list[int] = []
        self.probe_mocap_ids: list[int] = []
        for side in SIDES:
            for finger in FINGERS:
                body_id = int(
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        probe_body_name(side, finger),
                    )
                )
                geom_id = int(
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        probe_geom_name(side, finger),
                    )
                )
                if body_id < 0 or geom_id < 0:
                    raise ValueError(
                        "surface probe bodies are absent; load the scene with "
                        "add_surface_probes() before compiling it"
                    )
                mocap_id = int(model.body_mocapid[body_id])
                if mocap_id < 0:
                    raise ValueError(f"surface probe {probe_body_name(side, finger)!r} is not mocap")
                self.probe_geom_ids.append(geom_id)
                self.probe_mocap_ids.append(mocap_id)

        self.query_data = mujoco.MjData(model)
        self._has_state = False
        self._previous_normals: list[np.ndarray | None] = [None] * len(self.probe_geom_ids)

    def update(self, state: mujoco.MjData) -> None:
        """Copy generalized and mocap positions; leave the live simulator untouched."""
        if state.qpos.shape != self.query_data.qpos.shape:
            raise ValueError("surface query state does not match the MuJoCo model")
        np.copyto(self.query_data.qpos, state.qpos)
        if state.mocap_pos.shape != self.query_data.mocap_pos.shape:
            raise ValueError("surface query mocap state does not match the MuJoCo model")
        np.copyto(self.query_data.mocap_pos, state.mocap_pos)
        np.copyto(self.query_data.mocap_quat, state.mocap_quat)
        self._has_state = True

    def query(
        self,
        points_world: np.ndarray,
        *,
        probe_offset: int = 0,
    ) -> list[SurfaceHit]:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"surface points must have shape (K, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("surface query points contain non-finite coordinates")
        if probe_offset < 0 or probe_offset + len(points) > len(self.probe_geom_ids):
            raise ValueError("surface query points exceed the available probe set")
        if not self._has_state:
            raise RuntimeError("call MjGeomSurface.update() before query()")

        if not self.object_geom_ids:
            return [self._invalid_hit() for _ in points]

        for index, point in enumerate(points):
            mocap_id = self.probe_mocap_ids[probe_offset + index]
            self.query_data.mocap_pos[mocap_id] = point
        mujoco.mj_kinematics(self.model, self.query_data)

        hits: list[SurfaceHit] = []
        for index, point in enumerate(points):
            probe_index = probe_offset + index
            probe_geom = self.probe_geom_ids[probe_index]
            best_distance = self.distmax_m
            best_surface_point: np.ndarray | None = None
            best_object_id = -1
            fromto = np.zeros(6, dtype=np.float64)
            for object_geom in self.object_geom_ids:
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        self.query_data,
                        probe_geom,
                        object_geom,
                        self.distmax_m,
                        fromto,
                    )
                )
                if distance < best_distance:
                    best_distance = distance
                    best_surface_point = fromto[3:].copy()
                    best_object_id = object_geom

            if best_surface_point is None:
                hits.append(self._invalid_hit())
                continue

            signed_distance = best_distance + self.probe_radius_m
            delta = point - best_surface_point
            norm = float(np.linalg.norm(delta))
            if norm < 1.0e-6:
                normal = self._previous_normals[probe_index]
                if normal is None:
                    hits.append(self._invalid_hit())
                    continue
            else:
                sign = 1.0 if signed_distance >= 0.0 else -1.0
                normal = sign * delta / norm
            normal = np.asarray(normal, dtype=np.float64)
            normal_norm = float(np.linalg.norm(normal))
            if not np.isfinite(normal_norm) or normal_norm < 1.0e-8:
                hits.append(self._invalid_hit())
                continue
            normal = normal / normal_norm
            self._previous_normals[probe_index] = normal.copy()
            hits.append(
                SurfaceHit(
                    d=float(signed_distance),
                    s=best_surface_point,
                    n=normal,
                    obj_id=best_object_id,
                    valid=True,
                )
            )
        return hits

    def _invalid_hit(self) -> SurfaceHit:
        return SurfaceHit(
            d=self.distmax_m,
            s=None,
            n=None,
            obj_id=-1,
            valid=False,
        )
