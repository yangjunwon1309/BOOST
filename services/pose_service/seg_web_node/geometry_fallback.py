"""Bounded last-valid geometry hold for intermittent detector dropouts."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .grip_candidates import PersistentCylinderSurfaceTracker


class LastValidGeometryFallback:
    """Republish the last verified object geometry through short detector gaps."""

    def __init__(
        self,
        ttl_frames: int,
        score_decay: float,
        *,
        fallback_reason: str = "sam3_detection_missing",
    ) -> None:
        self.ttl_frames = max(0, int(ttl_frames))
        self.score_decay = float(np.clip(score_decay, 0.0, 1.0))
        self.fallback_reason = str(fallback_reason)
        self.entries: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _has_geometry(obj: dict[str, Any]) -> bool:
        center = obj.get("center_goal_m")
        if center is None:
            return False
        value = np.asarray(center, dtype=np.float64)
        return value.shape == (3,) and bool(np.isfinite(value).all())

    def complete_frame(
        self,
        current: list[dict[str, Any]],
        surface_tracker: PersistentCylinderSurfaceTracker,
        *,
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        if self.ttl_frames <= 0:
            return current
        output: list[dict[str, Any]] = []
        represented: set[str] = set()
        current_ids = {
            str(obj.get("track_id"))
            for obj in current
            if obj.get("track_id") is not None
        }

        for obj in current:
            track_id = str(obj.get("track_id", ""))
            if track_id and self._has_geometry(obj):
                fresh = copy.deepcopy(obj)
                fresh["geometry_fallback"] = False
                fresh["missing_frames"] = 0
                self.entries[track_id] = {
                    "object": fresh,
                    "missing": 0,
                    "base_score": float(obj.get("score", 0.0)),
                }
                output.append(fresh)
                represented.add(track_id)
                continue
            fallback = self._advance(track_id, surface_tracker, image_width, image_height)
            if fallback is not None:
                output.append(fallback)
                represented.add(track_id)
            else:
                output.append(obj)

        for track_id in list(self.entries):
            if track_id in represented or track_id in current_ids:
                continue
            fallback = self._advance(track_id, surface_tracker, image_width, image_height)
            if fallback is not None:
                output.append(fallback)
        return output

    def _advance(
        self,
        track_id: str,
        surface_tracker: PersistentCylinderSurfaceTracker,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any] | None:
        entry = self.entries.get(track_id)
        if entry is None:
            return None
        missing = int(entry["missing"]) + 1
        if missing > self.ttl_frames:
            del self.entries[track_id]
            return None
        cached = entry["object"]
        if (
            int(cached.get("mask_width", -1)) != int(image_width)
            or int(cached.get("mask_height", -1)) != int(image_height)
        ):
            del self.entries[track_id]
            return None
        entry["missing"] = missing
        fallback = copy.deepcopy(cached)
        fallback["score"] = float(entry["base_score"]) * self.score_decay**missing
        fallback["geometry_fallback"] = True
        fallback["fallback_reason"] = self.fallback_reason
        fallback["missing_frames"] = missing
        # Primitive coordinates were fitted in an older pelvis-identity frame.
        # Reusing them after pelvis motion would corrupt orientation belief.
        fallback.pop("surface_primitives", None)
        surface = surface_tracker.status(track_id)
        if surface is not None:
            fallback["surface_model"] = surface
        return fallback
