"""Contact-guided target generation layered on Mimic's sampling retargeter."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from mimic_retargeter_lab.retargeting.online import SamplingBasedRetargeter

from .surface.mujoco_client import MujocoSurfaceClient


FINGERS = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")


class ContactGuidedRetargeter(SamplingBasedRetargeter):
    """Use surface projections to adjust SBR keyvector targets before iCEM.

    The inherited robust loss and sampling loop are unchanged. Surface queries
    happen once per incoming hand frame in ``_compute_dexpilot_loss_terms``;
    only the resulting target array enters the batched candidate loss.
    """

    def __init__(
        self,
        *args: Any,
        side: str,
        surface_client: MujocoSurfaceClient,
        contact_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError(f"invalid contact-guidance hand side {side!r}")
        self.side = side
        self.surface_client = surface_client
        self._contact_on = float(contact_config.get("d_on", 0.01))
        self._contact_off = float(contact_config.get("d_off", 0.015))
        self._contact_epsilon = float(contact_config.get("epsilon", 0.003))
        if not (
            np.isfinite(self._contact_on)
            and np.isfinite(self._contact_off)
            and self._contact_on < self._contact_off
        ):
            raise ValueError("contact thresholds must be finite with d_on < d_off")
        if not np.isfinite(self._contact_epsilon) or self._contact_epsilon < 0.0:
            raise ValueError("contact epsilon must be finite and non-negative")
        self._contact_sequence = -1
        self._contact_active = np.zeros(5, dtype=bool)
        self._contact_alpha = np.zeros(5, dtype=np.float32)
        self._contact_distances = np.full(5, np.nan, dtype=np.float32)
        self._contact_geom_ids = np.full(5, -1, dtype=np.int32)
        super().__init__(*args, **kwargs)

    def set_input_sequence(self, sequence: int) -> None:
        self._contact_sequence = int(sequence)

    @property
    def contact_debug(self) -> dict[str, np.ndarray]:
        return {
            "alpha": self._contact_alpha.copy(),
            "distance": self._contact_distances.copy(),
            "active": self._contact_active.copy(),
            "geom_id": self._contact_geom_ids.copy(),
        }

    def _compute_dexpilot_loss_terms(
        self, src_keyvectors: dict[str, np.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        targets_jax, weights = super()._compute_dexpilot_loss_terms(src_keyvectors)
        targets = np.asarray(targets_jax, dtype=np.float32).copy()
        tip_keys = [f"WRIST_to_{finger}_TIP" for finger in FINGERS]
        tips_wrist = np.stack(
            [np.asarray(src_keyvectors[key], dtype=np.float32).reshape(3) for key in tip_keys]
        )

        reply = self.surface_client.query_hand_tips(
            self.side,
            tips_wrist,
            sequence=self._contact_sequence,
        )
        alpha = np.zeros(5, dtype=np.float32)
        # Invalid or unavailable surface samples force alpha to zero for this
        # frame, but do not erase the hysteresis state needed by the next valid
        # sample.
        active = self._contact_active.copy()
        distances = np.full(5, np.nan, dtype=np.float32)
        geom_ids = np.full(5, -1, dtype=np.int32)
        projected_tips = tips_wrist.copy()
        if reply is not None:
            rotation_world_from_wrist = reply.wrist_rotation_world
            for index, hit in enumerate(reply.hits):
                if not hit.valid or hit.n is None or not np.isfinite(hit.d):
                    continue
                distances[index] = float(hit.d)
                geom_ids[index] = int(hit.obj_id)
                was_active = bool(self._contact_active[index])
                if not was_active and hit.d < self._contact_on:
                    was_active = True
                elif was_active and hit.d > self._contact_off:
                    was_active = False
                active[index] = was_active
                if not was_active:
                    continue
                blend = np.clip(
                    (self._contact_off - hit.d)
                    / (self._contact_off - self._contact_on),
                    0.0,
                    1.0,
                )
                alpha[index] = float(blend)
                normal_wrist = rotation_world_from_wrist.T @ np.asarray(
                    hit.n, dtype=np.float32
                )
                norm = float(np.linalg.norm(normal_wrist))
                if not np.isfinite(norm) or norm < 1.0e-8:
                    active[index] = bool(self._contact_active[index])
                    alpha[index] = 0.0
                    continue
                normal_wrist /= norm
                projected_tips[index] = tips_wrist[index] - (
                    alpha[index] * (float(hit.d) + self._contact_epsilon)
                ) * normal_wrist

        self._contact_active = active
        self._contact_alpha = alpha
        self._contact_distances = distances
        self._contact_geom_ids = geom_ids

        target_indices = {key: index for index, key in enumerate(self._tgt_keys)}
        # Blend the existing wrist-to-tip target toward the surface-projected
        # fingertip. At alpha=0 this leaves every upstream target untouched.
        for index, finger in enumerate(FINGERS):
            key = f"WRIST_to_{finger}_TIP"
            target_index = target_indices.get(key)
            if target_index is None or alpha[index] <= 0.0:
                continue
            amount = alpha[index]
            targets[target_index] = (
                (1.0 - amount) * targets[target_index]
                + amount * projected_tips[index]
            )

        # Preserve the upstream DexPilot pinch target and its weight, while
        # blending its target vector toward the projected fingertip pair.
        for index, finger in enumerate(FINGERS[1:], start=1):
            key = f"THUMB_TIP_to_{finger}_TIP"
            target_index = target_indices.get(key)
            amount = float(max(alpha[0], alpha[index]))
            if target_index is None or amount <= 0.0:
                continue
            projected_vector = projected_tips[index] - projected_tips[0]
            targets[target_index] = (
                (1.0 - amount) * targets[target_index]
                + amount * projected_vector
            )

        return jnp.asarray(targets, dtype=jnp.float32), weights
