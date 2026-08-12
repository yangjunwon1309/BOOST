"""TASC-style convex intent features for discrete goal belief updates."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np


EPS = 1.0e-9
LOG = logging.getLogger(__name__)


def unit(vector: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= EPS:
        return None
    return arr / norm


def huber_like(error: float, *, k: float, delta: float, c: float = 0.0) -> float:
    e = max(0.0, float(error))
    delta = max(EPS, float(delta))
    k = float(k)
    c = float(c)
    if e <= delta:
        return 0.5 * (k / delta) * e * e + c * e
    linear_slope = k + c
    intercept = 0.5 * k * delta
    return linear_slope * e - intercept


def apply_translation_action(
    eef_position: np.ndarray,
    user_translation: np.ndarray,
    action_apply_time: float,
) -> np.ndarray:
    return (
        np.asarray(eef_position, dtype=np.float64).reshape(3)
        + np.asarray(user_translation, dtype=np.float64).reshape(3)
        * max(0.0, float(action_apply_time))
    )


def robust_standardize(
    values: np.ndarray,
    eps: float = 1.0e-6,
    *,
    min_signal_spread: float = 0.002,
    scale_floor: float = 0.01,
    clip: float = 2.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1 or not np.isfinite(arr).all():
        return np.zeros_like(arr)
    median = float(np.median(arr))
    spread = float(np.max(arr) - np.min(arr))
    if spread < max(0.0, float(min_signal_spread)):
        return np.zeros_like(arr)
    mad = float(np.median(np.abs(arr - median)))
    scale = max(1.4826 * mad, float(scale_floor), eps)
    standardized = (arr - median) / scale
    clip_limit = float(clip)
    if clip_limit > 0.0 and np.isfinite(clip_limit):
        standardized = np.clip(standardized, -clip_limit, clip_limit)
    return standardized


def logsumexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    peak = float(np.max(arr))
    return peak + float(np.log(np.sum(np.exp(arr - peak))))


def normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(log_probs, dtype=np.float64)
    return arr - logsumexp(arr)


def clip_log_probs(log_probs: np.ndarray, max_prob: float) -> np.ndarray:
    arr = normalize_log_probs(log_probs)
    max_prob = float(np.clip(max_prob, 0.5, 0.999999))
    probs = np.exp(arr)
    peak_index = int(np.argmax(probs))
    if probs[peak_index] <= max_prob:
        return arr
    if probs.size <= 1:
        return np.array([0.0], dtype=np.float64)
    remainder = 1.0 - max_prob
    others = np.delete(probs, peak_index)
    other_sum = float(np.sum(others))
    clipped = np.empty_like(probs)
    clipped[peak_index] = max_prob
    if other_sum <= EPS:
        clipped[:] = remainder / (probs.size - 1)
        clipped[peak_index] = max_prob
    else:
        scale = remainder / other_sum
        for index, prob in enumerate(probs):
            clipped[index] = max_prob if index == peak_index else prob * scale
    return normalize_log_probs(np.log(np.maximum(clipped, EPS)))


@dataclass(frozen=True)
class FeatureConfig:
    action_apply_time: float = 0.01
    w_distance: float = 1.4
    w_progress: float = 1.0
    w_agreement: float = 0.4
    w_reach: float = 1.0
    w_delta: float = 0.0
    distance_k: float = 1.5
    distance_delta: float = 0.10
    distance_c: float = 0.1
    agreement_lambda: float = 1.0
    agreement_kappa: float = 0.25
    delta_lambda: float = 0.3
    min_user_speed_mps: float = 0.005
    reach_min_m: float = 0.18
    reach_max_m: float = 0.55
    reach_k: float = 1.0
    reach_delta: float = 0.10
    w_orientation: float = 0.8
    ori_align_k: float = 1.0
    ori_align_delta: float = 0.30
    ori_hold_lambda: float = 1.0
    ori_converge_lambda: float = 0.5
    ori_omega_ref_rps: float = 0.5
    min_hand_omega_rps: float = 0.02

    @classmethod
    def from_dict(cls, data: dict | None) -> "FeatureConfig":
        data = dict(data or {})
        features = dict(data.get("features", data))
        distance = dict(features.get("distance", {}))
        ray = dict(features.get("ray", {}))
        if ray or "w_ray" in features:
            LOG.warning("TASC ray feature is deprecated and ignored; camera rays remain geometry-only")
        agreement = dict(features.get("agreement", {}))
        delta = dict(features.get("delta", {}))
        orientation = dict(features.get("orientation", {}))
        return cls(
            action_apply_time=float(data.get("action_apply_time", 0.01)),
            w_distance=float(features.get("w_distance", 1.4)),
            w_progress=float(features.get("w_progress", 1.0)),
            w_agreement=float(features.get("w_agreement", 0.4)),
            w_reach=float(features.get("w_reach", 1.0)),
            w_delta=float(features.get("w_delta", 0.0)),
            distance_k=float(distance.get("k", 1.5)),
            distance_delta=float(distance.get("delta", 0.10)),
            distance_c=float(distance.get("c", 0.1)),
            agreement_lambda=float(agreement.get("lambda_a", agreement.get("lambda", 1.0))),
            agreement_kappa=float(agreement.get("kappa", 0.25)),
            delta_lambda=float(delta.get("lambda_d", delta.get("lambda", 0.3))),
            min_user_speed_mps=float(features.get("min_user_speed_mps", 0.005)),
            reach_min_m=float(features.get("reach_min_m", 0.18)),
            reach_max_m=float(features.get("reach_max_m", 0.55)),
            reach_k=float(features.get("reach_k", 1.0)),
            reach_delta=float(features.get("reach_delta", 0.10)),
            w_orientation=float(features.get("w_orientation", 0.8)),
            ori_align_k=float(orientation.get("align_k", orientation.get("k", 1.0))),
            ori_align_delta=float(orientation.get("align_delta", orientation.get("delta", 0.30))),
            ori_hold_lambda=float(orientation.get("hold_lambda", 1.0)),
            ori_converge_lambda=float(orientation.get("converge_lambda", 0.5)),
            ori_omega_ref_rps=float(orientation.get("omega_ref_rps", 0.5)),
            min_hand_omega_rps=float(orientation.get("min_hand_omega_rps", 0.02)),
        )


class MultiFeatureCost:
    """Weighted hand-centric intent advantages for one 3-D target."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.previous_direction: np.ndarray | None = None

    def distance_advantage(
        self,
        target: np.ndarray,
        eef_position: np.ndarray,
        eef_after_action: np.ndarray,
    ) -> float:
        cfg = self.config
        target = np.asarray(target, dtype=np.float64).reshape(3)
        d_now = float(np.linalg.norm(np.asarray(eef_position).reshape(3) - target))
        d_after = float(np.linalg.norm(np.asarray(eef_after_action).reshape(3) - target))
        return huber_like(
            d_now, k=cfg.distance_k, delta=cfg.distance_delta, c=cfg.distance_c
        ) - huber_like(
            d_after, k=cfg.distance_k, delta=cfg.distance_delta, c=cfg.distance_c
        )

    def agreement_advantage(
        self,
        target: np.ndarray,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None,
    ) -> float:
        if user_translation is None:
            return 0.0
        u = np.asarray(user_translation, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(u)) < self.config.min_user_speed_mps:
            return 0.0
        target_dir = unit(np.asarray(target, dtype=np.float64).reshape(3) - eef_position)
        if target_dir is None:
            return 0.0
        u_opt = self.config.agreement_kappa * target_dir
        error = u - u_opt
        return -self.config.agreement_lambda * float(np.dot(error, error))

    def progress_advantage(
        self,
        target: np.ndarray,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None,
    ) -> float:
        """Speed-invariant directed progress from the hand to a candidate."""
        if user_translation is None:
            return 0.0
        velocity = np.asarray(user_translation, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(velocity)) < self.config.min_user_speed_mps:
            return 0.0
        direction = unit(velocity)
        target_direction = unit(
            np.asarray(target, dtype=np.float64).reshape(3)
            - np.asarray(eef_position, dtype=np.float64).reshape(3)
        )
        if direction is None or target_direction is None:
            return 0.0
        return float(np.dot(direction, target_direction))

    def reach_advantage(
        self, target: np.ndarray, shoulder_position: np.ndarray | None
    ) -> tuple[float, float | None]:
        """Penalize candidates outside the measured shoulder reach annulus."""
        if shoulder_position is None:
            return 0.0, None
        radius = float(np.linalg.norm(
            np.asarray(target, dtype=np.float64).reshape(3)
            - np.asarray(shoulder_position, dtype=np.float64).reshape(3)
        ))
        outside = max(0.0, radius - self.config.reach_max_m)
        inside = max(0.0, self.config.reach_min_m - radius)
        penalty = huber_like(outside, k=self.config.reach_k, delta=self.config.reach_delta)
        penalty += huber_like(inside, k=self.config.reach_k, delta=self.config.reach_delta)
        return -float(penalty), radius

    def delta_advantage(
        self,
        target: np.ndarray,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None,
    ) -> float:
        if self.config.w_delta == 0.0 or user_translation is None:
            return 0.0
        direction = unit(user_translation)
        if direction is None:
            return 0.0
        if self.previous_direction is None:
            self.previous_direction = direction
            return 0.0
        target_dir = unit(np.asarray(target, dtype=np.float64).reshape(3) - eef_position)
        if target_dir is None:
            return 0.0
        delta = direction - self.previous_direction
        return self.config.delta_lambda * float(np.dot(delta, target_dir))

    def finish_tick(self, user_translation: np.ndarray | None) -> None:
        if user_translation is None:
            return
        direction = unit(user_translation)
        if direction is not None:
            self.previous_direction = direction

    def orientation_breakdown(
        self,
        *,
        surface_normal: np.ndarray,
        hand_axis: np.ndarray,
        hand_angular_velocity: np.ndarray | None,
        relation: str,
    ) -> dict:
        """Return the independent orientation-intent feature diagnostic."""
        normal = unit(surface_normal)
        axis = unit(hand_axis)
        if normal is None or axis is None:
            return {
                "cos_c": None, "alignment_error": None, "relation": str(relation),
                "align_advantage": 0.0, "hold_advantage": 0.0,
                "converge_advantage": 0.0, "omega_norm": None,
                "weighted_total": 0.0,
            }
        relation = str(relation).lower()
        if relation not in {"parallel", "perpendicular"}:
            raise ValueError(f"unsupported orientation relation: {relation}")
        cfg = self.config
        cos_c = float(np.clip(np.dot(axis, normal), -1.0, 1.0))
        # Squared, sign-invariant alignment error is differentiable through
        # c=0, the perpendicular optimum.
        alignment_error = 1.0 - cos_c * cos_c if relation == "parallel" else cos_c * cos_c
        align = -huber_like(alignment_error, k=cfg.ori_align_k, delta=cfg.ori_align_delta)
        hold = 0.0
        converge = 0.0
        omega_norm: float | None = None
        if hand_angular_velocity is not None:
            omega = np.asarray(hand_angular_velocity, dtype=np.float64).reshape(3)
            if np.isfinite(omega).all():
                omega_norm = float(np.linalg.norm(omega))
                if omega_norm < cfg.min_hand_omega_rps:
                    hold = cfg.ori_hold_lambda * (1.0 - alignment_error)
                else:
                    dc = float(np.dot(np.cross(omega, axis), normal))
                    de = -2.0 * cos_c * dc if relation == "parallel" else 2.0 * cos_c * dc
                    converge = -cfg.ori_converge_lambda * de / max(EPS, cfg.ori_omega_ref_rps)
        total = align + hold + converge
        return {
            "cos_c": cos_c,
            "alignment_error": float(alignment_error),
            "relation": relation,
            "align_advantage": float(align),
            "hold_advantage": float(hold),
            "converge_advantage": float(converge),
            "omega_norm": omega_norm,
            "weighted_total": float(cfg.w_orientation * total),
        }

    def orientation_advantage(self, **kwargs) -> float:
        return float(self.orientation_breakdown(**kwargs)["weighted_total"])

    def advantage(
        self,
        target: np.ndarray,
        *,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None = None,
        shoulder_position: np.ndarray | None = None,
    ) -> float:
        return float(
            self.breakdown(
                target,
                eef_position=eef_position,
                user_translation=user_translation,
                shoulder_position=shoulder_position,
            )["weighted_total"]
        )

    def breakdown(
        self,
        target: np.ndarray,
        *,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None = None,
        shoulder_position: np.ndarray | None = None,
    ) -> dict:
        cfg = self.config
        target = np.asarray(target, dtype=np.float64).reshape(3)
        eef_position = np.asarray(eef_position, dtype=np.float64).reshape(3)
        user = (
            np.zeros(3, dtype=np.float64)
            if user_translation is None
            else np.asarray(user_translation, dtype=np.float64).reshape(3)
        )
        eef_after = apply_translation_action(
            eef_position, user, cfg.action_apply_time
        )

        d_now = float(np.linalg.norm(eef_position - target))
        d_after = float(np.linalg.norm(eef_after - target))
        v_dist = huber_like(
            d_now, k=cfg.distance_k, delta=cfg.distance_delta, c=cfg.distance_c
        )
        v_dist_after = huber_like(
            d_after, k=cfg.distance_k, delta=cfg.distance_delta, c=cfg.distance_c
        )
        a_dist = v_dist - v_dist_after

        agreement_dir = None
        agreement_cost = 0.0
        u_opt = None
        user_speed = float(np.linalg.norm(user))
        if user_translation is not None and user_speed >= cfg.min_user_speed_mps:
            target_dir = unit(target - eef_position)
            if target_dir is not None:
                agreement_dir = target_dir
                u_opt_arr = cfg.agreement_kappa * target_dir
                error = user - u_opt_arr
                agreement_cost = cfg.agreement_lambda * float(np.dot(error, error))
                u_opt = u_opt_arr.tolist()
        a_agreement = -agreement_cost
        a_progress = self.progress_advantage(target, eef_position, user_translation)
        a_reach, shoulder_distance = self.reach_advantage(target, shoulder_position)

        a_delta = self.delta_advantage(target, eef_position, user_translation)
        weighted = {
            "distance": cfg.w_distance * a_dist,
            "ray": 0.0,
            "progress": cfg.w_progress * a_progress,
            "agreement": cfg.w_agreement * a_agreement,
            "reach": cfg.w_reach * a_reach,
            "delta": cfg.w_delta * a_delta,
        }
        return {
            "target_m": target.tolist(),
            "eef_position_m": eef_position.tolist(),
            "eef_after_action_m": eef_after.tolist(),
            "user_translation_mps": (
                np.asarray(user_translation, dtype=np.float64).reshape(3).tolist()
                if user_translation is not None
                else None
            ),
            "distance_m": d_now,
            "distance_after_action_m": d_after,
            "distance_value": v_dist,
            "distance_value_after_action": v_dist_after,
            "distance_advantage": a_dist,
            "ray_cos": None,
            "ray_error_1_minus_cos": None,
            "ray_cost": 0.0,
            "ray_advantage": 0.0,
            "agreement_direction": (
                agreement_dir.tolist() if agreement_dir is not None else None
            ),
            "agreement_u_opt_mps": u_opt,
            "agreement_cost": agreement_cost,
            "agreement_advantage": a_agreement,
            "progress_advantage": a_progress,
            "shoulder_position_m": (
                None if shoulder_position is None
                else np.asarray(shoulder_position, dtype=np.float64).reshape(3).tolist()
            ),
            "shoulder_target_distance_m": shoulder_distance,
            "reach_advantage": a_reach,
            "delta_advantage": a_delta,
            "weights": {
                "distance": cfg.w_distance,
                "ray": 0.0,
                "progress": cfg.w_progress,
                "agreement": cfg.w_agreement,
                "reach": cfg.w_reach,
                "delta": cfg.w_delta,
            },
            "weighted_contributions": weighted,
            "weighted_total": float(sum(weighted.values())),
        }
