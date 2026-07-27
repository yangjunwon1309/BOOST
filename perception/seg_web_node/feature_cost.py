"""TASC-style convex intent features for discrete goal belief updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-9


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


def robust_standardize(values: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1 or not np.isfinite(arr).all():
        return np.zeros_like(arr)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad < eps:
        return arr - median
    return (arr - median) / (1.4826 * mad + eps)


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
    w_distance: float = 1.0
    w_ray: float = 0.8
    w_agreement: float = 0.6
    w_delta: float = 0.0
    distance_k: float = 1.5
    distance_delta: float = 0.10
    distance_c: float = 0.1
    ray_k: float = 1.0
    ray_delta: float = 0.15
    ray_c: float = 0.1
    agreement_lambda: float = 1.0
    agreement_kappa: float = 1.0
    delta_lambda: float = 0.3
    min_user_speed_mps: float = 0.005

    @classmethod
    def from_dict(cls, data: dict | None) -> "FeatureConfig":
        data = dict(data or {})
        features = dict(data.get("features", data))
        distance = dict(features.get("distance", {}))
        ray = dict(features.get("ray", {}))
        agreement = dict(features.get("agreement", {}))
        delta = dict(features.get("delta", {}))
        return cls(
            action_apply_time=float(data.get("action_apply_time", 0.01)),
            w_distance=float(features.get("w_distance", 1.0)),
            w_ray=float(features.get("w_ray", 0.8)),
            w_agreement=float(features.get("w_agreement", 0.6)),
            w_delta=float(features.get("w_delta", 0.0)),
            distance_k=float(distance.get("k", 1.5)),
            distance_delta=float(distance.get("delta", 0.10)),
            distance_c=float(distance.get("c", 0.1)),
            ray_k=float(ray.get("k_r", ray.get("k", 1.0))),
            ray_delta=float(ray.get("delta_r", ray.get("delta", 0.15))),
            ray_c=float(ray.get("c_r", ray.get("c", 0.1))),
            agreement_lambda=float(agreement.get("lambda_a", agreement.get("lambda", 1.0))),
            agreement_kappa=float(agreement.get("kappa", 1.0)),
            delta_lambda=float(delta.get("lambda_d", delta.get("lambda", 0.3))),
            min_user_speed_mps=float(features.get("min_user_speed_mps", 0.005)),
        )


class MultiFeatureCost:
    """Weighted distance, ray, and agreement advantage for one 3-D target."""

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

    def ray_advantage(
        self,
        target: np.ndarray,
        ray_origin: np.ndarray | None,
        ray_direction: np.ndarray | None,
    ) -> float:
        if ray_origin is None or ray_direction is None:
            return 0.0
        direction = unit(ray_direction)
        to_target = unit(np.asarray(target, dtype=np.float64).reshape(3) - ray_origin)
        if direction is None or to_target is None:
            return 0.0
        cos_psi = float(np.clip(np.dot(direction, to_target), -1.0, 1.0))
        e_ang = 1.0 - cos_psi
        cfg = self.config
        return -huber_like(e_ang, k=cfg.ray_k, delta=cfg.ray_delta, c=cfg.ray_c)

    def agreement_advantage(
        self,
        target: np.ndarray,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None,
        agreement_direction: np.ndarray | None = None,
    ) -> float:
        if user_translation is None:
            return 0.0
        u = np.asarray(user_translation, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(u)) < self.config.min_user_speed_mps:
            return 0.0
        if agreement_direction is None:
            direction_source = np.asarray(target, dtype=np.float64).reshape(3) - eef_position
        else:
            direction_source = agreement_direction
        target_dir = unit(direction_source)
        if target_dir is None:
            return 0.0
        u_opt = self.config.agreement_kappa * target_dir
        error = u - u_opt
        return -self.config.agreement_lambda * float(np.dot(error, error))

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

    def advantage(
        self,
        target: np.ndarray,
        *,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None = None,
        ray_origin: np.ndarray | None = None,
        ray_direction: np.ndarray | None = None,
        agreement_direction: np.ndarray | None = None,
    ) -> float:
        return float(
            self.breakdown(
                target,
                eef_position=eef_position,
                user_translation=user_translation,
                ray_origin=ray_origin,
                ray_direction=ray_direction,
                agreement_direction=agreement_direction,
            )["weighted_total"]
        )

    def breakdown(
        self,
        target: np.ndarray,
        *,
        eef_position: np.ndarray,
        user_translation: np.ndarray | None = None,
        ray_origin: np.ndarray | None = None,
        ray_direction: np.ndarray | None = None,
        agreement_direction: np.ndarray | None = None,
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

        ray_cos = None
        ray_error = None
        ray_cost = 0.0
        if ray_origin is not None and ray_direction is not None:
            direction = unit(ray_direction)
            to_target = unit(target - np.asarray(ray_origin, dtype=np.float64).reshape(3))
            if direction is not None and to_target is not None:
                ray_cos = float(np.clip(np.dot(direction, to_target), -1.0, 1.0))
                ray_error = 1.0 - ray_cos
                ray_cost = huber_like(
                    ray_error, k=cfg.ray_k, delta=cfg.ray_delta, c=cfg.ray_c
                )
        a_ray = -ray_cost

        agreement_dir = None
        agreement_cost = 0.0
        u_opt = None
        user_speed = float(np.linalg.norm(user))
        if user_translation is not None and user_speed >= cfg.min_user_speed_mps:
            if agreement_direction is None:
                direction_source = target - eef_position
            else:
                direction_source = agreement_direction
            target_dir = unit(direction_source)
            if target_dir is not None:
                agreement_dir = target_dir
                u_opt_arr = cfg.agreement_kappa * target_dir
                error = user - u_opt_arr
                agreement_cost = cfg.agreement_lambda * float(np.dot(error, error))
                u_opt = u_opt_arr.tolist()
        a_agreement = -agreement_cost

        a_delta = self.delta_advantage(target, eef_position, user_translation)
        weighted = {
            "distance": cfg.w_distance * a_dist,
            "ray": cfg.w_ray * a_ray,
            "agreement": cfg.w_agreement * a_agreement,
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
            "ray_cos": ray_cos,
            "ray_error_1_minus_cos": ray_error,
            "ray_cost": ray_cost,
            "ray_advantage": a_ray,
            "agreement_direction": (
                agreement_dir.tolist() if agreement_dir is not None else None
            ),
            "agreement_u_opt_mps": u_opt,
            "agreement_cost": agreement_cost,
            "agreement_advantage": a_agreement,
            "delta_advantage": a_delta,
            "weights": {
                "distance": cfg.w_distance,
                "ray": cfg.w_ray,
                "agreement": cfg.w_agreement,
                "delta": cfg.w_delta,
            },
            "weighted_contributions": weighted,
            "weighted_total": float(sum(weighted.values())),
        }
