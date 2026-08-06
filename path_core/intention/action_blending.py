from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


LEFT_WRIST_LINK = "left_wrist_yaw_link"
RIGHT_WRIST_LINK = "right_wrist_yaw_link"


@dataclass(frozen=True)
class SharedControlConfig:
    rho1: float = 0.55
    rho2: float = 0.85
    alpha_rate_up: float = 1.0
    alpha_rate_down: float = 3.0
    v_transfer_max: float = 0.25
    startup_hold_s: float = 2.0
    stale_after_s: float = 0.3
    d_max_m: float = 0.9
    reset_alpha_on_target_switch: bool = True
    ref_lpf_alpha: float = 0.3
    hand_h_margin_m: float = 0.05
    hand_switch_confirm_n: int = 10
    lock_hand_while_active: bool = True
    orientation_mode: str = "gmr_prior"
    waist_mode: str = "sonic_passthrough"


@dataclass(frozen=True)
class EETarget:
    position: np.ndarray
    confidence: float
    orientation_wxyz: np.ndarray | None = None
    object_id: str = ""
    grip_index: int = -1
    goal_uid: str = ""
    timestamp_us: int = 0

    @property
    def key(self) -> tuple[str]:
        # The tracked object identity is stable; the selected grip candidate can
        # legitimately jitter between neighboring surface samples on consecutive
        # RGB-D frames.  Treating grip_index as part of the target identity causes
        # alpha to reset repeatedly even when the user is still tracking the same
        # object.  The same is true for goal_uid in continuous streaming mode:
        # some producers stamp/refresh it per send.  The grip/contact point is
        # instead smoothed by p_contact_lpf, while explicit reset/new_goal in the
        # state machine still resets the blender when the user selects a new
        # object.
        return (str(self.object_id),)


@dataclass
class SharedControlResult:
    p_ref_left: np.ndarray
    p_ref_right: np.ndarray
    p_human_left: np.ndarray
    p_human_right: np.ndarray
    p_contact: np.ndarray | None
    alpha: float
    alpha_raw: float
    alpha_rate_eff: float
    gap_m: float
    confidence: float
    hand: str
    target_link: str
    gate_reason: str
    target_switched: bool
    object_id: str
    grip_index: int

    @property
    def active_p_ref(self) -> np.ndarray | None:
        if self.hand == "left":
            return self.p_ref_left
        if self.hand == "right":
            return self.p_ref_right
        return None


def confidence_to_alpha(confidence: float, rho1: float, rho2: float) -> float:
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(c) or rho2 <= rho1:
        return 0.0
    if c <= rho1:
        return 0.0
    if c >= rho2:
        return 1.0
    return float((c - rho1) / (rho2 - rho1))


def _finite_vec3(values: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def explicit_orientation_wxyz(payload: dict | None) -> np.ndarray | None:
    """Return a normalized explicit target orientation, or None if unspecified.

    Perception targets historically include an identity quaternion even when
    they do not estimate object/EE orientation.  Requiring orientation_valid
    prevents such placeholders from overriding the live operator wrist pose.
    """

    if not isinstance(payload, dict) or payload.get("orientation_valid") is not True:
        return None
    value = payload.get("quaternion", payload.get("target_quaternion"))
    try:
        quat = np.asarray(value, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(quat).all() or norm <= 1.0e-12:
        return None
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def quaternion_wxyz_to_rotation_matrix(
    quaternion_wxyz: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    quat = explicit_orientation_wxyz(
        {"orientation_valid": True, "quaternion": quaternion_wxyz}
    )
    if quat is None:
        raise ValueError("invalid explicit target quaternion")
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def apply_explicit_orientation_to_wrist_targets(
    left_target_tf: np.ndarray,
    right_target_tf: np.ndarray,
    *,
    selected_hand: str,
    orientation_wxyz: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply an explicit waist-frame EE orientation to only the selected hand."""

    left = np.asarray(left_target_tf, dtype=np.float64).reshape(4, 4).copy()
    right = np.asarray(right_target_tf, dtype=np.float64).reshape(4, 4).copy()
    if orientation_wxyz is None or selected_hand not in {"left", "right"}:
        return left, right, "human_wrist_prior"
    rotation = quaternion_wxyz_to_rotation_matrix(orientation_wxyz)
    if selected_hand == "left":
        left[:3, :3] = rotation
    else:
        right[:3, :3] = rotation
    return left, right, "explicit_static_target"


@dataclass
class HandSelector:
    h_margin_m: float = 0.05
    switch_confirm_n: int = 10
    lock_while_active: bool = True
    selected: str = "none"
    pending: str = "none"
    pending_count: int = 0

    def reset(self) -> None:
        self.selected = "none"
        self.pending = "none"
        self.pending_count = 0

    def update(
        self,
        *,
        p_contact: np.ndarray,
        measured_left: np.ndarray,
        measured_right: np.ndarray,
        alpha: float,
    ) -> str:
        d_left = float(np.linalg.norm(measured_left - p_contact))
        d_right = float(np.linalg.norm(measured_right - p_contact))
        nearest = "left" if d_left <= d_right else "right"
        if self.selected not in {"left", "right"}:
            self.selected = nearest
            self.pending = "none"
            self.pending_count = 0
            return self.selected
        if self.lock_while_active and alpha > 1.0e-6:
            self.pending = "none"
            self.pending_count = 0
            return self.selected

        current_d = d_left if self.selected == "left" else d_right
        other = "right" if self.selected == "left" else "left"
        other_d = d_right if other == "right" else d_left
        if other_d < current_d - max(0.0, float(self.h_margin_m)):
            if self.pending == other:
                self.pending_count += 1
            else:
                self.pending = other
                self.pending_count = 1
            if self.pending_count >= max(1, int(self.switch_confirm_n)):
                self.selected = other
                self.pending = "none"
                self.pending_count = 0
        else:
            self.pending = "none"
            self.pending_count = 0
        return self.selected


@dataclass
class TaskSpaceBlender:
    config: SharedControlConfig = field(default_factory=SharedControlConfig)
    alpha: float = 0.0
    p_contact_lpf: np.ndarray | None = None
    target_key: tuple[str] | None = None
    started_us: int | None = None
    max_observed_confidence: float = 0.0
    hand_selector: HandSelector = field(init=False)

    def __post_init__(self) -> None:
        if self.config.orientation_mode != "gmr_prior":
            raise NotImplementedError("shared-control v3 only implements orientation_mode=gmr_prior")
        self.hand_selector = HandSelector(
            h_margin_m=self.config.hand_h_margin_m,
            switch_confirm_n=self.config.hand_switch_confirm_n,
            lock_while_active=self.config.lock_hand_while_active,
        )

    def reset(self) -> None:
        self.alpha = 0.0
        self.p_contact_lpf = None
        self.target_key = None
        self.started_us = None
        self.max_observed_confidence = 0.0
        self.hand_selector.reset()

    def update(
        self,
        *,
        target: EETarget | None,
        p_human_left: np.ndarray,
        p_human_right: np.ndarray,
        p_measured_left: np.ndarray,
        p_measured_right: np.ndarray,
        now_us: int,
        dt: float,
        blending_enabled: bool = True,
        preferred_hand: str | None = None,
    ) -> SharedControlResult:
        if self.started_us is None:
            self.started_us = int(now_us)
        p_human_left = _finite_vec3(p_human_left)
        p_human_right = _finite_vec3(p_human_right)
        p_measured_left = _finite_vec3(p_measured_left)
        p_measured_right = _finite_vec3(p_measured_right)
        zero = np.zeros(3, dtype=np.float64)
        if p_human_left is None:
            p_human_left = zero.copy()
        if p_human_right is None:
            p_human_right = zero.copy()
        if p_measured_left is None:
            p_measured_left = p_human_left.copy()
        if p_measured_right is None:
            p_measured_right = p_human_right.copy()

        confidence = float(target.confidence) if target is not None else 0.0
        if np.isfinite(confidence):
            self.max_observed_confidence = max(self.max_observed_confidence, confidence)
        alpha_raw = confidence_to_alpha(confidence, self.config.rho1, self.config.rho2)

        p_contact_raw = _finite_vec3(target.position if target is not None else None)
        target_switched = False
        if target is None or p_contact_raw is None:
            gate_reason = "no_target"
            p_contact = self.p_contact_lpf
        else:
            key = target.key
            target_switched = self.target_key is not None and key != self.target_key
            if target_switched and self.config.reset_alpha_on_target_switch:
                self.alpha = 0.0
                self.hand_selector.reset()
                self.p_contact_lpf = p_contact_raw.copy()
            self.target_key = key
            if self.p_contact_lpf is None:
                self.p_contact_lpf = p_contact_raw.copy()
            else:
                beta = float(np.clip(self.config.ref_lpf_alpha, 0.0, 1.0))
                self.p_contact_lpf = beta * p_contact_raw + (1.0 - beta) * self.p_contact_lpf
            p_contact = self.p_contact_lpf
            gate_reason = "ok"

        startup_elapsed = (int(now_us) - int(self.started_us)) / 1.0e6
        if not blending_enabled:
            gate_reason = "disabled"
        elif startup_elapsed < max(0.0, self.config.startup_hold_s):
            gate_reason = "startup_hold"
        elif target is not None and target.timestamp_us > 0:
            age_s = (int(now_us) - int(target.timestamp_us)) / 1.0e6
            if age_s > self.config.stale_after_s:
                gate_reason = "stale"

        requested_hand = str(preferred_hand or "").strip().lower()
        if requested_hand not in {"left", "right"}:
            requested_hand = ""

        hand = self.hand_selector.selected
        gap_m = float("nan")
        if p_contact is not None and gate_reason not in {"no_target"}:
            if requested_hand:
                # If the web/SAM3 target explicitly requests a wrist side, obey
                # that side.  Nearest-hand auto selection is useful for generic
                # targets, but it is surprising and unsafe when the UI says
                # "left wrist" while the bridge silently routes to the right arm.
                if self.hand_selector.selected != requested_hand:
                    if self.hand_selector.selected in {"left", "right"} and self.alpha > 1.0e-6:
                        self.alpha = 0.0
                    self.hand_selector.selected = requested_hand
                    self.hand_selector.pending = "none"
                    self.hand_selector.pending_count = 0
                hand = requested_hand
            else:
                hand = self.hand_selector.update(
                    p_contact=p_contact,
                    measured_left=p_measured_left,
                    measured_right=p_measured_right,
                    alpha=self.alpha,
                )
            p_human = p_human_left if hand == "left" else p_human_right
            gap_m = float(np.linalg.norm(p_contact - p_human))
            if gap_m > self.config.d_max_m:
                gate_reason = "gap_too_large"
        else:
            hand = "none"

        if gate_reason != "ok":
            alpha_raw = 0.0

        rate_down = max(0.0, float(self.config.alpha_rate_down))
        rate_up_nominal = max(0.0, float(self.config.alpha_rate_up))
        if np.isfinite(gap_m) and gap_m > 1.0e-9:
            rate_up_eff = min(
                rate_up_nominal,
                max(0.0, float(self.config.v_transfer_max)) / max(gap_m, 1.0e-6),
            )
        else:
            rate_up_eff = rate_up_nominal
        lower = self.alpha - rate_down * max(0.0, float(dt))
        upper = self.alpha + rate_up_eff * max(0.0, float(dt))
        self.alpha = float(np.clip(alpha_raw, lower, upper))
        self.alpha = float(np.clip(self.alpha, 0.0, 1.0)) if np.isfinite(self.alpha) else 0.0

        p_ref_left = p_human_left.copy()
        p_ref_right = p_human_right.copy()
        if p_contact is not None and hand == "left":
            p_ref_left = self.alpha * p_contact + (1.0 - self.alpha) * p_human_left
        elif p_contact is not None and hand == "right":
            p_ref_right = self.alpha * p_contact + (1.0 - self.alpha) * p_human_right

        target_link = LEFT_WRIST_LINK if hand != "right" else RIGHT_WRIST_LINK
        return SharedControlResult(
            p_ref_left=p_ref_left,
            p_ref_right=p_ref_right,
            p_human_left=p_human_left,
            p_human_right=p_human_right,
            p_contact=None if p_contact is None else p_contact.copy(),
            alpha=self.alpha,
            alpha_raw=float(alpha_raw),
            alpha_rate_eff=float(rate_up_eff),
            gap_m=float(gap_m),
            confidence=confidence if np.isfinite(confidence) else 0.0,
            hand=hand,
            target_link=target_link,
            gate_reason=gate_reason,
            target_switched=target_switched,
            object_id=str(target.object_id) if target is not None else "",
            grip_index=int(target.grip_index) if target is not None else -1,
        )
