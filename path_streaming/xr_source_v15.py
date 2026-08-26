"""Source-neutral controller XR input for PATH v15.1.

This module intentionally has no dependency on the legacy Pico manager.  Both
the XRT comparison adapter and the Galaxy/WebXR server hand the same raw
OpenXR-basis SE(3) sample to the one waist-target geometry path below.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rotation


def _configure_vuer_server_tls() -> None:
    """Disable accidental mutual TLS for the self-hosted WebXR server.

    Vuer 0.0.60 requests a client certificate when no client CA is configured
    (``CERT_OPTIONAL``). Android/WebXR browsers may then offer a device
    certificate which the server cannot validate, closing the TLS handshake.
    PATH authenticates only the server certificate, so no-CA sessions must use
    ``CERT_NONE`` for client verification.
    """
    from vuer import base as vuer_base

    server_type = vuer_base.Server
    if getattr(server_type, "_path_tls_without_client_cert", False):
        return

    def run(self) -> None:
        async def init_server() -> None:
            runner = vuer_base.web.AppRunner(self.app)
            await runner.setup()
            if not self.cert:
                site = vuer_base.web.TCPSite(runner, self.host, self.port)
                await site.start()
                return

            ssl_context = vuer_base.ssl.create_default_context(
                vuer_base.ssl.Purpose.CLIENT_AUTH
            )
            ssl_context.load_cert_chain(certfile=self.cert, keyfile=self.key)
            if self.ca_cert:
                ssl_context.load_verify_locations(self.ca_cert)
                ssl_context.verify_mode = vuer_base.ssl.CERT_REQUIRED
            else:
                ssl_context.verify_mode = vuer_base.ssl.CERT_NONE
            site = vuer_base.web.TCPSite(
                runner, self.host, self.port, ssl_context=ssl_context
            )
            await site.start()

        event_loop = vuer_base.asyncio.get_event_loop()
        event_loop.run_until_complete(init_server())
        event_loop.run_forever()

    server_type.run = run
    server_type._path_tls_without_client_cert = True


T_ROBOT_OPENXR = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
T_OPENXR_ROBOT = np.linalg.inv(T_ROBOT_OPENXR)
WAIST_OFFSET_XYZ = np.array([0.15, 0.0, 0.45], dtype=np.float64)
CONST_HEAD_POSE = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]],
    dtype=np.float64,
)
CONST_LEFT_ARM_POSE = np.array(
    [[1, 0, 0, -0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]],
    dtype=np.float64,
)
CONST_RIGHT_ARM_POSE = np.array(
    [[1, 0, 0, 0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class XrSample:
    """One head-plus-controller input sample in the raw OpenXR basis."""

    head: np.ndarray
    left: np.ndarray
    right: np.ndarray
    head_valid: bool
    left_valid: bool
    right_valid: bool
    left_trigger: float
    right_trigger: float
    left_grip: float
    right_grip: float
    left_axis: tuple[float, float]
    right_axis: tuple[float, float]
    buttons: dict[str, bool]
    arrival_monotonic: float
    source_timestamp_ns: int | None


@dataclass
class ArrivalMonitor:
    """Inter-arrival statistics that distinguish jitter from frame dropouts."""

    BUCKETS_MS = (10, 20, 33, 50, 100, 200, 500, 1000)
    window: int = 3000
    gaps: collections.deque[float] = field(init=False)
    histogram: collections.Counter = field(default_factory=collections.Counter)
    max_gap_ms: float = 0.0
    previous: float | None = None

    def __post_init__(self) -> None:
        self.gaps = collections.deque(maxlen=self.window)

    def record(self, arrival_monotonic: float) -> None:
        if self.previous is not None:
            gap_ms = (float(arrival_monotonic) - self.previous) * 1000.0
            if math.isfinite(gap_ms) and gap_ms >= 0.0:
                self.gaps.append(gap_ms)
                self.max_gap_ms = max(self.max_gap_ms, gap_ms)
                bucket = next((edge for edge in self.BUCKETS_MS if gap_ms <= edge), "over")
                self.histogram[bucket] += 1
        self.previous = float(arrival_monotonic)

    def report(self) -> dict[str, Any]:
        if not self.gaps:
            return {}
        gaps = np.asarray(self.gaps, dtype=np.float64)
        return {
            "source_gap_median_ms": float(np.median(gaps)),
            "source_gap_p95_ms": float(np.percentile(gaps, 95)),
            "source_gap_p99_ms": float(np.percentile(gaps, 99)),
            "source_gap_max_ms": float(self.max_gap_ms),
            "source_dropouts_over_150ms": int(np.sum(gaps > 150.0)),
            "source_gap_histogram": dict(self.histogram),
        }


class XrSource(abc.ABC):
    """Raw OpenXR head and controller data only; never a body/skeleton API."""

    STALE_RELEASE_S = 0.20
    STALE_INVALID_S = 0.35

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def poll(self) -> XrSample | None:
        """Return a newly observed source sample, or None for a duplicate/no frame."""

    @abc.abstractmethod
    def latest(self) -> XrSample | None: ...

    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...

    @property
    def supports_body(self) -> bool:
        return False

    @classmethod
    def stale_safe(cls, sample: XrSample | None, now: float | None = None) -> XrSample | None:
        return _stale_safe_sample(
            sample,
            release_s=cls.STALE_RELEASE_S,
            invalid_s=cls.STALE_INVALID_S,
            now=now,
        )


def _stale_safe_sample(
    sample: XrSample | None,
    *,
    release_s: float,
    invalid_s: float,
    now: float | None = None,
) -> XrSample | None:
    """Expire operator consent before pose validity for any XR input path."""

    if invalid_s <= release_s:
        raise ValueError("invalid stale timeout must be greater than release timeout")
    if sample is None:
        return None
    age = (time.monotonic() if now is None else float(now)) - sample.arrival_monotonic
    if age <= release_s:
        return sample
    released = dataclasses.replace(
        sample,
        buttons={key: False for key in sample.buttons},
        left_trigger=0.0,
        right_trigger=0.0,
        left_grip=0.0,
        right_grip=0.0,
    )
    if age <= invalid_s:
        return released
    return dataclasses.replace(released, head_valid=False, left_valid=False, right_valid=False)


def _as_se3(value: Any) -> tuple[np.ndarray, bool]:
    """Validate a raw OpenXR transform without accepting reflections."""

    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(4, 4).copy()
    except (TypeError, ValueError):
        return np.eye(4, dtype=np.float64), False
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        return np.eye(4, dtype=np.float64), False
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or determinant <= 0.0:
        return np.eye(4, dtype=np.float64), False
    left, _, right_t = np.linalg.svd(matrix[:3, :3])
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) <= 0.0:
        return np.eye(4, dtype=np.float64), False
    matrix[:3, :3] = rotation
    return matrix, True


def _pose7_to_se3(value: Any) -> tuple[np.ndarray, bool]:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(7)
    except (TypeError, ValueError):
        return np.eye(4, dtype=np.float64), False
    if not np.isfinite(pose).all() or float(np.linalg.norm(pose[3:])) < 1.0e-8:
        return np.eye(4, dtype=np.float64), False
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform, True


def world_arm_to_waist_head_position(arm_world: np.ndarray, head_world: np.ndarray) -> np.ndarray:
    transform = np.asarray(arm_world, dtype=np.float64).reshape(4, 4).copy()
    head = np.asarray(head_world, dtype=np.float64).reshape(4, 4)
    transform[:3, 3] = transform[:3, 3] - head[:3, 3] + WAIST_OFFSET_XYZ
    return transform


def controller_waist_targets_se3(
    left_xr: np.ndarray,
    right_xr: np.ndarray,
    head_xr: np.ndarray,
    *,
    left_ok: bool = True,
    right_ok: bool = True,
    head_ok: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The sole raw-OpenXR-to-PATH waist geometry path for v15.1."""

    head = np.asarray(head_xr, dtype=np.float64).reshape(4, 4) if head_ok else CONST_HEAD_POSE
    left = np.asarray(left_xr, dtype=np.float64).reshape(4, 4) if left_ok else CONST_LEFT_ARM_POSE
    right = np.asarray(right_xr, dtype=np.float64).reshape(4, 4) if right_ok else CONST_RIGHT_ARM_POSE
    head_robot = T_ROBOT_OPENXR @ head @ T_OPENXR_ROBOT
    left_robot = T_ROBOT_OPENXR @ left @ T_OPENXR_ROBOT
    right_robot = T_ROBOT_OPENXR @ right @ T_OPENXR_ROBOT
    return (
        world_arm_to_waist_head_position(left_robot, head_robot),
        world_arm_to_waist_head_position(right_robot, head_robot),
        bool(head_ok and left_ok and right_ok),
    )


def _unit_interval(value: Any) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _axis(value: Any) -> tuple[float, float]:
    try:
        raw = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return (0.0, 0.0)
    if raw.size < 2 or not np.isfinite(raw[:2]).all():
        return (0.0, 0.0)
    return (float(raw[0]), float(raw[1]))


def _same_sample(first: XrSample, second: XrSample) -> bool:
    return (
        np.allclose(first.head, second.head, atol=1e-9)
        and np.allclose(first.left, second.left, atol=1e-9)
        and np.allclose(first.right, second.right, atol=1e-9)
        and np.allclose(
            [first.left_trigger, first.right_trigger, first.left_grip, first.right_grip, *first.left_axis, *first.right_axis],
            [second.left_trigger, second.right_trigger, second.left_grip, second.right_grip, *second.left_axis, *second.right_axis],
            atol=1e-9,
        )
        and first.buttons == second.buttons
        and (first.head_valid, first.left_valid, first.right_valid)
        == (second.head_valid, second.left_valid, second.right_valid)
    )


class _LatestSampleSource(XrSource):
    """Thread-safe duplicate suppression and two-stage stale handling."""

    def __init__(self) -> None:
        self._latest: XrSample | None = None
        self._lock = threading.Lock()

    def _accept(self, sample: XrSample) -> XrSample | None:
        with self._lock:
            if self._latest is not None and _same_sample(sample, self._latest):
                return None
            self._latest = sample
        return sample

    def latest(self) -> XrSample | None:
        with self._lock:
            sample = self._latest
        return self.stale_safe(sample)


class XrtSource(_LatestSampleSource):
    """Read-only XRoboToolkit adapter used only by the source-comparison tool."""

    def __init__(self) -> None:
        super().__init__()
        self._xrt: Any | None = None

    def start(self) -> None:
        try:
            import xrobotoolkit_sdk as xrt
        except ModuleNotFoundError as error:
            raise RuntimeError("xrobotoolkit_sdk is required for --xrt comparison") from error
        self._xrt = xrt
        xrt.init()

    def stop(self) -> None:
        self._xrt = None

    @property
    def connected(self) -> bool:
        return self._xrt is not None

    @property
    def supports_body(self) -> bool:
        return True

    def poll(self) -> XrSample | None:
        if self._xrt is None:
            return None
        head, head_ok = _pose7_to_se3(self._xrt.get_headset_pose())
        left, left_ok = _pose7_to_se3(self._xrt.get_left_controller_pose())
        right, right_ok = _pose7_to_se3(self._xrt.get_right_controller_pose())
        sample = XrSample(
            head=head,
            left=left,
            right=right,
            head_valid=head_ok,
            left_valid=left_ok,
            right_valid=right_ok,
            left_trigger=_unit_interval(self._xrt.get_left_trigger()),
            right_trigger=_unit_interval(self._xrt.get_right_trigger()),
            left_grip=_unit_interval(self._xrt.get_left_grip()),
            right_grip=_unit_interval(self._xrt.get_right_grip()),
            left_axis=_axis(self._xrt.get_left_axis()),
            right_axis=_axis(self._xrt.get_right_axis()),
            buttons={
                "A": bool(self._xrt.get_A_button()), "B": bool(self._xrt.get_B_button()),
                "X": bool(self._xrt.get_X_button()), "Y": bool(self._xrt.get_Y_button()),
                "left_menu": bool(getattr(self._xrt, "get_left_menu_button", lambda: False)()),
            },
            arrival_monotonic=time.monotonic(),
            source_timestamp_ns=int(self._xrt.get_time_stamp_ns()),
        )
        # XRT has a device clock.  A held-still controller still produces a
        # fresh source frame, which is essential for the XRT gap baseline and
        # must not be confused with TeleVuer's no-timestamp fallback.
        with self._lock:
            if (
                self._latest is not None
                and self._latest.source_timestamp_ns == sample.source_timestamp_ns
            ):
                return None
            self._latest = sample
        return sample


class TeleVuerSource(_LatestSampleSource):
    """Raw WebXR source.  Never use TeleVuerWrapper here: it transforms poses."""

    BASIS_CORRECTION = np.eye(4, dtype=np.float64)
    HAND_STALE_RELEASE_S = 0.10
    HAND_STALE_INVALID_S = 0.20

    def __init__(
        self,
        *,
        cert_file: Path,
        key_file: Path,
        televuer_root: Path | None = None,
        use_hand_tracking: bool = False,
        basis_correction: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.cert_file = Path(cert_file).expanduser()
        self.key_file = Path(key_file).expanduser()
        self.televuer_root = Path(televuer_root).expanduser() if televuer_root else None
        self.use_hand_tracking = bool(use_hand_tracking)
        correction = self.BASIS_CORRECTION if basis_correction is None else basis_correction
        self.basis_correction, correction_ok = _as_se3(correction)
        if not correction_ok:
            raise ValueError("TeleVuer basis correction must be a proper SE(3) transform")
        self._raw: Any | None = None

    def start(self) -> None:
        if self.televuer_root is not None:
            source_root = self.televuer_root / "src"
            if not source_root.is_dir():
                raise RuntimeError(f"TeleVuer source directory is missing: {source_root}")
            sys.path.insert(0, str(source_root))
        try:
            from televuer.televuer import TeleVuer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "televuer is required; install it with `pip install -e /path/to/televuer` "
                "or pass --televuer-root."
            ) from error
        if not self.cert_file.is_file() or not self.key_file.is_file():
            raise RuntimeError("TeleVuer requires readable --televuer-cert and --televuer-key files")
        _configure_vuer_server_tls()
        # TeleVuer v4.0 rejects img_shape=None even in pass-through mode.  A
        # 1x2 placeholder satisfies that constructor guard; pass-through with
        # ZMQ/WebRTC disabled creates no image transport or render path.
        self._raw = TeleVuer(
            use_hand_tracking=self.use_hand_tracking,
            binocular=True,
            img_shape=(1, 2),
            display_mode="pass-through",
            zmq=False,
            webrtc=False,
            cert_file=str(self.cert_file),
            key_file=str(self.key_file),
        )

    def stop(self) -> None:
        if self._raw is not None:
            self._raw.close()
        self._raw = None

    @property
    def connected(self) -> bool:
        return bool(self._raw is not None and getattr(self._raw, "motion_data_ready", False))

    def _correct(self, value: Any) -> tuple[np.ndarray, bool]:
        raw, valid = _as_se3(value)
        if not valid:
            return raw, False
        inverse = np.linalg.inv(self.basis_correction)
        return _as_se3(self.basis_correction @ raw @ inverse)

    def latest(self) -> XrSample | None:
        with self._lock:
            sample = self._latest
        if not self.use_hand_tracking:
            return self.stale_safe(sample)
        # Pinch is a weaker dead-man than a physical button.  Its consent
        # expires sooner, before the normal bridge stale path can hold it.
        return _stale_safe_sample(
            sample,
            release_s=self.HAND_STALE_RELEASE_S,
            invalid_s=self.HAND_STALE_INVALID_S,
        )

    def poll(self) -> XrSample | None:
        raw = self._raw
        if raw is None:
            return None
        ready = bool(getattr(raw, "motion_data_ready", False))
        head, head_ok = self._correct(getattr(raw, "head_pose", None))
        left, left_ok = self._correct(getattr(raw, "left_arm_pose", None))
        right, right_ok = self._correct(getattr(raw, "right_arm_pose", None))
        if self.use_hand_tracking:
            left_pinch = bool(getattr(raw, "left_hand_pinch", False)) if ready else False
            right_pinch = bool(getattr(raw, "right_hand_pinch", False)) if ready else False
            left_trigger = 1.0 if left_pinch else 0.0
            right_trigger = 1.0 if right_pinch else 0.0
            left_grip = _unit_interval(getattr(raw, "left_hand_squeezeValue", 0.0)) if ready else 0.0
            right_grip = _unit_interval(getattr(raw, "right_hand_squeezeValue", 0.0)) if ready else 0.0
            buttons = {
                # PATH's current gates are left=Y and right=B.  TeleVuer's
                # hand event already reports a discrete, tracker-qualified
                # pinch; no assumed landmark ordering is introduced here.
                "A": False,
                "B": right_pinch,
                "X": False,
                "Y": left_pinch,
                "left_menu": False,
            }
        else:
            left_trigger = _unit_interval(getattr(raw, "left_ctrl_triggerValue", 0.0)) if ready else 0.0
            right_trigger = _unit_interval(getattr(raw, "right_ctrl_triggerValue", 0.0)) if ready else 0.0
            left_grip = _unit_interval(getattr(raw, "left_ctrl_squeezeValue", 0.0)) if ready else 0.0
            right_grip = _unit_interval(getattr(raw, "right_ctrl_squeezeValue", 0.0)) if ready else 0.0
            buttons = {
                # WebXR xr-standard: left A/B fields are X/Y; right are A/B.
                "A": bool(getattr(raw, "right_ctrl_aButton", False)) if ready else False,
                "B": bool(getattr(raw, "right_ctrl_bButton", False)) if ready else False,
                "X": bool(getattr(raw, "left_ctrl_aButton", False)) if ready else False,
                "Y": bool(getattr(raw, "left_ctrl_bButton", False)) if ready else False,
                "left_menu": False,
            }
        sample = XrSample(
            head=head,
            left=left,
            right=right,
            head_valid=bool(ready and head_ok),
            left_valid=bool(ready and left_ok),
            right_valid=bool(ready and right_ok),
            left_trigger=left_trigger,
            right_trigger=right_trigger,
            left_grip=left_grip,
            right_grip=right_grip,
            left_axis=_axis(getattr(raw, "left_ctrl_thumbstickValue", None)) if ready and not self.use_hand_tracking else (0.0, 0.0),
            right_axis=_axis(getattr(raw, "right_ctrl_thumbstickValue", None)) if ready and not self.use_hand_tracking else (0.0, 0.0),
            buttons=buttons,
            arrival_monotonic=time.monotonic(),
            source_timestamp_ns=None,
        )
        return self._accept(sample)


class XrStreamReader:
    """Poll one source, preserve only new samples, and retain gap statistics."""

    def __init__(self, source: XrSource, *, poll_hz: float = 120.0) -> None:
        self.source = source
        self.poll_hz = max(20.0, float(poll_hz))
        self.monitor = ArrivalMonitor()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="xr-source-reader", daemon=True)

    def start(self) -> None:
        self.source.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.source.stop()

    def latest(self) -> XrSample | None:
        return self.source.latest()

    def _run(self) -> None:
        period = 1.0 / self.poll_hz
        deadline = time.monotonic()
        while not self._stop.is_set():
            sample = self.source.poll()
            if sample is not None:
                self.monitor.record(sample.arrival_monotonic)
            deadline += period
            wait = deadline - time.monotonic()
            if wait > 0.0:
                self._stop.wait(wait)
            elif wait < -period:
                deadline = time.monotonic()


def parse_basis_correction(values: list[float] | None) -> np.ndarray:
    if values is None:
        return np.eye(4, dtype=np.float64)
    matrix, valid = _as_se3(np.asarray(values, dtype=np.float64).reshape(4, 4))
    if not valid:
        raise ValueError("--basis-correction must be a finite proper 4x4 SE(3) matrix")
    return matrix
