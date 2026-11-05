"""Coordinate/gain receipt used by the FlashSAC G1 deployment path.

An identity template is useful for read-only shadow diagnosis only. Physical
LowCmd admission accepts either a named, operator-validated receipt or HOMIE's
direct Unitree state convention when its identity offset receipt is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import DreamWaQ29Contract, SDK_JOINT_NAMES


CALIBRATION_SCHEMA = "dreamwaq.g1_29dof.hardware_calibration.v1"

# Early FlashSAC receipts specify 1200/80 for the three spine motors; later
# deployment-matched runs serialize 300/3 directly.  Keep the physical ceiling
# here so either reviewed training receipt produces the same safe waist command
# in mock, shadow, and real modes.
_SPINE_SDK_SLICE = slice(12, 15)
_DEPLOYMENT_SPINE_KP = 300.0
_DEPLOYMENT_SPINE_KD = 3.0


def deployment_pd_gains(
    nominal_kp_sdk: Any,
    nominal_kd_sdk: Any,
    *,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical deployment gains without altering the export receipt.

    ``nominal_*`` remain the exact policy/calibration receipt values.  The
    deployment envelope only reduces the three spine motors; all other G1
    joints preserve the recorded gain and calibration scale.
    """

    kp = np.asarray(nominal_kp_sdk, dtype=np.float32).reshape(-1).copy()
    kd = np.asarray(nominal_kd_sdk, dtype=np.float32).reshape(-1).copy()
    if kp.shape != (29,) or kd.shape != (29,):
        raise ValueError("deployment gains must each contain 29 values")
    kp *= float(kp_scale)
    kd *= float(kd_scale)
    kp[_SPINE_SDK_SLICE] = _DEPLOYMENT_SPINE_KP * float(kp_scale)
    kd[_SPINE_SDK_SLICE] = _DEPLOYMENT_SPINE_KD * float(kd_scale)
    return kp, kd


@dataclass(frozen=True)
class HardwareCalibration:
    path: Path
    robot_serial: str
    contract_fingerprint: str
    encoder_zero_offsets_rad: np.ndarray
    nominal_kp_sdk: np.ndarray
    nominal_kd_sdk: np.ndarray
    kp_scale: float
    kd_scale: float
    imu_quaternion_order: str
    imu_quaternion_frame: str
    imu_angular_velocity_frame: str
    operator: str
    operator_validated: bool

    @property
    def hardware_ready(self) -> bool:
        placeholders = {"", "unset", "unknown", "todo"}
        return bool(
            self.operator_validated
            and self.operator.strip().lower() not in placeholders
            and self.robot_serial.strip().lower() not in placeholders
        )

    @property
    def homie_direct_state_compatible(self) -> bool:
        """Whether this receipt matches HOMIE's direct Unitree LowState path.

        HOMIE uses identity G1 motor order, firmware joint zeroes, a ``wxyz``
        body-to-world quaternion, and a body-frame gyro. This is narrower
        than ``hardware_ready`` and is the direct production state contract
        used by the bundled HOMIE G1 deployment.
        """

        return bool(
            np.allclose(self.encoder_zero_offsets_rad, 0.0, atol=1.0e-7, rtol=0.0)
            and self.imu_quaternion_order == "wxyz"
            and self.imu_quaternion_frame == "body_to_world"
            and self.imu_angular_velocity_frame == "body"
        )

    @property
    def kp_sdk(self) -> np.ndarray:
        return deployment_pd_gains(
            self.nominal_kp_sdk,
            self.nominal_kd_sdk,
            kp_scale=self.kp_scale,
            kd_scale=self.kd_scale,
        )[0]

    @property
    def kd_sdk(self) -> np.ndarray:
        return deployment_pd_gains(
            self.nominal_kp_sdk,
            self.nominal_kd_sdk,
            kp_scale=self.kp_scale,
            kd_scale=self.kd_scale,
        )[1]

    @classmethod
    def load(
        cls,
        path: Any,
        *,
        contract: DreamWaQ29Contract,
        require_hardware_ready: bool = False,
    ) -> "HardwareCalibration":
        raw = Path(path).expanduser()
        if raw.is_symlink() or not raw.is_file():
            raise ValueError("hardware calibration must be one regular file: %s" % raw)
        try:
            payload = json.loads(raw.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read G1 hardware calibration") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("unsupported G1 hardware calibration schema")
        if payload.get("contract_fingerprint") != contract.fingerprint:
            raise ValueError("hardware calibration was not reviewed for this exact policy/CENet contract")
        recorded_hashes = payload.get("artifact_sha256")
        if recorded_hashes is not None and dict(recorded_hashes) != dict(contract.artifact_sha256):
            raise ValueError("hardware calibration artifact hashes differ from the selected export")
        if payload.get("sdk_joint_names") != list(SDK_JOINT_NAMES):
            raise ValueError("hardware calibration joint order differs from Unitree G1 29")
        recorded_kp = payload.get("nominal_kp_sdk")
        recorded_kd = payload.get("nominal_kd_sdk")
        if recorded_kp is not None and not np.allclose(
            np.asarray(recorded_kp, dtype=np.float32).reshape(-1), contract.kp_sdk, atol=1.0e-7, rtol=0.0
        ):
            raise ValueError("hardware calibration nominal Kp differs from the FlashSAC contract")
        if recorded_kd is not None and not np.allclose(
            np.asarray(recorded_kd, dtype=np.float32).reshape(-1), contract.kd_sdk, atol=1.0e-7, rtol=0.0
        ):
            raise ValueError("hardware calibration nominal Kd differs from the FlashSAC contract")
        offsets = np.asarray(payload.get("encoder_zero_offsets_rad"), dtype=np.float32).reshape(-1)
        if offsets.shape != (29,) or not np.all(np.isfinite(offsets)):
            raise ValueError("encoder_zero_offsets_rad must contain 29 finite values")
        kp_scale = float(payload.get("kp_scale", float("nan")))
        kd_scale = float(payload.get("kd_scale", float("nan")))
        if not all(math.isfinite(value) and 0.0 < value <= 1.0 for value in (kp_scale, kd_scale)):
            raise ValueError("hardware gain scales must be in (0, 1]")
        # The generated template uses UNSET for these fields. The fixed G1
        # runtime convention permits read-only shadow diagnosis; real mode
        # additionally requires this receipt's operator validation.
        imu_order = str(payload.get("imu_quaternion_order", "wxyz")).strip().lower()
        imu_frame = str(payload.get("imu_quaternion_frame", "body_to_world")).strip().lower()
        gyro_frame = str(payload.get("imu_angular_velocity_frame", "body")).strip().lower()
        placeholders = {"", "unset", "unknown", "todo"}
        if imu_order in placeholders:
            imu_order = "wxyz"
        if imu_frame in placeholders:
            imu_frame = "body_to_world"
        if gyro_frame in placeholders:
            gyro_frame = "body"
        if imu_order != "wxyz":
            raise ValueError("DreamWaQ deployment requires Unitree wxyz quaternion order")
        if imu_frame != "body_to_world":
            raise ValueError("DreamWaQ deployment requires a body_to_world IMU quaternion")
        if gyro_frame != "body":
            raise ValueError("DreamWaQ deployment requires a body-frame Unitree gyroscope")
        result = cls(
            path=raw.resolve(),
            robot_serial=str(payload.get("robot_serial", "")).strip(),
            contract_fingerprint=contract.fingerprint,
            encoder_zero_offsets_rad=offsets.copy(),
            nominal_kp_sdk=contract.kp_sdk.copy(),
            nominal_kd_sdk=contract.kd_sdk.copy(),
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            imu_quaternion_order=imu_order,
            imu_quaternion_frame=imu_frame,
            imu_angular_velocity_frame=gyro_frame,
            operator=str(payload.get("operator", "")).strip(),
            operator_validated=payload.get("operator_validated") is True,
        )
        if require_hardware_ready and not result.hardware_ready:
            raise ValueError("hardware calibration is an unvalidated template; real LowCmd remains locked")
        return result


def write_calibration_template(
    path: Any,
    *,
    contract: DreamWaQ29Contract,
    overwrite: bool = False,
) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite calibration file: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "contract_fingerprint": contract.fingerprint,
        "artifact_sha256": dict(contract.artifact_sha256),
        "sdk_joint_names": list(SDK_JOINT_NAMES),
        "nominal_kp_sdk": [float(value) for value in contract.kp_sdk],
        "nominal_kd_sdk": [float(value) for value in contract.kd_sdk],
        "robot_serial": "UNSET",
        "encoder_zero_offsets_rad": [0.0] * 29,
        "kp_scale": 1.0,
        "kd_scale": 1.0,
        "imu_quaternion_order": "UNSET",
        "imu_quaternion_frame": "UNSET",
        "imu_angular_velocity_frame": "UNSET",
        "operator": "UNSET",
        "operator_validated": False,
        "note": (
            "Read-only shadow template only. Replace offsets, verify IMU conventions, "
            "set robot_serial/operator, and mark operator_validated before real LowCmd."
        ),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
