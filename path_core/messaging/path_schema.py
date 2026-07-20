from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np

SCHEMA_QP_REFERENCE_V1 = "path.qp_reference.v1"
QP_REFERENCE_TOPIC = "path_qp_reference"
CUROBO_PLAN_TOPIC = "curobo_plan"
PATH_GOAL_TOPIC = "path_goal"
STATIC_EE_TARGET_TOPIC = "path_static_ee_target"

PATH_STATES = {"teleop", "intention", "auto_planning", "damped_return", "idle"}
REFERENCE_TYPES = {"none", "sample", "bspline", "cartesian_target"}


def now_us() -> int:
    return time.monotonic_ns() // 1000


def _list29(values: Sequence[float] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 29:
        raise ValueError(f"q_ref must contain 29 joints, got {arr.shape[0]}")
    return arr.tolist()


def qp_sample_reference(
    q_ref: Sequence[float] | np.ndarray,
    *,
    state: str = "teleop",
    source: str = "gmr",
    control_mode: str = "teleop",
    feasible: bool = True,
    plan_id: int = 0,
    phase: float = 0.0,
    timestamp_us: int | None = None,
    **metadata,
) -> dict:
    if state not in PATH_STATES:
        raise ValueError(f"unknown PATH state: {state}")
    return {
        "schema": SCHEMA_QP_REFERENCE_V1,
        "reference_type": "sample",
        "state": state,
        "source": source,
        "control_mode": control_mode,
        "feasible": bool(feasible),
        "plan_id": int(plan_id),
        "phase": float(phase),
        "timestamp_us": int(timestamp_us if timestamp_us is not None else now_us()),
        "q_ref": _list29(q_ref),
        **metadata,
    }


def qp_bspline_reference(
    control_points: Iterable[Sequence[float] | np.ndarray],
    *,
    control_dt_sec: float,
    t_start_us: int,
    state: str = "auto_planning",
    source: str = "curobo",
    control_mode: str = "plan_exec",
    feasible: bool = True,
    plan_id: int = 0,
    phase: float = 0.0,
    timestamp_us: int | None = None,
    **metadata,
) -> dict:
    if state not in PATH_STATES:
        raise ValueError(f"unknown PATH state: {state}")
    rows = [_list29(row) for row in control_points]
    return {
        "schema": SCHEMA_QP_REFERENCE_V1,
        "reference_type": "bspline",
        "state": state,
        "source": source,
        "control_mode": control_mode,
        "feasible": bool(feasible),
        "plan_id": int(plan_id),
        "phase": float(phase),
        "timestamp_us": int(timestamp_us if timestamp_us is not None else now_us()),
        "t_start_us": int(t_start_us),
        "control_dt_sec": float(control_dt_sec),
        "control_points": rows,
        **metadata,
    }


def qp_cartesian_target_reference(
    *,
    target_link: str,
    target_position: Sequence[float] | np.ndarray,
    q_ref: Sequence[float] | np.ndarray,
    state: str = "auto_planning",
    source: str = "cartesian_servo",
    control_mode: str = "plan_exec",
    feasible: bool = True,
    plan_id: int = 0,
    phase: float = 0.0,
    timestamp_us: int | None = None,
    terminal_capture_radius_m: float = 0.0,
    **metadata,
) -> dict:
    if state not in PATH_STATES:
        raise ValueError(f"unknown PATH state: {state}")
    pos = np.asarray(target_position, dtype=np.float64).reshape(-1)
    if pos.shape[0] != 3:
        raise ValueError(f"target_position must contain 3 values, got {pos.shape[0]}")
    return {
        "schema": SCHEMA_QP_REFERENCE_V1,
        "reference_type": "cartesian_target",
        "state": state,
        "source": source,
        "control_mode": control_mode,
        "feasible": bool(feasible),
        "plan_id": int(plan_id),
        "phase": float(phase),
        "timestamp_us": int(timestamp_us if timestamp_us is not None else now_us()),
        "q_ref": _list29(q_ref),
        "terminal_target_link": str(target_link),
        "terminal_target_position": pos.tolist(),
        "terminal_capture_radius_m": float(terminal_capture_radius_m),
        **metadata,
    }


def qp_none_reference(
    *,
    state: str = "idle",
    source: str = "unknown",
    control_mode: str = "idle",
    feasible: bool = False,
    plan_id: int = 0,
    timestamp_us: int | None = None,
) -> dict:
    return {
        "schema": SCHEMA_QP_REFERENCE_V1,
        "reference_type": "none",
        "state": state,
        "source": source,
        "control_mode": control_mode,
        "feasible": bool(feasible),
        "plan_id": int(plan_id),
        "timestamp_us": int(timestamp_us if timestamp_us is not None else now_us()),
    }
