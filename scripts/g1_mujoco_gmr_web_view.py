#!/usr/bin/env python3
"""Realtime upper-body FK overlay for MuJoCo G1 state and SMPL->GMR reference.

The viewer subscribes to:
  - SONIC/MuJoCo debug ZMQ, usually tcp://127.0.0.1:5557/g1_debug
  - GMR reference ZMQ, usually tcp://127.0.0.1:5558/gmr_ref

It serves a small browser UI that draws the current upper-body FK and the
GMR-retargeted upper-body FK in the same canvas.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import msgpack
import numpy as np
import uvicorn
import zmq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


def _as_float_list(value: Any, min_len: int = 29) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if len(values) < min_len:
        return None
    return values[:min_len]


def _as_vec(value: Any, length: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if len(values) < length:
        return None
    return values[:length]


def _as_nonzero_vec(value: Any, length: int, eps: float = 1e-9) -> list[float] | None:
    values = _as_vec(value, length)
    if values is None:
        return None
    # Older/non-rebuilt real-LCM deploy binaries published default Eigen::Zero()
    # wrist telemetry. Treat that as missing so the viewer does not report a
    # bogus target error that disagrees with the QP telemetry.
    if sum(v * v for v in values) <= eps * eps:
        return None
    return values


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _unpack_topic_msg(raw: bytes, topic: str) -> dict[str, Any] | None:
    prefix = topic.encode()
    if not raw.startswith(prefix):
        return None
    try:
        payload = msgpack.unpackb(raw[len(prefix) :], raw=False)
    except (ValueError, msgpack.UnpackException):
        return None
    return payload if isinstance(payload, dict) else None


class LatestZmqTopic:
    def __init__(
        self,
        endpoint: str,
        topic: str,
        q_keys: tuple[str, ...],
        name: str,
    ) -> None:
        self.endpoint = endpoint
        self.topic = topic
        self.q_keys = q_keys
        self.name = name
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._latest_q: list[float] | None = None
        self._latest_q_key = ""
        self._latest_time = 0.0
        self._count = 0
        self._reported_q_key = False
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[{self.name}] subscribing {self.endpoint}/{self.topic}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        socket.connect(self.endpoint)
        try:
            while self._running:
                if not socket.poll(100, zmq.POLLIN):
                    continue
                raw = socket.recv()
                message = _unpack_topic_msg(raw, self.topic)
                if message is None:
                    continue
                q = None
                q_key = ""
                for key in self.q_keys:
                    q = _as_float_list(message.get(key))
                    if q is not None:
                        q_key = key
                        break
                now = time.monotonic()
                with self._lock:
                    self._latest = message
                    self._latest_q = q
                    self._latest_q_key = q_key
                    self._latest_time = now
                    self._count += 1
                    count = self._count
                if q_key and not self._reported_q_key:
                    self._reported_q_key = True
                    print(f"[{self.name}] receiving q from key='{q_key}'")
                elif count == 1 and not q_key:
                    print(
                        f"[{self.name}] received messages but no q key found; "
                        f"available keys={sorted(message.keys())[:20]}"
                    )
        finally:
            socket.close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            message = dict(self._latest) if self._latest is not None else None
            q = list(self._latest_q) if self._latest_q is not None else None
            q_key = self._latest_q_key
            age_ms = (
                1000.0 * (time.monotonic() - self._latest_time)
                if self._latest_time > 0.0
                else math.nan
            )
            count = self._count
        return {
            "message": message,
            "q": q,
            "q_key": q_key,
            "age_ms": age_ms,
            "count": count,
        }


class PathWebRecorder:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.expanduser()
        self._lock = threading.Lock()
        self._active = False
        self._rows: list[dict[str, Any]] = []
        self._telemetry_rows: list[dict[str, Any]] = []
        self._start_wall = 0.0
        self._session_dir: Path | None = None
        self._last_saved: dict[str, str] | None = None

    def start(self, label: str = "") -> dict[str, Any]:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip())
        name = f"{stamp}_{safe_label}" if safe_label else stamp
        session_dir = self.output_root / name
        session_dir.mkdir(parents=True, exist_ok=False)
        with self._lock:
            self._active = True
            self._rows = []
            self._telemetry_rows = []
            self._start_wall = time.time()
            self._session_dir = session_dir
            self._last_saved = None
        return {"recording": True, "session_dir": str(session_dir), "frames": 0}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._rows)
            telemetry_rows = list(self._telemetry_rows)
            session_dir = self._session_dir
            self._active = False
            self._rows = []
            self._telemetry_rows = []
        if session_dir is None:
            return {"recording": False, "frames": 0, "saved": False}
        csv_path = session_dir / "path_static_ee_tracking.csv"
        npz_path = session_dir / "path_static_ee_tracking.npz"
        telemetry_csv_path = session_dir / "path_ee_qp_telemetry.csv"
        if rows:
            fieldnames: list[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            arrays: dict[str, np.ndarray] = {}
            for key in fieldnames:
                values = [row.get(key, np.nan) for row in rows]
                try:
                    arrays[key] = np.asarray(values, dtype=np.float64)
                except (TypeError, ValueError):
                    arrays[key] = np.asarray(values, dtype=object)
            np.savez_compressed(npz_path, **arrays)
        if telemetry_rows:
            telemetry_fieldnames: list[str] = []
            for row in telemetry_rows:
                for key in row.keys():
                    if key not in telemetry_fieldnames:
                        telemetry_fieldnames.append(key)
            with telemetry_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=telemetry_fieldnames)
                writer.writeheader()
                writer.writerows(telemetry_rows)
        self._last_saved = {
            "session_dir": str(session_dir),
            "csv": str(csv_path),
            "npz": str(npz_path),
            "qp_telemetry_csv": str(telemetry_csv_path),
        }
        return {
            "recording": False,
            "frames": len(rows),
            "saved": bool(rows),
            **self._last_saved,
        }

    def append(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if not self._active:
                return
            elapsed = time.time() - self._start_wall
            index = len(self._rows)
        row = flatten_record_payload(payload, index=index, elapsed_sec=elapsed)
        telemetry_row = flatten_qp_telemetry_payload(
            payload, index=index, elapsed_sec=elapsed
        )
        with self._lock:
            if self._active:
                self._rows.append(row)
                self._telemetry_rows.append(telemetry_row)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "recording": self._active,
                "frames": len(self._rows),
                "session_dir": str(self._session_dir) if self._session_dir is not None else "",
                "last_saved": dict(self._last_saved) if self._last_saved else None,
            }


class MujocoWristFk:
    def __init__(self, model_xml: Path) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model_xml = model_xml.expanduser().resolve()
        old_cwd = Path.cwd()
        try:
            os.chdir(self.model_xml.parent)
            self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        finally:
            os.chdir(old_cwd)
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_ids: dict[str, int] = {}

    def wrist_position(self, q29: list[float] | np.ndarray, target_link: str) -> list[float] | None:
        q = np.asarray(q29, dtype=np.float64).reshape(-1)
        if q.shape[0] < 29 or not np.isfinite(q[:29]).all():
            return None
        body_id = self.body_ids.get(target_link)
        if body_id is None:
            body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, target_link)
            if body_id < 0:
                return None
            self.body_ids[target_link] = body_id
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[7:36] = q[:29]
        self.mujoco.mj_forward(self.model, self.data)
        pelvis = self.data.xpos[self.pelvis_id].copy() if self.pelvis_id >= 0 else np.zeros(3)
        pos = self.data.xpos[body_id].copy() - pelvis
        return pos.astype(float).tolist()


def _put_vec(row: dict[str, Any], prefix: str, values: list[float] | None) -> None:
    if values is None:
        return
    for i, value in enumerate(values):
        row[f"{prefix}_{i:02d}"] = float(value)


def _put_any_for_csv(row: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        row[key] = int(bool(value))
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        row[key] = float(value)
        return
    if isinstance(value, str):
        row[key] = value
        return
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (int, float, np.integer, np.floating, bool, np.bool_)) for item in value):
            for i, item in enumerate(value):
                row[f"{key}_{i:02d}"] = float(item)
        else:
            row[key] = json.dumps(json_safe(value), separators=(",", ":"))
        return
    if isinstance(value, dict):
        row[key] = json.dumps(json_safe(value), separators=(",", ":"))
        return
    row[key] = str(value)


def flatten_qp_telemetry_payload(
    payload: dict[str, Any], index: int, elapsed_sec: float
) -> dict[str, Any]:
    state_msg = payload.get("_state_msg")
    if not isinstance(state_msg, dict):
        state_msg = {}
    row: dict[str, Any] = {
        "frame": index,
        "elapsed_sec": elapsed_sec,
        "monotonic_us": time.monotonic_ns() // 1000,
        "state_age_ms": payload.get("state_age_ms", math.nan),
        "gmr_age_ms": payload.get("gmr_age_ms", math.nan),
    }
    # Keep the same raw telemetry that tools/listen_ee_qp_telemetry.py
    # summarizes, but write vectors as numbered CSV columns for easy plotting.
    for key in sorted(state_msg.keys()):
        if (
            key.startswith("ee_qp_")
            or key.startswith("base_")
            or key.startswith("body_torso_")
            or key in {"body_torso_ang_vel"}
        ):
            _put_any_for_csv(row, key, state_msg.get(key))
    return row


def flatten_record_payload(payload: dict[str, Any], index: int, elapsed_sec: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": index,
        "elapsed_sec": elapsed_sec,
        "monotonic_us": time.monotonic_ns() // 1000,
        "state_age_ms": payload.get("state_age_ms", math.nan),
        "gmr_age_ms": payload.get("gmr_age_ms", math.nan),
        "ee_qp_valid": int(bool(payload.get("ee_qp_valid", False))),
        "ee_qp_reference_type": payload.get("ee_qp_reference_type", ""),
        "ee_qp_plan_id": payload.get("ee_qp_plan_id", 0),
        "ee_qp_plan_phase": payload.get("ee_qp_plan_phase", math.nan),
        "ee_qp_terminal_target_active": int(bool(payload.get("ee_qp_terminal_target_active", False))),
        "ee_qp_terminal_target_side": payload.get("ee_qp_terminal_target_side", 0),
        "ee_qp_terminal_target_link": payload.get("ee_qp_terminal_target_link", ""),
        "qp_input_target_side": payload.get("qp_input_target_side", payload.get("ee_qp_terminal_target_side", 0)),
        "qp_input_target_link": payload.get("qp_input_target_link", payload.get("ee_qp_terminal_target_link", "")),
        "qp_input_coordinate_frame": payload.get("qp_input_coordinate_frame", "pelvis_identity_fk"),
        "ee_qp_terminal_target_distance_m": payload.get("ee_qp_terminal_target_distance_m", math.nan),
        "ee_qp_terminal_capture_radius_m": payload.get("ee_qp_terminal_capture_radius_m", math.nan),
        "ee_qp_servo_error_before_m": payload.get("ee_qp_servo_error_before_m", math.nan),
        "ee_qp_servo_error_after_m": payload.get("ee_qp_servo_error_after_m", math.nan),
        "left_target_error_m": payload.get("left_target_error_m", math.nan),
        "right_target_error_m": payload.get("right_target_error_m", math.nan),
        "active_target_error_m": payload.get("active_target_error_m", math.nan),
    }
    _put_vec(row, "body_q_measured", payload.get("q"))
    gmr_q_ref = payload.get("q_ref")
    qp_q_ref = payload.get("q_qp_ref")
    # Always keep the raw GMR reference. When QP is toggled off, ee_qp_q_ref is
    # absent, so keep bspline_q_ref backward-compatible by falling back to GMR.
    _put_vec(row, "gmr_q_ref", gmr_q_ref)
    _put_vec(row, "bspline_q_ref", qp_q_ref if qp_q_ref is not None else gmr_q_ref)
    _put_vec(row, "pd_q_corrected", payload.get("q_corrected"))
    _put_vec(row, "sonic_last_action", payload.get("last_action"))
    _put_vec(row, "terminal_target_position", payload.get("ee_qp_terminal_target_position"))
    _put_vec(row, "qp_input_target_position", payload.get("qp_input_target_position"))
    _put_vec(row, "ee_qp_left_wrist_position", payload.get("ee_qp_left_wrist_position"))
    _put_vec(row, "ee_qp_right_wrist_position", payload.get("ee_qp_right_wrist_position"))
    _put_vec(row, "ee_qp_active_wrist_position", payload.get("ee_qp_active_wrist_position"))
    _put_vec(row, "left_wrist_position", payload.get("left_wrist_position"))
    _put_vec(row, "right_wrist_position", payload.get("right_wrist_position"))
    return row


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>G1 MuJoCo / GMR Upper Body View</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:#101114; color:#e8e8e8; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
    min-height:100vh; display:flex; flex-direction:column;
  }
  header {
    height:48px; padding:0 18px; display:flex; align-items:center; gap:20px;
    background:#181b20; border-bottom:1px solid #2b3038;
  }
  h1 { font-size:16px; color:#f2f4f8; font-weight:650; }
  button {
    border:1px solid #3a4150; background:#20242c; color:#f1f5f9; padding:7px 10px;
    font-family:inherit; font-size:12px; cursor:pointer;
  }
  button:hover { background:#2a303a; }
  button.recording { border-color:#ef4444; color:#fecaca; }
  #status { margin-left:auto; color:#aeb6c2; font-size:12px; }
  main { flex:1; display:grid; grid-template-columns:minmax(640px, 1fr) 360px; gap:0; overflow:hidden; }
  #stage { display:flex; align-items:center; justify-content:center; padding:18px; }
  canvas { width:min(92vw, 960px); height:min(84vh, 760px); background:#f8fafc; border:1px solid #d9dee7; }
  aside { border-left:1px solid #2b3038; background:#15171b; padding:16px; overflow:auto; }
  h2 { font-size:13px; margin:0 0 10px; color:#dbeafe; }
  .legend { display:grid; gap:8px; margin-bottom:18px; font-size:12px; color:#c9d2df; }
  .swatch { display:inline-block; width:24px; height:3px; margin-right:8px; vertical-align:middle; }
  .current { background:#2563eb; }
  .ref { background:#16a34a; }
  .qpref { background:#f97316; }
  .corrected { background:#7c3aed; }
  .error { background:#ef4444; }
  .target { background:#dc2626; }
  .kv { font-size:12px; line-height:1.65; color:#c8d0dc; white-space:pre-wrap; }
  .warn { color:#f59e0b; }
  .ok { color:#86efac; }
</style>
</head>
<body>
<header>
  <h1>G1 Upper Body FK Overlay</h1>
  <button id="record-btn">Start record</button>
  <span id="record-status">record idle</span>
  <div id="status">connecting</div>
</header>
<main>
  <section id="stage">
    <canvas id="pose-canvas" width="960" height="760"></canvas>
  </section>
  <aside>
    <h2>Streams</h2>
    <div class="legend">
      <div><span class="swatch current"></span>MuJoCo measured q</div>
      <div><span class="swatch ref"></span>SMPL to GMR q_ref</div>
      <div><span class="swatch qpref"></span>QP active ref q</div>
      <div><span class="swatch corrected"></span>QP corrected PD target</div>
      <div><span class="swatch target"></span>static EE target</div>
      <div><span class="swatch error"></span>wrist FK error</div>
    </div>
    <h2>Telemetry</h2>
    <div id="info" class="kv">waiting for data</div>
  </aside>
</main>

<script>
const JOINT_NAMES_URDF = [
  "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
  "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
  "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
  "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
  "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
  "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
  "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
  "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
  "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
];

const JOINT_DATA = {
  "left_hip_pitch_joint":      ["pelvis","left_hip_pitch_link",    [0,0.064452,-0.1027],    [0,0,0],           [0,1,0]],
  "left_hip_roll_joint":       ["left_hip_pitch_link","left_hip_roll_link",  [0,0.052,-0.030465],[0,-0.1749,0],[1,0,0]],
  "left_hip_yaw_joint":        ["left_hip_roll_link","left_hip_yaw_link",   [0.025001,0,-0.12412],[0,0,0],    [0,0,1]],
  "left_knee_joint":           ["left_hip_yaw_link","left_knee_link",       [-0.078273,0.0021489,-0.17734],[0,0.1749,0],[0,1,0]],
  "left_ankle_pitch_joint":    ["left_knee_link","left_ankle_pitch_link",   [0,-9.4445e-5,-0.30001],[0,0,0],  [0,1,0]],
  "left_ankle_roll_joint":     ["left_ankle_pitch_link","left_ankle_roll_link",[0,0,-0.017558],[0,0,0],       [1,0,0]],
  "right_hip_pitch_joint":     ["pelvis","right_hip_pitch_link",   [0,-0.064452,-0.1027],   [0,0,0],           [0,1,0]],
  "right_hip_roll_joint":      ["right_hip_pitch_link","right_hip_roll_link",[0,-0.052,-0.030465],[0,-0.1749,0],[1,0,0]],
  "right_hip_yaw_joint":       ["right_hip_roll_link","right_hip_yaw_link", [0.025001,0,-0.12412],[0,0,0],   [0,0,1]],
  "right_knee_joint":          ["right_hip_yaw_link","right_knee_link",     [-0.078273,-0.0021489,-0.17734],[0,0.1749,0],[0,1,0]],
  "right_ankle_pitch_joint":   ["right_knee_link","right_ankle_pitch_link", [0,9.4445e-5,-0.30001],[0,0,0],  [0,1,0]],
  "right_ankle_roll_joint":    ["right_ankle_pitch_link","right_ankle_roll_link",[0,0,-0.017558],[0,0,0],    [1,0,0]],
  "waist_yaw_joint":           ["pelvis","waist_yaw_link",                  [0,0,0],         [0,0,0],           [0,0,1]],
  "waist_roll_joint":          ["waist_yaw_link","waist_roll_link",         [-0.0039635,0,0.035],[0,0,0],     [1,0,0]],
  "waist_pitch_joint":         ["waist_roll_link","torso_link",             [0,0,0.019],     [0,0,0],           [0,1,0]],
  "left_shoulder_pitch_joint": ["torso_link","left_shoulder_pitch_link",    [0.0039563,0.10022,0.23778],[0.27931,0,0],[0,1,0]],
  "left_shoulder_roll_joint":  ["left_shoulder_pitch_link","left_shoulder_roll_link",[0,0.038,-0.013831],[-0.27925,0,0],[1,0,0]],
  "left_shoulder_yaw_joint":   ["left_shoulder_roll_link","left_shoulder_yaw_link", [0,0.00624,-0.1032],[0,0,0],[0,0,1]],
  "left_elbow_joint":          ["left_shoulder_yaw_link","left_elbow_link",  [0.015783,0,-0.080518],[0,0,0], [0,1,0]],
  "left_wrist_roll_joint":     ["left_elbow_link","left_wrist_roll_link",    [0.100,0.00188791,-0.010],[0,0,0],[1,0,0]],
  "left_wrist_pitch_joint":    ["left_wrist_roll_link","left_wrist_pitch_link",[0.038,0,0],[0,0,0],           [0,1,0]],
  "left_wrist_yaw_joint":      ["left_wrist_pitch_link","left_wrist_yaw_link",[0.046,0,0],[0,0,0],            [0,0,1]],
  "right_shoulder_pitch_joint":["torso_link","right_shoulder_pitch_link",   [0.0039563,-0.10021,0.23778],[-0.27931,0,0],[0,1,0]],
  "right_shoulder_roll_joint": ["right_shoulder_pitch_link","right_shoulder_roll_link",[0,-0.038,-0.013831],[0.27925,0,0],[1,0,0]],
  "right_shoulder_yaw_joint":  ["right_shoulder_roll_link","right_shoulder_yaw_link",[0,-0.00624,-0.1032],[0,0,0],[0,0,1]],
  "right_elbow_joint":         ["right_shoulder_yaw_link","right_elbow_link",[0.015783,0,-0.080518],[0,0,0], [0,1,0]],
  "right_wrist_roll_joint":    ["right_elbow_link","right_wrist_roll_link",  [0.100,-0.00188791,-0.010],[0,0,0],[1,0,0]],
  "right_wrist_pitch_joint":   ["right_wrist_roll_link","right_wrist_pitch_link",[0.038,0,0],[0,0,0],         [0,1,0]],
  "right_wrist_yaw_joint":     ["right_wrist_pitch_link","right_wrist_yaw_link",[0.046,0,0],[0,0,0],          [0,0,1]],
};

const ORDER = [
  "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
  "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
  "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
  "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
  "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
  "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
  "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
  "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
  "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
];

const UPPER_BONES = [
  [null, "waist_yaw_joint"],
  ["waist_yaw_joint","waist_roll_joint"],
  ["waist_roll_joint","waist_pitch_joint"],
  ["waist_pitch_joint","left_shoulder_pitch_joint"],
  ["left_shoulder_pitch_joint","left_shoulder_roll_joint"],
  ["left_shoulder_roll_joint","left_shoulder_yaw_joint"],
  ["left_shoulder_yaw_joint","left_elbow_joint"],
  ["left_elbow_joint","left_wrist_roll_joint"],
  ["left_wrist_roll_joint","left_wrist_pitch_joint"],
  ["left_wrist_pitch_joint","left_wrist_yaw_joint"],
  ["waist_pitch_joint","right_shoulder_pitch_joint"],
  ["right_shoulder_pitch_joint","right_shoulder_roll_joint"],
  ["right_shoulder_roll_joint","right_shoulder_yaw_joint"],
  ["right_shoulder_yaw_joint","right_elbow_joint"],
  ["right_elbow_joint","right_wrist_roll_joint"],
  ["right_wrist_roll_joint","right_wrist_pitch_joint"],
  ["right_wrist_pitch_joint","right_wrist_yaw_joint"],
];

function mat4Identity() { return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
function mat4Mul(A, B) {
  const C = new Array(16).fill(0);
  for (let r=0;r<4;r++) for (let c=0;c<4;c++) for (let k=0;k<4;k++) C[r*4+c] += A[r*4+k]*B[k*4+c];
  return C;
}
function mat4Translation(x,y,z) { return [1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1]; }
function mat4RPY(r,p,y) {
  const cr=Math.cos(r),sr=Math.sin(r), cp=Math.cos(p),sp=Math.sin(p), cy=Math.cos(y),sy=Math.sin(y);
  return [
    cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, 0,
    sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, 0,
    -sp, cp*sr, cp*cr, 0,
    0,0,0,1
  ];
}
function mat4AxisAngle(ax,ay,az, angle) {
  const c=Math.cos(angle), s=Math.sin(angle), t=1-c;
  const l=Math.sqrt(ax*ax+ay*ay+az*az);
  if (l<1e-9) return mat4Identity();
  ax/=l; ay/=l; az/=l;
  return [
    t*ax*ax+c, t*ax*ay-s*az, t*ax*az+s*ay, 0,
    t*ax*ay+s*az, t*ay*ay+c, t*ay*az-s*ax, 0,
    t*ax*az-s*ay, t*ay*az+s*ax, t*az*az+c, 0,
    0,0,0,1
  ];
}
function applyMat4(M, x, y, z) {
  return [M[0]*x+M[1]*y+M[2]*z+M[3], M[4]*x+M[5]*y+M[6]*z+M[7], M[8]*x+M[9]*y+M[10]*z+M[11]];
}
function computeFK(q) {
  const qMap = {};
  for (let i=0; i<JOINT_NAMES_URDF.length; i++) qMap[JOINT_NAMES_URDF[i]] = q[i] || 0;
  const linkT = { "pelvis": mat4Identity() };
  const jointWorldPos = {};
  for (const jname of ORDER) {
    const [parent, child, xyz, rpy, axis] = JOINT_DATA[jname];
    const parentT = linkT[parent] || mat4Identity();
    const jointWorldT = mat4Mul(parentT, mat4Mul(mat4Translation(...xyz), mat4RPY(...rpy)));
    jointWorldPos[jname] = applyMat4(jointWorldT, 0, 0, 0);
    linkT[child] = mat4Mul(jointWorldT, mat4AxisAngle(...axis, qMap[jname] || 0));
  }
  jointWorldPos["pelvis"] = [0,0,0];
  return jointWorldPos;
}
function project(p, cx, cy, scale) {
  const [x, y, z] = p;
  return [cx - y * scale + x * scale * 0.12, cy - z * scale + x * scale * 0.05];
}
function dist(a, b) {
  if (!a || !b) return NaN;
  return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2);
}

const canvas = document.getElementById("pose-canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const infoEl = document.getElementById("info");
const recordBtn = document.getElementById("record-btn");
const recordStatusEl = document.getElementById("record-status");
const CX = canvas.width / 2;
const CY = canvas.height * 0.58;
const SCALE = 760;

function drawGrid() {
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e6e9ef";
  ctx.lineWidth = 1;
  for (let i=0; i<canvas.width; i+=50) { ctx.beginPath(); ctx.moveTo(i,0); ctx.lineTo(i,canvas.height); ctx.stroke(); }
  for (let i=0; i<canvas.height; i+=50) { ctx.beginPath(); ctx.moveTo(0,i); ctx.lineTo(canvas.width,i); ctx.stroke(); }
}
function drawSkeleton(pos, color, width, radius, dashed=false) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  if (dashed) ctx.setLineDash([9, 7]);
  for (const [from, to] of UPPER_BONES) {
    const a = from === null ? pos["pelvis"] : pos[from];
    const b = pos[to];
    if (!a || !b) continue;
    const [ax, ay] = project(a, CX, CY, SCALE);
    const [bx, by] = project(b, CX, CY, SCALE);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const joint of ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
    "left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
    "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]) {
    if (!pos[joint]) continue;
    const [x, y] = project(pos[joint], CX, CY, SCALE);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2*Math.PI);
    ctx.fill();
  }
  ctx.restore();
}
function drawErrorLine(a, b, color="#ef4444") {
  if (!a || !b) return;
  const [ax, ay] = project(a, CX, CY, SCALE);
  const [bx, by] = project(b, CX, CY, SCALE);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.3;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.restore();
}
function drawTarget(p, label) {
  if (!p) return;
  const [x, y] = project(p, CX, CY, SCALE);
  ctx.save();
  ctx.strokeStyle = "#dc2626";
  ctx.fillStyle = "#dc2626";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - 8, y);
  ctx.lineTo(x + 8, y);
  ctx.moveTo(x, y - 8);
  ctx.lineTo(x, y + 8);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, 2*Math.PI);
  ctx.fill();
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(label, x + 10, y - 8);
  ctx.restore();
}

function projectPlane(p, plane, cx, cy, scale) {
  const [x, y, z] = p;
  if (plane === "yz") return [cx - y * scale, cy - z * scale];
  if (plane === "xz") return [cx + x * scale, cy - z * scale];
  return [cx + x * scale, cy - y * scale];
}
function drawPointPlane(p, plane, cx, cy, scale, color, radius=4, cross=false) {
  if (!Array.isArray(p) || p.length < 3) return;
  const [px, py] = projectPlane(p, plane, cx, cy, scale);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.8;
  if (cross) {
    ctx.beginPath();
    ctx.moveTo(px - 6, py);
    ctx.lineTo(px + 6, py);
    ctx.moveTo(px, py - 6);
    ctx.lineTo(px, py + 6);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.arc(px, py, radius, 0, 2*Math.PI);
  ctx.fill();
  ctx.restore();
}
function drawLinePlane(a, b, plane, cx, cy, scale, color="#dc2626") {
  if (!Array.isArray(a) || !Array.isArray(b)) return;
  const [ax, ay] = projectPlane(a, plane, cx, cy, scale);
  const [bx, by] = projectPlane(b, plane, cx, cy, scale);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.0;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.restore();
}
function drawQpInputTargetPanel(target, activeWrist, targetSide, activeTargetErr) {
  if (!Array.isArray(target) || target.length < 3) return;
  const panelX = canvas.width - 278;
  const panelY = 22;
  const panelW = 254;
  const panelH = 218;
  const scale = 300;
  const views = [
    ["Y-Z", "yz", panelX + 67, panelY + 82],
    ["X-Z", "xz", panelX + 188, panelY + 82],
    ["X-Y", "xy", panelX + 128, panelY + 169],
  ];
  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.fillRect(panelX, panelY, panelW, panelH);
  ctx.strokeRect(panelX, panelY, panelW, panelH);
  ctx.fillStyle = "#111827";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText("QP input target (raw)", panelX + 10, panelY + 18);
  ctx.fillStyle = "#374151";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`xyz=[${target.map(v => v.toFixed(3)).join(", ")}]`, panelX + 10, panelY + 34);
  ctx.fillText(`${targetSide === 2 ? "right" : targetSide === 1 ? "left" : "active"} err=${Number.isFinite(activeTargetErr) ? (activeTargetErr*100).toFixed(2) : "n/a"}cm`, panelX + 10, panelY + 48);
  for (const [label, plane, cx, cy] of views) {
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 45, cy);
    ctx.lineTo(cx + 45, cy);
    ctx.moveTo(cx, cy - 45);
    ctx.lineTo(cx, cy + 45);
    ctx.stroke();
    ctx.fillStyle = "#6b7280";
    ctx.fillText(label, cx - 14, cy - 51);
    if (Array.isArray(activeWrist)) drawLinePlane(activeWrist, target, plane, cx, cy, scale, "#dc2626");
    if (Array.isArray(activeWrist)) drawPointPlane(activeWrist, plane, cx, cy, scale, "#2563eb", 3.5, false);
    drawPointPlane(target, plane, cx, cy, scale, "#dc2626", 3.5, true);
  }
  ctx.restore();
}
function drawFrame(data) {
  drawGrid();
  const q = data.q || new Array(29).fill(0);
  const qRef = data.q_ref;
  const qQpRef = data.q_qp_ref;
  const qCorrected = data.q_corrected;
  const terminalDistance = Number(data.ee_qp_terminal_target_distance_m);
  const qpInputTarget = Array.isArray(data.qp_input_target_position)
    ? data.qp_input_target_position
    : data.ee_qp_terminal_target_position;
  const target = Array.isArray(qpInputTarget) &&
    Number.isFinite(terminalDistance) && terminalDistance >= 0
    ? qpInputTarget : null;
  const targetSide = Number(data.ee_qp_terminal_target_side || 0);
  const cur = computeFK(q);
  if (Array.isArray(data.left_wrist_position) && data.left_wrist_position.length >= 3) {
    cur["left_wrist_yaw_joint"] = data.left_wrist_position.slice(0, 3);
  }
  if (Array.isArray(data.right_wrist_position) && data.right_wrist_position.length >= 3) {
    cur["right_wrist_yaw_joint"] = data.right_wrist_position.slice(0, 3);
  }
  const ref = qRef ? computeFK(qRef) : null;
  const qpRef = qQpRef ? computeFK(qQpRef) : null;
  const corrected = qCorrected ? computeFK(qCorrected) : null;
  if (ref) drawSkeleton(ref, "#16a34a", 2.0, 3.0, true);
  if (qpRef) drawSkeleton(qpRef, "#f97316", 1.9, 2.8, true);
  if (corrected) drawSkeleton(corrected, "#7c3aed", 1.7, 2.5, true);
  drawSkeleton(cur, "#2563eb", 2.4, 3.4, false);
  let lErr = NaN, rErr = NaN, lQpErr = NaN, rQpErr = NaN, lCorrectedErr = NaN, rCorrectedErr = NaN;
  let lTargetErr = NaN, rTargetErr = NaN, activeTargetErr = NaN;
  if (ref) {
    drawErrorLine(cur["left_wrist_yaw_joint"], ref["left_wrist_yaw_joint"], "#ef4444");
    drawErrorLine(cur["right_wrist_yaw_joint"], ref["right_wrist_yaw_joint"], "#ef4444");
    lErr = dist(cur["left_wrist_yaw_joint"], ref["left_wrist_yaw_joint"]);
    rErr = dist(cur["right_wrist_yaw_joint"], ref["right_wrist_yaw_joint"]);
  }
  if (qpRef) {
    drawErrorLine(cur["left_wrist_yaw_joint"], qpRef["left_wrist_yaw_joint"], "#f97316");
    drawErrorLine(cur["right_wrist_yaw_joint"], qpRef["right_wrist_yaw_joint"], "#f97316");
    lQpErr = dist(cur["left_wrist_yaw_joint"], qpRef["left_wrist_yaw_joint"]);
    rQpErr = dist(cur["right_wrist_yaw_joint"], qpRef["right_wrist_yaw_joint"]);
  }
  if (corrected) {
    lCorrectedErr = dist(cur["left_wrist_yaw_joint"], corrected["left_wrist_yaw_joint"]);
    rCorrectedErr = dist(cur["right_wrist_yaw_joint"], corrected["right_wrist_yaw_joint"]);
  }
  if (target) {
    drawTarget(target, targetSide === 2 ? "right QP target" : targetSide === 1 ? "left QP target" : "QP target");
    lTargetErr = dist(cur["left_wrist_yaw_joint"], target);
    rTargetErr = dist(cur["right_wrist_yaw_joint"], target);
    if (targetSide === 1) {
      activeTargetErr = lTargetErr;
      drawErrorLine(cur["left_wrist_yaw_joint"], target, "#dc2626");
    } else if (targetSide === 2) {
      activeTargetErr = rTargetErr;
      drawErrorLine(cur["right_wrist_yaw_joint"], target, "#dc2626");
    } else if (targetSide === 3) {
      activeTargetErr = Math.min(lTargetErr, rTargetErr);
      drawErrorLine(cur["left_wrist_yaw_joint"], target, "#dc2626");
      drawErrorLine(cur["right_wrist_yaw_joint"], target, "#dc2626");
    } else {
      activeTargetErr = Math.min(lTargetErr, rTargetErr);
      drawErrorLine(cur["left_wrist_yaw_joint"], target, "#dc2626");
      drawErrorLine(cur["right_wrist_yaw_joint"], target, "#dc2626");
    }
  }
  const serverLeftTargetErr = Number(data.left_target_error_m);
  const serverRightTargetErr = Number(data.right_target_error_m);
  const serverActiveTargetErr = Number(data.active_target_error_m);
  if (Number.isFinite(serverLeftTargetErr)) lTargetErr = serverLeftTargetErr;
  if (Number.isFinite(serverRightTargetErr)) rTargetErr = serverRightTargetErr;
  if (Number.isFinite(serverActiveTargetErr)) activeTargetErr = serverActiveTargetErr;
  if (target) {
    const activeWrist = targetSide === 2 ? cur["right_wrist_yaw_joint"] : cur["left_wrist_yaw_joint"];
    drawQpInputTargetPanel(target, activeWrist, targetSide, activeTargetErr);
  }
  ctx.fillStyle = "#111827";
  ctx.font = "16px ui-monospace, monospace";
  ctx.fillText("upper body only, QP internal wrist FK when available", 24, 34);
  ctx.font = "13px ui-monospace, monospace";
  ctx.fillText(`left wrist GMR/QP: ${Number.isFinite(lErr) ? (lErr*100).toFixed(2) : "n/a"} / ${Number.isFinite(lQpErr) ? (lQpErr*100).toFixed(2) : "n/a"} cm`, 24, 58);
  ctx.fillText(`right wrist GMR/QP: ${Number.isFinite(rErr) ? (rErr*100).toFixed(2) : "n/a"} / ${Number.isFinite(rQpErr) ? (rQpErr*100).toFixed(2) : "n/a"} cm`, 24, 78);
  ctx.fillText(`static target: ${Number.isFinite(activeTargetErr) ? (activeTargetErr*100).toFixed(2) : "n/a"} cm`, 24, 98);
  return {lErr, rErr, lQpErr, rQpErr, lCorrectedErr, rCorrectedErr, lTargetErr, rTargetErr, activeTargetErr};
}

function fmtAge(v) { return Number.isFinite(v) ? `${v.toFixed(1)} ms` : "none"; }
async function refreshRecordStatus() {
  const r = await fetch("/api/record/status");
  const status = await r.json();
  recordBtn.textContent = status.recording ? "Stop record" : "Start record";
  recordBtn.className = status.recording ? "recording" : "";
  recordStatusEl.textContent = status.recording ? `recording ${status.frames} frames` : "record idle";
}
recordBtn.onclick = async () => {
  const status = await (await fetch("/api/record/status")).json();
  if (status.recording) {
    const stopped = await (await fetch("/api/record/stop", {method:"POST"})).json();
    recordStatusEl.textContent = stopped.saved
      ? `saved ${stopped.frames} frames: ${stopped.csv} | qp: ${stopped.qp_telemetry_csv}`
      : "record stopped, no frames";
  } else {
    const started = await (await fetch("/api/record/start", {method:"POST"})).json();
    recordStatusEl.textContent = `recording to ${started.session_dir}`;
  }
  await refreshRecordStatus();
};
setInterval(refreshRecordStatus, 1000);
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/state`);
  ws.onopen = () => { statusEl.textContent = "connected"; statusEl.className = "ok"; };
  ws.onclose = () => { statusEl.textContent = "disconnected, retrying"; statusEl.className = "warn"; setTimeout(connect, 1000); };
  ws.onerror = () => { statusEl.textContent = "websocket error"; statusEl.className = "warn"; };
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    const err = drawFrame(data);
    infoEl.textContent =
`MuJoCo state: ${data.state_count} msgs
  q key: ${data.state_q_key || "missing"}
  age: ${fmtAge(data.state_age_ms)}

GMR reference: ${data.gmr_count} msgs
  q key: ${data.gmr_q_key || "missing"}
  age: ${fmtAge(data.gmr_age_ms)}
  frame: ${data.gmr_frame ?? "n/a"}
  mode/source: ${data.gmr_control_mode ?? "n/a"} / ${data.gmr_source ?? "n/a"}

QP active reference:
  q key: ${data.qp_ref_q_key || "missing"}
  mode/source: ${data.ee_qp_control_mode ?? "n/a"} / ${data.ee_qp_reference_source ?? "n/a"}
  type: ${data.ee_qp_reference_type ?? "n/a"}
  plan: ${data.ee_qp_plan_id ?? "n/a"} phase=${Number.isFinite(data.ee_qp_plan_phase) ? data.ee_qp_plan_phase.toFixed(3) : "n/a"}
  valid/fallback: ${data.ee_qp_valid ?? "n/a"} / ${data.ee_qp_fallback_reason ?? "n/a"}
  terminal active/link: ${data.ee_qp_terminal_target_active ?? "n/a"} / ${data.ee_qp_terminal_target_link ?? "n/a"}
  QP input target: ${Array.isArray(data.qp_input_target_position) ? data.qp_input_target_position.map(v=>v.toFixed(3)).join(", ") : "n/a"}
  QP input frame: ${data.qp_input_coordinate_frame ?? "n/a"}
  terminal dist deploy: ${Number.isFinite(data.ee_qp_terminal_target_distance_m) ? (data.ee_qp_terminal_target_distance_m*100).toFixed(2) : "n/a"} cm
  servo before/after: ${Number.isFinite(data.ee_qp_servo_error_before_m) ? (data.ee_qp_servo_error_before_m*100).toFixed(2) : "n/a"} / ${Number.isFinite(data.ee_qp_servo_error_after_m) ? (data.ee_qp_servo_error_after_m*100).toFixed(2) : "n/a"} cm

FK wrist error:
  left GMR: ${Number.isFinite(err.lErr) ? (err.lErr*100).toFixed(2) : "n/a"} cm
  right GMR: ${Number.isFinite(err.rErr) ? (err.rErr*100).toFixed(2) : "n/a"} cm
  left QP ref: ${Number.isFinite(err.lQpErr) ? (err.lQpErr*100).toFixed(2) : "n/a"} cm
  right QP ref: ${Number.isFinite(err.rQpErr) ? (err.rQpErr*100).toFixed(2) : "n/a"} cm
  left corrected target: ${Number.isFinite(err.lCorrectedErr) ? (err.lCorrectedErr*100).toFixed(2) : "n/a"} cm
  right corrected target: ${Number.isFinite(err.rCorrectedErr) ? (err.rCorrectedErr*100).toFixed(2) : "n/a"} cm
  left static target: ${Number.isFinite(err.lTargetErr) ? (err.lTargetErr*100).toFixed(2) : "n/a"} cm
  right static target: ${Number.isFinite(err.rTargetErr) ? (err.rTargetErr*100).toFixed(2) : "n/a"} cm
  active static target: ${Number.isFinite(err.activeTargetErr) ? (err.activeTargetErr*100).toFixed(2) : "n/a"} cm`;
  };
}
drawFrame({q:null, q_ref:null});
connect();
refreshRecordStatus();
</script>
</body>
</html>
"""


app = FastAPI()
state_sub: LatestZmqTopic | None = None
gmr_sub: LatestZmqTopic | None = None
recorder: PathWebRecorder | None = None
wrist_fk: MujocoWristFk | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(HTML)


@app.get("/health")
async def health() -> dict[str, Any]:
    state = state_sub.snapshot() if state_sub is not None else {}
    gmr = gmr_sub.snapshot() if gmr_sub is not None else {}
    state_msg = state.get("message") or {}
    return {
        "state_count": state.get("count", 0),
        "state_q_key": state.get("q_key"),
        "state_age_ms": state.get("age_ms"),
        "qp_ref_q_key": "ee_qp_q_ref" if _as_float_list(state_msg.get("ee_qp_q_ref")) is not None else None,
        "qp_corrected_q_key": "ee_qp_q_corrected" if _as_float_list(state_msg.get("ee_qp_q_corrected")) is not None else None,
        "gmr_count": gmr.get("count", 0),
        "gmr_q_key": gmr.get("q_key"),
        "gmr_age_ms": gmr.get("age_ms"),
        "record": recorder.status() if recorder is not None else {},
    }


@app.post("/api/record/start")
async def record_start() -> dict[str, Any]:
    if recorder is None:
        return {"recording": False, "error": "recorder not initialized"}
    return recorder.start("static_ee")


@app.post("/api/record/stop")
async def record_stop() -> dict[str, Any]:
    if recorder is None:
        return {"recording": False, "error": "recorder not initialized"}
    return recorder.stop()


@app.get("/api/record/status")
async def record_status() -> dict[str, Any]:
    if recorder is None:
        return {"recording": False, "frames": 0}
    return recorder.status()


@app.websocket("/ws/state")
async def state_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            state = state_sub.snapshot() if state_sub is not None else {}
            gmr = gmr_sub.snapshot() if gmr_sub is not None else {}
            state_msg = state.get("message") or {}
            gmr_msg = gmr.get("message") or {}
            q_measured = state.get("q")
            qp_ref = _as_float_list(state_msg.get("ee_qp_q_ref"))
            qp_corrected = _as_float_list(state_msg.get("ee_qp_q_corrected"))
            last_action = _as_float_list(state_msg.get("last_action"))
            terminal_position = _as_vec(state_msg.get("ee_qp_terminal_target_position"), 3)
            ee_qp_left_wrist_position = _as_nonzero_vec(state_msg.get("ee_qp_left_wrist_position"), 3)
            ee_qp_right_wrist_position = _as_nonzero_vec(state_msg.get("ee_qp_right_wrist_position"), 3)
            ee_qp_active_wrist_position = _as_nonzero_vec(state_msg.get("ee_qp_active_wrist_position"), 3)
            terminal_distance = state_msg.get("ee_qp_terminal_target_distance_m", math.nan)
            try:
                terminal_distance_float = float(terminal_distance)
            except (TypeError, ValueError):
                terminal_distance_float = math.nan
            if not math.isfinite(terminal_distance_float) or terminal_distance_float < 0.0:
                terminal_position = None
            target_side = state_msg.get("ee_qp_terminal_target_side", 0)
            try:
                target_side_int = int(target_side)
            except (TypeError, ValueError):
                target_side_int = 0
            target_link = state_msg.get("ee_qp_terminal_target_link")
            if not isinstance(target_link, str) or not target_link:
                if target_side_int == 1:
                    target_link = "left_wrist_yaw_link"
                elif target_side_int == 2:
                    target_link = "right_wrist_yaw_link"
                elif target_side_int == 3:
                    target_link = "both_wrist_yaw_link"
                else:
                    target_link = "none"
            left_wrist_position = None
            right_wrist_position = None
            left_target_error_m = math.nan
            right_target_error_m = math.nan
            active_target_error_m = math.nan
            if wrist_fk is not None and q_measured is not None:
                left_wrist_position = wrist_fk.wrist_position(q_measured, "left_wrist_yaw_link")
                right_wrist_position = wrist_fk.wrist_position(q_measured, "right_wrist_yaw_link")
            if ee_qp_left_wrist_position is not None:
                left_wrist_position = ee_qp_left_wrist_position
            if ee_qp_right_wrist_position is not None:
                right_wrist_position = ee_qp_right_wrist_position
            if terminal_position is not None:
                target_np = np.asarray(terminal_position, dtype=np.float64)
                if left_wrist_position is not None:
                    left_target_error_m = float(np.linalg.norm(np.asarray(left_wrist_position) - target_np))
                if right_wrist_position is not None:
                    right_target_error_m = float(np.linalg.norm(np.asarray(right_wrist_position) - target_np))
                if target_side_int == 1:
                    active_target_error_m = left_target_error_m
                elif target_side_int == 2:
                    active_target_error_m = right_target_error_m
                elif target_side_int == 3:
                    active_target_error_m = min(left_target_error_m, right_target_error_m)
                elif math.isfinite(left_target_error_m) or math.isfinite(right_target_error_m):
                    active_target_error_m = min(left_target_error_m, right_target_error_m)
            if terminal_position is not None and math.isfinite(terminal_distance_float):
                active_target_error_m = terminal_distance_float
                if target_side_int == 1:
                    left_target_error_m = terminal_distance_float
                elif target_side_int == 2:
                    right_target_error_m = terminal_distance_float
            payload = {
                "q": q_measured,
                "q_ref": gmr.get("q"),
                "q_qp_ref": qp_ref,
                "q_corrected": qp_corrected,
                "last_action": last_action,
                "left_wrist_position": left_wrist_position,
                "right_wrist_position": right_wrist_position,
                "ee_qp_left_wrist_position": ee_qp_left_wrist_position,
                "ee_qp_right_wrist_position": ee_qp_right_wrist_position,
                "ee_qp_active_wrist_position": ee_qp_active_wrist_position,
                "left_target_error_m": left_target_error_m,
                "right_target_error_m": right_target_error_m,
                "active_target_error_m": active_target_error_m,
                "state_q_key": state.get("q_key"),
                "gmr_q_key": gmr.get("q_key"),
                "qp_ref_q_key": "ee_qp_q_ref" if qp_ref is not None else "",
                "qp_corrected_q_key": "ee_qp_q_corrected" if qp_corrected is not None else "",
                "state_age_ms": state.get("age_ms"),
                "gmr_age_ms": gmr.get("age_ms"),
                "state_count": state.get("count", 0),
                "gmr_count": gmr.get("count", 0),
                "gmr_frame": gmr_msg.get("frame_index"),
                "gmr_source": gmr_msg.get("source"),
                "gmr_control_mode": gmr_msg.get("control_mode"),
                "ee_qp_valid": state_msg.get("ee_qp_valid"),
                "ee_qp_fallback_reason": state_msg.get("ee_qp_fallback_reason"),
                "ee_qp_reference_source": state_msg.get("ee_qp_reference_source"),
                "ee_qp_reference_type": state_msg.get("ee_qp_reference_type"),
                "ee_qp_control_mode": state_msg.get("ee_qp_control_mode"),
                "ee_qp_plan_id": state_msg.get("ee_qp_plan_id"),
                "ee_qp_plan_phase": state_msg.get("ee_qp_plan_phase"),
                "ee_qp_terminal_target_active": state_msg.get("ee_qp_terminal_target_active"),
                "ee_qp_terminal_target_side": target_side_int,
                "ee_qp_terminal_target_link": target_link,
                "ee_qp_terminal_target_distance_m": terminal_distance_float,
                "ee_qp_terminal_capture_radius_m": state_msg.get("ee_qp_terminal_capture_radius_m"),
                "ee_qp_terminal_target_position": terminal_position,
                "qp_input_target_position": terminal_position,
                "qp_input_target_side": target_side_int,
                "qp_input_target_link": target_link,
                "qp_input_coordinate_frame": state_msg.get(
                    "ee_qp_terminal_target_coordinate_frame",
                    "pelvis_identity_fk",
                ),
                "ee_qp_servo_error_before_m": state_msg.get("ee_qp_servo_error_before_m"),
                "ee_qp_servo_error_after_m": state_msg.get("ee_qp_servo_error_after_m"),
            }
            if recorder is not None:
                record_payload = dict(payload)
                record_payload["_state_msg"] = state_msg
                recorder.append(record_payload)
            await ws.send_text(json.dumps(json_safe(payload), allow_nan=False))
            await asyncio.sleep(1.0 / 30.0)
    except WebSocketDisconnect:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--state-topic", default="g1_debug")
    parser.add_argument("--gmr-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--gmr-topic", default="gmr_ref")
    parser.add_argument("--record-output-dir", default="data/path_web_records")
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("sonic_runtime/gear_sonic_deploy/g1/g1_29dof.xml"),
        help="MuJoCo model for server-side wrist FK and recording.",
    )
    return parser.parse_args()


def main() -> None:
    global state_sub, gmr_sub, recorder, wrist_fk
    args = parse_args()
    recorder = PathWebRecorder(Path(args.record_output_dir))
    try:
        wrist_fk = MujocoWristFk(args.model_xml)
        print(f"[web] loaded FK model {args.model_xml}")
    except Exception as exc:
        wrist_fk = None
        print(f"[web] WARNING: server-side FK disabled: {exc}")
    state_sub = LatestZmqTopic(
        args.state_endpoint,
        args.state_topic,
        (
            "body_q",
            "q_measured",
            "q",
            "sonic_body_q",
            "body_q_measured",
        ),
        "mujoco-state",
    )
    gmr_sub = LatestZmqTopic(
        args.gmr_endpoint,
        args.gmr_topic,
        (
            "q_ref",
            "q_mujoco",
            "q_plan",
        ),
        "gmr-ref",
    )
    state_sub.start()
    gmr_sub.start()
    print(f"[web] open http://127.0.0.1:{args.port}")
    print(f"[web] health http://127.0.0.1:{args.port}/health")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
