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
import json
import math
import threading
import time
from typing import Any

import msgpack
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
  .kv { font-size:12px; line-height:1.65; color:#c8d0dc; white-space:pre-wrap; }
  .warn { color:#f59e0b; }
  .ok { color:#86efac; }
</style>
</head>
<body>
<header>
  <h1>G1 Upper Body FK Overlay</h1>
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
function drawFrame(data) {
  drawGrid();
  const q = data.q || new Array(29).fill(0);
  const qRef = data.q_ref;
  const qQpRef = data.q_qp_ref;
  const qCorrected = data.q_corrected;
  const cur = computeFK(q);
  const ref = qRef ? computeFK(qRef) : null;
  const qpRef = qQpRef ? computeFK(qQpRef) : null;
  const corrected = qCorrected ? computeFK(qCorrected) : null;
  if (ref) drawSkeleton(ref, "#16a34a", 2.0, 3.0, true);
  if (qpRef) drawSkeleton(qpRef, "#f97316", 1.9, 2.8, true);
  if (corrected) drawSkeleton(corrected, "#7c3aed", 1.7, 2.5, true);
  drawSkeleton(cur, "#2563eb", 2.4, 3.4, false);
  let lErr = NaN, rErr = NaN, lQpErr = NaN, rQpErr = NaN, lCorrectedErr = NaN, rCorrectedErr = NaN;
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
  ctx.fillStyle = "#111827";
  ctx.font = "16px ui-monospace, monospace";
  ctx.fillText("upper body only, pelvis-local FK", 24, 34);
  ctx.font = "13px ui-monospace, monospace";
  ctx.fillText(`left wrist GMR/QP: ${Number.isFinite(lErr) ? (lErr*100).toFixed(2) : "n/a"} / ${Number.isFinite(lQpErr) ? (lQpErr*100).toFixed(2) : "n/a"} cm`, 24, 58);
  ctx.fillText(`right wrist GMR/QP: ${Number.isFinite(rErr) ? (rErr*100).toFixed(2) : "n/a"} / ${Number.isFinite(rQpErr) ? (rQpErr*100).toFixed(2) : "n/a"} cm`, 24, 78);
  return {lErr, rErr, lQpErr, rQpErr, lCorrectedErr, rCorrectedErr};
}

function fmtAge(v) { return Number.isFinite(v) ? `${v.toFixed(1)} ms` : "none"; }
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
  plan: ${data.ee_qp_plan_id ?? "n/a"} phase=${Number.isFinite(data.ee_qp_plan_phase) ? data.ee_qp_plan_phase.toFixed(3) : "n/a"}
  valid/fallback: ${data.ee_qp_valid ?? "n/a"} / ${data.ee_qp_fallback_reason ?? "n/a"}

FK wrist error:
  left GMR: ${Number.isFinite(err.lErr) ? (err.lErr*100).toFixed(2) : "n/a"} cm
  right GMR: ${Number.isFinite(err.rErr) ? (err.rErr*100).toFixed(2) : "n/a"} cm
  left QP ref: ${Number.isFinite(err.lQpErr) ? (err.lQpErr*100).toFixed(2) : "n/a"} cm
  right QP ref: ${Number.isFinite(err.rQpErr) ? (err.rQpErr*100).toFixed(2) : "n/a"} cm
  left corrected target: ${Number.isFinite(err.lCorrectedErr) ? (err.lCorrectedErr*100).toFixed(2) : "n/a"} cm
  right corrected target: ${Number.isFinite(err.rCorrectedErr) ? (err.rCorrectedErr*100).toFixed(2) : "n/a"} cm`;
  };
}
drawFrame({q:null, q_ref:null});
connect();
</script>
</body>
</html>
"""


app = FastAPI()
state_sub: LatestZmqTopic | None = None
gmr_sub: LatestZmqTopic | None = None


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
    }


@app.websocket("/ws/state")
async def state_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            state = state_sub.snapshot() if state_sub is not None else {}
            gmr = gmr_sub.snapshot() if gmr_sub is not None else {}
            state_msg = state.get("message") or {}
            gmr_msg = gmr.get("message") or {}
            qp_ref = _as_float_list(state_msg.get("ee_qp_q_ref"))
            qp_corrected = _as_float_list(state_msg.get("ee_qp_q_corrected"))
            payload = {
                "q": state.get("q"),
                "q_ref": gmr.get("q"),
                "q_qp_ref": qp_ref,
                "q_corrected": qp_corrected,
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
                "ee_qp_control_mode": state_msg.get("ee_qp_control_mode"),
                "ee_qp_plan_id": state_msg.get("ee_qp_plan_id"),
                "ee_qp_plan_phase": state_msg.get("ee_qp_plan_phase"),
            }
            await ws.send_text(json.dumps(payload, allow_nan=True))
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
    return parser.parse_args()


def main() -> None:
    global state_sub, gmr_sub
    args = parse_args()
    state_sub = LatestZmqTopic(
        args.state_endpoint,
        args.state_topic,
        (
            "body_q_measured",
            "body_q",
            "sonic_body_q",
            "q_measured",
            "q",
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
