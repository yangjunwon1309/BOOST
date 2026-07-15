"""
g1_web_server.py — G1 텔레오퍼레이션 웹 뷰어
Thor 보드에서 실행. LCM으로 G1_CAMERA_FRAME + G1_ROBOT_STATE 수신 후
FastAPI WebSocket + MJPEG으로 브라우저에 실시간 스트리밍.

실행:
  pip install fastapi uvicorn lcm
  python3 g1_web_server.py --lcm-url "udpm://239.255.76.67:7667?ttl=1" --port 8080

접속:
  http://192.168.0.27:8080  (Thor Wi-Fi IP)
"""

import argparse
import asyncio
import ctypes
import json
import struct
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

# ── LCM 수신 ──────────────────────────────────────────────────────────────────
import lcm

CAMERA_MAGIC = 0x4D524143
ROBOT_STATE_MAGIC = 0x31545347

class G1LcmReceiver:
    def __init__(self, lcm_url: str):
        self.lcm_url = lcm_url
        self._lc = lcm.LCM(lcm_url)
        self._lock = threading.Lock()

        self._latest_jpeg: Optional[bytes] = None
        self._latest_state: Optional[dict] = None
        self._frame_count = 0
        self._state_count = 0

        self._lc.subscribe("G1_CAMERA_FRAME", self._on_camera)
        self._lc.subscribe("G1_ROBOT_STATE",  self._on_state)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[lcm] Receiving on {self.lcm_url}")

    def stop(self):
        self._running = False

    def _recv_loop(self):
        while self._running:
            self._lc.handle_timeout(10)

    def _on_camera(self, channel, data):
        if len(data) < 24:
            return
        magic, w, h, jpeg_size = struct.unpack_from("<IIII", data, 0)
        ts = struct.unpack_from("<Q", data, 16)[0]
        if magic != CAMERA_MAGIC:
            return
        jpeg = bytes(data[24: 24 + jpeg_size])
        with self._lock:
            self._latest_jpeg = jpeg
            self._frame_count += 1

    def _on_state(self, channel, data):
        if len(data) < 33:
            return
        magic, version, flags = struct.unpack_from("<IHH", data, 0)
        if magic != ROBOT_STATE_MAGIC:
            return

        # RobotStatePacket 오프셋 (#pragma pack 1):
        # magic(4) + version(2) + flags(2) + sequence(8)
        # + lowstate_sequence(8) + send_time_us(8) + mode_machine(1) = 33
        off = 33
        base_quat  = list(struct.unpack_from("<4f", data, off)); off += 16
        base_gyro  = list(struct.unpack_from("<3f", data, off)); off += 12
        base_accel = list(struct.unpack_from("<3f", data, off)); off += 12
        torso_quat = list(struct.unpack_from("<4f", data, off)); off += 16
        torso_gyro = list(struct.unpack_from("<3f", data, off)); off += 12
        torso_accel= list(struct.unpack_from("<3f", data, off)); off += 12
        q          = list(struct.unpack_from("<29f", data, off)); off += 116
        dq         = list(struct.unpack_from("<29f", data, off)); off += 116
        tau        = list(struct.unpack_from("<29f", data, off))

        mode = struct.unpack_from("B", data, 32)[0]  # mode_machine at offset 32

        state = {
            "mode": mode,
            "base_quat": base_quat,
            "base_gyro": base_gyro,
            "q": q,
            "dq": dq,
        }
        with self._lock:
            self._latest_state = state
            self._state_count += 1

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_state(self) -> Optional[dict]:
        with self._lock:
            return self._latest_state

    @property
    def frame_count(self): return self._frame_count

    @property
    def state_count(self): return self._state_count


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI()
receiver: Optional[G1LcmReceiver] = None

# WebSocket 연결 관리
ws_clients: list[WebSocket] = []
ws_lock = asyncio.Lock()


# ── MJPEG 스트림 ──────────────────────────────────────────────────────────────
async def mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = receiver.get_jpeg() if receiver else None
        if jpeg:
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                   b"\r\n" + jpeg + b"\r\n")
        await asyncio.sleep(1 / 30)


@app.get("/stream")
async def camera_stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── WebSocket — joint state ───────────────────────────────────────────────────
@app.websocket("/ws/state")
async def state_ws(ws: WebSocket):
    await ws.accept()
    async with ws_lock:
        ws_clients.append(ws)
    try:
        while True:
            state = receiver.get_state() if receiver else None
            if state:
                await ws.send_text(json.dumps(state))
            await asyncio.sleep(1 / 30)  # 30Hz
    except WebSocketDisconnect:
        pass
    finally:
        async with ws_lock:
            ws_clients.remove(ws)


# ── 메인 HTML ─────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>G1 Teleoperation View</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#eee; font-family:monospace;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:8px 16px; background:#1a1a2e; display:flex;
           align-items:center; gap:16px; }
  header h1 { font-size:16px; color:#00d4ff; }
  #status { font-size:12px; color:#888; margin-left:auto; }
  #main { display:flex; flex:1; overflow:hidden; }

  /* 좌: 카메라 */
  #cam-panel { width:640px; min-width:640px; display:flex;
               flex-direction:column; align-items:center;
               justify-content:center; padding:16px; gap:8px; }
  #cam-panel h2 { font-size:13px; color:#00d4ff; }
  #cam-img { width:640px; height:480px; border:1px solid #333;
             background:#000; display:block; }

  /* 우: 3D 포즈 */
  #pose-panel { flex:1; display:flex; flex-direction:column;
                align-items:center; padding:16px; gap:8px; }
  #pose-panel h2 { font-size:13px; color:#00d4ff; }
  #pose-canvas { border:1px solid #333; background:#0a0a1a; }

  /* 하단 joint bar */
  #joint-bar { padding:8px 16px; background:#1a1a2e; font-size:10px;
               color:#666; overflow:hidden; white-space:nowrap; }
</style>
</head>
<body>
<header>
  <h1>🤖 G1 Teleoperation View</h1>
  <div id="status">Connecting...</div>
</header>
<div id="main">
  <div id="cam-panel">
    <h2>📷 Ego View (D435i IR, 640×480)</h2>
    <img id="cam-img" src="/stream" alt="camera stream">
  </div>
  <div id="pose-panel">
    <h2>🦾 Joint Pose (3D projection)</h2>
  <canvas id="pose-canvas" width="520" height="500"></canvas>  </div>
</div>
<div id="joint-bar" id="joint-bar">joint data loading...</div>

<script>
// ============================================================
// G1 정확한 FK — URDF g1_29dof_with_hand 기반
// 단위: meter, 좌표계: URDF (x=앞, y=좌, z=위)
// IsaacLab 인덱스 0~28 → URDF joint name 매핑
// ============================================================

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

const JOINT_NAMES_DISPLAY = [
  "L_HipPitch","L_HipRoll","L_HipYaw","L_Knee","L_AnklePitch","L_AnkleRoll",
  "R_HipPitch","R_HipRoll","R_HipYaw","R_Knee","R_AnklePitch","R_AnkleRoll",
  "WaistYaw","WaistRoll","WaistPitch",
  "L_ShldrPitch","L_ShldrRoll","L_ShldrYaw","L_Elbow","L_WristRoll","L_WristPitch","L_WristYaw",
  "R_ShldrPitch","R_ShldrRoll","R_ShldrYaw","R_Elbow","R_WristRoll","R_WristPitch","R_WristYaw"
];

// URDF에서 추출한 정확한 joint 데이터
// [parent_link, child_link, origin_xyz, origin_rpy, axis_xyz]
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

// 스켈레톤 뼈대 연결 [joint_name, color]
// child link의 origin이 joint 위치
const SKELETON_BONES = [
  // waist chain
  ["waist_yaw_joint","#4af"],
  ["waist_roll_joint","#4af"],
  ["waist_pitch_joint","#4af"],
  // L leg
  ["left_hip_pitch_joint","#6cf"],
  ["left_hip_roll_joint","#6cf"],
  ["left_hip_yaw_joint","#6cf"],
  ["left_knee_joint","#6cf"],
  ["left_ankle_pitch_joint","#6cf"],
  ["left_ankle_roll_joint","#6cf"],
  // R leg
  ["right_hip_pitch_joint","#f96"],
  ["right_hip_roll_joint","#f96"],
  ["right_hip_yaw_joint","#f96"],
  ["right_knee_joint","#f96"],
  ["right_ankle_pitch_joint","#f96"],
  ["right_ankle_roll_joint","#f96"],
  // L arm
  ["left_shoulder_pitch_joint","#4f8"],
  ["left_shoulder_roll_joint","#4f8"],
  ["left_shoulder_yaw_joint","#4f8"],
  ["left_elbow_joint","#4f8"],
  ["left_wrist_roll_joint","#4f8"],
  ["left_wrist_pitch_joint","#4f8"],
  // R arm
  ["right_shoulder_pitch_joint","#fa4"],
  ["right_shoulder_roll_joint","#fa4"],
  ["right_shoulder_yaw_joint","#fa4"],
  ["right_elbow_joint","#fa4"],
  ["right_wrist_roll_joint","#fa4"],
  ["right_wrist_pitch_joint","#fa4"],
];

// ── 수학 유틸리티 ─────────────────────────────────────────────────────────────
function mat4Identity() {
  return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
}

function mat4Mul(A, B) {
  const C = new Array(16).fill(0);
  for (let r=0;r<4;r++) for (let c=0;c<4;c++)
    for (let k=0;k<4;k++) C[r*4+c] += A[r*4+k]*B[k*4+c];
  return C;
}

function mat4Translation(x,y,z) {
  return [1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1];
}

// RPY → 회전 행렬 (URDF 규칙: R = Rz(yaw)*Ry(pitch)*Rx(roll))
function mat4RPY(r,p,y) {
  const cr=Math.cos(r),sr=Math.sin(r);
  const cp=Math.cos(p),sp=Math.sin(p);
  const cy=Math.cos(y),sy=Math.sin(y);
  return [
    cy*cp,  cy*sp*sr-sy*cr,  cy*sp*cr+sy*sr,  0,
    sy*cp,  sy*sp*sr+cy*cr,  sy*sp*cr-cy*sr,  0,
    -sp,    cp*sr,           cp*cr,            0,
    0,      0,               0,                1
  ];
}

// axis-angle 회전 행렬
function mat4AxisAngle(ax,ay,az, angle) {
  const c=Math.cos(angle), s=Math.sin(angle), t=1-c;
  const l=Math.sqrt(ax*ax+ay*ay+az*az);
  if (l<1e-9) return mat4Identity();
  ax/=l; ay/=l; az/=l;
  return [
    t*ax*ax+c,    t*ax*ay-s*az, t*ax*az+s*ay, 0,
    t*ax*ay+s*az, t*ay*ay+c,    t*ay*az-s*ax, 0,
    t*ax*az-s*ay, t*ay*az+s*ax, t*az*az+c,    0,
    0,            0,            0,            1
  ];
}

function applyMat4(M, x, y, z) {
  return [
    M[0]*x+M[1]*y+M[2]*z+M[3],
    M[4]*x+M[5]*y+M[6]*z+M[7],
    M[8]*x+M[9]*y+M[10]*z+M[11]
  ];
}

// ── FK 계산 ───────────────────────────────────────────────────────────────────
// link_name → world transform (4x4)
function computeFK(q) {
  // joint name → q value 매핑
  const qMap = {};
  for (let i=0; i<JOINT_NAMES_URDF.length; i++) {
    qMap[JOINT_NAMES_URDF[i]] = q[i] || 0;
  }

  // link world transform 캐시
  const linkT = { "pelvis": mat4Identity() };

  // joint 처리 순서 (부모 먼저)
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

  const jointWorldPos = {};

  for (const jname of ORDER) {
    const [parent, child, xyz, rpy, axis] = JOINT_DATA[jname];
    const parentT = linkT[parent] || mat4Identity();

    // joint frame = parent * T(origin) * R(rpy)
    const T = mat4Translation(...xyz);
    const R = mat4RPY(...rpy);
    const jointLocalT = mat4Mul(T, R);
    const jointWorldT = mat4Mul(parentT, jointLocalT);

    // joint world position
    const wp = applyMat4(jointWorldT, 0, 0, 0);
    jointWorldPos[jname] = wp;

    // child link transform = jointWorld * R(q_i around axis)
    const qi = qMap[jname] || 0;
    const Rq = mat4AxisAngle(...axis, qi);
    linkT[child] = mat4Mul(jointWorldT, Rq);
  }

  return jointWorldPos;
}

// ── 투영 (URDF: x=앞, y=좌, z=위 → 화면: 정면 뷰) ───────────────────────────
// 화면: px = center - y*scale (y축 = 좌우)
//       py = center - z*scale (z축 = 높이)
// x(앞뒤)는 약간 원근감으로 반영
function project(worldPos, cx, cy, scale) {
  const [x, y, z] = worldPos;
  const px = cx - y * scale + x * scale * 0.12;
  const py = cy - z * scale + x * scale * 0.05;
  return [px, py];
}

// ── 렌더링 ────────────────────────────────────────────────────────────────────
const canvas = document.getElementById("pose-canvas");
const ctx2d = canvas.getContext("2d");
const CX = canvas.width / 2;
const CY = canvas.height * 0.30;  // 0.55 → 0.42 (골반을 위로)
const SCALE = 350;

function drawPose(q) {
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);

  // 배경 그리드
  ctx2d.strokeStyle = "#1a1a2e";
  ctx2d.lineWidth = 1;
  for (let i=0;i<canvas.width;i+=40){ ctx2d.beginPath();ctx2d.moveTo(i,0);ctx2d.lineTo(i,canvas.height);ctx2d.stroke(); }
  for (let i=0;i<canvas.height;i+=40){ ctx2d.beginPath();ctx2d.moveTo(0,i);ctx2d.lineTo(canvas.width,i);ctx2d.stroke(); }

  const jointPos = computeFK(q);

  // 뼈대: parent joint → child joint
  const BONE_PAIRS = [
    // waist
    [null, "waist_yaw_joint","#4af"],
    ["waist_yaw_joint","waist_roll_joint","#4af"],
    ["waist_roll_joint","waist_pitch_joint","#4af"],
    // L leg
    [null,"left_hip_pitch_joint","#6cf"],
    ["left_hip_pitch_joint","left_hip_roll_joint","#6cf"],
    ["left_hip_roll_joint","left_hip_yaw_joint","#6cf"],
    ["left_hip_yaw_joint","left_knee_joint","#6cf"],
    ["left_knee_joint","left_ankle_pitch_joint","#6cf"],
    ["left_ankle_pitch_joint","left_ankle_roll_joint","#6cf"],
    // R leg
    [null,"right_hip_pitch_joint","#f96"],
    ["right_hip_pitch_joint","right_hip_roll_joint","#f96"],
    ["right_hip_roll_joint","right_hip_yaw_joint","#f96"],
    ["right_hip_yaw_joint","right_knee_joint","#f96"],
    ["right_knee_joint","right_ankle_pitch_joint","#f96"],
    ["right_ankle_pitch_joint","right_ankle_roll_joint","#f96"],
    // L arm
    ["waist_pitch_joint","left_shoulder_pitch_joint","#4f8"],
    ["left_shoulder_pitch_joint","left_shoulder_roll_joint","#4f8"],
    ["left_shoulder_roll_joint","left_shoulder_yaw_joint","#4f8"],
    ["left_shoulder_yaw_joint","left_elbow_joint","#4f8"],
    ["left_elbow_joint","left_wrist_roll_joint","#4f8"],
    ["left_wrist_roll_joint","left_wrist_pitch_joint","#4f8"],
    // R arm
    ["waist_pitch_joint","right_shoulder_pitch_joint","#fa4"],
    ["right_shoulder_pitch_joint","right_shoulder_roll_joint","#fa4"],
    ["right_shoulder_roll_joint","right_shoulder_yaw_joint","#fa4"],
    ["right_shoulder_yaw_joint","right_elbow_joint","#fa4"],
    ["right_elbow_joint","right_wrist_roll_joint","#fa4"],
    ["right_wrist_roll_joint","right_wrist_pitch_joint","#fa4"],
    // waist → hips
    ["waist_yaw_joint","left_hip_pitch_joint","#4af"],
    ["waist_yaw_joint","right_hip_pitch_joint","#4af"],
  ];

  // 골반(pelvis) 위치 = [0,0,0]
  const pelvisScreen = [CX, CY];

  for (const [from, to, color] of BONE_PAIRS) {
    const toPos = jointPos[to];
    if (!toPos) continue;
    const [tx, ty] = project(toPos, CX, CY, SCALE);

    let fx, fy;
    if (from === null) {
      [fx, fy] = pelvisScreen;
    } else {
      const fromPos = jointPos[from];
      if (!fromPos) continue;
      [fx, fy] = project(fromPos, CX, CY, SCALE);
    }

    ctx2d.beginPath();
    ctx2d.moveTo(fx, fy);
    ctx2d.lineTo(tx, ty);
    ctx2d.strokeStyle = color;
    ctx2d.lineWidth = 3;
    ctx2d.stroke();
  }

  // 관절 점
  ctx2d.fillStyle = "#fff";
  for (const [jname, pos] of Object.entries(jointPos)) {
    const [px, py] = project(pos, CX, CY, SCALE);
    ctx2d.beginPath();
    ctx2d.arc(px, py, 3.5, 0, Math.PI*2);
    ctx2d.fill();
  }

  // 골반 점
  ctx2d.beginPath();
  ctx2d.arc(CX, CY, 5, 0, Math.PI*2);
  ctx2d.fillStyle = "#ff4";
  ctx2d.fill();

  // 주요 joint 라벨
  const LABELS = ["left_knee_joint","right_knee_joint",
                  "left_elbow_joint","right_elbow_joint",
                  "waist_pitch_joint"];
  ctx2d.font = "9px monospace";
  ctx2d.fillStyle = "#888";
  for (const jname of LABELS) {
    const pos = jointPos[jname];
    if (!pos) continue;
    const [px, py] = project(pos, CX, CY, SCALE);
    const label = jname.replace("_joint","").replace(/_/g," ");
    ctx2d.fillText(label, px+5, py-5);
  }
}

// ── WebSocket 연결 ────────────────────────────────────────────────────────────
const statusEl = document.getElementById("status");
const jointBarEl = document.getElementById("joint-bar");
let lastQ = new Array(29).fill(0);

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws/state`);
  ws.onopen = () => { statusEl.textContent = "🟢 Connected"; };
  ws.onclose = () => {
    statusEl.textContent = "🔴 Disconnected — retrying...";
    setTimeout(connectWS, 1000);
  };
  ws.onerror = () => { statusEl.textContent = "⚠️ WS Error"; };
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    lastQ = d.q;
    drawPose(d.q);

    // joint bar 업데이트
    const txt = d.q.map((v,i) =>
      `${JOINT_NAMES_DISPLAY[i].padEnd(14)}${v.toFixed(3)}`
    ).join("  |  ");
    jointBarEl.textContent = txt;

    // status에 IMU 표시
    const [qw, qx, qy, qz] = d.base_quat;
    statusEl.textContent =
      `🟢 mode=${d.mode} | quat=[${qw.toFixed(2)},${qx.toFixed(2)},${qy.toFixed(2)},${qz.toFixed(2)}]`;
  };
}

connectWS();
drawPose(new Array(29).fill(0));
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.get("/health")
async def health():
    if not receiver:
        return {"status": "no receiver"}
    return {
        "status": "ok",
        "frames": receiver.frame_count,
        "states": receiver.state_count,
    }


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
def main():
    global receiver

    parser = argparse.ArgumentParser(description="G1 Web Teleoperation Server")
    parser.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    receiver = G1LcmReceiver(args.lcm_url)
    receiver.start()

    print(f"[web] Server starting on http://{args.host}:{args.port}")
    print(f"[web] Access from VR: http://192.168.0.27:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()