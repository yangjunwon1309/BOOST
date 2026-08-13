"""Live G1 RGB-D + SAM3 segmentation web server for Jetson Thor."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import lcm
import msgpack
import numpy as np
import uvicorn
import yaml
import zmq
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from .ee_target import GoalPublisher, Workspace, make_goal_payload
from .depth_alignment import align_depth_to_color
from .geometry import (
    CameraIntrinsics,
    transform_points,
)
from .lcm_protocol import (
    DEPTH_SOURCE_ALIGNED_TO_COLOR,
    DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    decode_calibration,
    decode_camera,
    decode_depth,
    decode_robot_state,
)
from .robot_geometry import (
    HAND_AXIS_LOCAL,
    RobotStateBuffer,
    angular_velocity_from_rotations,
    g1_hand_base_rotation,
    g1_grasp_center_position,
    g1_shoulder_position,
    hand_frame_profile,
    gravity_up_in_pelvis,
    pelvis_to_torso,
)
from .rgbd_sync import RgbdPair, RgbdSynchronizer, depth_at_rgb_size
from .object_point_filter import (
    ObjectPointFilterConfig,
    extract_filtered_object_cloud,
    project_camera_point_to_pixel,
)
from .sam2_backend import Sam2HieraTinyBackend
from .efficienttam_backend import EfficientTamTi512Backend
from .yolo11_backend import Yolo11SegBackend
from .sam3_backend import TensorRTSam3Backend, TorchSam3Backend
from .tasc_belief import (
    BeliefInputObject,
    HandInput,
    HierarchicalTascBelief,
    ParallelTascBelief,
    StableObjectIdTracker,
)
from .visualization import render_segmented_rgbd


LOG = logging.getLogger(__name__)
BELIEF_HAND_COLOR = {"left": (255, 190, 40), "right": (60, 160, 255)}
BELIEF_DIM_ALPHA = 0.40
BELIEF_ARMED_BOX_PX = 3
BELIEF_DIM_BOX_PX = 1
BELIEF_ARMED_CROSS_PX = 24
BELIEF_DIM_CROSS_PX = 14
BELIEF_DASH_PX = 9


def parse_cpu_set(spec: str) -> set[int]:
    """Parse Linux cpuset syntax such as ``8`` or ``9-13,15``."""
    cpus: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU id: {item}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set is empty")
    return cpus


def pin_current_thread_from_env(env_name: str) -> dict[str, Any]:
    """Pin only the calling native thread and return status metadata."""
    requested_text = os.environ.get(env_name, "").strip()
    allowed = set(os.sched_getaffinity(0))
    if not requested_text:
        return {
            "environment": env_name,
            "requested": None,
            "effective": sorted(allowed),
            "pinned": False,
        }
    requested = parse_cpu_set(requested_text)
    effective = requested & allowed
    if not effective:
        raise ValueError(
            f"{env_name}={requested_text} has no CPUs inside process affinity "
            f"{sorted(allowed)}"
        )
    os.sched_setaffinity(0, effective)
    return {
        "environment": env_name,
        "requested": requested_text,
        "effective": sorted(os.sched_getaffinity(0)),
        "pinned": True,
    }


def detect_lan_host() -> str:
    """Best-effort LAN IP used only for the printed browser URL."""
    candidates: list[str] = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            candidates.append(str(probe.getsockname()[0]))
        finally:
            probe.close()
    except OSError:
        pass
    try:
        hostname = socket.gethostname()
        _name, _aliases, addresses = socket.gethostbyname_ex(hostname)
        candidates.extend(addresses)
    except OSError:
        pass
    for candidate in candidates:
        if candidate and not candidate.startswith("127."):
            return candidate
    return "127.0.0.1"


def display_host(bind_host: str, advertise_host: str | None) -> str:
    if advertise_host:
        return advertise_host
    if bind_host in ("0.0.0.0", "::", ""):
        return detect_lan_host()
    return bind_host


class SharedControlHandSubscriber:
    """Read the state-machine A/B hold selection without affecting control."""

    def __init__(self, endpoint: str, topic: str) -> None:
        self.topic = str(topic)
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.socket.connect(str(endpoint))
        self.selected_hand = "none"
        self.active = False
        self.revision = 0

    def close(self) -> None:
        self.socket.close(0)

    def drain(self) -> bool:
        changed = False
        prefix = self.topic.encode("utf-8")
        while True:
            try:
                raw = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return changed
            try:
                if not raw.startswith(prefix):
                    continue
                payload = msgpack.unpackb(
                    raw[len(prefix) :], raw=False, strict_map_key=False
                )
                if not isinstance(payload, dict):
                    continue
                hand = str(payload.get("selected_hand", "none")).lower()
                if hand not in {"left", "right", "none"}:
                    continue
                active = bool(payload.get("active", False)) and hand != "none"
                next_hand = hand if active else "none"
                changed = changed or next_hand != self.selected_hand
                self.selected_hand = next_hand
                self.active = active
                self.revision = int(payload.get("revision", self.revision) or 0)
            except Exception:
                # A malformed diagnostic heartbeat must never stop SAM3 inference.
                continue


def ensure_port_available(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host and host != "0.0.0.0" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, int(port)))
        except OSError as error:
            raise RuntimeError(
                f"web port {host}:{port} is not available. "
                "Use --port <free_port> or stop the existing web server."
            ) from error


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>G1 SAM3 RGB-D View</title>
<style>
body{margin:0;background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif}
header{padding:12px 18px;background:#161b22;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;color:#58a6ff}.grow{flex:1}input{background:#0d1117;color:#fff;border:1px solid #30363d;border-radius:6px;padding:8px;width:230px}
button{background:#238636;color:white;border:0;border-radius:6px;padding:9px 14px;cursor:pointer}button.danger{background:#da3633}button:disabled{opacity:.45;cursor:not-allowed}
select{width:100%;margin:10px 0;background:#0d1117;color:#fff;border:1px solid #30363d;border-radius:6px;padding:8px}
main{padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:16px}
#view{width:100%;max-height:calc(100vh - 110px);object-fit:contain;background:#000;border:1px solid #30363d}table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}th,td{padding:5px;border-bottom:1px solid #30363d;text-align:left}.belief-selected{background:#22314a}.belief-dim{opacity:.55}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:12px}.ok{color:#3fb950}.bad{color:#f85149}pre{white-space:pre-wrap;line-height:1.5;color:#8b949e}.row{display:flex;gap:8px}.row button{flex:1}#targetmsg{min-height:38px;color:#8b949e;margin-top:10px}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>G1 · SAM3 RGB-D</h1><div class="grow"></div>
<input id="prompt" placeholder="person, box, tool, robot part"><button onclick="setPrompt()">Apply concepts</button></header>
<main><div><img id="view" src="/stream.mjpg"><div class="card"><b>Per-hand belief</b><table><thead><tr><th>object</th><th>L</th><th>R</th></tr></thead><tbody id="beliefrows"></tbody></table></div></div><aside>
<div class="card"><b>Belief target publisher</b>
<div id="targetmsg">Continuously publishing both hand belief targets; A/B in PATH selects the receiving arm.</div></div>
<div class="card"><b>Pipeline status</b><pre id="status">connecting…</pre></div></aside></main>
<script>
let state={};
async function api(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw new Error(data.detail||JSON.stringify(data));return data}
function renderTarget(s){state=s;const t=s.ee_target||{};const blockers=(t.blockers||[]).join('; ');const active=`Continuously publishing both hand belief targets at ${Number(t.publish_hz||0).toFixed(1)} Hz${t.last_published_label?` — ${t.last_published_label}`:''}`;document.getElementById('targetmsg').textContent=active||t.last_result||blockers;}
function beliefCell(view,selected,side){if(!view)return '<td class="belief-dim">-</td>';const p=Number(view.object_probability||0).toFixed(2);const j=view.joint_confidence==null?'-':Number(view.joint_confidence).toFixed(2);const c=side===selected?'belief-selected':'belief-dim';return `<td class="${c}">${p} / ${j}</td>`}
function renderBeliefTable(s){const selected=String((s.tasc_belief||{}).selected_hand||'none');const rows=(s.instances||[]).map(o=>{const h=o.tasc_hand_beliefs||{};return `<tr><td>${o.label||o.tasc_object_id||'object'}</td>${beliefCell(h.left,selected,'left')}${beliefCell(h.right,selected,'right')}</tr>`}).join('');document.getElementById('beliefrows').innerHTML=rows||'<tr><td colspan="3">no segmented belief candidates</td></tr>';}
async function update(){try{const s=await (await fetch('/api/status')).json();document.getElementById('status').textContent=JSON.stringify(s,null,2);renderTarget(s);renderBeliefTable(s);if(document.activeElement!==document.getElementById('prompt'))document.getElementById('prompt').value=(s.concepts||[]).join(', ');}catch(e){document.getElementById('status').textContent=String(e)}}
async function setPrompt(){const prompt=document.getElementById('prompt').value.trim();await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})});update()}
setInterval(update,1000);update();
</script></body></html>"""


def placeholder_jpeg(message: str, width: int = 960, height: int = 540) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(image, message, (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (180, 210, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("placeholder JPEG encoding failed")
    return encoded.tobytes()


class PromptRequest(BaseModel):
    prompt: str


class SendTargetRequest(BaseModel):
    object_index: int
    frame_timestamp_us: int
    label: str
    center_goal_m: list[float] | None = None


def load_target_geometry(
    camera_info_path: Path, extrinsics_path: Path
) -> tuple[CameraIntrinsics | None, np.ndarray | None, np.ndarray | None, list[str]]:
    blockers: list[str] = []
    intrinsics = None
    torso_camera = None
    static_goal_camera = None
    try:
        info = yaml.safe_load(camera_info_path.read_text())
        required = ("width", "height", "fx", "fy", "cx", "cy")
        if any(info.get(key) is None for key in required):
            raise ValueError("camera intrinsics contain null values")
        intrinsics = CameraIntrinsics(
            int(info["width"]), int(info["height"]), float(info["fx"]),
            float(info["fy"]), float(info["cx"]), float(info["cy"]),
        )
        intrinsics.validate()
    except Exception as error:
        # Production intrinsics arrive on G1_CAMERA_CALIBRATION. Keep this
        # file only as a backward-compatible source for already-aligned data.
        intrinsics = None
    try:
        data = yaml.safe_load(extrinsics_path.read_text())
        key = "T_torso_cam" if data.get("T_torso_cam") is not None else "T_goal_cam"
        if data.get(key) is None:
            raise ValueError("T_torso_cam is null (T_goal_cam legacy fallback is also null)")
        transform = np.asarray(data[key], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"{key} must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1.0e-6):
            raise ValueError(f"{key} has an invalid homogeneous last row")
        if key == "T_torso_cam":
            torso_camera = transform
        else:
            static_goal_camera = transform
    except Exception as error:
        blockers.append(f"goal transform unavailable: {error}")
    return intrinsics, torso_camera, static_goal_camera, blockers


def parse_concepts(text: str) -> list[str]:
    concepts = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not concepts:
        raise ValueError("at least one concept is required")
    if len(concepts) > 12:
        raise ValueError("at most 12 comma-separated concepts are allowed")
    return concepts


def suppress_duplicate_masks(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    iou_threshold: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not len(masks):
        return masks, boxes, scores, labels
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    keep: list[int] = []
    for index in np.argsort(-scores):
        duplicate = False
        for kept in keep:
            intersection = np.logical_and(masks[index], masks[kept]).sum()
            union = areas[index] + areas[kept] - intersection
            if union > 0 and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(index))
    keep_array = np.asarray(keep, dtype=np.int64)
    return masks[keep_array], boxes[keep_array], scores[keep_array], [labels[i] for i in keep]


def select_tracking_candidate(
    instances: list[dict[str, Any]],
    label: str,
    last_position: np.ndarray | None,
    max_jump_m: float,
) -> tuple[dict[str, Any] | None, str]:
    """Select the same labeled instance nearest the previous 3-D center."""
    matching = [
        obj for obj in instances
        if str(obj.get("label")) == str(label)
    ]
    if not matching:
        return None, "missing"
    valid = [
        obj for obj in matching
        if obj.get("center_goal_m") is not None
    ]
    if not valid:
        return None, "depth_unavailable"
    if last_position is None:
        return max(valid, key=lambda obj: float(obj.get("score", 0.0))), "ok"
    previous = np.asarray(last_position, dtype=np.float64).reshape(3)
    selected = min(
        valid,
        key=lambda obj: float(
            np.linalg.norm(
                np.asarray(obj["center_goal_m"], dtype=np.float64).reshape(3)
                - previous
            )
        ),
    )
    distance = float(
        np.linalg.norm(
            np.asarray(selected["center_goal_m"], dtype=np.float64).reshape(3)
            - previous
        )
    )
    if distance > max_jump_m:
        return None, "jump"
    return selected, "ok"


def _dashed_rectangle(
    image: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int],
    color: tuple[int, int, int], thickness: int, dash_px: int = BELIEF_DASH_PX,
) -> None:
    x0, y0 = top_left
    x1, y1 = bottom_right
    step = max(2, int(dash_px))
    for x in range(x0, x1, 2 * step):
        cv2.line(image, (x, y0), (min(x + step, x1), y0), color, thickness)
        cv2.line(image, (x, y1), (min(x + step, x1), y1), color, thickness)
    for y in range(y0, y1, 2 * step):
        cv2.line(image, (x0, y), (x0, min(y + step, y1)), color, thickness)
        cv2.line(image, (x1, y), (x1, min(y + step, y1)), color, thickness)


def _goal_point_to_pixel(
    position_goal_m: Any, t_goal_cam: np.ndarray | None, intrinsics: CameraIntrinsics | None,
) -> tuple[int, int] | None:
    if position_goal_m is None or t_goal_cam is None or intrinsics is None:
        return None
    try:
        homogeneous = np.ones(4, dtype=np.float64)
        homogeneous[:3] = np.asarray(position_goal_m, dtype=np.float64).reshape(3)
        camera = (np.linalg.inv(np.asarray(t_goal_cam, dtype=np.float64)) @ homogeneous)[:3]
        return project_camera_point_to_pixel(camera, intrinsics)
    except Exception:
        return None


def _draw_hand_label(
    layer: np.ndarray, side: str, target: dict[str, Any], color: tuple[int, int, int],
    *, anchor: tuple[int, int], armed: bool,
) -> None:
    parts = ["L" if side == "left" else "R"]
    confidence = target.get("joint_confidence")
    if isinstance(confidence, (int, float)):
        parts.append(f"{float(confidence):.2f}")
    grip = target.get("grip_index")
    if isinstance(grip, int) and grip >= 0:
        parts.append(f"g{grip}")
    relation = target.get("orientation_relation")
    if relation in {"parallel", "perpendicular"}:
        parts.append("par" if relation == "parallel" else "perp")
    if armed:
        parts.append("A/B")
    text = " ".join(parts)
    scale, thickness = (0.52, 2) if armed else (0.44, 1)
    if armed:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = anchor
        cv2.rectangle(layer, (x - 2, y - th - baseline - 2), (x + tw + 2, y + 2), color, -1)
        cv2.putText(layer, text, (x, y - baseline + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    else:
        cv2.putText(layer, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_orientation_arrow(
    layer: np.ndarray, target: dict[str, Any], color: tuple[int, int, int], thickness: int,
    t_goal_cam: np.ndarray | None, intrinsics: CameraIntrinsics | None, length_m: float = 0.08,
) -> None:
    origin = target.get("orientation_source_position_m")
    axis = target.get("orientation_axis_world")
    if origin is None or axis is None:
        return
    try:
        start = np.asarray(origin, dtype=np.float64).reshape(3)
        direction = np.asarray(axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1.0e-9:
            return
        end = start + direction / norm * float(length_m)
    except (TypeError, ValueError):
        return
    p0 = _goal_point_to_pixel(start, t_goal_cam, intrinsics)
    p1 = _goal_point_to_pixel(end, t_goal_cam, intrinsics)
    if p0 is not None and p1 is not None:
        cv2.arrowedLine(layer, p0, p1, color, max(1, thickness), cv2.LINE_AA, tipLength=0.3)


def _draw_hand_belief(
    layer: np.ndarray, objects: list[dict[str, Any]], side: str, target: dict[str, Any], *,
    armed: bool, t_goal_cam: np.ndarray | None, intrinsics: CameraIntrinsics | None,
    label_slot: int, draw_normal: bool,
) -> None:
    object_id = str(target.get("object_id") or "")
    if not object_id:
        return
    color = BELIEF_HAND_COLOR.get(side, (200, 200, 200))
    height, width = layer.shape[:2]
    box_px = BELIEF_ARMED_BOX_PX if armed else BELIEF_DIM_BOX_PX
    cross_px = BELIEF_ARMED_CROSS_PX if armed else BELIEF_DIM_CROSS_PX
    selected = next((obj for obj in objects if str(obj.get("tasc_object_id", "")) == object_id), None)
    if selected is not None:
        box = selected.get("box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box]
            x0, x1 = [int(v) for v in np.clip([x0, x1], 0, width - 1)]
            y0, y1 = [int(v) for v in np.clip([y0, y1], 0, height - 1)]
            if armed:
                cv2.rectangle(layer, (x0, y0), (x1, y1), color, box_px)
            else:
                _dashed_rectangle(layer, (x0, y0), (x1, y1), color, box_px)
            _draw_hand_label(layer, side, target, color, anchor=(x0, min(height - 6, y1 + 18 + label_slot * 20)), armed=armed)
    pixel = _goal_point_to_pixel(target.get("position_goal_m"), t_goal_cam, intrinsics)
    if pixel is not None:
        cv2.drawMarker(layer, pixel, color, cv2.MARKER_CROSS, cross_px, box_px + 1)
    if draw_normal and target.get("orientation_relation") in {"parallel", "perpendicular"}:
        _draw_orientation_arrow(layer, target, color, box_px, t_goal_cam, intrinsics)


def _draw_belief_legend(overlay: np.ndarray, hands: dict[str, Any], selected_hand: str) -> None:
    y = 20
    for side in ("left", "right"):
        color = BELIEF_HAND_COLOR[side]
        target = (hands.get(side) or {}).get("target") or {}
        state = "A/B" if side == selected_hand else "idle"
        if not target.get("object_id"):
            state = "no target"
        cv2.rectangle(overlay, (10, y - 10), (24, y + 2), color, -1)
        cv2.putText(overlay, f"{'L' if side == 'left' else 'R'} {state}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        y += 20


def draw_belief_overlay(
    overlay: np.ndarray,
    objects: list[dict[str, Any]],
    belief_telemetry: dict[str, Any],
    *,
    t_goal_cam: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
    draw_normals: bool = True,
) -> None:
    hands = belief_telemetry.get("hands") or {}
    if not isinstance(hands, dict):
        return
    selected_hand = str(belief_telemetry.get("selected_hand", "none")).lower()
    armed_sides, dim_sides = [], []
    for side in ("left", "right"):
        hand_data = hands.get(side)
        if not isinstance(hand_data, dict) or not (hand_data.get("target") or {}).get("object_id"):
            continue
        (armed_sides if side == selected_hand else dim_sides).append(side)
    if dim_sides:
        scratch = overlay.copy()
        for slot, side in enumerate(dim_sides):
            _draw_hand_belief(scratch, objects, side, hands[side]["target"], armed=False, t_goal_cam=t_goal_cam, intrinsics=intrinsics, label_slot=slot, draw_normal=draw_normals)
        cv2.addWeighted(scratch, BELIEF_DIM_ALPHA, overlay, 1.0 - BELIEF_DIM_ALPHA, 0.0, overlay)
    for slot, side in enumerate(armed_sides, start=len(dim_sides)):
        _draw_hand_belief(overlay, objects, side, hands[side]["target"], armed=True, t_goal_cam=t_goal_cam, intrinsics=intrinsics, label_slot=slot, draw_normal=draw_normals)
    _draw_belief_legend(overlay, hands, selected_hand)


def draw_belief_highlight(*args, **kwargs) -> None:
    """Compatibility wrapper for callers outside this module."""
    draw_belief_overlay(*args, **kwargs)


class SegmentationRuntime:
    def __init__(
        self,
        *,
        lcm_url: str,
        rgb_channel: str,
        depth_channel: str,
        calibration_channel: str,
        robot_state_channel: str,
        prompt: str,
        threshold: float,
        min_area_px: int,
        target_hz: float,
        sync_tolerance_ms: float,
        min_depth_m: float,
        max_depth_m: float,
        allow_unaligned: bool,
        checkpoint: Path | None,
        inference_backend: str,
        tensorrt_config: dict[str, Any],
        sam2_config: dict[str, Any],
        efficienttam_config: dict[str, Any],
        yolo11_config: dict[str, Any],
        depth_self_filter_config: dict[str, Any],
        object_point_filter_config: dict[str, Any],
        intrinsics: CameraIntrinsics | None,
        t_torso_cam: np.ndarray | None,
        static_t_goal_cam: np.ndarray | None,
        geometry_blockers: list[str],
        publisher: GoalPublisher | None,
        workspace: Workspace,
        target_config: dict[str, Any],
        belief_config: dict[str, Any],
        shared_hand_selection_endpoint: str,
        shared_hand_selection_topic: str,
    ):
        self.lcm_url = lcm_url
        self.rgb_channel = rgb_channel
        self.depth_channel = depth_channel
        self.calibration_channel = calibration_channel
        self.robot_state_channel = robot_state_channel
        self.threshold = threshold
        self.min_area_px = min_area_px
        self.target_hz = target_hz
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.allow_unaligned = allow_unaligned
        self.checkpoint = checkpoint
        self.inference_backend = inference_backend
        self.tensorrt_config = tensorrt_config
        self.sam2_config = sam2_config
        self.efficienttam_config = efficienttam_config
        self.yolo11_config = yolo11_config
        self.depth_self_filter_config = dict(depth_self_filter_config)
        self.object_point_filter = ObjectPointFilterConfig.from_dict(
            object_point_filter_config
        )
        self.intrinsics = intrinsics
        self.t_torso_cam = t_torso_cam
        self.static_t_goal_cam = static_t_goal_cam
        self.calibration = None
        self.state_buffer = RobotStateBuffer()
        self.torso_estimator = str(target_config.get("torso_estimator", "dual_imu"))
        self.robot_state_max_delta_ms = float(target_config.get("robot_state_max_delta_ms", 40.0))
        self.max_imu_fk_disagreement_deg = float(target_config.get("max_imu_fk_disagreement_deg", 12.0))
        self.max_waist_speed_rad_s = float(target_config.get("max_waist_speed_rad_s", 0.15))
        self.geometry_blockers = list(geometry_blockers)
        self.publisher = publisher
        self.workspace = workspace
        self.target_config = target_config
        self.belief_overlay_normals = bool(target_config.get("overlay_normals", True))
        self.tasc_gravity_warned = False
        self.tasc_hand_model = str(
            belief_config.get("hand_model", "dex1")
        ).strip().lower()
        self.tasc_hand_profile = hand_frame_profile(self.tasc_hand_model)
        # Perception and control must share one physical hand-axis convention.
        belief_config = dict(belief_config)
        orientation_config = dict(belief_config.get("orientation", {}))
        canonical_axis = HAND_AXIS_LOCAL[self.tasc_hand_model]
        configured_axis = orientation_config.get("hand_axis_local")
        if configured_axis is not None:
            try:
                if not np.allclose(
                    np.asarray(configured_axis, dtype=np.float64).reshape(3),
                    canonical_axis,
                    atol=1.0e-9,
                ):
                    LOG.warning(
                        "overriding tasc orientation.hand_axis_local=%s with canonical %s for %s",
                        configured_axis,
                        canonical_axis.tolist(),
                        self.tasc_hand_model,
                    )
            except (TypeError, ValueError):
                LOG.warning(
                    "invalid tasc orientation.hand_axis_local=%r; using canonical axis",
                    configured_axis,
                )
        orientation_config["hand_axis_local"] = canonical_axis.tolist()
        belief_config["orientation"] = orientation_config
        self.parallel_tasc_belief = ParallelTascBelief(belief_config)
        # Compatibility handle for legacy diagnostics. Runtime inference uses
        # parallel_tasc_belief for both hands below.
        self.tasc_belief = self.parallel_tasc_belief.trackers["left"]
        self.tasc_object_tracker = StableObjectIdTracker(
            max_distance_m=float(belief_config.get("object_track_max_distance_m", 0.25)),
            missing_ttl_frames=int(
                belief_config.get("object_track_missing_ttl_frames", 5)
            ),
        )
        self.tasc_use_contact_target = bool(belief_config.get("use_contact_target", True))
        self.tasc_pointcloud_max_points = int(
            belief_config.get(
                "object_max_points",
                belief_config.get("grip", {}).get("max_points", 8000),
            )
        )
        self.tasc_grasp_center_local_m = np.asarray(
            belief_config.get(
                "grasp_center_local_m",
                self.tasc_hand_profile["grasp_center_local"],
            ),
            dtype=np.float64,
        ).reshape(3)
        self.tasc_last_eef_by_link: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, int]] = {}
        self.shared_hand_selection = SharedControlHandSubscriber(
            shared_hand_selection_endpoint, shared_hand_selection_topic
        )
        self.target_side = str(target_config.get("side", "left"))
        self.target_link = str(
            target_config.get("target_link", f"{self.target_side}_wrist_yaw_link")
        )
        self.target_quaternion = tuple(
            float(value) for value in target_config.get("quaternion_wxyz", [1, 0, 0, 0])
        )
        self.target_offset = np.asarray(
            target_config.get("position_offset_goal_m", [0, 0, 0]), dtype=np.float64
        ).reshape(3)
        self.target_max_age_sec = float(target_config.get("max_frame_age_sec", 1.5))
        self.target_min_depth_points = int(target_config.get("min_depth_points", 100))
        self.target_bypass_checks = bool(target_config.get("bypass_checks", False))
        self.target_auto_publish_hz = float(
            target_config.get("auto_publish_hz", target_config.get("publish_hz", 5.0))
        )
        self.target_tracking_max_jump_m = float(
            target_config.get("tracking_max_jump_m", 0.25)
        )
        self.target_tracking_lost_grace_frames = int(
            target_config.get("tracking_lost_grace_frames", 3)
        )
        self.target_last_stream_publish_monotonic = 0.0
        # Belief targets are a continuous perception output.  The PATH bridge
        # alone decides whether A or B injects either hand target into IPOPT.
        self.belief_publishing_enabled = True
        self.last_belief_published_label: str | None = None
        self.target_lock = threading.Lock()
        self.target_tracking: dict[str, Any] | None = None
        self.target_tracking_missing_frames = 0
        self.last_processed_monotonic = 0.0
        self.geometry_update_hz = max(
            0.1, float(self.parallel_tasc_belief.config.geometry_update_hz)
        )
        self.belief_update_hz = max(
            0.1, float(self.parallel_tasc_belief.config.belief_update_hz)
        )
        self.sync = RgbdSynchronizer(sync_tolerance_ms, capacity=30)
        self.bus = lcm.LCM(lcm_url)
        self.lock = threading.Lock()
        self.pair_ready = threading.Condition(self.lock)
        self.frame_ready = threading.Condition(self.lock)
        self.running = False
        self.pending_pair: RgbdPair | None = None
        self.output_jpeg = placeholder_jpeg("Waiting for G1 RGB-D frames...")
        self.output_version = 0
        self.latest_belief_observation: dict[str, Any] | None = None
        self.latest_belief_geometry = None
        self.latest_belief_robot_state = None
        self.latest_visual: dict[str, Any] = {
            "masks": [],
            "boxes": [],
            "scores": [],
            "labels": [],
            "frame_timestamp_us": 0,
        }
        self.geometry_revision = 0
        self.belief_revision = 0
        self.concepts = parse_concepts(prompt)
        self.status: dict[str, Any] = {
            "model_ready": False,
            "inference_backend": inference_backend,
            "backend_metadata": None,
            "lcm_rgb_frames": 0,
            "lcm_depth_frames": 0,
            "lcm_calibration_frames": 0,
            "lcm_robot_state_frames": 0,
            "paired_frames": 0,
            "processed_frames": 0,
            "dropped_for_inference": 0,
            "decode_errors": 0,
            "last_error": None,
            "instances": [],
            "last_depth_source": None,
            "ee_target_last_result": None,
            "geometry_runtime_error": None,
            "alignment_ready": False,
            "robot_state_delta_ms": None,
            "torso_transform": None,
            "timing_ms": None,
            "live_video_frames": 0,
            "geometry_update_hz": self.geometry_update_hz,
            "belief_update_hz": self.belief_update_hz,
            "geometry_revision": 0,
            "belief_revision": 0,
            "thread_affinity": {},
            "tasc_belief": {"enabled": self.parallel_tasc_belief.config.enabled, "hands": {}},
        }
        self.threads: list[threading.Thread] = []
        self.bus.subscribe(rgb_channel, self._on_rgb)
        self.bus.subscribe(depth_channel, self._on_depth)
        self.bus.subscribe(calibration_channel, self._on_calibration)
        self.bus.subscribe(robot_state_channel, self._on_robot_state)

    def start(self) -> None:
        self.running = True
        self.threads = [
            threading.Thread(target=self._lcm_loop, name="lcm-rgbd", daemon=True),
            threading.Thread(target=self._inference_loop, name="sam3-inference", daemon=True),
            threading.Thread(target=self._belief_loop, name="tasc-belief", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def pin_current_thread(self, role: str, env_name: str) -> None:
        try:
            metadata = pin_current_thread_from_env(env_name)
        except Exception as error:
            metadata = {
                "environment": env_name,
                "pinned": False,
                "error": str(error),
            }
        metadata["native_thread_id"] = threading.get_native_id()
        with self.lock:
            affinity = dict(self.status.get("thread_affinity", {}))
            affinity[role] = metadata
            self.status["thread_affinity"] = affinity

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.pair_ready.notify_all()
            self.frame_ready.notify_all()
        for thread in self.threads:
            thread.join(timeout=3)
        if self.publisher is not None:
            self.publisher.close()
        self.shared_hand_selection.close()

    def _belief_target_link(self) -> str | None:
        return self.target_link

    def _belief_target_side(self) -> str:
        link = self._belief_target_link()
        return "right" if link and link.startswith("right_") else "left"

    def set_prompt(self, prompt: str) -> None:
        concepts = parse_concepts(prompt)
        with self.lock:
            self.concepts = concepts

    def snapshot(self) -> bytes:
        with self.lock:
            return self.output_jpeg

    def status_snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.status)
            blockers = self._target_blockers_locked()
            tracking = self.target_tracking
            result.update(
                concepts=list(self.concepts),
                rgb_queue=len(self.sync.rgb),
                depth_queue=len(self.sync.depth),
                sync_matched=self.sync.matched,
                sync_dropped_rgb=self.sync.dropped_rgb,
                sync_dropped_depth=self.sync.dropped_depth,
                allow_unaligned=self.allow_unaligned,
                calibration_channel=self.calibration_channel,
                calibration_id=(self.calibration.calibration_id if self.calibration else None),
                robot_state_channel=self.robot_state_channel,
                ee_target={
                    "publisher_enabled": self.publisher is not None,
                    "armed": False,
                    "belief_publishing": self.belief_publishing_enabled,
                    "tracking": tracking is not None,
                    "tracked_label": tracking.get("label") if tracking else None,
                    "tracked_index": tracking.get("last_index") if tracking else None,
                    "missing_frames": self.target_tracking_missing_frames,
                    "publish_hz": self.target_auto_publish_hz,
                    "goal_frame": "pico_head_position_waist",
                    "target_link": self._belief_target_link() or self.target_link,
                    "selected_hand": self.shared_hand_selection.selected_hand,
                    "selection_active": self.shared_hand_selection.active,
                    "selection_revision": self.shared_hand_selection.revision,
                    "bypass_checks": self.target_bypass_checks,
                    "blockers": blockers,
                    "last_published_label": self.last_belief_published_label,
                    "last_result": self.status.get("ee_target_last_result"),
                },
            )
            return json.loads(json.dumps(result))

    def _target_blockers_locked(self) -> list[str]:
        if self.target_bypass_checks:
            return []
        blockers = list(self.geometry_blockers)
        if self.status.get("geometry_runtime_error"):
            blockers.append(str(self.status["geometry_runtime_error"]))
        if self.publisher is None:
            blockers.append("publisher disabled; restart with --enable-ee-publishing")
        if not self.status.get("alignment_ready"):
            blockers.append("calibrated depth-to-color alignment is not ready")
        if not any(
            obj.get("center_goal_m") is not None
            for obj in self.status.get("instances", [])
        ):
            blockers.append("no segmented object has a valid goal-frame 3-D center")
        transform = self.status.get("torso_transform") or {}
        if float(transform.get("max_waist_speed_rad_s", 0.0)) > self.max_waist_speed_rad_s:
            blockers.append(
                f"waist is moving too fast for centimeter target timing "
                f"(limit {self.max_waist_speed_rad_s:.2f} rad/s)"
            )
        age = time.monotonic() - self.last_processed_monotonic
        if self.last_processed_monotonic <= 0 or age > self.target_max_age_sec:
            blockers.append(f"segmentation frame is stale (limit {self.target_max_age_sec:.1f}s)")
        return blockers

    def stop_target_tracking(
        self, reason: str = "Belief target publishing stopped."
    ) -> dict[str, Any]:
        with self.target_lock, self.lock:
            self.target_tracking = None
            self.target_tracking_missing_frames = 0
            self.status["ee_target_last_result"] = (
                "Belief target stream is continuous; manual tracking stopped."
            )
        return {"belief_publishing": True, "tracking": False, "sent_null": False}

    def start_belief_publishing(self) -> dict[str, Any]:
        with self.target_lock, self.lock:
            if self.publisher is None:
                raise ValueError("publisher is disabled; restart with --enable-ee-publishing")
            self.belief_publishing_enabled = True
            self.target_tracking = None
            self.target_tracking_missing_frames = 0
            self.target_last_stream_publish_monotonic = 0.0
            self.status["ee_target_last_result"] = (
                "Belief target publishing enabled; streaming best belief target or null fallback."
            )
            return {
                "belief_publishing": True,
                "target_link": self.target_link,
                "publish_hz": self.target_auto_publish_hz,
            }

    def _goal_payload_for_object_locked(
        self,
        selected: dict[str, Any],
        *,
        streaming_goal: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        center = (
            selected.get("target_contact_goal_m")
            if bool(getattr(self, "tasc_use_contact_target", False))
            else None
        ) or selected.get("center_goal_m")
        if center is None:
            raise ValueError("selected object has no valid goal-frame center")
        position = np.asarray(center, dtype=np.float64) + self.target_offset
        if not np.isfinite(position).all():
            raise ValueError("selected object has a non-finite goal-frame center")
        if self.target_bypass_checks:
            workspace_info = {
                "workspace_mode": "bypass",
                "workspace_reason": "validation_disabled_by_config",
                "workspace_clamped": False,
                "requested_position": position.tolist(),
            }
        else:
            workspace_info = self.workspace.validate(position, self.target_side)
        payload = make_goal_payload(
            position,
            side=self.target_side,
            target_link=self.target_link,
            quaternion=self.target_quaternion,
            duration_sec=float(self.target_config.get("duration_sec", 2.0)),
            final_hold_sec=float(self.target_config.get("final_hold_sec", 1.0)),
            workspace_info=workspace_info,
            metadata={
                "frame_timestamp_us": int(self.status["last_frame_ts"]),
                "object_index": int(selected["index"]),
                "label": str(selected["label"]),
                "score": float(selected["score"]),
                "center_camera_m": selected.get("center_camera_m"),
                "center_goal_m": selected.get("center_goal_m"),
                "target_contact_goal_m": selected.get("target_contact_goal_m"),
                "tasc_object_id": selected.get("tasc_object_id"),
                "tasc_selected_grip_index": selected.get("tasc_selected_grip_index"),
                "tasc_joint_confidence": selected.get("tasc_joint_confidence"),
                "streaming_goal": bool(streaming_goal),
            },
        )
        if (
            selected.get("target_contact_goal_m") is not None
            and bool(getattr(self, "tasc_use_contact_target", False))
        ):
            payload["source"] = "sam3_tasc_contact"
        if streaming_goal:
            payload["streaming_goal"] = True
            payload["goal_uid"] = (
                f"sam3_stream:{self.target_link}:"
                f"{str(selected.get('label', 'object'))}:"
                f"{int(selected.get('index', 0))}"
            )
        if selected.get("tasc_object_probability") is not None:
            payload["confidence"] = float(selected.get("tasc_object_probability"))
            payload["object_confidence"] = float(selected.get("tasc_object_probability"))
        if selected.get("tasc_joint_confidence") is not None:
            payload["joint_confidence"] = float(selected.get("tasc_joint_confidence"))
        return payload, position

    def _null_goal_payload_locked(self, reason: str, side: str | None = None) -> dict[str, Any]:
        target_side = side if side in {"left", "right"} else self._belief_target_side()
        target_link = f"{target_side}_wrist_yaw_link"
        return {
            "goal_uid": f"sam3_belief:null:{target_link}",
            "force_replan": False,
            "streaming_goal": True,
            "target_valid": False,
            "target_link": target_link,
            "side": target_side,
            "position": None,
            "quaternion": list(self.target_quaternion),
            "source": "sam3_tasc_belief_null",
            "coordinate_frame": "pico_head_position_waist",
            "target_position_frame": "pico_head_position_waist",
            "duration_sec": float(self.target_config.get("duration_sec", 2.0)),
            "final_hold_sec": float(self.target_config.get("final_hold_sec", 1.0)),
            "perception": {
                "frame_timestamp_us": int(self.status.get("last_frame_ts", 0) or 0),
                "reason": str(reason),
                "streaming_goal": True,
                "target_valid": False,
            },
        }

    def _belief_payload_for_hand_locked(
        self, target_side: str
    ) -> tuple[dict[str, Any], np.ndarray | None, str]:
        target_link = f"{target_side}_wrist_yaw_link"
        belief = self.status.get("tasc_belief") or {}
        hand_belief = (
            belief.get("hands", {}).get(target_side, {})
            if isinstance(belief, dict)
            else {}
        )
        target = hand_belief.get("target") if isinstance(hand_belief, dict) else None
        if not isinstance(target, dict):
            return self._null_goal_payload_locked("belief target unavailable", target_side), None, "null"
        try:
            position = np.asarray(target.get("position_goal_m"), dtype=np.float64).reshape(3)
        except Exception:
            return self._null_goal_payload_locked("belief target position unavailable", target_side), None, "null"
        position = position + self.target_offset
        if not np.isfinite(position).all():
            return self._null_goal_payload_locked("belief target position is non-finite", target_side), None, "null"

        object_id = str(target.get("object_id", ""))
        selected = next(
            (
                obj for obj in self.status.get("instances", [])
                if str(obj.get("tasc_object_id", "")) == object_id
            ),
            None,
        )
        label = str((selected or {}).get("label") or object_id or "belief_target")
        object_confidence = float(
            target.get(
                "object_confidence",
                (selected or {}).get(
                    "tasc_object_probability",
                    target.get("joint_confidence", 1.0),
                ),
            )
        )
        joint_confidence = float(target.get("joint_confidence", object_confidence))
        orientation_value = target.get("orientation_xyzw")
        has_orientation_quaternion = orientation_value is not None
        orientation_xyzw = np.asarray(
            orientation_value if has_orientation_quaternion else [
                self.target_quaternion[1], self.target_quaternion[2],
                self.target_quaternion[3], self.target_quaternion[0],
            ], dtype=np.float64,
        ).reshape(4)
        orientation_wxyz = (
            float(orientation_xyzw[3]), float(orientation_xyzw[0]),
            float(orientation_xyzw[1]), float(orientation_xyzw[2]),
        )
        if self.target_bypass_checks:
            workspace_info = {
                "workspace_mode": "bypass",
                "workspace_reason": "validation_disabled_by_config",
                "workspace_clamped": False,
                "requested_position": position.tolist(),
            }
        else:
            try:
                workspace_info = self.workspace.validate(position, target_side)
            except ValueError as error:
                return self._null_goal_payload_locked(str(error), target_side), None, "null"
        payload = make_goal_payload(
            position,
            side=target_side,
            target_link=target_link,
            quaternion=orientation_wxyz,
            duration_sec=float(self.target_config.get("duration_sec", 2.0)),
            final_hold_sec=float(self.target_config.get("final_hold_sec", 1.0)),
            workspace_info=workspace_info,
            metadata={
                "frame_timestamp_us": int(self.status.get("last_frame_ts", 0) or 0),
                "object_index": int((selected or {}).get("index", -1)),
                "label": label,
                "score": float((selected or {}).get("score", 0.0)),
                "center_camera_m": (selected or {}).get("center_camera_m"),
                "center_goal_m": (selected or {}).get("center_goal_m"),
                "target_contact_goal_m": target.get("position_goal_m"),
                "tasc_object_id": object_id,
                "tasc_selected_grip_index": target.get("grip_index"),
                "tasc_object_probability": object_confidence,
                "tasc_joint_confidence": joint_confidence,
                "orientation_object_id": target.get("orientation_object_id"),
                "orientation_relation": target.get("orientation_relation"),
                "orientation_axis_world": target.get("orientation_axis_world"),
                "orientation_confidence": target.get("orientation_confidence"),
                "orientation_source_position_m": target.get("orientation_source_position_m"),
                "orientation_policy": target.get("orientation_policy", "minimal_rotation"),
                "orientation_frame_world": target.get("orientation_frame_world"),
                "streaming_goal": True,
                "target_valid": True,
                "belief_selected": True,
            },
        )
        payload["source"] = "sam3_tasc_belief"
        payload["streaming_goal"] = True
        payload["target_valid"] = True
        payload["goal_uid"] = f"sam3_belief:{target_link}:{object_id or label}"
        # Shared-control alpha should answer "which object should we assist?",
        # not "how certain is the exact grip contact?".  Use p(object) for
        # alpha and keep p(object)*p(contact|object) as joint_confidence for
        # target/contact diagnostics.
        payload["confidence"] = object_confidence
        payload["object_confidence"] = object_confidence
        payload["joint_confidence"] = joint_confidence
        payload["object_id"] = object_id
        payload["grip_index"] = int(target.get("grip_index", -1))
        # Only an orientation posterior synthesized from a surface normal may
        # override the operator wrist orientation in the state machine.  The
        # generic quaternion in make_goal_payload is otherwise just a legacy
        # placeholder and must remain inactive.
        orientation_axis = target.get("orientation_axis_world")
        orientation_confidence = float(target.get("orientation_confidence", 0.0) or 0.0)
        axis_ok = False
        if orientation_axis is not None:
            try:
                axis = np.asarray(orientation_axis, dtype=np.float64).reshape(3)
                axis_ok = bool(np.isfinite(axis).all() and np.linalg.norm(axis) > 1.0e-6)
            except (TypeError, ValueError):
                axis_ok = False
        frame_ok = False
        try:
            frame = np.asarray(target.get("orientation_frame_world"), dtype=np.float64).reshape(3, 3)
            frame_ok = bool(np.isfinite(frame).all() and abs(float(np.linalg.det(frame))) > 0.5)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass
        payload["orientation_valid"] = bool(
            ((frame_ok and target.get("orientation_policy") == "grasp_frame")
             or (axis_ok and target.get("orientation_relation") in {"parallel", "perpendicular"}))
            and orientation_confidence >= float(self.target_config.get("orientation_min_confidence", 0.35))
        )
        payload["orientation_axis_world"] = None if not axis_ok else axis.tolist()
        payload["orientation_relation"] = target.get("orientation_relation")
        payload["orientation_confidence"] = orientation_confidence
        payload["orientation_policy"] = target.get("orientation_policy", "minimal_rotation")
        payload["orientation_frame_world"] = target.get("orientation_frame_world")
        payload["orientation_frame"] = "pelvis_identity_fk"
        payload["position_point"] = f"{self.tasc_hand_model}_grasp_center"
        if not has_orientation_quaternion:
            # The relationship normal is the only valid orientation control
            # representation at a perpendicular singularity.
            payload.pop("quaternion", None)
        return payload, position, label

    def _belief_payload_locked(self) -> tuple[dict[str, Any], np.ndarray | None, str]:
        """Publish both independent hand posteriors in one atomic message."""
        hand_payloads: dict[str, dict[str, Any]] = {}
        hand_positions: dict[str, np.ndarray | None] = {}
        hand_labels: dict[str, str] = {}
        for side in ("left", "right"):
            payload, position, label = self._belief_payload_for_hand_locked(side)
            hand_payloads[side] = payload
            hand_positions[side] = position
            hand_labels[side] = label

        primary_side = next(
            (side for side in ("left", "right") if hand_positions[side] is not None),
            "left",
        )
        payload = dict(hand_payloads[primary_side])
        payload["hand_targets"] = hand_payloads
        payload["target_valid"] = any(
            bool(item.get("target_valid", False)) for item in hand_payloads.values()
        )
        payload["source"] = "sam3_tasc_belief_dual_hand"
        payload["goal_uid"] = "sam3_belief:dual_hand"
        return payload, hand_positions[primary_side], hand_labels[primary_side]

    def _auto_publish_target_locked(self) -> tuple[GoalPublisher | None, dict[str, Any] | None, np.ndarray | None, str | None]:
        if self.publisher is None:
            return None, None, None, None
        now = time.monotonic()
        period = 1.0 / max(0.1, self.target_auto_publish_hz)
        if now - self.target_last_stream_publish_monotonic < period:
            return None, None, None, None
        blockers = self._target_blockers_locked()
        if blockers:
            payload = self._null_goal_payload_locked("; ".join(blockers))
            self.last_belief_published_label = None
            self.target_last_stream_publish_monotonic = now
            return self.publisher, payload, None, "null"
        payload, position, label = self._belief_payload_locked()
        self.target_last_stream_publish_monotonic = now
        self.last_belief_published_label = None if label == "null" else label
        return self.publisher, payload, position, label

    def send_object_target(
        self,
        object_index: int,
        frame_timestamp_us: int,
        label: str,
        center_goal_m: list[float] | None = None,
    ) -> dict[str, Any]:
        with self.target_lock:
            with self.lock:
                blockers = self._target_blockers_locked()
                if blockers:
                    raise ValueError("target blocked: " + "; ".join(blockers))
                if (
                    not self.target_bypass_checks
                    and int(self.status.get("last_frame_ts", -1)) != int(frame_timestamp_us)
                ):
                    raise ValueError("segmentation frame changed; reselect the object")
                selected = None
                if center_goal_m is not None:
                    requested_center = np.asarray(
                        center_goal_m, dtype=np.float64
                    ).reshape(-1)
                    if requested_center.shape == (3,) and np.isfinite(
                        requested_center
                    ).all():
                        selected, _ = select_tracking_candidate(
                            self.status.get("instances", []),
                            label,
                            requested_center,
                            self.target_tracking_max_jump_m,
                        )
                if selected is None:
                    selected = next(
                        (
                            obj
                            for obj in self.status.get("instances", [])
                            if int(obj["index"]) == int(object_index)
                            and str(obj.get("label")) == str(label)
                        ),
                        None,
                    )
                if selected is None:
                    # Inference normally updates faster than the browser status poll.
                    # Start from the highest-confidence current instance of the label.
                    selected, _ = select_tracking_candidate(
                        self.status.get("instances", []),
                        label,
                        None,
                        self.target_tracking_max_jump_m,
                    )
                if selected is None:
                    raise ValueError("selected object is no longer in the current frame")
                payload, position = self._goal_payload_for_object_locked(
                    selected, streaming_goal=True)
                publisher = self.publisher
                if publisher is None:
                    raise ValueError("publisher is disabled")
                self.target_tracking = {
                    "label": str(selected["label"]),
                    "last_index": int(selected["index"]),
                    "last_position": list(selected["center_goal_m"]),
                    "goal_uid": str(payload["goal_uid"]),
                }
                self.target_tracking_missing_frames = 0
                self.target_last_stream_publish_monotonic = time.monotonic()
            publisher.publish(
                payload,
                repeat=1,
                period_sec=1.0 / max(0.1, self.target_auto_publish_hz),
            )
            result = {
                "goal_uid": payload["goal_uid"],
                "target_link": self.target_link,
                "position_m": payload["position"],
                "label": selected["label"],
                "tracking": True,
            }
            with self.lock:
                self.status["ee_target_last_result"] = (
                    f"Continuously tracking {self.target_link} target "
                    f"{np.round(position, 4).tolist()} for {selected['label']}."
                )
            return result

    def wait_for_frame(self, previous_version: int, timeout: float = 1.0):
        with self.frame_ready:
            self.frame_ready.wait_for(
                lambda: self.output_version != previous_version or not self.running,
                timeout=timeout,
            )
            return self.output_version, self.output_jpeg

    def _record_error(self, error: Exception | str) -> None:
        with self.lock:
            self.status["decode_errors"] += 1
            self.status["last_error"] = str(error)

    def _submit_pair(self, pair: RgbdPair | None) -> None:
        if pair is None:
            return
        with self.pair_ready:
            if self.pending_pair is not None:
                self.status["dropped_for_inference"] += 1
            self.pending_pair = pair
            self.status["paired_frames"] += 1
            self.status["last_pair_delta_ms"] = pair.delta_us / 1000.0
            self.status["last_depth_source"] = pair.depth.source
            self.pair_ready.notify()

    def _on_rgb(self, _channel: str, data: bytes) -> None:
        try:
            frame = decode_camera(data)
            with self.lock:
                self.status["lcm_rgb_frames"] += 1
            self._submit_pair(self.sync.add_rgb(frame))
            self._publish_live_video(frame)
        except Exception as error:
            self._record_error(error)

    def _on_depth(self, _channel: str, data: bytes) -> None:
        try:
            frame = decode_depth(data, allow_unaligned=self.allow_unaligned)
            with self.lock:
                self.status["lcm_depth_frames"] += 1
            self._submit_pair(self.sync.add_depth(frame))
        except Exception as error:
            self._record_error(error)

    def _on_calibration(self, _channel: str, data: bytes) -> None:
        try:
            calibration = decode_calibration(data)
            with self.lock:
                self.calibration = calibration
                self.status["lcm_calibration_frames"] += 1
        except Exception as error:
            self._record_error(error)

    def _on_robot_state(self, _channel: str, data: bytes) -> None:
        try:
            state = decode_robot_state(data)
            with self.lock:
                self.state_buffer.add(state)
                self.latest_belief_robot_state = state
                self.status["lcm_robot_state_frames"] += 1
        except Exception as error:
            self._record_error(error)

    def _publish_live_video(self, frame) -> None:
        """Publish the newest RGB frame without waiting for segmentation.

        Segmentation metadata is intentionally cached at inference rate.  This
        path only redraws that newest annotation on the current camera image so
        the MJPEG stream remains tied to the camera's actual frame rate.
        """

        encoded = np.frombuffer(frame.jpeg, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("RGB JPEG decoding failed")
        with self.lock:
            visual = dict(self.latest_visual)
            masks = [np.asarray(mask).copy() for mask in visual["masks"]]
            boxes = [np.asarray(box, dtype=np.float64).copy() for box in visual["boxes"]]
            scores = list(visual["scores"])
            labels = list(visual["labels"])
            instances = [dict(item) for item in self.status.get("instances", [])]
            belief = dict(self.status.get("tasc_belief", {}))
            segment_timestamp_us = int(visual.get("frame_timestamp_us", 0))

        overlay = image.copy()
        height, width = overlay.shape[:2]
        for index, mask in enumerate(masks):
            if mask.shape != (height, width):
                continue
            color = np.asarray(
                [64, 64, 255] if index % 2 == 0 else [96, 220, 64],
                dtype=np.float32,
            )
            selected = np.asarray(mask, dtype=bool)
            overlay[selected] = (
                0.55 * overlay[selected].astype(np.float32) + 0.45 * color
            ).astype(np.uint8)
            if index >= len(boxes) or boxes[index].shape != (4,):
                continue
            x0, y0, x1, y1 = [int(round(value)) for value in boxes[index]]
            x0, x1 = np.clip([x0, x1], 0, width - 1)
            y0, y1 = np.clip([y0, y1], 0, height - 1)
            label = str(labels[index]) if index < len(labels) else "object"
            score = float(scores[index]) if index < len(scores) else 0.0
            cv2.rectangle(overlay, (x0, y0), (x1, y1), tuple(int(v) for v in color), 2)
            cv2.putText(
                overlay,
                f"{label} {score:.2f}",
                (x0, max(16, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                tuple(int(v) for v in color),
                2,
                cv2.LINE_AA,
            )
        draw_belief_overlay(
            overlay, instances, belief,
            t_goal_cam=visual.get("t_goal_cam"), intrinsics=visual.get("intrinsics"),
            draw_normals=self.belief_overlay_normals,
        )
        age_ms = (
            (int(frame.timestamp_us) - segment_timestamp_us) / 1000.0
            if segment_timestamp_us > 0
            else float("nan")
        )
        text = "live RGB | segmentation pending"
        if np.isfinite(age_ms):
            text = f"live RGB | segmentation age {age_ms:.0f}ms"
        cv2.putText(
            overlay, text, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2, cv2.LINE_AA,
        )
        ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("live RGB JPEG encoding failed")
        with self.frame_ready:
            self.output_jpeg = jpeg.tobytes()
            self.output_version += 1
            self.status["live_video_frames"] += 1
            self.frame_ready.notify_all()

    def _belief_hand_inputs(self, state) -> dict[str, HandInput]:
        hand_inputs: dict[str, HandInput] = {}
        if state is None:
            return hand_inputs
        for side in ("left", "right"):
            link = f"{side}_wrist_yaw_link"
            try:
                position = g1_grasp_center_position(
                    state.q,
                    link,
                    self.tasc_grasp_center_local_m,
                    self.tasc_hand_model,
                )
                rotation = g1_hand_base_rotation(
                    state.q, link, self.tasc_hand_model
                )
                previous = self.tasc_last_eef_by_link.get(link)
                velocity = None
                omega = None
                if previous is not None:
                    previous_position, previous_rotation, previous_omega, previous_ts = previous
                    dt = (state.timestamp_us - previous_ts) / 1.0e6
                    if 1.0e-4 <= dt <= 0.5:
                        velocity = (position - previous_position) / dt
                        raw_omega = angular_velocity_from_rotations(
                            previous_rotation, rotation, dt
                        )
                        if raw_omega is not None:
                            omega = (
                                raw_omega if previous_omega is None
                                else 0.25 * raw_omega + 0.75 * previous_omega
                            )
                self.tasc_last_eef_by_link[link] = (
                    position.copy(), rotation.copy(), omega, int(state.timestamp_us)
                )
                hand_inputs[side] = HandInput(
                    eef_position=position,
                    user_translation=velocity,
                    hand_rotation=rotation,
                    hand_axis=None,
                    approach_axis=rotation @ np.asarray(self.tasc_hand_profile["approach_axis_local"], dtype=np.float64),
                    angular_velocity=omega,
                    button_pressed=(self.shared_hand_selection.selected_hand == side),
                    shoulder_position=g1_shoulder_position(state.q, side),
                )
            except Exception as error:
                self._record_error(f"TASC {side} EE FK unavailable: {error}")
        return hand_inputs

    def _apply_belief_to_instances_locked(self, belief_telemetry: dict[str, Any]) -> None:
        hands = belief_telemetry.get("hands", {})
        by_hand_and_id = {
            side: {
                str(item.get("object_id")): item
                for item in hand_data.get("objects", [])
            }
            for side, hand_data in hands.items()
            if isinstance(hand_data, dict)
        }
        targets = {
            side: (hand_data.get("target") or {})
            for side, hand_data in hands.items()
            if isinstance(hand_data, dict)
        }
        for obj in self.status.get("instances", []):
            object_id = str(obj.get("tasc_object_id", ""))
            hand_views: dict[str, dict[str, Any]] = {}
            for side in ("left", "right"):
                belief_obj = by_hand_and_id.get(side, {}).get(object_id)
                if belief_obj is None:
                    continue
                target = targets.get(side, {})
                hand_views[side] = {
                    "object_probability": belief_obj.get("object_probability"),
                    "selected_grip_index": belief_obj.get("selected_grip_index"),
                    "target_contact_goal_m": belief_obj.get("selected_contact_goal_m"),
                    "joint_confidence": (
                        target.get("joint_confidence")
                        if target.get("object_id") == object_id
                        else None
                    ),
                }
            obj["tasc_hand_beliefs"] = hand_views
            if not hand_views:
                # This frame's segmentation is newer than the last geometry
                # frame. Do not present an unrelated stale target as its own.
                obj["tasc_object_probability"] = None
                obj["tasc_joint_confidence"] = None
                obj["target_contact_goal_m"] = None
                continue
            # Preserve the legacy scalar fields for existing web clients,
            # choosing the better of the two independent posteriors only for
            # display. A/B authority remains bridge-side.
            def _display_rank(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
                view = item[1]
                return (
                    float(view.get("joint_confidence") or 0.0),
                    float(view.get("object_probability") or 0.0),
                )

            ranked = sorted(hand_views.items(), key=_display_rank, reverse=True)
            display_side, display = ranked[0]
            if len(ranked) > 1 and _display_rank(ranked[0]) == _display_rank(ranked[1]):
                display_side = "tie"
            obj["tasc_display_hand"] = display_side
            obj["tasc_object_probability"] = display.get("object_probability")
            obj["tasc_selected_grip_index"] = display.get("selected_grip_index")
            obj["target_contact_goal_m"] = display.get("target_contact_goal_m")
            obj["tasc_joint_confidence"] = display.get("joint_confidence")

    def _belief_loop(self) -> None:
        self.pin_current_thread("tasc_belief", "G1_TASC_BELIEF_CPUSET")
        geometry_period = 1.0 / self.geometry_update_hz
        belief_period = 1.0 / self.belief_update_hz
        next_geometry = 0.0
        next_belief = 0.0
        while self.running:
            now = time.monotonic()
            if now < next_belief:
                time.sleep(min(0.005, next_belief - now))
                continue
            next_belief = max(next_belief + belief_period, now + belief_period)

            try:
                # This subscriber is intentionally consumed only here. pyzmq
                # sockets are not safe for concurrent recv() from inference
                # and belief threads.
                if self.shared_hand_selection.drain():
                    for tracker in self.parallel_tasc_belief.trackers.values():
                        tracker.feature_cost.previous_direction = None
                with self.lock:
                    observation = self.latest_belief_observation
                    robot_state = self.latest_belief_robot_state
                if observation is None:
                    continue
                if now >= next_geometry or self.latest_belief_geometry is None:
                    self.latest_belief_geometry = self.parallel_tasc_belief.build_shared_geometry(
                        observation["inputs"],
                        ray_origin=observation["ray_origin"],
                        ray_direction=observation["ray_direction"],
                        up_goal=observation.get("up_goal"),
                        stamp_us=observation["stamp_us"],
                    )
                    self.geometry_revision += 1
                    next_geometry = now + geometry_period
                hand_inputs = self._belief_hand_inputs(robot_state)
                if not hand_inputs:
                    continue
                per_hand = self.parallel_tasc_belief.update_hands(
                    self.latest_belief_geometry, hand_inputs, now_monotonic=now
                )
                selected_hand = self.shared_hand_selection.selected_hand
                selected_telemetry = per_hand.get(selected_hand)
                belief_telemetry = (
                    dict(selected_telemetry)
                    if selected_telemetry is not None
                    else {"enabled": True, "objects": [], "target": None}
                )
                belief_telemetry["hands"] = per_hand
                belief_telemetry["selected_hand"] = selected_hand
                belief_telemetry["selection_active"] = self.shared_hand_selection.active
                auto_publisher = auto_payload = auto_position = auto_label = None
                with self.target_lock, self.frame_ready:
                    self.status["tasc_belief"] = belief_telemetry
                    self.status["geometry_revision"] = self.geometry_revision
                    self.belief_revision += 1
                    self.status["belief_revision"] = self.belief_revision
                    self.status["belief_last_update_monotonic"] = now
                    self.status["belief_last_error"] = None
                    self._apply_belief_to_instances_locked(belief_telemetry)
                    auto_publisher, auto_payload, auto_position, auto_label = (
                        self._auto_publish_target_locked()
                    )
                if auto_publisher is not None and auto_payload is not None:
                    auto_publisher.publish(
                        auto_payload,
                        repeat=1,
                        period_sec=1.0 / max(0.1, self.target_auto_publish_hz),
                    )
                    with self.lock:
                        if auto_position is None or auto_payload.get("target_valid") is False:
                            self.status["ee_target_last_result"] = "Streaming null belief target."
                        else:
                            self.status["ee_target_last_result"] = (
                                f"Streaming belief target {np.round(auto_position, 4).tolist()} "
                                f"for {auto_label}."
                            )
            except Exception as error:
                # A malformed point cloud or a transient solver failure must
                # not kill the continuous two-hand worker.
                with self.lock:
                    self.status["belief_last_error"] = str(error)
                next_geometry = 0.0
                time.sleep(0.01)

    def _lcm_loop(self) -> None:
        self.pin_current_thread("lcm_rgbd", "G1_PERCEPTION_IO_CPUSET")
        while self.running:
            try:
                self.bus.handle_timeout(100)
            except Exception as error:
                self._record_error(error)
                time.sleep(0.1)

    def _load_model(self):
        if self.inference_backend == "yolo11-seg":
            options = self.yolo11_config
            return Yolo11SegBackend(
                checkpoint=Path(options.get("checkpoint", "/models/yolo11/yolo11n-seg.pt")),
                threshold=self.threshold,
                min_area_px=self.min_area_px,
                imgsz=int(options.get("imgsz", 640)),
                confidence=float(options.get("confidence", 0.80)),
                iou_threshold=float(options.get("iou_threshold", 0.50)),
                max_masks=int(options.get("max_masks", 20)),
                quantize=options.get("quantize", 16),
                track=dict(options.get("track", {})),
                allowed_classes=list(options.get("allowed_classes", [])),
                excluded_classes=list(options.get("excluded_classes", ["person"])),
                depth_self_filter=dict(
                    options.get("depth_self_filter", self.depth_self_filter_config)
                ),
            )
        if self.inference_backend == "sam2-tiny":
            options = self.sam2_config
            return Sam2HieraTinyBackend(
                checkpoint=Path(options.get("checkpoint", "/models/sam2/sam2.1_hiera_tiny.pt")),
                threshold=self.threshold,
                min_area_px=self.min_area_px,
                reinitialization=str(options.get("reinitialization", "central")),
                reinitialize_interval_frames=int(options.get("reinitialize_interval_frames", 15)),
                automatic_points_per_side=int(options.get("automatic_points_per_side", 8)),
                max_masks=int(options.get("max_masks", 12)),
                central=dict(options.get("central", {})),
                depth_self_filter=dict(
                    options.get("depth_self_filter", self.depth_self_filter_config)
                ),
            )
        if self.inference_backend == "efficienttam-ti-512":
            options = self.efficienttam_config
            return EfficientTamTi512Backend(
                checkpoint=Path(options.get("checkpoint", "/models/efficienttam/efficienttam_ti_512x512.pt")),
                threshold=self.threshold,
                min_area_px=self.min_area_px,
                reinitialization=str(options.get("reinitialization", "central")),
                reinitialize_interval_frames=int(options.get("reinitialize_interval_frames", 15)),
                automatic_points_per_side=int(options.get("automatic_points_per_side", 8)),
                max_masks=int(options.get("max_masks", 12)),
                central=dict(options.get("central", {})),
                depth_self_filter=dict(
                    options.get("depth_self_filter", self.depth_self_filter_config)
                ),
            )
        if self.inference_backend == "tensorrt":
            return TensorRTSam3Backend(
                engine_path=Path(self.tensorrt_config["engine"]),
                model_id=str(self.tensorrt_config.get("model_id", "facebook/sam3")),
                threshold=self.threshold,
                min_area_px=self.min_area_px,
                mask_threshold=float(self.tensorrt_config.get("mask_threshold", 0.5)),
                depth_self_filter=dict(self.depth_self_filter_config),
            )
        if self.inference_backend == "pytorch":
            return TorchSam3Backend(
                checkpoint=self.checkpoint,
                threshold=self.threshold,
                min_area_px=self.min_area_px,
                depth_self_filter=dict(self.depth_self_filter_config),
            )
        raise ValueError(f"unknown SAM3 inference backend: {self.inference_backend}")

    def _inference_loop(self) -> None:
        self.pin_current_thread("sam3_inference", "G1_SAM3_INFERENCE_CPUSET")
        try:
            backend = self._load_model()
            with self.lock:
                self.status["model_ready"] = True
                self.status["backend_metadata"] = backend.metadata
                self.status["last_error"] = None
        except Exception as error:
            self._record_error(f"SAM3 load failed: {error}")
            return

        minimum_period = 1.0 / self.target_hz
        last_started = 0.0
        while self.running:
            with self.pair_ready:
                self.pair_ready.wait_for(
                    lambda: self.pending_pair is not None or not self.running,
                    timeout=0.5,
                )
                if not self.running:
                    break
                pair = self.pending_pair
                self.pending_pair = None
                concepts = list(self.concepts)
                calibration = self.calibration
                state_match = self.state_buffer.nearest(
                    pair.rgb.timestamp_us, self.robot_state_max_delta_ms
                ) if pair is not None else None
            if pair is None:
                continue
            delay = minimum_period - (time.monotonic() - last_started)
            if delay > 0:
                time.sleep(delay)
            last_started = time.monotonic()
            try:
                frame_started = time.perf_counter()
                encoded = np.frombuffer(pair.rgb.jpeg, dtype=np.uint8)
                image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError("JPEG decoding failed")
                if pair.depth.source == DEPTH_SOURCE_RAW_DEPTH_OPTICAL:
                    if calibration is None:
                        raise ValueError("raw depth received before G1_CAMERA_CALIBRATION")
                    if (pair.rgb.width, pair.rgb.height) != (
                        calibration.color_width, calibration.color_height
                    ):
                        raise ValueError("RGB dimensions do not match transmitted color calibration")
                    depth = align_depth_to_color(pair.depth, calibration)
                    frame_intrinsics = CameraIntrinsics(
                        calibration.color_width, calibration.color_height,
                        *calibration.color_intrinsics,
                    )
                else:
                    depth = depth_at_rgb_size(pair)
                    frame_intrinsics = self.intrinsics
                    if calibration is not None:
                        frame_intrinsics = CameraIntrinsics(
                            calibration.color_width, calibration.color_height,
                            *calibration.color_intrinsics,
                        )
                geometry_runtime_error = None
                t_goal_cam = self.static_t_goal_cam
                transform_metadata = None
                state_for_belief = state_match[0] if state_match else None
                if self.t_torso_cam is not None:
                    if state_match is None:
                        geometry_runtime_error = "3-D goal unavailable: no timestamp-matched G1 robot state"
                        t_goal_cam = None
                    else:
                        state, state_delta_ms = state_match
                        try:
                            disagreement_limit_deg = (
                                float("inf")
                                if self.target_bypass_checks
                                else self.max_imu_fk_disagreement_deg
                            )
                            t_pelvis_torso, transform_metadata = pelvis_to_torso(
                                state, self.torso_estimator, disagreement_limit_deg
                            )
                            if self.target_bypass_checks and self.torso_estimator == "dual_imu":
                                transform_metadata = dict(transform_metadata)
                                transform_metadata["disagreement_check_bypassed"] = True
                                transform_metadata["configured_disagreement_limit_deg"] = (
                                    self.max_imu_fk_disagreement_deg
                                )
                            t_goal_cam = t_pelvis_torso @ self.t_torso_cam
                        except Exception as error:
                            geometry_runtime_error = f"3-D goal unavailable: {error}"
                            if self.target_bypass_checks:
                                try:
                                    t_pelvis_torso, transform_metadata = pelvis_to_torso(
                                        state, "joint_fk", self.max_imu_fk_disagreement_deg)
                                    transform_metadata = dict(transform_metadata)
                                    transform_metadata["fallback_from"] = self.torso_estimator
                                    transform_metadata["fallback_reason"] = str(error)
                                    t_goal_cam = t_pelvis_torso @ self.t_torso_cam
                                    geometry_runtime_error = (
                                        "3-D goal using joint_fk fallback: "
                                        f"{error}"
                                    )
                                except Exception:
                                    t_goal_cam = None
                            else:
                                t_goal_cam = None
                preprocess_finished = time.perf_counter()
                model_started = preprocess_finished
                if hasattr(backend, "set_depth_filter_context"):
                    backend.set_depth_filter_context(
                        image_bgr=image_bgr,
                        depth_raw=depth,
                        depth_scale_um=pair.depth.depth_scale_um,
                        intrinsics=frame_intrinsics,
                        t_goal_cam=t_goal_cam,
                        q=None if state_for_belief is None else state_for_belief.q,
                    )
                masks, boxes, scores, labels = backend.infer(
                    image_bgr,
                    concepts,
                    depth_raw=depth,
                    depth_scale_um=pair.depth.depth_scale_um,
                )
                model_finished = time.perf_counter()
                if len(masks):
                    masks, boxes, scores, labels = suppress_duplicate_masks(
                        masks, boxes, scores, labels
                    )
                overlay, objects = render_segmented_rgbd(
                    image_bgr, depth, masks, boxes, scores, labels,
                    pair.depth.depth_scale_um, self.min_depth_m, self.max_depth_m,
                )
                belief_ray_origin = None
                belief_ray_direction = None
                if t_goal_cam is not None:
                    belief_ray_origin = np.asarray(t_goal_cam[:3, 3], dtype=np.float64)
                    ray = np.asarray(t_goal_cam[:3, :3], dtype=np.float64) @ np.array(
                        [0.0, 0.0, 1.0], dtype=np.float64
                    )
                    ray_norm = float(np.linalg.norm(ray))
                    if ray_norm > 1.0e-9 and np.isfinite(ray_norm):
                        belief_ray_direction = ray / ray_norm
                if frame_intrinsics is not None:
                    tasc_inputs: list[BeliefInputObject] = []
                    pending_tasc_inputs: list[dict[str, Any]] = []
                    try:
                        depth_m = depth.astype(np.float32) * (
                            pair.depth.depth_scale_um / 1_000_000.0
                        )
                        for index, mask in enumerate(masks):
                            label_lower = str(labels[index]).lower()
                            cloud = extract_filtered_object_cloud(
                                depth_m,
                                mask,
                                frame_intrinsics,
                                self.object_point_filter,
                                min_depth_m=self.min_depth_m,
                                max_depth_m=self.max_depth_m,
                                min_valid_points=self.target_min_depth_points,
                                max_points=self.tasc_pointcloud_max_points,
                                erosion_px_override=0 if ("hole" in label_lower or "circle" in label_lower) else None,
                            )
                            objects[index]["center_camera_m"] = None
                            objects[index]["center_goal_m"] = None
                            objects[index]["center_pixel_uv"] = None
                            objects[index]["valid_depth_points"] = 0
                            objects[index]["point_filter"] = None
                            if cloud is None:
                                continue
                            point_camera = cloud.center_camera_m
                            pixel_uv = cloud.center_pixel_uv
                            objects[index]["center_camera_m"] = point_camera.tolist()
                            objects[index]["center_pixel_uv"] = list(pixel_uv)
                            objects[index]["valid_depth_points"] = cloud.valid_depth_points
                            objects[index]["median_depth_m"] = cloud.median_depth_m
                            objects[index]["point_filter"] = cloud.stats
                            if t_goal_cam is not None:
                                point_goal = transform_points(point_camera, t_goal_cam)
                                objects[index]["center_goal_m"] = point_goal.tolist()
                                points_goal = transform_points(
                                    cloud.points_camera_m,
                                    t_goal_cam,
                                )
                                pending_tasc_inputs.append(
                                    {
                                        "index": int(index),
                                        "label": str(labels[index]),
                                        "score": float(scores[index]),
                                        "center_goal_m": point_goal,
                                        "points_goal_m": points_goal,
                                    }
                                )
                            color = tuple(
                                int(value)
                                for value in np.asarray(
                                    [64, 64, 255] if index % 2 == 0 else [96, 220, 64]
                                )
                            )
                            cv2.circle(overlay, pixel_uv, 3, color, -1)
                    except Exception as error:
                        geometry_runtime_error = f"3-D center unavailable: {error}"
                    assignments = self.tasc_object_tracker.assign(pending_tasc_inputs)
                    for item in pending_tasc_inputs:
                        object_id = assignments.get(int(item["index"]))
                        if object_id is None:
                            continue
                        objects[int(item["index"])]["tasc_object_id"] = object_id
                        tasc_inputs.append(
                            BeliefInputObject(
                                object_id=object_id,
                                label=str(item["label"]),
                                score=float(item["score"]),
                                center_goal_m=item["center_goal_m"],
                                points_goal_m=item["points_goal_m"],
                            )
                        )
                else:
                    tasc_inputs = []
                    for obj in objects:
                        obj.update(
                            center_camera_m=None,
                            center_goal_m=None,
                            center_pixel_uv=None,
                            valid_depth_points=0,
                        )
                postprocess_finished = time.perf_counter()
                latency_ms = (postprocess_finished - model_started) * 1000.0
                frame_finished = time.perf_counter()
                timing_ms = {
                    "jpeg_decode_depth_align": round(
                        (preprocess_finished - frame_started) * 1000.0, 2
                    ),
                    "yolo_model": round((model_finished - model_started) * 1000.0, 2),
                    "rgbd_belief_overlay": round(
                        (postprocess_finished - model_finished) * 1000.0, 2
                    ),
                    "web_jpeg_encode": round(
                        (frame_finished - postprocess_finished) * 1000.0, 2
                    ),
                    "end_to_end": round((frame_finished - frame_started) * 1000.0, 2),
                }
                with self.frame_ready:
                    self.status["processed_frames"] += 1
                    self.status["backend_metadata"] = backend.metadata
                    self.status["last_frame_ts"] = pair.rgb.timestamp_us
                    self.status["last_inference_ms"] = round(latency_ms, 2)
                    self.status["timing_ms"] = timing_ms
                    self.status["instances"] = objects
                    self.status["depth_self_filter"] = (
                        backend.metadata.get("depth_self_filter")
                        if isinstance(backend.metadata, dict)
                        else None
                    )
                    self.status["geometry_runtime_error"] = geometry_runtime_error
                    self.status["alignment_ready"] = pair.depth.source == DEPTH_SOURCE_ALIGNED_TO_COLOR or (
                        pair.depth.source == DEPTH_SOURCE_RAW_DEPTH_OPTICAL and calibration is not None
                    )
                    self.status["robot_state_delta_ms"] = state_match[1] if state_match else None
                    self.status["torso_transform"] = transform_metadata
                    self.status["last_error"] = None
                    self.last_processed_monotonic = time.monotonic()
                    belief_up_goal = None
                    if state_for_belief is not None:
                        belief_up_goal = gravity_up_in_pelvis(
                            getattr(state_for_belief, "base_quat_wxyz", None)
                        )
                    if belief_up_goal is None and not self.tasc_gravity_warned:
                        LOG.warning("gravity up unavailable; normal snapping disabled")
                        self.tasc_gravity_warned = True
                    geometry = self.latest_belief_geometry
                    gravity_snap_count = sum(
                        int(bool(primitive.snapped_to_gravity))
                        for primitives in (geometry.surface_primitives.values() if geometry is not None else [])
                        for primitive in primitives
                    )
                    self.status["gravity_up_goal"] = (
                        None if belief_up_goal is None else belief_up_goal.tolist()
                    )
                    self.status["gravity_snap_count"] = int(gravity_snap_count)
                    self.latest_belief_observation = {
                        "inputs": tasc_inputs,
                        "ray_origin": belief_ray_origin,
                        "ray_direction": belief_ray_direction,
                        "stamp_us": int(pair.rgb.timestamp_us),
                        "up_goal": belief_up_goal,
                    }
                    self.latest_visual = {
                        "masks": [np.asarray(mask, dtype=bool).copy() for mask in masks],
                        "boxes": [np.asarray(box, dtype=np.float64).copy() for box in boxes],
                        "scores": [float(score) for score in scores],
                        "labels": [str(label) for label in labels],
                        "frame_timestamp_us": int(pair.rgb.timestamp_us),
                        "t_goal_cam": None if t_goal_cam is None else np.asarray(t_goal_cam).copy(),
                        "intrinsics": frame_intrinsics,
                    }
            except Exception as error:
                self._record_error(f"inference failed: {error}")


app = FastAPI(title="G1 SAM3 RGB-D View")
runtime: SegmentationRuntime | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/status")
def api_status():
    if runtime is None:
        raise HTTPException(503, "runtime is not initialized")
    return runtime.status_snapshot()


@app.post("/api/prompt")
def api_prompt(request: PromptRequest):
    if runtime is None:
        raise HTTPException(503, "runtime is not initialized")
    try:
        runtime.set_prompt(request.prompt)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {"concepts": runtime.status_snapshot()["concepts"]}


@app.post("/api/ee-target/send")
def api_ee_target_send(request: SendTargetRequest):
    if runtime is None:
        raise HTTPException(503, "runtime is not initialized")
    try:
        return runtime.send_object_target(
            request.object_index,
            request.frame_timestamp_us,
            request.label,
            request.center_goal_m,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/ee-target/start")
def api_ee_target_start():
    if runtime is None:
        raise HTTPException(503, "runtime is not initialized")
    try:
        return runtime.start_belief_publishing()
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/ee-target/stop")
def api_ee_target_stop():
    if runtime is None:
        raise HTTPException(503, "runtime is not initialized")
    return runtime.stop_target_tracking()


@app.get("/snapshot.jpg")
def snapshot():
    if runtime is None:
        raise HTTPException(503, "runtime is not initialized")
    return Response(runtime.snapshot(), media_type="image/jpeg")


async def mjpeg_frames():
    version = -1
    while runtime is not None and runtime.running:
        version, jpeg = await asyncio.to_thread(runtime.wait_for_frame, version, 1.0)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )


@app.get("/stream.mjpg")
def stream():
    return StreamingResponse(
        mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--lcm-url", default=None)
    parser.add_argument("--rgb-channel", default="G1_CAMERA_FRAME")
    parser.add_argument("--depth-channel", default="G1_DEPTH_FRAME")
    parser.add_argument("--calibration-channel", default="G1_CAMERA_CALIBRATION")
    parser.add_argument("--robot-state-channel", default="G1_ROBOT_STATE")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--concepts", default=None, help="Comma-separated text concepts")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-hz", type=float, default=None)
    parser.add_argument("--min-area-px", type=int, default=None)
    parser.add_argument("--allow-unaligned", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--backend", choices=("pytorch", "tensorrt", "sam2-tiny", "efficienttam-ti-512", "yolo11-seg"), default=None)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument(
        "--enable-ee-publishing",
        action="store_true",
        help="Enable continuous PATH EE publishing for a web-selected object.",
    )
    parser.add_argument(
        "--ee-dry-run",
        action="store_true",
        help="Exercise target tracking/payload validation without sending ZeroMQ messages.",
    )
    parser.add_argument(
        "--shared-hand-selection-endpoint",
        default=os.environ.get(
            "PATH_SHARED_HAND_SELECTION_ENDPOINT", "tcp://127.0.0.1:5567"
        ),
        help="State-machine PUB endpoint carrying the held A/B shared-control hand.",
    )
    parser.add_argument(
        "--shared-hand-selection-topic",
        default=os.environ.get(
            "PATH_SHARED_HAND_SELECTION_TOPIC", "path_shared_control_hand"
        ),
        help="State-machine A/B shared-control hand heartbeat topic.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "PATH_SAM3_WEB_BIND_HOST",
            os.environ.get("PATH_WEB_BIND_HOST", "0.0.0.0"),
        ),
    )
    parser.add_argument(
        "--advertise-host",
        default=os.environ.get(
            "PATH_SAM3_WEB_ADVERTISE_HOST",
            os.environ.get("PATH_WEB_ADVERTISE_HOST"),
        ),
        help=(
            "Host/IP printed for opening this page from another desktop. "
            "The server still binds to --host."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=(
            int(os.environ["PATH_SAM3_WEB_PORT"])
            if os.environ.get("PATH_SAM3_WEB_PORT")
            else None
        ),
    )
    return parser.parse_args()


def main() -> int:
    global runtime
    args = parse_args()
    config = yaml.safe_load(args.config.read_text()) if args.config.is_file() else {}
    web_port = int(args.port or config.get("web_port", 8080))
    try:
        ensure_port_available(args.host, web_port)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    depth_config = config.get("depth", {})
    segmentation_config = config.get("segmentation", {})
    tensorrt_config = dict(segmentation_config.get("tensorrt", {}))
    sam2_config = dict(segmentation_config.get("sam2", {}))
    efficienttam_config = dict(segmentation_config.get("efficienttam", {}))
    yolo11_config = dict(segmentation_config.get("yolo11", {}))
    depth_self_filter_config = dict(segmentation_config.get("depth_self_filter", {}))
    object_point_filter_config = dict(config.get("cloud", {}).get("object_filter", {}))
    target_config = config.get("ee_target", {})
    belief_config = config.get("tasc_belief", {})
    concepts = segmentation_config.get("concepts") or []
    checkpoint = args.checkpoint
    if checkpoint is None and segmentation_config.get("checkpoint"):
        checkpoint = Path(segmentation_config["checkpoint"])
    if checkpoint is not None and not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    inference_backend = args.backend or segmentation_config.get("backend", "pytorch")
    if args.engine is not None:
        tensorrt_config["engine"] = str(args.engine)
    if inference_backend == "tensorrt" and not tensorrt_config.get("engine"):
        raise SystemExit("TensorRT backend requires segmentation.tensorrt.engine or --engine")
    camera_info_path = Path(target_config.get("camera_info", "config/camera_info.yaml"))
    extrinsics_path = Path(target_config.get("extrinsics", "config/extrinsics.yaml"))
    intrinsics, t_torso_cam, static_t_goal_cam, geometry_blockers = load_target_geometry(
        camera_info_path, extrinsics_path
    )
    workspace_config = target_config.get("workspace", {})
    workspace = Workspace(
        x=tuple(workspace_config.get("x", [-0.15, 0.45])),
        left_y=tuple(workspace_config.get("left_y", [0.03, 0.50])),
        right_y=tuple(workspace_config.get("right_y", [-0.50, -0.03])),
        z=tuple(workspace_config.get("z", [-0.30, 0.35])),
        radius_m=float(workspace_config.get("radius_m", 0.62)),
    )
    publisher = None
    if args.enable_ee_publishing:
        publisher = GoalPublisher(
            goal_endpoint=str(
                target_config.get("goal_endpoint", "tcp://127.0.0.1:5564")
            ),
            goal_topic=str(target_config.get("goal_topic", "path_static_ee_target")),
            planner_endpoint=(
                str(target_config.get("planner_endpoint", "tcp://127.0.0.1:5561"))
                if target_config.get("send_planner_goal", False)
                else None
            ),
            planner_topic=str(target_config.get("planner_topic", "path_goal")),
            dry_run=args.ee_dry_run,
        )
    runtime = SegmentationRuntime(
        lcm_url=args.lcm_url or config.get("lcm_url", "udpm://239.255.76.67:7667?ttl=1"),
        rgb_channel=args.rgb_channel,
        depth_channel=args.depth_channel,
        calibration_channel=args.calibration_channel,
        robot_state_channel=args.robot_state_channel,
        prompt=args.concepts or args.prompt or ",".join(concepts) or "object",
        threshold=args.threshold if args.threshold is not None else segmentation_config.get("min_score", 0.5),
        min_area_px=args.min_area_px if args.min_area_px is not None else segmentation_config.get("min_area_px", 400),
        target_hz=args.target_hz if args.target_hz is not None else segmentation_config.get("target_hz", 3.5),
        sync_tolerance_ms=depth_config.get("sync_tol_ms", 50),
        min_depth_m=depth_config.get("min_depth_m", 0.15),
        max_depth_m=depth_config.get("max_depth_m", 3.0),
        allow_unaligned=args.allow_unaligned,
        checkpoint=checkpoint,
        inference_backend=inference_backend,
        tensorrt_config=tensorrt_config,
        sam2_config=sam2_config,
        efficienttam_config=efficienttam_config,
        yolo11_config=yolo11_config,
        depth_self_filter_config=depth_self_filter_config,
        object_point_filter_config=object_point_filter_config,
        intrinsics=intrinsics,
        t_torso_cam=t_torso_cam,
        static_t_goal_cam=static_t_goal_cam,
        geometry_blockers=geometry_blockers,
        publisher=publisher,
        workspace=workspace,
        target_config=target_config,
        belief_config=belief_config,
        shared_hand_selection_endpoint=args.shared_hand_selection_endpoint,
        shared_hand_selection_topic=args.shared_hand_selection_topic,
    )
    runtime.start()
    runtime.pin_current_thread("web_server", "G1_PERCEPTION_IO_CPUSET")
    advertised = display_host(args.host, args.advertise_host)
    print(f"[SAM3-web] serving on {args.host}:{web_port}", flush=True)
    print(f"[SAM3-web] open local  http://127.0.0.1:{web_port}", flush=True)
    print(f"[SAM3-web] open remote http://{advertised}:{web_port}", flush=True)
    print(f"[SAM3-web] status      http://{advertised}:{web_port}/api/status", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=web_port, log_level="info")
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
