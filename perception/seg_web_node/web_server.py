"""Live G1 RGB-D + SAM3 segmentation web server for Jetson Thor."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import lcm
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from .ee_target import GoalPublisher, Workspace, make_goal_payload
from .depth_alignment import align_depth_to_color
from .geometry import CameraIntrinsics, deproject, robust_mask_center, transform_points
from .lcm_protocol import (
    DEPTH_SOURCE_ALIGNED_TO_COLOR,
    DEPTH_SOURCE_RAW_DEPTH_OPTICAL,
    decode_calibration,
    decode_camera,
    decode_depth,
    decode_robot_state,
)
from .robot_geometry import RobotStateBuffer, g1_wrist_ee_position, pelvis_to_torso
from .rgbd_sync import RgbdPair, RgbdSynchronizer, depth_at_rgb_size
from .sam3_backend import TensorRTSam3Backend, TorchSam3Backend
from .tasc_belief import (
    BeliefInputObject,
    HierarchicalTascBelief,
    StableObjectIdTracker,
)
from .visualization import render_segmented_rgbd


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
#view{width:100%;max-height:calc(100vh - 110px);object-fit:contain;background:#000;border:1px solid #30363d}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:12px}.ok{color:#3fb950}.bad{color:#f85149}pre{white-space:pre-wrap;line-height:1.5;color:#8b949e}.row{display:flex;gap:8px}.row button{flex:1}#targetmsg{min-height:38px;color:#8b949e;margin-top:10px}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>G1 · SAM3 RGB-D</h1><div class="grow"></div>
<input id="prompt" placeholder="person, box, tool, robot part"><button onclick="setPrompt()">Apply concepts</button></header>
<main><div><img id="view" src="/stream.mjpg"></div><aside>
<div class="card"><b>Belief target publisher</b>
<div class="row"><button id="send" onclick="startBeliefPublishing()">Publish belief result</button><button id="stop" class="danger" onclick="stopBeliefPublishing()" disabled>Stop publishing</button></div>
<div id="targetmsg">Publishing is stopped. When enabled, the highest-belief contact target is streamed automatically; missing target streams null.</div></div>
<div class="card"><b>Pipeline status</b><pre id="status">connecting…</pre></div></aside></main>
<script>
let state={};
async function api(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw new Error(data.detail||JSON.stringify(data));return data}
function renderTarget(s){state=s;const t=s.ee_target||{};const enabled=!!t.belief_publishing;document.getElementById('send').disabled=enabled;document.getElementById('stop').disabled=!enabled;const blockers=(t.blockers||[]).join('; ');const active=enabled?`Publishing belief result at ${Number(t.publish_hz||0).toFixed(1)} Hz${t.last_published_label?` — ${t.last_published_label}`:''}`:'';document.getElementById('targetmsg').textContent=active||t.last_result||blockers||'Publishing is stopped. Enable it to stream belief target/null fallback.';}
async function update(){try{const s=await (await fetch('/api/status')).json();document.getElementById('status').textContent=JSON.stringify(s,null,2);renderTarget(s);if(document.activeElement!==document.getElementById('prompt'))document.getElementById('prompt').value=(s.concepts||[]).join(', ');}catch(e){document.getElementById('status').textContent=String(e)}}
async function setPrompt(){const prompt=document.getElementById('prompt').value.trim();await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})});update()}
async function startBeliefPublishing(){try{await api('/api/ee-target/start',{});await update()}catch(e){document.getElementById('targetmsg').textContent=e.message}}
async function stopBeliefPublishing(){try{await api('/api/ee-target/stop',{});await update()}catch(e){document.getElementById('targetmsg').textContent=e.message}}
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


def masked_goal_points(
    xyz_camera: np.ndarray,
    mask: np.ndarray,
    t_goal_cam: np.ndarray,
    *,
    max_points: int,
) -> np.ndarray:
    valid = mask.astype(bool) & np.isfinite(xyz_camera).all(axis=2)
    points = xyz_camera[valid]
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max(1, int(max_points))))
        points = points[::step][:max_points]
    return transform_points(points, t_goal_cam)


def draw_belief_highlight(
    overlay: np.ndarray,
    objects: list[dict[str, Any]],
    belief_telemetry: dict[str, Any],
) -> None:
    target = belief_telemetry.get("target") or {}
    object_id = target.get("object_id")
    if not object_id:
        return
    selected = next(
        (obj for obj in objects if obj.get("tasc_object_id") == object_id),
        None,
    )
    if selected is None:
        return
    height, width = overlay.shape[:2]
    color = (40, 255, 40)
    box = selected.get("box_xyxy")
    if isinstance(box, list) and len(box) == 4:
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
        x0, x1 = np.clip([x0, x1], 0, width - 1)
        y0, y1 = np.clip([y0, y1], 0, height - 1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
        confidence = target.get("joint_confidence")
        grip = target.get("grip_index")
        label = "belief"
        if isinstance(confidence, (int, float)):
            label += f" {float(confidence):.2f}"
        if grip is not None:
            label += f" grip {int(grip)}"
        cv2.putText(
            overlay,
            label,
            (x0, min(height - 6, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    pixel = selected.get("center_pixel_uv")
    if isinstance(pixel, list) and len(pixel) == 2:
        cv2.drawMarker(
            overlay,
            (int(pixel[0]), int(pixel[1])),
            color,
            cv2.MARKER_CROSS,
            24,
            3,
        )


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
        intrinsics: CameraIntrinsics | None,
        t_torso_cam: np.ndarray | None,
        static_t_goal_cam: np.ndarray | None,
        geometry_blockers: list[str],
        publisher: GoalPublisher | None,
        workspace: Workspace,
        target_config: dict[str, Any],
        belief_config: dict[str, Any],
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
        self.tasc_belief = HierarchicalTascBelief(belief_config)
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
        self.tasc_ee_offset_m = float(belief_config.get("ee_offset_m", 0.05))
        self.tasc_last_eef_position: np.ndarray | None = None
        self.tasc_last_eef_timestamp_us: int | None = None
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
        self.belief_publishing_enabled = bool(
            target_config.get("belief_publishing_enabled", False)
        )
        self.last_belief_published_label: str | None = None
        self.target_lock = threading.Lock()
        self.target_tracking: dict[str, Any] | None = None
        self.target_tracking_missing_frames = 0
        self.last_processed_monotonic = 0.0
        self.sync = RgbdSynchronizer(sync_tolerance_ms, capacity=30)
        self.bus = lcm.LCM(lcm_url)
        self.lock = threading.Lock()
        self.pair_ready = threading.Condition(self.lock)
        self.frame_ready = threading.Condition(self.lock)
        self.running = False
        self.pending_pair: RgbdPair | None = None
        self.output_jpeg = placeholder_jpeg("Waiting for G1 RGB-D frames...")
        self.output_version = 0
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
            "thread_affinity": {},
            "tasc_belief": self.tasc_belief.last_telemetry,
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
                    "goal_frame": "pelvis_identity_fk",
                    "target_link": self.target_link,
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
        publisher = None
        payload = None
        with self.target_lock, self.lock:
            self.belief_publishing_enabled = False
            self.target_tracking = None
            self.target_tracking_missing_frames = 0
            self.last_belief_published_label = None
            self.status["ee_target_last_result"] = reason
            publisher = self.publisher
            if publisher is not None:
                payload = self._null_goal_payload_locked(reason)
        if publisher is not None and payload is not None:
            publisher.publish(
                payload,
                repeat=1,
                period_sec=1.0 / max(0.1, self.target_auto_publish_hz),
            )
        return {"belief_publishing": False, "tracking": False, "sent_null": payload is not None}

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

    def _null_goal_payload_locked(self, reason: str) -> dict[str, Any]:
        return {
            "goal_uid": f"sam3_belief:null:{self.target_link}",
            "force_replan": False,
            "streaming_goal": True,
            "target_valid": False,
            "target_link": self.target_link,
            "side": self.target_side,
            "position": None,
            "quaternion": list(self.target_quaternion),
            "source": "sam3_tasc_belief_null",
            "coordinate_frame": "pelvis_identity_fk",
            "duration_sec": float(self.target_config.get("duration_sec", 2.0)),
            "final_hold_sec": float(self.target_config.get("final_hold_sec", 1.0)),
            "perception": {
                "frame_timestamp_us": int(self.status.get("last_frame_ts", 0) or 0),
                "reason": str(reason),
                "streaming_goal": True,
                "target_valid": False,
            },
        }

    def _belief_payload_locked(self) -> tuple[dict[str, Any], np.ndarray | None, str]:
        belief = self.status.get("tasc_belief") or {}
        target = belief.get("target") if isinstance(belief, dict) else None
        if not isinstance(target, dict):
            return self._null_goal_payload_locked("belief target unavailable"), None, "null"
        try:
            position = np.asarray(target.get("position_goal_m"), dtype=np.float64).reshape(3)
        except Exception:
            return self._null_goal_payload_locked("belief target position unavailable"), None, "null"
        position = position + self.target_offset
        if not np.isfinite(position).all():
            return self._null_goal_payload_locked("belief target position is non-finite"), None, "null"

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
        if self.target_bypass_checks:
            workspace_info = {
                "workspace_mode": "bypass",
                "workspace_reason": "validation_disabled_by_config",
                "workspace_clamped": False,
                "requested_position": position.tolist(),
            }
        else:
            try:
                workspace_info = self.workspace.validate(position, self.target_side)
            except ValueError as error:
                return self._null_goal_payload_locked(str(error)), None, "null"
        payload = make_goal_payload(
            position,
            side=self.target_side,
            target_link=self.target_link,
            quaternion=self.target_quaternion,
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
                "streaming_goal": True,
                "target_valid": True,
                "belief_selected": True,
            },
        )
        payload["source"] = "sam3_tasc_belief"
        payload["streaming_goal"] = True
        payload["target_valid"] = True
        payload["goal_uid"] = f"sam3_belief:{self.target_link}:{object_id or label}"
        # Shared-control alpha should answer "which object should we assist?",
        # not "how certain is the exact grip contact?".  Use p(object) for
        # alpha and keep p(object)*p(contact|object) as joint_confidence for
        # target/contact diagnostics.
        payload["confidence"] = object_confidence
        payload["object_confidence"] = object_confidence
        payload["joint_confidence"] = joint_confidence
        payload["object_id"] = object_id
        payload["grip_index"] = int(target.get("grip_index", -1))
        return payload, position, label

    def _auto_publish_target_locked(self) -> tuple[GoalPublisher | None, dict[str, Any] | None, np.ndarray | None, str | None]:
        if self.publisher is None:
            return None, None, None, None
        if self.belief_publishing_enabled:
            now = time.monotonic()
            period = 1.0 / max(0.1, self.target_auto_publish_hz)
            if now - self.target_last_stream_publish_monotonic < period:
                return None, None, None, None
            blockers = self._target_blockers_locked()
            if blockers:
                payload = self._null_goal_payload_locked("; ".join(blockers))
                self.target_last_stream_publish_monotonic = now
                self.last_belief_published_label = None
                return self.publisher, payload, None, "null"
            payload, position, label = self._belief_payload_locked()
            self.target_last_stream_publish_monotonic = now
            self.last_belief_published_label = None if label == "null" else label
            return self.publisher, payload, position, label
        tracking = self.target_tracking
        if tracking is None:
            return None, None, None, None
        selected, selection_state = select_tracking_candidate(
            self.status.get("instances", []),
            str(tracking["label"]),
            np.asarray(tracking["last_position"], dtype=np.float64),
            self.target_tracking_max_jump_m,
        )
        if selection_state == "missing":
            self.target_tracking_missing_frames += 1
            if (
                self.target_tracking_missing_frames
                >= self.target_tracking_lost_grace_frames
            ):
                label = str(tracking["label"])
                self.target_tracking = None
                self.status["ee_target_last_result"] = (
                    f"Continuous target stopped: {label} disappeared for "
                    f"{self.target_tracking_missing_frames} frames."
                )
            return None, None, None, None
        if selection_state == "depth_unavailable":
            self.status["ee_target_last_result"] = (
                f"Tracking {tracking['label']}; waiting for a valid 3-D center."
            )
            return None, None, None, None
        if selection_state == "jump":
            self.target_tracking_missing_frames += 1
            if (
                self.target_tracking_missing_frames
                >= self.target_tracking_lost_grace_frames
            ):
                label = str(tracking["label"])
                self.target_tracking = None
                self.status["ee_target_last_result"] = (
                    f"Continuous target stopped: {label} moved more than "
                    f"{self.target_tracking_max_jump_m:.2f} m between updates."
                )
            return None, None, None, None
        if selected is None:
            return None, None, None, None
        self.target_tracking_missing_frames = 0
        tracking["last_position"] = list(selected["center_goal_m"])
        tracking["last_index"] = int(selected["index"])
        now = time.monotonic()
        period = 1.0 / max(0.1, self.target_auto_publish_hz)
        if now - self.target_last_stream_publish_monotonic < period:
            return None, None, None, None
        blockers = self._target_blockers_locked()
        if blockers:
            self.status["ee_target_last_result"] = "Target stream blocked: " + "; ".join(blockers)
            return None, None, None, None
        payload, position = self._goal_payload_for_object_locked(
            selected, streaming_goal=True)
        payload["goal_uid"] = str(tracking["goal_uid"])
        self.target_last_stream_publish_monotonic = now
        label = str(selected.get("label", "object"))
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
                self.status["lcm_robot_state_frames"] += 1
        except Exception as error:
            self._record_error(error)

    def _lcm_loop(self) -> None:
        self.pin_current_thread("lcm_rgbd", "G1_PERCEPTION_IO_CPUSET")
        while self.running:
            try:
                self.bus.handle_timeout(100)
            except Exception as error:
                self._record_error(error)
                time.sleep(0.1)

    def _load_model(self):
        if self.inference_backend == "tensorrt":
            return TensorRTSam3Backend(
                engine_path=Path(self.tensorrt_config["engine"]),
                model_id=str(self.tensorrt_config.get("model_id", "facebook/sam3")),
                threshold=self.threshold,
                min_area_px=self.min_area_px,
                mask_threshold=float(self.tensorrt_config.get("mask_threshold", 0.5)),
            )
        if self.inference_backend == "pytorch":
            return TorchSam3Backend(
                checkpoint=self.checkpoint,
                threshold=self.threshold,
                min_area_px=self.min_area_px,
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
                started = time.perf_counter()
                masks, boxes, scores, labels = backend.infer(image_bgr, concepts)
                if len(masks):
                    masks, boxes, scores, labels = suppress_duplicate_masks(
                        masks, boxes, scores, labels
                    )
                overlay, objects = render_segmented_rgbd(
                    image_bgr, depth, masks, boxes, scores, labels,
                    pair.depth.depth_scale_um, self.min_depth_m, self.max_depth_m,
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
                                state, self.torso_estimator,
                                disagreement_limit_deg,
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
                belief_eef_position = None
                belief_eef_velocity = None
                belief_ray_origin = None
                belief_ray_direction = None
                if state_for_belief is not None:
                    try:
                        belief_eef_position = g1_wrist_ee_position(
                            state_for_belief.q,
                            self.target_link,
                            self.tasc_ee_offset_m,
                        )
                        previous_position = self.tasc_last_eef_position
                        previous_timestamp = self.tasc_last_eef_timestamp_us
                        if previous_position is not None and previous_timestamp is not None:
                            dt = (state_for_belief.timestamp_us - previous_timestamp) / 1.0e6
                            if 1.0e-4 <= dt <= 0.5:
                                belief_eef_velocity = (
                                    belief_eef_position - previous_position
                                ) / dt
                        self.tasc_last_eef_position = belief_eef_position.copy()
                        self.tasc_last_eef_timestamp_us = state_for_belief.timestamp_us
                    except Exception as error:
                        geometry_runtime_error = (
                            f"{geometry_runtime_error}; " if geometry_runtime_error else ""
                        ) + f"TASC EE FK unavailable: {error}"
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
                        xyz_camera = deproject(
                            depth,
                            frame_intrinsics,
                            pair.depth.depth_scale_um,
                            self.min_depth_m,
                            self.max_depth_m,
                        )
                        for index, mask in enumerate(masks):
                            center = robust_mask_center(
                                xyz_camera,
                                mask,
                                min_valid_points=self.target_min_depth_points,
                            )
                            objects[index]["center_camera_m"] = None
                            objects[index]["center_goal_m"] = None
                            objects[index]["center_pixel_uv"] = None
                            objects[index]["valid_depth_points"] = 0
                            if center is None:
                                continue
                            point_camera, pixel_uv, valid_count = center
                            objects[index]["center_camera_m"] = point_camera.tolist()
                            objects[index]["center_pixel_uv"] = list(pixel_uv)
                            objects[index]["valid_depth_points"] = valid_count
                            if t_goal_cam is not None:
                                point_goal = transform_points(point_camera, t_goal_cam)
                                objects[index]["center_goal_m"] = point_goal.tolist()
                                points_goal = masked_goal_points(
                                    xyz_camera,
                                    mask,
                                    t_goal_cam,
                                    max_points=self.tasc_pointcloud_max_points,
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
                            cv2.drawMarker(
                                overlay, pixel_uv, color, cv2.MARKER_CROSS, 18, 2
                            )
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
                if self.tasc_belief.config.enabled:
                    controller_xyzw = np.asarray(
                        [
                            self.target_quaternion[1],
                            self.target_quaternion[2],
                            self.target_quaternion[3],
                            self.target_quaternion[0],
                        ],
                        dtype=np.float64,
                    )
                    belief_telemetry = self.tasc_belief.update(
                        tasc_inputs,
                        eef_position_m=belief_eef_position,
                        user_translation_mps=belief_eef_velocity,
                        ray_origin_m=belief_ray_origin,
                        ray_direction=belief_ray_direction,
                        controller_orientation_xyzw=controller_xyzw,
                    )
                    by_id = {
                        str(item.get("object_id")): item
                        for item in belief_telemetry.get("objects", [])
                    }
                    for index, obj in enumerate(objects):
                        object_id = obj.get("tasc_object_id")
                        belief_obj = by_id.get(str(object_id)) if object_id is not None else None
                        if not belief_obj:
                            continue
                        obj["tasc_object_id"] = object_id
                        obj["tasc_object_probability"] = belief_obj.get(
                            "object_probability"
                        )
                        obj["tasc_selected_grip_index"] = belief_obj.get(
                            "selected_grip_index"
                        )
                        obj["target_contact_goal_m"] = belief_obj.get(
                            "selected_contact_goal_m"
                        )
                        obj["tasc_grip_candidates"] = belief_obj.get(
                            "grip_candidates", []
                        )
                        target = belief_telemetry.get("target") or {}
                        if target.get("object_id") == object_id:
                            obj["tasc_joint_confidence"] = target.get("joint_confidence")
                    if self.tasc_belief.should_log():
                        print(self.tasc_belief.format_log_line(), flush=True)
                        self.tasc_belief.mark_logged()
                    draw_belief_highlight(overlay, objects, belief_telemetry)
                else:
                    belief_telemetry = {"enabled": False, "objects": [], "target": None}
                latency_ms = (time.perf_counter() - started) * 1000.0
                cv2.putText(
                    overlay,
                    f"SAM3-{backend.name} {latency_ms:.0f}ms | {len(concepts)} concepts | pair {pair.delta_us / 1000:.1f}ms",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("overlay JPEG encoding failed")
                auto_publisher = None
                auto_payload = None
                auto_position = None
                auto_label = None
                with self.frame_ready:
                    self.output_jpeg = jpeg.tobytes()
                    self.output_version += 1
                    self.status["processed_frames"] += 1
                    self.status["backend_metadata"] = backend.metadata
                    self.status["last_frame_ts"] = pair.rgb.timestamp_us
                    self.status["last_inference_ms"] = round(latency_ms, 2)
                    self.status["instances"] = objects
                    self.status["geometry_runtime_error"] = geometry_runtime_error
                    self.status["alignment_ready"] = pair.depth.source == DEPTH_SOURCE_ALIGNED_TO_COLOR or (
                        pair.depth.source == DEPTH_SOURCE_RAW_DEPTH_OPTICAL and calibration is not None
                    )
                    self.status["robot_state_delta_ms"] = state_match[1] if state_match else None
                    self.status["torso_transform"] = transform_metadata
                    self.status["tasc_belief"] = belief_telemetry
                    self.status["last_error"] = None
                    self.last_processed_monotonic = time.monotonic()
                    auto_publisher, auto_payload, auto_position, auto_label = (
                        self._auto_publish_target_locked()
                    )
                    self.frame_ready.notify_all()
                if auto_publisher is not None and auto_payload is not None:
                    auto_publisher.publish(
                        auto_payload,
                        repeat=1,
                        period_sec=1.0 / max(0.1, self.target_auto_publish_hz),
                    )
                    with self.lock:
                        if auto_position is None or auto_payload.get("target_valid") is False:
                            reason = (
                                (auto_payload.get("perception") or {}).get("reason")
                                if isinstance(auto_payload, dict)
                                else None
                            )
                            self.status["ee_target_last_result"] = (
                                "Streaming null belief target"
                                + (f": {reason}" if reason else ".")
                            )
                        else:
                            self.status["ee_target_last_result"] = (
                                f"Streaming {self.target_link} belief target "
                                f"{np.round(auto_position, 4).tolist()} for {auto_label}."
                            )
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
    parser.add_argument("--backend", choices=("pytorch", "tensorrt"), default=None)
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
                if target_config.get("send_planner_goal", True)
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
        intrinsics=intrinsics,
        t_torso_cam=t_torso_cam,
        static_t_goal_cam=static_t_goal_cam,
        geometry_blockers=geometry_blockers,
        publisher=publisher,
        workspace=workspace,
        target_config=target_config,
        belief_config=belief_config,
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
