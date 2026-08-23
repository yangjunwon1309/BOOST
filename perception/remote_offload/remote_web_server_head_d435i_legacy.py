"""External desktop web viewer for remote perception Wi-Fi offload."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import socket
import threading
import time
from typing import Any

import cv2
import lcm
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from seg_web_node.lcm_protocol import decode_camera, decode_depth
from seg_web_node.rgbd_sync import depth_at_rgb_size
from seg_web_node.visualization import render_segmented_rgbd

from .protocol import (
    REMOTE_BELIEF_TELEMETRY_CHANNEL,
    REMOTE_CONCEPT_CONTROL_CHANNEL,
    REMOTE_DEPTH_CHANNEL,
    REMOTE_HEARTBEAT_CHANNEL,
    REMOTE_RESULT_CHANNEL,
    REMOTE_RGB_CHANNEL,
    decode_mask,
    normalize_concepts,
    now_us,
    pack_concept_control,
    unpack_belief_telemetry,
    unpack_heartbeat,
    unpack_result,
)

LOG = logging.getLogger("remote_offload.remote_web_server")


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PATH-G1 Remote Perception</title>
<style>
body{margin:0;background:#111318;color:#e8edf2;font:14px system-ui,sans-serif}
header{min-height:48px;padding:8px 16px;background:#1c232b;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:17px;margin:0;color:#8fd3ff}
main{padding:14px}
#view{width:100%;max-height:calc(100vh - 80px);object-fit:contain;background:#000;
border:1px solid #33404c}
.concepts{display:flex;gap:8px;align-items:center}.concepts input{width:min(360px,48vw);min-width:160px;padding:8px;
background:#10151a;color:#e8edf2;border:1px solid #3a4652;border-radius:5px}
.concepts button{padding:8px 12px;background:#1d6f9f;color:white;border:0;border-radius:5px;cursor:pointer}
#conceptMessage{color:#b8c4cf}
</style></head><body>
<header><h1>PATH-G1 Remote Perception</h1><div class="concepts"><input id="conceptInput" value="object" placeholder="bottle, cup">
<button id="conceptApply">Apply</button><span id="conceptMessage">waiting for worker</span></div></header>
<main><img id="view" src="/stream.mjpg"></main>
<script>
async function update(){
  try{
    const s=await (await fetch('/api/status')).json();
    const active=s.heartbeat?.active_concepts||[];
    const pending=s.heartbeat?.pending_concepts;
    const input=document.getElementById('conceptInput');
    if(document.activeElement!==input && active.length)input.value=active.join(', ');
    document.getElementById('conceptMessage').textContent=pending?.length?
      `pending: ${pending.join(', ')}`:`active: ${active.join(', ')||'unknown'}`;
  }catch(e){document.getElementById('conceptMessage').textContent=`status unavailable: ${e}`}
}
async function applyConcepts(){
  const message=document.getElementById('conceptMessage');
  try{
    const response=await fetch('/api/concepts',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({concepts:document.getElementById('conceptInput').value})});
    const data=await response.json(); if(!response.ok)throw new Error(data.detail||response.statusText);
    message.textContent=`sent: ${data.concepts.join(', ')}`;
  }catch(e){message.textContent=`error: ${e}`}
}
document.getElementById('conceptApply').addEventListener('click',applyConcepts);
document.getElementById('conceptInput').addEventListener('keydown',e=>{if(e.key==='Enter')applyConcepts()});
setInterval(update,1000);update();
</script></body></html>"""


def placeholder_jpeg(message: str, width: int = 960, height: int = 540) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        message,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (190, 220, 255),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("placeholder JPEG encoding failed")
    return encoded.tobytes()


def detect_lan_host() -> str:
    candidates: list[str] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            candidates.append(str(sock.getsockname()[0]))
        finally:
            sock.close()
    except OSError:
        pass
    for candidate in candidates:
        if candidate and not candidate.startswith("127."):
            return candidate
    return "127.0.0.1"


BELIEF_COLORS = {"left": (255, 170, 20), "right": (40, 140, 255)}
IPOPT_TARGET_COLOR = (70, 220, 70)
BELIEF_DIM_ALPHA = 0.75
CONTROL_STALE_AFTER_MS = 500.0


def _belief_for_object(
    telemetry: dict[str, Any] | None,
    side: str,
    track_id: str,
) -> dict[str, Any]:
    hand = ((telemetry or {}).get("hands") or {}).get(side) or {}
    probabilities = hand.get("object_probabilities") or {}
    selected = next(
        (
            obj
            for obj in hand.get("objects", [])
            if str(obj.get("object_id", "")) == str(track_id)
        ),
        {},
    )
    grips = selected.get("grip_probabilities") or []
    try:
        grip_index = int(selected.get("selected_grip_index", -1))
    except (TypeError, ValueError):
        grip_index = -1
    contact_probability = (
        float(grips[grip_index]) if 0 <= grip_index < len(grips) else 0.0
    )
    object_probability = float(
        selected.get("object_probability", probabilities.get(track_id, 0.0)) or 0.0
    )
    target = hand.get("target") or {}
    target_for_object = (
        target if str(target.get("object_id", "")) == str(track_id) else {}
    )
    return {
        "object_probability": object_probability,
        "contact_probability": contact_probability,
        "joint_probability": object_probability * contact_probability,
        "selected_contact_goal_m": selected.get("selected_contact_goal_m"),
        "selected_grip_index": grip_index,
        "orientation_relation": target_for_object.get("orientation_relation"),
        "orientation_confidence": target_for_object.get("orientation_confidence"),
        "orientation_valid": target_for_object.get("orientation_valid", False),
        "orientation_axis_world": target_for_object.get("orientation_axis_world"),
    }


def _display_posterior(
    telemetry: dict[str, Any] | None,
    track_id: str,
) -> float | None:
    """Use the armed hand's object posterior, else the best hand posterior."""
    hands = ((telemetry or {}).get("hands") or {})
    selected = str((telemetry or {}).get("selected_hand", "none"))
    selected_hand = hands.get(selected) or {}
    if selected in hands and selected_hand.get("armed"):
        value = _belief_for_object(telemetry, selected, track_id)[
            "object_probability"
        ]
        return float(value) if value > 0.0 else None
    values = [
        _belief_for_object(telemetry, side, track_id)["object_probability"]
        for side in ("left", "right")
        if side in hands
    ]
    best = max(values, default=0.0)
    return float(best) if best > 0.0 else None


def _draw_text_with_halo(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw legible coloured text without hiding the segmentation beneath it."""
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_dashed_rectangle(
    image: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x0, y0 = first
    x1, y1 = second
    for start in range(x0, x1 + 1, 12):
        cv2.line(image, (start, y0), (min(start + 6, x1), y0), color, thickness)
        cv2.line(image, (start, y1), (min(start + 6, x1), y1), color, thickness)
    for start in range(y0, y1 + 1, 12):
        cv2.line(image, (x0, start), (x0, min(start + 6, y1)), color, thickness)
        cv2.line(image, (x1, start), (x1, min(start + 6, y1)), color, thickness)


def _goal_to_pixel(
    point_goal_m: Any,
    t_goal_cam: Any,
    intrinsics: Any,
    *,
    camera_z_offset_m: float = 0.0,
) -> tuple[int, int] | None:
    """Project a goal-frame point into the frame that produced its geometry.

    ``camera_z_offset_m`` is visualization-only and is expressed in the
    camera's RealSense optical +Z direction.  It is used for an IPOPT wrist
    origin that may physically lie behind its wrist-mounted camera.
    """
    if point_goal_m is None or t_goal_cam is None or not intrinsics:
        return None
    try:
        homogeneous = np.ones(4, dtype=np.float64)
        homogeneous[:3] = np.asarray(point_goal_m, dtype=np.float64).reshape(3)
        transform = np.asarray(t_goal_cam, dtype=np.float64).reshape(4, 4)
        camera = (np.linalg.inv(transform) @ homogeneous)[:3]
        camera[2] += float(camera_z_offset_m)
        if not np.isfinite(camera).all() or camera[2] <= 1.0e-6:
            return None
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        if width <= 0 or height <= 0:
            return None
        u = round(
            float(intrinsics["fx"]) * camera[0] / camera[2]
            + float(intrinsics["cx"])
        )
        v = round(
            float(intrinsics["fy"]) * camera[1] / camera[2]
            + float(intrinsics["cy"])
        )
    except (TypeError, ValueError, KeyError, np.linalg.LinAlgError):
        return None
    return (
        int(np.clip(u, 0, width - 1)),
        int(np.clip(v, 0, height - 1)),
    )


def _project_axis_arrow(
    origin_goal_m: Any,
    axis_goal: Any,
    t_goal_cam: Any,
    intrinsics: Any,
    length_m: float = 0.08,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Project both endpoints of a physical 3-D axis segment."""
    try:
        start = np.asarray(origin_goal_m, dtype=np.float64).reshape(3)
        direction = np.asarray(axis_goal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(start).all() or not np.isfinite(norm) or norm < 1.0e-9:
            return None
        end = start + direction / norm * float(length_m)
    except (TypeError, ValueError):
        return None
    first = _goal_to_pixel(start, t_goal_cam, intrinsics)
    second = _goal_to_pixel(end, t_goal_cam, intrinsics)
    return None if first is None or second is None else (first, second)


def _projection_metadata(
    payload: dict[str, Any] | None,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, None
    raw = payload.get("t_goal_cam")
    intrinsics = payload.get("intrinsics")
    if not isinstance(raw, list) or len(raw) != 16 or not isinstance(intrinsics, dict):
        return None, None
    try:
        transform = np.asarray(raw, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(transform).all():
        return None, None
    return transform, intrinsics


def _draw_belief_legend(
    overlay: np.ndarray,
    _hands: dict[str, Any],
    control: dict[str, Any],
    selected_hand: str,
    stale: bool,
    *,
    legend_side: str | None = None,
    ipopt_camera_z_offset_m: float = 0.0,
) -> None:
    """Draw the hand-local visual key and the common IPOPT target key."""
    y = 22
    sides = (legend_side,) if legend_side in BELIEF_COLORS else ("left", "right")
    for side in sides:
        color = BELIEF_COLORS[side]
        initial = side[0].upper()
        cv2.drawMarker(
            overlay, (18, y - 4), color, cv2.MARKER_TILTED_CROSS, 14, 2
        )
        _draw_text_with_halo(
            overlay, f"{initial} human", (32, y), color, scale=0.48
        )
        cv2.drawMarker(
            overlay, (128, y - 4), color, cv2.MARKER_CROSS, 14, 2
        )
        _draw_text_with_halo(
            overlay, f"{initial} belief", (142, y), color, scale=0.48
        )
        y += 20
    cv2.drawMarker(
        overlay, (18, y - 4), IPOPT_TARGET_COLOR, cv2.MARKER_DIAMOND, 14, 2
    )
    ipopt_label = (
        "IPOPT target"
        if abs(float(ipopt_camera_z_offset_m)) <= 1.0e-9
        else f"IPOPT view +Z {float(ipopt_camera_z_offset_m):.2f}m"
    )
    _draw_text_with_halo(overlay, ipopt_label, (32, y), IPOPT_TARGET_COLOR, scale=0.48)
    del control, selected_hand, stale


def _draw_hand_markers(
    layer: np.ndarray,
    objects: list[dict[str, Any]],
    side: str,
    hand: dict[str, Any],
    control_hand: dict[str, Any],
    *,
    armed: bool,
    t_goal_cam: Any,
    intrinsics: Any,
    control_fresh: bool,
    ipopt_camera_z_offset_m: float,
) -> None:
    """Draw operator command, reference, and belief at grasp-centre points."""
    color = BELIEF_COLORS[side]
    initial = side[0].upper()
    if control_fresh:
        pixel = _goal_to_pixel(
            control_hand.get("human_grasp_goal_m"), t_goal_cam, intrinsics
        )
        if pixel is not None:
            cv2.drawMarker(layer, pixel, color, cv2.MARKER_TILTED_CROSS, 18, 2)
            _draw_text_with_halo(
                layer,
                f"{initial} human",
                (pixel[0] + 12, pixel[1] - 8),
                color,
                scale=0.46,
            )
    # This is the exact Cartesian target passed into the arm IPOPT solve:
    # VR/reference motion normally, or the selected shared target while its
    # switch is held.  It is independent of whether belief geometry exists.
    if control_fresh:
        ipopt_target = control_hand.get(
            "ipopt_wrist_target_m",
            control_hand.get("reference_grasp_goal_m"),
        )
        pixel = _goal_to_pixel(
            ipopt_target,
            t_goal_cam,
            intrinsics,
            camera_z_offset_m=ipopt_camera_z_offset_m,
        )
        if pixel is not None:
            cv2.drawMarker(
                layer, pixel, IPOPT_TARGET_COLOR, cv2.MARKER_DIAMOND, 20, 2
            )
            _draw_text_with_halo(
                layer,
                (
                    "IPOPT target"
                    if abs(float(ipopt_camera_z_offset_m)) <= 1.0e-9
                    else f"IPOPT +Z {float(ipopt_camera_z_offset_m):.2f}m"
                ),
                (pixel[0] + 12, pixel[1] + 16),
                IPOPT_TARGET_COLOR,
                scale=0.46,
            )

    target = hand.get("target") or {}
    object_id = str(target.get("object_id", ""))
    if not object_id:
        return
    obj = next(
        (item for item in objects if str(item.get("track_id", "")) == object_id),
        None,
    )
    box_px = 3 if armed else 1
    if obj is not None:
        box = obj.get("box_xyxy") or [0, 0, 0, 0]
        try:
            x0, y0, x1, y1 = [round(float(value)) for value in box]
        except (TypeError, ValueError):
            x0 = y0 = x1 = y1 = 0
        height, width = layer.shape[:2]
        x0, x1 = [int(value) for value in np.clip([x0, x1], 0, width - 1)]
        y0, y1 = [int(value) for value in np.clip([y0, y1], 0, height - 1)]
        if armed:
            cv2.rectangle(layer, (x0, y0), (x1, y1), color, box_px)
        else:
            _draw_dashed_rectangle(layer, (x0, y0), (x1, y1), color, box_px)

    belief = _belief_for_object(
        {"hands": {side: hand}}, side, object_id
    )
    contact_goal = target.get("position_goal_m")
    if contact_goal is None:
        selected = next(
            (
                value
                for value in (hand.get("objects") or [])
                if str(value.get("object_id", "")) == object_id
            ),
            {},
        )
        contact_goal = selected.get("selected_contact_goal_m")
    pixel = _goal_to_pixel(contact_goal, t_goal_cam, intrinsics)
    if pixel is None:
        return
    cv2.drawMarker(
        layer,
        pixel,
        color,
        cv2.MARKER_CROSS,
        26,
        box_px,
    )
    _draw_text_with_halo(
        layer,
        f"{initial} belief  {float(belief['contact_probability']):.2f}",
        (pixel[0] + 18, pixel[1] + 6),
        color,
        scale=0.50,
        thickness=2 if armed else 1,
    )


def _draw_remote_belief_overlay(
    overlay: np.ndarray,
    objects: list[dict[str, Any]],
    telemetry: dict[str, Any] | None,
    *,
    stale: bool,
    t_goal_cam: Any = None,
    intrinsics: Any = None,
    control: dict[str, Any] | None = None,
    legend_side: str | None = None,
    ipopt_camera_z_offset_m: float = 0.0,
) -> None:
    control = control if isinstance(control, dict) else {}
    control_hands = control.get("hands") or {}
    try:
        control_age_ms = float(control.get("age_ms"))
    except (TypeError, ValueError):
        control_age_ms = float("inf")
    control_fresh = control_age_ms <= CONTROL_STALE_AFTER_MS
    if not isinstance(telemetry, dict):
        _draw_belief_legend(
            overlay,
            {},
            control,
            "none",
            stale,
            legend_side=legend_side,
            ipopt_camera_z_offset_m=ipopt_camera_z_offset_m,
        )
        return
    hands = telemetry.get("hands") or {}
    selected_hand = str(telemetry.get("selected_hand", "none"))
    armed_sides: list[str] = []
    dim_sides: list[str] = []
    for side in ("left", "right"):
        hand = hands.get(side) or {}
        armed = side == selected_hand and bool(hand.get("armed", False)) and not stale
        (armed_sides if armed else dim_sides).append(side)

    if dim_sides:
        scratch = overlay.copy()
        for side in dim_sides:
            _draw_hand_markers(
                scratch,
                objects,
                side,
                hands.get(side) or {},
                control_hands.get(side) or {},
                armed=False,
                t_goal_cam=t_goal_cam,
                intrinsics=intrinsics,
                control_fresh=control_fresh,
                ipopt_camera_z_offset_m=ipopt_camera_z_offset_m,
            )
        cv2.addWeighted(
            scratch,
            BELIEF_DIM_ALPHA,
            overlay,
            1.0 - BELIEF_DIM_ALPHA,
            0.0,
            overlay,
        )

    for side in armed_sides:
        _draw_hand_markers(
            overlay,
            objects,
            side,
            hands.get(side) or {},
            control_hands.get(side) or {},
            armed=True,
            t_goal_cam=t_goal_cam,
            intrinsics=intrinsics,
            control_fresh=control_fresh,
            ipopt_camera_z_offset_m=ipopt_camera_z_offset_m,
        )
    _draw_belief_legend(
        overlay,
        hands,
        control,
        selected_hand,
        stale,
        legend_side=legend_side,
        ipopt_camera_z_offset_m=ipopt_camera_z_offset_m,
    )


class RemoteWebRuntime:
    def __init__(self, *, wifi_lcm_url: str, min_depth_m: float, max_depth_m: float) -> None:
        self.bus = lcm.LCM(wifi_lcm_url)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.render_ready = threading.Condition(self.lock)
        self.running = False
        self.render_requested = False
        self.rgb = None
        self.depth = None
        self.result: dict[str, Any] | None = None
        self.result_history: collections.OrderedDict[int, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self.result_history_max = 12
        # A telemetry/result sequence mismatch is normal on Wi-Fi.  Keep the
        # most recent matched geometry so valid belief targets do not flicker.
        self.last_belief_visual_objects: list[dict[str, Any]] = []
        self.last_belief_visual_t_goal_cam: np.ndarray | None = None
        self.last_belief_visual_intrinsics: dict[str, Any] | None = None
        self.heartbeat: dict[str, Any] | None = None
        self.belief_telemetry: dict[str, Any] | None = None
        self.belief_telemetry_received_monotonic: float | None = None
        self.output_jpeg = placeholder_jpeg("Waiting for remote RGB-D...")
        self.output_version = 0
        self.stats: dict[str, Any] = {
            "rgb_frames": 0,
            "depth_frames": 0,
            "results": 0,
            "heartbeats": 0,
            "belief_telemetry_packets": 0,
            "decode_errors": 0,
            "last_error": None,
            "objects": [],
            "concept_commands": 0,
        }
        self.pending_control_packets: list[bytes] = []
        self.concept_revision = int(time.time_ns() // 1000)
        self.threads: list[threading.Thread] = []
        self.bus.subscribe(REMOTE_RGB_CHANNEL, self._on_rgb)
        self.bus.subscribe(REMOTE_DEPTH_CHANNEL, self._on_depth)
        self.bus.subscribe(REMOTE_RESULT_CHANNEL, self._on_result)
        self.bus.subscribe(REMOTE_HEARTBEAT_CHANNEL, self._on_heartbeat)
        self.bus.subscribe(
            REMOTE_BELIEF_TELEMETRY_CHANNEL,
            self._on_belief_telemetry,
        )

    def start(self) -> None:
        self.running = True
        self.threads = [
            threading.Thread(
                target=self._lcm_loop, name="remote-web-lcm", daemon=True
            ),
            threading.Thread(
                target=self._render_loop, name="remote-web-render", daemon=True
            ),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.frame_ready.notify_all()
            self.render_ready.notify_all()
        for thread in self.threads:
            thread.join(timeout=2.0)

    def snapshot(self) -> bytes:
        with self.lock:
            return self.output_jpeg

    def wait_for_frame(self, previous_version: int, timeout: float = 1.0) -> tuple[int, bytes]:
        with self.frame_ready:
            self.frame_ready.wait_for(
                lambda: self.output_version != previous_version or not self.running,
                timeout=timeout,
            )
            return self.output_version, self.output_jpeg

    def status_snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.stats)
            result["heartbeat"] = self.heartbeat
            telemetry = self.belief_telemetry
            result["belief_telemetry"] = telemetry
            received_monotonic = self.belief_telemetry_received_monotonic
            if telemetry is not None and received_monotonic is not None:
                belief_age_ms = max(
                    0.0, (time.monotonic() - received_monotonic) * 1000.0
                )
                result["belief_telemetry_age_ms"] = belief_age_ms
                result["belief_telemetry_stale"] = belief_age_ms > 1500.0
                result["belief_selected_hand"] = telemetry.get(
                    "selected_hand", "none"
                )
            else:
                result["belief_telemetry_age_ms"] = None
                result["belief_telemetry_stale"] = True
                result["belief_selected_hand"] = "none"
            result_payload = self.result
            if result_payload is not None:
                result["result_sequence"] = result_payload.get("sequence")
                result["result_valid"] = result_payload.get("valid")
                result["result_error"] = result_payload.get("error")
                result["result_age_ms"] = (
                    time.time_ns() // 1000 - int(result_payload.get("published_timestamp_us", 0))
                ) / 1000.0
            return json.loads(json.dumps(result, default=str))

    def _request_render_locked(self) -> None:
        self.render_requested = True
        self.render_ready.notify()

    def _record_error(self, error: Exception | str) -> None:
        with self.lock:
            self.stats["decode_errors"] += 1
            self.stats["last_error"] = str(error)

    def set_concepts(self, value: Any) -> dict[str, Any]:
        concepts = normalize_concepts(value)
        with self.lock:
            self.concept_revision = max(
                self.concept_revision + 1, int(time.time_ns() // 1000)
            )
            payload = {
                "version": 1,
                "published_timestamp_us": now_us(),
                "revision": self.concept_revision,
                "concepts": concepts,
            }
            packet = pack_concept_control(payload)
            # Send redundantly on UDP multicast while retaining one revision;
            # the worker ignores duplicate revisions.
            self.pending_control_packets.extend([packet, packet, packet])
            self.stats["concept_commands"] += 1
        return payload

    def _on_rgb(self, _channel: str, data: bytes) -> None:
        try:
            rgb = decode_camera(data)
            with self.render_ready:
                self.rgb = rgb
                self.stats["rgb_frames"] += 1
                self._request_render_locked()
        except Exception as error:
            self._record_error(error)

    def _on_depth(self, _channel: str, data: bytes) -> None:
        try:
            depth = decode_depth(data, allow_unaligned=True)
            with self.render_ready:
                self.depth = depth
                self.stats["depth_frames"] += 1
                self._request_render_locked()
        except Exception as error:
            self._record_error(error)

    def _on_heartbeat(self, _channel: str, data: bytes) -> None:
        try:
            heartbeat = unpack_heartbeat(data)
            with self.lock:
                self.heartbeat = heartbeat
                self.stats["heartbeats"] += 1
        except Exception as error:
            self._record_error(error)

    def _on_result(self, _channel: str, data: bytes) -> None:
        try:
            payload = unpack_result(data)
            with self.render_ready:
                self.result = payload
                sequence = int(payload.get("sequence", -1))
                if sequence >= 0:
                    self.result_history[sequence] = payload
                    while len(self.result_history) > self.result_history_max:
                        self.result_history.popitem(last=False)
                self.stats["results"] += 1
                self._request_render_locked()
        except Exception as error:
            self._record_error(error)

    def _select_belief_pair(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
        """Return newest segmentation and the result described by telemetry."""
        with self.lock:
            telemetry = self.belief_telemetry
            newest = self.result
            history = dict(self.result_history)
        if not isinstance(telemetry, dict):
            return newest, None, "no_telemetry"
        try:
            sequence = int(telemetry.get("result_sequence", -1))
        except (TypeError, ValueError):
            sequence = -1
        matched = history.get(sequence)
        if matched is not None:
            return newest, matched, "matched"
        try:
            source_us = int(telemetry.get("source_rgb_timestamp_us", 0) or 0)
        except (TypeError, ValueError):
            source_us = 0
        if source_us:
            for payload in history.values():
                if int(payload.get("source_rgb_timestamp_us", 0) or 0) == source_us:
                    return newest, payload, "matched_ts"
        return newest, None, "unmatched"

    def _on_belief_telemetry(self, _channel: str, data: bytes) -> None:
        try:
            payload = unpack_belief_telemetry(data)
            with self.render_ready:
                self.belief_telemetry = payload
                self.belief_telemetry_received_monotonic = time.monotonic()
                self.stats["belief_telemetry_packets"] += 1
                self._request_render_locked()
        except Exception as error:
            self._record_error(error)

    def _render_loop(self) -> None:
        while self.running:
            with self.render_ready:
                self.render_ready.wait_for(
                    lambda: self.render_requested or not self.running,
                    timeout=0.5,
                )
                if not self.running:
                    break
                self.render_requested = False
            try:
                self._render()
            except Exception as error:
                self._record_error(error)

    def _render(self) -> None:
        newest, matched, match_state = self._select_belief_pair()
        with self.lock:
            rgb = self.rgb
            depth = self.depth
            belief_telemetry = self.belief_telemetry
            belief_received_monotonic = self.belief_telemetry_received_monotonic
        payload = newest
        if rgb is None:
            return
        encoded = np.frombuffer(rgb.jpeg, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("JPEG decoding failed")
        if depth is not None:
            try:
                pair_like = type("PairLike", (), {"rgb": rgb, "depth": depth})()
                depth_at_rgb = depth_at_rgb_size(pair_like)
                depth_scale_um = depth.depth_scale_um
            except Exception:
                depth_at_rgb = np.zeros(image_bgr.shape[:2], dtype=np.uint16)
                depth_scale_um = 1000
        else:
            depth_at_rgb = np.zeros(image_bgr.shape[:2], dtype=np.uint16)
            depth_scale_um = 1000
        masks, boxes, scores, labels = [], [], [], []
        objects_summary: list[dict[str, Any]] = []
        wire_objects = payload.get("objects", []) if isinstance(payload, dict) else []
        for obj in wire_objects:
            mask = decode_mask(obj["mask_data"], int(obj["mask_height"]), int(obj["mask_width"]))
            masks.append(mask)
            boxes.append(obj.get("box_xyxy", [0, 0, 0, 0]))
            scores.append(float(obj.get("score", 0.0)))
            labels.append(str(obj.get("label", "object")))
            summary = {
                    "track_id": obj.get("track_id"),
                    "label": obj.get("label"),
                    "score": obj.get("score"),
                    "box_xyxy": obj.get("box_xyxy"),
                    "center_pixel_uv": obj.get("center_pixel_uv"),
                    "median_depth_m": obj.get("median_depth_m"),
                    "center_camera_m": obj.get("center_camera_m"),
                    "center_goal_m": obj.get("center_goal_m"),
                    "target_contact_goal_m": obj.get("target_contact_goal_m"),
                    "contact_candidates": obj.get("contact_candidates", []),
                    "surface_primitives": obj.get("surface_primitives", []),
                    "surface_model": obj.get("surface_model"),
                    "point_filter": obj.get("point_filter"),
                    "geometry_fallback": obj.get("geometry_fallback", False),
                    "missing_frames": obj.get("missing_frames", 0),
                    "source_robot_state_delta_ms": obj.get(
                        "source_robot_state_delta_ms"
                    ),
                    "geometry_source_timestamp_us": obj.get(
                        "geometry_source_timestamp_us"
                    ),
                    "goal_frame": obj.get("goal_frame"),
                }
            summary["belief_left"] = _belief_for_object(
                belief_telemetry, "left", str(obj.get("track_id", ""))
            )
            summary["belief_right"] = _belief_for_object(
                belief_telemetry, "right", str(obj.get("track_id", ""))
            )
            objects_summary.append(summary)
        if masks:
            masks_array = np.asarray(masks, dtype=bool)
            boxes_array = np.asarray(boxes, dtype=np.float32)
            scores_array = np.asarray(scores, dtype=np.float32)
        else:
            height, width = image_bgr.shape[:2]
            masks_array = np.empty((0, height, width), dtype=bool)
            boxes_array = np.empty((0, 4), dtype=np.float32)
            scores_array = np.empty((0,), dtype=np.float32)
        overlay, _objects, instance_colors = render_segmented_rgbd(
            image_bgr,
            depth_at_rgb,
            masks_array,
            boxes_array,
            scores_array,
            labels,
            depth_scale_um,
            self.min_depth_m,
            self.max_depth_m,
            include_depth_panel=False,
            draw_score=False,
            draw_depth_text=False,
            return_instance_colors=True,
        )
        result_t_goal_cam, result_intrinsics = _projection_metadata(payload)
        for index, obj in enumerate(wire_objects):
            color = (
                instance_colors[index]
                if index < len(instance_colors)
                else (200, 200, 200)
            )
            box = obj.get("box_xyxy") or [0, 0, 0, 0]
            x0 = int(np.clip(round(float(box[0])), 0, overlay.shape[1] - 1))
            y0 = int(
                np.clip(round(float(box[1])) - 8, 14, overlay.shape[0] - 1)
            )
            posterior = _display_posterior(
                belief_telemetry, str(obj.get("track_id", ""))
            )
            label = str(obj.get("label", "object"))
            text = label if posterior is None else f"{label}  {posterior:.2f}"
            _draw_text_with_halo(overlay, text, (x0, y0), color, scale=0.58)
            primitives = obj.get("surface_primitives") or []
            if primitives:
                primitive = primitives[0]
                arrow = _project_axis_arrow(
                    primitive.get("origin_goal_m"),
                    primitive.get("axis_goal"),
                    result_t_goal_cam,
                    result_intrinsics,
                )
                if arrow is not None:
                    cv2.arrowedLine(
                        overlay,
                        arrow[0],
                        arrow[1],
                        color,
                        2,
                        cv2.LINE_AA,
                        tipLength=0.25,
                    )
        telemetry_stale = True
        if isinstance(belief_telemetry, dict) and belief_received_monotonic is not None:
            telemetry_age_ms = max(
                0.0, (time.monotonic() - belief_received_monotonic) * 1000.0
            )
            telemetry_stale = telemetry_age_ms > 1500.0
        belief_objects: list[dict[str, Any]] = []
        belief_t_goal_cam = None
        belief_intrinsics = None
        if matched is not None:
            matched_objects = [
                {
                    "track_id": obj.get("track_id"),
                    "box_xyxy": obj.get("box_xyxy"),
                }
                for obj in matched.get("objects", [])
                if isinstance(obj, dict)
            ]
            matched_t_goal_cam, matched_intrinsics = _projection_metadata(matched)
            if matched_objects and matched_t_goal_cam is not None and matched_intrinsics:
                self.last_belief_visual_objects = matched_objects
                self.last_belief_visual_t_goal_cam = matched_t_goal_cam.copy()
                self.last_belief_visual_intrinsics = dict(matched_intrinsics)
                belief_objects = matched_objects
                belief_t_goal_cam = matched_t_goal_cam
                belief_intrinsics = matched_intrinsics
        if belief_t_goal_cam is None:
            belief_objects = list(
                getattr(self, "last_belief_visual_objects", [])
            )
            belief_t_goal_cam = getattr(
                self, "last_belief_visual_t_goal_cam", None
            )
            belief_intrinsics = getattr(
                self, "last_belief_visual_intrinsics", None
            )
        raw_control = (
            belief_telemetry.get("control")
            if isinstance(belief_telemetry, dict)
            else None
        )
        control = dict(raw_control) if isinstance(raw_control, dict) else {}
        try:
            control_age_ms = float(control.get("age_ms"))
        except (TypeError, ValueError):
            control_age_ms = None
        if control_age_ms is not None and belief_received_monotonic is not None:
            control_age_ms += max(
                0.0, (time.monotonic() - belief_received_monotonic) * 1000.0
            )
        control["age_ms"] = control_age_ms
        _draw_remote_belief_overlay(
            overlay,
            belief_objects,
            belief_telemetry,
            stale=telemetry_stale,
            t_goal_cam=belief_t_goal_cam,
            intrinsics=belief_intrinsics,
            control=control,
        )
        result_age_ms = None
        if isinstance(payload, dict):
            result_age_ms = (
                time.time_ns() // 1000
                - int(payload.get("published_timestamp_us", 0) or 0)
            ) / 1000.0
        ok, jpg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("overlay JPEG encoding failed")
        with self.frame_ready:
            self.output_jpeg = jpg.tobytes()
            self.output_version += 1
            self.stats["objects"] = objects_summary
            self.stats["belief_match_state"] = match_state
            self.stats["result_age_ms"] = result_age_ms
            self.stats["control_age_ms"] = control_age_ms
            self.frame_ready.notify_all()

    def _lcm_loop(self) -> None:
        while self.running:
            try:
                with self.lock:
                    packet = (
                        self.pending_control_packets.pop(0)
                        if self.pending_control_packets
                        else None
                    )
                if packet is not None:
                    self.bus.publish(REMOTE_CONCEPT_CONTROL_CHANNEL, packet)
                self.bus.handle_timeout(100)
            except Exception as error:
                self._record_error(error)
                time.sleep(0.1)


def build_app(runtime: RemoteWebRuntime) -> FastAPI:
    app = FastAPI(title="PATH-G1 Remote Perception Webview")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/status")
    def status():
        return runtime.status_snapshot()

    @app.post("/api/concepts")
    async def concepts(request: Request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return runtime.set_concepts(payload.get("concepts"))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/stream.mjpg")
    def stream():
        def generator():
            version = -1
            while runtime.running:
                version, frame = runtime.wait_for_frame(version, timeout=1.0)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

        return StreamingResponse(
            generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runtime = RemoteWebRuntime(
        wifi_lcm_url=args.wifi_lcm_url,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    runtime.start()
    host = detect_lan_host() if args.host in ("0.0.0.0", "::", "") else args.host
    LOG.info("remote webview listening at http://%s:%s", host, args.port)
    try:
        uvicorn.run(build_app(runtime), host=args.host, port=args.port, log_level="info")
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
