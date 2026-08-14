"""External desktop web viewer for remote perception Wi-Fi offload."""

from __future__ import annotations

import argparse
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
header{height:48px;padding:0 16px;background:#1c232b;display:flex;align-items:center;gap:14px}
h1{font-size:17px;margin:0;color:#8fd3ff}
.pill{padding:3px 8px;border:1px solid #3a4652;border-radius:999px;color:#b8c4cf}
main{display:grid;grid-template-columns:minmax(0,1fr) 430px;gap:14px;padding:14px}
#view{width:100%;max-height:calc(100vh - 80px);object-fit:contain;background:#000;
border:1px solid #33404c}
.panel{background:#181e25;border:1px solid #33404c;border-radius:8px;padding:12px;
margin-bottom:12px}
pre{white-space:pre-wrap;color:#b8c4cf;line-height:1.45;margin:8px 0 0;max-height:38vh;overflow:auto}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}
th,td{border-bottom:1px solid #33404c;padding:5px;text-align:left}
.dim{opacity:.45}.armed{font-weight:700}.stale{color:#ffb86b}
.ok{color:#75d087}.bad{color:#ff7b72}
.concepts{display:flex;gap:8px;margin-top:8px}.concepts input{flex:1;min-width:0;padding:8px;
background:#10151a;color:#e8edf2;border:1px solid #3a4652;border-radius:5px}
.concepts button{padding:8px 12px;background:#1d6f9f;color:white;border:0;border-radius:5px;cursor:pointer}
#conceptMessage{display:block;margin-top:7px;color:#b8c4cf}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>PATH-G1 Remote Perception</h1><span id="link" class="pill">connecting</span></header>
<main><img id="view" src="/stream.mjpg"><aside>
<div class="panel"><b>SAM3 text labels</b>
<div class="concepts"><input id="conceptInput" value="object" placeholder="bottle, cup">
<button id="conceptApply">Apply</button></div><span id="conceptMessage">waiting for worker</span></div>
<div class="panel"><b>Two-hand belief</b><span id="beliefAge"></span>
<table><thead><tr><th>Object</th><th>Left p/joint</th><th>Right p/joint</th></tr></thead>
<tbody id="beliefRows"></tbody></table></div>
<div class="panel"><b>Geometry</b><pre id="objects">[]</pre></div>
<div class="panel"><b>Status</b><pre id="status">connecting...</pre></div>
</aside></main>
<script>
async function update(){
  try{
    const s=await (await fetch('/api/status')).json();
    const age=Number(s.result_age_ms ?? 1e9);
    document.getElementById('link').textContent=age<1500?'link live':'waiting';
    document.getElementById('link').className='pill '+(age<1500?'ok':'bad');
    document.getElementById('objects').textContent=JSON.stringify(s.objects||[],null,2);
    const stale=Boolean(s.belief_telemetry_stale);
    const selected=String(s.belief_selected_hand||'none');
    const beliefAge=Number(s.belief_telemetry_age_ms??1e9);
    const ageNode=document.getElementById('beliefAge');
    ageNode.textContent=` · ${Number.isFinite(beliefAge)?beliefAge.toFixed(0):'?'} ms${stale?' STALE':''}`;
    ageNode.className=stale?'stale':'';
    const rows=document.getElementById('beliefRows'); rows.innerHTML='';
    for(const o of (s.objects||[])){
      const tr=document.createElement('tr');
      const vals=[`${o.label||'object'}\n${o.track_id||''}`,
        `${Number(o.belief_left?.object_probability||0).toFixed(3)} / ${Number(o.belief_left?.joint_probability||0).toFixed(3)}`,
        `${Number(o.belief_right?.object_probability||0).toFixed(3)} / ${Number(o.belief_right?.joint_probability||0).toFixed(3)}`];
      vals.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;
        if(i===1)td.className=(selected==='left'&&!stale)?'armed':'dim';
        if(i===2)td.className=(selected==='right'&&!stale)?'armed':'dim';tr.appendChild(td)});
      rows.appendChild(tr);
    }
    document.getElementById('status').textContent=JSON.stringify(s,null,2);
    const active=s.heartbeat?.active_concepts||[];
    const pending=s.heartbeat?.pending_concepts;
    const input=document.getElementById('conceptInput');
    if(document.activeElement!==input && active.length)input.value=active.join(', ');
    document.getElementById('conceptMessage').textContent=pending?.length?
      `pending: ${pending.join(', ')}`:`active: ${active.join(', ')||'unknown'}`;
  }catch(e){document.getElementById('status').textContent=String(e)}
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


BELIEF_COLORS = {"left": (255, 180, 60), "right": (80, 180, 255)}


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


def _draw_remote_belief_overlay(
    overlay: np.ndarray,
    objects: list[dict[str, Any]],
    telemetry: dict[str, Any] | None,
    *,
    stale: bool,
) -> None:
    if not isinstance(telemetry, dict):
        return
    hands = telemetry.get("hands") or {}
    selected_hand = str(telemetry.get("selected_hand", "none"))
    for side in ("left", "right"):
        hand = hands.get(side) or {}
        target = hand.get("target") or {}
        object_id = str(target.get("object_id", ""))
        if not object_id:
            continue
        obj = next(
            (item for item in objects if str(item.get("track_id", "")) == object_id),
            None,
        )
        if obj is None:
            continue
        box = obj.get("box_xyxy") or [0, 0, 0, 0]
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
        height, width = overlay.shape[:2]
        x0, x1 = [int(v) for v in np.clip([x0, x1], 0, width - 1)]
        y0, y1 = [int(v) for v in np.clip([y0, y1], 0, height - 1)]
        armed = side == selected_hand and bool(hand.get("armed", False)) and not stale
        color = BELIEF_COLORS[side]
        if armed:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
        else:
            _draw_dashed_rectangle(overlay, (x0, y0), (x1, y1), color)
        probability = float(target.get("object_confidence", 0.0) or 0.0)
        joint = float(target.get("joint_confidence", 0.0) or 0.0)
        cv2.putText(
            overlay,
            f"{side[0].upper()} p={probability:.2f} j={joint:.2f}",
            (x0, min(height - 8, y1 + (18 if side == "left" else 36))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2 if armed else 1,
            cv2.LINE_AA,
        )
        pixel = obj.get("center_pixel_uv")
        if isinstance(pixel, list) and len(pixel) == 2:
            center = (int(pixel[0]), int(pixel[1]))
            cv2.drawMarker(
                overlay,
                center,
                color,
                cv2.MARKER_CROSS,
                18 if armed else 12,
                2 if armed else 1,
            )
            axis = target.get("orientation_axis_world")
            if isinstance(axis, list) and len(axis) == 3:
                end = (
                    int(center[0] + 35.0 * float(axis[0])),
                    int(center[1] - 35.0 * float(axis[1])),
                )
                cv2.arrowedLine(overlay, center, end, color, 2, tipLength=0.25)


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
                self.stats["results"] += 1
                self._request_render_locked()
        except Exception as error:
            self._record_error(error)

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
        with self.lock:
            rgb = self.rgb
            depth = self.depth
            payload = self.result
            belief_telemetry = self.belief_telemetry
            belief_received_monotonic = self.belief_telemetry_received_monotonic
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
        overlay, _objects = render_segmented_rgbd(
            image_bgr,
            depth_at_rgb,
            masks_array,
            boxes_array,
            scores_array,
            labels,
            depth_scale_um,
            self.min_depth_m,
            self.max_depth_m,
        )
        for index, obj in enumerate(wire_objects):
            box = obj.get("box_xyxy") or [8, 24 + index * 20, 8, 24 + index * 20]
            x0 = int(max(0, min(overlay.shape[1] - 1, round(float(box[0])))))
            y0 = int(max(18, min(overlay.shape[0] - 1, round(float(box[1])) - 20)))
            state_delta = obj.get("source_robot_state_delta_ms")
            state_text = "state=?" if state_delta is None else f"state={float(state_delta):.1f}ms"
            geometry_state = "fallback" if obj.get("geometry_fallback") else (
                "current" if obj.get("center_goal_m") is not None else "invalid"
            )
            text = f"{obj.get('track_id', '')} {geometry_state} {state_text}"
            cv2.putText(
                overlay,
                text,
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            pixel = obj.get("center_pixel_uv")
            if isinstance(pixel, list) and len(pixel) == 2:
                center = (int(pixel[0]), int(pixel[1]))
                candidates = obj.get("contact_candidates") or []
                for candidate_index, candidate in enumerate(candidates):
                    color = (60, max(80, 230 - candidate_index * 20), 255)
                    cv2.circle(
                        overlay,
                        (center[0] + candidate_index * 5, center[1]),
                        4,
                        color,
                        -1,
                    )
                primitives = obj.get("surface_primitives") or []
                if primitives:
                    axis = primitives[0].get("axis_goal")
                    if isinstance(axis, list) and len(axis) == 3:
                        end = (
                            int(center[0] + 35.0 * float(axis[0])),
                            int(center[1] - 35.0 * float(axis[1])),
                        )
                        cv2.arrowedLine(
                            overlay, center, end, (220, 100, 255), 2, tipLength=0.25
                        )
                surface = obj.get("surface_model")
                if isinstance(surface, dict):
                    axis = surface.get("axis_goal")
                    if isinstance(axis, list) and len(axis) == 3:
                        end = (
                            int(center[0] + 30.0 * float(axis[0])),
                            int(center[1] - 30.0 * float(axis[1])),
                        )
                        cv2.line(overlay, center, end, (80, 255, 120), 2)
        telemetry_stale = True
        if isinstance(belief_telemetry, dict) and belief_received_monotonic is not None:
            telemetry_age_ms = max(
                0.0, (time.monotonic() - belief_received_monotonic) * 1000.0
            )
            telemetry_stale = telemetry_age_ms > 1500.0
        _draw_remote_belief_overlay(
            overlay,
            objects_summary,
            belief_telemetry,
            stale=telemetry_stale,
        )
        result_age_ms = None
        if isinstance(payload, dict):
            result_age_ms = (
                time.time_ns() // 1000
                - int(payload.get("published_timestamp_us", 0) or 0)
            ) / 1000.0
        footer = (
            "live RGB | segmentation pending"
            if result_age_ms is None
            else f"live RGB | result age={result_age_ms:.0f}ms | belief={'STALE' if telemetry_stale else 'live'}"
        )
        cv2.putText(
            overlay,
            footer,
            (10, overlay.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        ok, jpg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("overlay JPEG encoding failed")
        with self.frame_ready:
            self.output_jpeg = jpg.tobytes()
            self.output_version += 1
            self.stats["objects"] = objects_summary
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
