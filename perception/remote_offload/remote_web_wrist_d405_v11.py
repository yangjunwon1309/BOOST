"""Live diagnostic viewer for PATH_V11_WRIST_* camera bundles."""

from __future__ import annotations

import argparse
import threading
import time
import zlib
from typing import Any

import cv2
import lcm
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .remote_web_server_head_d435i_legacy import (
    CONTROL_STALE_AFTER_MS,
    _draw_remote_belief_overlay,
    _goal_to_pixel,
    _ipopt_marker_style,
)
from .wrist_d405_protocol_v11 import (
    CAMERA_IDS,
    WRIST_BUNDLE_CHANNEL,
    WRIST_BELIEF_TELEMETRY_CHANNEL,
    WRIST_CONCEPT_CONTROL_CHANNEL,
    WRIST_HEARTBEAT_CHANNEL,
    WRIST_IPOPT_VISUAL_CHANNEL,
    WRIST_RESULT_CHANNEL,
    normalize_concepts,
    now_us,
    pack_concept_control,
    unpack_bundle,
    unpack_belief_telemetry,
    unpack_heartbeat,
    unpack_ipopt_visual,
    unpack_mask,
    unpack_result,
)


BELIEF_NORMAL_COLOR = (255, 70, 255)  # magenta, BGR
IPOPT_ORIENTATION_AXES = (
    ("x", (40, 40, 255)),  # red
    ("y", (40, 255, 40)),  # green
    ("z", (255, 180, 40)),  # blue
)


def _ipopt_orientation_axis_color(
    axis_color: tuple[int, int, int],
    marker_color: tuple[int, int, int],
    *,
    belief_applied: bool,
) -> tuple[int, int, int]:
    """Keep XYZ distinguishable while making active belief ownership visible."""
    if not belief_applied:
        return axis_color
    return tuple(
        round(0.35 * axis + 0.65 * marker)
        for axis, marker in zip(axis_color, marker_color)
    )


def _project_goal_axis(
    origin_goal_m: Any,
    axis_goal: Any,
    t_goal_cam: Any,
    intrinsics: Any,
    *,
    length_m: float,
    camera_z_offset_m: float = 0.0,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Project a goal-frame direction segment using the displayed camera pose."""
    try:
        origin = np.asarray(origin_goal_m, dtype=np.float64).reshape(3)
        axis = np.asarray(axis_goal, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if (
            not np.isfinite(origin).all()
            or not np.isfinite(axis).all()
            or axis_norm <= 1.0e-9
        ):
            return None
        endpoint = origin + axis * (float(length_m) / axis_norm)
    except (TypeError, ValueError):
        return None
    start = _goal_to_pixel(
        origin,
        t_goal_cam,
        intrinsics,
        camera_z_offset_m=camera_z_offset_m,
    )
    end = _goal_to_pixel(
        endpoint,
        t_goal_cam,
        intrinsics,
        camera_z_offset_m=camera_z_offset_m,
    )
    return None if start is None or end is None else (start, end)


def _draw_goal_axis(
    image: np.ndarray,
    origin_goal_m: Any,
    axis_goal: Any,
    t_goal_cam: Any,
    intrinsics: Any,
    *,
    color: tuple[int, int, int],
    length_m: float,
    thickness: int = 2,
    camera_z_offset_m: float = 0.0,
) -> tuple[int, int] | None:
    segment = _project_goal_axis(
        origin_goal_m,
        axis_goal,
        t_goal_cam,
        intrinsics,
        length_m=length_m,
        camera_z_offset_m=camera_z_offset_m,
    )
    if segment is None:
        return None
    start, end = segment
    cv2.arrowedLine(image, start, end, color, thickness, cv2.LINE_AA, tipLength=0.22)
    return start


def _draw_orientation_targets(
    image: np.ndarray,
    *,
    side: str,
    hand: dict[str, Any],
    control_hand: dict[str, Any],
    control_fresh: bool,
    belief_stale: bool,
    t_goal_cam: Any,
    intrinsics: Any,
    ipopt_camera_z_offset_m: float,
) -> None:
    """Overlay the belief contact normal and the exact IPOPT wrist-frame axes."""
    target = hand.get("target") if isinstance(hand, dict) else None
    if isinstance(target, dict) and not belief_stale and bool(target.get("orientation_valid")):
        origin = target.get("position_goal_m") or target.get("orientation_source_position_m")
        _draw_goal_axis(
            image,
            origin,
            target.get("orientation_axis_world"),
            t_goal_cam,
            intrinsics,
            color=BELIEF_NORMAL_COLOR,
            length_m=0.075,
        )

    if not control_fresh or not isinstance(control_hand, dict):
        return
    origin = control_hand.get("ipopt_wrist_target_m")
    try:
        rotation = np.asarray(control_hand.get("ipopt_wrist_rotation"), dtype=np.float64).reshape(3, 3)
        rotation_ok = bool(np.isfinite(rotation).all() and abs(float(np.linalg.det(rotation))) > 0.5)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        rotation_ok = False
    if origin is None or not rotation_ok:
        return
    marker_color, belief_applied = _ipopt_marker_style(side, control_hand)
    for axis_index, (_, axis_color) in enumerate(IPOPT_ORIENTATION_AXES):
        _draw_goal_axis(
            image,
            origin,
            rotation[:, axis_index],
            t_goal_cam,
            intrinsics,
            color=_ipopt_orientation_axis_color(
                axis_color, marker_color, belief_applied=belief_applied
            ),
            length_m=0.055,
            camera_z_offset_m=ipopt_camera_z_offset_m,
        )


# Kept deliberately byte-for-byte aligned with the legacy head-D435i viewer's
# user interface.  Only its data channels differ.
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


class WristV11WebViewer:
    BELIEF_STALE_MS = 500.0

    def __init__(self, lcm_url: str, *, ipopt_visual_camera_z_offset_m: float = 0.25) -> None:
        if not np.isfinite(ipopt_visual_camera_z_offset_m):
            raise ValueError("ipopt_visual_camera_z_offset_m must be finite")
        self.bus = lcm.LCM(lcm_url)
        # The wrist-yaw IPOPT target starts behind a forward-mounted D405 in
        # Q_READY. This is a display proxy only; it never changes the solver
        # target or the telemetry sent from the state machine.
        self.ipopt_visual_camera_z_offset_m = float(ipopt_visual_camera_z_offset_m)
        self.lock = threading.Condition()
        self.jpeg: bytes | None = None
        self.bundle_timestamp_us: int | None = None
        self.bundle_sequence: int | None = None
        self.object_count: int | None = None
        self.heartbeat: dict[str, Any] = {"active_concepts": [], "pending_concepts": None}
        # A viewer may restart while the worker remains running.  A wall-clock
        # revision stays newer than any previously acknowledged command.
        self.next_concept_revision = now_us()
        self.latest_images: dict[str, np.ndarray] | None = None
        self.objects: list[dict[str, Any]] = []
        self.latest_transforms: dict[str, dict[str, Any]] = {}
        self.belief: dict[str, Any] | None = None
        self.result_timestamp_us: int | None = None
        self.rendered_result_sequence: int | None = None
        self.result_source_bundle_sequence: int | None = None
        self.rendered_belief: dict[str, Any] | None = None
        self.ipopt_visual: dict[str, Any] | None = None
        self.ipopt_visual_received_monotonic: float | None = None
        self.bus.subscribe(WRIST_BUNDLE_CHANNEL, self._on_bundle)
        self.bus.subscribe(WRIST_RESULT_CHANNEL, self._on_result)
        self.bus.subscribe(WRIST_HEARTBEAT_CHANNEL, self._on_heartbeat)
        self.bus.subscribe(WRIST_BELIEF_TELEMETRY_CHANNEL, self._on_belief)
        self.bus.subscribe(WRIST_IPOPT_VISUAL_CHANNEL, self._on_ipopt_visual)

    def _on_bundle(self, _channel: str, data: bytes) -> None:
        try:
            bundle = unpack_bundle(data)
            images: dict[str, np.ndarray] = {}
            for camera_id in CAMERA_IDS:
                frame = bundle["cameras"][camera_id]
                image = cv2.imdecode(np.frombuffer(frame["color_jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"bad JPEG from {camera_id}")
                cv2.putText(image, f"{camera_id} {frame['serial']} {frame['width']}x{frame['height']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                images[camera_id] = image
            with self.lock:
                self.bundle_timestamp_us = int(bundle["bundle_timestamp_us"])
                self.bundle_sequence = int(bundle["sequence"])
                self.latest_images = images
                # Intrinsics belong to every RGB-D frame, not to a SAM object.
                # Keeping them from the bundle allows IPOPT projection even
                # when this camera currently has no segmentation result.
                for camera_id in CAMERA_IDS:
                    self.latest_transforms.setdefault(camera_id, {})[
                        "intrinsics"
                    ] = dict(bundle["cameras"][camera_id]["intrinsics"])
                self._render_locked()
        except Exception:
            return

    def _on_result(self, _channel: str, data: bytes) -> None:
        try:
            result = unpack_result(data)
            objects = [obj for obj in result.get("objects", []) if isinstance(obj, dict)]
            transforms = result.get("t_goal_cam") or {}
            with self.lock:
                self.objects = objects
                self.object_count = len(objects)
                self.latest_transforms = {
                    camera_id: {
                        "t_goal_cam": transforms.get(camera_id)
                        or self.latest_transforms.get(camera_id, {}).get("t_goal_cam"),
                        "intrinsics": next(
                            (
                                obj.get("intrinsics")
                                for obj in objects
                                if obj.get("camera_id") == camera_id
                                and obj.get("intrinsics")
                            ),
                            self.latest_transforms.get(camera_id, {}).get("intrinsics"),
                        ),
                    }
                    for camera_id in CAMERA_IDS
                }
                self.result_timestamp_us = int(
                    result.get("published_timestamp_us", 0) or 0
                )
                self.rendered_result_sequence = int(result.get("sequence", -1))
                source_sequence = result.get("source_bundle_sequence")
                self.result_source_bundle_sequence = (
                    int(source_sequence)
                    if isinstance(source_sequence, int) and source_sequence >= 0
                    else None
                )
                self._render_locked()
        except Exception:
            return

    def _on_heartbeat(self, _channel: str, data: bytes) -> None:
        try:
            heartbeat = unpack_heartbeat(data)
            with self.lock:
                self.heartbeat = heartbeat
                self.next_concept_revision = max(
                    self.next_concept_revision,
                    int(heartbeat.get("concept_revision", -1)) + 1,
                )
        except Exception:
            return

    def _on_belief(self, _channel: str, data: bytes) -> None:
        try:
            with self.lock:
                belief = unpack_belief_telemetry(data)
                self.belief = belief
                self.rendered_belief = belief
                self._render_locked()
        except Exception:
            return

    def _on_ipopt_visual(self, _channel: str, data: bytes) -> None:
        try:
            visual = unpack_ipopt_visual(data)
            with self.lock:
                self.ipopt_visual = visual
                # Publisher timestamps use Thor's monotonic clock and cannot
                # be compared with the laptop clock. Freshness is therefore
                # measured from the local LCM callback receipt time.
                self.ipopt_visual_received_monotonic = time.monotonic()
                self._render_locked()
        except Exception:
            return

    def _encode_locked(self, images: dict[str, np.ndarray]) -> None:
        stacked = np.vstack([images[camera_id] for camera_id in CAMERA_IDS])
        split_y = images["left_wrist"].shape[0]
        cv2.line(
            stacked,
            (0, split_y),
            (stacked.shape[1] - 1, split_y),
            (0, 0, 255),
            2,
        )
        ok, encoded = cv2.imencode(
            ".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if ok:
            self.jpeg = encoded.tobytes()
            self.lock.notify_all()

    def _belief_stale_locked(self) -> bool:
        belief = self.belief
        if not isinstance(belief, dict):
            return True
        published = int(belief.get("published_timestamp_us", 0) or 0)
        return published <= 0 or (now_us() - published) / 1000.0 > self.BELIEF_STALE_MS

    def _ipopt_control_locked(self) -> dict[str, Any] | None:
        visual = getattr(self, "ipopt_visual", None)
        if not isinstance(visual, dict):
            return None
        received_monotonic = getattr(
            self, "ipopt_visual_received_monotonic", None
        )
        if received_monotonic is None:
            return None
        age_ms = max(0.0, (time.monotonic() - received_monotonic) * 1000.0)
        return {
            "schema": visual.get("schema"),
            "state": visual.get("state"),
            "hands": visual.get("hands", {}),
            "age_ms": age_ms,
            "publisher_timestamp_us": visual.get("published_timestamp_us"),
        }

    def _ipopt_projection_status_locked(
        self, control: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not isinstance(control, dict):
            return {}
        hands = control.get("hands") or {}
        offset_m = float(getattr(self, "ipopt_visual_camera_z_offset_m", 0.25))
        status: dict[str, Any] = {}
        for camera_id, side in (
            ("left_wrist", "left"),
            ("right_wrist", "right"),
        ):
            metadata = self.latest_transforms.get(camera_id, {})
            transform_raw = metadata.get("t_goal_cam")
            intrinsics = dict(metadata.get("intrinsics") or {})
            if intrinsics:
                intrinsics.setdefault("cx", intrinsics.get("ppx"))
                intrinsics.setdefault("cy", intrinsics.get("ppy"))
            target = (hands.get(side) or {}).get("ipopt_wrist_target_m")
            rotation_raw = (hands.get(side) or {}).get("ipopt_wrist_rotation")
            try:
                rotation = np.asarray(rotation_raw, dtype=np.float64).reshape(3, 3)
                rotation_available = bool(
                    np.isfinite(rotation).all()
                    and abs(float(np.linalg.det(rotation))) > 0.5
                )
            except (TypeError, ValueError, np.linalg.LinAlgError):
                rotation_available = False
            camera_point = None
            try:
                homogeneous = np.ones(4, dtype=np.float64)
                homogeneous[:3] = np.asarray(target, dtype=np.float64).reshape(3)
                transform = np.asarray(transform_raw, dtype=np.float64).reshape(4, 4)
                camera_point = (np.linalg.inv(transform) @ homogeneous)[:3]
            except (TypeError, ValueError, np.linalg.LinAlgError):
                pass
            pixel = _goal_to_pixel(
                target,
                transform_raw,
                intrinsics or None,
                camera_z_offset_m=offset_m,
            )
            status[camera_id] = {
                "side": side,
                "target_pelvis_m": target,
                "transform_available": transform_raw is not None,
                "intrinsics_available": bool(intrinsics),
                "raw_camera_point_m": (
                    None if camera_point is None else camera_point.astype(float).tolist()
                ),
                "display_camera_z_m": (
                    None if camera_point is None else float(camera_point[2] + offset_m)
                ),
                "pixel_uv_local": None if pixel is None else list(pixel),
                "orientation_rotation_available": rotation_available,
            }
        return status

    def _render_locked(self) -> None:
        if self.latest_images is None:
            return
        images = {
            camera_id: image.copy()
            for camera_id, image in self.latest_images.items()
        }
        self._draw_sam3_masks_locked(images)
        self._draw_latest_belief_locked(images)
        self._encode_locked(images)

    def _draw_sam3_masks_locked(self, images: dict[str, np.ndarray]) -> None:
        """Alpha-blend the SAM3 local masks carried by the newest result.

        Masks are image-space data for ``result_source_bundle_sequence``.  The
        live viewer deliberately keeps the newest RGB frame for low-latency
        target projection, so ``status()`` exposes whether the mask and RGB
        sequences match.  The mask itself is still shown, just like its
        matching result bounding box, rather than silently dropping SAM3's
        segmentation whenever worker latency is non-zero.
        """
        colors = {
            "left_wrist": (255, 110, 25),
            "right_wrist": (25, 145, 255),
        }
        for obj in self.objects:
            camera_id = obj.get("camera_id")
            image = images.get(camera_id)
            if image is None or obj.get("mask_encoding") != "bitpack-zlib-row-major":
                continue
            payload = obj.get("mask_data")
            if not isinstance(payload, (bytes, bytearray)):
                continue
            height, width = image.shape[:2]
            try:
                mask = unpack_mask(bytes(payload), width=width, height=height)
            except (TypeError, ValueError, zlib.error):
                continue
            if not np.any(mask):
                continue
            mask_layer = image.copy()
            mask_layer[mask] = colors[camera_id]
            cv2.addWeighted(mask_layer, 0.38, image, 0.62, 0.0, image)

    def _draw_latest_belief_locked(self, images: dict[str, np.ndarray]) -> None:
        """Project pelvis-frame belief into the camera image displayed now."""
        belief = self.belief or {}
        hands = belief.get("hands") if isinstance(belief, dict) else None
        hands = hands if isinstance(hands, dict) else {}
        ipopt_control = self._ipopt_control_locked()
        control = ipopt_control or belief.get("control")
        if not hands and not isinstance(control, dict):
            return
        latest = self.latest_transforms or {}
        for camera_id in CAMERA_IDS:
            camera_side = "left" if camera_id == "left_wrist" else "right"
            transform = latest.get(camera_id, {}).get("t_goal_cam")
            intrinsics = dict(
                latest.get(camera_id, {}).get("intrinsics") or {}
            )
            if intrinsics:
                intrinsics.setdefault("cx", intrinsics.get("ppx"))
                intrinsics.setdefault("cy", intrinsics.get("ppy"))
            camera_objects = [
                obj for obj in self.objects if obj.get("camera_id") == camera_id
            ]
            camera_hands: dict[str, Any] = {}
            known = {str(obj.get("track_id", "")): obj for obj in camera_objects}
            for side in ("left", "right"):
                hand = dict(hands.get(side) or {})
                target = hand.get("target")
                if not isinstance(target, dict) or target.get("camera_id") != camera_id:
                    hand["target"] = None
                    camera_hands[side] = hand
                    continue
                target = dict(target)
                current = known.get(str(target.get("object_id", "")))
                # Boxes are image-frame measurements: bind them only to the
                # newest observation of the same track. Pelvis targets remain
                # valid without a box and are projected below with new pose.
                target["box_xyxy"] = current.get("box_xyxy") if current else None
                target.pop("t_goal_cam", None)
                target.pop("intrinsics", None)
                hand["target"] = target
                camera_hands[side] = hand
            _draw_remote_belief_overlay(
                images[camera_id],
                camera_objects,
                {**belief, "hands": camera_hands},
                stale=self._belief_stale_locked(),
                t_goal_cam=transform,
                intrinsics=intrinsics or None,
                control=control,
                legend_side=camera_side,
                ipopt_camera_z_offset_m=getattr(
                    self, "ipopt_visual_camera_z_offset_m", 0.25
                ),
                # Each local overlay is drawn before np.vstack. The left local
                # pixel therefore lands in the upper image and the right local
                # pixel is shifted automatically into the lower image.
                render_sides=(camera_side,),
            )
            control_hand = (
                (control.get("hands") or {}).get(camera_side, {})
                if isinstance(control, dict)
                else {}
            )
            try:
                control_fresh = float(control.get("age_ms")) <= CONTROL_STALE_AFTER_MS
            except (AttributeError, TypeError, ValueError):
                control_fresh = False
            _draw_orientation_targets(
                images[camera_id],
                side=camera_side,
                hand=camera_hands.get(camera_side, {}),
                control_hand=control_hand,
                control_fresh=control_fresh,
                belief_stale=self._belief_stale_locked(),
                t_goal_cam=transform,
                intrinsics=intrinsics or None,
                ipopt_camera_z_offset_m=getattr(
                    self, "ipopt_visual_camera_z_offset_m", 0.25
                ),
            )

    def lcm_loop(self) -> None:
        while True:
            self.bus.handle_timeout(100)

    def stream(self):
        previous: bytes | None = None
        while True:
            with self.lock:
                self.lock.wait_for(lambda: self.jpeg is not None and self.jpeg != previous, timeout=1.0)
                current = self.jpeg
            if current is None:
                continue
            previous = current
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(current)).encode() + b"\r\n\r\n" + current + b"\r\n"

    def status(self) -> dict[str, Any]:
        with self.lock:
            age = None if self.bundle_timestamp_us is None else round((time.time_ns() // 1000 - self.bundle_timestamp_us) / 1000.0, 1)
            hands = (
                (self.rendered_belief or {}).get("hands", {})
                if isinstance(self.rendered_belief, dict)
                else {}
            )
            ipopt_control = self._ipopt_control_locked()
            return {
                "heartbeat": dict(self.heartbeat),
                "bundle_sequence": self.bundle_sequence,
                "bundle_age_ms": age,
                "object_count": self.object_count,
                "belief_result_sequence": self.rendered_result_sequence,
                "mask_source_bundle_sequence": self.result_source_bundle_sequence,
                "mask_overlay_matches_live_rgb": (
                    self.result_source_bundle_sequence is not None
                    and self.result_source_bundle_sequence == self.bundle_sequence
                ),
                "overlay_uses_latest_camera_pose": bool(self.latest_transforms),
                "belief_stale": self._belief_stale_locked(),
                "ipopt_visual_age_ms": (
                    None
                    if ipopt_control is None
                    else round(float(ipopt_control["age_ms"]), 1)
                ),
                "ipopt_targets": (
                    {} if ipopt_control is None else ipopt_control.get("hands", {})
                ),
                "ipopt_visual_camera_z_offset_m": self.ipopt_visual_camera_z_offset_m,
                "ipopt_freshness_clock": "laptop_lcm_receipt_monotonic",
                "ipopt_projection": self._ipopt_projection_status_locked(
                    ipopt_control
                ),
                "camera_stack_order": ["left_wrist", "right_wrist"],
                "belief_targets": {
                    side: (hands.get(side) or {}).get("target")
                    for side in ("left", "right")
                },
            }

    def set_concepts(self, value: Any) -> list[str]:
        concepts = normalize_concepts(value)
        with self.lock:
            revision = self.next_concept_revision
            self.next_concept_revision += 1
            self.heartbeat = {
                **self.heartbeat,
                "pending_concepts": concepts,
            }
        packet = pack_concept_control(
            {
                "revision": revision,
                "published_timestamp_us": now_us(),
                "concepts": concepts,
            }
        )
        # LCM rides UDP multicast here.  Match the legacy viewer's redundant
        # transmission; the worker accepts a command only once by revision.
        for _ in range(3):
            self.bus.publish(WRIST_CONCEPT_CONTROL_CHANNEL, packet)
        return concepts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--ipopt-visual-camera-z-offset-m",
        type=float,
        default=0.25,
        help="Display-only D405 optical +Z offset for IPOPT wrist targets (default: 0.25).",
    )
    args = parser.parse_args()
    viewer = WristV11WebViewer(
        args.wifi_lcm_url,
        ipopt_visual_camera_z_offset_m=args.ipopt_visual_camera_z_offset_m,
    )
    threading.Thread(target=viewer.lcm_loop, daemon=True).start()
    app = FastAPI()
    app.get("/")(lambda: HTMLResponse(HTML))
    app.get("/api/status")(lambda: JSONResponse(viewer.status()))
    app.get("/stream.mjpg")(lambda: StreamingResponse(viewer.stream(), media_type="multipart/x-mixed-replace; boundary=frame"))
    @app.post("/api/concepts")
    async def set_concepts(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            return JSONResponse({"concepts": viewer.set_concepts(payload.get("concepts"))})
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
