"""Live diagnostic viewer for PATH_V11_WRIST_* camera bundles."""

from __future__ import annotations

import argparse
import collections
import threading
import time
from typing import Any

import cv2
import lcm
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .remote_web_server_head_d435i_legacy import _draw_remote_belief_overlay
from .wrist_d405_protocol_v11 import (
    CAMERA_IDS,
    WRIST_BUNDLE_CHANNEL,
    WRIST_BELIEF_TELEMETRY_CHANNEL,
    WRIST_CONCEPT_CONTROL_CHANNEL,
    WRIST_HEARTBEAT_CHANNEL,
    WRIST_RESULT_CHANNEL,
    normalize_concepts,
    now_us,
    pack_concept_control,
    unpack_bundle,
    unpack_belief_telemetry,
    unpack_heartbeat,
    unpack_mask,
    unpack_result,
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
    def __init__(self, lcm_url: str) -> None:
        self.bus = lcm.LCM(lcm_url)
        self.lock = threading.Condition()
        self.jpeg: bytes | None = None
        self.bundle_timestamp_us: int | None = None
        self.bundle_sequence: int | None = None
        self.object_count: int | None = None
        self.heartbeat: dict[str, Any] = {"active_concepts": [], "pending_concepts": None}
        # A viewer may restart while the worker remains running.  A wall-clock
        # revision stays newer than any previously acknowledged command.
        self.next_concept_revision = now_us()
        self.image_history: collections.OrderedDict[
            int, dict[str, np.ndarray]
        ] = collections.OrderedDict()
        self.result_history: collections.OrderedDict[int, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self.belief_history: collections.OrderedDict[int, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self.rendered_result_sequence: int | None = None
        self.rendered_belief: dict[str, Any] | None = None
        self.bus.subscribe(WRIST_BUNDLE_CHANNEL, self._on_bundle)
        self.bus.subscribe(WRIST_RESULT_CHANNEL, self._on_result)
        self.bus.subscribe(WRIST_HEARTBEAT_CHANNEL, self._on_heartbeat)
        self.bus.subscribe(WRIST_BELIEF_TELEMETRY_CHANNEL, self._on_belief)

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
                self._bounded_store(self.image_history, self.bundle_sequence, images)
                self._render_matching_locked()
        except Exception:
            return

    def _on_result(self, _channel: str, data: bytes) -> None:
        try:
            with self.lock:
                result = unpack_result(data)
                self._bounded_store(
                    self.result_history, int(result.get("sequence", -1)), result
                )
                self._render_matching_locked()
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
                sequence = int(belief.get("result_sequence", -1))
                if sequence >= 0:
                    self._bounded_store(self.belief_history, sequence, belief)
                self._render_matching_locked()
        except Exception:
            return

    @staticmethod
    def _bounded_store(
        history: collections.OrderedDict[int, Any], key: int, value: Any
    ) -> None:
        if key < 0:
            return
        history[key] = value
        history.move_to_end(key)
        while len(history) > 60:
            history.popitem(last=False)

    def _matched_triplet_locked(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None:
        """Return only an image/result/belief triple from the same result epoch."""
        for result_sequence in reversed(self.result_history):
            result = self.result_history[result_sequence]
            belief = self.belief_history.get(result_sequence)
            source_sequence = result.get("source_bundle_sequence")
            try:
                images = self.image_history.get(int(source_sequence))
            except (TypeError, ValueError):
                images = None
            if belief is not None and images is not None:
                return images, result, belief
        return None

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

    def _render_matching_locked(self) -> None:
        matched = self._matched_triplet_locked()
        if matched is None:
            # It is useful to keep video live while SAM3/belief catches up, but
            # never place stale masks or a belief marker onto that newer image.
            if not self.image_history:
                return
            latest_key = next(reversed(self.image_history))
            latest_images = self.image_history[latest_key]
            self.rendered_result_sequence = None
            self.rendered_belief = None
            self.object_count = None
            self._encode_locked(
                {camera_id: image.copy() for camera_id, image in latest_images.items()}
            )
            return
        source_images, result, belief = matched
        images = {
            camera_id: image.copy() for camera_id, image in source_images.items()
        }
        objects = list(result.get("objects", []))
        for obj in objects:
            camera_id = obj.get("camera_id")
            if camera_id not in images or obj.get("mask_encoding") != "bitpack-zlib-row-major":
                continue
            try:
                mask = unpack_mask(obj["mask_data"], width=int(obj["mask_width"]), height=int(obj["mask_height"]))
                image = images[camera_id]
                if mask.shape != image.shape[:2]:
                    continue
                color = np.array((40, 220, 40) if camera_id == "left_wrist" else (255, 140, 20), dtype=np.uint8)
                image[mask] = (0.55 * image[mask] + 0.45 * color).astype(np.uint8)
                x0, y0, _x1, _y1 = np.asarray(obj.get("box_xyxy"), dtype=int).reshape(4)
                cv2.putText(image, f"{obj.get('label', 'object')} {float(obj.get('score', 0.0)):.2f}", (max(0, x0), max(32, y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1, cv2.LINE_AA)
            except Exception:
                continue
        self._draw_belief_locked(images, objects, belief)
        self.rendered_result_sequence = int(result.get("sequence", -1))
        self.rendered_belief = belief
        self.object_count = len(objects)
        self._encode_locked(images)

    def _draw_belief_locked(
        self,
        images: dict[str, np.ndarray],
        objects: list[dict[str, Any]],
        belief: dict[str, Any],
    ) -> None:
        hands = belief.get("hands") if isinstance(belief, dict) else None
        if not isinstance(hands, dict):
            return
        for camera_id in CAMERA_IDS:
            camera_objects = [
                obj for obj in objects if obj.get("camera_id") == camera_id
            ]
            camera_hands: dict[str, Any] = {}
            projection: dict[str, Any] | None = None
            known_ids = {str(obj.get("track_id", "")) for obj in camera_objects}
            for side in ("left", "right"):
                hand = dict(hands.get(side) or {})
                target = hand.get("target")
                if not isinstance(target, dict) or target.get("camera_id") != camera_id:
                    hand["target"] = None
                    camera_hands[side] = hand
                    continue
                projection = target
                track_id = str(target.get("object_id", ""))
                if track_id and track_id not in known_ids:
                    camera_objects.append(
                        {
                            "track_id": track_id,
                            "box_xyxy": target.get("box_xyxy"),
                            "camera_id": camera_id,
                        }
                    )
                camera_hands[side] = hand
            if projection is None:
                projection = next(
                    (obj for obj in camera_objects if obj.get("t_goal_cam") and obj.get("intrinsics")),
                    None,
                )
            transform = projection.get("t_goal_cam") if projection else None
            intrinsics = dict(projection.get("intrinsics") or {}) if projection else None
            if intrinsics:
                intrinsics.setdefault("cx", intrinsics.get("ppx"))
                intrinsics.setdefault("cy", intrinsics.get("ppy"))
            _draw_remote_belief_overlay(
                images[camera_id],
                camera_objects,
                {**belief, "hands": camera_hands},
                stale=False,
                t_goal_cam=transform,
                intrinsics=intrinsics,
                control=belief.get("control"),
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
            return {
                "heartbeat": dict(self.heartbeat),
                "bundle_sequence": self.bundle_sequence,
                "bundle_age_ms": age,
                "object_count": self.object_count,
                "belief_result_sequence": self.rendered_result_sequence,
                "overlay_sequence_matched": self.rendered_result_sequence is not None,
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
    args = parser.parse_args()
    viewer = WristV11WebViewer(args.wifi_lcm_url)
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
