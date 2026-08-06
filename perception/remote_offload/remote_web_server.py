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
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from seg_web_node.lcm_protocol import decode_camera, decode_depth
from seg_web_node.rgbd_sync import depth_at_rgb_size
from seg_web_node.visualization import render_segmented_rgbd

from .protocol import (
    REMOTE_DEPTH_CHANNEL,
    REMOTE_HEARTBEAT_CHANNEL,
    REMOTE_RESULT_CHANNEL,
    REMOTE_RGB_CHANNEL,
    decode_mask,
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
main{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px;padding:14px}
#view{width:100%;max-height:calc(100vh - 80px);object-fit:contain;background:#000;
border:1px solid #33404c}
.panel{background:#181e25;border:1px solid #33404c;border-radius:8px;padding:12px;
margin-bottom:12px}
pre{white-space:pre-wrap;color:#b8c4cf;line-height:1.45;margin:8px 0 0}
.ok{color:#75d087}.bad{color:#ff7b72}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>PATH-G1 Remote Perception</h1><span id="link" class="pill">connecting</span></header>
<main><img id="view" src="/stream.mjpg"><aside>
<div class="panel"><b>Objects</b><pre id="objects">[]</pre></div>
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
    document.getElementById('status').textContent=JSON.stringify(s,null,2);
  }catch(e){document.getElementById('status').textContent=String(e)}
}
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


class RemoteWebRuntime:
    def __init__(self, *, wifi_lcm_url: str, min_depth_m: float, max_depth_m: float) -> None:
        self.bus = lcm.LCM(wifi_lcm_url)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.running = False
        self.rgb = None
        self.depth = None
        self.result: dict[str, Any] | None = None
        self.heartbeat: dict[str, Any] | None = None
        self.output_jpeg = placeholder_jpeg("Waiting for remote RGB-D...")
        self.output_version = 0
        self.stats: dict[str, Any] = {
            "rgb_frames": 0,
            "depth_frames": 0,
            "results": 0,
            "heartbeats": 0,
            "decode_errors": 0,
            "last_error": None,
            "objects": [],
        }
        self.thread: threading.Thread | None = None
        self.bus.subscribe(REMOTE_RGB_CHANNEL, self._on_rgb)
        self.bus.subscribe(REMOTE_DEPTH_CHANNEL, self._on_depth)
        self.bus.subscribe(REMOTE_RESULT_CHANNEL, self._on_result)
        self.bus.subscribe(REMOTE_HEARTBEAT_CHANNEL, self._on_heartbeat)

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._lcm_loop, name="remote-web-lcm", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.frame_ready.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

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
            result_payload = self.result
            if result_payload is not None:
                result["result_sequence"] = result_payload.get("sequence")
                result["result_valid"] = result_payload.get("valid")
                result["result_error"] = result_payload.get("error")
                result["result_age_ms"] = (
                    time.time_ns() // 1000 - int(result_payload.get("published_timestamp_us", 0))
                ) / 1000.0
            return json.loads(json.dumps(result, default=str))

    def _record_error(self, error: Exception | str) -> None:
        with self.lock:
            self.stats["decode_errors"] += 1
            self.stats["last_error"] = str(error)

    def _on_rgb(self, _channel: str, data: bytes) -> None:
        try:
            rgb = decode_camera(data)
            with self.lock:
                self.rgb = rgb
                self.stats["rgb_frames"] += 1
        except Exception as error:
            self._record_error(error)

    def _on_depth(self, _channel: str, data: bytes) -> None:
        try:
            depth = decode_depth(data, allow_unaligned=True)
            with self.lock:
                self.depth = depth
                self.stats["depth_frames"] += 1
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
            with self.lock:
                self.result = payload
                self.stats["results"] += 1
            self._render()
        except Exception as error:
            self._record_error(error)

    def _render(self) -> None:
        with self.lock:
            rgb = self.rgb
            depth = self.depth
            payload = self.result
        if rgb is None or depth is None or payload is None:
            return
        encoded = np.frombuffer(rgb.jpeg, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("JPEG decoding failed")
        pair_like = type("PairLike", (), {"rgb": rgb, "depth": depth})()
        depth_at_rgb = depth_at_rgb_size(pair_like)
        masks, boxes, scores, labels = [], [], [], []
        objects_summary: list[dict[str, Any]] = []
        for obj in payload.get("objects", []):
            mask = decode_mask(obj["mask_data"], int(obj["mask_height"]), int(obj["mask_width"]))
            masks.append(mask)
            boxes.append(obj.get("box_xyxy", [0, 0, 0, 0]))
            scores.append(float(obj.get("score", 0.0)))
            labels.append(str(obj.get("label", "object")))
            objects_summary.append(
                {
                    "track_id": obj.get("track_id"),
                    "label": obj.get("label"),
                    "score": obj.get("score"),
                    "median_depth_m": obj.get("median_depth_m"),
                    "center_camera_m": obj.get("center_camera_m"),
                    "center_goal_m": obj.get("center_goal_m"),
                    "target_contact_goal_m": obj.get("target_contact_goal_m"),
                    "contact_candidates": obj.get("contact_candidates", []),
                    "surface_model": obj.get("surface_model"),
                    "point_filter": obj.get("point_filter"),
                    "geometry_fallback": obj.get("geometry_fallback", False),
                    "missing_frames": obj.get("missing_frames", 0),
                }
            )
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
            depth.depth_scale_um,
            self.min_depth_m,
            self.max_depth_m,
        )
        for index, obj in enumerate(payload.get("objects", [])):
            box = obj.get("box_xyxy") or [8, 24 + index * 20, 8, 24 + index * 20]
            x0 = int(max(0, min(overlay.shape[1] - 1, round(float(box[0])))))
            y0 = int(max(18, min(overlay.shape[0] - 1, round(float(box[1])) - 20)))
            text = f"{obj.get('track_id', '')}"
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
