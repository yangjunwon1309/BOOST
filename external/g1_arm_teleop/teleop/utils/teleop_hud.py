"""Read-only teleop HUD for Vuer, independent of camera and robot loops.

Create TeleopHUD before TeleVuerWrapper and pass it as scene_overlay. The
producer only publishes a small JSON snapshot; each headset session renders
changed content at 2 Hz in a worker thread. No Vuer or font imports occur on
the producer path. A stale snapshot means UI updates stopped, not robot FAULT.
"""

import asyncio
from copy import deepcopy
from io import BytesIO
import json
import math
from multiprocessing import Array, Lock
from pathlib import Path
import sys
import time
from typing import NamedTuple


_FIELDS = ("schema", "task", "state", "record_enabled", "recording", "episode_id",
           "elapsed_s", "frame_count", "xr_ready", "home_fraction", "home_remaining_s",
           "home_confirmed", "summary", "notice", "error", "warnings", "tactile")
_SUMMARY_FIELDS = ("ready", "loading", "total", "success", "failure", "unlabeled",
                   "incomplete", "unknown", "last_saved", "pending_save", "pending_label_updates")
_STATUS = {
    "INITIALIZING": ("시작 준비", "INITIALIZING", (157, 176, 201)),
    "READY": ("준비 완료", "READY", (105, 192, 247)),
    "TRACKING": ("추적 중", "TRACKING", (76, 213, 171)),
    "RECORDING": ("녹화 중", "RECORDING", (255, 104, 108)),
    "WAITING_XR": ("XR 입력 대기", "WAITING FOR XR", (255, 197, 93)),
    "HOMING": ("홈 복귀 중", "RETURNING HOME", (255, 197, 93)),
    "SAVING": ("저장 중", "SAVING", (255, 197, 93)),
    "PAUSED": ("일시정지", "PAUSED", (157, 176, 201)),
    "STOPPING": ("종료 중", "STOPPING", (157, 176, 201)),
    "FAULT": ("오류로 중단", "FAULT", (255, 104, 108)),
}


def _payload_copy(payload):
    clean = {key: payload[key] for key in _FIELDS if key in payload}
    for key, limit in (("task", 160), ("notice", 400), ("error", 800)):
        if key in clean:
            clean[key] = str(clean[key])[:limit]
    if "warnings" in clean:
        warnings = clean["warnings"]
        warnings = [warnings] if isinstance(warnings, str) else warnings
        clean["warnings"] = [item[:240] for item in (warnings or ())
                             if isinstance(item, str) and item.strip()]
    if "summary" in clean:
        summary = clean["summary"]
        clean["summary"] = {key: summary[key] for key in _SUMMARY_FIELDS if key in summary}
    return clean


class SharedHUDSnapshot:
    """Bounded parent/child snapshot with nonblocking reads and writes.

    If a producer is interrupted while holding the lock, the reader retains
    its last complete local snapshot and can still display the stale marker.
    """

    def __init__(self, capacity=16384, stale_after_s=2.5):
        if capacity < 128 or not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError("HUD capacity and stale timeout must be positive")
        self._buffer = Array("c", int(capacity), lock=False)
        self._lock = Lock()
        self._stale_ns = int(stale_after_s * 1e9)
        self._last_read = None
        self.last_publish_error = None

    def publish(self, payload, now_ns=None):
        try:
            stamp = time.monotonic_ns() if now_ns is None else int(now_ns)
            encoded = json.dumps({"updated_monotonic_ns": stamp, "payload": _payload_copy(payload)},
                                 ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) >= len(self._buffer):
                raise ValueError("HUD snapshot exceeds its shared buffer")
        except Exception as error:
            self.last_publish_error = str(error)
            return False
        if not self._lock.acquire(False):
            return False
        try:
            self._buffer.value = encoded
        finally:
            self._lock.release()
        self.last_publish_error = None
        return True

    def snapshot(self, now_ns=None):
        if self._lock.acquire(False):
            try:
                encoded = self._buffer.value
            finally:
                self._lock.release()
            if encoded:
                try:
                    self._last_read = json.loads(encoded.decode("utf-8"))
                except (ValueError, UnicodeError):
                    pass
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        if self._last_read is None:
            return {"available": False, "payload": {}, "updated_monotonic_ns": None,
                    "age_s": None, "stale": False}
        stamp = self._last_read["updated_monotonic_ns"]
        age_ns = max(0, now_ns - stamp)
        return {"available": True, "payload": deepcopy(self._last_read["payload"]),
                "updated_monotonic_ns": stamp, "age_s": age_ns / 1e9,
                "stale": age_ns >= self._stale_ns}


def _count(value):
    try:
        return max(0, int(value))
    except (ValueError, TypeError, OverflowError):
        return None


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _view(snapshot):
    payload = snapshot["payload"]
    summary = payload.get("summary") or {}
    ready = bool(summary.get("ready"))
    unlabeled, unknown = _count(summary.get("unlabeled")), _count(summary.get("unknown"))
    last_saved = summary.get("last_saved") or {}
    last_outcome = last_saved.get("outcome")
    if last_outcome not in ("success", "failure", "unlabeled", "unknown"):
        result = last_saved.get("result")
        if isinstance(result, dict) and type(result.get("success")) is bool:
            expected = "success" if result["success"] else "failure"
            last_outcome = expected if result.get("label", expected) == expected else "unknown"
        else:
            last_outcome = "unlabeled" if result is None else "unknown"
    return {
        "available": snapshot["available"], "stale": snapshot["stale"],
        "task": str(payload.get("task") or "TELEOP"),
        "state": payload.get("state", "INITIALIZING"),
        "episode": _count(payload.get("episode_id")),
        "record_enabled": bool(payload.get("record_enabled", False)),
        "recording": bool(payload.get("recording", False)),
        "elapsed": _count(payload.get("elapsed_s")) or 0,
        "frames": _count(payload.get("frame_count")) or 0,
        "xr_ready": bool(payload.get("xr_ready", False)),
        "fraction": _finite(payload.get("home_fraction")),
        "remaining": _finite(payload.get("home_remaining_s")),
        "home_confirmed": payload.get("home_confirmed"),
        "counts": [_count(summary.get(name)) if ready else None
                   for name in ("total", "success", "failure")]
                  + [unlabeled + unknown if ready and unlabeled is not None and unknown is not None else None],
        "summary_ready": ready,
        "pending_save": bool(summary.get("pending_save")),
        "pending_labels": bool(summary.get("pending_label_updates")),
        "last_saved_episode": _count(last_saved.get("episode_id")),
        "last_saved_outcome": last_outcome,
        "notice": str(payload.get("notice") or ""),
        "error": str(payload.get("error") or ""),
        "warnings": list(payload.get("warnings") or []),
        "tactile": payload.get("tactile"),
    }


def _notice_text(view, ko):
    if view["stale"]:
        return "UI 상태 갱신이 지연되었습니다" if ko else "UI updates are stale; showing last known values"
    if view["error"]:
        return view["error"]
    if view["warnings"]:
        suffix = f" (+{len(view['warnings']) - 1})" if len(view["warnings"]) > 1 else ""
        return ("경고: " if ko else "WARNING: ") + view["warnings"][0] + suffix
    if view["pending_labels"]:
        return "라벨 저장 중" if ko else "LABEL UPDATE PENDING"
    if view["pending_save"]:
        return "에피소드 저장 중" if ko else "EPISODE SAVE PENDING"
    if not view["record_enabled"]:
        return "녹화 꺼짐 - 조작만 진행" if ko else "RECORDING OFF - teleoperation only"
    if not view["summary_ready"]:
        return "저장 기록 집계 중" if ko else "COUNTING SAVED EPISODES"
    if view["state"] == "PAUSED" and view["home_confirmed"] is False:
        return "홈 도착 미확인 - r로 직접 재개" if ko else "HOME UNCONFIRMED - press r to resume"
    if view["notice"]:
        return view["notice"]
    return ("XR 정상" if view["xr_ready"] else "XR 입력 대기") if ko else (
        "XR READY" if view["xr_ready"] else "WAITING FOR XR")


def _last_saved_text(view, ko):
    if view["last_saved_episode"] is None or view["pending_save"] or view["pending_labels"]:
        return None
    outcome = view["last_saved_outcome"]
    label = {"success": "성공", "failure": "실패", "unlabeled": "미분류", "unknown": "확인 필요"}[outcome] if ko else outcome.upper()
    prefix = "직전" if ko else "LAST"
    return f"{prefix} EP {view['last_saved_episode']:04d} {label}"


class HUDFrame(NamedTuple):
    src: bytes
    alpha_src: bytes
    preview_png: bytes
    language: str


class _HUDRenderer:
    WIDTH, HEIGHT = 720, 330

    def __init__(self, font_path=None):
        self.font_path = font_path
        self.fonts = None

    def _load_fonts(self):
        from PIL import ImageFont
        candidates = ([self.font_path] if self.font_path else [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ])
        korean = False
        selected = None
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                try:
                    ImageFont.truetype(candidate, 24)
                    selected, korean = candidate, True
                    break
                except OSError:
                    pass
        if selected is None:
            selected = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        try:
            self.fonts = {size: ImageFont.truetype(selected, size) for size in (18, 21, 25, 31, 38)}
        except OSError:
            self.fonts = {size: ImageFont.load_default() for size in (18, 21, 25, 31, 38)}
            korean = False
        self.korean = korean

    def render(self, view, preview=False):
        from PIL import Image, ImageDraw
        if self.fonts is None:
            self._load_fonts()
        ko = self.korean
        rgb = Image.new("RGB", (self.WIDTH, self.HEIGHT))
        alpha = Image.new("L", rgb.size, 0)
        draw, mask = ImageDraw.Draw(rgb), ImageDraw.Draw(alpha)
        draw.rounded_rectangle((7, 7, 713, 323), radius=20, fill=(12, 20, 33), outline=(67, 85, 107), width=2)
        mask.rounded_rectangle((7, 7, 713, 323), radius=20, fill=226)
        white, muted = (239, 245, 253), (153, 173, 198)

        def text(x, y, value, size=21, color=white, width=None):
            value = str(value)
            font = self.fonts[size]
            if not ko:
                value = value.encode("ascii", "replace").decode("ascii")
            if width is not None and draw.textlength(value, font=font) > width:
                while value and draw.textlength(value + "...", font=font) > width:
                    value = value[:-1]
                value += "..."
            draw.text((x, y), value, font=font, fill=color)
            mask.text((x, y), value, font=font, fill=255)

        episode = f"EP {view['episode']:04d}" if view["episode"] is not None else "EP --"
        text(28, 19, view["task"], 25, muted, 495)
        text(550, 19, episode, 25, white, 143)
        state = view["state"]
        status = _STATUS.get(state, ("상태 확인", "STATUS", muted))
        label, accent = status[0 if ko else 1], status[2]
        if not view["available"]:
            label, accent = ("상태 수신 대기" if ko else "WAITING FOR STATUS"), muted
        elif view["stale"] and state != "FAULT":
            label, accent = ("상태 갱신 지연" if ko else "STATUS UPDATE DELAYED"), (255, 197, 93)
        draw.ellipse((29, 76, 43, 90), fill=accent)
        mask.ellipse((29, 76, 43, 90), fill=255)
        text(55, 56, label, 31, accent, 390)
        if not view["record_enabled"]:
            detail = "녹화 꺼짐" if ko else "RECORDING OFF"
        elif state == "HOMING" and view["fraction"] is not None:
            fraction = max(0.0, min(1.0, view["fraction"]))
            detail = f"{fraction:.0%}"
            if view["remaining"] is not None:
                detail += f"  {max(0.0, view['remaining']):.1f}s"
        elif view["recording"]:
            detail = f"{view['elapsed'] // 60:02d}:{view['elapsed'] % 60:02d}  {view['frames']}f"
        else:
            detail = "녹화 대기" if ko else "NOT RECORDING"
        text(461, 67, detail, 21, muted, 228)

        labels = ("저장", "성공", "실패", "미분류") if ko else ("SAVED", "SUCCESS", "FAILURE", "UNLABELED")
        colors = (white, (82, 215, 163), (255, 150, 126), muted)
        for column, (name, value) in enumerate(zip(labels, view["counts"])):
            x = 26 + 172 * column
            draw.rounded_rectangle((x, 122, x + 156, 215), radius=11, fill=(23, 34, 51))
            text(x + 13, 127, name, 21, muted, 140)
            text(x + 13, 157, "--" if value is None else value, 38, colors[column], 140)

        tactile = view.get('tactile')
        notice_y = 223 if isinstance(tactile, dict) else 232
        badge = _last_saved_text(view, ko)
        text(29, notice_y, _notice_text(view, ko), 21,
             (255, 197, 93) if (view["stale"] or view["pending_labels"]
                               or (view["warnings"] and not view["error"])) else muted,
             361 if badge else 663)
        if badge:
            outcome = view["last_saved_outcome"]
            badge_color = (82, 215, 163) if outcome == "success" else (
                (255, 150, 126) if outcome == "failure" else muted)
            text(409, notice_y, badge, 21, badge_color, 283)
        if isinstance(tactile, dict):
            # Raw normal force and proximity, not contact classification.
            for side, label, y in (('left', 'L', 253), ('right', 'R', 274)):
                item = tactile.get(side) or {}
                frame = item.get('frame') or {}
                fingers = frame.get('fingers') or []
                if item.get('valid') and len(fingers) == 5:
                    forces = [(str(f['normal_raw']) + ('!' if f.get('status_flags', 0) else ''))
                              if f.get('normal_valid') else '--' for f in fingers]
                    proximity = [str(f['proximity_raw']) for f in fingers]
                    value = f"{label} F:" + '/'.join(forces) + "  P:" + '/'.join(proximity)
                    color = (255, 197, 93) if any(f.get('status_flags', 0) for f in fingers) else white
                else:
                    value = f"{label} TACTILE: --"
                    color = muted
                text(29, y, value, 18, color, 663)
        else:
            draw.line((28, 274, 692, 274), fill=(50, 67, 88), width=1)
        hint = "r 추적   s 녹화/저장   q 직전 실패   q 3초 종료" if ko else (
            "r track   s record/save   q fail last   hold q 3s exit")
        text(29, 300 if isinstance(tactile, dict) else 283, hint, 18, muted, 663)

        src, alpha_src, preview_buffer = BytesIO(), BytesIO(), BytesIO()
        rgb.save(src, format="JPEG", quality=90)
        alpha.save(alpha_src, format="PNG", compress_level=3)
        if preview:
            rgba = rgb.convert("RGBA")
            rgba.putalpha(alpha)
            rgba.save(preview_buffer, format="PNG", compress_level=3)
        return HUDFrame(src.getvalue(), alpha_src.getvalue(), preview_buffer.getvalue(), "ko" if ko else "en")


def render_hud(payload, *, stale=False, font_path=None):
    """Offline preview helper; rendering and encoding never touch robot APIs."""
    return _HUDRenderer(font_path).render(_view({"payload": _payload_copy(payload),
                                               "available": True, "stale": stale}), preview=True)


class TeleopHUD:
    def __init__(self, *, refresh_hz=2.0, stale_after_s=2.5, font_path=None,
                 element_key="brainco-teleop-hud", capacity=16384):
        if not math.isfinite(refresh_hz) or not 0 < refresh_hz <= 4:
            raise ValueError("HUD refresh_hz must be in (0, 4]")
        self._shared = SharedHUDSnapshot(capacity, stale_after_s)
        self._period = 1.0 / refresh_hz
        self._now_ns = time.monotonic_ns
        self.font_path = font_path
        self.element_key = element_key

    def publish(self, payload, now_ns=None):
        return self._shared.publish(payload, now_ns)

    def snapshot(self, now_ns=None):
        return self._shared.snapshot(now_ns)

    def _element(self, frame):
        from vuer.schemas import ImageBackground
        return ImageBackground(src=frame.src, alphaSrc=frame.alpha_src,
                               key=self.element_key, aspect=720 / 330,
                               height=0.195, distanceToCamera=0.75,
                               position=[-0.29, 0.23, 0.0], fixed=False,
                               interpolate=True, material={"depthWrite": False})

    @staticmethod
    def _connected(session):
        try:
            socket = session.vuer.ws.get(session.CURRENT_WS_ID)
            return socket is not None and not getattr(socket, "closed", False)
        except Exception:
            return True

    async def run(self, session):
        # Keep renderer/cache local to this session. A reconnect needs an
        # initial upsert even when no producer metric has changed.
        renderer, shown = None, None
        last_error_ns = None
        while self._connected(session):
            try:
                if renderer is None:
                    renderer = _HUDRenderer(self.font_path)
                view = _view(self.snapshot(self._now_ns()))
                if view != shown:
                    frame = await asyncio.to_thread(renderer.render, view)
                    if not self._connected(session):
                        return
                    session.upsert(self._element(frame), to="bgChildren")
                    shown = view
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # main_with_overlay shares this task with the pose/camera
                # session. A HUD failure must not end that session.
                now_ns = self._now_ns()
                if last_error_ns is None or now_ns - last_error_ns >= 5_000_000_000:
                    try:
                        print(f"[TeleopHUD] display update failed: {error}", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                    last_error_ns = now_ns
            await asyncio.sleep(self._period)
