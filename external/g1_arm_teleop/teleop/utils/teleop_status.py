"""Small operator-status snapshots; no rendering, file I/O, DDS, or commands."""

from collections.abc import Mapping
import logging
import math
import threading
import time


logger = logging.getLogger(__name__)


def _number(value, default=0.):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _summary_snapshot(summary):
    source = summary if isinstance(summary, Mapping) else {}
    result = {key: max(0, int(_number(source.get(key))))
              for key in ('total', 'success', 'failure', 'unlabeled', 'incomplete', 'unknown',
                          'pending_label_updates')}
    result.update(ready=bool(source.get('ready')), loading=bool(source.get('loading')),
                  pending_save=bool(source.get('pending_save')), last_saved=None)
    last = source.get('last_saved')
    if isinstance(last, Mapping):
        metadata = last.get('result')
        if isinstance(metadata, Mapping):
            trimmed = {key: metadata[key][:240] for key in ('label', 'reason')
                       if isinstance(metadata.get(key), str)}
            if isinstance(metadata.get('success'), bool):
                trimmed['success'] = metadata['success']
            metadata = trimmed
        else:
            metadata = None
        result['last_saved'] = dict(episode_id=last.get('episode_id'),
                                    outcome=str(last.get('outcome', 'unknown')),
                                    result=metadata)
    return result


def build_hud_status(*, task, initialized, start, stop, record_enabled, recording,
                     parking, xr_ready, has_started=False, episode_id=None, frame_count=0,
                     elapsed_s=0, summary=None, home_progress=None, error=None, warnings=None):
    """Derive display state without confusing intended pauses and XR loss."""
    summary = _summary_snapshot(summary)
    progress = home_progress if isinstance(home_progress, Mapping) else {}
    error_text = str(error)[:240] if error else ''
    warning_source = [warnings] if isinstance(warnings, str) else warnings
    warning_texts = [item[:240] for item in (warning_source or ())
                     if isinstance(item, str) and item.strip()]
    if error_text:
        state = 'FAULT'
    elif stop:
        state = 'STOPPING'
    elif parking or progress.get('active'):
        state = 'HOMING'
    elif not initialized:
        state = 'INITIALIZING'
    elif record_enabled and summary['pending_save'] and not recording:
        state = 'SAVING'
    elif start and not xr_ready:
        state = 'WAITING_XR'
    elif record_enabled and recording:
        state = 'RECORDING'
    elif start:
        state = 'TRACKING'
    elif has_started:
        state = 'PAUSED'
    else:
        state = 'READY'
    notices = {
        'INITIALIZING': '연결과 초기 자세를 준비하고 있습니다',
        'READY': 'r 추적 시작',
        'TRACKING': 's 녹화 시작' if record_enabled else '녹화 기능 꺼짐',
        'RECORDING': 's 에피소드 저장',
        'WAITING_XR': '양손을 카메라에 보여 주세요' + (' · 녹화 일시정지' if recording else ''),
        'HOMING': '홈 복귀 중 · 완료 후 r로 재개',
        'SAVING': '에피소드 저장을 마무리하고 있습니다',
        'PAUSED': 'r로 추적 재개 후 s 녹화',
        'STOPPING': '팔 복귀와 저장 정리 중',
        'FAULT': error_text,
    }
    notice = notices[state]
    if summary['pending_label_updates'] and state not in ('FAULT', 'STOPPING'):
        notice += ' · 직전 저장본 라벨 저장 중'
    return dict(
        schema=1, task=str(task)[:100], state=state, record_enabled=bool(record_enabled),
        recording=bool(recording), xr_ready=bool(xr_ready),
        episode_id=int(episode_id) if episode_id is not None else None,
        frame_count=max(0, int(_number(frame_count))), elapsed_s=max(0, int(_number(elapsed_s))),
        home_fraction=min(1., max(0., _number(progress.get('fraction')))) if progress else None,
        home_remaining_s=max(0., _number(progress.get('remaining_s'))) if progress else None,
        home_confirmed=(bool(progress.get('planned_complete') and progress.get('within_tolerance')
                             and not progress.get('aborted')) if progress else None),
        summary=summary, notice=notice, error=error_text, warnings=warning_texts,
    )


class TeleopStatusPublisher:
    """Poll convenience data on a daemon thread, independently of control."""

    def __init__(self, hud, source, frequency=2.0):
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError('HUD update frequency must be finite and positive')
        self.hud, self.source = hud, source
        self.period = 1. / frequency
        self._stop = threading.Event()
        self._thread = None
        self.last_error = None
        self._last_warning = float('-inf')

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self._stop.is_set():
            raise RuntimeError('HUD publisher is closed')
        if self.is_alive:
            return
        self._thread = threading.Thread(target=self._run, name='teleop-status', daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                payload = self.source()
                if self._stop.is_set():
                    break
                if self.hud.publish(payload) is False:
                    raise RuntimeError('HUD snapshot was not accepted')
                self.last_error = None
            except Exception as error:
                self.last_error = str(error)
                if started - self._last_warning >= 10.:
                    logger.warning('VR status update unavailable: %s', error)
                    self._last_warning = started
            self._stop.wait(max(.001, self.period - (time.monotonic() - started)))

    def close(self, timeout=1.0):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
        if self.is_alive:
            logger.warning('VR status source did not stop; control shutdown will continue')
