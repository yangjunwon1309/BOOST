import os
import copy
import cv2
import json
import datetime
import math
import numpy as np
import stat
import tempfile
import time
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import logging_mp
logger_mp = logging_mp.getLogger(__name__)


def _new_rerun_logger(**kwargs):
    # Recording to disk does not require the optional Rerun viewer.  Import it
    # only when the caller explicitly enables live visualization.
    from .rerun_visualizer import RerunLogger

    return RerunLogger(**kwargs)

class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None, frequency=30, image_size=[640, 480], rerun_log = True,
                 queue_size=256, compact_json=False, collect_stats=False, color_subdirs=None,
                 strict_io=False, enable_episode_summary=False):
        """
        image_size: [width, height]
        queue_size: bounded backlog of unwritten items; oldest is dropped when full
        compact_json: use the C JSON encoder on frame rows to reduce GIL/storage cost
        collect_stats: save measured cadence and writer backlog in recording_stats
        color_subdirs: optional color-key to relative directory mapping under colors/
        enable_episode_summary: background finalized-footer counts for a live HUD
        """
        self.color_subdirs = {}
        for key, directory in dict(color_subdirs or {}).items():
            directory = os.fspath(directory)
            normalized = os.path.normpath(directory)
            if (not directory or os.path.isabs(directory) or normalized in ('.', '..')
                    or normalized.startswith('..' + os.sep)):
                raise ValueError("color_subdirs must contain relative directories inside colors/")
            self.color_subdirs[key] = normalized
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size
        self.compact_json = compact_json
        self.collect_stats = collect_stats
        self.strict_io = bool(strict_io)

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...\n")
            self.rerun_logger = _new_rerun_logger(
                prefix="online/", IdxRangeBoundary=60, memory_limit="300MB")
            logger_mp.info("==> RerunLogger initializing ok.\n")
        
        self.item_id = -1
        self.episode_id = -1
        os.makedirs(self.task_dir, exist_ok=True)
        episode_ids = [int(name[8:]) for name in os.listdir(self.task_dir)
                       if name.startswith('episode_') and name[8:].isascii() and name[8:].isdigit()
                       and os.path.isdir(os.path.join(self.task_dir, name))]
        self.episode_id = max(episode_ids, default=-1)
        logger_mp.info(f"==> Last existing episode ID: {self.episode_id}\n")
        self.data_info()
        if self.color_subdirs:
            self.info['color_subdirs'] = self.color_subdirs.copy()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Bounded so a disk that cannot keep up costs frames instead of growing
        # RAM without limit. Absorbs short stalls; sustained drops mean the
        # storage cannot take the incoming rate.
        self.item_data_queue = Queue(maxsize=queue_size)
        self.queue_drops = 0
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.final_metadata = None
        self._episode_state_lock = Lock()
        self._result_updates = Queue()
        self._deferred_result_updates = []
        # Only files finalized by this writer can be relabeled. The offset is
        # the exact byte prefix containing info/text/data and optional stats.
        self._completed_episodes = {}
        self._accept_result_updates = True
        self._accept_frames = True
        self._closing = False
        self._closed = False
        self._close_error = None
        self._worker_failure = None
        self._active_result_path = None
        self._worker_wake = Event()
        self._abort_worker = Event()
        self._episode_summary_enabled = bool(enable_episode_summary)
        self._summary_lock = Lock()
        self._summary_records = {}
        self._summary_owned_paths = set()
        self._summary_counts = dict(success=0, failure=0, unlabeled=0, unknown=0, incomplete=0)
        self._summary_ready = False
        self._summary_last_path = None
        self._summary_snapshot = self._build_summary_snapshot()
        self._summary_thread = None
        self._reset_recording_stats()
        # Normal close explicitly flushes all work. A native filesystem call
        # that never returns must not hold interpreter shutdown indefinitely.
        self.worker_thread = Thread(target=self.process_queue, daemon=True)
        self.worker_thread.start()
        if self._episode_summary_enabled:
            self._summary_thread = Thread(target=self._scan_episode_summaries, daemon=True)
            self._summary_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        if self.strict_io and self._worker_failure is not None:
            path, detail = self._worker_failure
            raise RuntimeError(f"Episode writer failed for {path}: {detail}")
        return self.is_available

    def _build_summary_snapshot(self):
        """Publish immutable cached records; callers deepcopy outside the lock."""
        counts = self._summary_counts.copy()
        last = self._summary_records.get(self._summary_last_path)
        return dict(enabled=self._episode_summary_enabled, ready=self._summary_ready,
                    loading=self._episode_summary_enabled and not self._summary_ready,
                    total=sum(counts[key] for key in ('success', 'failure', 'unlabeled', 'unknown')),
                    **counts,
                    last_saved=(dict(episode_id=last['episode_id'], path=last['path'],
                                     outcome=last['outcome'],
                                     result=last['result'] if isinstance(last['result'], dict) else None)
                                if last is not None else None))

    def episode_summary(self):
        """Cheap HUD snapshot. Completion means a finalized footer, not a body audit.

        Historical discovery reads bounded tails on a separate daemon. Neither
        this method nor the control thread opens or parses episode files.
        """
        with self._summary_lock:
            snapshot = self._summary_snapshot
        result = copy.deepcopy(snapshot)
        result.update(pending_save=bool(self._episode_summary_enabled and self.need_save),
                      pending_label_updates=(self._result_updates.unfinished_tasks
                                             if self._episode_summary_enabled else 0))
        return result

    @staticmethod
    def _summary_outcome(has_result, result):
        if not has_result:
            return 'unlabeled'
        if not isinstance(result, dict) or type(result.get('success')) is not bool:
            return 'unknown'
        expected = 'success' if result['success'] else 'failure'
        if 'label' in result and result['label'] != expected:
            return 'unknown'
        return expected

    @staticmethod
    def _read_episode_summary(path, episode_id):
        """Parse only the root footer after the writer's column-zero data ]."""
        record = dict(path=os.path.abspath(path), episode_id=episode_id,
                      outcome='incomplete', result=None)
        try:
            with open(path, 'rb') as file:
                size = os.fstat(file.fileno()).st_size
                for limit in (64 * 1024, 256 * 1024):
                    file.seek(max(0, size - limit))
                    tail = file.read(limit)
                    marker = tail.rfind(b'\n]')
                    if marker < 0:
                        if size <= limit:
                            break
                        continue
                    footer = json.loads((b'{"data":[]' + tail[marker + 2:]).decode('utf-8'))
                    record['result'] = footer.get('result')
                    record['outcome'] = EpisodeWriter._summary_outcome('result' in footer, record['result'])
                    break
        except (OSError, UnicodeError, ValueError):
            pass
        return record

    def _install_summary_record(self, record, *, owned=False):
        if not self._episode_summary_enabled:
            return
        record = copy.deepcopy(record)
        path = record['path']
        with self._summary_lock:
            if not owned and path in self._summary_owned_paths:
                return
            if owned:
                self._summary_owned_paths.add(path)
            previous = self._summary_records.get(path)
            if previous is not None and previous['outcome'] in self._summary_counts:
                self._summary_counts[previous['outcome']] -= 1
            self._summary_records[path] = record
            if record['outcome'] in self._summary_counts:
                self._summary_counts[record['outcome']] += 1
            if record['outcome'] in ('success', 'failure', 'unlabeled', 'unknown'):
                last = self._summary_records.get(self._summary_last_path)
                if last is None or record['episode_id'] >= last['episode_id']:
                    self._summary_last_path = path
            self._summary_snapshot = self._build_summary_snapshot()

    def _completed_summary_record(self, path, result):
        if self._episode_summary_enabled:
            episode_id = int(os.path.basename(os.path.dirname(path)).split('_')[-1])
            self._install_summary_record(
                dict(path=os.path.abspath(path), episode_id=episode_id,
                     outcome=self._summary_outcome(result is not None, result), result=result), owned=True)

    def _scan_episode_summaries(self):
        try:
            with os.scandir(self.task_dir) as entries:
                paths = [(entry.path, int(entry.name[8:])) for entry in entries
                         if entry.is_dir() and entry.name.startswith('episode_')
                         and entry.name[8:].isdigit()]
            for directory, episode_id in paths:
                path = os.path.join(directory, 'data.json')
                self._install_summary_record(self._read_episode_summary(path, episode_id))
        except OSError as error:
            logger_mp.warning('Episode summary discovery failed for %s: %s', self.task_dir, error)
        finally:
            with self._summary_lock:
                self._summary_ready = True
                self._summary_snapshot = self._build_summary_snapshot()

    def _reset_recording_stats(self):
        self.written_frames = 0
        self.processing_errors = 0
        self.image_write_errors = 0
        self.max_queue_depth = 0
        self._episode_queue_drops_start = self.queue_drops
        self._writer_durations_ms = []
        self._first_action_us = None
        self._last_action_us = None
        self._action_intervals_ms = []

    def recording_stats(self):
        duration = ((self._last_action_us - self._first_action_us) / 1e6
                    if self._first_action_us is not None and self._last_action_us is not None else 0.0)
        def distribution(values):
            if not values:
                return None
            return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
                    "max": float(max(values))}
        return {
            "target_fps": self.frequency,
            "written_frames": self.written_frames,
            "attempted_frames": self.item_id + 1,
            "action_duration_s": duration,
            "observed_fps": ((len(self._action_intervals_ms)) / duration if duration > 0 else None),
            "action_interval_ms": distribution(self._action_intervals_ms),
            "intervals_over_1_5_periods": sum(dt > 1500.0 / self.frequency for dt in self._action_intervals_ms),
            "nonpositive_action_intervals": sum(dt <= 0 for dt in self._action_intervals_ms),
            "queue_drops": self.queue_drops - self._episode_queue_drops_start,
            "processing_errors": self.processing_errors,
            "image_write_errors": self.image_write_errors,
            "max_queue_depth": self.max_queue_depth,
            "writer_ms": distribution(self._writer_durations_ms),
            "compact_json": self.compact_json,
        }

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        with self._episode_state_lock:
            self._ensure_producer_open()
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.final_metadata = None
        self._reset_recording_stats()
        self.item_id = -1
        # Reserve a fresh directory atomically. Another writer may have chosen
        # the same next ID after our initial scan; never reopen its data.json.
        while True:
            self.episode_id += 1
            self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
            try:
                os.mkdir(self.episode_dir)
                break
            except FileExistsError:
                continue
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        if self._episode_summary_enabled:
            self._install_summary_record(dict(path=os.path.abspath(self.json_path), episode_id=self.episode_id,
                                               outcome='active', result=None), owned=True)
        os.makedirs(self.color_dir, exist_ok=True)
        for directory in set(self.color_subdirs.values()):
            os.makedirs(os.path.join(self.color_dir, directory), exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "x", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        self.first_item = True   # Flag to handle commas in JSON array

        if self.rerun_log:
            self.online_logger = _new_rerun_logger(
                prefix="online/", IdxRangeBoundary=60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None,
                 timestamps=None, annotation=None, telemetry=None):
        # Create the item data dictionary
        item_data = {
            'idx': None,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
        }
        if timestamps is not None:
            item_data['timestamps'] = timestamps
        if telemetry is not None:
            # Control timing and last-published command are annotations, never
            # extra channels in the state/action vectors used by training.
            item_data['telemetry'] = telemetry
        # Per-frame subtask label. Kept out of states/actions on purpose: it is an
        # annotation, not a robot channel, and putting it in either vector would
        # shift every block index downstream. Stored per frame rather than as a
        # list of ranges so that a data_editor trim -- which deep-copies frames
        # and renumbers idx -- carries it along instead of invalidating it.
        if annotation is not None:
            item_data['annotation'] = annotation
        # Serialize the short enqueue with close's producer barrier, so a
        # producer cannot append another row after close has drained the queue.
        with self._episode_state_lock:
            self._ensure_producer_open()
            self.item_id += 1
            item_data['idx'] = self.item_id
            # Never block the control loop on a disk backlog.
            try:
                self.item_data_queue.put_nowait(item_data)
            except Full:
                try:
                    self.item_data_queue.get_nowait()
                    self.item_data_queue.task_done()
                except Empty:
                    pass
                self.queue_drops += 1
                if self.queue_drops in (1, 10, 100) or self.queue_drops % 500 == 0:
                    logger_mp.warning(f"==> Writer queue full, dropped {self.queue_drops} item(s). "
                                      f"Storage cannot sustain the recording rate.")
                try:
                    self.item_data_queue.put_nowait(item_data)
                except Full:
                    pass
            if self.collect_stats:
                self.max_queue_depth = max(self.max_queue_depth, self.item_data_queue.qsize())
        self._worker_wake.set()

    def process_queue(self):
        try:
            while not self.stop_worker or not self.item_data_queue.empty():
                if self._abort_worker.is_set():
                    break
                self._worker_wake.clear()
                try:
                    item_data = self.item_data_queue.get_nowait()
                except Empty:
                    item_data = None
                if item_data is not None:
                    started = time.perf_counter_ns() if self.collect_stats else None
                    try:
                        self._process_item_data(item_data)
                    except Exception as error:
                        self.processing_errors += 1
                        logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {error}")
                        if self.strict_io:
                            raise
                    finally:
                        if started is not None:
                            self._writer_durations_ms.append((time.perf_counter_ns() - started) / 1e6)
                        self.item_data_queue.task_done()
                if self._abort_worker.is_set():
                    break
                if self.need_save and self.item_data_queue.empty():
                    self._save_episode()
                self._process_result_updates()
                if self.item_data_queue.empty() and not self.stop_worker:
                    self._worker_wake.wait(timeout=0.1)
        except BaseException as error:
            path = self._pending_path()
            with self._episode_state_lock:
                self._worker_failure = (path, f"{type(error).__name__}: {error}")
                self._accept_frames = self._accept_result_updates = False
                self.stop_worker = True
            logger_mp.error("Episode writer failed; incomplete recording %s: %s", path, error)
        finally:
            # No worker-local item will call task_done after this point. Drain
            # queued/deferred work so failure cannot leave a join counter stuck.
            self._discard_pending_work()

    def _ensure_producer_open(self):
        if self._worker_failure is not None:
            path, detail = self._worker_failure
            raise RuntimeError(f"Episode writer failed for {path}: {detail}")
        if not self._accept_frames:
            raise RuntimeError("episode writer is closing")

    def _pending_path(self):
        return self._active_result_path or getattr(self, 'json_path', self.task_dir)

    def _discard_pending_work(self):
        for _ in self._deferred_result_updates:
            self._result_updates.task_done()
        self._deferred_result_updates.clear()
        for queue in (self.item_data_queue, self._result_updates):
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break
                queue.task_done()

    def set_episode_success(self, json_path, success):
        """Queue a result correction without blocking the caller on file I/O.

        A request for the current episode waits for its footer to finish. A
        request for an older completed episode never touches the active file.
        """
        if not isinstance(success, bool):
            raise TypeError("episode success must be a bool")
        path = os.path.abspath(os.fspath(json_path))
        with self._episode_state_lock:
            self._ensure_producer_open()
            if not self._accept_result_updates:
                raise RuntimeError("episode writer is closing")
            self._result_updates.put_nowait((path, success))
        self._worker_wake.set()

    def _process_result_updates(self):
        pending, self._deferred_result_updates = self._deferred_result_updates, []
        try:
            while not self._abort_worker.is_set():
                if pending:
                    path, success = pending.pop(0)
                else:
                    try:
                        path, success = self._result_updates.get_nowait()
                    except Empty:
                        break
                completed = self._completed_episodes.get(path)
                if completed is None:
                    current = getattr(self, 'json_path', None)
                    if current is not None and path == os.path.abspath(current) and not self.is_available:
                        self._deferred_result_updates.append((path, success))
                        continue
                try:
                    self._active_result_path = path
                    if completed is None:
                        logger_mp.warning("Cannot update an episode not finalized by this writer: %s", path)
                    else:
                        self._update_episode_result(path, success, completed)
                except Exception as error:
                    logger_mp.error("Failed to update episode result %s: %s", path, error)
                finally:
                    self._active_result_path = None
                    self._result_updates.task_done()
        finally:
            self._deferred_result_updates.extend(pending)

    def _update_episode_result(self, path, success, completed):
        prefix_size, previous_result = completed
        if previous_result is not None and not isinstance(previous_result, dict):
            raise ValueError("episode result must be an object to update success")
        result = copy.deepcopy(previous_result) if previous_result is not None else {}
        result.update(success=success, label='success' if success else 'failure')
        if result == previous_result:
            return
        footer = (',\n"result": ' + json.dumps(result, ensure_ascii=False, indent=4) + '\n}').encode('utf-8')
        temporary_path = None
        try:
            with open(path, 'rb') as source:
                mode = stat.S_IMODE(os.fstat(source.fileno()).st_mode)
                with tempfile.NamedTemporaryFile(mode='wb', dir=os.path.dirname(path),
                                                 prefix='.data.json.result-', suffix='.tmp',
                                                 delete=False) as target:
                    temporary_path = target.name
                    remaining = prefix_size
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError("episode file is shorter than its saved data prefix")
                        target.write(chunk)
                        remaining -= len(chunk)
                    target.write(footer)
                    os.fchmod(target.fileno(), mode)
                    target.flush()
                    os.fsync(target.fileno())
            if self._abort_worker.is_set():
                raise RuntimeError("episode writer closed before result replacement completed")
            os.replace(temporary_path, path)
            temporary_path = None
            self._completed_episodes[path] = (prefix_size, result)
            self._completed_summary_record(path, result)
            logger_mp.info("Episode result updated: %s success=%s", path, success)
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                subdir = self.color_subdirs.get(color_key, '')
                relative_path = os.path.join('colors', subdir, color_name)
                color_path = os.path.join(self.episode_dir, relative_path)
                if isinstance(color, (bytes, bytearray, memoryview)):
                    # Already-encoded JPEG straight from the camera. Writing the
                    # original bytes avoids a decode/re-encode round trip and the
                    # generation loss that comes with it.
                    try:
                        with open(color_path, "wb") as image_file:
                            image_file.write(color)
                    except OSError as error:
                        self.image_write_errors += 1
                        logger_mp.info(f"Failed to save color image: {error}")
                        if self.strict_io:
                            raise
                elif not cv2.imwrite(color_path, color):
                    self.image_write_errors += 1
                    logger_mp.info(f"Failed to save color image.")
                    if self.strict_io:
                        raise OSError(f"Failed to save color image: {color_path}")
                item_data['colors'][color_key] = relative_path

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    self.image_write_errors += 1
                    logger_mp.info(f"Failed to save depth image.")
                    if self.strict_io:
                        raise OSError(f"Failed to save depth image: {depth_name}")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        # Encode before touching the file, so invalid values cannot leave a
        # dangling comma or partial row behind an otherwise valid prefix.
        row_json = json.dumps(item_data, ensure_ascii=False, allow_nan=not self.strict_io,
                              **({'separators': (',', ':')} if self.compact_json else {'indent': 4}))
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(row_json)
            self.first_item = False
        self.written_frames += 1
        if self.collect_stats:
            action_us = (item_data.get('timestamps') or {}).get('action_wall_time_us')
            if action_us is not None:
                if self._first_action_us is None:
                    self._first_action_us = action_us
                if self._last_action_us is not None:
                    self._action_intervals_ms.append((action_us - self._last_action_us) / 1000.0)
                self._last_action_us = action_us

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self, final_metadata=None):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self._request_save(final_metadata)

    def _request_save(self, final_metadata=None, *, from_close=False):
        with self._episode_state_lock:
            if not from_close:
                self._ensure_producer_open()
            elif self._worker_failure is not None:
                path, detail = self._worker_failure
                raise RuntimeError(f"Episode writer failed for {path}: {detail}")
            # The first request owns this footer. In particular, close() must
            # not replace its result or append another footer during a race.
            if self.is_available or self.need_save:
                return
            if final_metadata is not None:
                self.final_metadata = copy.deepcopy(final_metadata)
            self.need_save = True  # Set the save flag
        self._worker_wake.set()
        logger_mp.info(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        if self._abort_worker.is_set():
            return
        with self._episode_state_lock:
            json_path = os.path.abspath(self.json_path)
            final_metadata = copy.deepcopy(self.final_metadata)
        with open(json_path, "a", encoding="utf-8") as f:
            f.write("\n]")
            if self.collect_stats:
                stats = self.recording_stats()
                f.write(',\n"recording_stats": ')
                f.write(json.dumps(stats, ensure_ascii=False, indent=4))
                fps = stats['observed_fps']
                if fps is not None:
                    logger_mp.info(f"==> Recording rate: {fps:.3f}/{self.frequency:g} FPS, "
                                   f"written={self.written_frames}, queue_drops={stats['queue_drops']}, "
                                   f"processing_errors={self.processing_errors}, "
                                   f"image_write_errors={self.image_write_errors}")
                    if len(self._action_intervals_ms) >= 30 and fps < self.frequency * .95:
                        logger_mp.warning("==> Recording below target rate; inspect recording_stats "
                                          "for writer cost, long intervals and dropped frames.")
            # TextIO.tell() is a text cookie rather than a portable byte offset.
            # Flush before measuring the exact UTF-8 prefix copied on relabel.
            f.flush()
            prefix_size = os.fstat(f.fileno()).st_size
            if final_metadata is not None:
                # Optional, additive episode result. Existing teleop recordings
                # call save_episode() without it and retain their exact schema.
                f.write(",\n\"result\": ")
                f.write(
                    json.dumps(
                        final_metadata,
                        ensure_ascii=False,
                        indent=4,
                    )
                )
            f.write("\n}")      # Close the JSON object

        if self._abort_worker.is_set():
            return
        self._completed_summary_record(json_path, final_metadata)
        with self._episode_state_lock:
            # Publish the completed-file record before a new episode can start.
            self._completed_episodes[json_path] = (prefix_size, final_metadata)
            self.need_save = False     # Reset the save flag
            self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {json_path}.")

    def _check_close_worker(self, stage):
        if self._worker_failure is not None:
            path, detail = self._worker_failure
            raise RuntimeError(f"Episode writer failed while {stage}; incomplete recording {path}: {detail}")
        if not self.worker_thread.is_alive():
            raise RuntimeError(f"Episode writer stopped unexpectedly while {stage}: {self._pending_path()}")

    def _close_remaining(self, deadline, stage):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"EpisodeWriter close timed out while {stage}; pending recording {self._pending_path()}; "
                f"frames={self.item_data_queue.unfinished_tasks}, labels={self._result_updates.unfinished_tasks}")
        return remaining

    def _wait_close_queue(self, queue, deadline, stage):
        # Queue.join() has no timeout. Condition.wait releases the queue lock,
        # permitting the worker to finish while checking liveness and deadline.
        with queue.all_tasks_done:
            while queue.unfinished_tasks:
                self._check_close_worker(stage)
                queue.all_tasks_done.wait(timeout=min(.05, self._close_remaining(deadline, stage)))

    def close(self, timeout=10.0):
        """Flush accepted frames/results, or report failure within one deadline.

        A timeout cannot cancel an already executing native I/O call. The worker
        is daemonized and told to abort, so it cannot hold interpreter exit; no
        failed/unfinished episode is reported as successfully finalized.
        """
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("episode writer close timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        with self._episode_state_lock:
            if self._closed:
                if self._close_error is not None:
                    raise self._close_error
                return
            self._closing = True
            self._accept_frames = self._accept_result_updates = False
        try:
            self._check_close_worker('draining frames')
            self._wait_close_queue(self.item_data_queue, deadline, 'draining frames')
            self._check_close_worker('finalizing episode')
            if not self.is_available:
                self._request_save(from_close=True)
            while not self.is_available:
                self._check_close_worker('finalizing episode')
                time.sleep(min(.01, self._close_remaining(deadline, 'finalizing episode')))
            self._wait_close_queue(self._result_updates, deadline, 'flushing result labels')
            self._check_close_worker('stopping worker')
            self.stop_worker = True
            self._worker_wake.set()
            self.worker_thread.join(timeout=self._close_remaining(deadline, 'stopping worker'))
            if self.worker_thread.is_alive():
                self._close_remaining(deadline, 'stopping worker')
                raise TimeoutError(f"Episode writer did not stop: {self._pending_path()}")
        except (RuntimeError, TimeoutError) as error:
            self._close_error = error
            self._abort_worker.set()
            self.stop_worker = True
            self._worker_wake.set()
            logger_mp.error("EpisodeWriter close failed for %s: %s", self._pending_path(), error)
            raise
        finally:
            self._closed = True
