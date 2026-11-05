"""Keep camera processes and every existing thread on the camera CPUs."""

import argparse
import os
from pathlib import Path
import sys


DEFAULT_CAMERA_CPUS = (0, 1)


def parse_camera_cpus(value):
    cpus = set()
    try:
        for part in value.split(','):
            if '-' in part:
                first, last = (int(item) for item in part.split('-', 1))
                if first < 0 or last < first:
                    raise ValueError
                cpus.update(range(first, last + 1))
            else:
                cpu = int(part)
                if cpu < 0:
                    raise ValueError
                cpus.add(cpu)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("camera CPUs must look like '0,1' or '0-1'") from None
    if not cpus:
        raise argparse.ArgumentTypeError('camera CPU set cannot be empty')
    return tuple(sorted(cpus))


def _process_identity(path):
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[1]), int(fields[19])  # parent PID, process start time


def _camera_descendants(root_pid, proc):
    root_path = proc / str(root_pid)
    owner = root_path.stat().st_uid
    processes = {}
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid == owner:
                processes[int(path.name)] = _process_identity(path)
        except (OSError, ValueError, IndexError):
            continue
    if root_pid not in processes:
        raise ProcessLookupError(f'Camera process {root_pid} exited')
    selected = {root_pid}
    while True:
        expanded = selected | {pid for pid, (parent, _) in processes.items() if parent in selected}
        if expanded == selected:
            return {pid: processes[pid][1] for pid in selected}
        selected = expanded


def pin_camera_tree(cpus=DEFAULT_CAMERA_CPUS, *, root_pid=None, proc=Path('/proc')):
    """Pin the parent first, then all existing threads and descendants.

    Linux only changes the addressed thread's mask. New threads and child
    processes inherit their creator's mask; traversing the tree also repairs
    threads or workers created before this function was called.
    """
    target = set(cpus)
    if not target or any(cpu < 0 or cpu >= (os.cpu_count() or 1) for cpu in target):
        raise ValueError(f'Unavailable camera CPU set: {sorted(target)}')
    if not hasattr(os, 'sched_setaffinity'):
        raise RuntimeError('Camera CPU affinity requires Linux; use --no-affinity to opt out')
    root_pid = os.getpid() if root_pid is None else int(root_pid)
    pinned = {}
    # A second pass covers threads/children born while the first tree was read.
    for _ in range(2):
        selected = _camera_descendants(root_pid, proc)
        for pid in [root_pid] + sorted(set(selected) - {root_pid}):
            path = proc / str(pid)
            try:
                tasks = sorted(int(task.name) for task in (path / 'task').iterdir())
            except FileNotFoundError:
                continue
            # Set the process leader first so subsequently created work inherits it.
            for tid in [pid] + [task for task in tasks if task != pid]:
                try:
                    if _process_identity(path)[1] != selected[pid]:
                        break  # PID was reused; do not touch the new process.
                    os.sched_setaffinity(tid, target)
                    if os.sched_getaffinity(tid) != target:
                        raise RuntimeError(f'Camera thread {tid} did not accept CPU mask {sorted(target)}')
                    pinned.setdefault(pid, set()).add(tid)
                except (ProcessLookupError, FileNotFoundError):
                    continue
    return pinned


def bootstrap_camera_affinity(argv=None):
    """Apply the CLI default before imports can start native threads/workers."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if '--help' in argv or '-h' in argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--no-affinity', action='store_false', dest='affinity')
    parser.add_argument('--camera-cpus', type=parse_camera_cpus, default=DEFAULT_CAMERA_CPUS)
    args, _ = parser.parse_known_args(argv)
    if args.affinity:
        try:
            pin_camera_tree(args.camera_cpus)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(f'Cannot isolate camera CPUs: {exc}')
