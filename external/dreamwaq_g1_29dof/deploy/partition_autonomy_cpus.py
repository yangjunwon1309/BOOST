#!/usr/bin/env python3
"""Keep named navigation/camera/editor workloads off DreamWaQ timing CPUs.

System Python only. No DDS imports, signals to robot nodes, or motor writes.
The watchdog also covers nodes started later through the existing ROS aliases.
"""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import threading


ROLE_CPUS = {'camera': {0, 1}, 'slam': {2, 3}, 'background': {0, 1, 2, 3}}


def read_processes(proc=Path('/proc')):
    processes = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = tuple(value.decode('utf-8', 'replace')
                         for value in (entry / 'cmdline').read_bytes().split(b'\0') if value)
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if args:
                processes[int(entry.name)] = {'args': args, 'ppid': int(stat[1]),
                                              'start': int(stat[19])}
        except (OSError, ValueError, IndexError):
            continue
    return processes


def process_roles(processes):
    cameras = {pid for pid, p in processes.items() if any(
        arg in ('teleimager.image_server', 'teleimager.server_cli')
        or Path(arg).name == 'teleimager-server' for arg in p['args'])}
    while True:
        children = {pid for pid, p in processes.items() if p['ppid'] in cameras}
        expanded = cameras | children
        if expanded == cameras:
            break
        cameras = expanded
    roles = {pid: 'camera' for pid in cameras}
    for pid, p in processes.items():
        args = p['args']
        if pid in roles:
            continue
        if any(a.endswith('/url_lidar_slam/lio_node') for a in args):
            roles[pid] = 'slam'
        elif any(a.startswith('/home/unitree/.vscode-server/') for a in args):
            roles[pid] = 'background'
        elif any(a.startswith('/opt/ros/noetic/')
                 or a.startswith('url_navigation.interface.ros.')
                 or '/nature_ws/devel/lib/' in a
                 or Path(a).name in ('ros_autonomy_bridge.py',
                                     'dreamstep_odom_grav_to_g1.py',
                                     'dreamstep_odom_grav_to_g1_tilt.py',
                                     'goal_manager.py') for a in args):
            roles[pid] = 'background'
    return roles


def real_control_pids(processes):
    return [pid for pid, p in processes.items()
            if '--shadow' not in p['args'] and any(
                Path(a).name in ('dreamwaq-g1-29dof', 'run_dreamwaq.py') for a in p['args'])]


def partition(apply=False, proc=Path('/proc')):
    processes = read_processes(proc)
    roles = process_roles(processes)
    errors = []
    for pid, role in sorted(roles.items()):
        desired = ROLE_CPUS[role]
        try:
            tasks = list((proc / str(pid) / 'task').iterdir())
        except OSError:
            continue
        changed = 0
        observed = set()
        for task in tasks:
            try:
                tid = int(task.name)
                current = os.sched_getaffinity(tid)
                observed.update(current)
                if apply and current != desired:
                    # Reject PID reuse between discovery and mutation.
                    stat = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(stat[19]) != processes[pid]['start']:
                        break
                    os.sched_setaffinity(tid, desired)
                    changed += 1
            except ProcessLookupError:
                continue
            except (OSError, ValueError, IndexError) as exc:
                errors.append('pid=%d tid=%s: %s' % (pid, task.name, exc))
        if changed or not apply:
            print('[cpu-partition] pid=%d role=%s tasks=%d mask=%s changed=%d' %
                  (pid, role, len(tasks), ','.join(map(str, sorted(desired if apply else observed))), changed),
                  flush=True)
    for error in errors:
        print('[cpu-partition] ERROR ' + error, flush=True)
    return not errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='partition current matching processes once')
    mode.add_argument('--watch', action='store_true', help='also partition future matching processes')
    mode.add_argument('--check-idle', action='store_true', help='reject timing probes while real control is running')
    args = parser.parse_args(argv)
    if not set(range(8)).issubset(set(range(os.cpu_count() or 0))):
        parser.error('this G1 CPU profile requires CPUs 0-7')
    if args.check_idle:
        active = real_control_pids(read_processes())
        if active:
            parser.error('stop real DreamWaQ before a shadow timing probe; active PIDs: %s' % active)
        return 0
    if not args.watch:
        return 0 if partition(apply=args.apply) else 1
    os.sched_setaffinity(0, ROLE_CPUS['background'])
    cache = Path.home() / '.cache' / 'dreamwaq-autonomy-cpus'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('[cpu-partition] watchdog already running', flush=True)
            return 0
        (cache / 'watch.pid').write_text(str(os.getpid()) + '\n')
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        print('[cpu-partition] camera=0-1 slam=2-3 background=0-3; watchdog started', flush=True)
        while not stop.is_set():
            partition(apply=True)
            stop.wait(2.0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
