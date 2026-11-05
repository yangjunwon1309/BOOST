"""Small Linux CPU-affinity helpers for DreamWaQ's timing-sensitive threads."""

from __future__ import annotations

import os


def format_cpu_set(cpus) -> str:
    return "[" + ",".join(str(cpu) for cpu in sorted(cpus)) + "]"


def pin_current_thread(cpu: int | None) -> list[int] | None:
    """Pin only the calling Linux thread and return its effective CPU mask."""

    if cpu is None:
        try:
            return sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return None
    cpu = int(cpu)
    os.sched_setaffinity(0, {cpu})
    return sorted(os.sched_getaffinity(0))
