#!/usr/bin/env python3
"""Fail unless the active Pixi Thor environment can see JAX CUDA and SDK2."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


def main() -> int:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        raise SystemExit(f"Thor environment requires ARM64; active machine is {machine}")

    import jax
    import mujoco
    from mujoco import mjx  # noqa: F401
    import unitree_sdk2py  # noqa: F401

    try:
        devices = jax.devices("cuda")
    except RuntimeError as exc:
        raise SystemExit(f"JAX CUDA backend failed to initialize: {exc}") from exc
    if not devices:
        raise SystemExit("JAX sees no CUDA device; refusing a silent CPU-only hand runtime")

    print(f"python={sys.version.split()[0]} arch={machine}")
    print(f"jax={jax.__version__} devices={devices}")
    print(f"mujoco={mujoco.__version__} mjx=available")
    for package in ("mimic_retargeter_lab", "cyclonedds", "unitree_sdk2py"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "editable/source"
        print(f"{package}={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
