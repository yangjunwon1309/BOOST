#!/usr/bin/env python3
"""Check the ARM64 Televuer/PATH Pixi runtime, excluding JAX/CUDA workloads."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


def main() -> int:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        raise SystemExit(
            f"Thor teleoperation environment requires ARM64; active machine is {machine}"
        )

    for module_name in ("televuer", "unitree_sdk2py", "lcm", "pinocchio", "casadi"):
        importlib.import_module(module_name)

    print(f"python={sys.version.split()[0]} arch={machine}")
    for package in ("numpy", "televuer", "unitree-sdk2py", "pin", "casadi", "lcm"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "importable/source"
        print(f"{package}={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
