#!/usr/bin/env python3
"""Verify the exact Pixi replacement for the temporary v6 test environment."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys


EXPECTED = {
    "annotated-doc": "0.0.5",
    "annotated-types": "0.8.0",
    "anyio": "4.14.2",
    "click": "8.4.2",
    "fastapi": "0.141.1",
    "h11": "0.16.0",
    "idna": "3.18",
    "iniconfig": "2.3.0",
    "lcm": "1.5.2",
    "msgpack": "1.2.1",
    "numpy": "1.26.4",
    "opencv-python-headless": "4.11.0.86",
    "packaging": "26.3",
    "pillow": "10.2.0",
    "pluggy": "1.6.0",
    "pydantic": "2.13.4",
    "pydantic-core": "2.46.4",
    "pygments": "2.20.0",
    "pytest": "9.1.1",
    "pyyaml": "6.0.1",
    "pyzmq": "27.1.0",
    "ruff": "0.16.3",
    "scipy": "1.17.1",
    "starlette": "1.6.0",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.4",
    "uvicorn": "0.52.3",
}


def main() -> int:
    actual = {name: importlib.metadata.version(name) for name in EXPECTED}
    problems = [
        f"{name}: expected {EXPECTED[name]}, found {actual[name]}"
        for name in EXPECTED
        if actual[name] != EXPECTED[name]
    ]
    if sys.version_info[:3] != (3, 12, 3):
        problems.append(f"python: expected 3.12.3, found {platform.python_version()}")

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "executable": sys.executable,
                "packages": actual,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if problems:
        print("FAIL: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("PASS: Pixi host environment matches the temporary v6 environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
