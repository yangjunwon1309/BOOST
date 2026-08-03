#!/usr/bin/env python3
"""Launcher for the Thor-side remote SAM3 RGB-D Wi-Fi gateway."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.thor_gateway import main


if __name__ == "__main__":
    raise SystemExit(main())
