#!/usr/bin/env python3
"""Download the official COCO-pretrained YOLO11n-seg checkpoint into /models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/models/yolo11/yolo11n-seg.pt"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 1_000_000:
        print(f"[PASS] cached YOLO11n-seg: {output} ({output.stat().st_size} bytes)")
        return 0
    from ultralytics import YOLO

    previous = Path.cwd()
    try:
        os.chdir(output.parent)
        YOLO("yolo11n-seg.pt")  # Official Ultralytics asset download/cache path.
    finally:
        os.chdir(previous)
    downloaded = output.parent / "yolo11n-seg.pt"
    if downloaded != output:
        downloaded.replace(output)
    if not output.is_file() or output.stat().st_size <= 1_000_000:
        raise RuntimeError(f"YOLO11n-seg download did not produce {output}")
    print(f"[PASS] cached YOLO11n-seg: {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
