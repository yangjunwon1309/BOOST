#!/usr/bin/env python3
"""Cache the public EfficientTAM Tiny-512 checkpoint."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_ti_512x512.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/models/efficienttam/efficienttam_ti_512x512.pt"))
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    output = args.output
    if output.is_file() and output.stat().st_size > 10_000_000:
        print(f"[efficienttam] already cached: {output} ({output.stat().st_size} bytes)")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    print(f"[efficienttam] downloading {args.url} -> {output}", flush=True)
    with urllib.request.urlopen(args.url, timeout=120) as source, partial.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    partial.replace(output)
    print(f"[efficienttam] cached {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
