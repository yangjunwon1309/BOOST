#!/usr/bin/env python3
"""Download the public EfficientViT-SAM-L0 checkpoint into the mounted models volume."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_web_node.efficientvit_backend import DEFAULT_WEIGHT_URL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/models/efficientvit_sam/efficientvit_sam_l0.pt"))
    parser.add_argument("--url", default=DEFAULT_WEIGHT_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.is_file() and not args.force:
        print(f"[efficientvit] already cached: {output} ({output.stat().st_size} bytes)")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    print(f"[efficientvit] downloading {args.url} -> {output}", flush=True)
    with urllib.request.urlopen(args.url) as response, temporary.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    temporary.replace(output)
    print(f"[efficientvit] cached {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
