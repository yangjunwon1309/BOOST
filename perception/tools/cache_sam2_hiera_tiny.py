#!/usr/bin/env python3
"""Cache the public SAM2.1 Hiera-Tiny checkpoint in the mounted model volume."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_tiny.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/models/sam2/sam2.1_hiera_tiny.pt"),
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.is_file() and output.stat().st_size > 100_000_000:
        print(f"[sam2] already cached: {output} ({output.stat().st_size} bytes)")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    print(f"[sam2] downloading {args.url} -> {output}", flush=True)
    with urllib.request.urlopen(args.url, timeout=120) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(output)
    print(f"[sam2] cached {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
