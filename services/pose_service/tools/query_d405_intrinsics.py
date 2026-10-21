#!/usr/bin/env python3
"""Read D405 factory calibration with rs-enumerate-devices, not pyrealsense2.

Run this on Thor, or use ``--ssh user@thor`` from another machine.  It invokes
the librealsense command-line utility, which is independent of the Python
binding.  The tool never starts a video stream.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rs-enumerate",
        default="rs-enumerate-devices",
        help="Path/name of the rs-enumerate-devices executable.",
    )
    parser.add_argument("--json", action="store_true", help="Print parsed intrinsics as JSON.")
    parser.add_argument("--raw", action="store_true", help="Also print unparsed calibration output.")
    parser.add_argument(
        "--ssh",
        metavar="DESTINATION",
        help="Run the same query on DESTINATION over SSH.",
    )
    return parser.parse_args()


def _run_on_ssh(destination: str, args: argparse.Namespace) -> int:
    remote_args = ["python3", "-", "--rs-enumerate", args.rs_enumerate]
    if args.json:
        remote_args.append("--json")
    if args.raw:
        remote_args.append("--raw")
    try:
        result = subprocess.run(
            ["ssh", destination, *remote_args],
            input=Path(__file__).read_text(encoding="utf-8"),
            text=True,
        )
    except FileNotFoundError:
        print("ssh executable not found.", file=sys.stderr)
        return 4
    return result.returncode


def parse_intrinsics(text: str) -> list[dict[str, object]]:
    """Extract common librealsense calibration blocks without depending on its ABI."""

    records: list[dict[str, object]] = []
    pattern = re.compile(
        r'(?:Intrinsic|Intrinsics)\s+of\s+"(?P<stream>[^"]+)"'
        r'(?P<label>[^\{\n]*)\{(?P<body>.*?)\}',
        re.IGNORECASE | re.DOTALL,
    )
    field_names = {
        "width": ("width",),
        "height": ("height",),
        "fx": ("fx", "focal length x"),
        "fy": ("fy", "focal length y"),
        "ppx": ("ppx", "principal point x"),
        "ppy": ("ppy", "principal point y"),
    }
    for match in pattern.finditer(text):
        body = match.group("body")
        record: dict[str, object] = {"stream": match.group("stream").strip()}
        resolution = re.search(r"(\d+)\s*[xX]\s*(\d+)", match.group("label"))
        if resolution:
            record["width"] = int(resolution.group(1))
            record["height"] = int(resolution.group(2))
        for output_name, aliases in field_names.items():
            for alias in aliases:
                value = re.search(
                    rf"\b{re.escape(alias)}\b\s*[:=]\s*"
                    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                    body,
                    re.IGNORECASE,
                )
                if value:
                    numeric = float(value.group(1))
                    record[output_name] = int(numeric) if output_name in {"width", "height"} else numeric
                    break
        records.append(record)
    return records


def main() -> int:
    args = parse_args()
    if args.ssh:
        return _run_on_ssh(args.ssh, args)
    executable = shutil.which(args.rs_enumerate)
    if executable is None:
        print(
            "rs-enumerate-devices is not installed on Thor. Install/build librealsense "
            "command-line tools (not pyrealsense2), then re-run. For example, install the "
            "librealsense2-utils package appropriate for Thor's Ubuntu/JetPack release.",
            file=sys.stderr,
        )
        return 3
    completed = subprocess.run(
        [executable, "-c"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    output = completed.stdout
    records = parse_intrinsics(output)
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        if records:
            for record in records:
                print(json.dumps(record, sort_keys=True))
        else:
            print("No intrinsic blocks parsed; use --raw to inspect this utility version's output.")
    if args.raw:
        print(output, end="" if output.endswith("\n") else "\n")
    if completed.returncode:
        return completed.returncode
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
