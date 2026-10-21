#!/usr/bin/env python3
"""Compare raw XRT and raw TeleVuer OpenXR poses while devices are still.

Run this before using a non-identity --basis-correction.  A consistent 90/180
degree offset points to a basis mismatch or a transform that was already
applied upstream; a reflection is rejected by the source adapter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Permit direct execution without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boost.teleop.path_streaming.xr_source_v15 import ArrivalMonitor, TeleVuerSource, XrtSource, parse_basis_correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--televuer-root", type=Path, default=None)
    parser.add_argument(
        "--televuer-cert", type=Path,
        default=Path("~/.config/xr_teleoperate/cert.pem").expanduser(),
    )
    parser.add_argument(
        "--televuer-key", type=Path,
        default=Path("~/.config/xr_teleoperate/key.pem").expanduser(),
    )
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--basis-correction", type=float, nargs=16, default=None)
    return parser.parse_args()


def _rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(second, dtype=np.float64)[:3, :3] @ np.asarray(first, dtype=np.float64)[:3, :3].T
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main() -> int:
    args = parse_args()
    xrt = XrtSource()
    televuer = TeleVuerSource(
        cert_file=args.televuer_cert,
        key_file=args.televuer_key,
        televuer_root=args.televuer_root,
        basis_correction=parse_basis_correction(args.basis_correction),
    )
    xrt.start()
    televuer.start()
    xrt_latest = tv_latest = None
    xrt_arrivals = ArrivalMonitor()
    tv_arrivals = ArrivalMonitor()
    period = 1.0 / max(0.5, float(args.hz))
    deadline = time.monotonic() + max(0.0, float(args.duration_sec))
    next_report = time.monotonic() + 5.0
    try:
        while time.monotonic() < deadline:
            xrt_sample = xrt.poll()
            tv_sample = televuer.poll()
            if xrt_sample is not None:
                xrt_latest = xrt_sample
                xrt_arrivals.record(xrt_sample.arrival_monotonic)
            if tv_sample is not None:
                tv_latest = tv_sample
                tv_arrivals.record(tv_sample.arrival_monotonic)
            if xrt_latest is not None and tv_latest is not None:
                for name in ("head", "left", "right"):
                    first = getattr(xrt_latest, name)
                    second = getattr(tv_latest, name)
                    delta_p = second[:3, 3] - first[:3, 3]
                    delta_r = _rotation_delta_deg(first, second)
                    print(
                        f"{name:5s} dp={np.round(delta_p, 4)} m  dR={delta_r:6.2f} deg",
                        flush=True,
                    )
                print("---", flush=True)
            if time.monotonic() >= next_report:
                print(
                    f"arrival XRT={xrt_arrivals.report()} TeleVuer={tv_arrivals.report()}",
                    flush=True,
                )
                next_report += 5.0
            time.sleep(period)
    except KeyboardInterrupt:
        return 0
    finally:
        televuer.stop()
        xrt.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
