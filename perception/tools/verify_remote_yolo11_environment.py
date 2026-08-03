#!/usr/bin/env python3
"""Verify the remote x86/Titan XP YOLO11 PyTorch environment."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    import cv2
    import lcm
    import msgpack
    import numpy as np
    import torch
    import ultralytics

    print(f"python packages: numpy={np.__version__} opencv={cv2.__version__}")
    print(f"lcm={lcm.__file__} msgpack={msgpack.__version__}")
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    print(f"ultralytics={ultralytics.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; start the container with NVIDIA GPU access")
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    print(
        f"cuda_device={device_index} "
        f"name={device_name!r} "
        f"capability=sm_{capability[0]}{capability[1]}"
    )
    if capability < (6, 1):
        raise RuntimeError("YOLO11 remote worker expects Titan XP/Pascal sm_61 or newer")
    checkpoint = Path("/models/yolo11/yolo11n-seg.pt")
    if checkpoint.is_file():
        print(f"checkpoint={checkpoint} bytes={checkpoint.stat().st_size}")
    else:
        print(f"checkpoint missing: run python tools/cache_yolo11n_seg.py --output {checkpoint}")
    print("[PASS] remote YOLO11 environment imports and sees CUDA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
