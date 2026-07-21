#!/usr/bin/env python3
"""Fail-fast verification of the Thor SAM3 runtime without loading checkpoints."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def module_version(name: str) -> dict[str, str | None]:
    module = importlib.import_module(name)
    return {
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def main() -> int:
    import numpy as np
    import torch
    import lcm

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_check.returncode:
        raise RuntimeError(f"Python dependency conflicts:\n{pip_check.stdout}{pip_check.stderr}")

    if platform.machine() not in {"aarch64", "arm64"}:
        raise RuntimeError(f"Thor image must run as ARM64, got {platform.machine()}")
    if sys.version_info < (3, 12):
        raise RuntimeError(f"SAM3 requires Python >=3.12, got {sys.version}")
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(f"SAM3 pins NumPy <2, got {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA inside this container")

    lcm_url = "udpm://239.255.76.67:7667?ttl=1"
    lcm_handle = lcm.LCM(lcm_url)

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (11, 0):
        raise RuntimeError(f"Expected Thor compute capability (11, 0), got {capability}")

    # This executes a real kernel and catches images that report CUDA but lack sm_110 code.
    lhs = torch.randn((256, 256), device=device)
    rhs = torch.randn((256, 256), device=device)
    result = lhs @ rhs
    torch.cuda.synchronize(device)
    if not torch.isfinite(result).all().item():
        raise RuntimeError("CUDA matrix multiplication returned non-finite values")

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor

    packages = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "cv2",
        "lcm",
        "fastapi",
        "uvicorn",
        "yaml",
        "websockets",
        "huggingface_hub",
        "sam3",
    ):
        packages[name] = module_version(name)

    report = {
        "platform": {
            "machine": platform.machine(),
            "python": sys.version.replace("\n", " "),
        },
        "cuda": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": capability,
            "compiled_arches": torch.cuda.get_arch_list(),
            "kernel_test": "passed",
        },
        "sam3_api": {
            "build_sam3_image_model": str(inspect.signature(build_sam3_image_model)),
            "build_sam3_video_predictor": str(inspect.signature(build_sam3_video_predictor)),
            "Sam3Processor": str(inspect.signature(Sam3Processor)),
        },
        "checkpoint_state": {
            "hf_token_available": bool(os.getenv("HF_TOKEN")),
            "hf_cache": os.getenv("HF_HOME"),
            "torch_cache": os.getenv("TORCH_HOME"),
            "models_mount_writable": os.access(Path("/models"), os.W_OK),
        },
        "lcm": {"url": lcm_url, "handle": type(lcm_handle).__name__},
        "packages": packages,
        "pip_check": pip_check.stdout.strip(),
    }
    print(json.dumps(report, indent=2, default=str))
    print("\nPASS: Thor GPU kernel, runtime dependencies, and SAM3 API imports are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
