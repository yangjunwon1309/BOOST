#!/usr/bin/env python3
"""Fail-fast check for the isolated RTX 4060 remote SAM3 image."""

from __future__ import annotations

import inspect
import json
import platform
import sys


def main() -> int:
    if sys.version_info < (3, 12):
        raise RuntimeError(f"remote SAM3 requires Python >=3.12, got {sys.version}")

    import torch

    if torch.__version__.split("+", 1)[0] != "2.7.1":
        raise RuntimeError(f"expected PyTorch 2.7.1, got {torch.__version__}")
    if torch.version.cuda != "12.6":
        raise RuntimeError(f"expected PyTorch cu126, got CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with --gpus all")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 9):
        raise RuntimeError(f"expected RTX 4060 compute capability (8, 9), got {capability}")

    # Execute and synchronize a real BF16 CUDA kernel before importing SAM3.
    # This distinguishes a usable CUDA runtime from NVML-only/nvidia-smi access.
    lhs = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
    rhs = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
    result = lhs @ rhs
    torch.cuda.synchronize(device)
    if not torch.isfinite(result).all().item():
        raise RuntimeError("CUDA BF16 matrix multiplication returned non-finite values")

    import cv2
    import einops
    import lcm
    import msgpack
    import numpy as np
    import pycocotools
    import scipy
    import torchvision
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "compiled_arches": torch.cuda.get_arch_list(),
        "kernel_test": "BF16 matrix multiplication passed",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "einops": einops.__version__,
        "lcm": getattr(lcm, "__version__", "available"),
        "msgpack": msgpack.__version__,
        "pycocotools": str(pycocotools.__file__),
        "sam3": str(sam3.__file__),
        "sam3_builder": str(inspect.signature(build_sam3_image_model)),
        "sam3_processor": str(inspect.signature(Sam3Processor)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS: isolated remote SAM3 imports and RTX CUDA runtime are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
