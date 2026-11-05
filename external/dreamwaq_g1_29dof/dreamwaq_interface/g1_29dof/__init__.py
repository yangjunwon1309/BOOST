"""Unitree G1 29-DoF DreamWaQ sim-to-real runtime.

The package is dependency-light at import time.  PyTorch and Unitree SDK2 are
loaded only by the inference and physical-transport boundaries respectively.
"""

from .contract import DreamWaQ29Contract
from .runtime import DreamWaQ29Runtime

__all__ = ["DreamWaQ29Contract", "DreamWaQ29Runtime"]
