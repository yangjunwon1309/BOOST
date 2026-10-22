"""Simulation hand backend identifiers and the canonical Revo2 adapter."""

from boost.control.hand_mppi.revo2 import (
    REVO2_COMMAND_SUFFIXES,
    REVO2_JOINT_SUFFIXES,
    Revo2CommandAdapter,
)

DEX3_BACKEND = "dex3"
REVO2_BACKEND = "brainco_revo2"

__all__ = [
    "DEX3_BACKEND",
    "REVO2_BACKEND",
    "REVO2_COMMAND_SUFFIXES",
    "REVO2_JOINT_SUFFIXES",
    "Revo2CommandAdapter",
]
