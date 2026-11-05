"""Offline FlashSAC Isaac-play to standalone-runtime equivalence checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import DreamWaQ29Contract
from .inference import TorchScriptDreamWaQ
from .observation import FlashSACHistoryBuffer
from .recorder import SignalTraceRecorder


def _load(root: Path, name: str, count: int | None = None) -> np.ndarray:
    value = SignalTraceRecorder.load_signal(root, name)
    return value if count is None else value[:count]


def _error(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("trace parity shape mismatch: %s != %s" % (a.shape, b.shape))
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _split_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.shape != (96,):
        raise ValueError("FlashSAC trace raw_control_frame must contain 96 values")
    policy = np.concatenate((np.zeros(3, dtype=np.float32), raw[:6], raw[9:], raw[6:9])).astype(np.float32)
    measurable = np.concatenate((raw[:6], raw[9:])).astype(np.float32)
    return policy, measurable


def verify_play_trace(
    trace_root: Any,
    *,
    contract: DreamWaQ29Contract,
    backend: TorchScriptDreamWaQ,
    max_steps: int = 200,
    observation_atol: float = 2.0e-5,
    network_atol: float = 2.0e-4,
) -> dict[str, Any]:
    """Verify 96→99/93 split, term-major H=5 history and actor output.

    The Isaac trace writer must expose the FlashSAC signals below. Keeping raw
    control frames in the trace makes the history ordering test independent of
    a duplicate implementation of robot state mapping.
    """

    root = Path(trace_root).expanduser().resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("trace_schema") != "dreamwaq.g1_29dof.flashsac_play_trace.v1":
        raise ValueError("verify-play-trace requires a FlashSAC Isaac play trace")
    total, requested = int(metadata.get("steps", 0)), int(max_steps)
    if total <= 0 or requested <= 0:
        raise ValueError("trace and max_steps must both be non-empty")
    count = min(total, requested)
    names = ("raw_control_frame", "policy_current", "cenet_history", "v_hat", "z", "actor_input", "lower_action")
    signals = {name: _load(root, name, count) for name in names}
    if any(value.shape[0] != count for value in signals.values()):
        raise ValueError("trace signal streams do not have one common step count")

    history = FlashSACHistoryBuffer()
    maxima = {"policy_current": 0.0, "cenet_history": 0.0, "v_hat": 0.0, "z": 0.0, "actor_input": 0.0, "lower_action": 0.0}
    for index in range(count):
        policy, measurable = _split_raw(signals["raw_control_frame"][index])
        rebuilt_history = history.append(measurable)
        maxima["policy_current"] = max(maxima["policy_current"], _error(policy, signals["policy_current"][index]))
        maxima["cenet_history"] = max(maxima["cenet_history"], _error(rebuilt_history, signals["cenet_history"][index]))
        inferred = backend.infer(policy, rebuilt_history)
        for name, value in (("v_hat", inferred.v_hat), ("z", inferred.z), ("actor_input", inferred.actor_input), ("lower_action", inferred.lower_action)):
            maxima[name] = max(maxima[name], _error(value, signals[name][index]))

    observation_ok = max(maxima["policy_current"], maxima["cenet_history"]) <= float(observation_atol)
    network_ok = max(maxima[name] for name in ("v_hat", "z", "actor_input", "lower_action")) <= float(network_atol)
    result = {
        "status": "PASS" if observation_ok and network_ok else "FAIL",
        "checked_steps": count,
        "observation_atol": float(observation_atol),
        "network_atol": float(network_atol),
        "max_abs_error": maxima,
        "contract_fingerprint": contract.fingerprint,
    }
    if result["status"] != "PASS":
        raise RuntimeError("FlashSAC Isaac play/deployment parity failed: %s" % json.dumps(result, sort_keys=True))
    return result


__all__ = ["verify_play_trace"]
