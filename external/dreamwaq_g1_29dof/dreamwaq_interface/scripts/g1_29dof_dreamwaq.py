"""Offline tools for the FlashSAC DreamWaQ G1 29-DoF export.

Physical and read-only hardware execution intentionally live in the single
sequence-coupled ``dreamwaq-g1-29dof`` entry point. Keeping DDS ownership out
of this module prevents the removed latest-value state/action path from being
accidentally revived through ``python -m``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from dreamwaq_interface.g1_29dof.calibration import (
    deployment_pd_gains,
    write_calibration_template,
)
from dreamwaq_interface.g1_29dof.command import FixedCommandSource
from dreamwaq_interface.g1_29dof.contract import DreamWaQ29Contract
from dreamwaq_interface.g1_29dof.inference import TorchScriptDreamWaQ
from dreamwaq_interface.g1_29dof.runtime import (
    DreamWaQ29Runtime,
    create_deployment_recorder,
)
from dreamwaq_interface.g1_29dof.safety import SafetyLimits, SafetyMonitor
from dreamwaq_interface.g1_29dof.trace_verify import verify_play_trace
from dreamwaq_interface.g1_29dof.transport import MockPlantTransport


REMOVED_HARDWARE_COMMANDS = frozenset(("real", "shadow"))
HARDWARE_ENTRY_POINT = "dreamwaq-g1-29dof"


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return result


def _nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative finite number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative finite number")
    return result


def _motor_gain_scale(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("motor gain scale must be in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("motor gain scale must be in [0, 1]")
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export",
        required=True,
        type=Path,
        help="Export containing manifest.json/policy.pt/cenet_norm.pt",
    )
    parser.add_argument("--device", default="cpu", help="TorchScript device (normally cpu)")


def _load_backend(
    args: argparse.Namespace,
) -> tuple[DreamWaQ29Contract, TorchScriptDreamWaQ]:
    contract = DreamWaQ29Contract.load(args.export)
    backend = TorchScriptDreamWaQ(contract, device=args.device)
    backend.warmup(10)
    return contract, backend


def _close_backend(backend: object | None) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _safety_limits(args: argparse.Namespace) -> SafetyLimits:
    if str(args.safety_profile).lower() == "sim2sim":
        return SafetyLimits.sim2sim_compatible()
    return SafetyLimits()


def _report_gains(
    mode: str,
    scale: float,
    kp: np.ndarray,
    kd: np.ndarray,
    torque: np.ndarray,
) -> None:
    effective_kp = np.asarray(kp, dtype=np.float32) * float(scale)
    effective_kd = np.asarray(kd, dtype=np.float32) * float(scale)
    print(
        "[G1-29][%s] motor_gain_scale=%.3f effective_kp_max=%.3f "
        "effective_kd_max=%.3f spine_kp=%s spine_kd=%s spine_torque_limit_nm=%s"
        % (
            mode,
            scale,
            float(np.max(effective_kp)),
            float(np.max(effective_kd)),
            "/".join("%g" % value for value in effective_kp[12:15]),
            "/".join("%g" % value for value in effective_kd[12:15]),
            "/".join("%g" % value for value in torque[12:15]),
        ),
        flush=True,
    )


def validate(args: argparse.Namespace) -> int:
    contract, backend = _load_backend(args)
    try:
        result = backend.infer(
            np.zeros(contract.observation_dim, np.float32),
            np.zeros(contract.history_dim, np.float32),
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "contract": contract.metadata(),
                    "network_shapes": {
                        "v_hat": list(result.v_hat.shape),
                        "z": list(result.z.shape),
                        "actor_input": list(result.actor_input.shape),
                        "lower_action": list(result.lower_action.shape),
                    },
                    "warm_inference_ms": result.latency_ns / 1.0e6,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        _close_backend(backend)


def calibration_template(args: argparse.Namespace) -> int:
    contract = DreamWaQ29Contract.load(args.export)
    output = write_calibration_template(
        args.output,
        contract=contract,
        overwrite=args.overwrite,
    )
    print("Wrote non-ready calibration template: %s" % output)
    return 0


def verify_trace(args: argparse.Namespace) -> int:
    contract, backend = _load_backend(args)
    try:
        result = verify_play_trace(
            args.trace,
            contract=contract,
            backend=backend,
            max_steps=args.max_steps,
            observation_atol=args.observation_atol,
            network_atol=args.network_atol,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        _close_backend(backend)


def mock(args: argparse.Namespace) -> int:
    contract, backend = _load_backend(args)
    plant = MockPlantTransport(default_q_sdk=contract.default_q_sdk, mode_machine=5)
    kp, kd = deployment_pd_gains(contract.kp_sdk, contract.kd_sdk)
    runtime = DreamWaQ29Runtime(
        contract=contract,
        state_source=plant,
        command_sink=plant,
        backend=backend,
        safety=SafetyMonitor(
            limits=_safety_limits(args),
            expected_mode_machine=5,
            torque_limit_sdk=contract.torque_limit_sdk,
        ),
        kp_sdk=kp,
        kd_sdk=kd,
        motor_gain_scale=args.motor_gain_scale,
        recorder=(
            create_deployment_recorder(
                args.trace,
                contract,
                mode="mock",
                motor_gain_scale=args.motor_gain_scale,
            )
            if args.trace
            else None
        ),
    )
    try:
        _report_gains(
            "mock",
            args.motor_gain_scale,
            kp,
            kd,
            contract.torque_limit_sdk,
        )
        steps = runtime.run(
            command_source=FixedCommandSource((args.vx, args.vy, args.yaw)),
            duration_s=args.duration,
            status_every_s=args.status_every,
        )
        print("[G1-29][mock] PASS steps=%d" % steps)
        return 0
    finally:
        runtime.close()
        _close_backend(backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser(
        "validate-export",
        help="Validate export and run a TorchScript warmup",
    )
    _common(command)
    command.set_defaults(function=validate)

    command = sub.add_parser(
        "calibration-template",
        help="Write an identity calibration template",
    )
    _common(command)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--overwrite", action="store_true")
    command.set_defaults(function=calibration_template)

    command = sub.add_parser(
        "verify-play-trace",
        help="Compare a recorded trace with standalone inference",
    )
    _common(command)
    command.add_argument("--trace", required=True, type=Path)
    command.add_argument("--max-steps", type=int, default=200)
    command.add_argument("--observation-atol", type=float, default=2.0e-5)
    command.add_argument("--network-atol", type=float, default=2.0e-4)
    command.set_defaults(function=verify_trace)

    command = sub.add_parser("mock", help="Run against a deterministic CPU plant")
    _common(command)
    command.add_argument("--duration", type=_positive_float, default=5.0)
    command.add_argument("--vx", type=float, default=0.0)
    command.add_argument("--vy", type=float, default=0.0)
    command.add_argument("--yaw", type=float, default=0.0)
    command.add_argument("--trace", type=Path, default=None)
    command.add_argument("--status-every", type=_nonnegative_float, default=0.0)
    command.add_argument(
        "--safety-profile",
        choices=("hardware", "sim2sim"),
        default="hardware",
    )
    command.add_argument("--motor-gain-scale", type=_motor_gain_scale, default=1.0)
    command.set_defaults(function=mock)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in REMOVED_HARDWARE_COMMANDS:
        shadow_flag = " --shadow" if raw_args[0] == "shadow" else ""
        print(
            "Hardware command '%s' was removed from this offline module. "
            "Use '%s%s' so state/action sequence coupling is enforced."
            % (raw_args[0], HARDWARE_ENTRY_POINT, shadow_flag),
            file=sys.stderr,
        )
        return 2
    args = build_parser().parse_args(raw_args)
    try:
        return int(args.function(args))
    except KeyboardInterrupt:
        print("Interrupted; runtime shutdown requested.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            "G1 29-DoF offline tool error: %s: %s"
            % (type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
