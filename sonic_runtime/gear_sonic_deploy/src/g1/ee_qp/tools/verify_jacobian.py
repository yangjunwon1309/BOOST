#!/usr/bin/env python3
"""Compare pytorch_kinematics wrist Jacobians against central differences."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import torch
import pytorch_kinematics as pk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--end-link", required=True)
    parser.add_argument("--root-link", default="pelvis")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    model_dir = args.mjcf.resolve().parent
    xml = args.mjcf.read_text()
    asset_names = set(re.findall(r'file=["\']([^"\']+)["\']', xml))
    meshdir_match = re.search(r'meshdir=["\']([^"\']+)["\']', xml)
    meshdir = model_dir / (meshdir_match.group(1) if meshdir_match else "")
    assets = {}
    for name in asset_names:
        path = model_dir / name
        if not path.exists():
            path = meshdir / name
        assets[name] = path.read_bytes()
    chain = pk.build_chain_from_mjcf(xml, body=args.root_link, assets=assets)
    serial = pk.SerialChain(chain, args.end_link, args.root_link).to(
        dtype=torch.float64, device="cpu"
    )
    q = torch.zeros(len(serial.get_joint_parameter_names()), dtype=torch.float64)
    jacobian = serial.jacobian(q)[0, :3]

    finite_difference = torch.zeros_like(jacobian)
    for joint in range(q.numel()):
        q_plus = q.clone(); q_plus[joint] += args.eps
        q_minus = q.clone(); q_minus[joint] -= args.eps
        p_plus = serial.forward_kinematics(q_plus).get_matrix()[0, :3, 3]
        p_minus = serial.forward_kinematics(q_minus).get_matrix()[0, :3, 3]
        finite_difference[:, joint] = (p_plus - p_minus) / (2.0 * args.eps)

    error = (jacobian - finite_difference).abs()
    print("joints:", serial.get_joint_parameter_names())
    print("max_abs_error:", error.max().item())
    print("rms_error:", error.square().mean().sqrt().item())
    if error.max().item() > 1e-5:
        raise SystemExit("Jacobian validation failed")


if __name__ == "__main__":
    main()
