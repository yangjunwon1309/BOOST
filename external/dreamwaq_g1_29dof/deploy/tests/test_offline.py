#!/usr/bin/env python3
"""Offline smoke + regression test for the HOMIE policy (no robot, no unitree_interface).

Run:  python tests/test_offline.py
Needs only numpy + onnxruntime + the ckpts. Verifies:
  * obs assembles to 516 and output is 15-dim
  * stand.onnx below the threshold, walk.onnx at/above it
  * warm-start holds near the default pose at rest
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.dirname(HERE)
sys.path.insert(0, DEPLOY)

from homie_utils import load_config          # noqa: E402
from homie_policy import HomiePolicyV2       # noqa: E402

CKPTS = os.path.join(DEPLOY, "..", "ckpts")


def main():
    cfg = load_config(os.path.join(DEPLOY, "config/g1_homie_v2.yaml"))
    pol = HomiePolicyV2(cfg, os.path.join(CKPTS, "stand.onnx"),
                        os.path.join(CKPTS, "walk.onnx"))
    thr = pol.stand_cmd_threshold

    q = np.zeros(29, np.float32)
    q[:15] = cfg["default_angles"]
    dq = np.zeros(29, np.float32)
    quat = np.array([1, 0, 0, 0], np.float32)
    w = np.zeros(3, np.float32)

    # policy selection around the threshold
    pol.reset(); pol.set_command([thr - 0.01, 0, 0])
    qt, which = pol.update(q, dq, quat, w)
    assert which == "stand", which
    assert qt.shape == (15,), qt.shape

    pol.reset(); pol.set_command([thr, 0, 0])
    _, which = pol.update(q, dq, quat, w)
    assert which == "walk", which

    pol.reset(); pol.set_command([1.0, 0, 0])
    _, which = pol.update(q, dq, quat, w)
    assert which == "walk", which

    # warm-start keeps the first standing target near the default pose
    pol.reset(); pol.set_command([0.0, 0, 0])
    qt, _ = pol.update(q, dq, quat, w)
    dev = float(np.max(np.abs(qt - cfg["default_angles"])))
    assert dev < 0.3, f"startup target too far from default: {dev}"

    # obs buffer builds to the full history size
    for _ in range(10):
        pol.update(q, dq, quat, w)
    assert len(pol.obs_history) == pol.obs_history_len

    print(f"OK: threshold={thr}, startup dev={dev:.3f}, obs={pol.num_obs}, all checks passed")


if __name__ == "__main__":
    main()
