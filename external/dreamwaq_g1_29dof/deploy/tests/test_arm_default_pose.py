from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


_DREAMWAQ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DREAMWAQ_ROOT))

from dreamwaq_interface.g1_29dof.contract import (  # noqa: E402
    DEFAULT_Q_SDK,
    HOMIE_ARM_DEFAULT_Q_SDK,
)


def test_deployment_and_path_bridge_arm_defaults_are_zero_without_changing_policy_pose() -> None:
    config_path = (
        _DREAMWAQ_ROOT
        / "deploy"
        / "robot_control"
        / "configs"
        / "g1_homie.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    default_angles = np.asarray(config["default_angles"], dtype=np.float32)

    assert default_angles.shape == (29,)
    np.testing.assert_allclose(default_angles[:15], DEFAULT_Q_SDK[:15], atol=0.0)
    np.testing.assert_array_equal(default_angles[15:29], np.zeros(14, dtype=np.float32))
    np.testing.assert_array_equal(HOMIE_ARM_DEFAULT_Q_SDK, np.zeros(14, dtype=np.float32))
    np.testing.assert_allclose(
        DEFAULT_Q_SDK[15:29],
        [0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
         0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0],
        atol=0.0,
    )
