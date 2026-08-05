from pathlib import Path
import unittest

import numpy as np

from gear_sonic.utils.mujoco_sim.hand_backends import Revo2CommandAdapter


class Revo2HandBackendTest(unittest.TestCase):
    def test_binary_grip_static_targets(self):
        adapter = Revo2CommandAdapter()
        np.testing.assert_allclose(adapter.binary_target(False), np.zeros(6))
        np.testing.assert_allclose(
            adapter.binary_target(True),
            np.asarray([0.70, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )

    def test_legacy_dex3_binary_adapter(self):
        adapter = Revo2CommandAdapter()
        self.assertFalse(adapter.legacy_dex3_is_closed([0.0] * 7))
        self.assertTrue(
            adapter.legacy_dex3_is_closed([0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0])
        )

    def test_normalized_command_expands_to_eleven_joints(self):
        adapter = Revo2CommandAdapter()
        joint_targets = adapter.normalized_to_all_joints(adapter.binary_target(True))
        self.assertEqual(joint_targets.shape, (11,))
        self.assertTrue(np.all(np.isfinite(joint_targets)))
        self.assertTrue(np.all(joint_targets >= 0.0))

    def test_generated_revo2_model_dimensions(self):
        import mujoco

        model_path = (
            Path(__file__).resolve().parents[2]
            / "gear_sonic/data/robot_model/model_data/g1/scene_43dof_revo2.xml"
        )
        model = mujoco.MjModel.from_xml_path(str(model_path))
        joint_names = [model.joint(i).name for i in range(model.njnt)]
        actuator_names = [model.actuator(i).name for i in range(model.nu)]

        for side in ("left", "right"):
            hand_joints = [
                name
                for name in joint_names
                if name
                and name.startswith(f"{side}_")
                and any(
                    finger in name for finger in ("thumb", "index", "middle", "ring", "pinky")
                )
            ]
            hand_actuators = [
                name
                for name in actuator_names
                if name and name.startswith(f"{side}_revo2_")
            ]
            self.assertEqual(len(hand_joints), 11)
            self.assertEqual(len(hand_actuators), 6)


if __name__ == "__main__":
    unittest.main()
