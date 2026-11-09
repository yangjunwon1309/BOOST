# Upstream provenance

The arm controller, G1 IK, Televuer, episode writer, state probes, and XR
teleoperation recorder were copied from the local `url_g1_deploy` checkout at
commit `3d37aa43f88964fc4255857c840dd33dd08e6eda`.

The entry point is an arm-only adaptation of `teleop/teleop/teleop_ikea.py`:
Dex1 controller initialization, finger inputs, gripper probes, and finger
recording fields were removed. `left_ee/right_ee` retain the upstream wrist
target pose and optional FK pose schema, with finger qpos/qvel/torque empty.
Small runtime-path edits point IK at BOOST's pinned URDF/cache and make Rerun
and plotting imports optional when the headless recorder is used. The copied
Televuer license is in `televuer/LICENSE`; the upstream G1 teleoperation asset
license is in `LICENSE.xr_teleoperate`.
