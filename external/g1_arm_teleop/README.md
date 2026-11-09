# BOOST G1 arm teleoperation

This package adapts the G1_29 arm IK and recording path from `url_g1_deploy`
commit `3d37aa43f88964fc4255857c840dd33dd08e6eda`. The arm-only entry point
does not import or instantiate any hand/gripper controller. It preserves the
verified XR wrist-target, IK, waist-request, image, and episode-recording code.

`teleop/robot_control/robot_arm.py` publishes `rt/arm_sdk` in motion mode;
DreamWaQ reads those arm targets and remains the sole `rt/lowcmd` writer. The
vendored Televuer package is the revision used by the verified source runtime.
The G1 model cache is used for IK; if it must be rebuilt, the implementation
uses BOOST's `assets/robot/g1_body29_hand14.urdf` and meshes.

See [the G1 XR arm teleop guide](../../docs/g1_xr_arm_teleop.md) for launch,
topic ownership, recording layout, and keyboard controls.
