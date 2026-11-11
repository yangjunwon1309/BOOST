# Deployment layout

BOOST separates teleoperation, robot I/O, perception, and controller runtimes.

- `boost/teleop/`: Galaxy XR manager, PATH state machine, and teleoperation messages.
- `boost/control/arm/path_planning/`: PATH nonlinear bimanual arm IK.
- `boost/control/hand_mppi/`: Revo2 iCEM/MPOPI retargeting and BrainCo DDS mapping.
- `boost/io/g1_bridge/`: G1 arm-only adapter and GST1 state relay.
- `boost/sim/mujoco/`: G1/Revo2 scene, in-process DreamWaQ policy, DDS bridge, and surface queries.
- `external/dreamwaq_g1_29dof/`: selected DreamWaQ runtime and `new2_3_backpack_lower8cm_299999` default export.
- `external/g1_ws/`: target command, state, and camera schema references.
- `external/teleimager/`: G1 camera server, selected preset, and TIM2 protocol.
- `services/pose_service/`: Thor wrist relay, RGB/RGB-D perception, SAM3, and surface geometry.
- `configs/`: controller selection, camera relay settings, and machine-local defaults.

## Physical G1

PATH sends its 14 arm targets over PDU1/LCM. The G1 adapter publishes those
targets on `rt/arm_sdk` and relays robot state to Thor. DreamWaQ reads the arm
targets and `rt/lowstate`, merges body and arm commands, and owns the sole
hardware `rt/lowcmd` publisher. The external BrainCo hand service owns Revo2
serial I/O; Thor's hand-MPPI process publishes the separate hand DDS targets.
Teleimager runs as a single camera service on G1 and sends TIM2/ZMQ frames to
the Thor relay.

For a direct, finger-free XR arm route, `g1-arm-teleop` on Thor publishes
`rt/arm_sdk` from Televuer/G1_29 IK and DreamWaQ runs through
`run_g1_dreamwaq_body_controller.sh` on G1. This route and the PATH adapter
route are mutually exclusive because both publish `rt/arm_sdk`; its setup and
recording schema are in [`g1_xr_arm_teleop.md`](g1_xr_arm_teleop.md).

DreamWaQ and Teleimager use separate Python 3.10 environments outside the
repository, under `$HOME/.local/share/boost/external/venvs/` by default. Their
setup and launch scripts are in `scripts/`.

## Thor

`thor-teleop` isolates Televuer/PATH's NumPy 1 environment. `thor` isolates the
JAX/MJX hand sampler with NumPy 2 and CUDA 13. The processes exchange fresh
hand frames over the existing local ZMQ endpoint. These ARM64 environments are
declared separately in `pixi.toml`.

## Desktop simulation

The desktop simulation uses DDS loopback and its own LCM multicast group. It
evaluates DreamWaQ in-process and drives MuJoCo actuators; the hardware
controller and G1 camera server are not started by the sim tasks. Televuer,
PATH, and hand-MPPI dependencies use their existing Pixi environments.

For ownership and interfaces see [`control_ownership.md`](control_ownership.md).
For the selected export and hardware/simulation commands see
[`dreamwaq_integration.md`](dreamwaq_integration.md) and
[`g1_installation.md`](g1_installation.md).
