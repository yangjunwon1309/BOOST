# Deployment layout

BOOST keeps teleoperation, robot I/O, perception, and hardware-specific controller runtimes separate.

- `boost/teleop/`: Galaxy XR manager, PATH state machine, and teleop messages.
- `boost/control/arm/path_planning/`: PATH nonlinear bimanual arm IK.
- `boost/io/g1_bridge/`: G1 arm-only DDS adapter and GST1 state relay.
- `boost/sim/mujoco/`: G1/Revo2 MuJoCo scene, in-process DreamWaQ policy, DDS state bridge, and local surface-query service.
- `external/dreamwaq_g1_29dof/`: selected target DreamWaQ deployment source and the `new3_4_backpack_440000` export.
- `external/g1_ws/`: source snapshot and protocol reference. Its `g1_lowlevel_bridge` is not run with DreamWaQ because it writes `rt/lowcmd`.
- `external/teleimager/`: selected G1 camera server and TIM2 protocol source.
- `services/pose_service/`: Thor wrist-image relay, RGB/RGB-D perception, SAM3, and surface geometry.
- `configs/`: sim scene/controller selection, camera relay settings, and machine-local environment defaults.

The hardware low-level owner is the target DreamWaQ parent process. PATH's G1 bridge only publishes `rt/arm_sdk` and GST1 state; camera capture runs in Teleimager as its own service. DreamWaQ and Teleimager use separate Python 3.10 environments outside the repository under `$HOME/.local/share/boost/external/` by default. Their setup and launch scripts are in `scripts/`.

The desktop simulation remains isolated on DDS loopback and its own LCM multicast group. It evaluates DreamWaQ in-process and uses MuJoCo actuators; the hardware controller and camera server are not started by the sim task. Televuer, PATH dependencies, and the Revo2 MPPI environment remain managed by the existing Pixi desktop environments.

For command ownership, selected checkpoint hashes, camera protocol, and machine commands, see [`dreamwaq_integration.md`](dreamwaq_integration.md).
