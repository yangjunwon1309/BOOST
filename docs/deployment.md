# Deployment layout

<<<<<<< HEAD
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
=======
The real G1 teleoperation target keeps fast operator and optimization work on
Thor and keeps whole-body stabilization local to the robot.

- `boost/teleop/` and `boost/control/arm/`: Televuer/PATH input handling and
  G1 arm IK on Thor.
- `boost/control/hand_mppi/`: JAX/MJX sampling retargeter and Revo2 DDS adapter
  on Thor.
- `boost/io/g1_bridge/`: C++ Unitree SDK2 arm bridge on G1. It ends at
  `rt/arm_sdk` and publishes robot state for Thor.
- `boost/io/revo2_hand_service/`: contract for Unitree's external
  `brainco_hand_service`, which owns the Revo2 USB-serial devices on G1.
- `external/homie_g1_29dof/`: local HOMIE/WBC deployment. It is the only
  `rt/lowcmd` writer and merges body policy with the 14 G1 arm targets.
- `boost/sim/`: local MuJoCo/HOMIE/Revo2 test stack; it is isolated from the
  physical robot by its loopback DDS and LCM configuration.

## Real G1 command boundary

Thor sends arm-only LCM packets to the SDK2 bridge on G1. The bridge maps the
14 arm joints to `rt/arm_sdk`; HOMIE consumes them while it controls the legs
and waist and publishes the final 29-DoF body command to `rt/lowcmd`.

Revo2 is not part of that 29-motor body command. Thor's hand runner publishes
the two six-joint Revo2 targets directly to the BrainCo DDS command topics. The
G1 `brainco_hand_service` converts those topics to USB-serial commands and
publishes measured hand state back to Thor. Both DDS endpoints must use the
same domain and a network interface that can exchange traffic.

The desktop CUDA 12 hand environment remains available for simulation and
development. `pixi.toml` defines a separate `thor` ARM64 environment using JAX
CUDA 13 for JetPack 7. The lock can be generated on the desktop; install that
environment on Thor itself so Pixi selects ARM64 packages and the device's
runtime libraries. Bring-up order, commands, and preflight checks are in
[`g1_installation.md`](g1_installation.md).

## Current limits

The PATH arm bridge and Revo2 hand service use independent command paths and
independent watchdogs. A single HOMIE process does not command the six external
Revo2 actuators. The system still needs an on-robot hardware preflight for DDS
interface selection, service behavior, wrist attachment calibration, and
command loss behavior before physical teleoperation.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13
