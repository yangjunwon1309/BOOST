# Deployment layout

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
