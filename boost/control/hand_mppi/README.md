# Revo2 hand control integration

BOOST's BrainCo Revo2 integration is defined here; the upstream
[`mimic_retargeter_lab`](https://github.com/mimicrobotics/mimic_retargeter_lab)
repository is pinned as the `upstream/` submodule. Its current hand registry
does not include Revo2. The desktop simulator now uses BrainCo's official
Revo2 MJCF and mesh assets through the pinned `external/brainco-description`
submodule. The remaining work is a local `RobotHandModel` wrapper and
retargeter config here so the policy can produce the six Revo2 targets.

## Command and state contract

`revo2.py` is the single mapping between the upstream retargeter, BOOST, and the
existing MuJoCo simulator:

| Channel | DDS order | Commanded MJCF joint | Range |
|---|---|---|---|
| 0 | Thumb | `{side}_thumb_metacarpal_joint` | 0 to 1.57 rad |
| 1 | Thumb_aux | `{side}_thumb_proximal_joint` | 0 to 1.03 rad |
| 2 | Index | `{side}_index_proximal_joint` | 0 to 1.41 rad |
| 3 | Middle | `{side}_middle_proximal_joint` | 0 to 1.41 rad |
| 4 | Ring | `{side}_ring_proximal_joint` | 0 to 1.41 rad |
| 5 | Pinky | `{side}_pinky_proximal_joint` | 0 to 1.41 rad |

The command positions published on DDS are normalized to `[0, 1]`. Each side
uses Unitree `MotorCmds_` on `rt/brainco/{left,right}/cmd`; measured positions,
velocities, and estimated torque use `MotorStates_` on
`rt/brainco/{left,right}/state`. `BrainCoDDS` implements that transport. The
simulator uses the same constants and adapter. HOMIE remains the only writer to
`rt/lowcmd`; hand targets do not pass through PATH's 192-byte arm packet.

## Televuer input boundary

Televuer hand tracking provides 25 WebXR joint positions and orientations per
hand. The Galaxy XR launcher enables that input mode when
`PATH_TELEVUER_HAND_TRACKING=1` is set. PATH forwards the joint positions in
the existing `vr_pose_debug` ZMQ message. Schema version 1 uses the standard
25-joint WebXR order, float32 metres, and coordinate-frame id 1 for OpenXR world
after the configured basis correction. The hand sequence, monotonic timestamp,
and per-hand validity travel in the same message as the wrist target. PATH's
existing LCM packet still carries only the 14 arm joints and grip flags.

The low-level hand runner should subscribe to the same ZMQ endpoint, transform
valid landmarks into the configured retargeter frame, reject stale samples,
map retargeted Revo2 angles through `BrainCoRevo2Adapter`, then publish
normalized six-channel targets via `BrainCoDDS`. Televuer supplies hand pose
landmarks; the Mimic Revo2 retargeter remains responsible for converting them
into Revo2 joint targets.

## Next implementation sequence

1. Add a local `RobotHandModel` wrapper and Revo2 config that point to the
   official BrainCo MJCF, six commanded joints, and selected Televuer joint set.
2. Implement a runner here that subscribes to the versioned `vr_pose_debug`
   landmarks, loads the local Revo2 retargeter config, rejects
   stale or invalid frames, maps radians through `BrainCoRevo2Adapter`, and
   publishes at the configured hand update rate with `BrainCoDDS`.
3. Connect it to the local MuJoCo launcher and verify the full left/right path:
   WebXR/Televuer → PATH ZMQ → hand retargeter → BrainCo DDS → simulator state
   feedback. Keep HOMIE as the sole `rt/lowcmd` writer.

Initialize the pinned source and install the separate `hand` Pixi environment:
`git submodule update --init --recursive`, then `pixi install -e hand`. The
upstream project requires Python 3.12 and MuJoCo/MJX 3.6+, while the current PATH
simulation environment pins MuJoCo 3.5.0. Its license is CC BY-NC 4.0; retain
the submodule and attribution when using or redistributing its code.
Use the Pixi environment for all BOOST development commands; the upstream
repository's standalone environment instructions are not part of this workflow.
