# Control ownership

The physical G1 body path uses the backpack-trained DreamWaQ runtime. PATH
continues to own Galaxy XR arm teleoperation and IK. Revo2 hands remain on
their separate BrainCo command and state path.

| Joint/device group | Target producer | Final path |
| --- | --- | --- |
| 12 exported lower-body joints | Selected DreamWaQ export | DreamWaQ `rt/lowcmd` writer |
| Waist/spine | DreamWaQ export compiler and target config | DreamWaQ `rt/lowcmd` writer |
| G1 arms, motors 15-28 | PATH nonlinear IK | PDU1/LCM -> G1 adapter -> `rt/arm_sdk` -> DreamWaQ merge |
| G1 state | Unitree DDS | `rt/lowstate` -> DreamWaQ; adapter relays GST1 over LCM |
| BrainCo Revo2 hands | Thor hand MPPI | BrainCo DDS -> external hand serial service |
| G1 cameras | Teleimager | TIM2/ZMQ -> Thor relay -> wrist RGB consumer |
| Simulated Revo2 hands | BOOST binary grip adapter or hand MPPI, one at a time | Loopback BrainCo DDS channels |

DreamWaQ is the only hardware publisher to `rt/lowcmd`. The PATH G1 adapter
publishes arm targets on `rt/arm_sdk` and measured state on LCM; it does not
publish body `rt/lowcmd` or physical hand commands. Do not run
`g1_ws/g1_lowlevel_bridge` beside DreamWaQ: even its null-action smoke mode
publishes `rt/lowcmd`. GCA1 `G1_POLICY_ACTION` is a separate 29-joint policy
packet and does not replace PATH's PDU1 `G1_PATH_UPPER_COMMAND`.

```text
Galaxy XR -> PATH IK -> PDU1/LCM -> G1 adapter -> rt/arm_sdk --+
                                                               +-> DreamWaQ -> rt/lowcmd -> G1
G1 rt/lowstate ------------------------------------------------+

Galaxy XR hand input -> existing BOOST Revo2/MPOPI -> BrainCo DDS -> hand service
G1 Teleimager TIM2/ZMQ -> Thor relay -> wrist RGB/SAM3 path
```

The MuJoCo backend evaluates the same DreamWaQ policy and action compiler
in-process. It reads simulated G1 state and PATH arm targets on `rt/arm_sdk`,
then drives local actuators with the selected target gains and torque limits.
It uses isolated loopback DDS and never subscribes to or publishes a hardware
`rt/lowcmd` command.

Fresh PATH goals pass to IK without added reference smoothing or target
projection. Existing age, finite-value, and joint-limit checks remain active.
The hand retargeter publishes fresh bounded Revo2 targets directly; stale hand
input stops publication and leaves command-loss handling to the receiver.

See [`dreamwaq_integration.md`](dreamwaq_integration.md) for export provenance,
runtime setup, protocols, and launch commands.
