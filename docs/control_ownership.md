# Control ownership

BOOST keeps PATH's Galaxy XR arm teleoperation and IK. On G1, its 14 arm targets travel in the PATH PDU1 packet over LCM, then the G1 adapter publishes `rt/arm_sdk`. The DreamWaQ runtime reads that topic together with `rt/lowstate` and is the only hardware process that publishes `rt/lowcmd`.

| Joint/device group | Target producer | Final path |
| --- | --- | --- |
| 12 exported lower-body joints | Selected DreamWaQ export | DreamWaQ `rt/lowcmd` writer |
| Waist/spine | Export manifest action compiler and target config | DreamWaQ `rt/lowcmd` writer |
| Left/right arms, motors 15–28 | PATH nonlinear IK | `rt/arm_sdk` → DreamWaQ merge → `rt/lowcmd` |
| G1 state | Unitree DDS | `rt/lowstate` → DreamWaQ; G1 adapter also relays GST1 over LCM |
| Physical G1 cameras | Teleimager | TIM2/ZMQ → Thor relay → wrist-image LCM |
| Revo2 hands in MuJoCo | BOOST binary grip adapter or hand MPPI, one at a time | Simulated BrainCo DDS channels |

The G1 bridge publishes arm targets and relays state. It does not publish `rt/lowcmd`. `g1_ws/g1_lowlevel_bridge` is not part of the live command path: even its null-action smoke mode publishes `rt/lowcmd`, so it must not run next to DreamWaQ. The GCA1 `G1_POLICY_ACTION` packet is a separate 29-joint policy format; it does not replace PATH's PDU1 `G1_PATH_UPPER_COMMAND` packet.

```text
Galaxy XR → PATH IK → PDU1/LCM → G1 adapter → rt/arm_sdk ─┐
                                                          ├→ DreamWaQ → rt/lowcmd → G1
G1 rt/lowstate ───────────────────────────────────────────┘

Galaxy XR hand input → existing BOOST Revo2/MPOPI path → simulated Revo2

G1 Teleimager TIM2/ZMQ → Thor relay → pose service / RGB SAM3 path
```

The MuJoCo backend runs the same DreamWaQ policy and action compiler directly inside the simulator loop. It reads MuJoCo state and PATH adapter `rt/arm_sdk` targets, applies target PD gains and actuator torque limits, and never subscribes to or publishes hardware `rt/lowcmd`.

See [`dreamwaq_integration.md`](dreamwaq_integration.md) for installation, launch commands, protocol details, and selected export provenance.
