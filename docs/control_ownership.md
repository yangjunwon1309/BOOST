# Control ownership

PATH-G1 owns arm task control and produces arm-only targets. On G1, the PATH
bridge publishes those targets to `rt/arm_sdk`. The selected local
`homie_g1_29dof` deployment reads that topic, combines the arms with its
leg/waist policy output, and is the sole publisher to `rt/lowcmd`.

| Joint/device group | Target producer | Final SDK path |
| --- | --- | --- |
| Legs, motors 0-11 | HOMIE policy | HOMIE merged `rt/lowcmd` |
| Waist, motors 12-14 | HOMIE policy | HOMIE merged `rt/lowcmd` |
| Left/right arms, motors 15-28 | PATH nonlinear IK | `rt/arm_sdk` into HOMIE, then merged `rt/lowcmd` |
| Dex1 hands | PATH grip bits over LCM | G1 bridge publishes `rt/dex1/{left,right}/cmd` |
| Robot state | G1 bridge | Reads `rt/lowstate`, publishes `G1_ROBOT_STATE` |
| Camera masks, surface geometry, belief | Thor/remote SAM3 pipeline | Target and belief messages only |

The G1 bridge publishes arm targets on `rt/arm_sdk` and hand targets on the
separate Dex1 DDS topics. It does not publish `rt/lowcmd`. The HOMIE process
owns the final full-body `rt/lowcmd` writer and must be configured with
`arm_source: "arm_sdk"`.

```text
PICO / SAM3 belief target
          |
          v
PATH state machine + nonlinear arm IK (Thor)
          |
          v
G1_PATH_UPPER_COMMAND over LCM
          |
          v
G1 bridge: arm slots 15-28 -> rt/arm_sdk
          |
          v
HOMIE reads rt/arm_sdk and combines 14 arm + 15 leg/waist targets
          |
          v
HOMIE sole rt/lowcmd writer

PATH grip bits -----------------------> G1 bridge -> rt/dex1/*/cmd
```

Use the local `homie_g1_29dof` deployment for this interface. Do not run the
upstream HOMIE `g1_control` lowcmd writer at the same time: it accepts a
different LCM arm packet and also publishes `rt/lowcmd`. See
[`homie_integration.md`](homie_integration.md).
