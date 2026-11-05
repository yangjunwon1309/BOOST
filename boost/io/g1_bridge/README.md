# G1 arm and state adapter

This C++ Unitree SDK2 process runs on the G1 control computer. It receives
PATH's 14 arm joints in the 192-byte PDU1 `G1_PATH_UPPER_COMMAND` LCM packet,
publishes the arm targets on `rt/arm_sdk`, and relays measured state in the
461-byte GST1 `G1_ROBOT_STATE` packet. It does not publish `rt/lowcmd` or
command the physical Revo2 hands.

The hardware command chain is:

```text
Galaxy XR -> PATH arm IK -> PDU1/LCM -> this adapter -> rt/arm_sdk
                                                      |
                                                      v
                                           DreamWaQ merges arm + body
                                                      |
                                                      v
                                          sole rt/lowcmd writer -> G1
```

`G1_POLICY_ACTION` GCA1 is a separate 29-joint policy packet; it is not a
replacement for PATH's PDU1 arm packet. The schema snapshot is in
[`external/g1_ws`](../../../external/g1_ws). Do not start its
`g1_lowlevel_bridge` next to DreamWaQ: it also publishes `rt/lowcmd` in its
null-action smoke mode.

## Build on G1

Install CMake, a C++17 compiler, `pkg-config`, `liblcm-dev`, and Unitree SDK2
C++. Set `UNITREE_SDK2_ROOT` to the SDK source checkout or installed package:

```bash
UNITREE_SDK2_ROOT=/opt/unitree_robotics ./boost/io/g1_bridge/build.sh
```

## Launch

The wrapper defaults to dry-run. For the active controller pair, install the
DreamWaQ and Teleimager Python environments, then launch:

```bash
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_controller.sh
```

The launcher starts this adapter in live arm-only mode with
`--hand-backend none`, then starts the target DreamWaQ runtime as the only
`rt/lowcmd` writer. DreamWaQ begins in standing mode and follows the target
START then A enable procedure; X switches standing/walking. Revo2 uses the
separate external `brainco_hand_service` and the Thor hand-MPPI publisher.

To inspect the bridge alone without sending arm targets:

```bash
G1_NETWORK_INTERFACE=eth0 ./boost/io/g1_bridge/run.sh
```

The defaults are LCM URL `udpm://239.255.76.67:7667?ttl=1`, command channel
`G1_PATH_UPPER_COMMAND`, state channel `G1_ROBOT_STATE`, and state publication
at 100 Hz. Override them with `PATH_LCM_URL`, `G1_PATH_UPPER_CHANNEL`,
`G1_ROBOT_STATE_CHANNEL`, and `G1_LCM_STATE_RATE_HZ`.
