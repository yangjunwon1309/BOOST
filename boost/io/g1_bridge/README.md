# G1 arm and state adapter

<<<<<<< HEAD
This adapter runs on the G1 control computer. It receives PATH's 14 arm joints in the 192-byte PDU1 `G1_PATH_UPPER_COMMAND` LCM packet, maps them to `rt/arm_sdk`, and relays measured G1 state as the existing 461-byte GST1 `G1_ROBOT_STATE` packet. It does not publish `rt/lowcmd`.

The active hardware chain is:

```text
PATH IK → G1_PATH_UPPER_COMMAND / PDU1 → this adapter → rt/arm_sdk
                                                          ↓
                                              DreamWaQ merges arm + body
                                                          ↓
                                                  sole rt/lowcmd writer
```

The source schema snapshots from url_g1_deploy are in [`external/g1_ws`](../../../external/g1_ws). Its GCA1 `G1_POLICY_ACTION` is a separate 29-joint policy packet and is not substituted for PDU1. The target `g1_ws/g1_lowlevel_bridge` is not run with DreamWaQ: it publishes `rt/lowcmd`, including in its null-action smoke mode.

This BOOST adapter keeps its arm-only behavior and existing packet endian/layout. It publishes PATH's arm slots and Unitree's arm enable marker on `rt/arm_sdk`; DreamWaQ reads that arm topic and `rt/lowstate`. See [`docs/control_ownership.md`](../../../docs/control_ownership.md).

## Build on G1

Install CMake, a C++17 compiler, `pkg-config`, `liblcm-dev`, and Unitree SDK2 C++. Then build:
=======
This C++ Unitree SDK2 process runs on the G1 control computer. It receives
PATH's 14 arm joints over LCM, publishes them on `rt/arm_sdk`, and relays
measured G1 state to Thor for arm IK. It does not publish `rt/lowcmd` or hand
commands.

PATH sends `q`, `dq`, and `tau_ff` for arm motor slots 15-28 on
`G1_PATH_UPPER_COMMAND`. The bridge supplies the arm gains and SDK enable
marker on `rt/arm_sdk`. HOMIE consumes those fields, combines the arm targets
with its leg/waist policy output, and remains the sole writer to `rt/lowcmd`.
See [`../../../docs/control_ownership.md`](../../../docs/control_ownership.md).

BrainCo Revo2 is handled by Unitree's external
[`brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service)
and Thor's Revo2 hand-MPPI DDS publisher. The hand service connects to the
USB-serial interfaces and publishes measured state. This arm bridge has no
hand backend.

## Build on G1

Install CMake, a C++17 compiler, `pkg-config`, `liblcm-dev`, and Unitree SDK2.
The SDK can be a source checkout or installed SDK package:
>>>>>>> 228019f83417f222bc20e79b837865e645492e13

```bash
UNITREE_SDK2_ROOT=/opt/unitree_robotics ./boost/io/g1_bridge/build.sh
```

## Launch

<<<<<<< HEAD
The direct wrapper defaults to dry-run and does not publish arm targets. For the active DreamWaQ path, install the Python runtime and launch the single-owner pair with [`scripts/run_g1_dreamwaq_controller.sh`](../../../scripts/run_g1_dreamwaq_controller.sh):

```bash
BOOST_ALLOW_HARDWARE_CONTROL=1 G1_NETWORK_INTERFACE=eth0 \
  bash scripts/run_g1_dreamwaq_controller.sh
```

That launcher starts this adapter in live arm-only mode with `--hand-backend none`, then starts the target DreamWaQ runtime. DreamWaQ waits for the target START then A procedure before it takes body control. X switches standing/walking. Do not start another process that publishes `rt/lowcmd`.

To inspect the bridge alone without sending arm targets:

```bash
G1_NETWORK_INTERFACE=eth0 ./boost/io/g1_bridge/run.sh
```

The defaults are LCM URL `udpm://239.255.76.67:7667?ttl=1`, command channel `G1_PATH_UPPER_COMMAND`, state channel `G1_ROBOT_STATE`, and 100 Hz state publication. Override them with `PATH_LCM_URL`, `G1_PATH_UPPER_CHANNEL`, `G1_ROBOT_STATE_CHANNEL`, and `G1_LCM_STATE_RATE_HZ`.

The bridge retains its optional Dex1 DDS path for separate deployments. The DreamWaQ launcher passes `--hand-backend none`; this adapter does not drive physical BrainCo Revo2 hands.
=======
Dry-run is the default. Confirm state and LCM traffic before enabling arm SDK
ownership:

```bash
G1_NETWORK_INTERFACE=eth0 ./boost/io/g1_bridge/run.sh
G1_NETWORK_INTERFACE=eth0 G1_BRIDGE_LIVE=1 ./boost/io/g1_bridge/run.sh
```

Defaults: LCM URL `udpm://239.255.76.67:7667?ttl=1`, command channel
`G1_PATH_UPPER_COMMAND`, state channel `G1_ROBOT_STATE`, and state publication
at 100 Hz. Override with `PATH_LCM_URL`, `G1_PATH_UPPER_CHANNEL`,
`G1_ROBOT_STATE_CHANNEL`, and `G1_LCM_STATE_RATE_HZ`.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13
