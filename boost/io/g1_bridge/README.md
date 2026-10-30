# G1 arm bridge

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

```bash
UNITREE_SDK2_ROOT=/opt/unitree_robotics ./boost/io/g1_bridge/build.sh
```

## Run on G1

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
