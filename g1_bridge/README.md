# G1 arm bridge

This runs on the G1 control computer. It receives PATH's 14 arm joints over
LCM, applies the per-joint gains from the ported `g1_ws` configuration, and
publishes them on `rt/arm_sdk`. It also publishes measured G1 state on LCM for
PATH's arm IK. The bridge does not publish `rt/lowcmd`.

The source is ported from `C:\studioprojects\g1_ws\g1_ws`: the original
`src/g1_lowlevel_bridge.cpp` and its supporting headers are copied into this
folder. The original `g1_ws` checkout is not modified. The port adds direct
support for PATH's existing 192-byte `G1_PATH_UPPER_COMMAND` packet and
keeps the packaged process focused on arm SDK output, G1 state, and LCM. The
camera and hand DDS subsystems are not part of this bridge package. The bridge
only writes arm targets, so `path_direct_lcm_bridge` from `g1_ws` is not needed
on the robot.

## PATH and HOMIE command flow

PATH publishes `q`, `dq`, and `tau_ff` for arm motor slots 15-28 on
`G1_PATH_UPPER_COMMAND`. The G1 bridge sets `kp`/`kd` from its ported
`robot_parameters.hpp` values and publishes the arm slots plus Unitree's arm
enable marker on `rt/arm_sdk`. The local HOMIE process reads those fields and
carries them into its final `rt/lowcmd` command.

The local `homie_g1_29dof` deployment subscribes to `rt/arm_sdk`, combines the
arm targets with HOMIE's 15 leg/waist targets, and is the sole `rt/lowcmd`
writer. Run that WBC deployment separately. Do not run the upstream HOMIE
`g1_control` lowcmd writer at the same time; it has a different LCM arm input
interface. See [`../docs/control_ownership.md`](../docs/control_ownership.md).

## Build on G1

Install CMake, a C++17 compiler, `pkg-config`, `liblcm-dev`, and Unitree SDK2.
The SDK can be an SDK2 source checkout or an installed SDK package. It is not
vendored in PATH-G1.

```bash
UNITREE_SDK2_ROOT=/opt/unitree_robotics ./g1_bridge/build.sh
```

## Run on G1

The default is dry-run: receive PATH packets and publish G1 state, without
claiming arm SDK ownership. Set the robot network interface if it is not
`eth0`.

```bash
G1_NETWORK_INTERFACE=eth0 ./g1_bridge/run.sh
```

After checking the LCM command and state streams, enable live arm output:

```bash
G1_NETWORK_INTERFACE=eth0 G1_BRIDGE_LIVE=1 ./g1_bridge/run.sh
```

The defaults are LCM URL `udpm://239.255.76.67:7667?ttl=1`, command channel
`G1_PATH_UPPER_COMMAND`, state channel `G1_ROBOT_STATE`, and 100 Hz state
publication. Override them with `PATH_LCM_URL`, `G1_PATH_UPPER_CHANNEL`,
`G1_ROBOT_STATE_CHANNEL`, and `G1_LCM_STATE_RATE_HZ`.

The bridge validates but ignores the packet's grip flags. The separate Dex1
service remains responsible for hand control.
