# Real G1 teleoperation bring-up

This is the deployment plan for the physical G1 stack. It is not a record of a
completed hardware validation. The MuJoCo commands in
[`../boost/sim/README.md`](../boost/sim/README.md) are desktop-only and must not
be used to infer that a physical G1 is connected.

## Process split

| Machine | Process | Output |
| --- | --- | --- |
| Jetson Thor | Galaxy XR/Televuer manager and PATH state machine | Fresh wrist goals and 14-joint arm IK |
| Jetson Thor | Revo2 hand MPPI | Fresh six-joint targets per hand over Unitree DDS |
| G1 internal Orin | HOMIE/WBC | Sole writer to `rt/lowcmd`, merging legs/waist with arm SDK targets |
| G1 internal Orin | BOOST Unitree SDK2 arm bridge | `rt/arm_sdk` input to HOMIE and state relay back to Thor |
| G1 internal Orin | External `brainco_hand_service` | Revo2 USB-serial output and measured hand state topics |

The G1's `rt/lowcmd` body array is 29 joints. The external Revo2 hands are
separate actuators; their 12 targets are not merged into HOMIE's body array.
The arm bridge and hand service have separate command paths and watchdogs.
HOMIE must remain the only process writing `rt/lowcmd`.

## Prepare the Thor Pixi environments

Thor uses two isolated ARM64 Pixi environments because the pinned Televuer
package requires NumPy 1.x while JAX's CUDA 13 wheel requires NumPy 2. `thor-teleop`
runs Galaxy XR/Televuer and PATH arm IK on NumPy 1.26.4. `thor` runs the hand
MPPI sampler on JAX CUDA 13 and NumPy 2. The processes exchange fresh hand
frames over the existing local ZMQ endpoint on port 5566. The desktop `hand`
environment remains x86_64/CUDA 12. Neither Thor environment installs the
desktop Torch/GeoRT inference stack.

JetPack 7.2.1 currently ships CUDA 13.2.2 on Thor. JAX provides Linux ARM64
CUDA 13 wheels and recommends its pip-installed CUDA libraries; the CUDA 13
wheel requires a Linux NVIDIA driver at least 580 and CUDA 13-compatible
runtime libraries. See NVIDIA's [JetPack release
notes](https://developer.nvidia.com/embedded/jetpack/downloads) and the [JAX
installation guide](https://docs.jax.dev/en/latest/installation.html). The
Pixi environment does not install or replace the system JetPack driver.

On Thor, install Pixi into the user account if it is not already available, and
add its binary directory to the shell path:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd ~/BOOST
pixi install -e thor-teleop
pixi install -e thor
pixi run -e thor-teleop verify-thor-teleop
pixi run -e thor verify-thor
```

`verify-thor` requires an ARM64 host and a visible JAX CUDA device. If it fails,
do not start the hand publisher; inspect the installed JetPack release and
CUDA driver, and check whether `LD_LIBRARY_PATH` is overriding JAX's installed
CUDA libraries. No `uv`, `.venv_runtime`, Televuer checkout, or Unitree SDK
checkout is needed for the Pixi runtime. Televuer and Unitree SDK2 Python are
pinned Pixi dependencies. The G1 C++ arm bridge still needs Unitree SDK2 C++ on
the G1 computer.

## Prepare the G1 computer

Install the selected HOMIE deployment and start it as the only `rt/lowcmd`
writer. Configure HOMIE to consume `rt/arm_sdk` for the 14 arm joints; see
[`homie_integration.md`](homie_integration.md) and the checked-in config under
`external/homie_g1_29dof/deploy/robot_control/configs/`.

Build the arm bridge against the Unitree SDK2 C++ checkout already used on the
robot:

```bash
cd ~/BOOST
UNITREE_SDK2_ROOT=/path/to/unitree_sdk2 ./boost/io/g1_bridge/build.sh
```

The bridge is dry-run by default. First confirm it receives state and PATH
packets without claiming arm SDK ownership:

```bash
G1_NETWORK_INTERFACE=eth0 ./boost/io/g1_bridge/run.sh
```

After the network and ownership checks, enable the arm target path:

```bash
G1_NETWORK_INTERFACE=eth0 G1_BRIDGE_LIVE=1 ./boost/io/g1_bridge/run.sh
```

Install and start Unitree's external
[`brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service)
on the G1 computer connected to the Revo2 serial cables. It must subscribe to
`rt/brainco/{left,right}/cmd` and publish
`rt/brainco/{left,right}/state`. Use DDS domain 0 and select an interface that
can exchange DDS traffic with Thor. The bridge does not initialize or command
the hand service.

## Start the Thor teleoperation processes

Use three terminals on Thor. Televuer/PATH run in `thor-teleop`; hand MPPI
runs in `thor`. They share Televuer's loopback ZMQ endpoint, while the DDS
hand publisher selects the physical network interface explicitly.

```bash
cd ~/BOOST
pixi run -e thor-teleop galaxy-xr
```

```bash
cd ~/BOOST
pixi run -e thor-teleop path-state
```

```bash
cd ~/BOOST
BRAINCO_DDS_INTERFACE=eth0 pixi run -e thor hand-mppi-publish
```

Replace `eth0` with Thor's interface connected to the G1 network. Do not use
`lo`. Keep Unitree DDS domain 0 on Thor, G1, and the BrainCo service unless the
whole robot network is deliberately configured for another domain. The hand
runner accepts a Televuer frame only when it has a supported schema, finite
25×3 hand coordinates, a valid timestamp, and is within its 0.20 s freshness
window. A fresh retarget result is published directly after Revo2 joint-limit
validation. If input becomes stale, the runner stops publishing and the
BrainCo service's watchdog handles the loss.

PATH requires a monotonic source timestamp, accepts at most 0.20 s of target
age (and 0.10 s future-clock tolerance), and checks local receive age as well.
Past either age limit it stops publishing arm targets; the G1 arm bridge releases arm-SDK ownership after its
150 ms command timeout. Both age limits can be changed in the launch config
after measuring the actual Televuer packet gaps. They do not smooth fresh
targets.

Shared perception/contact targets are disabled by default so Y/B controller
buttons cannot replace the Televuer arm goals. To deliberately use the existing
shared-contact arm mode, set `PATH_ENABLE_SHARED_TARGET_ASSIST=1` when starting
`path-state`; then Y/B is an explicit dead-man selector for that alternate goal.

PATH publishes 14 arm targets over LCM channel `G1_PATH_UPPER_COMMAND`; the
G1 bridge republishes them on `rt/arm_sdk`. Thor and G1 must allow the configured
LCM multicast group `239.255.76.67:7667`. Revo2 commands and states use the
Unitree DDS topics listed in [`protocols.md`](protocols.md). Galaxy XR should
open the `https://<Thor-Wi-Fi-IP>:8012` address shown by the Televuer startup
log; use the address reachable from the headset, not `127.0.0.1`.

## Ordered hardware checks

1. Verify Thor and G1 select the intended wired robot interface; verify their
   DDS domain IDs match.
2. Start HOMIE and confirm it alone publishes `rt/lowcmd`.
3. Run the G1 arm bridge in dry-run and check `G1_ROBOT_STATE` reaches Thor.
4. Start the BrainCo service and confirm fresh state on both Revo2 DDS topics.
5. Start Televuer and PATH. Confirm headset wrist frames are fresh and the arm
   bridge receives PATH sequence numbers while still in dry-run.
6. Run hand MPPI without `--publish` first. Check both retargeter outputs and
   measured Revo2 states; then stop it.
7. Enable the bridge's live arm route and hand runner DDS publishing only
   after the robot is in a supported safe posture and each command topic has a
   single owner.

Do not run the MuJoCo simulator, its loopback publishers, or a second whole-body
controller on the physical DDS network. Check command age and ownership logs
before enabling motion. Stale or invalid frames are rejected; a fresh wrist
pose goes to IK without a moving reference, output smoothing, or target
projection. IK enforces model joint bounds and reports residuals without using
them to discard a finite solution.
