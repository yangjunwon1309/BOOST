# G1 and Thor teleoperation bring-up

This is the deployment plan for the physical G1 stack; it is not a record of
completed hardware validation. MuJoCo commands in
[../boost/sim/README.md](../boost/sim/README.md) run locally and do not
connect to a physical robot.

## Process ownership

| Machine | Process | Responsibility |
| --- | --- | --- |
| Jetson Thor | Galaxy XR / Televuer and PATH | Wrist goals, arm IK, and PDU1 commands |
| Jetson Thor | Revo2 hand MPPI | Fresh six-joint targets per hand over Unitree DDS |
| G1 internal Orin | DreamWaQ | Sole hardware writer to rt/lowcmd; body policy and arm-target merge |
| G1 internal Orin | BOOST Unitree SDK2 arm bridge | Reads PATH arm packets, publishes rt/arm_sdk, relays G1 state |
| G1 internal Orin | External brainco_hand_service | Revo2 serial I/O and measured hand-state topics |
| G1 internal Orin | Teleimager | Head and wrist RGB capture over TIM2/ZMQ |

The G1 body command has 29 joints. Revo2 hands are separate actuators and use
their own DDS topics. Do not start g1_ws/g1_lowlevel_bridge, an older body
controller, or another rt/lowcmd writer with DreamWaQ.

## Prepare Thor

Thor uses two ARM64 Pixi environments. thor-teleop keeps PATH/Televuer on
NumPy 1.26.4; thor runs the JAX/MJX hand sampler on NumPy 2 and CUDA 13. The
desktop hand environment is x86_64/CUDA 12 and is not used on Thor. The two
Thor processes exchange fresh hand frames over local ZMQ port 5566.

JetPack supplies the system CUDA driver. Check its compatibility with the
installed JAX CUDA 13 wheel before publishing hand targets. See NVIDIA's
[JetPack release notes](https://developer.nvidia.com/embedded/jetpack/downloads)
and the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

~~~bash
export PATH="$HOME/.pixi/bin:$PATH"
cd ~/BOOST
pixi install -e thor-teleop
pixi install -e thor
pixi run -e thor-teleop verify-thor-teleop
pixi run -e thor verify-thor
~~~

verify-thor requires ARM64 and a visible JAX CUDA device. Televuer and
Unitree SDK2 Python are Pixi dependencies. The G1 C++ arm bridge separately
requires the Unitree SDK2 C++ checkout already used on the robot.

## Prepare G1

The selected body runtime and backpack-trained export live under
external/dreamwaq_g1_29dof/; the active export is
new3_4_backpack_440000/exported. The target deployment retains upstream
filenames such as g1_homie.yaml and run_homie.py; these belong to the
DreamWaQ deployment package and do not indicate the retired BOOST HOMIE
controller.

From the BOOST checkout on G1, install the isolated runtime environments:

~~~bash
bash scripts/setup_dreamwaq_runtime.sh
bash scripts/setup_teleimager_runtime.sh
~~~

Build the arm bridge against the installed Unitree SDK2 C++ tree:

~~~bash
cd ~/BOOST
UNITREE_SDK2_ROOT=/path/to/unitree_sdk2 ./boost/io/g1_bridge/build.sh
~~~

Start Teleimager as the single camera owner:

~~~bash
bash scripts/run_g1_teleimager.sh
~~~

The combined DreamWaQ/controller launcher is gated. First check its dry-run
configuration and network interface, then enable hardware control when the
robot is in the supported posture and command ownership is confirmed:

~~~bash
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_controller.sh
~~~

DreamWaQ starts in standing mode and follows the target deployment START then
A enable procedure; X toggles standing/walking. This launcher starts the PATH
adapter on rt/arm_sdk and DreamWaQ as the sole rt/lowcmd owner. It does not
start the physical Dex1 or Revo2 service.

Install and start Unitree's external
[brainco_hand_service](https://github.com/unitreerobotics/brainco_hand_service)
on the G1 connected to the Revo2 serial cables. It subscribes to
rt/brainco/{left,right}/cmd and publishes
rt/brainco/{left,right}/state. Use DDS domain 0 and the interface connected
to Thor.

## Start Thor teleoperation

On Thor, start the following in separate terminals:

~~~bash
cd ~/BOOST
pixi run -e thor-teleop galaxy-xr
~~~

~~~bash
cd ~/BOOST
pixi run -e thor-teleop path-state
~~~

~~~bash
cd ~/BOOST
BRAINCO_DDS_INTERFACE=eth0 pixi run -e thor hand-mppi-publish
~~~

Replace eth0 with Thor's interface connected to G1. Do not use lo for the
physical hand publisher. Keep Unitree DDS domain 0 consistent across Thor,
G1, and the BrainCo service.

The hand runner accepts Televuer frames only when their schema, finite 25 x 3
landmarks, timestamp, and 0.20 s freshness bound are valid. It publishes
bounded fresh targets directly; stale input stops publication and leaves
command-loss handling to the BrainCo service. PATH applies its timestamp and
receive-age checks before publishing arm targets. Fresh wrist goals go to IK
without added output smoothing or retiming.

PATH publishes 14 arm targets on LCM channel G1_PATH_UPPER_COMMAND; the G1
adapter republishes them on rt/arm_sdk. Thor and G1 must permit the configured
LCM multicast group 239.255.76.67:7667. Revo2 topics are listed in
[protocols.md](protocols.md). Open the Galaxy XR URL printed by Televuer using
the Thor Wi-Fi address reachable by the headset.

## Bring-up checks

1. Confirm Thor and G1 use the intended robot interface and matching DDS domain.
2. Start the external BrainCo service and confirm fresh state from both hands.
3. Run the G1 arm bridge in dry-run; check state relay and PATH packet sequence.
4. Start Galaxy XR and PATH; confirm wrist frames and arm commands are fresh.
5. Run hand MPPI without publishing and inspect targets and measured hand state.
6. Enable DreamWaQ arm/body control and the hand publisher only after confirming
   command ownership and the robot's supported posture.

Do not run the MuJoCo simulator or its loopback publishers on the physical DDS
network. Check command age and ownership logs before enabling motion. Hardware
performance and physical bring-up remain unvalidated until these checks are
recorded on the robot.
