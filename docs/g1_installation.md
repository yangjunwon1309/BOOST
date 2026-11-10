# G1 and Thor teleoperation bring-up

This is the deployment plan for the physical G1 stack; it is not a record of
completed hardware validation. MuJoCo commands in
[../boost/sim/README.md](../boost/sim/README.md) run locally and do not
connect to a physical robot.

## Mutually exclusive arm command routes

DreamWaQ is the sole G1 hardware writer to `rt/lowcmd`. Choose one source for
its `rt/arm_sdk` arm targets:

| Route | Thor/XR host | G1 host | Hand control |
| --- | --- | --- | --- |
| PATH + Revo2 | Galaxy XR/PATH publishes PDU1; Revo2 MPPI may publish hand targets | BOOST arm bridge publishes `rt/arm_sdk`; DreamWaQ publishes `rt/lowcmd` | Optional external BrainCo service |
| Direct arm-only XR | `g1-arm-teleop` publishes `rt/arm_sdk` from Televuer + G1_29 IK | DreamWaQ body controller publishes `rt/lowcmd` | Disabled; no finger topics or commands |

Never run both arm publishers at once: the PATH adapter and direct XR process
both publish `rt/arm_sdk`. Do not start `g1_ws/g1_lowlevel_bridge`, another
body controller, or any other `rt/lowcmd` writer alongside DreamWaQ.

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
Unitree SDK2 Python are Pixi dependencies. The direct arm-only route needs
`thor-teleop`; the separate PATH/Revo2 route also uses the `thor` environment.
The G1 C++ arm bridge is needed only for the PATH route and requires the
Unitree SDK2 C++ checkout already used on the robot.

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

For the PATH route, build the arm bridge against the installed Unitree SDK2 C++ tree:

~~~bash
cd ~/BOOST
UNITREE_SDK2_ROOT=/path/to/unitree_sdk2 ./boost/io/g1_bridge/build.sh
~~~

Start Teleimager as the single camera owner:

~~~bash
bash scripts/run_g1_teleimager.sh
~~~

The combined DreamWaQ/PATH-controller launcher is gated. Check its config and
network interface, then enable hardware control when the robot is in the
supported posture and command ownership is confirmed:

~~~bash
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_controller.sh
~~~

DreamWaQ starts in standing mode and follows the target deployment START then
A enable procedure; X toggles standing/walking. This launcher starts the PATH
adapter on rt/arm_sdk and DreamWaQ as the sole rt/lowcmd owner. It does not
start the physical Dex1 or Revo2 service.

For the direct arm-only route, start the alternative body-only launcher on G1:

~~~bash
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_body_controller.sh
~~~

Then on Thor, use `pixi run -e thor-teleop g1-arm-teleop`. This direct route
does not run PATH, `g1_lowlevel_bridge`, or Revo2 hand MPPI. Follow
[`g1_xr_arm_teleop.md`](g1_xr_arm_teleop.md) for XR TLS, camera, recording,
keyboard, and data schema details. The two G1 launchers are alternatives; do
not start both.

Install and start Unitree's external
[brainco_hand_service](https://github.com/unitreerobotics/brainco_hand_service)
on the G1 connected to the Revo2 serial cables. It subscribes to
rt/brainco/{left,right}/cmd and publishes
rt/brainco/{left,right}/state. Use DDS domain 0 and the interface connected
to Thor.

## Start Thor teleoperation

On Thor, start the hand-tracking PATH stack in one interactive terminal:

~~~bash
cd ~/BOOST
pixi run -e thor-teleop path-teleop
~~~

Open the printed secure Televuer URL on Galaxy XR and wait until the PATH log
shows fresh XR wrist frames. Press `r` in that same terminal to start/resume
arm following, `s` to start a wrist experiment CSV, `s` again to stop and save,
or `q` to stop and exit. The CSV and summary JSON are written under
`$HOME/data/boost/path_wrist_experiments` by default; override that directory
with `BOOST_PATH_EXPERIMENT_DATA_ROOT`.

In a second Thor terminal, start Revo2 hand MPPI:

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

If G1 state does not appear in PATH, run the same read-only link report on both
hosts:

Thor:

~~~bash
PIXI_FROZEN=true pixi run -e thor-teleop bash scripts/diagnose_g1_teleop_links.sh 192.168.123.164
~~~

G1:

~~~bash
PATH_PYTHON="$HOME/miniconda3/envs/boost-py310/bin/python" bash scripts/diagnose_g1_teleop_links.sh 192.168.123.2
~~~

The report checks the wired route, listeners, running owners, and live LCM
counts for G1_ROBOT_STATE and G1_PATH_UPPER_COMMAND. On G1, set `PATH_PYTHON`
to the Conda/Pixi Python that imports `lcm` if the base Python does not.

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
