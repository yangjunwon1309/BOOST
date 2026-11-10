# BOOST

BOOST is the research codebase for object-centric teleoperation and
manipulation with a Unitree G1 and BrainCo Revo2 hands. The development scope
and staged implementation plan are in [BOOST_dev_plan.md](BOOST_dev_plan.md).

The current source is a reorganized PATH-G1 starting point. The teleop state
machine, XR streams, arm IK, G1 bridge, and wrist perception service have been
moved into the subsystem layout below. The local Revo2 MuJoCo path supports
binary grip commands and continuous landmark retargeting. The real G1 body
controller is the backpack-trained DreamWaQ deployment; its arm adapter and
the separate BrainCo Revo2 service have independent command paths. Hardware
control is still awaiting physical bring-up validation. FoundationPose and
object-aware MPPI remain future work; the direct arm-only XR recorder has a
documented whole-body episode schema.

Git workflow: main is the README-only landing branch; ongoing development is
on deploy. See [docs/git_workflow.md](docs/git_workflow.md) and
python scripts/boost_git.py --help.

## Repository layout

- boost/teleop/: PATH-derived state machine, XR input, and isolated legacy
  intention code. Its source pin and move map are in
  [boost/teleop/UPSTREAM.md](boost/teleop/UPSTREAM.md).
- boost/io/path_transport/: the PATH LCM protocol and ZMQ transport layer.
- boost/io/revo2_hand_service/: the adapter contract for the external
  BrainCo hand service; its driver source is not vendored.
- boost/control/hand_mppi/: the shared BrainCo DDS mapping, official Revo2
  MJCF adapter, and Televuer/override landmark retargeter, with
  mimic_retargeter_lab pinned as a submodule.
- boost/control/arm/: PATH-derived nonlinear G1 arm IK.
- boost/io/g1_bridge/: arm-only Unitree SDK2 bridge and GST1 state relay;
  physical Revo2 uses the separate BrainCo DDS service.
- external/dreamwaq_g1_29dof/: selected backpack-trained DreamWaQ runtime
  and export; its sim backend and G1 launch path are in
  [docs/dreamwaq_integration.md](docs/dreamwaq_integration.md).
- external/g1_arm_teleop/: G1_29 arm-only Galaxy XR teleop and recorder
  adapted from the locally verified `url_g1_deploy` snapshot.
- external/g1_ws/ and external/teleimager/: pinned target schema/reference
  sources and TIM2 camera server.
- services/pose_service/: the moved PATH wrist RGB-D/SAM3 service. It
  currently publishes surface and target data; FoundationPose pose estimation
  is planned work.
- configs/: BOOST configuration roots, with the existing teleop runtime
  settings under configs/teleop/.
- boost/sim/: MuJoCo assets and model work.
- assets/robot/: pinned G1 model and the BrainCo Revo2 description snapshot.
- scripts/, docs/, tests/, datasets/, and third_party/: launch tools,
  deployment references, verification, dataset guidance, and pinned external
  dependencies.

## Galaxy XR hand teleoperation

On Thor, start the combined hand-tracking/PATH/session recorder in an
interactive terminal, then run Revo2 MPPI in a second terminal:

~~~bash
pixi run -e thor-teleop path-teleop
~~~

~~~bash
BRAINCO_DDS_INTERFACE=eth0 pixi run -e thor hand-mppi-publish
~~~

After Galaxy XR hand tracking is connected, use `r` to start/resume, `s` to
start or stop a wrist experiment recording, and `q` to stop. The G1 PATH
adapter feeds `rt/arm_sdk`; target DreamWaQ owns `rt/lowcmd`. See
[docs/g1_installation.md](docs/g1_installation.md) and
[docs/dreamwaq_integration.md](docs/dreamwaq_integration.md).
Use `bash scripts/diagnose_g1_teleop_links.sh <peer-ip>` on Thor and G1 to
inspect the configured addresses, listeners, and G1 LCM state packets.
The alternative direct arm-only Galaxy XR route runs the Pixi task
`g1-arm-teleop` on Thor and `scripts/run_g1_dreamwaq_body_controller.sh` on
G1; it publishes arm targets directly to `rt/arm_sdk` and does not control fingers.
See [docs/g1_xr_arm_teleop.md](docs/g1_xr_arm_teleop.md).

## Local MuJoCo desktop

The local G1 + Revo2 model and DreamWaQ/PATH stack run in the Pixi sim
environment. Install and inspect the runtime with:

~~~bash
export PIXI_FROZEN=true
pixi install -e sim
pixi run -e sim setup-sim
~~~

Keep `PIXI_FROZEN=true` in each desktop shell so Pixi uses the committed sim
lock instead of re-solving the separate Thor NumPy 2 environment.

Start the desktop simulation without a headset with
pixi run -e sim mujoco-no-xr; use pixi run -e sim mujoco when Galaxy XR
certificates are configured. The Televuer manager always uses hand tracking;
press `r` in the PATH terminal to start following fresh XR wrist poses.
Use pixi run -e sim mujoco-hand-test to run the simulated G1 under DreamWaQ
while hand MPPI reads an editable JSON pose; it starts no XR, Televuer, or PATH
bridge. Use pixi run -e sim mujoco-hand-motion-test for a generated repeating
fist/open stream. The MuJoCo viewer keys drive local locomotion: W/S
forward/back, A/D left/right, Q/R yaw, O/P adjust both linear and angular
speed, and Space stops. Direction commands latch until reversed or stopped.
Use pixi run -e sim mujoco-hand-xr for the separate live Galaxy XR -> Televuer
-> PATH -> Mimic -> Revo2 simulation route. Both start the separate Pixi hand
environment and avoid competing Revo2 command publishers.
See [boost/sim/README.md](boost/sim/README.md) for network, channel, and
configuration details. The separate hand Pixi environment holds the Python
3.12 and GPU dependencies for the retargeter, without changing MuJoCo 3.5.0
in the current sim environment. Run git submodule update --init --recursive
followed by pixi install -e hand to install its pinned upstream and GPU
dependencies. The desktop dependency workflow is defined by pixi.toml and
pixi.lock; it does not require uv, .venv_runtime, or manual Televuer/Unitree
SDK source directories. If this shell cannot find Pixi, use
$HOME/.pixi/bin/pixi or add $HOME/.pixi/bin to PATH. The Galaxy XR URL, stack
shutdown, and port cleanup steps are in the
[MuJoCo + Galaxy XR runbook](docs/mujoco_galaxy_xr_runbook.md).

## Real G1 Thor runtime

Thor uses separate ARM64 thor-teleop (Televuer/PATH, NumPy 1) and thor
(hand MPPI, JAX CUDA 13/NumPy 2) Pixi environments because those dependencies
require different NumPy majors. On Jetson Thor, install both with
pixi install -e thor-teleop and pixi install -e thor, then run their
verify-thor-teleop and verify-thor tasks. They are separate from the x86
desktop hand environment; see [docs/g1_installation.md](docs/g1_installation.md)
for the physical bring-up sequence.
