# BOOST

BOOST is the research codebase for object-centric teleoperation and
manipulation with a Unitree G1 and BrainCo Revo2 hands. The development scope
and staged implementation plan are in [`BOOST_dev_plan.md`](BOOST_dev_plan.md).

The current source is a reorganized PATH-G1 starting point. The teleop state
machine, XR streams, arm IK, G1 bridge, and wrist perception service have been
moved into the subsystem layout below. The local Revo2 MuJoCo path supports
binary grip commands and continuous landmark retargeting. Physical Revo2
service validation, FoundationPose, object-aware MPPI, and the episode schema
remain future work.

Git workflow: `main` is the README-only landing branch; ongoing development is
on `deploy`. See [`docs/git_workflow.md`](docs/git_workflow.md) and
`python scripts/boost_git.py --help`.

## Repository layout

- `boost/teleop/`: PATH-derived state machine, XR input, and isolated legacy
  intention code. Its source pin and move map are in
  [`boost/teleop/UPSTREAM.md`](boost/teleop/UPSTREAM.md).
- `boost/io/path_transport/`: the PATH LCM protocol and ZMQ transport layer.
- `boost/io/revo2_hand_service/`: the planned adapter contract for the external
  BrainCo hand service; its driver source is not vendored.
- `boost/control/hand_mppi/`: the shared BrainCo DDS mapping, official Revo2
  MJCF adapter, and Televuer/override landmark retargeter, with
  `mimic_retargeter_lab` pinned as a submodule.
- `boost/control/arm/`: PATH-derived nonlinear G1 arm IK.
- `boost/io/g1_bridge/`: PATH-derived Unitree arm and Dex1 bridge. It still
  speaks the old Dex1 interface; Revo2 uses the separate
  `unitreerobotics/brainco_hand_service` DDS interface and needs a new adapter.
- `services/pose_service/`: the moved PATH wrist RGB-D/SAM3 service. It
  currently publishes surface and target data; FoundationPose pose estimation
  is planned work.
- `configs/`: BOOST configuration roots, with the existing teleop runtime
  settings under `configs/teleop/`.
- `boost/sim/`: MuJoCo assets and model work.
- `assets/robot/`: pinned G1 model and the BrainCo Revo2 description snapshot.
- `external/homie_g1_29dof/`: local HOMIE/WBC snapshot. HOMIE remains the sole
  `rt/lowcmd` writer; BOOST sends arm targets through `rt/arm_sdk`.
- `scripts/`, `docs/`, `tests/`, `datasets/`, and `third_party/`: launch tools,
  deployment references, verification, dataset guidance, and pinned external
  dependencies.

## Existing PATH teleop entry points

The current PATH-derived servers can be started from the repository root:

```bash
bash scripts/run_pico_manager.sh
bash scripts/run_state_machine_bridge.sh
```

Use `bash scripts/run_galaxy_xr_manager.sh` for the Galaxy XR input route.
These entry points retain the current PATH packet and controller behavior;
they do not send Revo2 hand commands. The service deployment notes are in
[`docs/deployment.md`](docs/deployment.md). The separate G1/Dex1 installation
document is retained as a PATH reference, not a BOOST Revo2 runbook.

## Local MuJoCo desktop

The local G1 + Revo2 model and HOMIE/PATH stack run in the Pixi `sim`
environment. Install and inspect the runtime with:

```bash
pixi install -e sim
pixi run -e sim setup-sim
```

Start the desktop simulation without a headset
with `pixi run -e sim mujoco-no-xr`; use `pixi run -e sim mujoco` when Galaxy XR
certificates are configured. To stream Televuer's 25 hand joints over PATH's
ZMQ pose message, set `PATH_TELEVUER_HAND_TRACKING=1` when starting `mujoco`.
Use `pixi run -e sim mujoco-hand-test` to run the simulated G1 under HOMIE
while hand MPPI reads an editable JSON pose; it starts no XR, Televuer, or PATH
bridge. The MuJoCo viewer keys drive local locomotion: W/S forward/back, A/D
left/right, Q/R yaw, O/P adjust both linear and angular speed, and Space stops.
Direction commands latch until reversed or stopped. Use
`pixi run -e sim mujoco-hand-xr` for the separate live
Galaxy XR → Televuer → PATH → Mimic → Revo2 simulation route. Both start the
separate Pixi `hand` environment and avoid competing Revo2 command publishers.
See [`boost/sim/README.md`](boost/sim/README.md)
for network, channel, and configuration details. The separate `hand` Pixi
environment holds the Python 3.12 and GPU dependencies for the retargeter,
without changing MuJoCo 3.5.0 in the current sim environment. Run
`git submodule update --init --recursive` followed by `pixi install -e hand` to
install its pinned upstream and GPU dependencies. The
desktop dependency workflow is defined by `pixi.toml` and `pixi.lock`; it does
not require the `uv` executable, `.venv_runtime`, or manual Televuer/Unitree
SDK source directories. If this shell cannot find Pixi, use
`$HOME/.pixi/bin/pixi` or add `$HOME/.pixi/bin` to `PATH`.
The Galaxy XR URL, stack shutdown, and port cleanup steps are in the
[`MuJoCo + Galaxy XR runbook`](docs/mujoco_galaxy_xr_runbook.md).
