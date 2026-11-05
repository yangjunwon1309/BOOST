# MuJoCo simulation

The Revo2 model uses the official BrainCo geometry, inertias, and joint limits from the pinned `external/brainco-description` submodule. BOOST adds G1 wrist mounts, six-channel actuators, and five distal-joint coupling constraints per hand. The scene loads as `nq=58`, `nv=57`, `nu=41`.

The active scene in `configs/sim/mujoco_revo2.yaml` is `mujoco/scene_29dof_with_revo2_table_room.xml`. It combines the G1/Revo2 model with the table room and its articulated object/can assets. The vacant G1/Revo2 model remains available in `configs/sim/mujoco_revo2_vacant.yaml`.

The active model includes the selected DreamWaQ backpack payload: a 3.5 kg box at the export metadata's nominal center `[-0.12464276, 0, 0.1]` m, with `[0.11, 0.18, 0.34]` m dimensions. This corresponds to the backpack-trained `new3_4_backpack_440000` export; no extra fixed compensation torque is added.

## Run the integrated scene

```bash
pixi install -e sim
pixi run -e sim setup-sim
pixi run -e sim mujoco
```

The `sim` environment installs the pinned MuJoCo/PATH packages, Unitree SDK2 Python, and the local DreamWaQ runtime. It does not install the G1 hardware runtime or Teleimager server. Those have separate Python 3.10 environments outside the repo.

- `pixi run -e sim mujoco-no-xr`: DreamWaQ MuJoCo and PATH bridge without Galaxy XR.
- `pixi run -e sim mujoco-headless`: same sim without XR or viewer.
- `pixi run -e sim mujoco-hand-test`: adds the Revo2 MPPI runner with a local JSON pose.
- `pixi run -e sim mujoco-hand-xr`: Galaxy XR hand tracking, contact query, and Revo2 MPPI.
- Set `MUJOCO_SIM_CONFIG=configs/sim/mujoco_revo2_vacant.yaml` to use the vacant scene.

The sim evaluates the selected DreamWaQ policy and action compiler in-process at the target 20 ms control period. MuJoCo state supplies policy observations; the PATH DDS bridge supplies arm targets on `rt/arm_sdk`. The sim applies target gains and torque limits to MuJoCo actuators. It neither subscribes to nor publishes hardware `rt/lowcmd`. The hardware DreamWaQ process is not started.

<<<<<<< HEAD
In the viewer, W/S, A/D, and Q/R request forward/back, lateral, and yaw velocity. O/P change the command scale, Space stops, and X toggles DreamWaQ standing/walking. Commands remain local on DDS loopback and the simulation LCM multicast URL.
=======
To stream a synthetic two-hand fist/open cycle through the same hand MPPI
runner, use the dedicated task:

```bash
pixi run -e sim mujoco-hand-motion-test
```

It repeats at 30 Hz: fist hold 1 s → linear open transition 1.5 s → open hold
1 s → linear close transition 1.5 s. The streamer interpolates both 25×3
landmark arrays and atomically replaces a temporary JSON file; the two source
templates remain unchanged. To run the generator by itself for a finite number
of cycles, use:

```bash
pixi run -e sim python scripts/stream_mujoco_hand_motion.py \
  --output /tmp/boost_hand_motion.json --cycles 1
```

The motion test uses the normal Revo2 hand retargeter and publishes each fresh
six-joint result directly after enforcing model joint limits. Retargeting still
maps the generated landmarks through the iCEM objective. It starts only the
local MuJoCo simulation, HOMIE's stand policy at
zero velocity, and hand MPPI on DDS `lo`, domain 0. MuJoCo holds its initial
pose for five seconds and until HOMIE is publishing body commands; no PATH arm
target is started, so HOMIE holds its configured arm pose without subscribing
to `rt/arm_sdk`. It does not start XR, Televuer, PATH, surface queries, or a
physical-robot publisher. `--motion` cannot be combined with `--override-file`;
use `mujoco-hand-test --override-file PATH` for a static editable pose. Ctrl+C
stops the streamer and other child processes and removes the temporary JSON
directory. See
[`docs/test_logs/mujoco_hand_motion_test.md`](../../docs/test_logs/mujoco_hand_motion_test.md)
for the motion-test checks.

With the viewer focused, W/S set forward/backward velocity, A/D set left/right,
and Q/R set left/right yaw. O/P lower/raise a shared speed scale from 20% to
100% in 10% steps. Direction inputs stay active until the opposite direction
or Space is pressed. The initial scale is 50%; the viewer overlay and periodic
sim log show the current `cmd_vel`.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13

The hand MPPI routes use the separate Pixi `hand` environment. In XR mode they disable PATH's binary grip publisher so only the MPPI process sends continuous Revo2 commands. The local JSON hand override defaults to `configs/hands/revo2_hand_override_fist.json`; select another with `--override-file PATH` or `MUJOCO_HAND_OVERRIDE_FILE`.

The surface query only targets the configured object geoms in `configs/sim/mujoco_revo2.yaml`. D405 RGB-D capture and depth-based BOOST estimation remain a separate perception path.

<<<<<<< HEAD
For XR certificate setup and occupied-port cleanup, see [the Galaxy XR runbook](../../docs/mujoco_galaxy_xr_runbook.md). Asset provenance is in [`mujoco/assets/ASSET_SOURCES.md`](mujoco/assets/ASSET_SOURCES.md).
=======
HOMIE is the sole `rt/lowcmd` writer: it combines its leg/waist policy output
with PATH's 14 arm targets and gains. MuJoCo applies
`tau + kp * (q_des - q) + kd * (dq_des - dq)` to the 29 body actuators and
publishes simulated state back to DDS. The simulator-only compatibility
adapter maps binary left/right grip flags to the six normalized BrainCo command
channels. The
`boost/control/hand_mppi` runner consumes either Televuer's 25 landmarks from
the existing `vr_pose_debug` ZMQ message or an editable local JSON pose, runs
Mimic's sampling-based retargeter against BrainCo's official Revo2 hand model,
then publishes six normalized DDS targets. Both MPPI launchers disable the
binary grip publisher so only one process writes the Revo2 command topics.

The simulation stack is adapted from PATH's Sonic runtime. The `sim` Pixi
environment uses the pinned MuJoCo/PATH Python dependencies declared in the
project's `sim` extra. See `configs/sim/mujoco_revo2.yaml` and
`configs/sim/homie_g1_mujoco.yaml` for model, hand, and HOMIE settings.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13
