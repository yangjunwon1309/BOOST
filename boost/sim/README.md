# MuJoCo simulation

The Revo2 model uses the pinned BrainCo geometry, inertias, and joint limits
from external/brainco-description. BOOST adds the G1 wrist mounts,
six-channel actuators, and distal-joint coupling constraints. The complete
scene currently has nq=76, nv=73, and nu=41.

configs/sim/mujoco_revo2.yaml selects the table-room scene with the articulated
box and can assets. configs/sim/mujoco_revo2_vacant.yaml selects the vacant
scene. Both use the backpack payload represented by the
new2_3_backpack_lower8cm_299999 DreamWaQ export metadata.

## Start the integrated scene

~~~bash
export PIXI_FROZEN=true
pixi install -e sim
pixi run -e sim setup-sim
pixi run -e sim mujoco
~~~

Keep `PIXI_FROZEN=true` exported in each desktop shell when running sim tasks;
it uses the committed desktop lock without resolving Thor's separate NumPy 2
runtime.

Available launch tasks:

- mujoco-no-xr: DreamWaQ MuJoCo and PATH bridge without Galaxy XR.
- mujoco-headless: same stack without XR or viewer.
- mujoco-hand-test: local JSON Revo2 pose and hand MPPI.
- mujoco-hand-motion-test: generated repeating fist/open hand input.
- mujoco-hand-xr: Galaxy XR hand tracking, contact query, and Revo2 MPPI.

Set MUJOCO_SIM_CONFIG=configs/sim/mujoco_revo2_vacant.yaml to select the
vacant scene. The desktop sim Pixi environment evaluates the selected
DreamWaQ policy and action compiler in-process at its configured control
period. MuJoCo supplies policy observations, while the PATH DDS bridge supplies
arm targets on rt/arm_sdk. The sim applies target gains and torque limits to
local actuators; it never subscribes to or publishes hardware rt/lowcmd.

With the viewer focused, W/S set forward/backward velocity, A/D set left/right,
and Q/R set left/right yaw. O/P adjust a shared speed scale from 20% to 100% in
10% steps. Direction inputs stay active until the opposite direction or Space
is pressed. The initial scale is 50%; the viewer overlay and periodic sim log
show the current cmd_vel. The default NEW2-3 export has no explicit
standing/walking observation, so X does not change its policy input. Exports
with an explicit locomotion-mode contract enable the X toggle.

## Revo2 hand motion test

Run a static editable pose:

~~~bash
pixi run -e sim mujoco-hand-test
~~~

The runner reads configs/hands/revo2_hand_override_fist.json by default and
reloads it while active. Select another file with --override-file PATH.
This route runs DreamWaQ in MuJoCo, holds its configured arm pose, and starts
no XR, Televuer, PATH bridge, or physical-robot publisher.

To stream a generated two-hand fist/open cycle through the same hand retargeter:

~~~bash
pixi run -e sim mujoco-hand-motion-test
~~~

At 30 Hz the generator holds a fist for 1 s, opens over 1.5 s, holds open for
1 s, then closes over 1.5 s. It interpolates both 25 x 3 landmark arrays and
atomically replaces a temporary JSON file; the source templates are unchanged.
To run the generator independently for a finite number of cycles:

~~~bash
pixi run -e sim python scripts/stream_mujoco_hand_motion.py \
  --output /tmp/boost_hand_motion.json --cycles 1
~~~

The test uses the usual hand MPPI optimizer and joint limits. DreamWaQ runs in
standing mode with its arm source set to hold; MuJoCo pauses its initial
physics for five seconds in motion mode. The launcher uses loopback DDS only
and starts no XR, Televuer, PATH, surface-query, or physical-robot process.
Ctrl+C stops the simulator, stream generator, and hand runner and removes the
temporary files. See
[docs/test_logs/mujoco_hand_motion_test.md](../../docs/test_logs/mujoco_hand_motion_test.md)
for the recorded smoke-test result.

The hand MPPI runner uses the separate Pixi hand environment. XR mode
disables PATH's binary grip publisher while MPPI publishes continuous Revo2
targets. The local JSON override defaults to
configs/hands/revo2_hand_override_fist.json; select another with
--override-file PATH or MUJOCO_HAND_OVERRIDE_FILE.

The surface query targets only configured object geoms in
configs/sim/mujoco_revo2.yaml. D405 RGB-D capture and depth-based BOOST
estimation remain a separate perception path. XR certificate setup and
occupied-port cleanup are covered by the
[Galaxy XR runbook](../../docs/mujoco_galaxy_xr_runbook.md). Asset provenance
is listed in [mujoco/assets/ASSET_SOURCES.md](mujoco/assets/ASSET_SOURCES.md).
