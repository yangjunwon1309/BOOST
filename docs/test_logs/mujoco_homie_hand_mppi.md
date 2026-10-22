# MuJoCo HOMIE stand and Revo2 hand MPPI test log

**Status:** Not run. This is a local MuJoCo simulation test; it does not target a
physical G1.

## Test intent

Start the G1 + Revo2 MuJoCo scene, have HOMIE ramp the simulated G1 to its
configured default standing pose, then run Revo2 hand MPPI against an editable
JSON landmark pose. HOMIE selects `stand.onnx` while keyboard velocity is zero
and `walk.onnx` when the viewer publishes a movement command. This validates
the HOMIE-to-simulator and MPPI-to-simulated-Revo2 DDS loops without Galaxy XR,
Televuer, PATH, or their topics.

## Record

| Field | Value |
|---|---|
| Date and local time | |
| BOOST revision | |
| Pixi lockfile revision | |
| Hand override file | `configs/hands/revo2_hand_override_fist.json` |
| Simulator viewer | Visible / Headless |
| HOMIE stand/walk policy | Pass / Fail |
| Left / right hand response | Pass / Fail / Not checked |
| Test result | Not run / Pass / Fail |

## Setup and command

From the BOOST repository root:

```bash
pixi install -e sim
pixi install -e hand
pixi run -e sim setup-sim
pixi run -e sim mujoco-hand-test
```

The default is a curled-hand pose so Revo2 movement is visible. To use a
different landmark file, set its path when launching the task:

```bash
MUJOCO_HAND_OVERRIDE_FILE=/path/to/hand_pose.json pixi run -e sim mujoco-hand-test
```

An open-hand 25-point pose is in `configs/hands/revo2_hand_override.json`.
The runner reloads the selected file while running, so it can be edited or
replaced to change the input pose. Each side may be `null` or a 25-by-3 array
of finite coordinates in metres; at least one side must contain points.

With the MuJoCo viewer focused, W/S command forward/backward, A/D command
left/right, and Q/R command left/right yaw. O/P decrease/increase the shared
linear and angular speed scale in 10% steps. Direction commands latch until
reversed; Space stops. The scale starts at 50% and ranges from 20% to 100%.

## Expected process and topic separation

- MuJoCo publishes simulated `rt/lowstate` and BrainCo hand state on the local
  DDS interface configured for the sim.
- HOMIE uses `--auto-start` and consumes keyboard `rt/nav_cmd` messages from
  the local MuJoCo simulator. At zero velocity it selects `stand.onnx`; a
  nonzero keyboard command selects `walk.onnx`.
- Hand MPPI reads the JSON override and publishes Revo2 targets to local
  `rt/brainco/{left,right}/cmd` DDS topics.
- No Galaxy XR manager, Televuer process, PATH bridge, physical wireless
  controller, physical network interface, or physical robot is started or
  addressed.

## Result notes

| Check | Expected | Actual / notes |
|---|---|---|
| Sim startup | G1 + Revo2 model loads and publishes simulated state | |
| HOMIE connection | HOMIE connects to local MuJoCo DDS state | |
| Stand/walk policy | HOMIE reports stand at zero command and walk after keyboard input | |
| Body command link | MuJoCo reports fresh `rt/lowcmd` samples | |
| MPPI input | Runner reports the selected override file and valid targets | |
| Revo2 command link | MuJoCo reports fresh left/right Revo2 DDS commands | |
| Revo2 response | Simulated hand state follows updated override landmarks | |
| DDS isolation | Interface is the configured local sim interface (default `lo`) | |
| Outcome and follow-up | | |

## Ending the test

Press Ctrl+C in the launcher terminal. It stops the local HOMIE, hand runner,
and MuJoCo processes together. HOMIE's shutdown sends damping to the simulated
G1; it has no effect on a physical robot.
