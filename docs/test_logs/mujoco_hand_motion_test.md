# MuJoCo synthetic Revo2 hand-motion test

**Status:** Headless smoke test passed on 2026-10-30. This is a local MuJoCo
test and does not target a physical G1 or Revo2 hand.

## Test intent

Run the normal hand MPPI retargeter using a generated, repeating two-hand
fist/open landmark stream. MuJoCo supplies fresh Revo2 state, HOMIE's stand
policy runs at zero velocity, and the hand runner publishes only to the
loopback DDS domain. MuJoCo holds its initial pose for five seconds and until
HOMIE is publishing body commands; with no PATH arm target, HOMIE keeps the
configured arm pose and does not subscribe to `rt/arm_sdk`. Galaxy XR,
Televuer, PATH arm tracking, and surface queries are not involved.

## Default motion

The generator reads the two existing 25×3 source files and leaves them
unchanged. At 30 Hz, the sequence is:

1. Hold fist for 1 s.
2. Linearly interpolate to open over 1.5 s.
3. Hold open for 1 s.
4. Linearly interpolate back to fist over 1.5 s.

The sequence repeats until Ctrl+C. `--cycles N` makes the standalone generator
stop after N complete cycles. The test launcher writes to a temporary JSON
file via atomic replacement and removes it on shutdown.

## Setup and command

From the BOOST repository root:

```bash
pixi install -e sim
pixi install -e hand
pixi run -e sim setup-sim
pixi run -e sim mujoco-hand-motion-test
```

To inspect a finite generated stream independently:

```bash
pixi run -e sim python scripts/stream_mujoco_hand_motion.py \
  --output /tmp/boost_hand_motion.json --cycles 1
```

The static `pixi run -e sim mujoco-hand-test` task remains available. In motion
mode, `--override-file` is rejected because the launcher owns a temporary
generated override file.

## Record

| Field | Value |
|---|---|
| Date and local time | 2026-10-30 (local smoke test) |
| BOOST revision | |
| Pixi lockfile revision | |
| Simulator viewer | Headless |
| Stream generator phases | Pass; all phases repeated |
| HOMIE default stand | Pass; base z ≈ 0.766 m, tilt under 1° |
| Left / right hand response | Pass; targets and states changed through fist/open cycle |
| MuJoCo lowcmd and hand DDS state | Pass; both hand outputs about 30 Hz |
| XR / Televuer / PATH arm processes absent | Pass; launcher did not start them |
| Test result | Pass; timed headless smoke run, stopped with Ctrl+C |

## Expected startup and shutdown

- Startup reports the synthetic temporary JSON input and explicitly says no
  Galaxy XR, Televuer, PATH arm tracking, or physical-robot process is started.
- The hand runner reports the temporary editable override path and
  `DDS publish=enabled` on `lo`, domain 0.
- HOMIE holds its configured arm pose without subscribing to `rt/arm_sdk`.
- MuJoCo reports fresh HOMIE `rt/lowcmd` and left/right Revo2 commands. HOMIE
  runs stand policy at zero command and keeps its arm targets at the default
  pose.
- Ctrl+C stops all launcher children and removes the temporary JSON directory.
- Post-shutdown inspection found no launcher processes or temporary motion
  directories.
- Revo2 outputs use the latest retargeter result directly after the official
  joint limits; there is no post-retarget smoothing or rate limiter. Actual
  updates can still run below the generated 30 Hz when retargeting takes longer.
