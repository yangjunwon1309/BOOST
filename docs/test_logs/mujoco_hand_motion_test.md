# MuJoCo synthetic Revo2 hand-motion test

**Status:** Launcher migrated to the DreamWaQ in-process simulator. The
DreamWaQ motion test has not been rerun. The earlier HOMIE-runtime smoke record
is preserved in `docs/test_logs/mujoco_homie_hand_mppi.md`.

## Test intent

The current launcher is intended to run the normal hand MPPI retargeter using
a generated, repeating two-hand fist/open landmark stream. MuJoCo evaluates
DreamWaQ in-process and publishes fresh Revo2 state; hand commands use loopback
DDS. DreamWaQ holds the configured arm pose, and MuJoCo pauses its initial
physics for five seconds in motion mode. Galaxy XR, Televuer, PATH arm
tracking, and surface queries are not involved. These behaviors still need a
new smoke run after the controller migration.

## Default motion

The generator reads the two existing 25 x 3 source files and leaves them
unchanged. At 30 Hz, the sequence is:

1. Hold fist for 1 s.
2. Linearly interpolate to open over 1.5 s.
3. Hold open for 1 s.
4. Linearly interpolate back to fist over 1.5 s.

The sequence repeats until Ctrl+C. --cycles N makes the standalone generator
stop after N complete cycles. The test launcher writes to a temporary JSON
file via atomic replacement and removes it on shutdown.

## Setup and command

From the BOOST repository root:

~~~bash
pixi install -e sim
pixi install -e hand
pixi run -e sim setup-sim
pixi run -e sim mujoco-hand-motion-test
~~~

To inspect a finite generated stream independently:

~~~bash
pixi run -e sim python scripts/stream_mujoco_hand_motion.py \
  --output /tmp/boost_hand_motion.json --cycles 1
~~~

The static pixi run -e sim mujoco-hand-test task remains available. In motion
mode, --override-file is rejected because the launcher owns a temporary
generated override file.

## Record

| Field | Value |
| --- | --- |
| Date and local time | 2026-10-30 (local smoke test) |
| BOOST revision | |
| Pixi lockfile revision | |
| Simulator viewer | Headless |
| Stream generator phases | Not rerun after DreamWaQ migration |
| DreamWaQ standing pose | Not rerun |
| Left / right hand response | Not rerun |
| MuJoCo state and hand DDS rate | Not rerun |
| XR / Televuer / PATH arm processes absent | Launcher is configured not to start them; not rerun |
| Test result | Not rerun after migration |

## Expected startup and shutdown

- Startup reports the synthetic temporary JSON input and says no XR, Televuer,
  PATH, or physical-robot process is started.
- The hand runner reports the temporary override path and DDS publishing on
  lo, domain 0.
- DreamWaQ holds its configured arm pose through the hold arm source.
- MuJoCo reports DreamWaQ policy status and left/right Revo2 commands.
- Ctrl+C stops the launcher children and removes the temporary JSON directory.
- Confirm no launcher processes or temporary motion directories remain after
  the next smoke run.
- Fresh retarget outputs are sent directly after official joint limits; there
  is no post-retarget smoothing or rate limiter. Updates can run below the
  generated 30 Hz when optimization takes longer.
