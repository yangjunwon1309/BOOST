# Revo2 hand retargeting

BOOST connects Televuer landmarks to BrainCo Revo2 with the pinned `mimic_retargeter_lab` checkout in `upstream/`. `brainco_revo2_model.py` uses the official single-hand MJCF and maps its eleven MJCF joints onto the six BrainCo command channels.

## Existing PATH and DDS contracts

| Direction | Contract |
|---|---|
| Televuer → PATH → runner | ZMQ `vr_pose_debug` on `tcp://127.0.0.1:5566` |
| PATH hand payload | Schema 1, two `(25, 3)` float32 arrays in metres, validity, sequence, monotonic timestamp, coordinate-frame id 1 |
| Runner → Revo2 | Unitree DDS `MotorCmds_` on `rt/brainco/{left,right}/cmd` |
| Revo2 → runner | Unitree DDS `MotorStates_` on `rt/brainco/{left,right}/state` |
| PATH arm route | PATH PDU1 packet → G1 `rt/arm_sdk` adapter → DreamWaQ body merge |

The runner checks schema, coordinate frame, age, validity, shape, and finite values before optimizing either hand. Its six radian targets are clamped to official BrainCo limits, converted to normalized `[0,1]` values, then rate-limited. Live publishing waits for fresh Revo2 state. It never publishes `rt/lowcmd`; DreamWaQ remains the sole hardware body writer.

<<<<<<< HEAD
## Runtime
=======
## Runtime behavior

`runner.py` uses Mimic's sampling-based iCEM retargeter with the local Revo2
model. Its six radian targets are clamped to official BrainCo limits, converted
through `BrainCoRevo2Adapter` to `[0, 1]`, and published directly. Live
publishing waits for fresh Revo2 state on each side and seeds the optimizer from
the measured pose. When Televuer input becomes stale, the runner stops
publishing; the BrainCo service owns its command-loss watchdog. There is no
post-retarget EMA, interpolation, or rate limiter. HOMIE remains the sole writer
to `rt/lowcmd`.

The runner is read-only unless `--publish` is supplied:
>>>>>>> 228019f83417f222bc20e79b837865e645492e13

```bash
pixi install -e hand
pixi run -e hand python -m boost.control.hand_mppi.runner
```

For the full Galaxy XR + Revo2 simulation, use `pixi run -e sim mujoco-hand-xr`. This disables PATH's binary grip DDS publisher while MPPI is active. The local JSON test uses `pixi run -e sim mujoco-hand-test`, defaults to `configs/hands/revo2_hand_override_fist.json`, and can be changed with `--override-file PATH`.

The XR hand route can query object surface hits from MuJoCo over local ZMQ. The active scene's collision geom names are configured in `configs/sim/mujoco_revo2.yaml`. D405 RGB-D geometry and depth synchronization remain separate from this simulation-only contact query.

<<<<<<< HEAD
See the [MuJoCo README](../../sim/README.md) for launch details and the [Galaxy XR runbook](../../../docs/mujoco_galaxy_xr_runbook.md) for the headset URL, shutdown, and port cleanup.
=======
This mode starts the runner in Pixi's separate `hand` environment and disables
PATH's binary grip DDS publisher to avoid competing writes to the Revo2 command
topics. `configs/hands/brainco_revo2_mppi.yaml` holds the model, optimizer,
input-age, update-rate, and command-rate settings.
The [MuJoCo + Galaxy XR runbook](../../../docs/mujoco_galaxy_xr_runbook.md)
covers the headset URL, shutdown, and port cleanup steps.

Surface-guided hand retargeting adds no new sample-wise cost term: it keeps
Mimic's original SBR/iCEM objective and sampling loop. Once per hand frame,
five fingertip queries can adjust the
wrist-to-tip and thumb-to-tip target vectors before sampling; surface lookup
does not run per candidate. The thresholds are `d_on=1 cm`, `d_off=1.5 cm`,
with a `3 mm` inward target offset. A missing or invalid surface sample sets
that fingertip's contact blend to zero, leaving its upstream SBR target
unchanged. There is no hand-target smoothing in this path.

MuJoCo queries use `mj_geomDistance` against configured collision geoms. A
real deployment path is also wired as an opt-in point-cloud SDF approximation:

1. Wrist D405 capture sends paired native-grid RGB and depth. The current mode
   uses the D405 left depth imager for both, so it does not resample depth onto
   a separate color grid; the SAM3 mask, depth grid, and intrinsics must match.
2. The SAM3 worker masks the object, filters depth points, and transforms the
   cloud into `pelvis_identity_fk` using camera intrinsics, the D405-to-wrist
   extrinsic, and robot state paired to exposure.
3. When explicitly enabled, the worker includes at most two score-ranked,
   1024-point clouds per camera in its LCM result. Thor caches only the latest accepted clouds from that hand's wrist
   camera and serves local ZMQ queries on port 5571. One same-side object is
   selected per hand frame (nearest the wrist target, optionally restricted by
   the SAM3 label configured on the gateway).
4. The hand runner transforms the five wrist-local fingertip targets with the
   fresh PATH waist-frame wrist pose. Nearest points and local PCA normals
   provide signed tangent-plane distances for contact-guided target generation.

The hardware query is a **local signed-distance approximation**, not a
watertight or global voxel SDF. Camera identity, depth/intrinsics alignment,
camera extrinsics, and the Televuer wrist-to-hand frame relation still need
physical calibration. Keep `contact_guidance.enabled: false` in
`configs/hands/brainco_revo2_mppi.yaml` until those checks are complete. The
default is off. Missing clouds, stale results, failed state pairing, or
insufficient point support produce no contact correction instead of reusing
an old surface.

After calibration, enable the hardware path by passing
`--publish-contact-clouds` to the remote D405 SAM3 worker and
`--hand-surface-query` to the Thor D405 gateway, then set
`contact_guidance.enabled: true` in the hand MPPI config. The query endpoint
is `tcp://127.0.0.1:5571` on Thor. The server accepts only current clouds,
exposure/state pairing error at or below 8 mm, and neighborhoods with enough
depth support. The input hand goal itself is not smoothed or retimed.

`mujoco-hand-xr` remains a separate explicit simulator opt-in: its launcher
passes `--surface-query-endpoint` to MuJoCo's collision-distance service. Set
`MUJOCO_SURFACE_QUERY_ENDPOINT` to change that local service address. This does
not enable the hardware point-cloud backend.

## MuJoCo MPPI test without XR or PATH

To hold the simulated G1 upright with HOMIE and exercise hand MPPI from a user
override pose, run:

```bash
pixi run -e sim mujoco-hand-test
```

This static test auto-starts HOMIE's stand/walk policy and begins with zero
keyboard velocity, which selects `stand.onnx`; viewer movement keys select
`walk.onnx`. The runner reads
`configs/hands/revo2_hand_override_fist.json` directly and reloads it while
active; `revo2_hand_override.json` provides an open-hand pose. Use
`--override-file PATH` to select another JSON file. This route starts no
Galaxy XR, Televuer, PATH bridge, or physical-robot transport.
The test record is
[`docs/test_logs/mujoco_homie_hand_mppi.md`](../../../docs/test_logs/mujoco_homie_hand_mppi.md).

For a repeated synthetic two-hand motion test, run:

```bash
pixi run -e sim mujoco-hand-motion-test
```

The launcher creates a temporary JSON override, starts the 30 Hz landmark
streamer, and feeds it to the same hand MPPI runner used by the static test.
Each five-second cycle holds the fist for 1 s, linearly opens over 1.5 s, holds
open for 1 s, and linearly closes over 1.5 s. Both hands interpolate all 25
three-dimensional points. The streamer's default input templates are
`configs/hands/revo2_hand_override_fist.json` and
`configs/hands/revo2_hand_override.json`; it never edits those templates.
`scripts/stream_mujoco_hand_motion.py --help` lists options for finite cycles,
timing, rate, source files, and output path. Its output updates use atomic file
replacement so the runner cannot observe a partially written JSON frame.

This test runs HOMIE's stand policy at zero command, keeps arm targets at the
configured default pose, and does not start Galaxy XR, Televuer, PATH arm
tracking, or a surface-query service. MuJoCo holds its initial pose for five
seconds and until HOMIE is publishing body commands, then starts physics. It
uses a temporary HOMIE config that holds arm joints at their configured default
and does not subscribe to `rt/arm_sdk`. It keeps the hand retargeter's
optimization and hard joint limits; fresh retarget outputs are published
directly. `--motion` and `--override-file` are mutually
exclusive. The original static test remains available as
`pixi run -e sim mujoco-hand-test`.

For a hardware runner, start the Televuer/PATH source and the BrainCo
`brainco_hand_service`, then run the hand environment with the G1/hand DDS
interface explicitly selected:

```bash
pixi run -e hand python -m boost.control.hand_mppi.runner \
  --publish --interface <dds-interface> --domain-id 0
```

On Jetson Thor, install both ARM64 environments: `thor-teleop` runs Televuer
and PATH with NumPy 1.26.4; `thor` runs the JAX CUDA 13 hand sampler with
NumPy 2. The two processes exchange hand frames on their existing local ZMQ
endpoint (`127.0.0.1:5566`). Use the guarded hand task, which requires a
non-loopback DDS interface:

```bash
pixi install -e thor-teleop
pixi install -e thor
BRAINCO_DDS_INTERFACE=eth0 pixi run -e thor hand-mppi-publish
```

See [`docs/g1_installation.md`](../../../docs/g1_installation.md) for the
JetPack compatibility notes and full arm/hand startup sequence.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13
