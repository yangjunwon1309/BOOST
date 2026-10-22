# Revo2 hand retargeting

BOOST connects Televuer landmarks to BrainCo Revo2 with the pinned
[`mimic_retargeter_lab`](https://github.com/mimicrobotics/mimic_retargeter_lab)
checkout in `upstream/`. The current Revo2 model wrapper lives in
`brainco_revo2_model.py`; it uses the official single-hand MJCF and maps its
eleven MJCF joints onto the six BrainCo command channels.

## Existing PATH and DDS contracts

| Direction | Contract |
|---|---|
| Televuer → PATH → runner | ZMQ `vr_pose_debug` on `tcp://127.0.0.1:5566` |
| PATH hand payload | Schema 1, two `(25, 3)` float32 arrays in metres, validity `[left, right]`, sequence, monotonic timestamp, coordinate-frame id 1 |
| Runner → Revo2 | Unitree DDS `MotorCmds_` on `rt/brainco/{left,right}/cmd` |
| Revo2 → runner | Unitree DDS `MotorStates_` on `rt/brainco/{left,right}/state` |
| PATH arm route | Existing arm packet and HOMIE `rt/lowcmd` ownership stay unchanged |

The runner treats Televuer positions as the same 25-point topology consumed by
Mimic's `ManusHandModel`; the WebXR orientation quaternions are not used by this
position-based retargeter. It checks the schema, coordinate frame, sample age,
validity, shape, and finiteness before optimizing either hand.

## Runtime behavior

`runner.py` uses Mimic's sampling-based iCEM retargeter with the local Revo2
model. Its six radian targets are clamped to official BrainCo limits, converted
through `BrainCoRevo2Adapter` to `[0, 1]`, then rate-limited per channel. Live
publishing waits for fresh Revo2 state on each side and seeds the optimizer from
the measured pose. After tracking has started, a stale hand stream moves toward
the configured open pose at the same rate limit. HOMIE remains the sole writer
to `rt/lowcmd`.

The runner is read-only unless `--publish` is supplied:

```bash
pixi install -e hand
pixi run -e hand python -m boost.control.hand_mppi.runner
```

Start the full desktop Galaxy XR → Televuer → PATH → HOMIE → MuJoCo loop with:

```bash
pixi run -e sim mujoco-hand-xr
```

This mode starts the runner in Pixi's separate `hand` environment and disables
PATH's binary grip DDS publisher to avoid competing writes to the Revo2 command
topics. `configs/hands/brainco_revo2_mppi.yaml` holds the model, optimizer,
input-age, update-rate, and command-rate settings.
The [MuJoCo + Galaxy XR runbook](../../../docs/mujoco_galaxy_xr_runbook.md)
covers the headset URL, shutdown, and port cleanup steps.

In `mujoco-hand-xr`, the runner also asks the MuJoCo process for object-surface
hits over a local ZMQ request/reply endpoint. The simulator uses ten invisible,
non-colliding mocap probes and `mj_geomDistance` against the configured object
collision geoms. Those five hits per hand adjust the upstream wrist-to-tip and
thumb-to-tip targets before iCEM sampling; SDF queries do not run per sample.
Contact projection uses `d_on=1 cm`, `d_off=1.5 cm`, and a `3 mm` inward target
offset. If a query is missing or invalid, that frame uses the original SBR
targets. The local `configs/sim/mujoco_revo2.yaml` currently has no object geom
names configured, so the new path is safely inactive until an object collision
geom is added to the scene and listed there. MuJoCo contact guidance is a sim
backend; mesh-grid and point-cloud backends for hardware have not been added.
Set `MUJOCO_SURFACE_QUERY_ENDPOINT` to change the local service address.

## MuJoCo MPPI test without XR or PATH

To hold the simulated G1 upright with HOMIE and exercise hand MPPI from a user
override pose, run:

```bash
pixi run -e sim mujoco-hand-test
```

This standalone test auto-starts HOMIE's default standing pose without loading
its locomotion policy. The runner reads
`configs/hands/revo2_hand_override_fist.json` directly and reloads it while
active; `revo2_hand_override.json` provides an open-hand pose. Use
`--override-file PATH` to select another JSON file. This route starts no
Galaxy XR, Televuer, PATH bridge, or physical-robot transport.
The test record is
[`docs/test_logs/mujoco_homie_hand_mppi.md`](../../../docs/test_logs/mujoco_homie_hand_mppi.md).

For a hardware runner, start the Televuer/PATH source and the BrainCo
`brainco_hand_service`, then run the hand environment with the G1/hand DDS
interface explicitly selected:

```bash
pixi run -e hand python -m boost.control.hand_mppi.runner \
  --publish --interface <dds-interface> --domain-id 0
```
