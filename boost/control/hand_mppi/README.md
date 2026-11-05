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

## Runtime

```bash
pixi install -e hand
pixi run -e hand python -m boost.control.hand_mppi.runner
```

For the full Galaxy XR + Revo2 simulation, use `pixi run -e sim mujoco-hand-xr`. This disables PATH's binary grip DDS publisher while MPPI is active. The local JSON test uses `pixi run -e sim mujoco-hand-test`, defaults to `configs/hands/revo2_hand_override_fist.json`, and can be changed with `--override-file PATH`.

The XR hand route can query object surface hits from MuJoCo over local ZMQ. The active scene's collision geom names are configured in `configs/sim/mujoco_revo2.yaml`. D405 RGB-D geometry and depth synchronization remain separate from this simulation-only contact query.

See the [MuJoCo README](../../sim/README.md) for launch details and the [Galaxy XR runbook](../../../docs/mujoco_galaxy_xr_runbook.md) for the headset URL, shutdown, and port cleanup.
