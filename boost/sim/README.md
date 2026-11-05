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

In the viewer, W/S, A/D, and Q/R request forward/back, lateral, and yaw velocity. O/P change the command scale, Space stops, and X toggles DreamWaQ standing/walking. Commands remain local on DDS loopback and the simulation LCM multicast URL.

The hand MPPI routes use the separate Pixi `hand` environment. In XR mode they disable PATH's binary grip publisher so only the MPPI process sends continuous Revo2 commands. The local JSON hand override defaults to `configs/hands/revo2_hand_override_fist.json`; select another with `--override-file PATH` or `MUJOCO_HAND_OVERRIDE_FILE`.

The surface query only targets the configured object geoms in `configs/sim/mujoco_revo2.yaml`. D405 RGB-D capture and depth-based BOOST estimation remain a separate perception path.

For XR certificate setup and occupied-port cleanup, see [the Galaxy XR runbook](../../docs/mujoco_galaxy_xr_runbook.md). Asset provenance is in [`mujoco/assets/ASSET_SOURCES.md`](mujoco/assets/ASSET_SOURCES.md).
