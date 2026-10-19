# HOMIE on G1 29DoF — onboard deployment

Runs the HOMIE `stand.onnx` / `walk.onnx` policy **directly on the G1**, talking
`rt/lowcmd` + `rt/lowstate` over DDS through `unitree_sdk2py` — the same DDS binding
xr_teleoperate uses. No LCM bridge, no C++ changes, and no `unitree_interface` dependency
— everything the robot needs lives in this package, and one DDS-owning process can safely
own both the leg/waist control loop and the `rt/arm_sdk` arm-teleop subscription (see
"Arms via xr_teleoperate" below for why that used to require two *DDS* processes).

`run_homie.py` is still a single command to launch, but it auto-spawns a **second OS
process for ONNX inference** (`homie_policy_process.py`, talking back over shared memory
-- see `policy_ipc.py`). This is unrelated to the DDS story above: it exists because
CPython's GIL means a slow `onnxruntime` call inside the same process as the `rt/lowcmd`
writer thread can stall that writer for tens of ms even with CPU clocks pinned, which was
measured to throttle teleop arm tracking well below xr_teleoperate's own publish rate.
Moving inference to its own process (own GIL) removes that stall entirely. The layout
mirrors the modular structure of `~/06_IKEA/TWIST2/deploy_real`.

Policy switch: **`‖cmd[vx,vy,wz]‖ < 0.2` → `stand.onnx`, otherwise `walk.onnx`**
(the threshold lives in `deploy/config/g1_homie_v2.yaml` as `stand_cmd_threshold`). The
authoritative robofinals policy uses 0.05; this deploy defaults to 0.2 as requested — see
`DEPLOY_GUIDE.md` §2.

**Deploying on hardware? Follow [`DEPLOY_GUIDE.md`](DEPLOY_GUIDE.md)** — always start with
the `--dry-run` (reads state + runs the policy at 50 Hz but keeps the robot limp).

## Layout

```
thirdparty/homie_g1_29dof/
├── ckpts/                         # stand.onnx, walk.onnx  (516 -> 15)
├── PLAN.md                        # original split-architecture design (superseded here)
└── deploy/
    ├── run_homie.py               # I/O process entry point: DDS + 50 Hz tick (real robot)
    ├── homie_policy_process.py    # policy process entry point: onnxruntime only, no DDS
    ├── policy_ipc.py              # shared-memory IPC between the two processes above
    ├── run_homie_sim.py           # sim2sim entry point (MuJoCo, single process)
    ├── homie_policy.py            # vendored G1HomiePolicyV2 (numpy + onnxruntime, no isaac/torch)
    ├── homie_utils.py             # gravity / quat helpers (vendored)
    ├── command_source.py          # wireless-controller sticks -> cmd_vel
    ├── arm_target_source.py       # 14-dim arm targets (hold / arm_sdk / redis)
    ├── config/g1_homie_v2.yaml    # policy spec: obs/action scales, defaults, threshold (authoritative)
    ├── robot_control/
    │   ├── config.py              # hardware config loader
    │   ├── g1_wrapper.py          # G1 DDS interface (unitree_sdk2py)     [real]
    │   ├── g1_mujoco_wrapper.py   # G1 MuJoCo interface, same API         [sim]
    │   └── configs/g1_homie.yaml  # 29-dim gains, joint map, cmd limits, arm source
    └── tests/test_offline.py      # runs without a robot
```

Only joints **0..14** (12 legs + 3 waist) are policy-controlled. Arms **15..28** are
driven by `arm_target_source` and hold the default pose unless an external channel feeds
14-dim targets.

## Run

```bash
cd thirdparty/homie_g1_29dof/deploy
python run_homie.py --net <iface> --dry-run  # ALWAYS FIRST: reads + computes, robot stays limp
python run_homie.py --net <iface>            # live; e.g. eth0 / enp...
# optional: --stand ../ckpts/stand.onnx --walk ../ckpts/walk.onnx --device cpu
```

Bring-up (safety-first, keep the e-stop in hand):
1. **START** on the remote → ramps from the current pose to the default pose.
2. **A** → starts the policy (eases in over `--ramp-time`, default 1 s).
3. Left stick = vx/vy, right stick = wz. Release sticks → `‖cmd‖=0` → stand.
4. **Select** (or Ctrl+C) → exits to zero-stiffness damping.

Flip `SIGN_VX/VY/WZ` in `command_source.py` if an axis is reversed on your unit.

### Sim2sim in MuJoCo (test before hardware)

Same policy/config/gains, MuJoCo backend instead of DDS. Run from an env that has
`mujoco` installed (e.g. your `~/01_NRF/url_sim2real` env) with a display:

```bash
cd thirdparty/homie_g1_29dof/deploy
python run_homie_sim.py    # defaults to the g1_29dof scene in url_sim2real
# or: --xml /path/to/scene_29dof.xml --init-height 0.78 --no-realtime
```

In the viewer: **W/S** = vx, **A/D** = vy, **Q/E** = wz, **space** = stand, close window
to quit. Watch the `[stand]`/`[walk]` label flip as `‖cmd‖` crosses the threshold. The
MuJoCo env (`robot_control/g1_mujoco_wrapper.py`) implements the same interface as the DDS
env, so the control loop is identical to hardware.

### Arms via xr_teleoperate

`arm_source: "arm_sdk"` is the **default** in `robot_control/configs/g1_homie.yaml`. Just
run xr_teleoperate normally (e.g. on the Thor board, publishing `rt/arm_sdk`) and start
`run_homie.py` — one process, one DDS binding (`unitree_sdk2py`) subscribes `rt/arm_sdk`
arm targets and packs them into our lowcmd alongside the leg/waist policy output. No
changes to xr_teleoperate, no second process, no Redis. Details in `DEPLOY_GUIDE.md` §5.

## Dependencies

- `numpy`, `onnxruntime`, `pyyaml` — policy + config (present on the dev box).
- `unitree_sdk2py` — the DDS binding for both `rt/lowcmd`/`rt/lowstate` control and the
  `rt/arm_sdk` teleop subscription. The only DDS dependency; `unitree_interface` is not
  used (see `robot_control/g1_wrapper.py` docstring for why mixing the two in one process
  is unsafe).
- `redis` — only if you deliberately switch `arm_source: redis` (legacy two-process
  bridge, see `deploy/arm_sdk_to_redis.py`); not needed for the default `arm_sdk` mode.
- `mujoco` — only for `run_homie_sim.py` (sim2sim); already in the `url_sim2real` env.

## Verification

`deploy/tests/test_offline.py` checks obs assembly (516), the 0.2 stand/walk switch,
and the warm-start, using only numpy + onnxruntime:

```bash
python deploy/tests/test_offline.py
```

The observation/action pipeline was cross-checked against the authoritative
`robofinals` `G1HomiePolicyV2` and matches to floating-point precision (~1e-7); the only
intended difference is the `stand_cmd_threshold` (0.05 → 0.2).

## Notes / gains

- Control rate 50 Hz (`simulation_dt 0.005 × control_decimation 4`).
- Leg/waist PD from `g1_homie_v2.yaml` (`kp 150/150/150/200/40/40 … 250`,
  `kd 2/2/2/4/2/2 … 5`). Arm PD (idx 15..28) is a conservative set from the TWIST2 G1
  config — confirm on hardware before removing support.
- HOMIE V2 uses the **base/pelvis IMU only**; the torso IMU is unused.
