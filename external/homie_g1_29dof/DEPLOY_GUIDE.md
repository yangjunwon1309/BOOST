# HOMIE G1 29DoF — Deployment Guide

Step-by-step for running `stand.onnx` / `walk.onnx` on a real Unitree G1 29DoF, onboard,
over DDS. Read all of §1–§2 before touching the robot. **Keep the hardware e-stop in
hand at every step.**

---

## 1. Prerequisites

On the robot's onboard computer (or the machine on the robot LAN that runs control):

- Python 3.8+ with `numpy`, `onnxruntime`, `pyyaml`, `unitree_sdk2py` (the `homie` conda
  env has all of these). `robot_control/g1_wrapper.py` uses `unitree_sdk2py` for
  `rt/lowcmd`/`rt/lowstate` -- there is no `unitree_interface` dependency, so nothing
  needs the TWIST2 native binding.
- For arm teleop: a running xr_teleoperate session publishing `rt/arm_sdk` (see §5). It
  uses the same `unitree_sdk2py` binding already required above, so arm teleop needs no
  extra native dependency and runs safely in the SAME process as the control loop.
- The robot in **low-level control mode** / developer mode so publishing `rt/lowcmd`
  actually drives the motors, and the built-in sport/motion service that fights low-level
  control is stopped.

The `homie` conda env (Python 3.8, aarch64) was created like this and has everything
above:

```bash
conda create -n homie python=3.8
conda activate homie
pip install cyclonedds==0.10.2 numpy onnxruntime pyyaml opencv-python
pip install -e /home/unitree/00_workspace/unitree_sdk2_python --no-deps
# only if you use the legacy Redis arm bridge (§5): pip install redis
```

Find your network interface to the robot:

```bash
ip -brief addr        # e.g. eth0, enpXsY — the NIC on the robot's 192.168.123.x subnet
```

Quick check that the policy files load (safe, no robot):

```bash
cd thirdparty/homie_g1_29dof/deploy
python tests/test_offline.py
```

Expect: `OK: threshold=0.05, ... all checks passed`.

---

## 2. What the controller does

- 50 Hz loop. Reads the wireless controller → `cmd_vel = [vx, vy, wz]`.
- `‖cmd_vel‖ < stand_cmd_threshold` (currently **0.05**, in `config/g1_homie_v2.yaml`) →
  `stand.onnx`; otherwise `walk.onnx`.
- Policy outputs 15 targets (12 legs + 3 waist, indices 0..14).
- Arms (15..28) come from the arm source: **hold default** (default) or **xr_teleoperate**
  via `rt/arm_sdk`.
- Sends absolute 29-DoF `q_target` with HOMIE PD gains on `rt/lowcmd`.

> The current snapshot and authoritative robofinals `G1HomiePolicyV2` both use
> `stand_cmd_threshold: 0.05`. Table-assembly teleoperation in
> `url_IKEA_challenge` uses a different policy and has no stand/walk ONNX switch.

---

## 2b. Sim2sim in MuJoCo (recommended before any hardware)

Validate the whole policy loop in MuJoCo first — same `HomiePolicyV2`, config, gains, and
stand/walk switch, just a MuJoCo backend instead of DDS. Run from an env with `mujoco`
installed and a display (e.g. your `~/01_NRF/url_sim2real` env):

```bash
cd thirdparty/homie_g1_29dof/deploy
python run_homie_sim.py
# uses ~/01_NRF/url_sim2real/.../g1/scene_29dof.xml by default
# tune: --xml <path> --init-height 0.78 --sim-dt 0.005 --no-realtime
```

In the viewer window: **W/S**=vx, **A/D**=vy, **Q/E**=wz, **space**=stand, close to quit.
Confirm the robot stands with `‖cmd‖=0` (`[stand]`), and that pushing `vx` past the
threshold flips to `[walk]` and it steps without falling. If it sags/drops at startup,
lower `--init-height` toward the HOMIE target (~0.74). Only move on to hardware once sim
stand + walk look stable.

## 3. Dry run first (no motion — do this every session)

The dry run connects, reads state, runs the policy at 50 Hz, and prints diagnostics, but
holds the robot **limp in damping** and never commands policy motion. Hang the robot on
its stand or have it supported.

```bash
cd thirdparty/homie_g1_29dof/deploy
python run_homie.py --net <iface> --dry-run
```

Verify from the console:
- It connects and prints the robot name / motor count (DDS OK).
- `grav_z` ≈ **-1.0** when the robot is upright (IMU + gravity projection OK). Tilt the
  base and watch it change.
- Move the left stick → `|cmd|` and `cmd=[vx,vy,wz]` change; label flips
  `stand → walk` as you cross the threshold. Release → back to `stand`.
- `legΔ` (policy target vs default) stays small (< ~0.3 rad) at rest — no wild outputs.
- If using arms: `armΔ` tracks your teleop motion; otherwise it stays 0.

Stop with **Select** or Ctrl+C (exits to damping). Only proceed when all of the above look
right.

---

## 4. Live run (supported, then free)

Keep the robot **hung / supported** for the first live run.

```bash
python run_homie.py --net <iface>
# optional: --stand ../ckpts/stand.onnx --walk ../ckpts/walk.onnx --device cpu --ramp-time 1.5
```

Bring-up sequence (prompted on the console):
1. **START** on the remote → ramps from the current pose to the default pose over ~2 s.
2. **A** → starts the policy. It eases from the default pose into policy output over
   `--ramp-time` (default 1 s).
3. Sticks: left = vx/vy, right = wz. Released sticks → `‖cmd‖ = 0` → `stand.onnx`.
4. Give a small forward `vx` past 0.05 → switches to `walk.onnx` and steps.
5. **Select** / Ctrl+C → exits to zero-stiffness damping.

Only remove the support once standing + small stepping look stable.

If any axis drives the wrong way, flip `SIGN_VX` / `SIGN_VY` / `SIGN_WZ` at the top of
`command_source.py`.

---

## 5. Arms via xr_teleoperate (default)

HOMIE owns the full `rt/lowcmd` (legs+waist+arms), so the usual `rt/arm_sdk` overlay is
not applied by firmware. Instead this controller **subscribes** `rt/arm_sdk` and packs the
arm joint targets xr_teleoperate computes into our lowcmd. No changes to xr_teleoperate.

`arm_source: "arm_sdk"` is already the default in `robot_control/configs/g1_homie.yaml`,
so normally you don't need to touch it:

1. Start xr_teleoperate as usual on the Thor board (its motion mode publishes 14 arm
   joint targets `sol_q` to `rt/arm_sdk`, order = left arm 7 then right arm 7 = indices
   15..28).
2. Start this controller (`run_homie.py --net <iface>`). It reads the arm targets each
   cycle; while the teleop enable weight (`kNotUsedJoint0.q`) is low it holds the last
   arm command, so a paused teleop never snaps the arms.

Both processes share DDS domain 0 on the same `--net` interface. The arm targets are the
IK solution `sol_q` from `xr_teleoperate/teleop/teleop_hand_and_arm.py` — the arms track
your controllers while HOMIE keeps the base balanced and walking.

> **Arm responsiveness / `rt/lowcmd` publish rate:** `robot_arm.py` on the teleop side
> publishes `rt/arm_sdk` at ~250 Hz (`control_dt = 1/250`), but the HOMIE leg/waist
> policy only ticks at 50 Hz. Naively sending one combined `rt/lowcmd` per policy tick
> would throttle the arm portion down to 50 Hz too -- a 5x drop from what xr_teleoperate
> alone (via the firmware's native `rt/arm_sdk` overlay) achieves. `run_homie.py` avoids
> this by running the actual `rt/lowcmd` publish on a **separate thread**
> (`G1RealWorldEnv.start_writer`, default `--writer-rate-hz 250`) that re-samples the arm
> source fresh on every write, while only the leg/waist portion is refreshed once per
> policy tick. Real achieved rate depends on CPU headroom (Python's GIL means the
> lowstate reader, arm_sdk reader, policy thread, and writer thread all take turns), but
> it's consistently well above the old 50 Hz ceiling. If arms still feel soft/slow after
> this, that's very likely tuned into xr_teleoperate itself (`robot_arm.py`'s
> `kp_low=80`/`kp_wrist=40`, well below the `kp_high=300` used for locked non-arm
> joints, plus its own `arm_velocity_limit` clip on the target *before* it ever reaches
> `rt/arm_sdk`) rather than anything in this repo.

> **Why this is safe in one process:** `run_homie.py`'s own `rt/lowcmd`/`rt/lowstate` I/O
> and the `rt/arm_sdk` subscription both go through `unitree_sdk2py` — the SAME DDS
> binding. Earlier revisions of this controller used `unitree_interface` (TWIST2's native
> pybind DDS module) for `rt/lowcmd`/`rt/lowstate` and only used `unitree_sdk2py` for the
> arm subscription, which meant loading two *different* native `libddsc.so` builds inside
> one process. That combination was confirmed (empirically, in this repo) to fail: a
> second DDS domain participant, from a different libddsc build than the first, cannot
> initialize on the same domain and raises `"[ChannelFactory] create domain error"`. Since
> `robot_control/g1_wrapper.py` now uses `unitree_sdk2py` exclusively, there's only ever
> one DDS binding loaded, and `ChannelFactoryInitialize` is an idempotent per-process
> singleton, so the arm source's own init call is a safe no-op.

Legacy alternative: `arm_source: "redis"` reads a 14-float JSON channel (`arm_redis_key`)
via a separate bridge process (`deploy/arm_sdk_to_redis.py`). This predates the
`unitree_sdk2py`-only rewrite above and is no longer necessary for safety — keep it only
if you have another reason to decouple the arm feed over Redis (e.g. a different board/
process boundary unrelated to the DDS-binding conflict).

---

## 6. Tuning / files

| What | Where |
|------|-------|
| stand/walk threshold | `deploy/config/g1_homie_v2.yaml` → `stand_cmd_threshold` |
| cmd_vel limits, deadzone | `deploy/robot_control/configs/g1_homie.yaml` → `max_cmd`, `cmd_deadzone` |
| PD gains (29-dim), default pose | `deploy/robot_control/configs/g1_homie.yaml` |
| arm source | `g1_homie.yaml` → `arm_source` |
| stick axis signs | `deploy/command_source.py` |
| startup ease-in | `--ramp-time` |

Do **not** edit the obs/action scales in `g1_homie_v2.yaml` — they are the contract the
ONNX files were trained with (verified byte-equivalent to the authoritative policy).

---

## 7. Safety checklist

- [ ] E-stop in hand.
- [ ] Robot supported/hung for dry run and first live run.
- [ ] Dry run passed (grav_z ≈ -1, cmd + switching correct, small legΔ).
- [ ] Correct `--net` interface; only one low-level publisher on `rt/lowcmd`.
- [ ] Firmware sport/motion service stopped so it doesn't fight low-level control.
- [ ] Arm teleop (if used) verified in dry run before the live run.
