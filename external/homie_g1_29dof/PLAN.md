# HOMIE on G1 29DoF — Deployment Plan

Run the HOMIE (`stand.onnx` / `walk.onnx`) policy on the Unitree G1 29DoF using the
existing `url_g1_deploy` split architecture: the **onboard** `g1_lowlevel_bridge` owns
the safety boundary and low-level PD, and a new **policy-board** package
(`thirdparty/homie_g1_29dof`) runs the ONNX and feeds joint targets over LCM.

Status: **PLAN — no code written yet.** Review before implementation.

---

## 0. Source of truth

The two ONNX files were produced by the robofinals `G1HomiePolicyV2` class:

- `url_IKEA_challenge/robofinals/robofinals/core/mdp/actions/wbc_policy/policy/g1_homie_policy.py` → `G1HomiePolicyV2`
- config: `.../wbc_policy/config/g1_homie_v2.yaml` (gains/defaults/scales)
- joint groups: `.../wbc_policy/utils/g1_supplemental_info.py`

The obs/action spec below is copied from that code and is authoritative for **these**
files. OpenHomie/TWIST2 are structural references only (their obs/gains differ).

### Model I/O (verified with onnxruntime)
- input  `[N, 516]`  (86 dims × 6 history frames)
- output `[N, 15]`   (12 legs + 3 waist). **Arms are NOT policy-controlled.**
- select `stand.onnx` when `‖cmd_xy_yaw‖ < 0.05`, else `walk.onnx`.

### Joint ordering — the key simplification
`body_actuated_joints` (robofinals) is **identical** to Unitree `G1JointIndex`:

| idx | joint | idx | joint |
|----|-------|----|-------|
| 0 | L hip pitch | 15 | L shoulder pitch |
| 1 | L hip roll | 16 | L shoulder roll |
| 2 | L hip yaw | 17 | L shoulder yaw |
| 3 | L knee | 18 | L elbow |
| 4 | L ankle pitch | 19 | L wrist roll |
| 5 | L ankle roll | 20 | L wrist pitch |
| 6 | R hip pitch | 21 | L wrist yaw |
| 7 | R hip roll | 22 | R shoulder pitch |
| 8 | R hip yaw | 23 | R shoulder roll |
| 9 | R knee | 24 | R shoulder yaw |
| 10 | R ankle pitch | 25 | R elbow |
| 11 | R ankle roll | 26 | R wrist roll |
| 12 | waist yaw | 27 | R wrist pitch |
| 13 | waist roll | 28 | R wrist yaw |
| 14 | waist pitch | | |

`get_joint_group_indices` returns **sorted** indices, so HOMIE "body" order == LCM
`q[29]` order == this table. **No permutation needed.**
- policy 15 outputs → hardware indices **0..14** (legs 0–11, waist 12–14)
- arms → hardware indices **15..28** (from external arm-target input)

---

## 1. Observation layout (single frame = 86; buffer = 86 × 6 = 516)

From `G1HomiePolicyV2.compute_observation`. `n_joints = 29`.

| slice | content | source | scale |
|-------|---------|--------|-------|
| 0:3   | `cmd[vx, vy, wz]` | command input | `cmd_scale=[2,2,0.5]` |
| 3:4   | base-height cmd | command input | 1 (`height_cmd=0.74`) |
| 4:7   | torso `[roll, pitch, yaw]` cmd | command input | 1 (`rpy_cmd=[0,0,0]`) |
| 7:10  | base angular velocity | `RobotStatePacket.base_gyro` | `ang_vel_scale=0.5` |
| 10:13 | projected gravity | from `base_quat` (w,x,y,z) | 1.0 |
| 13:42 | `q − default_angles` (29) | `RobotStatePacket.q` | `dof_pos_scale=1.0` |
| 42:71 | `dq` (29) | `RobotStatePacket.dq` | `dof_vel_scale=0.05` |
| 71:86 | previous 15-dim action | internal state | 1.0 |

Notes:
- `default_angles` is length-15; obs subtracts it only from the first 15 joints
  (`padded_defaults[15:]=0`), so arm entries in `q−default` are absolute.
- **V2 uses base IMU only** — torso IMU is unused. We only need `base_quat`,
  `base_gyro` from `RobotStatePacket`.
- History warm-start: at reset fill all 6 frames with the **current** frame (not
  zeros) — zeros are out-of-distribution for the standing policy.
- `gait_indices`/`clock_inputs` are computed in the code but **not** placed in the V2
  obs; they can be ignored.

### Action → joint target
```
q_target[0:15] = policy_out(15) * action_scale(0.25) + default_angles(15)   # legs + waist
q_target[15:29] = arm_target(14)                                            # external
```
`default_angles (15) = [-0.1,0,0,0.3,-0.2,0,  -0.1,0,0,0.3,-0.2,0,  0,0,0]`

---

## 2. The compatibility gap (must be handled)

The onboard bridge (`g1_lowlevel_bridge.cpp` + `policy_parameters.hpp`) currently:
1. interprets the packet `action[29]` as a **residual**:
   `q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i]`
2. applies its **own** armature-derived gains (hip kp≈99, knee≈99, ankle≈28, …).

HOMIE needs **absolute** targets and its **own** gains (legs `150/150/150/200/40/40`,
waist `250`; kd `2/2/2/4/2/2 … 5/5/5`). **The LCM packet carries no gain field**, so
gains must be set onboard. Two onboard changes are required:

### 2a. Add an absolute-target action mode (onboard)
Add a mode where the 29 packet floats are treated as **absolute `q_target`** and passed
through without the `default + scale` residual transform. Options:
- reuse a `flags` bit in `PolicyActionPacket` (e.g. `kActionFlagAbsoluteTarget`), or
- a `--action-encoding {residual,absolute}` CLI flag on the bridge.

Recommended: CLI flag (explicit, no per-packet ambiguity). Files:
`g1_ws/include/lcm_protocol.hpp`, `g1_ws/src/g1_lowlevel_bridge.cpp`.

### 2b. Set HOMIE gains (onboard)
Provide a HOMIE gain set for indices 0..14 from `g1_homie_v2.yaml`. **Open item:** arm
gains (15..28) are not in that config — source them from OpenHomie's upper-body PD (or
pick conservative stiff values, e.g. shoulder/elbow kp≈40–100, wrist kp≈20, and confirm
on hardware). Keep these behind the same flag so `null`/`lcm`-residual behavior is
unchanged. File: `g1_ws/include/policy_parameters.hpp` (add a parallel HOMIE table).

> Reinstate the disabled safety fallbacks (README §7) before running real motion:
> LowState stale timeout, action stale timeout → damping, and honor the emergency-stop
> flag. Currently only e-stop/power cut reliably stops the robot.

---

## 3. Policy-board package: `thirdparty/homie_g1_29dof`

```
thirdparty/homie_g1_29dof/
├── PLAN.md                     # this file
├── README.md
├── pyproject.toml              # deps: numpy, onnxruntime, pyyaml, lcm, torch(optional)
├── homie_g1_29dof/
│   ├── __init__.py
│   ├── config/g1_homie_v2.yaml # copied gains/defaults/scales (self-contained)
│   ├── lcm_protocol.py         # struct (un)pack mirroring lcm_protocol.hpp
│   ├── homie_policy.py         # vendored G1HomiePolicyV2 obs+inference (no isaac deps)
│   ├── command_source.py       # cmd[vx,vy,wz], height, rpy  (keyboard/joystick/LCM)
│   ├── arm_target_source.py    # external 14-dim arm targets (LCM channel)
│   ├── runner.py               # 50 Hz loop
│   └── models/                 # stand.onnx, walk.onnx (copied or symlinked)
└── scripts/run_homie.py
```

### 3a. `lcm_protocol.py` — must byte-match `g1_ws/include/lcm_protocol.hpp` (`#pragma pack(1)`)
- `RobotStatePacket` (subscribe `G1_ROBOT_STATE`), 461 bytes:
  `<I H H Q Q q B  4f 3f 3f  4f 3f 3f  29f 29f 29f`
  = magic, version, flags, sequence, lowstate_sequence, send_time_us, mode_machine,
  base_quat, base_gyro, base_accel, torso_quat, torso_gyro, torso_accel, q, dq, tau_est.
  Validate `magic == 0x31545347` ("GST1"), `version == 1`.
- `PolicyActionPacket` (publish `G1_POLICY_ACTION`), 140 bytes:
  `<I H H Q q 29f` = magic(`0x31414347` "GCA1"), version(1), flags, sequence,
  send_time_us, action[29].
  flags: `emergency_stop=1<<0`, `left_grip=1<<1`, `right_grip=1<<2` (+ new absolute-target
  bit if chosen in 2a).

### 3b. `homie_policy.py`
Vendor `G1HomiePolicyV2`'s `compute_observation` / `set_observation` / `get_action`,
stripped of isaac/robot_model deps. Since body order == hardware order, feed the packet
`q`/`dq` directly and build the obs dict as:
```
observation = {
  "q":  q29,  "dq": dq29,
  "floating_base_pose": [0,0,0, *base_quat],   # only [3:7] used
  "floating_base_vel":  [0,0,0, *base_gyro],   # only [3:6] used
}
```
Load both ONNX sessions; switch stand/walk on `‖cmd‖`. Output 15 → `q_target[0:15]`.

### 3c. `runner.py` (50 Hz — `simulation_dt 0.005 × control_decimation 4`)
Loop:
1. read latest `RobotStatePacket` (state pub is 100 Hz; use most recent).
2. read latest command + arm targets.
3. `policy.set_observation(obs)` → `policy.get_action()` → 15 leg/waist targets.
4. assemble `q_target[29]` = [15 policy][14 arm targets].
5. send absolute `q_target` (mode 2a) or, if not adding the onboard mode, invert the
   onboard residual transform per joint before publishing (fallback; still needs 2b gains).
6. publish `PolicyActionPacket` with sequence++ and grip/e-stop flags.
7. startup: hold at `default_angles` + current arm pose and ramp in over ~1–2 s.

### 3d. `arm_target_source.py` (external input — chosen requirement)
Subscribe a new LCM channel (e.g. `G1_ARM_TARGET`, 14 floats, hardware arm order 15..28)
fed by teleop/IK (TWIST2 or exoskeleton). On stale/no input, hold last target (or a safe
default arm pose). Keep the format documented so TWIST2 can publish to it.

---

## 4. Bring-up sequence (safety-first)

1. Build onboard bridge with LCM: verify `--action-source lcm --dry-run-actions` logs
   `First action received` while robot stays in damping (README §8).
2. Run `run_homie.py` against a **recorded/synthetic** state first; confirm 516-obs shape,
   50 Hz publish, sane `q_target` near default pose, correct packet size (140 B).
3. On robot, hung/supported: enable HOMIE gains + absolute mode with `cmd=0` (stand.onnx).
   Confirm it holds the default standing pose, arms follow arm-target input.
4. Small `vx` → confirm switch to walk.onnx and stepping. Keep e-stop in hand.
5. Only then remove support.

---

## 5. Open items to confirm before coding
- [ ] Onboard mode (2a): reuse a `flags` bit vs `--action-encoding` CLI flag.
- [ ] Arm PD gains (2b, idx 15..28) — source from OpenHomie or set conservative values.
- [ ] Command source: keyboard/joystick on the policy board, or an LCM command channel.
- [ ] Arm-target channel name/format and which system publishes it (TWIST2 vs exoskeleton).
- [ ] onnxruntime vs torch on the policy board (onnxruntime-only avoids the torch dep;
      the vendored code uses torch just for tensor plumbing — can be replaced with numpy).
- [ ] Reinstate onboard stale-timeout / e-stop fallbacks before real motion (README §7).
```
