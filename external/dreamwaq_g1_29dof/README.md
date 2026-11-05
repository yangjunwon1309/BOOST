# HOMIE-base DreamWaQ G1 29-DoF

ROS1 자율보행 연결과 실제 G1 실행 순서는 [ROS_AUTONOMY.md](ROS_AUTONOMY.md)를 참고하세요.

This package keeps the HOMIE hardware path and replaces only the policy process
with DreamWaQ.

- Parent process owns DDS, `rt/lowstate`, `rt/lowcmd`, and `rt/arm_sdk`.
- The policy runs in an isolated process. Each parent state carries a sequence
  and monotonic timestamp; only the action naming that exact state is accepted.
- The policy's previous-action observation is the last parent-accepted actor
  output, matching training and sim2sim. Writer targets remain separate
  diagnostics because startup blending and safety clipping change them.
- The HOMIE 250 Hz writer resamples the latest leg/waist policy target and the
  arm target. While inference is pending it holds the preceding safe target.
- The default action deadline is 15 ms. One miss holds the preceding target;
  three consecutive misses stop the writer and return the robot to damping.
  A writer-side action-age watchdog independently enters damping if the parent
  loop itself stalls, and keeps publishing damping until shutdown.
- Lower-body gains and the DreamWaQ default lower-body reference are in
  `deploy/robot_control/configs/g1_homie.yaml`; arm gains/targets remain owned
  by `rt/arm_sdk`.
- The previous DreamWaQ implementation is preserved in
  `../dreamwaq_g1_29dof_backup/`.

## Offline checks

```bash
cd /home/url/00_workspace/g1_deploy_dreamwaq_port
python -m compileall -q thirdparty/dreamwaq_g1_29dof
PYTHONPATH=thirdparty/dreamwaq_g1_29dof \
  python thirdparty/dreamwaq_g1_29dof/tests/test_g1_29dof_sim2real.py
PYTHONPATH=thirdparty/dreamwaq_g1_29dof/deploy \
  python thirdparty/dreamwaq_g1_29dof/deploy/tests/test_offline.py
```

## Bundled exports

The following five policy bundles are retained. Other checkpoint bundles
were removed from this branch at the user's request; their prior versions
remain recoverable from Git history. No history rewrite was performed.

| Export | Contents / usage |
| --- | --- |
| `new46_4_seed42_49999` | NEW46-4 final 50K run (iteration 49999), seed42; original training checkpoint and four export files; 15 actions, 30-frame/2790-D CENet history; current hardware loader expects 5 frames and rejects this export, so a separate compatibility update is required; not the default policy |
| `new2_3_main_plus50k_249999` | NEW2-3 main checkpoint and export |
| `new2_3_backpack_lower8cm_299999` | NEW2-3 backpack checkpoint, export and payload metadata; Nature switch target |
| `new3_4_widerange_quality_390000` | NEW3-4 checkpoint and export; default direct deployment policy |
| `new3_4_backpack_440000` | NEW3-4 backpack checkpoint, export and payload metadata; autonomy launcher policy |

Each bundle includes its resumable training checkpoint and deployment files.
Bundling does not apply the backpack MuJoCo model or modify hardware settings.

## G1 command

Install the local package into the G1 environment once, then run the HOMIE
bring-up. `START` performs the HOMIE default-pose ramp, `A` starts DreamWaQ,
and `Select` returns to damping.

The default export is now `new3_4_widerange_quality_390000`, from
`NEW3_4_widerange_quality_from240k_plus150k_server2_1001/final_390000`.
Its original manifest and TorchScript files are bundled unchanged, together
with the resumable `model_390000.pt` checkpoint.

NEW3 starts in **standing**. Press **X** once to enter walking and press it
again to return to standing. Holding X does not repeat the toggle. Standing
forces all three velocity commands to zero, including UDP/DDS commands and
wireless-stick input. Walking retains mode=1 even when the velocity command
is zero. Mode changes preserve the CENet history and accepted actor-output
previous-action history.

`--locomotion-mode walking` selects walking initially, matching the supplied
sim2sim mode. `--mode-toggle-button Y` or `B` changes the toggle button.
START, A and Select keep their existing bring-up and damping functions.
The same toggle and command gating apply in `--dry-run` and read-only
`--shadow`; shadow CSV records the requested mode and effective commands.

The actor current frame is 100-D, with mode at index 99; adding the 32-D
CENet latent gives a 132-D actor input. CENet retains five 93-D measurable
frames (465-D term-major history), without the mode or velocity command.
Standing=0 and walking=1 match the training/sim2sim export receipt.
The serialized waist gains are 300/3 and waist torque limits are 88/50/50 Nm.
Training command bounds are vx=[-0.8,1.0], vy=[-0.8,0.8], yaw=[-1.0,1.0];
the hardware config defaults remain 0.6/0.4/0.5. Explicit `--max-vx`,
`--max-vy` and `--max-yaw` set the requested caps. Each signed direction is
limited separately by the selected export's training range; the reverse bound
does not reduce the maximum forward command.

For NEW3_4, `--max-vx 0.95 --max-vy 0.75 --max-yaw 0.75` maps full stick
travel to forward **0.95 m/s**, reverse **0.8 m/s**, lateral **0.75 m/s**,
and yaw **0.75 rad/s**. A 10% center deadzone is followed by linear scaling
to each direction's own maximum. UDP/DDS commands use the same signed bounds;
standing still forces every velocity command to zero.

The sim2sim `--force_physics_substep_latency_ms 10` emulates action latency.
The hardware runtime measures its actual state/inference/writer timings and
does not insert an additional 10 ms sleep.

The bundled `g1_homie.yaml` uses `cmd_source: udp`: it accepts the XR
intervention client's body-frame `[vx, vy, wz]` JSON from Thor
`192.168.123.2` on G1 PC2 UDP port `15001`. Physical G1 wireless sticks always
override that feed, and the command becomes zero if no valid packet arrives for
0.3 seconds. Sender IP, session and sequence are checked before a command is
accepted. `cmd_source: dds` retains the legacy `rt/nav_cmd` option, and
`cmd_source: wireless` runs from the physical controller only.

`dreamwaq-g1-29dof` is the only hardware execution entry point. The
`dreamwaq_interface.scripts.g1_29dof_dreamwaq` module is offline-only and
rejects its removed `real`/`shadow` commands so the old latest-value IPC path
cannot be selected accidentally.

```bash
cd /home/unitree/00_workspace/g1_deploy/thirdparty/dreamwaq_g1_29dof
/home/unitree/miniconda3/envs/emer_tw/bin/python -m pip install -e . --no-deps

/home/unitree/miniconda3/envs/emer_tw/bin/dreamwaq-g1-29dof \
  --net eth0 \
  --export "$PWD/checkpoints/new3_4_widerange_quality_390000/exported" \
  --locomotion-mode standing \
  --mode-toggle-button X \
  --max-vx 0.95 --max-vy 0.75 --max-yaw 0.75 \
  --device cpu \
  --policy-timeout-ms 15 \
  --max-policy-timeouts 3
```

### Switch between DreamWaQ and Unitree regular walking

Add `--controller-switch` to the hardware command to enable **L1 + Y** as
the controller toggle. Release the buttons before pressing again. The first
DreamWaQ startup still uses START and A. Subsequent returns prepare the default
pose automatically using `--ramp-time`, then resume DreamWaQ in **standing**;
press X to walk. The DDS readers and policy process stay alive across toggles.

```bash
dreamwaq-g1-29dof --net eth0 \
  --export "$PWD/checkpoints/new3_4_widerange_quality_390000/exported" \
  --controller-switch \
  --locomotion-mode standing --mode-toggle-button X
```

Use `--initial-controller unitree` with `--controller-switch` to start in the
native controller and enter DreamWaQ with L1+Y. `--unitree-mode ai` selects the
native MotionSwitcher alias; `--controller-timeout 8` bounds each transition.
The feature is opt-in and rejected in shadow/dry-run mode.

On DreamWaQ -> Unitree, the LowCmd writer is joined and its final damping write
must finish before LowCmd is disabled. The existing native service is released
and restarted, including when `CheckMode` still says `ai` in Develop mode.
After locomotion state becomes ready, the runtime prepares native damping
(FSM 1) if the freshly started controller is in zero torque (FSM 0), then
locked standing (FSM 4) from FSM 0/1, and regular walking (FSM 500). Standing
FSM 4 can enter walking directly; an already walking FSM 500/501 needs no new
request. Each requested state must be confirmed before the next command.
All preparation runs inside a single **L1+Y** press. If the firmware does not
provide the `sport` RPC, it leaves the native controller active with DreamWaQ
LowCmd disabled and logs the manual remote sequence **L2+UP, then R1+X**.
This is reported as native control, without claiming walking was confirmed.

**This transition drops posture-support torque.** `suspend_lowcmd()` sends
zero-stiffness damping (`kp=0`, `tau_ff=0`) before stopping low-level ownership,
and the restarted native controller passes through zero torque/damping before
standing and walking. The observed loss of support is part of this implemented
sequence. A single-button transition and confirmed walking do not establish
continuous support. The upstream `rt/user_lowcmd` / user-control API offers a
different ownership path with the native service alive, but that path is not
implemented or validated on this robot in this runtime.

On this G1, an actual probe on 2026-10-02 returned **3203** from
`SwitchToInternalCtrl(WALKRUN)` (API 7111) while native FSM remained 500.
The SDK defines 3203 as `RPC_ERR_SERVER_API_NOT_IMPL`: the robot's server lacks
the return API, even though the installed Python client exposes the method.
Client SDK changes alone cannot implement it in the native controller.
No user-control entry was attempted, because its return path is unavailable.
The existing L1+Y implementation therefore still uses the restart sequence;
it has **not** been changed into a continuous-support switch.

To repeat the capability check after a native-controller update, stop DreamWaQ
and prepare native walking (FSM 500/501), then run:

```bash
/home/unitree/miniconda3/envs/emer_tw/bin/python deploy/check_live_controller_handoff.py \
  --net eth0 --report "$HOME/.cache/dreamwaq-live-handoff-check.json"
```

This sends only the internal WALKRUN request from an already walking controller;
it does not create a LowCmd publisher, release/restart AI, or enter user control.
An acknowledged return API is a prerequisite, not proof of a working full
handoff or continuous posture support. Unsupported or inconclusive checks exit
with a nonzero status.

The SDK's `3104` is a response timeout. Native startup can interrupt the RPC
service after `SelectMode(ai)` has already been sent. A timed-out transition
request is not resent, and its resulting controller/FSM state must be
confirmed. During that verification, transient read errors `3102` and `3104`
are retried within `--controller-timeout`; explicit server errors still fail
immediately. A timeout alone never confirms control ownership, and LowCmd stays
disabled if native startup cannot be verified. The same handling applies to
controller release and shutdown. Error codes are defined in the
[official SDK](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/rpc/internal.py).
MotionSwitcher busy responses (`7002`) explicitly reject a command; those
requests are retried within the transition deadline so a return/exit during
native startup can finish. Persistent busy status never enables LowCmd.

For locomotion-state reads, `7301` means `LocoState not available` in the
[G1 SDK](https://github.com/unitreerobotics/unitree_sdk2/blob/master/include/unitree/robot/g1/loco/g1_loco_error.hpp).
It can appear after MotionSwitcher has confirmed native startup but before
the native locomotion state is ready. The runtime retries this read within
`--controller-timeout` before requesting native motion, and while verifying
those transitions. It logs initialization once per wait, sends no FSM request
before state readiness, and retains the manual fallback if state never appears.
Other locomotion errors remain immediate failures.
FSM 4 can be reported before native standing preparation finishes. If a walking
request succeeds but a subsequent read still reports 4, the same FSM 500 target
is renewed within the deadline. Unknown outcomes from response timeouts are
never resent. Walking requests are renewed only while the observed FSM is 4.

On Unitree -> DreamWaQ, a confirmed `ReleaseMode` and fresh LowState are required
before any LowCmd write. CENet history, accepted actor history, published-command
telemetry and policy ease-in restart for the new controller session. Modified
X/Y button shortcuts do not toggle DreamWaQ's locomotion mode. This implements
a controller change with an intermediate damping/default-pose preparation;
continuous balance throughout the handoff has not been validated.

Select/Ctrl+C exits to damping from either controller. Native control is released
before the final LowCmd damping write. A failed release or a write still in flight
prevents ownership from being overridden and is reported in the runtime log.
`mode_machine=5` identifies the G1 hardware type; it is not a Develop-mode flag.

Offline transition tests (no robot I/O):

```bash
python -m unittest discover -s deploy/tests -p test_controller_handoff.py
```

Validation on 2026-10-02: all 32 handoff tests passed in the robot's `emer_tw`
environment, and both same-worker epoch-reset/sequence tests passed with the
currently deployed NEW3-2 tracking export. The workstation regression suite
had 41 passing tests and two failures: FP32 fused-action parity (3.46e-6 versus
3e-6 tolerance) and the unpinned worker's 15 ms deadline. Both failures were
also reproduced with the original code. The operator's hardware log confirmed
the L1+Y trigger and writer stop, then exposed a SelectMode/CheckMode 3104
timeout. A subsequent hardware log confirmed native startup and exposed the
initial GetFsmId 7301 response; later read-only probes found a responding sport
service and FSM 0. Timeout/restart and locomotion-state readiness recovery have
regression coverage. Hardware probes confirmed that FSM 0 ignores direct
requests for 4 and 500 despite SDK success responses. Native 0 -> 1 -> 4
transitions were confirmed; 4 -> 500 completed in 0.078 s, followed by three
FSM 500 readbacks. A full physical DreamWaQ/native round trip remains unverified.
The final automatic native restart/entry probe also confirmed walking: release
at 1.79 s, native startup at 9.18 s, FSM 1/4 preparation at 10.46/10.51 s, and
confirmed FSM 500 at 16.07 s from SDK-client construction. Three further reads
remained at 500. Approximately 14 s elapsed from release to confirmed walking;
the single button initiates the sequence but does not eliminate initialization
or physical standing preparation time.

On the 8-core G1, the default CPU layout is:

| CPUs | Work |
| --- | --- |
| 0–2 | onboard `teleimager-server` (its default affinity) |
| 3–5 | DreamWaQ DDS I/O mask; LowState reader on 3, `rt/arm_sdk` reader on 4, and 250 Hz LowCmd writer on 5 |
| 6 | DreamWaQ policy inference process |
| 7 | 50 Hz parent control loop |

DreamWaQ applies the 3–5 mask before initializing DDS channels, so middleware
threads inherit a CPU set separate from the image server. Its Python reader and
writer threads then pin themselves to their assigned individual CPUs. Startup
and runtime diagnostics print the requested layout and effective reader/writer
masks. The configured camera mask is reserved here; the teleimager process still
has to be started with its own 0–2 affinity.

Override `--camera-cpus`, `--io-cpus`, `--lowstate-reader-cpu`,
`--arm-reader-cpu`, `--writer-cpu`, `--policy-cpu`, or `--control-cpu` together
when intentionally benchmarking a different topology. The launcher rejects
overlapping CPU assignments and unavailable I/O CPUs. The worker polls for the
final 8 ms before each expected 50 Hz request (`--policy-prewake-us 8000`) so
the `schedutil` governor does not add model wake-up latency. Set
`--policy-prewake-us 0` only when intentionally benchmarking another topology.

Keep the robot suspended for the first test and begin with the gains in
`g1_homie.yaml`. Do not start a physical run from an automated shell session.

### Switch to a second DreamWaQ export

`--switch-export` loads a second export in its own policy process at startup.
Pressing and releasing `--switch-button` (default **Y**) on its own then hands
the robot to it, and the next press hands it back, without damping. Both
workers stay loaded. The LowCmd writer keeps holding the last accepted leg
target, the next state goes to the other worker, and the leg target blends from
that held pose to that policy's output over `--ramp-time`. Arm targets and arm
feed-forward torque are not part of the blend. The policy taking over starts
from a fresh CENet and actor history, as at bring-up, and in **standing** if
its export has an explicit locomotion mode.

The switch is accepted only while standing still: an explicit-mode export
(NEW3) is in standing, and the requested velocity command is zero before
standing gates it. The second export may have no standing mode and would act on
that command at once, so sticks must be centered and a ROS bridge in HOLD.
Otherwise the press is logged and ignored. It fires on release, and any other button seen during the
press cancels it, so **L1+Y** under `--controller-switch` never triggers it. That
handoff stays available after the switch and resumes the export now running.
A press during the leg blend is ignored too.

Switching to the second export latches the external hold (below), whoever
switched: it has no standing mode, so only the sticks may move it. The way back
cannot go by the command alone, because a policy with no standing mode can
step in place on a zero command. It also needs every leg joint slower than
`--switch-back-max-leg-dq` (0.5 rad/s) and the base slower than
`--switch-back-max-gyro` (0.3 rad/s) for `--switch-back-still-time` (0.5 s),
and `rt/arm_sdk` quiet: no sample that is both engaged and fresh, so the
mission client has exited. Then the arms ramp to the default pose (the
bring-up pose) over `--switch-back-arm-time` (3 s), together with their gains,
and earlier `rt/arm_sdk` samples are ignored; the next sender to engage takes
the arms again. The hold stays latched on the way back, so the bridge cannot
drive the robot while it is walked to the next start on the sticks; Thor's
`release_external` lifts it. These thresholds are not yet tuned on hardware;
a refused press logs the measured values.

Only these options change at the switch: `--switch-max-vx`, `--switch-max-vy`,
`--switch-max-yaw`, `--switch-arm-smoothing-tau` and
`--switch-disable-lower-joint-target-clip`. Unset values come from the hardware
config, as they would in a separate run. Everything else is shared. Both exports
must agree on action joints, default pose, gains and torque limits, because the
writer, the ramps and the gains are fixed at startup; the launcher refuses
otherwise. NEW3-4 `widerange_quality_390000` and NEW2-3 `main_plus50k_249999`
agree.

`deploy/run_dreamwaq_nature.sh` is `deploy/run_dreamwaq_autonomy.sh` with the
switch added: it runs that launcher unchanged, so the ROS bridge input on
`127.0.0.1:15002`, the CPU partition watchdog and layout, NEW3-4 backpack and
its 0.95/0.75/0.75 caps all come from there. Y switches to NEW2-3 backpack
`new2_3_backpack_lower8cm_299999` (0.3/0.2/0.4 caps, arm smoothing 0.04 s,
lower-body target clipping disabled), with the bridge held, and Y again
switches back. `--shadow` runs the autonomy probe without the switch. Extra
arguments are appended.

```bash
cd /home/unitree/00_workspace/g1_deploy/thirdparty/dreamwaq_g1_29dof
./deploy/run_dreamwaq_nature.sh
```

Offline tests (no robot I/O):
`python -m unittest discover -s deploy/tests -p test_policy_switch.py`.
The switch has not yet run on hardware.

### Remote requests from Thor (`./nature demo`)

The Nature launcher also listens on `192.168.123.164:15003` for Thor
(`192.168.123.2`) alone (`--remote-bind`, `--remote-port`,
`--remote-allowed-host`; `deploy/remote_mode.py`). Thor's `./nature demo` uses it
to drive the sequence the remote otherwise would. Every request names a state,
so a retry cannot toggle twice, and gets exactly one acknowledgement.

| action | effect | refused when |
| --- | --- | --- |
| `walking` / `standing` | sets NEW3's mode, as X would | the running export has no mode (after the switch) |
| `hold_external` | zeroes the bridge's velocity until `release_external`; the sticks still drive and damping still latches | the command source takes no external input |
| `release_external` | lets the bridge's velocity through again; standing still zeroes it until walking | the running export has no standing mode, or it is walking |
| `switch` | Y's switch to the second export, under Y's own checks; the way back is Y's alone | `hold_external` is not latched, the robot is walking, or a command is live |
| `status` | registers for status only | never |

A request the loop has not taken within 0.5 s is answered `expired`, so nothing
held back while Unitree owns the robot (L1+Y) fires later; requests are
answered `unavailable` then. Status goes back at 10 Hz: controller, export,
`switched`, `blending`, mode, `external_hold`, the bridge's last `state`/`reason`
and its age, and the requested and effective commands. That bridge reason is
how Thor sees the arrival (`operator_hold` after `ALIGNED`), and tells it apart
from a localisation dropout (`odometry_stale`), which also reads `hold`.

Tests: `python -m unittest discover -s deploy/tests -p test_remote_mode.py`
and `RemoteLoopTest` in `test_policy_switch.py`.

## Read-only latency shadow

Run this first on the onboard computer. Shadow mode constructs only the
`rt/lowstate` subscriber and, by default, the read-only arm source used by the
real controller: no `rt/lowcmd` publisher or writer is ever constructed. It
uses a production-shaped 250 Hz state reader and a publisher-free 250 Hz writer
cadence probe that reproduces arm lookup, 35-slot command construction, and CRC
calculation. Use `--no-shadow-production-load` only to
isolate policy/LowState timing from arm callback load.

```bash
/home/unitree/miniconda3/envs/emer_tw/bin/dreamwaq-g1-29dof \
  --net eth0 \
  --export "$PWD/checkpoints/p17_best_ee_off_move_0830/exported" \
  --device cpu \
  --shadow --shadow-warmup-s 5 --duration 60 \
  --policy-timeout-ms 15 \
  --shadow-expected-lowstate-hz 1000 \
  --shadow-report /tmp/dreamwaq_shadow_60s.json
```

The terminal prints `PASS_NO_DELAY`, `PASS_WITH_WARNINGS`, or
`FAIL_DELAY_DETECTED`. The JSON summary and per-tick CSV separate LowState age,
parent preparation, IPC wake-up, whole worker compute, TorchScript inference,
response transfer, response-to-next-250-Hz-slot latency, and the read-only arm
callback's response-to-ready latency. Any current-state
action timeout, sequence mismatch, worker state skip, LowState replay, excessive
state age, unexpectedly slow hardware-tick progress (DDS backlog/stall), or slow
shadow-writer handoff makes the run fail. Set `--shadow-expected-lowstate-hz` to
the robot firmware's actual LowState `tick` rate if it is not 1000 Hz.

A passing shadow proves the state/sequence/IPC/inference/pre-write software path
did not introduce an extra policy step under the measured load. Because the run
is read-only, it cannot measure the final DDS `Write()`/network/robot acceptance
latency, nor infer a constant LowState transport offset from an unsynchronized
hardware tick. Those limitations are also embedded in the JSON report.
