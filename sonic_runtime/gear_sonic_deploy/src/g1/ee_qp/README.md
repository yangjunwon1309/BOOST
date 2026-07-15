# G1 End-Effector QP Corrector

This directory contains the arms-only wrist-position QP corrector used by
`g1_deploy_onnx_ref`.

## Frames

- `q_sonic` is the 29-DoF post-processed SONIC position target.
- The QP consumes a generic motion reference. The current provider is the
  host-side `gmr_realtime_ee_reference.py` worker, which retargets the latest
  streamed SMPL-X frame with one persistent, warm-started GMR instance.
- The worker publishes `source`, `control_mode`, `feasible`, `plan_id`,
  `phase`, and the retargeted G1 29-DoF `q_ref` on
  `tcp://127.0.0.1:5558/gmr_ref`. The legacy `q_mujoco` key is still accepted.
- Future cuRobo plan-execution providers should publish the same schema with
  `source="curobo"` and `control_mode="plan_exec"`. An infeasible goal should
  publish `feasible=false`, in which case QP falls back without applying a
  planner correction.
- Deploy runs the same `RobotFkAdapter` on the reference pose and measured G1 pose, so both
  wrist positions use the same G1 model, tracking-point offsets, joint order,
  and pelvis-local frame.
- Deploy passes identity pelvis orientation to both wrist FK and the reference.
- Reference and measured G1 wrist positions are compared directly in the
  pelvis-local frame.
- Root translation and orientation error are therefore excluded from the QP
  task; only the arm joint configuration contributes to wrist-position error.
- If no fresh reference arrives within `reference_timeout_ms`, the configured
  invalid fallback is used. The default decays the previous correction to zero.

The task error and Jacobian are evaluated at the post-processed SONIC target.
The resulting correction is added to that same target, so the QP linearization
point and the commanded configuration have consistent semantics.
Joints outside the 14-arm decision vector, including the waist, are evaluated
at their measured positions. This makes waist tracking lag visible to wrist FK
without replacing SONIC's arm targets.

## Objective

For the 14 arm-joint correction `delta_q`, the QP minimizes

```text
0.5 w_p ||(J delta_q - e_ee) / position_tolerance||^2
+ 0.5 w_q ||(delta_q - (q_ref - q_SONIC)) / joint_tolerance||^2
+ 0.5 w_s ||delta_q - delta_q_previous||^2
```

The tolerances make the meter and radian residuals dimensionless before the
weights are compared. The reference joint delta is projected into the active joint,
correction, and rate bounds and supplied to ProxQP as its primal initial guess.
After solving, FK is evaluated again at `q_SONIC + delta_q`; a candidate that
does not improve the nonlinear wrist error is rejected.

`invalid_fallback.mode` accepts `sonic`, `hold`, or `decay`. `decay` uses
`invalid_fallback.decay_time_sec` as an exponential time constant and avoids an
instantaneous correction drop when a reference is stale or the solver fails.

Nonlinear validation starts with `backtracking_initial_alpha` (default `1.0`).
If the full correction does not improve FK error, it interpolates from the
previous correction toward the QP solution using `backtracking_factor` until
`backtracking_min_alpha`. A fully rejected candidate uses the configured
fallback without clearing the previous correction first.

Live 0.5-second telemetry summaries are available from the deploy ZMQ stream:

```bash
python gear_sonic/scripts/listen_ee_qp_telemetry.py \
  --endpoint tcp://127.0.0.1:5557 --interval 0.5
```

The listener reports fallback causes such as `stale_reference`,
`infeasible_bounds`, `solver_status`, `solve_timeout`, and
`candidate_rejected` over each interval.

The GMR worker consumes 50 Hz measured SMPL frames and publishes 100 Hz G1
references by solving once at the measured frame and once at a 10 ms
constant-velocity prediction. Override the defaults before starting PICO with
`GMR_EE_SOURCE_RATE_HZ` and `GMR_EE_OUTPUT_RATE_HZ`.

## Dependencies

System dependencies already used by deploy:

```bash
sudo apt install libeigen3-dev libyaml-cpp-dev
```

Python validation and the C++ ProxQP package are installed in an isolated venv:

```bash
cd gear_sonic_deploy/gear_sonic_deploy/src/g1/ee_qp
conda run -n unifolm-vla-thor python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r tools/requirements.txt
```

The wheel installs ProxSuite's C++ headers and CMake config under
`site-packages/cmeel.prefix`:

```bash
PROX_PREFIX=$(.venv/bin/python tools/print_proxsuite_prefix.py)
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PROX_PREFIX"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Validate the official `pytorch_kinematics` Jacobian against central finite
differences on the G1 MJCF:

```bash
MPLCONFIGDIR=/tmp/matplotlib-eeqp .venv/bin/python tools/verify_jacobian.py \
  --mjcf ../../../g1/g1_29dof.xml \
  --root-link pelvis \
  --end-link left_wrist_yaw_link
```

The pinned `pytorch_kinematics` source revision is required because the PyPI
wheel with the same version number does not expose its newer MJCF loader.

## Deploy integration

Build from the deploy project root, then append these options to the normal
`g1_deploy_onnx_ref` command:

```bash
just build Release
./target/release/g1_deploy_onnx_ref <interface> <policy> <motion> \
  --use-ee-qp \
  --ee-qp-config src/g1/ee_qp/config/ee_qp.yaml
```

The correction runs after SONIC action scaling and before
`motor_command_buffer_` is written. It is mutually exclusive with the legacy
upper-body IK modes. The LCM executable is not connected to this path yet.

`start_pico_manager.sh` starts the GMR worker alongside the PICO manager by
default. Set `GMR_EE_REFERENCE_ENABLED=0` only when the QP path is not needed.
