#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
exec taskset -c "${PATH_CONTROL_CPUSET:-6-7}" \
python -u path_streaming/path_state_machine_bridge.py \
  --state "${PATH_STATE:-teleop}" \
  --fps "${PATH_FPS:-50}" \
  $(if [[ "${PATH_PLANNER_LOWER_BODY_IDLE_DEFAULT:-${PATH_TELEOP_GMR_ARMS_ONLY_DEFAULT:-1}}" != "0" ]]; then printf '%s' "--planner-lower-body-idle-default"; else printf '%s' "--no-planner-lower-body-idle-default"; fi) \
  $(if [[ "${PATH_FREEZE_PLANNER_LOWER_BODY:-1}" != "0" ]]; then printf '%s' "--freeze-planner-lower-body"; else printf '%s' "--no-freeze-planner-lower-body"; fi) \
  $(if [[ "${PATH_FREEZE_PLANNER_WAIST:-1}" != "0" ]]; then printf '%s' "--freeze-planner-waist"; else printf '%s' "--no-freeze-planner-waist"; fi) \
  --pico-hand-timeout-sec "${PATH_PICO_HAND_TIMEOUT_SEC:-0.35}" \
  --smpl-timeout-sec "${PATH_SMPL_TIMEOUT_SEC:-0.20}" \
  --smpl-scale-x "${PATH_SMPL_SCALE_X:-0.85}" \
  --smpl-scale-y "${PATH_SMPL_SCALE_Y:-0.85}" \
  --smpl-scale-z "${PATH_SMPL_SCALE_Z:-0.85}" \
  --smpl-displacement-sign "${PATH_SMPL_DISPLACEMENT_SIGN:-1.0}" \
  --smpl-target-ema-alpha "${PATH_SMPL_TARGET_EMA_ALPHA:-0.35}" \
  --smpl-target-max-velocity-m-s "${PATH_SMPL_TARGET_MAX_VELOCITY_M_S:-0.25}" \
  --smpl-target-max-acceleration-m-s2 "${PATH_SMPL_TARGET_MAX_ACCELERATION_M_S2:-1.5}" \
  --smpl-jump-reject-m "${PATH_SMPL_JUMP_REJECT_M:-0.07}" \
  --smpl-max-anchor-offset-m "${PATH_SMPL_MAX_ANCHOR_OFFSET_M:-0.45}" \
  --phase-tracking-mode "${PATH_PHASE_TRACKING_MODE:-fixed_rate}" \
  --pico-endpoint "${PATH_PICO_INTERNAL_ENDPOINT:-tcp://127.0.0.1:5559}" \
  --button-endpoint "${PATH_PICO_BUTTON_ENDPOINT:-tcp://127.0.0.1:5565}" \
  --button-topic "${PATH_PICO_BUTTON_TOPIC:-pico_button_event}" \
  --plan-endpoint "${PATH_PLAN_ENDPOINT:-tcp://127.0.0.1:5562}" \
  --plan-topic "${PATH_PLAN_TOPIC:-curobo_plan}" \
  --static-target-bind "tcp://*:${PATH_STATIC_TARGET_PORT:-5564}" \
  --static-target-topic "${PATH_STATIC_TARGET_TOPIC:-path_static_ee_target}" \
  --static-target-timeout-sec "${PATH_STATIC_TARGET_TIMEOUT_SEC:-0.35}" \
  --static-target-hold-sec "${PATH_STATIC_TARGET_HOLD_SEC:-0.0}" \
  --auto-planning-mode "${PATH_AUTO_PLANNING_MODE:-shared_control}" \
  --shared-rho1 "${PATH_SHARED_RHO1:-0.60}" \
  --shared-rho2 "${PATH_SHARED_RHO2:-0.90}" \
  --shared-alpha-rate-up "${PATH_SHARED_ALPHA_RATE_UP:-1.0}" \
  --shared-alpha-rate-down "${PATH_SHARED_ALPHA_RATE_DOWN:-3.0}" \
  --shared-v-transfer-max "${PATH_SHARED_V_TRANSFER_MAX:-0.25}" \
  --shared-startup-hold-sec "${PATH_SHARED_STARTUP_HOLD_SEC:-2.0}" \
  --shared-stale-after-sec "${PATH_SHARED_STALE_AFTER_SEC:-0.3}" \
  --shared-d-max-m "${PATH_SHARED_D_MAX_M:-0.9}" \
  --shared-ref-lpf-alpha "${PATH_SHARED_REF_LPF_ALPHA:-0.3}" \
  --shared-hand-h-margin-m "${PATH_SHARED_HAND_H_MARGIN_M:-0.05}" \
  --shared-hand-switch-confirm-n "${PATH_SHARED_HAND_SWITCH_CONFIRM_N:-10}" \
  --shared-orientation-mode "${PATH_SHARED_ORIENTATION_MODE:-gmr_prior}" \
  --shared-waist-mode "${PATH_SHARED_WAIST_MODE:-sonic_passthrough}" \
  $(if [[ "${PATH_SHARED_LOCK_INACTIVE_ARM_Q_PRIOR:-1}" != "0" ]]; then printf '%s' "--shared-lock-inactive-arm-q-prior"; else printf '%s' "--no-shared-lock-inactive-arm-q-prior"; fi) \
  --sonic-port "${PATH_SONIC_PORT:-5556}" \
  --qp-port "${PATH_REFERENCE_PORT:-5560}" \
  --qp-topic "${PATH_REFERENCE_TOPIC:-path_qp_reference}" \
  --arm-nonlinear-ik-enabled \
  --arm-ik-urdf "${PATH_ARM_IK_URDF:-/home/url/00_workspace/xr_teleop/assets/g1/g1_body29_hand14.urdf}" \
  --arm-ik-model-dir "${PATH_ARM_IK_MODEL_DIR:-/home/url/00_workspace/xr_teleop/assets/g1}" \
  --arm-ik-max-iter "${PATH_ARM_IK_MAX_ITER:-30}" \
  --arm-ik-ee-offset-m "${PATH_ARM_IK_EE_OFFSET_M:-0.05}" \
  --arm-ik-orientation-full-error-m "${PATH_ARM_IK_ORIENTATION_FULL_ERROR_M:-0.01}" \
  --arm-ik-orientation-zero-error-m "${PATH_ARM_IK_ORIENTATION_ZERO_ERROR_M:-0.03}" \
  --arm-ik-orientation-min-scale "${PATH_ARM_IK_ORIENTATION_MIN_SCALE:-0.001}" \
  --arm-ik-residual-projection-threshold-m "${PATH_ARM_IK_RESIDUAL_PROJECTION_THRESHOLD_M:-0.03}" \
  --arm-ik-residual-success-threshold-m "${PATH_ARM_IK_RESIDUAL_SUCCESS_THRESHOLD_M:-0.05}" \
  --arm-ik-projection-step-m "${PATH_ARM_IK_PROJECTION_STEP_M:-0.05}" \
  ${PATH_AUTO_CLEAR_ON_TARGET_REACHED:+--auto-clear-on-target-reached} \
  ${PATH_CLEAR_PLAN_AFTER_TOTAL_DURATION:+--clear-plan-after-total-duration} \
  "$@"
