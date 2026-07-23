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
  --phase-tracking-mode "${PATH_PHASE_TRACKING_MODE:-fixed_rate}" \
  --pico-endpoint "${PATH_PICO_INTERNAL_ENDPOINT:-tcp://127.0.0.1:5559}" \
  --gmr-endpoint "${PATH_GMR_REF_ENDPOINT:-tcp://127.0.0.1:5558}" \
  --gmr-topic "${PATH_GMR_REF_TOPIC:-gmr_ref}" \
  --gmr-frame-tolerance "${PATH_GMR_FRAME_TOLERANCE:-1}" \
  --gmr-max-prediction-horizon-ms "${PATH_GMR_MAX_PREDICTION_HORIZON_MS:-1.0}" \
  --teleop-sync-buffer-min-frames "${PATH_TELEOP_SYNC_BUFFER_MIN_FRAMES:-2}" \
  --teleop-sync-drop-unmatched-after-sec "${PATH_TELEOP_SYNC_DROP_UNMATCHED_AFTER_SEC:-0.25}" \
  --plan-endpoint "${PATH_PLAN_ENDPOINT:-tcp://127.0.0.1:5562}" \
  --plan-topic "${PATH_PLAN_TOPIC:-curobo_plan}" \
  --static-target-bind "tcp://*:${PATH_STATIC_TARGET_PORT:-5564}" \
  --static-target-topic "${PATH_STATIC_TARGET_TOPIC:-path_static_ee_target}" \
  --static-target-timeout-sec "${PATH_STATIC_TARGET_TIMEOUT_SEC:-0.35}" \
  --static-target-hold-sec "${PATH_STATIC_TARGET_HOLD_SEC:-0.0}" \
  --sonic-port "${PATH_SONIC_PORT:-5556}" \
  --qp-port "${PATH_REFERENCE_PORT:-5560}" \
  --qp-topic "${PATH_REFERENCE_TOPIC:-path_qp_reference}" \
  ${PATH_OUTER_WAIST_ENABLED:+--outer-waist-enabled} \
  --outer-waist-update-hz "${PATH_OUTER_WAIST_UPDATE_HZ:-1.0}" \
  --outer-waist-enable-error-m "${PATH_OUTER_WAIST_ENABLE_ERROR_M:-0.045}" \
  --outer-waist-enable-dwell-sec "${PATH_OUTER_WAIST_ENABLE_DWELL_SEC:-0.50}" \
  --outer-waist-freeze-error-m "${PATH_OUTER_WAIST_FREEZE_ERROR_M:-0.025}" \
  --outer-waist-yaw-bound-rad "${PATH_OUTER_WAIST_YAW_BOUND_RAD:-0.15}" \
  --outer-waist-roll-bound-rad "${PATH_OUTER_WAIST_ROLL_BOUND_RAD:-0.02}" \
  --outer-waist-pitch-bound-rad "${PATH_OUTER_WAIST_PITCH_BOUND_RAD:-0.03}" \
  --outer-waist-yaw-velocity-rad-s "${PATH_OUTER_WAIST_YAW_VELOCITY_RAD_S:-0.06}" \
  --outer-waist-roll-velocity-rad-s "${PATH_OUTER_WAIST_ROLL_VELOCITY_RAD_S:-0.01}" \
  --outer-waist-pitch-velocity-rad-s "${PATH_OUTER_WAIST_PITCH_VELOCITY_RAD_S:-0.015}" \
  --outer-waist-yaw-samples "${PATH_OUTER_WAIST_YAW_SAMPLES:-31}" \
  $(if [[ "${PATH_ARM_NONLINEAR_IK_ENABLED:-1}" != "0" ]]; then printf '%s' "--arm-nonlinear-ik-enabled"; fi) \
  --arm-ik-urdf "${PATH_ARM_IK_URDF:-/home/url/00_workspace/xr_teleop/assets/g1/g1_body29_hand14.urdf}" \
  --arm-ik-model-dir "${PATH_ARM_IK_MODEL_DIR:-/home/url/00_workspace/xr_teleop/assets/g1}" \
  --arm-ik-max-iter "${PATH_ARM_IK_MAX_ITER:-30}" \
  --arm-ik-ee-offset-m "${PATH_ARM_IK_EE_OFFSET_M:-0.05}" \
  ${PATH_AUTO_CLEAR_ON_TARGET_REACHED:+--auto-clear-on-target-reached} \
  ${PATH_CLEAR_PLAN_AFTER_TOTAL_DURATION:+--clear-plan-after-total-duration} \
  "$@"
