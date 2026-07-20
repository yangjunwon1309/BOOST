#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
python -u path_streaming/path_state_machine_bridge.py \
  --state "${PATH_STATE:-teleop}" \
  --fps "${PATH_FPS:-50}" \
  --phase-tracking-mode "${PATH_PHASE_TRACKING_MODE:-fixed_rate}" \
  --pico-endpoint "${PATH_PICO_INTERNAL_ENDPOINT:-tcp://127.0.0.1:5559}" \
  --gmr-endpoint "${PATH_GMR_REF_ENDPOINT:-tcp://127.0.0.1:5558}" \
  --gmr-topic "${PATH_GMR_REF_TOPIC:-gmr_ref}" \
  --plan-endpoint "${PATH_PLAN_ENDPOINT:-tcp://127.0.0.1:5562}" \
  --plan-topic "${PATH_PLAN_TOPIC:-curobo_plan}" \
  --static-target-bind "tcp://*:${PATH_STATIC_TARGET_PORT:-5564}" \
  --static-target-topic "${PATH_STATIC_TARGET_TOPIC:-path_static_ee_target}" \
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
  ${PATH_AUTO_CLEAR_ON_TARGET_REACHED:+--auto-clear-on-target-reached} \
  ${PATH_CLEAR_PLAN_AFTER_TOTAL_DURATION:+--clear-plan-after-total-duration} \
  "$@"
