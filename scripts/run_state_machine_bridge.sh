#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/teleop/local.env"

cd "$PATH_G1_ROOT"
state_machine_command=(
"$PATH_PYTHON" -u boost/teleop/path_streaming/path_state_machine_bridge.py \
  --state "${PATH_STATE:-teleop}" \
  --fps "${PATH_FPS:-50}" \
  --pico-hand-timeout-sec "${PATH_PICO_HAND_TIMEOUT_SEC:-0.20}" \
  --pico-body-gap-hold-ms "${PATH_PICO_BODY_GAP_HOLD_MS:-200}" \
  --xr-future-tolerance-ms "${PATH_XR_FUTURE_TOLERANCE_MS:-100}" \
  --arm-reference-mode head_position \
  --xr-hand-endpoint "${PATH_XR_HAND_ENDPOINT:-tcp://127.0.0.1:5566}" \
  --static-target-bind "tcp://*:${PATH_STATIC_TARGET_PORT:-5564}" \
  --static-target-topic "${PATH_STATIC_TARGET_TOPIC:-path_static_ee_target}" \
  --static-target-timeout-sec "${PATH_STATIC_TARGET_TIMEOUT_SEC:-0.35}" \
  $(if [[ "${PATH_STATIC_TARGET_AUTO_ENTER:-0}" == "1" ]]; then printf '%s' "--static-target-auto-enter"; else printf '%s' "--no-static-target-auto-enter"; fi) \
  --static-target-hold-sec "${PATH_STATIC_TARGET_HOLD_SEC:-0.0}" \
  --gravity-filter-cutoff-hz "${PATH_GRAVITY_FILTER_CUTOFF_HZ:-8.0}" \
  $(if [[ "${PATH_GRAVITY_ACCEL_BIAS_ENABLED:-1}" != "0" ]]; then printf '%s' "--gravity-accel-bias-enabled"; else printf '%s' "--no-gravity-accel-bias-enabled"; fi) \
  --gravity-bias-time-constant-sec "${PATH_GRAVITY_BIAS_TIME_CONSTANT_SEC:-10.0}" \
  --gravity-bias-max-deg "${PATH_GRAVITY_BIAS_MAX_DEG:-3.0}" \
  --gravity-accel-norm-tolerance "${PATH_GRAVITY_ACCEL_NORM_TOLERANCE:-0.50}" \
  --gravity-bias-gyro-gate-rad-s "${PATH_GRAVITY_BIAS_GYRO_GATE_RAD_S:-0.05}" \
  --shared-stale-after-sec "${PATH_SHARED_STALE_AFTER_SEC:-0.3}" \
  --shared-d-max-m "${PATH_SHARED_D_MAX_M:-0.9}" \
  --shared-position-mode "${PATH_SHARED_POSITION_MODE:-replace}" \
  --shared-position-alpha "${PATH_SHARED_POSITION_ALPHA:-0.5}" \
  --shared-position-alpha-source "${PATH_SHARED_POSITION_ALPHA_SOURCE:-fixed}" \
  --shared-position-posterior-enter-confidence "${PATH_SHARED_POSITION_POSTERIOR_ENTER_CONFIDENCE:-0.45}" \
  --shared-position-posterior-exit-confidence "${PATH_SHARED_POSITION_POSTERIOR_EXIT_CONFIDENCE:-0.25}" \
  --shared-position-posterior-ema-alpha "${PATH_SHARED_POSITION_POSTERIOR_EMA_ALPHA:-0.15}" \
  --shared-position-posterior-max-rate-per-s "${PATH_SHARED_POSITION_POSTERIOR_MAX_RATE_PER_S:-1.2}" \
  --grasp-engagement-delay-sec "${PATH_GRASP_OVERRIDE_HOLD_SEC:-1.5}" \
  --grasp-closing-timeout-sec "${PATH_GRASP_CLOSING_TIMEOUT_SEC:-0.60}" \
  --end-effector-model brainco_revo2 \
  --shared-hand-h-margin-m "${PATH_SHARED_HAND_H_MARGIN_M:-0.05}" \
  --shared-hand-switch-confirm-n "${PATH_SHARED_HAND_SWITCH_CONFIRM_N:-10}" \
  --direct-lcm-url "${PATH_LCM_URL:-udpm://239.255.76.67:7667?ttl=1}" \
  --wrist-visual-lcm-url "${PATH_WRIST_VISUAL_LCM_URL:-${WIFI_LCM_URL:-udpm://239.255.77.67:7667?ttl=1}}" \
  --wrist-visual-channel "${PATH_WRIST_VISUAL_CHANNEL:-PATH_V11_WRIST_IPOPT_VISUAL}" \
  --wrist-record-control-channel "${PATH_WRIST_RECORD_CONTROL_CHANNEL:-PATH_V11_WRIST_EXPERIMENT_RECORD_CONTROL}" \
  --direct-upper-command-channel "${PATH_DIRECT_UPPER_CHANNEL:-G1_PATH_UPPER_COMMAND}" \
  --direct-state-channel "${PATH_LCM_STATE_CHANNEL:-G1_ROBOT_STATE}" \
  --direct-lcm-state \
  --qp-port "${PATH_REFERENCE_PORT:-5560}" \
  --qp-topic "${PATH_REFERENCE_TOPIC:-path_qp_reference}" \
  --shared-hand-selection-bind "${PATH_SHARED_HAND_SELECTION_BIND:-tcp://*:${PATH_SHARED_HAND_SELECTION_PORT:-5567}}" \
  --shared-hand-selection-topic "${PATH_SHARED_HAND_SELECTION_TOPIC:-path_shared_control_hand}" \
  --arm-nonlinear-ik-enabled \
  --arm-ik-urdf "${PATH_ARM_IK_URDF:-$PATH_G1_ROOT/assets/robot/g1_body29_hand14.urdf}" \
  --arm-ik-model-dir "${PATH_ARM_IK_MODEL_DIR:-$PATH_G1_ROOT/assets/robot}" \
  --arm-ik-max-iter "${PATH_ARM_IK_MAX_ITER:-30}" \
  --arm-ik-ee-offset-m "${PATH_ARM_IK_EE_OFFSET_M:-0.0}" \
  --arm-ik-orientation-full-error-m "${PATH_ARM_IK_ORIENTATION_FULL_ERROR_M:-0.01}" \
  --arm-ik-orientation-zero-error-m "${PATH_ARM_IK_ORIENTATION_ZERO_ERROR_M:-0.03}" \
  --arm-ik-orientation-min-scale "1.0" \
  --arm-ik-residual-warning-threshold-m "${PATH_ARM_IK_RESIDUAL_WARNING_THRESHOLD_M:-0.05}" \
  "$@"
)

if [[ "${PATH_AUTO_ARM_XR_TRACKING:-0}" == "1" ]]; then
  state_machine_command+=(--auto-arm-xr-tracking)
fi

if [[ "${PATH_ENABLE_SHARED_TARGET_ASSIST:-0}" == "1" ]]; then
  state_machine_command+=(--enable-shared-target-assist)
fi

if [[ -n "${PATH_CONTROL_CPUSET-6-7}" ]] && command -v taskset >/dev/null 2>&1; then
  exec taskset -c "$PATH_CONTROL_CPUSET" "${state_machine_command[@]}"
fi
exec "${state_machine_command[@]}"
