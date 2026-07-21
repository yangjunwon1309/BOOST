#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../configs/runtime/local.env"

cd "$PATH_SONIC_DEPLOY_ROOT"

ee_qp_args=()
if [[ "${EE_QP_ENABLED:-1}" == "1" ]]; then
  ee_qp_args+=(
    --use-ee-qp
    --ee-qp-config "${EE_QP_CONFIG:-src/g1/ee_qp/config/ee_qp_real.yaml}"
  )
  if [[ "${PATH_DYNAMIC_REFERENCE_ENABLED:-1}" == "1" ]]; then
    ee_qp_args+=(
      --path-use-dynamic-reference
      --path-reference-host "${PATH_REFERENCE_HOST:-127.0.0.1}"
      --path-reference-port "${PATH_REFERENCE_PORT:-5560}"
      --path-reference-topic "${PATH_REFERENCE_TOPIC:-path_qp_reference}"
      --path-reference-lookahead-ms "${PATH_REFERENCE_LOOKAHEAD_MS:-10}"
      --path-reference-heartbeat-timeout-ms "${PATH_REFERENCE_HEARTBEAT_TIMEOUT_MS:-500}"
    )
  fi
fi

lcm_args=(
  --input-type "${PATH_INPUT_TYPE:-zmq_manager}"
  --output-type "${PATH_OUTPUT_TYPE:-all}"
  --zmq-host "${PATH_SONIC_ZMQ_HOST:-127.0.0.1}"
  --lcm-url "${PATH_LCM_URL:-udpm://239.255.76.67:7667?ttl=1}"
  --lcm-action-channel "${PATH_LCM_ACTION_CHANNEL:-G1_POLICY_ACTION}"
  --lcm-state-channel "${PATH_LCM_STATE_CHANNEL:-G1_ROBOT_STATE}"
)

if [[ -n "${PATH_LCM_IFACE:-}" ]]; then
  lcm_args+=(--lcm-iface "$PATH_LCM_IFACE")
fi
if [[ -n "${PATH_LCM_IP:-}" ]]; then
  lcm_args+=(--lcm-ip "$PATH_LCM_IP")
fi
if [[ "${PATH_LCM_SKIP_NET_SETUP:-0}" == "1" ]]; then
  lcm_args+=(--skip-net-setup)
fi
if [[ "${PATH_DUMMY_STATE:-0}" == "1" ]]; then
  lcm_args+=(--dummy-state)
fi
if [[ "${PATH_DESK_POSE:-0}" == "1" ]]; then
  lcm_args+=(--desk-pose)
fi

bash deploy_lcm.sh \
  "${lcm_args[@]}" \
  "${ee_qp_args[@]}" \
  "$@"
