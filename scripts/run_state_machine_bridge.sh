#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
python -u path_streaming/path_state_machine_bridge.py \
  --state "${PATH_STATE:-teleop}" \
  --fps "${PATH_FPS:-50}" \
  --pico-endpoint "${PATH_PICO_INTERNAL_ENDPOINT:-tcp://127.0.0.1:5559}" \
  --gmr-endpoint "${PATH_GMR_REF_ENDPOINT:-tcp://127.0.0.1:5558}" \
  --gmr-topic "${PATH_GMR_REF_TOPIC:-gmr_ref}" \
  --plan-endpoint "${PATH_PLAN_ENDPOINT:-tcp://127.0.0.1:5562}" \
  --plan-topic "${PATH_PLAN_TOPIC:-curobo_plan}" \
  --sonic-port "${PATH_SONIC_PORT:-5556}" \
  --qp-port "${PATH_REFERENCE_PORT:-5560}" \
  --qp-topic "${PATH_REFERENCE_TOPIC:-path_qp_reference}" \
  "$@"
