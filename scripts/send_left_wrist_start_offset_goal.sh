#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
extra_args=()
if [[ "${PATH_GOAL_INTERACTIVE:-1}" != "0" ]]; then
  extra_args+=(
    --interactive
    --stream-hz "${PATH_GOAL_STREAM_HZ:-50}"
    --step-cm "${PATH_GOAL_STEP_CM:-1.0}"
  )
fi
if [[ "${PATH_GOAL_STREAM_INCLUDE_Q_SEED:-0}" == "1" ]]; then
  extra_args+=(--stream-include-q-seed)
fi

python tools/path_goal_sender.py \
  --from-state \
  --side "${PATH_GOAL_SIDE:-left}" \
  --dx "${PATH_GOAL_DX:-0.03}" \
  --dy "${PATH_GOAL_DY:-0.00}" \
  --dz "${PATH_GOAL_DZ:-0.00}" \
  --gmr-endpoint "${PATH_GMR_REF_ENDPOINT:-tcp://127.0.0.1:5558}" \
  --gmr-topic "${PATH_GMR_REF_TOPIC:-gmr_ref}" \
  --state-endpoint "${PATH_G1_DEBUG_ENDPOINT:-tcp://127.0.0.1:5557}" \
  --state-topic "${PATH_G1_DEBUG_TOPIC:-g1_debug}" \
  --goal-endpoint "tcp://127.0.0.1:${PATH_STATIC_TARGET_PORT:-5564}" \
  --goal-topic "${PATH_STATIC_TARGET_TOPIC:-path_static_ee_target}" \
  --model-xml "${PATH_ROBOT_XML:-$PATH_G1_ROOT/sonic_runtime/gear_sonic_deploy/g1/g1_29dof.xml}" \
  "${extra_args[@]}" \
  "$@"
