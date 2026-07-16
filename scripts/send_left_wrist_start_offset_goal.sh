#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
python tools/send_gmr_wrist_offset_goal.py \
  --target-link "${PATH_GOAL_TARGET_LINK:-left_wrist_yaw_link}" \
  --dx "${PATH_GOAL_DX:-0.03}" \
  --dy "${PATH_GOAL_DY:-0.00}" \
  --dz "${PATH_GOAL_DZ:-0.00}" \
  --gmr-endpoint "${PATH_GMR_REF_ENDPOINT:-tcp://127.0.0.1:5558}" \
  --gmr-topic "${PATH_GMR_REF_TOPIC:-gmr_ref}" \
  --goal-endpoint "tcp://127.0.0.1:${PATH_GOAL_PORT:-5561}" \
  --goal-topic "${PATH_GOAL_TOPIC:-path_goal}" \
  --model-xml "${PATH_ROBOT_XML:-$PATH_G1_ROOT/sonic_runtime/gear_sonic_deploy/g1/g1_29dof.xml}" \
  "$@"
