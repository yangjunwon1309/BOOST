#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
python -u path_planning/path_curobo_planning_server.py \
  --curobo-root "$PATH_CUROBO_ROOT" \
  --solver "${PATH_CUROBO_SOLVER:-trajopt}" \
  --robot "${PATH_CUROBO_ROBOT:-unitree_g1_29dof_retarget.yml}" \
  --num-seeds "${PATH_CUROBO_NUM_SEEDS:-4}" \
  --lbfgs-iters "${PATH_CUROBO_LBFGS_ITERS:-25}" \
  --output-port "${PATH_BSPLINE_PORT:-5560}" \
  --output-topic "${PATH_BSPLINE_TOPIC:-curobo_bspline}" \
  --goal-port "${PATH_GOAL_PORT:-5561}" \
  --goal-topic "${PATH_GOAL_TOPIC:-path_goal}" \
  --use-cuda-graph \
  "$@"

