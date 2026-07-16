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
  --publish-hz "${PATH_CUROBO_PUBLISH_HZ:-5}" \
  --duration-sec "${PATH_CUROBO_DURATION_SEC:-2.0}" \
  --control-dt-sec "${PATH_CUROBO_CONTROL_DT_SEC:-0.05}" \
  --min-replan-interval-sec "${PATH_CUROBO_MIN_REPLAN_INTERVAL_SEC:-2.0}" \
  --duplicate-goal-window-sec "${PATH_CUROBO_DUPLICATE_GOAL_WINDOW_SEC:-1.0}" \
  --duplicate-goal-eps-m "${PATH_CUROBO_DUPLICATE_GOAL_EPS_M:-0.002}" \
  --max-goal-offset-m "${PATH_CUROBO_MAX_GOAL_OFFSET_M:-0.10}" \
  --max-consecutive-failures "${PATH_CUROBO_MAX_CONSECUTIVE_FAILURES:-3}" \
  --infeasible-cooldown-sec "${PATH_CUROBO_INFEASIBLE_COOLDOWN_SEC:-1.0}" \
  --output-port "${PATH_PLAN_PORT:-5562}" \
  --output-topic "${PATH_PLAN_TOPIC:-curobo_plan}" \
  --goal-port "${PATH_GOAL_PORT:-5561}" \
  --goal-topic "${PATH_GOAL_TOPIC:-path_goal}" \
  --use-cuda-graph \
  "$@"
