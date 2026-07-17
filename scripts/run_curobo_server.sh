#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
extra_args=()
if [[ "${PATH_CUROBO_REJECT_OUTSIDE_WORKSPACE:-0}" == "1" ]]; then
  extra_args+=(--reject-outside-workspace)
fi
if [[ "${PATH_CUROBO_REJECT_LARGE_GOAL_OFFSET:-0}" == "1" ]]; then
  extra_args+=(--reject-large-goal-offset)
fi
if [[ "${PATH_CUROBO_ALLOW_REPLAN_DURING_EXECUTION:-0}" == "1" ]]; then
  extra_args+=(--allow-replan-during-execution)
fi

python -u path_planning/path_curobo_planning_server.py \
  --curobo-root "$PATH_CUROBO_ROOT" \
  --solver "${PATH_CUROBO_SOLVER:-trajopt}" \
  --robot "${PATH_CUROBO_ROBOT:-unitree_g1_29dof_retarget.yml}" \
  --num-seeds "${PATH_CUROBO_NUM_SEEDS:-4}" \
  --lbfgs-iters "${PATH_CUROBO_LBFGS_ITERS:-25}" \
  --publish-hz "${PATH_CUROBO_PUBLISH_HZ:-5}" \
  --duration-sec "${PATH_CUROBO_DURATION_SEC:-2.0}" \
  --min-duration-sec "${PATH_CUROBO_MIN_DURATION_SEC:-0.75}" \
  --max-duration-sec "${PATH_CUROBO_MAX_DURATION_SEC:-5.0}" \
  --final-hold-sec "${PATH_CUROBO_FINAL_HOLD_SEC:-0.75}" \
  --plan-clear-grace-sec "${PATH_CUROBO_PLAN_CLEAR_GRACE_SEC:-0.25}" \
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
  "${extra_args[@]}" \
  "$@"
