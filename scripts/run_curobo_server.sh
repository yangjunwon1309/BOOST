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
if [[ "${PATH_CUROBO_USE_CUDA_GRAPH:-0}" == "1" ]]; then
  extra_args+=(--use-cuda-graph)
fi
if [[ "${PATH_CUROBO_ALLOW_DURATION_EXTENSION:-1}" == "0" ]]; then
  extra_args+=(--no-allow-duration-extension)
fi

python -u path_planning/path_curobo_planning_server.py \
  --curobo-root "$PATH_CUROBO_ROOT" \
  --solver "${PATH_CUROBO_SOLVER:-trajopt}" \
  --endpoint-trajectory-source "${PATH_CUROBO_ENDPOINT_TRAJECTORY_SOURCE:-auto}" \
  --robot "${PATH_CUROBO_ROBOT:-unitree_g1_29dof_retarget.yml}" \
  --num-seeds "${PATH_CUROBO_NUM_SEEDS:-4}" \
  --lbfgs-iters "${PATH_CUROBO_LBFGS_ITERS:-25}" \
  --ik-position-tolerance-mm "${PATH_CUROBO_IK_POSITION_TOLERANCE_MM:-20.0}" \
  --max-joint-velocity-rad-s "${PATH_CUROBO_MAX_JOINT_VELOCITY_RAD_S:-0.60}" \
  --max-joint-acceleration-rad-s2 "${PATH_CUROBO_MAX_JOINT_ACCELERATION_RAD_S2:-2.0}" \
  --max-joint-jerk-rad-s3 "${PATH_CUROBO_MAX_JOINT_JERK_RAD_S3:-20.0}" \
  --publish-hz "${PATH_CUROBO_PUBLISH_HZ:-5}" \
  --stream-replan-hz "${PATH_CUROBO_STREAM_REPLAN_HZ:-1}" \
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
  "${extra_args[@]}" \
  "$@"
