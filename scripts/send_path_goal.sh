#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 X Y Z [extra path_goal_sender args...]" >&2
  echo "example: $0 0.25 0.12 0.18 --duration-sec 2.5 --final-hold-sec 1.0" >&2
  exit 2
fi

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

x="$1"; y="$2"; z="$3"; shift 3

cd "$PATH_G1_ROOT"
python tools/path_goal_sender.py \
  --target-link "${PATH_GOAL_TARGET_LINK:-left_wrist_yaw_link}" \
  --position "$x" "$y" "$z" \
  --goal-endpoint "tcp://127.0.0.1:${PATH_GOAL_PORT:-5561}" \
  --goal-topic "${PATH_GOAL_TOPIC:-path_goal}" \
  "$@"
