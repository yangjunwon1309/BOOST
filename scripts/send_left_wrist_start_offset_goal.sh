#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
python tools/send_mock_goal.py \
  --position 0.2197743 0.1486617 0.0952328 \
  --target-link left_wrist_yaw_link \
  "$@"

