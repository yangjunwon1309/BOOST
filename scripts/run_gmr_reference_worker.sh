#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
python -u sonic_runtime/gear_sonic/scripts/gmr_realtime_ee_reference.py \
  --gmr-root "$PATH_GMR_ROOT" \
  --pose-host "${PATH_PICO_INTERNAL_HOST:-127.0.0.1}" \
  --pose-port "${PATH_PICO_INTERNAL_PORT:-5559}" \
  --pose-topic "${PATH_PICO_POSE_TOPIC:-pose}" \
  --output-port "${PATH_GMR_REF_PORT:-5558}" \
  --output-topic "${PATH_GMR_REF_TOPIC:-gmr_ref}" \
  --source-rate-hz "${PATH_FPS:-50}" \
  --output-rate-hz "${PATH_GMR_REF_HZ:-100}" \
  "$@"
