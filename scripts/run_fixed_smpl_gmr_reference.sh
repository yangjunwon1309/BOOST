#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
exec taskset -c "${PATH_CONTROL_CPUSET:-6-7}" \
python -u path_streaming/fixed_smpl_gmr_reference.py \
  --recording-dir "${PATH_FIXED_GMR_RECORDING_DIR:-$PATH_GROOT_ROOT/gear_sonic/scripts/exp_qp_7}" \
  --recording-frame "${PATH_FIXED_GMR_RECORDING_FRAME:-0}" \
  --window-index "${PATH_FIXED_GMR_WINDOW_INDEX:-0}" \
  --gmr-root "${PATH_GMR_ROOT:-/home/url/GMR}" \
  --bind-host "${PATH_GMR_REF_BIND_HOST:-*}" \
  --output-port "${PATH_GMR_REF_PORT:-5558}" \
  --output-topic "${PATH_GMR_REF_TOPIC:-gmr_ref}" \
  --fps "${PATH_FIXED_GMR_HZ:-50}" \
  --warmup-sec "${PATH_FIXED_GMR_WARMUP_SEC:-0.5}" \
  "$@"
