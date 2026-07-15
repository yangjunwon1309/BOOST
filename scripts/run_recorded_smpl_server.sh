#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <recording_dir_or_npz> [extra path_smpl_thread_server.py args...]" >&2
  exit 2
fi

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
python -u path_streaming/path_smpl_thread_server.py \
  "$1" \
  --gmr-root "$PATH_GMR_ROOT" \
  "${@:2}"

