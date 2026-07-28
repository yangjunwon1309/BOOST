#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

cd "$PATH_G1_ROOT"
exec python path_streaming/pico_manager_thread_server.py \
  --manager \
  --port "${PATH_PICO_INTERNAL_PORT:-5559}" \
  --button_event_port "${PATH_PICO_BUTTON_PORT:-5565}" \
  --button_poll_hz "${PATH_PICO_BUTTON_POLL_HZ:-100}" \
  "$@"
