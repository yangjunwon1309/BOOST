#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/teleop/local.env"

cd "$PATH_G1_ROOT"
exec "$PATH_PYTHON" boost/teleop/path_streaming/galaxy_xr_manager_thread_server.py \
  --televuer-root "${PATH_TELEVUER_ROOT:?Set PATH_TELEVUER_ROOT to the televuer checkout}" \
  --televuer-cert "${PATH_TELEVUER_CERT:-$HOME/.config/xr_teleoperate/cert.pem}" \
  --televuer-key "${PATH_TELEVUER_KEY:-$HOME/.config/xr_teleoperate/key.pem}" \
  "$@"
