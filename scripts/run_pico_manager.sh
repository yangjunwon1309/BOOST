#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"

# The XRoboToolkit Python extension is built out-of-tree.  The shared-library
# directory comes from local.env, but Python also needs this build directory
# to import xrobotoolkit_sdk for the live Pico manager.
export PYTHONPATH="$PATH_XROBO_ROOT/build:${PYTHONPATH:-}"

cd "$PATH_G1_ROOT"
exec "$PATH_PYTHON" path_streaming/pico_manager_thread_server.py \
  --manager \
  --button_event_port "${PATH_PICO_BUTTON_PORT:-5565}" \
  --button_poll_hz "${PATH_PICO_BUTTON_POLL_HZ:-100}" \
  --vr_pose_debug_port "${PATH_PICO_VR_PORT:-5566}" \
  --vr_pose_debug_hz "${PATH_PICO_VR_HZ:-90}" \
  "$@"
