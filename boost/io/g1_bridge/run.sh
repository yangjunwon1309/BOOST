#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

bridge="$PATH_G1_ROOT/boost/io/g1_bridge/build/g1_lowlevel_bridge"
if [[ ! -x "$bridge" ]]; then
  echo "Bridge is not built; run boost/io/g1_bridge/build.sh first" >&2
  exit 2
fi

network_interface="${G1_NETWORK_INTERFACE:-eth0}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  network_interface="$1"
  shift
fi

args=(
  "$network_interface"
  --action-source lcm
  --lcm-url "${PATH_LCM_URL:-udpm://239.255.76.67:7667?ttl=1}"
  --lcm-action-channel "${G1_PATH_UPPER_CHANNEL:-G1_PATH_UPPER_COMMAND}"
  --publish-lcm-state
  --lcm-state-channel "${G1_ROBOT_STATE_CHANNEL:-G1_ROBOT_STATE}"
  --lcm-state-rate "${G1_LCM_STATE_RATE_HZ:-100}"
)
if [[ "${G1_BRIDGE_LIVE:-0}" != "1" ]]; then
  args+=(--dry-run-actions)
fi

exec "$bridge" "${args[@]}" "$@"
