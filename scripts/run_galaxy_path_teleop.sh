#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$BOOST_ROOT/configs/teleop/local.env"
export PATH_AUTO_ARM_XR_TRACKING=0

if [[ "${PIXI_PROJECT_ROOT:-}" != "$BOOST_ROOT" ]]; then
  echo "Run this stack through Pixi: pixi run -e thor-teleop path-teleop" >&2
  exit 2
fi
if [[ ! -t 0 ]]; then
  echo "PATH terminal controls need an interactive terminal (r/s/q)." >&2
  exit 2
fi

data_root="${BOOST_PATH_EXPERIMENT_DATA_ROOT:-$HOME/data/boost/path_wrist_experiments}"
mkdir -p "$data_root"

xr_pid=""
recorder_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "$xr_pid" "$recorder_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$xr_pid" "$recorder_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

cd "$BOOST_ROOT"
bash scripts/run_galaxy_xr_manager.sh &
xr_pid=$!
"$PATH_PYTHON" -u services/pose_service/tools/record_wrist_experiment_v11.py \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --g1-lcm-url "$PATH_LCM_URL" \
  --output "$data_root/path_wrist_experiment.csv" \
  --brainco-dds-interface "$BRAINCO_DDS_INTERFACE" \
  --hand-config "$BOOST_ROOT/configs/hands/brainco_revo2.yaml" \
  --teleimager-relay-config "$BOOST_TELEIMAGER_RELAY_CONFIG" \
  --event-toggle &
recorder_pid=$!

echo "Galaxy XR hand teleop stack is starting. In this terminal: r=start/resume, s=record start/stop, q=stop and exit."
echo "Teleop experiment trials: $data_root/path_wrist_experiment_NNN/"
echo "The Revo2 hand publisher is a separate thor environment: BRAINCO_DDS_INTERFACE=eth0 pixi run -e thor hand-mppi-publish"

bash scripts/run_state_machine_bridge.sh "$@"
