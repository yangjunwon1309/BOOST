#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"
source "$BOOST_ROOT/configs/sim/local.env"
cd "$BOOST_ROOT"

if [[ ! -x "$BOOST_ROOT/.venv_runtime/bin/python" ]]; then
  echo "Runtime is not installed. Run: bash scripts/setup_external_runtime.sh" >&2
  exit 2
fi
if [[ ! -d "$BOOST_HOMIE_ROOT/deploy" ]]; then
  echo "HOMIE checkout not found at $BOOST_HOMIE_ROOT" >&2
  exit 2
fi
if [[ ! -d "$PATH_TELEVUER_ROOT" ]]; then
  echo "Televuer checkout not found at $PATH_TELEVUER_ROOT; run scripts/setup_external_runtime.sh" >&2
  exit 2
fi
if [[ ! -f "$PATH_TELEVUER_CERT" || ! -f "$PATH_TELEVUER_KEY" ]]; then
  echo "Galaxy XR needs readable TLS files. Set PATH_TELEVUER_CERT and PATH_TELEVUER_KEY (defaults: $HOME/.config/xr_teleoperate/{cert,key}.pem)." >&2
  exit 2
fi

"$PATH_PYTHON" -c 'import mujoco, unitree_sdk2py, pinocchio, casadi, lcm, onnxruntime'
if [[ ! -x "$BOOST_ROOT/boost/io/g1_bridge/build/g1_lowlevel_bridge" ]]; then
  bash boost/io/g1_bridge/build.sh
fi

pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

start() {
  "$@" &
  pids+=("$!")
}

echo "Starting local simulation on DDS interface $SIM_DDS_INTERFACE, domain $SIM_DDS_DOMAIN_ID"
echo "LCM simulation transport: $PATH_LCM_URL"

start "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py \
  --config configs/sim/mujoco_revo2.yaml \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID"
sleep 2
start env G1_NETWORK_INTERFACE="$SIM_DDS_INTERFACE" G1_BRIDGE_LIVE=1 \
  bash boost/io/g1_bridge/run.sh --hand-backend none
sleep 1
start "$PATH_PYTHON" boost/sim/mujoco/path_revo2_grip_bridge.py \
  --config configs/sim/mujoco_revo2.yaml \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID"
start "$PATH_PYTHON" "$BOOST_HOMIE_ROOT/deploy/run_homie.py" \
  --net "$SIM_DDS_INTERFACE" --config configs/sim/homie_g1_mujoco.yaml
sleep 2
start bash scripts/run_state_machine_bridge.sh
start bash scripts/run_galaxy_xr_manager.sh --auto-tracking

echo "Stack is running. Press Ctrl-C to stop all local processes."
wait -n "${pids[@]}"
