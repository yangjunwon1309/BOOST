#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"
source "$BOOST_ROOT/configs/sim/local.env"
cd "$BOOST_ROOT"

usage() {
  cat <<'EOF'
Run the local G1 + Revo2 MuJoCo stack with HOMIE, PATH, and optional Galaxy XR.

Usage: pixi run -e sim mujoco [--no-xr] [--headless]

Options:
  --no-xr     Start MuJoCo, HOMIE, and the PATH/G1 bridges without a headset.
  --headless  Run the physics loop without opening the MuJoCo viewer.
  -h, --help  Show this help.
EOF
}

start_xr=1
headless=0
while (($#)); do
  case "$1" in
    --no-xr) start_xr=0 ;;
    --headless) headless=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "${PIXI_PROJECT_ROOT:-}" != "$BOOST_ROOT" ]]; then
  echo "Run this stack through Pixi: pixi run -e sim mujoco" >&2
  exit 2
fi
if [[ ! -d "$BOOST_HOMIE_ROOT/deploy" ]]; then
  echo "HOMIE snapshot not found at $BOOST_HOMIE_ROOT" >&2
  exit 2
fi
if ((start_xr)) && [[ ! -f "$PATH_TELEVUER_CERT" || ! -f "$PATH_TELEVUER_KEY" ]]; then
  echo "Galaxy XR needs readable TLS files trusted by the headset." >&2
  echo "Set PATH_TELEVUER_CERT and PATH_TELEVUER_KEY, or run with --no-xr." >&2
  exit 2
fi

"$PATH_PYTHON" -c 'import mujoco, unitree_sdk2py, pinocchio, casadi, lcm, onnxruntime'

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
sim_args=(
  "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py
  --config configs/sim/mujoco_revo2.yaml
  --interface "$SIM_DDS_INTERFACE"
  --domain-id "$SIM_DDS_DOMAIN_ID"
)
if ((headless)); then
  sim_args+=(--headless)
fi
start "${sim_args[@]}"
sleep 2
start "$PATH_PYTHON" boost/sim/mujoco/path_dds_bridge.py \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
  --lcm-url "$PATH_LCM_URL"
sleep 1
start "$PATH_PYTHON" boost/sim/mujoco/path_revo2_grip_bridge.py \
  --config configs/sim/mujoco_revo2.yaml \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID"
start "$PATH_PYTHON" "$BOOST_HOMIE_ROOT/deploy/run_homie.py" \
  --net "$SIM_DDS_INTERFACE" --config configs/sim/homie_g1_mujoco.yaml
sleep 2
start bash scripts/run_state_machine_bridge.sh
if ((start_xr)); then
  start bash scripts/run_galaxy_xr_manager.sh --auto-tracking
else
  echo "Galaxy XR is disabled; PATH is waiting for input on its configured ZMQ endpoints."
fi

echo "Stack is running. Press Ctrl-C to stop all local processes."
wait -n "${pids[@]}"
