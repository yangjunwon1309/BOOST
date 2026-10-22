#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"
source "$BOOST_ROOT/configs/sim/local.env"
# This test intentionally cannot be redirected onto a robot-facing interface.
SIM_DDS_INTERFACE=lo
SIM_DDS_DOMAIN_ID=0
export SIM_DDS_INTERFACE SIM_DDS_DOMAIN_ID
cd "$BOOST_ROOT"

sim_config="${MUJOCO_SIM_CONFIG:-configs/sim/mujoco_revo2.yaml}"
if [[ ! -f "$sim_config" ]]; then
  echo "MuJoCo sim config not found: $sim_config" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Run MuJoCo + HOMIE stand/walk + Revo2 hand MPPI with a local JSON pose.
This test starts no Galaxy XR, Televuer, PATH bridge, or physical-robot route.

Usage: pixi run -e sim mujoco-hand-test [--headless] [--override-file PATH]
EOF
}

headless=0
override_file="${MUJOCO_HAND_OVERRIDE_FILE:-configs/hands/revo2_hand_override_fist.json}"
while (($#)); do
  case "$1" in
    --headless) headless=1 ;;
    --override-file)
      (($# >= 2)) || { echo "--override-file requires a path" >&2; exit 2; }
      override_file="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "${PIXI_PROJECT_ROOT:-}" != "$BOOST_ROOT" ]]; then
  echo "Run this test through Pixi: pixi run -e sim mujoco-hand-test" >&2
  exit 2
fi
if [[ ! -d "$BOOST_HOMIE_ROOT/deploy" ]]; then
  echo "HOMIE snapshot not found at $BOOST_HOMIE_ROOT" >&2
  exit 2
fi
if [[ "$override_file" != /* ]]; then
  override_file="$BOOST_ROOT/$override_file"
fi
if [[ ! -f "$override_file" ]]; then
  echo "Hand override file not found: $override_file" >&2
  exit 2
fi

PIXI_CLI="${PIXI_CLI:-$(command -v pixi || true)}"
if [[ -z "$PIXI_CLI" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_CLI="$HOME/.pixi/bin/pixi"
fi
if [[ -z "$PIXI_CLI" ]]; then
  echo "Pixi executable not found; set PIXI_CLI or add ~/.pixi/bin to PATH." >&2
  exit 2
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

sim_args=(
  "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py
  --config "$sim_config"
  --interface "$SIM_DDS_INTERFACE"
  --domain-id "$SIM_DDS_DOMAIN_ID"
)
if ((headless)); then
  sim_args+=(--headless)
fi

echo "Starting local MuJoCo Revo2 simulation on $SIM_DDS_INTERFACE, domain $SIM_DDS_DOMAIN_ID"
echo "Starting HOMIE stand/walk policy; use the MuJoCo viewer keyboard for locomotion"
echo "W/S forward/back, A/D left/right, Q/R yaw, O/P speed, Space stop"
echo "Hand MPPI input: $override_file"
start "${sim_args[@]}"
sleep 2
start "$PATH_PYTHON" "$BOOST_HOMIE_ROOT/deploy/run_homie.py" \
  --net "$SIM_DDS_INTERFACE" \
  --config configs/sim/homie_g1_mujoco.yaml \
  --auto-start
# Let HOMIE finish its default-pose ramp and load the stand/walk policy before
# starting the JAX retargeter's first compilation.
sleep 3
start "$PIXI_CLI" run -e hand python -m boost.control.hand_mppi.runner \
  --config configs/hands/brainco_revo2_mppi.yaml \
  --input-mode override --override-file "$override_file" \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
  --publish

echo "MuJoCo hand test is running. Edit the override JSON to change hand pose; Ctrl-C stops all local processes."
wait -n "${pids[@]}"
