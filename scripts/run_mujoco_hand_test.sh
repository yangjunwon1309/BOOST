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
Run MuJoCo + HOMIE stand/walk + Revo2 hand MPPI with local JSON hand poses.
This test starts no Galaxy XR, Televuer, PATH bridge, or physical-robot route.

Usage: pixi run -e sim mujoco-hand-test [--headless] [--override-file PATH]
       pixi run -e sim mujoco-hand-motion-test [--headless]

--motion runs a synthetic repeated fist/open stream. It cannot be combined
with --override-file; use the separate stream script to select source files.
EOF
}

headless=0
motion=0
override_file_explicit=0
override_file="${MUJOCO_HAND_OVERRIDE_FILE:-configs/hands/revo2_hand_override_fist.json}"
while (($#)); do
  case "$1" in
    --headless) headless=1 ;;
    --motion) motion=1 ;;
    --override-file)
      (($# >= 2)) || { echo "--override-file requires a path" >&2; exit 2; }
      override_file="$2"
      override_file_explicit=1
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ((motion && override_file_explicit)); then
  echo "--motion cannot be combined with --override-file; motion mode generates its own temporary JSON input." >&2
  exit 2
fi

if [[ "${PIXI_PROJECT_ROOT:-}" != "$BOOST_ROOT" ]]; then
  echo "Run this test through Pixi: pixi run -e sim mujoco-hand-test" >&2
  exit 2
fi
if [[ ! -d "$BOOST_HOMIE_ROOT/deploy" ]]; then
  echo "HOMIE snapshot not found at $BOOST_HOMIE_ROOT" >&2
  exit 2
fi
motion_tmp_dir=""
if ((motion)); then
  override_file=""
else
  if [[ "$override_file" != /* ]]; then
    override_file="$BOOST_ROOT/$override_file"
  fi
  if [[ ! -f "$override_file" ]]; then
    echo "Hand override file not found: $override_file" >&2
    exit 2
  fi
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
homie_pid=""
signal_groups() {
  local signal="$1"
  local skip_homie="${2:-0}"
  local index
  for ((index=${#pids[@]} - 1; index >= 0; index--)); do
    if ((skip_homie)) && [[ "${pids[index]}" == "$homie_pid" ]]; then
      continue
    fi
    kill -"$signal" -- "-${pids[index]}" 2>/dev/null || true
  done
}

wait_for_groups() {
  local attempt
  local pid
  local alive
  for attempt in {1..50}; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    if ((alive == 0)); then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  local index
  # Let parent processes run their own cleanup first (notably HOMIE's policy
  # worker); escalate to the whole isolated group only if a child stays alive.
  if [[ -n "$homie_pid" ]]; then
    kill -INT "$homie_pid" 2>/dev/null || true
  fi
  signal_groups INT 1
  if ! wait_for_groups; then
    signal_groups TERM
    if ! wait_for_groups; then
      signal_groups KILL
    fi
  fi
  for ((index=${#pids[@]} - 1; index >= 0; index--)); do
    wait "${pids[index]}" 2>/dev/null || true
  done
  if [[ -n "$motion_tmp_dir" ]]; then
    rm -rf -- "$motion_tmp_dir"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

start() {
  setsid "$@" &
  pids+=("$!")
}

start_homie() {
  setsid "$@" &
  homie_pid="$!"
  pids+=("$homie_pid")
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
if ((motion)); then
  sim_args+=(--hold-start-delay-s 5.0)
fi

echo "Starting local MuJoCo Revo2 simulation on $SIM_DDS_INTERFACE, domain $SIM_DDS_DOMAIN_ID"
if ((motion)); then
  echo "Starting HOMIE stand-policy-only auto-start; zero command holds the configured stand"
  motion_tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/boost-mujoco-hand-motion.XXXXXX")"
  override_file="$motion_tmp_dir/override.json"
  homie_config="$motion_tmp_dir/homie_stand.yaml"
  "$PATH_PYTHON" - "$BOOST_ROOT/configs/sim/homie_g1_mujoco.yaml" "$homie_config" <<'PY'
import sys
from pathlib import Path

import yaml

source = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
source["arm_source"] = "hold"
Path(sys.argv[2]).write_text(
    yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
)
PY
  echo "Hand MPPI input: synthetic fist/open stream at 30 Hz -> temporary JSON"
  echo "HOMIE arms hold the configured default; rt/arm_sdk and PATH targets are not consumed"
  echo "No Galaxy XR, Televuer, PATH bridge, or physical-robot process is started"
  start "$PATH_PYTHON" scripts/stream_mujoco_hand_motion.py --output "$override_file"
  motion_pid="${pids[-1]}"
  for _ in {1..50}; do
    [[ -s "$override_file" ]] && break
    if ! kill -0 "$motion_pid" 2>/dev/null; then
      wait "$motion_pid"
      echo "Hand motion generator exited before creating its JSON input" >&2
      exit 1
    fi
    sleep 0.1
  done
  if [[ ! -s "$override_file" ]]; then
    echo "Timed out waiting for the hand motion generator to write its initial pose" >&2
    exit 1
  fi
else
  echo "Starting HOMIE stand/walk policy; use the MuJoCo viewer keyboard for locomotion"
  echo "W/S forward/back, A/D left/right, Q/R yaw, O/P speed, Space stop"
  echo "Hand MPPI input: $override_file"
  homie_config="$BOOST_ROOT/configs/sim/homie_g1_mujoco.yaml"
fi
start "${sim_args[@]}"
if ((!motion)); then
  sleep 2
fi
homie_args=(
  "$PATH_PYTHON" "$BOOST_HOMIE_ROOT/deploy/run_homie.py"
  --net "$SIM_DDS_INTERFACE"
  --config "$homie_config"
  --auto-start
)
if ((motion)); then
  homie_args+=(--stand-policy-only)
fi
# Start HOMIE immediately. Motion mode keeps simulation physics paused at its
# configured initial pose until HOMIE's first command, then uses stand.onnx at
# zero velocity. The hand runner starts after HOMIE's default-pose ramp.
start_homie "${homie_args[@]}"
sleep 3
start "$PIXI_CLI" run -e hand python -m boost.control.hand_mppi.runner \
  --config configs/hands/brainco_revo2_mppi.yaml \
  --input-mode override --override-file "$override_file" \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
  --publish

if ((motion)); then
  echo "MuJoCo hand motion test is running; Ctrl-C stops all local processes and removes the temporary JSON."
else
  echo "MuJoCo hand test is running. Edit the override JSON to change hand pose; Ctrl-C stops all local processes."
fi
wait -n "${pids[@]}"
