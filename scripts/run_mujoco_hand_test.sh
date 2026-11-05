#!/usr/bin/env bash
set -eo pipefail

BOOST_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"
source "$BOOST_ROOT/configs/sim/local.env"
# This test intentionally cannot be redirected onto a robot-facing interface.
SIM_DDS_INTERFACE=lo
SIM_DDS_DOMAIN_ID=0
export SIM_DDS_INTERFACE SIM_DDS_DOMAIN_ID
cd "$BOOST_ROOT"

sim_config="configs/sim/mujoco_revo2.yaml"
if [[ -n "$MUJOCO_SIM_CONFIG" ]]; then
  sim_config="$MUJOCO_SIM_CONFIG"
fi
if [[ "$sim_config" != /* ]]; then
  sim_config="$BOOST_ROOT/$sim_config"
fi
if [[ ! -f "$sim_config" ]]; then
  echo "MuJoCo sim config not found: $sim_config" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Run DreamWaQ in MuJoCo with Revo2 hand MPPI reading a local JSON pose.
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
override_file="configs/hands/revo2_hand_override_fist.json"
if [[ -n "$MUJOCO_HAND_OVERRIDE_FILE" ]]; then
  override_file="$MUJOCO_HAND_OVERRIDE_FILE"
fi
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

if [[ "$PIXI_PROJECT_ROOT" != "$BOOST_ROOT" ]]; then
  echo "Run this test through Pixi: pixi run -e sim mujoco-hand-test" >&2
  exit 2
fi
if [[ ! -f "$BOOST_DREAMWAQ_EXPORT/manifest.json" ]]; then
  echo "DreamWaQ export manifest not found at $BOOST_DREAMWAQ_EXPORT/manifest.json" >&2
  exit 2
fi

if [[ "$override_file" != /* ]]; then
  override_file="$BOOST_ROOT/$override_file"
fi
if ((!motion)) && [[ ! -f "$override_file" ]]; then
  echo "Hand override file not found: $override_file" >&2
  exit 2
fi

PIXI_CLI="$(command -v pixi || true)"
if [[ -z "$PIXI_CLI" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_CLI="$HOME/.pixi/bin/pixi"
fi
if [[ -z "$PIXI_CLI" ]]; then
  echo "Pixi executable not found; set PIXI_CLI or add ~/.pixi/bin to PATH." >&2
  exit 2
fi

motion_tmp_dir="$(mktemp -d /tmp/boost-mujoco-hand-test.XXXXXX)"
body_config="$motion_tmp_dir/dreamwaq_hold.yaml"
if ((motion)); then
  override_file="$motion_tmp_dir/override.json"
fi
body_config_source="$("$PATH_PYTHON" - "$BOOST_ROOT" "$sim_config" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
sim_config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
print(root / sim_config["body_config"])
PY
)"
"$PATH_PYTHON" - "$body_config_source" "$body_config" <<'PY'
import sys
from pathlib import Path
import yaml

source = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
source["arm_source"] = "hold"
Path(sys.argv[2]).write_text(
    yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
)
PY

pids=""
signal_groups() {
  local signal="$1"
  local pid
  for pid in $pids; do
    kill -"$signal" -- "-$pid" 2>/dev/null || true
  done
}

wait_for_groups() {
  local attempt
  local pid
  local alive
  for attempt in {1..50}; do
    alive=0
    for pid in $pids; do
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
  signal_groups INT
  if ! wait_for_groups; then
    signal_groups TERM
    if ! wait_for_groups; then
      signal_groups KILL
    fi
  fi
  local pid
  for pid in $pids; do
    wait "$pid" 2>/dev/null || true
  done
  rm -rf -- "$motion_tmp_dir"
  exit "$status"
}
trap cleanup EXIT INT TERM

start() {
  setsid "$@" &
  pids="$pids $!"
}

if ((motion)); then
  echo "Hand MPPI input: synthetic fist/open stream at 30 Hz -> temporary JSON"
  echo "DreamWaQ runs in-process with arm_source=hold; no PATH arm target is consumed"
  echo "No Galaxy XR, Televuer, PATH bridge, or physical-robot process is started"
  start "$PATH_PYTHON" scripts/stream_mujoco_hand_motion.py --output "$override_file"
  motion_pid="$!"
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
  echo "Hand MPPI input: $override_file"
fi

echo "Starting local MuJoCo Revo2 simulation on $SIM_DDS_INTERFACE, domain $SIM_DDS_DOMAIN_ID"
if ((headless && motion)); then
  start "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py --config "$sim_config" \
    --body-config "$body_config" --dreamwaq-export "$BOOST_DREAMWAQ_EXPORT" \
    --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
    --hold-start-delay-s 5.0 --headless
elif ((motion)); then
  start "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py --config "$sim_config" \
    --body-config "$body_config" --dreamwaq-export "$BOOST_DREAMWAQ_EXPORT" \
    --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
    --hold-start-delay-s 5.0
elif ((headless)); then
  start "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py --config "$sim_config" \
    --body-config "$body_config" --dreamwaq-export "$BOOST_DREAMWAQ_EXPORT" \
    --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" --headless
else
  start "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py --config "$sim_config" \
    --body-config "$body_config" --dreamwaq-export "$BOOST_DREAMWAQ_EXPORT" \
    --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID"
fi

echo "DreamWaQ is evaluated inside MuJoCo; no rt/lowcmd subscriber or hardware writer is started."
echo "Use W/S, A/D, Q/R, O/P and Space in the viewer to control locomotion."
sleep 2
start "$PIXI_CLI" run -e hand python -m boost.control.hand_mppi.runner \
  --config configs/hands/brainco_revo2_mppi.yaml \
  --input-mode override --override-file "$override_file" \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
  --publish

if ((motion)); then
  echo "MuJoCo hand motion test is running; Ctrl-C stops local processes and removes temporary files."
else
  echo "MuJoCo hand test is running. Edit the override JSON to change hand pose; Ctrl-C stops local processes."
fi
wait -n $pids
