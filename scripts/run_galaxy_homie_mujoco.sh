#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"
source "$BOOST_ROOT/configs/sim/local.env"
cd "$BOOST_ROOT"

sim_config="${MUJOCO_SIM_CONFIG:-configs/sim/mujoco_revo2.yaml}"
if [[ ! -f "$sim_config" ]]; then
  echo "MuJoCo sim config not found: $sim_config" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Run the local G1 + Revo2 MuJoCo stack with HOMIE, PATH, optional Galaxy XR,
and optional continuous Televuer-to-Revo2 hand retargeting.

Usage: pixi run -e sim mujoco [--no-xr] [--headless] [--hand-mppi-xr]

Options:
  --no-xr     Start MuJoCo, HOMIE, and the PATH/G1 bridges without a headset.
  --headless  Run the physics loop without opening the MuJoCo viewer.
  --hand-mppi-xr Enable Televuer hand tracking and the Mimic Revo2 DDS publisher.
  -h, --help  Show this help.
EOF
}

start_xr=1
headless=0
hand_mppi=0
while (($#)); do
  case "$1" in
    --no-xr) start_xr=0 ;;
    --headless) headless=1 ;;
    --hand-mppi-xr) hand_mppi=1 ;;
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
if ((start_xr)); then
  source "$BOOST_ROOT/scripts/ensure_televuer_tls.sh"
  ensure_televuer_tls
  "$PATH_PYTHON" - <<'PY'
import socket
import sys

ports = (5565, 5566, 8012)
busy = []
for port in ports:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        busy.append(port)
    finally:
        sock.close()

if busy:
    print(
        "Galaxy XR startup ports are already in use: "
        + ", ".join(map(str, busy))
        + ". Stop the old Televuer/PATH process; inspect owners with "
        + "`ss -ltnp '( sport = :5565 or sport = :5566 or sport = :8012 )'`.",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
fi
if ((hand_mppi && !start_xr)); then
  echo "Televuer-to-Revo2 retargeting needs Galaxy XR hand tracking; remove --no-xr." >&2
  exit 2
fi

if ((hand_mppi)); then
  export PATH_TELEVUER_HAND_TRACKING=1
  # Galaxy hand tracking has no physical A+X buttons. In this MuJoCo-only
  # launcher, enable arm tracking on a fresh Televuer wrist frame.
  export PATH_AUTO_ARM_PICO_TRACKING=1
  surface_query_endpoint="${MUJOCO_SURFACE_QUERY_ENDPOINT:-tcp://127.0.0.1:5570}"
  PIXI_CLI="${PIXI_CLI:-$(command -v pixi || true)}"
  if [[ -z "$PIXI_CLI" && -x "$HOME/.pixi/bin/pixi" ]]; then
    PIXI_CLI="$HOME/.pixi/bin/pixi"
  fi
  if [[ -z "$PIXI_CLI" ]]; then
    echo "Pixi executable not found; set PIXI_CLI or add ~/.pixi/bin to PATH." >&2
    exit 2
  fi
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
  last_started_pid=$!
  pids+=("$last_started_pid")
}

echo "Starting local simulation on DDS interface $SIM_DDS_INTERFACE, domain $SIM_DDS_DOMAIN_ID"
echo "LCM simulation transport: $PATH_LCM_URL"
sim_args=(
  "$PATH_PYTHON" boost/sim/mujoco/run_dds_sim.py
  --config "$sim_config"
  --interface "$SIM_DDS_INTERFACE"
  --domain-id "$SIM_DDS_DOMAIN_ID"
)
if ((headless)); then
  sim_args+=(--headless)
fi
if ((hand_mppi)); then
  sim_args+=(--surface-query-endpoint "$surface_query_endpoint")
fi
start "${sim_args[@]}"
sleep 2
start "$PATH_PYTHON" boost/sim/mujoco/path_dds_bridge.py \
  --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
  --lcm-url "$PATH_LCM_URL"
sleep 1
if ((hand_mppi)); then
  echo "Galaxy XR wrist poses feed PATH arm IK; PATH auto-arms on the first fresh XR frame."
  echo "Continuous Revo2 retargeting is enabled; PATH's legacy binary grip DDS publisher is disabled."
else
  start "$PATH_PYTHON" boost/sim/mujoco/path_revo2_grip_bridge.py \
    --config "$sim_config" \
    --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID"
fi
start "$PATH_PYTHON" "$BOOST_HOMIE_ROOT/deploy/run_homie.py" \
  --net "$SIM_DDS_INTERFACE" --config configs/sim/homie_g1_mujoco.yaml \
  --auto-start
sleep 2
start bash scripts/run_state_machine_bridge.sh
if ((start_xr)); then
  start bash scripts/run_galaxy_xr_manager.sh
  xr_manager_pid="$last_started_pid"
  sleep 1
  xr_manager_state="$(ps -o stat= -p "$xr_manager_pid" 2>/dev/null || true)"
  if [[ -z "$xr_manager_state" || "$xr_manager_state" == Z* ]]; then
    echo "Galaxy XR manager exited during startup; stopping the local stack." >&2
    exit 1
  fi
else
  echo "Galaxy XR is disabled; PATH is waiting for input on its configured ZMQ endpoints."
fi
if ((hand_mppi)); then
  sleep 1
  start "$PIXI_CLI" run -e hand python -m boost.control.hand_mppi.runner \
    --config configs/hands/brainco_revo2_mppi.yaml \
    --input-endpoint "${PATH_PICO_VR_ENDPOINT:-tcp://127.0.0.1:5566}" \
    --surface-query-endpoint "$surface_query_endpoint" \
    --interface "$SIM_DDS_INTERFACE" --domain-id "$SIM_DDS_DOMAIN_ID" \
    --publish
fi

echo "Stack is running. Press Ctrl-C to stop all local processes."
wait -n "${pids[@]}"
