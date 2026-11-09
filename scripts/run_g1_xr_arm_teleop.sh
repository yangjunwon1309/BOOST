#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"

if [[ "${PIXI_PROJECT_ROOT:-}" != "$BOOST_ROOT" ]]; then
  echo "Run this teleop on the XR/teleop host through Pixi: pixi run -e thor-teleop g1-arm-teleop" >&2
  exit 2
fi
if [[ "${BOOST_ALLOW_HARDWARE_CONTROL:-0}" != "1" ]]; then
  echo "Hardware control is gated. Set BOOST_ALLOW_HARDWARE_CONTROL=1 for an intentional G1 launch." >&2
  exit 2
fi

network_interface="${G1_NETWORK_INTERFACE:-eth0}"
teleop_python="${BOOST_G1_TELEOP_PYTHON:-$PATH_PYTHON}"
teleimager_host="${BOOST_G1_TELEIMAGER_HOST:-$G1_TELEIMAGER_HOST}"
xr_host="${BOOST_G1_XR_HOST:-}"
if [[ -z "$xr_host" ]] && command -v ip >/dev/null 2>&1; then
  xr_host="$(ip -o -4 addr show dev "$network_interface" | awk 'NR == 1 { split($4, ip, "/"); print ip[1] }')"
fi
if [[ -z "$xr_host" ]]; then
  echo "Set BOOST_G1_XR_HOST to the address the Galaxy XR headset can reach." >&2
  exit 2
fi
export BOOST_G1_XR_HOST="$xr_host"
data_root="${BOOST_G1_TELEOP_DATA_ROOT:-$HOME/data/boost/g1_teleop}"
task_name="${BOOST_G1_TELEOP_TASK_NAME:-g1_arm_teleop}"

if ! command -v "$teleop_python" >/dev/null 2>&1; then
  echo "Teleop Python not found: $teleop_python" >&2
  exit 2
fi
export PYTHONPATH="$BOOST_ROOT/external/g1_arm_teleop:$BOOST_ROOT/external/g1_arm_teleop/televuer/src:${PYTHONPATH:-}"

source "$BOOST_ROOT/scripts/ensure_televuer_tls.sh"
ensure_televuer_tls
export XR_TELEOP_CERT="$PATH_TELEVUER_CERT"
export XR_TELEOP_KEY="$PATH_TELEVUER_KEY"

# Televuer binds 8012. Refuse to steal a headset session from another process.
"$teleop_python" - <<'PY'
import socket
sock = socket.socket()
try:
    sock.bind(("0.0.0.0", 8012))
except OSError as error:
    raise SystemExit(f"XR port 8012 is already in use; stop the existing Televuer/PATH session: {error}")
finally:
    sock.close()
PY

# Camera config and model/runtime imports must be valid before arm publishers
# are constructed. Camera streams are served by the G1 host over TIM2.
"$teleop_python" - "$teleimager_host" <<'PY'
import sys
import casadi
import pinocchio
import televuer
import unitree_sdk2py
from teleop.utils.multi_host_image_client import request_camera_config

host = sys.argv[1]
config = request_camera_config(host)
for name in ("head_camera", "left_wrist_camera", "right_wrist_camera"):
    camera = config.get(name)
    if not isinstance(camera, dict):
        raise SystemExit(f"Teleimager {host} did not return {name} configuration")
    if not camera.get("enable_recording_zmq", False):
        raise SystemExit(f"{name} on {host} needs enable_recording_zmq: true for timestamped episode capture")
print(f"Verified teleop imports and timestamped camera config at {host}:60000")
PY

# On a shared host, catch the old PATH adapter: it also publishes rt/arm_sdk.
if command -v pgrep >/dev/null 2>&1 && pgrep -af '[g]1_lowlevel_bridge|[t]eleop_arm_only\.py|[t]eleop_ikea\.py|[t]eleop_hand_and_arm\.py' >/dev/null; then
  echo "A local G1 arm adapter or arm teleop is already running; stop it before starting this publisher." >&2
  exit 2
fi

args=(
  --arm G1_29
  --input-mode "${BOOST_G1_XR_INPUT_MODE:-controller}"
  --display-mode "${BOOST_G1_XR_DISPLAY_MODE:-pass-through}"
  --motion --record --headless --camera-clock-sync
  --network-interface "$network_interface"
  --img-server-ip "$teleimager_host"
  --task-dir "$data_root"
  --task-name "$task_name"
  --task-goal "${BOOST_G1_TELEOP_TASK_GOAL:-XR teleoperation of G1 arms}"
  --task-desc "${BOOST_G1_TELEOP_TASK_DESC:-G1_29 arm-only teleoperation with DreamWaQ lower-body control}"
  --task-steps "${BOOST_G1_TELEOP_TASK_STEPS:-approach; grasp; manipulate; release}"
)

echo "Starting arm-only Galaxy XR teleop on $network_interface."
echo "DreamWaQ must already own rt/lowcmd on the G1 host; this process publishes rt/arm_sdk."
echo "Record: [r] tracking, [s] start/save, [1..9]/[0] labels, hold [q] 3s to stop."
echo "XR URL: https://$xr_host:8012/"
echo "Data root: $data_root/$task_name"
exec "$teleop_python" "$BOOST_ROOT/external/g1_arm_teleop/teleop/teleop_arm_only.py" "${args[@]}" "$@"
