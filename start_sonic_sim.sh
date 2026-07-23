#!/usr/bin/env bash
# Start the SONIC MuJoCo/DDS simulation from PATH-G1.
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PATH_G1_ROOT/configs/runtime/local.env"
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"

export CYCLONEDDS_NETWORK_INTERFACE="${CYCLONEDDS_NETWORK_INTERFACE:-lo}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$PATH_G1_ROOT"
exec taskset -c "${PATH_SIM_CPUSET:-4-5}" \
  python sonic_runtime/gear_sonic/scripts/run_sim_loop.py \
  --enable-offscreen \
  --camera-port "${SONIC_SIM_CAMERA_PORT:-5555}" \
  --env_name "${SONIC_SIM_ENV_NAME:-default}" \
  --enable-image-publish \
  "$@"
