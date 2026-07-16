#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../configs/runtime/local.env"

cd "$PATH_SONIC_DEPLOY_ROOT"
PATH_DYNAMIC_REFERENCE_ENABLED="${PATH_DYNAMIC_REFERENCE_ENABLED:-1}" \
PATH_REFERENCE_PORT="${PATH_REFERENCE_PORT:-5560}" \
PATH_REFERENCE_TOPIC="${PATH_REFERENCE_TOPIC:-path_qp_reference}" \
bash teleop_sonic_mujoco_ik.sh "$@"

