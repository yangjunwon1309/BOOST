#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../configs/runtime/local.env"

cd "$PATH_SONIC_DEPLOY_ROOT"
PATH_BSPLINE_ENABLED="${PATH_BSPLINE_ENABLED:-1}" \
PATH_BSPLINE_PORT="${PATH_BSPLINE_PORT:-5560}" \
PATH_BSPLINE_TOPIC="${PATH_BSPLINE_TOPIC:-curobo_bspline}" \
bash teleop_sonic_mujoco_ik.sh "$@"

