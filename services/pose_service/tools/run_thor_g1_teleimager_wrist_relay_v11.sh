#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
perception_dir="$(cd -- "${script_dir}/.." && pwd)"
boost_root="$(cd -- "${perception_dir}/../.." && pwd)"
export BOOST_ROOT="$boost_root"
source "$BOOST_ROOT/configs/teleop/local.env"
cd "${perception_dir}"
exec "${PATH_PYTHON:-python}" -m remote_offload.thor_teleimager_wrist_relay_v11 \
  --config "$BOOST_TELEIMAGER_RELAY_CONFIG" "$@"
