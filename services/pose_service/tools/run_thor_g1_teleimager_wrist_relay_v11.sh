#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
perception_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${perception_dir}"
exec "${PATH_PYTHON:-python}" -m remote_offload.thor_teleimager_wrist_relay_v11 "$@"
