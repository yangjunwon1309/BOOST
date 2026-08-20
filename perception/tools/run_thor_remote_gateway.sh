#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/run_thor.sh" python -m remote_offload.thor_gateway_head_d435i_legacy "$@"
