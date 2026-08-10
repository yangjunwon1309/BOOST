#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cat <<'EOF'
[PATH-direct] SONIC policy is not started by this entrypoint.
[PATH-direct] Thor publishes G1_PATH_UPPER_COMMAND only; start the G1-local
[PATH-direct] path_direct_lcm_bridge and g1_lowlevel_bridge separately.
EOF

exec "$ROOT/scripts/run_state_machine_bridge.sh" "$@"
