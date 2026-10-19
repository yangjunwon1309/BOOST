#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cat <<'EOF'
[PATH-direct] Thor publishes the 14-arm G1_PATH_UPPER_COMMAND packet.
[PATH-direct] Start the packaged g1_bridge/run.sh process on the robot.
EOF

exec "$ROOT/scripts/run_state_machine_bridge.sh" "$@"
