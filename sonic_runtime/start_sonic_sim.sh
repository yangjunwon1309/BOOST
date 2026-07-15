#!/usr/bin/env bash
# Compatibility wrapper for users who run from sonic_runtime/.
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$PATH_G1_ROOT/start_sonic_sim.sh" "$@"

