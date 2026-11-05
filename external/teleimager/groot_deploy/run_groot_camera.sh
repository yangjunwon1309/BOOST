#!/usr/bin/env bash
set -euo pipefail

ADDON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ADDON_DIR}/hbvcam_4m2214hd_g1_groot.yaml"

if ! command -v teleimager-server >/dev/null 2>&1; then
  echo "teleimager-server is not available in the active G1 conda environment." >&2
  exit 1
fi

exec teleimager-server \
  --rs \
  --no-affinity \
  --config "${CONFIG_PATH}" \
  "$@"
