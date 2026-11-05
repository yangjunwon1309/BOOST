#!/usr/bin/env bash
set -euo pipefail

ADDON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TELEIMAGER_DIR="$(dirname "${ADDON_DIR}")"

if ! command -v teleimager-server >/dev/null 2>&1; then
  echo "teleimager-server is not available in the active G1 conda environment." >&2
  exit 1
fi

cd -- "${TELEIMAGER_DIR}"
exec teleimager-server --rs "$@"

