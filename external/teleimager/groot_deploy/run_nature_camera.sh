#!/usr/bin/env bash
set -euo pipefail

TELEIMAGER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TELEIMAGER_PYTHON="${TELEIMAGER_PYTHON:-/home/unitree/miniconda3/envs/teleimager/bin/python}"

if [[ ! -x "$TELEIMAGER_PYTHON" ]]; then
  echo "Teleimager Python not found: $TELEIMAGER_PYTHON" >&2
  exit 1
fi

# Pin both source and configuration to this checkout, independently of pip's
# editable install or the caller's activated conda environment.
export PYTHONPATH="$TELEIMAGER_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XR_TELEOP_CERT="${XR_TELEOP_CERT:-$TELEIMAGER_DIR/certs/g1-29/cert.pem}"
export XR_TELEOP_KEY="${XR_TELEOP_KEY:-$TELEIMAGER_DIR/certs/g1-29/key.pem}"
cd "$TELEIMAGER_DIR"
exec "$TELEIMAGER_PYTHON" -m teleimager.image_server \
  --rs --config "$TELEIMAGER_DIR/config/dex1_cam_nature.yaml" "$@"
