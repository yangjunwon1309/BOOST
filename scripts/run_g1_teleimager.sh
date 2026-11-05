#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"

if [[ ! -x "$BOOST_TELEIMAGER_BIN" ]]; then
  echo "Teleimager runtime missing; run scripts/setup_teleimager_runtime.sh first." >&2
  exit 2
fi
if [[ ! -f "$BOOST_TELEIMAGER_G1_CONFIG" ]]; then
  echo "Teleimager camera config not found: $BOOST_TELEIMAGER_G1_CONFIG" >&2
  exit 2
fi

echo "Starting the single G1 camera owner: Teleimager with TIM2 on recording ports."
echo "Head port 55558; left/right wrist ports 55559/55560 (from the selected preset)."
exec "$BOOST_TELEIMAGER_BIN" \
  --config "$BOOST_TELEIMAGER_G1_CONFIG" \
  --camera-cpus "$G1_TELEIMAGER_CAMERA_CPUSET" \
  "$@"
