#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${UNITREE_SDK2_ROOT:?Set UNITREE_SDK2_ROOT to the SDK2 source or install root}"

BRIDGE_DIR="$PATH_G1_ROOT/boost/io/g1_bridge"
cmake -S "$BRIDGE_DIR" -B "$BRIDGE_DIR/build" \
  -DUNITREE_SDK2_ROOT="$UNITREE_SDK2_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BRIDGE_DIR/build" --parallel
