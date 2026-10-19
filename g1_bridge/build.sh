#!/usr/bin/env bash
set -euo pipefail

PATH_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${UNITREE_SDK2_ROOT:?Set UNITREE_SDK2_ROOT to the SDK2 source or install root}"

cmake -S "$PATH_G1_ROOT/g1_bridge" -B "$PATH_G1_ROOT/g1_bridge/build" \
  -DUNITREE_SDK2_ROOT="$UNITREE_SDK2_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$PATH_G1_ROOT/g1_bridge/build" --parallel
