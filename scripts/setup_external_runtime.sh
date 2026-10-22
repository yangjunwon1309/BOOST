#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI_BIN="$(command -v pixi || true)"
if [[ -z "$PIXI_BIN" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
fi
if [[ -z "$PIXI_BIN" ]]; then
  echo "Pixi is required. Install Pixi, then run: pixi run -e sim setup-sim" >&2
  exit 2
fi

exec "$PIXI_BIN" run --manifest-path "$BOOST_ROOT/pixi.toml" -e sim setup-sim "$@"
