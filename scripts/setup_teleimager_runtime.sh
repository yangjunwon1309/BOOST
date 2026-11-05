#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"

python310_bin="${PYTHON310_BIN:-$(command -v python3.10 || true)}"
if [[ -z "$python310_bin" ]]; then
  echo "Python 3.10 is required; set PYTHON310_BIN to its executable." >&2
  exit 2
fi
if [[ ! -f "$BOOST_TELEIMAGER_ROOT/pyproject.toml" ]]; then
  echo "Teleimager source snapshot not found: $BOOST_TELEIMAGER_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$BOOST_TELEIMAGER_ENV_ROOT")"
"$python310_bin" -m venv "$BOOST_TELEIMAGER_ENV_ROOT"
"$BOOST_TELEIMAGER_ENV_ROOT/bin/python" -m pip install --upgrade pip setuptools wheel
"$BOOST_TELEIMAGER_ENV_ROOT/bin/python" -m pip install -e "$BOOST_TELEIMAGER_ROOT[server]"
"$BOOST_TELEIMAGER_ENV_ROOT/bin/python" -c 'import cv2, teleimager, zmq, yaml; print("Teleimager runtime imports OK; OpenCV", cv2.__version__)'
echo "Teleimager runtime installed at $BOOST_TELEIMAGER_ENV_ROOT"
