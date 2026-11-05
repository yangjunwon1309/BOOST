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
if [[ ! -d "$BOOST_DREAMWAQ_ROOT/deploy" ]]; then
  echo "DreamWaQ runtime snapshot not found: $BOOST_DREAMWAQ_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$BOOST_DREAMWAQ_ENV_ROOT")"
"$python310_bin" -m venv "$BOOST_DREAMWAQ_ENV_ROOT"
python="$BOOST_DREAMWAQ_PYTHON"
"$python" -m pip install --upgrade pip setuptools wheel
if [[ -n "${BOOST_TORCH_INDEX_URL:-}" ]]; then
  "$python" -m pip install --index-url "$BOOST_TORCH_INDEX_URL" 'torch>=2.1,<3'
else
  "$python" -m pip install 'torch>=2.1,<3'
fi
"$python" -m pip install \
  'numpy>=1.23,<2' \
  'PyYAML>=6,<7' \
  'cyclonedds==0.10.5' \
  'unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python.git@814556d15970dd2ecf1c9984e845ca02ab07e206'
"$python" -m pip install --no-deps -e "$BOOST_DREAMWAQ_ROOT"
"$python" -c 'import numpy, torch, yaml, cyclonedds, unitree_sdk2py; print("DreamWaQ runtime imports OK; torch", torch.__version__)'
echo "DreamWaQ runtime installed at $BOOST_DREAMWAQ_ENV_ROOT"
