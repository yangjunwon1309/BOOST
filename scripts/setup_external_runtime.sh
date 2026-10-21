#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_ROOT="$BOOST_ROOT/external"
TELEVUER_ROOT="${PATH_TELEVUER_ROOT:-$EXTERNAL_ROOT/televuer}"
UNITREE_SDK2PY_ROOT="${UNITREE_SDK2PY_ROOT:-$EXTERNAL_ROOT/unitree_sdk2_python}"
UNITREE_SDK2_ROOT="${UNITREE_SDK2_ROOT:-$EXTERNAL_ROOT/unitree_sdk2}"
VENV="$BOOST_ROOT/.venv_runtime"

command -v uv >/dev/null || {
  echo "uv is required; install it with https://docs.astral.sh/uv/" >&2
  exit 2
}
command -v git >/dev/null || {
  echo "git is required to fetch the external Unitree SDKs" >&2
  exit 2
}

mkdir -p "$EXTERNAL_ROOT"
if [[ ! -f "$TELEVUER_ROOT/pyproject.toml" ]]; then
  git clone --depth 1 https://github.com/unitreerobotics/televuer.git "$TELEVUER_ROOT"
fi
if [[ ! -f "$UNITREE_SDK2PY_ROOT/pyproject.toml" && ! -f "$UNITREE_SDK2PY_ROOT/setup.py" ]]; then
  git clone --depth 1 --recurse-submodules \
    https://github.com/unitreerobotics/unitree_sdk2_python.git "$UNITREE_SDK2PY_ROOT"
fi
if [[ ! -f "$UNITREE_SDK2_ROOT/CMakeLists.txt" ]]; then
  git clone --depth 1 --recurse-submodules \
    https://github.com/unitreerobotics/unitree_sdk2.git "$UNITREE_SDK2_ROOT"
fi

uv python install 3.10
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python "$(uv python find --no-project 3.10)" --prompt boost_runtime
fi

cd "$BOOST_ROOT"
uv pip install --python "$VENV/bin/python" -e ".[sim]"
uv pip install --python "$VENV/bin/python" -e "$TELEVUER_ROOT"
uv pip install --python "$VENV/bin/python" -e "$UNITREE_SDK2PY_ROOT"

"$VENV/bin/python" - <<'PY'
import mujoco
import pinocchio
import casadi
import lcm
import unitree_sdk2py
from pathlib import Path

scene = Path("boost/sim/mujoco/scene_29dof_with_revo2.xml")
model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
assert (model.nq, model.nv, model.nu) == (58, 57, 41)
print(f"Runtime ready: MuJoCo {mujoco.__version__}; G1+Revo2 nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
PY

echo "Runtime ready. Source configs/teleop/local.env before launching scripts."
