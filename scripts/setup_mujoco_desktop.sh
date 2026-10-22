#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${PIXI_PROJECT_ROOT:-}" != "$BOOST_ROOT" ]]; then
  echo "Run setup through Pixi: pixi run -e sim setup-sim" >&2
  exit 2
fi
cd "$BOOST_ROOT"
git submodule update --init --depth 1 --filter=blob:none external/brainco-description
python - <<'PY'
from pathlib import Path

import casadi
import lcm
import mujoco
import mujoco.viewer
import onnxruntime
import pinocchio
import unitree_sdk2py
import yaml
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

from boost.io.path_transport.g1_direct_protocol import pack_robot_state
from boost.io.unitree_crc import make_unitree_crc

with Path("configs/sim/mujoco_revo2.yaml").open("r", encoding="utf-8") as stream:
    sim_config = yaml.safe_load(stream)
scene = Path(sim_config["scene"])
model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
assert model.nu == 41, (model.nq, model.nv, model.nu)
print(
    f"Pixi MuJoCo runtime ready: MuJoCo {mujoco.__version__}; "
    f"scene={scene}; G1 + Revo2 nq/nv/nu={model.nq}/{model.nv}/{model.nu}"
)
crc = make_unitree_crc()
crc.Crc(unitree_hg_msg_dds__LowCmd_())
print("PATH LCM/DDS bridge ready: boost/sim/mujoco/path_dds_bridge.py")
PY

PIXI_CLI="$(command -v pixi || true)"
if [[ -z "$PIXI_CLI" && -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_CLI="$HOME/.pixi/bin/pixi"
fi
PIXI_CLI="${PIXI_CLI:-pixi}"
echo "Run: $PIXI_CLI run -e sim mujoco"
echo "Without a Galaxy XR headset: $PIXI_CLI run -e sim mujoco-no-xr"
