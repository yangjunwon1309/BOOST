#!/usr/bin/env bash
# Convenience launcher for run_homie.py -- activates the `homie` conda env and
# resolves paths relative to this script, so you don't have to `cd` here or
# type the full path to run_homie.py by hand.
#
# Usage:
#   ./run_homie.sh --net eth0                 # normal run
#   ./run_homie.sh --net eth0 --dry-run        # dry run (robot held in damping)
#   ./run_homie.sh --net lo --stand-only --auto-start --config configs/sim/homie_g1_mujoco.yaml
#   HOMIE_NET_IFACE=eth0 ./run_homie.sh         # omit --net, use the env var default
#   HOMIE_CONDA_ENV=other_env ./run_homie.sh --net eth0
#
# Any extra flags (--stand, --walk, --config, --ramp-time, ...) are forwarded
# to run_homie.py as-is.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${HOMIE_CONDA_ENV:-homie}"
NET_IFACE="${HOMIE_NET_IFACE:-eth0}"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
    echo "Error: conda not found on PATH. Activate/install conda first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$HERE"

args=("$@")
has_net=0
for a in "${args[@]:-}"; do
    if [[ "$a" == "--net" || "$a" == --net=* ]]; then
        has_net=1
        break
    fi
done

if [[ "$has_net" -eq 0 ]]; then
    args=(--net "$NET_IFACE" "${args[@]:-}")
fi

echo "[run_homie.sh] conda env: $CONDA_ENV"
echo "[run_homie.sh] running: python run_homie.py ${args[*]}"
exec python run_homie.py "${args[@]}"
