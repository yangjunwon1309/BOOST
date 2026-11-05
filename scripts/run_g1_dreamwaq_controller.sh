#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOST_ROOT
source "$BOOST_ROOT/configs/teleop/local.env"

if [[ "${BOOST_ALLOW_HARDWARE_CONTROL:-0}" != "1" ]]; then
  echo "Hardware control is gated. Set BOOST_ALLOW_HARDWARE_CONTROL=1 for an intentional G1 launch." >&2
  exit 2
fi
if [[ ! -x "$BOOST_DREAMWAQ_PYTHON" ]]; then
  echo "DreamWaQ runtime missing; run scripts/setup_dreamwaq_runtime.sh first." >&2
  exit 2
fi
if [[ ! -f "$BOOST_DREAMWAQ_CONFIG" || ! -f "$BOOST_DREAMWAQ_EXPORT/manifest.json" ]]; then
  echo "DreamWaQ config or selected backpack export is missing." >&2
  echo "config: $BOOST_DREAMWAQ_CONFIG" >&2
  echo "export: $BOOST_DREAMWAQ_EXPORT" >&2
  exit 2
fi
"$BOOST_DREAMWAQ_PYTHON" - "$BOOST_DREAMWAQ_EXPORT" <<'PY'
import json
import pathlib
import sys

export = pathlib.Path(sys.argv[1])
manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
payload_path = export / "backpack_payload.json"
if not payload_path.is_file():
    raise SystemExit("selected DreamWaQ export has no backpack training receipt")
payload = json.loads(payload_path.read_text(encoding="utf-8"))
if payload.get("backpack_enabled") is not True:
    raise SystemExit("selected DreamWaQ export is not marked as backpack-trained")
if manifest.get("num_actions") != 12:
    raise SystemExit(
        "this hardware integration requires the reviewed 12-action lower-body export; "
        f"got {manifest.get('num_actions')!r}"
    )
if manifest.get("action_contract_version") != "g1_lower_body_action_v3_canonical_waist_effort":
    raise SystemExit("selected DreamWaQ export does not match the reviewed waist-effort contract")
PY
if [[ "${G1_BRIDGE_LIVE:-1}" != "1" ]]; then
  echo "G1_BRIDGE_LIVE must be 1 so PATH arm targets reach rt/arm_sdk." >&2
  exit 2
fi
if command -v pgrep >/dev/null 2>&1 && pgrep -af 'run_dreamwaq\.py|dreamwaq-g1-29dof|run_homie\.py|g1_lowlevel_bridge' >/dev/null; then
  echo "A DreamWaQ/HOMIE/g1_ws low-level controller or bridge is already running." >&2
  echo "Stop the existing owner before starting this stack; there must be one rt/lowcmd writer." >&2
  exit 2
fi

network_interface="${G1_NETWORK_INTERFACE:-eth0}"
pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "Starting PATH arm/state adapter on $network_interface; it publishes rt/arm_sdk only."
G1_BRIDGE_LIVE=1 bash "$BOOST_ROOT/boost/io/g1_bridge/run.sh" \
  "$network_interface" --hand-backend none &
pids+=("$!")
sleep 1

echo "Starting the target DreamWaQ hardware runtime as the sole rt/lowcmd owner."
echo "Initial mode is standing. Use the target deployment's START then A procedure to enable control."
"$BOOST_DREAMWAQ_PYTHON" "$BOOST_DREAMWAQ_ROOT/deploy/run_dreamwaq.py" \
  --net "$network_interface" \
  --config "$BOOST_DREAMWAQ_CONFIG" \
  --export "$BOOST_DREAMWAQ_EXPORT" \
  --locomotion-mode standing \
  --mode-toggle-button X \
  --device cpu \
  --policy-timeout-ms 15 \
  --max-policy-timeouts 3 \
  --arm-tau-mode forward \
  "$@"
