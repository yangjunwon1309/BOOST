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

"$BOOST_DREAMWAQ_PYTHON" - "$BOOST_DREAMWAQ_CONFIG" "$BOOST_DREAMWAQ_EXPORT" <<'PY'
import json
import pathlib
import sys
import yaml

config_path, export_text = map(pathlib.Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if config.get("arm_source") != "arm_sdk":
    raise SystemExit("DreamWaQ config must set arm_source: arm_sdk to receive the remote XR arm targets")
export = json.loads((export_text / "manifest.json").read_text(encoding="utf-8"))
payload_path = export_text / "backpack_payload.json"
if not payload_path.is_file():
    raise SystemExit("selected DreamWaQ export has no backpack training receipt")
payload = json.loads(payload_path.read_text(encoding="utf-8"))
if payload.get("backpack_enabled") is not True:
    raise SystemExit("selected DreamWaQ export is not marked as backpack-trained")
if export.get("num_actions") != 12:
    raise SystemExit(f"expected the reviewed 12-action lower-body export, got {export.get('num_actions')!r}")
if export.get("action_contract_version") != "g1_lower_body_action_v3_canonical_waist_effort":
    raise SystemExit("selected DreamWaQ export does not match the reviewed waist-effort contract")
PY

# The old arm/state adapter publishes rt/arm_sdk. This script must be the only
# local body controller and deliberately does not start that adapter.
if command -v pgrep >/dev/null 2>&1 && pgrep -af '[r]un_dreamwaq\.py|[g]1_lowlevel_bridge|[r]un_homie\.py' >/dev/null; then
  echo "A G1 body controller or arm adapter is already running; stop it first." >&2
  exit 2
fi

network_interface="${G1_NETWORK_INTERFACE:-eth0}"
echo "Starting DreamWaQ on the G1 host as the sole rt/lowcmd writer."
echo "The remote Thor arm teleop must publish rt/arm_sdk; no PATH adapter is started here."
exec "$BOOST_DREAMWAQ_PYTHON" "$BOOST_DREAMWAQ_ROOT/deploy/run_dreamwaq.py" \
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
