#!/usr/bin/env bash
set -euo pipefail

BOOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BOOST_ROOT"

dds_interface="${BRAINCO_DDS_INTERFACE:-}"
if [[ -z "$dds_interface" ]]; then
  echo "Set BRAINCO_DDS_INTERFACE to the G1/Thor DDS network interface (for example, eth0)." >&2
  exit 2
fi
if [[ "$dds_interface" == "lo" || "$dds_interface" == "lo0" ]]; then
  echo "The Thor live hand publisher refuses loopback; use the shared robot network interface." >&2
  exit 2
fi

exec python -u -m boost.control.hand_mppi.runner \
  --publish \
  --interface "$dds_interface" \
  --domain-id "${BRAINCO_DDS_DOMAIN_ID:-0}" \
  --input-endpoint "${HAND_MPPI_INPUT_ENDPOINT:-${PATH_XR_HAND_ENDPOINT:-tcp://127.0.0.1:5566}}"
