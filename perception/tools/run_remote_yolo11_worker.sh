#!/usr/bin/env bash
set -euo pipefail

python -m remote_offload.remote_yolo11_worker "$@"
