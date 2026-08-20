#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${SAM3_DOCKER_IMAGE:-path-g1-remote-sam3-trt:rtx4060}"
exec docker run --rm --gpus all --network host \
    -v "${repo_dir}:/workspace/perception" \
    -w /workspace/perception "${image}" \
    python -m remote_offload.remote_sam3_wrist_d405_worker_v11 "$@"
