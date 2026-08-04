#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
perception_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${G1_PERCEPTION_IMAGE:-g1-perception-sam3:d435i-thor-jp71}"
container_name="${G1_PERCEPTION_CONTAINER_NAME:-g1-d435i-lcm-$$}"

exec docker run --rm -it \
    --name "${container_name}" \
    --gpus all \
    --privileged \
    --network host \
    --ipc host \
    -e HOME=/cache/torch/home \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v "${perception_dir}:/workspace/perception" \
    -v "${perception_dir}/.cache/huggingface:/cache/huggingface" \
    -v "${perception_dir}/.cache/torch:/cache/torch" \
    -v "${perception_dir}/models:/models" \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -w /workspace/perception \
    "${image}" python tools/realsense_lcm_bridge.py "$@"
