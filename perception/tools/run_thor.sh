#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
perception_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${G1_PERCEPTION_IMAGE:-g1-perception-sam3:thor-jp71}"
runtime_user="$(id -un)"

mkdir -p \
    "${perception_dir}/.cache/huggingface" \
    "${perception_dir}/.cache/torch/home" \
    "${perception_dir}/.cache/torch/xdg" \
    "${perception_dir}/.cache/torch/inductor" \
    "${perception_dir}/.cache/torch/triton" \
    "${perception_dir}/.cache/torch/cuda" \
    "${perception_dir}/models" \
    "${perception_dir}/registry_data"

if [ "$#" -eq 0 ]; then
    set -- python tools/verify_environment.py
fi

tty_args=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    tty_args=(-it)
fi

exec docker run --rm "${tty_args[@]}" \
    --name g1-perception-sam3 \
    --gpus all \
    --network host \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -e HOME=/cache/torch/home \
    -e USER="${runtime_user}" \
    -e LOGNAME="${runtime_user}" \
    -e XDG_CACHE_HOME=/cache/torch/xdg \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/torch/inductor \
    -e TRITON_CACHE_DIR=/cache/torch/triton \
    -e CUDA_CACHE_PATH=/cache/torch/cuda \
    -e HF_HOME=/cache/huggingface \
    -e TORCH_HOME=/cache/torch \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HF_TOKEN \
    -v "${perception_dir}:/workspace/perception" \
    -v "${perception_dir}/.cache/huggingface:/cache/huggingface" \
    -v "${perception_dir}/.cache/torch:/cache/torch" \
    -v "${perception_dir}/models:/models" \
    -v "${perception_dir}/registry_data:/workspace/perception/registry_data" \
    -w /workspace/perception \
    "${image}" "$@"
