#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${SAM3_DOCKER_IMAGE:-path-g1-remote-sam3-trt:rtx4060}"
lcm_url="${WIFI_LCM_URL:-udpm://239.255.77.67:7667?ttl=1}"
concepts="${SAM3_CONCEPTS:-object}"
inference_hz="${SAM3_INFERENCE_HZ:-8}"
engine_host="${SAM3_TRT_ENGINE_HOST:-${repo_dir}/models/sam3_trt/rtx4060/sam3_fp16_4out.engine}"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Docker access requires root, but sudo is unavailable." >&2
        exit 2
    fi
    docker_cmd=(sudo docker)
    if [[ -n "${HF_TOKEN:-}" ]]; then
        docker_cmd=(sudo --preserve-env=HF_TOKEN docker)
    fi
fi

if [[ ! -f "${engine_host}" ]]; then
    echo "TensorRT engine not found: ${engine_host}" >&2
    exit 2
fi

exec "${docker_cmd[@]}" run --rm --gpus all --network host \
    -e HF_TOKEN \
    -v "${repo_dir}:/workspace/perception" \
    -v "${engine_host}:/models/sam3_trt/remote.engine:ro" \
    -v "${HOME}/.cache/huggingface:/cache/huggingface" \
    -v "${HOME}/.cache/torch:/cache/torch" \
    "${image}" \
    python -m remote_offload.remote_sam3_belief_input_worker \
        --wifi-lcm-url "${lcm_url}" \
        --config config/pipeline.yaml \
        --backend tensorrt \
        --trt-engine /models/sam3_trt/remote.engine \
        --concepts "${concepts}" \
        --inference-hz "${inference_hz}" \
        --min-depth-points 80 \
        --sync-tolerance-ms 20
