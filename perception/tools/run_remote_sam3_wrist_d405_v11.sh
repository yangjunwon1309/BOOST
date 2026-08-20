#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${SAM3_DOCKER_IMAGE:-path-g1-remote-sam3-trt:rtx4060}"
lcm_url="${WIFI_LCM_URL:-udpm://239.255.77.67:7667?ttl=1}"
concepts="${SAM3_CONCEPTS:-object}"
engine_host="${SAM3_TRT_ENGINE_HOST:-${repo_dir}/models/sam3_trt/rtx4060/sam3_fp16_4out.engine}"
[[ -f "${engine_host}" ]] || { echo "TensorRT engine not found: ${engine_host}" >&2; exit 2; }
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    command -v sudo >/dev/null 2>&1 || { echo "Docker access requires root, but sudo is unavailable." >&2; exit 2; }
    docker_cmd=(sudo docker)
    [[ -n "${HF_TOKEN:-}" ]] && docker_cmd=(sudo --preserve-env=HF_TOKEN docker)
fi
exec "${docker_cmd[@]}" run --rm --gpus all --network host --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_TOKEN \
    -v "${repo_dir}:/workspace/perception" \
    -v "${engine_host}:/models/sam3_trt/remote.engine:ro" \
    -v "${HOME}/.cache/huggingface:/cache/huggingface" \
    -v "${HOME}/.cache/torch:/cache/torch" \
    -w /workspace/perception "${image}" \
    python -m remote_offload.remote_sam3_wrist_d405_worker_v11 \
      --wifi-lcm-url "${lcm_url}" --backend tensorrt \
      --trt-engine /models/sam3_trt/remote.engine --concepts "${concepts}" "$@"
