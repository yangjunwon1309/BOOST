#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${SAM3_DOCKER_IMAGE:-path-g1-remote-sam3-trt:rtx4060}"
lcm_url="${WIFI_LCM_URL:-udpm://239.255.77.67:7667?ttl=1}"
web_host="${SAM3_WEB_HOST:-0.0.0.0}"
web_port="${SAM3_WEB_PORT:-8082}"
ipopt_visual_camera_z_offset_m="${SAM3_IPOPT_VISUAL_CAMERA_Z_OFFSET_M:-0.25}"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    command -v sudo >/dev/null 2>&1 || { echo "Docker access requires root, but sudo is unavailable." >&2; exit 2; }
    docker_cmd=(sudo docker)
fi
exec "${docker_cmd[@]}" run --rm --network host \
    -v "${repo_dir}:/workspace/perception:ro" -w /workspace/perception "${image}" \
    python -m remote_offload.remote_web_wrist_d405_v11 \
      --wifi-lcm-url "${lcm_url}" --host "${web_host}" --port "${web_port}" \
      --ipopt-visual-camera-z-offset-m "${ipopt_visual_camera_z_offset_m}"
