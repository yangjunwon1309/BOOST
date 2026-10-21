#!/usr/bin/env bash
# Build the two Thor images used by wrist-D405 capture and local SAM3 inference.
#
# Default: build the GPU base image and its pyrealsense/D405 child image.
# --build-engine additionally exports SAM3 ONNX and builds a TensorRT plan on
# this Thor. TensorRT plans must never be copied from the RTX-4060 worker.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
perception_dir="$(cd -- "${script_dir}/.." && pwd)"

base_image="${G1_PERCEPTION_IMAGE:-g1-perception-sam3:thor-jp71}"
d405_image="${G1_PERCEPTION_D405_IMAGE:-g1-perception-sam3:d405i-thor-jp71}"
onnx_path="${SAM3_TRT_ONNX_PATH:-/models/sam3_trt/runtime4/sam3_4out.onnx}"
engine_path="${SAM3_TRT_ENGINE_PATH:-/models/sam3_trt/sam3_fp16_4out.engine}"
build_engine=0
overwrite_engine=0
no_cache=0

usage() {
    cat <<'EOF'
Usage: bash tools/build_thor_wrist_sam3_v11.sh [options]

Builds the Thor GPU base image and the D405/pyrealsense child image.

Options:
  --build-engine       Also export SAM3 ONNX and build the Thor TensorRT engine.
  --overwrite-engine   Allow replacement of an existing ONNX/engine artifact.
  --no-cache           Pass --no-cache to both Docker image builds.
  -h, --help           Show this help.

Environment overrides:
  G1_PERCEPTION_IMAGE          Base image tag (default g1-perception-sam3:thor-jp71)
  G1_PERCEPTION_D405_IMAGE     D405 image tag (default g1-perception-sam3:d405i-thor-jp71)
  SAM3_TRT_ONNX_PATH           Container-visible ONNX output path under /models
  SAM3_TRT_ENGINE_PATH         Container-visible TensorRT output path under /models

This script does not start, stop, rename, or remove any running container.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-engine) build_engine=1 ;;
        --overwrite-engine) overwrite_engine=1 ;;
        --no-cache) no_cache=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    command -v sudo >/dev/null 2>&1 || {
        echo "Docker access requires root or Docker-group membership." >&2
        exit 2
    }
    docker_cmd=(sudo docker)
fi

build_args=()
if (( no_cache )); then
    build_args+=(--no-cache)
fi

cd "${perception_dir}"
echo "[Thor SAM3] building base image: ${base_image}"
"${docker_cmd[@]}" build "${build_args[@]}" \
    -t "${base_image}" \
    -f docker/Dockerfile.thor .

echo "[Thor D405] building capture image: ${d405_image}"
"${docker_cmd[@]}" build "${build_args[@]}" \
    --build-arg "BASE_IMAGE=${base_image}" \
    -t "${d405_image}" \
    -f docker/Dockerfile.d435i .

if (( ! build_engine )); then
    cat <<EOF
[done] Images are ready.
  GPU SAM3 worker image : ${base_image}
  D405 gateway image    : ${d405_image}

Build the Thor-specific TensorRT plan when ready:
  bash tools/build_thor_wrist_sam3_v11.sh --build-engine
EOF
    exit 0
fi

# /models is bind-mounted to ${perception_dir}/models by run_thor.sh.
relative_onnx="${onnx_path#/models/}"
relative_engine="${engine_path#/models/}"
if [[ "${relative_onnx}" == "${onnx_path}" || "${relative_engine}" == "${engine_path}" ]]; then
    echo "SAM3_TRT_* paths must be under /models so run_thor.sh persists them." >&2
    exit 2
fi
host_onnx="${perception_dir}/models/${relative_onnx}"
host_engine="${perception_dir}/models/${relative_engine}"
if (( ! overwrite_engine )) && { [[ -e "${host_onnx}" ]] || [[ -e "${host_engine}" ]]; }; then
    echo "Refusing to overwrite existing SAM3 artifact. Re-run with --overwrite-engine:" >&2
    echo "  ${host_onnx}" >&2
    echo "  ${host_engine}" >&2
    exit 2
fi

engine_run_name="${G1_PERCEPTION_ENGINE_BUILD_CONTAINER_NAME:-g1-perception-sam3-engine-build}"
export G1_PERCEPTION_IMAGE="${base_image}"
export G1_PERCEPTION_CONTAINER_NAME="${engine_run_name}-export"
export G1_PERCEPTION_D405_IMAGE="${d405_image}"

echo "[Thor SAM3] exporting ONNX: ${onnx_path}"
./tools/run_thor.sh python tools/export_sam3_onnx.py \
    --output "${onnx_path}" --device cpu --overwrite

export G1_PERCEPTION_CONTAINER_NAME="${engine_run_name}-trt"
echo "[Thor SAM3] building TensorRT engine: ${engine_path}"
./tools/run_thor.sh bash tools/build_sam3_trt.sh \
    "${onnx_path}" "${engine_path}"

echo "[done] Thor TensorRT engine is ready: ${host_engine}"
