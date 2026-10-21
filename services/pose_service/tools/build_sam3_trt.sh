#!/usr/bin/env bash
set -euo pipefail

onnx_path="${1:-/models/sam3_trt/runtime4/sam3_4out.onnx}"
engine_path="${2:-/models/sam3_trt/sam3_fp16_4out.engine}"
mixed_path="${3:-${onnx_path%.onnx}_autocast.onnx}"
workspace_size="${SAM3_TRT_WORKSPACE:-4G}"

if [[ ! -f "$onnx_path" ]]; then
    echo "ONNX graph not found: $onnx_path" >&2
    exit 2
fi
if ! command -v trtexec >/dev/null 2>&1; then
    echo "trtexec is unavailable in this container" >&2
    exit 2
fi
mkdir -p "$(dirname "$engine_path")"

trt_major="$(python -c 'import tensorrt; print(tensorrt.__version__.split(".")[0])')"
if [[ -z "$trt_major" ]]; then
    echo "Could not determine the TensorRT major version" >&2
    exit 2
fi

build_onnx="$onnx_path"
precision_args=()
if (( trt_major >= 11 )); then
    echo "[SAM3-TRT] TensorRT 11 strongly typed mode: generating explicit FP16 graph"
    python tools/autocast_sam3_onnx.py \
        --input "$onnx_path" \
        --output "$mixed_path"
    build_onnx="$mixed_path"
else
    precision_args+=(--fp16)
fi

echo "[SAM3-TRT] TensorRT $(python -c 'import tensorrt; print(tensorrt.__version__)')"
echo "[SAM3-TRT] building $engine_path from $build_onnx"
echo "[SAM3-TRT] workspace pool limit: $workspace_size"
exec trtexec \
    --onnx="$build_onnx" \
    --saveEngine="$engine_path" \
    "${precision_args[@]}" \
    --builderOptimizationLevel=5 \
    --memPoolSize="workspace:${workspace_size}" \
    --profilingVerbosity=detailed \
    --skipInference
