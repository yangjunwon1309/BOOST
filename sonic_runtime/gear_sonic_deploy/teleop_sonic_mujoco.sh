#!/bin/bash
set -euo pipefail

# Works in the Docker path used for sim2sim.  On host, change /opt/trt_clean to
# $HOME/trt_clean if needed.
_requested_trt_root=${TensorRT_ROOT:-${TENSORRT_ROOT:-}}
_trt_candidates=(
  "${TensorRT_ROOT:-}"
  "${TENSORRT_ROOT:-}"
  "/opt/trt_clean"
  "/opt/TensorRT"
  "${HOME:-/root}/trt_clean"
  "${HOME:-/root}/TensorRT"
)

TensorRT_ROOT=""
for _trt_root in "${_trt_candidates[@]}"; do
  if [ -n "$_trt_root" ] \
    && [ -f "$_trt_root/include/NvInfer.h" ] \
    && compgen -G "$_trt_root/lib/libnvinfer.so*" >/dev/null; then
    TensorRT_ROOT="$_trt_root"
    break
  fi
done

if [ -n "$_requested_trt_root" ] && [ "$_requested_trt_root" != "$TensorRT_ROOT" ]; then
  echo "Ignoring invalid TensorRT_ROOT=$_requested_trt_root; using ${TensorRT_ROOT:-<none found>}." >&2
fi

if [ -z "$TensorRT_ROOT" ]; then
  echo "TensorRT header/library not found in any known location." >&2
  echo "Check that host ~/trt_clean is mounted to /opt/trt_clean in Docker." >&2
  exit 1
fi

export TensorRT_ROOT
export TENSORRT_ROOT=$TensorRT_ROOT
export LD_LIBRARY_PATH=$TensorRT_ROOT/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$TensorRT_ROOT/lib:${LIBRARY_PATH:-}
export CPATH=$TensorRT_ROOT/include:${CPATH:-}
export CMAKE_PREFIX_PATH=$TensorRT_ROOT:${CMAKE_PREFIX_PATH:-}
export CMAKE_INCLUDE_PATH=$TensorRT_ROOT/include:${CMAKE_INCLUDE_PATH:-}
export CMAKE_LIBRARY_PATH=$TensorRT_ROOT/lib:${CMAKE_LIBRARY_PATH:-}
export CYCLONEDDS_NETWORK_INTERFACE=lo

if [ ! -f "$TensorRT_ROOT/include/NvInfer.h" ]; then
  echo "TensorRT header not found: ${TensorRT_ROOT:-<none>}/include/NvInfer.h" >&2
  echo "Check that host ~/trt_clean is mounted to /opt/trt_clean in Docker." >&2
  exit 1
fi

if ! compgen -G "$TensorRT_ROOT/lib/libnvinfer.so*" >/dev/null; then
  echo "TensorRT library not found: $TensorRT_ROOT/lib/libnvinfer.so*" >&2
  echo "Check that host ~/trt_clean/lib is mounted to /opt/trt_clean/lib in Docker." >&2
  exit 1
fi

bash deploy.sh \
  --input-type zmq_manager \
  --zmq-host 127.0.0.1 \
  --zmq-topic pose \
  "$@" \
  sim
