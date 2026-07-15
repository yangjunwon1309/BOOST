#!/bin/bash
# start_sonic_deploy.sh
# 실행 위치: SONIC Docker container 안 (/workspace/gear_sonic_deploy)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EE_QP_CONFIG="$SCRIPT_DIR/src/g1/ee_qp/config/ee_qp.yaml"

# 1. 라이브러리 및 헤더 경로 설정
export LD_LIBRARY_PATH=/usr/lib:/host_nvidia_libs:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/opt/trt_clean/lib:${LIBRARY_PATH:-}
export CPATH=/opt/trt_clean/include:${CPATH:-}
export CYCLONEDDS_NETWORK_INTERFACE=lo

# 2. TensorRT 10.13.3 (Blackwell용) 런타임 링크 및 갱신
ln -sf /opt/trt_clean/lib/libnvinfer.so.10.13.3      /usr/lib/libnvinfer.so.10
ln -sf /opt/trt_clean/lib/libnvonnxparser.so.10.13.3 /usr/lib/libnvonnxparser.so.10
ln -sf /opt/trt_clean/lib/libnvparsers.so.10.13.3    /usr/lib/libnvparsers.so.10
ldconfig

# 3. 빌드 경로를 위한 추가 링크 (링커 에러 방지)
mkdir -p /usr/lib/tensorrt/lib
ln -sf /opt/trt_clean/lib/libnvonnxparser.so.10.13.3              /usr/lib/tensorrt/lib/libnvonnxparser.so
ln -sf /opt/trt_clean/lib/libnvinfer.so.10.13.3                   /usr/lib/tensorrt/lib/libnvinfer.so
ln -sf /opt/trt_clean/lib/libnvparsers.so.10.13.3                 /usr/lib/tensorrt/lib/libnvparsers.so
ln -sf /opt/trt_clean/lib/libnvinfer_builder_resource.so.10.13.3  /usr/lib/tensorrt/lib/libnvinfer_builder_resource.so
ldconfig

if [[ ! -f "$EE_QP_CONFIG" ]]; then
  echo "EE QP config not found: $EE_QP_CONFIG" >&2
  exit 1
fi


# 4. 소스 설정 및 실행
# source scripts/setup_env.sh
# bash deploy.sh --input-type keyboard \
#    --zmq-host 127.0.0.1 \
#    --zmq-port 5556 \
#    --zmq-topic pose \
#    --motion-data reference/kimodo \
#   sim

bash "$SCRIPT_DIR/deploy.sh" \
  --input-type keyboard \
  --zmq-host 127.0.0.1 \
  --zmq-port 5556 \
  --zmq-topic pose \
  --use-ee-qp \
  --motion-data reference/kimodo \
  --ee-qp-config "$EE_QP_CONFIG" \
  "$@" \
  sim


#bash ./deploy.sh --input-type zmq_manager sim
