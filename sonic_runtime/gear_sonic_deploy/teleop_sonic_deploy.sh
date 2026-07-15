#!/bin/bash
# teleop_sonic_deploy.sh
# 실행 위치: SONIC Docker container 안 (/workspace/gear_sonic_deploy)

# 1. 라이브러리 및 헤더 경로 설정
export LD_LIBRARY_PATH=/usr/lib:/host_nvidia_libs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/trt_clean/lib:$LIBRARY_PATH
export CPATH=/opt/trt_clean/include:$CPATH
# real LCM teleop에서는 CycloneDDS/MuJoCo loopback interface를 쓰지 않는다.
# g1_deploy_onnx_ref_lcm 내부에서 Unitree ChannelFactory는 thread utility용으로만 초기화된다.

IK_ARGS=("$@")

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

# 4. 소스 설정 및 실행
source scripts/setup_env.sh
bash deploy_lcm.sh --input-type zmq_manager \
  --zmq-host 127.0.0.1 \
  --lcm-url "udpm://239.255.76.67:7667?ttl=1" \
  --lcm-iface enP2p1s0 \
  --skip-net-setup \
  "${IK_ARGS[@]}"
