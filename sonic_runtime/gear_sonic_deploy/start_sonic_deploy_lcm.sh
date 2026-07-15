#!/bin/bash
# start_sonic_deploy_lcm.sh
# 실행 위치: SONIC Docker container 안 (/workspace/gear_sonic_deploy)
#
# 원본 start_sonic_deploy.sh 대비 변경:
#   - deploy.sh → deploy_lcm.sh
#   - sim 모드 제거 (LCM은 eth1 유선 직결, loopback 불필요)
#   - --dummy-state 플래그 추가 가능 (bridge 없이 패킷 송신만 검증 시)
#
# 환경 설정 블록(1~3)은 container 충돌 방지를 위해 원본 그대로 유지.

# ============================================================================
# 1. 라이브러리 및 헤더 경로 설정
# ============================================================================
export LD_LIBRARY_PATH=/usr/lib:/host_nvidia_libs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/trt_clean/lib:$LIBRARY_PATH
export CPATH=/opt/trt_clean/include:$CPATH

# DDS 루프백 설정 제거 — LCM 버전에서 CycloneDDS 불필요
# export CYCLONEDDS_NETWORK_INTERFACE=lo  ← 원본에 있었으나 제거

# ============================================================================
# 2. TensorRT 10.13.3 (Blackwell용) 런타임 링크 및 갱신
# ============================================================================
ln -sf /opt/trt_clean/lib/libnvinfer.so.10.13.3      /usr/lib/libnvinfer.so.10
ln -sf /opt/trt_clean/lib/libnvonnxparser.so.10.13.3 /usr/lib/libnvonnxparser.so.10
ln -sf /opt/trt_clean/lib/libnvparsers.so.10.13.3    /usr/lib/libnvparsers.so.10
ldconfig

# ============================================================================
# 3. 빌드 경로를 위한 추가 링크 (링커 에러 방지)
# ============================================================================
mkdir -p /usr/lib/tensorrt/lib
ln -sf /opt/trt_clean/lib/libnvonnxparser.so.10.13.3              /usr/lib/tensorrt/lib/libnvonnxparser.so
ln -sf /opt/trt_clean/lib/libnvinfer.so.10.13.3                   /usr/lib/tensorrt/lib/libnvinfer.so
ln -sf /opt/trt_clean/lib/libnvparsers.so.10.13.3                 /usr/lib/tensorrt/lib/libnvparsers.so
ln -sf /opt/trt_clean/lib/libnvinfer_builder_resource.so.10.13.3  /usr/lib/tensorrt/lib/libnvinfer_builder_resource.so
ldconfig

# ============================================================================
# 4. 소스 설정 및 실행
# ============================================================================
source scripts/setup_env.sh

# --- 실제 bridge 연결 실행 ---
# bridge(g1_lowlevel_bridge)가 G1 내부에서 먼저 실행되고 있어야 한다.
bash deploy_lcm.sh \
  --input-type zmq_manager \
  --zmq-host 127.0.0.1 \
  --lcm-url "udpm://239.255.76.67:7667?ttl=1" \
  --lcm-iface eth0

# --- bridge 없이 LCM 패킷 송신만 검증할 때 (--dummy-state 추가) ---
# 이 모드에서는:
#   - INIT → WAIT_FOR_CONTROL → CONTROL 상태까지 진입
#   - PolicyActionPacket이 LCM으로 정상 송신되는지 확인 가능
#   - lcm-spy 또는 lcm-logger로 패킷 수신 확인 가능
#   - 실제 모터 제어 없음 (bridge 없음)
#   - obs가 0-filled이므로 policy 추론 결과는 의미 없음

# bash deploy_lcm.sh \
#   --input-type keyboard \
#   --lcm-url "udpm://239.255.76.67:7667?ttl=1" \
#   --lcm-iface enP2p1s0 \
#   --skip-net-setup 