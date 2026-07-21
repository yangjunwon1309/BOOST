#!/bin/bash
set -e

# ============================================================================
# G1 Deploy LCM - Thor(외부 보드) 전용 배포 스크립트
# ============================================================================
# g1_deploy_onnx_ref_lcm 전용. DDS 없이 LCM UDP Multicast로 bridge와 통신한다.
#
# 원본 deploy.sh와의 차이:
#   - 네트워크 인터페이스 인자 불필요 (DDS 없음, 인자 자리만 유지)
#   - --disable-crc-check 제거 (LCM 버전에서 항상 true)
#   - sim/real 모드 대신 LCM URL로 통신 대상 지정
#   - 실행 바이너리: g1_deploy_onnx_ref_lcm
#   - 추가 인자: --lcm-url, --lcm-action-channel, --lcm-state-channel
#   - 네트워크 설정 (eth1 IP, 멀티캐스트 라우팅) 자동 적용
#   - input-type: keyboard / zmq / zmq_manager 만 지원 (gamepad 불가)
#
# Usage: ./deploy_lcm.sh [OPTIONS]
#
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Usage
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                  Show this help message"
    echo "  --cp, --checkpoint PATH     Checkpoint path (default: policy/release/model)"
    echo "  --obs-config PATH           Observation config YAML (default: policy/release/observation_config.yaml)"
    echo "  --planner PATH              Planner model path (default: planner/target_vel/V2/planner_sonic.onnx)"
    echo "  --motion-data PATH          Motion data directory (default: reference/example/)"
    echo "  --input-type TYPE           Input type: keyboard|zmq|zmq_manager (default: zmq_manager)"
    echo "                              ⚠️  gamepad 불가 — wireless_remote가 LCM 패킷에 없음"
    echo "  --output-type TYPE          Output type: zmq|all (default: all)"
    echo "  --zmq-host HOST             ZMQ input host (default: localhost)"
    echo "  --lcm-url URL               LCM URL (default: udpm://239.255.76.67:7667?ttl=1)"
    echo "  --lcm-iface IFACE           Thor의 LCM 전용 NIC (default: eth1)"
    echo "  --lcm-ip IP                 Thor의 LCM NIC에 할당할 IP (default: 10.0.0.1/24)"
    echo "  --lcm-action-channel NAME   Action 채널 (default: G1_POLICY_ACTION)"
    echo "  --lcm-state-channel NAME    State 채널 (default: G1_ROBOT_STATE)"
    echo "  --skip-net-setup            네트워크 설정 건너뜀 (이미 설정된 경우)"
    echo "  --dummy-state               bridge 없이 LCM 패킷 송신만 검증"
    echo "                              0-filled obs로 INIT→CONTROL 진입, lcm-spy로 확인 가능"
    echo "  --use-upper-ik              SONIC lower body + IK waist/arms teleop"
    echo "  --use-hand-ik               SONIC lower body + waist, IK arms/hands teleop"
    echo "  --use-hand-ik-origin        Legacy 50 Hz arms/hands IK override (no 500 Hz IK thread)"
    echo "  --use-ee-qp                 Enable PATH EE-QP correction"
    echo "  --ee-qp-config PATH         EE-QP config YAML (default: src/g1/ee_qp/config/ee_qp_real.yaml)"
    echo "  --path-use-dynamic-reference"
    echo "                               Read PATH state-machine QP references over ZMQ"
    echo "  --path-reference-host HOST  PATH QP reference host (default: 127.0.0.1)"
    echo "  --path-reference-port PORT  PATH QP reference port (default: 5560)"
    echo "  --path-reference-topic TOPIC"
    echo "                               PATH QP reference topic (default: path_qp_reference)"
    echo "  --path-reference-lookahead-ms MS"
    echo "                               PATH reference lookahead in ms (default: 10)"
    echo "  --path-reference-heartbeat-timeout-ms MS"
    echo "                               PATH reference heartbeat timeout in ms (default: 500)"
    echo "  --desk-pose                 INIT ramps to a hands-on-desk ready pose (arms bent) instead of default_angles"
    echo ""
    echo "Examples:"
    echo "  $0                                           # 기본값으로 실행"
    echo "  $0 --input-type keyboard                     # 키보드 입력"
    echo "  $0 --cp policy/release/model --obs-config policy/release/observation_config.yaml"
    echo "  $0 --lcm-iface enp3s0 --lcm-ip 10.0.0.1/24  # NIC 이름 직접 지정"
    echo "  $0 --skip-net-setup                          # 네트워크 이미 설정된 경우"
}

# ============================================================================
# Default Configuration
# ============================================================================

CHECKPOINT_DEFAULT="policy/release/model"
OBS_CONFIG_DEFAULT="policy/release/observation_config.yaml"
PLANNER_DEFAULT="planner/target_vel/V2/planner_sonic.onnx"
MOTION_DATA_DEFAULT="reference/kimodo/"
INPUT_TYPE_DEFAULT="zmq_manager"
OUTPUT_TYPE_DEFAULT="all"
ZMQ_HOST_DEFAULT="localhost"

# LCM 전용 기본값
LCM_URL_DEFAULT="udpm://239.255.76.67:7667?ttl=1"
LCM_IFACE_DEFAULT="eth1"
LCM_IP_DEFAULT="10.0.0.1/24"
LCM_MULTICAST_DEFAULT="239.255.76.67"
LCM_ACTION_CHANNEL_DEFAULT="G1_POLICY_ACTION"
LCM_STATE_CHANNEL_DEFAULT="G1_ROBOT_STATE"

CHECKPOINT="$CHECKPOINT_DEFAULT"
OBS_CONFIG="$OBS_CONFIG_DEFAULT"
PLANNER="$PLANNER_DEFAULT"
MOTION_DATA="$MOTION_DATA_DEFAULT"
INPUT_TYPE="$INPUT_TYPE_DEFAULT"
OUTPUT_TYPE="$OUTPUT_TYPE_DEFAULT"
ZMQ_HOST="$ZMQ_HOST_DEFAULT"
LCM_URL="$LCM_URL_DEFAULT"
LCM_IFACE="$LCM_IFACE_DEFAULT"
LCM_IP="$LCM_IP_DEFAULT"
LCM_MULTICAST="$LCM_MULTICAST_DEFAULT"
LCM_ACTION_CHANNEL="$LCM_ACTION_CHANNEL_DEFAULT"
LCM_STATE_CHANNEL="$LCM_STATE_CHANNEL_DEFAULT"
SKIP_NET_SETUP=false
DUMMY_STATE=false
USE_UPPER_IK=false
USE_HAND_IK=false
USE_HAND_IK_ORIGIN=false
USE_EE_QP=false
EE_QP_CONFIG="src/g1/ee_qp/config/ee_qp_real.yaml"
PATH_USE_DYNAMIC_REFERENCE=false
PATH_REFERENCE_HOST="127.0.0.1"
PATH_REFERENCE_PORT=5560
PATH_REFERENCE_TOPIC="path_qp_reference"
PATH_REFERENCE_LOOKAHEAD_MS=10
PATH_REFERENCE_HEARTBEAT_TIMEOUT_MS=500
DESK_POSE=false

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --cp|--checkpoint)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            CHECKPOINT="$2"; shift 2 ;;
        --obs-config)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            OBS_CONFIG="$2"; shift 2 ;;
        --planner)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            PLANNER="$2"; shift 2 ;;
        --motion-data)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            MOTION_DATA="$2"; shift 2 ;;
        --input-type)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            INPUT_TYPE="$2"; shift 2 ;;
        --output-type)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            OUTPUT_TYPE="$2"; shift 2 ;;
        --zmq-host)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            ZMQ_HOST="$2"; shift 2 ;;
        --lcm-url)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            LCM_URL="$2"; shift 2 ;;
        --lcm-iface)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            LCM_IFACE="$2"; shift 2 ;;
        --lcm-ip)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            LCM_IP="$2"; shift 2 ;;
        --lcm-action-channel)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            LCM_ACTION_CHANNEL="$2"; shift 2 ;;
        --lcm-state-channel)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            LCM_STATE_CHANNEL="$2"; shift 2 ;;
        --skip-net-setup)
            SKIP_NET_SETUP=true; shift ;;
        --dummy-state)
            DUMMY_STATE=true; shift ;;
        --use-upper-ik)
            USE_UPPER_IK=true; shift ;;
        --use-hand-ik)
            USE_HAND_IK=true; shift ;;
        --use-hand-ik-origin)
            USE_HAND_IK_ORIGIN=true; shift ;;
        --use-ee-qp)
            USE_EE_QP=true; shift ;;
        --ee-qp-config)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            EE_QP_CONFIG="$2"; shift 2 ;;
        --path-use-dynamic-reference)
            PATH_USE_DYNAMIC_REFERENCE=true; shift ;;
        --path-reference-host)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            PATH_REFERENCE_HOST="$2"; shift 2 ;;
        --path-reference-port)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            PATH_REFERENCE_PORT="$2"; shift 2 ;;
        --path-reference-topic)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            PATH_REFERENCE_TOPIC="$2"; shift 2 ;;
        --path-reference-lookahead-ms)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            PATH_REFERENCE_LOOKAHEAD_MS="$2"; shift 2 ;;
        --path-reference-heartbeat-timeout-ms)
            [[ -z "$2" ]] && { echo -e "${RED}Error: $1 requires an argument${NC}" >&2; exit 1; }
            PATH_REFERENCE_HEARTBEAT_TIMEOUT_MS="$2"; shift 2 ;;
        --desk-pose)
            DESK_POSE=true; shift ;;
        sim|real)
            echo -e "${YELLOW}⚠️  '$1' 모드는 LCM 버전에서 사용하지 않습니다. --lcm-url로 통신 대상을 지정하세요.${NC}"
            shift ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            show_usage
            exit 1 ;;
    esac
done

IK_MODE_COUNT=0
[[ "$USE_UPPER_IK" == true ]] && IK_MODE_COUNT=$((IK_MODE_COUNT + 1))
[[ "$USE_HAND_IK" == true ]] && IK_MODE_COUNT=$((IK_MODE_COUNT + 1))
[[ "$USE_HAND_IK_ORIGIN" == true ]] && IK_MODE_COUNT=$((IK_MODE_COUNT + 1))
[[ "$USE_EE_QP" == true ]] && IK_MODE_COUNT=$((IK_MODE_COUNT + 1))
if [[ "$IK_MODE_COUNT" -gt 1 ]]; then
    echo -e "${RED}Error: --use-upper-ik, --use-hand-ik, --use-hand-ik-origin, and --use-ee-qp cannot be used together${NC}" >&2
    exit 1
fi

# ============================================================================
# Input type validation — gamepad 차단
# ============================================================================

if [[ "$INPUT_TYPE" == "gamepad" || "$INPUT_TYPE" == "gamepad_manager" || "$INPUT_TYPE" == "interface_manager" ]]; then
    echo -e "${RED}❌ --input-type '$INPUT_TYPE' 는 LCM 버전에서 지원하지 않습니다.${NC}"
    echo "   이유: wireless_remote 필드가 RobotStatePacket에 없어 gamepad 입력 불가."
    echo "   지원 타입: keyboard, zmq, zmq_manager"
    exit 1
fi

# ============================================================================
# Derived Paths
# ============================================================================

CHECKPOINT_DECODER="${CHECKPOINT}_decoder.onnx"
CHECKPOINT_ENCODER="${CHECKPOINT}_encoder.onnx"

# ============================================================================
# Header
# ============================================================================

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    G1 DEPLOY LCM LAUNCHER (Thor)                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# Step 1: Prerequisites
# ============================================================================

echo -e "${BLUE}[Step 1/4]${NC} Checking prerequisites..."
echo ""

# TensorRT
if [ -z "$TensorRT_ROOT" ]; then
    echo -e "${YELLOW}⚠️  TensorRT_ROOT is not set.${NC}"
    for trt_path in /opt/trt_clean /opt/TensorRT "$HOME/trt_clean" "$HOME/TensorRT"; do
        if [ -f "$trt_path/include/NvInfer.h" ] && [ -d "$trt_path/lib" ]; then
            echo -e "${GREEN}   Found TensorRT at $trt_path - setting temporarily${NC}"
            export TensorRT_ROOT="$trt_path"
            export TENSORRT_ROOT="$TensorRT_ROOT"
            export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:${LD_LIBRARY_PATH:-}"
            export LIBRARY_PATH="$TensorRT_ROOT/lib:${LIBRARY_PATH:-}"
            export CPATH="$TensorRT_ROOT/include:${CPATH:-}"
            break
        fi
    done
fi

# LCM 설치 확인
if ! ldconfig -p 2>/dev/null | grep -q "liblcm" && ! pkg-config --exists lcm 2>/dev/null; then
    echo -e "${YELLOW}⚠️  LCM library not found. Checking common paths...${NC}"
    LCM_FOUND=false
    for path in /usr/lib /usr/local/lib /opt/lcm/lib; do
        if ls "$path"/liblcm* 2>/dev/null | grep -q .; then
            echo -e "${GREEN}   Found LCM at $path${NC}"
            LCM_FOUND=true
            break
        fi
    done
    if [[ "$LCM_FOUND" == false ]]; then
        echo -e "${RED}❌ LCM not found. Install with:${NC}"
        echo "   sudo apt-get install liblcm-dev"
        echo "   Or build from: https://github.com/lcm-proj/lcm"
        exit 1
    fi
else
    echo -e "${GREEN}✅ LCM library found${NC}"
fi

# 필수 파일 확인
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}❌ Missing: $1${NC}"; return 1
    else
        echo -e "${GREEN}✅ Found:   $1${NC}"; return 0
    fi
}

echo ""
echo "Checking required model files..."
MISSING_FILES=0
check_file "$CHECKPOINT_DECODER" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$CHECKPOINT_ENCODER" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$OBS_CONFIG"         || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$PLANNER"            || MISSING_FILES=$((MISSING_FILES + 1))

if [ -d "$MOTION_DATA" ]; then
    echo -e "${GREEN}✅ Found:   $MOTION_DATA${NC}"
else
    echo -e "${RED}❌ Missing directory: $MOTION_DATA${NC}"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some files are missing. You may need to run: git lfs pull${NC}"
fi

echo ""

# ============================================================================
# Step 2: Dependencies
# ============================================================================

echo -e "${BLUE}[Step 2/4]${NC} Checking/Installing dependencies..."

if ! command -v just &> /dev/null; then
    echo "Installing dependencies (just not found)..."
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}✅ just is already installed${NC}"
fi

DEPS_OK=true
for cmd in cmake clang git make; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${YELLOW}⚠️  $cmd not found${NC}"
        DEPS_OK=false; break
    fi
done

if [ "$DEPS_OK" = false ]; then
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}✅ All essential tools installed${NC}"
fi

echo ""

# ============================================================================
# Step 3: Environment Setup & Build
# ============================================================================

echo -e "${BLUE}[Step 3/4]${NC} Setting up environment and building..."

echo "Sourcing environment setup..."
set +e
source scripts/setup_env.sh
set -e

echo "Building the project..."
just build

echo ""

# ============================================================================
# Step 4: Network Setup + Deploy
# ============================================================================

echo -e "${BLUE}[Step 4/4]${NC} Ready to deploy!"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    DEPLOYMENT CONFIGURATION (LCM)                     ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Binary:              ${GREEN}g1_deploy_onnx_ref_lcm${NC}"
echo -e "  Decoder Model:       ${GREEN}$CHECKPOINT_DECODER${NC}"
echo -e "  Encoder Model:       ${GREEN}$CHECKPOINT_ENCODER${NC}"
echo -e "  Motion Data:         ${GREEN}$MOTION_DATA${NC}"
echo -e "  Obs Config:          ${GREEN}$OBS_CONFIG${NC}"
echo -e "  Planner:             ${GREEN}$PLANNER${NC}"
echo -e "  Input Type:          ${GREEN}$INPUT_TYPE${NC}"
echo -e "  Output Type:         ${GREEN}$OUTPUT_TYPE${NC}"
echo -e "  ZMQ Host:            ${GREEN}$ZMQ_HOST${NC}"
echo ""
echo -e "  LCM URL:             ${GREEN}$LCM_URL${NC}"
echo -e "  LCM NIC:             ${GREEN}$LCM_IFACE${NC}"
echo -e "  LCM IP:              ${GREEN}$LCM_IP${NC}"
echo -e "  Action Channel:      ${GREEN}$LCM_ACTION_CHANNEL${NC}"
echo -e "  State Channel:       ${GREEN}$LCM_STATE_CHANNEL${NC}"
echo -e "  Network Setup:       ${GREEN}$([ "$SKIP_NET_SETUP" = true ] && echo "SKIP" || echo "AUTO")${NC}"
echo -e "  Dummy State Mode:    ${GREEN}$([ "$DUMMY_STATE" = true ] && echo "ON (bridge 없이 패킷 송신 검증)" || echo "OFF (정상 모드)")${NC}"
if [[ "$USE_UPPER_IK" == true ]]; then
echo -e "  IK Mode:             ${GREEN}waist + arms/hands (lower body from SONIC)${NC}"
elif [[ "$USE_HAND_IK" == true ]]; then
echo -e "  IK Mode:             ${GREEN}arms/hands only (lower body + waist from SONIC)${NC}"
elif [[ "$USE_HAND_IK_ORIGIN" == true ]]; then
echo -e "  IK Mode:             ${GREEN}legacy 50 Hz arms/hands only (lower body + waist from SONIC)${NC}"
elif [[ "$USE_EE_QP" == true ]]; then
echo -e "  IK/QP Mode:          ${GREEN}PATH EE-QP correction${NC}"
echo -e "  EE-QP Config:        ${GREEN}$EE_QP_CONFIG${NC}"
if [[ "$PATH_USE_DYNAMIC_REFERENCE" == true ]]; then
echo -e "  PATH QP Reference:   ${GREEN}tcp://$PATH_REFERENCE_HOST:$PATH_REFERENCE_PORT/$PATH_REFERENCE_TOPIC${NC}"
echo -e "  PATH Lookahead:      ${GREEN}${PATH_REFERENCE_LOOKAHEAD_MS} ms${NC}"
echo -e "  PATH Heartbeat:      ${GREEN}${PATH_REFERENCE_HEARTBEAT_TIMEOUT_MS} ms${NC}"
else
echo -e "  PATH QP Reference:   ${YELLOW}OFF${NC}"
fi
else
echo -e "  IK Mode:             ${GREEN}OFF${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# 실행될 명령 미리보기
echo -e "${YELLOW}The following command will be executed:${NC}"
echo ""
echo -e "${BLUE}just run g1_deploy_onnx_ref_lcm dummy_iface $CHECKPOINT_DECODER $MOTION_DATA \\${NC}"
echo -e "${BLUE}    --obs-config $OBS_CONFIG \\${NC}"
echo -e "${BLUE}    --encoder-file $CHECKPOINT_ENCODER \\${NC}"
echo -e "${BLUE}    --planner-file $PLANNER \\${NC}"
echo -e "${BLUE}    --input-type $INPUT_TYPE \\${NC}"
echo -e "${BLUE}    --output-type $OUTPUT_TYPE \\${NC}"
echo -e "${BLUE}    --zmq-host $ZMQ_HOST --zmq-port 5556 \\${NC}"
echo -e "${BLUE}    --lcm-url $LCM_URL \\${NC}"
echo -e "${BLUE}    --lcm-action-channel $LCM_ACTION_CHANNEL \\${NC}"
echo -e "${BLUE}    --lcm-state-channel $LCM_STATE_CHANNEL${NC}"
if [[ "$DUMMY_STATE" == true ]]; then
echo -e "${BLUE}    --dummy-state${NC}"
fi
if [[ "$USE_UPPER_IK" == true ]]; then
echo -e "${BLUE}    --use-upper-ik${NC}"
fi
if [[ "$USE_HAND_IK" == true ]]; then
echo -e "${BLUE}    --use-hand-ik${NC}"
fi
if [[ "$USE_HAND_IK_ORIGIN" == true ]]; then
echo -e "${BLUE}    --use-hand-ik-origin${NC}"
fi
if [[ "$USE_EE_QP" == true ]]; then
echo -e "${BLUE}    --use-ee-qp --ee-qp-config $EE_QP_CONFIG \\${NC}"
if [[ "$PATH_USE_DYNAMIC_REFERENCE" == true ]]; then
echo -e "${BLUE}    --path-use-dynamic-reference \\${NC}"
echo -e "${BLUE}    --path-reference-host $PATH_REFERENCE_HOST --path-reference-port $PATH_REFERENCE_PORT \\${NC}"
echo -e "${BLUE}    --path-reference-topic $PATH_REFERENCE_TOPIC \\${NC}"
echo -e "${BLUE}    --path-reference-lookahead-ms $PATH_REFERENCE_LOOKAHEAD_MS \\${NC}"
echo -e "${BLUE}    --path-reference-heartbeat-timeout-ms $PATH_REFERENCE_HEARTBEAT_TIMEOUT_MS${NC}"
fi
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# 확인 프롬프트
if [[ "$DUMMY_STATE" == true ]]; then
    echo -e "${YELLOW}📋 DUMMY STATE MODE: bridge 없이 LCM 패킷 송신만 검증합니다.${NC}"
    echo -e "${YELLOW}   obs는 0-filled이므로 policy 추론 결과는 의미 없습니다.${NC}"
    echo -e "${YELLOW}   lcm-spy 또는 lcm-logger로 G1_POLICY_ACTION 채널을 모니터링하세요.${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING: This will start REAL robot control via LCM bridge!${NC}"
    echo -e "${YELLOW}   bridge(g1_lowlevel_bridge)가 G1 내부에서 먼저 실행 중이어야 합니다.${NC}"
fi
echo ""
read -p "$(echo -e ${GREEN}Proceed with deployment? [Y/n]: ${NC})" confirm

if ! ([[ "$confirm" =~ ^[Yy]$ ]] || [[ -z "$confirm" ]]); then
    echo ""
    echo -e "${YELLOW}Deployment cancelled.${NC}"
    exit 0
fi

echo ""

# ============================================================================
# Network Setup — eth1 IP 할당 및 멀티캐스트 라우팅
# ============================================================================

if [[ "$DUMMY_STATE" == true ]]; then
    echo -e "${YELLOW}[Net Setup]${NC} Skipped (--dummy-state mode, no bridge needed)"
elif [[ "$SKIP_NET_SETUP" == true ]]; then
    echo -e "${YELLOW}[Net Setup]${NC} Skipped (--skip-net-setup)"
else
    echo -e "${BLUE}[Net Setup]${NC} Configuring LCM network interface..."

    # NIC 존재 확인
    if ! ip link show "$LCM_IFACE" &>/dev/null; then
        echo -e "${RED}❌ Network interface '$LCM_IFACE' not found.${NC}"
        echo "   Available interfaces:"
        ip link show | awk '/^[0-9]+:/ {gsub(/:$/, "", $2); print "   " $2}'
        echo ""
        echo "   Use --lcm-iface <name> to specify the correct interface."
        exit 1
    fi

    echo "  Flushing existing IP on $LCM_IFACE..."
    sudo ip addr flush dev "$LCM_IFACE"

    echo "  Assigning $LCM_IP to $LCM_IFACE..."
    sudo ip addr add "$LCM_IP" dev "$LCM_IFACE"

    echo "  Bringing up $LCM_IFACE..."
    sudo ip link set "$LCM_IFACE" up

    echo "  Setting multicast route $LCM_MULTICAST → $LCM_IFACE..."
    sudo ip route replace "${LCM_MULTICAST}/32" dev "$LCM_IFACE"

    # 라우팅 확인
    echo "  Verifying multicast route:"
    ip route get "$LCM_MULTICAST" 2>/dev/null \
        && echo -e "  ${GREEN}✅ Multicast route OK${NC}" \
        || echo -e "  ${YELLOW}⚠️  Could not verify route (may still work)${NC}"

    echo ""
fi

# ============================================================================
# Launch
# ============================================================================

echo -e "${GREEN}🚀 Starting g1_deploy_onnx_ref_lcm...${NC}"
echo ""

DUMMY_ARG=""
if [[ "$DUMMY_STATE" == true ]]; then
    DUMMY_ARG="--dummy-state"
fi

IK_ARG=""
if [[ "$USE_UPPER_IK" == true ]]; then
    IK_ARG="--use-upper-ik"
elif [[ "$USE_HAND_IK" == true ]]; then
    IK_ARG="--use-hand-ik"
elif [[ "$USE_HAND_IK_ORIGIN" == true ]]; then
    IK_ARG="--use-hand-ik-origin"
fi

EE_QP_ARGS=()
if [[ "$USE_EE_QP" == true ]]; then
    EE_QP_ARGS+=(--use-ee-qp --ee-qp-config "$EE_QP_CONFIG")
    if [[ "$PATH_USE_DYNAMIC_REFERENCE" == true ]]; then
        EE_QP_ARGS+=(
            --path-use-dynamic-reference
            --path-reference-host "$PATH_REFERENCE_HOST"
            --path-reference-port "$PATH_REFERENCE_PORT"
            --path-reference-topic "$PATH_REFERENCE_TOPIC"
            --path-reference-lookahead-ms "$PATH_REFERENCE_LOOKAHEAD_MS"
            --path-reference-heartbeat-timeout-ms "$PATH_REFERENCE_HEARTBEAT_TIMEOUT_MS"
        )
    fi
fi

DESK_POSE_ARG=""
if [[ "$DESK_POSE" == true ]]; then
    DESK_POSE_ARG="--desk-pose"
fi

just run g1_deploy_onnx_ref_lcm "dummy_iface" "$CHECKPOINT_DECODER" "$MOTION_DATA" \
    --obs-config "$OBS_CONFIG" \
    --encoder-file "$CHECKPOINT_ENCODER" \
    --planner-file "$PLANNER" \
    --input-type "$INPUT_TYPE" \
    --output-type "$OUTPUT_TYPE" \
    --zmq-host "$ZMQ_HOST" --zmq-port 5556 \
    --lcm-url "$LCM_URL" \
    --lcm-action-channel "$LCM_ACTION_CHANNEL" \
    --lcm-state-channel "$LCM_STATE_CHANNEL" \
    --planner-fp16 \
    "${EE_QP_ARGS[@]}" \
    $DUMMY_ARG \
    $IK_ARG \
    $DESK_POSE_ARG
