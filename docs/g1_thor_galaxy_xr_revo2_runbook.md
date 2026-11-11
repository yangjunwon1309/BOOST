# Thor + G1 Galaxy XR / Revo2 실기 구동 가이드

Thor와 G1이 켜져 있고 BOOST 및 BrainCo hand service가 이미 설치된 상태에서 시작한다. 이 문서는 route 설정, 기존 serial 프로세스 정리, DreamWaQ/PATH/XR/손 구동 순서만 다룬다.

## 호스트와 연결값

| 장치 | 인터페이스 | 주소 | 용도 |
| --- | --- | --- | --- |
| Thor | `eth0` | `192.168.123.2` | G1 유선 DDS와 PATH LCM |
| G1 | `eth0` | `192.168.123.164` | Thor 유선 DDS와 PATH LCM |
| Thor Wi-Fi | Wi-Fi 인터페이스 | `192.168.0.16` | Galaxy XR HTTPS |

Galaxy XR은 `https://192.168.0.16:8012/`로 접속한다. Thor Wi-Fi 주소가 달라졌으면 Thor에서 `ip -brief -4 addr`로 현재 주소를 확인한다.

## 1. Thor와 G1의 PATH multicast 경로 설정

각 호스트에서 실행한다. `ip route replace` 설정은 재부팅 후 유지되지 않는다.

Thor:

```bash
sudo ip route replace 239.255.76.67/32 dev eth0 src 192.168.123.2
sudo ip route replace 239.255.77.67/32 dev wlP1p1s0 src 192.168.0.16
ip route get 239.255.76.67
ip route get 239.255.77.67
```

`wlP1p1s0`은 현재 Thor Wi-Fi 인터페이스 이름이다. 이름이나 주소가 바뀌면 `ip -brief -4 addr` 결과에 맞춰 두 번째 명령을 수정한다.

G1:

```bash
sudo ip route replace 239.255.76.67/32 dev eth0 src 192.168.123.164
ip route get 239.255.76.67
```

각 호스트에서 결과가 `dev eth0`과 해당 호스트의 `src` 주소를 가리켜야 한다. 주소가 맞지 않으면 BOOST 링크 진단으로 확인한다.

Thor:

```bash
cd ~/BOOST
PIXI_FROZEN=true pixi run -e thor-teleop bash scripts/diagnose_g1_teleop_links.sh 192.168.123.164
```

## 2. G1에서 BrainCo USB 손 서버 실행

G1에서 USB serial 포트들을 점유한 이전 프로세스를 정리한다. 이 명령은 `/dev/ttyUSB0`과 `/dev/ttyUSB1`을 사용 중인 프로세스만 강제 종료한다.

```bash
sudo fuser -v -k -KILL /dev/ttyUSB0 /dev/ttyUSB1
sleep 1
sudo fuser -v /dev/ttyUSB0 /dev/ttyUSB1
```

마지막 명령에 PID가 나오지 않는 것을 확인한 뒤 서버를 한 번만 실행한다.

```bash
sudo /home/unitree/brainco_hand_service/tactile_service/build/brainco_hand_server_tactile \
  --network_interface eth0
```

정상 로그에는 `/dev/ttyUSB*`별 왼손·오른손 `bound` 메시지와 두 worker가 모두 보여야 한다. 한 손만 연결되면 서버를 `Ctrl+C`로 멈춘 뒤 위 `fuser` 정리와 USB 확인을 다시 한다. 서버를 두 번 띄우지 않는다.

## 3. G1에서 DreamWaQ와 PATH arm bridge 실행

G1의 다른 터미널에서 `~/BOOST`의 DreamWaQ launcher를 한 번 실행한다. 이 launcher가 PATH arm/state bridge도 시작하므로 bridge를 따로 실행하지 않는다.

```bash
cd ~/BOOST
CYCLONEDDS_HOME="$(conda run -n boost-py310 python -c 'import sys; print(sys.prefix)')" \
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_controller.sh
```

G1 리모컨에서 `START`로 기본 자세 램프를 시작하고 자세가 잡힌 뒤 `A`로 DreamWaQ 정책을 시작한다. DreamWaQ와 BrainCo hand service는 각각 자기 터미널에서 계속 실행 상태로 둔다.

## 4. Thor에서 경로와 G1 상태 확인

DreamWaQ launcher가 bridge를 시작한 뒤 Thor에서 링크 진단을 실행한다.

```bash
cd ~/BOOST
PIXI_FROZEN=true pixi run -e thor-teleop bash scripts/diagnose_g1_teleop_links.sh 192.168.123.164
```

`G1_ROBOT_STATE` 패킷 수가 0보다 커야 한다. `G1_PATH_UPPER_COMMAND`는 PATH teleop 시작 후 증가한다. Galaxy XR의 8012 포트와 PATH용 5566 포트는 별도이며, 5566은 Thor 내부 ZMQ 연결이다.

이전 PATH/XR 실행이 남아 있을 때만 중복 manager를 정리한다. `q`로 기존 PATH 세션을 끝낸 뒤 확인한다.

```bash
ss -ltnp '( sport = :8012 or sport = :5566 )'
pgrep -af galaxy_xr_manager_thread_server.py
```

불필요한 manager가 남아 있으면 `ps -fp`로 명령을 확인하고, `galaxy_xr_manager_thread_server.py` 프로세스의 해당 PID만 `kill -TERM`으로 종료한다. 새 `path-teleop` 세션 하나만 실행한다.

## 5. Thor에서 Galaxy XR / PATH 시작

```bash
cd ~/BOOST
PIXI_FROZEN=true pixi run -e thor-teleop path-teleop
```

Galaxy XR에서 `https://192.168.0.16:8012/`에 접속한다. manager 로그의 `connected=True`, 좌우 `wrist_valid=(L:1 R:1)`와 PATH 로그의 최신 XR age를 확인한 뒤 **같은 PATH 터미널에서** `r`을 눌러 arm tracking을 시작한다.

- `r`: arm tracking 시작/재개
- `s`: 손목 실험 trial 디렉터리 기록 시작/중지 (`data.csv`, 요약, 카메라 JPEG)
- `q`: 기록을 닫고 PATH 세션 종료

팔 추종이 대기 상태면 PATH 로그에서 `g1_state_age`, `xr_source_age`, `max_arm_error`를 확인한다. `r`은 최신 G1 상태가 있고 양팔이 Q_READY에서 `0.100rad` 이내이며 최신 XR 프레임이 있을 때 활성화된다. 조건이 덜 갖춰졌을 때는 pending 상태에서 자동으로 기다린다. DreamWaQ 기본 팔 각도와 PATH의 영점 Q_READY를 0으로 맞췄지만, 정책의 서기·걷기 안정성은 MuJoCo에서 별도로 확인한다. 시작·재개·복귀 목표에는 속도 제한 감쇠가 적용되며, 상태 로그의 `xr_outliers`와 `xr_resyncs`는 손별 누적 횟수다.

각 `s` 기록은 `$HOME/data/boost/path_wrist_experiments/path_wrist_experiment_NNN/`에 저장된다. 안에는 `data.csv`, `summary.json`, head/좌우 손목 JPEG와 `images_index.csv`가 들어간다. 카메라가 연결되지 않으면 경고 후 CSV 기록은 계속된다.

## 6. Thor에서 Revo2 hand MPPI 시작

PATH/XR manager가 실행 중이고 양손 landmark가 들어오는 상태에서 Thor의 별도 터미널로 실행한다.

```bash
cd ~/BOOST
BRAINCO_DDS_INTERFACE=eth0 PIXI_FROZEN=true pixi run -e thor hand-mppi-publish
```

MPPI가 Revo2 명령을 DDS로 보내고, G1의 BrainCo server가 USB serial로 손에 전달한다. 손 상태가 없거나 한 손만 뜨면 G1 server 로그의 양쪽 `bound` 및 worker 상태와 Thor MPPI의 양손 state 수신을 확인한다.

## 종료

1. Thor hand MPPI 터미널에서 `Ctrl+C`.
2. Thor PATH 터미널에서 `q`.
3. G1 BrainCo server와 DreamWaQ 터미널에서 각각 `Ctrl+C`로 종료.

다음 실행에서 USB 점유가 남아 있으면 2절의 `fuser -k -KILL` 명령을 다시 수행한다. 다른 DreamWaQ 또는 `rt/lowcmd` writer를 동시에 실행하지 않는다.
