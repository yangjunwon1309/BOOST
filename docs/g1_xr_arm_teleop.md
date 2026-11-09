# 실제 G1 Galaxy XR 팔 teleop

이 실행 경로는 `url_g1_deploy`의 실기 검증된 G1_29 IK와 기록 흐름을
BOOST에 별도 포장합니다. 손가락/그리퍼 컨트롤러는 포함하지 않고,
MuJoCo용 PATH teleop 및 Revo2 MPPI 흐름도 바꾸지 않습니다.

## 실행 위치

DreamWaQ와 카메라 서버는 G1 호스트에서, Galaxy XR/Televuer 팔 teleop은
G1 DDS 네트워크에 연결된 제어 호스트(예: Thor)에서 실행합니다. 기본 카메라
설정은 head와 두 wrist 모두 timestamp가 포함된 TIM2 recording stream을
사용합니다.

### G1 호스트

```bash
bash scripts/setup_dreamwaq_runtime.sh
bash scripts/setup_teleimager_runtime.sh
bash scripts/run_g1_teleimager.sh
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_body_controller.sh
```

DreamWaQ body controller는 별도 터미널에 계속 둡니다. 기존
`run_g1_dreamwaq_controller.sh`를 여기서 실행하면 PATH adapter도 시작되어
`rt/arm_sdk`가 중복 발행되므로 사용하지 않습니다.

### Thor/XR teleop 호스트

```bash
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
BOOST_G1_XR_HOST=192.168.123.2 \
BOOST_G1_TELEIMAGER_HOST=192.168.123.164 \
pixi run -e thor-teleop g1-arm-teleop
```

`BOOST_G1_XR_HOST`에는 Galaxy XR에서 접속할 수 있는 teleop 서버 주소를
넣습니다. TLS 인증서는 `scripts/ensure_televuer_tls.sh`가 재사용하거나
생성합니다. Thor에서 만든 새 인증서라면 헤드셋이 해당 인증서를 신뢰하도록
등록해야 합니다. 기록 경로는 teleop 호스트의
`$HOME/data/boost/g1_teleop/g1_arm_teleop`이며
`BOOST_G1_TELEOP_DATA_ROOT`, `BOOST_G1_TELEOP_TASK_NAME`으로 바꿀 수 있습니다.

G1이 안정적으로 선 뒤 DreamWaQ 터미널에서 `START` 다음 `A`를 눌러 제어를
활성화합니다. XR URL을 헤드셋에서 열고 세션을 연결한 뒤 teleop 터미널에서
`r`을 눌러 팔 추종을 시작합니다. G1의 DreamWaQ 프로세스는 별도로
`Ctrl+C`로 끝내며, DreamWaQ 종료 damping 경로를 실행합니다.

## 명령 배선과 인덱스

```text
Galaxy XR wrist pose
  -> Televuer
  -> G1_29 IK (14 arm joints, slots 15..28)
  -> rt/arm_sdk
  -> DreamWaQ arm target reader
  -> merged G1 body command on rt/lowcmd
```

- teleop은 `--motion`을 강제합니다. 이 모드의 G1 arm controller는 `rt/arm_sdk`에 팔 타깃을 발행하고 `rt/lowcmd`에는 쓰지 않습니다.
- DreamWaQ가 `rt/lowcmd`의 유일한 writer입니다. 이 실행기는 PATH/G1 어댑터를 띄우지 않습니다. 그 어댑터도 `rt/arm_sdk` publisher이므로 동시에 실행하면 안 됩니다.
- 팔 관절 인덱스는 G1 공식 순서 15–28이며, 왼팔 7개 다음 오른팔 7개입니다. enable marker는 슬롯 29입니다. 허리 회전 요청은 슬롯 12와 enable 슬롯 30으로 전달되며 DreamWaQ가 적용합니다.
- teleop은 손 DDS 토픽, 손가락 관절 명령 또는 그리퍼 컨트롤러를 만들지 않습니다. IK URDF 안의 손 관절은 기구학용 고정 링크로만 남습니다.
- Galaxy XR의 stick 값은 데이터에 기록되지만 `--loco-client`는 금지되어 실제 base 이동 명령으로 전송되지 않습니다. 각 프레임의 `actions.base.cmd_vel_sent`는 `false`입니다. 하체 자세와 토크는 DreamWaQ가 제어합니다.

Thor의 8012 포트가 사용 중이거나 로컬 G1 arm adapter/teleop이 실행 중이면
teleop 시작을 거부합니다. G1 호스트에서 기존 DreamWaQ controller나
`g1_lowlevel_bridge`가 실행 중이면 body controller 시작을 거부합니다. 기존
`run_g1_dreamwaq_controller.sh`, PATH/G1 adapter, MuJoCo Galaxy XR 스택과
동시에 실행하지 마세요.

## 기록 형식

`EpisodeWriter`가 프레임을 백그라운드 큐로 넘겨 다음 구조에 기록합니다.

```text
$BOOST_G1_TELEOP_DATA_ROOT/$BOOST_G1_TELEOP_TASK_NAME/
  episode_0000/
    data.json
    colors/...
```

각 `data.json` 프레임에는 다음 내용이 들어갑니다.

- `states.left_arm/right_arm`: qpos, qvel, lowstate 기반 torque
- `actions.left_arm/right_arm`: IK가 만든 관절 목표와 feed-forward torque
- `actions.left_ee/right_ee.target_pose`: IK 직전 XR 손목 목표 pose; `target_pose_mat`도 함께 저장
- `states.body`: G1 35 슬롯 qpos/qvel/torque. `info.g1_body_joint_names`가 0–28의 Unitree 순서를 적고 29–34는 패딩입니다. `actions.body.qpos`는 기존 호환을 위해 관절값이 아닌 `[vx, vy, wz]`입니다.
- `states.base`, `states.torso`: pelvis/torso IMU, odometry, projected gravity 및 twist
- `ik_residual`, `waist`, 카메라 캡처 시각과 DDS 샘플 수신 시각/age
- `annotation.label_id`와 `annotation.label`: 현재 서브태스크 라벨
- `colors`: head 좌/우 이미지와 활성화된 wrist 이미지

손가락 상태/동작은 저장되지 않습니다. `states.*_ee.qpos`와
`actions.*_ee.qpos`는 빈 배열이고, `*_ee` 키는 XR 손목 목표와 선택적 FK
pose를 위한 것입니다. 따라서 Dex1/Revo2 손가락 채널을 기대하는 기존
변환기는 이 기록에 그대로 적용하면 안 됩니다.

TIM2 스트림이 제공될 때 이미지별 capture wall time/sequence/lag가 기록되고,
각 row에는 IK action 시각과 lowstate/odom/IMU 수신 시각도 저장됩니다. 기본
task steps는 `approach; grasp; manipulate; release`이며 실행 시
`BOOST_G1_TELEOP_TASK_STEPS`로 교체할 수 있습니다.

## 키보드

- `r`: XR 세션이 연결된 뒤 팔 추종 시작 또는 저장 후 재개
- `s`: 추종 중 녹화 시작; 녹화 중 다시 누르면 성공으로 저장을 요청하고 팔을 standby 자세로 이동. XR 입력이 끊긴 중에도 저장 가능
- `1`–`9`: 해당 task step 라벨 선택, `0`: unlabeled
- `c`: 고정 head anchor 재설정
- `w`: DreamWaQ에 오른쪽 허리 yaw 요청을 토글
- 짧은 `q`: 직전 저장 에피소드를 failure로 표시
- `q` 키 반복 입력이 이어지도록 3초 누름: teleop 정지 후 팔을 정리하고 종료
- `Ctrl+C`: 종료 및 recorder flush
- XR 오른쪽 컨트롤러 `A`: 같은 teleop 종료 요청

`s` 저장은 비동기입니다. 화면의 SAVING 상태가 사라지고 `episode_####`
완료 로그가 나온 뒤 파일이 닫힌 것입니다. 녹화 시작은 신선한 XR pose가
필요하지만, 진행 중인 기록은 XR 입력이 끊겨도 `s`로 저장할 수 있습니다.
저장한 에피소드는 기본 성공으로 끝나며, 직후 짧은 `q`로 마지막 에피소드를
failure로 바꿀 수 있습니다. `q`를 3초 누르면 종료하고, `Ctrl+C`도 종료 경로로
쓸 수 있습니다. XR 오른쪽 컨트롤러 `A`도 종료 요청을 보냅니다.

## 필터 경로

팔 IK에는 upstream의 4-sample weighted moving filter가 적용되고,
DreamWaQ config에는 기본 `arm_smoothing_tau: 0.04`가 있습니다. 따라서 DDS
전송 Hz가 높아도 실제 팔은 두 단계의 smoothing을 따라갑니다. 이는 이번
통합에서 수정하지 않았습니다.
