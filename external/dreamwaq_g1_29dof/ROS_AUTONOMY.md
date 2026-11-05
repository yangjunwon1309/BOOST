# DreamWaQ ROS1 자율보행 (G1 29DoF)

대상: `/home/unitree/00_workspace/g1_deploy/thirdparty/dreamwaq_g1_29dof`.

실제 설치된 경로는 다음과 같다.

```text
LiDAR / URL-LiDAR-SLAM → gotf → /dreamstep_odom_g1
TRG / gomg / goal_manager → /trg/output/multigoal_path
lpptest (pure_pursuit) → /cmd/velocity (geometry_msgs/Twist)
ROS 브리지 (system Python 3.8 / Noetic)
  → 127.0.0.1:15002 UDP → DreamWaQ (emer_tw) → rt/lowcmd
```

`gocontrol`은 Unitree 기본 보행 컨트롤러에 속도 명령을 보내므로 이 절차에서는 실행하지 않는다.
기존 Thor UDP 15001 설정은 그대로 두고, 자율보행 전용 `g1_ros_autonomy.yaml`을 선택한다.
브리지는 ROS 속도의 `linear.x / linear.y / angular.z`를 각각 `vx / vy / wz`로 전달한다.
조이스틱용 좌우 부호 반전을 ROS 명령에 적용하지 않는다.

## 준비 및 실행 순서

각 ROS 터미널에서 Noetic(기존 `.bashrc` 선택창에서는 2)을 선택하고 `master_g1`을 실행한다.
ROS Master는 `192.168.0.19:11311`, ROS IP는 `192.168.0.19`이다.
호스트 RViz에서도 같은 master를 사용하고 호스트 자신의 ROS IP를 지정한다.

1. 별도 터미널에서 `roscore`.
2. `lidar`.
3. 현장 맵에 맞는 SLAM 실행. 현재 장소의 `3_maploc.launch`는
   `bash deploy/run_slam_autonomy.sh`로 실행하면 SLAM CPU 2-3과 OpenMP 제한이 적용된다.
   제공된 다른 장소 설명의 `1_maploc.yaml / 2_maploc.yaml`을 현재 장소로 가정하지 않는다.
4. `gotf`. 고개를 젖히는 경우 실제 사용하는 tilt odometry topic을 브리지에도 지정한다.
5. 호스트 RViz에서 2D Pose Estimate. 위치와 진행 방향이 맵과 맞는지 확인한다.
6. `trgnature`, `gomg`, 필요 시 `goperception`.
7. 브리지 터미널:

   ```bash
   cd /home/unitree/00_workspace/g1_deploy/thirdparty/dreamwaq_g1_29dof
   bash deploy/run_ros_autonomy_bridge.sh
   # 항상 HOLD ON으로 시작한다. 아직 h를 누르지 않는다.
   ```

8. DreamWaQ 터미널:

   ```bash
   cd /home/unitree/00_workspace/g1_deploy/thirdparty/dreamwaq_g1_29dof
   bash deploy/run_dreamwaq_autonomy.sh
   ```

   위 스크립트는 사용자가 제공한 NEW3-4 export, 속도 상한, arm smoothing=0,
   CPU, 15 ms deadline, controller-switch 옵션에 자율보행 config를 추가한 것이다.
   직접 기존 명령을 사용하는 경우 다음 옵션만 추가한다.

   ```bash
   --config "$PWD/deploy/robot_control/configs/g1_ros_autonomy.yaml"
   ```

9. DreamWaQ의 안내에 따라 **START → A**로 자세 준비와 정책을 시작한다.
   **X**를 눌러 walking으로 바꾼다. 조이스틱이 중앙이면 브리지 HOLD가 자율보행 속도를 0으로 유지한다.
   기존 설명서의 `L2+B / L2+Up / R2+A / Start`는 예전 컨트롤러 조작법이며
   이 DreamWaQ 실행 경로에는 그대로 적용하지 않는다.
10. `lpptest`를 실행하고 `sendgoal`로 목표를 보낸다.
11. 브리지 터미널에서 **h**를 눌러 HOLD OFF. 새 속도 명령과 odometry가 도착하면 자율보행한다.

처음 현장 확인은 로봇을 지지한 상태에서 수행하고, 지면에서는 진행/좌우/회전 방향과
위치추정 일치를 먼저 확인한다. 소프트웨어 통합 검증만으로 실제 보행 안정성을 확인한 것은 아니다.

## 정지 및 재시작

- **h**: 자율보행 HOLD 토글. HOLD는 ROS 속도를 0으로 만든다. 물리 조이스틱은 HOLD보다 우선한다.
- **X**: DreamWaQ standing ↔ walking. standing은 모든 속도 입력을 0으로 만든다.
- **Select**: DreamWaQ 종료 및 damping.
- 브리지의 **d** 또는 `/dreamwaq/damping` (`std_msgs/Empty`): damping 요청을 래치한다.
  DreamWaQ 부모 루프, 자세 준비 취소 콜백, writer fault 경로가 이를 확인하고 종료한다.
  damping은 지지 토크를 떨어뜨리므로 로봇을 지지할 수 있는 상황에서 사용한다.
  다시 보행하려면 **브리지와 DreamWaQ 둘 다 재시작**한다.
- `--controller-switch`의 **L1+Y**는 기존 Unitree ↔ DreamWaQ 전환이다.
  자율보행 중 전환 전에 HOLD를 켠다. 기존 전환은 damping과 컨트롤러 재시작을 포함한다.
- 물리 조이스틱은 ROS 속도보다 항상 우선한다. HOLD, odometry 수신 중단, 브리지 미실행 중에도
  walking 상태에서는 수동 조종할 수 있다. damping 요청은 수동 조이스틱도 차단한다.
- 명령이 0.3초 이상 끊기면 자율보행 속도 0. odometry가 1.0초 이상 끊기거나
  브리지 UDP가 0.3초 이상 끊겨도 자율보행 속도 0이다. 수동 조이스틱은 계속 사용할 수 있다.
  odometry가 다시 도착하면 operator HOLD가 꺼진 경우 입력이 자동으로 재개될 수 있다.
- `/align/state`가 `ALIGNED`이면 HOLD ON으로 전환한다. 자동으로 해제하지 않는다.
  재수행 시 `resetgoal`, 시작점 복귀, HOLD 상태 확인, `sendgoal`, **h** 순서로 진행한다.
  수동 복귀는 walking 상태에서 조이스틱으로 수행한다. 자율보행 HOLD는 켜 두어도 된다.
- 브리지 Ctrl+C는 HOLD/속도 0을 요청한다. DreamWaQ Ctrl+C는 damping 종료이다.

키보드를 쓸 수 없는 경우:

```bash
rostopic pub -1 /dreamwaq/hold std_msgs/Bool 'data: true'
rostopic pub -1 /dreamwaq/hold std_msgs/Bool 'data: false'
rostopic pub -1 /dreamwaq/damping std_msgs/Empty '{}'
```

## 진단 및 모터 없는 검증

```bash
rostopic info /cmd/velocity
rostopic hz /dreamstep_odom_g1
rostopic echo /dreamwaq/autonomy_status
rostopic echo /align/state
tail -f /tmp/dreamwaq-g1-29dof-last.log
```

`autonomy_status.reason`은 `operator_hold`, `command_stale`, `odometry_stale`,
`active`, `damping` 중 하나다. `active`도 DreamWaQ에서 X로 walking을 선택해야 움직인다.
기존 pure-pursuit 설정은 최대 선속도 0.6 m/s와 각속도 0.5 rad/s이며,
DreamWaQ의 0.95 / 0.75 / 0.75 상한은 이를 확대하지 않고 수신 명령을 제한한다.
역방향 vx는 export 범위에 따라 최대 절댓값 0.8 m/s이다.

```bash
/home/unitree/miniconda3/envs/emer_tw/bin/python -m unittest discover \
  -s deploy/tests -p test_ros_autonomy.py -v
bash --noprofile --norc -c 'source /opt/ros/noetic/setup.bash; /usr/bin/python3 deploy/tests/check_ros_autonomy_pipeline.py'
```

두 번째 검증은 별도의 localhost ROS master와 임시 UDP 포트를 사용한다.
실제 ROS Twist 전달, HOLD, clipping, 입력 timeout, ALIGNED, 브리지 재시작,
damping 요청을 검증하며 Unitree SDK와 LowCmd publisher를 만들지 않는다.
이 검증은 현장 localization 정확도나 실제 모터 damping/보행을 검증하지 않는다.

tilt 모드 예:

```bash
bash deploy/run_ros_autonomy_bridge.sh --odom-topic /dreamstep_odom_tilt
```

기존 local planner의 odom topic도 같은 topic으로 설정해야 한다.

## 이번 설치의 검증 결과

- 자율보행 단위 테스트 11개, 기존 컨트롤러 전환 테스트 32개,
  기존 DDS/UDP/속도 상한 테스트 8개 통과.
- 별도의 ROS master에서 실제 ROS1 → UDP 통합 검증 통과. 실제 로봇 보행은 실행하지 않았다.
- 전체 기존 테스트에는 `test_observation_and_fused_action_match_sim2sim_assembly`의
  수치 허용오차 실패가 있다(최대 차이 약 6.56e-6, 허용오차 3e-6).
  수정 전 스냅샷에서도 같은 결과가 재현됐다.
- 기존 HOMIE ONNX용 `test_offline`은 `emer_tw`에 `onnxruntime`이 없어 import 단계에서 실패한다.
  이번 DreamWaQ ROS 브리지는 이 의존성을 사용하지 않는다.
- 기존 수정사항은 보존했으며, 덮어쓴 세 파일의 원본은
  `/home/unitree/.cache/dreamwaq-ros-autonomy-backup-20261002-194704/`에 보관했다.

## 조이스틱 및 15 ms 메시지 점검

초기 자율보행 연결에서는 HOLD와 브리지 timeout이 물리 조이스틱도 차단했다.
현재는 수정하여 물리 조이스틱이 ROS 명령, HOLD, odometry/브리지 수신 중단보다 우선한다.
damping 래치와 NEW3 standing은 수동 입력도 차단한다. 변경 후 DreamWaQ를 재실행하고
START → A → X 순서로 walking에 들어가면, 브리지를 실행하지 않아도 수동 조종할 수 있다.

`policy deadline miss ... deadline_ms=15.0`은 정책의 현재 상태에 대한 응답이 시간 안에
도착하지 않았다는 경고다. 현재 기존 런타임은 마지막으로 수락한 목표를 유지하며 계속 실행한다.
`--max-policy-timeouts 3`은 현재 코드에서는 진단 값이며, 3회 지연 시 자동 종료 조건이 아니다.
확인한 실기 로그는 timeout 누적 3,041회 뒤 Select 입력으로 종료했고, damping 전송을 기록했다.
추론 중앙값은 4.514 ms, 상태→응답 p95는 28.541 ms였다. 15 ms를 단순히 늘리지 않았다.

이전 5초 shadow 비교에서는 기존 CPU 배치가 16/250 timeout,
일부 SLAM·카메라·ROS·VS Code 작업을 임시로 CPU 0-4에 제한한 배치가 1/250 timeout이었다.
후자도 LowState age p99 17.050 ms 등 타이밍 조건을 만족하지 않아 FAIL_DELAY_DETECTED였다.
당시 CPU 경합을 줄여도 전체 타이밍 기준은 통과하지 못했다. 그 시험의 임시 affinity는 원복했다.
아래는 이후 적용한 CPU 분리와 최종 검증 결과다.

## 자율보행 CPU 분리와 검증

`run_dreamwaq_autonomy.sh`는 현재 프로세스의 CPU 배치를 적용하고, 새로 시작되는
SLAM·카메라·ROS·local planner·VS Code 프로세스와 모든 스레드를 2초 간격으로 확인하는
단일 CPU 감시기를 시작한다. 같은 사용자로 실행되는 대상 프로세스에 적용된다.
VS Code 터미널에서 실행해도 DreamWaQ는 CPU 0-7에서 시작한 뒤 각 역할을 분리한다.

| 역할 | CPU |
|---|---|
| 카메라 서버와 자식 프로세스 | 0-1 |
| SLAM의 모든 스레드 | 2-3 |
| ROS 브리지·local planner·기타 ROS·VS Code | 0-3 |
| arm SDK reader | 3 (SLAM과 공유) |
| LowState reader | 4 |
| LowCmd writer | 5 |
| 정책 프로세스 | 6 |
| 제어 루프 | 7 |

SLAM을 다음에 실행할 때는 기존 `slamcustom` 대신 다음 명령을 사용한다.
이 명령은 현재 장소의 `3_maploc.launch`를 사용하며, OpenMP worker 상한도 2로 제한한다.
이미 실행 중인 SLAM에는 CPU 감시기가 스레드별 affinity를 적용한다.

```bash
bash deploy/run_slam_autonomy.sh
```

브리지와 DreamWaQ는 각각 별도 터미널에서 실행한다.

```bash
bash deploy/run_ros_autonomy_bridge.sh
bash deploy/run_dreamwaq_autonomy.sh
```

자율보행 실행 스크립트는 Python 스레드 전환 간격을 1 ms로 설정하고 순환 GC를 비활성화한다.
참조 카운트에 따른 객체 해제는 계속 동작한다. 이 옵션들은 자율보행 스크립트에서 선택하며,
기존 직접 실행 명령의 기본 동작을 바꾸지 않는다. 15 ms deadline과 50 Hz 정책 주기는 유지한다.

모터 명령 없이 측정한 결과:

| 측정 | CPU 분리 전 (20초) | 최종 설정 (60초) |
|---|---:|---:|
| 측정 주기 수 | 999 | 3000 |
| 15 ms timeout | 21 | 0 |
| LowState age p99 | 14.155 ms | 4.444 ms |
| request roundtrip p99 | 14.403 ms | 7.405 ms |
| writer ready p99 | 11.223 ms | 4.611 ms |
| 판정 | FAIL_DELAY_DETECTED | PASS_NO_DELAY |

CPU 분리 후 local planner 누락을 보완하고, Python 전환 간격과 순환 GC 대기를 조정했다.
최종 측정에서는 SLAM odometry도 약 10 Hz로 정상 출력됐다. 측정 결과는
`/home/unitree/.cache/dreamwaq-cpu-timing-20261002/final_no_gc.json`과 같은 이름의 CSV에 있다.
원본 백업은 `/home/unitree/.cache/dreamwaq-cpu-partition-backup-20261002-w_hrpt1d/`에 있다.

타이밍을 다시 확인하려면 실제 DreamWaQ 제어 프로세스를 종료한 상태에서 실행한다.
실행 스크립트는 실제 제어가 실행 중이면 shadow 측정을 거절한다.

```bash
bash deploy/run_dreamwaq_autonomy.sh --shadow --duration 60 \
  --shadow-report /home/unitree/.cache/dreamwaq-cpu-timing-20261002/recheck.json
/usr/bin/python3 deploy/partition_autonomy_cpus.py
```

CPU 감시기 로그는 `~/.cache/dreamwaq-autonomy-cpus/watch.log`에 있다.
이 검증은 LowCmd publisher를 만들지 않았으며, DDS Write·네트워크 전송·실제 모터 보행은
검증 범위에 포함하지 않는다. 측정 구간의 통과가 모든 미래 부하에서의 실시간 보장을 뜻하지는 않는다.

## Local planner의 command_stale 수정

2026-10-02에 `/cmd/velocity`가 간헐적으로 0.3초 이상 끊기는 원인을 수정했다.
local planner가 대용량 그래프 시각화 메시지를 Python 객체로 디코딩하고 다시 인코딩하면서
명령 루프가 지연됐다. 실제 메시지 크기는 약 1.88 MB이고 초당 약 10회 들어왔다.
`url_navigation/interface/ros/ros1/nodes/lpp_node.py`의 그래프 입력을 `rospy.AnyMsg`로 받아
원본 바이트를 그대로 전달하도록 변경했다. 입력 MD5를 확인하며, 출력은 기존
`MarkerArray` 타입과 latch를 유지한다. 실제 입력·출력 601개의 해시도 모두 일치했다.

시작할 때 latched 경로가 TF보다 먼저 도착해 버려지던 문제도 수정했다.
최신 경로를 보관하고 TF가 준비되면 명령 루프에서 적용한다. TF 대기 중 새로운 경로가
들어오면 최신 경로를 사용한다. `configs/ros/ros1_lpp.yaml`의 `enable_path_view_viz`는
false로 설정했다. 이 지형 기반 경로 시각화 분석은 새 목표마다 1–2초 걸렸으며,
일반 경로 표시와 그래프 표시, 경로 추종 계산은 유지한다.

| 측정 | 수정 전 (10초) | 수정 후 (60초) |
|---|---:|---:|
| 속도 명령 수신 주기 | 약 13.9 Hz | 20.0 Hz |
| 최대 수신 간격 | 422.1 ms | 69.7 ms |
| 300 ms 초과 수신 간격 | 3회 | 0회 |
| local planner CPU 사용률 | 약 61% | 약 10% |

수정 후에는 1,200개 속도 명령과 3,000회 감시 판정에서 `command_stale`가 0회였다.
오도메트리는 약 10 Hz였다. 감시 시간은 원래의 0.3초를 유지했으므로 앞서 임시로 제안한
`--cmd-timeout 0.5`는 필요하지 않다. 모터 명령을 보내지 않는 관찰 측정이며,
실제 로봇의 정지 거리나 보행 안정성을 검증한 결과는 아니다.

기존 `lpptest`도 수정한 모듈과 설정을 사용한다. 현재 수정한 플래너가 이미 실행 중이면
중복 실행하지 않는다. 브리지는 다음 기본 명령으로 실행하고, 준비가 끝나면 `h`로 HOLD를 해제한다.

```bash
bash deploy/run_ros_autonomy_bridge.sh
```

그래프 전달·잘못된 타입 차단·latched 전달·TF 준비 전 경로 수신·최신 경로 보존에 대한
회귀 테스트 7개를 별도 loopback ROS master에서 통과했다.
테스트는 `/home/unitree/nature_task/url_navigation_pkg/tests/test_ros1_graph_relay.py`에 있다.
측정 결과는 `/home/unitree/.cache/dreamwaq-command-stale-20261002/final_timing.json`이고,
최초 원본 백업은 같은 디렉터리의 `backup-0vugw6yx/`에 있다.
