# BOOST 시작점과 외부 자산 출처

## PATH 기반 코드

- 저장소: `https://github.com/yangjunwon1309/PATH-G1`
- 시작 commit: `f98f8854cb45cc60afe1565f0eee6912d62aba64`
- 시작 당시 브랜치: `homie-g1-snapshot`
- BOOST 작업 브랜치: `boost`
- 작업 폴더: `C:/studioprojects/BOOST`
- 옮긴 모듈과 BOOST 내 새 위치는 [`boost/teleop/UPSTREAM.md`](boost/teleop/UPSTREAM.md)에 기록한다.

BOOST 작업 폴더는 위 commit을 기반으로 만든 별도 Git worktree다. PATH-G1 원본 작업 폴더는 수정하지 않았다.

## BrainCo Revo2 모델

- 원본: `https://github.com/BrainCoTech/brainco-description/tree/6db7e2ed1d864ee0a3ab6ff62d43a303787137f6/revo2_system`
- 원본 commit: `6db7e2ed1d864ee0a3ab6ff62d43a303787137f6`
- BOOST 내 위치: `assets/robot/brainco_revo2/revo2_system/`
- 포함 파일: 좌우 URDF, MuJoCo MJCF, USD, visual/collision mesh

원본 저장소에서 확인한 트리에는 라이선스 파일이 보이지 않았다. 공개 배포 전에 자산 재배포 권한을 확인한다.

## 손목 카메라 기준

- 좌우 D405는 각각 `left_wrist_yaw_link`, `right_wrist_yaw_link`에 고정된 것으로 다룬다.
- PATH의 카메라 frame·FK 연결은 `services/pose_service/config/wrist_d405_v11.yaml` 및 `services/pose_service/remote_offload/wrist_d405_geometry_pipeline_v11.py`를 기준으로 한다.
- 현재 PATH 설정의 좌우 카메라 serial 배정은 파일 주석상 현장 확인 전 provisional이다.

## Revo2 통신 기준

- 외부 서비스: `https://github.com/unitreerobotics/brainco_hand_service`
- 서비스는 USB-serial 손 명령을 DDS로 연결한다. 좌우별 `rt/brainco/{left,right}/cmd`와 `rt/brainco/{left,right}/state` 토픽을 사용한다.
- 손가락 인덱스: Thumb, Thumb_aux, Index, Middle, Ring, Pinky. 위치·속도는 `[0, 1]` 정규화 값이다.
- 서비스 소스는 이 worktree에 복사하지 않는다. BOOST 쪽 bridge는 서비스 DDS 인터페이스에 맞춰 연결한다.
- Revo2용 BOOST 어댑터의 토픽 계약은 [`boost/io/revo2_hand_service/README.md`](boost/io/revo2_hand_service/README.md)에 정리한다.
