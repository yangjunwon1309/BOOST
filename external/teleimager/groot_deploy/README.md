# G1: existing teleimager + GR00T camera add-on

설치 위치:

```text
/home/unitree/29_dof_g1_deploy/thirdparty/teleimager/groot_deploy/
```

G1의 기존 teleimager 코드와 conda 환경을 그대로 사용한다. 이 폴더에는
GR00T 모델이나 로봇 controller가 없다. 하체/균형 controller도 변경하지 않는다.

## 서버 실행

기존 teleimager conda 환경에서:

```bash
cd /home/unitree/29_dof_g1_deploy/thirdparty/teleimager/groot_deploy
./run_server.sh
```

이 스크립트는 부모 teleimager 폴더로 이동한 뒤 기존 명령을 그대로 실행한다.

```bash
teleimager-server --rs
```

기존 `cam_config_server.yaml`, 카메라 serial, 포트와 conda package를 변경하지
않는다.

## 세 카메라 검증

서버를 실행한 상태에서 G1의 다른 터미널을 열고 같은 conda 환경에서:

```bash
cd /home/unitree/29_dof_g1_deploy/thirdparty/teleimager/groot_deploy
python check_server.py
```

검사 항목:

- config service `127.0.0.1:60000`의 실제 응답
- head ZMQ 활성화, binocular 1280x480
- left/right wrist ZMQ 활성화, 각각 640x480
- 세 JPEG stream이 실제로 도착하고 decode되는지
- 각 stream의 teleimager 측정 FPS

`cam_config_client.yaml` fallback은 사용하지 않고 live config service에 직접
질의하므로, 서버가 꺼져 있는데 캐시 파일 때문에 통과하는 일이 없다.

검사가 통과하면 G1 쪽 준비는 끝이다. 이후 GR00T 서버, observation, action
실행은 모두 Thor의 기존 XR teleop 폴더에 추가한 `groot_deploy/`가 담당한다.

