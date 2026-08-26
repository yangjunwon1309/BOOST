# PATH 개발 명세서 v15.1 — XR 소스 추상화 (XRoboToolkit → TeleVuer)

**목표.** `pico_manager_thread_server.py`의 XR 입력을 장치 종속 SDK(`xrobotoolkit_sdk`)에서 **WebXR 기반 televuer**로 교체 가능하게 만든다. Galaxy XR을 포함한 모든 WebXR 장치에서 동작하는 것이 요구 조건이다.

**결론.** 가능하다. 그리고 **교체 표면이 매우 작다.**

**v15 → v15.1 개정.** v16(WebXR 드롭아웃 대응)의 세 항목을 조건부에서 **필수**로 옮겼다.

| 항목 | v15 | v15.1 | 이유 |
|---|---|---|---|
| 도착 간격 모니터 | v16 조건부 | **§6.3 필수** | 이것 없이는 v15의 성공 여부를 판정할 수단이 없다 |
| televuer 이미지 차단 | §3.2에 언급 | **§3.2 계약으로 승격** | 수정이 아니라 통합 시점의 설정 선택이다 |
| stale 안전 순서 | §5.2 단일 임계 | **§5.2 2단 순서** | 성능이 아니라 안전이며, 첫 실기부터 필요하다 |

v16의 나머지(감쇠 외삽, SSID 분리, TCP_NODELAY, 타임스탬프 패치)는 **§8.2의 측정 결과에 따라** 선택 적용한다.

---

## Part 0. 왜 작은가 — 기하 진입점이 하나뿐이다

PATH의 XR 데이터 경로는 다음 한 함수로 수렴한다.

```python
def controller_waist_targets(left_pose7, right_pose7, head_pose7) -> (left_waist, right_waist, ok):
    head_robot  = T_ROBOT_OPENXR @ head_xr  @ T_OPENXR_ROBOT
    left_robot  = T_ROBOT_OPENXR @ left_xr  @ T_OPENXR_ROBOT
    right_robot = T_ROBOT_OPENXR @ right_xr @ T_OPENXR_ROBOT
    left_waist  = world_arm_to_waist_head_position(left_robot,  head_robot)
    right_waist = world_arm_to_waist_head_position(right_robot, head_robot)
```

$$
\mathbf T^{\text{waist}}_i = \mathbf G\,\mathbf X_i\,\mathbf G^{-1},
\qquad
\mathbf p^{\text{waist}}_i = \mathbf p^R_i - \mathbf p^R_{\text{head}} + \mathbf o
$$

**이 아래로는 전부 소스 무관하다.** `T_ROBOT_OPENXR` 켤레변환, `WAIST_OFFSET_XYZ`, pelvis-identity 별칭, `LivePoseStreamer`, ZMQ 5559, bridge, IPOPT — 어느 것도 SDK를 알지 못한다.

지적하신 대로 **전신 데이터가 필요 없다.** 필요한 것은 머리 1 + 컨트롤러 2의 전역 SE(3)와 버튼·트리거뿐이다.

---

## Part 1. 실제 SDK 의존 표면

코드 전체에서 `xrt.*` 호출을 세면 18종이고, PATH가 실제로 쓰는 것은 14종이다.

| 분류 | 호출 | PATH 필요 | televuer 대응 |
|---|---|---|---|
| **포즈** | `get_headset_pose` | **필수** | 있음 |
| | `get_left_controller_pose` | **필수** | 있음 |
| | `get_right_controller_pose` | **필수** | 있음 |
| **아날로그** | `get_left_trigger` / `get_right_trigger` | **필수** (Dex1 grip) | WebXR `buttons[0].value` |
| | `get_left_grip` / `get_right_grip` | 필요 (손가락 데이터) | `buttons[1].value` |
| | `get_left_axis` / `get_right_axis` | 필요 (yaw, locomotion) | `axes[2..3]` |
| **버튼** | `get_A_button` / `get_B_button` | **필수** (A/B dead-man) | `buttons[4]` / `buttons[5]` |
| | `get_X_button` / `get_Y_button` | 필요 | 좌 컨트롤러 `buttons[4]` / `[5]` |
| | `get_left_menu_button` | 필요 | 장치별 |
| **전신** | `is_body_data_available` | **불필요** | 없음 |
| | `get_body_joints_pose` | **불필요** | 없음 |
| | `get_time_stamp_ns` | 대체 가능 | 도착 시각으로 |
| **초기화** | `init` | — | Vuer 시작 |

**전신 3종은 `_process_3pt_pose` / `ThreePointPose` / 시각화 테스트 경로에서만 쓰인다.** PATH 제어 경로는 이 셋을 전혀 참조하지 않는다.

---

## Part 2. [X-A] 어댑터 인터페이스

### 2.1 계약을 pose7이 아니라 **SE(3)** 로 정의한다

`_pico_pose_to_mat_openxr`가 `[x,y,z,qx,qy,qz,qw]` → 4×4를 한다. televuer는 이미 4×4를 주므로, pose7을 경유하면 **불필요한 왕복 손실**이 생긴다.

```python
@dataclass(frozen=True)
class XrSample:
    """One XR frame, already in the OpenXR basis (+x right, +y up, -z forward).

    The contract is SE(3), not the 7-vector the XRoboToolkit SDK happens to
    return: televuer produces matrices directly, and routing them through a
    quaternion would add a lossy round trip for no benefit.  The XRT adapter
    performs the 7-vector conversion internally instead.
    """
    head: np.ndarray                 # 4x4, OpenXR basis
    left: np.ndarray                 # 4x4
    right: np.ndarray                # 4x4
    head_valid: bool
    left_valid: bool
    right_valid: bool
    left_trigger: float              # [0, 1]
    right_trigger: float
    left_grip: float
    right_grip: float
    left_axis: tuple[float, float]   # thumbstick (x, y)
    right_axis: tuple[float, float]
    buttons: dict[str, bool]         # "A", "B", "X", "Y", "left_menu", ...
    arrival_monotonic: float
    source_timestamp_ns: int | None  # None if the source has no device clock


class XrSource(abc.ABC):
    """Head + two controller poses.  Deliberately no body/skeleton surface.

    PATH maps the operator's head-to-hand vector onto the robot's
    pelvis-to-wrist vector, so a full-body estimate is not merely unnecessary
    but adds a dependency that no WebXR device provides uniformly.
    """

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def latest(self) -> XrSample | None:
        """Newest sample, or None if nothing has arrived yet."""

    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...

    @property
    def supports_body(self) -> bool:
        return False
```

### 2.2 기존 코드 변경 최소화

`controller_waist_targets`에 4×4를 직접 받는 형제 함수를 둔다.

```python
def controller_waist_targets_se3(
    left_xr: np.ndarray, right_xr: np.ndarray, head_xr: np.ndarray,
    *, left_ok: bool = True, right_ok: bool = True, head_ok: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """SE(3) form of controller_waist_targets.  The 7-vector version delegates
    here after conversion, so both sources share one geometric path."""
    if not head_ok:  head_xr  = CONST_HEAD_POSE.copy()
    if not left_ok:  left_xr  = CONST_LEFT_ARM_POSE.copy()
    if not right_ok: right_xr = CONST_RIGHT_ARM_POSE.copy()
    head_robot  = T_ROBOT_OPENXR @ head_xr  @ T_OPENXR_ROBOT
    left_robot  = T_ROBOT_OPENXR @ left_xr  @ T_OPENXR_ROBOT
    right_robot = T_ROBOT_OPENXR @ right_xr @ T_OPENXR_ROBOT
    return (
        world_arm_to_waist_head_position(left_robot,  head_robot),
        world_arm_to_waist_head_position(right_robot, head_robot),
        bool(head_ok and left_ok and right_ok),
    )
```

기존 `controller_waist_targets`는 pose7 → SE(3) 변환 후 이 함수를 호출하는 얇은 래퍼가 된다. **기하는 한 곳에만 존재한다.**

---

## Part 3. [X-B] 두 어댑터

### 3.1 XRT 어댑터 (기존 동작 보존)

```python
class XrtSource(XrSource):
    """XRoboToolkit SDK adapter.  Preserves the current behaviour exactly."""

    def _poll(self) -> XrSample:
        head, head_ok   = _pico_pose_to_mat_openxr(xrt.get_headset_pose())
        left, left_ok   = _pico_pose_to_mat_openxr(xrt.get_left_controller_pose())
        right, right_ok = _pico_pose_to_mat_openxr(xrt.get_right_controller_pose())
        return XrSample(
            head=head, left=left, right=right,
            head_valid=head_ok, left_valid=left_ok, right_valid=right_ok,
            left_trigger=float(xrt.get_left_trigger()),
            right_trigger=float(xrt.get_right_trigger()),
            left_grip=float(xrt.get_left_grip()),
            right_grip=float(xrt.get_right_grip()),
            left_axis=tuple(xrt.get_left_axis()),
            right_axis=tuple(xrt.get_right_axis()),
            buttons={
                "A": bool(xrt.get_A_button()), "B": bool(xrt.get_B_button()),
                "X": bool(xrt.get_X_button()), "Y": bool(xrt.get_Y_button()),
                "left_menu": bool(xrt.get_left_menu_button()),
            },
            arrival_monotonic=time.monotonic(),
            source_timestamp_ns=int(xrt.get_time_stamp_ns()),
        )

    @property
    def supports_body(self) -> bool:
        return True          # only this source can feed the 3-point path
```

### 3.2 TeleVuer 어댑터

```python
class TeleVuerSource(XrSource):
    """WebXR adapter via televuer/Vuer.

    Device-agnostic by construction: the headset runs a browser, so Quest,
    Pico, Apple Vision Pro and Galaxy XR all present the same WebXR pose and
    gamepad surface.  That is the whole reason for this path -- the
    XRoboToolkit SDK is Pico-only.
    """

    def __init__(self, *, cert_file: Path, key_file: Path, ngrok: bool = False,
                 use_hand_tracking: bool = False,
                 button_map: str = "xr-standard"):
        from televuer import TeleVuerWrapper          # 실제 이름은 §8-1에서 확정
        self._wrapper = TeleVuerWrapper(
            binocular=True, use_hand_tracking=use_hand_tracking,
            img_shape=None,                            # PATH does not use televuer for images
            cert_file=str(cert_file), key_file=str(key_file), ngrok=ngrok,
        )
        ...

    def _poll(self) -> XrSample | None:
        data = self._wrapper.get_tele_data()
        if data is None:
            return None
        return XrSample(
            head=_as_se3(data.head_mat),
            left=_as_se3(data.left_wrist),
            right=_as_se3(data.right_wrist),
            head_valid=_finite_se3(data.head_mat),
            left_valid=_finite_se3(data.left_wrist),
            right_valid=_finite_se3(data.right_wrist),
            left_trigger=float(data.tele_state.left_trigger),
            right_trigger=float(data.tele_state.right_trigger),
            ...
            arrival_monotonic=time.monotonic(),
            source_timestamp_ns=None,          # WebXR gives no device clock here
        )
```

### 3.2.1 [필수] 이미지 경로를 반드시 끈다

**`img_shape=None`은 최적화가 아니라 계약이다.**

Vuer는 포즈와 이미지를 **같은 웹소켓**으로 보낸다. WSS는 TCP이므로 **head-of-line 블로킹**이 발생한다.

$$
t_{\text{block}} \approx \frac{S_{\text{image}}}{B_{\text{effective}}}
\approx \frac{150\,\mathrm{KB}\times 8}{15\,\mathrm{Mbit/s}} \approx 80\ \mathrm{ms}
$$

스테레오 JPEG 한 장이 포즈 메시지를 80 ms 막고, 30 Hz로 보내면 포즈가 계속 뒤에 밀린다. **로봇이 정지했다가 도약하는 증상이 정확히 이것이다.**

PATH는 자체 웹뷰(§v8, §v11)를 쓰므로 televuer의 이미지가 전혀 필요 없다. teleimager도 같은 이유로 통합하지 않는다.

```python
        self._wrapper = TeleVuerWrapper(
            binocular=True,
            use_hand_tracking=use_hand_tracking,
            # Not an optimisation: pose and image share one websocket, so an
            # image frame head-of-line blocks the pose stream by roughly
            # (image bytes / link rate).  PATH renders its own diagnostic view,
            # so this path is pure cost.
            img_shape=None,
            cert_file=str(cert_file), key_file=str(key_file), ngrok=ngrok,
        )
```

**조작자 시야가 필요하면 별도 경로**로 보낸다 — WebRTC(televuer v3.0 지원) 또는 완전히 다른 스트림. 어느 쪽이든 **포즈 웹소켓과 분리**하는 것이 요점이다.

### 3.3 선택

```python
parser.add_argument("--xr-source", choices=["xrt", "televuer"], default="xrt")
parser.add_argument("--televuer-cert", type=Path, default=Path("~/.config/xr_teleoperate/cert.pem").expanduser())
parser.add_argument("--televuer-key",  type=Path, default=Path("~/.config/xr_teleoperate/key.pem").expanduser())
parser.add_argument("--televuer-hand-tracking", action="store_true")
```

`supports_body`가 False인데 3점/SMPL 경로를 요구하면 **기동 시점에 명확히 실패**시킨다.

```python
        if args.stream_mode in BODY_STREAM_MODES and not source.supports_body:
            raise SystemExit(
                f"--xr-source {args.xr_source} provides head and controller poses only; "
                f"stream mode {args.stream_mode} requires body joints"
            )
```

---

## Part 4. [X-C] 좌표 규약 — 최대 위험 요소

### 4.1 위험의 구조

현재 코드는 SDK 포즈가 **OpenXR 기저**(+x 우, +y 상, −z 전방)라고 가정하고 켤레변환한다.

$$\mathbf T^{R} = \mathbf G\,\mathbf X\,\mathbf G^{-1}, \qquad \mathbf G = \mathbf T_{\text{ROBOT}\leftarrow\text{OPENXR}}$$

WebXR도 같은 기저이므로 **원리적으로는 호환된다.** 그러나 televuer는 `xr_teleoperate`의 부품이고, **그 안에서 이미 Unitree용 변환을 적용했을 수 있다.** 그렇다면 여기서 다시 켤레변환하면 **이중 변환**이 된다.

**이것이 이 작업의 유일한 실질 위험이다.** 나머지는 배선이다.

### 4.2 판별 — 원시 WebXR 포즈를 써야 한다

televuer의 `TeleData`에는 두 종류가 섞여 있을 수 있다.

| 종류 | 특징 | PATH에 필요 |
|---|---|---|
| 원시 WebXR 포즈 | 헤드셋 기준 세계 좌표, 변환 없음 | **이것** |
| Unitree 리타게팅 결과 | 이미 로봇 프레임, `T_to_unitree_*` 적용됨 | 아님 (이중 변환) |

`tv_wrapper.py`에서 `T_robot_openxr` / `T_to_unitree_*` 같은 상수를 적용하는 지점을 찾아, **그 이전 값**을 노출하도록 한다. 필요하면 `television.py`(원시 캡처)를 직접 쓴다.

### 4.3 검증 절차 — 정적 비교

두 소스를 **동시에** 돌릴 수 있으면 가장 확실하다.

```python
# tools/compare_xr_sources.py
"""Log both sources side by side while the controller is held still.

An axis or handedness mismatch is invisible in motion but obvious in a static
comparison, and a double transform shows up as a consistent rotation offset
rather than noise.
"""
    xrt_sample = xrt_source.latest()
    tv_sample = televuer_source.latest()
    for name in ("head", "left", "right"):
        a = getattr(xrt_sample, name); b = getattr(tv_sample, name)
        delta_p = b[:3, 3] - a[:3, 3]
        delta_r = np.degrees(np.linalg.norm(rotation_vector(b[:3, :3] @ a[:3, :3].T)))
        print(f"{name:6s} dp={np.round(delta_p, 4)}  dR={delta_r:6.2f} deg")
```

**판정 기준**

| 관측 | 해석 | 조치 |
|---|---|---|
| $\Delta p \approx 0$, $\Delta R \approx 0$ | 동일 규약 | 그대로 사용 |
| $\Delta R \approx 90°$ 또는 $180°$ (일정) | 기저 불일치 또는 이중 변환 | 상수 보정 행렬 도입 |
| 한 축의 부호만 반전 | 좌수/우수 불일치 | 반사 행렬 필요 — **`det = -1`이므로 켤레변환 전에 처리** |
| 무작위 | 필드 매핑 오류 | §8-1 재확인 |

동시 구동이 불가능하면 **알려진 자세**를 쓴다. 컨트롤러를 책상 위에 눕히고(중력 정렬) 각 축을 벽 방향에 맞춘 뒤 두 소스를 각각 기록해 비교한다.

### 4.4 보정 행렬 도입 지점

불일치가 확인되면 어댑터 내부에서 흡수한다. **`controller_waist_targets_se3`에는 절대 넣지 않는다** — 그러면 두 소스가 다른 기하 경로를 타게 된다.

```python
class TeleVuerSource(XrSource):
    # Absorbed inside the adapter so that both sources hand
    # controller_waist_targets_se3 an identical OpenXR-basis pose.  Putting it
    # downstream would fork the geometric path by source, which is exactly the
    # coupling this abstraction removes.
    BASIS_CORRECTION = np.eye(4)          # set from the 4.3 comparison
```

---

## Part 5. [X-D] 버튼 매핑

### 5.1 WebXR `xr-standard`가 대부분을 덮는다

WebXR Gamepad는 표준 매핑을 정의한다.

| 인덱스 | 역할 | PATH 용도 |
|---|---|---|
| `buttons[0]` | trigger (아날로그) | **Dex1 grip/ungrip** |
| `buttons[1]` | squeeze / grip (아날로그) | 손가락 데이터 |
| `buttons[2]` | touchpad | — |
| `buttons[3]` | thumbstick click | — |
| `buttons[4]` | A (우) / X (좌) | **A/B dead-man** |
| `buttons[5]` | B (우) / Y (좌) | |
| `axes[2], axes[3]` | thumbstick x, y | yaw, locomotion |

Quest 3와 Pico 4는 이 매핑을 따른다. **다만 장치별 예외가 있으므로 테이블로 두고 오버라이드 가능하게 한다.**

```python
XR_BUTTON_MAPS = {
    # WebXR "xr-standard" gamepad mapping.  Quest and Pico follow it; keep the
    # table so a device that does not can be corrected without touching the
    # control logic.
    "xr-standard": {
        "right": {"A": 4, "B": 5, "trigger": 0, "grip": 1, "stick_click": 3},
        "left":  {"X": 4, "Y": 5, "trigger": 0, "grip": 1, "stick_click": 3,
                  "left_menu": 6},
    },
}
```

### 5.2 A/B dead-man의 안전 요구 — 2단 해제 (필수)

A/B가 위치 보조의 유일한 동의 기준이고(§v13.1), 자세 권한은 **A/B와 무관하게 항상 적용**되므로(v13.1), 지령 단절의 결과가 이전보다 크다. **단일 임계로는 부족하다.**

**해제 순서가 규약이다.**

| 경과 | 조치 | 근거 |
|---|---|---|
| **> `STALE_RELEASE_S` (0.20 s)** | **버튼 전부 False, 트리거 0** | A/B 해제 → 위치 보조 램프다운 시작 |
| **> `STALE_INVALID_S` (0.35 s)** | **포즈 invalid** | bridge의 stale 경로로 참조 정지 |

**버튼 해제가 포즈 무효화보다 먼저여야 한다.** 순서가 반대면 보조가 걸린 채로 지령이 죽어, 참조가 마지막 belief 타겟을 향해 계속 끌린다.

```python
    STALE_RELEASE_S = 0.20        # buttons, triggers  -> release the dead-man
    STALE_INVALID_S = 0.35        # poses              -> hand over to bridge stale path

    def latest(self) -> XrSample | None:
        sample = self._latest
        if sample is None:
            return None
        age = time.monotonic() - sample.arrival_monotonic
        if age <= self.STALE_RELEASE_S:
            return sample
        # Over Wi-Fi a dropped connection is indistinguishable from a held
        # button, so consent expires first and the pose only afterwards.
        # Reversing the order would leave the assist engaged while the command
        # that justifies it is already dead.
        released = dataclasses.replace(
            sample,
            buttons={key: False for key in sample.buttons},
            left_trigger=0.0, right_trigger=0.0,
            left_grip=0.0, right_grip=0.0,
        )
        if age <= self.STALE_INVALID_S:
            return released
        return dataclasses.replace(
            released, left_valid=False, right_valid=False, head_valid=False
        )
```

`STALE_INVALID_S = 0.35`는 bridge의 `stale_after_s = 0.3`과 겹치도록 잡는다. **소스가 먼저 무효화하고 bridge가 뒤이어 확인**하는 순서다.

**v13.1 자세 권한과의 연결.** `rotation_command`가 낡으면 `rotation_target_from_relation`의 기준이 낡는다. `command_age_ms > 200`이면 bridge에서 자세 권한도 램프다운시킨다.

```python
                        # A stale command rotation makes the relation solve
                        # anchor to where the operator was, not where they are.
                        if control_age_ms > 200.0:
                            gate_weight = 0.0
```

---

## Part 6. [X-E] 지연·안정성 — v10.1에 영향

### 6.1 전송 경로가 달라진다

| | XRoboToolkit | TeleVuer |
|---|---|---|
| 경로 | 장치 SDK (유선/전용) | **Wi-Fi → WSS → Vuer** |
| 주기 | ~72–90 Hz | 60–90 Hz (렌더 루프 종속) |
| 지터 | 낮음 | **Wi-Fi 종속, 수십 ms 가능** |
| 장치 타임스탬프 | `get_time_stamp_ns` | **없음** |

**장치 타임스탬프가 없다는 것이 실질적 차이다.** 현재 `PicoReader`는 `stamp_ns` 변화로 중복 프레임을 걸러내는데, televuer에서는 그 기준이 사라진다. **도착 시각 + 포즈 변화량**으로 대체한다.

```python
        # No device clock on the WebXR path, so duplicate suppression falls
        # back to "the pose did not change".  A pure arrival-time filter would
        # pass a repeated frame that the browser re-sent.
        if previous is not None and np.allclose(sample.head, previous.head, atol=1e-9):
            continue
```

### 6.2 v10.1 파라미터 재검토가 필요하다

v10.1은 bridge가 **50 Hz**로 지령을 발행하고 게이트웨이가 **25 ms** 창으로 조회한다고 가정한다.

$$
\text{조회 성공} \iff \big|t_{\text{frame}} - t_{\text{command}}\big| \le 25\ \mathrm{ms}
$$

televuer의 Wi-Fi 지터가 25 ms를 넘으면 **조회 실패가 늘고 FK 폴백이 잦아진다.** 이는 v10.1이 없애려던 자기확증 루프를 부분적으로 되살린다.

**조치 순서**

1. `command_lookup_failures`와 `command_lookup_age_ms`를 먼저 측정한다 (v10.1 §3.3에 이미 있음)
2. 실패율이 5%를 넘으면 `belief_command_max_delta_ms`를 25 → 40으로 완화
3. bridge 발행은 **50 Hz를 유지**한다 — 소스가 60 Hz여도 bridge는 자체 루프이므로 보간·유지로 채운다

**소스 주기와 bridge 발행 주기를 혼동하지 말 것.** bridge는 항상 50 Hz로 발행하고, 소스가 느리면 같은 값을 반복 발행한다. v10.1의 링버퍼는 그 전제로 설계되어 있다.

### 6.3 [필수] 도착 간격 모니터 — 판정 수단

**이것 없이는 v15가 성공했는지 판단할 수 없다.** "버퍼링이 생기면 대응한다"의 기준이 주관적 체감이 되고, 지터와 드롭아웃을 구별할 수 없다.

| | 지터 | 드롭아웃 |
|---|---|---|
| 도착 간격 | 8–20 ms 산포 | 정상 → **200–500 ms 공백** → 정상 |
| 증상 | 미세한 떨림, 추종 지연 | **정지 후 도약** |
| 해법 | 외삽 (v16 §2) | **소스 수정** (v16 §1) |

두 실패 모드는 평균 지연 하나로는 구별되지 않으며 **정반대의 해법**을 요구한다.

```python
@dataclass
class ArrivalMonitor:
    """Inter-arrival histogram, to separate jitter from dropout.

    Both failure modes look identical in a mean-latency number and need
    opposite fixes: extrapolation erases jitter and barely touches a 300 ms
    gap, while a dedicated link erases the gap and does nothing for jitter.
    """

    BUCKETS_MS = (10, 20, 33, 50, 100, 200, 500, 1000)
    window: int = 3000
    gaps: collections.deque = field(init=False)
    histogram: collections.Counter = field(default_factory=collections.Counter)
    max_gap_ms: float = 0.0
    previous: float | None = None

    def __post_init__(self):
        self.gaps = collections.deque(maxlen=self.window)

    def record(self, arrival_monotonic: float) -> None:
        if self.previous is not None:
            gap_ms = (arrival_monotonic - self.previous) * 1000.0
            self.gaps.append(gap_ms)
            self.max_gap_ms = max(self.max_gap_ms, gap_ms)
            bucket = next((e for e in self.BUCKETS_MS if gap_ms <= e), "over")
            self.histogram[bucket] += 1
        self.previous = arrival_monotonic

    def report(self) -> dict:
        if not self.gaps:
            return {}
        gaps = np.asarray(self.gaps)
        return {
            "median_ms": float(np.median(gaps)),
            "p95_ms": float(np.percentile(gaps, 95)),
            "p99_ms": float(np.percentile(gaps, 99)),
            "max_ms": float(self.max_gap_ms),
            "dropouts_over_150ms": int(np.sum(gaps > 150.0)),
            "histogram": dict(self.histogram),
        }
```

`XrSource` 기반 클래스가 `record`를 호출하고, `report()`를 5초 로그와 발행 payload에 싣는다. **두 어댑터 모두 동일한 모니터를 쓴다** — XRT 기준선이 있어야 televuer 수치를 해석할 수 있다.

```python
                    "source_gap_median_ms": ...,
                    "source_gap_p99_ms": ...,
                    "source_gap_max_ms": ...,
                    "source_dropouts_over_150ms": ...,
```

**XRT 기준선을 반드시 먼저 기록한다.** televuer의 p99가 60 ms일 때, XRT가 15 ms였는지 50 ms였는지에 따라 해석이 완전히 달라진다. 기준선 없이 절대값만 보면 과잉 대응하게 된다.

---

## Part 7. [X-F] Galaxy XR 특이사항

### 7.1 컨트롤러 유무

Galaxy XR(Android XR)은 **핸드 트래킹이 기본**이고 컨트롤러는 선택이다. 컨트롤러가 없으면 PATH의 두 핵심 입력이 사라진다.

| PATH 요구 | 컨트롤러 있음 | 핸드 트래킹만 |
|---|---|---|
| 손목 SE(3) | `buttons` 있는 컨트롤러 포즈 | **손목 관절 포즈로 대체 가능** |
| A/B dead-man | `buttons[4]`, `[5]` | **핀치 제스처로 대체 필요** |
| Dex1 grip | `buttons[0]` 아날로그 | **핀치 거리 → 연속값** |

televuer가 `use_hand_tracking`을 지원하므로 손목 포즈는 얻을 수 있다. **문제는 A/B다.**

### 7.2 핸드 트래킹 시 A/B 대체안

```python
def pinch_to_deadman(hand_landmarks: np.ndarray, *, enter_m: float = 0.025,
                     exit_m: float = 0.040) -> bool:
    """Thumb-index pinch as the dead-man button, with hysteresis.

    Hand tracking has no discrete button, and a single distance threshold
    chatters at the boundary exactly where the operator holds a light pinch.
    The enter/exit gap is wider than the tracker's own jitter.
    """
```

**핀치를 dead-man으로 쓰는 것은 컨트롤러 버튼보다 안전성이 낮다.** 트래킹 소실 시 핀치가 "유지된 것처럼" 보일 수 있으므로, §5.2의 stale 처리를 **핸드 트래킹에서는 더 짧게**(0.1 s) 잡는다.

### 7.3 권장 — Galaxy XR에서도 컨트롤러 사용

Galaxy XR용 컨트롤러가 있으면 그것을 쓰는 편이 안전과 정밀도 양쪽에서 낫다. **핸드 트래킹은 폴백으로 두고, 논문 실험은 컨트롤러로 수행**하는 것을 권한다.

핸드 트래킹 경로는 v13.1의 자세 권한과도 상호작용한다 — 손목 자세 추정 잡음이 커지면 `orientation_confidence` 입력이 흔들린다.

---

## Part 8. 구현 순서

### 8.1 단계

| 순위 | 작업 | 파일 | 게이트 |
|---|---|---|---|
| **0** | televuer 필드 이름·규약 확인 (§10-1) | — | 이중 변환 여부 판정 |
| **1** | X-A `XrSource` / `XrSample` + `controller_waist_targets_se3` | `pico_manager_thread_server` | — |
| **2** | X-B `XrtSource` — **기존 동작 비트 단위 보존** | 동일 | ZMQ 5559 출력이 추상화 전후 동일 |
| **3** | **§6.3 도착 간격 모니터 — XRT 기준선 측정** | 동일 | p99·max 기록 |
| **4** | X-C 좌표 규약 비교 도구 + 검증 | `tools/compare_xr_sources.py` | $\Delta p<5$ mm, $\Delta R<2°$ |
| **5** | X-B `TeleVuerSource` (**§3.2.1 `img_shape=None`**) + §5.2 2단 stale | 동일 | 동작 확인 |
| **6** | **재측정** — televuer p99·max를 3단계 기준선과 비교 | — | §8.2 판정 |
| 7 | 판정 결과에 따라 v16 §1–3 선택 적용 | 네트워크, televuer 패치 | — |
| 8 | X-E v10.1 파라미터 조정 | bridge, gateway | 조회 실패율 < 5% |
| 9 | X-F 핸드 트래킹 폴백 (컨트롤러 불가 시에만) | 동일 | — |

**2단계를 단독으로 검증하십시오.** 추상화 도입만으로 XRT 동작이 바뀌면 이후 televuer 문제와 구별되지 않습니다.

**3단계의 XRT 기준선이 6단계 판정의 전제입니다.** 이것 없이 televuer 수치만 보면 정상인지 아닌지 알 수 없습니다.

**4단계가 5단계보다 먼저입니다.** 규약을 모른 채 televuer를 붙이면 "로봇이 이상하게 움직인다"만 남고 원인을 좁힐 수 없습니다.

### 8.2 6단계 판정 — v16 적용 여부

| 관측 (XRT 기준선 대비) | 분류 | 조치 |
|---|---|---|
| p99 < 40 ms, max < 100 ms | 정상 | **v16 불필요** |
| p99 정상, max 100–200 ms 산발 | 지터만 | v16 §2 **감쇠 외삽만** |
| max > 300 ms, **불규칙** | 네트워크 드롭아웃 | v16 §1.3 SSID 분리 → §1.2 TCP_NODELAY |
| max > 300 ms, **주기적이고 시간이 갈수록 악화** | **열 스로틀** | 운용 대응 + v16 §2 외삽으로 완화 |
| 히스토그램 이봉 | 둘 다 | 네트워크 먼저, 그다음 외삽 |

**`img_shape=None`(5단계)만으로 p99가 절반 이하로 떨어지면 §3.2.1이 원인이었고, 7단계 대부분이 불필요합니다.**

### 8.3 준비해 둘 것

측정 결과와 무관하게 **전용 5 GHz SSID를 미리 준비**하시길 권합니다. G1↔Thor↔4060이 이미 9 MB/s를 쓰고 있어(v9.1), 헤드셋이 같은 AP를 공유하면 포즈 스트림이 그 뒤에 줄을 섭니다. 지금 필요 없더라도 이후 해상도나 프레임률을 올리면 필요해집니다.

---

## Part 9. 검증

**추상화 (1–2단계)**

1. **비트 단위 동등.** `--xr-source xrt`로 60초 구동 시 ZMQ 5559의 `left_waist`/`right_waist`가 추상화 이전과 동일.
2. **전신 경로 차단.** `--xr-source televuer --stream-mode <body>`가 기동 시점에 명확한 메시지로 실패.

**좌표 규약 (3단계)**

3. **정적 비교.** 컨트롤러 정지 상태에서 두 소스의 $\Delta p < 5$ mm, $\Delta R < 2°$. 크면 §4.3 표로 원인 분류.
4. **축 검증.** 컨트롤러를 로봇 전방·좌측·상방으로 각각 이동시킬 때 `left_waist[:3,3]`의 증가 축이 $+x$·$+y$·$+z$와 일치.
5. **켤레변환 무결.** `det(T_ROBOT_OPENXR) = +1`이 유지되고, 보정 행렬 도입 시에도 최종 SE(3)의 회전부가 직교·$\det=+1$.

**모니터 (3단계)**

5b. **XRT 기준선.** `source_gap_*` 네 값이 5초 로그와 payload에 기록되고, XRT에서 p99·max가 확정됨.

**televuer 경로 (5단계)**

6. **연결 복원.** 헤드셋 브라우저를 새로고침해도 프로세스가 죽지 않고 재연결.
7. **stale 2단 순서.** Wi-Fi를 끊었을 때 **0.20 s에 버튼 전부 False**(A/B 해제, 트리거 0), **0.35 s에 포즈 invalid**. 순서가 뒤바뀌지 않음.
7b. **자세 권한 연동.** `command_age_ms > 200`에서 v13.1의 `gate_weight`가 0이 되고 `applied_angle_deg`가 0으로 램프다운.
8. **중복 억제.** 동일 포즈가 반복 도착할 때 `LivePoseStreamer`가 재발행하지 않음 (§6.1).
9. **주기.** bridge의 발행이 소스 주기와 무관하게 50 Hz를 유지.
9b. **이미지 차단 확인.** televuer 웹소켓 트래픽에 이미지 프레임이 없음. 있으면 §3.2.1 미적용.

**지연 (6단계)**

10. `command_lookup_failures`가 전체의 5% 미만, `command_lookup_age_ms` 중앙값이 15 ms 이하.
11. 실패율이 높으면 `belief_command_max_delta_ms` 조정 후 재측정.

**장치 (7단계)**

12. Quest 3 / Pico 4 / Galaxy XR 각각에서 3–5·7번 통과.
13. 핸드 트래킹 모드에서 핀치 dead-man의 히스테리시스가 채터하지 않음.

---

## Part 10. 확인 요청

1. **`televuer.TeleData`의 실제 필드 이름과 형상** — 아래로 즉시 확인 가능

```bash
pip install -e /path/to/televuer
python3 -c "
import inspect, televuer
from televuer import tv_wrapper
print([n for n in dir(televuer) if not n.startswith('_')])
src = inspect.getsource(tv_wrapper)
import re
for m in re.finditer(r'class (\w+).*?:|@dataclass|(\w+)\s*:\s*np\.ndarray', src):
    print(m.group(0))
print('--- transforms applied in the wrapper ---')
for line in src.splitlines():
    if any(k in line for k in ('T_robot', 'T_to_unitree', 'grd_yup2grd_zup', '@ ')):
        print(' ', line.strip())
"
```

**`grd_yup2grd_zup` 같은 상수가 보이면 §4.1의 이중 변환 위험이 현실이다.**

2. **`get_tele_data()`가 원시 WebXR 포즈를 노출하는지, 리타게팅 후 값만 주는지** — 후자면 `television.py`를 직접 써야 한다
3. **Galaxy XR용 컨트롤러 확보 가능 여부** — §7.3의 권장이 성립하는지
4. XRT 경로를 계속 유지할지, televuer로 완전 이관할지 — 유지하면 §2의 추상화가 영구 구조가 되고, 이관하면 2단계 이후 XRT 어댑터를 삭제할 수 있다
5. Vuer의 HTTPS/WSS 포트(8012)가 현재 네트워크 구성(G1 ↔ Thor ↔ 4060)에서 헤드셋에 도달 가능한지 — 방화벽과 서브넷 확인
