# PATH 개발 명세서 v14 — 노출 시각 기준 로봇 상태 정합

**문제.** 손목 카메라의 RGB-D 노출 시각과 FK에 쓰인 $\mathbf q$가 어긋나면 pelvis 프레임 타겟이 직접 틀어진다. 현재 구조는 시각 정합이 아니라 **"캡처 시점에 마지막으로 도착한 상태"를 보유(hold)** 한다.

**핵심 판단**

> 정합은 **Thor에서** 해야 한다. 워커가 아니다.
> 카메라 프레임과 로봇 상태가 **둘 다 Thor에 도착**하므로, 두 시각을 같은 클럭에서 비교할 수 있는 유일한 지점이 Thor다.
> 그리고 정합의 대부분은 **상수 바이어스**이므로, 보간보다 **바이어스 캘리브레이션**이 먼저다.

---

## Part 0. 오차 크기와 구조

### 0.1 위치 오차

$$
\lVert\Delta\mathbf p\rVert \approx \big(v_{\text{wrist}} + r\,\omega_{\text{wrist}}\big)\,\Delta t
$$

$r$은 손목에서 물체까지의 거리다(카메라가 손목에서 9 cm, 물체가 카메라에서 0.1–0.4 m).

| 상황 | $v$ | $\omega$ | $r$ | 계수 | 5 ms | 10 ms | 20 ms | 50 ms | 100 ms |
|---|---|---|---|---|---|---|---|---|---|
| 느린 접근 (보조 중) | 0.15 | 0.5 | 0.30 | 0.30 m/s | 0.15 | 0.30 | 0.60 | 1.50 | 3.00 cm |
| 통상 teleop | 0.30 | 1.0 | 0.30 | 0.60 | 0.30 | 0.60 | 1.20 | 3.00 | 6.00 cm |
| 빠른 재배향 | 0.50 | 2.0 | 0.30 | 1.10 | 0.55 | 1.10 | **2.20** | **5.50** | 11.0 cm |
| 빠른 + 먼 물체 | 0.50 | 2.0 | 0.50 | 1.50 | 0.75 | 1.50 | 3.00 | 7.50 | 15.0 cm |

### 0.2 법선 오차 — 이쪽이 더 아프다

$$\Delta\theta = \omega_{\text{wrist}}\,\Delta t$$

| $\omega$ | 5 ms | 10 ms | 20 ms | 50 ms | 100 ms |
|---|---|---|---|---|---|
| 0.5 rad/s | 0.14° | 0.29° | 0.57° | 1.43° | 2.86° |
| 1.0 | 0.29° | 0.57° | **1.15°** | 2.86° | 5.73° |
| 2.0 | 0.57° | 1.15° | 2.29° | **5.73°** | 11.5° |

**v4에서 시야 기준 5.7°를 손 기준 0.57°로 줄인 것이 여기서 상쇄된다.** 20 ms 정합 오차 + 1 rad/s면 1.15°로 이미 두 배이고, 50 ms면 5.73°로 개선분이 통째로 사라진다.

$\Delta\theta$는 v13.1의 자세 권한에 직접 들어가므로, **손목이 잘못된 방향으로 최대 40°까지 돌 수 있는 시스템에서 법선 입력이 5° 틀리는 것**은 그냥 두면 안 된다.

### 0.3 오차의 구성

$$
\Delta t = \underbrace{\Delta t_{\text{bias}}}_{\text{상수}} + \underbrace{\Delta t_{\text{jitter}}}_{\text{변동}} + \underbrace{\Delta t_{\text{hold}}}_{\text{현재 구조}}
$$

| 성분 | 원인 | 크기 | 해결 |
|---|---|---|---|
| $\Delta t_{\text{hold}}$ | "최신 보유" 구조 | **상태 발행 주기까지, 무경계에 가까움** | §1 링버퍼 — **필수** |
| $\Delta t_{\text{bias}}$ | G1→Thor 전송 − D405 USB 전송 | 수 ms ~ 수십 ms, **거의 상수** | §2 캘리브레이션 — **효과 최대** |
| $\Delta t_{\text{jitter}}$ | 두 전송의 변동 | 수 ms | §3 보간 (조건부) |

**$\Delta t_{\text{hold}}$와 $\Delta t_{\text{bias}}$가 지배적이고 둘 다 보간 없이 해결된다.** 보간은 셋째다.

---

## Part 1. [T-A] Thor에 로봇 상태 링버퍼 — 필수

### 1.1 현재 구조의 문제

```python
# thor_wrist_d405_gateway_v11
    def _on_robot_state(self, _channel, data):
        with self.lock:
            self.latest_robot_state_packet = (now_us(), bytes(data))   # 단순 보유
    ...
    def publish_once(self):
        ...
        state_received_us, state_packet = self.latest_robot_state_packet
```

`publish_once`가 `wait_for_bundle`을 **좌·우 순차로 블로킹 호출**하므로, 그 대기 동안 상태가 갱신되어도 이미 꺼낸 값을 쓴다. 대기 시간이 프레임 주기(15 Hz → 67 ms)에 이를 수 있다.

$$\Delta t_{\text{hold}} \sim \mathcal{O}(\text{프레임 주기}) = 67\ \mathrm{ms} \;\Rightarrow\; \text{7 cm, 7.7}°$$

**§0의 표에서 가장 나쁜 칸이 바로 이 구조에서 나온다.**

### 1.2 수정

```python
@dataclass
class RobotStateHistory:
    """Timestamped robot-state ring, sampled at exposure rather than held.

    The gateway blocks in wait_for_bundle for up to a frame period, so the
    packet grabbed before that wait can be a whole frame old.  Holding the
    newest packet therefore has no bound tied to the camera at all; a ring
    plus a timestamped lookup replaces an unbounded error with one bounded by
    the state publish period.
    """

    span_s: float = 2.0
    publish_hz: float = 500.0
    entries: collections.deque = field(init=False)

    def __post_init__(self):
        self.entries = collections.deque(maxlen=int(self.span_s * self.publish_hz))

    def add(self, received_us: int, packet: bytes, state) -> None:
        self.entries.append((int(received_us), packet, state))

    def nearest(self, timestamp_us: int) -> tuple[int, bytes, object, float] | None:
        if not self.entries:
            return None
        best = min(self.entries, key=lambda e: abs(e[0] - int(timestamp_us)))
        return best[0], best[1], best[2], abs(best[0] - int(timestamp_us)) / 1000.0
```

`_on_robot_state`에서 디코드까지 마쳐 링에 넣는다. 카메라 루프가 디코드 비용을 부담하지 않게 하기 위해서다.

### 1.3 잔여 오차는 상태 발행 주기가 결정한다

| 상태 발행 | 주기 | 최근접 최악 | 최근접 평균 |
|---|---|---|---|
| 50 Hz | 20.0 ms | 10.00 ms | 5.00 ms |
| 100 Hz | 10.0 ms | 5.00 ms | 2.50 ms |
| 200 Hz | 5.0 ms | 2.50 ms | 1.25 ms |
| **500 Hz** | 2.0 ms | **1.00 ms** | 0.50 ms |
| 1000 Hz | 1.0 ms | 0.50 ms | 0.25 ms |

**G1 상태가 200 Hz 이상이면 최근접만으로 2.5 ms 이하가 되어 §3의 보간이 불필요하다.** 발행 주기 확인이 §3 결정의 전제다.

---

## Part 2. [T-B] 바이어스 캘리브레이션 — 효과가 가장 크다

### 2.1 두 클럭을 정렬할 필요가 없다

기존 진단에서 "G1 클럭과 Thor 벽시계를 비교할 수 없다"고 판단했는데, 그건 `state.timestamp_us`를 쓰려 할 때의 이야기다.

**양쪽 도착 시각을 Thor의 단조 클럭으로 찍으면 같은 클럭이 된다.** 남는 것은 두 전송 지연의 차이뿐이고, 그것은 거의 상수다.

$$
t^{\text{Thor}}_{\text{state arrival}} = t_{\text{measure}} + d_{\text{G1}\to\text{Thor}}
$$
$$
t^{\text{Thor}}_{\text{frame arrival}} = t_{\text{exposure}} + d_{\text{USB}}
$$
$$
\Rightarrow\quad
\Delta t_{\text{bias}} = d_{\text{G1}\to\text{Thor}} - d_{\text{USB}}
$$

**따라서 조회 키를 한 상수만큼 이동하면 된다.**

```python
        lookup_us = int(frame_arrival_us) + int(self.state_lookup_bias_us)
        match = self.state_history.nearest(lookup_us)
```

### 2.2 먼저 확인 — librealsense 전역 시각

D405가 `RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`을 주면 SDK가 이미 device↔host 클럭 회귀를 유지하고 있으므로, `frame.get_timestamp()`가 **host epoch 밀리초**로 나온다. 그러면 USB 지연 성분이 이미 제거된다.

```bash
python3 -c "
import pyrealsense2 as rs
p = rs.pipeline(); prof = p.start(rs.config())
dev = prof.get_device()
for s in dev.query_sensors():
    if s.supports(rs.option.global_time_enabled):
        print(s.get_info(rs.camera_info.name), 'global_time =', s.get_option(rs.option.global_time_enabled))
f = p.wait_for_frames().get_depth_frame()
import time
print('domain :', f.get_frame_timestamp_domain())
print('frame  :', f.get_timestamp())          # ms
print('host   :', time.time()*1000.0)         # ms, same epoch if GLOBAL_TIME
p.stop()
"
```

| 도메인 | 의미 | 조치 |
|---|---|---|
| `GLOBAL_TIME` | 이미 host epoch — **frame timestamp를 그대로 쓴다** | $d_{\text{USB}}$ 제거됨 |
| `SYSTEM_TIME` | host 수신 시각 (SDK 콜백 시점) | 현재와 동일, 바이어스 캘리브 필요 |
| `HARDWARE_CLOCK` | 장치 클럭 | 전역 시각을 켜거나 회귀 추정 |

**`GLOBAL_TIME`이면 §2.3의 캘리브레이션이 $d_{\text{G1}\to\text{Thor}}$ 하나만 다루면 되어 훨씬 단순해진다.**

### 2.3 캘리브레이션 — 정지 물체의 pelvis 좌표 분산 최소화

외부 장비가 필요 없고, 목적함수가 태스크와 직결된다.

**절차**

1. 정지한 물체 하나를 두고 팔을 다양하게 흔들며 60초 기록한다. **손목 회전이 큰 동작을 포함**해야 한다(회전이 바이어스에 가장 민감).
2. 각 프레임에 대해 `points_camera_m`, `frame_arrival_us`, 그리고 로봇 상태 링 전체를 저장한다.
3. 오프라인에서 $\tau$를 스윕하며 pelvis 좌표 중심의 분산을 계산한다.

$$
\tau^{\ast} = \arg\min_{\tau}\ \sum_{k}\Big\lVert \mathbf T_{\text{goal}\leftarrow\text{cam}}\big(\mathbf q(t_k+\tau)\big)\,\bar{\mathbf p}^{\text{cam}}_k - \bar{\mathbf p}^{\text{goal}}\Big\rVert^2
$$

```python
def calibrate_state_lookup_bias(records, history, extrinsics, camera_id,
                                sweep_ms=(-100, 100), step_ms=1.0):
    """Find the state-lookup offset that makes a static object stationary.

    A static object has a fixed pelvis-frame centroid by definition, so the
    offset that minimises its variance is the offset that pairs each frame
    with the joint state it was actually exposed at.  The objective is the
    quantity the belief consumes, so no external timing reference is needed
    and no clock-domain conversion is assumed.
    """
    best = (None, float("inf"))
    for tau_ms in np.arange(sweep_ms[0], sweep_ms[1] + step_ms, step_ms):
        centres = []
        for record in records:
            match = history.nearest(record.frame_arrival_us + int(tau_ms * 1000))
            if match is None:
                continue
            transform = t_goal_camera(camera_id, match[2], extrinsics)
            centres.append(transform_points(record.centre_camera_m, transform))
        if len(centres) < 20:
            continue
        variance = float(np.mean(np.var(np.asarray(centres), axis=0)))
        if variance < best[1]:
            best = (float(tau_ms), variance)
    return best
```

**해석**: 최소점이 뚜렷한 골이면(양쪽으로 분산이 확실히 증가) 신뢰할 수 있다. 평평하면 손목 회전이 부족했던 것이므로 데이터를 다시 받는다.

$\tau^\ast$가 양수면 카메라 프레임보다 **나중** 상태를 써야 한다는 뜻이고, 이는 $d_{\text{G1}\to\text{Thor}} > d_{\text{USB}}$를 의미한다.

**좌·우 카메라를 각각 캘리브레이션한다.** USB 경로가 다르면 $d_{\text{USB}}$가 다르다.

---

## Part 3. [T-C] 보간 — 조건부

### 3.1 필요 여부는 상태 발행 주기가 결정한다

$$
\Delta t_{\text{nearest}}^{\max} = \frac{1}{2 f_{\text{state}}}
$$

| $f_{\text{state}}$ | 최악 | 빠른 재배향 시 오차 | 판정 |
|---|---|---|---|
| 500 Hz | 1.0 ms | 0.11 cm, 0.11° | **보간 불필요** |
| 200 Hz | 2.5 ms | 0.28 cm, 0.29° | 불필요 |
| 100 Hz | 5.0 ms | 0.55 cm, 0.57° | 경계 |
| 50 Hz | 10.0 ms | **1.10 cm, 1.15°** | **필요** |

**§1을 넣고 실측한 뒤 판단하십시오.** G1이 고주기로 발행하면 이 절은 건너뛴다.

### 3.2 필요하면 — 관절 각 unwrap 후 선형 보간

```python
def interpolate_q(q0: np.ndarray, t0_us: int, q1: np.ndarray, t1_us: int,
                  target_us: int) -> np.ndarray:
    """Linear joint interpolation with wrap handling.

    Revolute joints can cross +-pi between samples; interpolating the raw
    values then sweeps almost a full turn.  Unwrapping the second sample
    relative to the first keeps the interpolation on the short arc.
    """
    span = float(t1_us - t0_us)
    if span <= 0.0:
        return np.asarray(q0, dtype=np.float64).copy()
    alpha = float(np.clip((target_us - t0_us) / span, 0.0, 1.0))
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    delta = np.mod(q1 - q0 + np.pi, 2.0 * np.pi) - np.pi
    return q0 + alpha * delta
```

**외삽은 하지 않는다.** `alpha`를 $[0,1]$로 clip하고, 목표 시각이 링 바깥이면 조회 실패로 처리한다. 외삽은 관절 속도 잡음을 그대로 증폭한다.

`base_quat_wxyz`는 선형 보간이 아니라 slerp을 써야 한다. 다만 IMU가 고주기면 차이가 무시할 수준이므로, **먼저 최근접으로 두고 필요 시에만** 추가한다.

---

## Part 4. [T-D] 어디까지 Thor가 하고 무엇을 보낼 것인가

### 4.1 결론 — Thor가 **정합만** 하고, FK는 워커가 유지

세 선택지가 있다.

| 안 | 번들 내용 | 장점 | 단점 |
|---|---|---|---|
| (a) 현행 | `robot_state_packet` 1개 | 최소 | **정합이 워커에서 불가** |
| **(b) 권장** | **카메라별 `q_at_exposure` (29 float) + `base_quat_at_exposure`** | 정합은 Thor, FK·외부파라미터는 워커 유지 | 프로토콜 소폭 확장 |
| (c) | 카메라별 `T_goal_cam` (16 float) | 워커 FK 불필요 | 외부파라미터가 Thor로 이동, 캘리브 반영이 재시작 필요 |

**(b)를 권한다.** 정합은 두 클럭이 만나는 Thor에서만 가능하지만, 외부 파라미터는 자주 바뀌는 값이라 워커 설정에 두는 편이 낫다. 추가 전송량은 카메라당 $33 \times 8 = 264$ B, 프레임당 528 B로 무시할 수준이다.

### 4.2 번들 스키마 확장

```python
                    "cameras": {
                        camera_id: {
                            ...,
                            # Paired at Thor, where the frame arrival and the
                            # state arrival share one monotonic clock.  The
                            # worker cannot do this: it sees neither clock.
                            "q_at_exposure": [...],              # 29 floats
                            "base_quat_at_exposure_wxyz": [...], # 4 floats
                            "state_pairing": {
                                "lookup_us": int,
                                "state_arrival_us": int,
                                "residual_ms": float,        # |lookup - arrival|
                                "bias_us": int,              # applied tau*
                                "method": "nearest" | "interpolated",
                                "wrist_speed_mps": float,    # from finite difference
                                "wrist_omega_rps": float,
                                "estimated_error_m": float,  # (v + r*w) * residual
                            },
                        }
                    }
```

`estimated_error_m`을 번들이 직접 계산해 싣는 것이 핵심이다. **워커와 웹뷰가 "이 프레임의 기하를 얼마나 믿어도 되는가"를 하나의 숫자로 볼 수 있다.**

### 4.3 워커 측 변경

`process_latest`가 `paired_state.q` 대신 카메라별 `q_at_exposure`를 쓴다.

```python
        # One FK per camera, from that camera's own exposure-paired state.
        # The two wrists are captured at slightly different times, so a single
        # shared q would reintroduce the skew this pairing removes.
        t_goal_cameras = {}
        for camera_id in CAMERA_IDS:
            q = np.asarray(frames[camera_id]["q_at_exposure"], dtype=np.float64)
            t_goal_cameras[camera_id] = (
                g1_link_transforms(q)[CAMERA_TO_WRIST_LINK[camera_id]]
                @ self.extrinsics[camera_id]
            )
```

**좌·우가 서로 다른 $\mathbf q$를 쓴다.** 캡처 시각이 다르므로 이것이 옳다. v11 감사 D-2의 "프레임당 1회 FK"는 이제 **카메라당 1회**가 된다(2회/프레임).

`up_goal`도 카메라별 `base_quat_at_exposure_wxyz`에서 계산한다.

---

## Part 5. [T-E] 정합 품질 게이트

v11의 `robot_state_max_delta_ms`를 대체한다. 판정 기준을 **시간이 아니라 추정 위치 오차**로 바꾸는 것이 핵심이다.

$$
\hat\epsilon = \big(v_{\text{wrist}} + r\,\omega_{\text{wrist}}\big)\,\Delta t_{\text{residual}}
$$

```python
                # Gate on the estimated geometric error, not on elapsed time.
                # 20 ms is harmless while the arm is still and unacceptable
                # during a fast reorientation; a fixed time threshold has to
                # be tuned for the worst case and then wastes good frames.
                if pairing["estimated_error_m"] > self.pairing_max_error_m:
                    self.stats["pairing_rejects"] += 1
                    continue
```

| 파라미터 | 기본 | 근거 |
|---|---|---|
| `pairing_max_error_m` | **0.008** | 복셀 5 mm와 grip 병합 반경 30 mm 사이 |
| `pairing_max_residual_ms` | 15.0 | 이차 안전망 (속도 추정 실패 시) |
| `pairing_max_extrapolation_ms` | 0.0 | **외삽 금지** |

$r$은 해당 프레임의 오브젝트 중앙 depth를 쓴다. 오브젝트가 없으면 게이트를 적용하지 않는다.

---

## Part 6. 이것이 해결하지 못하는 것

**SAM3 지연으로 인한 마스크 노후는 별개 문제다.** 정합이 완벽해도 마스크는 여전히 100–200 ms 전 프레임의 것이다.

| 문제 | 원인 | 해결 |
|---|---|---|
| pelvis 타겟이 어긋남 | **q 정합** | **본 명세서** |
| 마스크가 최신 영상과 어긋나 보임 | SAM3 지연 | 웹뷰: 소스 프레임 캐시 (진단) |
| 오래된 마스크에 최신 depth 적용 시 바닥 선택 | 마스크·depth 에폭 혼용 | **절대 금지** — 같은 번들만 사용 |

**세 번째가 특히 중요하다.** 현재 코드는 같은 번들의 마스크와 depth만 쓰므로 안전하다. 지연을 줄이려고 "최신 depth + 과거 마스크"를 시도하면 물체가 이동한 만큼 바닥을 집는다.

실시간 개선이 필요하면 **SAM3를 저주기 키프레임으로 쓰고 그 사이를 광류/트래커로 마스크를 최신 프레임에 전달**한 뒤, **그 최신 프레임의 depth와 $\mathbf q$로** 접촉점을 재계산해야 한다. 별도 명세로 다룬다.

---

## Part 7. 구현 순서

| 순위 | 작업 | 파일 | 효과 | 규모 |
|---|---|---|---|---|
| **0** | 진단만 — `state_pairing` 필드 기록, 개입 없음 | gateway, protocol | 현황 측정 | 소 |
| **1** | T-A 링버퍼 + 최근접 | gateway | $\Delta t_{\text{hold}}$ 제거 | 소 |
| **2** | T-B 전역 시각 확인 + 바이어스 캘리브레이션 | 확인 + 오프라인 | $\Delta t_{\text{bias}}$ 제거 | 중 |
| **3** | T-D 번들 스키마 + 워커 배선 | protocol, gateway, worker | 정합 반영 | 중 |
| **4** | T-E 오차 기반 게이트 | worker | 나쁜 프레임 차단 | 소 |
| 5 | T-C 보간 | gateway | $\Delta t_{\text{jitter}}$ | 소 |

**0번을 먼저 하십시오.** 개입 없이 `residual_ms`와 `estimated_error_m`만 기록하면, 현재 오차가 실제로 얼마인지와 §3의 보간이 필요한지가 동시에 드러난다. **7–10 cm 오차가 정말 정합에서 오는지도 이 단계에서 확정된다.**

1번은 0번과 함께 넣어도 안전하다. 구조 개선일 뿐 판정을 바꾸지 않는다.

---

## Part 8. 검증

1. **정합 잔차.** `residual_ms` 중앙값이 상태 발행 주기의 절반 이하.
2. **hold 제거.** `residual_ms`의 최대가 프레임 주기(67 ms)가 아니라 상태 주기 수준으로 떨어짐.
3. **바이어스 골.** $\tau$ 스윕 곡선에 뚜렷한 최소가 존재하고, $\tau^\ast$ 양쪽 ±20 ms에서 분산이 2배 이상 증가.
4. **정지 물체 안정성.** 팔을 흔드는 동안 정지 물체의 `center_goal_m` 표준편차가 **1 cm 이하**. 수정 전에는 5–10 cm여야 한다(회귀 대조).
5. **법선 안정성.** 같은 조건에서 `surface_primitives[0].axis_goal`의 각도 표준편차가 2° 이하.
6. **좌우 독립.** 두 카메라의 `q_at_exposure`가 서로 다르고, 각각의 `residual_ms`가 독립적으로 작다.
7. **게이트 선택성.** 팔이 정지한 상태에서 `pairing_rejects`가 0, 빠른 재배향 중에만 증가.
8. **외삽 금지.** 링버퍼 범위 밖 조회 시 프레임이 거부되고 외삽이 발생하지 않음.

---

## Part 9. 확인 요청

1. **G1 로봇 상태 LCM 발행 주기** — §3의 보간 필요 여부를 결정한다
2. **D405 timestamp domain** — `GLOBAL_TIME`이면 §2가 크게 단순해진다 (§2.2의 명령)
3. `wait_for_bundle`이 논블로킹 최신 반환인지 블로킹 대기인지 — 후자면 $\Delta t_{\text{hold}}$가 §1.1의 추정대로 크다
4. 좌·우 D405가 같은 USB 컨트롤러에 물려 있는지 — 다르면 $d_{\text{USB}}$가 달라 카메라별 바이어스가 필요하다
5. 현재 관측된 7–10 cm 오차가 **정지 상태에서도** 나타나는지 — 나타난다면 정합이 아니라 외부 파라미터(§v11 CAD) 문제다
