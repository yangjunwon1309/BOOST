# Hand MPPI 접촉 유도 비용 개발 계획

- 버전: v0.2 (2026-10-22) — 3.4절 Backend A를 MuJoCo 3.16.0에서 실측 검증, 질의 코드 수정
- 대상: Codex 및 협업자
- 상위 문서: `BOOST_dev_plan.md` 6.1.3절 (이 문서가 해당 절의 상세 명세를 대체함)

---

## 0. 현재 상태와 목표

**현재 상태**
- mimic_retargeter_lab의 Sampling-Based Retargeter(Smooth Operator, 이하 SBR) 구조에 BrainCo Revo2 embodiment 연결 완료
- MuJoCo에서 정상 동작 확인 (원본 SBR keyvector 비용 기준)

**목표**
- 원본 비용에 **물체 표면 기준 접촉 유도**를 추가한다.
- 조건: **샘플당 연산은 원본과 같은 차수로 유지**한다. 물체 정보는 샘플링 루프 밖의 "목표 생성" 단계에서만 사용한다.
- 이 단계의 용도는 **teleop**이다. 물체 좌표계 keypoint는 쓰지 않는다 (2.5절).

---

## 1. 설계 원칙

1. **비용 형태는 원본 SBR과 같고, 목표값만 바뀐다.** 모든 손가락이 물체에서 멀면($\alpha_i=0$) 출력이 원본과 정확히 같아야 한다 (회귀 테스트 기준).
2. **표면 질의는 제어 주기당 손가락 수(한 손 5개)만큼만** 한다. 샘플마다 SDF를 조회하지 않는다.
3. **표면 질의는 인터페이스 뒤에 둔다.** sim에서는 MuJoCo 충돌 기하로, 실물에서는 메쉬 SDF 또는 SAM3 점군으로 교체한다.

---

## 2. 비용 수식

### 2.1 기호 (한 손 기준, 양손은 독립적으로 동일)

| 기호 | 의미 |
|---|---|
| $q \in \mathbb R^{6}$ | 손 관절 (최적화 변수) |
| $q_{\text{prev}}$ | 직전 제어 주기의 해 |
| $X \in SE(3)$ | 손목(손 base) pose, world 기준. 팔 IK가 정한 값으로 **고정 입력** |
| $^{w}p_i(q)$ | FK로 구한 손끝 $i$ 위치, 손목 좌표계. $i=1$은 엄지 |
| $^{w}h_i$ | 사람 손끝을 기존 Kabsch-Umeyama 보정으로 로봇 손목 좌표계에 옮긴 값 |
| $v^{\text{SBR}}_{1i}$ | 원본 SBR의 엄지–손가락 keyvector 목표 (pinch 규칙 포함, 기존 코드 그대로) |
| $\phi(x)$ | 물체 표면까지의 부호 거리 (밖 +, 안 −) |
| $s(x)$, $n(x)$ | $x$에서 가장 가까운 표면점, 바깥 방향 단위 법선 |

손목 좌표계의 기준점은 **기존 코드의 keyvector 기준 frame을 그대로 사용**한다 (손바닥 site 등).

### 2.2 목표 생성 (제어 주기당 1회, 샘플링 루프 밖)

손가락 $i = 1..5$ 마다:

$$
\tilde x_i = X\,{}^{w}h_i, \qquad d_i = \phi(\tilde x_i), \qquad s_i = s(\tilde x_i), \qquad n_i = n(\tilde x_i)
$$

$$
p^{*}_i = \tilde x_i - \alpha_i\,(d_i + \varepsilon)\,n_i
$$

- $\alpha_i = 0$이면 $p^*_i = \tilde x_i$ (사람 손끝 그대로)
- $\alpha_i = 1$이면 $p^*_i = s_i - \varepsilon\,n_i$ (표면점에서 $\varepsilon$만큼 안쪽). 위치 제어 손에서 파지력을 만드는 설정값

**활성화 $\alpha_i$ (hysteresis + 연속 보간)**

상태 변수 $a_i \in \{0, 1\}$를 손가락마다 유지한다.

$$
a_i \leftarrow \begin{cases} 1 & d_i < d_{\text{on}} \\ 0 & d_i > d_{\text{off}} \\ a_i & \text{otherwise}\end{cases}
\qquad
\alpha_i = a_i\cdot\operatorname{clip}\!\left(\frac{d_{\text{off}} - d_i}{d_{\text{off}} - d_{\text{on}}},\,0,\,1\right)
$$

$d_{\text{on}} < d_{\text{off}}$. 활성화된 뒤에는 $d_i \le d_{\text{on}}$에서 $\alpha_i = 1$, 경계 사이에서는 선형으로 줄어든다.

**손목 좌표계 목표**

$$
{}^{w}p^{*}_i = X^{-1}p^{*}_i
$$

$$
\bar\alpha_{1i} = \max(\alpha_1, \alpha_i), \qquad
v^{*}_{1i} = (1-\bar\alpha_{1i})\,v^{\text{SBR}}_{1i} + \bar\alpha_{1i}\,\big({}^{w}p^{*}_i - {}^{w}p^{*}_1\big)
$$

엄지나 손가락 $i$ 중 하나라도 표면에 활성화되면, 엄지–손가락 벡터 목표를 **투영된 목표점끼리의 벡터**로 옮긴다. 사람은 햅틱이 없어 물체 폭보다 손가락을 더 오므리는 경향이 있으므로, 원본 벡터 목표를 유지하면 손가락을 물체 안쪽으로 과하게 당긴다.

**표면 질의 실패 처리**
- 질의 결과가 없거나(거리 상한 밖) 법선이 정의되지 않으면 $\alpha_i = 0$으로 처리한다.
- $s_i, n_i$에는 짧은 지수 이동 평균을 건다 (sim에서는 끄고, 실물 점군 backend에서만 켬).

### 2.3 샘플당 비용

$$
J(q) = \sum_{i=1}^{5} w_i\,\rho_\beta\big({}^{w}p_i(q) - {}^{w}p^{*}_i\big)
+ w_{th}\sum_{i=2}^{5}\rho_\beta\big(v_{1i}(q) - v^{*}_{1i}\big)
+ w_v\,\lVert q - q_{\text{prev}}\rVert^2
$$

- $v_{1i}(q) = {}^{w}p_i(q) - {}^{w}p_1(q)$
- $\rho_\beta$: 기존 코드의 robust 손실(smooth-L1) 그대로
- 첫 번째 항: 기존 코드에 손목/손바닥→손끝 keyvector 항이 있으면 그 목표를 $^{w}p^*_i$로 교체한다. 없으면 이 항을 새로 추가한다 (FK 결과를 재사용하므로 샘플당 추가 연산은 거리 5개).
- 기존 코드의 다른 항(중립 자세 정규화 등)은 그대로 둔다. 제거 여부는 6.3절 실험으로 결정한다.

최적화 알고리즘(warm start, $N$ 샘플, $M$ elite, softmax 가중 평균과 공분산 갱신 $L$회, 관절 한계 clip)은 **변경하지 않는다**.

### 2.4 선택 항: 접평면 침투 hinge (기본 비활성)

침투가 관찰될 때만 켠다. 목표 생성 단계에서 얻은 $s_i, n_i$로 손끝의 부호 거리를 접평면으로 근사한다.

$$
\hat\phi_i(q) = n_i^\top\big(X\,{}^{w}p_i(q) - s_i\big), \qquad
J_{\text{pen}}(q) = \lambda_{pen}\sum_{i:\,\alpha_i>0}\max\big(0,\,-\hat\phi_i(q) - \varepsilon_{\max}\big)^2
$$

샘플당 내적 5번이다. $\varepsilon_{\max} > \varepsilon$로 두어 목표 침투 깊이보다 더 깊이 들어가는 해만 벌한다.

### 2.5 teleop에서 물체 좌표계를 쓰지 않는 이유

목표를 물체 좌표계로 옮겨도 강체 변환 $\hat T_o = (R, t)$에 대해

$$
\lVert \hat T_o^{-1}a - \hat T_o^{-1}b\rVert = \lVert R^\top(a-b)\rVert = \lVert a - b\rVert
$$

이므로 등방성 손실의 값은 변하지 않는다. teleop에서 물체 정보가 의미를 갖는 곳은 **목표점을 표면으로 옮기는 2.2절**뿐이다. 물체 좌표계 목표는 이후 runtime 단계(그래프 subgoal)에서 $\tilde x_i$를 만드는 방식으로만 들어온다. 비용은 그대로 두고 목표 생성기만 교체한다.

---

## 3. MuJoCo에서 표면 질의를 얻는 방법

### 3.1 요구사항

1. 제어 주기당 한 손 5개(양손 10개) 점에 대해 $d_i$, $s_i$, $n_i$를 반환
2. **sim이 실제로 충돌 계산에 쓰는 기하와 일치**해야 함. 비용의 표면과 시뮬레이션의 접촉 표면이 다르면, 목표를 표면 안쪽 $\varepsilon$에 두어도 실제 접촉력이 의도와 달라짐
3. 실물로 옮길 때 backend만 바꿔 쓸 수 있어야 함

### 3.2 후보 비교

| 방식 | 내용 | sim 충돌 기하와 일치 | 실물 이전 | 구현량 |
|---|---|---|---|---|
| **A. `mj_geomDistance` + probe** | 손끝 목표 위치에 작은 probe sphere를 두고, MuJoCo의 geom 간 부호 거리 함수로 물체 geom까지의 거리와 최근접점을 얻음 | **정확히 일치** | 불가 (sim 전용) | 작음 |
| B. 해석적 primitive SDF | 물체 collision geom이 box/sphere/capsule 등일 때 닫힌 형태의 SDF | 일치 (primitive일 때만) | 가능 (pose만 있으면) | 중간 |
| C. 메쉬 격자 SDF | 물체 메쉬로 물체 좌표계 SDF 격자를 미리 계산, 런타임에 pose로 변환 후 삼선형 보간 | 메쉬가 오목하면 **불일치** (아래 참고) | **가능** (FoundationPose pose) | 중간 |
| D. MuJoCo SDF plugin | MuJoCo의 SDF 충돌 plugin | 일치 | 불가 | 큼, Python에서 점 질의 경로 불확실 |
| E. 점군 표면 | 렌더링 depth(sim) 또는 SAM3 depth(실물)에서 KD-tree 최근접점과 PCA 법선 | 근사 | **가능** (PATH 방식) | 중간 |

**C가 sim 충돌과 어긋나는 이유**: MuJoCo는 메쉬 geom의 충돌을 **볼록 껍질(convex hull)**로 계산한다. 오목한 메쉬(뚜껑이 열린 박스 등)로 격자 SDF를 만들면, 비용은 오목한 실제 형상을 보는데 시뮬레이션은 볼록 껍질로 접촉하게 된다.

### 3.3 권장

- **sim 개발·검증: A (`mj_geomDistance` + probe)를 기본 backend로 쓴다.** 충돌 기하와 정확히 일치하고, primitive와 메쉬(볼록 껍질)를 같은 코드로 처리하며, 추가 기하 코드가 필요 없다.
- **실물 이전 대비: C(메쉬 SDF + FoundationPose)와 E(SAM3 점군)를 같은 인터페이스로 구현**하고, sim에서 A를 정답으로 삼아 오차를 측정한다. 실물에서 어떤 backend를 쓸지는 이 측정 결과로 정한다.
- D는 쓰지 않는다.

### 3.4 Backend A 구현 상세

**MJCF 수정**
- probe body를 손가락 수만큼 추가한다 (양손 10개): `mocap="true"` body 안에 반지름 $r_p$ = 0.5 mm sphere geom
  - `contype="0" conaffinity="0"` (시뮬레이션 접촉에 영향 없음)
  - `group="5"`, `rgba="0 0 0 0"` (렌더링 제외)
- 물체 geom 목록: 물체 body에 속한 collision geom (`contype` 또는 `conaffinity`가 0이 아닌 것). 손, 로봇, 바닥, 테이블은 제외
- 물체 collision geom의 `margin`은 0으로 둔다 (margin이 있으면 접촉이 표면보다 먼저 생김)

**질의용 MjData 분리**
- 시뮬레이션 stepping에 쓰는 `MjData`와 별도로 **질의 전용 `MjData`**를 하나 둔다.
- 매 제어 주기: 시뮬레이션 데이터의 `qpos`를 복사하고, **probe 10개의 `mocap_pos`를 한꺼번에** $\tilde x_i$로 설정한 뒤 `mj_kinematics`를 **한 번만** 호출해 geom pose를 갱신한다.
- 이렇게 하면 probe 위치 설정이 시뮬레이션 상태를 건드리지 않는다.

**거리 질의** (Python, MuJoCo 3.16.0에서 동작 확인)

```python
def query(m, d_q, probe_gid, obj_gids, x_tilde, distmax, r_p, prev_normal):
    """d_q: mocap_pos 설정 + mj_kinematics가 끝난 질의 전용 MjData."""
    fromto = np.zeros(6)
    best_dist, best_s, best_g = distmax, None, -1
    for g in obj_gids:
        dist = mujoco.mj_geomDistance(m, d_q, probe_gid, g, distmax, fromto)
        if dist < best_dist:               # distmax 이상이면 fromto는 0으로 채워짐 → 반환값으로만 판정
            best_dist, best_s, best_g = dist, fromto[3:].copy(), g
    if best_s is None:                     # distmax 안에 물체 없음 → alpha_i = 0
        return SurfaceHit(d=distmax, s=None, n=None, obj_id=-1, valid=False)
    d = best_dist + r_p                    # probe 반지름 보정 → 점 x_tilde의 부호 거리
    s = best_s                             # 물체 표면 위 최근접점 (침투 시에도 표면 위)
    v = x_tilde - s
    nv = np.linalg.norm(v)
    n = prev_normal if nv < 1e-6 else np.sign(d) * v / nv   # 바깥 법선 (안쪽이면 부호 반전)
    return SurfaceHit(d=d, s=s, n=n, obj_id=best_g, valid=True)
```

- `distmax` = $d_{\text{off}}$ + 여유 (예: 3 cm). 그보다 먼 geom은 계산을 생략하고 `distmax`를 반환한다.
- 여러 geom의 최솟값을 취하면 합집합 물체(박스 벽 여러 개 + 뚜껑)의 부호 거리 $\phi_{A\cup B} = \min(\phi_A, \phi_B)$가 된다. 밖에서는 정확하고, 두 geom이 겹치는 안쪽에서는 근사이지만 $\varepsilon$ 수준 깊이에서는 문제없다.

**검증 결과 (MuJoCo 3.16.0, 단위 테스트로 그대로 옮길 것)**

시험 모델: box geom (half-size 5×3×2 cm), 볼록 메쉬(사면체), `contype=0 conaffinity=0`인 반지름 0.5 mm mocap sphere probe.

| # | 확인 항목 | 결과 |
|---|---|---|
| 1 | `contype=0` probe에도 거리 계산 | **동작함.** `mj_geomDistance`는 충돌 필터와 무관 |
| 2 | 바깥 점 거리, 해석적 box SDF와 비교 (면·모서리·꼭짓점 영역 3점) | 10.0000 / 11.1803 / 8.6603 mm, **해석해와 일치** (< 0.1 μm) |
| 3 | 침투 점 (깊이 3 mm, 10 mm) | −3.0000 / −10.0000 mm, **해석해와 일치**. `fromto[3:]`는 표면 위 최근접점, 위 코드로 구한 법선이 바깥 방향 |
| 4 | `distmax` 밖의 점 | 반환값 = `distmax`, **`fromto`는 0으로 채워짐** (이전 값 유지 아님) → 반환값으로 판정해야 함 |
| 5 | 메쉬 geom (볼록 껍질) | 사면체 경사면까지 17.3205 mm, 해석해 $0.03/\sqrt3$와 일치 |
| 6 | 연산 시간 (CPU) | `mj_kinematics` 1회 + 질의 2회 = 9.7 μs. probe마다 kinematics를 따로 불러도 10개에 약 0.1 ms. **kinematics를 1회로 묶으면 더 작음** |

결론: 제어 주기당 표면 질의 비용은 0.1 ms 이하로, 샘플링 최적화(ms 단위)에 비해 무시할 수 있다.

**남은 확인 항목 (실제 BOOST 모델에서)**
1. Revo2가 붙은 G1 모델에서 질의 전용 `MjData` 복사(`qpos`만) 후 결과가 시뮬레이션 상태와 같은지
2. 박스 뚜껑 hinge가 움직일 때 뚜껑 geom 질의가 따라오는지 (`qpos` 복사로 해결되어야 함)
3. 실제 마우스 메쉬에서 볼록 껍질과 시각 메쉬의 차이가 손끝 접촉 부위에서 몇 mm인지

### 3.5 Backend C 구현 상세 (실물 이전 대비, 2순위)

- 물체 메쉬(물체 좌표계, 미터 단위)가 watertight인지 확인하고, 아니면 수리
- 바운딩 박스 + 3 cm 여백 영역에 1–2 mm 해상도 SDF 격자를 미리 계산해 `.npz`로 저장 (격자 원점, 해상도, 메쉬 해시 포함)
- 질의: $x_o = T_o^{-1}x$ → 삼선형 보간으로 $\phi$, 보간 계수로 해석적 gradient → $n = \nabla\phi/\lVert\nabla\phi\rVert$, $s = x - \phi\,n$
- 관절이 있는 물체(박스 본체 + 뚜껑): 부품마다 격자를 따로 두고 각 부품 pose로 질의한 뒤 최솟값
- pose 출처: sim에서는 물체 body의 GT pose (`xpos`, `xquat`), 실물에서는 FoundationPose 추정값
- sim에서 볼록하지 않은 형상은 A와 차이가 날 수 있으므로, 비교는 볼록한 부품 위주로 하고 차이가 나는 영역을 기록

### 3.6 Backend E 구현 상세 (실물 이전 대비, 2순위)

- 입력: 물체 마스크가 적용된 depth 점군 (sim은 MuJoCo 렌더링 depth + segmentation, 실물은 SAM3 + depth)
- KD-tree 최근접점 $s$, k-NN PCA 법선을 카메라 쪽으로 정렬해 바깥 법선으로 사용
- 부호: $\operatorname{sign}(n^\top(x - s))$
- 손에 가려진 프레임은 표면을 갱신하지 않고 마지막 정상 표면 유지

### 3.7 인터페이스

```python
@dataclass
class SurfaceHit:
    d: float            # 부호 거리 [m]
    s: np.ndarray       # (3,) 표면 최근접점, world
    n: np.ndarray       # (3,) 바깥 단위 법선, world
    obj_id: int         # 가장 가까운 물체 ID
    valid: bool

class SurfaceQuery(Protocol):
    def update(self, state) -> None: ...          # 프레임마다 물체 pose·표면 갱신
    def query(self, x: np.ndarray) -> list[SurfaceHit]: ...   # x: (K, 3) world
```

구현체: `MjGeomSurface` (A), `MeshGridSDF` (C), `PointCloudSurface` (E). 설정 파일에서 선택한다.

---

## 4. 시뮬레이션 모델 설정 주의

| 항목 | 권장 |
|---|---|
| 박스 collision 기하 | 벽·바닥·뚜껑을 **box geom 여러 개**로 구성 (메쉬 하나로 두면 볼록 껍질 때문에 내부 공간이 막힘) |
| 마우스 collision 기하 | 메쉬 그대로 (거의 볼록이므로 볼록 껍질로 충분). 필요하면 볼록 분해 |
| 뚜껑 | 박스 본체에 hinge joint로 연결된 별도 body |
| 물체 geom margin | 0 |
| 손 actuator | 위치 제어. 목표 침투 $\varepsilon$와 손가락 관절 gain이 실제 파지력을 결정하므로, $\varepsilon$ sweep 실험에서 함께 기록 |
| 마찰 | 손가락–물체 마찰 계수를 실물에 가깝게 설정 (들어 올리기 검증에 영향) |

---

## 5. 구현 작업 목록 (순서대로)

| # | 작업 | 산출물 |
|---|---|---|
| 1 | 물체 MJCF 정리: 박스(box geom 조합 + 뚜껑 hinge), 마우스(메쉬), margin 0 | `assets/objects/*.xml` |
| 2 | probe mocap body 10개 추가 (MJCF 생성 시 자동 삽입) | MJCF 생성 코드 |
| 3 | `SurfaceQuery` 인터페이스 + `MjGeomSurface` (질의 전용 MjData 포함) | `surface/base.py`, `surface/mj_geom.py` |
| 4 | 3.4절 검증 표 1–5번을 단위 테스트로 이식 + 남은 확인 항목 3개 | `tests/test_surface_mj.py` |
| 5 | 목표 생성기: $\tilde x_i$, $\alpha_i$ hysteresis, $p^*_i$, $v^*_{1i}$ 혼합 | `targets/contact_projection.py` |
| 6 | 비용 수정: 손끝 목표 교체 또는 추가, 벡터 목표 교체. 샘플링 루프 내부 코드는 목표 배열만 바꿔 받도록 | 기존 비용 모듈 |
| 7 | 설정 항목 추가 (6절 표) | config yaml |
| 8 | 로깅 필드 추가 (7절) | recorder |
| 9 | 6절 검증 시나리오 실행 스크립트 | `scripts/eval_contact_cost.py` |
| 10 | (2순위) `MeshGridSDF`, `PointCloudSurface` 구현과 A 대비 오차 측정 | `surface/mesh_grid.py`, `surface/point_cloud.py` |

---

## 6. 검증 계획

### 6.1 회귀 테스트 (가장 먼저)

- 물체를 손에서 멀리 둔 상태(모든 $\alpha_i = 0$)에서 **수정 전 코드와 출력 $q$가 일치**해야 한다 (같은 난수 seed 기준, 수치 오차 범위 내).

### 6.2 시나리오

| 시나리오 | 확인 내용 |
|---|---|
| 자유 공간 pinch | 원본 SBR의 pinch 동작 유지 |
| 박스 옆면 접근 → 파지 | $\alpha_i$가 hysteresis대로 켜지는지, 손끝이 표면 안쪽 $\varepsilon$에 수렴하는지 |
| 파지 후 들어 올리기 | 미끄러짐 없이 유지 |
| 사람이 손가락을 과하게 오므린 입력 | 손가락이 물체를 관통하지 않고 표면에 머무는지 |
| 놓기 | 손가락이 $d_{\text{off}}$ 밖으로 나가면 원본 동작으로 부드럽게 복귀 |
| 마우스 집기 | 메쉬(볼록 껍질) 물체에서 동작 |

입력은 녹화된 사람 손 데이터 재생 또는 합성 궤적을 쓴다.

### 6.3 측정 지표

| 지표 | 기준 (초안) |
|---|---|
| 제어 주기 계산 시간 | 수정 전 대비 증가분 < 10% |
| 활성 손가락의 표면 오차 $\lvert d_i(q) + \varepsilon\rvert$ | 기록 (목표 수 mm 이내) |
| 비활성 손가락 침투 깊이 | 기록, 2 mm 이상이 잦으면 2.4절 hinge 활성화 |
| 파지 유지 성공률 | 시나리오별 기록 |
| jitter (손끝 이산 가속도 제곱) | 수정 전과 비교 |
| $\varepsilon$ sweep (1, 3, 5 mm) | 파지 성공률, 침투, 접촉력 |
| 중립 자세 정규화 on/off | 결과 차이 없으면 off |

---

## 7. 로깅 필드 (상위 문서 7.3절 (5) `mppi/` 그룹 변경)

| 필드 | shape (양손) | 설명 |
|---|---|---|
| `tip_human_world` | (N, 2, 5, 3) | $\tilde x_i$ |
| `tip_target_world` | (N, 2, 5, 3) | $p^*_i$ |
| `surf_dist` | (N, 2, 5) | $d_i$ |
| `surf_point` | (N, 2, 5, 3) | $s_i$ |
| `surf_normal` | (N, 2, 5, 3) | $n_i$ |
| `surf_obj` | (N, 2, 5) int | 가장 가까운 물체 ID |
| `alpha` | (N, 2, 5) | $\alpha_i$ |
| `active` | (N, 2, 5) bool | hysteresis 상태 $a_i$ |
| `surface_backend` | scalar | `mj_geom` / `mesh_grid` / `point_cloud` |
| `cost_terms` | (N, 2, 3) | 손끝 항, 벡터 항, 속도 항 값 |
| `solve_ms` | (N,) | 제어 주기 계산 시간 |

`tip_target_obj`는 삭제한다.

---

## 8. 파라미터 초기값

| 파라미터 | 초기값 | 근거 |
|---|---|---|
| $d_{\text{on}}$ | 1.0 cm | 원본 SBR의 project distance |
| $d_{\text{off}}$ | 1.5 cm | 원본 SBR의 escape distance |
| $\varepsilon$ | 3 mm | sweep으로 조정 |
| $w_i$ | 1.0 | 원본 기본 keyvector 가중치 |
| $w_{th}$ | 2.0 | 원본 primary keyvector 가중치 |
| $w_v$ | 0.1 | 원본 속도 가중치 |
| $r_p$ (probe 반지름) | 0.5 mm | |
| `distmax` | 3 cm | |
| $\lambda_{pen}$, $\varepsilon_{\max}$ | 0 (비활성), 6 mm | 필요 시 활성화 |

---

## 9. 이후 단계와의 연결

- **실물 teleop**: backend를 `PointCloudSurface`(SAM3) 또는 `MeshGridSDF`(FoundationPose)로 교체. 비용과 목표 생성기는 그대로.
- **runtime (손 expert, 그래프)**: $\tilde x_i$를 사람 손끝 대신 그래프 subgoal의 물체 좌표계 손끝 목표를 $\hat T_o$로 사상한 값으로 교체. 목표 생성기 인터페이스 하나로 teleop용과 runtime용을 바꿔 끼운다.
