# BOOST 개발 계획서

- 버전: v0.2 (2026-10-22) — 손 MPPI 비용을 표면 투영 목표 방식으로 변경 (상세: `BOOST_hand_mppi_contact_plan.md`)
- 목적: Codex 및 협업자 공유용 개발 기준 문서
- 목표 학회: IROS 2027 (논문 마감 2027-03-01)
- 읽는 순서: 0–1 (무엇을 만드는가) → 2–5 (구조, 환경, git 규칙) → 6 (단계별 작업) → **7 (teleop 데이터 명세, 가장 중요)**

---

## 0. 한 줄 요약

고정된 VLA(GR00T N1.7, 팔 담당) 위에서, **물체 기준 해석적 손목 보정**과 **contact-indexed state graph**를 이용해 **손 expert(diffusion policy)를 실물 rollout으로 점진적으로 fine-tuning**한다. 시연 ~30개만으로 시연에 없던 물체 pose에서도 포장 과제(마우스 박스)를 성공시키는 것이 목표다.

- 로봇: Unitree G1 29 DoF, 양손 BrainCo 6 DoF
- 과제: 포장 공정 (박스 잡기 → 뚜껑 열기 → 마우스 넣기 → 뚜껑 닫기 → 내려놓기). 물체는 고정(박스, 마우스), **물체 pose만 변화**
- 손 embodiment는 현재 BrainCo 6 DoF 하나. 단, 손에 의존하는 코드는 반드시 `HandSpec` 뒤에 둔다 (3.3절)

---

## 1. 방법 요약 (구현자가 알아야 할 최소한)

### 1.1 실행 시 손목·손 명령

$$
x_j = x^{V}_j \oplus \delta\xi^{\text{anc}}_j \oplus \delta\xi^{\theta}_j, \qquad j = 1..H
$$

| 항 | 의미 | 출처 | 학습 |
|---|---|---|---|
| $x^{V}_{1:H}$ | 팔(손목) chunk | 고정 VLA | 초기 1회 SFT 후 고정 |
| $\delta\xi^{\text{anc}}$ | 물체 기준 해석적 손목 보정 | 그래프 subgoal의 기준 좌표계 손목 pose를 **추정 물체 pose**로 world에 사상 | 없음 |
| $\delta\xi^{\theta}$ | 접촉 미세 보정 | 손 expert 출력 | 그래프 가치 가중 SFT |
| $q_{1:H}$ | 손 관절 목표 (6 DoF × 2) | 0차: GMM retrieval / 이후: 손 expert | 위와 동일 |

저수준에서는 손 MPPI가 손끝 목표(teleop: 사람 손끝을 물체 표면으로 투영한 점 / runtime: 그래프 subgoal에서 온 점)를 받아 실제 손 명령을 만든다 (6.1.3절).

### 1.2 상태 그래프

- 노드 $v = (s^{m}_v,\ s^{w}_v,\ \kappa_v,\ k_v,\ \phi_v,\ \tau_v)$, $n$ 프레임 간격 keyframe
  - $s^{m}_v = (b,\ q_R,\ q_L,\ \{{}^{o}x_w\}_{o},\ {}^{\text{box}}T_{\text{mouse}})$ — **매칭용** (간선, 최근접 탐색, 가중치 유사도)
  - $s^{w}_v = ({}^{\text{pel}}x_w,\ \{{}^{\text{pel}}T_o\}_o)$ — **매칭에 쓰지 않음** (기준 좌표계 통계, IK 도달성 확인)
  - $\kappa_v = (q_{1:H},\ {}^{a_k}x_{w,1:H})$ — payload
  - $k_v$: 접촉 세그먼트, $\phi_v \in [0,1]$: 세그먼트 내 위상, $\tau_v$: 정규화 시간
- 간선: 시연 순서 간선 + $\{d(s^m_u,s^m_v)<\mathrm{Tol},\ b_u=b_v,\ k_u=k_v,\ \tau_v\ge\tau_u,\ n(u)\ne n(v)\}$, 가중치 1
- 진행도 $c(v)$: goal까지 최단 hop 수. 실패 rollout 노드는 goal 간선 없이 편입
- 모드 분할: 세그먼트 $k$ (접촉 패턴) → 기준 좌표계 $a_k = \arg\min_f \operatorname{tr}(W\Sigma^{(k)}_f)$ → 파지 성분 GMM

### 1.3 학습

- 실행된 transition 가치: $\tilde Q(v_i) = -1 - c(v_{i+1})$
- GSR식 가중치 재분배: $w(u) = \sum_{v:u\in\mathcal N(v)} \dfrac{\exp[\beta_1 S(u,v)+\beta_2\tilde Q(u)]}{\sum_{u'}\exp[\beta_1 S(u',v)+\beta_2\tilde Q(u')]}$
- 손 expert loss: $\mathcal L(\theta) = \mathbb E\big[w(u)\,\lVert\epsilon-\epsilon_\theta(\cdot,k,c_u)\rVert^2\big]$

---

## 2. 시스템 및 프로세스 구조

의존성이 무거운 구성요소는 별도 프로세스(서비스)로 분리한다.

```
┌────────────────────────── G1 로봇 / 손 드라이버 ──────────────────────────┐
│  unitree g1 sdk (arm only topic), BrainCo hand driver, head RGB-D camera   │
└───────────────▲──────────────────────────────▲─────────────────────────────┘
                │ arm q_cmd / q_meas            │ hand q_cmd / q_meas
┌───────────────┴──────────────┐  ┌────────────┴──────────────────────────────┐
│ control 프로세스 (core env)   │  │ hand MPPI 프로세스 (core env)              │
│  arm IK (PATH 이식), chunk 실행│  │  SBR 비용 + 표면 투영 손끝 목표            │
└───────────────▲──────────────┘  └────────────▲──────────────────────────────┘
                │                               │
┌───────────────┴───────────────────────────────┴──────────────────────────────┐
│ teleop / runtime 프로세스 (core env)                                          │
│  XR 입력 → 목표 생성 → recorder                                               │
│  (이후 단계) graph localize, δξ^anc, retrieval / 손 expert                     │
└──────▲──────────────────────────────────────────────▲────────────────────────┘
       │ 물체 pose, 표면 SDF                             │ VLA chunk (Stage 3 이후)
┌──────┴──────────────────────────┐        ┌───────────┴───────────────────────┐
│ pose 서비스 (pose env)           │        │ vla 서비스 (vla env)               │
│  SAM3 마스크 + 표면 피팅 (PATH)   │        │  GR00T N1.7 추론 서버              │
│  FoundationPose 추정·tracking    │        │                                    │
└─────────────────────────────────┘        └────────────────────────────────────┘
```

- 프로세스 간 통신은 **PATH가 현재 쓰는 전송 계층을 그대로 재사용**한다 (방식은 PATH 코드 기준으로 확인 후 이 문서에 기록).
- 모든 메시지는 단일 시계 기준 타임스탬프를 포함한다 (7.1절).

---

## 3. 레포 전략과 구조

### 3.1 PATH와의 관계

- PATH 정리본에 `path-icra2027` tag를 찍고 **동결**한다 (ICRA 재현용).
- BOOST의 teleop은 **PATH 정리본을 출발점으로 가져와 BOOST 안에서 독립적으로 수정**한다. FoundationPose 통합, 의도 추론 제거, MPPI 비용 변경 등 수정 폭이 크므로 의존성으로 두지 않는다.
- 가져온 PATH commit hash를 `boost/teleop/UPSTREAM.md`에 기록한다. 히스토리를 보존하고 싶으면 `git subtree`로 가져와도 된다.

### 3.2 외부 의존성

| 대상 | 관리 | 비고 |
|---|---|---|
| Isaac-GR00T | `third_party/Isaac-GR00T` submodule, **본인 fork의 특정 commit** | 코드 복사 금지. 수정은 fork에서 |
| FoundationPose | `third_party/FoundationPose` submodule, 특정 commit | CUDA 확장 빌드는 pixi task로 |

BOOST 안에 두는 GR00T 관련 코드는 **데이터 변환기, modality 설정, 추론 클라이언트**뿐이다.

### 3.3 디렉터리 구조

```
boost/
├── README.md
├── pixi.toml / pixi.lock          # 환경 3개: core, vla, pose
├── configs/
│   ├── hands/brainco_6dof.yaml        # DoF, 관절 한계, 손끝 정의, 접촉 판정 임계값·hysteresis
│   ├── robots/g1_29dof.yaml           # 카메라 extrinsic 체인, 손 장착 offset
│   ├── tasks/packaging.yaml           # 물체 목록, 메쉬 경로, canonical frame, 대칭군, 언어 지시문
│   ├── teleop/default.yaml            # MPPI 가중치, target_mode, 녹화 주기
│   ├── graph/default.yaml             # n, Tol, α, K, β1, β2
│   └── expert/default.yaml
├── boost/
│   ├── common/        # 타입(Frame, Episode, ObjectPose, Chunk, NodeState), SE(3) 유틸, 시계
│   ├── embodiment/    # HandSpec 인터페이스 + brainco_6dof 구현
│   ├── io/            # 로봇·서비스 통신 (PATH 전송 계층)
│   ├── perception/    # pose 서비스 클라이언트, 물체 레지스트리(메쉬, canonical frame, 대칭)
│   ├── control/
│   │   ├── arm/       # 팔 IK (PATH 이식), chunk 실행기
│   │   └── hand_mppi/ # SBR(Smooth Operator) 구조 + surface/(SurfaceQuery backends) + targets/(목표 생성기)
│   ├── teleop/        # XR teleop, 목표 생성, UPSTREAM.md, legacy/(의도 추론 격리)
│   ├── sim/           # MuJoCo 환경 (G1 + BrainCo + 박스·마우스), 렌더 카메라
│   ├── data/
│   │   ├── schema.py          # 표준 에피소드 형식 (단일 원본)
│   │   ├── recorder.py
│   │   ├── reader.py          # 7.5절 추적표의 모든 상태를 재구성하는 API
│   │   ├── manifest.py
│   │   └── to_lerobot.py      # → GR00T 학습 형식
│   ├── graph/         # features, segment, anchor, build, value, localize, weights, diagnostics
│   ├── policy/        # anchor_target, retrieval, expert/, runtime
│   ├── vla/           # GR00T 클라이언트, modality 설정
│   └── eval/          # pose 격자 샘플러, 평가 프로토콜, 사람 시간 기록
├── services/
│   ├── pose_service/  # SAM3 + 표면 피팅 + FoundationPose 서버
│   └── vla_service/   # GR00T 추론 서버 래퍼
├── scripts/           # teleop_sim.py, teleop_real.py, build_graph.py, run_round.py, train_expert.py ...
├── datasets/          # 데이터셋 manifest(yaml)만 커밋. 실제 데이터는 외부 저장소
├── third_party/       # Isaac-GR00T, FoundationPose (submodule)
└── tests/             # 짧은 녹화 fixture 기반 오프라인 테스트
```

### 3.4 HandSpec (손 embodiment 격리)

```python
class HandSpec(Protocol):
    name: str                      # "brainco_6dof"
    dof: int                       # 6 (손 하나 기준)
    joint_limits: np.ndarray       # (dof, 2), raw 단위
    def normalize(self, q_raw) -> np.ndarray: ...             # → [0, 1]^dof
    def fingertips(self, q, T_wrist) -> np.ndarray: ...       # (5, 3) world, FK
    def keypoints(self, q, T_wrist) -> np.ndarray: ...        # (K, 3) 손바닥·관절 포함 keypoint
    def contact(self, q_cmd, q_meas, aux=None) -> np.ndarray: ...  # (5,) 손가락별 접촉 비트
    def retarget(self, human_kp) -> np.ndarray: ...           # teleop용 초기값
```

- `graph/`, `control/hand_mppi/`, `policy/expert/`는 `HandSpec`만 참조한다.
- 그래프 artifact와 expert 체크포인트에 `hand_spec.name`과 설정 해시를 기록하고, 로드 시 불일치하면 오류를 낸다.

---

## 4. 환경 (pixi)

환경은 **core, vla, pose** 세 개이며, 하나의 `pixi.toml`에 feature로 정의하고 `pixi.lock` 하나로 고정한다. pixi는 환경마다 따로 의존성을 풀기 때문에 서로 다른 torch/CUDA 버전이 공존할 수 있다.

```toml
[workspace]
name = "boost"
channels = ["conda-forge", "nvidia"]
platforms = ["linux-64"]          # linux-aarch64(Thor)는 6.x절 TBD 참고

[system-requirements]
cuda = "12"

[feature.core.dependencies]
python = "3.11.*"
numpy = "*"
# torch, mujoco, zarr, hydra-core, scipy ...

[feature.core.pypi-dependencies]
boost = { path = ".", editable = true }

[feature.vla.dependencies]
python = "3.10.*"                 # GR00T 요구 버전에 맞춤 (확인 필요)

[feature.vla.pypi-dependencies]
gr00t = { path = "third_party/Isaac-GR00T", editable = true }

[feature.pose.dependencies]
python = "3.10.*"                 # FoundationPose 요구 버전에 맞춤 (확인 필요)

[feature.pose.tasks]
build-foundationpose = "bash services/pose_service/build_ext.sh"

[environments]
core = ["core"]
vla  = ["vla"]
pose = ["pose"]
```

- 실행 예: `pixi run -e core python scripts/teleop_sim.py`, `pixi run -e pose python -m services.pose_service`
- `pixi.lock`은 반드시 커밋한다.
- **Thor(linux-aarch64) 배포 위험**: Jetson 계열의 GPU용 torch는 JetPack 전용 빌드라 conda-forge/PyPI로 해결되지 않을 수 있다. Thor에서 돌릴 서비스(pose, vla)는 NVIDIA 컨테이너 사용을 대안으로 둔다. 현재는 x86 + RTX 3080 기준으로 개발한다.

---

## 5. Git 운영 규칙

| 브랜치 / tag | 용도 | 규칙 |
|---|---|---|
| `main` | README와 문서만 | 코드 push 금지 |
| `develop` | **기본 작업 브랜치** (GitHub default branch로 설정) | 항상 sim에서 동작하는 상태 유지 |
| `feat/<name>` | 선택적 단기 작업 브랜치 | develop으로 merge 후 삭제 |
| tag `<feature>-v<maj>.<min>` | 기능 완성 시점 표시 (annotated tag) | 예: `teleop-v0.1` |
| `snap/<feature>-v<maj>.<min>` | 기능 완성 시점 스냅샷 브랜치 | **수정 금지**. 긴급 수정은 `snap/<...>-p1` 새 브랜치 |

- 기능이 완성되면 같은 commit에 **tag와 스냅샷 브랜치를 함께** 만들고, 작업은 develop에서 계속한다.
- 브랜치와 tag를 **똑같은 이름으로 만들면 git ref가 모호해지므로**(ambiguous refname) 스냅샷 브랜치는 반드시 `snap/` 접두사를 붙인다.
- 커밋 금지: 데이터, 체크포인트, 메쉬 원본 대용량 파일. `.gitignore`에 `data/`, `checkpoints/`, `outputs/`.
- 모든 학습·rollout 실행은 `(BOOST commit, 데이터셋 ID, 설정 해시)`를 기록한다.

계획된 tag:

| tag | 완료 조건 |
|---|---|
| `teleop-sim-v0.1` | 6.1.5절 MuJoCo 검증 통과 |
| `teleop-v0.1` | 실물 teleop + 시연 30개 수집 완료 |
| `graph-v0.1` | 6.2절 오프라인 진단 통과 |
| `vla-v0.1` | 팔 SFT + 추론 서버 + pilot 완료 |
| `round0-v0.1` | 0차 수집 완료 |
| `expert-v0.1` | 손 expert BC + guidance |
| `rounds-v0.1` | 1–3차 라운드 완료 |

---

## 6. 단계별 계획

| 단계 | 기간 (목표) | 내용 | tag |
|---|---|---|---|
| **1. teleop** | 10/20 – 11/30 | PATH 이식, FoundationPose 통합, MPPI 비용 변경, recorder, **MuJoCo 검증 → 실물**, 시연 30개 | `teleop-sim-v0.1`, `teleop-v0.1` |
| 2. graph | 11/24 – 12/7 | 오프라인 그래프, 세그먼트, 기준 좌표계, 진단 (로봇 불필요, 1단계와 병행) | `graph-v0.1` |
| 3. VLA | 11/24 – 12/7 | LeRobot 변환, 팔 SFT, 추론 서버, 고정 VLA pilot | `vla-v0.1` |
| 4. round 0 | 12/8 – 12/28 | runtime: 고정 VLA + $\delta\xi^{\text{anc}}$ + retrieval + MPPI | `round0-v0.1` |
| 5. expert | 12/22 – 1/18 | diffusion BC → guidance → 1–3차 라운드 | `expert-v0.1`, `rounds-v0.1` |
| 6. eval | 1/19 – 2/15 | 베이스라인, ablation, 사람 시간 비교 | — |
| 7. writing | 2/16 – 3/1 | 논문, 영상 | — |

### 6.1 Stage 1: teleop (현재 단계)

#### 6.1.1 PATH 이식과 정리

- PATH 정리본을 `boost/teleop/`, `boost/control/arm/`, `boost/io/`, `services/pose_service/`로 나눠 가져온다.
- **의도 추론(belief inference)**: 삭제하지 않고 `boost/teleop/legacy/`로 격리하고 설정 플래그 `teleop.use_intent_inference=false`를 기본으로 둔다. 손 MPPI의 표면 투영 목표가 실물에서 안정화되면 삭제 여부를 결정한다.
- WBC는 별도 레포를 전제로 하고, g1 sdk로 보내는 토픽은 arm only로 유지한다 (PATH 정리 기준 유지).

#### 6.1.2 Perception: SAM3 표면 + FoundationPose

두 출력의 역할을 명확히 나눈다.

| 출력 | 생성 | 좌표계 | 사용처 |
|---|---|---|---|
| 물체 표면 (point cloud / SDF) | SAM3 마스크 + depth 표면 피팅 (**PATH 기존 그대로**) | world(pelvis) | 손 MPPI 목표 생성(손끝을 표면으로 투영) |
| 물체 6D pose $\hat T_o$ | FoundationPose (**신규**), SAM3 마스크로 초기화 | canonical 물체 좌표계 → pelvis | 그래프 상태, 해석적 손목 보정. **teleop 제어 루프에는 쓰지 않고 녹화만** |

- FoundationPose 메쉬: 박스는 실측 치수로 CAD, 마우스는 스캔 또는 SAM 3D로 1회 생성. canonical frame 정의와 대칭군을 `configs/tasks/packaging.yaml`에 기록.
- 실행 방식: **pose 추정(estimation)은 에피소드 시작과 tracking 실패 시에만**, 이후 프레임은 **tracking**. 물체당 tracker 하나 (박스, 마우스).
- 손에 가려지는 구간 대비: tracking 신뢰도가 낮아지면 SAM3 마스크로 재초기화. 물체가 손에 쥐어진 뒤에는 "파지 순간의 손-물체 상대 pose 유지" 가정으로 손 FK에서 물체 pose를 전파하는 fallback을 둔다.
- 메쉬가 생기므로, MPPI SDF를 "메쉬를 $\hat T_o$로 배치한 SDF"로도 만들 수 있다. SAM3 표면과 교차 검증용으로 둘 다 계산하되, 기본값은 PATH 방식(SAM3 표면) 유지.
- 속도는 8절 참고.

#### 6.1.3 손 MPPI 비용: 표면 투영 손끝 목표

**상세 명세는 `BOOST_hand_mppi_contact_plan.md`가 기준이다.** 요약:

- SBR(Smooth Operator) 비용의 **형태는 그대로** 두고, 손끝·엄지벡터 **목표값만** 바꾼다. 샘플당 연산은 원본과 같은 차수.
- 목표 생성(제어 주기당 1회): 사람 손끝 $\tilde x_i$의 표면 부호 거리 $d_i$, 최근접점 $s_i$, 법선 $n_i$를 구하고, hysteresis 활성화 $\alpha_i$로 $p^*_i = \tilde x_i - \alpha_i(d_i+\varepsilon)n_i$ (표면 안쪽 $\varepsilon$, 파지력 설정값).
- 샘플당 비용: $J(q) = \sum_i w_i\rho_\beta({}^{w}p_i(q) - {}^{w}p^*_i) + w_{th}\sum_{i\ge2}\rho_\beta(v_{1i}(q) - v^*_{1i}) + w_v\lVert q-q_{\text{prev}}\rVert^2$, 한 시점 문제($H_m=1$).
- 이전 초안의 물체 좌표계 keypoint 항, squeeze 항, 샘플별 SDF 침투 항은 **삭제**. teleop에서 물체 좌표계 정합은 강체 변환 불변성 때문에 의미가 없다 (상세 문서 2.5절). 침투가 관찰되면 접평면 hinge만 선택적으로 켠다.
- 표면 질의는 `SurfaceQuery` 인터페이스 뒤에 둔다. **sim: `mj_geomDistance` + probe (MuJoCo 충돌 기하와 정확히 일치, 실측 검증 완료)**, 실물: SAM3 점군 또는 메쉬 SDF + FoundationPose.
- 이전의 목표 모드 A/B 구분과 PATH 접촉점 추론 경로는 **이 투영 방식 하나로 통합**한다.
- **사람 원본 keypoint, 투영 목표, MPPI 출력 $q$, 실행된 $q$를 모두 녹화**한다 (7.3절).

#### 6.1.4 Recorder

- 7절 명세를 그대로 구현한다. 녹화는 별도 스레드/프로세스로, 제어 루프를 막지 않는다.
- 녹화 직후 7.6절 QA 검사를 자동 실행하고, 실패하면 경고한다.

#### 6.1.5 MuJoCo 검증 (첫 번째 마일스톤, `teleop-sim-v0.1`)

| 항목 | 방법 | 통과 기준 (초안, 조정 가능) |
|---|---|---|
| 모델 로드 | G1 + BrainCo 6 DoF MJCF, 박스·마우스 메쉬 | `HandSpec.fingertips`가 MuJoCo 사이트와 일치 (< 1 mm) |
| 회귀 | 물체가 멀 때(모든 $\alpha_i=0$) 수정 전 SBR과 출력 $q$ 일치 | 같은 seed에서 수치 오차 이내 |
| MPPI 실시간성 | 수정 전 대비 제어 주기 계산 시간 | 증가분 < 10% (기존 주기 유지) |
| 표면 질의 | `mj_geomDistance` backend 단위 테스트 (상세 문서 3.4절) | 해석해와 일치, 질의 < 0.1 ms/주기 |
| 표면 추종 | 녹화 재생·합성 궤적으로 박스 옆면 파지 | 활성 손가락 $\lvert d_i(q)+\varepsilon\rvert$ 기록 |
| 파지 유지 | 물체를 쥐고 들어 올리기 | 미끄러짐·과침투 없이 유지 |
| $\varepsilon$ sweep | 1, 3, 5 mm | 파지 성공률·침투·접촉력으로 기본값 결정 |
| 표면 노이즈 강건성 | 표면점·법선에 노이즈 주입 (실물 점군 backend 대비) | 노이즈 크기별 성공률 곡선 확보 |
| (선택) FoundationPose | MuJoCo 렌더 RGB-D로 pose 추정, GT와 비교 | 오차 통계 확보 |
| **recorder 추적성** | sim 에피소드 녹화 → `data/reader.py`로 7.5절 모든 상태 재구성 | **단위 테스트 통과** (가장 중요) |

#### 6.1.6 실물 이전과 수집

- 실물 teleop 확인 후 **pose 정확도를 fiducial로 1회 측정** (fiducial은 정책 카메라에 보이지 않는 위치에 부착하거나 평가 전용 에피소드에서만 사용).
- 시연 30개: 물체 pose를 학습 영역 $\mathcal R_{\text{train}}$ 안에서 분산 배치. 각 물체 pose는 7.4절 메타데이터에 기록.
- pilot용 **학습 영역 밖 pose 목록**도 이때 정의해 둔다.

### 6.2 Stage 2: graph (오프라인, 로봇 불필요)

- `graph/features.py`: reader가 준 프레임에서 $s^m$, $s^w$ 계산 (`HandSpec` 사용)
- `graph/segment.py`: 접촉 비트 → hysteresis·최소 지속 시간 → 세그먼트 $k$, 위상 $\phi$
- `graph/anchor.py`: 대칭 정규화 → 세그먼트 끝 손목 pose의 좌표계별 분산 → $a_k$ → 파지 성분 GMM
- `graph/build.py`, `value.py`, `localize.py`, `weights.py`
- `graph/diagnostics.py`: leave-one-demo-out으로 (1) 커버리지, (2) $c$의 단조성, (3) 해석적 손목 목표 오차
- 완료 기준: 세 진단 수치 확보, 그래프 artifact에 데이터셋 ID·손 이름·설정 해시 기록

### 6.3 Stage 3: VLA (vla env)

- `data/to_lerobot.py`로 변환, GR00T NEW_EMBODIMENT modality 설정 (팔만)
- 팔 SFT (LLM 동결, projector·DiT 파인튜닝) → 고정
- vla 서비스 기동
- **pilot**: 학습 영역 밖 pose에서 고정 VLA rollout, 실패를 reach / 접촉 전환 / 접촉 정렬로 분류, 파지 직전 손목 오차 vs 손가락 오차 측정

### 6.4 Stage 4: round 0

- `policy/anchor_target.py`: $\delta\xi^{\text{anc}}$ (chunk 내 시간 배분 $\alpha(j)$, 신뢰도 gating)
- `policy/retrieval.py`: 손 관절만 GMM retrieval ($w_v \propto \exp(-d/T + \beta_2\tilde Q)$), chunk 단위 성분 고정
- `policy/runtime.py`: VLA chunk + $\delta\xi^{\text{anc}}$ + retrieval → arm IK + 손 MPPI
- 성공·실패 rollout 모두 녹화 (7.7절 추가 필드), 그래프 갱신, 가중치 계산
- 이 상태가 곧 **MimicGen식 베이스라인** ($\delta\xi^{\theta}=0$)

### 6.5 Stage 5: hand expert

- 입력 $c_t = (\psi(I_t), s^m_t, s_t, \Delta s_t, m_t\odot x^V_{t:t+H}, \delta\xi^{\text{anc}}_{t:t+H})$, 출력 $\kappa = (\delta\xi^{\theta}_{1:H}, q_{1:H})$
- 가중 diffusion BC (teleop + 0차) → guidance ($D_V = \sum\lVert\delta\xi^\theta\rVert^2$, $D_o$ = 기준 좌표계 Mahalanobis) → **추론 지연 측정** → 1–3차 라운드
- teleop 데이터의 $\delta\xi^{\text{anc}}$는 leave-one-demo-out 통계로 계산 (라벨 누설 방지)

### 6.6 Stage 6: eval

- 베이스라인: VLA 팔+손 공동 SFT, 앵커만(MimicGen식), VLA 없이 앵커 재생, 성공 필터 vs 그래프 가중, 시연 수 곡선(10/20/30)
- 사람 시간(teleop 시간, 라벨링, reset)을 모든 조건에서 기록

---

## 7. Teleop 데이터 명세 ★

**원칙**
1. 녹화 형식은 하나(표준 에피소드)이고, 그래프·학습·평가에 쓰는 모든 것은 이 원본에서 재생성 가능해야 한다.
2. **나중에 다시 수집할 수 없는 정보는 전부 녹화한다** (사람 원본 입력, 명령과 실행값 모두, 원시 센서).
3. 파생값(세그먼트, 노드, 가중치 등)은 녹화하지 않는다. 계산 코드와 설정으로 재현한다.

### 7.1 공통 규약

| 항목 | 규약 |
|---|---|
| 기준 좌표계 | `pel`: G1 pelvis frame (모든 world 값의 기준). `cam`: 머리 카메라 광학 frame. `o_box`, `o_mouse`: 메쉬의 canonical frame. `wR`, `wL`: 손 장착 플랜지(손목) frame |
| pose 표현 | 위치 (3,) [m] + quaternion (4,) **wxyz** (MuJoCo 규약). 저장 시 float64 |
| 단위 | SI (m, rad, s, N). 손 관절은 raw 단위와 `[0,1]` 정규화 값 둘 다 저장 |
| 시간 | 모든 스트림은 **단일 단조 시계**(로봇 PC 기준) 타임스탬프 `t` [s, float64]를 가짐. 카메라 시계와의 offset은 메타데이터에 기록 |
| 좌우 | 모든 손 관련 필드는 `R`, `L` 두 벌 |
| 결측 | 값이 없으면 NaN + `valid` 플래그 |

### 7.2 저장 형식

에피소드 하나 = 디렉터리 하나.

```
<dataset_root>/<dataset_id>/<episode_id>/
├── meta.json            # 7.4절
├── streams.zarr/        # 7.3절 각 그룹 (그룹마다 자체 타임스탬프 t)
└── video/
    ├── head_rgb.mp4     # 무손실 또는 고품질 (프레임 타임스탬프는 zarr에)
    └── head_depth.zarr  # uint16 [mm], 정렬된 depth
```

- 이미지는 원본 해상도로 저장한다. GR00T 입력 크기 변환은 `to_lerobot.py`에서 한다.
- 그룹별 주기가 다르므로 **재샘플링하지 않고 원본 주기 그대로** 저장한다. 정렬은 reader가 타임스탬프로 한다.

### 7.3 스트림 목록

#### (1) 카메라 `camera/` — 30 Hz (카메라 기본 주기)

| 필드 | shape | 설명 | 용도 |
|---|---|---|---|
| `t` | (N,) | 프레임 타임스탬프 | 정렬 |
| `rgb_index` | (N,) | `head_rgb.mp4` 프레임 번호 | VLA·expert 이미지 입력 $\psi(I_t)$ |
| `depth_index` | (N,) | depth 프레임 번호 | 표면 재피팅, pose 재추정 |
| `T_pel_cam` | (N, 7) | 카메라 pose (머리·허리 관절 FK + 보정값) | 모든 관측의 pelvis 변환 |

#### (2) 물체 인식 `perception/` — 카메라 주기

| 필드 | shape | 설명 | 용도 |
|---|---|---|---|
| `t` | (N,) | | |
| `obj/<name>/T_pel_obj` | (N, 7) | FoundationPose 6D pose (pelvis 기준) | $^{\text{pel}}T_o$, $^{o}x_w$, $^{\text{box}}T_{\text{mouse}}$, 해석적 보정 |
| `obj/<name>/score` | (N,) | 추정·tracking 신뢰도 | gating, QA |
| `obj/<name>/mode` | (N,) int | 0: estimation, 1: tracking, 2: 손 FK 전파 fallback | 신뢰도 해석 |
| `obj/<name>/reinit` | (N,) bool | 재초기화 발생 여부 | QA |
| `obj/<name>/mask_index` | (N,) | SAM3 마스크 (별도 저장, 압축) | 재처리 |
| `obj/<name>/surface` | 가변 | SAM3 표면 피팅 파라미터 또는 점군 (PATH 형식) | MPPI SDF 재현 |

#### (3) 팔 `arm/` — 제어 주기 (PATH 기준, 예: 100 Hz 이상)

| 필드 | shape | 설명 | 용도 |
|---|---|---|---|
| `t` | (N,) | | |
| `q_meas` | (N, n_arm) | 실측 관절각 (팔 7×2 + 허리 등 FK 체인에 필요한 관절 전부) | $^{\text{pel}}x_w$ FK |
| `dq_meas` | (N, n_arm) | 실측 관절 속도 | |
| `tau_meas` | (N, n_arm) | 관절 토크 (가능하면) | 접촉 보조 |
| `q_cmd` | (N, n_arm) | IK 출력 명령 | VLA 행동 라벨 후보 |
| `T_pel_wR_target`, `T_pel_wL_target` | (N, 7) | IK에 들어간 손목 목표 | **VLA 팔 행동 라벨 (EE 공간)** |
| `T_pel_wR`, `T_pel_wL` | (N, 7) | 실측 손목 pose (FK) | $^{\text{pel}}x_w$, $^{o}x_w$ |
| `ik_residual` | (N, 2) | IK 잔차 | QA |

#### (4) 손 `hand/` — 손 MPPI 제어 주기

| 필드 | shape | 설명 | 용도 |
|---|---|---|---|
| `t` | (N,) | | |
| `q_meas_raw`, `q_meas` | (N, 2, 6) | 실측 관절 (raw, `[0,1]`) | $q_R, q_L$ ($s^m$) |
| `q_cmd` | (N, 2, 6) | **MPPI 출력** (실제로 손에 보낸 명령) | 손 행동 라벨 후보 1, 접촉 판정 |
| `q_goal` | (N, 2, 6) | MPPI 이전의 retarget 목표 | 손 행동 라벨 후보 2 |
| `current` / `force` | (N, 2, 6) | 모터 전류 또는 힘 센서 (BrainCo 제공 범위 확인) | 접촉 판정 보조 |
| `contact_raw` | (N, 2, 5) | `HandSpec.contact` 즉시값 (hysteresis 전) | 세그먼트 계산 원천 |
| `contact_cont` | (N, 2, 5) | 연속 접촉 강도 (추종 오차 크기 등) | expert 입력 $s_t$ |
| `tip_world` | (N, 2, 5, 3) | FK 손끝 위치 (pelvis) | 검증용 (재계산 가능하지만 QA 편의상 저장) |

#### (5) 손 MPPI 내부 `mppi/` — 손 MPPI 제어 주기

| 필드 | shape | 설명 | 용도 |
|---|---|---|---|
| `t` | (N,) | | |
| `tip_human_world` | (N, 2, 5, 3) | 사람 손끝 $\tilde x_i$ (pelvis) | 투영 전 목표, 재처리 |
| `tip_target_world` | (N, 2, 5, 3) | 투영 목표 $p^*_i$ (pelvis) | 라벨·분석 |
| `surf_dist` | (N, 2, 5) | $d_i$ | 접촉 판정 보조, 세그먼트 검증 |
| `surf_point`, `surf_normal` | (N, 2, 5, 3) | $s_i$, $n_i$ | 재처리 |
| `surf_obj` | (N, 2, 5) int | 가장 가까운 물체 ID | **기준 물체 추적** (그래프 anchor 후보) |
| `alpha`, `active` | (N, 2, 5) | $\alpha_i$, hysteresis 상태 $a_i$ | 투영 강도 |
| `surface_backend` | scalar | `mj_geom` / `mesh_grid` / `point_cloud` | |
| `cost_terms` | (N, 2, 3) | 손끝·벡터·속도 항 값 | 진단 |
| `solve_ms` | (N,) | 제어 주기 계산 시간 | 실시간성 QA |

#### (6) 사람 입력 `xr/` — XR 기본 주기

| 필드 | shape | 설명 | 용도 |
|---|---|---|---|
| `t` | (N,) | | |
| `T_head` | (N, 7) | 사람 머리 pose | 재처리 |
| `T_wrist_R`, `T_wrist_L` | (N, 7) | 사람 손목 pose (로봇 좌표로 변환 전 원본 + 변환 후 둘 다) | 재retarget |
| `hand_kp_R`, `hand_kp_L` | (N, K_xr, 7) | 사람 손 관절 keypoint 원본 (OpenXR 손 관절 전체) | **재retarget, 투영 목표 재계산** |
| `buttons` | (N, ·) | 버튼, 트리거 상태 | 시작·종료·라벨 이벤트 |

#### (7) 이벤트 `events/`

| 필드 | 설명 |
|---|---|
| `start`, `end` | 에피소드 시작·종료 시각 |
| `operator_marks` | 운영자가 누른 표시 (선택, 예: "파지 완료") |
| `intervention` | 사람 개입 발생 시각 (teleop에서는 없음, 라운드 rollout에서 사용) |

### 7.4 메타데이터 `meta.json`

| 키 | 예시 / 설명 |
|---|---|
| `episode_id`, `dataset_id` | 불변 ID |
| `task` | `packaging` |
| `instruction` | VLA용 언어 지시문 (고정 문장) |
| `hand_spec` | `{name: brainco_6dof, config_hash: ...}` |
| `robot` | `g1_29dof`, 펌웨어 버전 |
| `objects` | 물체별 `{name, mesh_id, mesh_hash, canonical_frame, symmetry}` |
| `init_object_pose` | 에피소드 시작 시 물체별 `T_pel_obj` (pose 격자 관리용) |
| `region` | `train` / `shift_L1` / `shift_L2` ... |
| `calibration` | 카메라 intrinsic, `T_head_link_cam`, 손 장착 offset `T_flange_hand`, 카메라-로봇 시계 offset, 보정 일시 |
| `success` | 사람이 확인한 성공 여부 (bool) |
| `failure_note` | 실패 시 짧은 메모 (선택) |
| `source` | `teleop` / `round0` / `round1` ... |
| `operator` | 운영자 ID |
| `human_time` | `{teleop_s, label_s, reset_s}` (사람 시간 비교 실험용) |
| `code` | `{boost_commit, path_upstream_commit, foundationpose_commit, gr00t_commit}` |
| `configs` | 사용한 설정 파일 해시 (teleop, hand, task) |
| `rates` | 그룹별 실제 평균 주기 |

### 7.5 그래프·학습 상태 ↔ 원천 필드 추적표

**이 표의 모든 행이 `data/reader.py`로 재구성되어야 한다 (6.1.5절 단위 테스트).**

| 대상 | 기호 | 계산식 | 원천 필드 |
|---|---|---|---|
| 접촉 비트 | $b$ | `contact_raw`에 hysteresis·최소 지속 시간 적용 | `hand/contact_raw` (+ `q_cmd`, `q_meas`, `current`로 재계산 가능) |
| 손 관절 | $q_R, q_L$ | 그대로 | `hand/q_meas` |
| 손목 pose (pelvis) | $^{\text{pel}}x_w$ | 그대로 또는 FK | `arm/T_pel_w*`, `arm/q_meas` |
| 물체 pose (pelvis) | $^{\text{pel}}T_o$ | 그대로 | `perception/obj/*/T_pel_obj` |
| 물체 좌표계 손목 | $^{o}x_w$ | $(^{\text{pel}}T_o)^{-1}\,{}^{\text{pel}}x_w$ (대칭 정규화 포함) | 위 두 행 + `meta.objects.symmetry` |
| 두 물체 상대 pose | $^{\text{box}}T_{\text{mouse}}$ | $(^{\text{pel}}T_{\text{box}})^{-1}\,{}^{\text{pel}}T_{\text{mouse}}$ | `perception/obj/*` |
| 매칭 상태 | $s^m$ | $(b, q_R, q_L, \{^{o}x_w\}, {}^{\text{box}}T_{\text{mouse}})$ | 위 행들 |
| world 상태 | $s^w$ | $(^{\text{pel}}x_w, \{^{\text{pel}}T_o\})$ | 위 행들 |
| 세그먼트, 위상 | $k, \phi$ | $b$의 변화 지점으로 분할, 세그먼트 내 정규화 시간 | $b$, 타임스탬프 |
| 정규화 시간 | $\tau$ | $(t - t_{\text{start}})/(t_{\text{end}} - t_{\text{start}})$ | `events/start,end` |
| 손 행동 chunk | $q_{1:H}$ | keyframe부터 $H$ 스텝 | `hand/q_cmd` 또는 `hand/q_goal` (설정으로 선택) |
| 기준 좌표계 손목 chunk | $^{a_k}x_{w,1:H}$ | $a_k$ 결정 후 변환 | 손목 pose, 물체 pose |
| 손끝 keypoint | $^{o}p_i$ | `HandSpec.fingertips` 후 물체 좌표계 변환 | `hand/q_meas`, 손목 pose, 물체 pose |
| 연속 접촉 벡터 | $s_t, \Delta s_t$ | 그대로, 차분 | `hand/contact_cont` |
| 이미지 임베딩 | $\psi(I_t)$ | 인코더로 계산 (저장 안 함) | `camera/rgb_index` |
| 에피소드 성공 | — | 그대로 | `meta.success` |
| 진행도 | $c(v)$ | 그래프 위 최단 hop 수 | 그래프 + `meta.success` |
| VLA 학습 데이터 | — | LeRobot 변환 | `camera/*`, `arm/q_meas`, `arm/T_pel_w*_target`, `meta.instruction` |
| VLA chunk (teleop 데이터에 대해) | $x^V$ | 고정 VLA를 녹화 관측에 offline 실행 | `camera/*`, `arm/q_meas`, `meta.instruction` |
| 해석적 보정 | $\delta\xi^{\text{anc}}$ | 그래프 통계(leave-one-demo-out) + 물체 pose + $x^V$ | 위 행들 |
| 학습 손목 잔차 라벨 | $\delta\xi^{\theta}$ | $x^{\text{exec}}_w \ominus (x^V \oplus \delta\xi^{\text{anc}})$ | 위 행들 |

### 7.6 수집 시점 QA (녹화 직후 자동 검사)

- 모든 그룹의 타임스탬프 단조 증가, 그룹별 평균 주기가 설정값의 ±10% 이내
- 손 MPPI `solve_ms` p95가 설정 제어 주기 이내
- 물체 pose `score`가 임계값 미만인 프레임 비율, `reinit` 횟수
- 접촉 직전 구간(손끝–표면 거리 < 3 cm)에서 pose 결측 비율
- `meta.success`가 기입되었는지
- reader로 7.5절 전 항목 재구성 성공

### 7.7 라운드 rollout 추가 필드 (Stage 4 이후, `policy/` 그룹)

| 필드 | 설명 |
|---|---|
| `round`, `sampler` | 라운드 번호, `retrieval` / `expert` |
| `vla_chunk` | 받은 VLA chunk $x^V_{1:H}$와 질의 시각 |
| `delta_anc` | $\delta\xi^{\text{anc}}_{1:H}$, gating 값, 기준 좌표계 $a_k$, 선택된 파지 성분 $g^*$ |
| `delta_theta` | $\delta\xi^{\theta}_{1:H}$ |
| `q_target` | 손 expert / retrieval이 낸 $q_{1:H}$ |
| `node_hat` | 위치한 노드 ID, 거리, 재위치화 여부 |
| `subgoal` | subgoal 노드 ID |
| `guidance` | $\lambda_V, \lambda_o$, 후보 수 $N$, 선택된 후보 |
| `model` | expert 체크포인트 ID, 그래프 artifact ID |
| `infer_ms` | 각 모듈 추론 시간 |

---

## 8. FoundationPose 속도 메모

- 가중치 파일 크기(~180 MB)는 속도를 결정하지 않는다. 비용의 대부분은 **pose 추정 단계에서 다수의 pose 가설을 렌더링·정제·채점하는 과정**이고, tracking은 이전 프레임 pose 하나만 정제하므로 훨씬 가볍다.
- NVIDIA NGC 모델 카드의 TensorRT FP16 trtexec 측정(네트워크만, 전처리·렌더링 제외): pose 추정 Jetson Orin 4.6 QPS, RTX 4060Ti 8.7 QPS / tracking Jetson Orin 746 QPS, RTX 4060Ti 1599 QPS.
- Isaac ROS 문서는 Jetson Thor에서 tracking이 120 FPS를 넘는다고 기술한다.
- 결론: **tracking은 카메라 주기(30 Hz)를 충분히 넘길 것으로 예상**되고, **pose 추정은 수 Hz 수준이라 에피소드 시작·재초기화에만 사용**한다. RTX 3080 수치는 공개 자료에서 찾지 못했으므로 **직접 측정**해 이 문서에 기록한다. 원본 PyTorch 구현은 TensorRT 경로보다 느릴 수 있으므로, 실시간 서비스는 TensorRT(또는 Isaac ROS 구현) 사용을 우선 검토한다.

---

## 9. 미결정 사항

| 항목 | 결정 시점 |
|---|---|
| PATH 전송 계층 방식 (ZMQ 등) 확인 및 메시지 스키마 확정 | Stage 1 첫 주 |
| 손 MPPI 투영 파라미터 $\varepsilon$, $d_{\text{on}}/d_{\text{off}}$ 기본값 | `teleop-sim-v0.1` |
| 실물 표면 backend: SAM3 점군 vs 메쉬 SDF + FoundationPose | 실물 이전 시 (sim에서 `mj_geom` 대비 오차 측정 후) |
| 손 행동 라벨: `q_cmd` vs `q_goal` | Stage 5 시작 전 (둘 다 녹화하므로 지연 가능) |
| keyframe 간격 $n$, chunk 길이 $H$ | Stage 2 |
| BrainCo 전류·힘 센서 제공 범위 | Stage 1 |
| 머리 카메라 외에 손목 카메라 사용 여부 | Stage 1 (사용 시 7.3절에 그룹 추가) |
| Thor(aarch64) 배포 경로 (pixi vs NVIDIA 컨테이너) | Stage 3 |
| 마우스 메쉬 획득 방법 (스캔 vs SAM 3D) | Stage 1 |
