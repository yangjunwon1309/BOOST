"""fk_error_analysis.py — GMR retarget + recorded motion FK link-position error.

이 스크립트는 4단계 워크플로우(GMR retarget → lag scan → segment 추출 →
FK link error)를 하나의 파이프라인으로 통합한 것이다. joint-angle 오차
계산/저장은 생략하고, GMR reference와 recorded q를 MuJoCo FK에 통과시켜
얻은 key-link 3D position의 오차(L2 norm, cm)만 계산/저장한다.

파이프라인:
  0) (--gmr-ref-dof-pos/--gmr-pkl이 없을 때만) session_selected/ 의
     pose_smpl_pose/pose_smpl_joints/pose_frame_index/pose_body_quat_w 를
     읽어 다중 핸들(record tick × hand)을 frame_index 단위로 평균내고
     SMPL-X npz(pose_body/root_orient/trans)로 변환 → GMR로 G1에 retarget
     → gmr_g1_ref.pkl / gmr_g1_ref_dof_pos.npy 저장. 이미 두 파일이
     존재하면 이 단계는 건너뛰고 바로 읽어서 사용한다(매번 재실행 방지).
  1) session_selected/ 에서 q(measured), action(SONIC target, optional),
     frame_index, record_monotonic_time 로드
  2) GMR reference dof_pos(.npy)와 -50~+50 프레임 lag scan으로 최적 시간 정렬
     (joint-angle RMSE가 가장 작은 lag를 채택 — 이 단계의 "오차"는 정렬용 보조
     지표일 뿐, 결과로 저장하지 않는다)
  3) 정렬된 시계열에서 시간 불연속(dt), GMR 자체의 급격한 점프(ref_step),
     joint 오차 급증(q_mae spike) 기준으로 "나쁜 구간"을 찾아 제외하고,
     남은 연속 구간 중 최소 길이를 만족하는 것만 세그먼트로 채택
  4) 각 세그먼트에 대해 GMR과 recorded q를 각각 MuJoCo FK에 넣어 key-link
     world position을 얻고, 그 차이(L2 norm)를 MAE/RMSE/P95/Max(cm)로 집계
     (base_local: root identity 고정 / root_oriented: 실측 root quat 적용)

GMR retarget(Stage 0)과 FK 평가(Stage 4)는 **동일한 --model-xml**을 사용한다.
GMR repo의 robot 키(예: unitree_g1)가 내부적으로 별도 asset 경로를 쓰는
경우를 피하기 위해, GMR 호출 시 robot config의 xml 경로를 --model-xml로
오버라이드한다 (see `retarget_with_gmr()` / `_resolve_gmr_xml_override`).

경로를 모를 때는 find_gmr_assets.sh로 먼저 후보를 탐색한다:
  bash find_gmr_assets.sh ~

실행 예 (retarget 결과가 이미 있는 경우):
  python fk_error_analysis.py \
      --session-dir ~/GR00T-WholeBodyControl/session_selected \
      --gmr-ref-dof-pos ~/gmr_ref_from_session/gmr_g1_ref_dof_pos.npy \
      --gmr-pkl ~/gmr_ref_from_session/gmr_g1_ref.pkl \
      --model-xml ~/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
      --output-dir ~/gmr_ref_from_session/fk_analysis

실행 예 (retarget을 이 스크립트가 직접 수행, 결과 없을 때 자동 실행):
  python fk_error_analysis.py \
      --session-dir ~/GR00T-WholeBodyControl/session_selected \
      --gmr-ref-dof-pos ~/gmr_ref_from_session/gmr_g1_ref_dof_pos.npy \
      --gmr-pkl ~/gmr_ref_from_session/gmr_g1_ref.pkl \
      --model-xml ~/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
      --output-dir ~/gmr_ref_from_session/fk_analysis \
      --gmr-repo-root ~/GMR \
      --gmr-robot unitree_g1 \
      --gmr-target-fps 50
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# GIF 시각화 (--make-gif 사용 시에만 import)
def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    return plt, FuncAnimation, PillowWriter

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
JOINT_XML_NAMES = [j + "_joint" for j in JOINT_NAMES]

DEFAULT_KEY_BODIES = [
    "pelvis",
    "torso_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link",
    "left_knee_link", "left_toe_link",
    "right_knee_link", "right_toe_link",
]

# GMR's smplx_to_g1.json applies this local rotation to the SMPL pelvis target.
# pose_body_quat_w comes from motion-lib and is already a world-frame root-body
# quaternion. Remove the offset before using it as SMPL-X global_orient so GMR
# applies the coordinate conversion exactly once.
GMR_SMPLX_TO_G1_ROOT_OFFSET_WXYZ = np.array(
    [0.5, -0.5, -0.5, -0.5], dtype=np.float64
)


# ──────────────────────────────────────────────────────────────────────────
# Stage 0: GMR retarget (PICO smpl_pose/smpl_joints → G1 dof_pos), optional
# ──────────────────────────────────────────────────────────────────────────
# session_to_smplx_npz.py + run_gmr_keepfps.py 의 로직을 그대로 가져온 것.
# --gmr-ref-dof-pos / --gmr-pkl 둘 다 이미 존재하면 이 단계는 완전히 건너뛴다.
def session_to_smplx_npz(session_dir: Path, smpl_pose_key: str,
                          smpl_joints_key: str, frame_index_key: str,
                          base_quat_key: str, out_npz: Path,
                          fps: float,
                          root_quat_convention: str = "robot_world") -> None:
    """session_selected의 다중 핸들(record tick × hand) SMPL 샘플을
    frame_index 단위로 평균내어 SMPL-X 표준 npz(pose_body/root_orient/trans)로
    변환한다. (원본: session_to_smplx_npz.py, 로직 동일)"""
    pose = np.load(session_dir / f"{smpl_pose_key}.npy")      # (T,H,21,3)
    joints = np.load(session_dir / f"{smpl_joints_key}.npy")  # (T,H,24,3)
    frame_idx = np.load(session_dir / f"{frame_index_key}.npy")  # (T,H)

    quat_path = session_dir / f"{base_quat_key}.npy"
    quat = np.load(quat_path) if quat_path.exists() else None  # (T,H,4), wxyz

    n_frames = int(frame_idx.max()) + 1
    sum_pose = np.zeros((n_frames, 21, 3), dtype=np.float64)
    sum_trans = np.zeros((n_frames, 3), dtype=np.float64)
    cnt = np.zeros((n_frames,), dtype=np.float64)
    sum_quat = np.zeros((n_frames, 4), dtype=np.float64) if quat is not None else None

    t_dim, h_dim = frame_idx.shape
    for t in range(t_dim):
        for h in range(h_dim):
            k = int(frame_idx[t, h])
            sum_pose[k] += pose[t, h]
            sum_trans[k] += joints[t, h, 0]  # pelvis/root joint
            if quat is not None:
                q = quat[t, h].astype(np.float64)
                if np.linalg.norm(q) > 1e-8:
                    q = q / np.linalg.norm(q)
                    if cnt[k] > 0 and np.dot(sum_quat[k], q) < 0:
                        q = -q
                    sum_quat[k] += q
            cnt[k] += 1

    valid = cnt > 0
    for k in range(n_frames):
        if not valid[k]:
            prev = max(0, k - 1)
            sum_pose[k] = sum_pose[prev]
            sum_trans[k] = sum_trans[prev]
            if sum_quat is not None:
                sum_quat[k] = sum_quat[prev]
            cnt[k] = 1

    pose_body = (sum_pose / cnt[:, None, None]).reshape(n_frames, 63).astype(np.float32)
    trans = (sum_trans / cnt[:, None]).astype(np.float32)
    trans = trans - trans[:1]

    root_orient = np.zeros((n_frames, 3), dtype=np.float32)
    if sum_quat is not None:
        try:
            from scipy.spatial.transform import Rotation as R
            q_wxyz = sum_quat / np.maximum(
                np.linalg.norm(sum_quat, axis=1, keepdims=True), 1e-8
            )
            root_rotation = R.from_quat(q_wxyz[:, [1, 2, 3, 0]])
            if root_quat_convention == "robot_world":
                offset = R.from_quat(
                    GMR_SMPLX_TO_G1_ROOT_OFFSET_WXYZ[[1, 2, 3, 0]]
                )
                root_rotation = root_rotation * offset.inv()
            elif root_quat_convention != "smpl":
                raise ValueError(
                    f"Unknown root quaternion convention: {root_quat_convention}"
                )
            root_orient = root_rotation.as_rotvec().astype(np.float32)

            source_euler = R.from_quat(
                q_wxyz[0, [1, 2, 3, 0]]
            ).as_euler("xyz", degrees=True)
            smpl_euler = root_rotation[0].as_euler("xyz", degrees=True)
            print(f"[retarget] root quaternion convention={root_quat_convention}; "
                  f"frame0 source xyz(deg)={source_euler.round(2)}, "
                  f"SMPL global_orient xyz(deg)={smpl_euler.round(2)}")
        except Exception as exc:
            print(f"[retarget] warn: failed to convert root quat, using zero "
                  f"root_orient: {exc!r}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        gender=np.array("neutral"),
        betas=np.zeros((10,), dtype=np.float32),
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        mocap_frame_rate=np.array(fps),
        mocap_framerate=np.array(fps),
        fps=np.array(fps),
        root_quat_convention=np.array(root_quat_convention),
    )
    print(f"[retarget] saved smplx npz: {out_npz} "
          f"(frames={n_frames}, pose_body={pose_body.shape}, trans={trans.shape})")


def retarget_with_gmr(
    smplx_npz: Path, gmr_repo_root: Path, model_xml: Path,
    robot: str, target_fps: int, save_pkl: Path,
) -> Path:
    """GMR(GeneralMotionRetargeting)을 호출해 SMPL-X npz를 robot dof_pos로
    retarget한다. (원본: run_gmr_keepfps.py, 로직 동일)

    --model-xml을 GMR의 robot config가 가리키는 xml 경로와 일치시키기 위해,
    GMR import 전에 PYTHONPATH/cwd를 gmr_repo_root로 맞추고, 가능하면
    robot config의 xml 경로를 model_xml로 오버라이드한다.
    """
    gmr_repo_root = gmr_repo_root.expanduser().resolve()
    if str(gmr_repo_root) not in sys.path:
        sys.path.insert(0, str(gmr_repo_root))

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import (
        load_smplx_file,
        get_smplx_data_offline_fast,
    )

    _resolve_gmr_xml_override(robot, model_xml)

    smplx_folder = gmr_repo_root / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        str(smplx_npz), smplx_folder
    )
    frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=target_fps
    )

    retarget = GMR(actual_human_height=actual_human_height, src_human="smplx",
                    tgt_robot=robot)

    qpos_list = []
    for i, fr in enumerate(frames):
        if i % 100 == 0:
            print(f"[retarget] {i}/{len(frames)}")
        qpos_list.append(retarget.retarget(fr))

    qpos = np.asarray(qpos_list)
    motion = {
        "fps": fps,
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],  # wxyz -> xyzw
        "dof_pos": qpos[:, 7:],
    }

    save_pkl.parent.mkdir(parents=True, exist_ok=True)
    with save_pkl.open("wb") as f:
        pickle.dump(motion, f)
    dof_pos_path = save_pkl.with_name(save_pkl.stem + "_dof_pos.npy")
    np.save(dof_pos_path, motion["dof_pos"])
    print(f"[retarget] saved {save_pkl} and {dof_pos_path} "
          f"(dof_pos={motion['dof_pos'].shape}, fps={fps})")
    return dof_pos_path


def _resolve_gmr_xml_override(robot: str, model_xml: Path) -> None:
    """GMR의 robot config가 --model-xml과 다른 경로를 바라보고 있으면 경고.

    GMR 버전에 따라 robot xml 경로를 노출하는 API가 다를 수 있어, 가능한
    한 비파괴적으로(설정을 시도하되 실패해도 계속 진행) 처리한다.
    """
    try:
        from general_motion_retargeting import RobotMotionViewer  # noqa: F401
    except Exception:
        pass
    try:
        from general_motion_retargeting.params import ROBOT_XML_DICT
        configured = ROBOT_XML_DICT.get(robot)
        if configured is not None:
            configured_path = Path(configured).expanduser().resolve()
            if configured_path != model_xml.expanduser().resolve():
                print(f"[retarget] note: GMR's default xml for robot={robot!r} is "
                      f"{configured_path}, while --model-xml is {model_xml}. "
                      f"Overriding ROBOT_XML_DICT[{robot!r}] to use --model-xml "
                      f"for consistency with the FK evaluation stage.")
                ROBOT_XML_DICT[robot] = str(model_xml.expanduser().resolve())
    except Exception as exc:
        print(f"[retarget] note: could not inspect/override GMR's robot xml "
              f"path automatically ({exc!r}); GMR will use its own default "
              f"xml for robot={robot!r}. Make sure it matches --model-xml, "
              f"or FK results may be inconsistent between retarget and "
              f"evaluation.")


def ensure_gmr_reference(
    session_dir: Path, output_work_dir: Path,
    gmr_ref_dof_pos: Path, gmr_pkl: Path | None,
    gmr_repo_root: Path | None, model_xml: Path,
    robot: str, target_fps: int,
    smpl_pose_key: str, smpl_joints_key: str,
    frame_index_key: str, base_quat_key: str,
    root_quat_convention: str,
) -> tuple[Path, Path | None]:
    """gmr_ref_dof_pos(+gmr_pkl)이 이미 존재하면 그대로 사용, 없으면
    session_selected로부터 retarget을 실행해서 생성한다."""
    dof_pos_exists = gmr_ref_dof_pos.exists()
    pkl_exists = gmr_pkl is not None and gmr_pkl.exists()

    if dof_pos_exists and (gmr_pkl is None or pkl_exists):
        print(f"[retarget] using existing reference: {gmr_ref_dof_pos}"
              + (f", {gmr_pkl}" if gmr_pkl else ""))
        return gmr_ref_dof_pos, gmr_pkl

    if gmr_repo_root is None:
        raise SystemExit(
            f"GMR reference not found ({gmr_ref_dof_pos}) and --gmr-repo-root "
            f"was not given, so retargeting cannot be run automatically. "
            f"Either generate the reference first, or pass --gmr-repo-root."
        )

    print(f"[retarget] reference not found, running GMR retarget from "
          f"session: {session_dir}")
    smplx_npz = output_work_dir / "session_smplx.npz"
    session_to_smplx_npz(
        session_dir, smpl_pose_key, smpl_joints_key, frame_index_key,
        base_quat_key, smplx_npz, fps=float(target_fps),
        root_quat_convention=root_quat_convention,
    )

    save_pkl = gmr_pkl if gmr_pkl is not None else (output_work_dir / "gmr_g1_ref.pkl")
    dof_pos_path = retarget_with_gmr(
        smplx_npz, gmr_repo_root, model_xml, robot, target_fps, save_pkl,
    )
    return dof_pos_path, save_pkl


# ──────────────────────────────────────────────────────────────────────────
# Stage 1: load recorded session + GMR reference
# ──────────────────────────────────────────────────────────────────────────
def load_session(session_dir: Path, q_key: str, action_key: str | None,
                  frame_index_key: str, time_key: str):
    q = np.load(session_dir / f"{q_key}.npy")
    frame_idx = np.load(session_dir / f"{frame_index_key}.npy")
    if frame_idx.ndim > 1:
        frame_idx = frame_idx[:, 0]
    t = np.load(session_dir / f"{time_key}.npy").reshape(-1)

    action = None
    if action_key is not None:
        action_path = session_dir / f"{action_key}.npy"
        if action_path.exists():
            action = np.load(action_path)
        else:
            print(f"[warn] action key not found, skipping action alignment: "
                  f"{action_path}")
    return q, action, frame_idx, t


def load_gmr_ref(ref_dof_pos_path: Path) -> np.ndarray:
    return np.load(ref_dof_pos_path)


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: lag scan (joint-angle based, alignment-only — not saved as a result)
# ──────────────────────────────────────────────────────────────────────────
def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    e = a - b
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
    }


def align_with_lag(base_rows: np.ndarray, valid_ref: np.ndarray,
                    frame_idx: np.ndarray, ref: np.ndarray,
                    sig: np.ndarray, lag: int):
    sig_rows = base_rows + lag
    valid = valid_ref & (sig_rows >= 0) & (sig_rows < len(sig))
    b = base_rows[valid]
    s = sig_rows[valid]
    return b, s, ref[frame_idx[b]], sig[s]


def scan_best_lag(base_rows: np.ndarray, valid_ref: np.ndarray,
                   frame_idx: np.ndarray, ref: np.ndarray, sig: np.ndarray,
                   lag_range: int = 50) -> tuple[int, list[tuple[int, float, float]]]:
    rows = []
    for lag in range(-lag_range, lag_range + 1):
        _, _, r, s = align_with_lag(base_rows, valid_ref, frame_idx, ref, sig, lag)
        if len(r) == 0:
            continue
        m = metrics(s, r)
        rows.append((lag, m["mae"], m["rmse"]))
    if not rows:
        raise RuntimeError("No valid frames found during lag scan.")
    best = min(rows, key=lambda x: x[2])
    return best[0], rows


# ──────────────────────────────────────────────────────────────────────────
# Stage 3: segment extraction (discontinuity / spike / GMR-jump based)
# ──────────────────────────────────────────────────────────────────────────
def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def find_segments(
    q_mae: np.ndarray, dt: np.ndarray, ref_step: np.ndarray,
    min_frames: int, min_duration_sec: float, dilate_radius: int,
    dt_max_sec: float, time_rel: np.ndarray,
) -> list[tuple[int, int]]:
    q_err_thr = max(np.percentile(q_mae, 98), np.median(q_mae) + 6 * mad(q_mae))
    ref_step_thr = max(np.percentile(ref_step, 99),
                        np.median(ref_step) + 10 * mad(ref_step))

    bad = np.zeros(len(q_mae), dtype=bool)
    bad |= dt > dt_max_sec
    bad |= q_mae > q_err_thr
    bad |= ref_step > ref_step_thr

    bad_dilated = bad.copy()
    for i in np.where(bad)[0]:
        bad_dilated[max(0, i - dilate_radius): min(len(bad), i + dilate_radius + 1)] = True
    bad = bad_dilated

    segments: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(~bad):
        if ok and start is None:
            start = i
        if (not ok or i == len(bad) - 1) and start is not None:
            end = i if not ok else i + 1
            n = end - start
            dur = time_rel[end - 1] - time_rel[start] if n > 1 else 0.0
            if n >= min_frames and dur >= min_duration_sec:
                segments.append((start, end))
            start = None

    print(f"[segments] q_err_thr={q_err_thr * 180 / np.pi:.3f} deg, "
          f"ref_step_thr={ref_step_thr:.4f} rad, segments={len(segments)}")
    return segments


# ──────────────────────────────────────────────────────────────────────────
# Stage 4: FK link-position error (the only error metric we keep/save)
# ──────────────────────────────────────────────────────────────────────────
class FkEvaluator:
    def __init__(self, model_xml: Path, key_bodies: list[str]):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)

        self.body_ids: dict[str, int] = {}
        for b in key_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if bid >= 0:
                self.body_ids[b] = bid
            else:
                print(f"[warn] body '{b}' not found in model, skipping.")
        if not self.body_ids:
            raise RuntimeError("None of the requested key bodies exist in the model.")

        self.joint_qpos_addr: dict[str, int] = {}
        self.free_addr: int | None = None
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            jtype = self.model.jnt_type[j]
            addr = self.model.jnt_qposadr[j]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                self.free_addr = addr
            elif name in JOINT_XML_NAMES:
                self.joint_qpos_addr[name] = addr

        missing = [n for n in JOINT_XML_NAMES if n not in self.joint_qpos_addr]
        if missing:
            raise RuntimeError(f"Missing joints in model: {missing}")

        print(f"[fk] model: {model_xml}")
        print(f"[fk] nq={self.model.nq}, nv={self.model.nv}, "
              f"bodies={list(self.body_ids.keys())}")

    @staticmethod
    def _norm_quat(q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        return np.array([1.0, 0.0, 0.0, 0.0]) if n < 1e-9 else q / n

    def fk(self, q29: np.ndarray, root_quat_wxyz: np.ndarray | None = None,
           mode: str = "base_local") -> dict[str, np.ndarray]:
        self.data.qpos[:] = 0.0
        if self.free_addr is not None:
            self.data.qpos[self.free_addr:self.free_addr + 3] = 0.0
            if mode == "root_oriented" and root_quat_wxyz is not None:
                self.data.qpos[self.free_addr + 3:self.free_addr + 7] = self._norm_quat(root_quat_wxyz)
            else:
                self.data.qpos[self.free_addr + 3:self.free_addr + 7] = np.array(
                    [1.0, 0.0, 0.0, 0.0]
                )

        for col, jn in enumerate(JOINT_XML_NAMES):
            self.data.qpos[self.joint_qpos_addr[jn]] = q29[col]

        self.mujoco.mj_forward(self.model, self.data)

        root_bid = self.body_ids.get("pelvis", next(iter(self.body_ids.values())))
        root_pos = self.data.xpos[root_bid].copy()

        pos = {}
        for name, bid in self.body_ids.items():
            pos[name] = self.data.xpos[bid].copy() - root_pos
        return pos

def evaluate_segment_fk(
    fk_eval: FkEvaluator,
    seg_id: int, s: int, e: int,
    base_rows: np.ndarray, frame_idx: np.ndarray, q_rows: np.ndarray, time_rel: np.ndarray,
    gmr_ref: np.ndarray, q: np.ndarray,
    gmr_root_wxyz: np.ndarray | None, q_base_quat_wxyz: np.ndarray | None,
    modes: list[str], output_dir: Path,
) -> list[list]:
    summary_rows: list[list] = []

    for mode in modes:
        if mode == "root_oriented" and (
            gmr_root_wxyz is None or q_base_quat_wxyz is None
        ):
            print(f"[fk] seg={seg_id} mode={mode} skipped "
                  f"(missing root orientation data)")
            continue

        err_by_body = {b: [] for b in fk_eval.body_ids}
        frame_rows = []
        long_path = output_dir / f"segment_{seg_id:02d}_fk_positions_{mode}.csv"

        with long_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "time_sec", "body",
                "gmr_x", "gmr_y", "gmr_z",
                "q_x", "q_y", "q_z",
                "err_x", "err_y", "err_z", "err_norm_m",
            ])

            for i in range(s, e):
                base_row = base_rows[i]
                gi = int(frame_idx[base_row])
                qi = int(q_rows[i])
                t_sec = float(time_rel[i])

                if mode == "base_local":
                    gp = fk_eval.fk(gmr_ref[gi], None, mode="base_local")
                    qp = fk_eval.fk(q[qi], None, mode="base_local")

                elif mode == "root_oriented":
                    gmr_quat = gmr_root_wxyz[gi]
                    q_quat = q_base_quat_wxyz[qi]
                    gp = fk_eval.fk(gmr_ref[gi], gmr_quat, mode="root_oriented")
                    qp = fk_eval.fk(q[qi], q_quat, mode="root_oriented")

                else:
                    continue

                norms = []
                for b in fk_eval.body_ids:
                    err = qp[b] - gp[b]
                    n = float(np.linalg.norm(err))
                    err_by_body[b].append(n)
                    norms.append(n)
                    w.writerow([
                        t_sec, b,
                        gp[b][0], gp[b][1], gp[b][2],
                        qp[b][0], qp[b][1], qp[b][2],
                        err[0], err[1], err[2], n,
                    ])
                frame_rows.append([t_sec, float(np.mean(norms)),
                                    math.sqrt(float(np.mean(np.square(norms))))])

        metric_path = output_dir / f"segment_{seg_id:02d}_fk_body_metrics_{mode}.csv"
        with metric_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["segment", "mode", "body", "mae_cm", "rmse_cm", "p95_cm", "max_cm"])
            for b, vals in err_by_body.items():
                vals_arr = np.asarray(vals)
                mae = float(vals_arr.mean() * 100)
                rmse = float(math.sqrt(np.mean(vals_arr * vals_arr)) * 100)
                p95 = float(np.percentile(vals_arr, 95) * 100)
                mx = float(vals_arr.max() * 100)
                w.writerow([seg_id, mode, b, mae, rmse, p95, mx])
                summary_rows.append([seg_id, mode, b, mae, rmse, p95, mx])

        np.savetxt(
            output_dir / f"segment_{seg_id:02d}_fk_frame_metrics_{mode}.csv",
            np.asarray(frame_rows),
            delimiter=",",
            header="time_sec,mean_body_error_m,rmse_body_error_m",
            comments="",
        )

    return summary_rows


# ──────────────────────────────────────────────────────────────────────────
# GIF 시각화 — FK link position을 3D 스켈레톤 애니메이션으로 저장
# ──────────────────────────────────────────────────────────────────────────
# fk_error_analysis.py가 저장한 segment_XX_fk_positions_{mode}.csv를 읽어
# GMR(파랑)과 measured G1 q(주황)를 같은 3D 좌표계에 겹쳐 그리고 GIF로 저장.

_GIF_BONES = [
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
    ("pelvis", "left_knee_link"),
    ("left_knee_link", "left_toe_link"),
    ("pelvis", "right_knee_link"),
    ("right_knee_link", "right_toe_link"),
]


def _load_fk_positions_csv(
    path: Path,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """CSV를 times, body_names, gmr[F,B,3], measured[F,B,3]로 읽는다."""
    times: list[float] = []
    body_names: list[str] = []
    gmr_rows: list[list[np.ndarray]] = []
    q_rows: list[list[np.ndarray]] = []
    current_time: float | None = None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_sec"])
            if current_time is None or t != current_time:
                times.append(t)
                gmr_rows.append([])
                q_rows.append([])
                current_time = t
            if len(times) == 1:
                body_names.append(row["body"])
            gmr_rows[-1].append(np.fromiter(
                (float(row["gmr_x"]), float(row["gmr_y"]), float(row["gmr_z"])),
                dtype=np.float64,
            ))
            q_rows[-1].append(np.fromiter(
                (float(row["q_x"]), float(row["q_y"]), float(row["q_z"])),
                dtype=np.float64,
            ))
    if not times:
        return np.empty(0), [], np.empty((0, 0, 3)), np.empty((0, 0, 3))
    return np.asarray(times), body_names, np.asarray(gmr_rows), np.asarray(q_rows)


def _gif_axis_limits(
    gmr: np.ndarray, measured: np.ndarray, margin: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.concatenate((gmr.reshape(-1, 3), measured.reshape(-1, 3)), axis=0)
    center = (arr.min(0) + arr.max(0)) / 2
    half   = max((arr.max(0) - arr.min(0)).max() / 2, 0.05) + margin
    return center - half, center + half


def make_segment_gif(
    csv_path: Path,
    output_path: Path,
    fps: int = 15,
    stride: int = 2,
    elev: float = 15.0,
    azim: float = -60.0,
) -> None:
    plt, FuncAnimation, PillowWriter = _import_matplotlib()

    times, body_names, gmr, measured = _load_fk_positions_csv(csv_path)
    times, gmr, measured = times[::stride], gmr[::stride], measured[::stride]
    if len(times) == 0:
        print(f"[gif] no frames to animate: {csv_path}")
        return

    lo, hi = _gif_axis_limits(gmr, measured)
    mode_label = csv_path.stem.split("_fk_positions_")[-1]
    seg_id     = csv_path.stem.split("_")[1]
    body_index = {name: i for i, name in enumerate(body_names)}
    bone_indices = [
        (body_index[parent], body_index[child])
        for parent, child in _GIF_BONES
        if parent in body_index and child in body_index
    ]

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)

    gmr_scatter = ax.scatter([], [], [], c="#1f77b4", s=28, depthshade=True,
                             label="GMR (retargeted SMPL)")
    q_scatter = ax.scatter([], [], [], c="#ff7f0e", s=28, depthshade=True,
                           label="Measured G1 q")
    gmr_lines = [ax.plot([], [], [], color="#1f77b4", linewidth=2.0,
                         alpha=0.85)[0] for _ in bone_indices]
    q_lines = [ax.plot([], [], [], color="#ff7f0e", linewidth=2.0,
                       alpha=0.85)[0] for _ in bone_indices]
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")

    def set_points(scatter, points: np.ndarray) -> None:
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    def set_bones(lines, points: np.ndarray) -> None:
        for line, (parent, child) in zip(lines, bone_indices):
            segment = points[[parent, child]]
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])

    def update(i: int):
        set_points(gmr_scatter, gmr[i]); set_points(q_scatter, measured[i])
        set_bones(gmr_lines, gmr[i]); set_bones(q_lines, measured[i])
        title.set_text(f"seg {seg_id} [{mode_label}]  t={times[i]:.2f}s  "
                       f"({i+1}/{len(times)})")
        return (gmr_scatter, q_scatter, title, *gmr_lines, *q_lines)

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 // fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[gif] saved {output_path}  "
          f"({len(times)} frames @ {fps}fps, stride={stride}, "
          f"span={times[0]:.2f}s~{times[-1]:.2f}s)")


# ──────────────────────────────────────────────────────────────────────────
# Optional: root orientation sources (for root_oriented FK mode)
# ──────────────────────────────────────────────────────────────────────────
def load_gmr_root_wxyz(gmr_pkl_path: Path | None) -> np.ndarray | None:
    if gmr_pkl_path is None:
        return None
    with gmr_pkl_path.open("rb") as f:
        gmr_motion = pickle.load(f)
    root_xyzw = np.asarray(gmr_motion["root_rot"])
    return root_xyzw[:, [3, 0, 1, 2]]


def load_state_base_quat_wxyz(session_dir: Path, key: str) -> np.ndarray | None:
    path = session_dir / f"{key}.npy"
    if not path.exists():
        return None
    return np.load(path)


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GMR retarget + recorded motion — FK link-position error "
                    "(joint-angle error scan is alignment-only and not saved)."
    )
    p.add_argument("--session-dir", type=Path,
                    help="session_selected 디렉토리 (q/action/frame_index/time .npy 위치, "
                         "그리고 retarget이 필요하면 pose_smpl_pose/pose_smpl_joints도). "
                         "record_sonic_motion_triplet.py 실행 결과물.")
    p.add_argument("--gmr-ref-dof-pos", type=Path,
                    help="GMR retarget 결과 dof_pos .npy ((T,29)) 경로. "
                         "이 파일이 이미 존재하면 그대로 읽어서 사용한다. "
                         "존재하지 않고 --gmr-repo-root가 주어지면, 이 스크립트가 "
                         "session-dir로부터 자동으로 retarget을 실행해 이 경로에 저장한다.")
    p.add_argument("--gmr-pkl", type=Path, default=None,
                    help="GMR 결과 .pkl (root_rot 포함). root_oriented FK 모드에 필요. "
                         "없고 retarget을 자동 실행하는 경우 <output-dir>/gmr_g1_ref.pkl로 "
                         "생성된다. 미리 만들어진 게 있으면 그 경로를 지정.")
    p.add_argument(
        "--model-xml", type=Path,
        help="G1 29dof MJCF xml 경로. GMR retarget(Stage 0)과 FK 평가(Stage 4) "
             "양쪽에서 동일하게 사용된다 — 둘이 다르면 좌표계/링크 이름이 어긋나 "
             "결과가 부정확해진다. 권장: ~/GMR/assets/unitree_g1/g1_mocap_29dof.xml "
             "(GMR retarget 결과와 정합성을 보장하려면 GMR 자체 asset을 권장).",
    )
    p.add_argument("--output-dir", required=True, type=Path)

    # GMR retarget (Stage 0) — gmr-ref-dof-pos가 없을 때만 사용됨
    p.add_argument("--gmr-repo-root", type=Path, default=None,
                    help="GMR 레포 루트 (예: ~/GMR). --gmr-ref-dof-pos가 없을 때 "
                         "자동 retarget을 실행하려면 필요.")
    p.add_argument("--gmr-robot", default="unitree_g1",
                    help="GMR retarget 대상 robot 키 (GMR의 tgt_robot 인자).")
    p.add_argument("--gmr-target-fps", type=int, default=50,
                    help="GMR retarget 및 SMPL-X 변환 시 목표 FPS.")
    p.add_argument("--smpl-pose-key", default="pose_smpl_pose",
                    help="session_dir 안의 SMPL pose_body(.npy, (T,H,21,3)) 파일명.")
    p.add_argument("--smpl-joints-key", default="pose_smpl_joints",
                    help="session_dir 안의 SMPL joints(.npy, (T,H,24,3)) 파일명.")
    p.add_argument("--smpl-base-quat-key", default="pose_body_quat_w",
                    help="session_dir 안의 PICO 측 root quaternion(wxyz, (T,H,4)) "
                         "파일명. retarget 시 root_orient 계산에 사용 (선택, "
                         "없으면 root_orient=0으로 처리).")
    p.add_argument(
        "--smpl-root-quat-convention",
        choices=["robot_world", "smpl"],
        default="robot_world",
        help="--smpl-base-quat-key의 좌표 규약. robot_world(기본)는 "
             "motion-lib body_quat_w처럼 이미 robot/world 축인 quaternion이며 "
             "GMR pelvis offset의 역회전을 적용한다. smpl은 입력을 SMPL "
             "native global_orient로 그대로 사용한다.",
    )
    p.add_argument("--force-retarget", action="store_true",
                    help="--gmr-ref-dof-pos/--gmr-pkl이 이미 있어도 강제로 재실행.")

    p.add_argument("--q-key", default="state_body_q",
                    help="session_dir 안의 measured joint q .npy 파일명(확장자 제외).")
    p.add_argument("--action-key", default="state_last_action",
                    help="SONIC policy target .npy 파일명. lag scan 보조용 "
                         "(없으면 --no-action-lag-scan로 생략).")
    p.add_argument("--no-action-lag-scan", action="store_true",
                    help="action 기반 lag scan을 생략 (q만으로 정렬).")
    p.add_argument("--frame-index-key", default="pose_frame_index")
    p.add_argument("--time-key", default="record_monotonic_time")
    p.add_argument("--base-quat-key", default="state_base_quat",
                    help="measured root quaternion(wxyz) .npy 파일명. "
                         "root_oriented FK 모드에 필요.")

    p.add_argument("--lag-range", type=int, default=50,
                    help="lag scan 범위 (±프레임).")
    p.add_argument("--dt-max-sec", type=float, default=0.08,
                    help="이 값보다 큰 시간 간격은 불연속으로 간주.")
    p.add_argument("--min-segment-frames", type=int, default=100)
    p.add_argument("--min-segment-duration-sec", type=float, default=2.0)
    p.add_argument("--dilate-radius", type=int, default=15,
                    help="불연속/스파이크 주변 제외 마진(프레임).")

    p.add_argument("--key-bodies", nargs="+", default=DEFAULT_KEY_BODIES,
                    help="FK로 비교할 body 이름 목록.")
    p.add_argument("--fk-modes", nargs="+",
                    default=["base_local", "root_oriented"],
                    choices=["base_local", "root_oriented"],
                    help="평가할 FK 모드. 기본값으로 두 모드 모두 실행. "
                         "base_local: root identity 고정(순수 자세 비교). "
                         "root_oriented: GMR root와 measured root quaternion 적용.")

    # GIF 시각화
    p.add_argument("--make-gif", action="store_true",
                    help="FK 평가 완료 후 각 segment × mode에 대해 3D 스켈레톤 "
                         "애니메이션 GIF를 자동 생성한다. "
                         "--gif-segments / --gif-modes 로 범위를 제한할 수 있다.")
    p.add_argument("--gif-only", action="store_true",
                    help="FK 분석을 실행하지 않고 output-dir의 기존 position CSV로 "
                         "GIF만 생성한다. 이때 session/GMR/model 인자는 불필요하다.")
    p.add_argument("--gif-segments", nargs="+", type=int, default=None,
                    help="GIF를 생성할 segment id 목록. 생략 시 모든 segment. "
                         "예: --gif-segments 0 2 5")
    p.add_argument("--gif-modes", nargs="+",
                    default=["base_local", "root_oriented"],
                    choices=["base_local", "root_oriented"],
                    help="GIF를 생성할 FK 모드 목록. 기본: 두 모드 모두.")
    p.add_argument("--gif-fps", type=int, default=15)
    p.add_argument("--gif-stride", type=int, default=2,
                    help="프레임 thinning. 원본 50Hz라 2~5 권장.")
    p.add_argument("--gif-elev", type=float, default=15.0)
    p.add_argument("--gif-azim", type=float, default=-60.0)

    return p.parse_args()


def generate_requested_gifs(
    args: argparse.Namespace,
    available_segment_ids: list[int] | None = None,
    allowed_modes: list[str] | None = None,
) -> None:
    """기존 FK position CSV에서 요청된 GIF를 생성한다."""
    if available_segment_ids is None:
        available_segment_ids = sorted({
            int(path.name.split("_")[1])
            for mode in args.gif_modes
            for path in args.output_dir.glob(f"segment_*_fk_positions_{mode}.csv")
        })
    segment_ids = args.gif_segments or available_segment_ids
    modes = args.gif_modes if allowed_modes is None else [
        mode for mode in args.gif_modes if mode in allowed_modes
    ]

    try:
        _import_matplotlib()
    except ImportError:
        print("[gif] matplotlib or pillow not found. "
              "Install with: pip install matplotlib pillow")
        return

    generated = 0
    for seg_id in segment_ids:
        for mode in modes:
            csv_path = args.output_dir / f"segment_{seg_id:02d}_fk_positions_{mode}.csv"
            if not csv_path.exists():
                print(f"[gif] {csv_path.name} not found, skip.")
                continue
            gif_path = args.output_dir / f"segment_{seg_id:02d}_{mode}.gif"
            make_segment_gif(
                csv_path, gif_path,
                fps=args.gif_fps,
                stride=args.gif_stride,
                elev=args.gif_elev,
                azim=args.gif_azim,
            )
            generated += 1
    if generated == 0:
        print("[gif] no matching FK position CSV files found.")


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.gif_only:
        generate_requested_gifs(args)
        return

    required = {
        "--session-dir": args.session_dir,
        "--gmr-ref-dof-pos": args.gmr_ref_dof_pos,
        "--model-xml": args.model_xml,
    }
    missing_args = [name for name, value in required.items() if value is None]
    if missing_args:
        raise SystemExit(f"Missing required arguments: {', '.join(missing_args)}")

    # 사전 점검: 일반 분석에서는 session/model-xml이 있어야 함
    missing = []
    if not args.session_dir.exists():
        missing.append(
            f"--session-dir not found: {args.session_dir}\n"
            f"  → run record_sonic_motion_triplet.py first to produce session_selected/."
        )
    if not args.model_xml.exists():
        missing.append(f"--model-xml not found: {args.model_xml}")
    if missing:
        print("[fk_error_analysis] Missing required inputs:\n")
        for m in missing:
            print(f"  - {m}\n")
        print("Run `bash find_gmr_assets.sh ~` again after generating the "
              "missing files, then re-run this script with the correct paths.")
        raise SystemExit(1)

    # Stage 0 — GMR retarget (없으면 자동 실행, 있으면 그대로 사용)
    if args.force_retarget:
        if args.gmr_ref_dof_pos.exists():
            args.gmr_ref_dof_pos.unlink()
        if args.gmr_pkl is not None and args.gmr_pkl.exists():
            args.gmr_pkl.unlink()

    gmr_ref_dof_pos_path, gmr_pkl_path = ensure_gmr_reference(
        session_dir=args.session_dir,
        output_work_dir=args.output_dir,
        gmr_ref_dof_pos=args.gmr_ref_dof_pos,
        gmr_pkl=args.gmr_pkl,
        gmr_repo_root=args.gmr_repo_root,
        model_xml=args.model_xml,
        robot=args.gmr_robot,
        target_fps=args.gmr_target_fps,
        smpl_pose_key=args.smpl_pose_key,
        smpl_joints_key=args.smpl_joints_key,
        frame_index_key=args.frame_index_key,
        base_quat_key=args.smpl_base_quat_key,
        root_quat_convention=args.smpl_root_quat_convention,
    )
    args.gmr_ref_dof_pos = gmr_ref_dof_pos_path
    args.gmr_pkl = gmr_pkl_path

    # Stage 1 — load
    q, action, frame_idx, t = load_session(
        args.session_dir, args.q_key,
        None if args.no_action_lag_scan else args.action_key,
        args.frame_index_key, args.time_key,
    )
    gmr_ref = load_gmr_ref(args.gmr_ref_dof_pos)
    print(f"[load] gmr_ref={gmr_ref.shape}, q={q.shape}"
          + (f", action={action.shape}" if action is not None else ""))

    base_rows = np.arange(len(frame_idx))
    valid_ref = (frame_idx >= 0) & (frame_idx < len(gmr_ref))
    print(f"[load] valid rows: {int(valid_ref.sum())}/{len(valid_ref)}")

    # Stage 2 — lag scan (alignment only, joint-angle metric not saved)
    q_lag, _ = scan_best_lag(base_rows, valid_ref, frame_idx, gmr_ref, q,
                              lag_range=args.lag_range)
    print(f"[align] best q lag: {q_lag} frames")

    align_lag = q_lag
    if action is not None:
        action_lag, _ = scan_best_lag(base_rows, valid_ref, frame_idx, gmr_ref, action,
                                       lag_range=args.lag_range)
        print(f"[align] best action lag: {action_lag} frames (reported only, "
              f"q_lag is used for segmenting/FK)")

    q_rows = base_rows + align_lag
    valid = valid_ref & (q_rows >= 0) & (q_rows < len(q))
    base = base_rows[valid]
    q_rows = q_rows[valid]

    gmr_aligned = gmr_ref[frame_idx[base]]
    q_aligned = q[q_rows]
    q_err = q_aligned - gmr_aligned
    q_mae_frame = np.mean(np.abs(q_err), axis=1)

    ref_time = t[base]
    ref_time_rel = ref_time - ref_time[0]
    dt = np.r_[0.0, np.diff(ref_time)]
    ref_step = np.r_[0.0, np.linalg.norm(np.diff(gmr_aligned, axis=0), axis=1)]

    # Stage 3 — segments
    segments = find_segments(
        q_mae_frame, dt, ref_step,
        min_frames=args.min_segment_frames,
        min_duration_sec=args.min_segment_duration_sec,
        dilate_radius=args.dilate_radius,
        dt_max_sec=args.dt_max_sec,
        time_rel=ref_time_rel,
    )
    for sid, (s, e) in enumerate(segments):
        print(f"  seg {sid:02d}: {ref_time_rel[s]:8.3f}s - {ref_time_rel[e-1]:8.3f}s "
              f"dur={ref_time_rel[e-1]-ref_time_rel[s]:7.3f}s frames={e-s:5d}")

    if not segments:
        print("[warn] No valid segments found — nothing to evaluate with FK.")
        return

    # Stage 4 — FK link-position error
    gmr_root_wxyz = load_gmr_root_wxyz(args.gmr_pkl)
    q_base_quat_wxyz = load_state_base_quat_wxyz(args.session_dir, args.base_quat_key)

    fk_modes = list(args.fk_modes)
    if "root_oriented" in fk_modes and (gmr_root_wxyz is None or q_base_quat_wxyz is None):
        print("[warn] root_oriented FK requested but root orientation data "
              "unavailable (need --gmr-pkl and a valid --base-quat-key); "
              "this mode will be skipped per-segment.")

    fk_eval = FkEvaluator(args.model_xml, args.key_bodies)

    summary_rows: list[list] = []
    for seg_id, (s, e) in enumerate(segments):
        summary_rows += evaluate_segment_fk(
            fk_eval, seg_id, s, e,
            base, frame_idx, q_rows, ref_time_rel,
            gmr_ref, q,
            gmr_root_wxyz, q_base_quat_wxyz,
            fk_modes, args.output_dir,
        )

    with (args.output_dir / "fk_summary_all_segments.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "mode", "body", "mae_cm", "rmse_cm", "p95_cm", "max_cm"])
        w.writerows(summary_rows)

    print(f"\n[done] saved to: {args.output_dir}")
    for mode in fk_modes:
        rows = [r for r in summary_rows if r[1] == mode]
        if not rows:
            continue
        print(f"\ntop FK body errors, {mode}:")
        for row in sorted(rows, key=lambda x: -x[4])[:20]:
            print(f"seg={row[0]} {row[2]:24s} MAE={row[3]:6.2f}cm "
                  f"RMSE={row[4]:6.2f}cm P95={row[5]:6.2f}cm")

    if args.make_gif:
        generate_requested_gifs(
            args,
            available_segment_ids=list(range(len(segments))),
            allowed_modes=fk_modes,
        )


if __name__ == "__main__":
    main()
