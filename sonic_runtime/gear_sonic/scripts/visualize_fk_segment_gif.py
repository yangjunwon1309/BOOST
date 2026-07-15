"""visualize_fk_segment_gif.py — FK link position을 3D로 시각화해 GIF로 저장.

fk_error_analysis.py가 생성한 segment_XX_fk_positions_{mode}.csv를 읽어,
GMR retarget 결과(파란색 스켈레톤)와 측정된 G1 q(주황색 스켈레톤)를
같은 3D 좌표계에 겹쳐 그리고, 시간축을 따라 애니메이션 GIF로 저장한다.

기본은 base_local 모드를 사용한다 — root orientation 차이를 제거하고
순수하게 G1 신체 비례 기준 자세(팔다리 형태)만 비교하기 위함.
root_oriented로 비교하고 싶으면 --mode root_oriented로 전환 가능.

key-link 뼈대 연결은 fk_error_analysis.py의 DEFAULT_KEY_BODIES 구조
(pelvis - torso - 양 어깨/팔꿈치/손목, pelvis - 양 무릎/발끝)를 그대로 따른다.

실행 예:
  python visualize_fk_segment_gif.py \
      --fk-dir ~/gmr_ref_from_session/fk_analysis \
      --segment 2 \
      --output ~/gmr_ref_from_session/fk_analysis/seg02_compare.gif

  # root_oriented 모드, 속도/품질 조절
  python visualize_fk_segment_gif.py \
      --fk-dir ~/gmr_ref_from_session/fk_analysis \
      --segment 5 --mode root_oriented \
      --fps 15 --stride 2 \
      --output ~/gmr_ref_from_session/fk_analysis/seg05_root_oriented.gif
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection 등록용)
import numpy as np


# fk_error_analysis.py의 DEFAULT_KEY_BODIES와 동일한 순서/이름
KEY_BODIES = [
    "pelvis",
    "torso_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link",
    "left_knee_link", "left_toe_link",
    "right_knee_link", "right_toe_link",
]

# 뼈대 연결: (parent, child). pelvis가 루트(원점, FK가 이미 pelvis 기준 상대좌표로 줌)
BONES = [
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


def load_segment_csv(path: Path) -> dict[float, dict[str, dict[str, np.ndarray]]]:
    """time_sec -> body -> {'gmr': (3,), 'q': (3,)} 형태로 로드."""
    frames: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_sec"])
            body = row["body"]
            gmr_pos = np.array([float(row["gmr_x"]), float(row["gmr_y"]), float(row["gmr_z"])])
            q_pos = np.array([float(row["q_x"]), float(row["q_y"]), float(row["q_z"])])
            frames.setdefault(t, {})[body] = {"gmr": gmr_pos, "q": q_pos}
    return frames


def compute_axis_limits(frames: dict, margin: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """모든 프레임/모든 body의 gmr+q position을 모아 공통 axis limit을 구한다."""
    all_pts = []
    for t, bodies in frames.items():
        for body, d in bodies.items():
            all_pts.append(d["gmr"])
            all_pts.append(d["q"])
    all_pts = np.asarray(all_pts)
    lo = all_pts.min(axis=0) - margin
    hi = all_pts.max(axis=0) + margin
    # 세 축 범위를 동일하게 맞춰 비율 왜곡 방지
    center = (lo + hi) / 2
    half_range = max((hi - lo).max() / 2, 0.05)
    lo = center - half_range
    hi = center + half_range
    return lo, hi


def draw_skeleton(ax, bodies: dict[str, dict[str, np.ndarray]], key: str,
                   color: str, label: str, marker_size: float = 30) -> None:
    """key='gmr' 또는 'q' 스켈레톤을 그린다."""
    # 관절 점
    pts = np.array([bodies[b][key] for b in KEY_BODIES if b in bodies])
    names = [b for b in KEY_BODIES if b in bodies]
    if len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=marker_size,
               depthshade=True, label=label)

    # 뼈대 선
    for parent, child in BONES:
        if parent in bodies and child in bodies:
            p = bodies[parent][key]
            c = bodies[child][key]
            ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                    color=color, linewidth=2.0, alpha=0.85)


def make_gif(
    frames: dict[float, dict[str, dict[str, np.ndarray]]],
    output_path: Path,
    fps: int,
    stride: int,
    elev: float,
    azim: float,
    title_prefix: str,
) -> None:
    times = sorted(frames.keys())
    times = times[::stride]
    if not times:
        raise SystemExit("No frames to animate after applying --stride.")

    lo, hi = compute_axis_limits(frames)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(i: int):
        ax.cla()
        t = times[i]
        bodies = frames[t]

        draw_skeleton(ax, bodies, "gmr", color="#1f77b4", label="GMR (retargeted SMPL)")
        draw_skeleton(ax, bodies, "q", color="#ff7f0e", label="Measured G1 q")

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"{title_prefix}  t={t:.2f}s  (frame {i+1}/{len(times)})")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper left", fontsize=8)
        return ax,

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps)
    writer = PillowWriter(fps=fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"[gif] saved {output_path}  ({len(times)} frames @ {fps}fps, "
          f"stride={stride}, span={times[0]:.2f}s~{times[-1]:.2f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize GMR vs measured G1 FK link positions as a 3D GIF."
    )
    p.add_argument("--fk-dir", required=True, type=Path,
                    help="fk_error_analysis.py의 --output-dir (segment_XX_fk_positions_*.csv가 있는 곳).")
    p.add_argument("--segment", type=int, required=True,
                    help="시각화할 segment id (fk_error_analysis.py 콘솔 출력의 seg 번호).")
    p.add_argument("--mode", choices=["base_local", "root_oriented"], default="base_local",
                    help="어느 FK 모드 결과를 쓸지. base_local 권장 — root orientation "
                         "차이를 제거하고 신체 형태(팔다리 자세)만 비교.")
    p.add_argument("--output", type=Path, default=None,
                    help="저장할 .gif 경로. 기본: <fk-dir>/segment_XX_<mode>.gif")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--stride", type=int, default=2,
                    help="프레임 thinning. 1=모든 프레임, 2=한 칸씩 건너뜀 등. "
                         "원본이 50Hz라 stride=2~5 정도가 보기 좋다.")
    p.add_argument("--elev", type=float, default=15.0, help="3D 시점 고도각(도).")
    p.add_argument("--azim", type=float, default=-60.0, help="3D 시점 방위각(도).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.fk_dir / f"segment_{args.segment:02d}_fk_positions_{args.mode}.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            f"Run fk_error_analysis.py first with --fk-modes including "
            f"'{args.mode}', or check --segment/--mode."
        )

    print(f"[gif] loading {csv_path}")
    frames = load_segment_csv(csv_path)
    print(f"[gif] loaded {len(frames)} time steps, "
          f"{len(KEY_BODIES)} key bodies")

    output_path = args.output or (
        args.fk_dir / f"segment_{args.segment:02d}_{args.mode}.gif"
    )
    title_prefix = f"segment {args.segment:02d} ({args.mode})"

    make_gif(
        frames, output_path,
        fps=args.fps, stride=args.stride,
        elev=args.elev, azim=args.azim,
        title_prefix=title_prefix,
    )


if __name__ == "__main__":
    main()
