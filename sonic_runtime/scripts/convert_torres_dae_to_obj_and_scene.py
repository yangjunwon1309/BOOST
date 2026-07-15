#!/usr/bin/env python3
"""Prepare the car4/Torres MuJoCo scene from the IsaacLab-style asset.

The source car asset in ``thirdparty/emergence_ggyu_sub`` is authored as
Collada/DAE and IsaacLab converts it through URDF/USD with convex-decomposition
collision.  MuJoCo does not load DAE meshes in this setup, and plain mesh geoms
collide as convex hulls, so this helper builds a MuJoCo-friendly hybrid:

* OBJ visual meshes, transformed to match the IsaacLab upright car frame.
* Optional bodyless collision mode so cabin entry is not blocked by car-board
  hulls such as the steering wheel, dashboard, and cabin trim.
* Per-object/material convex hull OBJ collision pieces generated with SciPy.
* A native MuJoCo hinge for the driver door and handle.

Run from the repository root:

  source .venv_sim/bin/activate
  python scripts/convert_torres_dae_to_obj_and_scene.py --skip-convert

Use ``--skip-convert`` when OBJ files already exist.  Without it, ``assimp`` must
be installed and available on PATH.

To consume an IsaacLab-exported OBJ directory directly:

  python scripts/convert_torres_dae_to_obj_and_scene.py \
    --obj-dir thirdparty/emergence_ggyu_sub/assets/car/torres_mini_proxy/isaaclab/texture_dae

To rebuild from the older assimp-generated OBJ files with richer source
materials:

  python scripts/convert_torres_dae_to_obj_and_scene.py --detail_mode
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError


BODY_SCALE = 1.0996936170828284#*1.2
HANDLE_SCALE = 0.8421896393360485#*1.2
DAE_BAKE_EULER = (math.pi / 2.0, 0.0, 0.0)
OBJ_BAKE_EULER = (0.0, 0.0, 0.0)
MESH_EULER = (0.0, 0.0, 0.0)
# CAR_WORLD_POS = (0.9, 0.0, 0.691892660713678)
CAR_WORLD_POS = (0.9, 0.0, 0.691892660713678)
DOOR_HINGE_POS = (-1.0737739406004048, 0.5477275064157311, 0.0)
DOOR_VISUAL_POS = (1.0737739406004048, -0.5477275064157311, 0.0)
HANDLE_BASE_POS = (1.0637739406004048, -0.5477275064157311, 0.005)
DOOR_LIMIT = 1.5079644737231006
HANDLE_LIMIT = 0.17453
HANDLE_GRASP_REGION_RADIUS = 0.01
DRIVER_ENTRY_CLEARANCE_MIN = np.array([-0.35, 0.0, -0.62], dtype=float)
DRIVER_ENTRY_CLEARANCE_MAX = np.array([0.90, 0.70, 0.30], dtype=float)
DRIVER_ENTRY_MIN_OVERLAP = np.array([0.12, 0.12, 0.12], dtype=float)

@dataclass(frozen=True)
class MeshSpec:
    name: str
    obj_name: str
    dae_candidates: tuple[str, ...]
    scale: float
    max_collision_pieces: int
    min_collision_faces: int


@dataclass(frozen=True)
class VisualPiece:
    path: Path
    material: str | None
    rgba: str | None = None


MESH_SPECS = {
    "body": MeshSpec(
        name="body",
        obj_name="car_board.obj",
        dae_candidates=("car_board.dae", "car-board.dae"),
        scale=BODY_SCALE,
        max_collision_pieces=256,
        min_collision_faces=12,
    ),
    "door": MeshSpec(
        name="door",
        obj_name="car_door.obj",
        dae_candidates=("car_door.dae", "car-door.dae"),
        scale=BODY_SCALE,
        max_collision_pieces=64,
        min_collision_faces=8,
    ),
    "handle": MeshSpec(
        name="handle",
        obj_name="handle.obj",
        dae_candidates=("handle.dae", "_9968005_handle.dae", "9968005-handle.dae"),
        scale=HANDLE_SCALE,
        max_collision_pieces=2,
        min_collision_faces=4,
    ),
    "handle_lock": MeshSpec(
        name="handle_lock",
        obj_name="handle_lock.obj",
        dae_candidates=("handle_lock.dae", "_9968005_lock.dae", "9968005-lock.dae"),
        scale=HANDLE_SCALE,
        max_collision_pieces=2,
        min_collision_faces=4,
    ),
}


def fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def sanitize_name(value: str, max_len: int = 56) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value)
    return (safe.strip("_") or "part")[:max_len]


def material_for_source(source_material: str) -> str:
    material = source_material.lower()
    if "glass" in material:
        return "torres_glass_mat"
    if "tire" in material:
        return "torres_tire_mat"
    if "black" in material:
        return "torres_black_mat"
    if "charcoal" in material or "darkgray" in material or "dark_gray" in material:
        return "torres_charcoal_mat"
    if "carpaint" in material or "flakes" in material:
        return "torres_carpaint_mat"
    if "aluminum" in material or "silver" in material:
        return "torres_silver_mat"
    if "white" in material:
        return "torres_white_mat"
    return "torres_body_mat"


def rotation_matrix_from_euler(euler: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = euler
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def find_deploy_dir(root: Path) -> Path:
    nested = root / "gear_sonic_deploy" / "gear_sonic_deploy"
    flat = root / "gear_sonic_deploy"
    if (nested / "g1").is_dir():
        return nested
    if (flat / "g1").is_dir():
        return flat
    raise SystemExit("Cannot find deploy g1 directory under gear_sonic_deploy.")


def find_dae_dir(root: Path, deploy_dir: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        deploy_dir / "g1" / "meshes" / "torres_mini_proxy" / "texture_dae",
        root / "thirdparty" / "emergence_ggyu_sub" / "assets" / "car4" / "_isaaclab_training_usd" / "input" / "texture_dae",
        root / "thirdparty" / "emergence_ggyu_sub" / "assets" / "car4" / "_isaaclab_usd" / "input" / "texture_dae",
        root / "thirdparty" / "emergence_ggyu_sub" / "assets" / "car4" / "texture_dae",
    ]
    for candidate in candidates:
        if all(any((candidate / dae_name).is_file() for dae_name in spec.dae_candidates) for spec in MESH_SPECS.values()):
            return candidate
    msg = ["Could not find all Torres/car4 DAE files. Tried:"]
    msg += [f"  - {candidate}" for candidate in candidates]
    raise SystemExit("\n".join(msg))


def find_obj_dir(root: Path, deploy_dir: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        root
        / "thirdparty"
        / "emergence_ggyu_sub"
        / "assets"
        / "car"
        / "torres_mini_proxy"
        / "isaaclab"
        / "texture_dae",
        deploy_dir / "g1" / "meshes" / "torres_mini_proxy" / "obj",
    ]
    for candidate in candidates:
        if all((candidate / spec.obj_name).is_file() for spec in MESH_SPECS.values()):
            return candidate
    msg = ["Could not find all Torres OBJ files. Tried:"]
    msg += [f"  - {candidate}" for candidate in candidates]
    raise SystemExit("\n".join(msg))


def find_source_dae(dae_dir: Path, spec: MeshSpec) -> Path:
    for dae_name in spec.dae_candidates:
        candidate = dae_dir / dae_name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing DAE for {spec.name} in {dae_dir}")


def run_assimp(assimp: str, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.obj")
    if tmp.exists():
        tmp.unlink()
    cmd = [assimp, "export", str(src), str(tmp)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not tmp.is_file() or tmp.stat().st_size == 0:
        raise RuntimeError(f"assimp did not create {tmp}")
    tmp.replace(dst)


def parse_mtl_colors(path: Path) -> dict[str, str]:
    colors: dict[str, str] = {}
    current: str | None = None
    kd = (0.55, 0.57, 0.54)
    alpha = 1.0

    def flush() -> None:
        if current is None:
            return
        colors[current] = f"{kd[0]:.9g} {kd[1]:.9g} {kd[2]:.9g} {alpha:.9g}"

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith("newmtl "):
                flush()
                current = line.split(None, 1)[1].strip()
                kd = (0.55, 0.57, 0.54)
                alpha = 1.0
            elif line.startswith("Kd "):
                parts = line.split()
                if len(parts) >= 4:
                    kd = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif line.startswith("d "):
                parts = line.split()
                if len(parts) >= 2:
                    alpha = float(parts[1])
            elif line.startswith("Tr "):
                parts = line.split()
                if len(parts) >= 2:
                    alpha = 1.0 - float(parts[1])
    flush()
    return colors


def collect_obj_material_colors(obj_paths: Iterable[Path]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for obj_path in obj_paths:
        with obj_path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                if not line.startswith("mtllib "):
                    continue
                for mtl_name in line.split()[1:]:
                    mtl_path = obj_path.parent / mtl_name
                    if mtl_path.is_file():
                        colors.update(parse_mtl_colors(mtl_path))
    return colors


def bake_obj_orientation(src: Path, dst: Path, rotation: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="replace") as infile, dst.open("w", encoding="utf-8") as outfile:
        outfile.write("# MuJoCo-upright double-sided OBJ generated from raw converted OBJ\n")
        outfile.write(f"# Source: {src.as_posix()}\n")
        for line in infile:
            if line.startswith("v "):
                parts = line.split()
                vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                vec = rotation @ vec
                outfile.write(f"v {vec[0]:.9g} {vec[1]:.9g} {vec[2]:.9g}\n")
            elif line.startswith("vn "):
                parts = line.split()
                normal = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                normal = rotation @ normal
                norm = np.linalg.norm(normal)
                if norm > 0.0:
                    normal = normal / norm
                outfile.write(f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n")
            elif line.startswith("vt "):
                outfile.write(line)
            elif line.startswith("f "):
                tokens = [token for token in line.split()[1:] if token.split("/", 1)[0]]
                if len(tokens) >= 3:
                    outfile.write("f " + " ".join(tokens) + "\n")
                    outfile.write("f " + " ".join(reversed(tokens)) + "\n")
            elif line.startswith("mtllib "):
                # MuJoCo materials are assigned in MJCF.  Leaving stale MTL names
                # from assimp can make debugging confusing, so omit them.
                continue
            else:
                outfile.write(line)


def parse_obj_groups(path: Path) -> tuple[np.ndarray, dict[str, set[int]], dict[str, int]]:
    vertices: list[tuple[float, float, float]] = []
    group_vertices: dict[str, set[int]] = defaultdict(set)
    group_faces: dict[str, int] = defaultdict(int)
    current_object = "default"
    current_group = "default"
    current_material = "default"

    def collision_group_name() -> str:
        parts = []
        if current_object != "default":
            parts.append(current_object)
        if current_group != "default":
            parts.append(current_group)
        if current_material != "default":
            parts.append(current_material)
        return "__".join(parts) or "default"

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("o "):
                current_object = line[2:].strip() or "default"
            elif line.startswith("g "):
                current_group = line[2:].strip() or "default"
            elif line.startswith("usemtl "):
                current_material = line.split(None, 1)[1].strip() or "default"
            elif line.startswith("f "):
                indices: list[int] = []
                for token in line.split()[1:]:
                    raw = token.split("/", 1)[0]
                    if not raw:
                        continue
                    idx = int(raw)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    indices.append(idx)
                if len(indices) < 3:
                    continue
                group_name = collision_group_name()
                group_vertices[group_name].update(indices)
                group_faces[group_name] += 1

    if not vertices:
        raise RuntimeError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64), group_vertices, group_faces


def parse_obj_vertex_index(token: str, vertex_count: int) -> int | None:
    raw = token.split("/", 1)[0]
    if not raw:
        return None
    idx = int(raw)
    if idx < 0:
        return vertex_count + idx
    return idx - 1


def split_visual_meshes_by_group_material(
    obj_path: Path,
    out_dir: Path,
    part_name: str,
    *,
    min_faces: int = 4,
    material_rgba: dict[str, str] | None = None,
) -> list[VisualPiece]:
    vertices: list[tuple[float, float, float]] = []
    pieces: dict[tuple[str, str], list[list[int]]] = {}
    current_group = "default"
    current_material = "default"

    with obj_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("g "):
                current_group = line[2:].strip() or "default"
            elif line.startswith("usemtl "):
                current_material = line.split(None, 1)[1].strip() or "default"
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.split()[1:]:
                    idx = parse_obj_vertex_index(token, len(vertices))
                    if idx is not None:
                        face.append(idx)
                if len(face) >= 3:
                    pieces.setdefault((current_group, current_material), []).append(face)

    visual_pieces: list[VisualPiece] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for piece_id, ((group_name, source_material), faces) in enumerate(pieces.items()):
        if len(faces) < min_faces:
            continue

        local_index: dict[int, int] = {}
        local_vertices: list[tuple[float, float, float]] = []
        local_faces: list[list[int]] = []
        for face in faces:
            local_face: list[int] = []
            for global_idx in face:
                if global_idx < 0 or global_idx >= len(vertices):
                    continue
                if global_idx not in local_index:
                    local_index[global_idx] = len(local_vertices) + 1
                    local_vertices.append(vertices[global_idx])
                local_face.append(local_index[global_idx])
            if len(local_face) >= 3:
                local_faces.append(local_face)

        if len(local_vertices) < 3 or not local_faces:
            continue

        safe_group = sanitize_name(group_name, max_len=40)
        safe_mat = sanitize_name(source_material, max_len=32)
        out_path = out_dir / f"{part_name}_{piece_id:03d}_{safe_group}_{safe_mat}.obj"
        with out_path.open("w", encoding="utf-8") as file:
            file.write("# MuJoCo visual mesh split from source OBJ group/material\n")
            file.write(f"# Source: {obj_path.as_posix()}\n")
            file.write(f"# Group: {group_name}\n")
            file.write(f"# Source material: {source_material}\n")
            for x, y, z in local_vertices:
                file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for face in local_faces:
                file.write("f " + " ".join(str(idx) for idx in face) + "\n")
        rgba = material_rgba.get(source_material) if material_rgba is not None else None
        visual_pieces.append(
            VisualPiece(
                out_path,
                material=None if rgba is not None else material_for_source(source_material),
                rgba=rgba,
            )
        )

    return visual_pieces


def group_score(points: np.ndarray, face_count: int, scale: float) -> float:
    extent = (points.max(axis=0) - points.min(axis=0)) * scale
    if np.count_nonzero(extent > 0.015) < 2:
        return 0.0
    volume = float(np.prod(np.maximum(extent, 0.015)))
    return volume * (1.0 + math.log1p(face_count))


def select_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points

    extrema: set[int] = set()
    for axis in range(3):
        extrema.add(int(np.argmin(points[:, axis])))
        extrema.add(int(np.argmax(points[:, axis])))

    rng = np.random.default_rng(seed)
    remaining = np.setdiff1d(np.arange(len(points)), np.fromiter(extrema, dtype=np.int64), assume_unique=False)
    sample_count = max(0, max_points - len(extrema))
    if len(remaining) > sample_count:
        remaining = rng.choice(remaining, size=sample_count, replace=False)
    chosen = np.concatenate([np.fromiter(extrema, dtype=np.int64), remaining])
    return points[np.sort(chosen)]


def write_convex_hull_obj(points: np.ndarray, path: Path, max_points: int, seed: int) -> bool:
    unique = np.unique(np.round(points, decimals=8), axis=0)
    if len(unique) < 4:
        return False
    unique = select_points(unique, max_points=max_points, seed=seed)
    try:
        hull = ConvexHull(unique, qhull_options="QJ Pp")
    except QhullError:
        return False

    if len(hull.vertices) < 4 or len(hull.simplices) < 4:
        return False

    hull_points = unique
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("# Generated convex hull collision mesh for MuJoCo\n")
        for x, y, z in hull_points:
            file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for tri in hull.simplices:
            a, b, c = tri + 1
            file.write(f"f {a} {b} {c}\n")
    return True


def generate_collision_meshes(
    obj_path: Path,
    out_dir: Path,
    spec: MeshSpec,
    max_hull_points: int,
) -> list[Path]:
    vertices, group_vertices, group_faces = parse_obj_groups(obj_path)
    scored_groups: list[tuple[float, str, np.ndarray]] = []

    for group_name, indices in group_vertices.items():
        face_count = group_faces[group_name]
        if face_count < spec.min_collision_faces or len(indices) < 4:
            continue
        idx = np.fromiter(indices, dtype=np.int64)
        points = vertices[idx]
        score = group_score(points, face_count=face_count, scale=spec.scale)
        if score <= 0.0:
            continue
        scored_groups.append((score, group_name, points))

    scored_groups.sort(key=lambda item: item[0], reverse=True)
    selected = scored_groups[: spec.max_collision_pieces]

    written: list[Path] = []
    for piece_id, (_, group_name, points) in enumerate(selected):
        safe_group = "".join(ch if ch.isalnum() else "_" for ch in group_name)[:48]
        rel_path = out_dir / f"{spec.name}_{piece_id:03d}_{safe_group}.obj"
        if write_convex_hull_obj(points, rel_path, max_points=max_hull_points, seed=piece_id + len(spec.name) * 1000):
            written.append(rel_path)

    return written


def clean_collision_dir(collision_dir: Path) -> None:
    collision_dir.mkdir(parents=True, exist_ok=True)
    for child in collision_dir.iterdir():
        if child.is_file() and child.suffix.lower() == ".obj":
            child.unlink()


def clean_visual_dir(visual_dir: Path) -> None:
    visual_dir.mkdir(parents=True, exist_ok=True)
    for child in visual_dir.iterdir():
        if child.is_file() and child.suffix.lower() == ".obj":
            child.unlink()


def mesh_asset(
    name: str,
    file_path: str,
    scale: float,
    maxhullvert: int | None = None,
    inertia: str | None = None,
) -> str:
    extra = f' maxhullvert="{maxhullvert}"' if maxhullvert is not None else ""
    if inertia is not None:
        extra += f' inertia="{inertia}"'
    scale_str = f"{scale:.16g} {scale:.16g} {scale:.16g}"
    return f'    <mesh name="{name}" file="{file_path}" scale="{scale_str}"{extra}/>'


def geom_mesh(
    name: str,
    mesh: str,
    *,
    pos: tuple[float, float, float] | None = None,
    material: str | None = None,
    rgba: str | None = None,
    contact: bool,
    contype: int | None = None,
    conaffinity: int | None = None,
    density_zero: bool = True,
) -> str:
    attrs = [f'name="{name}"', 'type="mesh"', f'mesh="{mesh}"', f'euler="{fmt(MESH_EULER)}"']
    if pos is not None:
        attrs.append(f'pos="{fmt(pos)}"')
    if material is not None:
        attrs.append(f'material="{material}"')
    if rgba is not None:
        attrs.append(f'rgba="{rgba}"')
    if contact:
        attrs.append(f'contype="{1 if contype is None else contype}"')
        attrs.append(f'conaffinity="{1 if conaffinity is None else conaffinity}"')
        attrs.append('group="3"')
    else:
        attrs.append('contype="0" conaffinity="0"')
    if density_zero:
        attrs.append('density="0"')
    return "        <geom " + " ".join(attrs) + "/>"


def obj_aabb_center(paths: Iterable[Path], scale: float) -> tuple[float, float, float]:
    mins, maxs = [], []
    for path in paths:
        if not path.is_file():
            continue
        vertices, _, _ = parse_obj_groups(path)
        mins.append(vertices.min(axis=0))
        maxs.append(vertices.max(axis=0))

    if not mins:
        return (0.0, 0.0, 0.0)

    center = (np.min(np.vstack(mins), axis=0) + np.max(np.vstack(maxs), axis=0)) * 0.5
    center = center * scale
    return (float(center[0]), float(center[1]), float(center[2]))


def obj_scaled_aabb(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    vertices, _, _ = parse_obj_groups(path)
    vertices = vertices * scale
    return vertices.min(axis=0), vertices.max(axis=0)


def blocks_driver_entry_clearance(path: Path, scale: float) -> bool:
    mins, maxs = obj_scaled_aabb(path, scale)
    overlap = np.minimum(maxs, DRIVER_ENTRY_CLEARANCE_MAX) - np.maximum(
        mins, DRIVER_ENTRY_CLEARANCE_MIN
    )
    return bool(np.all(overlap >= DRIVER_ENTRY_MIN_OVERLAP))


def filter_driver_entry_collision_hulls(paths: list[Path], scale: float) -> tuple[list[Path], list[Path]]:
    kept: list[Path] = []
    removed: list[Path] = []
    for path in paths:
        if blocks_driver_entry_clearance(path, scale):
            removed.append(path)
        else:
            kept.append(path)
    return kept, removed


def scene_xml(
    visual_files: dict[str, list[VisualPiece]],
    collision_files: dict[str, list[Path]],
    collision_alpha: float,
    show_visual_hulls: bool,
    car_collision_mode: str,
) -> str:
    body_scale = BODY_SCALE
    handle_scale = HANDLE_SCALE
    mesh_dir = Path("torres_mini_proxy")
    collision_mat = f"1.0 0.55 0.1 {collision_alpha:g}"
    handle_region_pos = obj_aabb_center(collision_files.get("handle", []), handle_scale)
    include_hull_collisions = car_collision_mode == "hulls"

    asset_lines = [
        mesh_asset("torres_handle_visual_mesh", "torres_mini_proxy/mujoco_obj/handle.obj", handle_scale),
        mesh_asset("torres_handle_lock_visual_mesh", "torres_mini_proxy/mujoco_obj/handle_lock.obj", handle_scale),
    ]

    geom_lines_by_part: dict[str, list[str]] = defaultdict(list)
    visual_lines_by_part: dict[str, list[str]] = defaultdict(list)
    visual_hull_lines_by_part: dict[str, list[str]] = defaultdict(list)

    for part_name in ("body", "door"):
        spec = MESH_SPECS[part_name]
        for piece_id, piece in enumerate(visual_files.get(part_name, [])):
            mesh_name = f"torres_{part_name}_visual_mesh_{piece_id:03d}"
            rel_file = (mesh_dir / "visual" / piece.path.name).as_posix()
            asset_lines.append(mesh_asset(mesh_name, rel_file, spec.scale, inertia="shell"))
            visual_lines_by_part[part_name].append(
                geom_mesh(
                    f"torres_{part_name}_visual_{piece_id:03d}",
                    mesh_name,
                    contact=False,
                    material=piece.material,
                    rgba=piece.rgba,
                )
            )

    if not visual_lines_by_part["body"]:
        asset_lines.append(mesh_asset("torres_body_visual_mesh", "torres_mini_proxy/mujoco_obj/car_board.obj", body_scale))
        visual_lines_by_part["body"].append(
            geom_mesh(
                "torres_body_visual",
                "torres_body_visual_mesh",
                contact=False,
                material="torres_body_mat",
            )
        )

    if not visual_lines_by_part["door"]:
        asset_lines.append(mesh_asset("torres_door_visual_mesh", "torres_mini_proxy/mujoco_obj/car_door.obj", body_scale))
        visual_lines_by_part["door"].append(
            geom_mesh(
                "torres_door_visual",
                "torres_door_visual_mesh",
                contact=False,
                material="torres_door_mat",
            )
        )

    for part_name, files in collision_files.items():
        include_part_collision = include_hull_collisions or (
            car_collision_mode == "bodyless" and part_name != "body"
        )
        needs_hull_asset = include_part_collision or (
            show_visual_hulls and part_name in {"body", "door"}
        )
        if not needs_hull_asset:
            continue

        spec = MESH_SPECS[part_name]
        collision_type = {
            "body": 2,
            "door": 4,
            "handle": 8,
            "handle_lock": 16,
        }[part_name]
        for piece_id, path in enumerate(files):
            mesh_name = f"torres_{part_name}_collision_mesh_{piece_id:03d}"
            rel_file = (mesh_dir / "collision" / path.name).as_posix()
            asset_lines.append(mesh_asset(mesh_name, rel_file, spec.scale, maxhullvert=64))
            if show_visual_hulls and part_name in {"body", "door"}:
                visual_hull_lines_by_part[part_name].append(
                    geom_mesh(
                        f"torres_{part_name}_visual_hull_{piece_id:03d}",
                        mesh_name,
                        contact=False,
                        material="torres_body_hull_mat" if part_name == "body" else "torres_door_hull_mat",
                    )
                )
            if include_part_collision:
                geom_lines_by_part[part_name].append(
                    geom_mesh(
                        f"torres_{part_name}_collision_{piece_id:03d}",
                        mesh_name,
                        contact=True,
                        material="torres_collision_mat",
                        contype=collision_type,
                        conaffinity=1,
                    )
                )

    body_collision_geoms = "\n".join(geom_lines_by_part["body"])
    body_visual_geoms = "\n".join(visual_lines_by_part["body"])
    body_visual_hull_geoms = "\n".join(visual_hull_lines_by_part["body"])
    door_visual_geoms = "\n".join(
        line.replace("<geom ", f'<geom pos="{fmt(DOOR_VISUAL_POS)}" ', 1)
        for line in visual_lines_by_part["door"]
    )
    door_collision_geoms = "\n".join(
        line.replace("<geom ", f'<geom pos="{fmt(DOOR_VISUAL_POS)}" ', 1)
        for line in geom_lines_by_part["door"]
    )
    door_visual_hull_geoms = "\n".join(
        line.replace("<geom ", f'<geom pos="{fmt(DOOR_VISUAL_POS)}" ', 1)
        for line in visual_hull_lines_by_part["door"]
    )
    lock_collision_geoms = "\n".join(geom_lines_by_part["handle_lock"])
    handle_collision_geoms = "\n".join(geom_lines_by_part["handle"])
    asset_block = "\n".join(asset_lines)

    return f'''<mujoco model="g1_29dof_torres_mesh_hinge_scene">
  <!--
    G1 + IsaacLab-aligned car4/Torres visual mesh + configurable collision scene.

    Generated by scripts/convert_torres_dae_to_obj_and_scene.py.
    The original DAE meshes are converted to OBJ for MuJoCo.  By default,
    collision uses generated convex hull pieces; bodyless collision mode keeps
    the car-board/base hulls visual-only while retaining door and handle
    collision.
  -->
  <include file="g1_29dof_with_hand.xml"/>

  <statistic center="0.25 0.35 0.75" extent="4.5"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="torres_body_mat" rgba="0.55 0.57 0.54 1" emission="0.18"/>
    <material name="torres_door_mat" rgba="0.48 0.50 0.48 1" emission="0.18"/>
    <material name="torres_carpaint_mat" rgba="0.28 0.32 0.27 1" emission="0.08"/>
    <material name="torres_charcoal_mat" rgba="0.10 0.11 0.10 1" emission="0.04"/>
    <material name="torres_black_mat" rgba="0.025 0.025 0.022 1"/>
    <material name="torres_tire_mat" rgba="0.015 0.015 0.014 1"/>
    <material name="torres_silver_mat" rgba="0.62 0.64 0.61 1" emission="0.05"/>
    <material name="torres_white_mat" rgba="0.82 0.84 0.80 1" emission="0.06"/>
    <material name="torres_glass_mat" rgba="0.13 0.20 0.24 0.42" emission="0.04"/>
    <material name="torres_body_hull_mat" rgba="0.38 0.40 0.38 1" emission="0.12"/>
    <material name="torres_door_hull_mat" rgba="0.46 0.48 0.46 1" emission="0.12"/>
    <material name="torres_handle_mat" rgba="0.72 0.72 0.68 1"/>
    <material name="torres_handle_grasp_region_mat" rgba="0.1 0.8 1.0 0.28" emission="0.25"/>
    <material name="seat_support_mat" rgba="0 0 0 0"/>
    <material name="torres_collision_mat" rgba="{collision_mat}"/>

{asset_block}
  </asset>

  <worldbody>
    <light pos="0 0 3.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.8 0.02 0.01"/>

    <body name="torres_car4" pos="{fmt(CAR_WORLD_POS)}">
{body_visual_geoms}
{body_visual_hull_geoms}
{body_collision_geoms}

      <body name="torres_driver_door_body" pos="{fmt(DOOR_HINGE_POS)}">
        <inertial pos="0 -0.32 0" mass="18" diaginertia="0.55 0.45 0.12"/>
        <joint name="torres_driver_door_hinge" type="hinge" axis="0 0 -1" range="0 {DOOR_LIMIT:.10g}"
          limited="true" damping="8" frictionloss="0.8" armature="0.02"/>
{door_visual_geoms}
{door_visual_hull_geoms}
{door_collision_geoms}

        <body name="torres_handle_lock_body" pos="{fmt(HANDLE_BASE_POS)}">
          <inertial pos="0.08 0 0" mass="0.6" diaginertia="0.002 0.002 0.002"/>
          <geom name="torres_handle_lock_visual" type="mesh" mesh="torres_handle_lock_visual_mesh"
            euler="{fmt(MESH_EULER)}" material="torres_handle_mat" contype="0" conaffinity="0" density="0"/>
{lock_collision_geoms}
          <body name="torres_handle_body" pos="0 0 0">
            <inertial pos="0.08 0 0" mass="0.4" diaginertia="0.001 0.001 0.001"/>
            <joint name="torres_driver_handle_hinge" type="hinge" axis="0 -1 0" range="0 {HANDLE_LIMIT:.10g}"
              limited="true" damping="1.5" frictionloss="0.15" armature="0.002"/>
            <geom name="torres_handle_grasp_region_marker" type="sphere" pos="{fmt(handle_region_pos)}"
              size="{HANDLE_GRASP_REGION_RADIUS:.10g}" material="torres_handle_grasp_region_mat"
              contype="0" conaffinity="0" density="0"/>
            <geom name="torres_handle_visual" type="mesh" mesh="torres_handle_visual_mesh"
              euler="{fmt(MESH_EULER)}" material="torres_handle_mat" contype="0" conaffinity="0" density="0"/>
{handle_collision_geoms}
          </body>
        </body>
      </body>

      <!-- Seat-only collision support; the visible car shell stays visual-only. -->
      <geom name="driver_seat_cushion_support" type="box" pos="0.02 0.67 -0.3973090552"
        size="0.30 0.25 0.055" material="seat_support_mat" contype="32" conaffinity="1" group="3" friction="1.0 0.02 0.01"/>
      <geom name="driver_seat_back_support" type="box" pos="-0.26 0.67 -0.1473090552"
        size="0.055 0.25 0.30" euler="0 -0.22 0" material="seat_support_mat" contype="32" conaffinity="1" group="3"/>

      <site name="driver_doorway_target" type="sphere" pos="0.18 0.29 -0.4773090552" size="0.001" rgba="1.0 0.55 0.1 0"/>
      <site name="driver_seat_target" type="sphere" pos="0.02 0.67 -0.3173090552" size="0.001" rgba="0.1 0.8 0.2 0"/>
    </body>

    <site name="com_marker" pos="0.1 0 0" size="0.001" rgba="1 0 0 0" type="sphere"/>
  </worldbody>

  <default>
    <geom friction="0.7 0.02 0.01"/>
  </default>
</mujoco>
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="GR00T-WholeBodyControl root")
    ap.add_argument("--dae-dir", default=None, help="Directory containing car4/Torres DAE files")
    ap.add_argument("--obj-dir", default=None, help="Directory containing car_board/car_door/handle/handle_lock OBJ files")
    ap.add_argument(
        "--detail-mode",
        "--detail_mode",
        action="store_true",
        dest="detail_mode",
        help="use the older deploy OBJ files with their source MTL colors and dense collision pieces",
    )
    ap.add_argument("--assimp", default=None, help="assimp executable path; defaults to PATH lookup")
    ap.add_argument("--skip-convert", action="store_true", help="only regenerate collision meshes and scene XML")
    ap.add_argument("--skip-collision", action="store_true", help="do not regenerate convex collision OBJ files")
    ap.add_argument("--max-hull-points", type=int, default=2500, help="max sampled points per generated convex hull")
    ap.add_argument("--collision-alpha", type=float, default=0.0, help="debug alpha for generated collision geoms")
    ap.add_argument("--show-visual-hulls", action="store_true", help="draw convex collision hulls as a visual fallback")
    ap.add_argument(
        "--car-collision-mode",
        choices=("bodyless", "hulls"),
        default="hulls",
        help="bodyless removes car-board/base collision while retaining door/handle hulls",
    )
    ap.add_argument(
        "--keep-entry-collision-hulls",
        action="store_true",
        help="keep body collision hulls that intrude into the driver doorway/cabin clearance volume",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    deploy_dir = find_deploy_dir(root)
    mesh_root = deploy_dir / "g1" / "meshes" / "torres_mini_proxy"
    obj_dir = mesh_root / "obj"
    collision_dir = mesh_root / "collision"
    visual_dir = mesh_root / "visual"
    obj_dir.mkdir(parents=True, exist_ok=True)

    if args.detail_mode and args.obj_dir:
        raise SystemExit("--detail_mode and --obj-dir are mutually exclusive.")
    if args.detail_mode and args.dae_dir:
        raise SystemExit("--detail_mode and --dae-dir are mutually exclusive.")

    source_obj_dir: Path | None = None
    dae_dir: Path | None = None
    source_mode = "current IsaacLab OBJ"
    bake_euler = OBJ_BAKE_EULER
    if args.detail_mode:
        source_obj_dir = find_obj_dir(root, deploy_dir, obj_dir.as_posix())
        source_mode = "detail deploy OBJ"
        bake_euler = DAE_BAKE_EULER
    elif args.obj_dir:
        source_obj_dir = find_obj_dir(root, deploy_dir, args.obj_dir)
        source_mode = "explicit OBJ"
    elif args.dae_dir:
        dae_dir = find_dae_dir(root, deploy_dir, args.dae_dir)
        source_mode = "DAE conversion"
        bake_euler = DAE_BAKE_EULER
    else:
        source_obj_dir = find_obj_dir(root, deploy_dir, None)

    assimp = args.assimp or shutil.which("assimp")
    if not assimp and not args.skip_convert and source_obj_dir is None:
        raise SystemExit("assimp not found. Use --skip-convert with existing OBJ files, pass --obj-dir, or install assimp-utils.")

    print(f"Root:          {root}")
    print(f"Deploy dir:    {deploy_dir}")
    print(f"Source mode:   {source_mode}")
    if dae_dir is not None:
        print(f"DAE dir:       {dae_dir}")
    if source_obj_dir is not None:
        print(f"Source OBJ dir:{source_obj_dir}")
    print(f"OBJ dir:       {obj_dir}")
    mujoco_obj_dir = mesh_root / "mujoco_obj"
    print(f"MuJoCo OBJ dir:{mujoco_obj_dir}")
    print(f"Visual dir:    {visual_dir}")
    print(f"Collision dir: {collision_dir}")
    print(f"Collision mode:{args.car_collision_mode}")
    print(f"Baked euler:   {fmt(bake_euler)}")
    print(f"Car pos:       {fmt(CAR_WORLD_POS)}")

    if source_obj_dir is not None:
        raw_obj_paths = {name: source_obj_dir / spec.obj_name for name, spec in MESH_SPECS.items()}
    else:
        if not args.skip_convert:
            assert dae_dir is not None
            assert assimp is not None
            for spec in MESH_SPECS.values():
                run_assimp(assimp, find_source_dae(dae_dir, spec), obj_dir / spec.obj_name)
        raw_obj_paths = {name: obj_dir / spec.obj_name for name, spec in MESH_SPECS.items()}

    missing = [path for path in raw_obj_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing OBJ files: " + ", ".join(str(path) for path in missing))
    material_rgba = collect_obj_material_colors(raw_obj_paths.values())
    if material_rgba:
        print(f"Source MTL colors: {len(material_rgba)}")

    rotation = rotation_matrix_from_euler(bake_euler)
    obj_paths = {name: mujoco_obj_dir / spec.obj_name for name, spec in MESH_SPECS.items()}
    for name, raw_path in raw_obj_paths.items():
        bake_obj_orientation(raw_path, obj_paths[name], rotation)

    clean_visual_dir(visual_dir)
    visual_files: dict[str, list[VisualPiece]] = {name: [] for name in MESH_SPECS}
    for name in ("body", "door"):
        visual_files[name] = split_visual_meshes_by_group_material(
            obj_paths[name],
            visual_dir,
            name,
            material_rgba=material_rgba,
        )
        print(f"Generated split visual pieces for {name}: {len(visual_files[name])}")

    collision_files: dict[str, list[Path]] = {name: [] for name in MESH_SPECS}
    if not args.skip_collision:
        clean_collision_dir(collision_dir)
        for name, spec in MESH_SPECS.items():
            print(f"Generating convex collision pieces for {name}...")
            collision_files[name] = generate_collision_meshes(
                obj_paths[name],
                collision_dir,
                spec,
                max_hull_points=args.max_hull_points,
            )
            print(f"  wrote {len(collision_files[name])} pieces")
    else:
        for name in MESH_SPECS:
            collision_files[name] = sorted(collision_dir.glob(f"{name}_*.obj"))

    if not args.keep_entry_collision_hulls:
        collision_files["body"], removed = filter_driver_entry_collision_hulls(
            collision_files["body"],
            MESH_SPECS["body"].scale,
        )
        if removed:
            print(
                "Filtered driver-entry body collision hulls: "
                + ", ".join(path.name for path in removed)
            )

    scene_path = deploy_dir / "g1" / "scene_29dof_torres_mesh_hinge.xml"
    scene_path.write_text(
        scene_xml(
            visual_files,
            collision_files,
            collision_alpha=args.collision_alpha,
            show_visual_hulls=args.show_visual_hulls,
            car_collision_mode=args.car_collision_mode,
        ),
        encoding="utf-8",
    )
    print(f"Wrote scene: {scene_path}")
    print("Next:")
    print("  source .venv_sim/bin/activate")
    print("  python gear_sonic/scripts/run_g1_torres_mesh_hinge_teleop_mujoco.py")


if __name__ == "__main__":
    main()
