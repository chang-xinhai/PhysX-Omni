"""
Multi-part rigid object water drop simulation.

Supports:
- Loading multiple GLB parts from a scene directory (flat or objs/ layout)
- Preserving relative positions from GLB coordinates
- Collision detection and intelligent separation
- Auto-scaling to fit container
- Per-part density from basic_info.json or material_metric_json/origin/{source}/{id}.json

Examples:
  python multi_part_water_drop.py \\
    --scene-dir physx_result/<method>/<object_id> \\
    --metric-json-dir physx_result/material_metric_json/origin/<source_name> \\
    --output-dir benchmark/benchmark_assets/material_videos/water \\
    --solver mpm \\
    --record
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import genesis as gs


def _try_tqdm(iterable, total: Optional[int] = None, desc: str = "", disable: bool = False):
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, disable=disable, file=sys.stderr, dynamic_ncols=True)


def generate_distinct_color(index: int, total: int) -> Tuple[float, float, float, float]:
    """Generate visually distinct colors using HSV color space."""
    if total == 1:
        return (0.8, 0.6, 0.4, 1.0)

    hue = (index / total) % 1.0
    saturation = 0.7
    value = 0.9

    h_i = int(hue * 6)
    f = hue * 6 - h_i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)

    if h_i == 0:
        r, g, b = value, t, p
    elif h_i == 1:
        r, g, b = q, value, p
    elif h_i == 2:
        r, g, b = p, value, t
    elif h_i == 3:
        r, g, b = p, q, value
    elif h_i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q

    return (r, g, b, 1.0)


TANK_HALF_XY = 0.50
WATER_HEIGHT = 0.50
MPM_LOWER = (-TANK_HALF_XY - 0.20, -TANK_HALF_XY - 0.20, -0.20)
MPM_UPPER = (TANK_HALF_XY + 0.20, TANK_HALF_XY + 0.20, 1.50)


def load_basic_info(scene_dir: Path, metric_json_dir: Optional[Path] = None) -> Dict:
    """Load part metadata: prefer metric_json_dir / {scene_dir.name}.json, else scene basic_info.json."""
    if metric_json_dir is not None:
        mp = metric_json_dir / f"{scene_dir.name}.json"
        if mp.is_file():
            with open(mp, "r", encoding="utf-8") as f:
                return json.load(f)
    info_path = scene_dir / "basic_info.json"
    if info_path.is_file():
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parse_density_g_cm3(raw: Any, default: float = 1.0) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return default
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return default


def density_kg_m3_for_part(part_id: int, parts_list: List[Dict]) -> float:
    """Match rigid part id to metric JSON entry by ``label`` (fallback: list index order)."""
    density_g_cm3 = 1.0
    for entry in parts_list:
        label = entry.get("label")
        if label is not None and int(label) == int(part_id):
            density_g_cm3 = _parse_density_g_cm3(entry.get("density", 1.0))
            break
    else:
        if 0 <= part_id < len(parts_list):
            density_g_cm3 = _parse_density_g_cm3(parts_list[part_id].get("density", 1.0))
    return max(density_g_cm3 * 1000.0, 50.0)


def load_parts(scene_dir: Path, rotate_x_deg: float = -90.0) -> List[Dict]:
    """
    Load all GLB parts and apply X-axis rotation.

    Layouts supported:
    - ``scene_dir/objs/<id>/<id>.glb`` (watertight export)
    - ``scene_dir/<id>/<id>.glb`` (legacy flat layout)
    """
    parts: List[Dict] = []

    objs_root = scene_dir / "objs"
    if objs_root.is_dir():
        search_roots = [objs_root]
    else:
        search_roots = [scene_dir]

    for root in search_roots:
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            part_id_str = subdir.name
            if not part_id_str.isdigit():
                continue
            glb_path = subdir / f"{part_id_str}.glb"
            if not glb_path.is_file():
                continue

            mesh = trimesh.load(str(glb_path), force="mesh")

            if rotate_x_deg != 0.0:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    np.radians(rotate_x_deg), [1, 0, 0]
                )
                mesh.apply_transform(rotation_matrix)

            parts.append({
                "id": int(part_id_str),
                "path": glb_path,
                "mesh": mesh,
                "bounds": mesh.bounds.copy(),
                "centroid": mesh.centroid.copy(),
            })

    parts.sort(key=lambda p: p["id"])
    return parts


def compute_overall_bounds(parts: List[Dict]) -> np.ndarray:
    """Compute union bounding box of all parts."""
    all_bounds = np.array([p["bounds"] for p in parts])
    overall_min = all_bounds[:, 0, :].min(axis=0)
    overall_max = all_bounds[:, 1, :].max(axis=0)
    return np.array([overall_min, overall_max])


def check_aabb_collision(bounds_a: np.ndarray, bounds_b: np.ndarray, margin: float = 0.0) -> bool:
    """Check if two AABB boxes collide (with optional margin)."""
    for axis in range(3):
        if bounds_a[1][axis] + margin < bounds_b[0][axis] - margin:
            return False
        if bounds_a[0][axis] - margin > bounds_b[1][axis] + margin:
            return False
    return True


def compute_separation_vector(bounds_a: np.ndarray, bounds_b: np.ndarray) -> np.ndarray:
    """Compute minimal separation vector to resolve collision."""
    overlaps = []
    for axis in range(3):
        overlap_min = min(bounds_a[1][axis], bounds_b[1][axis]) - max(bounds_a[0][axis], bounds_b[0][axis])
        overlaps.append(overlap_min)

    min_axis = int(np.argmin(overlaps))
    min_overlap = overlaps[min_axis]

    center_a = (bounds_a[0] + bounds_a[1]) / 2.0
    center_b = (bounds_b[0] + bounds_b[1]) / 2.0
    direction = np.sign(center_b[min_axis] - center_a[min_axis])

    sep_vec = np.zeros(3)
    sep_vec[min_axis] = direction * (min_overlap / 2.0 + 0.01)
    return sep_vec


def resolve_collisions(parts: List[Dict], collision_margin: float = 0.005, max_iterations: int = 10):
    """Iteratively resolve collisions by moving parts."""
    for iteration in range(max_iterations):
        collision_found = False

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if check_aabb_collision(parts[i]["bounds"], parts[j]["bounds"], collision_margin):
                    collision_found = True
                    sep_vec = compute_separation_vector(parts[i]["bounds"], parts[j]["bounds"])

                    parts[i]["centroid"] -= sep_vec
                    parts[i]["bounds"] -= sep_vec

                    parts[j]["centroid"] += sep_vec
                    parts[j]["bounds"] += sep_vec

        if not collision_found:
            print(f"[collision] resolved in {iteration + 1} iterations")
            break
    else:
        print("[collision] warning: max iterations reached, some collisions may remain")


def auto_scale_to_container(parts: List[Dict], tank_half_xy: float, water_height: float) -> float:
    """Compute scale factor to fit all parts in container."""
    overall_bounds = compute_overall_bounds(parts)

    xy_extent = max(
        overall_bounds[1][0] - overall_bounds[0][0],
        overall_bounds[1][1] - overall_bounds[0][1]
    )
    z_extent = overall_bounds[1][2] - overall_bounds[0][2]

    usable_xy = tank_half_xy * 2.0 * 0.9
    usable_z = water_height * 0.9

    scale_xy = usable_xy / xy_extent if xy_extent > usable_xy else 1.0
    scale_z = usable_z / z_extent if z_extent > usable_z else 1.0

    scale_factor = min(scale_xy, scale_z, 1.0)

    if scale_factor < 1.0:
        print(f"[auto-scale] scaling parts by {scale_factor:.3f} to fit container")
        for part in parts:
            part["centroid"] *= scale_factor
            part["bounds"] *= scale_factor
            part["mesh"].apply_scale(scale_factor)

    return scale_factor


def center_parts_xy(parts: List[Dict]) -> Tuple[float, float]:
    """Translate all parts so union XY center is around tank center (0, 0)."""
    overall_bounds = compute_overall_bounds(parts)
    center_xy = 0.5 * (overall_bounds[0][:2] + overall_bounds[1][:2])
    delta_xy = -center_xy
    if np.linalg.norm(delta_xy) > 0:
        for part in parts:
            part["centroid"][:2] += delta_xy
            part["bounds"][:, :2] += delta_xy
    return float(delta_xy[0]), float(delta_xy[1])


def export_centered_meshes(parts: List[Dict], scene_name: str) -> Path:
    """
    Export centered temp OBJ meshes and store world centroids in part metadata.
    Genesis then loads centered mesh with pos=world_centroid, scale=1.0.
    """
    tmp_dir = Path(tempfile.gettempdir()) / f"water_drop_centered_{scene_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        mesh = part["mesh"].copy()
        world_centroid = np.array(part["centroid"], dtype=float)
        mesh.apply_translation(-world_centroid)
        obj_path = tmp_dir / f"part_{part['id']}_centered.obj"
        mesh.export(str(obj_path))
        part["centered_path"] = obj_path
        part["world_centroid"] = world_centroid
    return tmp_dir


def compute_release_positions(parts: List[Dict], drop_height: float) -> List[Tuple[float, float, float]]:
    """Compute release positions for all parts."""
    overall_bounds = compute_overall_bounds(parts)
    overall_z_min = overall_bounds[0][2]

    release_positions = []
    for part in parts:
        z_offset = part["world_centroid"][2] - overall_z_min
        release_z = WATER_HEIGHT + drop_height + z_offset

        release_positions.append((
            float(part["world_centroid"][0]),
            float(part["world_centroid"][1]),
            release_z
        ))

    return release_positions


def add_container_walls(scene, wall_alpha: float):
    wall_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
    t = 0.02
    h = 0.90
    zc = h / 2.0

    wall_surface = gs.surfaces.Default(color=(0.72, 0.76, 0.82, wall_alpha), vis_mode="visual")

    wall_specs = [
        ((TANK_HALF_XY + t / 2.0, 0.0, zc), (t, TANK_HALF_XY * 2.0 + t * 2.0, h)),
        ((-(TANK_HALF_XY + t / 2.0), 0.0, zc), (t, TANK_HALF_XY * 2.0 + t * 2.0, h)),
        ((0.0, TANK_HALF_XY + t / 2.0, zc), (TANK_HALF_XY * 2.0, t, h)),
        ((0.0, -(TANK_HALF_XY + t / 2.0), zc), (TANK_HALF_XY * 2.0, t, h)),
    ]

    for pos, size in wall_specs:
        scene.add_entity(
            material=wall_material,
            morph=gs.morphs.Box(pos=pos, size=size, fixed=True),
            surface=wall_surface,
        )


def build_mpm_scene(args, parts: List[Dict], release_positions: List[Tuple], basic_info: Dict):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
            gravity=(0, 0, -9.8)
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=MPM_LOWER,
            upper_bound=MPM_UPPER,
            grid_density=32,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.3, 2.3, 1.6),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_up=(0, 0, 1),
            res=(1280, 720),
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    add_container_walls(scene, wall_alpha=args.wall_alpha if args.show_walls else 0.0)

    scene.add_entity(
        material=gs.materials.MPM.Liquid(rho=1000.0, sampler="pbs"),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, WATER_HEIGHT / 2.0),
            size=(TANK_HALF_XY * 1.8, TANK_HALF_XY * 1.8, WATER_HEIGHT),
        ),
        surface=gs.surfaces.Rough(color=(0.15, 0.5, 0.95, args.water_alpha), vis_mode="particle"),
    )

    part_entities = []
    parts_list = basic_info.get("parts", [])
    total_parts = len(parts)

    for i, (part, pos) in enumerate(zip(parts, release_positions)):
        density_kg_m3 = density_kg_m3_for_part(part["id"], parts_list)
        color = generate_distinct_color(i, total_parts)

        entity = scene.add_entity(
            material=gs.materials.Rigid(rho=density_kg_m3, needs_coup=True, coup_friction=0.0),
            morph=gs.morphs.Mesh(file=str(part["centered_path"]), pos=pos, euler=(0, 0, 0), scale=1.0),
            surface=gs.surfaces.Default(color=color, vis_mode="visual"),
        )
        part_entities.append(entity)

    return scene, part_entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--solver", choices=["mpm"], default="mpm", help="Only MPM supported for now")
    parser.add_argument("--scene-dir", type=Path, required=True,
                        help="Scene root (numeric subdirs or objs/ with GLBs)")
    parser.add_argument(
        "--metric-json-dir",
        type=Path,
        default=None,
        help="Directory containing <scene_id>.json (e.g. json_for_metric/origin/dataset_name)",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Video output directory (default: <this_dir>/output)")
    parser.add_argument(
        "--video-filename",
        type=str,
        default=None,
        help="Output mp4 filename only (no path); default multi_part_water_drop_<scene>_mpm.mp4",
    )
    parser.add_argument("--rotate-x", type=float, default=-90.0)
    parser.add_argument("--drop-height", type=float, default=0.12)
    parser.add_argument("--collision-margin", type=float, default=0.005)
    parser.add_argument("--auto-scale", action="store_true", default=False)
    parser.add_argument("--wall-alpha", type=float, default=0.35)
    parser.add_argument("--show-walls", action="store_true", default=False)
    parser.add_argument("--water-alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render one frame every N simulation steps when recording (default: 1 = render every step)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Skip rendering for the first N simulation steps (recording still starts after build)",
    )
    parser.add_argument("--progress-interval", type=int, default=50,
                        help="Print / tqdm update every N simulation steps")
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {args.scene_dir}")
    if args.render_every < 1:
        raise ValueError("--render-every must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")

    out_dir = args.output_dir if args.output_dir is not None else Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_name = args.scene_dir.name
    if args.video_filename:
        video_path = out_dir / args.video_filename
    else:
        video_path = out_dir / f"multi_part_water_drop_{scene_name}_mpm.mp4"

    print(f"[config] scene_dir={args.scene_dir}, solver={args.solver}, rotate_x={args.rotate_x}°", flush=True)
    print(f"[config] metric_json_dir={args.metric_json_dir}, record={args.record}, video={video_path}", flush=True)

    parts = load_parts(args.scene_dir, rotate_x_deg=args.rotate_x)
    if not parts:
        raise ValueError(f"No GLB parts found under {args.scene_dir} (check objs/ or flat layout)")

    print(f"[parts] loaded {len(parts)} parts: {[p['id'] for p in parts]}", flush=True)

    basic_info = load_basic_info(args.scene_dir, metric_json_dir=args.metric_json_dir)

    resolve_collisions(parts, collision_margin=args.collision_margin)

    if args.auto_scale:
        auto_scale_to_container(parts, TANK_HALF_XY, WATER_HEIGHT)

    delta_x, delta_y = center_parts_xy(parts)
    print(f"[geometry] center shift delta_xy=({delta_x:.4f}, {delta_y:.4f})", flush=True)

    tmp_mesh_dir = export_centered_meshes(parts, scene_name=scene_name)
    print(f"[geometry] centered temp meshes at: {tmp_mesh_dir}", flush=True)

    release_positions = compute_release_positions(parts, args.drop_height)

    overall_bounds = compute_overall_bounds(parts)
    print(f"[geometry] overall bounds: min={overall_bounds[0]}, max={overall_bounds[1]}", flush=True)
    print(f"[geometry] release positions: {release_positions}", flush=True)

    print(
        "[sim] gs.init + scene.build() 可能占用较长时间（GPU/CUDA、编译着色器等）...",
        flush=True,
    )
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="info")

    scene, _part_entities = build_mpm_scene(args, parts, release_positions, basic_info)

    cam = None
    if args.record:
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(2.3, 2.3, 1.6),
            lookat=(0.0, 0.0, 0.3),
            up=(0, 0, 1),
            fov=40,
        )

    scene.build()
    print("[sim] scene.build() 完成，开始步进...", flush=True)

    if args.record and cam is not None:
        cam.start_recording()

    n_steps = args.steps if "PYTEST_VERSION" not in os.environ else 5
    print(f"[sim] running {n_steps} steps")

    step_iter = range(n_steps)
    if args.progress_interval > 0:
        step_iter = _try_tqdm(step_iter, total=n_steps, desc=f"sim[{scene_name}]", disable=False)
    has_tqdm_bar = hasattr(step_iter, "set_postfix")

    i = -1
    try:
        for i in step_iter:
            scene.step()
            if cam is not None:
                # Decouple simulation stepping and rendering frequency for speed.
                should_render = (i >= args.warmup_steps) and (((i - args.warmup_steps) % args.render_every) == 0)
                if should_render:
                    cam.render(rgb=True)
            if args.progress_interval > 0 and has_tqdm_bar:
                if (i + 1) % args.progress_interval == 0 or i == 0:
                    step_iter.set_postfix(step=f"{i + 1}/{n_steps}")
            elif args.progress_interval > 0 and (i % args.progress_interval == 0):
                print(f"  step {i}/{n_steps}", file=sys.stderr)
    except Exception as e:
        at = i if i >= 0 else 0
        print(f"[error] simulation crashed at step {at}/{n_steps}: {e}")
        if cam is not None and args.record:
            print("[save] saving partial video before exit...")
            cam.stop_recording(save_to_filename=str(video_path), fps=args.fps)
            print(f"Partial video saved to {video_path}")
        raise

    if cam is not None and args.record:
        cam.stop_recording(save_to_filename=str(video_path), fps=args.fps)
        print(f"Video saved to {video_path}")

    print("Done.")


if __name__ == "__main__":
    main()
