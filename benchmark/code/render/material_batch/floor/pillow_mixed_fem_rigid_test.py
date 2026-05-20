"""
Mixed FEM (soft) + Rigid simulation test.

- Reads basic_info.json from a case dir and parses density in g/cm^3 or kg/m^3 strings.
- Splits each part into FEM or Rigid by Young's modulus threshold (in GPa).
- Optionally loads extra rigid mesh(es) (e.g. an .obj from results/.../objs/...).
- Applies a -90 deg rotation around X for .glb (configurable) to fix up-axis.
- Preserves the relative XY layout of parts; only lifts the whole group above ground.
- Resolves initial AABB overlaps by pushing along the axis with smallest penetration.
- Camera auto-frames the entire scene bounding box.
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _pyglet_headless_before_genesis() -> None:
    """Genesis 会导入 pyglet；在无 DISPLAY 的 Linux 上 X11/libXrender 可能加载失败
    （AttributeError: NoneType ... XRenderFindVisualFormat）。在 import genesis 前打开
    pyglet 无头模式可规避。若需强制走 X11，可设环境变量 GENESIS_FORCE_X11=1。"""
    if os.environ.get("GENESIS_FORCE_X11"):
        return
    if sys.platform != "linux":
        return
    disp = os.environ.get("DISPLAY", "").strip()
    if disp:
        return
    os.environ.setdefault("PYGLET_HEADLESS", "1")


_pyglet_headless_before_genesis()
import genesis as gs
import trimesh

DEFAULT_CASE_DIR = Path("physx_result/example_case")
DEFAULT_SCALE = 1.0
DEFAULT_E_THRESHOLD_GPA = 1.0
DEFAULT_GLB_ROT_X = 90.0   # degrees, applied around X for glb (Y-up -> Z-up)
DEFAULT_OBJ_ROT_X = 0.0
DEFAULT_TMP_MESH_DIR = Path(__file__).resolve().parent / "_tmp_meshes_fem"


def _safe_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return token or "case"


def make_task_tmp_mesh_dir(base_dir: Path, source_id: str) -> Path:
    """Create a per-process temp mesh directory.

    The batch runner launches many Genesis processes concurrently. Using one
    shared directory with names such as part_0_centered.obj lets workers
    overwrite each other's temporary OBJ files, which can make TetGen process
    the wrong mesh or a partially written mesh. Keep the user-facing
    --tmp-mesh-dir as a root, then isolate each sample/process underneath it.
    """
    return base_dir / f"{_safe_path_token(source_id)}_pid{os.getpid()}"


def _parse_density_to_kgm3(value):
    """Parse kg/m^3 for bare numbers; strings may omit unit or use g/cm^3 / kg/m^3."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported density type: {type(value)}")
    text = value.strip().lower().replace(" ", "")
    m = re.match(r"^([+-]?[0-9]*\.?[0-9]+)(g/cm\^3|kg/m\^3)?$", text)
    if not m:
        raise ValueError(f"Cannot parse density: {value}")
    number = float(m.group(1))
    unit = m.group(2)
    if unit in (None, "kg/m^3"):
        return number
    if unit == "g/cm^3":
        return number * 1000.0
    raise ValueError(f"Unsupported density unit: {value}")


def part_mesh_path(case_dir: Path, label):
    return case_dir / str(label) / f"{label}.glb"


def part_mesh_path_from_scene(scene_dir: Path, label):
    return scene_dir / "objs" / str(label) / f"{label}.glb"


def _normalize_part_record(part: Dict[str, Any], label: Any, mesh_path: Path) -> Dict[str, Any]:
    e_gpa = float(part["Young's Modulus (GPa)"])
    nu = float(part["Poisson's Ratio"])
    rho = _parse_density_to_kgm3(part["density"])
    return {
        "label": label,
        "name": part.get("name", f"part_{label}"),
        "mesh_path": mesh_path,
        "E_gpa": e_gpa,
        "E": e_gpa * 1e9,
        "nu": nu,
        "rho": rho,
    }


def _load_parts_from_data(data: Dict[str, Any], mesh_path_fn) -> List[Dict[str, Any]]:
    parts = data.get("parts", [])
    loaded = []
    for idx, part in enumerate(parts):
        label = part.get("label", idx)
        mesh_path = mesh_path_fn(label)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh for label {label}: {mesh_path}")
        loaded.append(_normalize_part_record(part, label, mesh_path))
    return loaded


def load_parts(case_dir: Path):
    json_path = case_dir / "basic_info.json"
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _load_parts_from_data(data, lambda label: part_mesh_path(case_dir, label))


def load_parts_from_watertight(scene_dir: Path, metric_json_dir: Path):
    json_path = metric_json_dir / f"{scene_dir.name}.json"
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _load_parts_from_data(data, lambda label: part_mesh_path_from_scene(scene_dir, label))


def bake_and_export(mesh_path: Path, scale: float, rot_x_deg: float,
                    tmp_dir: Path, tag: str):
    """Bake rotation/scale, recenter the mesh, and export a temporary OBJ.

    Genesis then loads the temporary mesh with ``pos=local_centroid``,
    ``euler=(0,0,0)``, and ``scale=1``. This avoids the offset coupling between
    a user-provided position and a mesh centroid under rotation.

    Returns: (obj_path, aabb_min, aabb_max, local_centroid)
        - aabb_min/max: AABB before recentering in target world coordinates.
        - local_centroid: centroid before recentering, used as the target part position.
    """
    m = trimesh.load(str(mesh_path), force="mesh")
    if rot_x_deg:
        T = trimesh.transformations.rotation_matrix(
            np.deg2rad(rot_x_deg), [1.0, 0.0, 0.0]
        )
        m.apply_transform(T)
    if scale != 1.0:
        m.apply_scale(scale)

    local_centroid = np.array(m.centroid)
    aabb_min = np.array(m.bounds[0])
    aabb_max = np.array(m.bounds[1])

    m.apply_translation(-local_centroid)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    obj_path = tmp_dir / f"part_{tag}_centered.obj"
    m.export(str(obj_path))
    return obj_path, aabb_min, aabb_max, local_centroid



def classify(part, e_threshold_gpa: float):
    return "rigid" if part["E_gpa"] >= e_threshold_gpa else "fem"


def resolve_overlaps(items, eps=1e-3, max_iter=50):
    """Greedy AABB separation. items have aabb_min, aabb_max, offset (already
    initialized to mesh centroid so original baked offsets are preserved).
    Pushes along the axis with smallest penetration depth (minimum movement).
    """
    n = len(items)
    for _ in range(max_iter):
        any_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                # AABB are in pre-offset local-world coords; the world AABB is
                # aabb + (offset - centroid) since pos=offset means the mesh's
                # centroid moves from `centroid` to `offset`.
                shift_i = items[i]["offset"] - items[i]["centroid"]
                shift_j = items[j]["offset"] - items[j]["centroid"]
                a_min = items[i]["aabb_min"] + shift_i
                a_max = items[i]["aabb_max"] + shift_i
                b_min = items[j]["aabb_min"] + shift_j
                b_max = items[j]["aabb_max"] + shift_j
                overlap = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
                if np.all(overlap > 0):
                    any_overlap = True
                    axis = int(np.argmin(overlap))
                    push = overlap[axis] + eps
                    a_c = 0.5 * (a_min[axis] + a_max[axis])
                    b_c = 0.5 * (b_min[axis] + b_max[axis])
                    if b_c >= a_c:
                        items[j]["offset"][axis] += push / 2
                        items[i]["offset"][axis] -= push / 2
                    else:
                        items[j]["offset"][axis] -= push / 2
                        items[i]["offset"][axis] += push / 2
        if not any_overlap:
            return True
    return False



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vis", action="store_true", default=False)
    ap.add_argument("--cpu", action="store_true", default=False)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    ap.add_argument("--scene-dir", type=Path, default=None,
                    help="Watertight scene dir containing objs/<label>/<label>.glb")
    ap.add_argument("--metric-json-dir", type=Path, default=None,
                    help="Directory containing metric json files named <scene_id>.json")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--e-threshold", type=float, default=DEFAULT_E_THRESHOLD_GPA)
    ap.add_argument("--glb-rot-x", type=float, default=DEFAULT_GLB_ROT_X,
                    help="Degrees rotation around X applied to .glb meshes")
    ap.add_argument("--obj-rot-x", type=float, default=DEFAULT_OBJ_ROT_X,
                    help="Degrees rotation around X applied to .obj extra rigid")
    ap.add_argument("--drop-height", type=float, default=0.3)
    ap.add_argument("--cam-dist-scale", type=float, default=1.8,
                    help="Distance scale for recording camera auto-framing")
    ap.add_argument("--cam-min-dist", type=float, default=1.5,
                    help="Minimum camera distance in recording mode")
    ap.add_argument("--cam-fov", type=float, default=45.0,
                    help="Recording camera field of view (degrees)")
    ap.add_argument("--cam-ground-z", type=float, default=0.0,
                    help="Ground plane z used for framing falling motion")
    ap.add_argument("--cam-ground-margin", type=float, default=0.15,
                    help="Extra margin below ground when framing recording camera")
    ap.add_argument("--cam-lookat-z-bias", type=float, default=-0.1,
                    help="Additive z bias applied to recording camera lookat center")
    ap.add_argument("--extra-rigid", type=Path, default=None)
    ap.add_argument("--extra-rigid-rho", type=float, default=2000.0)
    ap.add_argument("--record", action="store_true", default=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render one frame every N simulation steps when recording",
    )
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Skip rendering during the first N simulation steps",
    )
    ap.add_argument(
        "--video-path",
        type=Path,
        default=Path("benchmark/benchmark_assets/material_videos_v2/floor/example.mp4"),
    )
    ap.add_argument("--tmp-mesh-dir", type=Path, default=DEFAULT_TMP_MESH_DIR,
                    help="Temp directory for centered .obj exports fed to Genesis.")
    args = ap.parse_args()
    if args.render_every < 1:
        raise ValueError("--render-every must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.cam_dist_scale <= 0:
        raise ValueError("--cam-dist-scale must be > 0")
    if args.cam_min_dist <= 0:
        raise ValueError("--cam-min-dist must be > 0")
    if args.cam_fov <= 0 or args.cam_fov >= 180:
        raise ValueError("--cam-fov must be in (0, 180)")
    if args.cam_ground_margin < 0:
        raise ValueError("--cam-ground-margin must be >= 0")

    if args.scene_dir is not None or args.metric_json_dir is not None:
        if args.scene_dir is None or args.metric_json_dir is None:
            raise ValueError("--scene-dir and --metric-json-dir must be set together")
        parts = load_parts_from_watertight(args.scene_dir, args.metric_json_dir)
        print(f"Scene: {args.scene_dir}, metric_json_dir: {args.metric_json_dir}")
        source_id = f"{args.scene_dir.parent.name}_{args.scene_dir.name}"
    else:
        parts = load_parts(args.case_dir)
        print(f"Case: {args.case_dir}")
        source_id = args.case_dir.name
    task_tmp_mesh_dir = make_task_tmp_mesh_dir(args.tmp_mesh_dir, source_id)
    print(f"Temp mesh dir: {task_tmp_mesh_dir}")
    print(f"Loaded {len(parts)} parts. E threshold = {args.e_threshold} GPa")
    for p in parts:
        kind = classify(p, args.e_threshold)
        print(f"  label={p['label']} name={p['name']:32s} "
              f"E={p['E_gpa']:.4f} GPa nu={p['nu']} rho={p['rho']} -> {kind}")

    # Build entity descriptors (parts + optional extra rigid), compute world AABB.
    items = []
    for i, p in enumerate(parts):
        kind = classify(p, args.e_threshold)
        obj_path, a_min, a_max, centroid = bake_and_export(
            p["mesh_path"], args.scale, args.glb_rot_x,
            task_tmp_mesh_dir, tag=str(p["label"]),
        )
        items.append({
            "kind": kind,
            "color_idx": i,
            "mesh_path": obj_path,
            "E": p["E"], "nu": p["nu"], "rho": p["rho"],
            "name": p["name"], "label": p["label"],
            "aabb_min": a_min, "aabb_max": a_max,
            "centroid": centroid,
            "offset": centroid.copy(),  # world-target centroid (preserves relative layout)
        })

    if args.extra_rigid is not None:
        if args.extra_rigid.exists():
            obj_path, a_min, a_max, centroid = bake_and_export(
                args.extra_rigid, args.scale, args.obj_rot_x,
                task_tmp_mesh_dir, tag="extra",
            )
            items.append({
                "kind": "rigid",
                "color_idx": -1,
                "mesh_path": obj_path,
                "rho": args.extra_rigid_rho,
                "name": "extra_rigid",
                "label": "extra",
                "aabb_min": a_min, "aabb_max": a_max,
                "centroid": centroid,
                "offset": centroid.copy(),
            })
        else:
            print(f"[WARN] extra-rigid not found: {args.extra_rigid}")


    # Resolve any initial AABB overlaps by minimum-penetration push.
    ok = resolve_overlaps(items)
    print(f"Overlap resolution converged: {ok}")

    # Lift the whole group above ground (preserve relative XY/Z among parts).
    global_z_min = min(
        (it["aabb_min"][2] + (it["offset"][2] - it["centroid"][2])) for it in items
    )
    dz_lift = -global_z_min + args.drop_height
    for it in items:
        it["offset"][2] += dz_lift


    # Init genesis after we are ready.
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="info")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-4, substeps=10, gravity=(0, 0, -9.8)),
        fem_options=gs.options.FEMOptions(
            damping=20.0, use_implicit_solver=True,
            damping_alpha=3.0, damping_beta=1e-2,
        ),

        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 3.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_up=(0, 0, 1),
            res=(960, 640),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    fem_color = [
        (0.9, 0.75, 0.8, 1.0),
        (0.7, 0.85, 0.95, 1.0),
        (0.85, 0.95, 0.75, 1.0),
    ]
    rigid_color = (0.6, 0.6, 0.65, 1.0)
    extra_color = (0.85, 0.5, 0.3, 1.0)

    for it in items:
        pos = tuple(float(x) for x in it["offset"])
        # Mesh has been pre-rotated, pre-scaled, and re-centered to origin in
        # bake_and_export(); Genesis loads it as-is at world position `pos`.
        morph = gs.morphs.Mesh(
            file=str(it["mesh_path"]),
            pos=pos, scale=1.0, euler=(0.0, 0.0, 0.0),
        )

        if it["kind"] == "fem":
            color = fem_color[it["color_idx"] % len(fem_color)]
            scene.add_entity(
                material=gs.materials.FEM.Elastic(
                    E=it["E"], nu=it["nu"], rho=it["rho"], model="linear_corotated",
                ),
                morph=morph,
                surface=gs.surfaces.Default(color=color, vis_mode="visual"),
            )
        else:
            color = extra_color if it["color_idx"] == -1 else rigid_color
            scene.add_entity(
                material=gs.materials.Rigid(
                    rho=it["rho"], needs_coup=True, coup_friction=0.3,
                ),
                morph=morph,
                surface=gs.surfaces.Default(color=color, vis_mode="visual"),
            )
        print(f"  place [{it['kind']}] {it['name']} pos={pos}")

    # Camera auto-frame from final world AABB.
    cam = None
    if args.record:
        all_min = np.min([
            it["aabb_min"] + (it["offset"] - it["centroid"]) for it in items
        ], axis=0)
        all_max = np.max([
            it["aabb_max"] + (it["offset"] - it["centroid"]) for it in items
        ], axis=0)

        # Expand the recording frustum for falling motion:
        # include ground and a small below-ground margin so objects are less likely
        # to leave the frame after contact/settling.
        all_min[2] = min(all_min[2], args.cam_ground_z - args.cam_ground_margin)
        center = 0.5 * (all_min + all_max)
        center[2] += args.cam_lookat_z_bias
        diag = float(np.linalg.norm(all_max - all_min))
        dist = max(diag * args.cam_dist_scale, args.cam_min_dist)
        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(float(center[0] + dist * 0.7),
                 float(center[1] + dist * 0.7),
                 float(center[2] + dist * 0.6)),
            lookat=(float(center[0]), float(center[1]), float(center[2])),
            up=(0, 0, 1), fov=args.cam_fov, GUI=False,
        )

    scene.build()

    if cam is not None:
        cam.start_recording()

    n_steps = args.steps if "PYTEST_VERSION" not in os.environ else 5
    last_step = -1
    recording_stopped = False
    try:
        for step in range(n_steps):
            last_step = step
            scene.step()
            if cam is not None:
                should_render = (step >= args.warmup_steps) and (
                    ((step - args.warmup_steps) % args.render_every) == 0
                )
                if should_render:
                    cam.render()
            if step % 100 == 0:
                print(f"Step {step}/{n_steps}")
    except Exception as e:
        at = last_step if last_step >= 0 else 0
        print(f"[error] simulation crashed at step {at}/{n_steps}: {e}")
        print(traceback.format_exc())
        if cam is not None:
            print("[save] saving partial video before exit...")
            cam.stop_recording(save_to_filename=str(args.video_path), fps=args.fps)
            recording_stopped = True
            print(f"Partial video saved to: {args.video_path}")
        raise

    if cam is not None and not recording_stopped:
        cam.stop_recording(save_to_filename=str(args.video_path), fps=args.fps)
        print(f"Video saved to: {args.video_path}")


if __name__ == "__main__":
    main()
