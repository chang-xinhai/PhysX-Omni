"""
Blender script: render 24 fixed-grid views.
Called by render_fixed_grid.py via:
  blender --background --python blender_script/render_fixed_grid.py -- \
          --object <path> --output_folder <path>

Output per object:
  <output_folder>/
    elev-15_azim000.png
    elev-15_azim045.png
    ...  (24 images total)
    transforms.json              (NeRF-style camera params)
"""

import argparse, sys, os, math, glob
from typing import Dict, Callable, Tuple

import bpy
from mathutils import Vector
import numpy as np
import json

# ──────────────────────────── constants ────────────────────────────
VIEWS_JSON   = "views_fixed_grid.json"   # written by render_fixed_grid.py
OBJ_IMPORT_OP = bpy.ops.wm.obj_import if hasattr(bpy.ops.wm, "obj_import") else bpy.ops.import_scene.obj

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj":  OBJ_IMPORT_OP,
    "glb":  bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd":  bpy.ops.import_scene.usd,
    "fbx":  bpy.ops.import_scene.fbx,
    "stl":  bpy.ops.import_mesh.stl,
    "ply":  bpy.ops.import_mesh.ply,
}


# ──────────────────────────── scene helpers ────────────────────────

def init_render(resolution: int = 512) -> None:
    sc = bpy.context.scene
    sc.render.engine                         = "CYCLES"
    sc.render.resolution_x                   = resolution
    sc.render.resolution_y                   = resolution
    sc.render.resolution_percentage          = 100
    sc.render.image_settings.file_format     = "PNG"
    sc.render.image_settings.color_mode      = "RGBA"
    sc.render.film_transparent               = True

    requested_device                         = os.environ.get("BLENDER_DEVICE", "GPU").upper()
    sc.cycles.device                         = "GPU" if requested_device == "GPU" else "CPU"
    sc.cycles.samples                        = 128
    sc.cycles.filter_type                    = "BOX"
    sc.cycles.filter_width                   = 1
    sc.cycles.diffuse_bounces                = 1
    sc.cycles.glossy_bounces                 = 1
    sc.cycles.transparent_max_bounces        = 3
    sc.cycles.transmission_bounces           = 3
    sc.cycles.use_denoising                  = True

    if requested_device == "GPU":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.get_devices()
            prefs.compute_device_type = "CUDA"
        except Exception:
            sc.cycles.device = "CPU"


def render_with_fallback() -> None:
    """Render one frame; fallback to CPU if GPU backend fails."""
    try:
        bpy.ops.render.render(write_still=True)
        return
    except RuntimeError as exc:
        err = str(exc)
        markers = ("CUDA kernel", "Unsupported PTX", "OPTIX", "HIP", "METAL")
        if bpy.context.scene.cycles.device == "GPU" and any(marker in err for marker in markers):
            print("[WARN] GPU render failed, fallback to CPU.")
            bpy.context.scene.cycles.device = "CPU"
            bpy.ops.render.render(write_still=True)
            return
        raise


def import_single_object(path: str) -> None:
    ext = path.split(".")[-1].lower()
    if ext not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {path}")
    fn = IMPORT_FUNCTIONS[ext]
    if ext in {"glb", "gltf"}:
        fn(filepath=path, merge_vertices=True, import_shading="NORMALS")
    elif ext == "obj":
        if hasattr(bpy.ops.wm, "obj_import"):  # Blender 3.3+ new API
            bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Z", up_axis="Y")
        else:                                   # Blender < 3.3 legacy
            bpy.ops.import_scene.obj(filepath=path, axis_forward="-Z", axis_up="Y")
    else:
        fn(filepath=path)


def init_scene() -> None:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)
    for tex in bpy.data.textures:
        bpy.data.textures.remove(tex, do_unlink=True)
    for img in bpy.data.images:
        bpy.data.images.remove(img, do_unlink=True)


def init_camera() -> bpy.types.Object:
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera         = cam
    cam.data.sensor_height            = cam.data.sensor_width = 32
    constraint                        = cam.constraints.new(type="TRACK_TO")
    constraint.track_axis             = "TRACK_NEGATIVE_Z"
    constraint.up_axis                = "UP_Z"
    target = bpy.data.objects.new("CamTarget", None)
    target.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(target)
    constraint.target = target
    return cam


def init_lighting() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    key = bpy.data.objects.new("KeyLight", bpy.data.lights.new("KeyLight", type="POINT"))
    bpy.context.collection.objects.link(key)
    key.data.energy  = 1000
    key.location     = (4, 1, 6)

    top = bpy.data.objects.new("TopLight", bpy.data.lights.new("TopLight", type="AREA"))
    bpy.context.collection.objects.link(top)
    top.data.energy  = 10000
    top.location     = (0, 0, 10)
    top.scale        = (100, 100, 100)

    bot = bpy.data.objects.new("BotLight", bpy.data.lights.new("BotLight", type="AREA"))
    bpy.context.collection.objects.link(bot)
    bot.data.energy  = 1000
    bot.location     = (0, 0, -10)


def load_objects(object_path: str) -> None:
    """Load mesh files from object_path (single file or directory).

    Directory detection priority (mirrors render_mobility.py):
      1. objs/**/*.obj  — ours_* / articulateanything_* (nested sub-dirs)
      2. [0-9]*.glb     — PhysXverse partglb
      3. [0-9]*/*.obj   — PhysXverse partobj
      4. objs/*.obj     — partseg flat structure
    """
    init_scene()

    if os.path.isfile(object_path):
        import_single_object(object_path)
        return

    # Directory mode: four-level probe
    obj_files = sorted(glob.glob(os.path.join(object_path, "objs", "**", "*.obj"), recursive=True))
    if not obj_files:
        obj_files = sorted(glob.glob(os.path.join(object_path, "[0-9]*.glb")))
    if not obj_files:
        obj_files = sorted(glob.glob(os.path.join(object_path, "[0-9]*", "*.obj")))
    if not obj_files:
        obj_files = sorted(glob.glob(os.path.join(object_path, "objs", "*.obj")))

    if not obj_files:
        raise ValueError(f"No supported mesh files found in directory: {object_path}")

    for f in obj_files:
        import_single_object(f)


def scene_bbox() -> Tuple[Vector, Vector]:
    bbox_min = Vector((math.inf,  math.inf,  math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    for obj in bpy.context.scene.objects:
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            bbox_min = Vector(map(min, zip(bbox_min, world)))
            bbox_max = Vector(map(max, zip(bbox_max, world)))
    return bbox_min, bbox_max


def normalize_scene() -> Tuple[float, Vector]:
    roots = [o for o in bpy.context.scene.objects if not o.parent]
    if len(roots) > 1:
        parent = bpy.data.objects.new("SceneRoot", None)
        bpy.context.scene.collection.objects.link(parent)
        for o in roots:
            o.parent = parent
        root = parent
    else:
        root = roots[0]

    bbox_min, bbox_max = scene_bbox()
    scale  = 1.0 / max(bbox_max - bbox_min)
    root.scale *= scale
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    root.matrix_world.translation += offset
    return scale, offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = [[rt[i][j] for j in range(3)] + [pos[i]] for i in range(3)]
    matrix.append([0, 0, 0, 1])
    return matrix


# ──────────────────────────── main ────────────────────────────────

def main(arg: argparse.Namespace) -> None:
    os.makedirs(arg.output_folder, exist_ok=True)

    init_render(resolution=arg.resolution)
    load_objects(arg.object)
    norm_scale, norm_offset = normalize_scene()
    cam = init_camera()
    init_lighting()
    print("[INFO] Scene ready.")

    with open(VIEWS_JSON, "r", encoding="utf-8") as f:
        views = json.load(f)
    assert len(views) == 24, f"Expected 24 views, got {len(views)}"

    to_export = {
        "aabb":   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale":  float(norm_scale),
        "offset": [float(norm_offset.x), float(norm_offset.y), float(norm_offset.z)],
        "frames": [],
    }

    for view in views:
        yaw    = view["yaw"]
        pitch  = view["pitch"]
        radius = view["radius"]
        fov    = view["fov"]
        elev   = view["elev_deg"]
        azim   = view["azim_deg"]

        # Spherical → Cartesian (Z-up)
        cam.location = (
            radius * math.cos(yaw) * math.cos(pitch),
            radius * math.sin(yaw) * math.cos(pitch),
            radius * math.sin(pitch),
        )
        # focal length from sensor width = 32 mm
        cam.data.lens = 16.0 / math.tan(fov / 2.0)

        # e.g. "elev-15_azim000.png" or "elev020_azim045.png"
        sign   = "-" if elev < 0 else ""
        fname  = f"elev{sign}{abs(elev):02d}_azim{azim:03d}.png"
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, fname)

        render_with_fallback()
        bpy.context.view_layer.update()

        to_export["frames"].append({
            "file_path":       fname,
            "elev_deg":        elev,
            "azim_deg":        azim,
            "camera_angle_x":  fov,
            "transform_matrix": get_transform_matrix(cam),
        })

        print(f"  [✓] elev={elev:+3d}°  azim={azim:3d}°  → {fname}")

    with open(os.path.join(arg.output_folder, "transforms.json"), "w") as f:
        json.dump(to_export, f, indent=4)
    print(f"[INFO] transforms.json saved to {arg.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",        type=str, required=True,
                        help="Path to OBJ folder (containing objs/) or single 3D file.")
    parser.add_argument("--output_folder", type=str, default="./renders_fixed_grid")
    parser.add_argument("--resolution",    type=int, default=512)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    main(args)
