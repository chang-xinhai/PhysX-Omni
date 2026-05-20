#!/usr/bin/env python3
"""Non-simulation articulation demo renderer for MJCF XML and URDF inputs.

Behavior:
- No physics stepping (`mj_step`) is used.
- For each movable joint, motion mode is configurable:
    - `return`: initial -> target -> initial
    - `hold`: initial -> target -> hold(target), then hard-cut reset for next joint
- Supports multi-view rendering in one side-by-side panel video.
- Supports single-file mode and batch mode over subfolders.
- URDF inputs are converted to temporary visual-only MJCF when
  ``--urdf-visual-only`` is enabled.
"""
# Example:
# python3 benchmark/code/render/kinematic/kinematic_articulation_demo.py \
#   --batch-input-dir demo_scale/demo_sacle_2 --batch-xml-name basic.xml \
#   --output-root render_outputs/kinematic_multiview_worldfree \
#   --input-root demo_scale --return-mode hold --include-root-free \
#   --view 135,-18 --view 45,-18 --hold-start 0.08 --move-duration 0.8 \
#   --hold-end 0.8 --fps 30



import argparse
import math
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio.v2 as iio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402


def ease_in_out(alpha):
    alpha = min(1.0, max(0.0, float(alpha)))
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / n
    half = 0.5 * float(angle)
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def quat_slerp(q0, q1, t):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    t = float(min(1.0, max(0.0, t)))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return quat_normalize((1.0 - t) * q0 + t * q1)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) < 1e-12:
        return q0.copy()
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return quat_normalize(s0 * q0 + s1 * q1)


def joint_qpos_width(jtype):
    if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def resolve_output_path(
    xml_path,
    out_path,
    output_root,
    input_root,
    filename="kinematic_demo.mp4",
    drop_xml_parent_name=None,
    drop_xml_parent_levels=0,
):
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    xml_path = Path(xml_path)
    output_root = Path(output_root)
    input_root = Path(input_root)
    try:
        rel_parent = xml_path.parent
        if drop_xml_parent_name and rel_parent.name == str(drop_xml_parent_name):
            rel_parent = rel_parent.parent
        for _ in range(max(0, int(drop_xml_parent_levels))):
            rel_parent = rel_parent.parent
        rel = rel_parent.relative_to(input_root)
        out = output_root.joinpath(*rel.parts) / filename
    except ValueError:
        out = output_root / xml_path.parent.name / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def resolve_asset_dir(
    xml_path,
    asset_dir=None,
    asset_dir_from_sample_objs=False,
    asset_dir_from_xml_parent=False,
):
    if asset_dir is not None:
        return Path(asset_dir)
    if asset_dir_from_xml_parent:
        return Path(xml_path).parent
    if not asset_dir_from_sample_objs:
        return None
    for parent in Path(xml_path).parents:
        candidate = parent / "objs"
        if candidate.is_dir():
            return candidate
    return None


def load_mujoco_assets(asset_dir, extra_asset_paths=None):
    assets = {}
    if asset_dir is not None:
        asset_dir = Path(asset_dir)
        if not asset_dir.is_dir():
            raise ValueError(f"asset dir not found: {asset_dir}")
        for path in sorted(p for p in asset_dir.iterdir() if p.is_file()):
            if path.name in assets:
                raise ValueError(f"duplicate asset basename in {asset_dir}: {path.name}")
            assets[path.name] = path.read_bytes()
        image_dir = asset_dir.parent / "images"
        if image_dir.is_dir():
            for path in sorted(p for p in image_dir.iterdir() if p.is_file()):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                assets.setdefault(path.name, path.read_bytes())

    for path in extra_asset_paths or []:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"extra asset file not found: {path}")
        assets.setdefault(path.name, path.read_bytes())

    return assets or None


def make_ours_visual_element():
    visual = ET.Element("visual")
    ET.SubElement(visual, "global", {"realtime": "1"})
    ET.SubElement(visual, "quality", {"shadowsize": "16384", "numslices": "28", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"diffuse": "2 2 2", "specular": "0.5 0.5 0.5", "active": "1"})
    ET.SubElement(visual, "rgba", {"fog": "0 1 0 1", "haze": "1 0 0 1"})
    return visual


def add_or_replace_child(parent, child):
    for old in list(parent):
        if old.tag == child.tag:
            parent.remove(old)
            break
    parent.insert(0, child)


def parse_float_list(text, default, expected_len):
    if text is None:
        return list(default)
    try:
        values = [float(x) for x in str(text).split()]
    except ValueError:
        return list(default)
    if len(values) != expected_len:
        return list(default)
    return values


def format_float_list(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def quat_from_rpy(roll, pitch, yaw):
    qx = quat_from_axis_angle([1.0, 0.0, 0.0], roll)
    qy = quat_from_axis_angle([0.0, 1.0, 0.0], pitch)
    qz = quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
    return quat_normalize(quat_mul(quat_mul(qz, qy), qx))


def origin_attrs(origin):
    if origin is None:
        return {"pos": "0 0 0", "quat": "1 0 0 0"}
    xyz = parse_float_list(origin.get("xyz"), [0.0, 0.0, 0.0], 3)
    rpy = parse_float_list(origin.get("rpy"), [0.0, 0.0, 0.0], 3)
    quat = quat_from_rpy(rpy[0], rpy[1], rpy[2])
    return {"pos": format_float_list(xyz), "quat": format_float_list(quat)}


def safe_mjcf_name(prefix, raw):
    keep = []
    for ch in str(raw):
        if ch.isalnum() or ch in "_-":
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    return f"{prefix}_{name}" if name else prefix


def obj_has_enough_vertices(asset_dir, filename):
    if asset_dir is None:
        return True
    path = Path(asset_dir) / Path(filename).name
    if not path.is_file() or path.suffix.lower() != ".obj":
        return True
    count = 0
    try:
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    count += 1
                    if count >= 4:
                        return True
    except OSError:
        return False
    return False


def parse_mtl_file(path):
    materials = {}
    current = None
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()
    except OSError:
        return materials

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0]
        if key == "newmtl" and len(parts) >= 2:
            current = " ".join(parts[1:])
            materials[current] = {"kd": None, "alpha": 1.0, "map_kd": None}
        elif current and key == "Kd" and len(parts) >= 4:
            try:
                materials[current]["kd"] = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                pass
        elif current and key in {"d", "Tr"} and len(parts) >= 2:
            try:
                alpha = float(parts[1])
                materials[current]["alpha"] = 1.0 - alpha if key == "Tr" else alpha
            except ValueError:
                pass
        elif current and key == "map_Kd" and len(parts) >= 2:
            # Keep the original relative path; it is resolved relative to the OBJ/MTL folder.
            materials[current]["map_kd"] = " ".join(parts[1:])
    return materials


def read_obj_material_info(asset_dir, filename):
    if asset_dir is None:
        return None
    obj_path = Path(asset_dir) / Path(filename).name
    if not obj_path.is_file():
        return None

    mtllibs = []
    first_usemtl = None
    current = None
    face_counts = {}
    try:
        for line in obj_path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "mtllib" and len(parts) >= 2:
                mtllibs.append(" ".join(parts[1:]))
            elif parts[0] == "usemtl" and len(parts) >= 2:
                current = " ".join(parts[1:])
                if first_usemtl is None:
                    first_usemtl = current
                face_counts.setdefault(current, 0)
            elif parts[0] == "f" and current:
                face_counts[current] = face_counts.get(current, 0) + 1
    except OSError:
        return None

    if not mtllibs:
        return None
    material_name = None
    if face_counts:
        material_name = max(face_counts.items(), key=lambda item: item[1])[0]
    elif first_usemtl:
        material_name = first_usemtl
    if material_name is None:
        return None

    merged = {}
    for mtllib in mtllibs:
        merged.update(parse_mtl_file(Path(asset_dir) / mtllib))
    info = merged.get(material_name)
    if not info:
        return None

    kd = info.get("kd") or [0.5, 0.5, 0.5]
    alpha = max(0.0, min(1.0, float(info.get("alpha", 1.0))))
    texture = None
    if info.get("map_kd"):
        texture_path = (Path(asset_dir) / info["map_kd"]).resolve()
        if texture_path.is_file():
            texture = texture_path.name
    return {"rgba": [kd[0], kd[1], kd[2], alpha], "texture": texture}


def urdf_visual_material_info(visual):
    material = visual.find("material")
    if material is None:
        return None
    color = material.find("color")
    if color is None or not color.get("rgba"):
        return None
    rgba = parse_float_list(color.get("rgba"), [0.5, 0.5, 0.5, 1.0], 4)
    return {"rgba": rgba, "texture": None}


def collect_urdf_visual_meshes(link, asset_dir):
    visuals = []
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            continue
        filename = Path(mesh.get("filename")).name
        if not obj_has_enough_vertices(asset_dir, filename):
            continue
        scale = parse_float_list(mesh.get("scale"), [1.0, 1.0, 1.0], 3)
        visuals.append(
            {
                "filename": filename,
                "scale": scale,
                "origin": visual.find("origin"),
                "urdf_material": urdf_visual_material_info(visual),
            }
        )
    return visuals


def convert_urdf_to_visual_mjcf(urdf_path, out_path, asset_dir=None):
    tree = ET.parse(urdf_path)
    urdf_root = tree.getroot()
    links = {link.get("name"): link for link in urdf_root.findall("link") if link.get("name")}

    joints_by_child = {}
    children_by_parent = {}
    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        if child_name in joints_by_child:
            raise ValueError(f"URDF body has multiple parents: {child_name}")
        joints_by_child[child_name] = joint
        children_by_parent.setdefault(parent_name, []).append(child_name)

    mj = ET.Element("mujoco", {"model": urdf_root.get("name") or Path(urdf_path).stem})
    compiler_attrs = {
        "angle": "radian",
        "inertiafromgeom": "false",
        "balanceinertia": "true",
        "boundmass": "1e-6",
        "boundinertia": "1e-6",
    }
    if asset_dir is not None:
        # Keep GLB/OBJ mesh loading file-based. MuJoCo can fail to decode GLB
        # when it is provided through the in-memory assets dict.
        compiler_attrs["meshdir"] = str(Path(asset_dir).resolve())
        compiler_attrs["texturedir"] = str(Path(asset_dir).resolve())
    ET.SubElement(mj, "compiler", compiler_attrs)
    asset = ET.SubElement(mj, "asset")
    worldbody = ET.SubElement(mj, "worldbody")

    mesh_names = {}
    material_names = {}
    texture_names = {}
    obj_material_cache = {}
    mesh_counter = 0
    material_counter = 0
    texture_counter = 0

    def mesh_name_for(visual):
        nonlocal mesh_counter
        key = (visual["filename"], tuple(float(x) for x in visual["scale"]))
        if key in mesh_names:
            return mesh_names[key]
        mesh_counter += 1
        name = safe_mjcf_name("mesh", f"{mesh_counter}_{Path(visual['filename']).stem}")
        attrs = {"name": name, "file": visual["filename"], "inertia": "shell"}
        if any(abs(float(v) - 1.0) > 1e-9 for v in visual["scale"]):
            attrs["scale"] = format_float_list(visual["scale"])
        ET.SubElement(asset, "mesh", attrs)
        mesh_names[key] = name
        return name

    def material_info_for(visual):
        key = visual["filename"]
        if key not in obj_material_cache:
            obj_material_cache[key] = read_obj_material_info(asset_dir, key)
        return obj_material_cache[key] or visual.get("urdf_material")

    def material_name_for(visual):
        nonlocal material_counter, texture_counter
        info = material_info_for(visual)
        if not info:
            return None

        rgba = [float(x) for x in info.get("rgba", [0.5, 0.5, 0.5, 1.0])]
        rgba = [max(0.0, min(1.0, x)) for x in rgba]
        texture = info.get("texture")
        key = (tuple(round(x, 6) for x in rgba), texture)
        if key in material_names:
            return material_names[key]

        texture_name = None
        if texture:
            if texture in texture_names:
                texture_name = texture_names[texture]
            else:
                texture_counter += 1
                texture_name = safe_mjcf_name("tex", f"{texture_counter}_{Path(texture).stem}")
                ET.SubElement(asset, "texture", {"name": texture_name, "type": "2d", "file": texture})
                texture_names[texture] = texture_name

        material_counter += 1
        name = safe_mjcf_name("mat", f"{material_counter}_{Path(visual['filename']).stem}")
        attrs = {"name": name, "rgba": format_float_list(rgba)}
        if texture_name:
            attrs.update({"texture": texture_name, "texuniform": "false"})
        ET.SubElement(asset, "material", attrs)
        material_names[key] = name
        return name

    def add_body(parent_xml, link_name, joint=None):
        attrs = {"name": safe_mjcf_name("body", link_name)}
        if joint is not None:
            attrs.update(origin_attrs(joint.find("origin")))
        body = ET.SubElement(parent_xml, "body", attrs)
        ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": "1", "diaginertia": "1 1 1"})

        if joint is not None and joint.get("type") not in {None, "fixed"}:
            jtype = joint.get("type")
            joint_attrs = {"name": safe_mjcf_name("joint", joint.get("name") or f"{link_name}_joint")}
            if jtype in {"revolute", "continuous"}:
                joint_attrs["type"] = "hinge"
            elif jtype == "prismatic":
                joint_attrs["type"] = "slide"
            elif jtype == "floating":
                joint_attrs["type"] = "free"
            else:
                joint_attrs["type"] = "hinge"
            axis = joint.find("axis")
            if axis is not None and axis.get("xyz") and joint_attrs["type"] != "free":
                joint_attrs["axis"] = format_float_list(parse_float_list(axis.get("xyz"), [1.0, 0.0, 0.0], 3))
            limit = joint.find("limit")
            if joint_attrs["type"] != "free" and limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
                lower = float(limit.get("lower"))
                upper = float(limit.get("upper"))
                if upper < lower:
                    lower, upper = upper, lower
                if abs(upper - lower) > 1e-12:
                    joint_attrs["limited"] = "true"
                    joint_attrs["range"] = f"{lower:.9g} {upper:.9g}"
            ET.SubElement(body, "joint", joint_attrs)

        link = links.get(link_name)
        if link is not None:
            for vidx, visual in enumerate(collect_urdf_visual_meshes(link, asset_dir), start=1):
                geom_attrs = {
                    "name": safe_mjcf_name("geom", f"{link_name}_{vidx}"),
                    "type": "mesh",
                    "mesh": mesh_name_for(visual),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                }
                material_name = material_name_for(visual)
                if material_name:
                    geom_attrs["material"] = material_name
                else:
                    geom_attrs["rgba"] = "0.5 0.5 0.5 1"
                geom_attrs.update(origin_attrs(visual["origin"]))
                ET.SubElement(body, "geom", geom_attrs)

        for child_name in children_by_parent.get(link_name, []):
            add_body(body, child_name, joints_by_child.get(child_name))
        return body

    roots = [name for name in links if name not in joints_by_child]
    if not roots and links:
        roots = [next(iter(links))]
    for root_name in roots:
        add_body(worldbody, root_name, None)

    ET.ElementTree(mj).write(out_path, encoding="utf-8", xml_declaration=True)


def inject_ours_scene_preset(xml_path, out_path, desert_asset_name=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    add_or_replace_child(root, make_ours_visual_element())

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(1 if root.find("compiler") is not None else 0, asset)

    def has_asset(tag, name):
        return any(child.tag == tag and child.get("name") == name for child in asset)

    if desert_asset_name and not any(child.tag == "texture" and child.get("type") == "skybox" for child in asset):
        ET.SubElement(
            asset,
            "texture",
            {"type": "skybox", "file": desert_asset_name, "gridsize": "3 4", "gridlayout": ".U..LFRB.D.."},
        )
    if not has_asset("texture", "plane"):
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "plane",
                "type": "2d",
                "builtin": "checker",
                "rgb1": ".1 .1 .1",
                "rgb2": ".5 .5 .5",
                "width": "512",
                "height": "512",
                "mark": "cross",
                "markrgb": ".8 .8 .8",
            },
        )
    if not has_asset("material", "plane"):
        ET.SubElement(
            asset,
            "material",
            {"name": "plane", "reflectance": "0.3", "texture": "plane", "texrepeat": "1 1", "texuniform": "true"},
        )

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if not any(child.tag == "geom" and child.get("name") == "floor" for child in worldbody):
        worldbody.insert(
            0,
            ET.Element(
                "geom",
                {"name": "floor", "pos": "0 0 0", "size": "0 0 .25", "type": "plane", "material": "plane", "condim": "6"},
            ),
        )
    if not any(child.tag == "light" for child in worldbody):
        worldbody.insert(
            1,
            ET.Element(
                "light",
                {
                    "directional": "true",
                    "ambient": ".3 .3 .3",
                    "pos": "30 30 30",
                    "dir": "0 -2 -1",
                    "diffuse": ".5 .5 .5",
                    "specular": ".5 .5 .5",
                },
            ),
        )

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def load_model_with_optional_scene(
    xml_path,
    assets=None,
    scene_preset="none",
    scene_desert_path=None,
    urdf_visual_only=False,
    visual_asset_dir=None,
):
    temp_dir = None
    source_xml = Path(xml_path)
    if urdf_visual_only:
        temp_dir = tempfile.TemporaryDirectory(prefix="aa_urdf_visual_")
        source_xml = Path(temp_dir.name) / "visual_only.xml"
        convert_urdf_to_visual_mjcf(xml_path, source_xml, asset_dir=visual_asset_dir)

    if scene_preset == "none":
        if assets is None:
            return mujoco.MjModel.from_xml_path(str(source_xml)), temp_dir
        return mujoco.MjModel.from_xml_path(str(source_xml), assets), temp_dir
    if scene_preset != "ours_xml":
        raise ValueError(f"Unsupported scene preset: {scene_preset}")

    if assets is None:
        base_model = mujoco.MjModel.from_xml_path(str(source_xml))
    else:
        base_model = mujoco.MjModel.from_xml_path(str(source_xml), assets)
    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="aa_urdf_scene_")
    base_xml = Path(temp_dir.name) / "base.xml"
    scene_xml = Path(temp_dir.name) / "scene.xml"
    mujoco.mj_saveLastXML(str(base_xml), base_model)
    inject_ours_scene_preset(
        xml_path=base_xml,
        out_path=scene_xml,
        desert_asset_name=Path(scene_desert_path).name if scene_desert_path else None,
    )
    if assets is None:
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
    else:
        model = mujoco.MjModel.from_xml_path(str(scene_xml), assets)
    return model, temp_dir


def discover_batch_xml_paths(batch_input_dir, preferred_xml_name="basic.xml"):
    batch_input_dir = Path(batch_input_dir)
    if not batch_input_dir.is_dir():
        raise ValueError(f"batch input dir not found: {batch_input_dir}")

    xml_paths = []
    for child in sorted(batch_input_dir.iterdir()):
        if not child.is_dir():
            continue
        preferred = child / preferred_xml_name
        if preferred.is_file():
            xml_paths.append(preferred)
            continue
        candidates = sorted(child.glob("*.xml"))
        if candidates:
            xml_paths.append(candidates[0])
    return xml_paths


def parse_view_spec(spec):
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"Invalid --view '{spec}'. Expected 'azimuth,elevation[,distance_scale]'."
        )
    try:
        azimuth = float(parts[0])
        elevation = float(parts[1])
        distance_scale = float(parts[2]) if len(parts) == 3 else 2.4
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --view '{spec}': {exc}") from exc
    if distance_scale <= 0:
        raise argparse.ArgumentTypeError(f"Invalid --view '{spec}': distance_scale must be > 0")
    return (azimuth, elevation, distance_scale)


def build_cameras(model, view_specs=None):
    specs = list(view_specs) if view_specs else [(135.0, -18.0, 2.4)]
    cameras = []
    for azimuth, elevation, distance_scale in specs:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = model.stat.center
        cam.distance = float(distance_scale) * model.stat.extent
        cam.azimuth = float(azimuth)
        cam.elevation = float(elevation)
        cameras.append(cam)
    return cameras


def visible_nonplane_geom_ids(model):
    plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
    return [
        gid
        for gid in range(model.ngeom)
        if int(model.geom_type[gid]) != plane_type and float(model.geom_rgba[gid][3]) > 1e-6
    ]


def top_level_body_for_geom(model, gid):
    bid = int(model.geom_bodyid[gid])
    if bid <= 0:
        return None
    while int(model.body_parentid[bid]) > 0:
        bid = int(model.body_parentid[bid])
    return bid if bid > 0 else None


def lift_model_above_ground(model, data, ground_z=0.0, clearance=0.02):
    """Lift visible object root bodies so their conservative bottom is above ground.

    The estimate uses geom center minus geom bounding radius. This is deliberately
    conservative: it may leave a little extra clearance, but it avoids rendering
    generated assets half buried under the checker plane.
    """
    mujoco.mj_forward(model, data)
    geom_ids = visible_nonplane_geom_ids(model)
    if not geom_ids:
        return 0.0, None

    bottoms = []
    root_bodies = set()
    world_geom_ids = []
    for gid in geom_ids:
        bottoms.append(float(data.geom_xpos[gid][2]) - float(model.geom_rbound[gid]))
        root = top_level_body_for_geom(model, gid)
        if root is not None:
            root_bodies.add(root)
        elif int(model.geom_bodyid[gid]) == 0:
            world_geom_ids.append(gid)

    if not bottoms or (not root_bodies and not world_geom_ids):
        return 0.0, None

    min_bottom = min(bottoms)
    target_bottom = float(ground_z) + max(0.0, float(clearance))
    delta = target_bottom - min_bottom
    if delta <= 0.0:
        return 0.0, min_bottom

    for bid in sorted(root_bodies):
        model.body_pos[bid][2] += delta
    for gid in world_geom_ids:
        model.geom_pos[gid][2] += delta
    mujoco.mj_forward(model, data)
    return delta, min_bottom


def render_multiview_frame(renderers, cameras, data):
    frames = []
    for renderer, camera in zip(renderers, cameras):
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    if len(frames) == 1:
        return frames[0]
    return np.concatenate(frames, axis=1)


def build_focus_geom_ids(model, events):
    seed_bodies = set()
    for ev in events:
        jid = int(ev["joint_id"])
        seed_bodies.add(int(model.jnt_bodyid[jid]))

    if not seed_bodies:
        seed_bodies = set(range(1, model.nbody))

    focus_bodies = set(seed_bodies)
    for bid in range(model.nbody):
        parent = bid
        while parent > 0:
            parent = int(model.body_parentid[parent])
            if parent in seed_bodies:
                focus_bodies.add(bid)
                break

    plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
    geom_ids = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid not in focus_bodies:
            continue
        if int(model.geom_type[gid]) == plane_type:
            continue
        if float(model.geom_rgba[gid][3]) <= 1e-6:
            continue
        geom_ids.append(gid)

    if geom_ids:
        return geom_ids

    return [
        gid
        for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) > 0 and int(model.geom_type[gid]) != plane_type
    ]


def collect_motion_spheres(model, data, initial_qpos, events, return_mode):
    geom_ids = build_focus_geom_ids(model, events)
    if not geom_ids:
        return None, None

    centers_all = []
    radii_all = []

    def append_state(event=None, alpha=0.0):
        data.qpos[:] = initial_qpos
        if event is not None:
            apply_event_qpos(data, event, alpha)
        mujoco.mj_forward(model, data)

        centers = np.asarray(data.geom_xpos[geom_ids], dtype=float).copy()
        radii = np.asarray(model.geom_rbound[geom_ids], dtype=float).copy()
        tiny = radii < 1e-8
        if np.any(tiny):
            sizes = np.asarray(model.geom_size[geom_ids], dtype=float)
            radii[tiny] = np.maximum(1e-4, np.linalg.norm(sizes[tiny], axis=1))

        centers_all.append(centers)
        radii_all.append(radii)

    append_state(event=None, alpha=0.0)
    if events:
        up_alphas = np.linspace(0.0, 1.0, num=7)
        down_alphas = np.linspace(1.0, 0.0, num=5)
        for ev in events:
            for a in up_alphas:
                append_state(event=ev, alpha=float(a))
            if return_mode == "return":
                for a in down_alphas:
                    append_state(event=ev, alpha=float(a))

    centers = np.concatenate(centers_all, axis=0)
    radii = np.concatenate(radii_all, axis=0)
    return centers, radii


def camera_fits_spheres(model, data, camera, width, height, centers, radii, margin, scene, option, perturb):
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        perturb,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL.value,
        scene,
    )

    cam = scene.camera[0]
    pos = np.asarray(cam.pos, dtype=float)
    forward = np.asarray(cam.forward, dtype=float)
    up = np.asarray(cam.up, dtype=float)

    fn = np.linalg.norm(forward)
    un = np.linalg.norm(up)
    if fn < 1e-12 or un < 1e-12:
        return False
    forward = forward / fn
    up = up / un

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-12:
        return False
    right = right / rn

    near = max(1e-9, float(cam.frustum_near))
    tan_half_y = float(cam.frustum_top) / near
    tan_half_x = tan_half_y * (float(width) / float(height))
    tan_half_x *= float(margin)
    tan_half_y *= float(margin)

    for p, r in zip(centers, radii):
        rel = p - pos
        z = float(np.dot(rel, forward))
        x = abs(float(np.dot(rel, right)))
        y = abs(float(np.dot(rel, up)))

        if z <= r + 1e-6:
            return False
        if x + r > z * tan_half_x:
            return False
        if y + r > z * tan_half_y:
            return False

    return True


def adapt_cameras_to_motion(model, data, cameras, initial_qpos, events, return_mode, width, height, margin=0.96):
    centers, radii = collect_motion_spheres(model, data, initial_qpos, events, return_mode)
    if centers is None or len(centers) == 0:
        return False

    mins = np.min(centers - radii[:, None], axis=0)
    maxs = np.max(centers + radii[:, None], axis=0)
    lookat = 0.5 * (mins + maxs)

    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=max(1000, model.ngeom * 3))

    for cam in cameras:
        cam.lookat[:] = lookat

        high = max(float(cam.distance), 1e-3)
        for _ in range(32):
            cam.distance = high
            if camera_fits_spheres(model, data, cam, width, height, centers, radii, margin, scene, option, perturb):
                break
            high *= 1.5

        low = 0.0
        for _ in range(32):
            mid = 0.5 * (low + high)
            cam.distance = mid
            if camera_fits_spheres(model, data, cam, width, height, centers, radii, margin, scene, option, perturb):
                high = mid
            else:
                low = mid

        cam.distance = high

    return True


def find_base_body_id(model):
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and bname.lower() == "base":
            return bid
    return None


def build_target_for_joint(model, data, jid, event_index, extent, include_root_free, base_body_id):
    jtype = int(model.jnt_type[jid])
    qadr = int(model.jnt_qposadr[jid])
    width = joint_qpos_width(jtype)
    initial = np.asarray(data.qpos[qadr : qadr + width], dtype=float).copy()

    JOINT_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
    JOINT_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
    JOINT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
    JOINT_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)

    sign = 1.0 if (event_index % 2 == 0) else -1.0

    if jtype in (JOINT_HINGE, JOINT_SLIDE):
        val0 = float(initial[0])
        limited = bool(model.jnt_limited[jid])
        low, high = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        if jtype == JOINT_HINGE:
            base_amp = math.radians(60.0)
            if limited:
                span = max(1e-6, high - low)
                amp = min(base_amp, 0.98 * span)
            else:
                amp = base_amp
        else:
            base_amp = 0.15 * max(0.2, extent)
            if limited:
                span = max(1e-6, high - low)
                amp = min(base_amp, 0.98 * span)
            else:
                amp = base_amp

        if limited:
            margin = 0.03 * max(1e-6, high - low)
            lo = low + margin
            hi = high - margin
            # Prefer the farther side to maximize visible coverage of joint range.
            if abs(hi - val0) >= abs(lo - val0):
                target_val = hi
            else:
                target_val = lo
        else:
            target_val = val0 + sign * amp

        target = initial.copy()
        target[0] = target_val
        return target

    if jtype == JOINT_BALL:
        axes = (
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        axis = axes[event_index % len(axes)]
        angle = sign * math.radians(45.0)
        delta = quat_from_axis_angle(axis, angle)
        target = quat_mul(delta, quat_normalize(initial))
        return quat_normalize(target)

    if jtype == JOINT_FREE:
        bid = int(model.jnt_bodyid[jid])
        if (base_body_id is not None) and (bid == int(base_body_id)):
            # Keep the canonical base body fixed.
            return None
        if not include_root_free:
            return None
        lift = max(0.12, 0.25 * max(0.2, extent))
        target = initial.copy()
        target[:3] = initial[:3]
        target[2] = initial[2] + lift
        target[3:7] = quat_normalize(initial[3:7])
        return target

    return None


def build_joint_events(model, data, include_root_free=False):
    events = []
    extent = float(model.stat.extent)
    base_body_id = find_base_body_id(model)
    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        target = build_target_for_joint(
            model,
            data,
            jid,
            len(events),
            extent,
            include_root_free,
            base_body_id,
        )
        if target is None:
            continue
        qadr = int(model.jnt_qposadr[jid])
        width = joint_qpos_width(jtype)
        initial = np.asarray(data.qpos[qadr : qadr + width], dtype=float).copy()
        if np.allclose(initial, target, atol=1e-10):
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])) or "unknown_body"
        events.append(
            {
                "joint_id": jid,
                "joint_name": jname,
                "body_name": bname,
                "joint_type": jtype,
                "qadr": qadr,
                "initial": initial,
                "target": target,
            }
        )
    return events


def apply_event_qpos(data, event, alpha):
    alpha = float(min(1.0, max(0.0, alpha)))
    jtype = int(event["joint_type"])
    qadr = int(event["qadr"])
    initial = np.asarray(event["initial"], dtype=float)
    target = np.asarray(event["target"], dtype=float)

    if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        data.qpos[qadr] = (1.0 - alpha) * initial[0] + alpha * target[0]
        return

    if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
        data.qpos[qadr : qadr + 4] = quat_slerp(initial, target, alpha)
        return

    if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
        data.qpos[qadr : qadr + 3] = (1.0 - alpha) * initial[:3] + alpha * target[:3]
        data.qpos[qadr + 3 : qadr + 7] = quat_slerp(initial[3:7], target[3:7], alpha)
        return


def render_kinematic_demo(
    xml_path,
    out_path,
    fps=30,
    hold_start=0.2,
    move_duration=0.8,
    hold_end=0.2,
    include_root_free=False,
    return_mode="return",
    view_specs=None,
    panel_width=None,
    panel_height=None,
    camera_mode="adaptive",
    camera_margin=0.96,
    asset_dir=None,
    asset_dir_from_sample_objs=False,
    asset_dir_from_xml_parent=False,
    urdf_visual_only=False,
    scene_preset="none",
    scene_desert_path=None,
    lift_above_ground=True,
    ground_z=0.0,
    ground_clearance=0.02,
):
    if return_mode not in {"return", "hold"}:
        raise ValueError(f"Unsupported return_mode: {return_mode}")
    if camera_mode not in {"adaptive", "fixed"}:
        raise ValueError(f"Unsupported camera_mode: {camera_mode}")
    if not (0.0 < float(camera_margin) <= 1.0):
        raise ValueError(f"camera_margin must be in (0, 1], got {camera_margin}")

    resolved_asset_dir = resolve_asset_dir(
        xml_path=xml_path,
        asset_dir=asset_dir,
        asset_dir_from_sample_objs=asset_dir_from_sample_objs,
        asset_dir_from_xml_parent=asset_dir_from_xml_parent,
    )
    extra_assets = [scene_desert_path] if scene_desert_path else []
    use_file_based_urdf_assets = (
        urdf_visual_only
        and resolved_asset_dir is not None
        and any(Path(resolved_asset_dir).glob("*.glb"))
    )
    assets = None
    if not use_file_based_urdf_assets:
        assets = load_mujoco_assets(
            resolved_asset_dir,
            extra_asset_paths=extra_assets,
        )
    model, temp_scene_dir = load_model_with_optional_scene(
        xml_path=xml_path,
        assets=assets,
        scene_preset=scene_preset,
        scene_desert_path=scene_desert_path,
        urdf_visual_only=urdf_visual_only,
        visual_asset_dir=resolved_asset_dir,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lift_delta = 0.0
    min_bottom_before_lift = None
    if lift_above_ground:
        lift_delta, min_bottom_before_lift = lift_model_above_ground(
            model=model,
            data=data,
            ground_z=ground_z,
            clearance=ground_clearance,
        )

    initial_qpos = data.qpos.copy()
    events = build_joint_events(
        model,
        data,
        include_root_free=include_root_free,
    )
    auto_enabled_root_free = False
    if (not events) and (not include_root_free):
        # Fallback to include free-part motion when no other event exists.
        events = build_joint_events(
            model,
            data,
            include_root_free=True,
        )
        auto_enabled_root_free = bool(events)
    cameras = build_cameras(model, view_specs=view_specs)

    print("xml =", xml_path)
    print("output =", out_path)
    if resolved_asset_dir is not None:
        print("asset_dir =", resolved_asset_dir)
        print("asset_count =", len(assets) if assets is not None else "file_based")
    print("urdf_visual_only =", urdf_visual_only)
    print("scene_preset =", scene_preset)
    print("lift_above_ground =", lift_above_ground)
    if lift_above_ground:
        print("ground_z =", ground_z)
        print("ground_clearance =", ground_clearance)
        print("min_bottom_before_lift =", min_bottom_before_lift)
        print("lift_delta =", lift_delta)
    print("movable_joints =", len(events))
    if auto_enabled_root_free:
        print("auto_fallback = include_root_free enabled because no events were found")
    print("return_mode =", return_mode)
    print("view_count =", len(cameras))
    if events:
        for i, ev in enumerate(events, start=1):
            jtype = int(ev["joint_type"])
            print(
                f"joint {i}: id={ev['joint_id']} name={ev['joint_name']} "
                f"body={ev['body_name']} type={jtype}"
            )
    else:
        print("no movable joints found, writing a short static clip.")

    width = int(panel_width) if panel_width is not None else int(model.vis.global_.offwidth)
    height = int(panel_height) if panel_height is not None else int(model.vis.global_.offheight)
    if width <= 0:
        width = 1280
    if height <= 0:
        height = 720

    camera_adapted = False
    if camera_mode == "adaptive":
        camera_adapted = adapt_cameras_to_motion(
            model=model,
            data=data,
            cameras=cameras,
            initial_qpos=initial_qpos,
            events=events,
            return_mode=return_mode,
            width=width,
            height=height,
            margin=camera_margin,
        )
    print("camera_mode =", camera_mode)
    if camera_mode == "adaptive":
        print("camera_margin =", camera_margin)
        print("camera_adapted =", camera_adapted)
    print(
        "views =",
        [
            f"(az={c.azimuth:.1f}, el={c.elevation:.1f}, dist={c.distance:.3f})"
            for c in cameras
        ],
    )

    n_hold0 = max(1, int(round(hold_start * fps)))
    n_up = max(1, int(round(move_duration * fps)))
    n_down = max(1, int(round(move_duration * fps))) if return_mode == "return" else 0
    n_hold1 = max(1, int(round(hold_end * fps)))

    with ExitStack() as stack:
        renderers = [
            stack.enter_context(mujoco.Renderer(model, width=width, height=height))
            for _ in cameras
        ]
        with iio.get_writer(str(out_path), fps=fps) as writer:
            if not events:
                n_static = max(1, int(round(1.0 * fps)))
                for _ in range(n_static):
                    data.qpos[:] = initial_qpos
                    mujoco.mj_forward(model, data)
                    writer.append_data(render_multiview_frame(renderers, cameras, data))
            else:
                for ev in events:
                    # Hold initial.
                    for _ in range(n_hold0):
                        data.qpos[:] = initial_qpos
                        apply_event_qpos(data, ev, 0.0)
                        mujoco.mj_forward(model, data)
                        writer.append_data(render_multiview_frame(renderers, cameras, data))

                    # Move to target.
                    for k in range(n_up):
                        t = 1.0 if n_up <= 1 else k / (n_up - 1)
                        alpha = ease_in_out(t)
                        data.qpos[:] = initial_qpos
                        apply_event_qpos(data, ev, alpha)
                        mujoco.mj_forward(model, data)
                        writer.append_data(render_multiview_frame(renderers, cameras, data))

                    if return_mode == "return":
                        # Return to initial.
                        for k in range(n_down):
                            t = 1.0 if n_down <= 1 else k / (n_down - 1)
                            alpha = 1.0 - ease_in_out(t)
                            data.qpos[:] = initial_qpos
                            apply_event_qpos(data, ev, alpha)
                            mujoco.mj_forward(model, data)
                            writer.append_data(render_multiview_frame(renderers, cameras, data))

                        # Hold initial again.
                        for _ in range(n_hold1):
                            data.qpos[:] = initial_qpos
                            apply_event_qpos(data, ev, 0.0)
                            mujoco.mj_forward(model, data)
                            writer.append_data(render_multiview_frame(renderers, cameras, data))
                    else:
                        # Hold target and then hard-cut reset at next event start.
                        for _ in range(n_hold1):
                            data.qpos[:] = initial_qpos
                            apply_event_qpos(data, ev, 1.0)
                            mujoco.mj_forward(model, data)
                            writer.append_data(render_multiview_frame(renderers, cameras, data))

    print("saved:", out_path, "size=", out_path.stat().st_size)
    if temp_scene_dir is not None:
        temp_scene_dir.cleanup()
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Non-simulation articulation demo with optional multi-view side-by-side rendering."
    )
    parser.add_argument("--xml", type=str, default=None, help="Single-file mode: path to one MJCF XML or URDF.")
    parser.add_argument(
        "--batch-input-dir",
        type=str,
        default=None,
        help="Batch mode: input folder whose subfolders each contain one XML/URDF.",
    )
    parser.add_argument(
        "--batch-xml-name",
        type=str,
        default="basic.xml",
        help="Preferred XML/URDF filename in each subfolder. Relative nested paths are allowed.",
    )
    parser.add_argument("--batch-limit", type=int, default=None, help="Optional max number of subfolders in batch mode.")
    parser.add_argument("--batch-num-shards", type=int, default=1, help="Split batch XML list into this many modulo shards.")
    parser.add_argument("--batch-shard-index", type=int, default=0, help="Modulo shard index to render, in [0, batch-num-shards).")
    parser.add_argument("--skip-existing", action="store_true", help="In batch mode, skip samples whose output video already exists.")
    parser.add_argument("--out", type=str, default=None, help="Single-file output path.")
    parser.add_argument(
        "--asset-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of OBJ/MTL/texture files to pass to MuJoCo as in-memory assets. "
            "This is useful for URDF files whose mesh filenames are relative or basename-only."
        ),
    )
    parser.add_argument(
        "--asset-dir-from-sample-objs",
        action="store_true",
        help=(
            "Resolve assets from the nearest ancestor directory named `objs`. "
            "For ArticulateAnything URDFs this maps */<object_id>/joint_actor/.../mobility.urdf "
            "to */<object_id>/objs without editing the URDF file."
        ),
    )
    parser.add_argument(
        "--asset-dir-from-xml-parent",
        action="store_true",
        help=(
            "Resolve mesh assets from the XML/URDF parent directory. "
            "This supports URDFs whose mesh references are basename-only without editing the URDF."
        ),
    )
    parser.add_argument(
        "--urdf-visual-only",
        action="store_true",
        help=(
            "Convert URDF to a temporary MJCF with visual meshes only. "
            "Collision tags are ignored, mesh assets use shell inertia, and invalid tiny meshes are skipped. "
            "This is intended for kinematic visualization only; original URDF/OBJ files are not modified."
        ),
    )
    parser.add_argument(
        "--scene-preset",
        choices=["none", "ours_xml"],
        default="none",
        help="Optional visual scene preset. `ours_xml` injects the same skybox/checker floor/light style used by generated XML outputs into a temporary MJCF copy.",
    )
    parser.add_argument(
        "--scene-desert-path",
        type=str,
        default=None,
        help="Path to desert.png for --scene-preset ours_xml skybox. The file is passed as an in-memory MuJoCo asset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="render_outputs/kinematic_demo",
        help="Auto-output root when --out is not provided.",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="demo_scale",
        help="Path-mapping root for auto output structure.",
    )
    parser.add_argument(
        "--drop-xml-parent-name",
        type=str,
        default=None,
        help=(
            "When auto-building output paths, drop this final XML parent directory name. "
            "For example, use `urdf` so */<object_id>/urdf/model.urdf outputs to "
            "<output-root>/<object_id>/kinematic_demo.mp4 instead of <output-root>/<object_id>/urdf/..."
        ),
    )
    parser.add_argument(
        "--drop-xml-parent-levels",
        type=int,
        default=0,
        help=(
            "When auto-building output paths, drop this many parent directories above the XML/URDF parent. "
            "Use 1 for <object_id>/mesh/basic.urdf and 3 for "
            "<object_id>/joint_actor/iter_0/seed_0/mobility.urdf."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-start", type=float, default=0.2)
    parser.add_argument("--move-duration", type=float, default=0.8)
    parser.add_argument(
        "--hold-end",
        type=float,
        default=0.2,
        help="In return mode: post-return hold(initial). In hold mode: hold(target).",
    )
    parser.add_argument(
        "--return-mode",
        type=str,
        choices=["return", "hold"],
        default="return",
        help="Joint motion mode: 'return' keeps old behavior; 'hold' avoids animated return.",
    )
    parser.add_argument(
        "--view",
        action="append",
        type=parse_view_spec,
        default=None,
        help=(
            "Camera spec: 'azimuth,elevation[,distance_scale]'. Repeat this flag for multi-view. "
            "Example: --view 135,-18 --view 45,-18"
        ),
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=None,
        help="Per-view render width. If unset, uses MJCF offscreen width.",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=None,
        help="Per-view render height. If unset, uses MJCF offscreen height.",
    )
    parser.add_argument(
        "--camera-mode",
        type=str,
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="Camera strategy. 'adaptive' fits all moving parts tightly inside frame.",
    )
    parser.add_argument(
        "--camera-margin",
        type=float,
        default=0.96,
        help="Frustum safety margin in adaptive mode (0,1]. Lower means more border.",
    )
    parser.add_argument(
        "--include-root-free",
        action="store_true",
        help="If set, animate free-part joints while keeping the base body fixed.",
    )
    parser.add_argument(
        "--no-lift-above-ground",
        action="store_true",
        help="Disable the default conservative lift that moves visible object geoms above the ground plane.",
    )
    parser.add_argument("--ground-z", type=float, default=0.0, help="Ground plane z used by --lift-above-ground.")
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.02,
        help="Minimum conservative clearance above ground after lifting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    common_kwargs = dict(
        fps=args.fps,
        hold_start=args.hold_start,
        move_duration=args.move_duration,
        hold_end=args.hold_end,
        include_root_free=args.include_root_free,
        return_mode=args.return_mode,
        view_specs=args.view,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        camera_mode=args.camera_mode,
        camera_margin=args.camera_margin,
        asset_dir=args.asset_dir,
        asset_dir_from_sample_objs=args.asset_dir_from_sample_objs,
        asset_dir_from_xml_parent=args.asset_dir_from_xml_parent,
        urdf_visual_only=args.urdf_visual_only,
        scene_preset=args.scene_preset,
        scene_desert_path=args.scene_desert_path,
        lift_above_ground=not args.no_lift_above_ground,
        ground_z=args.ground_z,
        ground_clearance=args.ground_clearance,
    )

    if args.batch_input_dir is not None:
        if args.xml is not None:
            raise ValueError("Use either --xml or --batch-input-dir, not both.")
        if args.out is not None:
            raise ValueError("--out is single-file only; batch mode uses auto output path.")

        xml_paths = discover_batch_xml_paths(args.batch_input_dir, preferred_xml_name=args.batch_xml_name)
        if args.batch_num_shards <= 0:
            raise ValueError("--batch-num-shards must be > 0.")
        if not (0 <= args.batch_shard_index < args.batch_num_shards):
            raise ValueError("--batch-shard-index must be in [0, --batch-num-shards).")
        if args.batch_num_shards > 1:
            xml_paths = [
                path for idx, path in enumerate(xml_paths)
                if idx % args.batch_num_shards == args.batch_shard_index
            ]
        if args.batch_limit is not None and args.batch_limit > 0:
            xml_paths = xml_paths[: args.batch_limit]
        if not xml_paths:
            raise ValueError(f"No XML/URDF found in subfolders of: {args.batch_input_dir}")

        print(f"batch_input_dir = {args.batch_input_dir}")
        print(f"batch_shard = {args.batch_shard_index}/{args.batch_num_shards}")
        print(f"batch_xml_count = {len(xml_paths)}")
        failures = []
        for idx, xml_path in enumerate(xml_paths, start=1):
            try:
                out_path = resolve_output_path(
                    xml_path=xml_path,
                    out_path=None,
                    output_root=args.output_root,
                    input_root=args.input_root,
                    filename="kinematic_demo.mp4",
                    drop_xml_parent_name=args.drop_xml_parent_name,
                    drop_xml_parent_levels=args.drop_xml_parent_levels,
                )
                if args.skip_existing and out_path.is_file():
                    print(f"[{idx}/{len(xml_paths)}] skip existing: {out_path}", flush=True)
                    continue
                print(f"[{idx}/{len(xml_paths)}] rendering: {xml_path}")
                render_kinematic_demo(xml_path=xml_path, out_path=out_path, **common_kwargs)
            except Exception as exc:
                failures.append((str(xml_path), str(exc)))
                print(f"[{idx}/{len(xml_paths)}] failed: {xml_path} | {exc}")
        print(f"batch_done: success={len(xml_paths) - len(failures)} fail={len(failures)}")
        if failures:
            raise RuntimeError(f"Batch finished with {len(failures)} failures.")
        return

    if args.xml is None:
        raise ValueError("Please provide --xml (single mode) or --batch-input-dir (batch mode).")

    xml_path = Path(args.xml)
    out_path = resolve_output_path(
        xml_path=xml_path,
        out_path=args.out,
        output_root=args.output_root,
        input_root=args.input_root,
        filename="kinematic_demo.mp4",
        drop_xml_parent_name=args.drop_xml_parent_name,
        drop_xml_parent_levels=args.drop_xml_parent_levels,
    )
    render_kinematic_demo(xml_path=xml_path, out_path=out_path, **common_kwargs)


if __name__ == "__main__":
    main()
