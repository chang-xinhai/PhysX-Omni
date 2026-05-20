import os
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def convert_file_path_to_abs(elem, xml_path):
    if "file" in elem.attrib:
        file_path = elem.attrib["file"]
        if not os.path.isabs(file_path):
            elem.attrib["file"] = str((xml_path.parent / file_path).resolve())

    for child in elem:
        convert_file_path_to_abs(child, xml_path)


def apply_mesh_scale(elem, scale_factor):
    """
    Apply object-level scale to all <mesh scale="...">.
    If mesh has no scale, create scale="{s} {s} {s}".
    """
    if elem.tag == "mesh":
        if "scale" in elem.attrib:
            old_scale = [float(x) for x in elem.attrib["scale"].split()]
            if len(old_scale) == 1:
                old_scale = old_scale * 3
            new_scale = [x * scale_factor for x in old_scale]
        else:
            new_scale = [scale_factor, scale_factor, scale_factor]

        elem.attrib["scale"] = f"{new_scale[0]} {new_scale[1]} {new_scale[2]}"

    for child in elem:
        apply_mesh_scale(child, scale_factor)


def _parse_float_list(value):
    return [float(x) for x in value.split()]


def _format_float_list(values):
    return " ".join(f"{x:g}" for x in values)


def _scale_attr(elem, attr, scale_factor, expected_lengths=None):
    if attr not in elem.attrib:
        return

    values = _parse_float_list(elem.attrib[attr])
    if expected_lengths is not None and len(values) not in expected_lengths:
        return

    elem.attrib[attr] = _format_float_list([x * scale_factor for x in values])


def apply_kinematic_scale(elem, scale_factor, is_root=True):
    """
    Scale object-local spatial parameters together with mesh scale.

    Rotational parameters, such as hinge joint ranges and axes, are left
    unchanged. Slide joint ranges are linear distances, so they are scaled.
    """
    if elem.tag in {"body", "geom", "site", "camera", "light", "inertial"}:
        # The root body pose is replaced by the scene placement later, so keep
        # that external placement in scene units while scaling all child offsets.
        if not (is_root and elem.tag == "body"):
            _scale_attr(elem, "pos", scale_factor, expected_lengths={3})

    if elem.tag in {"geom", "site"}:
        _scale_attr(elem, "size", scale_factor)
        _scale_attr(elem, "fromto", scale_factor, expected_lengths={6})

    if elem.tag == "joint":
        _scale_attr(elem, "pos", scale_factor, expected_lengths={3})
        if elem.attrib.get("type") == "slide":
            _scale_attr(elem, "range", scale_factor, expected_lengths={2})

    for child in elem:
        apply_kinematic_scale(child, scale_factor, is_root=False)


def prefix_references(elem, prefix):
    attrs_to_prefix = {
        "name",
        "class",
        "mesh",
        "material",
        "texture",
    }

    for attr in attrs_to_prefix:
        if attr in elem.attrib:
            elem.attrib[attr] = prefix + elem.attrib[attr]

    for child in elem:
        prefix_references(child, prefix)


def set_body_pose(body, pos, euler=None, name=None):
    body = copy.deepcopy(body)

    if name is not None:
        body.set("name", name)

    body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")

    if euler is not None:
        body.set("euler", f"{euler[0]} {euler[1]} {euler[2]}")

    return body


def merge_mujoco_xmls(
    xml_paths,
    positions,
    output_path,
    eulers=None,
    scales=None,
    model_name="merged_scene",
):
    assert len(xml_paths) == len(positions), "xml_paths and positions must have same length."

    if eulers is None:
        eulers = [(0, 0, 0)] * len(xml_paths)

    if scales is None:
        scales = [1.0] * len(xml_paths)

    assert len(eulers) == len(xml_paths), "eulers and xml_paths must have same length."
    assert len(scales) == len(xml_paths), "scales and xml_paths must have same length."

    root = ET.Element("mujoco", {"model": model_name})

    ET.SubElement(root, "compiler", {
        "angle": "radian",
        "autolimits": "true",
    })

    ET.SubElement(root, "option", {
        "timestep": "0.002",
        "gravity": "0 0 -9.81",
        "wind": "0 0 0",
        "integrator": "implicitfast",
        "density": "1.225",
        "viscosity": "1.8e-05",
    })

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"realtime": "1"})
    ET.SubElement(visual, "quality", {
        "shadowsize": "16384",
        "numslices": "28",
        "offsamples": "4",
    })
    ET.SubElement(visual, "headlight", {
        "diffuse": "2 2 2",
        "specular": "0.5 0.5 0.5",
        "active": "1",
    })

    asset_out = ET.SubElement(root, "asset")
    default_out = ET.SubElement(root, "default")
    worldbody_out = ET.SubElement(root, "worldbody")

    ET.SubElement(asset_out, "texture", {
        "name": "plane",
        "type": "2d",
        "builtin": "checker",
        "rgb1": ".1 .1 .1",
        "rgb2": ".5 .5 .5",
        "width": "512",
        "height": "512",
        "mark": "cross",
        "markrgb": ".8 .8 .8",
    })

    ET.SubElement(asset_out, "material", {
        "name": "plane",
        "reflectance": "0.3",
        "texture": "plane",
        "texrepeat": "1 1",
        "texuniform": "true",
    })

    ET.SubElement(default_out, "geom", {
        "solref": ".5e-4",
        "solimp": "0.9 0.99 1e-4",
        "fluidcoef": "0.5 0.25 0.5 2.0 1.0",
    })

    ET.SubElement(worldbody_out, "geom", {
        "name": "floor",
        "pos": "0 0 0",
        "size": "0 0 .25",
        "type": "plane",
        "material": "plane",
        "condim": "6",
    })

    ET.SubElement(worldbody_out, "light", {
        "directional": "true",
        "ambient": ".3 .3 .3",
        "pos": "30 30 30",
        "dir": "0 -2 -1",
        "diffuse": ".5 .5 .5",
        "specular": ".5 .5 .5",
    })

    skybox_added = False

    for idx, xml_path in enumerate(xml_paths):
        xml_path = Path(xml_path).resolve()
        tree = ET.parse(xml_path)
        obj_root = tree.getroot()

        prefix = f"obj{idx}_"
        scale_factor = float(scales[idx])

        asset = obj_root.find("asset")
        if asset is not None:
            for child in asset:
                child_new = copy.deepcopy(child)

                if child_new.tag == "texture" and child_new.attrib.get("type") == "skybox":
                    if skybox_added:
                        continue
                    skybox_added = True

                convert_file_path_to_abs(child_new, xml_path)
                apply_mesh_scale(child_new, scale_factor)
                prefix_references(child_new, prefix)

                asset_out.append(child_new)

        default = obj_root.find("default")
        if default is not None:
            for child in default:
                child_new = copy.deepcopy(child)

                if child_new.tag == "geom":
                    continue

                prefix_references(child_new, prefix)
                default_out.append(child_new)

        worldbody = obj_root.find("worldbody")
        if worldbody is not None:
            bodies = worldbody.findall("body")

            for b_idx, body in enumerate(bodies):
                body_new = copy.deepcopy(body)
                apply_kinematic_scale(body_new, scale_factor)
                prefix_references(body_new, prefix)

                body_new = set_body_pose(
                    body_new,
                    pos=positions[idx],
                    euler=eulers[idx],
                    name=f"object_{idx}_{b_idx}",
                )

                worldbody_out.append(body_new)

    indent(root)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ET.ElementTree(root).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"Saved merged scene XML to: {output_path}")


if __name__ == "__main__":
    basepath = "./ours_demo"


    namelist = [
        "068f17e7bcf74d04bacf5524c4a2079d",
        "5aee96056818436fb578e45371c25783",
        "65e2f7b0452046be8ee948b49ee17ef4",
    ]  # object name

    #assume coordinate of object_1 is (x,y,z) using code in ./applications_scene.
    #positions should be (z,x,-y)

    positions = [
        (8.17, -5, 1.5),   
        (8.0, 0, 0.07),
        (9.53, 4, 1.3),
    ] # object location   

    eulers = [
        (0, 0, 1.73),
        (0, 0, 0),
        (0, 0, 0),
    ] # object angle

    scales = [
        1.0,
        1.0,
        1.0,
    ] # object scale

    # Prevent the object from being underground.
    positions = [
        (x, y, 7 - z)
        for x, y, z in positions
    ]

    xml_paths = []
    for name in namelist:
        xml_paths.append(os.path.join(basepath, name, "basic.xml"))

    merge_mujoco_xmls(
        xml_paths=xml_paths,
        positions=positions,
        eulers=eulers,
        scales=scales,
        output_path="./scene.xml",
        model_name="merged_scene",
    )
