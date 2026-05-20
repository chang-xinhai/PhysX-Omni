"""
nvdiffrast-based mesh renderer.

Camera: orbit trajectory matching render_video_gt() in render_utils.py
  - yaw  : 0 → 2π (linear)
  - pitch: 0.25 + 0.5*sin(t) (oscillating)
  - r=2, fov=40°

Uses utils3d (already in trellis5 env) for camera matrices to ensure
correct coordinate conventions with nvdiffrast.

Run with:
  python3 render.py \
      --input  path/to/mesh.glb \
      --output path/to/output_dir \
      --num_frames 30 \
      --resolution 512
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
import utils3d.torch as utils3d
from PIL import Image


# ---------------------------------------------------------------------------
# Camera: orbit trajectory  (matches render_video_gt in render_utils.py)
# ---------------------------------------------------------------------------

def orbit_cameras(num_frames: int = 30, r: float = 2.0, fov_deg: float = 40.0, device: str = 'cuda'):
    """
    Generate orbit camera extrinsics + intrinsics.

    yaw  : linspace(0, 2π, num_frames)
    pitch: 0.25 + 0.5 * sin(linspace(0, 2π, num_frames))

    Returns:
        extrinsics : list of (4,4) w2c tensors
        intrinsics : list of (3,3) tensors  (normalised, cx=cy=0.5)
        fov_rad    : scalar float
    """
    ts = torch.linspace(0, 2 * math.pi, num_frames)
    yaws   = ts
    pitchs = 0.25 + 0.5 * torch.sin(ts)

    fov_rad = torch.deg2rad(torch.tensor(float(fov_deg))).to(device)
    target  = torch.zeros(3, device=device)
    up      = torch.tensor([0.0, 0.0, 1.0], device=device)

    extrinsics = []
    intrinsics = []
    for yaw, pitch in zip(yaws.tolist(), pitchs.tolist()):
        orig = torch.tensor([
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ], dtype=torch.float32, device=device) * r

        extr = utils3d.extrinsics_look_at(orig, target, up)        # (4,4) w2c
        intr = utils3d.intrinsics_from_fov_xy(fov_rad, fov_rad)    # (3,3)
        extrinsics.append(extr)
        intrinsics.append(intr)

    return extrinsics, intrinsics, float(fov_rad.item())


# ---------------------------------------------------------------------------
# Projection matrix  (same as mesh_renderer.py in Render_physx)
# ---------------------------------------------------------------------------

def intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """Normalised 3×3 intrinsics → 4×4 OpenGL-style projection matrix."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    proj = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    proj[0, 0] =  2 * fx
    proj[1, 1] =  2 * fy
    proj[0, 2] =  2 * cx - 1
    proj[1, 2] = -2 * cy + 1
    proj[2, 2] = far / (far - near)
    proj[2, 3] = near * far / (near - far)
    proj[3, 2] = 1.0
    return proj


# ---------------------------------------------------------------------------
# Mesh loading & normalisation
# ---------------------------------------------------------------------------

def load_mesh(filepath: str):
    """
    Load GLB/OBJ with trimesh, merge into single mesh.
    Returns vertices, faces, vertex_normals, and colour info.
    """
    scene = trimesh.load(filepath, force='scene')
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()          # trimesh ≥ 4.x (no deprecation warning)
    else:
        mesh = scene

    vertices  = np.array(mesh.vertices,       dtype=np.float32)
    faces     = np.array(mesh.faces,          dtype=np.int32)
    vert_norm = np.array(mesh.vertex_normals,  dtype=np.float32)

    # --- colour: prefer UV texture, fall back to vertex colours ---
    has_texture = False
    uv = tex_image = vert_colors = None

    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        try:
            mat = mesh.visual.material
            img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
            if img is not None:
                tex_image   = np.array(img.convert('RGB'), dtype=np.uint8)
                uv          = np.array(mesh.visual.uv, dtype=np.float32)
                has_texture = True
        except Exception:
            pass

    if not has_texture:
        try:
            colors      = mesh.visual.to_color().vertex_colors   # (V, 4) uint8
            vert_colors = colors[:, :3].astype(np.float32) / 255.0
        except Exception:
            vert_colors = np.full((len(vertices), 3), 0.7, dtype=np.float32)

    return vertices, faces, vert_norm, has_texture, uv, tex_image, vert_colors


def normalize_mesh(vertices: np.ndarray):
    """Scale + translate so mesh fits in [-0.5, 0.5]^3."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center   = (bbox_min + bbox_max) / 2.0
    extent   = (bbox_max - bbox_min).max()
    scale    = 1.0 / extent if extent > 0 else 1.0
    return (vertices - center) * scale, scale, center


# ---------------------------------------------------------------------------
# Lambertian shading  (world-space normals)
# ---------------------------------------------------------------------------

def phong_shading(normals_img: torch.Tensor, mask: torch.Tensor, cam_pos: torch.Tensor) -> torch.Tensor:
    """
    Phong shading: ambient + Lambertian diffuse + specular highlight.
    normals_img : (1, H, W, 3) world-space normals, values in [-1, 1]
    mask        : (1, H, W, 1)
    cam_pos     : (3,) camera world position (for specular)
    Returns     : (1, H, W, 3) shading multiplier in [0, 1]
    """
    lights = [
        (torch.tensor([4.0,  1.0,  6.0], device=normals_img.device), 0.55, 0.30),  # dir, diffuse, specular
        (torch.tensor([0.0,  0.0, 10.0], device=normals_img.device), 0.20, 0.05),
        (torch.tensor([0.0,  0.0,-10.0], device=normals_img.device), 0.08, 0.00),
    ]
    ambient   = 0.35   # higher ambient for metallic appearance
    shininess = 32.0

    view_dir = F.normalize(cam_pos, dim=0).reshape(1, 1, 1, 3)
    shading  = torch.full_like(normals_img[..., :1], ambient)

    for light_dir, diff_int, spec_int in lights:
        l    = F.normalize(light_dir, dim=0).reshape(1, 1, 1, 3)
        diff = torch.clamp((normals_img * l).sum(dim=-1, keepdim=True), 0.0, 1.0)
        shading = shading + diff * diff_int

        if spec_int > 0:
            reflect = 2.0 * (normals_img * l).sum(dim=-1, keepdim=True) * normals_img - l
            spec    = torch.clamp((reflect * view_dir).sum(dim=-1, keepdim=True), 0.0, 1.0)
            spec    = spec ** shininess
            shading = shading + spec * spec_int

    return torch.clamp(shading, 0.0, 1.0).expand_as(normals_img) * mask


# ---------------------------------------------------------------------------
# Core render
# ---------------------------------------------------------------------------

def render_mesh(
    filepath:    str,
    savepath:    str,
    num_frames:  int   = 30,
    resolution:  int   = 512,
    ssaa:        int   = 2,
    r:           float = 2.0,
    fov_deg:     float = 40.0,
    near:        float = 0.5,
    far:         float = 10.0,
    device:      str   = 'cuda',
):
    """
    Render orbit-trajectory frames of a GLB/OBJ using nvdiffrast.

    Saves:
        {savepath}/{i:03d}.png      RGBA images
        {savepath}/transforms.json  NeRF-format camera params
    """
    os.makedirs(savepath, exist_ok=True)

    # ── load & normalise ──────────────────────────────────────────────────
    vertices, faces, vert_norm, has_texture, uv, tex_image, vert_colors = load_mesh(filepath)
    vertices, norm_scale, norm_offset = normalize_mesh(vertices)

    verts_t  = torch.tensor(vertices,  dtype=torch.float32, device=device)
    faces_t  = torch.tensor(faces,     dtype=torch.int32,   device=device)
    vnorm_t  = F.normalize(torch.tensor(vert_norm, dtype=torch.float32, device=device), dim=-1)

    if has_texture:
        # GLB UV: v=0 at top (image convention)
        # nvdiffrast dr.texture: v=0 at bottom (OpenGL convention)
        # → flip V axis
        uv_flipped = uv.copy()
        uv_flipped[:, 1] = 1.0 - uv[:, 1]
        uv_t  = torch.tensor(uv_flipped, dtype=torch.float32, device=device)
        tex_t = torch.tensor(tex_image,  dtype=torch.float32, device=device) / 255.0
    else:
        vcol_t = torch.tensor(vert_colors, dtype=torch.float32, device=device)

    # ── cameras ───────────────────────────────────────────────────────────
    extrinsics, intrinsics, fov_rad = orbit_cameras(num_frames, r, fov_deg, device)

    # ── rasterise ─────────────────────────────────────────────────────────
    glctx  = dr.RasterizeCudaContext(device=device)
    frames = []
    res_aa = resolution * ssaa

    for idx, (w2c, intr) in enumerate(zip(extrinsics, intrinsics)):
        proj  = intrinsics_to_projection(intr, near, far)
        mvp   = (proj @ w2c).unsqueeze(0)                           # (1,4,4)

        v_h    = torch.cat([verts_t, torch.ones_like(verts_t[:, :1])], dim=-1)  # (V,4)
        v_clip = (mvp @ v_h.T).permute(0, 2, 1).contiguous()       # (1,V,4)

        rast, _ = dr.rasterize(glctx, v_clip, faces_t.contiguous(), (res_aa, res_aa))

        mask = (rast[..., -1:] > 0).float()                         # (1,H,W,1)

        # vertex normals (smooth shading)
        norm_img = dr.interpolate(vnorm_t.unsqueeze(0).contiguous(), rast, faces_t)[0]
        norm_img = F.normalize(norm_img, dim=-1)                     # (1,H,W,3)

        # colour
        if has_texture:
            texc, _   = dr.interpolate(uv_t.unsqueeze(0).contiguous(), rast, faces_t)
            color_img = dr.texture(tex_t.unsqueeze(0).contiguous(), texc, filter_mode='linear')
        else:
            color_img = dr.interpolate(vcol_t.unsqueeze(0).contiguous(), rast, faces_t)[0]

        # shading — pass camera world position for specular
        cam_pos = torch.inverse(w2c)[:3, 3]
        shading   = phong_shading(norm_img, mask, cam_pos)
        rgb       = torch.clamp(color_img * shading, 0.0, 1.0)

        # RGBA + antialias
        rgba = torch.cat([rgb, mask], dim=-1)                        # (1,H,W,4)
        rgba = dr.antialias(rgba, rast, v_clip, faces_t)

        if ssaa > 1:
            rgba = F.interpolate(
                rgba.permute(0, 3, 1, 2),
                (resolution, resolution),
                mode='bilinear', align_corners=False, antialias=True
            ).permute(0, 2, 3, 1)

        rgba_np = (rgba[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgba_np, 'RGBA').save(os.path.join(savepath, f'{idx:03d}.png'))

        # c2w for transforms.json
        c2w = torch.inverse(w2c).cpu().numpy().tolist()
        frames.append({
            'file_path':        f'{idx:03d}.png',
            'camera_angle_x':   fov_rad,
            'transform_matrix': c2w,
        })

    transforms = {
        'aabb':   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        'scale':  float(norm_scale),
        'offset': norm_offset.tolist(),
        'frames': frames,
    }
    with open(os.path.join(savepath, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f'[render] {num_frames} frames → {savepath}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Orbit-camera nvdiffrast renderer')
    parser.add_argument('--input',       type=str,   required=True)
    parser.add_argument('--output',      type=str,   required=True)
    parser.add_argument('--num_frames',  type=int,   default=30)
    parser.add_argument('--resolution',  type=int,   default=512)
    parser.add_argument('--ssaa',        type=int,   default=2)
    parser.add_argument('--r',           type=float, default=2.0)
    parser.add_argument('--fov',         type=float, default=40.0)
    args = parser.parse_args()

    render_mesh(
        args.input, args.output,
        num_frames=args.num_frames,
        resolution=args.resolution,
        ssaa=args.ssaa,
        r=args.r,
        fov_deg=args.fov,
    )
