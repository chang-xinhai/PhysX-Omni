import glob
import os
import json
import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from depth_anything_3.api import DepthAnything3


def random_color_from_name(name: str):

    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return rng.integers(0, 256, size=3, dtype=np.uint8)


def load_view_masks(mask_root, view_idx):

    view_dir = os.path.join(mask_root + "_" + str(view_idx))
    masks_dict = {}

    if not os.path.isdir(view_dir):
        print(f"[Warning] mask dir not found: {view_dir}")
        return masks_dict

    for mask_path in sorted(glob.glob(os.path.join(view_dir, "*.png"))):
        tag = Path(mask_path).stem
        mask = np.array(Image.open(mask_path).convert("L"))
        masks_dict[tag] = mask > 0

    return masks_dict


def resize_mask_to_depth(mask, depth_shape):
    """
    mask: [H0, W0] bool/uint8
    depth_shape: (H, W)
    """
    h, w = depth_shape
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
    mask_resized = np.array(mask_img) > 0
    return mask_resized


def get_object_size_type(mask):
    """
    - small: area_ratio < 0.1
    - medium: 0.1 <= area_ratio < 0.4
    - large: area_ratio >= 0.4
    """
    h, w = mask.shape
    total_area = float(h * w)
    obj_area = float(mask.sum())
    area_ratio = obj_area / max(total_area, 1.0)

    if area_ratio < 0.1:
        size_type = "small"
    elif area_ratio < 0.4:
        size_type = "medium"
    else:
        size_type = "large"

    return size_type, area_ratio


def mask_bbox(mask):
    """
    mask: [H, W] bool
    return: (x0, y0, x1, y1), where x1/y1 are exclusive bounds
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return (x0, y0, x1, y1)


def bbox_change_metrics(mask_a, mask_b):

    box_a = mask_bbox(mask_a)
    box_b = mask_bbox(mask_b)

    if box_a is None or box_b is None:
        return {
            "valid": False,
            "iou": 0.0,
            "dw": 1.0,
            "dh": 1.0,
            "da": 1.0,
        }

    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    aw = max(ax1 - ax0, 1)
    ah = max(ay1 - ay0, 1)
    bw = max(bx1 - bx0, 1)
    bh = max(by1 - by0, 1)

    area_a = aw * ah
    area_b = bw * bh

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(ix1 - ix0, 0)
    ih = max(iy1 - iy0, 0)
    inter = iw * ih
    union = area_a + area_b - inter
    iou = inter / max(union, 1)

    dw = abs(bw - aw) / max(aw, 1)
    dh = abs(bh - ah) / max(ah, 1)
    da = abs(area_b - area_a) / max(area_a, 1)

    return {
        "valid": True,
        "iou": float(iou),
        "dw": float(dw),
        "dh": float(dh),
        "da": float(da),
    }


def bbox_change_is_large(mask_before, mask_after, iou_thresh=0.75, delta_thresh=0.25):

    m = bbox_change_metrics(mask_before, mask_after)
    if not m["valid"]:
        return True

    if m["iou"] < iou_thresh:
        return True
    if m["dw"] > delta_thresh or m["dh"] > delta_thresh or m["da"] > 0.4:
        return True
    return False


def changed_region_fg_ratio(component_mask, eroded_mask):

    removed = component_mask & (~eroded_mask)
    box = mask_bbox(removed)
    if box is None:
        return 1.0

    x0, y0, x1, y1 = box
    region = component_mask[y0:y1, x0:x1]
    fg_ratio = float(region.sum()) / max(region.size, 1)
    return fg_ratio


def get_component_masks(mask):
 
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)

    components = []
    for lab in range(1, num_labels):
        comp = labels == lab
        if comp.sum() > 0:
            components.append(comp)
    return components


def multiscale_component_erosion(mask, kernels=(1, 3, 5), fg_ratio_thresh=0.3):

    h, w = mask.shape
    out_mask = np.zeros((h, w), dtype=bool)

    components = get_component_masks(mask)
    debug_info = []

    kernels = sorted(set(kernels), reverse=True)

    for comp_idx, comp in enumerate(components):
        comp_u8 = comp.astype(np.uint8)
        chosen_mask = comp.copy()
        chosen_k = 1
        decision = "keep_original"

        for k in kernels:
            kernel = np.ones((k, k), np.uint8)
            eroded = cv2.erode(comp_u8, kernel, iterations=1).astype(bool)

            if eroded.sum() == 0:
                continue

            large_change = bbox_change_is_large(comp, eroded)

            if not large_change:
                chosen_mask = eroded
                chosen_k = k
                decision = f"use_k_{k}"
                break
            else:
                fg_ratio = changed_region_fg_ratio(comp, eroded)
                if fg_ratio < fg_ratio_thresh:
                    chosen_mask = comp
                    chosen_k = 1
                    decision = f"thin_parts_keep_original_fg_ratio_{fg_ratio:.3f}"
                    break
                else:
                    continue

        out_mask |= chosen_mask

        debug_info.append(
            {
                "component_id": comp_idx,
                "orig_area": int(comp.sum()),
                "final_area": int(chosen_mask.sum()),
                "chosen_k": int(chosen_k),
                "decision": decision,
            }
        )

    return out_mask, debug_info


def remove_depth_edges_adaptive(depth, mask, base_grad_thresh=0.05):

    size_type, _ = get_object_size_type(mask)

    if size_type == "small":
        return mask

    depth = depth.astype(np.float32, copy=False)
    gx = np.abs(np.gradient(depth, axis=1))
    gy = np.abs(np.gradient(depth, axis=0))
    grad = np.sqrt(gx ** 2 + gy ** 2)

    if size_type == "medium":
        grad_thresh = base_grad_thresh * 1.4
    else:
        grad_thresh = base_grad_thresh

    clean_mask = mask & (grad < grad_thresh)
    return clean_mask


def chunked_knn_mean_distance(points, k, chunk_size=2048):

    points = np.asarray(points, dtype=np.float32)
    n = len(points)

    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    if n < k + 1:
        k = max(1, n - 1)

    knn_mean = np.empty((n,), dtype=np.float32)

    all_sq = np.sum(points * points, axis=1, dtype=np.float32)  # [N]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[start:end]  # [B, 3]
        chunk_sq = np.sum(chunk * chunk, axis=1, dtype=np.float32)[:, None]  # [B, 1]

        # dist^2 = ||a||^2 + ||b||^2 - 2 a·b
        dist_sq = chunk_sq + all_sq[None, :] - 2.0 * (chunk @ points.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)

        row_idx = np.arange(end - start)
        col_idx = np.arange(start, end)
        dist_sq[row_idx, col_idx] = np.inf

        part = np.partition(dist_sq, kth=k - 1, axis=1)[:, :k]
        part = np.sqrt(part, dtype=np.float32)
        knn_mean[start:end] = part.mean(axis=1, dtype=np.float32)

        del chunk, chunk_sq, dist_sq, part
        gc.collect()

    return knn_mean


def remove_outliers_knn_adaptive(points, colors=None, area_ratio=None, k_default=16):

    points = np.asarray(points, dtype=np.float32)
    n = len(points)

    if n < 30:
        return points, colors

    if area_ratio is None:
        area_ratio = 0.2

    if area_ratio < 0.1:
        k = min(max(4, n // 30), 8)
        std_ratio = 3.0
    elif area_ratio < 0.4:
        k = min(max(6, n // 25), 12)
        std_ratio = 2.7
    else:
        k = min(k_default, max(8, n // 20))
        std_ratio = 2.5

    if n < k + 2:
        return points, colors

    knn_mean = chunked_knn_mean_distance(points, k=k, chunk_size=2048)
    mu = knn_mean.mean()
    sigma = knn_mean.std()

    if sigma < 1e-12:
        keep = np.ones(n, dtype=bool)
    else:
        keep = knn_mean < (mu + std_ratio * sigma)

    points_filtered = points[keep]
    colors_filtered = colors[keep] if colors is not None else None

    del knn_mean, keep
    gc.collect()

    return points_filtered, colors_filtered


def depth_aware_bbox_filter(points, colors=None, area_ratio=None):

    points = np.asarray(points, dtype=np.float32)
    n = len(points)

    if n < 30:
        return points, colors

    z = points[:, 2].astype(np.float32, copy=False)
    z_min = np.percentile(z, 5)
    z_max = np.percentile(z, 95)

    if z_max - z_min < 1e-8:
        return points, colors

    z_clip = np.clip(z, z_min, z_max)
    z_norm = (z_clip - z_min) / (z_max - z_min)  # near -> 0, far -> 1

    if area_ratio is None:
        area_ratio = 0.2

    if area_ratio < 0.1:
        k = 6
        near_std_ratio = 3.5
        far_std_ratio = 2.6
    elif area_ratio < 0.4:
        k = 10
        near_std_ratio = 3.2
        far_std_ratio = 2.4
    else:
        k = 16
        near_std_ratio = 3.0
        far_std_ratio = 2.2

    k = min(k, max(4, n - 2))
    if n < k + 2:
        return points, colors

    knn_mean = chunked_knn_mean_distance(points, k=k, chunk_size=2048)

    mu = knn_mean.mean()
    sigma = knn_mean.std()

    if sigma < 1e-12:
        keep = np.ones(n, dtype=bool)
    else:
        point_std_ratio = near_std_ratio * (1.0 - z_norm) + far_std_ratio * z_norm
        threshold = mu + point_std_ratio * sigma
        keep = knn_mean < threshold

    points_filtered = points[keep]
    colors_filtered = colors[keep] if colors is not None else None

    del z, z_clip, z_norm, knn_mean, keep
    gc.collect()

    return points_filtered, colors_filtered


def compute_robust_aabb(points, low=1.0, high=99.0):

    points = np.asarray(points, dtype=np.float32)
    mins = np.percentile(points, low, axis=0)
    maxs = np.percentile(points, high, axis=0)
    center = (mins + maxs) / 2.0
    extent = maxs - mins
    return center, extent, mins, maxs


def sample_line_segment(p0, p1, step):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    length = np.linalg.norm(p1 - p0)
    n = max(int(np.ceil(length / max(step, 1e-8))) + 1, 2)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    pts = (1.0 - t) * p0[None, :] + t * p1[None, :]
    return pts


def build_bbox_edge_points(mins, maxs, step=None, color=(255, 0, 0)):
    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)

    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()

    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    diag = np.linalg.norm(maxs - mins)
    if step is None:
        step = max(diag / 80.0, 1e-4)

    all_pts = []
    for i0, i1 in edges:
        pts = sample_line_segment(corners[i0], corners[i1], step=step)
        all_pts.append(pts)

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.repeat(np.array(color, dtype=np.uint8)[None, :], len(all_pts), axis=0)
    return all_pts, all_cols


def depth_to_point_cloud_with_mask(
    depth,
    image,
    K,
    object_mask,
    extrinsic_w2c=None,
    conf=None,
    conf_thresh=None,
    depth_min=1e-6,
):

    h, w = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32, copy=False)

    valid = (z > depth_min) & object_mask

    if conf is not None and conf_thresh is not None:
        valid = valid & (conf > conf_thresh)

    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    u = u[valid].astype(np.float32, copy=False)
    v = v[valid].astype(np.float32, copy=False)
    z = z[valid].astype(np.float32, copy=False)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    colors_rgb = image[valid].astype(np.float32, copy=False) / 255.0

    if extrinsic_w2c is not None:
        if extrinsic_w2c.shape == (3, 4):
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :4] = extrinsic_w2c
        elif extrinsic_w2c.shape == (4, 4):
            w2c = extrinsic_w2c.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unexpected extrinsic shape: {extrinsic_w2c.shape}")

        c2w = np.linalg.inv(w2c)
        points_cam_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1
        )
        points_world_h = (c2w @ points_cam_h.T).T
        points_world = points_world_h[:, :3].astype(np.float32, copy=False)

        del w2c, c2w, points_cam_h, points_world_h
    else:
        points_world = points_cam

    return points_world, colors_rgb


def depth_to_point_cloud(depth, image, K, extrinsic_w2c=None, conf=None, conf_thresh=None, depth_min=1e-6):

    h, w = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32, copy=False)

    mask = z > depth_min
    if conf is not None and conf_thresh is not None:
        mask = mask & (conf > conf_thresh)

    u = u[mask].astype(np.float32, copy=False)
    v = v[mask].astype(np.float32, copy=False)
    z = z[mask].astype(np.float32, copy=False)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    colors = image[mask].astype(np.float32, copy=False) / 255.0

    if extrinsic_w2c is not None:
        if extrinsic_w2c.shape == (3, 4):
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :4] = extrinsic_w2c
        elif extrinsic_w2c.shape == (4, 4):
            w2c = extrinsic_w2c.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unexpected extrinsic shape: {extrinsic_w2c.shape}")

        c2w = np.linalg.inv(w2c)
        points_cam_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1
        )
        points_world_h = (c2w @ points_cam_h.T).T
        points_world = points_world_h[:, :3].astype(np.float32, copy=False)

        del w2c, c2w, points_cam_h, points_world_h
    else:
        points_world = points_cam

    return points_world, colors

def refine_bbox_by_depth_slices(points, n_slices=10, keep_ratio=0.8, depth_axis=2):

    points = np.asarray(points, dtype=np.float32)

    if len(points) == 0:
        return points, {
            "valid": False,
            "reason": "empty_points"
        }

    if len(points) < n_slices:
        return points, {
            "valid": True,
            "reason": "too_few_points_use_all",
            "num_points_total": int(len(points)),
            "num_points_kept": int(len(points)),
        }

    depth_vals = points[:, depth_axis]
    z_min = float(depth_vals.min())
    z_max = float(depth_vals.max())

    if z_max - z_min < 1e-8:
        return points, {
            "valid": True,
            "reason": "flat_depth_use_all",
            "num_points_total": int(len(points)),
            "num_points_kept": int(len(points)),
            "depth_min": z_min,
            "depth_max": z_max,
        }


    edges = np.linspace(z_min, z_max, n_slices + 1, dtype=np.float32)

    total_points = len(points)
    target_points = max(1, int(np.ceil(total_points * keep_ratio)))

    selected_mask = np.zeros(total_points, dtype=bool)
    cumulative = 0
    chosen_slice_idx = n_slices - 1

    for i in range(n_slices):
        z0 = edges[i]
        z1 = edges[i + 1]

        if i == n_slices - 1:
            slice_mask = (depth_vals >= z0) & (depth_vals <= z1)
        else:
            slice_mask = (depth_vals >= z0) & (depth_vals < z1)

        selected_mask |= slice_mask
        cumulative = int(selected_mask.sum())

        if cumulative >= target_points:
            chosen_slice_idx = i
            break

    refined_points = points[selected_mask]

    if len(refined_points) < 10:
        refined_points = points
        return refined_points, {
            "valid": True,
            "reason": "refined_too_few_use_all",
            "num_points_total": int(total_points),
            "num_points_kept": int(len(refined_points)),
            "target_points": int(target_points),
        }

    return refined_points, {
        "valid": True,
        "reason": "success",
        "num_points_total": int(total_points),
        "num_points_kept": int(len(refined_points)),
        "target_points": int(target_points),
        "keep_ratio": float(keep_ratio),
        "n_slices": int(n_slices),
        "depth_min": z_min,
        "depth_max": z_max,
        "chosen_slice_idx": int(chosen_slice_idx),
        "chosen_depth_max": float(edges[chosen_slice_idx + 1]),
    }

def save_ply(filename, points, colors=None):
    """
    Save point cloud to ASCII PLY
    """
    assert points.ndim == 2 and points.shape[1] == 3

    if colors is None:
        colors = np.ones_like(points, dtype=np.uint8) * 255
    else:
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


# =========================
# main
# =========================

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

example_path = "outputs"
mask_root = os.path.join(example_path, "mask")
save_object_dir = "outputs/ply"
os.makedirs(save_object_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(example_path, "raw_image.jpg")))

prediction = model.inference(images)

# -------- full point cloud --------
all_points = []
all_colors = []

for i in range(len(images)):
    img = prediction.processed_images[i]
    depth = prediction.depth[i]
    conf = prediction.conf[i]
    K = prediction.intrinsics[i]
    ext = prediction.extrinsics[i]

    points_world, colors = depth_to_point_cloud(
        depth=depth,
        image=img,
        K=K,
        extrinsic_w2c=ext,
        conf=conf,
        conf_thresh=0.1,
        depth_min=1e-6,
    )

    all_points.append(points_world)
    all_colors.append(colors)

all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)
save_ply(os.path.join(save_object_dir, "pointcloud_fused.ply"), all_points, all_colors)

# -------- segmented point cloud --------
all_seg_points = []
all_seg_colors = []

object_points_dict = {}
object_colors_dict = {}
object_area_ratio_dict = {}

for i in range(len(images)):
    img = prediction.processed_images[i]
    depth = prediction.depth[i]
    conf = prediction.conf[i]
    K = prediction.intrinsics[i]
    ext = prediction.extrinsics[i]

    masks_dict = load_view_masks(mask_root, i)

    if len(masks_dict) == 0:
        print(f"[Info] no masks for view {i:03d}, skip")
        continue

    for tag, obj_mask in masks_dict.items():
        obj_mask = resize_mask_to_depth(obj_mask, depth.shape)

        size_type_before, area_ratio_before = get_object_size_type(obj_mask)

        obj_mask, erosion_debug = multiscale_component_erosion(
            obj_mask,
            kernels=(1, 3),
            fg_ratio_thresh=0.3,
        )

        obj_mask = remove_depth_edges_adaptive(depth, obj_mask, base_grad_thresh=0.05)

        if obj_mask.sum() == 0:
            print(f"[Info] empty cleaned mask for view {i:03d} | {tag}, skip")
            continue

        size_type_after, area_ratio_after = get_object_size_type(obj_mask)

        if area_ratio_after < 0.1:
            conf_thresh = 0.10
        elif area_ratio_after < 0.4:
            conf_thresh = 0.18
        else:
            conf_thresh = 0.25

        points_world, _ = depth_to_point_cloud_with_mask(
            depth=depth,
            image=img,
            K=K,
            object_mask=obj_mask,
            extrinsic_w2c=ext,
            conf=conf,
            conf_thresh=conf_thresh,
            depth_min=1e-6,
        )

        if len(points_world) == 0:
            continue

        points_world, _ = remove_outliers_knn_adaptive(
            points_world,
            colors=None,
            area_ratio=area_ratio_after,
            k_default=16,
        )

        if len(points_world) == 0:
            continue

        seg_color = random_color_from_name(tag)
        seg_colors = np.repeat(seg_color[None, :], len(points_world), axis=0)

        all_seg_points.append(points_world)
        all_seg_colors.append(seg_colors)

        if tag not in object_points_dict:
            object_points_dict[tag] = []
            object_colors_dict[tag] = []
            object_area_ratio_dict[tag] = []

        object_points_dict[tag].append(points_world)
        object_colors_dict[tag].append(seg_colors)
        object_area_ratio_dict[tag].append(area_ratio_after)

        print(
            f"view {i:03d} | {tag}: {len(points_world)} points | "
            f"size_before={size_type_before} | area_ratio_before={area_ratio_before:.4f} | "
            f"size_after={size_type_after} | area_ratio_after={area_ratio_after:.4f} | "
            f"conf_thresh={conf_thresh}"
        )
        print(f"[ErodeDebug] view {i:03d} | {tag} | {erosion_debug}")

        del obj_mask, points_world, seg_colors
        gc.collect()

    del masks_dict
    gc.collect()

if len(all_seg_points) > 0:
    all_seg_points = np.concatenate(all_seg_points, axis=0)
    all_seg_colors = np.concatenate(all_seg_colors, axis=0)
    save_ply(os.path.join(save_object_dir, "segmentation_fused.ply"), all_seg_points, all_seg_colors)
    print(f"Saved segmentation_fused.ply: {len(all_seg_points)} points")

# -------- bbox and visualization --------
bboxes_dict = {}
all_bbox_vis_points = []
all_bbox_vis_colors = []

for tag in object_points_dict:
    pts = np.concatenate(object_points_dict[tag], axis=0)
    cols = np.concatenate(object_colors_dict[tag], axis=0)
    mean_area_ratio = float(np.mean(object_area_ratio_dict[tag]))

    if len(pts) == 0:
        print(f"[Info] {tag} has no valid points, skip")
        continue


    save_ply(os.path.join(save_object_dir, f"{tag}.ply"), pts, cols)
    print(f"Saved {tag}.ply: {len(pts)} points")

   
    bbox_pts_filtered, _ = depth_aware_bbox_filter(
        pts,
        colors=None,
        area_ratio=mean_area_ratio,
    )

    if len(bbox_pts_filtered) < 10:
        bbox_pts_filtered = pts


    bbox_pts_refined, depth_refine_debug = refine_bbox_by_depth_slices(
        bbox_pts_filtered,
        n_slices=20,
        keep_ratio=0.9,
        depth_axis=2,   
    )

    if len(bbox_pts_refined) >= 10:
        center, extent, mins, maxs = compute_robust_aabb(
            bbox_pts_refined,
            low=1.0,
            high=99.0,
        )

        if mean_area_ratio < 0.1:
            size_type = "small"
        elif mean_area_ratio < 0.4:
            size_type = "medium"
        else:
            size_type = "large"

        bboxes_dict[tag] = {
            "center": center.tolist(),
            "extent": extent.tolist(),
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
            "num_points_seg": int(len(pts)),
            "num_points_bbox_filtered": int(len(bbox_pts_filtered)),
            "num_points_bbox_refined": int(len(bbox_pts_refined)),
            "mean_area_ratio": mean_area_ratio,
            "size_type": size_type,
            "depth_refine_debug": depth_refine_debug,
        }

        print(
            f"[BBox] {tag} center={center}, extent={extent}, size={size_type}, "
            f"bbox_points_refined={len(bbox_pts_refined)}/{len(pts)}, "
            f"depth_refine={depth_refine_debug}"
        )

        bbox_vis_pts, bbox_vis_cols = build_bbox_edge_points(
            mins, maxs, step=None, color=(255, 0, 0)
        )

        pts_vis = np.concatenate([pts, bbox_vis_pts], axis=0)
        cols_vis = np.concatenate([cols, bbox_vis_cols], axis=0)
        save_ply(os.path.join(save_object_dir, f"{tag}_with_bbox.ply"), pts_vis, cols_vis)

        all_bbox_vis_points.append(bbox_vis_pts)
        all_bbox_vis_colors.append(bbox_vis_cols)
    else:
        print(f"[Info] {tag} has too few points for bbox after depth refinement")

    del pts, cols
    if "bbox_pts_filtered" in locals():
        del bbox_pts_filtered
    if "bbox_pts_refined" in locals():
        del bbox_pts_refined
    gc.collect()

bbox_json_path = os.path.join(save_object_dir, "bboxes.json")
with open(bbox_json_path, "w", encoding="utf-8") as f:
    json.dump(bboxes_dict, f, indent=2, ensure_ascii=False)

print(f"Saved bbox json to: {bbox_json_path}")

if len(all_seg_points) > 0 and len(all_bbox_vis_points) > 0:
    all_bbox_vis_points = np.concatenate(all_bbox_vis_points, axis=0)
    all_bbox_vis_colors = np.concatenate(all_bbox_vis_colors, axis=0)

    merged_vis_points = np.concatenate([all_seg_points, all_bbox_vis_points], axis=0)
    merged_vis_colors = np.concatenate([all_seg_colors, all_bbox_vis_colors], axis=0)

    save_ply(
        os.path.join(save_object_dir, "segmentation_with_bboxes.ply"),
        merged_vis_points,
        merged_vis_colors,
    )
    print("Saved segmentation_with_bboxes.ply")

gc.collect()
