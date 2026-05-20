"""
自适应并行渲染脚本 —— 统一入口

支持三种视角模式（--view_mode）：
  random      : Hammersley 球面随机视角（默认，25 帧）
  orbit       : 环绕轨道视角（yaw 0→2π, pitch 波动, 30 帧, r=2, fov=40°）
  fixed_grid  : 固定网格视角（3仰角 × 8方位 = 24 帧）

支持动态 N-GPU（--gpus 0 1 2 3），所有数据集 job 统一分配到各 GPU。

自适应并行：根据 mesh 文件大小动态调整 worker 数，避免大文件 OOM。

GT 相机对齐（--gt_output_root）：
  random 模式下，若指定 --gt_output_root，则每个 item 从对应 GT 的
  transforms.json 中提取相机位置，复用相同视角，而非重新随机生成。
  GT transforms.json 中的 transform_matrix 第 4 列即相机位置 (x,y,z)，
  可直接反推 yaw/pitch/radius；fov 取 camera_angle_x 字段。

示例：
  # GT 渲染（random，生成随机相机）
  python render_adaptive.py \
      --dataset_roots physx_result/PhysX-Mobility/partseg \
      --gpus 0 \
      --view_mode random \
      --output_root benchmark/benchmark_assets/rendered_views/description/gt

  # 非 GT 渲染（复用 GT 相机）
  python render_adaptive.py \
      --dataset_roots physx_result/ours_mobility_181500 \
      --gpus 0 \
      --view_mode random \
      --gt_output_root benchmark/benchmark_assets/rendered_views/description/gt/gt_dataset \
      --output_root benchmark/benchmark_assets/rendered_views/description
"""

import argparse
import json
import math
import os
import glob as g
from multiprocessing import Process, Queue
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 视角生成
# ---------------------------------------------------------------------------

VIEWS_JSON           = os.path.join(os.path.dirname(__file__), "views.json")
VIEWS_FIXED_GRID_JSON = os.path.join(os.path.dirname(__file__), "views_fixed_grid.json")


def _make_random_views(num_views: int = 25) -> list:
    """Hammersley 球面随机视角，返回 views 列表。"""
    from utils.sampling import sphere_hammersley_sequence
    offset = (np.random.rand(), np.random.rand())
    yaws, pitchs = [], []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)

    fov_min, fov_max = 10, 70
    r_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    r_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min, k_max = 1 / r_max ** 2, 1 / r_min ** 2
    ks     = np.random.uniform(k_min, k_max, num_views)
    radii  = [1 / np.sqrt(k) for k in ks]
    fovs   = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radii]

    return [{"yaw": y, "pitch": p, "radius": r, "fov": f}
            for y, p, r, f in zip(yaws, pitchs, radii, fovs)]


def _write_random_views(num_views: int = 25) -> None:
    """Hammersley 球面随机视角，写入 views.json。"""
    views = _make_random_views(num_views)
    with open(VIEWS_JSON, "w", encoding="utf-8") as fp:
        json.dump(views, fp, indent=4)


def _make_orbit_views(num_frames: int = 30, r: float = 2.0, fov_deg: float = 40.0) -> list:
    """环绕轨道视角，返回 views 列表。"""
    fov_rad = math.radians(fov_deg)
    ts = np.linspace(0, 2 * math.pi, num_frames)
    return [
        {"yaw":   float(math.pi / 2 - t),
         "pitch": float(0.25 + 0.5 * math.sin(t)),
         "radius": r, "fov": fov_rad}
        for t in ts
    ]


def _write_orbit_views(num_frames: int = 30, r: float = 2.0, fov_deg: float = 40.0) -> None:
    """环绕轨道视角，写入 views.json。"""
    views = _make_orbit_views(num_frames, r, fov_deg)
    with open(VIEWS_JSON, "w", encoding="utf-8") as fp:
        json.dump(views, fp, indent=4)


def _write_fixed_grid_views(fov_deg: float = 40.0) -> None:
    """固定网格视角，写入 views_fixed_grid.json。"""
    elevations = [-15, 20, 45]
    azimuths   = [0, 45, 90, 135, 180, 225, 270, 315]
    fov_rad    = math.radians(fov_deg)
    # random 公式: r = sqrt(3)/2 / sin(fov/2) 确保 unit cube 对角线恰好填满画面
    radius     = math.sqrt(3) / 2 / math.sin(fov_rad / 2)
    views = [
        {"yaw": float(math.radians(azim)), "pitch": float(math.radians(elev)),
         "radius": radius, "fov": fov_rad, "elev_deg": elev, "azim_deg": azim}
        for elev in elevations for azim in azimuths
    ]
    with open(VIEWS_FIXED_GRID_JSON, "w", encoding="utf-8") as fp:
        json.dump(views, fp, indent=4)


# ---------------------------------------------------------------------------
# GT 相机复用（random 模式）
# ---------------------------------------------------------------------------

def _extract_views_from_transforms(transforms_path: str) -> list:
    """从 GT 的 transforms.json 提取相机参数，转换为 views.json 格式。

    transform_matrix 第 4 列（索引 3）即相机位置 (x, y, z)，反推：
      radius = sqrt(x²+y²+z²)
      pitch  = arcsin(z / radius)
      yaw    = arctan2(y, x)
    fov 直接取 camera_angle_x 字段。
    """
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    views = []
    for frame in transforms["frames"]:
        m = frame["transform_matrix"]
        x, y, z = m[0][3], m[1][3], m[2][3]
        radius = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        if radius < 1e-6:
            continue
        pitch = math.asin(max(-1.0, min(1.0, z / radius)))
        yaw   = math.atan2(y, x)
        fov   = frame["camera_angle_x"]
        views.append({"yaw": yaw, "pitch": pitch, "radius": radius, "fov": fov})
    return views


def _find_gt_transforms(gt_output_root: str, item_id: str) -> Optional[str]:
    """在 gt_output_root 下找 item_id 对应的 transforms.json。

    支持两种 GT 输出结构：
      - 目录型：gt_output_root/<item_id>/transforms.json      （partseg 等）
      - 平铺文件型：gt_output_root/<item_id>.glb/transforms.json（wholeglb 等）
    """
    # 目录型
    p = os.path.join(gt_output_root, item_id, "transforms.json")
    if os.path.exists(p):
        return p
    # 平铺文件型（.glb / .gltf / .obj）
    for ext in (".glb", ".gltf", ".obj"):
        p = os.path.join(gt_output_root, item_id + ext, "transforms.json")
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Blender 调用
# ---------------------------------------------------------------------------

def _get_blender_cmd():
    blender_bin = os.path.expanduser(os.environ.get("BLENDER_BIN", "blender"))
    if not os.path.exists(blender_bin):
        blender_bin = "blender"
    blender_lib = os.path.expanduser(os.environ.get("BLENDER_LD_LIBRARY_PATH", ""))
    if not os.path.isdir(blender_lib):
        blender_lib = ""
    return blender_bin, blender_lib


def _render_one(savepath: str, mesh_path: str, view_mode: str,
                num_views: int, rotate: int, rotate_x: float = 0.0,
                resolution: int = 512,
                gt_transforms_path: Optional[str] = None) -> None:
    """渲染单个 mesh（在 worker 进程内调用）。

    gt_transforms_path: 若提供，random 模式下从 GT transforms.json 提取相机，
                        而非重新随机生成。orbit / fixed_grid 模式忽略此参数。
    views 写入 {savepath}/views_input.json（per-item 独立文件，避免并发竞争）。
    """
    from subprocess import call

    os.makedirs(savepath, exist_ok=True)
    blender_bin, blender_lib = _get_blender_cmd()

    if view_mode == "fixed_grid":
        script = os.path.join(os.path.dirname(__file__),
                              "blender_script", "render_fixed_grid.py")
        _write_fixed_grid_views()
    else:
        script = os.path.join(os.path.dirname(__file__),
                              "blender_script", "render_mobility.py")
        # 构建 views 列表
        if view_mode == "orbit":
            views = _make_orbit_views(num_frames=num_views)
        elif gt_transforms_path is not None:
            # random + GT 对齐：从 GT transforms.json 提取相机位置
            views = _extract_views_from_transforms(gt_transforms_path)
        else:
            # random：随机生成
            views = _make_random_views(num_views=num_views)
        # 写入 per-item views 文件（避免多进程并发写同一全局文件）
        views_input = os.path.join(savepath, "views_input.json")
        with open(views_input, "w", encoding="utf-8") as fp:
            json.dump(views, fp, indent=4)

    cmd = [blender_bin, "--background", "--python", script, "--",
           "--object", os.path.expanduser(mesh_path),
           "--output_folder", savepath]
    if view_mode != "fixed_grid":
        cmd += ["--rotate", str(rotate)]
        if rotate_x:
            cmd += ["--rotate_x", str(rotate_x)]
        cmd += ["--views_json", views_input]
    cmd += ["--resolution", str(resolution)]

    env = os.environ.copy()
    if blender_lib:
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = blender_lib if not ld else f"{blender_lib}:{ld}"

    ret = call(cmd, env=env)
    if ret != 0:
        raise RuntimeError(f"Blender failed (exit {ret}) for {mesh_path}")
    transforms_file = os.path.join(savepath, "transforms.json")
    if not os.path.exists(transforms_file):
        raise RuntimeError(f"Blender exited 0 but transforms.json not written for {mesh_path}")


# ---------------------------------------------------------------------------
# Job 收集 & 大小分类
# ---------------------------------------------------------------------------

def _get_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024


def _size_category(size_mb: float) -> str:
    if size_mb < 5:   return "small"
    if size_mb < 10:  return "medium"
    if size_mb < 15:  return "large"
    return "xlarge"


def _find_mesh(item_dir: str):
    """
    按优先级探测 item 目录下的 mesh 文件或目录，返回路径或 None。
      1. mesh.glb    — physxanything_*（单 GLB 文件）
      2. texture.glb — physxgen_*, gt_verse（单 GLB 文件）
      3. merged.obj  — gt_mobility, ours_*, articulateanything_* 合并版
      4. objs/       — ours_*, articulateanything_* 原始多 part 目录
                       Blender 会递归 glob objs/**/*.obj 分别加载每个 part，
                       避免合并导致的 UV 重叠纹理问题
    返回文件路径（单文件格式）或目录路径（multi-part objs/ 格式）。
    """
    # 单文件格式（按优先级）
    for fname in ("mesh.glb", "texture.glb", "sample.glb", "merged.obj"):
        p = os.path.join(item_dir, fname)
        if os.path.isfile(p):
            return p
    # 多 part 目录格式（原始数据，不合并，避免 UV 重叠）
    #   objs/           — ours_*, articulateanything_*: objs/**/*.obj
    #   partglb/0.glb…  — PhysXverse partglb: 数字命名的多 GLB 文件
    #   0/0.obj…        — PhysXverse partobj: 数字命名子目录各含 obj+atlas.png
    if os.path.isdir(os.path.join(item_dir, "objs")):
        return item_dir
    # 数字命名的多 GLB（partglb 结构：0.glb, 1.glb, …）
    if g.glob(os.path.join(item_dir, "[0-9]*.glb")):
        return item_dir
    # 数字命名子目录含 OBJ（partobj 结构：0/<n>.obj, 1/<n>.obj, …）
    if g.glob(os.path.join(item_dir, "[0-9]*", "*.obj")):
        return item_dir
    return None


# 支持的 flat 文件扩展名（item 本身就是 mesh 文件）
_FLAT_EXTS = (".glb", ".gltf", ".obj")


def collect_jobs(dataset_roots: List[str], output_root: str, allowed_ids: set = None) -> dict:
    """
    遍历所有 dataset_roots，收集未完成的 mesh job，按大小分类。

    支持两种 item 结构：
      A. 目录结构 <root>/<item_id>/<mesh>  — 大多数数据集
         检测优先级：mesh.glb > texture.glb > sample.glb > merged.obj
                    > objs/ > [0-9]*.glb > [0-9]*/*.obj
      B. 平铺文件结构 <root>/<item_id>.glb  — PhysXverse/wholeglb
         item_id = 文件名去掉扩展名，mesh_path = 文件本身
    """
    jobs_by_size = {"small": [], "medium": [], "large": [], "xlarge": []}

    for dataset_root in dataset_roots:
        dataset_name = os.path.basename(dataset_root.rstrip("/"))
        out_dir = os.path.join(output_root, dataset_name)

        for item in sorted(os.listdir(dataset_root)):
            item_path = os.path.join(dataset_root, item)

            if os.path.isdir(item_path):
                # 标准目录结构
                item_id   = item
                mesh_path = _find_mesh(item_path)
            elif os.path.isfile(item_path) and item.lower().endswith(_FLAT_EXTS):
                # 平铺文件结构（如 wholeglb/<item_id>.glb）
                item_id   = os.path.splitext(item)[0]
                mesh_path = item_path
            else:
                continue

            if mesh_path is None:
                continue  # 不支持的格式，跳过
            if allowed_ids is not None and item_id not in allowed_ids:
                continue  # 不在 JSON key 白名单里，跳过

            save_dir  = os.path.join(out_dir, item)
            done_file = os.path.join(save_dir, "transforms.json")
            if os.path.exists(done_file):
                continue  # 断点续跑

            size_mb  = _get_size_mb(mesh_path)
            category = _size_category(size_mb)
            jobs_by_size[category].append((mesh_path, save_dir, size_mb, item_id))

    return jobs_by_size


def worker(worker_id: str, gpu_id: int, job_queue: Queue,
           view_mode: str, num_views: int, rotate: int,
           rotate_x: float = 0.0, resolution: int = 512,
           gt_output_root: Optional[str] = None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["BLENDER_DEVICE"] = "GPU"

    rendered = 0
    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:
                break
            mesh_path, save_dir, size_mb, item_id = job
            print(f"[{worker_id}] START {os.path.basename(save_dir)} ({size_mb:.1f}MB)")
            # random 模式下查找 GT transforms
            gt_transforms_path = None
            if view_mode == "random" and gt_output_root:
                gt_transforms_path = _find_gt_transforms(gt_output_root, item_id)
                if gt_transforms_path is None:
                    print(f"[{worker_id}] SKIP  {os.path.basename(save_dir)}: GT transforms not found for {item_id}")
                    continue
            try:
                _render_one(save_dir, mesh_path, view_mode, num_views, rotate,
                            rotate_x, resolution, gt_transforms_path)
                rendered += 1
                print(f"[{worker_id}] DONE  {os.path.basename(save_dir)} (total: {rendered})")
            except Exception as e:
                print(f"[{worker_id}] ERROR {os.path.basename(save_dir)}: {e}")
        except Exception:
            break
    print(f"[{worker_id}] Finished, rendered {rendered} meshes")


def render_on_gpu(gpu_id: int, jobs_by_size: dict, worker_config: dict,
                  view_mode: str, num_views: int, rotate: int,
                  rotate_x: float = 0.0, resolution: int = 512,
                  gt_output_root: Optional[str] = None) -> None:
    """在单个 GPU 上，按大小类别依次渲染（大 → 小）。"""
    for category in ["xlarge", "large", "medium", "small"]:
        jobs = jobs_by_size.get(category, [])
        if not jobs:
            continue
        n_workers = worker_config.get(category, 4)
        print(f"\n[GPU {gpu_id}] {category.upper()}: {len(jobs)} meshes, {n_workers} workers")

        q = Queue()
        for job in jobs:
            q.put(job)
        for _ in range(n_workers):
            q.put(None)

        procs = []
        for i in range(n_workers):
            wid = f"GPU{gpu_id}-{category[:1].upper()}{i}"
            p = Process(target=worker,
                        args=(wid, gpu_id, q, view_mode, num_views, rotate,
                              rotate_x, resolution, gt_output_root))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print(f"[GPU {gpu_id}] All done")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive multi-GPU renderer (random / orbit / fixed_grid)"
    )

    # 数据集：支持多个路径
    parser.add_argument("--dataset_roots", nargs="+", required=True,
                        help="One or more dataset root directories (each containing <item>/mesh.glb)")
    parser.add_argument("--filter_json", nargs="+", type=str, default=None,
                        help="只渲染这些 JSON 文件 key 对应的 item（可多个，取并集）")
    parser.add_argument("--output_root", type=str,
                        default="benchmark/benchmark_assets/rendered_views/description")

    # GPU：动态 N 个
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1],
                        help="GPU IDs to use (e.g. --gpus 0 1 2 3)")

    # 视角模式
    parser.add_argument("--view_mode", type=str, default="random",
                        choices=["random", "orbit", "fixed_grid"],
                        help="Camera view mode")
    parser.add_argument("--num_views", type=int, default=25,
                        help="Number of views (random/orbit modes; fixed_grid always 24)")
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--rotate_x", type=float, default=0.0,
                        help="Rotate N deg around X axis (90 = Y-up/Z-fwd -> Z-up/Y-back)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution (default 512)")
    parser.add_argument("--gt_output_root", type=str, default=None,
                        help="random 模式下 GT 输出根目录。提供后每个 item 从对应 GT "
                             "transforms.json 中提取相机位置，找不到则跳过该 item。"
                             "格式：<gt_output_root>/<item_id>/transforms.json 或 "
                             "<gt_output_root>/<item_id>.glb/transforms.json")

    # 自适应 worker 配置
    parser.add_argument("--workers_small",  type=int, default=8)
    parser.add_argument("--workers_medium", type=int, default=6)
    parser.add_argument("--workers_large",  type=int, default=4)
    parser.add_argument("--workers_xlarge", type=int, default=3)

    args = parser.parse_args()

    worker_config = {
        "small":  args.workers_small,
        "medium": args.workers_medium,
        "large":  args.workers_large,
        "xlarge": args.workers_xlarge,
    }

    print("=== 自适应并行渲染 ===")
    print(f"  view_mode      : {args.view_mode}")
    print(f"  num_views      : {args.num_views}")
    print(f"  gpus           : {args.gpus}")
    print(f"  dataset_roots  : {args.dataset_roots}")
    print(f"  gt_output_root : {args.gt_output_root}")
    print(f"  worker cfg     : {worker_config}")
    print()

    os.makedirs(args.output_root, exist_ok=True)

    # 读取 JSON 白名单
    allowed_ids = None
    if args.filter_json:
        import json as _json
        allowed_ids = set()
        for jpath in args.filter_json:
            d = _json.load(open(jpath))
            allowed_ids.update(d.keys())
        print("Filter: " + str(len(allowed_ids)) + " allowed item IDs from " + str(args.filter_json))

    # 收集所有 job
    all_jobs = collect_jobs(args.dataset_roots, args.output_root, allowed_ids=allowed_ids)
    total = sum(len(v) for v in all_jobs.values())
    print(f"Total pending jobs: {total}")
    for cat in ["xlarge", "large", "medium", "small"]:
        print(f"  {cat}: {len(all_jobs[cat])}")

    if total == 0:
        print("[INFO] Nothing to render.")
        return

    n_gpus = len(args.gpus)

    # 将 job 按类别均匀分配到各 GPU（轮询分片）
    def split_jobs_by_gpu(jobs_by_size: dict, n: int) -> List[dict]:
        shards = [{"small": [], "medium": [], "large": [], "xlarge": []} for _ in range(n)]
        for category, jobs in jobs_by_size.items():
            for idx, job in enumerate(jobs):
                shards[idx % n][category].append(job)
        return shards

    shards = split_jobs_by_gpu(all_jobs, n_gpus)

    # 每个 GPU 启动一个进程
    procs = []
    for i, gpu_id in enumerate(args.gpus):
        p = Process(target=render_on_gpu,
                    args=(gpu_id, shards[i], worker_config,
                          args.view_mode, args.num_views, args.rotate,
                          args.rotate_x, args.resolution, args.gt_output_root))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = [p for p in procs if p.exitcode != 0]
    if failed:
        print(f"[WARN] {len(failed)} GPU process(es) exited with errors.")
    else:
        print("\n[DONE] All rendering finished.")


if __name__ == "__main__":
    main()
