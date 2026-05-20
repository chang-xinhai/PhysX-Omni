"""
Multi-worker parallel renderer using nvdiffrast.

Distributes mesh rendering jobs across multiple GPU workers
using multiprocessing.Process + Queue.
"""

import argparse
import os
from multiprocessing import Process, Queue
from typing import List, Tuple


def collect_mesh_jobs(dataset_root: str, output_root: str) -> List[Tuple[str, str]]:
    """Collect (mesh_path, save_dir) pairs from dataset_root."""
    jobs: List[Tuple[str, str]] = []
    for item in sorted(os.listdir(dataset_root)):
        item_dir = os.path.join(dataset_root, item)
        if not os.path.isdir(item_dir):
            continue
        # Try common mesh filenames
        mesh_path = None
        for name in ("mesh.glb", "mesh.obj", "model.glb", "model.obj"):
            candidate = os.path.join(item_dir, name)
            if os.path.isfile(candidate):
                mesh_path = candidate
                break
        # Also accept if the item itself is a file
        if mesh_path is None:
            for ext in (".glb", ".obj"):
                candidate = os.path.join(dataset_root, item)
                if candidate.endswith(ext) and os.path.isfile(candidate):
                    mesh_path = candidate
                    break
        if mesh_path is None:
            continue
        save_dir = os.path.join(output_root, item)
        jobs.append((mesh_path, save_dir))
    return jobs


def worker(worker_id: str, gpu_id: int, job_queue: Queue,
           num_views: int, resolution: int, ssaa: int) -> None:
    """Worker process: pick jobs from queue and render."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import render inside worker so CUDA context is per-process
    from render import render_mesh

    rendered = 0
    while True:
        try:
            job = job_queue.get(timeout=2)
        except Exception:
            break
        if job is None:
            break

        mesh_path, save_dir = job
        os.makedirs(save_dir, exist_ok=True)

        # Skip if already rendered
        done_file = os.path.join(save_dir, "transforms.json")
        if os.path.exists(done_file):
            print(f"[{worker_id}] SKIP {mesh_path}")
            continue

        print(f"[{worker_id}] START {mesh_path}")
        try:
            render_mesh(mesh_path, save_dir, num_views=num_views,
                        resolution=resolution, ssaa=ssaa)
            rendered += 1
            print(f"[{worker_id}] DONE {mesh_path} (total: {rendered})")
        except Exception as e:
            print(f"[{worker_id}] ERROR {mesh_path}: {e}")

    print(f"[{worker_id}] Finished, rendered {rendered} meshes")


def render_dataset_parallel(dataset_root: str, output_root: str,
                            gpu_id: int, num_workers: int,
                            num_views: int, resolution: int, ssaa: int) -> None:
    """Launch workers for one GPU."""
    jobs = collect_mesh_jobs(dataset_root, output_root)
    os.makedirs(output_root, exist_ok=True)
    print(f"[INFO] GPU {gpu_id}: {len(jobs)} jobs, {num_workers} workers")

    job_queue = Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(num_workers):
        job_queue.put(None)

    workers = []
    for i in range(num_workers):
        wid = f"GPU{gpu_id}-W{i}"
        p = Process(target=worker,
                    args=(wid, gpu_id, job_queue, num_views, resolution, ssaa))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    print(f"[INFO] GPU {gpu_id}: All workers finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-worker nvdiffrast renderer")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory containing mesh subdirectories")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for rendered outputs")
    parser.add_argument("--gpu0", type=int, default=0,
                        help="First GPU id")
    parser.add_argument("--gpu1", type=int, default=1,
                        help="Second GPU id")
    parser.add_argument("--workers_per_gpu", type=int, default=4,
                        help="Number of parallel workers per GPU")
    parser.add_argument("--num_views", type=int, default=25,
                        help="Number of views to render per mesh")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ssaa", type=int, default=2)
    args = parser.parse_args()

    # Collect all jobs, then split evenly between two GPUs
    all_jobs = collect_mesh_jobs(args.dataset_root, args.output_root)
    mid = len(all_jobs) // 2
    jobs_gpu0 = all_jobs[:mid]
    jobs_gpu1 = all_jobs[mid:]

    def run_jobs(gpu_id, jobs, num_workers, num_views, resolution, ssaa):
        """Feed pre-split jobs into a queue and launch workers."""
        os.makedirs(args.output_root, exist_ok=True)
        print(f"[INFO] GPU {gpu_id}: {len(jobs)} jobs, {num_workers} workers")
        job_queue = Queue()
        for job in jobs:
            job_queue.put(job)
        for _ in range(num_workers):
            job_queue.put(None)
        workers = []
        for i in range(num_workers):
            wid = f"GPU{gpu_id}-W{i}"
            p = Process(target=worker,
                        args=(wid, gpu_id, job_queue, num_views, resolution, ssaa))
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
        print(f"[INFO] GPU {gpu_id}: All workers finished")

    p0 = Process(target=run_jobs,
                 args=(args.gpu0, jobs_gpu0, args.workers_per_gpu,
                       args.num_views, args.resolution, args.ssaa))
    p1 = Process(target=run_jobs,
                 args=(args.gpu1, jobs_gpu1, args.workers_per_gpu,
                       args.num_views, args.resolution, args.ssaa))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("[DONE] All rendering finished")


if __name__ == "__main__":
    main()
