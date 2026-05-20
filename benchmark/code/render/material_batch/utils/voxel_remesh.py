"""
Voxel-based mesh remesher.

Goal
----
Take an arbitrary input mesh (possibly self-intersecting, possibly non-manifold,
possibly with thin shells) and produce a NEW mesh that is simultaneously:

  * watertight        (closed surface, no holes)
  * manifold          (every edge shared by exactly two faces)
  * self-intersection free
  * single connected component (optional, default: keep largest component)

Method
------
Voxelize the surface, fill the interior, then extract an iso-surface via
marching cubes from the resulting occupancy / signed-distance grid.

This trades geometric fidelity (sharp features get rounded; thin shells smaller
than the voxel pitch disappear) for a guaranteed-valid PLC suitable for
constrained Delaunay tetrahedralization (TetGen / fTetWild / etc.).

Two backends:
  - trimesh : mesh.voxelized(pitch).fill().marching_cubes      (default)
  - open3d  : signed-distance field via raycasting + marching cubes

Usage
-----
  # Single file
  python voxel_remesh.py -i input.glb -o output.obj --pitch 0.005

  # Auto-pitch from bbox diagonal (default 1/256)
  python voxel_remesh.py -i input.glb -o output.obj --rel-pitch 0.004

  # Batch over a directory tree, mirror layout
  python voxel_remesh.py -i case_dir/objs -o cleaned_dir --pitch 0.005

  # Process one material-rendering case
  python voxel_remesh.py \
      -i physx_result/ours_verse_181500/<id>/objs \
      -o physx_result/ours_verse_181500/<id>/objs_clean \
      --rel-pitch 0.004

  # Batch: mirror floor/water watertight layout from raw OBJ release dirs into watertightFix
  python voxel_remesh.py --watertight-batch \
      --output-root physx_result/watertightFix_max3000 \
      --max-faces 3000

  # If logs appear stuck, use unbuffered stdout:  python -u ...  or  PYTHONUNBUFFERED=1

  # Parallel batch (default 32 workers; use --workers 1 for sequential)
  python voxel_remesh.py --watertight-batch \
      --output-root physx_result/watertightFix_max3000 \
      --max-faces 3000

  # Same with explicit dataset roots (each root is .../<dataset_name>/ with scene_id/objs/...)
  python voxel_remesh.py --sources physx_result/ours_mobility_181500 \\
      --output-root physx_result/watertightFix_max3000 \\
      --max-faces 3000
"""

import argparse
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Iterator, List, Optional, Tuple

# Heavy deps; on slow NFS / cold cache this can take minutes before any other output.
print(
    "[voxel_remesh] importing numpy/trimesh… (no output here usually means this step is slow)",
    flush=True,
)
import numpy as np
import trimesh


SUPPORTED_EXTS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}

_PHYSX_RESULT_ROOT = Path("physx_result")
_DEFAULT_MAX_FACES = 3000
_DEFAULT_SIMPLIFY_AGG = 1.0

# Same six dataset roots as used when building .../watertight (raw OBJ / mixed releases).
_DEFAULT_WATERTIGHT_SOURCES: List[Path] = [
    _PHYSX_RESULT_ROOT / "output_physxanything_inthewild",
    _PHYSX_RESULT_ROOT / "output_physxanything_mobility",
    _PHYSX_RESULT_ROOT / "output_physxanything_verse",
    _PHYSX_RESULT_ROOT / "ours_inthewild_181500",
    _PHYSX_RESULT_ROOT / "ours_mobility_181500",
    _PHYSX_RESULT_ROOT / "ours_verse_181500",
]
_DEFAULT_WATERTIGHT_OUTPUT_ROOT = _PHYSX_RESULT_ROOT / "watertightFix_max3000"
_DEFAULT_BATCH_WORKERS = 64

# Per-part file preference (same stem as part folder name), aligned with batch rglob dedupe.
_PART_MESH_PRIORITY = [".glb", ".gltf", ".obj", ".ply", ".stl", ".off"]


@dataclass
class ProcessOptions:
    """Picklable subset of CLI args for ProcessPoolExecutor workers."""

    backend: str
    pitch: float
    rel_pitch: float
    keep_largest: bool
    max_faces: int
    simplify_agg: float
    pymeshfix: bool


def process_options_from_args(ns: argparse.Namespace) -> ProcessOptions:
    return ProcessOptions(
        backend=ns.backend,
        pitch=float(ns.pitch),
        rel_pitch=float(ns.rel_pitch),
        keep_largest=bool(ns.keep_largest),
        max_faces=int(ns.max_faces),
        simplify_agg=float(ns.simplify_agg),
        pymeshfix=bool(ns.pymeshfix),
    )


def pick_mesh_in_part_dir(part_dir: Path) -> Optional[Path]:
    """`<dataset>/<scene>/objs/<part_id>/` — choose one mesh named `<part_id>.<ext>`."""
    part_id = part_dir.name
    for ext in _PART_MESH_PRIORITY:
        p = part_dir / f"{part_id}{ext}"
        if p.is_file():
            return p
    return None


def iter_watertight_style_parts(dataset_root: Path) -> Iterator[Tuple[str, str, Path]]:
    """Same traversal as watertight GLB scan, but input may be obj/glb/… per part."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(str(dataset_root))
    for scene_dir in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not scene_dir.is_dir():
            continue
        objs = scene_dir / "objs"
        if not objs.is_dir():
            continue
        for part_dir in sorted(objs.iterdir(), key=lambda p: p.name):
            if not part_dir.is_dir():
                continue
            mesh = pick_mesh_in_part_dir(part_dir)
            if mesh is not None:
                yield scene_dir.name, part_dir.name, mesh


def derive_watertight_fix_output(
    dataset_root: Path, scene_id: str, part_id: str, out_root: Path, output_ext: str
) -> Path:
    """Mirror .../watertight/<dataset>/<scene>/objs/<part>/<part>.<ext> under out_root."""
    ds_name = dataset_root.name
    ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    return out_root / ds_name / scene_id / "objs" / part_id / f"{part_id}{ext}"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def voxel_remesh_trimesh(mesh: trimesh.Trimesh, pitch: float) -> trimesh.Trimesh:
    """Surface-voxelize, fill interior, then marching-cubes back to a mesh.

    NOTE: trimesh.VoxelGrid.marching_cubes returns vertices in voxel-INDEX
    space (0..N) and does not always apply the world transform reliably across
    versions. We therefore call scikit-image's marching_cubes directly on the
    dense occupancy grid and apply the world origin + pitch ourselves so the
    output mesh stays in the SAME world coordinates as the input.
    """
    from skimage import measure

    vg = mesh.voxelized(pitch=pitch).fill()
    matrix = np.asarray(vg.encoding.dense).astype(np.float32)
    # Pad with zeros so the iso-surface closes at the boundary even when the
    # object touches the grid edges (otherwise MC produces open holes there).
    matrix = np.pad(matrix, 1, mode="constant", constant_values=0.0)

    # World origin of voxel (0,0,0) BEFORE padding. After padding, voxel index
    # (i,j,k) corresponds to world (i-1, j-1, k-1) in voxel units.
    if hasattr(vg, "origin"):
        grid_origin = np.asarray(vg.origin, dtype=np.float64)
    else:
        # Fall back to the transform's translation column.
        grid_origin = np.asarray(vg.transform)[:3, 3].astype(np.float64)

    verts, faces, _, _ = measure.marching_cubes(
        matrix, level=0.5, spacing=(pitch, pitch, pitch)
    )
    # Undo the +1 padding shift, then place into world coords.
    verts = verts - np.array([pitch, pitch, pitch]) + grid_origin
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return out


def voxel_remesh_open3d(mesh: trimesh.Trimesh, pitch: float,
                         band_voxels: int = 4) -> trimesh.Trimesh:
    """Compute a signed-distance grid via Open3D raycasting, then run marching
    cubes via trimesh's voxelgrid->mesh path. Slower but uses true SDF, so the
    iso-surface is closer to the original surface than pure occupancy.
    """
    import open3d as o3d
    from skimage import measure

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    bmin, bmax = mesh.bounds
    pad = pitch * band_voxels
    bmin = bmin - pad
    bmax = bmax + pad
    nx = int(np.ceil((bmax[0] - bmin[0]) / pitch)) + 1
    ny = int(np.ceil((bmax[1] - bmin[1]) / pitch)) + 1
    nz = int(np.ceil((bmax[2] - bmin[2]) / pitch)) + 1

    xs = bmin[0] + np.arange(nx) * pitch
    ys = bmin[1] + np.arange(ny) * pitch
    zs = bmin[2] + np.arange(nz) * pitch
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).astype(np.float32)
    pts = grid.reshape(-1, 3)

    sdf = scene.compute_signed_distance(pts).numpy().reshape(nx, ny, nz)

    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0, spacing=(pitch, pitch, pitch))
    verts = verts + bmin                       # shift back to world coords
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


# ---------------------------------------------------------------------------
# Optional decimation (quadric edge collapse) to cap output face count
# ---------------------------------------------------------------------------
def simplify_to_max_faces(
    mesh: trimesh.Trimesh,
    max_faces: int,
    simplify_agg: float = _DEFAULT_SIMPLIFY_AGG,
) -> trimesh.Trimesh:
    """Reduce face count to <= max_faces via quadric decimation.

    Tries backends in this order:
      1. fast_simplification directly. In practice this preserves watertightness
         better than trimesh's wrapper for our voxel-remeshed meshes.
      2. trimesh.Trimesh.simplify_quadric_decimation
      3. open3d's TriangleMesh.simplify_quadric_decimation directly
    """
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh

    try:
        import fast_simplification

        verts, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            target_count=int(max_faces),
            agg=float(simplify_agg),
            verbose=False,
        )
        simp = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if len(simp.faces) > 0:
            return simp
    except Exception as e:  # noqa: BLE001
        print(f"[simplify] fast_simplification failed: {e}; falling back to trimesh")

    if hasattr(mesh, "simplify_quadric_decimation"):
        try:
            simp = mesh.simplify_quadric_decimation(face_count=int(max_faces))
            if simp is not None and len(simp.faces) > 0:
                return simp
        except Exception as e:
            print(f"[simplify] trimesh backend failed: {e}; falling back to open3d")

    try:
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        simp_o3d = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=int(max_faces))
        simp_o3d.remove_unreferenced_vertices()
        simp_o3d.remove_degenerate_triangles()
        return trimesh.Trimesh(
            vertices=np.asarray(simp_o3d.vertices),
            faces=np.asarray(simp_o3d.triangles),
            process=True,
        )
    except ImportError:
        print("[simplify] open3d not installed; skipping decimation.")
        return mesh
    except Exception as e:
        print(f"[simplify] open3d backend failed: {e}; returning unsimplified mesh.")
        return mesh


def repair_pymeshfix(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Run pymeshfix to fill holes and remove self-intersections.

    pymeshfix.MeshFix detects self-intersecting/non-manifold patches, removes
    them, then re-fills resulting holes; output is designed to be watertight,
    manifold, and self-intersection free (single connected component).
    """
    try:
        import pymeshfix
    except ImportError:
        print("[pymeshfix] not installed; skip repair. (pip install pymeshfix)")
        return mesh
    try:
        fx = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        try:
            fx.repair(verbose=False)
        except TypeError:
            fx.repair()
        # API names differ across pymeshfix versions:
        #   newer (0.18+): .points, .faces
        #   older       : .v, .f
        verts = getattr(fx, "points", None)
        if verts is None:
            verts = getattr(fx, "v")
        faces = getattr(fx, "faces", None)
        if faces is None:
            faces = getattr(fx, "f")
        out = trimesh.Trimesh(vertices=np.asarray(verts),
                              faces=np.asarray(faces), process=True)
        trimesh.repair.fix_normals(out)
        return out
    except Exception as e:
        print(f"[pymeshfix] repair failed: {e}; returning input unchanged.")
        return mesh


# ---------------------------------------------------------------------------
# Cleanup / verification
# ---------------------------------------------------------------------------
def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh
    comps = sorted(comps, key=lambda m: m.volume if m.is_volume else m.area, reverse=True)
    return comps[0]


def post_clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Use modern trimesh API (older `remove_duplicate_faces` / `remove_degenerate_faces`
    # were removed). `process(validate=True)` does merge_vertices, dedupe, drops
    # degenerate triangles, and fixes winding where possible.
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def verify(mesh: trimesh.Trimesh, tag: str = "") -> dict:
    bmin, bmax = mesh.bounds
    info = {
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "volume": float(mesh.volume) if mesh.is_volume else None,
    }
    print(f"[verify{(' '+tag) if tag else ''}] "
          f"V={info['n_vertices']} F={info['n_faces']} "
          f"watertight={info['is_watertight']} winding_ok={info['is_winding_consistent']} "
          f"euler={info['euler_number']} volume={info['volume']}")
    print(f"           bbox min=({bmin[0]:.4f},{bmin[1]:.4f},{bmin[2]:.4f}) "
          f"max=({bmax[0]:.4f},{bmax[1]:.4f},{bmax[2]:.4f})")
    return info


# ---------------------------------------------------------------------------
# Pitch helpers
# ---------------------------------------------------------------------------
def auto_pitch(mesh: trimesh.Trimesh, rel_pitch: float) -> float:
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    return diag * rel_pitch


def run_remesh_backend(mesh: trimesh.Trimesh, backend: str, pitch: float) -> trimesh.Trimesh:
    if backend == "trimesh":
        return voxel_remesh_trimesh(mesh, pitch)
    if backend == "open3d":
        return voxel_remesh_open3d(mesh, pitch)
    raise ValueError(f"unknown backend: {backend}")


def prepare_remesh_candidate(
    mesh: trimesh.Trimesh,
    opts: ProcessOptions,
    pitch: float,
) -> trimesh.Trimesh:
    remeshed = run_remesh_backend(mesh, opts.backend, pitch)
    if opts.keep_largest:
        remeshed = keep_largest_component(remeshed)
    return post_clean(remeshed)


def enforce_face_budget(
    source_mesh: trimesh.Trimesh,
    remeshed: trimesh.Trimesh,
    initial_pitch: float,
    opts: ProcessOptions,
) -> trimesh.Trimesh:
    """Keep FEM proxy meshes reasonably small without sacrificing watertightness.

    Low-aggressiveness quadric simplification preserves shape but sometimes
    refuses to reach a strict face budget. High-aggressiveness simplification
    can hit the budget but often breaks watertightness. For FEM/TetGen, a
    coarser voxel remesh is usually the safer fallback: it is smoother, but it
    remains closed and avoids huge tetrahedralizations.
    """
    if opts.max_faces <= 0 or len(remeshed.faces) <= opts.max_faces:
        return remeshed

    before = len(remeshed.faces)
    was_watertight = bool(remeshed.is_watertight)
    simplified = simplify_to_max_faces(remeshed, opts.max_faces, opts.simplify_agg)
    simplified = post_clean(simplified)
    print(
        f"[simplify] {before} -> {len(simplified.faces)} faces "
        f"(target<={opts.max_faces}, agg={opts.simplify_agg:g}, "
        f"watertight={simplified.is_watertight})"
    )
    if len(simplified.faces) <= opts.max_faces and (
        not was_watertight or simplified.is_watertight
    ):
        return simplified

    reason = "did not reach face budget"
    if was_watertight and not simplified.is_watertight:
        reason = "broke watertightness"
    print(f"[budget] simplification {reason}; trying adaptive coarser voxel remesh")

    best = remeshed
    pitch = initial_pitch
    for attempt in range(1, 9):
        current_faces = max(len(best.faces), len(remeshed.faces), 1)
        ratio = float(np.sqrt(current_faces / max(opts.max_faces, 1)))
        pitch *= max(1.20, min(2.00, ratio * 1.08))
        cand = prepare_remesh_candidate(source_mesh, opts, pitch)
        print(
            f"[budget] attempt={attempt} pitch={pitch:.6g} "
            f"faces={len(cand.faces)} watertight={cand.is_watertight}"
        )
        if cand.is_watertight:
            if (not best.is_watertight) or len(cand.faces) < len(best.faces):
                best = cand
            if len(cand.faces) <= opts.max_faces:
                return cand

    if len(simplified.faces) < len(best.faces) and (
        not was_watertight or simplified.is_watertight
    ):
        return simplified
    print(
        f"[warn] face budget not fully met; using best watertight candidate "
        f"faces={len(best.faces)} target={opts.max_faces}"
    )
    return best


# ---------------------------------------------------------------------------
# Process one file
# ---------------------------------------------------------------------------
def process_one(in_path: Path, out_path: Path, opts: ProcessOptions) -> None:
    print(f"\n=== {in_path} -> {out_path} ===")
    mesh = trimesh.load(str(in_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        print(f"[skip] could not load a triangle mesh from {in_path}")
        return
    verify(mesh, "input")

    pitch = opts.pitch if opts.pitch > 0 else auto_pitch(mesh, opts.rel_pitch)
    print(f"[remesh] backend={opts.backend}  pitch={pitch:.6g}  "
          f"(bbox diag={float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])):.4f})")

    remeshed = prepare_remesh_candidate(mesh, opts, pitch)
    remeshed = enforce_face_budget(mesh, remeshed, pitch, opts)
    if opts.pymeshfix:
        before = (len(remeshed.vertices), len(remeshed.faces))
        remeshed = repair_pymeshfix(remeshed)
        remeshed = post_clean(remeshed)
        print(f"[pymeshfix] V/F {before[0]}/{before[1]} -> "
              f"{len(remeshed.vertices)}/{len(remeshed.faces)}")
    verify(remeshed, "output")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    remeshed.export(str(out_path))
    print(f"[save] {out_path}")


def _process_one_worker(payload: Tuple[str, str, ProcessOptions]) -> Tuple[str, Optional[str]]:
    """Top-level entry for ProcessPoolExecutor (must be picklable)."""
    in_s, out_s, opts = payload
    try:
        process_one(Path(in_s), Path(out_s), opts)
        return (in_s, None)
    except Exception as e:  # noqa: BLE001
        return (in_s, f"{type(e).__name__}: {e}")


def run_batch_jobs(
    jobs: List[Tuple[Path, Path]],
    opts: ProcessOptions,
    workers: int,
) -> Tuple[int, List[Tuple[str, str]]]:
    """Run process_one for each (input, output) pair; returns (n_ok, [(path, err), ...])."""
    if not jobs:
        return 0, []
    if workers <= 1:
        n_ok = 0
        errors: List[Tuple[str, str]] = []
        for inp, outp in jobs:
            try:
                process_one(inp, outp, opts)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                errors.append((str(inp), f"{type(e).__name__}: {e}"))
        return n_ok, errors
    payloads = [(str(inp), str(outp), opts) for inp, outp in jobs]
    errors: List[Tuple[str, str]] = []
    n_ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_one_worker, p) for p in payloads]
        for fut in as_completed(futs):
            in_s, err = fut.result()
            if err:
                errors.append((in_s, err))
            else:
                n_ok += 1
    return n_ok, errors


def run_watertight_batch_streaming(
    source_list: List[Path],
    out_root: Path,
    opts: ProcessOptions,
    output_ext: str,
    skip_existing: bool,
    workers: int,
    progress_every: int,
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """Enumerate datasets in a background thread; process jobs as soon as they appear.

    Returns (n_ok, total_jobs_submitted, errors).
    """
    job_q: Queue[Optional[Tuple[str, str]]] = Queue(maxsize=max(512, workers * 64))

    def fill_queue() -> None:
        try:
            for ds_root in source_list:
                if not ds_root.is_dir():
                    print(f"[warn] skip missing dataset root: {ds_root}", file=sys.stderr)
                    continue
                print(
                    f"[dataset] scanning {ds_root} … "
                    f"(streaming jobs; workers may already be running)",
                    flush=True,
                )
                try:
                    for scene_id, part_id, in_mesh in iter_watertight_style_parts(ds_root):
                        out_f = derive_watertight_fix_output(
                            ds_root, scene_id, part_id, out_root, output_ext
                        )
                        if skip_existing and out_f.is_file():
                            continue
                        job_q.put((str(in_mesh), str(out_f)))
                except FileNotFoundError as e:
                    print(f"[warn] {e}", file=sys.stderr)
                print(
                    f"[dataset] {ds_root.name}  enumeration done for this root",
                    flush=True,
                )
        finally:
            job_q.put(None)

    enum_thread = threading.Thread(target=fill_queue, daemon=True, name="watertight-enumerator")
    enum_thread.start()

    errors: List[Tuple[str, str]] = []
    n_ok = 0
    total_jobs = 0
    start_time = time.time()
    last_progress_done = 0

    def report_progress(pending_len: int = 0, force: bool = False) -> None:
        nonlocal last_progress_done
        if progress_every <= 0:
            return
        done_count = n_ok + len(errors)
        if not force and done_count - last_progress_done < progress_every:
            return
        elapsed = max(time.time() - start_time, 1e-6)
        rate = done_count / elapsed
        print(
            f"[progress] done={done_count} submitted={total_jobs} pending={pending_len} "
            f"ok={n_ok} failed={len(errors)} rate={rate:.2f}/s elapsed={elapsed/60:.1f}min",
            flush=True,
        )
        last_progress_done = done_count

    if workers <= 1:
        while True:
            item = job_q.get()
            if item is None:
                break
            total_jobs += 1
            in_s, out_s = item
            try:
                process_one(Path(in_s), Path(out_s), opts)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                errors.append((in_s, f"{type(e).__name__}: {e}"))
            report_progress()
        enum_thread.join(timeout=3600.0)
        report_progress(force=True)
        return n_ok, total_jobs, errors

    max_pending = max(workers * 8, 64)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pending = set()
        while True:
            item = job_q.get()
            if item is None:
                break
            total_jobs += 1
            in_s, out_s = item
            fut = ex.submit(_process_one_worker, (in_s, out_s, opts))
            pending.add(fut)
            while len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    in_r, err = f.result()
                    if err:
                        errors.append((in_r, err))
                    else:
                        n_ok += 1
                report_progress(pending_len=len(pending))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                in_r, err = f.result()
                if err:
                    errors.append((in_r, err))
                else:
                    n_ok += 1
            report_progress(pending_len=len(pending))

    enum_thread.join(timeout=3600.0)
    report_progress(force=True)
    return n_ok, total_jobs, errors


def derive_output_path(in_path: Path, in_root: Path, out_root: Path,
                       output_ext: str) -> Path:
    rel = in_path.relative_to(in_root)
    out = out_root / rel
    if output_ext:
        out = out.with_suffix(output_ext)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, default=None,
                    help="Input mesh file or directory (not used with --watertight-batch / --sources).")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output mesh file or directory (not used with watertight batch mode).")
    ap.add_argument(
        "--watertight-batch",
        action="store_true",
        help="Process default raw dataset roots (ours_* + output_physxanything_*) into "
             "watertightFix-style tree; use --output-root (default watertightFix_max3000).",
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=None,
        metavar="DIR",
        help="Dataset root(s): each contains <scene_id>/objs/<part_id>/<part>.(obj|glb|…). "
             "Implies batch layout; overrides --watertight-batch default list.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_WATERTIGHT_OUTPUT_ROOT,
        help="Watertight-batch output root (default: watertightFix_max3000 under physx_result).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="In watertight batch mode, skip if destination mesh file already exists.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_BATCH_WORKERS,
        help="Parallel worker processes for batch modes (--watertight-batch, --sources, "
             "or directory -i). "
             f"Default {_DEFAULT_BATCH_WORKERS}. Use --workers 1 for sequential. "
             "Ignored when -i is a single file.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a streaming progress line after this many finished meshes in watertight batch mode. 0 disables it.",
    )
    ap.add_argument("--backend", choices=["trimesh", "open3d"], default="trimesh")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="Absolute voxel size (world units). 0 => use --rel-pitch.")
    ap.add_argument("--rel-pitch", type=float, default=1.0 / 256.0,
                    help="Relative voxel size = rel_pitch * bbox_diagonal. "
                         "Default 1/256 ~= moderate detail.")
    ap.add_argument("--output-ext", type=str, default=".glb",
                    help="Output extension when batch mode is used (.glb/.obj/.ply/.stl).")
    ap.add_argument("--keep-largest", action="store_true", default=True,
                    help="Keep only the largest connected component (default: on).")
    ap.add_argument("--no-keep-largest", dest="keep_largest", action="store_false")
    ap.add_argument("--max-faces", type=int, default=_DEFAULT_MAX_FACES,
                    help="Cap output face count via quadric decimation. 0 = no cap.")
    ap.add_argument("--simplify-agg", type=float, default=_DEFAULT_SIMPLIFY_AGG,
                    help="fast_simplification aggressiveness for --max-faces. "
                         "Lower preserves geometry more but can be slower. Default 1.0.")
    ap.add_argument("--pymeshfix", action="store_true", default=False,
                    help="After (optional) decimation, run pymeshfix.MeshFix to fill "
                         "small holes and remove residual self-intersections.")
    args = ap.parse_args()

    batch_mode = bool(args.watertight_batch or args.sources)
    if batch_mode:
        print(
            "[watertight-batch] resolving source paths (large NFS trees can take a bit)…",
            flush=True,
        )
        source_list = (
            [p.expanduser().resolve() for p in args.sources]
            if args.sources
            else [p.resolve() for p in _DEFAULT_WATERTIGHT_SOURCES]
        )
        out_root = args.output_root.expanduser().resolve()
        opts = process_options_from_args(args)
        print(
            f"[watertight-batch] output root: {out_root}  workers={args.workers}  "
            f"max_faces={opts.max_faces} simplify_agg={opts.simplify_agg:g} rel_pitch={opts.rel_pitch:g} "
            f"(enumerate runs in background; processing overlaps with scanning next root)",
            flush=True,
        )
        n_ok, total_jobs, errors = run_watertight_batch_streaming(
            source_list,
            out_root,
            opts,
            args.output_ext,
            args.skip_existing,
            args.workers,
            args.progress_every,
        )
        for path, msg in errors:
            print(f"[error] {path}: {msg}", file=sys.stderr)
        print(
            f"\n[watertight-batch done] ok={n_ok}  failed={len(errors)}  "
            f"total_jobs={total_jobs}  root={out_root}",
            flush=True,
        )
        return

    if args.input is None or args.output is None:
        ap.error("either use -i/-o, or --watertight-batch / --sources ... with --output-root")

    in_path = args.input.expanduser().resolve()
    out_path = args.output.expanduser().resolve()

    if not in_path.exists():
        print(f"[fatal] input does not exist: {in_path}", file=sys.stderr)
        sys.exit(1)

    opts = process_options_from_args(args)

    if in_path.is_file():
        process_one(in_path, out_path, opts)
        return

    # Directory mode: walk recursively, mirror layout to out_path/
    files = [p for p in in_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    # Deduplicate: when multiple meshes share the same parent + stem
    # (e.g. <dir>/0.glb and <dir>/0.obj are clearly the same part), keep
    # only one in this preference order: glb > gltf > obj > ply > stl > off.
    ext_priority = {".glb": 0, ".gltf": 1, ".obj": 2, ".ply": 3, ".stl": 4, ".off": 5}
    best = {}
    for f in files:
        key = (f.parent, f.stem)
        rank = ext_priority.get(f.suffix.lower(), 99)
        if key not in best or rank < ext_priority.get(best[key].suffix.lower(), 99):
            best[key] = f
    files = sorted(best.values())
    if not files:
        print(f"[fatal] no supported meshes found under {in_path}", file=sys.stderr)
        sys.exit(2)
    print(f"[batch] found {len(files)} mesh files under {in_path}  workers={args.workers}")
    jobs = [(f, derive_output_path(f, in_path, out_path, args.output_ext)) for f in files]
    n_ok, errors = run_batch_jobs(jobs, opts, args.workers)
    for path, msg in errors:
        print(f"[error] {path}: {msg}", file=sys.stderr)
    print(f"\n[done] ok={n_ok}  failed={len(errors)}")


if __name__ == "__main__":
    main()
