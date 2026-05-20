#!/usr/bin/env python3
"""Prepare colored affordance heatmap assets from grayscale affordance renders.

The PhysX result folders store per-view affordance maps as grayscale PNGs under:

  <result_root>/<object_id>/affordance/*.png

The old Render_physx scripts define a `draw_heatmap` helper that maps scalar
values through a jet colormap with a black zero background, but their batch loops
write grayscale images. This script applies the same benchmark-facing idea and
normalizes assets into:

  <output_root>/<method>/<dataset>/<object_id>/
    affordance_heatmap_views/*.png
    affordance_heatmap_grid.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


SOURCES = {
    ("ours", "mobility"): "ours_mobility_181500",
    ("ours", "verse"): "ours_verse_181500",
    ("physxanything", "mobility"): "output_physxanything_mobility",
    ("physxanything", "verse"): "output_physxanything_verse",
    ("physxgen", "mobility"): "outputs_physxgen_mobility",
    ("physxgen", "verse"): "outputs_physxgen_verse",
    ("physxanything", "inthewild"): "output_physxanything_inthewild",
    ("physxgen", "inthewild"): "outputs_physxgen_inthewild",
    ("ours", "inthewild"): "ours_inthewild_181500",
}


METHOD_ALIASES = {
    "physanything": "physxanything",
    "physgen": "physxgen",
}


def numeric_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def jet_with_black_zero(gray: np.ndarray) -> np.ndarray:
    """Map [0, 255] grayscale to RGB jet colors, with exact zero as black."""
    x = gray.astype(np.float32)
    if x.max(initial=0) > 1.0:
        x /= 255.0
    x = np.clip(x, 0.0, 1.0)

    # Matplotlib-like jet approximation: blue -> cyan -> yellow -> red.
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[x <= 0.0] = 0.0
    return np.round(rgb * 255.0).astype(np.uint8)


def read_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img)


def write_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def make_grid(image_paths, out_path: Path, columns: int, tile_size: int):
    if not image_paths:
        return
    tiles = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        if tile_size > 0:
            img = img.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        tiles.append(img)

    columns = max(1, columns)
    rows = math.ceil(len(tiles) / columns)
    width, height = tiles[0].size
    grid = Image.new("RGB", (columns * width, rows * height), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * width
        y = (idx // columns) * height
        grid.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def iter_affordance_dirs(result_root: Path):
    if not result_root.is_dir():
        return []
    return sorted(p for p in result_root.glob("*/affordance") if p.is_dir())


def prepare_one(aff_dir: Path, out_dir: Path, grid_num_images: int, grid_columns: int, tile_size: int, skip_existing: bool):
    source_images = sorted([p for p in aff_dir.glob("*.png") if p.is_file()], key=numeric_key)
    if not source_images:
        return {"status": "missing_source_images", "views": 0}

    views_dir = out_dir / "affordance_heatmap_views"
    colored_paths = []
    for src in source_images:
        dst = views_dir / src.name
        if skip_existing and dst.is_file():
            colored_paths.append(dst)
            continue
        gray = read_gray(src)
        rgb = jet_with_black_zero(gray)
        write_rgb(dst, rgb)
        colored_paths.append(dst)

    if grid_num_images < 0:
        grid_paths = colored_paths
    else:
        grid_paths = colored_paths[:grid_num_images]
    grid_path = out_dir / "affordance_heatmap_grid.png"
    if grid_paths and not (skip_existing and grid_path.is_file()):
        make_grid(grid_paths, grid_path, columns=grid_columns, tile_size=tile_size)

    return {"status": "ready", "views": len(colored_paths)}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert grayscale affordance maps to colored heatmap assets.")
    parser.add_argument("--physx-result-root", default="physx_result")
    parser.add_argument(
        "--output-root",
        default="benchmark/benchmark_assets/affordance_heatmaps",
    )
    parser.add_argument("--methods", nargs="+", default=["ours", "physxanything", "physxgen"])
    parser.add_argument("--datasets", nargs="+", default=["mobility", "verse"])
    parser.add_argument("--grid-num-images", type=int, default=8, help="Use first N views in grid; negative means all.")
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256, help="Grid tile size; <=0 keeps original view size.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    physx_root = Path(args.physx_result_root)
    output_root = Path(args.output_root)

    total = 0
    ready = 0
    missing = 0
    methods = [METHOD_ALIASES.get(method, method) for method in args.methods]
    for method in methods:
        for dataset in args.datasets:
            rel = SOURCES.get((method, dataset))
            if rel is None:
                print(f"[skip] unknown source method={method} dataset={dataset}", flush=True)
                continue
            result_root = physx_root / rel
            aff_dirs = iter_affordance_dirs(result_root)
            print(f"[source] method={method} dataset={dataset} dirs={len(aff_dirs)} root={result_root}", flush=True)
            for aff_dir in aff_dirs:
                object_id = aff_dir.parent.name
                out_dir = output_root / method / dataset / object_id
                stat = prepare_one(
                    aff_dir=aff_dir,
                    out_dir=out_dir,
                    grid_num_images=args.grid_num_images,
                    grid_columns=args.grid_columns,
                    tile_size=args.tile_size,
                    skip_existing=args.skip_existing,
                )
                total += 1
                if stat["status"] == "ready":
                    ready += 1
                else:
                    missing += 1
                    print(f"[warn] {method}/{dataset}/{object_id}: {stat['status']}", flush=True)
            print(f"[done] method={method} dataset={dataset}", flush=True)

    print(f"total={total} ready={ready} missing={missing} output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
