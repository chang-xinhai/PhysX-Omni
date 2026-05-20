#!/usr/bin/env python3
"""Normalize existing demo PNGs into benchmark condition image layout.

The benchmark expects:

  <output-root>/<dataset>/<object_id>/first_frame.png

This helper copies or symlinks flat demo images such as
`physx_result/demo_verse/<object_id>.png` into that layout. It is especially
useful for PhysXverse, where the current local dataset snapshot has `finaljson`
and `partglb` but no URDF directory.
"""

import argparse
import shutil
from pathlib import Path


def iter_demo_pngs(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"demo image dir not found: {input_dir}")
    return sorted(p for p in input_dir.glob("*.png") if p.is_file())


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare benchmark condition images from flat demo PNGs.")
    parser.add_argument("--input-dir", required=True, help="Flat directory containing <object_id>.png images.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. mobility, verse, inthewild.")
    parser.add_argument(
        "--output-root",
        default="benchmark/benchmark_assets/condition_images",
    )
    parser.add_argument("--output-name", default="first_frame.png")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying files.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root) / args.dataset
    count = 0
    skipped = 0
    for png in iter_demo_pngs(args.input_dir):
        out_path = output_root / png.stem / args.output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and args.skip_existing:
            skipped += 1
            continue
        if out_path.exists() or out_path.is_symlink():
            out_path.unlink()
        if args.symlink:
            out_path.symlink_to(png.resolve())
        else:
            shutil.copy2(png, out_path)
        count += 1
    print(f"prepared={count} skipped={skipped} dataset={args.dataset} output_root={output_root}")


if __name__ == "__main__":
    main()
