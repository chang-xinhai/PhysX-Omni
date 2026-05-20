#!/usr/bin/env python3
"""Build Quality / Consistency manifests from rendered multi-view PNG folders."""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ROOT = "benchmark/benchmark_assets/rendered_views/description"
DEFAULT_EXPECTED_ROOT_BASE = "physx_result"


DEFAULT_EXPECTED_ROOTS = {
    "ours_mobility_181500": "ours_mobility_181500",
    "output_physxanything_mobility": "output_physxanything_mobility",
    "outputs_physxgen_mobility": "outputs_physxgen_mobility",
    "output_articulateanything_mobility": "output_articulateanything_mobility",
    "ours_verse_181500": "ours_verse_181500",
    "output_physxanything_verse": "output_physxanything_verse",
    "outputs_physxgen_verse": "outputs_physxgen_verse",
    "output_articulateanything_verse": "output_articulateanything_verse",
    "ours_inthewild_181500": "ours_inthewild_181500",
    "output_physxanything_inthewild": "output_physxanything_inthewild",
    "outputs_physxgen_inthewild": "outputs_physxgen_inthewild",
    "output_articulateanything_inthewild": "output_articulateanything_inthewild",
}


METRIC_CONFIG = {
    "rqs": {"metric": "rqs", "num_views": 4},
    "quality": {"metric": "rqs", "num_views": 4},
    "mcs": {"metric": "mcs", "num_views": 8},
    "consistency": {"metric": "mcs", "num_views": 8},
}


def numeric_path_key(path):
    path = Path(path)
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def select_render_view_paths(paths, num_paths):
    paths = sorted([Path(p) for p in paths], key=numeric_path_key)
    if len(paths) <= num_paths:
        return paths
    if num_paths == 1:
        return [paths[0]]
    indices = [round(i * (len(paths) - 1) / (num_paths - 1)) for i in range(num_paths)]
    return [paths[i] for i in indices]


def is_object_dir(path):
    name = Path(path).name
    return not name.startswith(".") and not name.endswith("_mask")


def parse_source(value):
    parts = str(value).split(":")
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--source must have format <folder_name>:<method>:<dataset>"
        )
    folder_name, method, dataset = [part.strip() for part in parts]
    return folder_name, method, dataset


def expected_root_for_source(folder_name):
    rel = DEFAULT_EXPECTED_ROOTS.get(folder_name)
    if rel is None:
        return None
    path = Path(DEFAULT_EXPECTED_ROOT_BASE) / rel
    return path if path.is_dir() else None


def build_rows(root, sources, metric_name, limit=None, skip_incomplete=False):
    cfg = METRIC_CONFIG[metric_name]
    rows = []
    missing_rows = []
    root = Path(root)
    for folder_name, method, dataset in sources:
        source_root = root / folder_name
        if not source_root.is_dir():
            raise NotADirectoryError(f"source folder not found: {source_root}")
        render_dirs = {p.name: p for p in source_root.iterdir() if p.is_dir() and is_object_dir(p)}
        expected_root = expected_root_for_source(folder_name)
        if expected_root is not None:
            expected_dirs = {p.name: p for p in expected_root.iterdir() if p.is_dir() and is_object_dir(p)}
            object_ids = sorted(expected_dirs.keys(), key=lambda x: numeric_path_key(Path(x)))
        else:
            expected_dirs = {}
            object_ids = sorted(render_dirs.keys(), key=lambda x: numeric_path_key(Path(x)))
        if limit is not None:
            object_ids = object_ids[:limit]
        for object_id in object_ids:
            render_dir = render_dirs.get(object_id)
            expected_dir = expected_dirs.get(object_id)
            sample_dir = render_dir or expected_dir or (source_root / object_id)
            pngs = sorted([p for p in render_dir.glob("*.png") if p.is_file()], key=numeric_path_key) if render_dir else []
            selected = select_render_view_paths(pngs, cfg["num_views"])
            if not pngs:
                status = "missing_render_views"
                ready = False
                render_missing_score_zero = True
            elif len(pngs) < cfg["num_views"]:
                status = "partial_render_views"
                ready = True
                render_missing_score_zero = False
            else:
                status = "ready"
                ready = True
                render_missing_score_zero = False

            if render_missing_score_zero:
                missing_rows.append(
                    {
                        "metric": cfg["metric"],
                        "method": method,
                        "dataset": dataset,
                        "object_id": object_id,
                        "sample_id": object_id,
                        "relative_dir": f"{cfg['metric']}/{method}/{dataset}/{object_id}",
                        "source_folder": folder_name,
                        "source_result_dir": str(sample_dir.resolve()),
                        "num_render_views_available": len(pngs),
                        "num_render_views_required": cfg["num_views"],
                        "num_render_views_selected": 0,
                        "view_image_paths": [],
                        "ready": ready,
                        "status": status,
                        "render_missing_score_zero": True,
                    }
                )

            rows.append(
                {
                    "metric": cfg["metric"],
                    "method": method,
                    "dataset": dataset,
                    "object_id": object_id,
                    "sample_id": object_id,
                    "relative_dir": f"{cfg['metric']}/{method}/{dataset}/{object_id}",
                    "source_folder": folder_name,
                    "source_result_dir": str(sample_dir.resolve()),
                    "num_render_views_available": len(pngs),
                    "num_render_views_required": cfg["num_views"],
                    "num_render_views_selected": len(selected),
                    "view_image_paths": [str(p.resolve()) for p in selected],
                    "ready": ready,
                    "status": status,
                    "render_missing_score_zero": render_missing_score_zero,
                }
            )
    return rows, missing_rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "method",
        "dataset",
        "object_id",
        "sample_id",
        "relative_dir",
        "source_folder",
        "source_result_dir",
        "num_render_views_available",
        "num_render_views_required",
        "num_render_views_selected",
        "view_image_paths",
        "ready",
        "status",
        "render_missing_score_zero",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["view_image_paths"] = json.dumps(out["view_image_paths"], ensure_ascii=False)
            writer.writerow(out)


def write_missing_jsonl(rows, path):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build RQS/MCS render-view manifest rows.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--metric", choices=sorted(METRIC_CONFIG), required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        help="Source mapping: <folder_name>:<method>:<dataset>",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--missing-jsonl", default=None)
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Deprecated no-op; missing render rows are kept and marked for zero scoring.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-source sample limit for smoke tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows, missing_rows = build_rows(
        args.root,
        args.source,
        args.metric,
        limit=args.limit,
        skip_incomplete=args.skip_incomplete,
    )
    write_jsonl(rows, args.output_jsonl)
    write_csv(rows, args.output_csv)
    write_missing_jsonl(missing_rows, args.missing_jsonl)
    print(f"metric={args.metric} rows={len(rows)} missing={len(missing_rows)}")
    print(f"jsonl={args.output_jsonl}")
    print(f"csv={args.output_csv}")
    if args.missing_jsonl:
        print(f"missing_jsonl={args.missing_jsonl}")


if __name__ == "__main__":
    main()
