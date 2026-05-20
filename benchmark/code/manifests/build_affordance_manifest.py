#!/usr/bin/env python3
"""Build APS benchmark manifest from prepared affordance heatmap assets."""

import argparse
import csv
import json
from pathlib import Path


DATASETS = {
    "mobility": {
        "ours": "ours_mobility_181500",
        "physxanything": "output_physxanything_mobility",
        "physxgen": "outputs_physxgen_mobility",
    },
    "verse": {
        "ours": "ours_verse_181500",
        "physxanything": "output_physxanything_verse",
        "physxgen": "outputs_physxgen_verse",
    },
    "inthewild": {
        "physxanything": "output_physxanything_inthewild",
        "physxgen": "outputs_physxgen_inthewild",
        "ours": "ours_inthewild_181500",
    },
}


METHOD_ALIASES = {
    "physanything": "physxanything",
    "physgen": "physxgen",
}


AFFORDANCE_ZERO_STATUS_TOKENS = {
    "missing_affordance_heatmap_views",
    "insufficient_affordance_heatmap_views",
}


def sample_dirs(root: Path):
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def numeric_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def condition_image_path(condition_image_root, dataset, object_id):
    return Path(condition_image_root) / dataset / str(object_id) / "first_frame.png"


def heatmap_dir(heatmap_root, method, dataset, object_id):
    root = Path(heatmap_root)
    canonical = root / method / dataset / str(object_id)
    if canonical.exists():
        return canonical
    if method == "physxanything":
        legacy = root / "physanything" / dataset / str(object_id)
        if legacy.exists():
            return legacy
    return canonical


def select_evenly_spaced_paths(paths, num_paths):
    paths = sorted([Path(p) for p in paths], key=numeric_key)
    if num_paths <= 0 or len(paths) <= num_paths:
        return paths, list(range(len(paths)))
    if num_paths == 1:
        return [paths[0]], [0]
    indices = [round(i * (len(paths) - 1) / (num_paths - 1)) for i in range(num_paths)]
    return [paths[i] for i in indices], indices


def build_row(condition_image_root, heatmap_root, method, dataset, sample_dir, num_views):
    object_id = sample_dir.name
    image_path = condition_image_path(condition_image_root, dataset, object_id)
    out_dir = heatmap_dir(heatmap_root, method, dataset, object_id)
    views_dir = out_dir / "affordance_heatmap_views"
    grid_path = out_dir / "affordance_heatmap_grid.png"

    status = []
    if not image_path.is_file():
        status.append("missing_condition_image")
    all_view_paths = sorted([p for p in views_dir.glob("*.png") if p.is_file()], key=numeric_key)
    if num_views > 0 and len(all_view_paths) < num_views:
        status.append("insufficient_affordance_heatmap_views")
    view_paths, sampled_indices = select_evenly_spaced_paths(all_view_paths, num_views)
    if not view_paths:
        status.append("missing_affordance_heatmap_views")
    if not grid_path.is_file():
        status.append("missing_affordance_heatmap_grid")
    missing_affordance_reasons = [x for x in status if x in AFFORDANCE_ZERO_STATUS_TOKENS]

    row = {
        "metric": "aps",
        "method": method,
        "dataset": dataset,
        "object_id": object_id,
        "sample_id": object_id,
        "image_path": str(image_path),
        "condition_image": str(image_path),
        "affordance_view_paths": [str(p) for p in view_paths],
        "affordance_heatmap_grid": str(grid_path) if grid_path.is_file() else "",
        "relative_dir": f"aps/{method}/{dataset}/{object_id}",
        "source_result_dir": str(sample_dir),
        "source_affordance_dir": str(sample_dir / "affordance"),
        "num_affordance_views": len(view_paths),
        "num_affordance_views_available": len(all_view_paths),
        "affordance_view_sample_indices": sampled_indices,
        "affordance_view_sampling": "uniform_numeric",
        "aps_missing_score_zero": bool(missing_affordance_reasons),
        "missing_affordance_reason": ";".join(missing_affordance_reasons),
        "ready": not status,
        "status": "ready" if not status else ";".join(status),
    }
    return row


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
        "image_path",
        "affordance_heatmap_grid",
        "affordance_view_paths",
        "relative_dir",
        "source_result_dir",
        "source_affordance_dir",
        "num_affordance_views",
        "num_affordance_views_available",
        "affordance_view_sample_indices",
        "affordance_view_sampling",
        "aps_missing_score_zero",
        "missing_affordance_reason",
        "ready",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["affordance_view_paths"] = json.dumps(out.get("affordance_view_paths", []), ensure_ascii=False)
            out["affordance_view_sample_indices"] = json.dumps(
                out.get("affordance_view_sample_indices", []), ensure_ascii=False
            )
            writer.writerow({k: out.get(k, "") for k in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Build APS pairs manifest for prepared affordance heatmaps.")
    parser.add_argument("--physx-result-root", default="physx_result")
    parser.add_argument(
        "--condition-image-root",
        default="benchmark/benchmark_assets/condition_images",
    )
    parser.add_argument(
        "--heatmap-root",
        default="benchmark/benchmark_assets/affordance_heatmaps",
    )
    parser.add_argument(
        "--output-jsonl",
        default="benchmark/benchmark_manifests/affordance_pairs.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark/benchmark_manifests/affordance_pairs.csv",
    )
    parser.add_argument("--num-views", type=int, default=8, help="Number of colored heatmap views to include after uniform numeric sampling; <=0 uses all.")
    parser.add_argument("--methods", nargs="+", default=["ours", "physxanything", "physxgen"], help="Generation methods to include.")
    parser.add_argument("--datasets", nargs="+", default=["mobility", "verse"], help="Datasets to include.")
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument(
        "--missing-affordance-only",
        action="store_true",
        help=(
            "Emit only APS rows whose generated affordance evidence is missing/insufficient. "
            "multi.py will auto-score these rows as APS=0 without running the VLM."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    physx_root = Path(args.physx_result_root)
    selected_methods = {METHOD_ALIASES.get(method, method) for method in args.methods}
    selected_datasets = set(args.datasets)
    for dataset, methods in DATASETS.items():
        if dataset not in selected_datasets:
            continue
        for method, rel_dir in methods.items():
            if method not in selected_methods:
                continue
            result_root = physx_root / rel_dir
            for sample_dir in sample_dirs(result_root):
                rows.append(
                    build_row(
                        condition_image_root=args.condition_image_root,
                        heatmap_root=args.heatmap_root,
                        method=method,
                        dataset=dataset,
                        sample_dir=sample_dir,
                        num_views=args.num_views,
                    )
                )

    if args.missing_affordance_only:
        output_rows = [r for r in rows if r.get("aps_missing_score_zero")]
    elif args.ready_only:
        output_rows = [r for r in rows if r["ready"]]
    else:
        output_rows = rows
    write_jsonl(output_rows, args.output_jsonl)
    write_csv(output_rows, args.output_csv)

    print(f"rows_total={len(rows)} rows_written={len(output_rows)} ready={sum(1 for r in rows if r['ready'])}")
    by_key = {}
    for row in rows:
        key = (row["method"], row["dataset"])
        stats = by_key.setdefault(key, {"total": 0, "ready": 0, "statuses": {}})
        stats["total"] += 1
        stats["ready"] += int(row["ready"])
        for status in row["status"].split(";"):
            if status:
                stats["statuses"][status] = stats["statuses"].get(status, 0) + 1
    for key, stats in sorted(by_key.items()):
        method, dataset = key
        print(f"{method},{dataset}: total={stats['total']} ready={stats['ready']} statuses={stats['statuses']}")
    print(f"jsonl={args.output_jsonl}")
    print(f"csv={args.output_csv}")


if __name__ == "__main__":
    main()
