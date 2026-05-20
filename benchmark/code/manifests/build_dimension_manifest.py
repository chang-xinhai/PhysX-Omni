#!/usr/bin/env python3
"""Build DQS prior manifest for generated results.

The VLM stage only estimates an image-based dimension prior from the shared
condition image. Algorithm dimensions are kept in the manifest and consumed by
`score_dimension_results.py`.
"""

import argparse
import csv
import json
from pathlib import Path

DATASETS = {
    "mobility": {
        "ours": {"dir": "ours_mobility_181500", "source": "basic_info_json"},
        "physxanything": {"dir": "output_physxanything_mobility", "source": "basic_info_json"},
        "physxgen": {"dir": "outputs_physxgen_mobility", "source": "scale_npy"},
    },
    "verse": {
        "ours": {"dir": "ours_verse_181500", "source": "basic_info_json"},
        "physxanything": {"dir": "output_physxanything_verse", "source": "basic_info_json"},
        "physxgen": {"dir": "outputs_physxgen_verse", "source": "scale_npy"},
    },
    "inthewild": {
        "physxanything": {"dir": "output_physxanything_inthewild", "source": "basic_info_json"},
        "physxgen": {"dir": "outputs_physxgen_inthewild", "source": "scale_npy"},
        "ours": {"dir": "ours_inthewild_181500", "source": "basic_info_json"},
    },
}


METHOD_ALIASES = {
    "physanything": "physxanything",
    "physgen": "physxgen",
}


def sample_dirs(root: Path):
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def condition_image_path(condition_image_root, dataset, object_id):
    return Path(condition_image_root) / dataset / str(object_id) / "first_frame.png"


def load_basic_info(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_basic_info_txt(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "name":
                data["object_name"] = value
            elif key == "category":
                data["category"] = value
            elif key == "dimension":
                data["dimension"] = value
    return data


def load_scale(path: Path):
    import numpy as np

    value = np.load(path, allow_pickle=False)
    return float(value.item())


def build_row(condition_image_root, method, dataset, sample_dir):
    object_id = sample_dir.name
    image_path = condition_image_path(condition_image_root, dataset, object_id)
    source_cfg = DATASETS[dataset][method]
    source_type = source_cfg["source"]
    status = []

    if not image_path.is_file():
        status.append("missing_condition_image")

    algorithm_json_path = ""
    algorithm_info_path = ""
    algorithm_scale_path = ""
    algorithm_dimension = ""
    algorithm_generated_max_dimension_cm = ""
    algorithm_dimension_defaulted = False
    object_name = ""
    category = ""
    output_source_type = source_type

    if source_type == "basic_info_json":
        json_path = sample_dir / "basic_info.json"
        txt_path = sample_dir / "basic_info.txt"
        algorithm_json_path = str(json_path)
        if json_path.is_file():
            algorithm_info_path = str(json_path)
            output_source_type = "basic_info_json"
            try:
                data = load_basic_info(json_path)
                algorithm_dimension = str(data.get("dimension") or "")
                object_name = str(data.get("object_name") or data.get("name") or "")
                category = str(data.get("category") or "")
                if not algorithm_dimension:
                    algorithm_dimension = "0*0*0"
                    algorithm_dimension_defaulted = True
                    output_source_type = "default_zero"
            except Exception as exc:
                algorithm_dimension = "0*0*0"
                algorithm_dimension_defaulted = True
                output_source_type = "default_zero"
                status.append(f"invalid_basic_info_json:{type(exc).__name__}")
        elif txt_path.is_file():
            algorithm_info_path = str(txt_path)
            output_source_type = "basic_info_txt"
            try:
                data = load_basic_info_txt(txt_path)
                algorithm_dimension = str(data.get("dimension") or "")
                object_name = str(data.get("object_name") or "")
                category = str(data.get("category") or "")
                if not algorithm_dimension:
                    algorithm_dimension = "0*0*0"
                    algorithm_dimension_defaulted = True
                    output_source_type = "default_zero"
            except Exception as exc:
                algorithm_dimension = "0*0*0"
                algorithm_dimension_defaulted = True
                output_source_type = "default_zero"
                status.append(f"invalid_basic_info_txt:{type(exc).__name__}")
        else:
            algorithm_dimension = "0*0*0"
            algorithm_dimension_defaulted = True
            output_source_type = "default_zero"
    elif source_type == "scale_npy":
        path = sample_dir / "scale.npy"
        algorithm_scale_path = str(path)
        if not path.is_file():
            algorithm_generated_max_dimension_cm = 0.0
            algorithm_dimension = "0"
            algorithm_dimension_defaulted = True
            output_source_type = "scale_npy_default_zero"
        else:
            try:
                algorithm_generated_max_dimension_cm = load_scale(path)
                if algorithm_generated_max_dimension_cm <= 0:
                    algorithm_generated_max_dimension_cm = 0.0
                    algorithm_dimension = "0"
                    algorithm_dimension_defaulted = True
                    output_source_type = "scale_npy_default_zero"
            except Exception as exc:
                algorithm_generated_max_dimension_cm = 0.0
                algorithm_dimension = "0"
                algorithm_dimension_defaulted = True
                output_source_type = "scale_npy_default_zero"
                status.append(f"invalid_scale_npy:{type(exc).__name__}")
    else:
        status.append(f"unknown_algorithm_dimension_source:{source_type}")

    non_ready = [
        item
        for item in status
        if item == "missing_condition_image" or item.startswith("unknown_algorithm_dimension_source")
    ]
    ready = not non_ready
    status_text = "ready" if ready and not status else ";".join(status)
    if ready and status:
        status_text = f"ready_with_notes;{status_text}"

    row = {
        "metric": "dqs",
        "method": method,
        "dataset": dataset,
        "object_id": object_id,
        "sample_id": object_id,
        "image_path": str(image_path),
        "condition_image": str(image_path),
        "relative_dir": f"dqs/{method}/{dataset}/{object_id}",
        "source_result_dir": str(sample_dir),
        "algorithm_dimension_source": output_source_type,
        "algorithm_json_path": algorithm_json_path,
        "algorithm_info_path": algorithm_info_path,
        "algorithm_scale_path": algorithm_scale_path,
        "algorithm_dimension": algorithm_dimension,
        "algorithm_generated_max_dimension_cm": algorithm_generated_max_dimension_cm,
        "algorithm_dimension_defaulted": algorithm_dimension_defaulted,
        "object_name": object_name,
        "category": category,
        "ready": ready,
        "status": status_text,
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
        "relative_dir",
        "source_result_dir",
        "algorithm_dimension_source",
        "algorithm_json_path",
        "algorithm_info_path",
        "algorithm_scale_path",
        "algorithm_dimension",
        "algorithm_generated_max_dimension_cm",
        "algorithm_dimension_defaulted",
        "object_name",
        "category",
        "ready",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_unique_prior_rows(rows):
    unique = {}
    for row in rows:
        if not row.get("ready"):
            continue
        key = (row["dataset"], row["object_id"])
        if key in unique:
            continue
        unique[key] = {
            "metric": "dqs_prior",
            "method": "shared",
            "dataset": row["dataset"],
            "object_id": row["object_id"],
            "sample_id": row["object_id"],
            "image_path": row["image_path"],
            "condition_image": row["condition_image"],
            "relative_dir": f"dqs_prior/{row['dataset']}/{row['object_id']}",
            "source_result_dir": "",
            "algorithm_dimension_source": "",
            "algorithm_json_path": "",
            "algorithm_info_path": "",
            "algorithm_scale_path": "",
            "algorithm_dimension": "",
            "algorithm_generated_max_dimension_cm": "",
            "algorithm_dimension_defaulted": "",
            "object_name": "",
            "category": "",
            "ready": True,
            "status": "ready",
        }
    return [unique[key] for key in sorted(unique)]


def parse_args():
    parser = argparse.ArgumentParser(description="Build DQS prior pairs manifest.")
    parser.add_argument("--physx-result-root", default="physx_result")
    parser.add_argument(
        "--condition-image-root",
        default="benchmark/benchmark_assets/condition_images",
    )
    parser.add_argument(
        "--output-jsonl",
        default="benchmark/benchmark_manifests/dimension_pairs.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark/benchmark_manifests/dimension_pairs.csv",
    )
    parser.add_argument("--methods", nargs="+", default=["ours", "physxanything", "physxgen"])
    parser.add_argument("--datasets", nargs="+", default=["mobility", "verse"])
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument(
        "--unique-priors",
        action="store_true",
        help=(
            "Emit one shared image-prior row per dataset/object_id instead of one row per method. "
            "Use this for the VLM dimension_prior pass; score_dimension_results.py can reuse the "
            "shared prior for all method rows."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    physx_root = Path(args.physx_result_root)
    selected_methods = {METHOD_ALIASES.get(method, method) for method in args.methods}
    selected_datasets = set(args.datasets)
    rows = []
    for dataset, methods in DATASETS.items():
        if dataset not in selected_datasets:
            continue
        for method, cfg in methods.items():
            if method not in selected_methods:
                continue
            result_root = physx_root / cfg["dir"]
            for sample_dir in sample_dirs(result_root):
                rows.append(
                    build_row(
                        condition_image_root=args.condition_image_root,
                        method=method,
                        dataset=dataset,
                        sample_dir=sample_dir,
                    )
                )

    if args.unique_priors:
        output_rows = build_unique_prior_rows(rows)
    else:
        output_rows = [r for r in rows if r["ready"]] if args.ready_only else rows
    write_jsonl(output_rows, args.output_jsonl)
    write_csv(output_rows, args.output_csv)

    print(f"rows_total={len(rows)} rows_written={len(output_rows)} ready={sum(1 for r in rows if r['ready'])}")
    if args.unique_priors:
        print(f"unique_prior_rows={len(output_rows)}")
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
