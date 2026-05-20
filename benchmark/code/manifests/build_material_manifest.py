#!/usr/bin/env python3
"""Build MPS/material benchmark manifest rows.

Each material row needs:
  - condition image;
  - Video A: floor/ground-impact video;
  - Video B: water-entry video;
  - compact material-parameter JSON text from each generated result folder's
    basic_info.json, injected into the prompt.

If either material video is missing or invalid, the row is kept with
``mps_missing_score_zero=true`` so multi.py can deterministically write MPS=0.
"""

import argparse
import csv
import json
from pathlib import Path


DATASETS = {
    "mobility": {
        "ours": {
            "source_folder": "ours_mobility_181500",
            "floor_folder": "ours_mobility",
        },
        "physxanything": {
            "source_folder": "output_physxanything_mobility",
            "floor_folder": "physxanything_mobility",
        },
    },
    "verse": {
        "ours": {
            "source_folder": "ours_verse_181500",
            "floor_folder": "ours_verse",
        },
        "physxanything": {
            "source_folder": "output_physxanything_verse",
            "floor_folder": "physxanything_verse",
        },
    },
    "inthewild": {
        "ours": {
            "source_folder": "ours_inthewild_181500",
            "floor_folder": "ours_inthewild",
        },
        "physxanything": {
            "source_folder": "output_physxanything_inthewild",
            "floor_folder": "physxanything_inthewild",
        },
    },
}


METHOD_ALIASES = {
    "physanything": "physxanything",
}


MISSING_ZERO_TOKENS = {
    "missing_floor_video",
    "missing_water_video",
    "invalid_floor_video",
    "invalid_water_video",
    "insufficient_material_videos",
    "missing_material_json",
    "invalid_material_json",
}


def condition_image_path(physx_result_root: Path, dataset: str, object_id: str) -> Path:
    return physx_result_root / f"demo_{dataset}" / f"{object_id}.png"


def valid_file(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def sample_dirs(result_root: Path):
    if not result_root.is_dir():
        return []
    return sorted(
        [p for p in result_root.iterdir() if p.is_dir()],
        key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name),
    )


def compact_material_parameters(json_path: Path) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = {
        "object_name": data.get("object_name"),
        "category": data.get("category"),
        "dimension": data.get("dimension"),
        "parts": [],
    }
    for part in data.get("parts", []) or []:
        if not isinstance(part, dict):
            continue
        out["parts"].append(
            {
                "label": part.get("label"),
                "name": part.get("name"),
                "material": part.get("material"),
                "density": part.get("density"),
                "Young's Modulus (GPa)": part.get("Young's Modulus (GPa)"),
                "Poisson's Ratio": part.get("Poisson's Ratio"),
                "priority_rank": part.get("priority_rank"),
            }
        )
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def material_parameters_from_sample_dir(sample_dir: Path):
    json_path = sample_dir / "basic_info.json"
    if json_path.is_file():
        try:
            return compact_material_parameters(json_path), json_path, None
        except Exception as exc:  # noqa: BLE001
            return "", json_path, f"invalid_material_json_detail:{type(exc).__name__}"

    txt_path = sample_dir / "basic_info.txt"
    if txt_path.is_file():
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:  # noqa: BLE001
            return "", txt_path, f"invalid_material_txt_detail:{type(exc).__name__}"
        if text:
            return text, txt_path, None
        return "", txt_path, "invalid_material_txt_detail:empty"

    return "", json_path, "missing_material_json"


def build_row(
    *,
    physx_result_root: Path,
    water_root: Path,
    floor_root: Path,
    method: str,
    dataset: str,
    source_folder: str,
    floor_folder: str,
    sample_dir: Path,
    object_id: str,
    min_video_bytes: int,
):
    image_path = condition_image_path(physx_result_root, dataset, object_id)
    material_parameters, metric_json_path, material_error = material_parameters_from_sample_dir(sample_dir)
    floor_video = floor_root / floor_folder / f"{object_id}_floor.mp4"
    water_video = water_root / source_folder / f"{source_folder}_{object_id}_mpm.mp4"

    status = []
    if not image_path.is_file():
        status.append("missing_condition_image")
    if material_error == "missing_material_json":
        status.append("missing_material_json")
    elif material_error:
        status.append("invalid_material_json")
        status.append(material_error)

    floor_ok = valid_file(floor_video, min_video_bytes)
    water_ok = valid_file(water_video, min_video_bytes)
    if not floor_video.is_file():
        status.append("missing_floor_video")
    elif not floor_ok:
        status.append("invalid_floor_video")
    if not water_video.is_file():
        status.append("missing_water_video")
    elif not water_ok:
        status.append("invalid_water_video")
    if not (floor_ok and water_ok):
        status.append("insufficient_material_videos")

    missing_reasons = [token for token in status if token in MISSING_ZERO_TOKENS]
    ready = not status
    video_paths = [str(floor_video), str(water_video)] if floor_ok and water_ok else []
    return {
        "metric": "mps",
        "method": method,
        "dataset": dataset,
        "object_id": object_id,
        "sample_id": object_id,
        "image_path": str(image_path),
        "condition_image": str(image_path),
        "video_paths": video_paths,
        "floor_video_path": str(floor_video),
        "water_video_path": str(water_video),
        "material_parameters_path": str(metric_json_path),
        "material_parameters": material_parameters,
        "relative_dir": f"mps/{method}/{dataset}/{object_id}",
        "source_folder": source_folder,
        "source_result_dir": str(sample_dir),
        "floor_folder": floor_folder,
        "num_material_videos": len(video_paths),
        "num_material_videos_required": 2,
        "mps_missing_score_zero": bool(missing_reasons),
        "missing_material_reason": ";".join(dict.fromkeys(missing_reasons)),
        "ready": ready,
        "status": "ready" if ready else ";".join(dict.fromkeys(status)),
    }


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "method",
        "dataset",
        "object_id",
        "image_path",
        "video_paths",
        "floor_video_path",
        "water_video_path",
        "material_parameters_path",
        "relative_dir",
        "source_folder",
        "floor_folder",
        "num_material_videos",
        "num_material_videos_required",
        "mps_missing_score_zero",
        "missing_material_reason",
        "ready",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["video_paths"] = json.dumps(out.get("video_paths", []), ensure_ascii=False)
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Build MPS/material pairs manifest.")
    parser.add_argument("--physx-result-root", default="physx_result")
    parser.add_argument(
        "--water-root",
        default="benchmark/benchmark_assets/material_videos/water",
    )
    parser.add_argument(
        "--floor-root",
        default="benchmark/benchmark_assets/material_videos_v2/floor",
    )
    parser.add_argument("--methods", nargs="+", default=["ours", "physxanything"])
    parser.add_argument("--datasets", nargs="+", default=["mobility", "verse", "inthewild"])
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional per method/dataset limit.")
    parser.add_argument("--min-video-bytes", type=int, default=1024)
    parser.add_argument(
        "--output-jsonl",
        default="benchmark/benchmark_manifests/material_pairs.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark/benchmark_manifests/material_pairs.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    physx_result_root = Path(args.physx_result_root)
    water_root = Path(args.water_root)
    floor_root = Path(args.floor_root)
    methods = {METHOD_ALIASES.get(method, method) for method in args.methods}
    datasets = set(args.datasets)

    rows = []
    for dataset, method_map in DATASETS.items():
        if dataset not in datasets:
            continue
        for method, cfg in method_map.items():
            if method not in methods:
                continue
            result_root = physx_result_root / cfg["source_folder"]
            files = sample_dirs(result_root)
            if args.limit is not None:
                files = files[: args.limit]
            for sample_dir in files:
                rows.append(
                    build_row(
                        physx_result_root=physx_result_root,
                        water_root=water_root,
                        floor_root=floor_root,
                        method=method,
                        dataset=dataset,
                        source_folder=cfg["source_folder"],
                        floor_folder=cfg["floor_folder"],
                        sample_dir=sample_dir,
                        object_id=sample_dir.name,
                        min_video_bytes=args.min_video_bytes,
                    )
                )

    if args.missing_only:
        output_rows = [row for row in rows if row.get("mps_missing_score_zero")]
    elif args.ready_only:
        output_rows = [row for row in rows if row.get("ready")]
    else:
        output_rows = rows

    write_jsonl(output_rows, Path(args.output_jsonl))
    write_csv(output_rows, Path(args.output_csv))

    print(
        f"rows_total={len(rows)} rows_written={len(output_rows)} "
        f"ready={sum(1 for row in rows if row['ready'])}"
    )
    by_key = {}
    for row in rows:
        key = (row["method"], row["dataset"])
        stats = by_key.setdefault(key, {"total": 0, "ready": 0, "statuses": {}})
        stats["total"] += 1
        stats["ready"] += int(row["ready"])
        for status in row["status"].split(";"):
            if status:
                stats["statuses"][status] = stats["statuses"].get(status, 0) + 1
    for (method, dataset), stats in sorted(by_key.items()):
        print(f"{method},{dataset}: total={stats['total']} ready={stats['ready']} statuses={stats['statuses']}")
    print(f"jsonl={args.output_jsonl}")
    print(f"csv={args.output_csv}")


if __name__ == "__main__":
    main()
