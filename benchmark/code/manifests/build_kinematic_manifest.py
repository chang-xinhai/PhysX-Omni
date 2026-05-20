#!/usr/bin/env python3
"""Build a KPS manifest from physx_result method outputs.

The output JSONL can be passed to multi.py with --pairs-manifest after required
condition images and videos exist.

Rows use the normalized fields:
  metric, method, dataset, object_id, image_path, video_path, relative_dir, status
"""

import argparse
import csv
import json
from pathlib import Path


DATASETS = {
    "mobility": {
        "articulateanything": "output_articulateanything_mobility",
        "physxanything": "output_physxanything_mobility",
        "physxgen": "outputs_physxgen_mobility",
        "ours": "ours_mobility_181500",
    },
    "verse": {
        "articulateanything": "output_articulateanything_verse",
        "physxanything": "output_physxanything_verse",
        "physxgen": "outputs_physxgen_verse",
        "ours": "ours_verse_181500",
    },
    "inthewild": {
        "articulateanything": "output_articulateanything_inthewild",
        "physxanything": "output_physxanything_inthewild",
        "physxgen": "outputs_physxgen_inthewild",
        "ours": "ours_inthewild_181500",
    },
}


def sample_dirs(root):
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def condition_image_path(condition_image_root, dataset, object_id):
    return Path(condition_image_root) / dataset / str(object_id) / "first_frame.png"


def candidate_video_paths(video_root, method, dataset, object_id):
    root = Path(video_root)
    canonical = root / method / dataset / str(object_id) / "kinematic_demo.mp4"
    candidates = [canonical]
    # Early benchmark runs used "physanything" in asset paths. Keep this
    # compatibility read-only so old rendered evidence remains discoverable.
    if method == "physxanything":
        candidates.append(root / "physanything" / dataset / str(object_id) / "kinematic_demo.mp4")
    return candidates


def expected_video_path(video_root, method, dataset, object_id):
    candidates = candidate_video_paths(video_root, method, dataset, object_id)
    canonical = candidates[0]
    if canonical.is_file():
        return canonical
    for path in candidates[1:]:
        if path.is_file():
            return path
    return canonical


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.is_file():
            return path
    return None


def find_articulateanything_urdf(sample_dir):
    candidates = [
        sample_dir / "joint_actor" / "iter_0" / "seed_0" / "mobility.urdf",
        sample_dir / "joint_actor" / "iter_0" / "seed_0" / "basic.urdf",
        sample_dir / "mobility.urdf",
        sample_dir / "basic.urdf",
    ]
    found = first_existing(candidates)
    if found:
        return found
    urdfs = sorted((sample_dir / "joint_actor").glob("**/*.urdf")) if (sample_dir / "joint_actor").is_dir() else []
    return urdfs[0] if urdfs else candidates[0]


def resolve_existing_video(video_root, fallback_video_roots, method, dataset, object_id):
    for path in candidate_video_paths(video_root, method, dataset, object_id):
        if path.is_file():
            return path, "primary", ""
    for fallback_root in fallback_video_roots:
        for path in candidate_video_paths(fallback_root, method, dataset, object_id):
            if path.is_file():
                return path, "fallback", str(path)
    return None, "", ""


def build_row(physx_root, condition_image_root, video_root, fallback_video_roots, method, dataset, sample_dir):
    object_id = sample_dir.name
    image_path = condition_image_path(condition_image_root, dataset, object_id)
    expected_path = expected_video_path(video_root, method, dataset, object_id)
    existing_video_path, video_source, fallback_video_path = resolve_existing_video(
        video_root=video_root,
        fallback_video_roots=fallback_video_roots,
        method=method,
        dataset=dataset,
        object_id=object_id,
    )
    status = []

    if not image_path.is_file():
        status.append("missing_condition_image")

    row = {
        "metric": "kps",
        "method": method,
        "dataset": dataset,
        "object_id": object_id,
        "sample_id": object_id,
        "image_path": str(image_path),
        "condition_image": str(image_path),
        "video_path": str(existing_video_path) if existing_video_path else "",
        "expected_video_path": str(expected_path),
        "fallback_video_path": fallback_video_path,
        "video_source": video_source,
        "relative_dir": f"kps/{method}/{dataset}/{object_id}",
        "source_result_dir": str(sample_dir),
        "source_xml": "",
        "source_urdf": "",
        "status": "",
    }

    if video_source == "fallback":
        status.append("uses_fallback_video")

    if method == "articulateanything":
        source_urdf = find_articulateanything_urdf(sample_dir)
        row["source_urdf"] = str(source_urdf)
        if not source_urdf.is_file():
            status.append("missing_source_urdf")
        if not row["video_path"]:
            status.append("needs_render_video")
    else:
        source_xml = sample_dir / "basic.xml"
        source_urdf = sample_dir / "basic.urdf"
        if method == "physxgen":
            source_xml = sample_dir / "mesh" / "basic.xml"
            source_urdf = sample_dir / "mesh" / "basic.urdf"
        row["source_xml"] = str(source_xml)
        row["source_urdf"] = str(source_urdf)
        if method == "physxgen":
            if not source_urdf.is_file():
                status.append("missing_source_urdf")
        elif not source_xml.is_file():
            status.append("missing_source_xml_has_urdf" if source_urdf.is_file() else "missing_source_xml")
        if not row["video_path"]:
            status.append("needs_render_video")

    if not row["video_path"]:
        status.append("missing_video_path")

    non_blocking = {
        "uses_single_aa_video_directly",
        "uses_fallback_video",
        "missing_source_xml",
        "missing_source_xml_has_urdf",
    }
    blocking = [s for s in status if s not in non_blocking]
    row["ready"] = not blocking
    row["status"] = "ready" if row["ready"] else ";".join(status)
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
        "video_path",
        "expected_video_path",
        "fallback_video_path",
        "video_source",
        "relative_dir",
        "source_result_dir",
        "source_xml",
        "source_urdf",
        "ready",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Build KPS/VAPS pairs manifest for physx_result outputs.")
    parser.add_argument("--physx-result-root", default="physx_result")
    parser.add_argument(
        "--condition-image-root",
        default="benchmark/benchmark_assets/condition_images",
    )
    parser.add_argument(
        "--video-root",
        default="benchmark/benchmark_assets/kinematic_videos",
    )
    parser.add_argument(
        "--fallback-video-root",
        action="append",
        default=[],
        help=(
            "Additional rendered-video root to consult after --video-root. "
            "May be passed multiple times; useful for URDF fallback videos."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        default="benchmark/benchmark_manifests/kinematic_pairs.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark/benchmark_manifests/kinematic_pairs.csv",
    )
    parser.add_argument("--ready-only", action="store_true", help="Only write rows that are ready for multi.py.")
    parser.add_argument("--methods", nargs="+", default=None, help="Optional method filter.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset filter.")
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Only write rows whose resolved video_path comes from --fallback-video-root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    physx_root = Path(args.physx_result_root).expanduser().resolve()
    condition_image_root = Path(args.condition_image_root).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    fallback_video_roots = [Path(p).expanduser().resolve() for p in args.fallback_video_root]
    rows = []
    method_filter = set(args.methods or [])
    dataset_filter = set(args.datasets or [])
    for dataset, methods in DATASETS.items():
        if dataset_filter and dataset not in dataset_filter:
            continue
        for method, rel_dir in methods.items():
            if method_filter and method not in method_filter:
                continue
            result_root = physx_root / rel_dir
            for sample_dir in sample_dirs(result_root):
                rows.append(
                    build_row(
                        physx_root=physx_root,
                        condition_image_root=condition_image_root,
                        video_root=video_root,
                        fallback_video_roots=fallback_video_roots,
                        method=method,
                        dataset=dataset,
                        sample_dir=sample_dir,
                    )
                )

    output_rows = [r for r in rows if r["ready"]] if args.ready_only else rows
    if args.fallback_only:
        output_rows = [r for r in output_rows if r.get("video_source") == "fallback"]
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
