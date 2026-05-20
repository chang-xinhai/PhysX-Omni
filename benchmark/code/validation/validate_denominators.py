#!/usr/bin/env python3
"""Validate benchmark denominators across manifests, raw outputs, and summaries.

This script is intentionally dependency-light. It answers one practical
question before paper/table reporting:

  For each method/dataset/metric, did the final summary use the same denominator
  as the manifest, including deterministic zero rows for missing evidence?

It reports expected manifest counts, missing-zero counts, parsed raw result
counts, duplicate object scores, malformed raw outputs, and summary CSV counts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


METRIC_CANONICAL = {
    "rqs": "RQS",
    "quality": "RQS",
    "mcs": "MCS",
    "consistency": "MCS",
    "dcs": "DCS",
    "description": "DCS",
    "aps": "APS",
    "affordance": "APS",
    "kps": "KPS",
    "vaps": "KPS",
    "dqs": "DQS",
    "dimension": "DQS",
    "mps": "MPS",
    "material": "MPS",
}

SKIP_METRICS = {"DQS_PRIOR"}

MISSING_ZERO_FIELDS = [
    "render_missing_score_zero",
    "aps_missing_score_zero",
    "kps_missing_score_zero",
    "dcs_missing_score_zero",
    "mps_missing_score_zero",
]


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]


def load_aggregate_module(repo_root: Path):
    path = repo_root / "benchmark" / "code" / "aggregation" / "aggregate_vlm_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_vlm_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import aggregation module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def canonical_metric(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in {"RQS", "MCS", "DCS", "DQS", "APS", "KPS", "VAPS", "MPS", "DQS_PRIOR"}:
        return "KPS" if upper == "VAPS" else upper
    return METRIC_CANONICAL.get(raw.lower(), upper)


def canonical_method(value: Any) -> str:
    method = str(value or "").strip()
    if method == "physanything":
        return "physxanything"
    if method == "physgen":
        return "physxgen"
    return method


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return list(read_csv_rows(path))
    return list(read_jsonl(path))


def iter_manifest_paths(paths: List[str], manifest_root: str | None) -> List[Path]:
    out = [Path(p) for p in paths]
    if manifest_root:
        root = Path(manifest_root)
        out.extend(sorted(root.glob("*.jsonl")))
    # Prefer JSONL over CSV when both exist for the same stem.
    best: Dict[Tuple[Path, str], Path] = {}
    for path in out:
        if not path.exists() or path.name.endswith("_missing.jsonl"):
            continue
        key = (path.parent.resolve(), path.stem)
        old = best.get(key)
        if old is None or (old.suffix.lower() == ".csv" and path.suffix.lower() == ".jsonl"):
            best[key] = path
    return sorted(best.values())


def is_manifest_missing_zero(row: Dict[str, Any]) -> bool:
    if any(truthy(row.get(field)) for field in MISSING_ZERO_FIELDS):
        return True
    source = str(row.get("algorithm_dimension_source") or "").lower()
    if source in {"default_zero", "scale_npy_default_zero"}:
        return True
    if truthy(row.get("algorithm_dimension_defaulted")):
        return True
    return False


def manifest_counts(manifest_paths: List[Path]):
    counts: Counter[Tuple[str, str, str]] = Counter()
    ready: Counter[Tuple[str, str, str]] = Counter()
    missing_zero: Counter[Tuple[str, str, str]] = Counter()
    row_ids: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)
    manifest_rows = 0

    for path in manifest_paths:
        for row in read_manifest(path):
            metric = canonical_metric(row.get("metric") or path.stem)
            if not metric or metric in SKIP_METRICS:
                continue
            method = canonical_method(row.get("method"))
            dataset = str(row.get("dataset") or "").strip()
            object_id = str(row.get("object_id") or row.get("sample_id") or "").strip()
            if not method or not dataset or not object_id:
                continue
            key = (method, dataset, metric)
            counts[key] += 1
            manifest_rows += 1
            row_ids[key].add(object_id)
            if truthy(row.get("ready")):
                ready[key] += 1
            if is_manifest_missing_zero(row):
                missing_zero[key] += 1
    return counts, ready, missing_zero, row_ids, manifest_rows


def read_summary_counts(path: str | None):
    counts: Dict[Tuple[str, str, str], int] = {}
    means: Dict[Tuple[str, str, str], str] = {}
    if not path or not Path(path).is_file():
        return counts, means
    for row in read_csv_rows(Path(path)):
        key = (
            canonical_method(row.get("method")),
            str(row.get("dataset") or ""),
            canonical_metric(row.get("metric")),
        )
        try:
            counts[key] = int(float(row.get("count") or 0))
        except ValueError:
            counts[key] = 0
        means[key] = str(row.get("mean") or row.get("score_mean") or "")
    return counts, means


def read_object_counts(path: str | None):
    counts: Counter[Tuple[str, str, str]] = Counter()
    ids: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)
    if not path or not Path(path).is_file():
        return counts, ids
    for row in read_csv_rows(Path(path)):
        key = (
            canonical_method(row.get("method")),
            str(row.get("dataset") or ""),
            canonical_metric(row.get("metric")),
        )
        object_id = str(row.get("object_id") or "")
        if not all(key) or not object_id:
            continue
        counts[key] += 1
        ids[key].add(object_id)
    return counts, ids


def raw_counts(results_roots: List[str], repo_root: Path):
    results_roots = [root for root in results_roots if Path(root).exists()]
    if not results_roots:
        return Counter(), Counter(), Counter(), {}, [], 0
    agg = load_aggregate_module(repo_root)
    rows, errors = agg.parse_results(results_roots)
    before: Counter[Tuple[str, str, str]] = Counter()
    auto: Counter[Tuple[str, str, str]] = Counter()
    per_object: Counter[Tuple[str, str, str, str]] = Counter()
    for row in rows:
        key = (
            canonical_method(row.get("method")),
            str(row.get("dataset") or ""),
            canonical_metric(row.get("metric")),
        )
        if not all(key):
            continue
        before[key] += 1
        if row.get("_auto_scored"):
            auto[key] += 1
        per_object[(key[0], key[1], key[2], str(row.get("object_id")))] += 1

    dedup_rows = agg.deduplicate_rows(rows)
    dedup: Counter[Tuple[str, str, str]] = Counter()
    dedup_ids: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)
    for row in dedup_rows:
        key = (
            canonical_method(row.get("method")),
            str(row.get("dataset") or ""),
            canonical_metric(row.get("metric")),
        )
        if not all(key):
            continue
        dedup[key] += 1
        dedup_ids[key].add(str(row.get("object_id")))

    duplicate_keys: Counter[Tuple[str, str, str]] = Counter()
    for method, dataset, metric, _object_id in per_object:
        key = (method, dataset, metric)
        if per_object[(method, dataset, metric, _object_id)] > 1:
            duplicate_keys[key] += 1

    return before, dedup, auto, dedup_ids, errors, sum(before.values())


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark denominator consistency.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest JSONL/CSV. Can be repeated.")
    parser.add_argument(
        "--manifest-root",
        default="benchmark/benchmark_manifests",
        help="Directory whose *.jsonl manifests should be included. Set empty to disable.",
    )
    parser.add_argument("--results-root", action="append", default=[], help="Raw VLM output root. Can be repeated.")
    parser.add_argument(
        "--object-csv",
        default="benchmark/benchmark_results/object_level_scores/object_scores_long.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="benchmark/benchmark_results/dataset_level_scores/dataset_metric_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark/benchmark_results/logs/denominator_validation.csv",
    )
    parser.add_argument(
        "--errors-jsonl",
        default="benchmark/benchmark_results/logs/denominator_validation_errors.jsonl",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if count mismatches are found.")
    args = parser.parse_args()

    repo_root = repo_root_from_this_file()
    manifest_root = args.manifest_root if args.manifest_root else None
    manifests = iter_manifest_paths(args.manifest, manifest_root)
    expected, ready, manifest_zero, manifest_ids, manifest_row_count = manifest_counts(manifests)
    raw_before, raw_dedup, raw_auto, raw_ids, raw_errors, raw_row_count = raw_counts(args.results_root, repo_root)
    object_counts, object_ids = read_object_counts(args.object_csv)
    summary_counts, summary_means = read_summary_counts(args.summary_csv)

    all_keys = sorted(set(expected) | set(raw_dedup) | set(object_counts) | set(summary_counts))
    rows = []
    mismatch_count = 0
    for key in all_keys:
        method, dataset, metric = key
        expected_count = expected.get(key, 0)
        raw_count = raw_dedup.get(key, 0)
        object_count = object_counts.get(key, 0)
        summary_count = summary_counts.get(key, 0)
        missing_expected_ids = sorted(manifest_ids.get(key, set()) - raw_ids.get(key, set()))
        extra_raw_ids = sorted(raw_ids.get(key, set()) - manifest_ids.get(key, set()))
        mismatch = (
            (expected_count and raw_count != expected_count)
            or (expected_count and object_count and object_count != expected_count)
            or (expected_count and summary_count and summary_count != expected_count)
            or bool(extra_raw_ids)
        )
        mismatch_count += int(bool(mismatch))
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "metric": metric,
                "expected_count": expected_count,
                "manifest_ready_count": ready.get(key, 0),
                "manifest_missing_zero_count": manifest_zero.get(key, 0),
                "raw_metric_rows_before_dedup": raw_before.get(key, 0),
                "raw_dedup_count": raw_count,
                "raw_auto_zero_count": raw_auto.get(key, 0),
                "object_score_count": object_count,
                "summary_count": summary_count,
                "summary_mean": summary_means.get(key, ""),
                "duplicate_object_result_keys": max(0, raw_before.get(key, 0) - raw_count),
                "missing_raw_after_dedup": max(0, expected_count - raw_count),
                "extra_raw_after_dedup": max(0, raw_count - expected_count) if expected_count else raw_count,
                "missing_object_ids_sample": ";".join(missing_expected_ids[:20]),
                "extra_raw_object_ids_sample": ";".join(extra_raw_ids[:20]),
                "count_mismatch": bool(mismatch),
            }
        )

    fieldnames = [
        "method",
        "dataset",
        "metric",
        "expected_count",
        "manifest_ready_count",
        "manifest_missing_zero_count",
        "raw_metric_rows_before_dedup",
        "raw_dedup_count",
        "raw_auto_zero_count",
        "object_score_count",
        "summary_count",
        "summary_mean",
        "duplicate_object_result_keys",
        "missing_raw_after_dedup",
        "extra_raw_after_dedup",
        "missing_object_ids_sample",
        "extra_raw_object_ids_sample",
        "count_mismatch",
    ]
    write_csv(Path(args.output_csv), rows, fieldnames)

    err_path = Path(args.errors_jsonl)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    with err_path.open("w", encoding="utf-8") as handle:
        meta = {
            "manifest_files": [str(path) for path in manifests],
            "manifest_rows_counted": manifest_row_count,
            "raw_metric_rows_before_dedup": raw_row_count,
            "raw_parse_error_count": len(raw_errors),
            "count_mismatch_groups": mismatch_count,
        }
        handle.write(json.dumps({"type": "meta", **meta}, ensure_ascii=False) + "\n")
        for err in raw_errors:
            handle.write(json.dumps({"type": "raw_parse_error", **err}, ensure_ascii=False) + "\n")

    print(
        f"groups={len(rows)} mismatches={mismatch_count} "
        f"manifest_rows={manifest_row_count} raw_rows={raw_row_count} raw_errors={len(raw_errors)}"
    )
    print(f"validation_csv={args.output_csv}")
    print(f"errors_jsonl={args.errors_jsonl}")
    if args.strict and mismatch_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
