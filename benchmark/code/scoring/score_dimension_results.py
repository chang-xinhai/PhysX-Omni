#!/usr/bin/env python3
"""Compute deterministic DQS from VLM dimension priors and algorithm dimensions."""

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

def extract_json_object(raw):
    if raw is None:
        raise ValueError("empty output")
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def parse_dimension_string(value):
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value or ""))]
    if len(nums) < 3:
        raise ValueError(f"dimension string has fewer than 3 numbers: {value!r}")
    nums = nums[:3]
    if any(x <= 0 or not math.isfinite(x) for x in nums):
        raise ValueError(f"dimension string has non-positive/non-finite numbers: {value!r}")
    return sorted(nums, reverse=True)


def zero_dimension(source, algorithm_dimension="0", failure_reason=None):
    return {
        "source": source,
        "generated_dimensions_cm_sorted": [0.0, 0.0, 0.0],
        "generated_max_dimension_cm": 0.0,
        "algorithm_dimension": algorithm_dimension,
        "dimension_output_status": "failed" if failure_reason else "ok",
        "dimension_output_failure_reason": failure_reason,
    }


def load_scale(path):
    import numpy as np

    value = float(np.load(path, allow_pickle=False).item())
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"invalid scale value: {value}")
    return value


def load_algorithm_dimension(row):
    source = row.get("algorithm_dimension_source")
    if source in {"basic_info_json", "basic_info_txt"}:
        dim = row.get("algorithm_dimension")
        if not dim and source == "basic_info_json" and row.get("algorithm_json_path"):
            with open(row["algorithm_json_path"], "r", encoding="utf-8") as f:
                dim = json.load(f).get("dimension")
        try:
            dims = parse_dimension_string(dim)
        except ValueError as exc:
            if "fewer than 3 numbers" in str(exc):
                return zero_dimension(
                    f"{source}_invalid_zero",
                    algorithm_dimension=dim,
                    failure_reason="dimension string has fewer than 3 numbers",
                )
            raise
        return {
            "source": source,
            "generated_dimensions_cm_sorted": dims,
            "generated_max_dimension_cm": dims[0],
            "algorithm_dimension": dim,
            "dimension_output_status": "ok",
            "dimension_output_failure_reason": None,
        }
    if source in {"default_zero", "scale_npy_default_zero"}:
        return zero_dimension(source)
    if source == "scale_npy":
        scale_path = row.get("algorithm_scale_path")
        if not scale_path:
            return zero_dimension("scale_npy_default_zero")
        max_dim = load_scale(scale_path)
        return {
            "source": source,
            "generated_dimensions_cm_sorted": [max_dim],
            "generated_max_dimension_cm": max_dim,
            "algorithm_dimension": str(max_dim),
            "dimension_output_status": "ok",
            "dimension_output_failure_reason": None,
        }
    raise ValueError(f"unknown algorithm_dimension_source: {source}")


def verdict_from_score(score):
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "fair"
    if score >= 40:
        return "poor"
    return "bad"


def compute_dqs(g, v):
    symmetric_error = 2.0 * abs(g - v) / (g + v)
    if symmetric_error >= 0.8:
        dqs = 0.0
    else:
        dqs = 100.0 * (1.0 - symmetric_error / 0.8)
    return symmetric_error, round(dqs, 2)


def load_manifest(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def result_key_from_payload(payload, result_path):
    ctx = payload.get("benchmark_context") or {}
    return (
        str(ctx.get("method") or ""),
        str(ctx.get("dataset") or ""),
        str(ctx.get("object_id") or ctx.get("sample_id") or payload.get("video_id") or result_path.parent.name),
    )


def collect_priors(results_root):
    priors = {}
    shared_priors = {}
    errors = []
    for result_path in sorted(Path(results_root).rglob("result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"result_json": str(result_path), "error": f"read_json: {exc}"})
            continue
        key = result_key_from_payload(payload, result_path)
        for item in payload.get("results", []):
            if item.get("error"):
                errors.append({"result_json": str(result_path), "turn_id": item.get("turn_id"), "error": item.get("error")})
                continue
            try:
                parsed = extract_json_object(item.get("output"))
            except Exception as exc:
                errors.append({"result_json": str(result_path), "turn_id": item.get("turn_id"), "error": f"parse_output: {exc}"})
                continue
            if parsed.get("task") != "dimension_prior":
                continue
            dims = parsed.get("estimated_dimensions_cm_sorted")
            max_dim = parsed.get("max_dimension_cm")
            if not isinstance(dims, list) or len(dims) != 3:
                errors.append({"result_json": str(result_path), "turn_id": item.get("turn_id"), "error": "invalid estimated_dimensions_cm_sorted"})
                continue
            try:
                dims = [float(x) for x in dims]
                max_dim = float(max_dim)
            except Exception as exc:
                errors.append({"result_json": str(result_path), "turn_id": item.get("turn_id"), "error": f"invalid_numeric_prior: {exc}"})
                continue
            if any(x <= 0 or not math.isfinite(x) for x in dims) or max_dim <= 0 or not math.isfinite(max_dim):
                errors.append({"result_json": str(result_path), "turn_id": item.get("turn_id"), "error": "non_positive_or_non_finite_prior"})
                continue
            priors[key] = {
                "prior": parsed,
                "result_json": str(result_path),
                "turn_id": item.get("turn_id"),
            }
            method, dataset, object_id = key
            shared_key = (dataset, object_id)
            if method in {"", "shared", "__shared__"}:
                shared_priors[shared_key] = priors[key]
            else:
                shared_priors.setdefault(shared_key, priors[key])
    return priors, shared_priors, errors


def write_multi_like_result(row, prior_info, score_json, output_run_dir):
    rel = Path(f"dqs/{row['method']}/{row['dataset']}/{row['object_id']}")
    out_dir = output_run_dir.joinpath(*rel.parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    payload = {
        "run_id": output_run_dir.name,
        "video_path": None,
        "video_id": row["object_id"],
        "video_relative_dir": str(rel),
        "paired_image_path": row.get("image_path"),
        "affordance_view_paths": [],
        "benchmark_context": row,
        "sampling": {"mode": "no_video"},
        "turns_template": [],
        "results": [
            {
                "turn_id": "dimension_scoring",
                "turn_index": 0,
                "prompt_ref_id": "deterministic_dimension_scoring",
                "input_modalities": {"image": False, "affordance_view_images": 0, "video": False},
                "output": json.dumps(score_json, ensure_ascii=False),
                "error": None,
                "kv_cache_hit": None,
                "elapsed_sec": 0.0,
            }
        ],
        "pair_error": None,
        "elapsed_sec": 0.0,
        "dimension_prior_result_json": prior_info["result_json"],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_csv(rows, path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["dataset"]), []).append(float(row["DQS"]))
    out = []
    for (method, dataset), values in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "dataset": dataset,
                "metric": "DQS",
                "count": len(values),
                "mean": round(statistics.fmean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) >= 2 else 0.0,
            }
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Score DQS deterministically from dimension prior VLM outputs.")
    parser.add_argument("--manifest", default="benchmark/benchmark_manifests/dimension_pairs_ready.jsonl")
    parser.add_argument("--prior-results-root", required=True, help="multi.py run dir or raw output root containing dimension_prior result.json files.")
    parser.add_argument(
        "--output-root",
        default="benchmark/benchmark_results/raw_vlm_outputs/dimension_scoring",
    )
    parser.add_argument(
        "--object-csv",
        default="benchmark/benchmark_results/object_level_scores/dimension_scores.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="benchmark/benchmark_results/dataset_level_scores/dimension_metric_summary.csv",
    )
    parser.add_argument(
        "--errors-jsonl",
        default="benchmark/benchmark_results/logs/dimension_scoring_errors.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_rows = load_manifest(args.manifest)
    priors, shared_priors, errors = collect_priors(args.prior_results_root)
    run_ts = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    output_run_dir = Path(args.output_root) / run_ts
    scored_rows = []

    for row in manifest_rows:
        key = (str(row["method"]), str(row["dataset"]), str(row["object_id"]))
        shared_key = (str(row["dataset"]), str(row["object_id"]))
        prior_info = priors.get(key) or shared_priors.get(shared_key)
        if prior_info is None:
            errors.append({"method": row["method"], "dataset": row["dataset"], "object_id": row["object_id"], "error": "missing_dimension_prior"})
            continue
        try:
            prior = prior_info["prior"]
            estimated_dims = [float(x) for x in prior["estimated_dimensions_cm_sorted"]]
            estimated_max = float(prior["max_dimension_cm"])
            generated = load_algorithm_dimension(row)
            generated_max = float(generated["generated_max_dimension_cm"])
            symmetric_error, dqs = compute_dqs(generated_max, estimated_max)
            score_json = {
                "task": "dimension_scoring",
                "object_summary": prior.get("object_summary", ""),
                "generated_dimensions_cm_sorted": generated["generated_dimensions_cm_sorted"],
                "generated_max_dimension_cm": round(generated_max, 5),
                "estimated_dimensions_cm_sorted": estimated_dims,
                "estimated_max_dimension_cm": round(estimated_max, 5),
                "symmetric_error": round(symmetric_error, 5),
                "symmetric_error_percent": round(100.0 * symmetric_error, 2),
                "DQS": dqs,
                "verdict": verdict_from_score(dqs),
                "algorithm_dimension_source": generated["source"],
                "algorithm_dimension": generated["algorithm_dimension"],
                "dimension_output_status": generated.get("dimension_output_status", "ok"),
                "dimension_output_failure_reason": generated.get("dimension_output_failure_reason"),
            }
            result_path = write_multi_like_result(row, prior_info, score_json, output_run_dir)
            scored_rows.append(
                {
                    "method": row["method"],
                    "dataset": row["dataset"],
                    "object_id": row["object_id"],
                    "DQS": dqs,
                    "generated_max_dimension_cm": round(generated_max, 5),
                    "estimated_max_dimension_cm": round(estimated_max, 5),
                    "symmetric_error": round(symmetric_error, 5),
                    "algorithm_dimension_source": generated["source"],
                    "dimension_output_status": generated.get("dimension_output_status", "ok"),
                    "dimension_output_failure_reason": generated.get("dimension_output_failure_reason") or "",
                    "dimension_prior_result_json": prior_info["result_json"],
                    "dqs_result_json": str(result_path),
                }
            )
        except Exception as exc:
            errors.append({"method": row["method"], "dataset": row["dataset"], "object_id": row["object_id"], "error": f"score_failed: {type(exc).__name__}: {exc}"})

    write_csv(
        scored_rows,
        args.object_csv,
        [
            "method",
            "dataset",
            "object_id",
            "DQS",
            "generated_max_dimension_cm",
            "estimated_max_dimension_cm",
            "symmetric_error",
            "algorithm_dimension_source",
            "dimension_output_status",
            "dimension_output_failure_reason",
            "dimension_prior_result_json",
            "dqs_result_json",
        ],
    )
    write_csv(summarize(scored_rows), args.summary_csv, ["method", "dataset", "metric", "count", "mean", "std"])
    errors_path = Path(args.errors_jsonl)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")

    print(
        f"manifest_rows={len(manifest_rows)} priors={len(priors)} "
        f"shared_priors={len(shared_priors)} scored={len(scored_rows)} errors={len(errors)}"
    )
    print(f"raw_dqs_run={output_run_dir}")
    print(f"object_csv={args.object_csv}")
    print(f"summary_csv={args.summary_csv}")
    print(f"errors_jsonl={args.errors_jsonl}")


if __name__ == "__main__":
    main()
