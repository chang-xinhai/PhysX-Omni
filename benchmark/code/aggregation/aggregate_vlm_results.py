#!/usr/bin/env python3
"""Aggregate VLM result.json files into benchmark CSV tables.

The script reads multi.py outputs, extracts the strict JSON emitted by the VLM,
and writes:
  1. object-level long rows, one row per parsed metric output;
  2. dataset-level metric summary, grouped by method/dataset/metric.
  3. dataset-level submetric summary for metrics with structured subscores.

It understands the current task schemas:
  affordance_scoring -> APS
  vaps_scoring -> KPS/VAPS plus S_prior/S_reveal/S_global
  dimension_scoring -> DQS
  description_mask_scoring -> DCS directly on a 0-100 scale
  material_scoring -> deterministic MPS from Young's modulus / Poisson's ratio / density subscores
  generic score in [1, 5] -> normalized 25 * (score - 1)
"""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import statistics
from pathlib import Path


TASK_TO_METRIC = {
    "affordance_scoring": "APS",
    "vaps_scoring": "KPS",
    "dimension_scoring": "DQS",
    "material_scoring": "MPS",
    "description_mask_scoring": "DCS",
}


GENERIC_TURN_METRICS = {
    "rqs": "RQS",
    "render_quality": "RQS",
    "mcs": "MCS",
    "multi_view": "MCS",
    "dcs": "DCS",
    "description": "DCS",
}


METHOD_ALIASES = {
    "physanything": "physxanything",
    "physgen": "physxgen",
}


EXCLUDED_DIR_NAMES = {"excluded_raw_vlm_outputs"}


def canonical_method(value):
    method = str(value or "unknown_method")
    return METHOD_ALIASES.get(method, method)


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
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(candidate, strict=False)


def recover_material_scoring_object(raw):
    """Recover MPS subscores from malformed material_scoring JSON text.

    A few long material outputs are truncated or contain unescaped control
    characters. For benchmark aggregation we only need the explicit 1-5
    subscore fields; MPS is recomputed deterministically later.
    """
    text = str(raw or "")
    if "material_scoring" not in text:
        raise ValueError("not a material_scoring output")

    def score_for(section):
        pattern = r'"' + re.escape(section) + r'"\s*:\s*\{.*?"score"\s*:\s*([1-5])'
        match = re.search(pattern, text, flags=re.S)
        if not match:
            raise ValueError(f"missing recoverable score for {section}")
        return int(match.group(1))

    youngs_score = score_for("youngs_modulus_evaluation")
    poisson_score = score_for("poisson_ratio_evaluation")
    density_score = score_for("density_evaluation")
    weighted_score = 0.4 * youngs_score + 0.2 * poisson_score + 0.4 * density_score
    return {
        "task": "material_scoring",
        "youngs_modulus_evaluation": {
            "score": youngs_score,
            "confidence": "unknown",
            "reason": "Recovered from malformed raw VLM JSON.",
        },
        "poisson_ratio_evaluation": {
            "score": poisson_score,
            "confidence": "unknown",
            "reason": "Recovered from malformed raw VLM JSON.",
        },
        "density_evaluation": {
            "score": density_score,
            "confidence": "unknown",
            "reason": "Recovered from malformed raw VLM JSON.",
        },
        "material_weighted_score": weighted_score,
        "MPS": 25.0 * (weighted_score - 1.0),
        "overall_confidence": "unknown",
        "overall_reason": "Recovered material subscores from malformed raw VLM JSON.",
        "parse_recovered": True,
    }


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def score_1_to_5_to_100(value):
    score = safe_float(value)
    if score is None:
        return None
    return 25.0 * (score - 1.0)


def infer_metric_from_turn(turn_id):
    turn = str(turn_id or "").lower()
    for key, metric in GENERIC_TURN_METRICS.items():
        if key in turn:
            return metric
    return None


def extract_metric(parsed, turn_id):
    task = parsed.get("task") if isinstance(parsed, dict) else None
    metric = TASK_TO_METRIC.get(task)
    value = None
    extras = {}

    if task == "affordance_scoring":
        value = safe_float(parsed.get("APS"))
    elif task == "vaps_scoring":
        aggregates = parsed.get("aggregates") or {}
        value = safe_float(aggregates.get("VAPS"))
        extras["S_prior"] = safe_float(aggregates.get("S_prior"))
        extras["S_reveal"] = safe_float(aggregates.get("S_reveal"))
        extras["S_global"] = safe_float(aggregates.get("S_global"))
    elif task == "dimension_scoring":
        value = safe_float(parsed.get("DQS"))
    elif task == "material_scoring":
        youngs = parsed.get("youngs_modulus_evaluation") or {}
        poisson = parsed.get("poisson_ratio_evaluation") or {}
        density = parsed.get("density_evaluation") or {}
        youngs_raw = safe_float(youngs.get("score"))
        poisson_raw = safe_float(poisson.get("score"))
        density_raw = safe_float(density.get("score"))
        extras["youngs_modulus_score"] = score_1_to_5_to_100(youngs.get("score"))
        extras["poisson_ratio_score"] = score_1_to_5_to_100(poisson.get("score"))
        extras["density_score"] = score_1_to_5_to_100(density.get("score"))
        if youngs_raw is not None and poisson_raw is not None and density_raw is not None:
            weighted_score = 0.4 * youngs_raw + 0.2 * poisson_raw + 0.4 * density_raw
            value = 25.0 * (weighted_score - 1.0)
        else:
            value = safe_float(parsed.get("MPS"))
    elif task == "description_mask_scoring":
        value = safe_float(parsed.get("DCS"))
        extras["alignment_score"] = safe_float(parsed.get("alignment_score"))
        extras["precision_score"] = safe_float(parsed.get("precision_score"))
    elif isinstance(parsed, dict) and "score" in parsed:
        metric = infer_metric_from_turn(turn_id)
        score = safe_float(parsed.get("score"))
        if metric == "DCS" and score is not None and score > 5:
            value = score
        elif score is not None:
            value = 25.0 * (score - 1.0)

    verdict = parsed.get("verdict") if isinstance(parsed, dict) else None
    if verdict is None and task == "vaps_scoring":
        verdict = (parsed.get("aggregates") or {}).get("verdict")
    return metric, value, task, verdict, extras


def context_from_payload(payload, result_path):
    ctx = payload.get("benchmark_context") or {}
    rel = str(payload.get("video_relative_dir") or "")
    parts = [p for p in Path(rel).parts if p not in {".", ""}]

    metric = ctx.get("metric")
    method = ctx.get("method")
    dataset = ctx.get("dataset")
    object_id = ctx.get("object_id") or ctx.get("sample_id") or payload.get("video_id")

    if len(parts) >= 4 and parts[0].lower() in {"kps", "aps", "dqs", "rqs", "mcs", "dcs", "mps"}:
        metric = metric or parts[0]
        method = method or parts[1]
        dataset = dataset or parts[2]
        object_id = object_id or parts[3]
    elif len(parts) >= 3:
        method = method or parts[-3]
        dataset = dataset or parts[-2]
        object_id = object_id or parts[-1]

    return {
        "metric_hint": str(metric).upper() if metric else "",
        "method": canonical_method(method),
        "dataset": dataset or "unknown_dataset",
        "object_id": object_id or result_path.parent.name,
    }


def is_excluded_path(path):
    return any(part in EXCLUDED_DIR_NAMES for part in Path(path).parts)


def iter_result_jsons(path):
    path = Path(path)
    if path.is_file():
        if path.name == "result.json" and not is_excluded_path(path):
            yield path
    elif path.is_dir():
        rg_bin = shutil.which("rg")
        if rg_bin:
            try:
                proc = subprocess.run(
                    [rg_bin, "--files", str(path)],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                candidates = [Path(line) for line in proc.stdout.splitlines() if line.endswith("/result.json")]
                for result_path in sorted(p for p in candidates if not is_excluded_path(p)):
                    yield result_path
                return
            except subprocess.CalledProcessError:
                pass
        yield from sorted(p for p in path.rglob("result.json") if not is_excluded_path(p))
    else:
        raise FileNotFoundError(path)


def parse_results(results_roots):
    if isinstance(results_roots, (str, Path)):
        results_roots = [results_roots]
    rows = []
    errors = []
    for results_root in results_roots:
        for result_path in iter_result_jsons(results_root):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append({"result_json": str(result_path), "error": f"read_json: {exc}"})
                continue
            ctx = context_from_payload(payload, result_path)
            for item in payload.get("results", []):
                turn_id = item.get("turn_id")
                if item.get("error"):
                    errors.append({"result_json": str(result_path), "turn_id": turn_id, "error": item.get("error")})
                    continue
                try:
                    parsed = extract_json_object(item.get("output"))
                except Exception as exc:
                    metric_hint = str(ctx.get("metric_hint") or "").upper()
                    if metric_hint == "MPS" or str(turn_id or "").lower() == "material_scoring":
                        try:
                            parsed = recover_material_scoring_object(item.get("output"))
                        except Exception as recover_exc:
                            errors.append(
                                {
                                    "result_json": str(result_path),
                                    "turn_id": turn_id,
                                    "error": f"parse_output: {exc}",
                                    "recover_error": str(recover_exc),
                                }
                            )
                            continue
                    else:
                        errors.append({"result_json": str(result_path), "turn_id": turn_id, "error": f"parse_output: {exc}"})
                        continue
                metric, value, task, verdict, extras = extract_metric(parsed, turn_id)
                if metric is None and ctx["metric_hint"]:
                    metric = ctx["metric_hint"]
                if metric is None or value is None:
                    continue
                row = {
                    "method": ctx["method"],
                    "dataset": ctx["dataset"],
                    "object_id": ctx["object_id"],
                    "metric": metric,
                    "score": round(float(value), 4),
                    "task": task or "",
                    "verdict": verdict or "",
                    "turn_id": turn_id or "",
                    "result_json": str(result_path),
                    "video_path": payload.get("video_path") or "",
                    "video_paths": json.dumps(payload.get("video_paths") or [], ensure_ascii=False),
                    "paired_image_path": payload.get("paired_image_path") or "",
                    "pair_error": payload.get("pair_error") or "",
                    "_auto_scored": bool(item.get("auto_scored") or payload.get("auto_scored_missing_render_views") or payload.get("auto_scored_missing_video") or payload.get("auto_scored_missing_affordance") or payload.get("auto_scored_missing_description_images") or payload.get("auto_scored_missing_material_videos") or payload.get("auto_scored_invalid_json")),
                    "_result_mtime": result_path.stat().st_mtime,
                }
                for key, extra_value in extras.items():
                    row[key] = extra_value
                rows.append(row)
    return rows, errors


def deduplicate_rows(rows):
    """Keep one score per method/dataset/object/metric.

    Repeated rows happen when a metric is rerun or when a deterministic
    missing-evidence zero backfill intentionally replaces an older scoring
    protocol. Keep the newest result file. If two files have the same mtime,
    prefer a real VLM row over an auto-scored backfill.
    """
    best = {}
    for row in rows:
        key = (row["method"], row["dataset"], row["object_id"], row["metric"])
        rank = (float(row.get("_result_mtime") or 0.0), 0 if row.get("_auto_scored") else 1)
        current = best.get(key)
        if current is None or rank > current[0]:
            best[key] = (rank, row)
    return [item[1] for item in best.values()]


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
        key = (row["method"], row["dataset"], row["metric"])
        groups.setdefault(key, []).append(float(row["score"]))
    out = []
    for (method, dataset, metric), values in sorted(groups.items()):
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) >= 2 else 0.0
        out.append(
            {
                "method": method,
                "dataset": dataset,
                "metric": metric,
                "count": len(values),
                "mean": round(mean, 4),
                "std": round(std, 4),
            }
        )
    return out


SUBMETRIC_FIELDS = {
    "KPS": ["S_prior", "S_reveal", "S_global"],
    "DCS": ["alignment_score", "precision_score"],
    "MPS": ["youngs_modulus_score", "poisson_ratio_score", "density_score"],
}


def summarize_submetrics(rows):
    groups = {}
    for row in rows:
        metric = row["metric"]
        fields = SUBMETRIC_FIELDS.get(metric)
        if not fields:
            continue
        has_any_submetric = any(safe_float(row.get(field)) is not None for field in fields)
        if not has_any_submetric:
            continue
        key = (row["method"], row["dataset"], metric)
        groups.setdefault(key, []).append(row)

    out = []
    for (method, dataset, metric), items in sorted(groups.items()):
        score_values = [float(row["score"]) for row in items if safe_float(row.get("score")) is not None]
        row_out = {
            "method": method,
            "dataset": dataset,
            "metric": metric,
            "count": len(items),
            "score_mean": round(statistics.fmean(score_values), 4) if score_values else "",
            "score_std": round(statistics.stdev(score_values), 4) if len(score_values) >= 2 else 0.0,
        }
        for field in SUBMETRIC_FIELDS[metric]:
            values = [safe_float(item.get(field)) for item in items]
            values = [value for value in values if value is not None]
            row_out[f"{field}_mean"] = round(statistics.fmean(values), 4) if values else ""
            row_out[f"{field}_std"] = round(statistics.stdev(values), 4) if len(values) >= 2 else 0.0 if values else ""
            row_out[f"{field}_count"] = len(values)
        out.append(row_out)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multi.py VLM result JSONs into benchmark CSVs.")
    parser.add_argument(
        "--results-root",
        action="append",
        required=True,
        help="Run directory, result.json, or root to scan. Can be repeated.",
    )
    parser.add_argument(
        "--object-csv",
        default="benchmark/benchmark_results/object_level_scores/object_scores_long.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="benchmark/benchmark_results/dataset_level_scores/dataset_metric_summary.csv",
    )
    parser.add_argument(
        "--submetric-csv",
        default="benchmark/benchmark_results/dataset_level_scores/dataset_submetric_summary.csv",
    )
    parser.add_argument(
        "--errors-jsonl",
        default="benchmark/benchmark_results/logs/aggregate_errors.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, errors = parse_results(args.results_root)
    rows = deduplicate_rows(rows)
    object_fields = [
        "method",
        "dataset",
        "object_id",
        "metric",
        "score",
        "S_prior",
        "S_reveal",
        "S_global",
        "alignment_score",
        "precision_score",
        "youngs_modulus_score",
        "poisson_ratio_score",
        "density_score",
        "task",
        "verdict",
        "turn_id",
        "result_json",
        "video_path",
        "video_paths",
        "paired_image_path",
        "pair_error",
    ]
    write_csv(rows, args.object_csv, object_fields)
    summary_rows = summarize(rows)
    write_csv(summary_rows, args.summary_csv, ["method", "dataset", "metric", "count", "mean", "std"])
    submetric_rows = summarize_submetrics(rows)
    submetric_fields = [
        "method",
        "dataset",
        "metric",
        "count",
        "score_mean",
        "score_std",
        "S_prior_mean",
        "S_prior_std",
        "S_prior_count",
        "S_reveal_mean",
        "S_reveal_std",
        "S_reveal_count",
        "S_global_mean",
        "S_global_std",
        "S_global_count",
        "alignment_score_mean",
        "alignment_score_std",
        "alignment_score_count",
        "precision_score_mean",
        "precision_score_std",
        "precision_score_count",
        "youngs_modulus_score_mean",
        "youngs_modulus_score_std",
        "youngs_modulus_score_count",
        "poisson_ratio_score_mean",
        "poisson_ratio_score_std",
        "poisson_ratio_score_count",
        "density_score_mean",
        "density_score_std",
        "density_score_count",
    ]
    write_csv(submetric_rows, args.submetric_csv, submetric_fields)

    err_path = Path(args.errors_jsonl)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    with err_path.open("w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")

    print(
        f"object_rows={len(rows)} summary_rows={len(summary_rows)} "
        f"submetric_rows={len(submetric_rows)} errors={len(errors)}"
    )
    print(f"object_csv={args.object_csv}")
    print(f"summary_csv={args.summary_csv}")
    print(f"submetric_csv={args.submetric_csv}")
    print(f"errors_jsonl={args.errors_jsonl}")


if __name__ == "__main__":
    main()
