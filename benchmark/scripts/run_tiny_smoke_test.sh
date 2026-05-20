#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
cd "${ROOT}"

TINY_ROOT="${TINY_ROOT:-benchmark/tiny_example/generated}"
RESULT_ROOT="${TINY_ROOT}/benchmark_results"
MANIFEST_ROOT="${TINY_ROOT}/benchmark_manifests"
RENDER_ROOT="${TINY_ROOT}/rendered_views"

rm -rf "${TINY_ROOT}"
mkdir -p \
  "${RENDER_ROOT}/tiny_ours/0001" \
  "${MANIFEST_ROOT}" \
  "${RESULT_ROOT}/raw_vlm_outputs/tiny_run/rqs/ours/mobility/0001"

python3 - <<'PY'
from pathlib import Path
import base64

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
root = Path("benchmark/tiny_example/generated/rendered_views/tiny_ours/0001")
for idx in range(4):
    (root / f"{idx:03d}.png").write_bytes(png)
PY

python3 benchmark/code/manifests/build_render_view_manifest.py \
  --root "${RENDER_ROOT}" \
  --metric rqs \
  --source tiny_ours:ours:mobility \
  --output-jsonl "${MANIFEST_ROOT}/rqs.jsonl" \
  --output-csv "${MANIFEST_ROOT}/rqs.csv"

python3 - <<'PY'
from pathlib import Path
import json

manifest = json.loads(Path("benchmark/tiny_example/generated/benchmark_manifests/rqs.jsonl").read_text().splitlines()[0])
out = Path("benchmark/tiny_example/generated/benchmark_results/raw_vlm_outputs/tiny_run/rqs/ours/mobility/0001/result.json")
payload = {
    "run_id": "tiny_run",
    "video_path": None,
    "video_id": "0001",
    "video_relative_dir": "rqs/ours/mobility/0001",
    "paired_image_path": "",
    "benchmark_context": manifest,
    "sampling": {"mode": "tiny_smoke"},
    "turns_template": [],
    "results": [
        {
            "turn_id": "render_quality",
            "turn_index": 0,
            "prompt_ref_id": "tiny_fake_rqs",
            "input_modalities": {"image": True, "video": False},
            "output": json.dumps({"score": 5, "reason": "tiny smoke test"}, ensure_ascii=False),
            "error": None,
            "elapsed_sec": 0.0,
        }
    ],
    "pair_error": None,
    "elapsed_sec": 0.0,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

python3 benchmark/code/aggregation/aggregate_vlm_results.py \
  --results-root "${RESULT_ROOT}/raw_vlm_outputs" \
  --object-csv "${RESULT_ROOT}/object_scores.csv" \
  --summary-csv "${RESULT_ROOT}/summary.csv" \
  --submetric-csv "${RESULT_ROOT}/submetric_summary.csv" \
  --errors-jsonl "${RESULT_ROOT}/aggregate_errors.jsonl"

python3 benchmark/code/validation/validate_denominators.py \
  --manifest "${MANIFEST_ROOT}/rqs.jsonl" \
  --manifest-root "" \
  --results-root "${RESULT_ROOT}/raw_vlm_outputs" \
  --object-csv "${RESULT_ROOT}/object_scores.csv" \
  --summary-csv "${RESULT_ROOT}/summary.csv" \
  --output-csv "${RESULT_ROOT}/denominator_validation.csv" \
  --errors-jsonl "${RESULT_ROOT}/denominator_validation_errors.jsonl" \
  --strict

python3 - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("benchmark/tiny_example/generated/benchmark_results/summary.csv").open()))
assert len(rows) == 1, rows
row = rows[0]
assert row["method"] == "ours"
assert row["dataset"] == "mobility"
assert row["metric"] == "RQS"
assert int(row["count"]) == 1
assert abs(float(row["mean"]) - 100.0) < 1e-6
print("[tiny-smoke] ok: aggregation and denominator validation passed")
PY

echo "[tiny-smoke] outputs:"
echo "  ${RESULT_ROOT}/summary.csv"
echo "  ${RESULT_ROOT}/denominator_validation.csv"
