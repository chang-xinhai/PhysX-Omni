#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
cd "$ROOT"

INTERVAL_SEC="${LIVE_AGG_INTERVAL_SEC:-600}"
RESULTS_ROOT="${LIVE_AGG_RESULTS_ROOT:-benchmark/benchmark_results/raw_vlm_outputs}"
RESULTS_ROOTS="${LIVE_AGG_RESULTS_ROOTS:-${RESULTS_ROOT}}"
OBJECT_CSV="${LIVE_AGG_OBJECT_CSV:-benchmark/benchmark_results/object_level_scores/object_scores_live.csv}"
SUMMARY_CSV="${LIVE_AGG_SUMMARY_CSV:-benchmark/benchmark_results/dataset_level_scores/dataset_metric_summary_live.csv}"
SUBMETRIC_CSV="${LIVE_AGG_SUBMETRIC_CSV:-benchmark/benchmark_results/dataset_level_scores/dataset_submetric_summary_live.csv}"
ERRORS_JSONL="${LIVE_AGG_ERRORS_JSONL:-benchmark/benchmark_results/logs/aggregate_errors_live.jsonl}"

mkdir -p \
  "$(dirname "${OBJECT_CSV}")" \
  "$(dirname "${SUMMARY_CSV}")" \
  "$(dirname "${SUBMETRIC_CSV}")" \
  "$(dirname "${ERRORS_JSONL}")" \
  benchmark/logs

echo "[live-aggregate] started $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "[live-aggregate] interval_sec=${INTERVAL_SEC}"
echo "[live-aggregate] results_roots=${RESULTS_ROOTS}"
echo "[live-aggregate] object_csv=${OBJECT_CSV}"
echo "[live-aggregate] summary_csv=${SUMMARY_CSV}"
echo "[live-aggregate] submetric_csv=${SUBMETRIC_CSV}"
echo "[live-aggregate] errors_jsonl=${ERRORS_JSONL}"

while true; do
  echo "[live-aggregate] tick $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  result_count=0
  aggregate_args=()
  for root in ${RESULTS_ROOTS}; do
    aggregate_args+=(--results-root "${root}")
    if [ -e "${root}" ]; then
      count_for_root="$(find "${root}" -path '*/result.json' -type f 2>/dev/null | wc -l | tr -d ' ')"
      result_count=$((result_count + count_for_root))
    fi
  done
  echo "[live-aggregate] raw_result_count=${result_count}"

  if [ "${result_count}" -gt 0 ]; then
    tmp_object="${OBJECT_CSV}.tmp"
    tmp_summary="${SUMMARY_CSV}.tmp"
    tmp_submetric="${SUBMETRIC_CSV}.tmp"
    tmp_errors="${ERRORS_JSONL}.tmp"

    if python3 benchmark/code/aggregation/aggregate_vlm_results.py \
      "${aggregate_args[@]}" \
      --object-csv "${tmp_object}" \
      --summary-csv "${tmp_summary}" \
      --submetric-csv "${tmp_submetric}" \
      --errors-jsonl "${tmp_errors}"; then
      mv "${tmp_object}" "${OBJECT_CSV}"
      mv "${tmp_summary}" "${SUMMARY_CSV}"
      mv "${tmp_submetric}" "${SUBMETRIC_CSV}"
      mv "${tmp_errors}" "${ERRORS_JSONL}"
      echo "[live-aggregate] updated live outputs"
    else
      echo "[live-aggregate] aggregate failed; keeping previous live outputs"
      rm -f "${tmp_object}" "${tmp_summary}" "${tmp_submetric}" "${tmp_errors}"
    fi
  else
    echo "[live-aggregate] no raw result JSONs yet; skipping"
  fi

  sleep "${INTERVAL_SEC}"
done
