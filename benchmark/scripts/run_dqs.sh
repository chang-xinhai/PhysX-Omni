#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

python3 benchmark/code/manifests/build_dimension_manifest.py \
  --physx-result-root "${PHYSX_RESULT_ROOT}" \
  --condition-image-root "${CONDITION_IMAGE_ROOT}" \
  --output-jsonl "${MANIFEST_ROOT}/dqs.jsonl" \
  --output-csv "${MANIFEST_ROOT}/dqs.csv" \
  --methods ${METHODS} \
  --datasets ${DATASETS}

python3 benchmark/code/manifests/build_dimension_manifest.py \
  --physx-result-root "${PHYSX_RESULT_ROOT}" \
  --condition-image-root "${CONDITION_IMAGE_ROOT}" \
  --output-jsonl "${MANIFEST_ROOT}/dqs_prior.jsonl" \
  --output-csv "${MANIFEST_ROOT}/dqs_prior.csv" \
  --methods ${METHODS} \
  --datasets ${DATASETS} \
  --unique-priors

DQS_MANIFEST="${MANIFEST_ROOT}/dqs.jsonl"
DQS_PRIOR_MANIFEST="${MANIFEST_ROOT}/dqs_prior.jsonl"

run_vlm "${DQS_PRIOR_MANIFEST}" benchmark/prompts/prompts_dimension.yaml

if [ "${RUN_VLM}" != "0" ]; then
  python3 benchmark/code/scoring/score_dimension_results.py \
    --manifest "${DQS_MANIFEST}" \
    --prior-results-root "${RESULT_ROOT}/raw_vlm_outputs" \
    --output-root "${RESULT_ROOT}/raw_vlm_outputs/dimension_scoring" \
    --object-csv "${RESULT_ROOT}/object_level_scores/dimension_scores.csv" \
    --summary-csv "${RESULT_ROOT}/dataset_level_scores/dimension_metric_summary.csv" \
    --errors-jsonl "${RESULT_ROOT}/logs/dimension_scoring_errors.jsonl"
fi

aggregate_and_validate
