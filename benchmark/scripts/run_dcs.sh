#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

eval "SOURCE_ARGS=($(source_args_description))"
eval "LIMIT_ARGS=($(limit_args))"
python3 benchmark/code/manifests/build_description_mask_manifest.py \
  --render-root "${RENDERED_VIEW_ROOT}" \
  --result-root "${PHYSX_RESULT_ROOT}" \
  --description-root "${DESCRIPTION_ROOT}" \
  "${SOURCE_ARGS[@]}" \
  --output-jsonl "${MANIFEST_ROOT}/dcs.jsonl" \
  --output-csv "${MANIFEST_ROOT}/dcs.csv" \
  --missing-jsonl "${MANIFEST_ROOT}/dcs_missing.jsonl" \
  "${LIMIT_ARGS[@]}"

run_vlm "${MANIFEST_ROOT}/dcs.jsonl" benchmark/prompts/prompts_description.yaml
aggregate_and_validate
