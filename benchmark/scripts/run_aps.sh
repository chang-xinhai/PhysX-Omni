#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

python3 benchmark/code/assets/prepare_affordance_heatmaps.py \
  --physx-result-root "${PHYSX_RESULT_ROOT}" \
  --output-root "${AFFORDANCE_HEATMAP_ROOT}" \
  --methods ${METHODS} \
  --datasets ${DATASETS} \
  --skip-existing

python3 benchmark/code/manifests/build_affordance_manifest.py \
  --physx-result-root "${PHYSX_RESULT_ROOT}" \
  --condition-image-root "${CONDITION_IMAGE_ROOT}" \
  --heatmap-root "${AFFORDANCE_HEATMAP_ROOT}" \
  --output-jsonl "${MANIFEST_ROOT}/aps.jsonl" \
  --output-csv "${MANIFEST_ROOT}/aps.csv" \
  --methods ${METHODS} \
  --datasets ${DATASETS}

run_vlm "${MANIFEST_ROOT}/aps.jsonl" benchmark/prompts/prompts_affordance.yaml
aggregate_and_validate
