#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ "${RUN_RENDER:-0}" = "1" ]; then
  ROOTS=()
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      folder="$(source_folder "${method}" "${dataset}")"
      [ -d "${PHYSX_RESULT_ROOT}/${folder}" ] && ROOTS+=("${PHYSX_RESULT_ROOT}/${folder}")
    done
  done
  if [ "${#ROOTS[@]}" -gt 0 ]; then
    python3 benchmark/code/render/views/render_adaptive.py \
      --dataset_roots "${ROOTS[@]}" \
      --output_root "${RENDERED_VIEW_ROOT}" \
      --gpus ${GPUS:-0} \
      --view_mode orbit \
      --num_views "${NUM_RENDER_VIEWS:-30}"
  fi
fi

eval "SOURCE_ARGS=($(source_args_render))"
eval "LIMIT_ARGS=($(limit_args))"
python3 benchmark/code/manifests/build_render_view_manifest.py \
  --root "${RENDERED_VIEW_ROOT}" \
  --metric rqs \
  "${SOURCE_ARGS[@]}" \
  --output-jsonl "${MANIFEST_ROOT}/rqs.jsonl" \
  --output-csv "${MANIFEST_ROOT}/rqs.csv" \
  "${LIMIT_ARGS[@]}"

run_vlm "${MANIFEST_ROOT}/rqs.jsonl" benchmark/prompts/prompts_quality.yaml
aggregate_and_validate
