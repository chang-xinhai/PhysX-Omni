#!/usr/bin/env bash
set -euo pipefail
DEFAULT_METHODS="ours physxanything"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MATERIAL_VIDEO_ROOT="${MATERIAL_VIDEO_ROOT:-${MATERIAL_VIDEO_ROOT_CFG}}"
FLOOR_VIDEO_ROOT="${FLOOR_VIDEO_ROOT:-${MATERIAL_FLOOR_VIDEO_ROOT_CFG}}"
WATER_VIDEO_ROOT="${WATER_VIDEO_ROOT:-${MATERIAL_WATER_VIDEO_ROOT_CFG}}"

if [ "${RUN_WATERTIGHT:-0}" = "1" ]; then
  python3 benchmark/code/render/material_batch/utils/voxel_remesh.py \
    --watertight-batch \
    --output-root "${WATERTIGHT_ROOT}" \
    --max-faces "${MAX_FACES:-3000}" \
    --workers "${WATERTIGHT_WORKERS:-64}" \
    --skip-existing
fi

if [ "${RENDER_MATERIAL:-0}" = "1" ]; then
  PAIRS=()
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      PAIRS+=("${method}-${dataset}")
    done
  done
  IFS=,; PAIR_CSV="${PAIRS[*]}"; unset IFS

  python3 benchmark/code/render/material_batch/floor/multi_gpu_batch.py \
    --config benchmark/code/render/material_batch/floor/config.toml \
    --pairs "${PAIR_CSV}" \
    --gpus "${GPUS_CSV:-${GPUS:-0}}" \
    --resume --skip-existing --skip-failed

  python3 benchmark/code/render/material_batch/water/multi_gpu_batch.py \
    --config benchmark/code/render/material_batch/water/config.toml \
    --pairs "${PAIR_CSV}" \
    --gpus "${GPUS_CSV:-${GPUS:-0}}" \
    --resume --skip-existing
fi

eval "LIMIT_ARGS=($(limit_args))"
python3 benchmark/code/manifests/build_material_manifest.py \
  --physx-result-root "${PHYSX_RESULT_ROOT}" \
  --water-root "${WATER_VIDEO_ROOT}" \
  --floor-root "${FLOOR_VIDEO_ROOT}" \
  --output-jsonl "${MANIFEST_ROOT}/mps.jsonl" \
  --output-csv "${MANIFEST_ROOT}/mps.csv" \
  --methods ${METHODS} \
  --datasets ${DATASETS} \
  "${LIMIT_ARGS[@]}"

run_vlm "${MANIFEST_ROOT}/mps.jsonl" benchmark/prompts/prompts_material.yaml --video-num-frames "${MPS_VIDEO_NUM_FRAMES:-20}"
aggregate_and_validate
