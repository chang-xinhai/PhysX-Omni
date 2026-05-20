#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-benchmark/configs/paths.example.yaml}"

cfg() {
  local key="$1"
  local default_value="${2:-}"
  python3 - "$CONFIG" "$key" "$default_value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]
value = default
if path.is_file():
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip() == key:
            value = v.strip().strip("'\"")
            break
print(value)
PY
}

PHYSX_RESULT_ROOT="${PHYSX_RESULT_ROOT:-$(cfg physx_result_root physx_result)}"
ASSET_ROOT="${ASSET_ROOT:-$(cfg benchmark_asset_root benchmark/benchmark_assets)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$(cfg benchmark_manifest_root benchmark/benchmark_manifests)}"
RESULT_ROOT="${RESULT_ROOT:-$(cfg benchmark_result_root benchmark/benchmark_results)}"
MODEL_ID="${MODEL_ID:-$(cfg vlm_model_path Qwen/Qwen3.5-122B-A10B)}"
CONDITION_IMAGE_ROOT="${CONDITION_IMAGE_ROOT:-$(cfg condition_image_root "${ASSET_ROOT}/condition_images")}"
RENDERED_VIEW_ROOT="${RENDERED_VIEW_ROOT:-$(cfg rendered_view_root "${ASSET_ROOT}/rendered_views/description")}"
AFFORDANCE_HEATMAP_ROOT="${AFFORDANCE_HEATMAP_ROOT:-$(cfg affordance_heatmap_root "${ASSET_ROOT}/affordance_heatmaps")}"
KINEMATIC_VIDEO_ROOT="${KINEMATIC_VIDEO_ROOT:-$(cfg kinematic_video_root "${ASSET_ROOT}/kinematic_videos")}"
MATERIAL_VIDEO_ROOT_CFG="$(cfg material_video_root "${ASSET_ROOT}/material_videos")"
MATERIAL_FLOOR_VIDEO_ROOT_CFG="$(cfg material_floor_video_root "${ASSET_ROOT}/material_videos_v2/floor")"
MATERIAL_WATER_VIDEO_ROOT_CFG="$(cfg material_water_video_root "${ASSET_ROOT}/material_videos/water")"
DESCRIPTION_ROOT="${DESCRIPTION_ROOT:-$(cfg description_metadata_root benchmark/assets/description)}"
MATERIAL_JSON_ROOT="${MATERIAL_JSON_ROOT:-$(cfg material_metric_json_root "${PHYSX_RESULT_ROOT}/material_metric_json/origin")}"
WATERTIGHT_ROOT="${WATERTIGHT_ROOT:-$(cfg watertight_mesh_root "${PHYSX_RESULT_ROOT}/watertightFix_max3000")}"

METHODS="${METHODS:-${DEFAULT_METHODS:-ours physxanything physxgen}}"
DATASETS="${DATASETS:-${DEFAULT_DATASETS:-mobility verse inthewild}}"
RUN_VLM="${RUN_VLM:-1}"
if [ "${RUN_VLM}" = "0" ]; then
  RUN_AGGREGATE="${RUN_AGGREGATE:-0}"
  RUN_VALIDATE="${RUN_VALIDATE:-0}"
else
  RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
  RUN_VALIDATE="${RUN_VALIDATE:-1}"
fi
LIMIT="${LIMIT:-}"

mkdir -p "${ASSET_ROOT}" "${MANIFEST_ROOT}" "${RESULT_ROOT}" benchmark/logs

source_folder() {
  local method="$1"
  local dataset="$2"
  case "${method}" in
    ours) echo "ours_${dataset}_181500" ;;
    physxanything|physanything) echo "output_physxanything_${dataset}" ;;
    physxgen|physgen) echo "outputs_physxgen_${dataset}" ;;
    articulateanything) echo "output_articulateanything_${dataset}" ;;
    *) echo "${method}_${dataset}" ;;
  esac
}

source_args_render() {
  local args=()
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      local folder
      folder="$(source_folder "${method}" "${dataset}")"
      args+=(--source "${folder}:${method}:${dataset}")
    done
  done
  printf '%q ' "${args[@]}"
}

source_args_description() {
  local args=()
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      local folder
      folder="$(source_folder "${method}" "${dataset}")"
      args+=(--source "${folder}:${folder}:${method}:${dataset}")
    done
  done
  printf '%q ' "${args[@]}"
}

limit_args() {
  if [ -n "${LIMIT}" ]; then
    printf '%q ' --limit "${LIMIT}"
  fi
}

run_vlm() {
  local manifest="$1"
  local prompt="$2"
  shift 2
  if [ "${RUN_VLM}" = "0" ]; then
    echo "[skip] RUN_VLM=0: ${manifest}"
    return 0
  fi
  python3 benchmark/code/vlm/multi.py \
    --model-id "${MODEL_ID}" \
    --pairs-manifest "${manifest}" \
    --prompts-file "${prompt}" \
    --output-root "${RESULT_ROOT}/raw_vlm_outputs" \
    "$@"
}

aggregate_results() {
  if [ "${RUN_AGGREGATE}" = "0" ]; then
    echo "[skip] RUN_AGGREGATE=0"
    return 0
  fi
  if ! find "${RESULT_ROOT}/raw_vlm_outputs" -path '*/result.json' -type f -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] no raw result.json files under ${RESULT_ROOT}/raw_vlm_outputs"
    return 0
  fi
  python3 benchmark/code/aggregation/aggregate_vlm_results.py \
    --results-root "${RESULT_ROOT}/raw_vlm_outputs" \
    --object-csv "${RESULT_ROOT}/object_level_scores/object_scores_long.csv" \
    --summary-csv "${RESULT_ROOT}/dataset_level_scores/dataset_metric_summary.csv" \
    --submetric-csv "${RESULT_ROOT}/dataset_level_scores/dataset_submetric_summary.csv" \
    --errors-jsonl "${RESULT_ROOT}/logs/aggregate_errors.jsonl"
}

validate_denominators() {
  if [ "${RUN_VALIDATE}" = "0" ]; then
    echo "[skip] RUN_VALIDATE=0"
    return 0
  fi
  python3 benchmark/code/validation/validate_denominators.py \
    --manifest-root "${MANIFEST_ROOT}" \
    --results-root "${RESULT_ROOT}/raw_vlm_outputs" \
    --object-csv "${RESULT_ROOT}/object_level_scores/object_scores_long.csv" \
    --summary-csv "${RESULT_ROOT}/dataset_level_scores/dataset_metric_summary.csv" \
    --output-csv "${RESULT_ROOT}/logs/denominator_validation.csv" \
    --errors-jsonl "${RESULT_ROOT}/logs/denominator_validation_errors.jsonl"
}

aggregate_and_validate() {
  aggregate_results
  validate_denominators
}
