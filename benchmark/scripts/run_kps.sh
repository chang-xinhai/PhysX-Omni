#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

KPS_VIDEO_ROOT="${KPS_VIDEO_ROOT:-${KINEMATIC_VIDEO_ROOT}}"
RENDER_KPS="${RENDER_KPS:-0}"

python3 benchmark/code/manifests/build_kinematic_manifest.py \
  --physx-result-root "${PHYSX_RESULT_ROOT}" \
  --condition-image-root "${CONDITION_IMAGE_ROOT}" \
  --video-root "${KPS_VIDEO_ROOT}" \
  --output-jsonl "${MANIFEST_ROOT}/kps_source.jsonl" \
  --output-csv "${MANIFEST_ROOT}/kps_source.csv" \
  --methods ${METHODS} \
  --datasets ${DATASETS}

if [ "${RENDER_KPS}" = "1" ]; then
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      folder="$(source_folder "${method}" "${dataset}")"
      input_dir="${PHYSX_RESULT_ROOT}/${folder}"
      [ -d "${input_dir}" ] || continue
      output_dir="${KPS_VIDEO_ROOT}/${method}/${dataset}"
      case "${method}" in
        ours|physxanything|physanything)
          python3 benchmark/code/render/kinematic/kinematic_articulation_demo.py \
            --batch-input-dir "${input_dir}" \
            --batch-xml-name basic.xml \
            --output-root "${output_dir}" \
            --input-root "${input_dir}" \
            --return-mode hold \
            --include-root-free \
            --scene-preset ours_xml \
            --view 135,-18 \
            --view 45,-18 \
            --skip-existing
          ;;
        physxgen|physgen)
          python3 benchmark/code/render/kinematic/kinematic_articulation_demo.py \
            --batch-input-dir "${input_dir}" \
            --batch-xml-name mesh/basic.urdf \
            --output-root "${output_dir}" \
            --input-root "${input_dir}" \
            --drop-xml-parent-levels 1 \
            --return-mode hold \
            --include-root-free \
            --urdf-visual-only \
            --asset-dir-from-xml-parent \
            --scene-preset ours_xml \
            --view 135,-18 \
            --view 45,-18 \
            --skip-existing
          ;;
        articulateanything)
          python3 benchmark/code/render/kinematic/kinematic_articulation_demo.py \
            --batch-input-dir "${input_dir}" \
            --batch-xml-name joint_actor/iter_0/seed_0/mobility.urdf \
            --output-root "${output_dir}" \
            --input-root "${input_dir}" \
            --drop-xml-parent-levels 3 \
            --return-mode hold \
            --include-root-free \
            --urdf-visual-only \
            --asset-dir-from-sample-objs \
            --scene-preset ours_xml \
            --view 135,-18 \
            --view 45,-18 \
            --skip-existing
          ;;
      esac
    done
  done
fi

python3 benchmark/code/manifests/build_kinematic_manifest.py \
  --physx-result-root "${PHYSX_RESULT_ROOT}" \
  --condition-image-root "${CONDITION_IMAGE_ROOT}" \
  --video-root "${KPS_VIDEO_ROOT}" \
  --output-jsonl "${MANIFEST_ROOT}/kps.jsonl" \
  --output-csv "${MANIFEST_ROOT}/kps.csv" \
  --methods ${METHODS} \
  --datasets ${DATASETS}

run_vlm "${MANIFEST_ROOT}/kps.jsonl" benchmark/prompts/prompts_vaps_english.yaml
aggregate_and_validate
