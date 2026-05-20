#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for dataset in ${DATASETS}; do
  input_dir="${PHYSX_RESULT_ROOT}/demo_${dataset}"
  if [ ! -d "${input_dir}" ]; then
    echo "[warn] missing demo image directory: ${input_dir}"
    continue
  fi
  python3 benchmark/code/assets/prepare_demo_condition_images.py \
    --input-dir "${input_dir}" \
    --dataset "${dataset}" \
    --output-root "${CONDITION_IMAGE_ROOT}" \
    --symlink --skip-existing
done
