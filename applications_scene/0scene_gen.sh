#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$APP_DIR/.." && pwd)"
MICROMAMBA="${MICROMAMBA:-$HOME/.local/bin/micromamba}"
VLM_SITE="$("$MICROMAMBA" run -n physx-vlm python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="$VLM_SITE/torch/lib:$VLM_SITE/nvidia/cublas/lib:$VLM_SITE/nvidia/cuda_runtime/lib:$VLM_SITE/nvidia/cudnn/lib:$VLM_SITE/nvidia/cusparse/lib:$VLM_SITE/nvidia/cusolver/lib:$VLM_SITE/nvidia/nccl/lib:$VLM_SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

INPUT="${1:-$ROOT_DIR/demo/microwave_7221_urdf_1024.png}"
PROMPT="${2:-microwave}"
OUTPUT="${3:-$APP_DIR/outputs_example}"

"$MICROMAMBA" run -n scene env \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  python "$APP_DIR/1automatic_label_seg.py" \
    --input_image "$INPUT" \
    --output_dir "$OUTPUT" \
    --text_prompt "$PROMPT" \
    --box_threshold 0.20 \
    --text_threshold 0.15 \
    --target_bbox_ratio 0.65 \
    --device cuda

echo "Scene preprocessing output: $OUTPUT"
