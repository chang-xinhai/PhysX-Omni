#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run.sh INPUT [--output outputs] [--model pretrain] [--download]
  ./run.sh --input INPUT [--output outputs] [--model pretrain] [--download]

INPUT can be a single image file or a directory of images. The script runs the
full README inference pipeline through MuJoCo XML and ROS URDF generation:
  python download.py        # optional, enabled by --download or missing model dir
  python 1vlm_demo.py       # VLM inference
  python 2infer_geo.py      # decoder inference
  python 3jsongen_update.py # convert to URDF & XML

Examples:
  ./run.sh demo/microwave_7221_urdf_1024.png
  ./run.sh demo/microwave_7221_urdf_1024.png --output outputs
  ./run.sh --input demo --output outputs
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INPUT=""
OUTPUT="outputs"
MODEL="pretrain"
DOWNLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input|-i)
      INPUT="$2"
      shift 2
      ;;
    --output|-o)
      OUTPUT="$2"
      shift 2
      ;;
    --model|-m)
      MODEL="$2"
      shift 2
      ;;
    --download)
      DOWNLOAD=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$INPUT" ]]; then
        echo "Unexpected extra positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      INPUT="$1"
      shift
      ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Missing INPUT. Pass a single image or an image directory." >&2
  usage >&2
  exit 2
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MICROMAMBA="${MICROMAMBA:-$HOME/.local/bin/micromamba}"
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_vlm() {
  "$MICROMAMBA" run -n physx-vlm "$@"
}

run_omni() {
  "$MICROMAMBA" run -n physx-omni "$@"
}

WORK_INPUT="$(mktemp -d -t physx-input-XXXXXX)"
SELECTED_BASE="$(mktemp -d -t physx-selected-XXXXXX)"
cleanup() {
  rm -rf "$WORK_INPUT" "$SELECTED_BASE"
}
trap cleanup EXIT

echo "[input]  $INPUT"
echo "[output] $OUTPUT"
echo "[model]  $MODEL"
echo "[hf]     $HF_ENDPOINT"
echo "[cuda]   $CUDA_VISIBLE_DEVICES"

echo "[prepare] normalize input image(s) to $WORK_INPUT"
run_vlm python - "$INPUT" "$WORK_INPUT" <<'PY'
import os
import shutil
import sys
from pathlib import Path
from PIL import Image

src = Path(sys.argv[1]).expanduser()
dst = Path(sys.argv[2])
exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

if not src.exists():
    raise SystemExit(f'input does not exist: {src}')

if src.is_file():
    files = [src]
else:
    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts)

if not files:
    raise SystemExit(f'no supported image files found in: {src}')

names = []
for path in files:
    out = dst / f'{path.stem}.png'
    if path.suffix.lower() == '.png':
        shutil.copy2(path, out)
    else:
        Image.open(path).convert('RGB').save(out)
    names.append(path.stem)

print('prepared=' + ','.join(names))
PY

mapfile -t NAMES < <(find "$WORK_INPUT" -maxdepth 1 -type f -name '*.png' -printf '%f\n' | sed 's/\.png$//' | sort)
if [[ "${#NAMES[@]}" -eq 0 ]]; then
  echo "No staged images found." >&2
  exit 1
fi
printf '[samples] %s\n' "${NAMES[*]}"
mkdir -p "$OUTPUT"
printf '[sample output] %s\n' "${NAMES[@]/#/$OUTPUT/}"

if [[ "$DOWNLOAD" == 1 || ! -d "$MODEL" ]]; then
  echo "[1/4] python download.py"
  run_omni python download.py
else
  echo "[1/4] skip download.py (model dir exists; pass --download to refresh)"
fi

echo "[2/4] python 1vlm_demo.py"
run_vlm python 1vlm_demo.py \
  --imagepath "$WORK_INPUT" \
  --modelpath "$MODEL" \
  --savedir "$OUTPUT" \
  --attn "${PHYSX_VLM_ATTN:-sdpa}" \
  --device-map "${PHYSX_VLM_DEVICE_MAP:-cuda}"

for name in "${NAMES[@]}"; do
  sample_dir="$OUTPUT/$name"
  if [[ ! -d "$sample_dir" ]]; then
    echo "Expected VLM output is missing: $sample_dir" >&2
    exit 1
  fi
  ln -s "$(realpath "$sample_dir")" "$SELECTED_BASE/$name"
done

echo "[3/4] python 2infer_geo.py"
run_omni python 2infer_geo.py --outputpath "$SELECTED_BASE"

echo "[4/4] python 3jsongen_update.py"
run_omni python 3jsongen_update.py --basepath "$SELECTED_BASE"

echo "[validate] MuJoCo XML and URDF parse"
run_omni python - "$SELECTED_BASE" <<'PY'
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco

base = Path(sys.argv[1])
count = 0
for name in sorted(os.listdir(base)):
    sample = base / name
    xml_path = sample / 'basic.xml'
    urdf_path = sample / 'basic.urdf'
    if not xml_path.exists() or not urdf_path.exists():
        raise SystemExit(f'missing XML/URDF for {name}: {xml_path}, {urdf_path}')
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    ET.parse(urdf_path)
    print(f'{name}: xml={xml_path.resolve()} urdf={urdf_path.resolve()} bodies={model.nbody} joints={model.njnt} geoms={model.ngeom} meshes={model.nmesh}')
    count += 1
print(f'loaded_xml={count}')
PY
