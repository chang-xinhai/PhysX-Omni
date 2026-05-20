# Benchmark Installation

The benchmark has four execution layers:

1. lightweight manifest / aggregation scripts;
2. VLM inference with PyTorch and Transformers;
3. Blender multi-view rendering for RQS, MCS, and DCS;
4. MuJoCo and Genesis rendering/simulation for KPS and MPS.

Install only the layers you need. For example, denominator validation and
manifest building use only the Python standard library plus small image
dependencies, while VLM and material simulations need GPU runtimes.

## System Requirements

Recommended:

- Linux with NVIDIA driver >= 535;
- CUDA 12.x runtime compatible with your PyTorch build;
- Python 3.11;
- `ffmpeg` available on `PATH`;
- Blender 3.6 LTS or newer for rendered view generation;
- EGL/OpenGL headless libraries for MuJoCo / Genesis rendering.

Ubuntu packages commonly needed on headless servers:

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg git git-lfs wget unzip \
  libegl1 libgl1 libglvnd0 libglx0 libopengl0 \
  libx11-6 libxext6 libxrender1 libxrandr2 libxi6 libxcursor1 \
  libxinerama1 libxxf86vm1 libsm6 libice6
```

## Conda Environment

```bash
conda env create -f benchmark/environment.yml
conda activate physx-omni-benchmark
```

If `genesis-world` is installed from a local checkout instead of PyPI, install it
after activating the environment:

```bash
pip install -e external/genesis-world
```

## Blender

Install Blender separately. Either make `blender` available on `PATH`, or export:

```bash
export BLENDER_BIN=external/blender/blender
export BLENDER_DEVICE=GPU
```

If Blender needs libraries from a conda environment:

```bash
export BLENDER_LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
```

## Headless Rendering

For MuJoCo / Genesis on a server without an X display:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYGLET_HEADLESS=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Quick checks:

```bash
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

python3 - <<'PY'
import mujoco, imageio, numpy
print("mujoco ok")
PY

python3 - <<'PY'
import trimesh
print("trimesh ok")
PY
```

For MPS only:

```bash
python3 - <<'PY'
import genesis
print("genesis ok")
PY
```

For VLM inference, make sure the model is available through Hugging Face or a
local cache:

```bash
export HF_HOME=hf_cache
```

Then pass the model id or local model path to scripts with:

```bash
MODEL_ID=Qwen/Qwen3.5-122B-A10B
```
