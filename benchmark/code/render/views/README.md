# Multi-view Render Toolkit

This folder contains the Blender-based renderer used to create the rendered views consumed by RQS, MCS, and DCS.

Run commands from the repository root.

## Files

```text
benchmark/code/render/views/
├── render_adaptive.py
├── run_render.sh
├── config.sh
├── views.json
├── views_fixed_grid.json
├── blender_script/
│   ├── render_mobility.py
│   ├── render_fixed_grid.py
│   └── io_scene_usdz.zip
├── nvdiffrast/
└── utils/
```

## Environment

```bash
export BLENDER_BIN=blender
export BLENDER_DEVICE=GPU
```

If Blender needs a custom runtime library path, set it explicitly:

```bash
export BLENDER_LD_LIBRARY_PATH=<conda_env>/lib
```

## Recommended Command

```bash
python3 benchmark/code/render/views/render_adaptive.py \
  --dataset_roots physx_result/ours_mobility_181500 \
  --output_root benchmark/benchmark_assets/rendered_views/description \
  --gpus 0 1 2 3 \
  --view_mode orbit \
  --num_views 30
```

## View Modes

| Mode | Use | Views |
|---|---|---:|
| `orbit` | benchmark evaluation | usually 30 |
| `random` | diverse training/debug views | configurable |
| `fixed_grid` | fixed camera grid | 24 |

## Output

Each object is written as:

```text
benchmark/benchmark_assets/rendered_views/description/<source_name>/<object_id>/
├── 000.png
├── 001.png
├── ...
└── transforms.json
```

