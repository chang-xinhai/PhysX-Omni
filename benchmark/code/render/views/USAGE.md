# Render Usage

Run from the repository root.

```bash
export BLENDER_BIN=blender
export BLENDER_DEVICE=GPU
```

Render one source:

```bash
python3 benchmark/code/render/views/render_adaptive.py \
  --dataset_roots physx_result/ours_mobility_181500 \
  --output_root benchmark/benchmark_assets/rendered_views/description \
  --gpus 0 1 \
  --view_mode orbit \
  --num_views 30
```

Render several sources:

```bash
python3 benchmark/code/render/views/render_adaptive.py \
  --dataset_roots \
    physx_result/ours_mobility_181500 \
    physx_result/output_physxanything_mobility \
    physx_result/outputs_physxgen_mobility \
  --filter_json benchmark/assets/description/descript_mobility.json \
  --output_root benchmark/benchmark_assets/rendered_views/description \
  --gpus 0 1 2 3 \
  --view_mode orbit \
  --num_views 30
```

Use `benchmark/code/render/views/run_render.sh` only after editing `benchmark/code/render/views/config.sh` or overriding its variables through the environment.

