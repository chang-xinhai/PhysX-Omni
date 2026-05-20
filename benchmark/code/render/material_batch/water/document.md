# Water Material Video Rendering

This renderer creates water-entry videos for MPS. It expects watertight,
face-limited meshes under:

```text
physx_result/watertightFix_max3000/<source_folder>/<object_id>/objs/...
```

and material metric JSON files under:

```text
physx_result/material_metric_json/origin/<source_folder>/<object_id>.json
```

The default output folder is:

```text
benchmark/benchmark_assets/material_videos/water
```

Prepare watertight meshes first:

```bash
python3 benchmark/code/render/material_batch/utils/voxel_remesh.py \
  --watertight-batch \
  --output-root physx_result/watertightFix_max3000 \
  --max-faces 3000 \
  --workers 64 \
  --skip-existing
```

Then render one or more method/dataset pairs:

```bash
python3 benchmark/code/render/material_batch/water/multi_gpu_batch.py \
  --config benchmark/code/render/material_batch/water/config.toml \
  --pairs ours-mobility,physxanything-mobility \
  --gpus 0,1,2,3 \
  --resume --skip-existing
```
