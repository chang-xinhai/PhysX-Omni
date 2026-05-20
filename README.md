<div align="left">
<h1 align="center">PhysX-Omni: Unified Simulation-Ready Physical 3D Generation
for Rigid, Deformable, and Articulated Objects
</h1>
<p align="center"><a href="https://arxiv.org/abs/"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://physx-omni.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=homepage&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/PhysX-Omni/PhysXVerse'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<a href='https://youtu.be/ZCgj4ffz4yk'><img src='https://img.shields.io/youtube/views/ZCgj4ffz4yk'></a>
<div>
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="img/teaser.png">
</div>




## 🏆 News

- We release the code of PhysX-Omni, PhysXVerse, and  PhysX-Bench 🎉

## I. PhysX-Omni

### Installation

1. Clone the repo:

```
git clone --recurse-submodules https://github.com/physx-omni/PhysX-Omni.git
cd PhysX-Omni 
```

2. Create a new conda environment named `physx-anything` and install the dependencies:

```bash
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

**Note**: The detailed usage of `setup.sh` can be found at [TRELLIS](https://github.com/microsoft/TRELLIS)

3. Install the dependencies for Qwen2.5:

```bash
pip install transformers==4.50.0
pip install qwen-vl-utils
pip install 'accelerate>=0.26.0'
```

**Note**: We release the `requirements.txt` file. You can install all dependencies by running:

```bash
conda create -n physx-omni python=3.10
conda activate physx-omni
pip install -r requirements.txt
```

### Training

1. Download PhysX datasets from [PhysXNet](https://huggingface.co/datasets/Caoza/PhysX-3D), [PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility), and [PhysXVerse](https://huggingface.co/datasets/PhysX-Omni/PhysXVerse)

2. Run the preprocessing script for PhysXVerse. 

   ```python
   cd dataset
   python 1voxel_verse.py
   python 2encode_representation_64_finetune
   python 3generate_data_new_64_finetune_rle.py
   ```

   **Note**: Here is a template for you to check the format: [template](https://github.com/ziangcao0312/PhysX-Anything/blob/main/dataset/training_data_template.json).

   **Note**: Preprocess the PhysXNet and PhysX-Mobility follows [PhysX-Anything](https://github.com/ziangcao0312/PhysX-Anything)

3. Render the conditioning images (25 images per object) based on your requirements. 

   For PhysX-Mobility and PhysXVerse, we use [dataset_toolkits/render_cond_mobility.py](https://github.com/ziangcao0312/PhysX-Anything/tree/main/dataset_toolits) to generate the conditioning images. 

   For PhysXNet, please check [PhysX-3D/dataset_toolkits/precess.sh](https://github.com/ziangcao0312/PhysX-3D/blob/main/dataset_toolkits/precess.sh)

4. Set the path in train [configuration](https://github.com/ziangcao0312/PhysX-Omni/blob/main/qwen-vl-finetune/qwenvl/data/__init__.py)

   ```python
   PHYSXNET = {
       "annotation_path": "xx", #json file path
       "data_path": "xx",  # conditioning image path
   }
   
   PHYSXMOBILITY = {
       "annotation_path": "xx", #json file path
       "data_path": "xx",  # conditioning image path
   }
   
   PHYSXVERSE = {
       "annotation_path": "xx", #json file path
       "data_path": "xx",  # conditioning image path
   }
   ```

5. Finetune the model

   ```
   cd qwen-vl-finetune
   sbatch scripts/train_physx.sh
   ```

### Inference

1. Download the pre-train model from [huggingface](https://huggingface.co/PhysX-Omni/PhysX-Omni).

```bash
python download.py
```

2. Run the inference code

```bash
python 1vlm_demo.py            # vlm inference
    
python 2infer_geo.py           # decoder inference

python 3jsongen_update.py      # convert to URDF & XML
```

## II. PhysX-Bench

This repository includes the PhysX-Omni benchmark code under [`benchmark/`](benchmark/).

See [`benchmark/README.md`](benchmark/README.md) for the benchmark file structure, asset generation pipeline, VLM evaluation commands, denominator validation, and aggregation workflow.

For environment setup, see [`benchmark/INSTALL.md`](benchmark/INSTALL.md).

## III. PhysXVerse

For more details about our proposed dataset including dataset structure and annotation, please see this  [PhysXVerse](https://huggingface.co/datasets/PhysX-Omni/PhysXVerse), [PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility) and [PhysXNet](https://huggingface.co/datasets/Caoza/PhysX-3D).

## IV. Other Tools

We provide `convert_objects2scene.py`, which converts individual objects into a simulation-ready scene. In addition, we build a simple scene generation pipeline in `applications_scene` based on existing works.

### Acknowledgement

The data and code is based on [PartNet-mobility](https://sapien.ucsd.edu/browse), [Qwen](https://github.com/QwenLM/Qwen3-VL), [TRELLIS](https://github.com/microsoft/TRELLIS), [Depth-Anything](https://github.com/ByteDance-Seed/depth-anything-3), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [CAST](https://github.com/FishWoWater/CAST). We would like to express our sincere thanks to the contributors.

## :newspaper_roll: License

Distributed under the S-Lab License. See `LICENSE` for more information.

<div align="center">
  <a href="https://info.flagcounter.com/CFxN"><img src="https://s01.flagcounter.com/map/CFxN/size_s/txt_000000/border_CCCCCC/pageviews_0/viewers_0/flags_0/" alt="Flag Counter" border="0"></a>
</div>
