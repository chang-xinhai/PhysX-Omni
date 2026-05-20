### Installation

1. Depth-Anything:

```
conda create -n scene python=3.10
pip install xformers torch\>=2 torchvision
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL
```

2. Grounded-Segment-Anything

```bash
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch]
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

**Note**: We release the `requirements.txt` file. 

3. Qwen-Image

```bash
conda activate qwen_image
pip install git+https://github.com/huggingface/diffusers
pip install transformers>=4.51.3
```



### Inference

1. Download the pre-train model.

```bash
cd Grounded-Segment-Anything

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

2. Run the inference code

```bash
conda activate scene
python 1automatic_label_seg.py

conda activate qwen_image
python 2image_inpainting.py
python 3vlm_filter.py

conda activate scene
python 4layout_bbox.py
```



