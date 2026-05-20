conda activate scene
python 1automatic_label_seg.py

conda activate qwen_image
python 2image_inpainting.py
python 3vlm_filter.py


conda activate scene
python 4layout_bbox.py




