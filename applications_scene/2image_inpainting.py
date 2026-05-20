import os
from PIL import Image
import torch
import json
from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

input_dir='outputs/crop_masked_images_rgba'
output_dir='outputs/inpainting_0'
os.makedirs(os.path.join(output_dir), exist_ok=True)

with open(os.path.join('outputs','label.json'), 'r') as file:
    labeldata = json.load(file)



namelist=os.listdir(input_dir)
for name in namelist:
    label_name=labeldata['mask'][int(name.split('.')[0])+1]['label']

    prompt = "Given the visible part of an occluded object ("+label_name+"), reconstruct the same object as a complete, realistic "+label_name
    image = Image.open(os.path.join(input_dir,name)).convert("RGB")
    
    print(name,label_name)

    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(os.path.join(output_dir,name))
   