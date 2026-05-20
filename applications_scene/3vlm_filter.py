from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import ipdb
import numpy as np
from PIL import Image
import trimesh
import logging
import argparse
import re

import json




def generate_save(model,messages,save_dir,save_name='test',save=True):


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    #ipdb.set_trace()

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)


    generated_ids = model.generate(**inputs, do_sample=True,max_length=5000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if save:
        with open(os.path.join(save_dir,save_name+'.txt'),'w') as file:
            file.write( output_text[0])
    return output_text[0]


if __name__ == '__main__':


    imagepath='./outputs/inpainting_0'
    savepath='outputs/filter'
    os.makedirs(os.path.join(savepath), exist_ok=True)
    with open(os.path.join('outputs','label.json'), 'r') as file:
        labeldata = json.load(file)

    modelpath="Qwen/Qwen2.5-VL-7B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                modelpath,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
    min_pixels = 65536
    max_pixels = 262144

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor.image_processor.min_pixels=min_pixels
    processor.image_processor.max_pixels=max_pixels
    processor.image_processor.size["shortest_edge"]=min_pixels
    processor.image_processor.size["longest_edge"]=max_pixels
    

    namelist=os.listdir(imagepath)

    for name in namelist:
        label_name=labeldata['mask'][int(name.split('.')[0])+1]['label']
        
        basicqu="Analyze the image of "+label_name+". Check whether the image contains only this single object and it is not background name such as sky, floor, and wall. If yes, output 1. If no, output 0. Do not output any explanation."


        im_resized = Image.open(os.path.join(imagepath,name))

        messages = [

            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": im_resized.convert("RGB"),
                    },
                    {"type": "text", "text": basicqu},
                ],
            }
        ]
        basicoutput=generate_save(model,messages,savepath,name.split('.')[0])
        

