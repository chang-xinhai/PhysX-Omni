import argparse
import json
import os

from diffusers import QwenImageEditPipeline
from PIL import Image
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="outputs/crop_masked_images_rgba")
    parser.add_argument("--output_dir", default="outputs/inpainting_0")
    parser.add_argument("--label_json", default="outputs/label.json")
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    label_json = resolve_path(args.label_json)
    os.makedirs(output_dir, exist_ok=True)

    with open(label_json, "r", encoding="utf-8") as file:
        labeldata = json.load(file)

    pipeline = QwenImageEditPipeline.from_pretrained(args.model)
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=None)

    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        label_name = labeldata["mask"][int(os.path.splitext(name)[0]) + 1]["label"]
        prompt = (
            "Given the visible part of an occluded object "
            f"({label_name}), reconstruct the same object as a complete, "
            f"realistic {label_name}"
        )
        image = Image.open(os.path.join(input_dir, name)).convert("RGB")

        print(name, label_name)
        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": torch.manual_seed(args.seed),
            "true_cfg_scale": args.true_cfg_scale,
            "negative_prompt": " ",
            "num_inference_steps": args.num_inference_steps,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
            output.images[0].save(os.path.join(output_dir, name))


if __name__ == "__main__":
    main()
