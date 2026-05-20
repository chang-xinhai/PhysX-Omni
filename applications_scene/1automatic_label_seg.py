import argparse
import os
import sys
import json
from dataclasses import dataclass, asdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "Grounded-Segment-Anything")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import ipdb
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything.segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor,
)

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS


@dataclass
class CropInfo:
    index: int
    bbox_xyxy: tuple
    bbox_w: int
    bbox_h: int
    bbox_area: int
    square_side_before_expand: int
    final_side: int
    bbox_ratio_in_final: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    bbox_in_final_xyxy: tuple
    label: str = ""


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags, mask_list, box_list, label_list):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])

    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "mask.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    json_data = {
        "tags": tags,
        "mask": [{"value": value, "label": "background"}],
    }

    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]
        json_data["mask"].append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )

    with open(os.path.join(output_dir, "label.json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def save_mask_slices(masks, save_dir, squeeze_channel=True, to_uint8=True):
    ensure_dir(save_dir)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    for i in range(len(masks)):
        mask = masks[i]

        if squeeze_channel and mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if squeeze_channel and mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if to_uint8:
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8) * 255
            else:
                mask = mask.astype(np.float32)
                if mask.max() <= 1.0:
                    mask = mask * 255.0
                mask = np.clip(mask, 0, 255).astype(np.uint8)

        save_path = os.path.join(save_dir, f"{i}.png")
        Image.fromarray(mask).save(save_path)
        print(f"Saved: {save_path}")


def save_masked_image_slices(image_rgb, masks, save_dir, transparent_bg=False):
    ensure_dir(save_dir)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    image_rgb = np.asarray(image_rgb)
    h, w = image_rgb.shape[:2]

    for i in range(len(masks)):
        mask = masks[i]

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if mask.dtype != np.bool_:
            mask = mask > 0.5

        if transparent_bg:
            alpha = (mask.astype(np.uint8) * 255)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = image_rgb
            rgba[..., 3] = alpha
            out = Image.fromarray(rgba, mode="RGBA")
        else:
            masked = np.zeros_like(image_rgb, dtype=np.uint8)
            masked[mask] = image_rgb[mask]
            out = Image.fromarray(masked)

        save_path = os.path.join(save_dir, f"{i}.png")
        out.save(save_path)
        print(f"Saved masked image: {save_path}")


def mask_to_bool(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.dtype == np.bool_:
        return mask
    return mask > 0.5


def compute_tight_bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def center_square_from_bbox(x0, y0, x1, y1):
    bw = x1 - x0
    bh = y1 - y0
    side = max(bw, bh)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    sx0 = int(np.floor(cx - side / 2.0))
    sy0 = int(np.floor(cy - side / 2.0))
    sx1 = sx0 + side
    sy1 = sy0 + side
    return sx0, sy0, sx1, sy1, side


def crop_with_zero_padding(arr, x0, y0, side, fill_value=0):

    H, W = arr.shape[:2]
    x1 = x0 + side
    y1 = y0 + side

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(W, x1)
    src_y1 = min(H, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if arr.ndim == 2:
        out = np.full((side, side), fill_value, dtype=arr.dtype)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    else:
        c = arr.shape[2]
        if np.isscalar(fill_value):
            fill = [fill_value] * c
        else:
            fill = fill_value
        out = np.zeros((side, side, c), dtype=arr.dtype)
        for k in range(c):
            out[..., k] = fill[k]
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    return out, pad_left, pad_top, pad_right, pad_bottom


def make_square_expand_crop_from_mask(
    image_rgb,
    mask_bool,
    target_bbox_ratio=0.75,
    label="",
    index=0,
):

    bbox = compute_tight_bbox_from_mask(mask_bool)
    if bbox is None:
        return None, None, None

    H, W = mask_bool.shape
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    bbox_area = bw * bh

    sq_x0, sq_y0, sq_x1, sq_y1, side0 = center_square_from_bbox(x0, y0, x1, y1)

    required_side = int(np.ceil(np.sqrt(bbox_area / float(target_bbox_ratio))))
    final_side = max(side0, required_side)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    fx0 = int(np.floor(cx - final_side / 2.0))
    fy0 = int(np.floor(cy - final_side / 2.0))

    crop_rgb, pad_left, pad_top, pad_right, pad_bottom = crop_with_zero_padding(
        image_rgb, fx0, fy0, final_side, fill_value=0
    )
    crop_mask, _, _, _, _ = crop_with_zero_padding(
        mask_bool.astype(np.uint8) * 255, fx0, fy0, final_side, fill_value=0
    )

    crop_mask_bool = crop_mask > 0
    alpha = (crop_mask_bool.astype(np.uint8) * 255)

    crop_rgba = np.zeros((final_side, final_side, 4), dtype=np.uint8)
    crop_rgba[..., :3] = crop_rgb
    crop_rgba[..., 3] = alpha

    bbox_in_final_x0 = x0 - fx0
    bbox_in_final_y0 = y0 - fy0
    bbox_in_final_x1 = x1 - fx0
    bbox_in_final_y1 = y1 - fy0

    bbox_ratio_in_final = bbox_area / float(final_side * final_side)

    info = CropInfo(
        index=index,
        bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
        bbox_w=int(bw),
        bbox_h=int(bh),
        bbox_area=int(bbox_area),
        square_side_before_expand=int(side0),
        final_side=int(final_side),
        bbox_ratio_in_final=float(bbox_ratio_in_final),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        pad_right=int(pad_right),
        pad_bottom=int(pad_bottom),
        bbox_in_final_xyxy=(
            int(bbox_in_final_x0),
            int(bbox_in_final_y0),
            int(bbox_in_final_x1),
            int(bbox_in_final_y1),
        ),
        label=label,
    )

    return crop_rgba, crop_mask.astype(np.uint8), info


def save_cropped_rgba_from_masks(
    image_rgb,
    masks,
    save_rgba_dir,
    save_mask_dir,
    save_meta_path,
    labels=None,
    target_bbox_ratio=0.65,
):
    ensure_dir(save_rgba_dir)
    ensure_dir(save_mask_dir)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    image_rgb = np.asarray(image_rgb).astype(np.uint8)
    meta = []

    for i in range(len(masks)):
        label = labels[i] if labels is not None and i < len(labels) else str(i)
        mask_bool = mask_to_bool(masks[i])

        result = make_square_expand_crop_from_mask(
            image_rgb=image_rgb,
            mask_bool=mask_bool,
            target_bbox_ratio=target_bbox_ratio,
            label=label,
            index=i,
        )

        if result[0] is None:
            print(f"[Warning] mask {i} is empty, skip.")
            continue

        crop_rgba, crop_mask, info = result

        rgba_path = os.path.join(save_rgba_dir, f"{i}.png")
        mask_path = os.path.join(save_mask_dir, f"{i}.png")

        crop_rgba_512 = Image.fromarray(crop_rgba, mode="RGBA").resize(
            (512, 512), resample=Image.BILINEAR
        )
        crop_mask_512 = Image.fromarray(crop_mask, mode="L").resize(
            (512, 512), resample=Image.NEAREST
        )

        crop_rgba_512.save(rgba_path)
        crop_mask_512.save(mask_path)

        meta.append(asdict(info))
        print(
            f"Saved crop {i}: side={info.final_side}, "
            f"bbox_ratio={info.bbox_ratio_in_final:.4f}, "
            f"label={info.label}"
        )

    with open(save_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved crop metadata to: {save_meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--input_image",
        type=str,
        default="demo.png",
        help="path to image file",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument(
        "--ram_checkpoint",
        type=str,
        default="Grounded-Segment-Anything/ckpt/ram_swin_large_14m.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--grounded_checkpoint",
        type=str,
        default="Grounded-Segment-Anything/ckpt/groundingdino_swint_ogc.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="Grounded-Segment-Anything/ckpt/sam_vit_h_4b8939.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--sam_hq_checkpoint",
        type=str,
        default=None,
        help="path to sam-hq checkpoint file",
    )
    parser.add_argument(
        "--use_sam_hq",
        action="store_true",
        help="using sam-hq for prediction",
    )
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="outputs",
        help="output directory",
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--target_bbox_ratio",
        type=float,
        default=0.65,
        help="target bbox area / final image area",
    )
    args = parser.parse_args()

    config_file = args.config
    ram_checkpoint = args.ram_checkpoint
    grounded_checkpoint = args.grounded_checkpoint
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    split = args.split
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    target_bbox_ratio = args.target_bbox_ratio

    ensure_dir(output_dir)

    # load image
    image_pil, image_transformed = load_image(image_path)

    # load grounding model
    model = load_model(config_file, grounded_checkpoint, device=device)


    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    
    # load model
    ram_model = ram(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()

    ram_model = ram_model.to(device)
    raw_image = image_pil.resize(
                    (384, 384))
    raw_image  = transform(raw_image).unsqueeze(0).to(device)

    res = inference_ram(raw_image , ram_model)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    tags=res[0].replace(' |', ',')
    tags_chinese=res[1].replace(' |', ',')

    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])
    

    

    # grounding dino
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model,
        image_transformed,
        tags,
        box_threshold,
        text_threshold,
        device=device,
    )

    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    image_rgb = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_rgb.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    save_mask_slices(masks, save_dir=os.path.join(output_dir, "mask_0"))
    save_masked_image_slices(
        image_rgb,
        masks,
        save_dir=os.path.join(output_dir, "masked_image_0"),
        transparent_bg=False,
    )

    save_cropped_rgba_from_masks(
        image_rgb=image_rgb,
        masks=masks,
        save_rgba_dir=os.path.join(output_dir, "crop_masked_images_rgba"),
        save_mask_dir=os.path.join(output_dir, "crop_masks"),
        save_meta_path=os.path.join(output_dir, "crop_meta.json"),
        labels=pred_phrases,
        target_bbox_ratio=target_bbox_ratio,
    )

    # visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        
    for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "automatic_label_output.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
    plt.close()

    save_mask_data(output_dir, tags, masks, boxes_filt, pred_phrases)