from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
import os
import numpy as np
from PIL import Image
import trimesh
from rembg import remove
import logging
import argparse
import re
from typing import List, Dict, Tuple, Optional
import shutil

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ----------------------------
# Compact run list: "start[ length];start[ length];..."
# omit length if == 1
# ----------------------------
def runs_to_compact_str(runs: np.ndarray) -> str:
    runs = np.asarray(runs)
    if runs.size == 0:
        return ""
    if runs.ndim != 2 or runs.shape[1] != 2:
        raise ValueError(f"runs must be (K,2), got {runs.shape}")
    runs = runs.astype(np.int64, copy=False)

    items = []
    for s, L in runs:
        s = int(s); L = int(L)
        if L == 1:
            items.append(f"{s}")
        else:
            items.append(f"{s} {L}")
    return ";".join(items)

def compact_str_to_runs(s: str) -> np.ndarray:
    s = (s or "").strip()
    if not s:
        return np.zeros((0, 2), dtype=np.int64)

    rows: List[Tuple[int, int]] = []
    # allow both ';' and ',' as separators
    for it in re.split(r"[;,]", s):
        it = it.strip()
        if not it:
            continue
        parts = it.split()
        if len(parts) == 1:
            start = int(parts[0])
            length = 1
        elif len(parts) == 2:
            start = int(parts[0])
            length = int(parts[1])
        else:
            raise ValueError(f"bad run item: '{it}'")
        rows.append((start, length))

    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)

def _runs_set(runs: np.ndarray) -> set:
    runs = np.asarray(runs)
    if runs.size == 0:
        return set()
    if runs.ndim != 2 or runs.shape[1] != 2:
        raise ValueError(f"runs must be (K,2), got {runs.shape}")
    return set(map(tuple, runs.astype(np.int64, copy=False)))

def runs_similarity(r1: np.ndarray, r2: np.ndarray) -> float:
    s1, s2 = _runs_set(r1), _runs_set(r2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    denom = max(len(s1), len(s2))
    return inter / denom if denom > 0 else 1.0

def _int_to_label(i: int) -> str:
    # a,b,c,...,z,aa,ab,...
    s = ""
    i += 1
    while i > 0:
        i -= 1
        s = chr(ord('a') + (i % 26)) + s
        i //= 26
    return s

# ----------------------------
# Decode lossless template format
# ----------------------------
_LAYER_RE = re.compile(
    r"^\s*(\d+)\s*[：:]\s*layer\s+([a-z]+)"
    r"(?:\s*\+\[(.*?)\])?"
    r"(?:\s*-\[(.*?)\])?\s*$",
    re.IGNORECASE
)
_TEMPLATE_RE = re.compile(r"^\s*([a-z]+)\s*[：:]\s*(.*?)\s*$", re.IGNORECASE)





######################

def _clean_line_soft(line: str) -> str:
    """
    Soft sanitize a whole line:
    keep only a safe charset to avoid weird symbols breaking regex parsing.
    """
    # allow: letters/digits/space, colon, full-width colon, brackets, plus/minus, semicolon, comma, dot
    # dot is harmless; you can remove it if you never use it
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t:：[];+-,.")
    return "".join(ch for ch in line if ch in allowed)

def _clean_runs_payload(payload: str) -> str:
    """
    Sanitize run-list payload inside template / +[...] / -[...]
    We only keep digits, spaces, ';' and ','.
    Any other char (including '-') is dropped.
    """
    allowed = set("0123456789 \t;,\n")
    payload = "".join(ch for ch in payload if ch in allowed)
    # normalize separators/spaces a bit
    payload = payload.replace("\t", " ").strip()
    payload = re.sub(r"\s+", " ", payload)
    payload = re.sub(r"[;,]\s*[;,]+", ";", payload)  # collapse repeated separators
    payload = payload.strip(";, ")
    return payload

def string_to_runs_by_z_lossless_robust(
    text: str,
    *,
    D: Optional[int] = 64,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Robust parser for old lossless template format.

    Supported main forms:
        a:10 2;20 3
        0:layer a
        1:layer a +[30 2;40] -[20 3]

    Also tolerant to noisy variants:
        0:l_a
        0:l-a
        0:l a
        Chinese punctuation / markdown fence / extra symbols

    Robustness policy:
      - only parse layers in [0, 63]
      - if one small run chunk is malformed:
            first delete unknown symbols and retry
            if still invalid, skip only that chunk
      - if one line is malformed, skip only that line
      - if layer template ref is invalid / missing, skip only that layer
    """

    def _log(msg: str):
        if verbose:
            print(msg)

    def normalize_text(s: str) -> str:
        s = s or ""
        replace_map = {
            "：": ":",
            "；": ";",
            "，": ",",
            "【": "[",
            "】": "]",
            "（": "(",
            "）": ")",
            "\t": " ",
            "\r": "\n",
        }
        for a, b in replace_map.items():
            s = s.replace(a, b)

        s = re.sub(r"```(?:txt|text|python)?", "", s, flags=re.IGNORECASE)
        s = s.replace("```", "")

        # normalize layer forms
        s = re.sub(r"\blayer\s+([a-z]+)\b", r"layer \1", s, flags=re.IGNORECASE)
        s = re.sub(r"\bl[\s\-_]+([a-z]+)\b", r"layer \1", s, flags=re.IGNORECASE)

        s = re.sub(r"\+\s*\[", "+[", s)
        s = re.sub(r"-\s*\[", "-[", s)

        s = "\n".join(line.strip() for line in s.splitlines())
        return s

    def split_kv_line(line: str):
        if ":" not in line:
            return None, None
        left, right = line.split(":", 1)
        return left.strip(), right.strip()

    def normalize_label(s: str) -> Optional[str]:
        s = (s or "").strip().lower()
        s = s.replace("-", " ")
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()

        m = re.fullmatch(r"layer\s+([a-z]+)", s)
        if m:
            return m.group(1)

        m = re.fullmatch(r"([a-z]+)", s)
        if m:
            return m.group(1)

        return None

    def sanitize_run_chunk(part: str) -> str:
        """
        Keep likely useful chars only:
          digits, minus, spaces, comma, semicolon, brackets
        Then normalize to a plain token like '12 3' or '12'
        """
        part = part or ""

        replace_map = {
            "：": " ",
            "；": ";",
            "，": " ",
            "【": " ",
            "】": " ",
            "（": " ",
            "）": " ",
            "[": " ",
            "]": " ",
            "(": " ",
            ")": " ",
        }
        for a, b in replace_map.items():
            part = part.replace(a, b)

        part = re.sub(r"[^0-9\-\s,;]+", " ", part)
        part = part.replace(",", " ")
        part = re.sub(r"\s+", " ", part).strip()
        return part

    def parse_one_run_chunk_with_repair(part: str) -> Optional[Tuple[int, int]]:
        """
        Old run token forms:
          '12'    -> (12,1)
          '12 3'  -> (12,3)

        If malformed:
          1) raw integer extraction
          2) sanitize and retry
          3) fail => None
        """
        raw = (part or "").strip()
        if not raw:
            return None

        nums = re.findall(r"-?\d+", raw)
        if len(nums) == 1:
            try:
                return (int(nums[0]), 1)
            except Exception:
                pass
        elif len(nums) >= 2:
            try:
                return (int(nums[0]), int(nums[1]))
            except Exception:
                pass

        repaired = sanitize_run_chunk(raw)
        if repaired and repaired != raw:
            nums = re.findall(r"-?\d+", repaired)
            if len(nums) == 1:
                try:
                    _log(f"[repair run] {raw!r} -> {repaired!r}")
                    return (int(nums[0]), 1)
                except Exception:
                    pass
            elif len(nums) >= 2:
                try:
                    _log(f"[repair run] {raw!r} -> {repaired!r}")
                    return (int(nums[0]), int(nums[1]))
                except Exception:
                    pass

        return None

    def robust_compact_str_to_runs(s: str) -> np.ndarray:
        """
        Parse run list robustly.
        Small chunk boundaries are cut by ; , newline.
        One bad chunk only affects itself.
        """
        s = (s or "").strip()
        if not s:
            return np.zeros((0, 2), dtype=np.int64)

        rows: List[Tuple[int, int]] = []
        for part in re.split(r"[;,\n]+", s):
            part = part.strip()
            if not part:
                continue

            item = parse_one_run_chunk_with_repair(part)
            if item is None:
                _log(f"[skip run token] cannot repair: {part!r}")
                continue

            start, length = item
            if start < 0:
                _log(f"[skip run token] negative start: {item}")
                continue
            if length <= 0:
                _log(f"[skip run token] non-positive length: {item}")
                continue

            rows.append((start, length))

        if not rows:
            return np.zeros((0, 2), dtype=np.int64)

        return np.asarray(rows, dtype=np.int64).reshape(-1, 2)

    def extract_layer_components(body: str):
        """
        Parse layer body robustly.

        Examples:
          'layer a'
          'layer a +[10 2;20] -[30 4]'
          'l_a +[10 2]'
          'a +[10]'   # fallback, though old format prefers 'layer a'
        """
        body = (body or "").strip()
        if not body:
            return None, "", ""

        body = re.sub(r"\bl[\s\-_]+([a-z]+)\b", r"layer \1", body, flags=re.IGNORECASE)

        m = re.match(r"^\s*(layer\s+[a-z]+|[a-z]+)\b", body, flags=re.IGNORECASE)
        if not m:
            return None, "", ""

        raw_label = m.group(1)
        label = normalize_label(raw_label)
        if label is None:
            return None, "", ""

        rest = body[m.end():]

        add_segments = []
        rem_segments = []

        i = 0
        n = len(rest)
        while i < n:
            ch = rest[i]
            if ch not in "+-":
                i += 1
                continue

            sign = ch
            i += 1

            while i < n and rest[i].isspace():
                i += 1

            if i < n and rest[i] == "[":
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if rest[j] == "[":
                        depth += 1
                    elif rest[j] == "]":
                        depth -= 1
                    j += 1

                if depth == 0:
                    seg = rest[i + 1:j - 1]
                    if sign == "+":
                        add_segments.append(seg)
                    else:
                        rem_segments.append(seg)
                    i = j
                    continue
                else:
                    # unmatched bracket: read until next sign
                    j = i + 1
                    while j < n and rest[j] not in "+-":
                        j += 1
                    seg = rest[i + 1:j]
                    if sign == "+":
                        add_segments.append(seg)
                    else:
                        rem_segments.append(seg)
                    i = j
                    continue

            # no bracket: read until next sign
            j = i
            while j < n and rest[j] not in "+-":
                j += 1
            seg = rest[i:j]
            if sign == "+":
                add_segments.append(seg)
            else:
                rem_segments.append(seg)
            i = j

        add_str = ";".join(s.strip() for s in add_segments if s.strip())
        rem_str = ";".join(s.strip() for s in rem_segments if s.strip())
        return label, add_str, rem_str

    text = normalize_text(text).strip()
    if not text:
        return [] if D is None else [np.zeros((0, 2), dtype=np.int64) for _ in range(D)]

    max_layers = 64 if D is None else min(int(D), 64)

    templates: Dict[str, np.ndarray] = {}
    layer_defs: Dict[int, Tuple[str, str, str]] = {}

    # first pass: parse loosely
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        left, right = split_kv_line(line)
        if left is None:
            _log(f"[skip line] no colon: {line!r}")
            continue

        # layer line
        if re.fullmatch(r"\d+", left):
            try:
                z = int(left)
            except Exception:
                _log(f"[skip layer line] bad z index: {line!r}")
                continue

            if not (0 <= z < max_layers):
                _log(f"[skip layer line] z out of range [0,{max_layers-1}]: {z}")
                continue

            label, add_str, rem_str = extract_layer_components(right)
            if label is None:
                _log(f"[skip layer line] cannot parse layer ref: {line!r}")
                continue

            layer_defs[z] = (label, add_str, rem_str)
            continue

        # template line
        if re.fullmatch(r"[a-z]+", left, flags=re.IGNORECASE):
            key = left.lower()
            try:
                templates[key] = robust_compact_str_to_runs(right)
            except Exception as e:
                _log(f"[skip template line] {line!r} -> {e}")
            continue

        _log(f"[skip line] unknown lhs: {line!r}")

    # output init
    out_D = max_layers if D is not None else (max(layer_defs.keys()) + 1 if layer_defs else 0)
    runs_by_z: List[np.ndarray] = [np.zeros((0, 2), dtype=np.int64) for _ in range(out_D)]

    # second pass: resolve layers
    for z in range(out_D):
        if z not in layer_defs:
            continue

        label, add_str, rem_str = layer_defs[z]
        if label not in templates:
            _log(f"[skip layer] undefined template '{label}' for z={z}")
            continue

        try:
            base = _runs_set(templates[label])
        except Exception as e:
            _log(f"[skip layer] bad base template '{label}' for z={z}: {e}")
            continue

        # local add parse
        adds = set()
        if add_str:
            try:
                add_runs = robust_compact_str_to_runs(add_str)
                adds = _runs_set(add_runs)
            except Exception as e:
                _log(f"[skip add part] z={z}: {e}")

        # local remove parse
        rems = set()
        if rem_str:
            try:
                rem_runs = robust_compact_str_to_runs(rem_str)
                rems = _runs_set(rem_runs)
            except Exception as e:
                _log(f"[skip remove part] z={z}: {e}")

        final = (base | adds) - rems

        if not final:
            runs_by_z[z] = np.zeros((0, 2), dtype=np.int64)
        else:
            try:
                runs_by_z[z] = np.array(sorted(list(final)), dtype=np.int64).reshape(-1, 2)
            except Exception as e:
                _log(f"[skip layer finalize] z={z}: {e}")
                continue

    return runs_by_z



#####################
def decode_voxel_2drle_by_z(
    runs_by_z: List[np.ndarray],
    shape: Tuple[int, int, int] = (64, 64, 64),
    *,
    validate: bool = True,
) -> np.ndarray:
    """
    Decode 64 z-slices of 2D RLE back to voxel coords (x,y,z).

    Args:
        runs_by_z: list length D; each (Kz,2) is (start,length)
        shape: (D,H,W)
        validate: check decoded coords within bounds

    Returns:
        coords: (N,3) int64 array (x,y,z)
    """
    D, H, W = shape
    if len(runs_by_z) != D:
        raise ValueError(f"runs_by_z must have length D={D}, got {len(runs_by_z)}")

    coords_chunks = []

    for zi in range(D):
        runs = np.asarray(runs_by_z[zi])
        if runs.size == 0:
            continue
        if runs.ndim != 2 or runs.shape[1] != 2:
            raise ValueError(f"runs_by_z[{zi}] must be (K,2), got {runs.shape}")
        if validate and not np.issubdtype(runs.dtype, np.integer):
            raise TypeError(f"runs_by_z[{zi}] must be integer dtype, got {runs.dtype}")

        runs = runs.astype(np.int64, copy=False)
        starts = runs[:, 0]
        lengths = runs[:, 1]

        if validate:
            valid_run_mask = (starts >= 0) & (lengths > 0)
            if not np.all(valid_run_mask):
                starts = starts[valid_run_mask]
                lengths = lengths[valid_run_mask]
            if starts.size == 0:
                continue

        # expand runs -> idx2d
        total = int(lengths.sum())
        idx2d = np.empty(total, dtype=np.int64)
        p = 0
        for s, L in zip(starts, lengths):
            idx2d[p:p+L] = np.arange(s, s + L, dtype=np.int64)
            p += L

        if validate:
            in_range = (idx2d >= 0) & (idx2d < W * H)
            idx2d = idx2d[in_range]
            if idx2d.size == 0:
                continue

        # idx2d -> (x,y) under x-fastest
        y = idx2d // W
        x = idx2d % W
        z = np.full_like(x, zi, dtype=np.int64)

        if validate:
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x, y, z = x[ok], y[ok], z[ok]
            if x.size == 0:
                continue

        coords_chunks.append(np.stack([x, y, z], axis=1))

    if not coords_chunks:
        return np.zeros((0, 3), dtype=np.int64)

    return np.concatenate(coords_chunks, axis=0).astype(np.int64)





def addmessage(message,before,after):
    answer={}
    answer['role']='assistant'
    answer['content']=[{"type": "text", "text": before}]
    question={}
    question['role']='user'
    question['content']=[{"type": "text", "text": after}]
    newmessage=message.copy()
    newmessage.append(answer)
    newmessage.append(question)
    return newmessage



def generate_save(model,messages,save_dir,save_name='test',save=True):

# Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=False,temperature=0,max_length=32768)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagepath", type=str, default='demo')
    parser.add_argument("--modelpath", type=str, default='pretrain')
    args = parser.parse_args()
    save_part_ply=True


    basepath=args.imagepath
    namelist=os.listdir(basepath)
    

    logger = get_logger('exp_1qwen_demo.log',verbosity=1)
    logger.info('start')
    

    modelpath=args.modelpath
    savedir='ours_demo'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                modelpath,
                torch_dtype=torch.bfloat16,
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
    

    for name in namelist:
        name=name[:-4]
 
        logger.info('begin: '+name)
        
        save_dir=os.path.join(savedir,name)

        if os.path.exists(os.path.join(save_dir,'allind.npy')):
            logger.info('skip success: '+name)
            continue


        os.makedirs(os.path.join(save_dir), exist_ok=True)

        image_path = os.path.join(basepath,name+'.png')
        

        shutil.copy(image_path, os.path.join(save_dir,'cond_img.png'))


        with open(os.path.join('./dataset/example_64_finetune_rle.txt'), "r", encoding="utf-8") as f:
            basicqu = f.read()

        output_image = Image.open(image_path)
        im_resized = output_image.resize((512, 512), Image.LANCZOS)

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
        
        try:


            basicoutput=generate_save(model,messages,save_dir,'basic_info')
            index=0
            while 'l_'+str(index) in basicoutput:
                index+=1
            logger.info('overall: basic_info')

            allcoord=[]
            for part in range(index):

                question="Based on the structured description of l_"+str(part)+", generate its 3D voxel (grid=64) in the 3D RLE (linear scan) format. Output one run per line as: start_index length"
                messages1=addmessage(messages,basicoutput,question)
                output1=generate_save(model,messages1,save_dir,'coord_'+str(part),save=True)
                print(len(messages1))

                runs_by_z2 = string_to_runs_by_z_lossless_robust(output1, D=64)
                voxels_back = decode_voxel_2drle_by_z(runs_by_z2, shape=(64,64,64))

                allcoord.append(voxels_back)
                np.save(os.path.join(save_dir,'ind_'+str(part)+'.npy'),voxels_back)
                if save_part_ply and len(voxels_back)!=0:
                    partply=trimesh.points.PointCloud(voxels_back)
                    partply.export(os.path.join(save_dir,'ind_'+str(part)+'.ply'))

                logger.info('part: '+str(part))

            np.save(os.path.join(save_dir,'allind.npy'),np.concatenate(allcoord))
            logger.info('success: '+name)
        except:
            logger.info('error: '+name)
