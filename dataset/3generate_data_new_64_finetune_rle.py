import os
import json
import numpy as np
from pathlib import Path
import ipdb
import argparse
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

import logging

import re
from typing import List, Dict, Tuple, Optional



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
# Lossless template encoding with delta
# ----------------------------
def runs_by_z_to_string_lossless(
    runs_by_z: List[np.ndarray],
    *,
    similarity_threshold: float = 0.90,
    use_chinese_colon: bool = True,
) -> str:
    """
    Lossless compression:
      - Define templates a,b,c... (exact runs)
      - For each layer, choose best template if sim>=threshold
      - Store layer as 'layer a +[adds] -[removes]' so it's exact (no point loss)
      - length==1 omitted in run lists

    Output:
      a：<runs>
      ...
      0：layer a +[...]-[...]
      ...
    """
    colon = "：" if use_chinese_colon else ":"
    D = len(runs_by_z)

    templates: List[np.ndarray] = []
    labels: List[str] = []

    layer_lines: List[str] = []

    for z in range(D):
        r = np.asarray(runs_by_z[z], dtype=np.int64)
        # find best template
        best_i = None
        best_sim = -1.0
        for i, t in enumerate(templates):
            sim = runs_similarity(r, t)
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i is None or best_sim < similarity_threshold:
            # make new template = exactly this layer (lossless)
            label = _int_to_label(len(templates))
            templates.append(r)
            labels.append(label)
            # exact reference (no delta)
            layer_lines.append(f"{z}{colon}layer {label}")
        else:
            label = labels[best_i]
            t = templates[best_i]
            s_r = _runs_set(r)
            s_t = _runs_set(t)

            adds = np.array(sorted(list(s_r - s_t)), dtype=np.int64).reshape(-1, 2) if (s_r - s_t) else np.zeros((0,2), np.int64)
            rems = np.array(sorted(list(s_t - s_r)), dtype=np.int64).reshape(-1, 2) if (s_t - s_r) else np.zeros((0,2), np.int64)

            add_str = runs_to_compact_str(adds)
            rem_str = runs_to_compact_str(rems)

            extra = ""
            if add_str:
                extra += f" +[{add_str}]"
            if rem_str:
                extra += f" -[{rem_str}]"

            layer_lines.append(f"{z}{colon}layer {label}{extra}")

    # template definitions first
    lines = []
    for label, t in zip(labels, templates):
        lines.append(f"{label}{colon}{runs_to_compact_str(t)}")
    lines.extend(layer_lines)
    return "\n".join(lines)

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

def string_to_runs_by_z_lossless(
    text: str,
    *,
    D: Optional[int] = None
) -> List[np.ndarray]:
    """
    Parse lossless format back to runs_by_z.
    Accepts ':' or '：'.
    """
    text = (text or "").strip()
    if not text:
        return [] if D is None else [np.zeros((0,2), np.int64) for _ in range(D)]

    templates: Dict[str, np.ndarray] = {}
    layer_defs: Dict[int, Tuple[str, str, str]] = {}  # z -> (label, add_str, rem_str)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # try layer line first
        m = _LAYER_RE.match(line)
        if m:
            z = int(m.group(1))
            label = m.group(2).lower()
            add_str = (m.group(3) or "").strip()
            rem_str = (m.group(4) or "").strip()
            layer_defs[z] = (label, add_str, rem_str)
            continue

        # template line
        m2 = _TEMPLATE_RE.match(line)
        if m2:
            key = m2.group(1).lower()
            body = (m2.group(2) or "").strip()
            # avoid confusing numeric layer like "0: ..."
            if re.fullmatch(r"\d+", key):
                raise ValueError(f"line looks like layer but not parsed: {line}")
            templates[key] = compact_str_to_runs(body)
            continue

        raise ValueError(f"cannot parse line: {line}")

    if D is None:
        D = max(layer_defs.keys()) + 1 if layer_defs else 0

    runs_by_z: List[np.ndarray] = [np.zeros((0,2), dtype=np.int64) for _ in range(D)]
    for z in range(D):
        if z not in layer_defs:
            continue
        label, add_str, rem_str = layer_defs[z]
        if label not in templates:
            raise ValueError(f"layer {z} references undefined template '{label}'")
        base = _runs_set(templates[label])
        adds = _runs_set(compact_str_to_runs(add_str)) if add_str else set()
        rems = _runs_set(compact_str_to_runs(rem_str)) if rem_str else set()

        final = (base | adds) - rems
        if not final:
            runs_by_z[z] = np.zeros((0,2), dtype=np.int64)
        else:
            runs_by_z[z] = np.array(sorted(list(final)), dtype=np.int64).reshape(-1, 2)

    return runs_by_z

# ----------------------------
# Helper: check roundtrip equality at runs level
# ----------------------------
def assert_runs_by_z_equal(a: List[np.ndarray], b: List[np.ndarray]):
    if len(a) != len(b):
        raise AssertionError(f"len mismatch: {len(a)} vs {len(b)}")
    for i, (ra, rb) in enumerate(zip(a, b)):
        sa, sb = _runs_set(ra), _runs_set(rb)
        if sa != sb:
            diff1 = len(sa - sb)
            diff2 = len(sb - sa)
            raise AssertionError(f"slice {i} mismatch: a-b={diff1}, b-a={diff2}")


###########################################


def encode_voxel_2drle_by_z(
    coords: np.ndarray,
    shape: Tuple[int, int, int] = (64, 64, 64),  # (D,H,W) = (z,y,x)
    *,
    validate: bool = True,
    unique: bool = True,
) -> List[np.ndarray]:
    """
    Encode 3D voxel coords (x,y,z) into 64 z-slices of 2D RLE (linear scan).
    For each z slice, flatten (y,x) with x-fastest:
        idx2d = x + W*y
    Save only 1-runs: (start, length).

    Args:
        coords: (N,3) int array, rows are (x,y,z) in [0,63]
        shape: (D,H,W)
        validate: bounds/dtype checks
        unique: remove duplicate voxels

    Returns:
        runs_by_z: list length D, each element is (Kz,2) int64 array
    """
    D, H, W = shape
    coords = np.asarray(coords)

    # init 64 empty slices
    runs_by_z: List[np.ndarray] = [np.zeros((0, 2), dtype=np.int64) for _ in range(D)]
    if coords.size == 0:
        return runs_by_z

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3), got {coords.shape}")
    if validate and not np.issubdtype(coords.dtype, np.integer):
        raise TypeError(f"coords dtype must be integer, got {coords.dtype}")

    x = coords[:, 0].astype(np.int64)
    y = coords[:, 1].astype(np.int64)
    z = coords[:, 2].astype(np.int64)

    if validate:
        if (x < 0).any() or (x >= W).any() or (y < 0).any() or (y >= H).any() or (z < 0).any() or (z >= D).any():
            raise ValueError("coords out of bounds for given shape")

    # Process each slice
    for zi in range(D):
        mask = (z == zi)
        if not np.any(mask):
            continue

        idx2d = x[mask] + W * y[mask]  # x-fastest linear index in [0, W*H-1]
        idx2d = np.unique(idx2d) if unique else np.sort(idx2d)

        if idx2d.size == 0:
            continue

        # Build runs from sorted indices
        # break when idx2d[i] != idx2d[i-1] + 1
        breaks = np.where(idx2d[1:] != idx2d[:-1] + 1)[0] + 1
        starts_pos = np.concatenate(([0], breaks))
        ends_pos = np.concatenate((breaks, [idx2d.size]))  # exclusive

        starts = idx2d[starts_pos]
        lengths = ends_pos - starts_pos

        runs_by_z[zi] = np.stack([starts, lengths], axis=1).astype(np.int64)

    return runs_by_z


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
            if (starts < 0).any():
                raise ValueError(f"slice z={zi}: start must be >= 0")
            if (lengths <= 0).any():
                raise ValueError(f"slice z={zi}: length must be >= 1")

        # expand runs -> idx2d
        total = int(lengths.sum())
        idx2d = np.empty(total, dtype=np.int64)
        p = 0
        for s, L in zip(starts, lengths):
            idx2d[p:p+L] = np.arange(s, s + L, dtype=np.int64)
            p += L

        if validate:
            if (idx2d >= W * H).any():
                raise ValueError(f"slice z={zi}: idx2d out of range (>= {W*H})")

        # idx2d -> (x,y) under x-fastest
        y = idx2d // W
        x = idx2d % W
        z = np.full_like(x, zi, dtype=np.int64)

        if validate:
            if (x < 0).any() or (x >= W).any() or (y < 0).any() or (y >= H).any():
                raise ValueError(f"slice z={zi}: decoded (x,y) out of bounds")

        coords_chunks.append(np.stack([x, y, z], axis=1))

    if not coords_chunks:
        return np.zeros((0, 3), dtype=np.int64)

    return np.concatenate(coords_chunks, axis=0).astype(np.int64)


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
def gen_conver(oriconv,question,answer):
    conversation1={}
    conversation1['from']="human"
    conversation1['value']=question
    conversation2={}
    conversation2['from']="gpt"
    conversation2['value']=answer
    oriconv.append(conversation1)
    oriconv.append(conversation2)
    return oriconv


def runs_to_lines(runs: np.ndarray) -> str:
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 2:
        raise ValueError(f"runs must be (N,2), got {runs.shape}")
    runs = runs.astype(np.int64, copy=False)
    return "\n".join(f"{s} {l}" for s, l in runs)

def lines_to_runs(s: str) -> np.ndarray:
    s = s.strip()
    if not s:
        return np.zeros((0, 2), dtype=np.int64)
    rows = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split()
        rows.append((int(a), int(b)))
    return np.asarray(rows, dtype=np.int64)




parser = argparse.ArgumentParser()
parser.add_argument("--ind", type=int, default=0)
parser.add_argument("--range", type=int, default=1800)
args = parser.parse_args()


    
alldata=[]
txtname='txt_rep_64_finetune_verse'
basepath='./tmp_verse'
savepath='trainset_64_final_verse'

voxel_path=os.path.join(basepath,'partobj')
json_path=os.path.join(basepath,'finaljson')
os.makedirs(savepath, exist_ok=True)
namelist=os.listdir(txtname)
namelist=namelist[args.ind*args.range:(args.ind+1)*args.range]

alllength=[]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
logger = get_logger('info'+str(args.ind)+'.log',verbosity=1)
logger.info('start')

for name in namelist:
    name=name[:-4]
    with open(os.path.join(json_path,name+'.json'),'r') as f:
        jsondata=json.load(f)
    with open(os.path.join(txtname,name+'.txt'), "r", encoding="utf-8") as f:
        content = f.read()
    with open(os.path.join('example_64_finetune_rle.txt'), "r", encoding="utf-8") as f:
        basicqu = f.read()

    for part in range(len(jsondata['parts'])):

        for ind in range(0,25):
            dataseq={}
            dataseq['id']=name+'_'+str(ind).zfill(3)
            dataseq['image']=os.path.join(name,str(ind).zfill(3)+'.png')
            dataseq['conversations']=[]
            dataseq['data_source']='physx'

            gen_conver(dataseq['conversations'],\
                    "<image>\n"+basicqu,\
                    content
            )

            coords=np.load(os.path.join(voxel_path,name,'64','ind_'+str(part)+'.npy')).astype(np.int64)
            runs = encode_voxel_2drle_by_z(coords, shape=(64,64,64))

            txt = runs_by_z_to_string_lossless(runs, similarity_threshold=0.50,use_chinese_colon=False)
            runs_by_z2 = string_to_runs_by_z_lossless(txt, D=64)
            coords_rec = decode_voxel_2drle_by_z(runs_by_z2, shape=(64,64,64))
            
            idx = coords[:, 0] + 64 * (coords[:, 1] + 64 * coords[:, 2])
            idx2 = coords_rec[:, 0] + 64 * (coords_rec[:, 1] + 64 * coords_rec[:, 2])

            logger.info(name+'_'+str(part)+'_'+str(ind)+'_same: '+str(np.array_equal(np.unique(idx), np.unique(idx2))))


            
            gen_conver(dataseq['conversations'],\
                    "Based on the structured description of l_"+str(part)+", generate its 3D voxel (grid=64) in the 3D RLE (linear scan) format. Output one run per line as: start_index length",\
                    txt
            )

            tokennum=len(tokenizer(str(txt)+"Based on the structured description of l_"+str(part)+", generate its 3D voxel (grid=64) in the 3D RLE (linear scan) format. Output one run per line as: start_index length"+"<image>\n"+basicqu+content)["input_ids"])
            alllength.append(tokennum)
            dataseq['meshlength']=tokennum

            if tokennum<20000:


                alldata.append(dataseq)
            
            
    logger.info(name)

with open(os.path.join(savepath,'training_set_'+str(args.ind)+'_randompart.json'), 'w') as json_file:
    json.dump(alldata, json_file, indent=4)   


