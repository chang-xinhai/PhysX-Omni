#!/usr/bin/env python3
"""Build DCS manifest rows from one full-color render paired with one predicted part mask."""

import argparse
import csv
import json
import re
import struct
import zlib
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_RENDER_ROOT = "benchmark/benchmark_assets/rendered_views/description"
DEFAULT_DESCRIPTION_ROOT = "benchmark/benchmark_assets/description"
DEFAULT_RESULT_ROOT = "physx_result"


DEFAULT_SOURCES = [
    "ours_mobility_181500:ours_mobility_181500:ours:mobility",
    "output_physxanything_mobility:output_physxanything_mobility:physxanything:mobility",
    "outputs_physxgen_mobility:outputs_physxgen_mobility:physxgen:mobility",
    "ours_verse_181500:ours_verse_181500:ours:verse",
    "output_physxanything_verse:output_physxanything_verse:physxanything:verse",
    "outputs_physxgen_verse:outputs_physxgen_verse:physxgen:verse",
    "ours_inthewild_181500:ours_inthewild_181500:ours:inthewild",
    "output_physxanything_inthewild:output_physxanything_inthewild:physxanything:inthewild",
    "outputs_physxgen_inthewild:outputs_physxgen_inthewild:physxgen:inthewild",
]


def numeric_path_key(path):
    path = Path(path)
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return (0, int(match.group(1)), path.name)
    try:
        return (0, int(path.stem), path.name)
    except ValueError:
        return (1, path.name)


def numeric_index(path):
    path = Path(path)
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def iter_png_rows(path):
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG file: {path}")

    offset = 8
    width = height = bit_depth = color_type = interlace = None
    idat = []
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            idat.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if bit_depth != 8 or interlace not in (0, None):
        raise ValueError(f"unsupported PNG format for mask black check: {path}")
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"unsupported PNG color type {color_type} for mask black check: {path}")
    channels = channels_by_type[color_type]
    bytes_per_pixel = channels
    row_len = width * channels
    raw = zlib.decompress(b"".join(idat))

    prev = bytearray(row_len)
    pos = 0
    for _row_index in range(height):
        filter_type = raw[pos]
        pos += 1
        scan = bytearray(raw[pos : pos + row_len])
        pos += row_len
        recon = bytearray(row_len)
        for i, value in enumerate(scan):
            left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            up = prev[i]
            upper_left = prev[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            if filter_type == 0:
                recon[i] = value
            elif filter_type == 1:
                recon[i] = (value + left) & 0xFF
            elif filter_type == 2:
                recon[i] = (value + up) & 0xFF
            elif filter_type == 3:
                recon[i] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                recon[i] = (value + paeth_predictor(left, up, upper_left)) & 0xFF
            else:
                raise ValueError(f"unsupported PNG filter {filter_type} in {path}")
        prev = recon
        yield recon, channels, color_type


def mask_white_ratio(path, black_threshold):
    if Image is not None:
        with Image.open(path) as image:
            hist = image.convert("L").histogram()
        total_pixels = sum(hist)
        if total_pixels <= 0:
            return 0.0
        white_pixels = sum(hist[int(black_threshold) + 1 :])
        return white_pixels / total_pixels

    white_pixels = 0
    total_pixels = 0
    for row, channels, color_type in iter_png_rows(path):
        total_pixels += len(row) // channels
        if color_type == 0:
            white_pixels += sum(1 for value in row if value > black_threshold)
        elif color_type == 2:
            for i in range(0, len(row), channels):
                if row[i] > black_threshold or row[i + 1] > black_threshold or row[i + 2] > black_threshold:
                    white_pixels += 1
        elif color_type == 4:
            for i in range(0, len(row), channels):
                if row[i] > black_threshold:
                    white_pixels += 1
        elif color_type == 6:
            for i in range(0, len(row), channels):
                if row[i] > black_threshold or row[i + 1] > black_threshold or row[i + 2] > black_threshold:
                    white_pixels += 1
    if total_pixels <= 0:
        return 0.0
    return white_pixels / total_pixels


def parse_source(value):
    parts = str(value).split(":")
    if len(parts) != 4 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--source must have format <render_folder>:<result_folder>:<method>:<dataset>"
        )
    return tuple(part.strip() for part in parts)


def load_description_map(description_root, dataset):
    path = Path(description_root) / f"descript_{dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(f"description JSON not found for dataset={dataset}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"description JSON must be an object/dict: {path}")
    return {str(k): str(v) for k, v in data.items() if str(v).strip()}, path


def list_pngs(path):
    if not path.is_dir():
        return []
    return sorted([p for p in path.glob("*.png") if p.is_file()], key=numeric_path_key)


def paired_view_candidates(color_dir, mask_dir, black_threshold):
    color_by_index = {numeric_index(p): p for p in list_pngs(color_dir)}
    pairs = []
    for mask_path in list_pngs(mask_dir):
        idx = numeric_index(mask_path)
        color_path = color_by_index.get(idx)
        if color_path is None:
            continue
        pairs.append(
            {
                "index": idx,
                "color_path": color_path,
                "mask_path": mask_path,
                "white_ratio": mask_white_ratio(mask_path, black_threshold),
            }
        )
    return sorted(pairs, key=lambda item: numeric_path_key(item["mask_path"]))


def build_rows(
    render_root,
    result_root,
    description_root,
    sources,
    mask_subdir,
    black_threshold,
    seed,
    limit=None,
    object_id_filter=None,
):
    render_root = Path(render_root)
    result_root = Path(result_root)
    rows = []
    missing_rows = []
    description_cache = {}

    for render_folder, result_folder, method, dataset in sources:
        color_source_root = render_root / render_folder
        result_source_root = result_root / result_folder
        if not color_source_root.is_dir() and not result_source_root.is_dir():
            raise NotADirectoryError(
                f"neither color render folder nor result folder exists: "
                f"{color_source_root} ; {result_source_root}"
            )
        if dataset not in description_cache:
            description_cache[dataset] = load_description_map(description_root, dataset)
        description_map, description_json_path = description_cache[dataset]

        if result_source_root.is_dir():
            object_ids = sorted(
                [p.name for p in result_source_root.iterdir() if p.is_dir() and not p.name.endswith("_mask")],
                key=lambda name: numeric_path_key(Path(name)),
            )
        else:
            object_ids = sorted(
                [p.name for p in color_source_root.iterdir() if p.is_dir() and not p.name.endswith("_mask")],
                key=lambda name: numeric_path_key(Path(name)),
            )
        if object_id_filter:
            keep = set(str(x) for x in object_id_filter)
            object_ids = [object_id for object_id in object_ids if object_id in keep]
        if limit is not None and limit > 0:
            object_ids = object_ids[:limit]

        for object_id in object_ids:
            color_dir = color_source_root / object_id
            result_dir = result_source_root / object_id
            mask_dir = result_dir / mask_subdir
            reference_description = description_map.get(object_id, "")

            color_pngs = list_pngs(color_dir)
            mask_pngs = list_pngs(mask_dir)
            status = []
            if not color_pngs:
                status.append("missing_color_render_views")
            if not mask_pngs:
                status.append("missing_description_masks")
            if not reference_description:
                status.append("missing_reference_description")

            candidates = []
            nonblack_candidates = []
            selected = []
            selection_mode = "missing"
            if color_pngs and mask_pngs:
                candidates = paired_view_candidates(color_dir, mask_dir, black_threshold)
                if not candidates:
                    status.append("missing_paired_description_views")
                else:
                    nonblack_candidates = [item for item in candidates if item["white_ratio"] > 0.0]
                    best = max(candidates, key=lambda item: item["white_ratio"])
                    selected = [best]
                    if best["white_ratio"] > 0.0:
                        selection_mode = "largest_white_ratio_mask"
                    else:
                        selection_mode = "all_masks_black_largest_ratio_fallback"
                        status.append("all_masks_black")

            missing_evidence = (
                not color_pngs
                or not mask_pngs
                or not reference_description
                or ("missing_paired_description_views" in status)
            )
            ready = not missing_evidence
            dcs_missing_score_zero = missing_evidence

            color_paths = [str(item["color_path"].resolve()) for item in selected]
            mask_paths = [str(item["mask_path"].resolve()) for item in selected]
            row = {
                "metric": "dcs",
                "method": method,
                "dataset": dataset,
                "object_id": object_id,
                "sample_id": object_id,
                "part_id": object_id,
                "relative_dir": f"dcs_mask/{method}/{dataset}/{object_id}",
                "render_folder": render_folder,
                "result_folder": result_folder,
                "source_result_dir": str(result_dir.resolve()),
                "color_render_dir": str(color_dir.resolve()),
                "mask_dir": str(mask_dir.resolve()),
                "description_json": str(description_json_path.resolve()),
                "reference_description": reference_description,
                "num_color_views_available": len(color_pngs),
                "num_description_mask_views_available": len(mask_pngs),
                "num_paired_views_available": len(candidates),
                "num_nonblack_masks_available": len(nonblack_candidates),
                "num_description_views_required": 1,
                "num_description_views_selected": len(selected),
                "view_pair_indices": [item["index"] for item in selected],
                "selected_mask_white_ratio": selected[0]["white_ratio"] if selected else None,
                "view_image_paths": color_paths,
                "mask_image_paths": mask_paths,
                "selection_mode": selection_mode,
                "ready": ready,
                "status": "ready" if not status else ";".join(status),
                "dcs_missing_score_zero": dcs_missing_score_zero,
            }
            rows.append(row)
            if not ready:
                missing_rows.append(row)

    return rows, missing_rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "method",
        "dataset",
        "object_id",
        "part_id",
        "relative_dir",
        "render_folder",
        "result_folder",
        "source_result_dir",
        "color_render_dir",
        "mask_dir",
        "description_json",
        "reference_description",
        "num_color_views_available",
        "num_description_mask_views_available",
        "num_paired_views_available",
        "num_nonblack_masks_available",
        "num_description_views_required",
        "num_description_views_selected",
        "view_pair_indices",
        "selected_mask_white_ratio",
        "view_image_paths",
        "mask_image_paths",
        "selection_mode",
        "ready",
        "status",
        "dcs_missing_score_zero",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("view_pair_indices", "view_image_paths", "mask_image_paths"):
                out[key] = json.dumps(out.get(key, []), ensure_ascii=False)
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Build paired full-render + mask DCS manifest rows.")
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--description-root", default=DEFAULT_DESCRIPTION_ROOT)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        default=[],
        help="Source mapping: <render_folder>:<result_folder>:<method>:<dataset>.",
    )
    parser.add_argument("--mask-subdir", default="description")
    parser.add_argument(
        "--num-views",
        type=int,
        default=1,
        help="Deprecated alias kept for scripts; this mask-DCS builder always selects one color/mask pair.",
    )
    parser.add_argument("--black-threshold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument(
        "--object-id",
        action="append",
        default=[],
        help="Restrict the manifest to one object id. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-jsonl",
        default="benchmark/benchmark_manifests/description_mask_pairs_all_current.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark/benchmark_manifests/description_mask_pairs_all_current.csv",
    )
    parser.add_argument(
        "--missing-jsonl",
        default="benchmark/benchmark_results/logs/description_mask_manifest_missing_all_current.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ready_only and args.missing_only:
        raise ValueError("--ready-only and --missing-only are mutually exclusive")
    sources = args.source or [parse_source(value) for value in DEFAULT_SOURCES]
    rows, missing_rows = build_rows(
        render_root=args.render_root,
        result_root=args.result_root,
        description_root=args.description_root,
        sources=sources,
        mask_subdir=args.mask_subdir,
        black_threshold=args.black_threshold,
        seed=args.seed,
        limit=args.limit,
        object_id_filter=args.object_id,
    )
    if args.ready_only:
        output_rows = [row for row in rows if row["ready"]]
    elif args.missing_only:
        output_rows = missing_rows
    else:
        output_rows = rows
    write_jsonl(output_rows, args.output_jsonl)
    write_csv(output_rows, args.output_csv)
    if args.missing_jsonl:
        write_jsonl(missing_rows, args.missing_jsonl)
    print(
        f"rows_total={len(rows)} rows_written={len(output_rows)} "
        f"ready={sum(1 for row in rows if row['ready'])} missing={len(missing_rows)}"
    )
    print(
        "selection_modes="
        + json.dumps(
            {
                key: sum(1 for row in rows if row["selection_mode"] == key)
                for key in sorted({row["selection_mode"] for row in rows})
            },
            ensure_ascii=False,
        )
    )
    print(f"jsonl={args.output_jsonl}")
    print(f"csv={args.output_csv}")
    if args.missing_jsonl:
        print(f"missing_jsonl={args.missing_jsonl}")


if __name__ == "__main__":
    main()
