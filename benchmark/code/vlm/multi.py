"""
Multi-turn multimodal inference runner for Qwen3.5-style VLM models.

Core behavior:
- Turn composition is YAML-driven (text + optional image/video fields).
- Conversation context is preserved within one sample pair.
- Different sample pairs do not share context.

Common commands:
1) Batch mode (paired image mirrored from --image-input-dir):
   python benchmark/code/vlm/multi.py \
     --input-dir ./render_outputs/v4/demo_sacle_2 \
     --image-input-dir ./image \
     --prompts-file ./benchmark/prompts/prompts_vaps.yaml

2) Single mode with manual video + manual image path:
   python benchmark/code/vlm/multi.py \
     --input-dir ./render_outputs/v4/demo_sacle_2 \
     --single-seq-path ./render_outputs/v4/demo_sacle_2/<sample_id>/xxx.mp4 \
     --single-image-path ./image_multi_view/<sample_id>/first_frame_scan/az_000.png \
     --prompts-file ./benchmark/prompts/prompts_vaps.yaml

3) Affordance mode (original image + multi-view heatmaps):
   - YAML turn should include:
       image_from_input: true
       affordance_views_from_input: true
   python benchmark/code/vlm/multi.py \
     --input-dir ./render_outputs/v4/demo_sacle_2 \
     --image-input-dir ./image \
     --affordance-views-input-dir ./outputs_affordance_views/demo_sacle_2_8views \
     --affordance-views-subdir affordance_heatmap_views \
     --affordance-views-num-images 8 \
     --prompts-file ./benchmark/prompts/prompts_affordance.yaml
"""

import argparse
import csv
import atexit
import fcntl
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def to_jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def _sanitize_for_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def acquire_model_run_lock(model_id):
    lock_name = f"phys_anything_multi_{_sanitize_for_filename(model_id)}.lock"
    lock_dir = Path(os.environ.get("PHYSX_OMNI_LOCK_DIR", "benchmark/logs/locks"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / lock_name
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        holder = lock_file.read().strip() or "unknown"
        lock_file.close()
        raise RuntimeError(
            f"Another multi.py process is already running for model `{model_id}`. "
            f"Lock file: {lock_path}. Holder: {holder}"
        ) from exc

    lock_file.seek(0)
    lock_file.truncate(0)
    lock_file.write(f"pid={os.getpid()} ts={datetime.now(timezone.utc).isoformat()}")
    lock_file.flush()

    def _release_lock():
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_file.close()
        except Exception:
            pass

    atexit.register(_release_lock)
    return str(lock_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video inference for Qwen3.5-style multimodal models.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-122B-A10B", help="HF model id.")
    parser.add_argument("--input-dir", default=None, help="Root directory to scan videos from. Optional when --pairs-manifest is used.")
    parser.add_argument(
        "--single-seq-path",
        default=None,
        help=(
            "Optional single-sample path (sample folder or video file). "
            "If provided, only one paired sequence is inferred."
        ),
    )
    parser.add_argument(
        "--single-image-path",
        default=None,
        help=(
            "Optional manual image path for single-sequence inference. "
            "Used when turn config has image_from_input: true. "
            "Must be used together with --single-seq-path or --pairs-manifest."
        ),
    )
    parser.add_argument(
        "--pairs-manifest",
        default=None,
        help=(
            "Optional JSON/JSONL/CSV manifest with explicit per-row video_path/video_paths, image_path, "
            "view_image_paths, affordance_view_paths, and method/dataset/object_id metadata. "
            "When set, --input-dir is not required."
        ),
    )
    parser.add_argument("--prompts-file", required=True, help="YAML file with top-level `turns` list.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens for generation.")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Optional target sampling fps for video frames in processor (mutually exclusive with --video-num-frames).",
    )
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=None,
        help="Optional fixed number of sampled video frames in processor (mutually exclusive with --video-fps).",
    )
    parser.add_argument(
        "--disable-adaptive-video-sampling",
        action="store_true",
        help=(
            "Disable duration-adaptive video sampling. "
            "By default, when --video-fps/--video-num-frames are unset, num_frames is auto-scaled with video duration."
        ),
    )
    parser.add_argument(
        "--adaptive-base-fps",
        type=float,
        default=10.0,
        help="Base fps used to convert video duration into adaptive num_frames when auto-sampling is enabled.",
    )
    parser.add_argument(
        "--adaptive-frames-per-part",
        type=int,
        default=8,
        help="Minimum frame budget per estimated moving part under adaptive sampling.",
    )
    parser.add_argument(
        "--adaptive-part-seconds",
        type=float,
        default=1.68,
        help=(
            "Estimated seconds occupied by one part motion segment in rendered videos. "
            "Used to map duration -> estimated part count."
        ),
    )
    parser.add_argument(
        "--adaptive-min-frames",
        type=int,
        default=32,
        help="Lower bound of adaptive num_frames.",
    )
    parser.add_argument(
        "--adaptive-max-frames",
        type=int,
        default=1024,
        help="Upper bound of adaptive num_frames.",
    )
    parser.add_argument(
        "--adaptive-fallback-frames",
        type=int,
        default=64,
        help="Fallback num_frames when duration probing fails under adaptive sampling.",
    )
    parser.add_argument("--video-glob", default="**/*.mp4", help="Glob pattern under input-dir.")
    parser.add_argument(
        "--image-input-dir",
        default=None,
        help="Root directory of paired images (mirrors input-dir structure). Required if any turn uses image_from_input: true.",
    )
    parser.add_argument(
        "--image-name",
        default="first_frame.png",
        help="Paired image filename under mirrored subfolder when image_from_input is enabled.",
    )
    parser.add_argument(
        "--affordance-views-input-dir",
        default=None,
        help=(
            "Root directory of mirrored affordance multi-view images. "
            "Expected per-sample leaf: <root>/<sample_id>/<affordance-views-subdir>/..."
        ),
    )
    parser.add_argument(
        "--affordance-views-subdir",
        default="affordance_heatmap_views",
        help="Subdirectory name under each sample folder that stores affordance view images.",
    )
    parser.add_argument(
        "--affordance-views-glob",
        default="*.png",
        help="Glob for affordance view files inside each sample's affordance views directory.",
    )
    parser.add_argument(
        "--affordance-views-num-images",
        type=int,
        default=8,
        help=(
            "How many affordance view images to feed after numeric filename sorting. "
            "When >0, views are sampled uniformly across the full view sequence; "
            "set <=0 to feed all matched images."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="benchmark/benchmark_results/raw_vlm_outputs",
        help="Directory root for result files.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/processor/config from local HF cache only (no network requests).",
    )
    parser.add_argument(
        "--debug-input-shapes",
        action="store_true",
        help="Print processor/model input tensor shapes for each inference turn.",
    )
    return parser.parse_args()


def load_turns(prompts_file):
    prompts_path = Path(prompts_file)
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with prompts_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    turns_raw = data.get("turns")
    if not isinstance(turns_raw, list) or len(turns_raw) < 1:
        raise ValueError("Prompts file must contain top-level `turns` list with at least 1 item.")

    turns = []
    for idx, item in enumerate(turns_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Turn item at index {idx} must be a dict.")

        text = item.get("text")
        if text is None:
            text = item.get("prompt")
        turn_id = item.get("id") or f"turn_{idx + 1}"
        image = item.get("image")
        static_image_paths = item.get("static_image_paths") or []
        image_from_input = bool(item.get("image_from_input", False))
        view_images_from_input = bool(item.get("view_images_from_input", False))
        mask_images_from_input = bool(item.get("mask_images_from_input", False))
        affordance_views_from_input = bool(item.get("affordance_views_from_input", False))
        video_from_input = bool(item.get("video_from_input", False))
        videos_from_input = bool(item.get("videos_from_input", False))

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Turn text at index {idx} is empty or invalid.")
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError(f"Turn id at index {idx} is empty or invalid.")

        resolved_image = None
        if image is not None and image_from_input:
            raise ValueError(f"Turn index {idx}: use either `image` or `image_from_input`, not both.")
        if image is not None:
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"`image` at turn index {idx} must be a non-empty string path.")
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = (prompts_path.parent / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file for turn index {idx} not found: {image_path}")
            resolved_image = str(image_path)

        if not isinstance(static_image_paths, list):
            raise ValueError(f"`static_image_paths` at turn index {idx} must be a list of image paths.")
        resolved_static_image_paths = []
        for image_idx, image_value in enumerate(static_image_paths):
            if not isinstance(image_value, str) or not image_value.strip():
                raise ValueError(
                    f"`static_image_paths[{image_idx}]` at turn index {idx} must be a non-empty string path."
                )
            static_path = Path(image_value)
            if not static_path.is_absolute():
                static_path = (prompts_path.parent / static_path).resolve()
            if not static_path.is_file():
                raise FileNotFoundError(f"Static image file for turn index {idx} not found: {static_path}")
            resolved_static_image_paths.append(str(static_path))

        turns.append(
            {
                "id": turn_id.strip(),
                "text": text,
                "image": resolved_image,
                "static_image_paths": resolved_static_image_paths,
                "image_from_input": image_from_input,
                "view_images_from_input": view_images_from_input,
                "mask_images_from_input": mask_images_from_input,
                "affordance_views_from_input": affordance_views_from_input,
                "video_from_input": video_from_input,
                "videos_from_input": videos_from_input,
            }
        )
    return turns


def discover_videos(input_dir, video_glob):
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")

    videos = sorted([p for p in root.glob(video_glob) if p.is_file()])
    return root, videos


def resolve_single_video(input_root, videos, single_seq_path):
    target = Path(single_seq_path)
    if target.is_file():
        if target in videos:
            return target
        if target.suffix.lower() == ".mp4":
            return target
        raise FileNotFoundError(f"`--single-seq-path` points to a file that is not a video: {target}")

    sample_id = target.name
    same_id_videos = [v for v in videos if v.parent.name == sample_id]
    if not same_id_videos:
        raise FileNotFoundError(
            f"No video found under input-dir `{input_root}` for sample id `{sample_id}` "
            f"(from --single-seq-path={single_seq_path})."
        )
    if len(same_id_videos) > 1:
        print(
            f"[Warn] Multiple videos matched sample id `{sample_id}`, using first: {same_id_videos[0]}",
            flush=True,
        )
    return same_id_videos[0]


def resolve_model_source(model_id, local_files_only):
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)

    if not local_files_only:
        return model_id

    if "/" not in model_id:
        return model_id
    org, name = model_id.split("/", 1)
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}"
    ref_path = cache_dir / "refs" / "main"
    if not ref_path.is_file():
        return model_id
    revision = ref_path.read_text(encoding="utf-8").strip()
    if not revision:
        return model_id
    snapshot_dir = cache_dir / "snapshots" / revision
    if snapshot_dir.is_dir():
        return str(snapshot_dir)
    return model_id


def resolve_paired_image_path(video_path, input_root, image_input_root, image_name):
    if image_input_root is None:
        raise ValueError(
            "This config uses `image_from_input: true`, but `--image-input-dir` was not provided."
        )

    video_path = Path(video_path)
    input_root = Path(input_root)
    image_input_root = Path(image_input_root)
    rel_parent = video_path.parent.relative_to(input_root)
    image_path = image_input_root.joinpath(*rel_parent.parts) / image_name
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Paired image not found for video `{video_path}`. Expected: {image_path}"
        )
    return str(image_path)


def resolve_single_image_path(single_image_path):
    image_path = Path(single_image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"`--single-image-path` not found: {image_path}")
    return str(image_path.resolve())


def numeric_path_key(path):
    path = Path(path)
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def select_evenly_spaced_paths(paths, num_paths, label="paths"):
    paths = sorted([Path(p) for p in paths], key=numeric_path_key)
    if num_paths <= 0 or len(paths) <= num_paths:
        return paths
    if num_paths == 1:
        return [paths[0]]
    indices = [round(i * (len(paths) - 1) / (num_paths - 1)) for i in range(num_paths)]
    selected = [paths[i] for i in indices]
    print(
        f"[Sampling] {label}: evenly sampled {len(selected)}/{len(paths)} views "
        f"indices={indices}",
        flush=True,
    )
    return selected


def require_min_paths(paths, num_paths, source):
    if num_paths > 0 and len(paths) < num_paths:
        raise ValueError(
            f"Expected at least {num_paths} affordance view files, "
            f"but found {len(paths)} under: {source}"
        )


def resolve_affordance_view_paths(
    video_path,
    input_root,
    affordance_views_input_dir,
    affordance_views_subdir,
    affordance_views_glob,
    affordance_views_num_images,
):
    if affordance_views_input_dir is None:
        raise ValueError(
            "This config uses `affordance_views_from_input: true`, "
            "but `--affordance-views-input-dir` was not provided."
        )

    video_path = Path(video_path)
    input_root = Path(input_root)
    affordance_views_input_dir = Path(affordance_views_input_dir)
    rel_parent = video_path.parent.relative_to(input_root)
    views_dir = affordance_views_input_dir.joinpath(*rel_parent.parts) / affordance_views_subdir
    if not views_dir.is_dir():
        raise FileNotFoundError(f"Affordance views directory not found for video `{video_path}`: {views_dir}")

    view_paths = sorted([p for p in views_dir.glob(affordance_views_glob) if p.is_file()], key=numeric_path_key)
    if not view_paths:
        raise FileNotFoundError(
            f"No affordance view files matched `{affordance_views_glob}` under directory: {views_dir}"
        )

    require_min_paths(view_paths, affordance_views_num_images, views_dir)
    view_paths = select_evenly_spaced_paths(
        view_paths,
        affordance_views_num_images,
        label=f"affordance_views:{views_dir}",
    )

    return [str(p) for p in view_paths]


def probe_video_duration_seconds(video_path):
    """Return video duration in seconds via ffprobe, then torchvision fallback."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        proc = None

    if proc is not None and proc.returncode == 0:
        raw = (proc.stdout or "").strip()
        if raw:
            try:
                duration = float(raw)
            except ValueError:
                duration = None
            if duration is not None and math.isfinite(duration) and duration > 0.0:
                return duration

    # ffprobe may be unavailable in some containers; fallback to torchvision metadata.
    try:
        from torchvision.io import read_video_timestamps  # pylint: disable=import-outside-toplevel

        pts, fps = read_video_timestamps(str(video_path), pts_unit="sec")
        if pts:
            if len(pts) >= 2:
                duration = float(pts[-1]) - float(pts[0])
            elif fps and fps > 0:
                duration = 1.0 / float(fps)
            else:
                duration = None
            if duration is not None and math.isfinite(duration) and duration > 0.0:
                return duration
    except Exception:
        pass

    return None


def resolve_video_sampling(video_path, args):
    """
    Resolve per-video sampling settings.

    Priority:
    1) Explicit --video-fps / --video-num-frames
    2) Duration-adaptive num_frames (default)
    3) Legacy processor defaults when adaptive is disabled
    """
    if args.video_fps is not None:
        return {
            "mode": "manual_fps",
            "video_fps": float(args.video_fps),
            "video_num_frames": None,
            "duration_sec": None,
            "estimated_parts": None,
        }

    if args.video_num_frames is not None:
        return {
            "mode": "manual_num_frames",
            "video_fps": None,
            "video_num_frames": int(args.video_num_frames),
            "duration_sec": None,
            "estimated_parts": None,
        }

    if args.disable_adaptive_video_sampling:
        return {
            "mode": "processor_default",
            "video_fps": None,
            "video_num_frames": None,
            "duration_sec": None,
            "estimated_parts": None,
        }

    duration_sec = probe_video_duration_seconds(video_path)
    if duration_sec is None:
        fallback_frames = max(1, int(args.adaptive_fallback_frames))
        return {
            "mode": "adaptive_fallback",
            "video_fps": None,
            "video_num_frames": fallback_frames,
            "duration_sec": None,
            "estimated_parts": None,
        }

    part_seconds = max(1e-6, float(args.adaptive_part_seconds))
    est_parts = max(1, int(round(duration_sec / part_seconds)))

    frames_from_duration = int(math.ceil(duration_sec * float(args.adaptive_base_fps)))
    frames_from_parts = int(est_parts * int(args.adaptive_frames_per_part))

    min_frames = max(1, int(args.adaptive_min_frames))
    max_frames = max(min_frames, int(args.adaptive_max_frames))
    target_frames = max(min_frames, frames_from_duration, frames_from_parts)
    target_frames = min(max_frames, target_frames)

    return {
        "mode": "adaptive_duration",
        "video_fps": None,
        "video_num_frames": int(target_frames),
        "duration_sec": float(duration_sec),
        "estimated_parts": int(est_parts),
    }


def resolve_video_sampling_for_paths(video_paths, args):
    """Resolve one shared processor sampling config for one or more videos.

    The processor accepts a single videos_kwargs block for all videos in the
    turn. For material evaluation, callers should usually pass
    --video-num-frames explicitly so each short video is uniformly sampled with
    the same fixed frame count. If adaptive sampling is left enabled, use the
    maximum frame budget across all provided videos.
    """
    paths = [Path(p) for p in video_paths or [] if p]
    if not paths:
        return sampling_not_applicable()
    if len(paths) == 1:
        sampling = resolve_video_sampling(paths[0], args)
        sampling["video_paths_count"] = 1
        sampling["durations_sec"] = [sampling.get("duration_sec")]
        return sampling

    per_video = [resolve_video_sampling(path, args) for path in paths]
    first = dict(per_video[0])
    first["video_paths_count"] = len(paths)
    first["durations_sec"] = [item.get("duration_sec") for item in per_video]
    first["estimated_parts_per_video"] = [item.get("estimated_parts") for item in per_video]
    if args.video_num_frames is not None:
        first["mode"] = "manual_num_frames_multi_video"
        first["video_num_frames"] = int(args.video_num_frames)
        first["video_fps"] = None
        return first
    if args.video_fps is not None:
        first["mode"] = "manual_fps_multi_video"
        first["video_fps"] = float(args.video_fps)
        first["video_num_frames"] = None
        return first
    if args.disable_adaptive_video_sampling:
        first["mode"] = "processor_default_multi_video"
        first["video_fps"] = None
        first["video_num_frames"] = None
        return first
    budgets = [item.get("video_num_frames") for item in per_video if item.get("video_num_frames") is not None]
    first["mode"] = "adaptive_duration_multi_video"
    first["video_fps"] = None
    first["video_num_frames"] = max(budgets) if budgets else int(args.adaptive_fallback_frames)
    first["duration_sec"] = max(
        [float(x) for x in first["durations_sec"] if x is not None],
        default=None,
    )
    first["estimated_parts"] = None
    return first


def render_prompt_text(text, task):
    replacements = {
        "reference_description": task.get("reference_description"),
        "object_id": task.get("object_id"),
        "sample_id": task.get("sample_id"),
        "method": task.get("method"),
        "dataset": task.get("dataset"),
        "part_id": task.get("part_id"),
        "material_parameters": task.get("material_parameters"),
    }
    rendered = str(text)
    for key, value in replacements.items():
        if value is not None:
            rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def build_user_message(
    turn,
    video_path,
    video_paths,
    paired_image_path,
    view_image_paths,
    mask_image_paths,
    affordance_view_paths,
    task=None,
):
    content = []
    for static_image in turn.get("static_image_paths", []):
        content.append({"type": "image", "image": static_image})
    if turn.get("view_images_from_input", False):
        for view_image in view_image_paths:
            content.append({"type": "image", "image": view_image})
    if turn.get("mask_images_from_input", False):
        for mask_image in mask_image_paths:
            content.append({"type": "image", "image": mask_image})
    if turn.get("image") is not None:
        content.append({"type": "image", "image": turn["image"]})
    if turn.get("image_from_input", False):
        content.append({"type": "image", "image": paired_image_path})
    if turn.get("affordance_views_from_input", False):
        for affordance_image in affordance_view_paths:
            content.append({"type": "image", "image": affordance_image})
    if turn.get("video_from_input", False):
        content.append({"type": "video", "video": str(video_path)})
    if turn.get("videos_from_input", False):
        for item_video_path in video_paths or []:
            content.append({"type": "video", "video": str(item_video_path)})
    content.append({"type": "text", "text": render_prompt_text(turn["text"], task or {})})
    return {"role": "user", "content": content}


def debug_tensor_shapes(label, inputs):
    parts = []
    for key in sorted(inputs.keys()):
        value = inputs[key]
        if isinstance(value, torch.Tensor):
            detail = f"{key}.shape={tuple(value.shape)} dtype={value.dtype}"
            if key.endswith("grid_thw"):
                try:
                    detail += f" values={value.detach().cpu().tolist()}"
                except Exception:
                    pass
            parts.append(detail)
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}.len={len(value)}")
        else:
            parts.append(f"{key}.type={type(value).__name__}")
    print(f"[Input Debug] {label} " + " | ".join(parts), flush=True)


def run_single_inference(
    model,
    processor,
    messages,
    max_new_tokens,
    cache_state,
    video_fps=None,
    video_num_frames=None,
    debug_input_shapes=False,
    debug_label="",
):
    template_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    processor_kwargs = {}
    videos_kwargs = {}
    if video_fps is not None:
        videos_kwargs["fps"] = float(video_fps)
        videos_kwargs["num_frames"] = None
    if video_num_frames is not None:
        videos_kwargs["num_frames"] = int(video_num_frames)
        # Override video processor default fps to avoid mutual-exclusion conflict.
        videos_kwargs["fps"] = None
    if videos_kwargs:
        processor_kwargs["videos_kwargs"] = videos_kwargs

    template_kwargs["processor_kwargs"] = processor_kwargs

    full_inputs = processor.apply_chat_template(messages, **template_kwargs)
    full_inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in full_inputs.items()}
    if debug_input_shapes:
        debug_tensor_shapes(debug_label or "full_inputs", full_inputs)

    effective_inputs = dict(full_inputs)
    cache_hit = False

    prev_past = cache_state.get("past_key_values")
    prev_sequence_ids = cache_state.get("sequence_ids")
    if prev_past is not None and prev_sequence_ids is not None:
        cur_ids = full_inputs["input_ids"]
        prev_len = int(prev_sequence_ids.shape[1])
        if cur_ids.shape[1] >= prev_len and torch.equal(cur_ids[:, :prev_len], prev_sequence_ids):
            delta_ids = cur_ids[:, prev_len:]
            if delta_ids.shape[1] > 0:
                effective_inputs = {"input_ids": delta_ids, "past_key_values": prev_past}
                if "attention_mask" in full_inputs:
                    effective_inputs["attention_mask"] = full_inputs["attention_mask"]
                for k, v in full_inputs.items():
                    if k not in {"input_ids", "attention_mask"}:
                        effective_inputs[k] = v
                cache_hit = True
    if debug_input_shapes and cache_hit:
        debug_tensor_shapes((debug_label or "effective_inputs") + " cache_delta", effective_inputs)

    with torch.inference_mode():
        generation_output = model.generate(
            **effective_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            return_dict_in_generate=True,
        )

    prompt_len = int(effective_inputs["input_ids"].shape[1])
    generated_ids_trimmed = generation_output.sequences[:, prompt_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    new_past = getattr(generation_output, "past_key_values", None)
    if new_past is not None:
        if cache_hit:
            sequence_ids = torch.cat([prev_sequence_ids, generation_output.sequences], dim=1)
        else:
            sequence_ids = generation_output.sequences
        cache_state["past_key_values"] = new_past
        cache_state["sequence_ids"] = sequence_ids.detach().clone()
    else:
        cache_state["past_key_values"] = None
        cache_state["sequence_ids"] = None

    if isinstance(output_text, list) and len(output_text) == 1:
        return output_text[0], cache_hit
    return output_text, cache_hit



def _coerce_manifest_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return [str(v) for v in loaded if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]


def _resolve_manifest_path(value, manifest_dir):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    return str(path)


def _existing_manifest_file(value, manifest_dir, label):
    resolved = _resolve_manifest_path(value, manifest_dir)
    if resolved is None:
        return None
    path = Path(resolved)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest {label} not found: {path}")
    return str(path)


def _default_relative_dir(item, sample_id):
    parts = []
    for key in ("metric", "method", "dataset"):
        value = str(item.get(key, "")).strip()
        if value:
            parts.append(_sanitize_for_filename(value))
    parts.append(_sanitize_for_filename(sample_id or "sample"))
    return "/".join(parts)


def load_pairs_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pairs manifest not found: {manifest_path}")
    manifest_dir = manifest_path.parent
    suffix = manifest_path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        raw_items = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    raw_items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {manifest_path}:{line_no}: {exc}") from exc
    elif suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            raw_items = list(csv.DictReader(f))
    else:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            raw_items = data.get("pairs") or data.get("items") or data.get("manifest")
        else:
            raw_items = data

    if not isinstance(raw_items, list):
        raise ValueError("Pairs manifest must be a list, or a dict containing `pairs`/`items`/`manifest`.")

    tasks = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest row {idx} must be an object/dict.")

        video_path = item.get("video_path") or item.get("video")
        video_paths = _coerce_manifest_list(item.get("video_paths"))
        image_path = item.get("image_path") or item.get("condition_image") or item.get("paired_image_path")
        sample_id = item.get("object_id") or item.get("sample_id") or item.get("video_id")
        if not sample_id and video_path:
            sample_id = Path(str(video_path)).parent.name
        if not sample_id:
            sample_id = f"sample_{idx:06d}"

        aff_paths = _coerce_manifest_list(item.get("affordance_view_paths"))
        aff_paths = [_existing_manifest_file(x, manifest_dir, "affordance_view_paths") for x in aff_paths]
        view_paths = _coerce_manifest_list(item.get("view_image_paths"))
        view_paths = [_existing_manifest_file(x, manifest_dir, "view_image_paths") for x in view_paths]
        mask_paths = _coerce_manifest_list(item.get("mask_image_paths"))
        mask_paths = [_existing_manifest_file(x, manifest_dir, "mask_image_paths") for x in mask_paths]

        task = dict(item)
        task["sample_id"] = str(sample_id)
        task["object_id"] = str(item.get("object_id") or sample_id)
        task["video_id"] = str(item.get("video_id") or sample_id)
        task["video_path"] = _existing_manifest_file(video_path, manifest_dir, "video_path") if video_path else None
        task["video_paths"] = [
            _existing_manifest_file(x, manifest_dir, "video_paths") for x in video_paths
        ]
        task["image_path"] = _existing_manifest_file(image_path, manifest_dir, "image_path") if image_path else None
        task["view_image_paths"] = view_paths
        task["mask_image_paths"] = mask_paths
        task["affordance_view_paths"] = aff_paths
        task["relative_dir"] = str(item.get("relative_dir") or _default_relative_dir(item, sample_id))
        tasks.append(task)

    return tasks


def make_video_task(video_path, input_root):
    video_path = Path(video_path)
    rel_parent = video_path.parent.relative_to(input_root)
    return {
        "sample_id": video_path.parent.name,
        "object_id": video_path.parent.name,
        "video_id": video_path.parent.name,
        "video_path": str(video_path),
        "image_path": None,
        "view_image_paths": [],
        "mask_image_paths": [],
        "affordance_view_paths": [],
        "relative_dir": str(rel_parent),
    }


def sampling_not_applicable():
    return {
        "mode": "no_video",
        "video_fps": None,
        "video_num_frames": None,
        "duration_sec": None,
        "estimated_parts": None,
    }


class PairCompletedWithoutModel(Exception):
    """Internal control flow for deterministic non-VLM pair outputs."""


APS_MISSING_ZERO_STATUS_TOKENS = {
    "missing_affordance_heatmap_views",
    "insufficient_affordance_heatmap_views",
}


KPS_MISSING_VIDEO_ZERO_STATUS_TOKENS = {
    "missing_video_path",
    "needs_render_video",
    "missing_aa_joint_video",
    "invalid_video",
    "black_video",
}


MPS_MISSING_VIDEO_ZERO_STATUS_TOKENS = {
    "missing_floor_video",
    "missing_water_video",
    "invalid_floor_video",
    "invalid_water_video",
    "insufficient_material_videos",
}


RENDER_MISSING_ZERO_STATUS_TOKENS = {
    "missing_render_views",
}


DCS_MISSING_ZERO_STATUS_TOKENS = {
    "missing_description_images",
    "missing_description_masks",
}


MIN_VALID_KPS_VIDEO_BYTES = 1024


def manifest_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def manifest_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def aps_missing_affordance_zero_reason(task, required_num_views):
    """Return a reason when an APS row should be deterministically scored as 0.

    This is for generated affordance evidence that is missing because the method
    did not produce heatmaps. Missing condition images remain normal errors.
    """
    metric = str(task.get("metric") or "").strip().lower()
    rel = str(task.get("relative_dir") or "").strip().lower()
    is_aps = metric in {"aps", "affordance", "affordance_scoring"} or rel.startswith("aps/")
    if not is_aps:
        return None

    explicit_zero = manifest_bool(task.get("aps_missing_score_zero"))
    status = str(task.get("status") or "")
    status_tokens = {x.strip() for x in status.split(";") if x.strip()}
    reasons = sorted(status_tokens.intersection(APS_MISSING_ZERO_STATUS_TOKENS))

    paths = task.get("affordance_view_paths") or []
    available = manifest_int(task.get("num_affordance_views_available"), len(paths))
    required = int(required_num_views or 0)
    if required > 0 and len(paths) < required and available < required:
        reasons.append(f"available_affordance_views={available}<required={required}")
    if not paths and (explicit_zero or reasons or not manifest_bool(task.get("ready"))):
        reasons.append("no_affordance_view_paths")

    if explicit_zero or reasons:
        return ";".join(dict.fromkeys(reasons)) or "missing_affordance_evidence"
    return None


def missing_affordance_zero_output(task, reason):
    object_id = task.get("object_id") or task.get("sample_id") or "unknown_object"
    return {
        "task": "affordance_scoring",
        "object_summary": f"Object {object_id}; generated affordance evidence is missing.",
        "expected_high_affordance_regions": [],
        "major_alignment": [],
        "major_misalignment": ["missing generated affordance heatmap"],
        "APS": 0.0,
        "verdict": "bad",
        "reasoning": f"APS is set to 0 because generated affordance evidence is missing: {reason}",
    }


def is_kps_task(task):
    metric = str(task.get("metric") or "").strip().lower()
    rel = str(task.get("relative_dir") or "").strip().lower()
    return metric in {"kps", "vaps", "vaps_scoring", "kinematic"} or rel.startswith("kps/")


def kps_missing_video_zero_reason(task):
    """Return a reason when a KPS row should be deterministically scored as 0.

    This is for active KPS rows whose generated articulation video evidence is
    missing or known invalid. Missing condition images remain normal errors.
    """
    if not is_kps_task(task):
        return None

    explicit_zero = manifest_bool(task.get("kps_missing_score_zero"))
    status = str(task.get("status") or "")
    status_tokens = {x.strip() for x in status.split(";") if x.strip()}
    reasons = sorted(status_tokens.intersection(KPS_MISSING_VIDEO_ZERO_STATUS_TOKENS))

    video_path = task.get("video_path")
    if not video_path:
        reasons.append("no_video_path")
    else:
        video_file = Path(str(video_path))
        if not video_file.is_file():
            reasons.append(f"video_file_not_found:{video_path}")
        else:
            try:
                video_size = video_file.stat().st_size
            except OSError:
                video_size = -1
            if video_size < MIN_VALID_KPS_VIDEO_BYTES:
                reasons.append(f"video_file_too_small_bytes={video_size}")

    if explicit_zero or reasons:
        return ";".join(dict.fromkeys(reasons)) or "missing_video_evidence"
    return None


def missing_kps_video_zero_output(task, reason):
    object_id = task.get("object_id") or task.get("sample_id") or "unknown_object"
    method = task.get("method") or "unknown_method"
    dataset = task.get("dataset") or "unknown_dataset"
    return {
        "task": "vaps_scoring",
        "object_summary": (
            f"KPS could not be evaluated for {method}/{dataset}/{object_id} because no valid "
            "articulation video evidence is available."
        ),
        "parts": [],
        "revealed_entities": [],
        "aggregates": {
            "num_parts_total": 0,
            "num_parts_scored": 0,
            "num_revealed_entities": 0,
            "S_prior": None,
            "S_reveal": None,
            "S_global": 0,
            "VAPS": 0.0,
            "KPS": 0.0,
            "verdict": "failed_missing_video",
        },
        "reasoning": f"KPS is set to 0 because generated articulation video evidence is missing: {reason}",
        "auto_scored_missing_video": True,
    }


def is_mps_task(task):
    metric = str(task.get("metric") or "").strip().lower()
    rel = str(task.get("relative_dir") or "").strip().lower()
    return metric in {"mps", "material", "material_scoring"} or rel.startswith("mps/")


def mps_missing_video_zero_reason(task):
    """Return a reason when material videos are missing and MPS should be 0."""
    if not is_mps_task(task):
        return None
    explicit_zero = manifest_bool(task.get("mps_missing_score_zero"))
    status = str(task.get("status") or "")
    status_tokens = {x.strip() for x in status.split(";") if x.strip()}
    reasons = sorted(status_tokens.intersection(MPS_MISSING_VIDEO_ZERO_STATUS_TOKENS))
    video_paths = [p for p in task.get("video_paths") or [] if p]
    if len(video_paths) < 2:
        reasons.append(f"available_material_videos={len(video_paths)}<required=2")
    for idx, video_path in enumerate(video_paths[:2]):
        video_file = Path(str(video_path))
        label = "floor" if idx == 0 else "water"
        if not video_file.is_file():
            reasons.append(f"{label}_video_file_not_found:{video_path}")
        else:
            try:
                video_size = video_file.stat().st_size
            except OSError:
                video_size = -1
            if video_size < MIN_VALID_KPS_VIDEO_BYTES:
                reasons.append(f"{label}_video_file_too_small_bytes={video_size}")
    if explicit_zero or reasons:
        return ";".join(dict.fromkeys(reasons)) or "missing_material_video_evidence"
    return None


def missing_material_videos_zero_output(task, reason):
    object_id = task.get("object_id") or task.get("sample_id") or "unknown_object"
    method = task.get("method") or "unknown_method"
    dataset = task.get("dataset") or "unknown_dataset"
    return {
        "task": "material_scoring",
        "used_high_confidence_priors": [],
        "given_material_parameters_summary": str(task.get("material_parameters") or ""),
        "part_matching_between_image_and_video": "not evaluable because one or both required material videos are missing",
        "youngs_modulus_evaluation": {
            "score": 1,
            "confidence": "low",
            "reason": f"Ground-impact material evidence is missing or invalid for {method}/{dataset}/{object_id}.",
        },
        "poisson_ratio_evaluation": {
            "score": 1,
            "confidence": "low",
            "reason": "Compression/lateral deformation evidence is missing or invalid.",
        },
        "density_evaluation": {
            "score": 1,
            "confidence": "low",
            "reason": "Water-entry/buoyancy material evidence is missing or invalid.",
        },
        "MPS": 0.0,
        "overall_confidence": "low",
        "overall_reason": f"MPS is set to 0 because required material videos are missing: {reason}",
        "video_artifact_notes": "missing required material simulation video evidence",
        "uncertainty": "all material video evidence is incomplete",
        "auto_scored_missing_material_videos": True,
    }


def render_metric_name(task):
    metric = str(task.get("metric") or "").strip().lower()
    rel = str(task.get("relative_dir") or "").strip().lower()
    if metric in {"rqs", "quality", "render_quality", "render_quality_score"} or rel.startswith("rqs/"):
        return "RQS"
    if metric in {"mcs", "consistency", "multi_view", "multi_view_consistency"} or rel.startswith("mcs/"):
        return "MCS"
    return None


def render_missing_views_zero_reason(task):
    """Return a reason when an RQS/MCS row has no rendered views and should score 0."""
    metric = render_metric_name(task)
    if metric is None:
        return None

    paths = task.get("view_image_paths") or []
    if paths:
        return None

    explicit_zero = manifest_bool(task.get("render_missing_score_zero"))
    status = str(task.get("status") or "")
    status_tokens = {x.strip() for x in status.split(";") if x.strip()}
    reasons = sorted(status_tokens.intersection(RENDER_MISSING_ZERO_STATUS_TOKENS))
    available = manifest_int(task.get("num_render_views_available"), 0)
    required = manifest_int(task.get("num_render_views_required"), 0)
    if available <= 0:
        reasons.append("available_render_views=0")
    if required > 0:
        reasons.append(f"required_render_views={required}")
    if explicit_zero:
        reasons.append("render_missing_score_zero=true")
    if not manifest_bool(task.get("ready")):
        reasons.append("ready=false")

    return ";".join(dict.fromkeys(reasons)) or "missing_render_views"


def missing_render_views_zero_output(task, metric, reason):
    object_id = task.get("object_id") or task.get("sample_id") or "unknown_object"
    method = task.get("method") or "unknown_method"
    dataset = task.get("dataset") or "unknown_dataset"
    base = {
        "score": 1,
        "reason": (
            f"{metric} is set to 0 for {method}/{dataset}/{object_id} because no rendered "
            f"view images are available: {reason}"
        ),
        "auto_scored_missing_render_views": True,
    }
    if metric == "RQS":
        base.update(
            {
                "clarity": "not evaluable because no rendered views are available",
                "edge_sharpness": "not evaluable because no rendered views are available",
                "detail_visibility": "not evaluable because no rendered views are available",
                "texture_clarity": "not evaluable because no rendered views are available",
                "artifacts": "not evaluable because no rendered views are available",
            }
        )
    else:
        base.update(
            {
                "global_view_consistency": "not evaluable because no rendered views are available",
                "view_specific_failures": "not evaluable because no rendered views are available",
                "surface_appearance_coherence": "not evaluable because no rendered views are available",
                "failure_views": [],
            }
        )
    return base


def is_dcs_task(task):
    metric = str(task.get("metric") or "").strip().lower()
    rel = str(task.get("relative_dir") or "").strip().lower()
    return metric in {"dcs", "description", "description_scoring"} or rel.startswith("dcs/")


def dcs_missing_images_zero_reason(task):
    """Return a reason when a DCS row has no part/render images and should score 0."""
    if not is_dcs_task(task):
        return None
    paths = task.get("view_image_paths") or []
    mask_paths = task.get("mask_image_paths") or []
    has_mask_field = "mask_image_paths" in task or "num_description_mask_views_available" in task
    if paths and (not has_mask_field or mask_paths):
        return None

    explicit_zero = manifest_bool(task.get("dcs_missing_score_zero"))
    status = str(task.get("status") or "")
    status_tokens = {x.strip() for x in status.split(";") if x.strip()}
    reasons = sorted(status_tokens.intersection(DCS_MISSING_ZERO_STATUS_TOKENS))
    available = manifest_int(task.get("num_description_views_available"), 0)
    masks_available = manifest_int(task.get("num_description_mask_views_available"), 0)
    required = manifest_int(task.get("num_description_views_required"), 0)
    if available <= 0:
        reasons.append("available_description_views=0")
    if has_mask_field and masks_available <= 0:
        reasons.append("available_description_mask_views=0")
    if required > 0:
        reasons.append(f"required_description_views={required}")
    if explicit_zero:
        reasons.append("dcs_missing_score_zero=true")
    if not manifest_bool(task.get("ready")):
        reasons.append("ready=false")

    if explicit_zero or reasons:
        return ";".join(dict.fromkeys(reasons)) or "missing_description_images"
    return None


def missing_description_images_zero_output(task, reason):
    object_id = task.get("object_id") or task.get("sample_id") or "unknown_object"
    method = task.get("method") or "unknown_method"
    dataset = task.get("dataset") or "unknown_dataset"
    reference = task.get("reference_description") or ""
    return {
        "task": "description_mask_scoring",
        "score": 1,
        "DCS": 0,
        "alignment_level": "poor",
        "alignment_score": 0,
        "alignment_reason": (
            f"No evaluable generated description/mask evidence is available for "
            f"{method}/{dataset}/{object_id}: {reason}"
        ),
        "precision_level": "poor",
        "precision_score": 0,
        "precision_reason": (
            f"No evaluable generated description/mask evidence is available for "
            f"{method}/{dataset}/{object_id}: {reason}"
        ),
        "reason": (
            f"DCS is set to 0 for {method}/{dataset}/{object_id} because no highlighted "
            f"description images are available: {reason}"
        ),
        "part_identity": "not evaluable because no highlighted description images are available",
        "shape_structure": "not evaluable because no highlighted description images are available",
        "color_material": "not evaluable because no highlighted description images are available",
        "texture_detail": "not evaluable because no highlighted description images are available",
        "spatial_relation": "not evaluable because no highlighted description images are available",
        "missing_or_wrong_generated_features": "missing generated description image evidence",
        "uncertainty": "all visual evidence is missing",
        "reference_description": reference,
        "auto_scored_missing_description_images": True,
    }


def auto_score_reason(task, required_num_views):
    render_reason = render_missing_views_zero_reason(task)
    if render_reason is not None:
        return {"metric": render_metric_name(task), "reason": render_reason}
    dcs_reason = dcs_missing_images_zero_reason(task)
    if dcs_reason is not None:
        return {"metric": "DCS", "reason": dcs_reason}
    aps_reason = aps_missing_affordance_zero_reason(task, required_num_views)
    if aps_reason is not None:
        return {"metric": "APS", "reason": aps_reason}
    kps_reason = kps_missing_video_zero_reason(task)
    if kps_reason is not None:
        return {"metric": "KPS", "reason": kps_reason}
    mps_reason = mps_missing_video_zero_reason(task)
    if mps_reason is not None:
        return {"metric": "MPS", "reason": mps_reason}
    return None


def video_scoring_turn(turns):
    for turn_idx, turn in enumerate(turns):
        if turn.get("video_from_input", False) or turn.get("videos_from_input", False):
            return turn_idx, turn
    return max(0, len(turns) - 1), turns[-1]


def main():
    args = parse_args()
    if args.single_image_path is not None and args.single_seq_path is None and args.pairs_manifest is None:
        raise ValueError("`--single-image-path` must be used with `--single-seq-path` or `--pairs-manifest`.")
    if args.video_fps is not None and args.video_num_frames is not None:
        raise ValueError("`--video-fps` and `--video-num-frames` are mutually exclusive.")
    if args.adaptive_base_fps <= 0:
        raise ValueError("`--adaptive-base-fps` must be > 0.")
    if args.adaptive_frames_per_part <= 0:
        raise ValueError("`--adaptive-frames-per-part` must be > 0.")
    if args.adaptive_part_seconds <= 0:
        raise ValueError("`--adaptive-part-seconds` must be > 0.")
    if args.adaptive_min_frames <= 0:
        raise ValueError("`--adaptive-min-frames` must be > 0.")
    if args.adaptive_max_frames < args.adaptive_min_frames:
        raise ValueError("`--adaptive-max-frames` must be >= `--adaptive-min-frames`.")
    if args.adaptive_fallback_frames <= 0:
        raise ValueError("`--adaptive-fallback-frames` must be > 0.")
    if args.affordance_views_num_images == 0:
        raise ValueError("`--affordance-views-num-images` must be > 0 or < 0 (for all).")
    if args.pairs_manifest is not None and args.single_seq_path is not None:
        raise ValueError("Use either `--pairs-manifest` or `--single-seq-path`, not both.")

    turns = load_turns(args.prompts_file)
    uses_image_from_input = any(t.get("image_from_input", False) for t in turns)
    uses_view_images_from_input = any(t.get("view_images_from_input", False) for t in turns)
    uses_mask_images_from_input = any(t.get("mask_images_from_input", False) for t in turns)
    uses_affordance_views_from_input = any(t.get("affordance_views_from_input", False) for t in turns)
    uses_video_from_input = any(t.get("video_from_input", False) for t in turns)
    uses_videos_from_input = any(t.get("videos_from_input", False) for t in turns)
    if args.single_image_path is not None and not uses_image_from_input:
        raise ValueError(
            "`--single-image-path` is provided, but prompts have no `image_from_input: true` turn."
        )
    if uses_image_from_input and args.single_image_path is None and args.image_input_dir is None and args.pairs_manifest is None:
        raise ValueError(
            "Prompts use `image_from_input: true`, but no image source was provided. "
            "Set `--image-input-dir`, `--single-image-path`, or per-row manifest image_path."
        )
    if uses_affordance_views_from_input and args.affordance_views_input_dir is None and args.pairs_manifest is None:
        raise ValueError(
            "Prompts use `affordance_views_from_input: true`, but no affordance source was provided. "
            "Set `--affordance-views-input-dir` or per-row manifest affordance_view_paths."
        )
    if uses_view_images_from_input and args.pairs_manifest is None:
        raise ValueError(
            "Prompts use `view_images_from_input: true`; provide per-row manifest view_image_paths "
            "through `--pairs-manifest`."
        )
    if uses_mask_images_from_input and args.pairs_manifest is None:
        raise ValueError(
            "Prompts use `mask_images_from_input: true`; provide per-row manifest mask_image_paths "
            "through `--pairs-manifest`."
        )

    input_root = None
    tasks = []
    if args.pairs_manifest is not None:
        tasks = load_pairs_manifest(args.pairs_manifest)
        if not tasks:
            raise RuntimeError(f"No rows found in pairs manifest: {args.pairs_manifest}")
        if args.input_dir is not None:
            input_root = Path(args.input_dir)
        print(f"Loaded manifest pairs: {len(tasks)}", flush=True)
    else:
        if args.input_dir is None:
            raise ValueError("`--input-dir` is required unless `--pairs-manifest` is used.")
        input_root, videos = discover_videos(args.input_dir, args.video_glob)
        if not videos:
            raise RuntimeError(f"No videos found with pattern `{args.video_glob}` under: {input_root}")
        if args.single_seq_path is not None:
            single_video = resolve_single_video(input_root=input_root, videos=videos, single_seq_path=args.single_seq_path)
            videos = [single_video]
            print(f"Single sequence mode enabled: {args.single_seq_path}", flush=True)
            print(f"Resolved single video: {single_video}", flush=True)
        tasks = [make_video_task(video_path=v, input_root=input_root) for v in videos]

    if args.single_image_path is not None and len(tasks) != 1:
        raise ValueError("`--single-image-path` is only supported when exactly one pair is selected.")

    print(f"Discovered pairs: {len(tasks)}")
    print(f"Loaded turns: {len(turns)}")

    auto_score_reasons = [auto_score_reason(task, args.affordance_views_num_images) for task in tasks]
    requires_model = any(reason is None for reason in auto_score_reasons)

    model_source = None
    config = None
    processor = None
    model = None
    loading_info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    if requires_model:
        lock_path = acquire_model_run_lock(args.model_id)
        print(f"[Init] Acquired run lock: {lock_path}", flush=True)

        model_source = resolve_model_source(args.model_id, args.local_files_only)
        print(f"[Init] Model source: {model_source}", flush=True)
        print(f"[Init] Loading config...", flush=True)
        config = AutoConfig.from_pretrained(model_source, local_files_only=args.local_files_only)
        print(f"[Init] Loading processor...", flush=True)
        processor = AutoProcessor.from_pretrained(model_source, local_files_only=args.local_files_only)
        print(f"[Init] Loading model weights...", flush=True)
        model, loading_info = AutoModelForImageTextToText.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_loading_info=True,
            local_files_only=args.local_files_only,
        )
        print("[Init] Model ready.", flush=True)
    else:
        print("[Init] All pairs can be auto-scored without loading the VLM model.", flush=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / args.model_id / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp_utc": run_ts,
        "run_id": f"run_{run_ts}",
        "run_dir": str(run_dir),
        "model_id": args.model_id,
        "model_source": model_source,
        "config_model_type": getattr(config, "model_type", None) if config is not None else None,
        "config_architectures": getattr(config, "architectures", None) if config is not None else None,
        "model_type": type(model).__name__ if model is not None else None,
        "loading_info": {
            "missing_keys": loading_info.get("missing_keys", []),
            "unexpected_keys": loading_info.get("unexpected_keys", []),
            "mismatched_keys": loading_info.get("mismatched_keys", []),
            "error_msgs": loading_info.get("error_msgs", []),
        },
        "args": {
            "input_dir": str(input_root) if input_root else None,
            "pairs_manifest": str(Path(args.pairs_manifest)) if args.pairs_manifest else None,
            "single_seq_path": str(Path(args.single_seq_path)) if args.single_seq_path else None,
            "single_image_path": str(Path(args.single_image_path)) if args.single_image_path else None,
            "prompts_file": str(Path(args.prompts_file)),
            "max_new_tokens": args.max_new_tokens,
            "video_fps": args.video_fps,
            "video_num_frames": args.video_num_frames,
            "disable_adaptive_video_sampling": args.disable_adaptive_video_sampling,
            "adaptive_base_fps": args.adaptive_base_fps,
            "adaptive_frames_per_part": args.adaptive_frames_per_part,
            "adaptive_part_seconds": args.adaptive_part_seconds,
            "adaptive_min_frames": args.adaptive_min_frames,
            "adaptive_max_frames": args.adaptive_max_frames,
            "adaptive_fallback_frames": args.adaptive_fallback_frames,
            "video_glob": args.video_glob,
            "image_input_dir": str(Path(args.image_input_dir)) if args.image_input_dir else None,
            "image_name": args.image_name,
            "affordance_views_input_dir": (
                str(Path(args.affordance_views_input_dir)) if args.affordance_views_input_dir else None
            ),
            "affordance_views_subdir": args.affordance_views_subdir,
            "affordance_views_glob": args.affordance_views_glob,
            "affordance_views_num_images": args.affordance_views_num_images,
            "output_root": str(Path(args.output_root)),
            "local_files_only": args.local_files_only,
            "debug_input_shapes": args.debug_input_shapes,
            "do_sample": False,
            "temperature": None,
            "conversation_mode": "yaml_driven_turn_modalities",
        },
        "num_pairs": len(tasks),
        "num_turns": len(turns),
        "num_tasks": len(tasks) * len(turns),
        "requires_model": requires_model,
        "num_auto_scored_missing_affordance_planned": sum(
            1 for item in auto_score_reasons if item is not None and item["metric"] == "APS"
        ),
        "num_auto_scored_missing_video_planned": sum(
            1 for item in auto_score_reasons if item is not None and item["metric"] == "KPS"
        ),
        "num_auto_scored_missing_render_views_planned": sum(
            1 for item in auto_score_reasons if item is not None and item["metric"] in {"RQS", "MCS"}
        ),
        "num_auto_scored_missing_description_images_planned": sum(
            1 for item in auto_score_reasons if item is not None and item["metric"] == "DCS"
        ),
        "num_auto_scored_missing_material_videos_planned": sum(
            1 for item in auto_score_reasons if item is not None and item["metric"] == "MPS"
        ),
    }

    pair_index = []
    total_turn_errors = 0
    total_pair_errors = 0
    total_auto_scored_missing_affordance = 0
    total_auto_scored_missing_video = 0
    total_auto_scored_missing_render_views = 0
    total_auto_scored_missing_description_images = 0
    total_auto_scored_missing_material_videos = 0
    single_image_override = None
    if args.single_image_path is not None:
        single_image_override = resolve_single_image_path(args.single_image_path)
        print(f"Single image override enabled: {single_image_override}", flush=True)

    pbar = tqdm(tasks, total=len(tasks), desc="Pairs", unit="pair")
    for task in pbar:
        video_path = Path(task["video_path"]) if task.get("video_path") else None
        video_paths = [Path(p) for p in task.get("video_paths", []) if p]
        video_id = str(task.get("video_id") or task.get("sample_id") or "sample")
        rel_text = str(task.get("relative_dir") or _default_relative_dir(task, video_id))
        video_rel_parent = Path(rel_text)
        pbar.set_postfix_str(video_id)
        pair_start = time.perf_counter()

        pair_out_dir = run_dir.joinpath(*video_rel_parent.parts)
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        pair_json_path = pair_out_dir / "result.json"

        pair_payload = {
            "run_id": run_meta["run_id"],
            "video_path": str(video_path) if video_path else None,
            "video_paths": [str(p) for p in video_paths],
            "video_id": video_id,
            "video_relative_dir": str(video_rel_parent),
            "paired_image_path": None,
            "view_image_paths": [],
            "mask_image_paths": [],
            "affordance_view_paths": [],
            "benchmark_context": task,
            "sampling": None,
            "turns_template": turns,
            "results": [],
            "pair_error": None,
            "elapsed_sec": None,
        }

        try:
            auto_item = auto_score_reason(task, args.affordance_views_num_images)
            if auto_item is not None and auto_item["metric"] in {"RQS", "MCS"}:
                pair_payload["sampling"] = sampling_not_applicable()
                pair_payload["view_image_paths"] = []
                pair_payload["auto_scored_missing_render_views"] = True
                pair_payload["missing_render_views_reason"] = auto_item["reason"]
                total_auto_scored_missing_render_views += 1
                print(
                    f"[AutoScore] video={video_id} metric={auto_item['metric']} score=0 "
                    f"reason={auto_item['reason']}",
                    flush=True,
                )
                for turn_idx, turn in enumerate(turns):
                    output_json = missing_render_views_zero_output(task, auto_item["metric"], auto_item["reason"])
                    pair_payload["results"].append(
                        {
                            "turn_id": turn["id"],
                            "turn_index": turn_idx,
                            "prompt_ref_id": turn["id"],
                            "input_modalities": {
                                "image": bool(turn.get("image") is not None or turn.get("image_from_input", False)),
                                "static_images": len(turn.get("static_image_paths", [])),
                                "view_images": 0,
                                "affordance_view_images": 0,
                                "video": False,
                            },
                            "output": json.dumps(output_json, ensure_ascii=False),
                            "error": None,
                            "kv_cache_hit": False,
                            "elapsed_sec": 0.0,
                            "auto_scored": True,
                            "auto_score_reason": auto_item["reason"],
                        }
                    )
                raise PairCompletedWithoutModel()

            if auto_item is not None and auto_item["metric"] == "DCS":
                pair_payload["sampling"] = sampling_not_applicable()
                pair_payload["view_image_paths"] = []
                pair_payload["auto_scored_missing_description_images"] = True
                pair_payload["missing_description_images_reason"] = auto_item["reason"]
                total_auto_scored_missing_description_images += 1
                print(
                    f"[AutoScore] video={video_id} metric=DCS score=0 reason={auto_item['reason']}",
                    flush=True,
                )
                for turn_idx, turn in enumerate(turns):
                    output_json = missing_description_images_zero_output(task, auto_item["reason"])
                    pair_payload["results"].append(
                        {
                            "turn_id": turn["id"],
                            "turn_index": turn_idx,
                            "prompt_ref_id": turn["id"],
                            "input_modalities": {
                                "image": bool(turn.get("image") is not None or turn.get("image_from_input", False)),
                                "static_images": len(turn.get("static_image_paths", [])),
                                "view_images": 0,
                                "affordance_view_images": 0,
                                "video": False,
                            },
                            "output": json.dumps(output_json, ensure_ascii=False),
                            "error": None,
                            "kv_cache_hit": False,
                            "elapsed_sec": 0.0,
                            "auto_scored": True,
                            "auto_score_reason": auto_item["reason"],
                        }
                    )
                raise PairCompletedWithoutModel()

            if auto_item is not None and auto_item["metric"] == "KPS":
                paired_image_path = None
                if uses_image_from_input:
                    if single_image_override is not None:
                        paired_image_path = str(single_image_override)
                    elif task.get("image_path"):
                        paired_image_path = str(Path(task["image_path"]))
                    elif task.get("condition_image"):
                        paired_image_path = str(Path(task["condition_image"]))
                pair_payload["paired_image_path"] = paired_image_path
                pair_payload["sampling"] = sampling_not_applicable()
                pair_payload["auto_scored_missing_video"] = True
                pair_payload["missing_video_zero_reason"] = auto_item["reason"]
                total_auto_scored_missing_video += 1
                print(
                    f"[AutoScore] video={video_id} metric=KPS score=0 reason={auto_item['reason']}",
                    flush=True,
                )
                turn_idx, turn = video_scoring_turn(turns)
                output_json = missing_kps_video_zero_output(task, auto_item["reason"])
                pair_payload["results"].append(
                    {
                        "turn_id": turn["id"],
                        "turn_index": turn_idx,
                        "prompt_ref_id": turn["id"],
                        "input_modalities": {
                            "image": bool(turn.get("image") is not None or turn.get("image_from_input", False)),
                            "static_images": len(turn.get("static_image_paths", [])),
                            "view_images": 0,
                            "affordance_view_images": 0,
                            "video": False,
                        },
                        "output": json.dumps(output_json, ensure_ascii=False),
                        "error": None,
                        "kv_cache_hit": False,
                        "elapsed_sec": 0.0,
                        "auto_scored": True,
                        "auto_score_reason": auto_item["reason"],
                    }
                )
                raise PairCompletedWithoutModel()

            if auto_item is not None and auto_item["metric"] == "MPS":
                paired_image_path = None
                if uses_image_from_input:
                    if single_image_override is not None:
                        paired_image_path = str(single_image_override)
                    elif task.get("image_path"):
                        paired_image_path = str(Path(task["image_path"]))
                    elif task.get("condition_image"):
                        paired_image_path = str(Path(task["condition_image"]))
                pair_payload["paired_image_path"] = paired_image_path
                pair_payload["sampling"] = sampling_not_applicable()
                pair_payload["auto_scored_missing_material_videos"] = True
                pair_payload["missing_material_videos_reason"] = auto_item["reason"]
                total_auto_scored_missing_material_videos += 1
                print(
                    f"[AutoScore] video={video_id} metric=MPS score=0 reason={auto_item['reason']}",
                    flush=True,
                )
                turn_idx, turn = video_scoring_turn(turns)
                output_json = missing_material_videos_zero_output(task, auto_item["reason"])
                pair_payload["results"].append(
                    {
                        "turn_id": turn["id"],
                        "turn_index": turn_idx,
                        "prompt_ref_id": turn["id"],
                        "input_modalities": {
                            "image": bool(turn.get("image") is not None or turn.get("image_from_input", False)),
                            "static_images": len(turn.get("static_image_paths", [])),
                            "view_images": 0,
                            "affordance_view_images": 0,
                            "video": False,
                            "videos": 0,
                        },
                        "output": json.dumps(output_json, ensure_ascii=False),
                        "error": None,
                        "kv_cache_hit": False,
                        "elapsed_sec": 0.0,
                        "auto_scored": True,
                        "auto_score_reason": auto_item["reason"],
                    }
                )
                raise PairCompletedWithoutModel()

            if uses_video_from_input and video_path is None:
                raise ValueError(f"Pair `{video_id}` requires video_from_input but has no video_path.")
            if video_path is not None and not video_path.is_file():
                raise FileNotFoundError(f"Video file not found for pair `{video_id}`: {video_path}")
            if uses_videos_from_input:
                if not video_paths:
                    raise ValueError(f"Pair `{video_id}` requires videos_from_input but has no video_paths.")
                for item_video_path in video_paths:
                    if not item_video_path.is_file():
                        raise FileNotFoundError(f"Video file not found for pair `{video_id}`: {item_video_path}")

            conversation = []
            cache_state = {"past_key_values": None, "sequence_ids": None}
            paired_image_path = None
            view_image_paths = []
            mask_image_paths = []
            affordance_view_paths = []
            if uses_image_from_input:
                if single_image_override is not None:
                    paired_image_path = single_image_override
                elif task.get("image_path"):
                    paired_image_path = str(Path(task["image_path"]))
                else:
                    if video_path is None or input_root is None:
                        raise ValueError(
                            f"Pair `{video_id}` has no manifest image_path and cannot use mirrored image lookup."
                        )
                    paired_image_path = resolve_paired_image_path(
                        video_path=video_path,
                        input_root=input_root,
                        image_input_root=args.image_input_dir,
                        image_name=args.image_name,
                    )
            pair_payload["paired_image_path"] = paired_image_path
            if uses_view_images_from_input:
                if task.get("view_image_paths"):
                    view_image_paths = [str(Path(p)) for p in task["view_image_paths"]]
                else:
                    raise ValueError(f"Pair `{video_id}` has no manifest view_image_paths.")
            pair_payload["view_image_paths"] = view_image_paths
            if uses_mask_images_from_input:
                if task.get("mask_image_paths"):
                    mask_image_paths = [str(Path(p)) for p in task["mask_image_paths"]]
                else:
                    raise ValueError(f"Pair `{video_id}` has no manifest mask_image_paths.")
            pair_payload["mask_image_paths"] = mask_image_paths
            if auto_item is not None and auto_item["metric"] == "APS":
                sampling = sampling_not_applicable()
                pair_payload["sampling"] = sampling
                pair_payload["affordance_view_paths"] = []
                pair_payload["auto_scored_missing_affordance"] = True
                pair_payload["missing_affordance_reason"] = auto_item["reason"]
                total_auto_scored_missing_affordance += 1
                print(
                    f"[AutoScore] video={video_id} metric=APS score=0 reason={auto_item['reason']}",
                    flush=True,
                )
                for turn_idx, turn in enumerate(turns):
                    output_json = missing_affordance_zero_output(task, auto_item["reason"])
                    pair_payload["results"].append(
                        {
                            "turn_id": turn["id"],
                            "turn_index": turn_idx,
                            "prompt_ref_id": turn["id"],
                            "input_modalities": {
                                "image": bool(turn.get("image") is not None or turn.get("image_from_input", False)),
                                "static_images": len(turn.get("static_image_paths", [])),
                            "view_images": (
                                len(view_image_paths) if turn.get("view_images_from_input", False) else 0
                            ),
                            "mask_images": (
                                len(mask_image_paths) if turn.get("mask_images_from_input", False) else 0
                            ),
                            "affordance_view_images": 0,
                            "video": bool(turn.get("video_from_input", False)),
                            },
                            "output": json.dumps(output_json, ensure_ascii=False),
                            "error": None,
                            "kv_cache_hit": False,
                            "elapsed_sec": 0.0,
                            "auto_scored": True,
                            "auto_score_reason": auto_item["reason"],
                        }
                    )
                raise PairCompletedWithoutModel()
            if uses_affordance_views_from_input:
                if task.get("affordance_view_paths"):
                    manifest_affordance_paths = [Path(p) for p in task["affordance_view_paths"]]
                    require_min_paths(manifest_affordance_paths, args.affordance_views_num_images, f"manifest pair `{video_id}`")
                    affordance_view_paths = [
                        str(p)
                        for p in select_evenly_spaced_paths(
                            manifest_affordance_paths,
                            args.affordance_views_num_images,
                            label=f"manifest_affordance_views:{video_id}",
                        )
                    ]
                else:
                    if video_path is None or input_root is None:
                        raise ValueError(
                            f"Pair `{video_id}` has no manifest affordance_view_paths and cannot use mirrored lookup."
                        )
                    affordance_view_paths = resolve_affordance_view_paths(
                        video_path=video_path,
                        input_root=input_root,
                        affordance_views_input_dir=args.affordance_views_input_dir,
                        affordance_views_subdir=args.affordance_views_subdir,
                        affordance_views_glob=args.affordance_views_glob,
                        affordance_views_num_images=args.affordance_views_num_images,
                    )
            pair_payload["affordance_view_paths"] = affordance_view_paths

            if uses_videos_from_input:
                sampling = resolve_video_sampling_for_paths(video_paths=video_paths, args=args)
            elif uses_video_from_input:
                sampling = resolve_video_sampling(video_path=video_path, args=args)
            else:
                sampling = sampling_not_applicable()
            pair_payload["sampling"] = sampling
            print(
                "[Sampling] "
                f"video={video_id} mode={sampling['mode']} "
                f"duration_sec={sampling['duration_sec']} "
                f"estimated_parts={sampling['estimated_parts']} "
                f"fps={sampling['video_fps']} num_frames={sampling['video_num_frames']}",
                flush=True,
            )

            for turn_idx, turn in enumerate(turns):
                turn_start = time.perf_counter()
                print(
                    f"[Turn Start] video={video_id} turn={turn['id']} idx={turn_idx}",
                    flush=True,
                )
                user_message = build_user_message(
                    turn=turn,
                    video_path=video_path,
                    video_paths=video_paths,
                    paired_image_path=paired_image_path,
                    view_image_paths=view_image_paths,
                    mask_image_paths=mask_image_paths,
                    affordance_view_paths=affordance_view_paths,
                    task=task,
                )
                conversation.append(user_message)

                result_item = {
                    "turn_id": turn["id"],
                    "turn_index": turn_idx,
                    "prompt_ref_id": turn["id"],
                    "input_modalities": {
                        "image": bool(turn.get("image") is not None or turn.get("image_from_input", False)),
                        "static_images": len(turn.get("static_image_paths", [])),
                        "view_images": (
                            len(view_image_paths) if turn.get("view_images_from_input", False) else 0
                        ),
                        "mask_images": (
                            len(mask_image_paths) if turn.get("mask_images_from_input", False) else 0
                        ),
                        "affordance_view_images": (
                            len(affordance_view_paths) if turn.get("affordance_views_from_input", False) else 0
                        ),
                        "video": bool(turn.get("video_from_input", False)),
                        "videos": len(video_paths) if turn.get("videos_from_input", False) else 0,
                    },
                    "output": None,
                    "error": None,
                    "kv_cache_hit": None,
                    "elapsed_sec": None,
                }

                try:
                    output, kv_cache_hit = run_single_inference(
                        model=model,
                        processor=processor,
                        messages=conversation,
                        max_new_tokens=args.max_new_tokens,
                        cache_state=cache_state,
                        video_fps=sampling["video_fps"],
                        video_num_frames=sampling["video_num_frames"],
                        debug_input_shapes=args.debug_input_shapes,
                        debug_label=f"video={video_id} turn={turn['id']} full_inputs",
                    )
                    result_item["output"] = output
                    result_item["kv_cache_hit"] = bool(kv_cache_hit)
                    conversation.append({"role": "assistant", "content": [{"type": "text", "text": output}]})
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    is_oom = "out of memory" in str(exc).lower()
                    if is_oom:
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        cache_state["past_key_values"] = None
                        cache_state["sequence_ids"] = None
                        try:
                            output, kv_cache_hit = run_single_inference(
                                model=model,
                                processor=processor,
                                messages=conversation,
                                max_new_tokens=args.max_new_tokens,
                                cache_state=cache_state,
                                video_fps=sampling["video_fps"],
                                video_num_frames=sampling["video_num_frames"],
                                debug_input_shapes=args.debug_input_shapes,
                                debug_label=f"video={video_id} turn={turn['id']} retry_full_inputs",
                            )
                            result_item["output"] = output
                            result_item["kv_cache_hit"] = bool(kv_cache_hit)
                            conversation.append({"role": "assistant", "content": [{"type": "text", "text": output}]})
                            result_item["error"] = None
                        except Exception as retry_exc:
                            result_item["error"] = (
                                f"{msg} | retry_failed: {type(retry_exc).__name__}: {retry_exc}"
                            )
                            result_item["kv_cache_hit"] = False
                            total_turn_errors += 1
                    else:
                        result_item["error"] = msg
                        result_item["kv_cache_hit"] = False
                        total_turn_errors += 1

                result_item["elapsed_sec"] = round(time.perf_counter() - turn_start, 4)
                pair_payload["results"].append(result_item)
                print(
                    f"[Turn Done] video={video_id} turn={turn['id']} "
                    f"elapsed={result_item['elapsed_sec']}s "
                    f"error={result_item['error'] is not None} "
                    f"kv_cache_hit={result_item['kv_cache_hit']}",
                    flush=True,
                )

        except PairCompletedWithoutModel:
            pass
        except Exception as exc:
            pair_payload["pair_error"] = f"{type(exc).__name__}: {exc}"
            total_pair_errors += 1

        pair_payload["elapsed_sec"] = round(time.perf_counter() - pair_start, 4)
        with pair_json_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(pair_payload), f, ensure_ascii=False, indent=2)

        pair_index.append(
            {
                "video_id": video_id,
                "object_id": task.get("object_id"),
                "method": task.get("method"),
                "dataset": task.get("dataset"),
                "metric": task.get("metric"),
                "video_path": str(video_path) if video_path else None,
                "video_paths": [str(p) for p in video_paths],
                "view_image_count": len(pair_payload.get("view_image_paths") or []),
                "mask_image_count": len(pair_payload.get("mask_image_paths") or []),
                "relative_dir": str(video_rel_parent),
                "result_json": str(pair_json_path),
                "pair_error": pair_payload["pair_error"],
                "turn_error_count": sum(1 for x in pair_payload["results"] if x["error"] is not None),
                "auto_scored_missing_affordance": bool(pair_payload.get("auto_scored_missing_affordance")),
                "auto_scored_missing_video": bool(pair_payload.get("auto_scored_missing_video")),
                "auto_scored_missing_render_views": bool(pair_payload.get("auto_scored_missing_render_views")),
                "auto_scored_missing_description_images": bool(
                    pair_payload.get("auto_scored_missing_description_images")
                ),
                "auto_scored_missing_material_videos": bool(
                    pair_payload.get("auto_scored_missing_material_videos")
                ),
            }
        )
    pbar.close()

    run_meta["num_pair_errors"] = total_pair_errors
    run_meta["num_turn_errors"] = total_turn_errors
    run_meta["num_auto_scored_missing_affordance"] = total_auto_scored_missing_affordance
    run_meta["num_auto_scored_missing_video"] = total_auto_scored_missing_video
    run_meta["num_auto_scored_missing_render_views"] = total_auto_scored_missing_render_views
    run_meta["num_auto_scored_missing_description_images"] = total_auto_scored_missing_description_images
    run_meta["num_auto_scored_missing_material_videos"] = total_auto_scored_missing_material_videos

    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(run_meta), f, ensure_ascii=False, indent=2)
    with (run_dir / "pairs_index.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(pair_index), f, ensure_ascii=False, indent=2)

    print(f"Saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
