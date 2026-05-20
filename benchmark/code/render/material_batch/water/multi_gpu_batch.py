#!/usr/bin/env python3
"""
Run multi_part_water_drop.py over many scenes in parallel (one process per GPU).

Usage:
  python benchmark/code/render/material_batch/water/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/water/config.toml
  python benchmark/code/render/material_batch/water/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/water/config.toml \
      --pairs ours-mobility,physxanything-mobility --gpus 0,1,2,3
  python benchmark/code/render/material_batch/water/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/water/config.toml --dry-run

Extra Genesis args after the batch flags are forwarded to multi_part_water_drop.py, e.g.:
  python benchmark/code/render/material_batch/water/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/water/config.toml -- --auto-scale --show-walls
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore
except ImportError as e:
    raise SystemExit(
        "Need TOML parser: use Python 3.11+ or `pip install tomli`"
    ) from e


Task = Tuple[str, str, str, Path, Path, str, Path]


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def list_scene_dirs(watertight: Path) -> List[Path]:
    scenes: List[Path] = []
    for p in watertight.iterdir():
        if not p.is_dir():
            continue
        if not p.name.isdigit():
            continue
        scenes.append(p)
    scenes.sort(key=lambda x: int(x.name))
    return scenes


def sim_to_cli_flags(sim: Dict[str, Any]) -> List[str]:
    if not sim:
        return []
    out: List[str] = []
    m = {
        "steps": ("--steps", int),
        "fps": ("--fps", int),
        "render_every": ("--render-every", int),
        "warmup_steps": ("--warmup-steps", int),
        "progress_interval": ("--progress-interval", int),
        "rotate_x": ("--rotate-x", float),
        "drop_height": ("--drop-height", float),
        "collision_margin": ("--collision-margin", float),
        "wall_alpha": ("--wall-alpha", float),
        "water_alpha": ("--water-alpha", float),
    }
    for key, (flag, cast) in m.items():
        if key not in sim:
            continue
        out.extend([flag, str(cast(sim[key]))])

    if sim.get("cpu"):
        out.append("--cpu")
    if sim.get("vis") or sim.get("viewer"):
        out.append("--vis")
    if sim.get("no_record") or sim.get("no-record"):
        out.append("--no-record")
    if sim.get("auto_scale") or sim.get("auto-scale"):
        out.append("--auto-scale")
    if sim.get("show_walls") or sim.get("show-walls"):
        out.append("--show-walls")
    return out


def infer_method_dataset(name: str) -> Tuple[str, str]:
    if name.startswith("ours_"):
        return "ours", name[len("ours_") :].removesuffix("_181500")
    if name.startswith("output_physxanything_"):
        return "physxanything", name[len("output_physxanything_") :]
    if name.startswith("outputs_physxgen_"):
        return "physxgen", name[len("outputs_physxgen_") :]
    return "", ""


def parse_pair_token(token: str) -> Tuple[str, str]:
    token = token.strip()
    if not token:
        raise argparse.ArgumentTypeError("empty method-dataset pair")
    if ":" in token:
        method, dataset = token.rsplit(":", 1)
    elif "-" in token:
        method, dataset = token.rsplit("-", 1)
    else:
        raise argparse.ArgumentTypeError(
            f"invalid pair '{token}', expected METHOD-DATASET or METHOD:DATASET"
        )
    method = method.strip()
    dataset = dataset.strip()
    if not method or not dataset:
        raise argparse.ArgumentTypeError(f"invalid pair '{token}'")
    return method, dataset


def requested_pairs_from_args(args: argparse.Namespace) -> List[Tuple[str, str]]:
    tokens: List[str] = []
    if args.pairs:
        tokens.extend([x.strip() for x in str(args.pairs).split(",") if x.strip()])
    return [parse_pair_token(token) for token in tokens]


def collect_tasks(
    cfg: Dict[str, Any],
    limit_per_root: Union[int, None],
    method_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
) -> Tuple[List[Task], Path]:
    defaults = cfg.get("defaults", {})
    out_dir = Path(
        defaults.get(
            "output_dir",
            str(Path(__file__).resolve().parent / "output"),
        )
    )
    tasks: List[Task] = []
    roots = cfg.get("roots", [])
    for entry in roots:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "dataset")
        inferred_method, inferred_dataset = infer_method_dataset(str(name))
        method = str(entry.get("method", inferred_method))
        dataset = str(entry.get("dataset", inferred_dataset))
        if method_filter and method != method_filter:
            continue
        if dataset_filter and dataset != dataset_filter:
            continue
        wt = Path(entry["watertight"])
        mj = Path(entry["metric_json"])
        root_out_dir = out_dir / str(entry.get("output_name", name))
        if not wt.is_dir():
            print(f"[warn] skip {name}: watertight not found: {wt}", file=sys.stderr)
            continue
        if not mj.is_dir():
            print(f"[warn] skip {name}: metric_json not found: {mj}", file=sys.stderr)
            continue
        scenes = list_scene_dirs(wt)
        if limit_per_root is not None:
            scenes = scenes[: limit_per_root]
        for sd in scenes:
            vid = f"{name}_{sd.name}_mpm.mp4"
            tasks.append((str(name), method, dataset, sd.resolve(), mj.resolve(), vid, root_out_dir))
    return tasks, out_dir


def _task_key(task: Task) -> str:
    _root_name, method, dataset, scene_dir, _, _, _ = task
    return f"{method}/{dataset}/{scene_dir.name}"


def _load_done_from_state(state_file: Path) -> set[str]:
    done: set[str] = set()
    if not state_file.exists():
        return done
    try:
        with state_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") == "ok" and row.get("task"):
                    done.add(str(row["task"]))
    except OSError as e:
        print(f"[warn] cannot read state file {state_file}: {e}", file=sys.stderr)
    return done


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _gpu_worker(
    gpu: str,
    script: Path,
    chunk: List[Task],
    sim_flags: List[str],
    forward: List[str],
    state_file: Path,
) -> Tuple[int, int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    ok = fail = 0
    for root_name, method, dataset, scene_dir, metric_json, video_name, out_dir in chunk:
        task = f"{method}/{dataset}/{scene_dir.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            "--scene-dir",
            str(scene_dir),
            "--metric-json-dir",
            str(metric_json),
            "--output-dir",
            str(out_dir),
            "--video-filename",
            video_name,
            *sim_flags,
            *forward,
        ]
        print(f"[GPU {gpu}] {method}/{dataset}/{scene_dir.name} → {out_dir / video_name}", flush=True)
        try:
            r = subprocess.run(cmd, env=env)
            code = r.returncode
        except Exception as e:
            code = -999
            print(f"[GPU {gpu}] ERROR {task}: {e}", flush=True)

        if code == 0:
            ok += 1
            status = "ok"
        else:
            fail += 1
            status = "fail"
            print(f"[GPU {gpu}] FAILED {method}/{dataset}/{scene_dir.name} (exit {code})", flush=True)

        try:
            _append_jsonl(
                state_file,
                {
                    "task": task,
                    "root": root_name,
                    "method": method,
                    "dataset": dataset,
                    "scene": scene_dir.name,
                    "video": str(out_dir / video_name),
                    "gpu": gpu,
                    "status": status,
                    "exit_code": code,
                },
            )
        except OSError as e:
            print(f"[warn] write state failed for {task}: {e}", flush=True)
    return ok, fail


def _pool_entry(payload: Tuple[str, Path, List[Task], List[str], List[str], Path]) -> Tuple[int, int]:
    try:
        return _gpu_worker(*payload)
    except Exception as e:
        gpu, _, chunk, _, _, _ = payload
        print(f"[GPU {gpu}] worker crashed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 0, len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU batch for multi_part_water_drop.py")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-per-root", type=int, default=None)
    parser.add_argument("--gpus", type=str, action="store", default=None, help="Comma-separated, e.g. 0,1,2")
    parser.add_argument("--method", type=str, default=None, help="Only run one method, e.g. ours or physxanything")
    parser.add_argument("--dataset", type=str, default=None, help="Only run one dataset, e.g. mobility, verse, inthewild")
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated queue of METHOD-DATASET pairs, e.g. ours-verse,ours-inthewild.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-file", type=Path, default=None, help="JSONL status file path")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already done in prior runs")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip task if target video file already exists in output dir",
    )
    args, forward = parser.parse_known_args()
    if forward and forward[0] == "--":
        forward = forward[1:]

    cfg = load_toml(args.config)
    gpus_sec = cfg.get("gpus", {})
    devices = args.gpus
    if devices is not None:
        gpu_list = [x.strip() for x in devices.split(",") if x.strip()]
    else:
        raw = gpus_sec.get("devices", [0])
        gpu_list = [str(x) for x in raw]

    if not gpu_list:
        print("[error] [gpus].devices is empty and --gpus not set", file=sys.stderr)
        sys.exit(1)

    pair_queue = requested_pairs_from_args(args)
    if pair_queue and (args.method or args.dataset):
        raise ValueError("Use either --pairs or --method/--dataset, not both.")
    requested = pair_queue or [(args.method, args.dataset)]

    sim_flags = sim_to_cli_flags(cfg.get("sim", {}))
    script = Path(__file__).resolve().parent / "multi_part_water_drop.py"
    exit_code = 0

    for index, (method, dataset) in enumerate(requested, start=1):
        tasks, out_root = collect_tasks(cfg, args.limit_per_root, method, dataset)
        if not tasks:
            print(f"[queue] {index}/{len(requested)} {method or 'all'}_{dataset or 'all'}: no tasks.", flush=True)
            continue
        state_file = args.state_file or (out_root / "batch_state.jsonl")
        out_root.mkdir(parents=True, exist_ok=True)
        label = f"{method or 'all'}_{dataset or 'all'}"
        print(f"[queue] {index}/{len(requested)} start {label}", flush=True)

        if args.skip_existing:
            before = len(tasks)
            tasks = [t for t in tasks if not (t[6] / t[5]).exists()]
            print(f"[batch:{label}] skip-existing filtered: {before} -> {len(tasks)}", flush=True)

        if args.resume:
            done = _load_done_from_state(state_file)
            before = len(tasks)
            tasks = [t for t in tasks if _task_key(t) not in done]
            print(f"[batch:{label}] resume filtered: {before} -> {len(tasks)}", flush=True)

        if not tasks:
            print(f"[batch:{label}] no pending tasks after filtering.", flush=True)
            continue

        print(f"[batch:{label}] {len(tasks)} scenes → {out_root}, GPUs={gpu_list}", flush=True)
        if args.dry_run:
            for _root_name, method_i, dataset_i, scene_dir, metric_json, video_name, out_dir in tasks[:20]:
                print(
                    f"  dry-run: {method_i}/{dataset_i} scene={scene_dir.name} "
                    f"metric={metric_json} video={out_dir / video_name}"
                )
            if len(tasks) > 20:
                print(f"  ... and {len(tasks) - 20} more")
            continue

        chunks: List[List[Task]] = [[] for _ in gpu_list]
        for i, task in enumerate(tasks):
            chunks[i % len(gpu_list)].append(task)

        payloads = [
            (gpu_list[i], script, chunks[i], sim_flags, forward, state_file)
            for i in range(len(gpu_list))
            if chunks[i]
        ]

        if len(payloads) == 1:
            ok, fail = _gpu_worker(*payloads[0])
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(payloads)) as pool:
                results = pool.map(_pool_entry, payloads)
            ok = sum(r[0] for r in results)
            fail = sum(r[1] for r in results)

        print(f"[batch:{label}] done: ok={ok} fail={fail}", flush=True)
        if fail:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
