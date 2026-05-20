#!/usr/bin/env python3
"""
Run pillow_mixed_fem_rigid_test.py over many cases in parallel (one process per GPU).

Usage:
  python benchmark/code/render/material_batch/floor/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/floor/config.toml
  python benchmark/code/render/material_batch/floor/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/floor/config.toml \
      --pairs ours-mobility,physxanything-mobility --gpus 0,1,2,3
  python benchmark/code/render/material_batch/floor/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/floor/config.toml --dry-run

Extra pillow_mixed_fem_rigid_test.py args after batch flags are forwarded, e.g.:
  python benchmark/code/render/material_batch/floor/multi_gpu_batch.py \
      --config benchmark/code/render/material_batch/floor/config.toml -- --obj-rot-x 10
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore
except ImportError as e:
    raise SystemExit("Need TOML parser: use Python 3.11+ or `pip install tomli`") from e


Task = Tuple[str, str, str, Path, Optional[Path], str, Path]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FPS_RE = re.compile(r"Running at\s+([0-9]+(?:\.[0-9]+)?)\s+FPS", re.IGNORECASE)
STEP_RE = re.compile(r"\b[Ss]tep\s+([0-9]+)\s*/\s*([0-9]+)")


def clean_console_text(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "").strip()


def parse_child_progress(line: str) -> Tuple[Optional[float], Optional[str]]:
    clean = clean_console_text(line)
    fps_match = FPS_RE.search(clean)
    fps = float(fps_match.group(1)) if fps_match else None
    step_match = STEP_RE.search(clean)
    step = f"{step_match.group(1)}/{step_match.group(2)}" if step_match else None
    return fps, step


def safe_log_name(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("_") or "task"


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _is_case_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (p / "basic_info.json").is_file()


def list_cases_under_root(root: Path) -> List[Path]:
    cases: List[Path] = []
    for p in root.iterdir():
        if _is_case_dir(p):
            cases.append(p)
    cases.sort(key=lambda x: x.name)
    return cases


def list_scene_dirs(watertight: Path) -> List[Path]:
    scenes: List[Path] = []
    for p in watertight.iterdir():
        if not p.is_dir():
            continue
        scenes.append(p)
    scenes.sort(key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name))
    return scenes


def sim_to_cli_flags(sim: Dict[str, Any]) -> List[str]:
    if not sim:
        return []

    sim = sim_with_derived_steps(sim)
    out: List[str] = []
    numeric_map = {
        "steps": ("--steps", int),
        "fps": ("--fps", int),
        "render_every": ("--render-every", int),
        "warmup_steps": ("--warmup-steps", int),
        "scale": ("--scale", float),
        "e_threshold": ("--e-threshold", float),
        "glb_rot_x": ("--glb-rot-x", float),
        "obj_rot_x": ("--obj-rot-x", float),
        "drop_height": ("--drop-height", float),
        "extra_rigid_rho": ("--extra-rigid-rho", float),
        "cam_dist_scale": ("--cam-dist-scale", float),
        "cam_min_dist": ("--cam-min-dist", float),
        "cam_fov": ("--cam-fov", float),
        "cam_ground_z": ("--cam-ground-z", float),
        "cam_ground_margin": ("--cam-ground-margin", float),
        "cam_lookat_z_bias": ("--cam-lookat-z-bias", float),
    }
    for key, (flag, cast) in numeric_map.items():
        if key in sim:
            out.extend([flag, str(cast(sim[key]))])

    if sim.get("cpu"):
        out.append("--cpu")
    if sim.get("vis"):
        out.append("--vis")
    if bool(sim.get("record", True)):
        out.append("--record")

    if "extra_rigid" in sim:
        out.extend(["--extra-rigid", str(Path(sim["extra_rigid"]))])
    if "tmp_mesh_dir" in sim:
        out.extend(["--tmp-mesh-dir", str(Path(sim["tmp_mesh_dir"]))])

    return out


def sim_with_derived_steps(sim: Dict[str, Any]) -> Dict[str, Any]:
    """Derive simulation steps from requested recorded video duration.

    The floor runner records one frame every `render_every` simulation steps
    after `warmup_steps`, then writes the video at `fps`. Therefore a requested
    `duration_sec` maps to:

        steps = warmup_steps + ceil(duration_sec * fps) * render_every

    The derived value intentionally overrides `steps` when a duration is set,
    because duration is the more user-facing control for benchmark videos.
    """
    out = dict(sim or {})
    duration = None
    for key in ("duration_sec", "render_duration_sec", "video_duration_sec"):
        if key in out and out[key] is not None:
            duration = float(out[key])
            break
    if duration is None:
        return out
    if duration <= 0:
        raise ValueError("duration_sec must be > 0")
    fps = int(out.get("fps", 30))
    render_every = int(out.get("render_every", 1))
    warmup_steps = int(out.get("warmup_steps", 0))
    if fps <= 0:
        raise ValueError("fps must be > 0 when duration_sec is set")
    if render_every <= 0:
        raise ValueError("render_every must be > 0 when duration_sec is set")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0 when duration_sec is set")
    out["steps"] = warmup_steps + int(math.ceil(duration * fps)) * render_every
    return out


def sim_with_cli_overrides(sim: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(sim or {})
    override_map = {
        "duration_sec": "duration_sec",
        "steps": "steps",
        "fps": "fps",
        "render_every": "render_every",
        "warmup_steps": "warmup_steps",
        "cam_dist_scale": "cam_dist_scale",
        "cam_min_dist": "cam_min_dist",
        "cam_fov": "cam_fov",
        "cam_ground_z": "cam_ground_z",
        "cam_ground_margin": "cam_ground_margin",
        "cam_lookat_z_bias": "cam_lookat_z_bias",
    }
    for arg_name, sim_key in override_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            out[sim_key] = value
    return out


def infer_method_dataset(name: str) -> Tuple[str, str]:
    if name.startswith("ours_"):
        dataset = name[len("ours_") :].removesuffix("_181500")
        return "ours", dataset
    if name.startswith("output_physxanything_"):
        dataset = name[len("output_physxanything_") :]
        return "physxanything", dataset
    if name.startswith("outputs_physxgen_"):
        dataset = name[len("outputs_physxgen_") :]
        return "physxgen", dataset
    return name, "unknown"


def method_dataset_dir_name(method: str, dataset: str) -> str:
    return f"{method}_{dataset}"


def parse_pair_token(token: str) -> Tuple[str, str]:
    token = str(token).strip()
    if not token:
        raise argparse.ArgumentTypeError("empty method-dataset pair")
    if ":" in token:
        method, dataset = token.rsplit(":", 1)
    elif "-" in token:
        method, dataset = token.rsplit("-", 1)
    else:
        raise argparse.ArgumentTypeError(
            f"invalid pair `{token}`; expected METHOD-DATASET or METHOD:DATASET, e.g. ours-verse"
        )
    method = method.strip()
    dataset = dataset.strip()
    if not method or not dataset:
        raise argparse.ArgumentTypeError(f"invalid pair `{token}`")
    return method, dataset


def requested_pairs_from_args(args: argparse.Namespace) -> List[Tuple[str, str]]:
    tokens: List[str] = []
    if args.pairs:
        tokens.extend([x.strip() for x in str(args.pairs).split(",") if x.strip()])
    for item in args.pair or []:
        tokens.extend([x.strip() for x in str(item).split(",") if x.strip()])
    return [parse_pair_token(token) for token in tokens]


def _task_video_name(scene_or_case_dir: Path) -> str:
    return f"{scene_or_case_dir.name}_floor.mp4"


def _task_key(task: Task) -> str:
    _root_name, method, dataset, scene_or_case_dir, _, _, _ = task
    return f"{method}/{dataset}/{scene_or_case_dir.name}"


def is_existing_video_ok(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _load_tasks_from_state(state_file: Path, statuses: set[str]) -> set[str]:
    tasks: set[str] = set()
    if not state_file.exists():
        return tasks
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
                if str(row.get("status")) in statuses and row.get("task"):
                    tasks.add(str(row["task"]))
    except OSError as e:
        print(f"[warn] cannot read state file {state_file}: {e}", file=sys.stderr)
    return tasks


def _load_done_from_state(state_file: Path) -> set[str]:
    return _load_tasks_from_state(state_file, {"ok"})


def _load_failed_from_state(state_file: Path) -> set[str]:
    return _load_tasks_from_state(state_file, {"fail"})


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


class ProgressReporter:
    def __init__(self, total: int, label: str):
        self.total = total
        self.label = label
        self.count = 0
        self.ok = 0
        self.fail = 0
        self.workers: Dict[str, Dict[str, Any]] = {}
        self._last_print = 0.0
        self._bar = None
        try:
            from tqdm.auto import tqdm  # type: ignore

            self._bar = tqdm(
                total=total,
                desc=f"batch:{label}",
                unit="case",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        except Exception:
            self._bar = None
            print(f"[progress:{label}] 0/{total} ok=0 fail=0", flush=True)

    def _worker_postfix(self) -> str:
        parts: List[str] = []
        for gpu in sorted(self.workers, key=lambda x: (len(x), x)):
            state = self.workers[gpu]
            fps = state.get("fps")
            step = state.get("step")
            task = str(state.get("task", "idle")).split("/")[-1]
            if fps is None:
                value = f"g{gpu}:{task}"
            else:
                value = f"g{gpu}:{fps:.2f}fps"
            if step:
                value += f"@{step}"
            parts.append(value)
        return " ".join(parts)

    def handle_event(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type", "done"))
        gpu = str(event.get("gpu", ""))
        task = str(event.get("task", ""))
        if event_type in {"start", "heartbeat"} and gpu:
            state = self.workers.setdefault(gpu, {})
            if task:
                state["task"] = task
            if event.get("fps") is not None:
                state["fps"] = float(event["fps"])
            if event.get("step"):
                state["step"] = str(event["step"])
            if self._bar is not None:
                self._bar.set_postfix_str(
                    f"ok={self.ok} fail={self.fail} {self._worker_postfix()}",
                    refresh=True,
                )
            else:
                self._fallback_print(force=False)
            return
        self.update(str(event.get("status")), task, gpu=gpu)

    def update(self, status: str, task: str, gpu: str = "") -> None:
        self.count += 1
        if status == "ok":
            self.ok += 1
        else:
            self.fail += 1
        if gpu and gpu in self.workers:
            self.workers[gpu]["task"] = "idle"
            self.workers[gpu].pop("step", None)
        if self._bar is not None:
            self._bar.set_postfix_str(
                f"ok={self.ok} fail={self.fail} last={task[-24:]} {self._worker_postfix()}",
                refresh=True,
            )
            self._bar.update(1)
            return
        self._fallback_print(force=(self.count == self.total), last=task)

    def _fallback_print(self, force: bool = False, last: str = "") -> None:
        now = time.time()
        if force or now - self._last_print >= 2.0:
            self._last_print = now
            print(
                f"[progress:{self.label}] {self.count}/{self.total} "
                f"ok={self.ok} fail={self.fail} last={last} {self._worker_postfix()}",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def collect_tasks(
    cfg: Dict[str, Any],
    limit_per_root: Union[int, None],
    method_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
) -> Tuple[List[Task], Path]:
    defaults = cfg.get("defaults", {})
    output_root = Path(
        defaults.get("output_root")
        or defaults.get("output_dir")
        or str(Path(__file__).resolve().parent / "output")
    )

    tasks: List[Task] = []

    # Explicit single-case mode.
    for entry in cfg.get("cases", []):
        if not isinstance(entry, dict):
            continue
        root_name = str(entry.get("name", "case"))
        inferred_method, inferred_dataset = infer_method_dataset(root_name)
        method = str(entry.get("method", inferred_method))
        dataset = str(entry.get("dataset", inferred_dataset))
        if method_filter and method != method_filter:
            continue
        if dataset_filter and dataset != dataset_filter:
            continue
        out_dir = output_root / str(entry.get("output_name", method_dataset_dir_name(method, dataset)))
        case_dir = Path(entry["case_dir"])
        if not _is_case_dir(case_dir):
            print(f"[warn] skip case {root_name}: invalid case_dir {case_dir}", file=sys.stderr)
            continue
        video_name = str(entry.get("video_name", _task_video_name(case_dir)))
        tasks.append((root_name, method, dataset, case_dir.resolve(), None, video_name, out_dir))

    # Root scan mode.
    for entry in cfg.get("roots", []):
        if not isinstance(entry, dict):
            continue
        root_name = str(entry.get("name", "dataset"))
        inferred_method, inferred_dataset = infer_method_dataset(root_name)
        method = str(entry.get("method", inferred_method))
        dataset = str(entry.get("dataset", inferred_dataset))
        if method_filter and method != method_filter:
            continue
        if dataset_filter and dataset != dataset_filter:
            continue
        out_dir = output_root / str(entry.get("output_name", method_dataset_dir_name(method, dataset)))
        # Legacy mode: root path where each case contains basic_info.json.
        if "path" in entry:
            root = Path(entry["path"])
            if not root.is_dir():
                print(f"[warn] skip root {root_name}: not found {root}", file=sys.stderr)
                continue
            cases = list_cases_under_root(root)
            if limit_per_root is not None:
                cases = cases[:limit_per_root]
            for case_dir in cases:
                tasks.append((root_name, method, dataset, case_dir.resolve(), None, _task_video_name(case_dir), out_dir))
            continue

        # Watertight+metric_json mode (same style as water batch config).
        if "watertight" in entry and "metric_json" in entry:
            watertight = Path(entry["watertight"])
            metric_json = Path(entry["metric_json"])
            if not watertight.is_dir():
                print(f"[warn] skip root {root_name}: watertight not found {watertight}", file=sys.stderr)
                continue
            if not metric_json.is_dir():
                print(f"[warn] skip root {root_name}: metric_json not found {metric_json}", file=sys.stderr)
                continue
            scenes = list_scene_dirs(watertight)
            valid_scenes: List[Path] = []
            for scene_dir in scenes:
                mj = metric_json / f"{scene_dir.name}.json"
                if not mj.is_file():
                    print(
                        f"[warn] skip scene {root_name}/{scene_dir.name}: metric json not found {mj}",
                        file=sys.stderr,
                    )
                    continue
                valid_scenes.append(scene_dir)
            if limit_per_root is not None:
                valid_scenes = valid_scenes[:limit_per_root]
            for scene_dir in valid_scenes:
                tasks.append((root_name, method, dataset, scene_dir.resolve(), metric_json.resolve(), _task_video_name(scene_dir), out_dir))
            continue

        print(
            f"[warn] skip root {root_name}: need either 'path' or ('watertight' + 'metric_json')",
            file=sys.stderr,
        )

    return tasks, output_root


def env_overrides_from_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    raw = cfg.get("env", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[str(key)] = "true" if value else "false"
        else:
            out[str(key)] = str(value)
    return out


def build_child_env(gpu: str, extra_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    # Genesis/pyglet should not try to use an X11 display on headless workers.
    # Keep this as a default so config can still override it explicitly.
    env.setdefault("PYGLET_HEADLESS", "true")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.update(extra_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def preflight_environment(python_bin: str, script: Path, gpu: str, extra_env: Dict[str, str]) -> bool:
    env = build_child_env(gpu, extra_env)
    code = (
        "import os\n"
        "import pyglet\n"
        "print('PYGLET_HEADLESS=', os.environ.get('PYGLET_HEADLESS'))\n"
        "print('pyglet.options.headless=', pyglet.options.get('headless'))\n"
        "import genesis as gs\n"
        "print('genesis import ok')\n"
    )
    result = subprocess.run(
        [python_bin, "-c", code],
        env=env,
        cwd=str(script.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode == 0:
        print("[preflight] render environment import check passed", flush=True)
        return True
    print("[preflight] render environment import check failed:", file=sys.stderr, flush=True)
    print(result.stdout, file=sys.stderr, flush=True)
    print(
        "[preflight] This usually means OpenGL/EGL/X11 runtime libraries are missing. "
        "Install libEGL/libGL/libXrender or use --skip-preflight only if you know "
        "the worker environment is already configured.",
        file=sys.stderr,
        flush=True,
    )
    return False


def _gpu_worker(
    gpu: str,
    python_bin: str,
    script: Path,
    chunk: List[Task],
    sim_flags: List[str],
    forward: List[str],
    state_file: Path,
    extra_env: Dict[str, str],
    child_log_dir: Path,
    progress_queue: Optional[Any] = None,
) -> Tuple[int, int]:
    env = build_child_env(gpu, extra_env)
    child_log_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for root_name, method, dataset, scene_or_case_dir, metric_json_dir, video_name, out_dir in chunk:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [python_bin, str(script)]
        if metric_json_dir is None:
            cmd.extend(["--case-dir", str(scene_or_case_dir)])
        else:
            cmd.extend(
                [
                    "--scene-dir",
                    str(scene_or_case_dir),
                    "--metric-json-dir",
                    str(metric_json_dir),
                ]
            )
        cmd.extend(["--video-path", str(out_dir / video_name), *sim_flags, *forward])
        task = f"{method}/{dataset}/{scene_or_case_dir.name}"
        child_log = child_log_dir / f"gpu{gpu}_{safe_log_name(task)}.log"
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "start",
                    "gpu": gpu,
                    "task": task,
                    "video": str(out_dir / video_name),
                    "log": str(child_log),
                }
            )
        else:
            print(f"[GPU {gpu}] {task} -> {out_dir / video_name} log={child_log}", flush=True)
        try:
            with child_log.open("w", encoding="utf-8", errors="replace") as log_f:
                log_f.write("[cmd] " + " ".join(cmd) + "\n")
                log_f.flush()
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                last_emit = 0.0
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_f.write(line)
                    log_f.flush()
                    fps, step = parse_child_progress(line)
                    if progress_queue is not None and (fps is not None or step is not None):
                        now = time.time()
                        if now - last_emit >= 1.0:
                            progress_queue.put(
                                {
                                    "type": "heartbeat",
                                    "gpu": gpu,
                                    "task": task,
                                    "fps": fps,
                                    "step": step,
                                }
                            )
                            last_emit = now
                code = proc.wait()
        except Exception as e:
            code = -999
            with child_log.open("a", encoding="utf-8", errors="replace") as log_f:
                log_f.write(f"\n[worker-error] {type(e).__name__}: {e}\n")
                log_f.write(traceback.format_exc())
            if progress_queue is None:
                print(f"[GPU {gpu}] ERROR {task}: {e}", flush=True)

        if code == 0:
            ok += 1
            status = "ok"
        else:
            fail += 1
            status = "fail"
            if progress_queue is None:
                print(f"[GPU {gpu}] FAILED {dataset}/{scene_or_case_dir.name} (exit {code}); log={child_log}", flush=True)

        try:
            _append_jsonl(
                state_file,
                {
                    "task": task,
                    "root": root_name,
                    "method": method,
                    "dataset": dataset,
                    "scene_or_case": scene_or_case_dir.name,
                    "video": str(out_dir / video_name),
                    "gpu": gpu,
                    "status": status,
                    "exit_code": code,
                    "log": str(child_log),
                },
            )
        except OSError as e:
            print(f"[warn] write state failed for {task}: {e}", flush=True)
        if progress_queue is not None:
            progress_queue.put({"type": "done", "gpu": gpu, "task": task, "status": status})
        else:
            print(
                f"[GPU {gpu}] progress {ok + fail}/{len(chunk)} ok={ok} fail={fail}",
                flush=True,
            )
    return ok, fail


def _pool_entry(
    payload: Tuple[str, str, Path, List[Task], List[str], List[str], Path, Dict[str, str], Path, Optional[Any]]
) -> Tuple[int, int]:
    try:
        return _gpu_worker(*payload)
    except Exception as e:
        gpu, _, _, chunk, _, _, _, _, _, progress_queue = payload
        print(f"[GPU {gpu}] worker crashed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        if progress_queue is not None:
            for _ in chunk:
                progress_queue.put({"type": "done", "gpu": gpu, "task": f"gpu-{gpu}/worker_crash", "status": "fail"})
        return 0, len(chunk)


def resolve_gpu_list(cfg: Dict[str, Any], gpus_arg: Optional[str]) -> List[str]:
    gpus_sec = cfg.get("gpus", {})
    if gpus_arg is not None:
        gpu_list = [x.strip() for x in gpus_arg.split(",") if x.strip()]
    else:
        raw = gpus_sec.get("devices", [0])
        gpu_list = [str(x) for x in raw]
    if not gpu_list:
        raise ValueError("[gpus].devices is empty and --gpus not set")
    return gpu_list


def run_task_batch(
    *,
    tasks: List[Task],
    output_root: Path,
    combo_label: str,
    gpu_list: List[str],
    python_bin: str,
    script: Path,
    sim_flags: List[str],
    forward: List[str],
    extra_env: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[int, int]:
    if not script.is_file():
        print(f"[error] script not found: {script}", file=sys.stderr)
        return 0, 1
    output_root.mkdir(parents=True, exist_ok=True)

    state_file = args.state_file or (output_root / f"batch_state_{combo_label}.jsonl")
    child_log_dir = args.child_log_dir or (state_file.parent / "child_logs" / combo_label)

    if not tasks:
        print(f"[batch:{combo_label}] no tasks to run.", flush=True)
        return 0, 1

    if args.skip_existing:
        before = len(tasks)
        tasks = [t for t in tasks if not is_existing_video_ok(t[6] / t[5], args.min_video_bytes)]
        print(
            f"[batch:{combo_label}] skip-existing filtered: {before} -> {len(tasks)} "
            f"(min_video_bytes={args.min_video_bytes})",
            flush=True,
        )

    if args.resume:
        done = _load_done_from_state(state_file)
        before = len(tasks)
        tasks = [t for t in tasks if _task_key(t) not in done]
        print(f"[batch:{combo_label}] resume filtered: {before} -> {len(tasks)}", flush=True)

    if args.skip_failed:
        failed = _load_failed_from_state(state_file)
        before = len(tasks)
        tasks = [t for t in tasks if _task_key(t) not in failed]
        print(f"[batch:{combo_label}] skip-failed filtered: {before} -> {len(tasks)}", flush=True)

    if not tasks:
        print(f"[batch:{combo_label}] no pending tasks after filtering.", flush=True)
        return 0, 0

    print(f"[batch:{combo_label}] {len(tasks)} cases -> {output_root}, GPUs={gpu_list}", flush=True)
    if args.dry_run:
        for _root_name, method, dataset, scene_or_case_dir, metric_json_dir, video_name, out_dir in tasks[:20]:
            if metric_json_dir is None:
                print(f"  dry-run: {method}/{dataset} case={scene_or_case_dir} video={out_dir / video_name}")
            else:
                print(
                    f"  dry-run: {method}/{dataset} scene={scene_or_case_dir} "
                    f"metric_json_dir={metric_json_dir} video={out_dir / video_name}"
                )
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        return 0, 0

    chunks: List[List[Task]] = [[] for _ in gpu_list]
    for i, task in enumerate(tasks):
        chunks[i % len(gpu_list)].append(task)

    payloads = [
        (gpu_list[i], python_bin, script, chunks[i], sim_flags, forward, state_file, extra_env, child_log_dir, None)
        for i in range(len(gpu_list))
        if chunks[i]
    ]

    if len(payloads) == 1:
        ok, fail = _gpu_worker(*payloads[0])
    else:
        ctx = mp.get_context("spawn")
        progress = ProgressReporter(len(tasks), combo_label)
        try:
            with ctx.Manager() as manager:
                progress_queue = manager.Queue()
                payloads = [
                    (gpu_list[i], python_bin, script, chunks[i], sim_flags, forward, state_file, extra_env, child_log_dir, progress_queue)
                    for i in range(len(gpu_list))
                    if chunks[i]
                ]
                with ctx.Pool(processes=len(payloads)) as pool:
                    async_result = pool.map_async(_pool_entry, payloads)
                    while not async_result.ready():
                        try:
                            event = progress_queue.get(timeout=1.0)
                        except Empty:
                            continue
                        progress.handle_event(dict(event))
                    results = async_result.get()
                    while progress.count < len(tasks):
                        try:
                            event = progress_queue.get_nowait()
                        except Empty:
                            break
                        progress.handle_event(dict(event))
        finally:
            progress.close()
        ok = sum(r[0] for r in results)
        fail = sum(r[1] for r in results)

    print(f"[batch:{combo_label}] done: ok={ok} fail={fail}", flush=True)
    return ok, fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU batch for pillow_mixed_fem_rigid_test.py")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-per-root", type=int, default=None)
    parser.add_argument("--gpus", type=str, action="store", default=None, help="Comma-separated, e.g. 0,1,2")
    parser.add_argument("--method", type=str, default=None, help="Only run one method, e.g. ours or physxanything")
    parser.add_argument("--dataset", type=str, default=None, help="Only run one dataset, e.g. mobility, verse, inthewild")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Target recorded video duration in seconds; overrides --steps by deriving steps from fps/render_every/warmup.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override [sim].steps.")
    parser.add_argument("--fps", type=int, default=None, help="Override [sim].fps.")
    parser.add_argument("--render-every", type=int, default=None, help="Override [sim].render_every.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override [sim].warmup_steps.")
    parser.add_argument("--cam-dist-scale", type=float, default=None, help="Override recording camera distance scale.")
    parser.add_argument("--cam-min-dist", type=float, default=None, help="Override recording camera minimum distance.")
    parser.add_argument("--cam-fov", type=float, default=None, help="Override recording camera field of view in degrees.")
    parser.add_argument("--cam-ground-z", type=float, default=None, help="Override ground z used for camera framing.")
    parser.add_argument("--cam-ground-margin", type=float, default=None, help="Override below-ground camera framing margin.")
    parser.add_argument("--cam-lookat-z-bias", type=float, default=None, help="Override additive z bias for camera lookat.")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Queue one METHOD-DATASET pair. Can be repeated, e.g. --pair ours-verse --pair ours-inthewild.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated queue of METHOD-DATASET pairs, e.g. ours-verse,ours-inthewild.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the one-time pyglet/genesis import check before launching worker tasks.",
    )
    parser.add_argument("--state-file", type=Path, default=None, help="JSONL status file path")
    parser.add_argument(
        "--child-log-dir",
        type=Path,
        default=None,
        help="Directory for per-sample child process logs. Defaults to <state-file-dir>/child_logs/<pair>.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip tasks already done in prior runs")
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip tasks previously recorded as failed in --state-file.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip task if target video file already exists and is at least --min-video-bytes",
    )
    parser.add_argument(
        "--min-video-bytes",
        type=int,
        default=1024,
        help="Minimum video file size treated as completed by --skip-existing. Default 1024.",
    )
    args, forward = parser.parse_known_args()
    if forward and forward[0] == "--":
        forward = forward[1:]

    cfg = load_toml(args.config)
    try:
        pair_queue = requested_pairs_from_args(args)
        if pair_queue and (args.method or args.dataset):
            raise ValueError("Use either --pairs/--pair queue mode or --method/--dataset single-filter mode, not both.")
        gpu_list = resolve_gpu_list(cfg, args.gpus)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        sim_cfg = sim_with_cli_overrides(cfg.get("sim", {}), args)
        sim_cfg = sim_with_derived_steps(sim_cfg)
        sim_flags = sim_to_cli_flags(sim_cfg)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[sim] flags: {' '.join(sim_flags)}", flush=True)
    defaults = cfg.get("defaults", {})
    python_bin = str(defaults.get("python_bin") or defaults.get("python") or sys.executable)
    script = Path(
        defaults.get("script_path")
        or (Path(__file__).resolve().parent / "pillow_mixed_fem_rigid_test.py")
    )
    extra_env = env_overrides_from_config(cfg)
    if not args.dry_run and not args.skip_preflight:
        if not preflight_environment(python_bin, script, gpu_list[0], extra_env):
            sys.exit(2)

    if pair_queue:
        requested = pair_queue
    else:
        requested = [(args.method, args.dataset)]

    total_ok = 0
    total_fail = 0
    for index, (method, dataset) in enumerate(requested, start=1):
        combo_label = method_dataset_dir_name(method, dataset) if method and dataset else (
            str(method or dataset or "all")
        )
        if pair_queue:
            print(f"[queue] {index}/{len(requested)} start {combo_label}", flush=True)
        tasks, output_root = collect_tasks(cfg, args.limit_per_root, method, dataset)
        ok, fail = run_task_batch(
            tasks=tasks,
            output_root=output_root,
            combo_label=combo_label,
            gpu_list=gpu_list,
            python_bin=python_bin,
            script=script,
            sim_flags=sim_flags,
            forward=forward,
            extra_env=extra_env,
            args=args,
        )
        total_ok += ok
        total_fail += fail
        if pair_queue:
            print(f"[queue] {index}/{len(requested)} done {combo_label}: ok={ok} fail={fail}", flush=True)

    if pair_queue:
        print(f"[queue] all done: ok={total_ok} fail={total_fail}", flush=True)
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
