"""Formal six-task VoxeLibre zero-shot evaluation for Minecraft GameWAMs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gamewam.envs.voxelibre.tasks import TASK_SPECS, build_episode_scene, resolve_tasks
from gamewam.utils.eval_trace import RolloutTraceLogger
from gamewam.utils.ray_lifecycle import (
    packed_gpu_actor_capacity,
    shutdown_actor_pool,
    wait_actor_pool_ready,
)

logger = logging.getLogger(__name__)

# A removed node took roughly one second to reach the rendered client mesh in
# the pinned Craftium/VoxeLibre runtime. Keep another half-second so completion
# is visible rather than appearing only in the final video frame.
POST_SUCCESS_CAPTURE_SECONDS = 1.5


def _compact_craftium_startup_output(text: str, max_lines: int = 24) -> str:
    """Keep useful native startup diagnostics without flooding Ray's stdout."""
    unique_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line or line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)
    if len(unique_lines) > max_lines:
        head_count = max_lines // 2
        tail_count = max_lines - head_count
        omitted = len(unique_lines) - max_lines
        unique_lines = (
            unique_lines[:head_count]
            + [f"... {omitted} startup lines omitted ..."]
            + unique_lines[-tail_count:]
        )
    return " | ".join(unique_lines)


def _dump_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def _load_json(path: str | Path, default: Any) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _task_summary(results: list[dict[str, Any]], target: int) -> dict[str, Any]:
    successes = [item for item in results if bool(item["success"])]
    success_steps = [int(item["steps"]) for item in successes]
    return {
        "episodes_completed": len(results),
        "episodes_target": int(target),
        "successes": len(successes),
        "success_rate": len(successes) / len(results) if results else None,
        "success_rate_wilson95": _wilson_interval(len(successes), len(results)),
        "average_steps_on_success": (
            float(np.mean(success_steps)) if success_steps else None
        ),
        "median_steps_on_success": (
            float(np.median(success_steps)) if success_steps else None
        ),
        "average_episode_steps": (
            float(np.mean([int(item["steps"]) for item in results]))
            if results
            else None
        ),
        "gui_open_episode_rate": (
            float(np.mean([int(item["gui_open_steps"]) > 0 for item in results]))
            if results
            else None
        ),
        "camera_clip_rate": (
            float(
                sum(int(item["camera_clipped_steps"]) for item in results)
                / max(1, sum(int(item["steps"]) for item in results))
            )
            if results
            else None
        ),
    }


def summarize(
    tasks: list[str],
    results_by_task: dict[str, list[dict[str, Any]]],
    episodes_per_task: int,
) -> dict[str, Any]:
    task_items = {
        name: {
            "table_label": TASK_SPECS[name].table_label,
            "instruction": TASK_SPECS[name].instruction,
            **_task_summary(results_by_task.get(name, []), episodes_per_task),
        }
        for name in tasks
    }
    all_results = [item for name in tasks for item in results_by_task.get(name, [])]
    return {
        "evaluation": "GameWAM VoxeLibre zero-shot",
        "scope": "six controlled non-GUI primitive tasks; no VoxeLibre training or tuning",
        "primary_metric": "task_success_rate",
        "secondary_metric": "average_steps_on_success",
        "tasks": task_items,
        "overall_micro": _task_summary(
            all_results, len(tasks) * int(episodes_per_task)
        ),
        "episodes_completed": len(all_results),
        "episodes_target": len(tasks) * int(episodes_per_task),
    }


def write_reports(
    output_dir: str | Path,
    tasks: list[str],
    results_by_task: dict[str, list[dict[str, Any]]],
    episodes_per_task: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary = summarize(tasks, results_by_task, episodes_per_task)
    _dump_json(summary, output_dir / "summary.json")

    fields = (
        "task_name",
        "episode_id",
        "success",
        "steps",
        "end_reason",
        "total_reward",
        "wall_time_sec",
        "environment_seed",
        "model_seed",
        "target_x",
        "target_z",
        "view_yaw_offset_deg",
        "gui_open_steps",
        "camera_clipped_steps",
    )
    tmp_csv = output_dir / "episodes.csv.tmp"
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in tasks:
            for item in results_by_task.get(name, []):
                writer.writerow({field: item.get(field) for field in fields})
    os.replace(tmp_csv, output_dir / "episodes.csv")

    lines = [
        "GameWAM VoxeLibre zero-shot evaluation",
        "Primary: success rate. Secondary: average environment steps on success.",
        "",
        f"{'Task':<18} {'Success':>11} {'Rate':>9} {'95% Wilson CI':>20} {'Steps/success':>15}",
    ]
    for name in tasks:
        item = summary["tasks"][name]
        rate = item["success_rate"]
        interval = item["success_rate_wilson95"]
        avg_steps = item["average_steps_on_success"]
        ci_text = (
            f"[{100*interval[0]:.1f}, {100*interval[1]:.1f}]" if interval else "n/a"
        )
        lines.append(
            f"{item['table_label']:<18} "
            f"{item['successes']:>4}/{item['episodes_completed']:<6} "
            f"{100*rate if rate is not None else float('nan'):>8.1f}% "
            f"{ci_text:>20} "
            f"{avg_steps if avg_steps is not None else float('nan'):>15.1f}"
        )
    overall = summary["overall_micro"]
    lines.extend(
        [
            "",
            (
                f"Micro overall: {overall['successes']}/{overall['episodes_completed']} "
                f"({100*overall['success_rate']:.1f}%)"
                if overall["success_rate"] is not None
                else "Micro overall: no completed episodes"
            ),
            "GUI-open and camera-clipping rates in summary.json are diagnostics only.",
        ]
    )
    text_path = output_dir / "summary.txt"
    tmp_text = text_path.with_suffix(".txt.tmp")
    tmp_text.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_text, text_path)
    return summary


class EpisodeWorker:
    """Ray actor that loads one model and reuses it across episodes."""

    def __init__(
        self,
        *,
        cfg_dict: dict[str, Any],
        ckpt: str,
        dataset_stats_path: str,
        device: str,
        voxelibre_game_dir: str,
        craftium_minetest_dir: str | None,
        run_dir_prefix: str | None,
        render_size: tuple[int, int],
        fps: int,
        max_steps: int | None,
        save_video: bool,
        keep_runtime_dir: bool,
        init_frames: int,
        mt_listen_timeout_ms: int,
        env_start_retries: int,
    ):
        from omegaconf import OmegaConf

        from gamewam.runtime import build_inference

        cfg = OmegaConf.create(cfg_dict)
        cfg.inference.checkpoint_path = ckpt
        cfg.inference.dataset_stats_path = dataset_stats_path
        cfg.inference.device = device
        cfg.inference.mode = "rollout"
        self.inference = build_inference(cfg)
        self.base_seed = int(cfg.inference.get("seed", 0) or 0)
        self.ckpt = ckpt
        self.dataset_stats_path = dataset_stats_path
        self.device = device
        self.eval_cfg = OmegaConf.to_container(cfg.inference, resolve=True)
        self.voxelibre_game_dir = voxelibre_game_dir
        self.craftium_minetest_dir = craftium_minetest_dir
        self.run_dir_prefix = run_dir_prefix
        self.render_size = tuple(render_size)
        cursor_reference_size = cfg.data.common.processor.get(
            "cursor_reference_size", self.render_size
        )
        self.cursor_reference_size = tuple(map(float, cursor_reference_size))
        self.fps = int(fps)
        self.max_steps = max_steps
        self.save_video = bool(save_video)
        self.keep_runtime_dir = bool(keep_runtime_dir)
        self.init_frames = int(init_frames)
        self.mt_listen_timeout_ms = int(mt_listen_timeout_ms)
        self.env_start_retries = int(env_start_retries)

    def ready(self) -> bool:
        return True

    def _build_and_reset(self, task_name: str, episode_id: str, scene):
        from gamewam.envs.voxelibre.env import VoxeLibreGameEnv

        task = TASK_SPECS[task_name]
        last_error: Exception | None = None
        for attempt in range(self.env_start_retries + 1):
            env = VoxeLibreGameEnv(
                task=task,
                scene=scene,
                voxelibre_game_dir=self.voxelibre_game_dir,
                craftium_minetest_dir=self.craftium_minetest_dir,
                render_size=self.render_size,
                cursor_reference_size=self.cursor_reference_size,
                fps=self.fps,
                max_steps=self.max_steps,
                run_dir_prefix=self.run_dir_prefix,
                keep_runtime_dir=self.keep_runtime_dir,
                init_frames=self.init_frames,
                mt_listen_timeout_ms=self.mt_listen_timeout_ms,
            )
            startup_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(startup_output), contextlib.redirect_stderr(
                    startup_output
                ):
                    obs = env.reset()
                return env, obs, scene, attempt
            except Exception as exc:
                last_error = exc
                env.close()
                diagnostic = _compact_craftium_startup_output(
                    startup_output.getvalue()
                )
                logger.warning(
                    "[%s] ep=%s Craftium startup attempt %d/%d failed: %s%s",
                    task_name,
                    episode_id,
                    attempt + 1,
                    self.env_start_retries + 1,
                    exc,
                    f"; native output: {diagnostic}" if diagnostic else "",
                )
        assert last_error is not None
        raise last_error

    def run_episode(self, task_name: str, episode_id: str, episode_dir: str):
        from gamewam.envs.minecraft.action_space import (
            BINARY_KEYS,
            action_to_vector,
            vector_to_action,
        )
        from gamewam.utils.video_io import save_mp4

        episode_id = str(episode_id)
        episode_path = Path(episode_dir)
        episode_path.mkdir(parents=True, exist_ok=True)
        inference = self.inference
        task = TASK_SPECS[task_name]
        scene = build_episode_scene(task_name, episode_id, self.base_seed)
        actual_max_steps = int(self.max_steps or task.max_steps)
        inference.seed = scene.model_seed
        trace = RolloutTraceLogger(
            episode_path,
            action_dim=22,
            action_names=("camera_pitch", "camera_yaw", *BINARY_KEYS),
            binary_action_indices=range(2, 22),
        )
        trace.write_config(
            {
                "task_name": task_name,
                "episode_id": episode_id,
                "instruction": task.instruction,
                "ckpt": self.ckpt,
                "dataset_stats_path": self.dataset_stats_path,
                "device": self.device,
                "max_steps": actual_max_steps,
                "save_video": self.save_video,
                "task_seed": scene.task_seed,
                "environment_seed": scene.environment_seed,
                "model_seed": scene.model_seed,
                "target_x": scene.target_x,
                "target_z": scene.target_z,
                "view_yaw_offset_deg": scene.view_yaw_offset_deg,
                "eval_cfg": self.eval_cfg,
                "env": {
                    "render_size": list(self.render_size),
                    "cursor_reference_size": list(self.cursor_reference_size),
                    "fps": self.fps,
                    "init_frames": self.init_frames,
                    "mt_listen_timeout_ms": self.mt_listen_timeout_ms,
                    "env_start_retries": self.env_start_retries,
                },
            }
        )
        inference.inference_logger = trace.on_inference
        env = None
        frames: list[np.ndarray] = []
        total_reward = 0.0
        steps = 0
        gui_open_steps = 0
        camera_clipped_steps = 0
        wall_start = time.perf_counter()
        try:
            env, obs, scene, startup_retries_used = self._build_and_reset(
                task_name, episode_id, scene
            )
            inference.reset()

            while not obs.done and steps < actual_max_steps:
                if self.save_video and obs.image is not None:
                    frames.append(obs.image)
                action_vec = np.asarray(
                    inference.get_action(obs, task.instruction), dtype=np.float32
                ).reshape(-1)
                if action_vec.shape != (22,) or not np.isfinite(action_vec).all():
                    raise ValueError(
                        f"Minecraft policy must emit finite 22-D actions, got {action_vec}"
                    )
                env_action = vector_to_action(
                    action_vec, binarize=True, hotbar_argmax=True
                )
                obs = env.step(env_action)
                if env.last_effective_action is None:
                    raise RuntimeError(
                        "VoxeLibre env did not report its executed action"
                    )
                executed_action = env.last_effective_action
                executed_action_vec = action_to_vector(executed_action)
                trace.on_step(
                    step_idx=steps,
                    action_vec=action_vec,
                    env_action=env_action,
                    executed_action_vec=executed_action_vec,
                    executed_env_action=executed_action,
                    reward=obs.reward,
                    done=obs.done,
                    info={
                        "is_gui_open": bool(obs.info.get("is_gui_open", False)),
                        "current_hotbar": obs.info.get("current_hotbar"),
                        "pitch": obs.info.get("pitch"),
                        "yaw": obs.info.get("yaw"),
                        "camera_clipped": bool(env.last_camera_clipped),
                    },
                )
                steps += 1
                total_reward += float(obs.reward)
                gui_open_steps += int(bool(obs.info.get("is_gui_open", False)))
                camera_clipped_steps += int(env.last_camera_clipped)
                if obs.reward > 0:
                    break

            if self.save_video and obs.image is not None:
                frames.append(obs.image)
            success = total_reward > 0
            if self.save_video and success:
                # The server-side success callback can precede the client-side
                # render of the changed node/entity by multiple transport
                # frames. This neutral tail is visualization-only: it is
                # neither counted nor recorded in the model's action history.
                capture_frames = max(
                    1, math.ceil(self.fps * POST_SUCCESS_CAPTURE_SECONDS)
                )
                frames.extend(env.capture_post_success_frames(capture_frames))
            timeout = not success and not obs.done and steps >= actual_max_steps
            result = {
                "task_name": task_name,
                "episode_id": episode_id,
                "instruction": task.instruction,
                "success": success,
                "steps": steps,
                "end_reason": (
                    "success"
                    if success
                    else ("max_steps" if timeout else "environment_done")
                ),
                "total_reward": float(total_reward),
                "wall_time_sec": time.perf_counter() - wall_start,
                "task_seed": scene.task_seed,
                "environment_seed": scene.environment_seed,
                "model_seed": scene.model_seed,
                "target_x": scene.target_x,
                "target_z": scene.target_z,
                "view_yaw_offset_deg": scene.view_yaw_offset_deg,
                "startup_retries_used": startup_retries_used,
                "gui_open_steps": gui_open_steps,
                "camera_clipped_steps": camera_clipped_steps,
            }
            if self.save_video and frames:
                save_mp4(frames, str(episode_path / "episode.mp4"), fps=self.fps)
            trace.finalize(result)
            (episode_path / "episode_error.json").unlink(missing_ok=True)
            return result
        except Exception as exc:
            trace.finalize(
                {
                    "task_name": task_name,
                    "episode_id": episode_id,
                    "instruction": task.instruction,
                    "success": False,
                    "end_reason": "error",
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc),
                    "steps": steps,
                    "total_reward": total_reward,
                    "wall_time_sec": time.perf_counter() - wall_start,
                    "task_seed": scene.task_seed,
                    "environment_seed": scene.environment_seed,
                    "model_seed": scene.model_seed,
                }
            )
            _dump_json(
                {
                    "task_name": task_name,
                    "episode_id": episode_id,
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc),
                    "traceback": traceback.format_exc(),
                    "steps": steps,
                    "total_reward": total_reward,
                    "wall_time_sec": time.perf_counter() - wall_start,
                },
                episode_path / "episode_error.json",
            )
            raise
        finally:
            inference.inference_logger = None
            trace.close()
            if env is not None:
                env.close()

    def shutdown(self) -> None:
        self.inference.close()


def _compose_config(args: argparse.Namespace) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [f"task={args.task_config}"] if args.task_config else []
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"), version_base=None
    ):
        cfg = compose(config_name="sim_voxelibre", overrides=overrides)
    cfg.inference.checkpoint_path = str(Path(args.ckpt).resolve())
    cfg.inference.dataset_stats_path = str(Path(args.dataset_stats_path).resolve())
    cfg.inference.device = args.device
    cfg.inference.seed = int(args.seed)
    if args.num_inference_steps is not None:
        cfg.inference.num_inference_steps = int(args.num_inference_steps)
    if args.text_cfg_scale is not None:
        cfg.inference.text_cfg_scale = float(args.text_cfg_scale)
    processor = cfg.data.common.processor
    if int(processor.action_output_dim) != 22:
        raise ValueError(
            "VoxeLibre zero-shot evaluation requires Minecraft action_output_dim=22"
        )
    if int(processor.proprio_output_dim) != 15:
        raise ValueError(
            "VoxeLibre zero-shot evaluation requires Minecraft proprio_output_dim=15"
        )
    proprio_encoder_type = str(processor.proprio_encoder_type)
    if proprio_encoder_type not in {
        "generic_masked_linear",
        "minecraft_grouped_cursor",
    }:
        raise ValueError(
            "VoxeLibre zero-shot evaluation requires a validity-aware Minecraft "
            "proprio contract; expected generic_masked_linear or "
            f"minecraft_grouped_cursor, got {proprio_encoder_type!r}"
        )
    cursor_reference_size = processor.get("cursor_reference_size")
    if cursor_reference_size is None or len(cursor_reference_size) != 2:
        raise ValueError(
            "VoxeLibre zero-shot evaluation requires cursor_reference_size=[width, height]"
        )
    cursor_reference_size = tuple(map(float, cursor_reference_size))
    if any(not math.isfinite(value) or value <= 1 for value in cursor_reference_size):
        raise ValueError(
            "VoxeLibre cursor_reference_size values must be finite and greater than one"
        )
    if not bool(cfg.inference.get("binarize", True)) or not bool(
        cfg.inference.get("hotbar_argmax", True)
    ):
        raise ValueError(
            "VoxeLibre zero-shot evaluation requires Minecraft binary action decoding "
            "and hotbar argmax"
        )
    if (
        cfg.inference.prompt_template
        != "Control the Minecraft player to complete the task: {task}"
    ):
        raise ValueError(
            "VoxeLibre zero-shot evaluation pins the Minecraft rollout prompt template; "
            f"got {cfg.inference.prompt_template!r}"
        )
    return OmegaConf.to_container(cfg, resolve=True)


def _validate_resume_meta(path: Path, requested: dict[str, Any]) -> None:
    existing = _load_json(path, None)
    if existing is None:
        _dump_json(requested, path)
        return
    if existing != requested:
        differing = sorted(
            key
            for key in set(existing) | set(requested)
            if existing.get(key) != requested.get(key)
        )
        raise RuntimeError(
            f"Resume metadata mismatch for {differing}. Use the original settings "
            "or a new --output_dir."
        )


def _write_driver_error(
    output_dir: Path,
    task_name: str,
    episode_id: str,
    exc: Exception,
    actor_index: int | None = None,
) -> None:
    _dump_json(
        {
            "task_name": task_name,
            "episode_id": episode_id,
            "actor_index": actor_index,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": dt.datetime.now().isoformat(),
        },
        output_dir / "evaluation_error.json",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument("--voxelibre_game_dir", required=True)
    parser.add_argument("--craftium_minetest_dir", default=None)
    parser.add_argument(
        "--task_config",
        default=None,
        help="Optional Hydra task-group override.",
    )
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episodes_per_task", type=int, default=20)
    parser.add_argument("--split_number", type=int, default=8)
    parser.add_argument("--gpus_per_worker", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_dir_prefix", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render_width", type=int, default=640)
    parser.add_argument("--render_height", type=int, default=360)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--text_cfg_scale", type=float, default=None)
    parser.add_argument(
        "--save_video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--keep_runtime_dir", action="store_true")
    parser.add_argument("--init_frames", type=int, default=30)
    parser.add_argument("--mt_listen_timeout_ms", type=int, default=180_000)
    parser.add_argument("--env_start_retries", type=int, default=2)
    parser.add_argument("--actor_start_timeout_sec", type=float, default=1800.0)
    parser.add_argument("--ray_progress_timeout_sec", type=float, default=3600.0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = resolve_tasks(args.tasks)
    for name in (
        "episodes_per_task",
        "fps",
        "render_width",
        "render_height",
        "init_frames",
        "mt_listen_timeout_ms",
    ):
        if int(getattr(args, name)) <= 0 and name != "init_frames":
            raise ValueError(f"--{name} must be positive")
    if args.init_frames < 0 or args.split_number < 0 or args.env_start_retries < 0:
        raise ValueError(
            "init_frames, split_number and env_start_retries must be non-negative"
        )
    if args.render_width <= 1 or args.render_height <= 1:
        raise ValueError("render_width and render_height must both exceed one")
    if (
        not math.isfinite(args.actor_start_timeout_sec)
        or args.actor_start_timeout_sec <= 0
        or not math.isfinite(args.ray_progress_timeout_sec)
        or args.ray_progress_timeout_sec <= 0
    ):
        raise ValueError("actor and progress timeouts must be finite and positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be positive when supplied")
    if args.gpus_per_worker is not None and (
        not math.isfinite(args.gpus_per_worker) or args.gpus_per_worker <= 0
    ):
        raise ValueError("--gpus_per_worker must be finite and positive when supplied")
    for path_name in ("ckpt", "dataset_stats_path"):
        if not Path(getattr(args, path_name)).is_file():
            raise FileNotFoundError(
                f"--{path_name} not found: {getattr(args, path_name)}"
            )
    game_dir = Path(args.voxelibre_game_dir)
    if not (game_dir / "game.conf").is_file():
        raise FileNotFoundError(f"Invalid --voxelibre_game_dir: {game_dir}")
    if importlib.util.find_spec("craftium") is None:
        raise RuntimeError(
            "Craftium is not installed; install pinned Craftium v0.0.1 first"
        )
    from gamewam.envs.voxelibre.env import (
        CRAFTIUM_PLACE_EDGE_MARKER_VALUE,
        CRAFTIUM_XVFB_FRAMEBUFFER_MARKER_VALUE,
        SUPPORTED_CRAFTIUM_VERSION,
        SUPPORTED_VOXELIBRE_VERSION,
        _load_craftium_env_class,
        _read_voxelibre_version,
        _validate_craftium_build_root,
    )

    _load_craftium_env_class()
    craftium_root, craftium_binary = _validate_craftium_build_root(
        args.craftium_minetest_dir
    )
    detected_voxelibre_version = _read_voxelibre_version(game_dir.resolve())
    if detected_voxelibre_version != SUPPORTED_VOXELIBRE_VERSION:
        raise RuntimeError(
            f"This suite requires VoxeLibre {SUPPORTED_VOXELIBRE_VERSION}, got "
            f"{detected_voxelibre_version}."
        )

    cfg_dict = _compose_config(args)
    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "evaluate_results"
        / "voxelibre_zero_shot"
        / f"{Path(args.ckpt).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite_version": 1,
        "scene_contract_version": 1,
        "resolved_config_sha256": _json_sha256(cfg_dict),
        "checkpoint": _file_identity(args.ckpt),
        "dataset_stats": _file_identity(args.dataset_stats_path),
        "voxelibre_game_dir": str(game_dir.resolve()),
        "voxelibre_version": detected_voxelibre_version,
        "craftium_version": SUPPORTED_CRAFTIUM_VERSION,
        "craftium_place_edge_patch": CRAFTIUM_PLACE_EDGE_MARKER_VALUE,
        "craftium_xvfb_framebuffer_patch": (
            CRAFTIUM_XVFB_FRAMEBUFFER_MARKER_VALUE
        ),
        "craftium_runtime_root": str(craftium_root),
        "craftium_binary": _file_identity(craftium_binary),
        "tasks": tasks,
        "task_contract": {
            name: {
                "instruction": TASK_SPECS[name].instruction,
                "kind": TASK_SPECS[name].kind,
                "target": TASK_SPECS[name].target,
                "inventory": list(TASK_SPECS[name].inventory),
                "selected_hotbar": int(TASK_SPECS[name].selected_hotbar),
                "structure": TASK_SPECS[name].structure,
                "max_steps": int(TASK_SPECS[name].max_steps),
            }
            for name in tasks
        },
        "episodes_per_task": int(args.episodes_per_task),
        "base_seed": int(args.seed),
        "task_config": args.task_config,
        "max_steps_override": args.max_steps,
        "render_size": [int(args.render_width), int(args.render_height)],
        "cursor_reference_size": list(
            map(float, cfg_dict["data"]["common"]["processor"]["cursor_reference_size"])
        ),
        "fps": int(args.fps),
        "save_video": bool(args.save_video),
        "video_contract_version": 2,
        "post_success_capture_seconds": (
            POST_SUCCESS_CAPTURE_SECONDS if args.save_video else 0.0
        ),
        "init_frames": int(args.init_frames),
        "num_inference_steps": cfg_dict["inference"]["num_inference_steps"],
        "text_cfg_scale": cfg_dict["inference"]["text_cfg_scale"],
        "prompt_template": cfg_dict["inference"]["prompt_template"],
        "action_decoding": {"binarize": True, "hotbar_argmax": True},
        "camera": {"vertical_fov_deg": 90.0, "mouse_sensitivity": 0.2},
        "craftium_contract": (
            "v0.0.1 raw Dict action + required virtual PLACE edge patch"
        ),
        "action_adapter_contract_version": 1,
    }
    _validate_resume_meta(output_dir / "run_meta.json", meta)
    _dump_json(cfg_dict, output_dir / "resolved_config.json")

    results_by_task: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[str, str]] = []
    for task_name in tasks:
        results_path = output_dir / task_name / "results.json"
        results = _load_json(results_path, [])
        results.sort(key=lambda item: int(item["episode_id"]))
        completed = {str(item["episode_id"]) for item in results}
        results_by_task[task_name] = results
        for episode_id in range(args.episodes_per_task):
            if str(episode_id) not in completed:
                pending.append((task_name, str(episode_id)))
    write_reports(output_dir, tasks, results_by_task, args.episodes_per_task)
    if not pending:
        logger.info(
            "All requested VoxeLibre episodes are already complete: %s", output_dir
        )
        return summarize(tasks, results_by_task, args.episodes_per_task)

    worker_kwargs = {
        "cfg_dict": cfg_dict,
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_stats_path": str(Path(args.dataset_stats_path).resolve()),
        "device": args.device,
        "voxelibre_game_dir": str(game_dir.resolve()),
        "craftium_minetest_dir": args.craftium_minetest_dir,
        "run_dir_prefix": args.run_dir_prefix,
        "render_size": (args.render_width, args.render_height),
        "fps": args.fps,
        "max_steps": args.max_steps,
        "save_video": args.save_video,
        "keep_runtime_dir": args.keep_runtime_dir,
        "init_frames": args.init_frames,
        "mt_listen_timeout_ms": args.mt_listen_timeout_ms,
        "env_start_retries": args.env_start_retries,
    }

    def episode_dir(task_name: str, episode_id: str) -> str:
        path = output_dir / task_name / episode_id
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def handle_result(result: dict[str, Any]) -> None:
        task_name = result["task_name"]
        results_by_task[task_name].append(result)
        results_by_task[task_name].sort(key=lambda item: int(item["episode_id"]))
        _dump_json(results_by_task[task_name], output_dir / task_name / "results.json")
        summary = write_reports(
            output_dir, tasks, results_by_task, args.episodes_per_task
        )
        item = summary["tasks"][task_name]
        logger.info(
            "[%s] ep=%s success=%s steps=%d; %d/%d successful (%d/%d complete)",
            task_name,
            result["episode_id"],
            result["success"],
            result["steps"],
            item["successes"],
            item["episodes_completed"],
            summary["episodes_completed"],
            summary["episodes_target"],
        )

    if args.split_number == 0:
        worker = EpisodeWorker(**worker_kwargs)
        try:
            for task_name, episode_id in pending:
                try:
                    handle_result(
                        worker.run_episode(
                            task_name, episode_id, episode_dir(task_name, episode_id)
                        )
                    )
                except Exception as exc:
                    _write_driver_error(output_dir, task_name, episode_id, exc)
                    raise
        finally:
            worker.shutdown()
    else:
        import ray

        gpus_per_worker = (
            float(args.gpus_per_worker)
            if args.gpus_per_worker is not None
            else (1.0 if args.device.startswith("cuda") else 0.0)
        )
        max_workers = min(args.split_number, len(pending))
        if gpus_per_worker > 0 and args.device != "cuda":
            raise ValueError(
                "Ray GPU actors require --device cuda (without an explicit index)"
            )
        if gpus_per_worker > 0:
            gpu_count = torch.cuda.device_count()
            capacity = packed_gpu_actor_capacity(gpu_count, gpus_per_worker)
            if capacity < 1:
                raise RuntimeError(
                    f"gpus_per_worker={gpus_per_worker} cannot fit on {gpu_count} visible GPU(s)"
                )
            max_workers = min(max_workers, capacity)
        options = {"num_gpus": gpus_per_worker} if gpus_per_worker > 0 else {}
        actors: list[Any] = []
        in_flight: dict[Any, tuple[int, str, str]] = {}
        queue_index = 0

        def submit(actor_index: int) -> None:
            nonlocal queue_index
            if queue_index >= len(pending):
                return
            task_name, episode_id = pending[queue_index]
            queue_index += 1
            ref = actors[actor_index].run_episode.remote(
                task_name, episode_id, episode_dir(task_name, episode_id)
            )
            in_flight[ref] = (actor_index, task_name, episode_id)

        try:
            ray.init(ignore_reinit_error=True, num_cpus=max_workers)
            WorkerActor = (
                ray.remote(**options)(EpisodeWorker)
                if options
                else ray.remote(EpisodeWorker)
            )
            for _ in range(max_workers):
                actors.append(WorkerActor.remote(**worker_kwargs))
            wait_actor_pool_ready(ray, actors, timeout_s=args.actor_start_timeout_sec)
            for actor_index in range(len(actors)):
                submit(actor_index)
            while in_flight:
                ready, _ = ray.wait(
                    list(in_flight),
                    num_returns=1,
                    timeout=args.ray_progress_timeout_sec,
                )
                if not ready:
                    active = list(in_flight.values())
                    raise TimeoutError(
                        f"No VoxeLibre episode completed for {args.ray_progress_timeout_sec:g}s; "
                        f"active={active}"
                    )
                ref = ready[0]
                actor_index, task_name, episode_id = in_flight.pop(ref)
                try:
                    handle_result(ray.get(ref))
                except Exception as exc:
                    _write_driver_error(
                        output_dir, task_name, episode_id, exc, actor_index
                    )
                    raise
                submit(actor_index)
        finally:
            for ref in list(in_flight):
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    pass
            shutdown_actor_pool(ray, actors, logger=logger)

    summary = write_reports(output_dir, tasks, results_by_task, args.episodes_per_task)
    logger.info("VoxeLibre zero-shot evaluation complete: %s", output_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        run(parse_args(argv))
    except Exception:
        logger.exception("VoxeLibre zero-shot evaluation failed")
        raise


if __name__ == "__main__":
    main()
