"""GameWAM evaluation on the four GameTARS VizDoom maps."""

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gamewam.envs.vizdoom.assets import GAMETARS_SCENARIOS
from gamewam.envs.vizdoom.tasks import TASK_PARAPHRASES
from gamewam.utils.eval_trace import RolloutTraceLogger
from gamewam.utils.ray_lifecycle import shutdown_actor_pool

logger = logging.getLogger(__name__)


def _dump_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp_path, path)


def _load_json(path, default):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def _derive_episode_seed(base_seed, scenario, episode_id, domain):
    digest = hashlib.blake2b(
        f"{base_seed}|{scenario}|{episode_id}|{domain}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _resolve_scenarios(value):
    if value.strip().lower() == "all":
        return list(GAMETARS_SCENARIOS)
    scenarios = [item.strip() for item in value.split(",") if item.strip()]
    if not scenarios:
        raise ValueError("--scenarios must be 'all' or a non-empty comma-separated list.")
    unknown = [item for item in scenarios if item not in GAMETARS_SCENARIOS]
    if unknown:
        raise ValueError(
            f"Unknown VizDoom scenario(s): {unknown}. "
            f"Expected a subset of {GAMETARS_SCENARIOS}."
        )
    if len(set(scenarios)) != len(scenarios):
        raise ValueError(f"Duplicate scenario in --scenarios: {scenarios}")
    return scenarios


def _mean_metric(results, key):
    values = [float(item[key]) for item in results if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def _summarize(scenarios, results_by_scenario, episodes_per_map):
    maps = {}
    for scenario in scenarios:
        results = results_by_scenario.get(scenario, [])
        rewards = np.asarray(
            [float(item["episode_reward"]) for item in results],
            dtype=np.float64,
        )
        if rewards.size:
            mean = float(rewards.mean())
            std = float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0
            sem = std / math.sqrt(rewards.size)
            ci95 = [mean - 1.96 * sem, mean + 1.96 * sem]
            reward_stats = {
                "average_episode_reward": mean,
                "episode_reward_std": std,
                "episode_reward_sem": sem,
                "episode_reward_ci95": ci95,
                "episode_reward_median": float(np.median(rewards)),
                "episode_reward_min": float(rewards.min()),
                "episode_reward_max": float(rewards.max()),
            }
        else:
            reward_stats = {
                "average_episode_reward": None,
                "episode_reward_std": None,
                "episode_reward_sem": None,
                "episode_reward_ci95": None,
                "episode_reward_median": None,
                "episode_reward_min": None,
                "episode_reward_max": None,
            }
        maps[scenario] = {
            "episodes_completed": len(results),
            "episodes_target": int(episodes_per_map),
            **reward_stats,
            "diagnostics": {
                "average_kills": _mean_metric(results, "kills"),
                "average_episode_steps": _mean_metric(results, "steps"),
                "average_final_health": _mean_metric(results, "final_health"),
                "average_final_ammo": _mean_metric(results, "final_ammo"),
                "timeout_rate": (
                    float(np.mean([bool(item["timeout"]) for item in results]))
                    if results
                    else None
                ),
            },
        }
    return {
        "benchmark": "GameTARS VizDoom",
        "primary_metric": "average_episode_reward",
        "episode_metric": "cumulative_environment_reward",
        "maps": maps,
        "episodes_completed": sum(len(results_by_scenario.get(s, [])) for s in scenarios),
        "episodes_target": len(scenarios) * int(episodes_per_map),
    }


def _write_reports(output_dir, scenarios, results_by_scenario, episodes_per_map):
    output_dir = Path(output_dir)
    summary = _summarize(scenarios, results_by_scenario, episodes_per_map)
    _dump_json(summary, output_dir / "summary.json")

    fields = [
        "scenario",
        "episode_id",
        "episode_reward",
        "steps",
        "kills",
        "final_health",
        "final_ammo",
        "timeout",
        "end_reason",
        "wall_time_sec",
        "env_seed",
        "model_seed",
    ]
    csv_tmp = output_dir / "episodes.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in scenarios:
            for result in results_by_scenario.get(scenario, []):
                writer.writerow({key: result.get(key) for key in fields})
    os.replace(csv_tmp, output_dir / "episodes.csv")

    lines = [
        "GameTARS VizDoom evaluation",
        "Primary metric: average cumulative environment reward per episode",
        "",
        f"{'Map':<20} {'Episodes':>10} {'Avg reward':>14} {'Std':>12} {'Avg kills':>12} {'Avg steps':>12}",
    ]
    for scenario in scenarios:
        item = summary["maps"][scenario]
        avg = item["average_episode_reward"]
        std = item["episode_reward_std"]
        diagnostics = item["diagnostics"]
        lines.append(
            f"{scenario:<20} "
            f"{item['episodes_completed']:>4}/{item['episodes_target']:<5} "
            f"{avg if avg is not None else float('nan'):>14.4f} "
            f"{std if std is not None else float('nan'):>12.4f} "
            f"{diagnostics['average_kills'] if diagnostics['average_kills'] is not None else float('nan'):>12.3f} "
            f"{diagnostics['average_episode_steps'] if diagnostics['average_episode_steps'] is not None else float('nan'):>12.1f}"
        )
    lines.extend(
        [
            "",
            "Kills, steps, health, ammo and timeout rate are diagnostics; they are not part of the benchmark score.",
            "Scores are not averaged across maps because their reward scales may differ.",
        ]
    )
    text_path = output_dir / "summary.txt"
    tmp_path = text_path.with_suffix(".txt.tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_path, text_path)
    return summary


def _safe_game_variable(env, name):
    try:
        return float(env.get_game_variable(name))
    except Exception as exc:
        logger.warning("Could not read final VizDoom variable %s: %s", name, exc)
        return None


class EpisodeWorker:
    def __init__(
        self,
        cfg_dict,
        ckpt,
        dataset_stats_path,
        device,
        max_steps,
        save_video,
        prompt_index,
        window_visible,
        fps,
    ):
        from omegaconf import OmegaConf
        from gamewam.runtime import build_inference

        cfg = OmegaConf.create(cfg_dict)
        cfg.inference.checkpoint_path = ckpt
        cfg.inference.dataset_stats_path = dataset_stats_path
        cfg.inference.device = device
        cfg.inference.mode = "rollout"

        self.inference = build_inference(cfg)
        self.ckpt = ckpt
        self.dataset_stats_path = dataset_stats_path
        self.device = device
        self.max_steps = int(max_steps)
        self.save_video = bool(save_video)
        self.prompt_index = int(prompt_index)
        self.window_visible = bool(window_visible)
        self.fps = int(fps)
        self.base_seed = int(cfg.inference.get("seed", 0) or 0)
        self.eval_cfg = OmegaConf.to_container(cfg.inference, resolve=True)

    def run_episode(self, scenario, episode_id, episode_dir):
        from gamewam.envs.vizdoom.action_space import (
            ACTION_KEYS,
            action_to_vector,
            vector_to_action,
        )
        from gamewam.envs.vizdoom.env import VizDoomGameEnv
        from gamewam.utils.video_io import save_mp4

        episode_id = str(episode_id)
        episode_dir = Path(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        model_seed = _derive_episode_seed(
            self.base_seed, scenario, episode_id, "model"
        )
        env_seed = _derive_episode_seed(
            self.base_seed, scenario, episode_id, "environment"
        )
        instruction = TASK_PARAPHRASES[scenario][self.prompt_index]
        inference = self.inference
        inference.seed = model_seed
        trace = RolloutTraceLogger(
            episode_dir,
            action_dim=9,
            action_names=ACTION_KEYS,
            binary_action_indices=range(8),
        )
        trace.write_config(
            {
                "scenario": scenario,
                "episode_id": episode_id,
                "instruction": instruction,
                "ckpt": self.ckpt,
                "dataset_stats_path": self.dataset_stats_path,
                "device": self.device,
                "max_steps": self.max_steps,
                "save_video": self.save_video,
                "prompt_index": self.prompt_index,
                "model_seed": model_seed,
                "environment_seed": env_seed,
                "eval_cfg": self.eval_cfg,
                "env": {
                    "frame_skip": 1,
                    "state_dim": 4,
                    "obs_mode": "rgb",
                    "screen_size": [480, 640],
                    "fps": self.fps,
                    "window_visible": self.window_visible,
                },
            }
        )
        inference.inference_logger = trace.on_inference
        env = None
        frames = []
        summed_reward = 0.0
        steps = 0
        wall_start = time.perf_counter()

        try:
            env = VizDoomGameEnv(
                scenario=scenario,
                frame_skip=1,
                state_dim=4,
                seed=env_seed,
                window_visible=self.window_visible,
                obs_mode="rgb",
                screen_size=(480, 640),
            )
            obs = env.reset()
            inference.reset()
            while not obs.done and steps < self.max_steps:
                if self.save_video and obs.image is not None:
                    frames.append(obs.image)
                action_vec = np.asarray(
                    inference.get_action(obs, instruction),
                    dtype=np.float32,
                ).reshape(-1)
                if action_vec.shape != (9,):
                    raise ValueError(
                        f"VizDoom policy must emit a 9-dim action, got {action_vec.shape}."
                    )
                env_action = vector_to_action(action_vec, binarize=True)
                executed_action_vec = action_to_vector(env_action)
                obs = env.step(env_action)
                summed_reward += float(obs.reward)
                trace.on_step(
                    step_idx=steps,
                    action_vec=action_vec,
                    env_action=env_action,
                    executed_action_vec=executed_action_vec,
                    executed_env_action=env_action,
                    reward=obs.reward,
                    done=obs.done,
                    info={"game_vars": obs.info.get("game_vars", {})},
                )
                steps += 1

            timeout = not obs.done and steps >= self.max_steps
            engine_reward = env.get_total_reward()
            if not np.isclose(engine_reward, summed_reward, rtol=1e-6, atol=1e-5):
                raise RuntimeError(
                    "VizDoom cumulative reward mismatch: "
                    f"engine={engine_reward}, summed_steps={summed_reward}."
                )
            result = {
                "scenario": scenario,
                "episode_id": episode_id,
                "episode_reward": float(engine_reward),
                "steps": steps,
                "kills": _safe_game_variable(env, "KILLCOUNT"),
                "final_health": _safe_game_variable(env, "HEALTH"),
                "final_ammo": _safe_game_variable(env, "AMMO2"),
                "timeout": timeout,
                "end_reason": "max_steps" if timeout else "environment_done",
                "wall_time_sec": time.perf_counter() - wall_start,
                "env_seed": env_seed,
                "model_seed": model_seed,
                "instruction": instruction,
            }
            if self.save_video and frames:
                save_mp4(frames, str(episode_dir / "episode.mp4"), fps=self.fps)
            trace.finalize(result)
            (episode_dir / "episode_error.json").unlink(missing_ok=True)
            return result
        except Exception as exc:
            trace.finalize(
                {
                    "scenario": scenario,
                    "episode_id": episode_id,
                    "success": False,
                    "end_reason": "error",
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc),
                    "steps": steps,
                    "summed_reward": summed_reward,
                    "wall_time_sec": time.perf_counter() - wall_start,
                    "env_seed": env_seed,
                    "model_seed": model_seed,
                }
            )
            _dump_json(
                {
                    "scenario": scenario,
                    "episode_id": episode_id,
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc),
                    "traceback": traceback.format_exc(),
                    "steps": steps,
                    "summed_reward": summed_reward,
                    "wall_time_sec": time.perf_counter() - wall_start,
                    "env_seed": env_seed,
                    "model_seed": model_seed,
                },
                episode_dir / "episode_error.json",
            )
            raise
        finally:
            inference.inference_logger = None
            trace.close()
            if env is not None:
                env.close()

    def shutdown(self):
        self.inference.close()


def _write_error_report(output_dir, scenario, episode_id, exc, actor_idx=None):
    _dump_json(
        {
            "scenario": scenario,
            "episode_id": str(episode_id),
            "actor_idx": actor_idx,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": dt.datetime.now().isoformat(),
        },
        Path(output_dir) / "evaluation_error.json",
    )


def _validate_resume_meta(path, new_meta):
    old_meta = _load_json(path, None)
    if old_meta is None:
        _dump_json(new_meta, path)
        return
    for key, value in new_meta.items():
        if old_meta.get(key) != value:
            raise RuntimeError(
                f"Resume mismatch on {key!r}: existing={old_meta.get(key)!r}, "
                f"requested={value!r}. Use a different --output_dir."
            )


def _compose_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = []
    if args.task_config:
        overrides.append(f"task={args.task_config}")
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"),
        version_base=None,
    ):
        cfg = compose(config_name="sim_vizdoom", overrides=overrides)
    cfg.inference.checkpoint_path = str(Path(args.ckpt).resolve())
    cfg.inference.dataset_stats_path = str(Path(args.dataset_stats_path).resolve())
    cfg.inference.device = args.device
    cfg.inference.seed = args.seed
    if args.num_inference_steps is not None:
        cfg.inference.num_inference_steps = args.num_inference_steps
    if args.text_cfg_scale is not None:
        cfg.inference.text_cfg_scale = args.text_cfg_scale

    processor = cfg.data.common.processor
    if int(processor.action_output_dim) != 9:
        raise ValueError(
            f"VizDoom evaluation requires action_output_dim=9, got {processor.action_output_dim}."
        )
    if int(processor.proprio_output_dim) != 4:
        raise ValueError(
            f"VizDoom evaluation requires proprio_output_dim=4, got {processor.proprio_output_dim}."
        )
    return OmegaConf.to_container(cfg, resolve=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GameWAM on GameTARS's four VizDoom maps. The primary "
            "metric is average cumulative environment reward per map."
        )
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument(
        "--task_config",
        default=None,
        help="Optional Hydra task-group override.",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="'all' or a comma-separated subset of the four benchmark maps.",
    )
    parser.add_argument("--episodes_per_map", type=int, default=5)
    parser.add_argument(
        "--split_number",
        type=int,
        default=4,
        help="Maximum concurrent workers; 0 runs in the main process.",
    )
    parser.add_argument("--gpus_per_worker", type=float, default=None)
    parser.add_argument("--ray_progress_timeout_sec", type=float, default=3600.0)
    parser.add_argument("--max_steps", type=int, default=2100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--window_visible", action="store_true")
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=0,
        help="Fixed task paraphrase index (0-4) used for every episode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--text_cfg_scale", type=float, default=None)
    parser.add_argument("--fps", type=int, default=35)
    return parser.parse_args(argv)


def run(args):
    scenarios = _resolve_scenarios(args.scenarios)
    if args.episodes_per_map <= 0:
        raise ValueError("--episodes_per_map must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")
    if args.split_number < 0:
        raise ValueError("--split_number cannot be negative.")
    if args.gpus_per_worker is not None and args.gpus_per_worker < 0:
        raise ValueError("--gpus_per_worker cannot be negative.")
    if not 0 <= args.prompt_index < 5:
        raise ValueError("--prompt_index must be between 0 and 4.")
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not Path(args.dataset_stats_path).is_file():
        raise FileNotFoundError(
            f"Dataset stats not found: {args.dataset_stats_path}"
        )
    if importlib.util.find_spec("vizdoom") is None:
        raise RuntimeError(
            "VizDoom is not installed in the active Python environment. "
            "Install the project-compatible version, e.g. vizdoom==1.2.4."
        )

    cfg_dict = _compose_config(args)
    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "evaluate_results"
        / "vizdoom"
        / f"{Path(args.ckpt).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_stats_path": str(Path(args.dataset_stats_path).resolve()),
        "scenarios": scenarios,
        "episodes_per_map": int(args.episodes_per_map),
        "max_steps": int(args.max_steps),
        "base_seed": int(args.seed),
        "prompt_index": int(args.prompt_index),
        "task_config": args.task_config,
        "num_inference_steps": cfg_dict["inference"]["num_inference_steps"],
        "text_cfg_scale": cfg_dict["inference"]["text_cfg_scale"],
    }
    _validate_resume_meta(output_dir / "run_meta.json", meta)

    results_by_scenario = {}
    pending = []
    for scenario in scenarios:
        result_path = output_dir / scenario / "results.json"
        results = _load_json(result_path, [])
        results.sort(key=lambda item: int(item["episode_id"]))
        completed_ids = {str(item["episode_id"]) for item in results}
        results_by_scenario[scenario] = results
        for episode_id in range(args.episodes_per_map):
            if str(episode_id) not in completed_ids:
                pending.append((scenario, str(episode_id)))

    _write_reports(
        output_dir, scenarios, results_by_scenario, args.episodes_per_map
    )
    if not pending:
        logger.info("All requested VizDoom episodes are already complete.")
        return _summarize(
            scenarios, results_by_scenario, args.episodes_per_map
        )

    gpus_per_worker = (
        float(args.gpus_per_worker)
        if args.gpus_per_worker is not None
        else (1.0 if args.device.startswith("cuda") else 0.0)
    )
    worker_kwargs = {
        "cfg_dict": cfg_dict,
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_stats_path": str(Path(args.dataset_stats_path).resolve()),
        "device": args.device,
        "max_steps": args.max_steps,
        "save_video": args.save_video,
        "prompt_index": args.prompt_index,
        "window_visible": args.window_visible,
        "fps": args.fps,
    }

    def episode_dir(scenario, episode_id):
        path = output_dir / scenario / episode_id
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def handle_result(result):
        scenario = result["scenario"]
        results_by_scenario[scenario].append(result)
        results_by_scenario[scenario].sort(
            key=lambda item: int(item["episode_id"])
        )
        _dump_json(
            results_by_scenario[scenario],
            output_dir / scenario / "results.json",
        )
        summary = _write_reports(
            output_dir, scenarios, results_by_scenario, args.episodes_per_map
        )
        item = summary["maps"][scenario]
        logger.info(
            "[%s] ep=%s reward=%.4f steps=%d; average=%.4f (%d/%d)",
            scenario,
            result["episode_id"],
            result["episode_reward"],
            result["steps"],
            item["average_episode_reward"],
            item["episodes_completed"],
            item["episodes_target"],
        )

    if args.split_number == 0:
        worker = EpisodeWorker(**worker_kwargs)
        try:
            for scenario, episode_id in pending:
                try:
                    handle_result(
                        worker.run_episode(
                            scenario,
                            episode_id,
                            episode_dir(scenario, episode_id),
                        )
                    )
                except Exception as exc:
                    _write_error_report(
                        output_dir, scenario, episode_id, exc
                    )
                    raise
        finally:
            worker.shutdown()
    else:
        max_workers = args.split_number
        if gpus_per_worker > 0 and torch.cuda.is_available():
            gpu_bound = int(torch.cuda.device_count() / gpus_per_worker)
            if gpu_bound < 1:
                raise RuntimeError(
                    f"gpus_per_worker={gpus_per_worker} exceeds "
                    f"{torch.cuda.device_count()} visible GPU(s)."
                )
            max_workers = min(max_workers, gpu_bound)

        import ray

        ray.init(
            ignore_reinit_error=True,
            num_cpus=min(args.split_number, len(pending)),
        )
        remote_options = (
            {"num_gpus": gpus_per_worker} if gpus_per_worker > 0 else {}
        )
        WorkerActor = ray.remote(**remote_options)(EpisodeWorker)
        actors = [
            WorkerActor.remote(**worker_kwargs)
            for _ in range(min(max_workers, len(pending)))
        ]
        queue_index = 0
        in_flight = {}

        def submit(actor_index):
            nonlocal queue_index
            if queue_index >= len(pending):
                return
            scenario, episode_id = pending[queue_index]
            queue_index += 1
            ref = actors[actor_index].run_episode.remote(
                scenario,
                episode_id,
                episode_dir(scenario, episode_id),
            )
            in_flight[ref] = (actor_index, scenario, episode_id)

        for actor_index in range(len(actors)):
            submit(actor_index)

        try:
            while in_flight:
                ready, _ = ray.wait(
                    list(in_flight),
                    num_returns=1,
                    timeout=args.ray_progress_timeout_sec,
                )
                if not ready:
                    actor_index, scenario, episode_id = next(
                        iter(in_flight.values())
                    )
                    exc = TimeoutError(
                        "No VizDoom episode completed for "
                        f"{args.ray_progress_timeout_sec:g} seconds."
                    )
                    _write_error_report(
                        output_dir,
                        scenario,
                        episode_id,
                        exc,
                        actor_idx=actor_index,
                    )
                    raise exc
                ref = ready[0]
                actor_index, scenario, episode_id = in_flight.pop(ref)
                try:
                    result = ray.get(ref)
                except Exception as exc:
                    _write_error_report(
                        output_dir,
                        scenario,
                        episode_id,
                        exc,
                        actor_idx=actor_index,
                    )
                    raise
                handle_result(result)
                submit(actor_index)
        finally:
            for ref in list(in_flight):
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    pass
            shutdown_actor_pool(ray, actors, logger=logger)

    summary = _write_reports(
        output_dir, scenarios, results_by_scenario, args.episodes_per_map
    )
    logger.info("VizDoom evaluation complete: %s", output_dir)
    return summary


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    try:
        run(args)
    except Exception:
        logger.exception("VizDoom evaluation failed.")
        raise


if __name__ == "__main__":
    main()
