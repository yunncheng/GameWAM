#!/usr/bin/env python3
"""Parallel, exact-target APPO data collection for the four VizDoom maps."""

from __future__ import annotations

import argparse
import errno
import json
import math
import multiprocessing as mp
import os
import queue
import random
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PROJECT_ROOT = _HERE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from appo_common import SCENARIO_TO_SF_ENV, passes_quality_gate  # noqa: E402
from collect_appo_data import (  # noqa: E402
    _load_policy,
    _open_dataset,
    _parse_model_args,
    _rollout_with_loaded,
)


def planned_attempts(remaining: int, keep_top_ratio: float, workers: int) -> int:
    """Size a ranking batch so its top fraction can fill ``remaining``."""

    if remaining <= 0:
        return 0
    return max(int(workers), int(math.ceil(remaining / keep_top_ratio)))


def deterministic_for_attempt(attempt_index: int, fraction: float) -> bool:
    """Stable deterministic/stochastic mix that remains stable after resume."""

    return attempt_index % 100 < int(round(100 * fraction))


def next_attempt_index(output_dir: Path, existing: int) -> int:
    """Find a non-repeating rollout index from persisted episode metadata."""

    episodes = output_dir / "meta" / "episodes.jsonl"
    persisted: list[int] = []
    if episodes.is_file():
        with episodes.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if "rollout_index" in row:
                        persisted.append(int(row["rollout_index"]))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
    if persisted:
        return max(persisted) + 1
    # Episodes without rollout IDs reserve a seed range based on the existing
    # count, preventing collisions with prior collection attempts.
    return max(0, int(existing) * 1000)


def select_batch(
    results: list[dict[str, Any]],
    *,
    scenario: str,
    remaining: int,
    keep_top_ratio: float,
    min_score: float | None,
) -> tuple[list[dict[str, Any]], int]:
    """Apply hard gates, then retain the best requested fraction."""

    passed = [
        result
        for result in results
        if "error" not in result
        and passes_quality_gate(result["metrics"], scenario=scenario)
        and (min_score is None or float(result["score"]) >= min_score)
    ]
    passed.sort(
        key=lambda item: (float(item["score"]), float(item["metrics"]["position_cells"])),
        reverse=True,
    )
    keep = min(remaining, int(math.ceil(keep_top_ratio * len(passed))))
    return passed[:keep], len(passed)


def _episode_seed(base_seed: int, scenario_index: int, attempt_index: int) -> int:
    # VizDoom accepts a signed 32-bit seed. Scenario bands make cross-map
    # provenance explicit while staying well inside that range.
    return int(base_seed + scenario_index * 100_000_000 + attempt_index)


def _set_cpu_limits(threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    value = str(max(1, int(threads)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value


def _worker_loop(
    worker_id: int,
    scenario: str,
    model_dir: str,
    task_queue: Any,
    result_queue: Any,
    max_agent_steps: int,
    temp_dir: str,
    threads_per_worker: int,
    video_crf: int,
    video_preset: str,
) -> None:
    """Load one expert, then service rollout requests until shutdown."""

    _set_cpu_limits(threads_per_worker)
    try:
        import torch

        torch.set_num_threads(max(1, int(threads_per_worker)))
        torch.set_num_interop_threads(1)
        cfg, env, env_info, actor, repeat = _load_policy(Path(model_dir), scenario, 1)
    except Exception:  # noqa: BLE001
        result_queue.put({"worker_fatal": worker_id, "error": traceback.format_exc()})
        return

    try:
        while True:
            task = task_queue.get()
            if task is None:
                return
            attempt_index, seed, deterministic = task
            stem = Path(temp_dir) / f"w{worker_id:02d}_a{attempt_index:09d}"
            video_path = stem.with_suffix(".mp4")
            arrays_path = stem.with_suffix(".npz")
            try:
                candidate = _rollout_with_loaded(
                    scenario=scenario,
                    rollout_index=attempt_index,
                    seed=seed,
                    deterministic=deterministic,
                    max_agent_steps=max_agent_steps,
                    fps=35,
                    video_path=video_path,
                    humanized=False,
                    humanized_turn_scale=1.0,
                    humanized_keep_speed=True,
                    cfg=cfg,
                    env=env,
                    env_info=env_info,
                    actor=actor,
                    policy_action_repeat=repeat,
                    capture_frame_skip=1,
                    video_crf=video_crf,
                    video_preset=video_preset,
                )
                # These arrays are small compared with the video. Avoiding
                # ZIP deflate saves CPU on every candidate worker.
                np.savez(
                    arrays_path,
                    actions=candidate.actions,
                    states=candidate.states,
                    state_is_pad=candidate.state_is_pad,
                )
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "rollout_index": attempt_index,
                        "rollout_seed": seed,
                        "deterministic": deterministic,
                        "video_path": str(video_path),
                        "arrays_path": str(arrays_path),
                        "length": int(len(candidate.actions)),
                        "score": float(candidate.score),
                        "shaped_reward": float(candidate.shaped_reward),
                        "metrics": candidate.metrics,
                        "policy_action_repeat": int(candidate.policy_action_repeat),
                        "video_crf": int(video_crf),
                        "video_preset": str(video_preset),
                    }
                )
            except Exception:  # noqa: BLE001
                video_path.unlink(missing_ok=True)
                arrays_path.unlink(missing_ok=True)
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "rollout_index": attempt_index,
                        "rollout_seed": seed,
                        "error": traceback.format_exc(),
                    }
                )
    finally:
        env.close()


def _remove_result_files(result: dict[str, Any]) -> None:
    for key in ("video_path", "arrays_path"):
        value = result.get(key)
        if value:
            Path(value).unlink(missing_ok=True)


def _write_results(
    *,
    dataset: Any,
    output_dir: Path,
    scenario: str,
    chosen: list[dict[str, Any]],
    instruction_rng: random.Random,
) -> int:
    from PIL import Image as PILImage
    from gamewam.envs.vizdoom.tasks import get_task_paraphrases

    dummy = PILImage.new("RGB", (1, 1))
    paraphrases = get_task_paraphrases(scenario)
    written = 0
    for result in chosen:
        episode_idx = int(dataset.meta.total_episodes)
        for video_key in dataset.meta.video_keys:
            destination = output_dir / dataset.meta.get_video_file_path(episode_idx, video_key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = Path(result["video_path"])
            try:
                # Temporary rollouts live beside the output directory by
                # default. An atomic rename avoids reading and writing every
                # accepted MP4 a second time on Lustre.
                os.replace(source, destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # Explicit --temp-dir may point at another filesystem.
                shutil.copy2(source, destination)
                source.unlink()

        with np.load(result["arrays_path"]) as arrays:
            actions = arrays["actions"]
            states = arrays["states"]
            masks = arrays["state_is_pad"]
            dataset.episode_buffer = dataset.create_episode_buffer()
            instruction = instruction_rng.choice(paraphrases)
            for index in range(len(actions)):
                dataset.add_frame(
                    {
                        "observation.images": dummy,
                        "observation.state": states[index],
                        "observation.state_dim_is_pad": masks[index],
                        "action": actions[index],
                    },
                    task=[instruction, instruction],
                )
        dataset.save_episode(
            raw_file_name=f"{scenario}_{episode_idx:06d}",
            extra_fields={
                "scenario": scenario,
                "policy": "sample_factory_appo_rnn",
                "capture_mode": "tick_smooth_native",
                "capture_frame_skip": 1,
                "policy_action_repeat": result["policy_action_repeat"],
                "humanized": False,
                "true_objective": result["score"],
                "shaped_reward": result["shaped_reward"],
                "rollout_index": result["rollout_index"],
                "rollout_seed": result["rollout_seed"],
                "deterministic": result["deterministic"],
                "worker_id": result["worker_id"],
                "video_crf": result["video_crf"],
                "video_preset": result["video_preset"],
                **result["metrics"],
            },
        )
        written += 1
    return written


def _scenario_coordinator(
    scenario: str,
    scenario_index: int,
    model_dir: str,
    args: argparse.Namespace,
) -> None:
    """Own one scenario dataset, its workers, filtering, ranking and writes."""

    _set_cpu_limits(args.threads_per_worker)
    worker_count = int(args.workers_by_scenario.get(scenario, args.workers_per_scenario))
    attempt_batch_size = int(args.attempt_batch_size or max(8, worker_count * 4))
    output_dir = Path(args.output_dir).resolve() / scenario
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / "meta" / "info.json"
    episodes_path = output_dir / "meta" / "episodes.jsonl"
    if info_path.is_file() != episodes_path.is_file():
        raise RuntimeError(f"Partially written dataset at {output_dir}; repair or move it first")

    existing = 0
    if info_path.is_file():
        with info_path.open("r", encoding="utf-8") as handle:
            existing = int(json.load(handle).get("total_episodes", 0))
    target = int(args.target_episodes_per_scenario)
    if existing >= target:
        print(f"[{scenario}] already has {existing} episodes (target={target}); nothing to do", flush=True)
        return

    attempt_index = next_attempt_index(output_dir, existing)
    attempts_used = 0
    dataset = None
    instruction_rng = random.Random(args.seed + scenario_index * 10_000 + existing)
    min_score = args.min_scores.get(scenario)
    temp_parent = Path(args.temp_dir).resolve() if args.temp_dir else output_dir.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")

    with tempfile.TemporaryDirectory(prefix=f"appo_{scenario}_", dir=temp_parent) as temp_dir:
        # Keep this queue unbounded: if every worker fails during policy
        # initialization, a bounded queue could block the coordinator before
        # it gets a chance to consume the fatal result messages.
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    scenario,
                    model_dir,
                    task_queue,
                    result_queue,
                    args.max_agent_steps,
                    temp_dir,
                    args.threads_per_worker,
                    args.video_crf,
                    args.video_preset,
                ),
                name=f"appo-{scenario}-w{worker_id}",
            )
            for worker_id in range(worker_count)
        ]
        for worker in workers:
            worker.start()

        def enqueue_wave(remaining: int) -> list[tuple[int, int, bool]]:
            nonlocal attempt_index, attempts_used
            batch_size = planned_attempts(remaining, args.keep_top_ratio, worker_count)
            # Rank in bounded waves. A production target can be tens of
            # thousands of episodes; materializing all target/ratio
            # candidates at once would consume unbounded temporary disk.
            batch_size = min(batch_size, attempt_batch_size)
            available = args.max_attempts_per_scenario - attempts_used
            batch_size = min(batch_size, available)
            if batch_size <= 0:
                raise RuntimeError(
                    f"[{scenario}] exhausted {args.max_attempts_per_scenario} attempts at "
                    f"{existing}/{target} accepted episodes"
                )
            tasks: list[tuple[int, int, bool]] = []
            for _ in range(batch_size):
                deterministic = deterministic_for_attempt(
                    attempt_index, args.deterministic_fraction
                )
                seed = _episode_seed(args.seed, scenario_index, attempt_index)
                tasks.append((attempt_index, seed, deterministic))
                attempt_index += 1
            for task in tasks:
                task_queue.put(task)
            attempts_used += len(tasks)
            return tasks

        def receive_wave(tasks: list[tuple[int, int, bool]]) -> list[dict[str, Any]]:
            batch: list[dict[str, Any]] = []
            while len(batch) < len(tasks):
                try:
                    result = result_queue.get(timeout=10)
                except queue.Empty:
                    dead = [worker for worker in workers if not worker.is_alive()]
                    if dead:
                        codes = [(worker.name, worker.exitcode) for worker in dead]
                        raise RuntimeError(f"[{scenario}] rollout workers died: {codes}")
                    continue
                if "worker_fatal" in result:
                    raise RuntimeError(
                        f"[{scenario}] worker {result['worker_fatal']} failed to initialize:\n"
                        f"{result['error']}"
                    )
                batch.append(result)
            return batch

        try:
            tasks = enqueue_wave(target - existing)
            while existing < target:
                batch = receive_wave(tasks)
                remaining = target - existing
                chosen, passed_count = select_batch(
                    batch,
                    scenario=scenario,
                    remaining=remaining,
                    keep_top_ratio=args.keep_top_ratio,
                    min_score=min_score,
                )
                errors = sum("error" in result for result in batch)
                print(
                    f"[{scenario}] attempts={attempts_used} batch={len(batch)} "
                    f"gate_pass={passed_count} selected={len(chosen)} "
                    f"errors={errors} progress={existing}/{target}",
                    flush=True,
                )
                for result in batch:
                    if "error" in result:
                        print(
                            f"[{scenario}] attempt {result.get('rollout_index')} failed: "
                            f"{str(result['error']).splitlines()[-1]}",
                            flush=True,
                        )

                # Start the next rollout wave before serial LeRobot writes.
                # While the coordinator builds parquet/meta, workers keep the
                # CPU and encoders busy instead of waiting at a batch barrier.
                projected_total = existing + len(chosen)
                next_tasks = (
                    enqueue_wave(target - projected_total)
                    if projected_total < target
                    else None
                )

                if chosen:
                    if dataset is None:
                        dataset = _open_dataset(output_dir, 35, Path(chosen[0]["video_path"]))
                    written = _write_results(
                        dataset=dataset,
                        output_dir=output_dir,
                        scenario=scenario,
                        chosen=chosen,
                        instruction_rng=instruction_rng,
                    )
                    existing += written
                    print(f"[{scenario}] wrote {written}; total={existing}/{target}", flush=True)

                for result in batch:
                    _remove_result_files(result)
                if existing < target:
                    assert next_tasks is not None
                    tasks = next_tasks
        finally:
            for _ in workers:
                try:
                    task_queue.put_nowait(None)
                except queue.Full:
                    break
            for worker in workers:
                worker.join(timeout=20)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            task_queue.close()
            result_queue.close()

    if existing != target:
        raise RuntimeError(f"[{scenario}] expected exactly {target}, got {existing}")
    print(f"[{scenario}] DONE: exactly {existing} episodes at {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only native 35 FPS APPO collection with exact targets"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", action="append", required=True, metavar="SCENARIO=PATH")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIO_TO_SF_ENV))
    parser.add_argument("--target-episodes-per-scenario", type=int, required=True)
    parser.add_argument("--workers-per-scenario", type=int, default=2)
    parser.add_argument(
        "--workers-by-scenario",
        type=json.loads,
        default={},
        metavar="JSON",
        help="Optional per-map worker counts; overrides --workers-per-scenario",
    )
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--video-preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"),
        default="medium",
    )
    parser.add_argument(
        "--attempt-batch-size",
        type=int,
        default=0,
        help="Ranking wave per scenario; 0 chooses max(8, workers*4)",
    )
    parser.add_argument("--keep-top-ratio", type=float, default=0.5)
    parser.add_argument("--deterministic-fraction", type=float, default=0.75)
    parser.add_argument(
        "--min-scores",
        type=json.loads,
        default={},
        metavar="JSON",
        help='Optional per-map thresholds, e.g. \'{"battle1": 20}\'',
    )
    parser.add_argument("--max-agent-steps", type=int, default=600)
    parser.add_argument(
        "--max-attempts-per-scenario",
        type=int,
        default=0,
        help="0 chooses max(100, target*20)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()

    if args.target_episodes_per_scenario <= 0:
        parser.error("--target-episodes-per-scenario must be positive")
    if args.workers_per_scenario <= 0 or args.threads_per_worker <= 0:
        parser.error("worker and thread counts must be positive")
    if args.max_agent_steps <= 0:
        parser.error("--max-agent-steps must be positive")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be in [0, 51]")
    if not 0.0 < args.keep_top_ratio <= 1.0:
        parser.error("--keep-top-ratio must be in (0, 1]")
    if not 0.0 <= args.deterministic_fraction <= 1.0:
        parser.error("--deterministic-fraction must be in [0, 1]")
    if not isinstance(args.min_scores, dict):
        parser.error("--min-scores must decode to a JSON object")
    if not isinstance(args.workers_by_scenario, dict):
        parser.error("--workers-by-scenario must decode to a JSON object")
    try:
        args.min_scores = {str(key): float(value) for key, value in args.min_scores.items()}
    except (TypeError, ValueError):
        parser.error("all --min-scores values must be numeric")
    try:
        args.workers_by_scenario = {
            str(key): int(value) for key, value in args.workers_by_scenario.items()
        }
    except (TypeError, ValueError):
        parser.error("all --workers-by-scenario values must be integers")
    if any(value <= 0 for value in args.workers_by_scenario.values()):
        parser.error("all --workers-by-scenario values must be positive")

    # Set thread-related environment variables before spawning.  Spawned
    # interpreters import NumPy while loading this module, so doing this only
    # inside the worker function would be too late for some BLAS runtimes.
    _set_cpu_limits(args.threads_per_worker)
    models = _parse_model_args(args.model)
    scenarios = args.scenarios or list(SCENARIO_TO_SF_ENV)
    missing = [scenario for scenario in scenarios if scenario not in models]
    if missing:
        parser.error(f"missing --model entries for {missing}")
    unknown_scores = sorted(set(args.min_scores) - set(SCENARIO_TO_SF_ENV))
    if unknown_scores:
        parser.error(f"unknown scenarios in --min-scores: {unknown_scores}")
    unknown_workers = sorted(set(args.workers_by_scenario) - set(SCENARIO_TO_SF_ENV))
    if unknown_workers:
        parser.error(f"unknown scenarios in --workers-by-scenario: {unknown_workers}")
    max_workers = max(
        [args.workers_by_scenario.get(scenario, args.workers_per_scenario) for scenario in scenarios]
    )
    if args.attempt_batch_size < 0:
        parser.error("--attempt-batch-size cannot be negative")
    if args.attempt_batch_size and args.attempt_batch_size < max_workers:
        parser.error("--attempt-batch-size must be at least the largest scenario worker count")
    if args.max_attempts_per_scenario < 0:
        parser.error("--max-attempts-per-scenario cannot be negative")
    args.max_attempts_per_scenario = args.max_attempts_per_scenario or max(
        100, args.target_episodes_per_scenario * 20
    )

    ctx = mp.get_context("spawn")
    coordinators = []
    for scenario_index, scenario in enumerate(scenarios):
        process = ctx.Process(
            target=_scenario_coordinator,
            args=(scenario, scenario_index, str(models[scenario]), args),
            name=f"appo-{scenario}-coordinator",
        )
        process.start()
        coordinators.append(process)
    failed = []
    for process in coordinators:
        process.join()
        if process.exitcode != 0:
            failed.append((process.name, process.exitcode))
    if failed:
        raise SystemExit(f"Scenario coordinators failed: {failed}")
    print("ALL SCENARIOS DONE", flush=True)


if __name__ == "__main__":
    main()
