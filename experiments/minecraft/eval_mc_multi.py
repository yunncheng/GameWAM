"""GameWAM Minecraft multi-task evaluation with Ray parallelism."""

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.minecraft.utils.draw_utils import (
    load_json,
    dump_json,
    show_success_rate,
    compute_success_rate,
    print_summary_table,
)
from experiments.minecraft.utils.task_sets import (
    TASK_SET_CHOICES,
    resolve_tasks,
    task_category,
)
from experiments.minecraft.utils.eval_lifecycle import (
    cleanup_minecraft_instances,
    consume_retry,
    enqueue_retry_next,
    is_completed_episode_result,
    is_retryable_mc_result,
    is_retryable_minecraft_error,
    make_retryable_mc_result,
)
from gamewam.utils.ray_lifecycle import (
    packed_gpu_actor_capacity,
    shutdown_actor_pool,
    wait_actor_pool_ready,
)
from gamewam.utils.video_io import resize_frame_to_size

logger = logging.getLogger(__name__)


def _derive_episode_seed(base_seed: int, episode_id: str, domain: str, task_name: str = "") -> int:
    """Derive a deterministic, well-mixed per-episode sub-seed for a named domain.

    Uses blake2b with an 8-byte digest so adjacent (base_seed, episode_id) pairs
    never produce correlated sequences across domains or episodes. The result is
    masked into the non-negative signed-64 range accepted by
    ``torch.Generator.manual_seed`` and ``random.Random``.

    ``task_name`` is mixed into the digest so that episode 0 of task A and
    episode 0 of task B receive independent RNG streams (model noise, task
    RNG, inventory, mobs, episode).
    """
    h = hashlib.blake2b(
        f"{base_seed}|{task_name}|{episode_id}|{domain}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def _resolve_gpus_per_worker(device: str, gpus_per_worker):
    """Choose a sensible GPU allocation default for Ray workers."""
    if gpus_per_worker is not None:
        return float(gpus_per_worker)
    return 1.0 if str(device).startswith("cuda") else 0.0


def _to_jsonable(obj):
    """Recursively coerce numpy/torch/Path objects to JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _chunk_stats(chunk: np.ndarray) -> dict:
    """Compress an action chunk [T, 22] into a small stats dict.

    We intentionally avoid persisting the raw tensor (disk-cheap "standard"
    verbosity). These stats are enough to answer: "is this chunk saturating
    camera?", "which binary dims are actually firing?", "is there variance
    or is the model predicting a constant?".
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2:
        return {"error": f"unexpected shape {tuple(chunk.shape)}"}
    stats = {
        "mean": chunk.mean(axis=0).tolist(),
        "std": chunk.std(axis=0).tolist(),
        "min": chunk.min(axis=0).tolist(),
        "max": chunk.max(axis=0).tolist(),
    }
    if chunk.shape[1] >= 3:
        # Dims 2.. are binary in the MC action space. Report how often each
        # fires above the canonical 0.5 threshold — a view of what the model
        # actually predicted (the env binarizes via vector_to_action).
        stats["binary_active_ratio"] = (chunk[:, 2:] >= 0.5).mean(axis=0).tolist()
    return stats


class EpisodeLogger:
    """Persist per-episode inference metadata under ``ep_dir``.

    Writes four artifacts:

    - ``config.json`` — full static config snapshot (prompt, eval_cfg,
      task_config, ...). One-shot at episode start.
    - ``inference_trace.jsonl`` — one line per policy replan; includes the
      exact prompt sent to the model and action-chunk summary stats.
    - ``step_log.jsonl`` — one line per environment step with the executed
      22-dim action vector, decoded env_action, reward, done.
    - ``episode_summary.json`` — one-shot at episode end with outcome.

    All writes are guarded by try/except: logger failures must never
    abort inference. Files are line-flushed so a worker crash leaves a
    useful partial trace on disk.
    """

    def __init__(self, ep_dir: str, fps: int = 20):
        self.ep_dir = ep_dir
        self.fps = int(fps)
        os.makedirs(ep_dir, exist_ok=True)
        self._trace_path = os.path.join(ep_dir, "inference_trace.jsonl")
        self._step_path = os.path.join(ep_dir, "step_log.jsonl")
        self._config_path = os.path.join(ep_dir, "config.json")
        self._summary_path = os.path.join(ep_dir, "episode_summary.json")
        self._pred_video_path = os.path.join(ep_dir, "pred_video.mp4")
        self._pred_video_frames = []
        self._pred_video_size = None
        self._pred_video_error = None
        try:
            self._trace_f = open(self._trace_path, "w", encoding="utf-8")
            self._step_f = open(self._step_path, "w", encoding="utf-8")
        except Exception as e:
            logger.warning("EpisodeLogger: failed to open log files in %s: %s", ep_dir, e)
            self._trace_f = None
            self._step_f = None
        self.replan_count = 0
        self._current_replan_idx = -1  # -1 until the first replan lands
        self.step_count = 0
        self._total_infer_latency = 0.0  # accumulated model inference time
        self._total_predicted_steps = 0   # sum of prediction_horizon across replans
        self._total_executed_steps = 0    # sum of execution_horizon across replans

    @property
    def current_replan_idx(self) -> int:
        """Index of the most recent replan; ``-1`` before any replan fires."""
        return self._current_replan_idx

    def write_config(self, cfg: dict) -> None:
        try:
            payload = dict(cfg)
            payload["episode_logger_version"] = 4
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("EpisodeLogger.write_config failed: %s", e)

    def _to_pil_frame(self, frame) -> Image.Image:
        return frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame))

    def _normalize_pred_video_frame(self, frame) -> Image.Image:
        frame = self._to_pil_frame(frame)
        if self._pred_video_size is None:
            self._pred_video_size = frame.size
        if frame.size != self._pred_video_size:
            frame = resize_frame_to_size(frame, self._pred_video_size)
        return frame

    def _append_pred_video(self, obs_frame, frames, skip_initial_frames: int = 0) -> Optional[str]:
        if frames is None:
            return None
        try:
            pred_frames = list(frames)[int(skip_initial_frames):]
            if not pred_frames:
                return None
            # Model predictions define the artifact resolution (normally
            # 224x224). Resize the raw Minecraft observation with the exact
            # no-crop rule used by GameWAMInference._build_image_tensor.
            pred_frames = [
                self._normalize_pred_video_frame(frame) for frame in pred_frames
            ]
            if not self._pred_video_frames and obs_frame is not None:
                self._pred_video_frames.append(
                    self._normalize_pred_video_frame(obs_frame)
                )
            self._pred_video_frames.extend(pred_frames)
            return self._pred_video_path if self._pred_video_frames else None
        except Exception as e:
            self._pred_video_error = f"{type(e).__name__}: {e}"
            logger.warning("EpisodeLogger._append_pred_video failed: %s", e)
            return None

    def _save_pred_video(self) -> Optional[str]:
        if not self._pred_video_frames:
            return None
        try:
            from gamewam.utils.video_io import save_mp4

            save_mp4(self._pred_video_frames, self._pred_video_path, fps=self.fps)
            return self._pred_video_path
        except Exception as e:
            self._pred_video_error = f"{type(e).__name__}: {e}"
            logger.warning("EpisodeLogger._save_pred_video failed: %s", e)
            return None

    def on_inference(self, payload: dict) -> None:
        """Inference callback — runs inside GameWAMInference._infer_action_chunk*."""
        if self._trace_f is None:
            return
        try:
            replan_idx = self.replan_count
            chunk = payload.get("action_chunk")
            stats = _chunk_stats(chunk) if chunk is not None else None
            pred_video = payload.get("pred_video")
            self._append_pred_video(
                payload.get("obs_image"),
                pred_video,
                int(payload.get("pred_video_skip_initial_frames", 0)),
            )
            self._total_infer_latency += float(payload.get("infer_latency_sec", 0.0) or 0.0)
            self._total_predicted_steps += int(payload.get("prediction_horizon", 0) or 0)
            self._total_executed_steps += int(payload.get("execution_horizon", 0) or 0)
            record = {
                "replan_idx": replan_idx,
                "env_step": int(payload.get("step_count", self.step_count)),
                "timestamp": time.time(),
                "prompt": payload.get("prompt"),
                "task_str": payload.get("task_str"),
                "fine_task": payload.get("fine_task"),
                "coarse_task": payload.get("coarse_task"),
                "num_inference_steps": payload.get("num_inference_steps"),
                "text_cfg_scale": payload.get("text_cfg_scale"),
                "sampling_seed": payload.get("sampling_seed"),
                "base_seed": payload.get("base_seed"),
                "global_replan_index": payload.get("global_replan_index"),
                "prediction_horizon": payload.get("prediction_horizon"),
                "execution_horizon": payload.get("execution_horizon"),
                "infer_latency_sec": payload.get("infer_latency_sec"),
                "action_chunk_shape": (
                    list(chunk.shape) if hasattr(chunk, "shape") else None
                ),
                "action_chunk_stats": stats,
                "action_debug": payload.get("action_debug"),
                "action_route_mask": payload.get("action_route_mask"),
                "action_route_logit": payload.get("action_route_logit"),
                "ar_cycle_step": payload.get("ar_cycle_step"),
                "ar_block_index": payload.get("ar_block_index"),
                "proprio_expected": payload.get("proprio_expected"),
                "proprio_actual": payload.get("proprio_actual"),
                "proprio_raw_indices": payload.get("proprio_raw_indices"),
                "runtime_state": payload.get("runtime_state"),
                "clean_latents_in_cache": payload.get("clean_latents_in_cache"),
                "clean_kv_tokens": payload.get("clean_kv_tokens"),
                "ar_history_mode": payload.get("ar_history_mode"),
            }
            self._trace_f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
            self._trace_f.flush()
            self._current_replan_idx = replan_idx
            self.replan_count += 1
        except Exception as e:
            logger.warning("EpisodeLogger.on_inference failed: %s", e)

    def on_step(
        self,
        step_idx: int,
        action_vec: np.ndarray,
        env_action: dict,
        reward: float,
        done: bool,
        is_post_success_pause: bool = False,
    ) -> None:
        if self._step_f is None:
            return
        try:
            record = {
                "step": int(step_idx),
                "from_replan_idx": self._current_replan_idx,
                "action_vec": np.asarray(action_vec, dtype=np.float32).tolist(),
                "env_action": _to_jsonable(env_action),
                "reward": float(reward),
                "done": bool(done),
                "post_success_pause": bool(is_post_success_pause),
            }
            self._step_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._step_f.flush()
            self.step_count = step_idx + 1
        except Exception as e:
            logger.warning("EpisodeLogger.on_step failed at step %s: %s", step_idx, e)

    def finalize(self, summary: dict) -> None:
        try:
            self._save_pred_video()
            payload = dict(summary)
            payload.setdefault("num_replans", self.replan_count)
            payload.setdefault("episode_logger_version", 4)
            # Pure model inference time (sum of per-replan infer_latency_sec).
            infer_time_sec = float(self._total_infer_latency)
            payload.setdefault("infer_time_sec", round(infer_time_sec, 4))
            # Both frequencies use pure model inference time as the denominator.
            # plan_hz  = prediction_horizon * num_replans / infer_time  (model outputs 16 steps/replan)
            # exec_hz  = execution_horizon * num_replans / infer_time   (only 8 steps/replan are actually used)
            if infer_time_sec > 0:
                payload.setdefault(
                    "plan_hz",
                    round(self._total_predicted_steps / infer_time_sec, 4),
                )
                payload.setdefault(
                    "exec_hz",
                    round(self._total_executed_steps / infer_time_sec, 4),
                )
            else:
                payload.setdefault("plan_hz", 0.0)
                payload.setdefault("exec_hz", 0.0)
            with open(self._summary_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("EpisodeLogger.finalize failed: %s", e)
        finally:
            for f in (self._trace_f, self._step_f):
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass
            self._trace_f = None
            self._step_f = None


# --- Episode worker ---


class EpisodeWorker:
    """Holds a loaded inference object and runs episodes reusing it."""

    def __init__(
        self,
        cfg_dict: dict,
        ckpt: str,
        dataset_stats_path: str,
        device: str,
        difficulty: str,
        max_steps: Optional[int],
        binarize: bool,
        hotbar_argmax: bool,
        render_size: tuple,
        fps: int,
        save_video: bool,
    ):
        from omegaconf import OmegaConf
        from gamewam.runtime import build_inference

        cfg = OmegaConf.create(cfg_dict)
        cfg.inference.checkpoint_path = ckpt
        cfg.inference.dataset_stats_path = dataset_stats_path
        cfg.inference.device = device
        cfg.inference.mode = "rollout"

        self.cfg = cfg
        self.ckpt = ckpt
        self.dataset_stats_path = dataset_stats_path
        self.device = device
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.binarize = binarize
        self.hotbar_argmax = hotbar_argmax
        self.render_size = render_size
        self.fps = fps
        self.save_video = save_video
        self.base_seed = int(cfg.inference.get("seed", 0) or 0)

        # Heavy: load model once (tens of seconds)
        self.inference = build_inference(cfg)

    def ready(self) -> bool:
        """Readiness barrier: callable only after actor construction finishes."""
        return True

    def process_id(self) -> int:
        return os.getpid()

    def _cleanup_minecraft_instances(self) -> None:
        try:
            cleanup_minecraft_instances(parent_pid=os.getpid(), logger=logger)
        except Exception:
            logger.exception("Failed to clean up actor-owned Minecraft instances")

    def run_episode(
        self,
        task_name: str,
        episode_id: str,
        video_dir: str,
    ):
        """Run one episode and return ``(success, steps, episode_id)``.

        Reuses ``self.inference`` (loaded once in ``__init__``). Transient
        Minecraft failures return an explicit retry sentinel; all other
        exceptions propagate to the driver for fail-fast shutdown.
        """
        from omegaconf import OmegaConf
        from gamewam.envs.minecraft.env import MinecraftGameEnv
        from gamewam.envs.minecraft.action_space import vector_to_action
        from gamewam.envs.minecraft.tasks.manager import choose_available_task
        import random

        cfg = self.cfg
        inference = self.inference

        # Derive per-episode sub-seeds from the user-supplied base seed.
        # Each domain gets an independent, well-mixed sub-seed so that
        # (a) the same (base_seed, task_name, episode_id) reproduces the
        # exact same episode, and (b) different episodes / different
        # tasks / different domains never share correlated randomness.
        model_seed = _derive_episode_seed(self.base_seed, str(episode_id), "model", task_name)
        task_seed = _derive_episode_seed(self.base_seed, str(episode_id), "task", task_name)
        inventory_seed = _derive_episode_seed(self.base_seed, str(episode_id), "inventory", task_name)
        mobs_seed = _derive_episode_seed(self.base_seed, str(episode_id), "mobs", task_name)
        episode_seed = _derive_episode_seed(self.base_seed, str(episode_id), "episode", task_name)
        episode_rng = random.Random(episode_seed)

        # Update the inference sampling seed for this episode.  ``reset()``
        # (called below) re-initialises the seed index, so per-replan
        # diffusion noise is derived from this per-episode base.
        inference.seed = model_seed

        ep_logger = EpisodeLogger(video_dir, fps=self.fps)
        bootstrap_start = time.perf_counter()

        def _record_bootstrap_error(exc: Exception, stage: str) -> None:
            ep_logger.finalize({
                "episode_id": str(episode_id),
                "task_name": task_name,
                "success": False,
                "steps": 0,
                "total_reward": 0.0,
                "end_reason": "bootstrap_error",
                "bootstrap_stage": stage,
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
                "traceback": traceback.format_exc(),
                "wall_time_sec": time.perf_counter() - bootstrap_start,
            })

        inference.inference_logger = ep_logger.on_inference

        env = None
        current_stage = "resolve_task"
        try:
            # Resolve task configuration
            task_config = choose_available_task(
                task_name, difficulty=self.difficulty, seed=task_seed,
            )
            actual_max_steps = int(
                self.max_steps or task_config.get("maximum_steps", 600)
            )
            if actual_max_steps <= 0:
                raise ValueError(
                    f"Minecraft episode max_steps must be positive, got {actual_max_steps}"
                )

            # Inject per-episode seeds into task_config so env.reset() can
            # forward them to InitInventoryCallback and SummonMobsCallback.
            task_config.setdefault("callback", {}).setdefault("init_inventory", {})["seed"] = inventory_seed
            task_config["callback"]["mobs_seed"] = mobs_seed

            # Build env (per-episode: Malmo port must be unique)
            current_stage = "build_environment"
            cursor_reference_size = getattr(
                inference.processor, "cursor_reference_size", None
            )
            env = MinecraftGameEnv(
                task_config=task_config,
                render_size=self.render_size,
                fps=self.fps,
                seed=task_config.get("seed", 0),
                cursor_reference_size=cursor_reference_size,
            )

            # Write config.json before the loop starts
            fine_instruction = task_config["task_description"]
            coarse_instruction = None
            task_str_preview = inference._build_task_string(fine_instruction, coarse_instruction)
            final_prompt_preview = inference.prompt_template.format(task=task_str_preview)
            used_dual_task_branch = False
            ep_logger.write_config({
                "episode_id": str(episode_id),
                "task_name": task_name,
                "difficulty": self.difficulty,
                "ckpt": self.ckpt,
                "dataset_stats_path": self.dataset_stats_path,
                "device": self.device,
                "max_steps": int(actual_max_steps),
                "requested_max_steps": self.max_steps,
                "save_video": bool(self.save_video),
                "prompt": {
                    "template": inference.prompt_template,
                    "fine_instruction": fine_instruction,
                    "coarse_instruction": coarse_instruction,
                    "used_dual_task_branch": used_dual_task_branch,
                    "task_str": task_str_preview,
                    "final_prompt": final_prompt_preview,
                },
                "eval_cfg": OmegaConf.to_container(cfg.inference, resolve=True),
                "action_space": {
                    "binarize": bool(self.binarize),
                    "hotbar_argmax": bool(self.hotbar_argmax),
                },
                "env": {
                    "render_size": list(self.render_size),
                    "cursor_reference_size": (
                        list(env.cursor_reference_size)
                        if env.cursor_reference_size is not None else None
                    ),
                    "fps": int(self.fps),
                    "seed": task_config.get("seed", 0),
                },
                "task_config": task_config,
            })

            # Reset for new episode
            current_stage = "reset"
            obs = env.reset()
            inference.reset()

        except Exception as exc:
            traceback_text = traceback.format_exc()
            _record_bootstrap_error(exc, current_stage)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self._cleanup_minecraft_instances()
            if is_retryable_minecraft_error(exc, current_stage):
                return make_retryable_mc_result(
                    task_name=task_name,
                    episode_id=str(episode_id),
                    stage=current_stage,
                    exc=exc,
                    steps=0,
                    traceback_text=traceback_text,
                )
            raise

        # Run episode loop
        total_reward = 0.0
        step_count = 0
        frames = []
        end_reason = "max_steps"
        wall_start = time.perf_counter()

        try:
            for step in range(actual_max_steps):
                if self.save_video and obs.image is not None:
                    frames.append(obs.image)

                current_stage = "inference"
                action_vec = inference.get_action(
                    obs,
                    task_config["task_description"],
                    coarse_instruction=coarse_instruction,
                )
                env_action = vector_to_action(action_vec, binarize=self.binarize, hotbar_argmax=self.hotbar_argmax)
                current_stage = "env_step"
                obs = env.step(env_action)
                total_reward += obs.reward
                step_count += 1

                ep_logger.on_step(
                    step_idx=step,
                    action_vec=action_vec,
                    env_action=env_action,
                    reward=obs.reward,
                    done=obs.done,
                    is_post_success_pause=False,
                )

                # The episode ends at the first positive reward.
                if obs.reward > 0:
                    end_reason = "success"
                    break
                if obs.done:
                    end_reason = "done"
                    break

            # Capture last frame
            if self.save_video and obs.image is not None:
                frames.append(obs.image)

            success = total_reward > 0

            # Post-success pause: a random noop tail so the completion
            # action is visible in the saved video.
            post_success_pause_steps = 0
            post_success_pause_error = None
            if success:
                end_pause = episode_rng.randint(5, 20)
                noop_action = vector_to_action(
                    np.zeros(22, dtype=np.float32), binarize=self.binarize, hotbar_argmax=self.hotbar_argmax)
                for pause_i in range(end_pause):
                    if obs.done:
                        break
                    try:
                        obs = env.step(noop_action)
                    except Exception as pause_exc:
                        if not is_retryable_minecraft_error(pause_exc, "env_step"):
                            raise
                        # The benchmark outcome was already established by a
                        # positive reward. A transport failure in the optional
                        # recording pause must not erase or rerun that success.
                        post_success_pause_error = {
                            "error_type": type(pause_exc).__name__,
                            "error_msg": str(pause_exc),
                        }
                        logger.warning(
                            "[%s] ep=%s MC environment failed during post-success "
                            "pause; preserving the successful rollout: %s",
                            task_name, episode_id, pause_exc,
                        )
                        break
                    post_success_pause_steps += 1
                    ep_logger.on_step(
                        step_idx=step_count + pause_i,
                        action_vec=np.zeros(22, dtype=np.float32),
                        env_action=noop_action,
                        reward=obs.reward,
                        done=obs.done,
                        is_post_success_pause=True,
                    )
                    if self.save_video and obs.image is not None:
                        frames.append(obs.image)
                    if obs.done:
                        break

            env.close()
            env = None
            self._cleanup_minecraft_instances()

            # Save video (non-fatal: a failed encode should not discard a
            # completed episode or abort the evaluation driver)
            if self.save_video and frames:
                try:
                    from gamewam.utils.video_io import save_mp4
                    video_path = os.path.join(video_dir, "episode.mp4")
                    save_mp4([Image.fromarray(f) for f in frames], video_path, fps=self.fps)
                except Exception as video_exc:
                    print(f"  [WARN] episode.mp4 save failed for {episode_id}: {video_exc}", file=sys.stderr)

            ep_logger.finalize({
                "episode_id": str(episode_id),
                "task_name": task_name,
                "success": bool(success),
                "steps": int(step_count),
                "total_reward": float(total_reward),
                "end_reason": end_reason,
                "wall_time_sec": time.perf_counter() - wall_start,
                "post_success_pause_steps": int(post_success_pause_steps),
                "post_success_pause_error": post_success_pause_error,
            })

            return (success, step_count, str(episode_id))

        except Exception as exc:
            traceback_text = traceback.format_exc()
            # Persist whatever we have so debugging isn't blind, then
            # classify it while the original exception is still available.
            ep_logger.finalize({
                "episode_id": str(episode_id),
                "task_name": task_name,
                "success": False,
                "steps": int(step_count),
                "total_reward": float(total_reward),
                "end_reason": "error",
                "error_phase": current_stage,
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
                "traceback": traceback.format_exc(),
                "wall_time_sec": time.perf_counter() - wall_start,
            })
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            self._cleanup_minecraft_instances()
            if is_retryable_minecraft_error(exc, current_stage):
                return make_retryable_mc_result(
                    task_name=task_name,
                    episode_id=str(episode_id),
                    stage=current_stage,
                    exc=exc,
                    steps=step_count,
                    traceback_text=traceback_text,
                )
            raise

    def shutdown(self):
        """Release inference resources. Safe to call multiple times."""
        self._cleanup_minecraft_instances()
        if hasattr(self, "inference"):
            try:
                self.inference.close()
            except Exception:
                pass


# --- Multi-task orchestrator ---


def multi_evaluate(args):
    """Run multi-task evaluation with Ray parallelism."""
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir
    from omegaconf import OmegaConf

    if not np.isfinite(float(args.ray_progress_timeout_sec)) or float(
        args.ray_progress_timeout_sec
    ) <= 0:
        raise ValueError("ray_progress_timeout_sec must be positive")
    if args.max_steps is not None and int(args.max_steps) <= 0:
        raise ValueError("max_steps must be positive when specified")
    if int(args.mc_env_max_retries) < 0:
        raise ValueError("mc_env_max_retries must be non-negative")
    if int(args.episodes_per_task) <= 0:
        raise ValueError("episodes_per_task must be positive")
    if int(args.split_number) < 0:
        raise ValueError("split_number must be non-negative")
    if not np.isfinite(float(args.actor_init_timeout_sec)) or float(
        args.actor_init_timeout_sec
    ) <= 0:
        raise ValueError("actor_init_timeout_sec must be positive")

    # Load Hydra config
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    compose_overrides = []
    if args.task_config:
        # Override the hydra `task` group so AR / non-AR checkpoints can be
        # evaluated without hand-editing sim_minecraft.yaml. The task config
        # pulls in its own /model and /data overrides (e.g. wan_uncond),
        # which transitively select mask_mode, history, etc.
        compose_overrides.append(f"task={args.task_config}")
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name="sim_minecraft", overrides=compose_overrides)

    # Serialize the full cfg for Ray. Worker reconstructs it via
    # OmegaConf.create() and calls runtime.build_inference(cfg).
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if args.disable_history:
        # Counterfactual evaluation preserves the checkpoint and AR geometry
        # while disabling only the history prefix module.
        cfg_dict["model"]["history_config"] = None

    # Apply CLI overrides to cfg.inference, the source of truth for
    # model-side runtime settings.
    inference_cfg = cfg_dict.get("inference", {})
    inference_cfg.update({
        "device": args.device,
        "checkpoint_path": args.ckpt,
        "dataset_stats_path": args.dataset_stats_path,
        "prompt_template": (
            args.prompt_template
            if args.prompt_template is not None
            else inference_cfg.get("prompt_template")
        ),
    })
    cfg_dict["inference"] = inference_cfg

    binarize = cfg.inference.get("binarize", True)
    hotbar_argmax = cfg.inference.get("hotbar_argmax", True)
    render_size = tuple(cfg.inference.render_size)
    fps = cfg.inference.fps

    tasks = resolve_tasks(args.task_set, args.tasks)
    gpus_per_worker = _resolve_gpus_per_worker(args.device, args.gpus_per_worker)
    # NOTE: output_dir is created lazily below, right before the first
    # episode is submitted. Do not pre-create it here — we want an aborted
    # or fully-resumed run to leave no trace on disk.

    if not np.isfinite(gpus_per_worker) or gpus_per_worker < 0:
        raise ValueError(
            f"gpus_per_worker must be finite and >= 0, got {gpus_per_worker}"
        )
    if gpus_per_worker > 0 and not str(args.device).startswith("cuda"):
        raise ValueError("gpus_per_worker > 0 requires --device to be a CUDA device")
    if (
        args.split_number > 0
        and str(args.device).startswith("cuda")
        and gpus_per_worker == 0
    ):
        raise ValueError(
            "Ray CUDA evaluation requires gpus_per_worker > 0 so actors are "
            "GPU-scheduled and isolated; use --split_number 0 for single-process debug"
        )
    if str(args.device).startswith("cuda") and torch.cuda.device_count() == 0:
        raise RuntimeError("--device requests CUDA but no visible GPU is available")

    logger.info(
        "Resolved task_set=%s | num_tasks=%d | split_number=%d | gpus_per_worker=%s",
        args.task_set,
        len(tasks),
        args.split_number,
        gpus_per_worker,
    )

    worker_init_kwargs = dict(
        cfg_dict=cfg_dict,
        ckpt=args.ckpt,
        dataset_stats_path=args.dataset_stats_path,
        device=args.device,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        binarize=binarize,
        hotbar_argmax=hotbar_argmax,
        render_size=render_size,
        fps=fps,
        save_video=args.save_video,
    )

    task_states = {}
    pending_jobs = []  # FIFO: task-major order (task1 all eps, then task2, ...)
    for task_name in tasks:
        task_dir = os.path.join(args.output_dir, task_name.replace(":", "_"))
        log_path = os.path.join(task_dir, "end.json")

        # Only probe existing results — do NOT create the task_dir yet.
        # load_json handles the missing-file case; dump_json / makedirs will
        # create it lazily when we actually have something to write.
        raw_results = load_json(log_path, default=[])
        done_results = []
        seen_done_ids = set()
        for row in raw_results:
            if not is_completed_episode_result(row):
                continue
            row_id = str(row[2])
            row_index = int(row_id)
            if not 0 <= row_index < args.episodes_per_task:
                raise RuntimeError(
                    f"Out-of-range episode id {row_id!r} in {log_path!r}"
                )
            if row_id in seen_done_ids:
                raise RuntimeError(
                    f"Duplicate completed episode id {row_id!r} in {log_path!r}"
                )
            seen_done_ids.add(row_id)
            done_results.append(row)
        migrated_count = len(raw_results) - len(done_results)
        if migrated_count:
            logger.warning(
                "Task %s: treating %d historical zero-step failure(s) as pending",
                task_name, migrated_count,
            )
        done_ids = {str(r[2]) for r in done_results}
        undone_ids = [i for i in range(args.episodes_per_task) if str(i) not in done_ids]
        task_states[task_name] = {
            "task_dir": task_dir,
            "log_path": log_path,
            "done_results": done_results,
        }

        if not undone_ids:
            logger.info(f"Task {task_name}: all {args.episodes_per_task} episodes done, skipping.")
            continue

        logger.info(
            f"Task {task_name}: {len(undone_ids)} episodes remaining "
            f"({len(done_ids)} already done)"
        )
        # NOTE: do NOT pre-create ep_dir (or task_dir) here — both are created
        # lazily at submission time so an aborted / unused run leaves zero
        # empty directories behind.
        for ep_id in undone_ids:
            pending_jobs.append((task_name, str(ep_id)))

    # Create the root output dir only after we know there's work (or at
    # least that this run was legitimately launched). task_dirs / ep_dirs
    # below will create it transitively if needed.
    if pending_jobs:
        os.makedirs(args.output_dir, exist_ok=True)

    # Resume-safety: refuse to mix runs that differ on fairness-critical
    # fields.  Write / validate run_meta.json inside output_dir so a later
    # resume call can detect ckpt / dataset_stats / difficulty / seed /
    # task_config / episodes_per_task drift and abort loudly instead of
    # silently producing a polluted summary.
    run_meta_path = os.path.join(args.output_dir, "run_meta.json")
    new_run_meta = {
        "ckpt": os.path.realpath(args.ckpt),
        "dataset_stats_path": os.path.realpath(args.dataset_stats_path),
        "difficulty": args.difficulty,
        "task_config": args.task_config,
        "episodes_per_task": int(args.episodes_per_task),
        "base_seed": int(inference_cfg.get("seed", 0) or 0),
        "task_set": args.task_set,
        "tasks": list(tasks),
        "max_steps": args.max_steps,
        "save_video": bool(args.save_video),
    }
    if os.path.exists(run_meta_path):
        old_meta = load_json(run_meta_path, default={})
        for key in (
            "ckpt",
            "dataset_stats_path",
            "difficulty",
            "task_config",
            "episodes_per_task",
            "base_seed",
        ):
            if old_meta.get(key) != new_run_meta.get(key):
                raise RuntimeError(
                    f"Resume mismatch on '{key}': "
                    f"existing={old_meta.get(key)!r} vs new={new_run_meta.get(key)!r}. "
                    f"Refusing to mix runs in {args.output_dir!r}. "
                    f"Use a different --output_dir or delete the directory."
                )
    elif pending_jobs:
        dump_json(new_run_meta, run_meta_path)

    # A root error file describes only the invocation that wrote it. Once a
    # resume attempt starts, remove stale status files; a new fatal failure
    # below will write a fresh one.
    if pending_jobs:
        for stale_name in ("evaluation_error.json", "recoverable_mc_failures.json"):
            stale_path = os.path.join(args.output_dir, stale_name)
            try:
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            except OSError as exc:
                logger.warning("Failed to remove stale %s: %s", stale_path, exc)

    logger.info("Total pending episodes across tasks: %d", len(pending_jobs))

    # Incremental summary machinery
    # Tasks are grouped by the benchmark category (Embodied / GUI / Combat,
    # same as the summary table).  As soon as every task in a category is
    # complete, that category is folded into summary.txt / summary.json —
    # no need to wait for the whole run.
    task_groups = {}
    for task_name in tasks:
        task_groups.setdefault(task_category(task_name), []).append(task_name)
    completed_groups = []  # benchmark categories, in completion order
    imaged_tasks = set()

    def _task_complete(task_name: str) -> bool:
        return len(task_states[task_name]["done_results"]) >= args.episodes_per_task

    def _emit_task_image(task_name: str) -> None:
        """Write image.png + log the rate once the task is complete."""
        if task_name in imaged_tasks:
            return
        imaged_tasks.add(task_name)
        state = task_states[task_name]
        done_results = state["done_results"]
        show_success_rate(done_results, os.path.join(state["task_dir"], "image.png"))
        rate = compute_success_rate(done_results)
        logger.info(f"Task {task_name}: success_rate={rate:.1%} ({len(done_results)} episodes)")

    def _write_group_summary(trigger: str) -> None:
        """Regenerate summary.txt/json over all completed groups."""
        summary_tasks = [t for g in completed_groups for t in task_groups[g]]
        print_summary_table(args.output_dir, summary_tasks)
        logger.info(
            "Summary updated (%s): %d/%d tasks across categories [%s]",
            trigger, len(summary_tasks), len(tasks), ", ".join(completed_groups),
        )

    # Resume case: categories already fully done from a previous run get
    # their per-task images and an initial summary right away.
    for group, group_tasks in task_groups.items():
        if all(_task_complete(t) for t in group_tasks):
            for t in group_tasks:
                _emit_task_image(t)
            completed_groups.append(group)
    if completed_groups and pending_jobs:
        _write_group_summary("resume")

    def _handle_result(task_name: str, ep_id: str, result):
        """Validate and record a terminal rollout result into ``end.json``."""
        if not is_completed_episode_result(result):
            raise RuntimeError(f"Worker returned non-terminal result: {result!r}")
        if str(result[2]) != str(ep_id):
            raise RuntimeError(
                "Episode worker returned mismatched result identity: "
                f"submitted={ep_id!r}, returned={result[2]!r}"
            )
        state = task_states[task_name]
        if any(str(existing[2]) == str(ep_id) for existing in state["done_results"]):
            raise RuntimeError(
                f"Refusing to append duplicate episode {ep_id!r} for {task_name!r}"
            )
        state["done_results"].append(result)
        state["done_results"].sort(key=lambda r: int(r[2]))
        dump_json(state["done_results"], state["log_path"])
        logger.info(
            f"  [{task_name}] ep={result[2]} success={result[0]} steps={result[1]}"
        )
        # Task finished → image.png; whole category finished → refresh summary.
        if _task_complete(task_name):
            _emit_task_image(task_name)
            group = task_category(task_name)
            if group not in completed_groups and all(
                _task_complete(t) for t in task_groups[group]
            ):
                completed_groups.append(group)
                _write_group_summary(f"group '{group}' complete")

    def _make_ep_dir(task_name: str, ep_id: str) -> str:
        # Creates task_dir + ep_dir in one shot, right before submission.
        ep_dir = os.path.join(task_states[task_name]["task_dir"], ep_id)
        os.makedirs(ep_dir, exist_ok=True)
        return ep_dir

    def _write_eval_error_report(
        task_name, ep_id, exc, actor_idx=None, traceback_text=None
    ):
        """Persist a root-level evaluation_error.json for post-mortem."""
        import datetime
        report = {
            "task_name": task_name,
            "episode_id": str(ep_id),
            "actor_idx": actor_idx,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "traceback": traceback_text or traceback.format_exc(),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "evaluation_error.json"), "w",
                      encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as report_exc:
            logger.warning("Failed to write evaluation_error.json: %s", report_exc)

    _MAX_ENV_RETRIES = int(args.mc_env_max_retries)
    _ep_retries = {}

    def _register_mc_failure(task_name, ep_id, result):
        if not is_retryable_mc_result(result):
            raise TypeError(f"Expected retryable MC result, got {result!r}")
        if result.get("task_name") != task_name or str(result.get("episode_id")) != str(ep_id):
            raise RuntimeError(
                "Episode worker returned mismatched retry identity: "
                f"submitted={(task_name, str(ep_id))!r}, "
                f"returned={(result.get('task_name'), result.get('episode_id'))!r}"
            )
        requeue, failure_count = consume_retry(
            _ep_retries,
            task_name,
            str(ep_id),
            max_retries=_MAX_ENV_RETRIES,
        )
        record = dict(result)
        record.update({
            "failure_count": failure_count,
            "max_retries": _MAX_ENV_RETRIES,
            "will_requeue": bool(requeue),
            "timestamp": time.time(),
        })
        try:
            with open(
                os.path.join(args.output_dir, "mc_env_retry_history.jsonl"),
                "a",
                encoding="utf-8",
            ) as retry_file:
                retry_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write MC retry history: %s", exc)
        return requeue, failure_count

    if args.split_number <= 0:
        # Single-process: one worker loads the model once and runs all
        # episodes sequentially, reusing the loaded inference object.
        try:
            worker = EpisodeWorker(**worker_init_kwargs)
        except Exception as exc:
            logger.error("Single-process worker initialization failed: %r", exc)
            _write_eval_error_report("__worker_init__", "-1", exc)
            raise
        job_list = list(pending_jobs)
        job_idx = 0
        while job_idx < len(job_list):
            task_name, ep_id = job_list[job_idx]
            job_idx += 1
            ep_dir = _make_ep_dir(task_name, ep_id)
            logger.info(f"  [{task_name}] Episode {ep_id} (single-process)...")
            try:
                result = worker.run_episode(
                    task_name=task_name,
                    episode_id=ep_id,
                    video_dir=ep_dir,
                )
                if is_retryable_mc_result(result):
                    requeue, failure_count = _register_mc_failure(task_name, ep_id, result)
                    if requeue:
                        logger.warning(
                            "  [%s] ep=%s transient MC env failure, re-queuing "
                            "(failure %d/%d retries)",
                            task_name, ep_id, failure_count, _MAX_ENV_RETRIES,
                        )
                        enqueue_retry_next(
                            job_list, job_idx, (task_name, ep_id)
                        )
                    else:
                        exhausted_exc = RuntimeError(
                            f"Minecraft environment retries exhausted for "
                            f"{task_name} episode {ep_id} after {failure_count} failures; "
                            f"last failure at {result['stage']}: "
                            f"{result['error_type']}: {result['error_msg']}"
                        )
                        logger.error("  %s — aborting.", exhausted_exc)
                        _write_eval_error_report(
                            task_name,
                            ep_id,
                            exhausted_exc,
                            traceback_text=result["traceback"],
                        )
                        worker.shutdown()
                        sys.exit(1)
                    continue
                _handle_result(task_name, ep_id, result)
            except Exception as exc:
                logger.error(f"  [{task_name}] ep={ep_id} failed: {exc!r} — aborting.")
                _write_eval_error_report(task_name, ep_id, exc)
                try:
                    worker.shutdown()
                except Exception:
                    pass
                sys.exit(1)
        worker.shutdown()
    elif pending_jobs:
        import ray
        ray_remote_options = {}
        if gpus_per_worker > 0:
            ray_remote_options["num_gpus"] = gpus_per_worker

        # Actor pool scheduler (task-major FIFO sliding window)
        # Keep up to `num_actors` actors alive, each loading the model
        # once in __init__ and reusing it across all episodes assigned to
        # that actor.  As soon as an actor finishes an episode, feed it
        # the next pending job so we never wait for a full batch wave.
        #
        # Actor count is bounded by GPU resources AND pending jobs:
        #   - If jobs <= actors: one actor per job (no reuse, but no waste)
        #   - If jobs > actors: actors pull from a shared queue → reuse
        requested_actors = min(args.split_number, len(pending_jobs))
        max_concurrent = requested_actors
        # Fractional GPU resources are packed per physical device. For example,
        # 0.19 permits floor(1 / 0.19) == 5 actors per GPU, not 1 / 0.19
        # fragments pooled across the whole node.
        if gpus_per_worker > 0 and torch.cuda.is_available():
            gpu_total = torch.cuda.device_count()
            gpu_bound = packed_gpu_actor_capacity(gpu_total, gpus_per_worker)
            if gpu_bound == 0:
                raise ValueError(
                    f"gpus_per_worker={gpus_per_worker} exceeds available GPU "
                    f"total ({gpu_total}). Cannot allocate even one actor — "
                    f"reduce --gpus_per_worker."
                )
            if requested_actors > gpu_bound:
                raise ValueError(
                    "Requested Ray actor count cannot be scheduled with the "
                    "visible per-GPU resources: "
                    f"min(split_number={args.split_number}, pending_jobs="
                    f"{len(pending_jobs)})={requested_actors}, but "
                    f"{gpu_total} GPU(s) with gpus_per_worker="
                    f"{gpus_per_worker} can host at most {gpu_bound} actor(s). "
                    f"Use --split {gpu_bound} or lower."
                )
        num_actors = max_concurrent
        ray.init(
            ignore_reinit_error=True,
            num_cpus=min(args.split_number, len(pending_jobs)),
        )
        EpisodeWorkerActor = ray.remote(**ray_remote_options)(EpisodeWorker)
        actors = []
        actor_pids = []
        try:
            for _ in range(num_actors):
                actors.append(EpisodeWorkerActor.remote(**worker_init_kwargs))
            wait_actor_pool_ready(
                ray,
                actors,
                timeout_s=args.actor_init_timeout_sec,
            )
            actor_pids = ray.get([actor.process_id.remote() for actor in actors])
        except BaseException:
            shutdown_actor_pool(ray, actors, logger=logger)
            raise
        logger.info(
            "%d Ray actors are ready (model loaded once per actor).",
            num_actors,
        )

        # Job queue as a list + index pointer. Transient MC environment
        # failures are inserted at the next unconsumed position; any other
        # exception (actor crash, CUDA OOM, ...) triggers fail-fast shutdown.
        job_queue = list(pending_jobs)
        job_idx = 0
        in_flight = {}  # ref -> (actor_idx, task_name, ep_id)
        abort_immediately = False

        def _submit_next(actor_idx):
            nonlocal job_idx
            if job_idx >= len(job_queue):
                return False
            task_name, ep_id = job_queue[job_idx]
            job_idx += 1
            ep_dir = _make_ep_dir(task_name, ep_id)
            ref = actors[actor_idx].run_episode.remote(
                task_name=task_name,
                episode_id=ep_id,
                video_dir=ep_dir,
            )
            in_flight[ref] = (actor_idx, task_name, ep_id)
            logger.info(f"  submit [{task_name}] ep={ep_id} (actor={actor_idx}, in_flight={len(in_flight)})")
            return True

        consecutive_timeouts = 0
        _MAX_CONSECUTIVE_TIMEOUTS = 10
        try:
            # Keep initial submission inside the cleanup boundary so a local
            # filesystem/Ray submission failure cannot leak existing actors.
            for actor_idx in range(num_actors):
                if not _submit_next(actor_idx):
                    break

            while in_flight:
                ready, _ = ray.wait(
                    list(in_flight.keys()),
                    num_returns=1,
                    timeout=args.ray_progress_timeout_sec,
                )
                if not ready:
                    consecutive_timeouts += 1
                    stuck = [(a, t, e) for a, t, e in in_flight.values()]
                    logger.warning(
                        "No episode completed for %gs (%d/%d), %d still active: %s",
                        args.ray_progress_timeout_sec,
                        consecutive_timeouts, _MAX_CONSECUTIVE_TIMEOUTS,
                        len(in_flight), stuck,
                    )
                    if consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
                        timeout_exc = TimeoutError(
                            "No Ray episode completed for "
                            f"{args.ray_progress_timeout_sec * _MAX_CONSECUTIVE_TIMEOUTS:g}s"
                        )
                        first_actor, first_task, first_ep = next(iter(in_flight.values()))
                        logger.error("Too many consecutive timeouts, aborting.")
                        abort_immediately = True
                        _write_eval_error_report(
                            first_task, first_ep, timeout_exc, actor_idx=first_actor,
                        )
                        for other_ref in list(in_flight.keys()):
                            try:
                                ray.cancel(other_ref)
                            except Exception:
                                pass
                        raise timeout_exc
                    continue
                consecutive_timeouts = 0
                for ref in ready:
                    actor_idx, task_name, ep_id = in_flight.pop(ref)
                    try:
                        result = ray.get(ref)
                    except Exception as exc:
                        logger.error(
                            f"  [{task_name}] ep={ep_id} actor={actor_idx} "
                            f"failed: {exc!r} — aborting."
                        )
                        abort_immediately = True
                        _write_eval_error_report(task_name, ep_id, exc, actor_idx=actor_idx)
                        for other_ref in list(in_flight.keys()):
                            try:
                                ray.cancel(other_ref)
                            except Exception:
                                pass
                        sys.exit(1)
                    if is_retryable_mc_result(result):
                        requeue, failure_count = _register_mc_failure(task_name, ep_id, result)
                        if requeue:
                            logger.warning(
                                "  [%s] ep=%s transient MC env failure, re-queuing "
                                "(failure %d/%d retries)",
                                task_name, ep_id, failure_count, _MAX_ENV_RETRIES,
                            )
                            enqueue_retry_next(
                                job_queue, job_idx, (task_name, ep_id)
                            )
                            _submit_next(actor_idx)
                        else:
                            exhausted_exc = RuntimeError(
                                f"Minecraft environment retries exhausted for "
                                f"{task_name} episode {ep_id} after {failure_count} failures; "
                                f"last failure at {result['stage']}: "
                                f"{result['error_type']}: {result['error_msg']}"
                            )
                            logger.error("  %s — aborting.", exhausted_exc)
                            abort_immediately = True
                            _write_eval_error_report(
                                task_name,
                                ep_id,
                                exhausted_exc,
                                actor_idx=actor_idx,
                                traceback_text=result["traceback"],
                            )
                            for other_ref in list(in_flight.keys()):
                                try:
                                    ray.cancel(other_ref)
                                except Exception:
                                    pass
                            sys.exit(1)
                        continue
                    _handle_result(task_name, ep_id, result)
                    # Refill: feed next job to this now-free actor.
                    _submit_next(actor_idx)
        finally:
            try:
                shutdown_actor_pool(
                    ray,
                    actors,
                    close_timeout_s=0.0 if abort_immediately else 30.0,
                    logger=logger,
                )
            finally:
                for actor_pid in actor_pids:
                    cleanup_minecraft_instances(
                        parent_pid=actor_pid, logger=logger, attempts=10
                    )

    # Only publish final task images and aggregate metrics after every episode
    # has a terminal rollout result. Partial summaries are easy to mistake for
    # official benchmark numbers.
    for task_name in tasks:
        _emit_task_image(task_name)
    print_summary_table(args.output_dir, tasks)
    logger.info(f"All results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GameWAM multi-task MC evaluation (Ray parallel)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_stats_path", type=str, required=True, help="Path to dataset_stats.json")
    parser.add_argument("--task_config", type=str, default=None,
                        help="Hydra `task` group override (e.g. 'mc_uncond_ar_224_4e-5'). "
                             "Selects model/data transitively. Default: use the task "
                             "selected by sim_minecraft.yaml.")
    parser.add_argument("--task_set", type=str, default="custom", choices=TASK_SET_CHOICES,
                        help="Task source: custom CSV, canonical mini benchmark, or all registered tasks")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names; required when --task_set=custom")
    parser.add_argument("--difficulty", type=str, default="zero")
    parser.add_argument("--episodes_per_task", type=int, default=30,
                        help="Number of episodes per task")
    parser.add_argument("--split_number", type=int, default=6,
                        help="Max concurrent Ray workers per batch (0 = single-process debug mode)")
    parser.add_argument(
        "--actor_init_timeout_sec",
        type=float,
        default=1800.0,
        help="Fail if all Ray actors have not loaded their models within this many seconds.",
    )
    parser.add_argument(
        "--ray_progress_timeout_sec",
        type=float,
        default=3600.0,
        help=(
            "Progress-check interval in seconds; abort after 10 consecutive "
            "intervals without any completed episode."
        ),
    )
    parser.add_argument(
        "--mc_env_max_retries",
        type=int,
        default=2,
        help="Additional retries per episode for transient MineStudio/Malmo failures.",
    )
    parser.add_argument("--gpus_per_worker", type=float, default=None,
                        help="Ray GPU resources per worker (default: 1 for CUDA, 0 otherwise)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: from task config)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument(
        "--disable_history",
        action="store_true",
        help="Evaluation-only AR counterfactual: disable history prefix while "
             "keeping checkpoint, AR geometry and current observations fixed.",
    )

    parser.add_argument("--prompt_template", type=str, default=None,
                        help="Prompt template used to train the checkpoint. Current "
                             "Minecraft configs use \"Control the Minecraft player to "
                             "complete the task: {task}\". MUST contain '{task}'. If "
                             "omitted, uses inference.prompt_template from the sim config. "
                             "Mismatch silently breaks text conditioning.")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./evaluate_results/minecraft/{args.task_set}_{time.strftime('%Y%m%d_%H%M%S')}"

    def _terminate(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _terminate)

    cleanup_minecraft_instances(logger=logger, attempts=10)
    try:
        multi_evaluate(args)
    finally:
        cleanup_minecraft_instances(
            parent_pid=os.getpid(), logger=logger, attempts=10
        )
        cleanup_minecraft_instances(logger=logger, attempts=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
