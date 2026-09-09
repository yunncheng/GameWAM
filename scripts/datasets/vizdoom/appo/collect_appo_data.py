#!/usr/bin/env python3
"""Collect high-quality four-map VizDoom trajectories with APPO-RNN experts.

This collector intentionally runs the policy in Sample Factory's own wrapped
environment. A policy action is chosen at the checkpoint's original cadence
(normally every four Doom ticks), but the environment advances one tick at a
time and records every intermediate frame at 35 FPS. This matches Sample
Factory's smooth evaluation semantics without changing the policy observation
or recurrent-state cadence. Model inference is always on CPU.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SRC_ROOT = _PROJECT_ROOT / "src"
for path in (_PROJECT_ROOT, _SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from appo_common import (
    ACTION_DIM,
    ACTION_KEYS,
    SCENARIO_TO_SF_ENV,
    canonical_action_from_flat,
    humanize_flat_action,
    passes_quality_gate,
    state_from_info,
    trajectory_quality,
)
from gamewam.datasets.schemas.vizdoom import (
    attach_or_validate_vizdoom_state_schema,
    vizdoom_state_features,
)


@dataclass
class Candidate:
    scenario: str
    rollout_index: int
    video_path: Path
    actions: np.ndarray
    states: np.ndarray
    state_is_pad: np.ndarray
    score: float
    shaped_reward: float
    metrics: dict[str, float]
    policy_action_repeat: int
    capture_frame_skip: int
    humanized: bool


class StreamingMp4:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: int,
        *,
        crf: int = 18,
        preset: str = "medium",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.count = 0
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", str(crf),
                "-preset", str(preset), "-threads", "1",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"Unexpected render shape {rgb.shape}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(rgb.tobytes())
        self.count += 1

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.stdin = None
        stderr = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} (rc={code}): {stderr}")


def _parse_model_args(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be SCENARIO=PATH, got {value!r}")
        scenario, path = value.split("=", 1)
        if scenario not in SCENARIO_TO_SF_ENV:
            raise ValueError(f"Unknown scenario {scenario!r}")
        result[scenario] = Path(path).resolve()
    return result


def _first_info(infos: Any) -> dict[str, Any]:
    if isinstance(infos, (list, tuple)):
        return dict(infos[0])
    return dict(infos or {})


def _action_components(processed_actions: Any) -> list[int]:
    if isinstance(processed_actions, list):
        return [int(np.asarray(component).reshape(-1)[0]) for component in processed_actions]
    array = np.asarray(processed_actions)
    if array.ndim == 2:
        array = array[0]
    return [int(value) for value in array.reshape(-1)]


def _load_policy(experiment_dir: Path, scenario: str, capture_frame_skip: int):
    import torch
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.utils.attr_dict import AttrDict
    from sf_examples.vizdoom.train_vizdoom import (
        parse_vizdoom_cfg,
        register_vizdoom_components,
    )

    expected_env = SCENARIO_TO_SF_ENV[scenario]
    register_vizdoom_components()
    argv = [
        f"--env={expected_env}",
        f"--experiment={experiment_dir.name}",
        f"--train_dir={experiment_dir.parent}",
        "--device=cpu",
        "--env_gpu_actions=False",
        "--env_gpu_observations=False",
    ]
    cfg = load_from_checkpoint(parse_vizdoom_cfg(argv=argv, evaluation=True))
    if cfg.env != expected_env:
        raise ValueError(f"{experiment_dir} was trained for {cfg.env}, expected {expected_env}")
    training_frameskip = int(cfg.env_frameskip)
    if training_frameskip % capture_frame_skip != 0:
        raise ValueError(
            f"Checkpoint frameskip {training_frameskip} is not divisible by "
            f"capture frameskip {capture_frame_skip}"
        )
    policy_action_repeat = training_frameskip // capture_frame_skip
    cfg.env_frameskip = capture_frame_skip
    cfg.eval_env_frameskip = capture_frame_skip
    cfg.device = "cpu"
    cfg.env_gpu_actions = False
    cfg.env_gpu_observations = False
    cfg.num_envs = 1

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode="rgb_array",
    )
    env_info = extract_env_info(env, cfg)

    # Policy observations still pass through the checkpoint's 128x72 resize,
    # but VizDoom renders at 640x480 so GameWAM receives natural high-res RGB.
    import vizdoom as vzd

    doom = env.unwrapped
    if getattr(doom, "initialized", False):
        raise RuntimeError("VizDoom initialized before capture resolution override")
    doom.screen_w = 640
    doom.screen_h = 480
    doom.screen_resolution = vzd.ScreenResolution.RES_640X480
    doom.calc_observation_space()

    actor = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor.eval()
    actor.model_to_device(torch.device("cpu"))
    checkpoints = sorted((experiment_dir / "checkpoint_p0").glob("best_prepared_*.pth"))
    if not checkpoints:
        checkpoints = Learner.get_checkpoints(
            experiment_dir / "checkpoint_p0", "checkpoint_prepared_*"
        )
    if not checkpoints:
        raise FileNotFoundError(f"No prepared checkpoint found below {experiment_dir}")
    checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    actor.load_state_dict(checkpoint["model"], strict=True)
    return cfg, env, env_info, actor, policy_action_repeat


def _rollout_with_loaded(
    *,
    scenario: str,
    rollout_index: int,
    seed: int,
    deterministic: bool,
    max_agent_steps: int,
    fps: int,
    video_path: Path,
    humanized: bool,
    humanized_turn_scale: float,
    humanized_keep_speed: bool,
    cfg: Any,
    env: Any,
    env_info: Any,
    actor: Any,
    policy_action_repeat: int,
    capture_frame_skip: int,
    video_crf: int = 18,
    video_preset: str = "medium",
) -> Candidate:
    """Run one episode with a preloaded policy/env pair.

    Parallel workers call this repeatedly so checkpoint loading and environment
    construction happen once per worker rather than once per episode.
    """

    import torch
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
    from sample_factory.model.model_utils import get_rnn_size

    doom = env.unwrapped
    doom.seed(seed)
    obs, infos = env.reset()
    current_info = _first_info(infos)
    rnn_states = torch.zeros((1, get_rnn_size(cfg)), dtype=torch.float32)

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    state_masks: list[np.ndarray] = []
    total_reward = 0.0
    terminal_info = current_info
    writer = StreamingMp4(
        video_path,
        width=640,
        height=480,
        fps=fps,
        crf=video_crf,
        preset=video_preset,
    )
    try:
        with torch.no_grad():
            finished = False
            for _ in range(max_agent_steps):
                normalized = prepare_and_normalize_obs(actor, obs)
                outputs = actor(normalized, rnn_states)
                policy_actions = outputs["actions"]
                if deterministic:
                    policy_actions = argmax_actions(actor.action_distribution())
                if policy_actions.ndim == 1:
                    policy_actions = unsqueeze_tensor(policy_actions, dim=-1)
                processed = preprocess_actions(env_info, policy_actions)
                components = _action_components(processed)
                rnn_states = outputs["new_rnn_states"]
                policy_flat = np.asarray(
                    doom._convert_actions(components), dtype=np.float32
                )

                # Sample Factory's official smooth evaluation advances the
                # environment at one tick and repeats the chosen policy action
                # four times. Record each tick and its actual action label.
                for repeat_index in range(policy_action_repeat):
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("Sample Factory env.render() returned None")
                    flat = policy_flat
                    if humanized:
                        flat = humanize_flat_action(
                            scenario,
                            policy_flat,
                            turn_scale=humanized_turn_scale,
                            keep_speed=humanized_keep_speed,
                            repeat_index=repeat_index,
                        )

                    writer.write(frame)
                    action = canonical_action_from_flat(scenario, flat)
                    state, state_is_pad = state_from_info(current_info)
                    actions.append(action)
                    states.append(state)
                    state_masks.append(state_is_pad)

                    # VizdoomEnv explicitly supports externally supplied flat
                    # actions. This ensures the governed action, not the raw
                    # policy output, is what the game executes.
                    doom._actions_flattened = flat.tolist()
                    obs, reward, terminated, truncated, infos = env.step(processed)
                    total_reward += float(reward.reshape(-1)[0].item())
                    terminal_info = _first_info(infos)
                    current_info = dict(terminal_info.get("reset_info", terminal_info))
                    done = bool(make_dones(terminated, truncated).reshape(-1)[0].item())
                    if done:
                        finished = True
                        break
                if finished:
                    break
    finally:
        writer.close()

    action_array = np.stack(actions)
    state_array = np.stack(states)
    mask_array = np.stack(state_masks)
    score = float(terminal_info.get("true_objective", terminal_info.get("USER2", total_reward)))
    metrics = trajectory_quality(action_array, state_array, mask_array)
    return Candidate(
        scenario=scenario,
        rollout_index=rollout_index,
        video_path=video_path,
        actions=action_array,
        states=state_array,
        state_is_pad=mask_array,
        score=score,
        shaped_reward=total_reward,
        metrics=metrics,
        policy_action_repeat=policy_action_repeat,
        capture_frame_skip=capture_frame_skip,
        humanized=humanized,
    )


def _rollout(
    scenario: str,
    experiment_dir: Path,
    rollout_index: int,
    seed: int,
    deterministic: bool,
    max_agent_steps: int,
    fps: int,
    video_path: Path,
    capture_frame_skip: int,
    humanized: bool,
    humanized_turn_scale: float,
    humanized_keep_speed: bool,
) -> Candidate:
    """Run one serial rollout with an isolated policy environment."""

    cfg, env, env_info, actor, policy_action_repeat = _load_policy(
        experiment_dir, scenario, capture_frame_skip
    )
    try:
        return _rollout_with_loaded(
            scenario=scenario,
            rollout_index=rollout_index,
            seed=seed,
            deterministic=deterministic,
            max_agent_steps=max_agent_steps,
            fps=fps,
            video_path=video_path,
            humanized=humanized,
            humanized_turn_scale=humanized_turn_scale,
            humanized_keep_speed=humanized_keep_speed,
            cfg=cfg,
            env=env,
            env_info=env_info,
            actor=actor,
            policy_action_repeat=policy_action_repeat,
            capture_frame_skip=capture_frame_skip,
        )
    finally:
        env.close()


def _make_features(video_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images": {
            "dtype": "video", "shape": (3, 480, 640),
            "names": ["channels", "height", "width"], "info": video_info,
        },
        **vizdoom_state_features(),
        "action": {
            "dtype": "float32", "shape": (ACTION_DIM,), "names": list(ACTION_KEYS),
        },
    }


def _open_dataset(output_dir: Path, fps: int, probe_video: Path):
    from scripts.datasets.common.video import build_video_info
    from gamewam.datasets.lerobot.lerobot_dataset import LeRobotDataset

    meta = output_dir / "meta"
    complete = (meta / "info.json").is_file() and (meta / "episodes.jsonl").is_file()
    if complete:
        dataset = LeRobotDataset(
            repo_id=str(output_dir), root=output_dir,
            is_compute_episode_stats_image=False,
        )
        existing_shape = tuple(dataset.meta.features["action"]["shape"])
        if existing_shape != (ACTION_DIM,):
            raise ValueError(f"Existing dataset action shape is {existing_shape}, expected {(ACTION_DIM,)}")
        if int(dataset.meta.fps) != int(fps):
            raise ValueError(
                f"Existing dataset FPS is {dataset.meta.fps}, requested {fps}; "
                "write smooth capture to a new output directory"
            )
        attach_or_validate_vizdoom_state_schema(dataset)
        return dataset
    if (meta / "info.json").exists():
        raise RuntimeError(f"Partially written dataset at {output_dir}; move or repair it first")
    dataset = LeRobotDataset.create(
        repo_id=str(output_dir), root=output_dir, fps=fps,
        features=_make_features(build_video_info(probe_video, fps=fps)),
        robot_type="vizdoom_appo", use_videos=True,
        is_compute_episode_stats_image=False,
    )
    attach_or_validate_vizdoom_state_schema(dataset)
    return dataset


def collect(args: argparse.Namespace) -> int:
    # Defense-in-depth: even a CUDA-enabled torch build must not see a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model_dirs = _parse_model_args(args.model)
    scenarios = args.scenarios or list(SCENARIO_TO_SF_ENV)
    missing = [scenario for scenario in scenarios if scenario not in model_dirs]
    if missing:
        raise ValueError(f"Missing --model entries for {missing}")

    random.seed(args.seed)
    selected: list[Candidate] = []
    temp_parent = Path(args.temp_dir) if args.temp_dir else None
    with tempfile.TemporaryDirectory(prefix="gamewam_appo_", dir=temp_parent) as temp:
        temp_root = Path(temp)
        for scenario_index, scenario in enumerate(scenarios):
            candidates: list[Candidate] = []
            print(f"\n=== {scenario}: APPO CPU rollouts ===")
            deterministic_count = int(round(args.episodes_per_scenario * args.deterministic_fraction))
            for episode in range(args.episodes_per_scenario):
                candidate = _rollout(
                    scenario=scenario,
                    experiment_dir=model_dirs[scenario],
                    rollout_index=episode,
                    seed=args.seed + 10000 * scenario_index + episode,
                    deterministic=episode < deterministic_count,
                    max_agent_steps=args.max_agent_steps,
                    fps=args.fps,
                    video_path=temp_root / f"{scenario}_{episode:04d}.mp4",
                    capture_frame_skip=args.capture_frame_skip,
                    humanized=args.humanized,
                    humanized_turn_scale=args.humanized_turn_scale,
                    humanized_keep_speed=args.humanized_keep_speed,
                )
                metrics = candidate.metrics
                quality_ok = passes_quality_gate(metrics, scenario=scenario)
                print(
                    f"  ep={episode:03d} T={len(candidate.actions):4d} score={candidate.score:7.2f} "
                    f"reward={candidate.shaped_reward:7.2f} det={episode < deterministic_count} "
                    f"idle={metrics['idle_ratio']:.3f} flip={metrics['turn_flip_ratio']:.3f} "
                    f"stuck={metrics['forward_stuck_ratio']:.3f} cells={metrics['position_cells']:.0f} "
                    f"gate={'pass' if quality_ok else 'DROP'}"
                )
                if quality_ok and (args.min_score is None or candidate.score >= args.min_score):
                    candidates.append(candidate)
            candidates.sort(key=lambda item: (item.score, item.metrics["position_cells"]), reverse=True)
            keep = max(1, math.ceil(args.keep_top_ratio * len(candidates))) if candidates else 0
            chosen = candidates[:keep]
            print(f"  selected {len(chosen)}/{args.episodes_per_scenario} rollouts")
            selected.extend(chosen)

        if not selected:
            raise RuntimeError("No rollout passed the score and trajectory-quality gates")

        output_dir = Path(args.output_dir).resolve()
        dataset = _open_dataset(output_dir, args.fps, selected[0].video_path)
        from PIL import Image as PILImage
        from gamewam.envs.vizdoom.tasks import get_task_paraphrases

        dummy = PILImage.new("RGB", (1, 1))
        rng = random.Random(args.seed)
        for candidate in selected:
            episode_idx = dataset.meta.total_episodes
            for video_key in dataset.meta.video_keys:
                destination = output_dir / dataset.meta.get_video_file_path(episode_idx, video_key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate.video_path, destination)
            dataset.episode_buffer = dataset.create_episode_buffer()
            instruction = rng.choice(get_task_paraphrases(candidate.scenario))
            for index in range(len(candidate.actions)):
                dataset.add_frame(
                    {
                        "observation.images": dummy,
                        "observation.state": candidate.states[index],
                        "observation.state_dim_is_pad": candidate.state_is_pad[index],
                        "action": candidate.actions[index],
                    },
                    task=[instruction, instruction],
                )
            dataset.save_episode(
                raw_file_name=f"{candidate.scenario}_{episode_idx:06d}",
                extra_fields={
                    "scenario": candidate.scenario,
                    "policy": "sample_factory_appo_rnn",
                    "capture_mode": (
                        "tick_smooth"
                        if candidate.capture_frame_skip == 1
                        else "legacy_step"
                    ),
                    "capture_frame_skip": candidate.capture_frame_skip,
                    "policy_action_repeat": candidate.policy_action_repeat,
                    "humanized": candidate.humanized,
                    "true_objective": candidate.score,
                    "shaped_reward": candidate.shaped_reward,
                    **candidate.metrics,
                },
            )
        print(f"\nWrote {len(selected)} APPO episodes to {output_dir}")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", action="append", required=True, metavar="SCENARIO=PATH")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIO_TO_SF_ENV))
    parser.add_argument("--episodes-per-scenario", type=int, default=20)
    parser.add_argument("--keep-top-ratio", type=float, default=0.5)
    parser.add_argument("--deterministic-fraction", type=float, default=0.75)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--max-agent-steps", type=int, default=600,
        help="Maximum policy decisions; smooth capture stores up to 4x frames",
    )
    parser.add_argument("--fps", type=int, default=35)
    parser.add_argument(
        "--capture-frame-skip", type=int, default=1, choices=(1, 4),
        help="Engine ticks per stored frame; 1 is smooth 35 FPS, 4 is legacy 9 FPS",
    )
    parser.add_argument(
        "--humanized", action="store_true",
        help="Apply an action governor; this does not require human recordings",
    )
    parser.add_argument("--humanized-turn-scale", type=float, default=0.5)
    parser.add_argument(
        "--humanized-keep-speed", action="store_true",
        help="Keep the Battle SPEED button; humanized mode suppresses it by default",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()
    if not 0.0 < args.keep_top_ratio <= 1.0:
        parser.error("--keep-top-ratio must be in (0, 1]")
    if not 0.0 <= args.deterministic_fraction <= 1.0:
        parser.error("--deterministic-fraction must be in [0, 1]")
    if not 0.0 < args.humanized_turn_scale <= 1.0:
        parser.error("--humanized-turn-scale must be in (0, 1]")
    expected_fps = 35 if args.capture_frame_skip == 1 else 9
    if args.fps != expected_fps:
        parser.error(
            f"--capture-frame-skip={args.capture_frame_skip} requires --fps={expected_fps} "
            "to preserve VizDoom wall-clock time"
        )
    collect(args)


if __name__ == "__main__":
    main()
