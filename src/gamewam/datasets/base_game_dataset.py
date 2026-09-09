"""BaseGameDataset — low-level game data loader for GameWAM.

Wraps :class:`MultiLeRobotDataset` to provide the sample-dict interface
expected by :class:`GameVideoDataset`.
"""

import math
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf
from tqdm import tqdm

from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from gamewam.utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5


def _accumulate_route_stats(
    cur_action: torch.Tensor,
    route_label: torch.Tensor,
    route_min: Dict[str, list],
    route_max: Dict[str, list],
    route_mean: Dict[str, list],
    route_var: Dict[str, list],
    route_raw: Dict[str, list],
    route_sum: Dict[str, torch.Tensor],
    route_sumsq: Dict[str, torch.Tensor],
    route_count: Dict[str, torch.Tensor],
    route: str,
    action_key: str = "default",
    route_valid: Optional[torch.Tensor] = None,
) -> None:
    """Append per-route action stats for one episode.

    ``cur_action`` is 2D ``[T, D]`` (synthetic) or 3D ``[T, action_size, D]``
    (production sliding-window). ``route_label`` shares the leading two
    axes and is bool. Tokens go to ``"gui"`` when ``route_label=True``,
    else ``"gameplay"``. 3D inputs are flattened to 2D so each
    ``(i, w)`` pair contributes independently.
    """
    # Flatten 3D → 2D so all window tokens contribute independently.
    if cur_action.ndim == 3:
        if cur_action.shape[0] != route_label.shape[0] or cur_action.shape[1] != route_label.shape[1]:
            logger.warning(
                "_accumulate_route_stats: shape mismatch after "
                "sliding window — action=%s, route_label=%s "
                "(route=%s); expected matching "
                "[T, action_size, ...] shapes.",
                tuple(cur_action.shape), tuple(route_label.shape), route,
            )
            return
        cur_action = cur_action.reshape(-1, cur_action.shape[-1])
        route_label = route_label.reshape(-1, 1)
        if route_valid is not None:
            route_valid = route_valid.reshape(-1, 1)
    if cur_action.ndim != 2 or route_label.ndim != 2 or route_label.shape[-1] != 1:
        logger.warning(
            "_accumulate_route_stats: skipping token with shape "
            "action=%s, route_label=%s (route=%s); expected "
            "action [T, D] and route_label [T, 1].",
            tuple(cur_action.shape), tuple(route_label.shape), route,
        )
        return
    T = int(cur_action.shape[0])
    if int(route_label.shape[0]) != T:
        logger.warning(
            "_accumulate_route_stats: token count mismatch — "
            "action has T=%d, route_label has T=%d (route=%s). "
            "Skipping token.",
            T, int(route_label.shape[0]), route,
        )
        return
    if route_valid is not None:
        if route_valid.ndim != 2 or route_valid.shape != route_label.shape:
            logger.warning(
                "_accumulate_route_stats: route_valid shape mismatch — "
                "route_label=%s, route_valid=%s (route=%s). Skipping token.",
                tuple(route_label.shape), tuple(route_valid.shape), route,
            )
            return
    mask = route_label[:, 0].bool()
    valid_mask = torch.ones_like(mask) if route_valid is None else route_valid[:, 0].bool()
    is_gui_route = (route == "gui")
    selected = (mask if is_gui_route else ~mask) & valid_mask
    if not bool(selected.any()):
        return
    selected_action = cur_action[selected]
    route_min[action_key].append(selected_action.amin(0))
    route_max[action_key].append(selected_action.amax(0))
    route_mean[action_key].append(selected_action.mean(0))
    route_var[action_key].append(selected_action.var(0, unbiased=False))
    route_raw[action_key].append(selected_action.clone().detach())
    flat = selected_action.reshape(-1, selected_action.shape[-1]).double()
    cur_sum = flat.sum(dim=0)
    cur_sumsq = (flat * flat).sum(dim=0)
    cur_count = torch.tensor([flat.shape[0]], dtype=torch.float64)
    if action_key not in route_sum:
        route_sum[action_key] = cur_sum
        route_sumsq[action_key] = cur_sumsq
        route_count[action_key] = cur_count
    else:
        route_sum[action_key] += cur_sum
        route_sumsq[action_key] += cur_sumsq
        route_count[action_key] += cur_count


def build_delta_timestamps(
    image_meta: List[Dict[str, Any]],
    state_meta: List[Dict[str, Any]],
    action_meta: List[Dict[str, Any]],
    *,
    obs_size: int,
    past_obs_size: int,
    action_size: int,
    past_action_size: int,
    frame_stride: int,
    fps: float,
    video_only: bool,
) -> Dict[str, List[float]]:
    """Build the per-key ``delta_timestamps`` dict LeRobot uses to sample
    ``(state, action, image)`` for a single clip.

    Resolution matters under ``frame_stride > 1``:
      * image — obs resolution: ``obs_size`` rows spaced ``frame_stride`` apart.
      * state — action resolution: ``action_size + 1`` rows every frame. The
        trailing state row is what the "state at the moment before
        action[t]" convention requires for ``t = action_size``. The
        per-action route mask and the proprio encoder both need
        action-aligned state.
      * action — action resolution: ``action_size`` rows every frame.
    """
    delta_timestamps: Dict[str, List[float]] = {}
    for meta in image_meta:
        key = meta["key"]
        meta["lerobot_key"] = (
            f"observation.images.{key}" if key != "default" else "observation.images"
        )
        delta_timestamps[meta["lerobot_key"]] = [
            (t * frame_stride) / fps
            for t in range(-past_obs_size, -past_obs_size + obs_size)
        ]

    if not video_only:
        for meta in state_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"observation.state.{key}" if key != "default" else "observation.state"
            )
            delta_timestamps[meta["lerobot_key"]] = [
                t / fps
                for t in range(-past_action_size, -past_action_size + action_size + 1)
            ]

        for meta in action_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"action.{key}" if key != "default" else "action"
            )
            delta_timestamps[meta["lerobot_key"]] = [
                t / fps
                for t in range(-past_action_size, -past_action_size + action_size)
            ]
    return delta_timestamps


class BaseGameDataset(torch.utils.data.Dataset):
    """Low-level game dataset backed by LeRobot parquet+mp4 format.

    This is the GameWAM counterpart of GameWAM's ``BaseLerobotDataset``.
    It is consumed by :class:`GameVideoDataset` rather than used directly.
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0,  # Excludes the current frame
        obs_size: int = 1,
        past_obs_size: int = 0,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        frame_stride: int = 1,
        sample_interval: int = 4,
        event_anchor_enabled: bool = False,
        event_anchor_radius: Optional[int] = None,
        sample_interval_event: Optional[int] = None,
        sample_interval_normal: Optional[int] = None,
        fine_task_resample_enabled: bool = False,
        fine_task_resample_mix_original: float = 0.5,
        fine_task_resample_gamma: float = 0.5,
        fine_task_resample_max_expected_repeat: float = 5.0,
        seed: int = 42,
        video_only: bool = False,
        minecraft_cursor_reference_size: Optional[Sequence[float]] = None,
        state_schema: Optional[str] = None,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size >= obs_size - 1, "action_size must cover at least the obs transitions"

        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.is_training_set = is_training_set
        self.frame_stride = frame_stride
        self.sample_interval = sample_interval
        self.val_set_proportion = val_set_proportion
        self.video_only = video_only
        self.return_images = True
        self.processor = None

        self.event_anchor_enabled = event_anchor_enabled
        self.event_anchor_radius = event_anchor_radius
        self.fine_task_resample_enabled = bool(fine_task_resample_enabled)
        if self.fine_task_resample_enabled:
            for name, value in (
                ("fine_task_resample_mix_original", fine_task_resample_mix_original),
                ("fine_task_resample_gamma", fine_task_resample_gamma),
                ("fine_task_resample_max_expected_repeat", fine_task_resample_max_expected_repeat),
            ):
                if value is None:
                    raise ValueError(
                        f"`{name}` must be set when `fine_task_resample_enabled=True`."
                    )
            self.fine_task_resample_mix_original = float(fine_task_resample_mix_original)
            self.fine_task_resample_gamma = float(fine_task_resample_gamma)
            self.fine_task_resample_max_expected_repeat = float(fine_task_resample_max_expected_repeat)
            if not 0.0 <= self.fine_task_resample_mix_original <= 1.0:
                raise ValueError("fine_task_resample_mix_original must be in [0, 1].")
            if self.fine_task_resample_gamma < 0:
                raise ValueError("fine_task_resample_gamma must be >= 0.")
            if self.fine_task_resample_max_expected_repeat < 1:
                raise ValueError("fine_task_resample_max_expected_repeat must be >= 1.")
        else:
            self.fine_task_resample_mix_original = None
            self.fine_task_resample_gamma = None
            self.fine_task_resample_max_expected_repeat = None
        self.sample_interval_event = (
            sample_interval_event if sample_interval_event is not None else sample_interval
        )
        self.sample_interval_normal = (
            sample_interval_normal if sample_interval_normal is not None else sample_interval
        )

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        # Build LeRobotDatasetMetadata for each dir
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            meta = LeRobotDatasetMetadata(repo_id=ds_dir, root=ds_root)
            metas.append(meta)
        self.dataset_metas = tuple(metas)
        if minecraft_cursor_reference_size is not None:
            from .schemas.minecraft import (
                validate_minecraft_state_schema,
            )

            for meta in metas:
                validate_minecraft_state_schema(
                    features=meta.features,
                    info=meta.info,
                    cursor_reference_size=minecraft_cursor_reference_size,
                    dataset_name=str(meta.root),
                )
        if state_schema is not None:
            if state_schema != "vizdoom":
                raise ValueError(f"Unknown state_schema {state_schema!r}.")
            from .schemas.vizdoom import validate_vizdoom_state_schema

            for meta in metas:
                validate_vizdoom_state_schema(
                    meta.features,
                    dataset_name=str(meta.root),
                )

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]
        self.has_state_field_valid = (
            not video_only
            and all("observation.state_field_valid" in m.features for m in metas)
        )

        # Build delta_timestamps for LeRobot indexing
        delta_timestamps = build_delta_timestamps(
            image_meta=self.image_meta,
            state_meta=self.state_meta,
            action_meta=self.action_meta,
            obs_size=obs_size,
            past_obs_size=past_obs_size,
            action_size=action_size,
            past_action_size=past_action_size,
            frame_stride=frame_stride,
            fps=fps,
            video_only=video_only,
        )
        if self.has_state_field_valid:
            delta_timestamps["observation.state_field_valid"] = [
                t / fps
                for t in range(-past_action_size, -past_action_size + action_size + 1)
            ]

        # Train/val split by episode
        episodes = {}
        if val_set_proportion < 1e-6:
            for m in metas:
                episodes[m.repo_id] = list(range(m.total_episodes))
        else:
            for m in metas:
                split_idx = int(m.total_episodes * (1 - val_set_proportion))
                episode_indices = list(range(m.total_episodes))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if is_training_set:
                    episodes[m.repo_id] = episode_indices[:split_idx]
                else:
                    episodes[m.repo_id] = episode_indices[split_idx:]

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )
        self._dataset_frame_ends = np.cumsum(
            [int(dataset.num_frames) for dataset in self.multi_dataset._datasets],
            dtype=np.int64,
        )
        self._episode_from_by_dataset = [
            dataset.episode_data_index["from"].numpy()
            for dataset in self.multi_dataset._datasets
        ]
        self._episode_to_by_dataset = [
            dataset.episode_data_index["to"].numpy()
            for dataset in self.multi_dataset._datasets
        ]

        # Build episode_data_index for stats computation
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index += dataset.num_frames

        self.episode_data_index = {
            "from": torch.cat([d["from"] for d in episode_data_index]),
            "to": torch.cat([d["to"] for d in episode_data_index]),
        }

        logger.info(
            f"BaseGameDataset: {self.multi_dataset.num_episodes} episodes, "
            f"{self.multi_dataset.num_frames} frames ({'train' if is_training_set else 'val'})"
        )

        self.sample_fine_tasks: List[str] = []
        self.sample_starts = np.asarray(
            self._build_sample_index(), dtype=np.int64
        )
        self._fine_task_sampler_weights: Optional[torch.Tensor] = None
        if self.fine_task_resample_enabled:
            self._fine_task_sampler_weights = self._build_fine_task_sampler_weights()
        if self.event_anchor_enabled:
            logger.info(
                f"BaseGameDataset: event-anchored sampling on, "
                f"radius={self.event_anchor_radius}, "
                f"stride_event={self.sample_interval_event}, "
                f"stride_normal={self.sample_interval_normal}, "
                f"total samples={len(self.sample_starts)}."
            )

    # Sample construction helpers

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key]
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, \
            f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        assert state.shape[-1] == raw_shape, \
            f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = (image * 255).to(torch.uint8)
        return image

    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample

    # Episode-level data access (for stats computation)

    def _get_episode_data(self, episode_idx: int) -> dict:
        """Load all actions and states from a single episode via parquet."""
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        valid_action_starts: Optional[int] = None
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            # A training sample starting at row ``i`` needs ``action_size``
            # actions *and* the trailing observation at ``i + action_size``.
            # Therefore only ``N - action_size`` starts are reachable by
            # ``_build_sample_index``.  Keeping all N replicated windows here
            # heavily overweights the final (untrainable) action in stats.
            cur_valid_starts = max(int(a.shape[0]) - int(self.action_size), 0)
            if valid_action_starts is None:
                valid_action_starts = cur_valid_starts
            elif valid_action_starts != cur_valid_starts:
                raise ValueError(
                    "Episode action keys have inconsistent row counts: "
                    f"expected {valid_action_starts} valid starts, got "
                    f"{cur_valid_starts} for key {meta['key']!r}."
                )
            a = _sliding_window_with_replication(a, self.action_size)[:cur_valid_starts]
            action[meta["key"]] = a.float()
        # State and its per-field validity are frame-level episode data, so
        # both remain at the full episode length N. Only the action-window axis
        # is cropped to the N-action_size trainable starts above.
        state_dim_is_pad = lerobot_sample.get(f"{self.state_meta[0]['lerobot_key']}_dim_is_pad")
        state_field_valid = lerobot_sample.get("observation.state_field_valid")
        if state_field_valid is not None:
            state_rows = {int(value.shape[0]) for value in state.values()}
            if len(state_rows) != 1:
                raise ValueError(
                    "Episode state keys have inconsistent row counts: "
                    f"{sorted(state_rows)}."
                )
            expected_rows = next(iter(state_rows))
            if int(state_field_valid.shape[0]) != expected_rows:
                raise ValueError(
                    "Episode state/state_field_valid row mismatch: "
                    f"state={expected_rows}, "
                    f"field_valid={state_field_valid.shape[0]}, "
                    f"episode_idx={episode_idx}."
                )
            state_field_valid = state_field_valid.unsqueeze(1)

        return {
            "action": action,
            "state": state,
            "state_dim_is_pad": state_dim_is_pad,
            "state_field_valid": state_field_valid,
            "stats_valid_action_starts": int(valid_action_starts or 0),
        }

    # Dataset statistics (normalization)

    def get_dataset_stats(self, preprocessor) -> dict:
        """Compute global dataset statistics for normalization.

        When ``route_aware_action_normalization=True`` (GUI route head
        enabled) this also writes per-route action stats under
        ``stats["action_by_route"]`` with ``gameplay`` and ``gui`` keys.
        The route label is read from the *raw* (pre-transform) episode
        state. Action and state share the same ``N`` parquet rows: the
        full route window is built first, then cropped to the same
        trainable starts as action before transformed tokens are split
        into route buckets.
        """
        state_raw = defaultdict(list)
        state_min = defaultdict(list)
        state_max = defaultdict(list)
        state_mean = defaultdict(list)
        state_var = defaultdict(list)
        state_global_sum: Dict[str, torch.Tensor] = {}
        state_global_sumsq: Dict[str, torch.Tensor] = {}
        state_global_count: Dict[str, torch.Tensor] = {}

        action_raw = defaultdict(list)
        action_min = defaultdict(list)
        action_max = defaultdict(list)
        action_mean = defaultdict(list)
        action_var = defaultdict(list)
        action_global_sum: Dict[str, torch.Tensor] = {}
        action_global_sumsq: Dict[str, torch.Tensor] = {}
        action_global_count: Dict[str, torch.Tensor] = {}

        # Per-route accumulators (only used when route_aware_action_normalization).
        route_aware = bool(
            getattr(preprocessor, "route_aware_action_normalization", False)
        )
        route_state_key = getattr(preprocessor, "gui_label_state_key", "default")
        route_dim_index = getattr(preprocessor, "gui_label_dim_index", None)
        use_stepwise_action_norm = bool(
            getattr(preprocessor, "use_stepwise_action_norm", False)
        )
        if route_aware and use_stepwise_action_norm:
            raise NotImplementedError(
                "Route-aware action normalization with stepwise norm is not "
                "supported yet. Set use_stepwise_action_norm=false or "
                "route_aware_action_normalization=false."
            )
        if route_aware and (route_dim_index is None or route_state_key is None):
            raise ValueError(
                "`route_aware_action_normalization=True` requires "
                "`gui_label_dim_index` and `gui_label_state_key` to be set "
                "on the preprocessor."
            )
        if route_aware and len(self.action_meta) != 1:
            raise ValueError(
                "Route-aware action normalization requires exactly one combined "
                f"action key, got {len(self.action_meta)}."
            )
        route_action_key = self.action_meta[0]["key"] if route_aware else None
        route_keys = ("gameplay", "gui")
        action_route_min: Dict[str, Dict[str, list]] = {k: defaultdict(list) for k in route_keys}
        action_route_max: Dict[str, Dict[str, list]] = {k: defaultdict(list) for k in route_keys}
        action_route_mean: Dict[str, Dict[str, list]] = {k: defaultdict(list) for k in route_keys}
        action_route_var: Dict[str, Dict[str, list]] = {k: defaultdict(list) for k in route_keys}
        action_route_raw: Dict[str, Dict[str, list]] = {k: defaultdict(list) for k in route_keys}
        action_route_sum: Dict[str, Dict[str, torch.Tensor]] = {
            k: {} for k in route_keys
        }
        action_route_sumsq: Dict[str, Dict[str, torch.Tensor]] = {
            k: {} for k in route_keys
        }
        action_route_count: Dict[str, Dict[str, torch.Tensor]] = {
            k: {} for k in route_keys
        }

        episodes_num = self.multi_dataset.num_episodes

        dist_state = PartialState()
        rank = dist_state.process_index
        world_size = dist_state.num_processes
        my_episode_idxs = list(range(rank, episodes_num, world_size))

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            action_values = list((batch.get("action") or {}).values())
            if not action_values:
                raise ValueError(f"Episode {episode_idx} contains no action tensors.")
            action_starts = int(action_values[0].shape[0])
            if any(int(value.shape[0]) != action_starts for value in action_values[1:]):
                raise ValueError(
                    f"Episode {episode_idx} has inconsistent action start counts."
                )
            # Episodes no longer than action_size cannot produce a training
            # sample (the trailing observation is also required).  Excluding
            # them here keeps normalization support identical to training.
            if action_starts <= 0:
                return episode_idx, None, None
            # Snapshot raw (pre-transform) route label so per-route
            # action stats align with the action token at index [0:T].
            route_label = None
            route_valid = None
            if route_aware:
                state_dict = batch.get("state") or {}
                raw_state = state_dict.get(route_state_key)
                if raw_state is None:
                    raise KeyError(
                        f"Route state key {route_state_key!r} is missing from "
                        f"episode {episode_idx}; available keys={list(state_dict)}."
                    )
                full_route_label = _build_episode_route_window(
                    raw_state,
                    route_dim_index=int(route_dim_index),
                    window_size=self.action_size,
                    expected_starts=int(raw_state.shape[0]),
                )
                route_label = full_route_label[:action_starts]
                state_field_valid = batch.get("state_field_valid")
                if state_field_valid is not None:
                    full_route_valid = _build_episode_route_window(
                        state_field_valid,
                        route_dim_index=int(route_dim_index),
                        window_size=self.action_size,
                        expected_starts=int(raw_state.shape[0]),
                    )
                    route_valid = full_route_valid[:action_starts]
            batch = preprocessor.action_state_transform(batch)
            return episode_idx, batch, (route_label, route_valid)

        def accumulate_global_stats(
            sum_dict: Dict[str, torch.Tensor],
            sumsq_dict: Dict[str, torch.Tensor],
            count_dict: Dict[str, torch.Tensor],
            key: str,
            tensor: torch.Tensor,
        ):
            flat = tensor.reshape(-1, tensor.shape[-1]).double()
            cur_sum = flat.sum(dim=0)
            cur_sumsq = (flat * flat).sum(dim=0)
            cur_count = torch.tensor([flat.shape[0]], dtype=torch.float64)
            if key not in sum_dict:
                sum_dict[key] = cur_sum
                sumsq_dict[key] = cur_sumsq
                count_dict[key] = cur_count
            else:
                sum_dict[key] += cur_sum
                sumsq_dict[key] += cur_sumsq
                count_dict[key] += cur_count

        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(
                my_episode_idxs,
                desc=f"Iterating dataset (rank {rank}/{world_size}) for normalization",
            ):
                episode_idx, batch, route_pair = process_episode(episode_idx)
                if batch is None:
                    continue
                route_label, route_valid = route_pair if route_pair is not None else (None, None)
                state_dim_is_pad = batch.get("state_dim_is_pad")
                for meta in self.state_meta:
                    key = meta["key"]
                    if state_dim_is_pad is not None and state_dim_is_pad.all():
                        continue
                    cur_state: torch.Tensor = batch["state"][key]
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0, unbiased=False))
                    state_raw[key].append(cur_state.clone().detach())
                    accumulate_global_stats(
                        state_global_sum, state_global_sumsq, state_global_count, key, cur_state
                    )
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key]
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0, unbiased=False))
                    action_raw[key].append(cur_action.clone().detach())
                    accumulate_global_stats(
                        action_global_sum, action_global_sumsq, action_global_count, key, cur_action
                    )
                    if route_aware and route_label is not None:
                        _accumulate_route_stats(
                            cur_action, route_label,
                            action_route_min["gameplay"], action_route_max["gameplay"],
                            action_route_mean["gameplay"], action_route_var["gameplay"],
                            action_route_raw["gameplay"],
                            action_route_sum["gameplay"], action_route_sumsq["gameplay"],
                            action_route_count["gameplay"],
                            route="gameplay",
                            action_key=key,
                            route_valid=route_valid,
                        )
                        _accumulate_route_stats(
                            cur_action, route_label,
                            action_route_min["gui"], action_route_max["gui"],
                            action_route_mean["gui"], action_route_var["gui"],
                            action_route_raw["gui"],
                            action_route_sum["gui"], action_route_sumsq["gui"],
                            action_route_count["gui"],
                            route="gui",
                            action_key=key,
                            route_valid=route_valid,
                        )

        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in my_episode_idxs]

                for future in tqdm(
                    as_completed(futures),
                    total=len(my_episode_idxs),
                    desc=f"Iterating dataset (rank {rank}/{world_size}) for normalization",
                ):
                    try:
                        episode_idx, batch, route_pair = future.result()
                        if batch is None:
                            continue
                        route_label, route_valid = route_pair if route_pair is not None else (None, None)
                        state_dim_is_pad = batch.get("state_dim_is_pad")
                        for meta in self.state_meta:
                            key = meta["key"]
                            if state_dim_is_pad is not None and state_dim_is_pad.all():
                                continue
                            cur_state: torch.Tensor = batch["state"][key]
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0, unbiased=False))
                            state_raw[key].append(cur_state.clone().detach())
                            accumulate_global_stats(
                                state_global_sum, state_global_sumsq, state_global_count, key, cur_state
                            )
                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key]
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0, unbiased=False))
                            action_raw[key].append(cur_action.clone().detach())
                            accumulate_global_stats(
                                action_global_sum, action_global_sumsq, action_global_count, key, cur_action
                            )
                            if route_aware and route_label is not None:
                                _accumulate_route_stats(
                                    cur_action, route_label,
                                    action_route_min["gameplay"], action_route_max["gameplay"],
                                    action_route_mean["gameplay"], action_route_var["gameplay"],
                                    action_route_raw["gameplay"],
                                    action_route_sum["gameplay"], action_route_sumsq["gameplay"],
                                    action_route_count["gameplay"],
                                    route="gameplay",
                                    action_key=key,
                                    route_valid=route_valid,
                                )
                                _accumulate_route_stats(
                                    cur_action, route_label,
                                    action_route_min["gui"], action_route_max["gui"],
                                    action_route_mean["gui"], action_route_var["gui"],
                                    action_route_raw["gui"],
                                    action_route_sum["gui"], action_route_sumsq["gui"],
                                    action_route_count["gui"],
                                    route="gui",
                                    action_key=key,
                                    route_valid=route_valid,
                                )

                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        print(traceback.format_exc())
                        raise e

        norm_default_mode = getattr(preprocessor, "norm_default_mode", "q01/q99")
        norm_exception_mode = getattr(preprocessor, "norm_exception_mode", None)
        state_quantile_dims = {
            meta["key"]: _quantile_dims_for_meta("state", meta, norm_default_mode, norm_exception_mode)
            for meta in self.state_meta
        }
        action_quantile_dims = {
            meta["key"]: _quantile_dims_for_meta("action", meta, norm_default_mode, norm_exception_mode)
            for meta in self.action_meta
        }

        def _compute_quantiles(raw_dict, quantile_dims_by_key, min_dict, max_dict):
            result_q01 = {}
            result_q99 = {}
            metas = self.state_meta if raw_dict is state_raw else self.action_meta
            for meta in metas:
                key = meta["key"]
                if not min_dict[key] or not max_dict[key]:
                    raise ValueError(
                        f"No trainable samples are available for normalization key {key!r}."
                    )
                fill_min = torch.stack(min_dict[key]).amin(0)
                fill_max = torch.stack(max_dict[key]).amax(0)
                result_q01[key] = _distributed_quantile_dim0_exact_tail(
                    raw_dict[key],
                    0.01,
                    quantile_dims=quantile_dims_by_key.get(key),
                    fill=fill_min,
                )
                result_q99[key] = _distributed_quantile_dim0_exact_tail(
                    raw_dict[key],
                    0.99,
                    quantile_dims=quantile_dims_by_key.get(key),
                    fill=fill_max,
                )
            return result_q01, result_q99

        if world_size > 1:
            state_dicts = (state_min, state_max, state_mean, state_var)
            action_dicts = (action_min, action_max, action_mean, action_var)
            for meta in self.state_meta:
                k = meta["key"]
                for d in state_dicts:
                    d[k] = _all_gather_tensor_list(d[k])
                zero_sum = torch.zeros(_meta_dim_count(meta), dtype=torch.float64)
                zero_count = torch.zeros(1, dtype=torch.float64)
                state_global_sum[k] = torch.stack(
                    _all_gather_tensor_list([state_global_sum.get(k, zero_sum)])
                ).sum(0)
                state_global_sumsq[k] = torch.stack(
                    _all_gather_tensor_list([state_global_sumsq.get(k, zero_sum)])
                ).sum(0)
                state_global_count[k] = torch.stack(
                    _all_gather_tensor_list([state_global_count.get(k, zero_count)])
                ).sum(0)
            for meta in self.action_meta:
                k = meta["key"]
                for d in action_dicts:
                    d[k] = _all_gather_tensor_list(d[k])
                zero_sum = torch.zeros(_meta_dim_count(meta), dtype=torch.float64)
                zero_count = torch.zeros(1, dtype=torch.float64)
                action_global_sum[k] = torch.stack(
                    _all_gather_tensor_list([action_global_sum.get(k, zero_sum)])
                ).sum(0)
                action_global_sumsq[k] = torch.stack(
                    _all_gather_tensor_list([action_global_sumsq.get(k, zero_sum)])
                ).sum(0)
                action_global_count[k] = torch.stack(
                    _all_gather_tensor_list([action_global_count.get(k, zero_count)])
                ).sum(0)

            # Per-route cross-rank aggregation. The gather must be
            # unconditional — a sparse rank contributes an empty list
            # so the union on every rank reflects *all* ranks' data.
            if route_aware:
                _gather_route_stats_across_ranks(
                    action_route_min, action_route_max,
                    action_route_mean, action_route_var,
                    action_route_raw, action_route_sum,
                    action_route_sumsq, action_route_count,
                    route_keys, action_dim=int(self.action_meta[0]["raw_shape"]),
                    action_key=route_action_key,
                )

        # Quantiles must be computed from the global row population. Averaging
        # rank-local quantiles is not a quantile and is especially wrong for
        # uneven or sparse rank shards. The helper exchanges only the exact
        # order-statistic tails needed for q01/q99.
        state_stepwise_q01, state_stepwise_q99 = _compute_quantiles(
            state_raw, state_quantile_dims, state_min, state_max
        )
        action_stepwise_q01, action_stepwise_q99 = _compute_quantiles(
            action_raw, action_quantile_dims, action_min, action_max
        )

        stats: Dict[str, Any] = {
            "state": defaultdict(dict),
            "action": defaultdict(dict),
            "num_episodes": episodes_num,
            "num_transition": self.multi_dataset.num_frames,
        }
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = state_stepwise_q01[key]
            stats["state"][key]["stepwise_q99"] = state_stepwise_q99[key]
            stats["state"][key]["global_q01"] = state_stepwise_q01[key].amin(0)
            stats["state"][key]["global_q99"] = state_stepwise_q99[key].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
            ) = get_stepwise_mean_std(state_mean[key], state_var[key])
            (
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_global_mean_std(
                state_global_sum,
                state_global_sumsq,
                state_global_count,
                key,
                stats["state"][key]["stepwise_mean"].dtype,
            )

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = action_stepwise_q01[key]
            stats["action"][key]["stepwise_q99"] = action_stepwise_q99[key]
            stats["action"][key]["global_q01"] = action_stepwise_q01[key].amin(0)
            stats["action"][key]["global_q99"] = action_stepwise_q99[key].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
            ) = get_stepwise_mean_std(action_mean[key], action_var[key])
            (
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_global_mean_std(
                action_global_sum,
                action_global_sumsq,
                action_global_count,
                key,
                stats["action"][key]["stepwise_mean"].dtype,
            )

        # Per-route action stats for route_aware_action_normalization.
        # ``stats["action_by_route"][route][action_key]`` is a flat dict
        # of the same shape that ``LinearNormalizer.normalizers["action"][key]``
        # consumes, plus route-level ``"count"`` / ``"empty"`` bookkeeping.
        if route_aware:
            stats["action_by_route"] = {}
            for route in route_keys:
                route_block: Dict[str, Any] = {}
                # Count-based empty check: a presence-only check would
                # mis-classify a globally empty route or a sparse rank's
                # empty local list, falling through to ``torch.stack`` on
                # an empty list.
                count_t = action_route_count[route].get(route_action_key)
                empty_route = (
                    count_t is None
                    or float(count_t.double().sum().item()) <= 0.0
                )
                route_block = _finalize_route_stats(
                    route=route,
                    empty_route=empty_route,
                    action_meta=self.action_meta,
                    action_route_min=action_route_min[route],
                    action_route_max=action_route_max[route],
                    action_route_mean=action_route_mean[route],
                    action_route_var=action_route_var[route],
                    action_route_raw=action_route_raw[route],
                    action_route_sum=action_route_sum[route],
                    action_route_sumsq=action_route_sumsq[route],
                    action_route_count=action_route_count[route],
                )
                stats["action_by_route"][route] = route_block
            # Metadata so the load side can validate the
            # route_aware_action_normalization config.
            stats["route_aware_action_normalization"] = {
                "enabled": True,
                "route_state_key": route_state_key,
                "gui_label_dim_index": int(route_dim_index),
                "continuous_dims": [
                    int(i)
                    for i, is_continuous in enumerate(
                        getattr(preprocessor, "dim_is_continuous", []) or []
                    )
                    if bool(is_continuous)
                ],
                "action_dim": sum(
                    int(meta["raw_shape"]) for meta in self.action_meta
                ),
            }

        return stats

    # Processor management

    def set_processor(self, processor):
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def _resolve_sample_episode(self, sample_idx: int):
        """Resolve a global sample index to its underlying dataset/episode metadata."""
        sample_idx = int(sample_idx)
        if not hasattr(self, "_dataset_frame_ends"):
            datasets = self.multi_dataset._datasets
            self._dataset_frame_ends = np.cumsum(
                [int(dataset.num_frames) for dataset in datasets], dtype=np.int64
            )
            self._episode_from_by_dataset = [
                dataset.episode_data_index["from"].numpy() for dataset in datasets
            ]
            self._episode_to_by_dataset = [
                dataset.episode_data_index["to"].numpy() for dataset in datasets
            ]
        dataset_pos = int(np.searchsorted(
            self._dataset_frame_ends, sample_idx, side="right"
        ))
        if dataset_pos >= len(self.multi_dataset._datasets):
            raise IndexError(
                f"sample_idx={sample_idx} out of bounds for episode resolution."
            )
        dataset = self.multi_dataset._datasets[dataset_pos]
        dataset_start = (
            0 if dataset_pos == 0 else int(self._dataset_frame_ends[dataset_pos - 1])
        )
        local_idx = sample_idx - dataset_start
        ep_to = self._episode_to_by_dataset[dataset_pos]
        ep_pos = int(np.searchsorted(ep_to, local_idx, side="right"))
        if ep_pos >= len(ep_to):
            raise IndexError(
                f"sample_idx={sample_idx} maps outside selected episodes "
                f"for dataset {dataset.repo_id}."
            )
        ep_from = int(self._episode_from_by_dataset[dataset_pos][ep_pos])
        frame_idx = local_idx - ep_from
        episode_idx = (
            int(dataset.episodes[ep_pos])
            if dataset.episodes is not None
            else ep_pos
        )
        ep_meta = dataset.meta.episodes.get(episode_idx, {}) or {}
        return dataset, episode_idx, ep_from, frame_idx, ep_meta

    def get_vae_cache_key(self, sample_idx: int, resolved_episode=None) -> str:
        """Return a split-stable physical key for a window start."""
        if resolved_episode is None:
            resolved_episode = self._resolve_sample_episode(sample_idx)
        dataset, episode_idx, _ep_from, frame_idx, _ep_meta = resolved_episode
        return f"{dataset.repo_id}::episode={episode_idx}::frame={frame_idx}"

    # Dataset interface

    def _iter_ep_meta(self):
        """Yield (global_ep_idx, ep_meta_dict) over the selected episodes of
        every underlying LeRobotDataset, in the order MultiLeRobotDataset
        concatenates them. Event metadata comes from the converter-written
        `episodes.jsonl` (see scripts/datasets/minecraft/converters/vpt_event.py).
        """
        global_ep_idx = 0
        for ds in self.multi_dataset._datasets:
            sub_indices = ds.episodes if ds.episodes is not None else list(range(ds.meta.total_episodes))
            for sub_idx in sub_indices:
                ep_meta = ds.meta.episodes.get(sub_idx, {})
                yield global_ep_idx, ep_meta
                global_ep_idx += 1

    @staticmethod
    def _fine_task_from_ep_meta(ep_meta: Dict[str, Any]) -> str:
        tasks = ep_meta.get("tasks", [])
        return str(tasks[-1]) if tasks else ""

    @staticmethod
    def _apply_probability_caps(
        probs: Dict[str, float],
        caps: Dict[str, float],
    ) -> Dict[str, float]:
        final: Dict[str, float] = {}
        remaining = set(probs)
        remaining_mass = 1.0

        while remaining:
            base_mass = sum(probs[k] for k in remaining)
            if base_mass <= 0:
                uniform = remaining_mass / len(remaining)
                for k in remaining:
                    final[k] = min(caps[k], uniform)
                break

            capped_now = []
            scale = remaining_mass / base_mass
            for k in remaining:
                p = probs[k] * scale
                if p > caps[k]:
                    final[k] = caps[k]
                    remaining_mass -= caps[k]
                    capped_now.append(k)

            if not capped_now:
                for k in remaining:
                    final[k] = probs[k] * scale
                break

            for k in capped_now:
                remaining.remove(k)

        norm = sum(final.values())
        if norm <= 0:
            raise ValueError("Fine-task resampling probabilities sum to zero.")
        return {k: v / norm for k, v in final.items()}

    def _build_fine_task_sampler_weights(self) -> torch.Tensor:
        if len(self.sample_fine_tasks) != len(self.sample_starts):
            raise RuntimeError(
                "Fine-task label count does not match sample starts: "
                f"{len(self.sample_fine_tasks)} vs {len(self.sample_starts)}."
            )
        if not self.sample_fine_tasks:
            raise RuntimeError("fine_task_resample_enabled=True but no fine-task labels were built.")

        counts = Counter(self.sample_fine_tasks)
        total = float(len(self.sample_fine_tasks))
        mix_original = self.fine_task_resample_mix_original
        gamma = self.fine_task_resample_gamma

        tail_scores = {task: float(count) ** gamma for task, count in counts.items()}
        tail_norm = sum(tail_scores.values())
        target = {}
        for task, count in counts.items():
            p_data = float(count) / total
            p_tail = tail_scores[task] / tail_norm
            target[task] = mix_original * p_data + (1.0 - mix_original) * p_tail

        caps = {
            task: min(1.0, self.fine_task_resample_max_expected_repeat * float(count) / total)
            for task, count in counts.items()
        }
        target = self._apply_probability_caps(target, caps)
        per_sample_weight = {task: target[task] / float(count) for task, count in counts.items()}
        weights = torch.tensor(
            [per_sample_weight[task] for task in self.sample_fine_tasks],
            dtype=torch.float32,
        )
        weights /= weights.sum()

        raw_max = max(counts.values())
        raw_min = min(counts.values())
        effective_counts = {task: target[task] * total for task in counts}
        logger.info(
            "Fine-task resampling enabled: %d samples, %d fine tasks, "
            "mix_original=%.3f gamma=%.3f max_expected_repeat=%.2f, "
            "raw_count[min/median/max]=%d/%.1f/%d, effective_count[min/median/max]=%.1f/%.1f/%.1f.",
            len(self.sample_fine_tasks),
            len(counts),
            mix_original,
            gamma,
            self.fine_task_resample_max_expected_repeat,
            raw_min,
            float(np.median(list(counts.values()))),
            raw_max,
            min(effective_counts.values()),
            float(np.median(list(effective_counts.values()))),
            max(effective_counts.values()),
        )
        return weights

    def get_sampler_weights(self) -> Optional[torch.Tensor]:
        return self._fine_task_sampler_weights

    def get_task_sampling_spec(self):
        """Return FineTaskSamplingSpec for the task-quota-cyclic sampler."""
        if not self.fine_task_resample_enabled:
            return None
        from gamewam.utils.samplers import FineTaskSamplingSpec

        counts = Counter(self.sample_fine_tasks)
        task_names = sorted(counts.keys())
        name_to_id = {name: i for i, name in enumerate(task_names)}

        task_ids_per_sample = torch.tensor(
            [name_to_id[t] for t in self.sample_fine_tasks], dtype=torch.long,
        )
        task_counts = torch.tensor([counts[n] for n in task_names], dtype=torch.long)

        indices_by_task: List[torch.Tensor] = []
        for name in task_names:
            task_id = name_to_id[name]
            local_indices = torch.where(task_ids_per_sample == task_id)[0]
            indices_by_task.append(local_indices)

        return FineTaskSamplingSpec(
            task_names=tuple(task_names),
            indices_by_task=tuple(indices_by_task),
            task_counts=task_counts,
            gamma=self.fine_task_resample_gamma,
            mix=self.fine_task_resample_mix_original,
            max_expected_repeat=self.fine_task_resample_max_expected_repeat,
        )

    def _build_sample_index(self) -> List[int]:
        """Materialize per-sample global frame-start indices for __getitem__.

        When event anchoring is disabled, this reproduces the original stride-
        based subsampling (`sample_idx = idx * sample_interval`). When enabled,
        each clip contributes:
          - dense starts (stride = sample_interval_event) whose obs window
            overlaps [event - radius, event + radius];
          - sparse starts (stride = sample_interval_normal) outside that band.
        Duplicate starts within the same clip are deduplicated.
        """
        self.sample_fine_tasks = []
        obs_span = (self.obs_size - 1) * self.frame_stride + 1
        from_ = self.episode_data_index["from"].tolist()
        to_ = self.episode_data_index["to"].tolist()

        if not self.event_anchor_enabled:
            step = max(1, int(self.sample_interval))
            starts: List[int] = []
            for global_ep_idx, ep_meta in self._iter_ep_meta():
                ep_from = from_[global_ep_idx]
                ep_to = to_[global_ep_idx]
                clip_length = ep_to - ep_from
                if clip_length < obs_span:
                    continue
                max_start = clip_length - obs_span
                ep_starts = [ep_from + s for s in range(0, max_start + 1, step)]
                starts.extend(ep_starts)
                if self.fine_task_resample_enabled:
                    fine_task = self._fine_task_from_ep_meta(ep_meta)
                    self.sample_fine_tasks.extend([fine_task] * len(ep_starts))
            return starts

        if self.event_anchor_radius is None:
            raise ValueError(
                "event_anchor_enabled=True requires event_anchor_radius to be set."
            )
        radius = int(self.event_anchor_radius)
        stride_event = max(1, int(self.sample_interval_event))
        stride_normal = max(1, int(self.sample_interval_normal))

        starts: List[int] = []
        no_event_meta_count = 0
        for global_ep_idx, ep_meta in self._iter_ep_meta():
            ep_from = from_[global_ep_idx]
            ep_to = to_[global_ep_idx]
            clip_length = ep_to - ep_from
            if clip_length < obs_span:
                continue

            max_start = clip_length - obs_span  # inclusive upper bound (clip-local)

            # Dense sampling is anchored on the rightmost same-key
            # event in the sub-clip (the one that passed Gate A/B).
            # The list `pre_anchor_event_frame_idxs_in_clip` covers
            # EARLIER same-key events — those are for label
            # generation, not for sampling density, since each
            # would be its own anchor in its own sub-clip.
            event_anchor = ep_meta.get("event_anchor_frame_idx_in_clip")
            if event_anchor is None:
                # Fallback: uniform stride for sub-clips lacking
                # event-anchor metadata.
                no_event_meta_count += 1
                ep_starts = [ep_from + s for s in range(0, max_start + 1, stride_normal)]
            else:
                event_anchor = int(event_anchor)
                # Dense band: sample starts whose obs window
                # overlaps the anchor's [anchor - radius,
                # anchor + radius] interval.
                event_lo = max(0, event_anchor - radius - obs_span + 1)
                event_hi = min(max_start, event_anchor + radius)
                event_region = set()
                if event_hi >= event_lo:
                    for s in range(event_lo, event_hi + 1, stride_event):
                        event_region.add(s)

                # Normal starts: uniformly spaced over the full clip,
                # but drop any that fall inside the dense band.
                clip_starts = set()
                for s in range(0, max_start + 1, stride_normal):
                    if event_lo <= s <= event_hi:
                        continue
                    clip_starts.add(s)

                merged = sorted(event_region | clip_starts)
                ep_starts = [ep_from + s for s in merged]

            starts.extend(ep_starts)
            if self.fine_task_resample_enabled:
                fine_task = self._fine_task_from_ep_meta(ep_meta)
                self.sample_fine_tasks.extend([fine_task] * len(ep_starts))

        if no_event_meta_count:
            logger.warning(
                f"BaseGameDataset: {no_event_meta_count} episode(s) lacked "
                f"`event_anchor_frame_idx_in_clip`; they fall back to uniform stride."
            )
        return starts

    def __len__(self):
        return len(self.sample_starts)

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        sample_idx = int(self.sample_starts[idx])
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    f"Error loading sample {sample_idx} (attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                sample_idx = int(self.sample_starts[np.random.randint(len(self))])
                print(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {idx}."
            ) from last_exception

        resolved_episode = self._resolve_sample_episode(sample_idx)

        # Build sample dict matching the format expected by GameWAMProcessor
        sample = {
            "idx": sample_idx,
            "vae_cache_key": self.get_vae_cache_key(sample_idx, resolved_episode),
            "images": {},
            "image_is_pad": lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"],
            "video_only": self.video_only,
            "skip_images": not self.return_images,
        }
        try:
            _ds, _ep_idx, _ep_from, _frame_idx, _ep_meta = resolved_episode
            sample["clip_frame_idx"] = int(_frame_idx)
            evt_idxs = _ep_meta.get("pre_anchor_event_frame_idxs_in_clip")
            if evt_idxs is not None:
                sample["pre_anchor_event_frame_idxs_in_clip"] = [int(x) for x in evt_idxs]
            evt_times = _ep_meta.get("pre_anchor_event_times")
            if evt_times is not None:
                sample["pre_anchor_event_times"] = [int(t) for t in evt_times]
            evt_anchor = _ep_meta.get("event_anchor_frame_idx_in_clip")
            if evt_anchor is not None:
                sample["event_anchor_frame_idx_in_clip"] = int(evt_anchor)
            anchor_time = _ep_meta.get("event_anchor_time")
            if anchor_time is not None:
                sample["event_anchor_time"] = int(anchor_time)
            event_key = _ep_meta.get("event_key")
            if event_key is not None:
                sample["event_key"] = event_key
        except IndexError:
            pass
        if "task" in lerobot_sample:
            sample["task"] = lerobot_sample["task"]
        if "coarse_task" in lerobot_sample:
            sample["coarse_task"] = lerobot_sample["coarse_task"]

        if self.video_only:

            if self.return_images:
                for meta in self.image_meta:
                    sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
        else:
            sample["task"] = lerobot_sample["task"]
            sample["action"] = {}
            sample["state"] = {}
            for meta in self.state_meta:
                sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
            for meta in self.action_meta:
                sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)
            if self.return_images:
                for meta in self.image_meta:
                    sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
            sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
            sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
            sample["state_dim_is_pad"] = lerobot_sample.get(f"{self.state_meta[0]['lerobot_key']}_dim_is_pad")
            sample["state_field_valid"] = lerobot_sample.get("observation.state_field_valid")

        sample = self._get_additional_data(sample, lerobot_sample)

        if not self.video_only:
            for key in lerobot_sample:
                if key not in sample and "observation" not in key and "action" not in key:
                    sample[key] = lerobot_sample[key]

        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        return sample


def get_stepwise_mean_std(means: List[torch.Tensor], vars: List[torch.Tensor]):
    """Stepwise (per-episode) mean/std aggregator.

    Aggregates per-episode means/vars into a stepwise mean and std;
    used by ``get_dataset_stats`` and ``_finalize_route_stats``.
    """
    means = torch.stack(means)
    vars = torch.stack(vars)
    stepwise_mean = means.mean(0)
    stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
    return stepwise_mean, stepwise_std


def get_global_mean_std(
    sum_dict: Dict[str, torch.Tensor],
    sumsq_dict: Dict[str, torch.Tensor],
    count_dict: Dict[str, torch.Tensor],
    key: str,
    dtype: torch.dtype,
):
    """Global mean/std from per-rank sum / sumsq / count.

    Used by ``get_dataset_stats`` for the global statistics.
    """
    count = count_dict[key].double().sum().clamp_min(1.0)
    total = sum_dict[key].double()
    total_squares = sumsq_dict[key].double()
    mean = total / count
    var_num = (total_squares - (total * total) / count).clamp_min(0.0)
    if count.item() > 1:
        std = torch.sqrt(var_num / (count - 1.0))
    else:
        std = torch.zeros_like(mean)
    return mean.to(dtype), std.to(dtype)


def _all_gather_tensor_list(local_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """Gather per-rank tensor lists onto every rank via all_gather_object.

    Returns local_list unchanged when distributed is not initialized.
    Returned order is rank-major; downstream stats (mean/amin/etc.) are
    order-invariant so this is safe.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_list

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return local_list

    cpu_local = [t.detach().cpu() for t in local_list]
    gathered: List[Optional[List[torch.Tensor]]] = [None] * world_size
    torch.distributed.all_gather_object(gathered, cpu_local)

    full: List[torch.Tensor] = []
    for chunk in gathered:
        if chunk:
            full.extend(chunk)
    return full


def _gather_route_stats_across_ranks(
    action_route_min: Dict[str, Dict[str, list]],
    action_route_max: Dict[str, Dict[str, list]],
    action_route_mean: Dict[str, Dict[str, list]],
    action_route_var: Dict[str, Dict[str, list]],
    action_route_raw: Dict[str, Dict[str, list]],
    action_route_sum: Dict[str, Dict[str, torch.Tensor]],
    action_route_sumsq: Dict[str, Dict[str, torch.Tensor]],
    action_route_count: Dict[str, Dict[str, torch.Tensor]],
    route_keys: List[str],
    action_dim: int,
    action_key: str = "default",
) -> None:
    """Per-route cross-rank all-gather for the route-aware action
    normalizer stats. Mutates the per-route dicts in place so they
    contain the *union* of all ranks' contributions on every rank.

    Sparse ranks contribute empty lists or zero tensors so the all-gather
    returns the global value rather than a rank-local empty value.
    """
    for route in route_keys:
        # Per-list dicts: ensure every rank has an action-key entry,
        # even if empty. The empty list is a valid contribution to
        # ``_all_gather_tensor_list`` (it just doesn't extend the
        # union).
        for d in (
            action_route_min[route],
            action_route_max[route],
            action_route_mean[route],
            action_route_var[route],
            action_route_raw[route],
        ):
            d.setdefault(action_key, [])
            d[action_key] = _all_gather_tensor_list(d[action_key])
        # Sum / sumsq / count: zero-placeholder for sparse ranks so
        # the stack + sum produces the correct global value.
        zero_sum = torch.zeros(action_dim, dtype=torch.float64)
        zero_count = torch.zeros(1, dtype=torch.float64)
        local_sum = action_route_sum[route].get(action_key, zero_sum)
        local_sumsq = action_route_sumsq[route].get(action_key, zero_sum)
        local_count = action_route_count[route].get(action_key, zero_count)
        action_route_sum[route][action_key] = torch.stack(
            _all_gather_tensor_list([local_sum])
        ).sum(0)
        action_route_sumsq[route][action_key] = torch.stack(
            _all_gather_tensor_list([local_sumsq])
        ).sum(0)
        action_route_count[route][action_key] = torch.stack(
            _all_gather_tensor_list([local_count])
        ).sum(0)


def _finalize_route_stats(
    *,
    route: str,
    empty_route: bool,
    action_meta: List[Dict[str, Any]],
    action_route_min: Dict[str, list],
    action_route_max: Dict[str, list],
    action_route_mean: Dict[str, list],
    action_route_var: Dict[str, list],
    action_route_raw: Dict[str, list],
    action_route_sum: Dict[str, torch.Tensor],
    action_route_sumsq: Dict[str, torch.Tensor],
    action_route_count: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Build the per-route stats block for ``action_by_route``.

    Called by ``BaseGameDataset.get_dataset_stats``.

    ``empty_route`` must be the *count-based* check the caller computes
    via ``action_route_count[route].get(action_key)`` — a presence-only
    check mis-classifies sparse ranks in the distributed case. When
    ``empty_route`` is True, the block is a single zero-tensor entry per
    dim with ``count=0, empty=True`` so the load side can fail-fast.
    When False, builds the nested per-action-key stats. A per-key empty
    list is also handled (skipped) so ``torch.stack`` on an empty list
    never fires.
    """
    route_block: Dict[str, Any] = {}
    if empty_route:
        # No tokens in this route — count=0 lets the load side fail-fast.
        action_dim = int(action_meta[0]["raw_shape"])
        zero = torch.zeros(action_dim)
        action_key = action_meta[0]["key"]
        route_block[action_key] = {
            "global_min": zero.clone(),
            "global_max": zero.clone(),
            "global_q01": zero.clone(),
            "global_q99": zero.clone(),
            "global_mean": zero.clone(),
            "global_std": zero.clone(),
        }
        route_block["count"] = 0
        route_block["empty"] = True
        return route_block
    # Build per-action-key stats under the nested key structure.
    for meta in action_meta:
        key = meta["key"]
        if key not in action_route_min:
            continue
        route_min_list = action_route_min[key]
        route_max_list = action_route_max[key]
        route_mean_list = action_route_mean[key]
        route_var_list = action_route_var[key]
        route_raw_list = action_route_raw[key]
        # Skip keys with empty lists (e.g. a per-key empty list in
        # distributed setting) to avoid ``torch.stack`` on an empty list.
        if not route_min_list or not route_max_list:
            continue
        # Per-route stepwise q01/q99 from concatenated tokens.
        if route_raw_list:
            all_route_data = torch.cat(route_raw_list, dim=0)
            route_q01 = _quantile_dim0_exact_tail(
                all_route_data, 0.01, fill=all_route_data.amin(dim=0)
            )
            route_q99 = _quantile_dim0_exact_tail(
                all_route_data, 0.99, fill=all_route_data.amax(dim=0)
            )
        else:
            action_dim = int(meta.get("raw_shape", 0))
            z = torch.zeros(action_dim)
            route_q01 = z.clone()
            route_q99 = z.clone()
        stepwise_min = torch.stack(route_min_list).amin(0)
        stepwise_max = torch.stack(route_max_list).amax(0)
        stepwise_mean, stepwise_std = get_stepwise_mean_std(
            route_mean_list, route_var_list
        )
        global_mean, global_std = get_global_mean_std(
            action_route_sum,
            action_route_sumsq,
            action_route_count,
            key,
            stepwise_mean.dtype,
        )
        route_block[key] = {
            "global_min": stepwise_min,
            "global_max": stepwise_max,
            "global_q01": route_q01,
            "global_q99": route_q99,
            "global_mean": global_mean,
            "global_std": global_std,
        }
    # Bookkeeping at the route level. By construction there is one
    # action key in the production config; sum across them if more.
    first_key = action_meta[0]["key"]
    route_block["count"] = int(
        action_route_count[first_key].double().sum().item()
    )
    route_block["empty"] = False
    return route_block


def _mode_is_q01_q99(mode: Any) -> bool:
    return str(mode).strip().lower() == "q01/q99"


def _as_plain_mode(mode: Any) -> Any:
    if OmegaConf.is_config(mode):
        return OmegaConf.to_container(mode, resolve=True)
    return mode


def _meta_dim_count(meta: Dict[str, Any]) -> int:
    raw_shape = meta["raw_shape"]
    if isinstance(raw_shape, (list, tuple)):
        return int(raw_shape[-1])
    return int(raw_shape)


def _quantile_dims_for_meta(
    field: str,
    meta: Dict[str, Any],
    default_mode: Any,
    exception_mode: Optional[Dict[str, Dict[str, Any]]],
) -> List[int]:
    """Return last-dimension indices whose normalizer mode needs q01/q99."""
    key = meta["key"]
    cur_mode = default_mode
    if exception_mode is not None and field in exception_mode and key in exception_mode[field]:
        cur_mode = exception_mode[field][key]

    cur_mode = _as_plain_mode(cur_mode)
    default_mode = _as_plain_mode(default_mode)
    n_dim = _meta_dim_count(meta)

    if _mode_is_q01_q99(cur_mode):
        return list(range(n_dim))
    if not isinstance(cur_mode, list):
        return []

    dims: set[int] = set()
    covered: set[int] = set()
    for entry in cur_mode:
        entry_dims = [int(d) for d in entry.get("dims", [])]
        covered.update(entry_dims)
        if _mode_is_q01_q99(entry.get("mode")):
            dims.update(entry_dims)

    if _mode_is_q01_q99(default_mode):
        dims.update(d for d in range(n_dim) if d not in covered)
    return sorted(dims)


def _topk_dim0_exact_tail(data: torch.Tensor, k: int, *, largest: bool) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(int(k), int(data.shape[0]))
    try:
        return torch.topk(data.float(), k=k, dim=0, largest=largest, sorted=False).values
    except RuntimeError as exc:
        logger.warning(
            "Direct topk failed while computing exact normalizer quantile; "
            "falling back to chunked tail merge. Error: %s",
            exc,
        )

    chunk_rows = 1_000_000
    tail = None
    for start in range(0, data.shape[0], chunk_rows):
        chunk = data[start:start + chunk_rows].float()
        chunk_k = min(k, int(chunk.shape[0]))
        chunk_tail = torch.topk(chunk, k=chunk_k, dim=0, largest=largest, sorted=False).values
        candidate = chunk_tail if tail is None else torch.cat([tail, chunk_tail], dim=0)
        tail = torch.topk(
            candidate,
            k=min(k, int(candidate.shape[0])),
            dim=0,
            largest=largest,
            sorted=False,
        ).values
    return tail


def _quantile_dim0_exact_tail(
    data: torch.Tensor,
    q: float,
    *,
    quantile_dims: Optional[List[int]] = None,
    fill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute exact linear-interpolated dim-0 quantile via tail selection.

    This avoids ``torch.quantile``'s oversized-input path while still using all
    rows. ``quantile_dims`` selects last-dimension entries that actually need
    q01/q99; other entries are copied from ``fill`` to preserve stats shape.
    """
    if data.shape[0] <= 0:
        raise ValueError("Cannot compute quantile for an empty tensor.")

    if quantile_dims is not None:
        if len(quantile_dims) == 0:
            if fill is None:
                raise ValueError("fill is required when quantile_dims is empty.")
            return fill.clone()
        dims_t = torch.as_tensor(quantile_dims, dtype=torch.long, device=data.device)
        source = data.index_select(-1, dims_t)
    else:
        dims_t = None
        source = data

    n = int(source.shape[0])
    pos = float(q) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    weight = pos - lo

    if q <= 0.5:
        tail = _topk_dim0_exact_tail(source, hi + 1, largest=False).sort(dim=0).values
        lo_value = tail[lo]
        hi_value = tail[hi]
    else:
        tail_start = lo
        tail = _topk_dim0_exact_tail(source, n - tail_start, largest=True).sort(dim=0).values
        lo_value = tail[lo - tail_start]
        hi_value = tail[hi - tail_start]

    value = lo_value if hi == lo else lo_value + (hi_value - lo_value) * weight
    value = value.to(data.dtype)

    if dims_t is None:
        return value
    if fill is None:
        raise ValueError("fill is required when quantile_dims selects a subset.")
    result = fill.clone()
    result.index_copy_(-1, dims_t, value)
    return result


def _distributed_quantile_dim0_exact_tail(
    tensor_list: List[torch.Tensor],
    q: float,
    *,
    quantile_dims: Optional[List[int]] = None,
    fill: torch.Tensor,
) -> torch.Tensor:
    """Exact global dim-0 quantile without gathering the full dataset.

    Each rank contributes only the smallest/largest local tail that can
    contain the requested global order statistics.  This is exact for the
    same linear interpolation used by :func:`_quantile_dim0_exact_tail`, and
    handles ranks with zero eligible episodes.
    """
    if quantile_dims is not None and len(quantile_dims) == 0:
        return fill.clone()

    local_data = torch.cat(tensor_list, dim=0) if tensor_list else None
    local_n = 0 if local_data is None else int(local_data.shape[0])
    gathered_counts = _all_gather_tensor_list(
        [torch.tensor([local_n], dtype=torch.int64)]
    )
    global_n = sum(int(count.item()) for count in gathered_counts)
    if global_n <= 0:
        raise ValueError("Cannot compute a distributed quantile with zero rows.")

    pos = float(q) * (global_n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    weight = pos - lo
    needed = hi + 1 if q <= 0.5 else global_n - lo

    local_tail: List[torch.Tensor] = []
    dims_t = None
    if local_data is not None:
        source = local_data
        if quantile_dims is not None:
            dims_t = torch.as_tensor(
                quantile_dims, dtype=torch.long, device=local_data.device
            )
            source = local_data.index_select(-1, dims_t)
        local_tail = [
            _topk_dim0_exact_tail(
                source,
                min(needed, local_n),
                largest=q > 0.5,
            )
        ]

    gathered_tail = _all_gather_tensor_list(local_tail)
    if not gathered_tail:
        raise ValueError("Distributed quantile tail gather returned no rows.")
    tail = torch.cat(gathered_tail, dim=0).sort(dim=0).values
    if q <= 0.5:
        lo_value = tail[lo]
        hi_value = tail[hi]
    else:
        tail_start = global_n - int(tail.shape[0])
        lo_value = tail[lo - tail_start]
        hi_value = tail[hi - tail_start]
    value = lo_value if lo == hi else lo_value + (hi_value - lo_value) * weight
    value = value.to(device=fill.device, dtype=fill.dtype)

    if quantile_dims is None:
        return value
    result = fill.clone()
    result.index_copy_(
        -1,
        torch.as_tensor(quantile_dims, dtype=torch.long, device=result.device),
        value.to(result.device),
    )
    return result


def _sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Sliding window with last-row replication for out-of-bounds indices
    along dim 0 (the time axis).

    Accepts 2D ``[N, D]`` → ``[N, window_size, D]`` (per-frame action)
    or 3D ``[N, K, D]`` → ``[N, window_size, K, D]``. The window
    axis is inserted at position 1, so a 2D route label of shape ``[N, 1]``
    windows to ``[N, action_size, 1]`` aligned with the action's
    ``[N, action_size, D]``.
    """
    assert x.dim() in (2, 3) and window_size > 0
    N = int(x.shape[0])
    i_indices = torch.arange(N).unsqueeze(1)
    j_indices = torch.arange(window_size).unsqueeze(0)
    indices = torch.clamp(i_indices + j_indices, min=0, max=N - 1)
    return x[indices]


def _build_episode_route_window(
    raw_state: torch.Tensor,
    *,
    route_dim_index: int,
    window_size: int,
    expected_starts: Optional[int] = None,
) -> torch.Tensor:
    """Build the route label aligned with episode-level action windows.

    ``_get_episode_data`` loads action and state from the same parquet
    rows. This helper deliberately builds the full ``N`` route windows;
    the stats caller then removes starts that cannot produce a complete
    training sample. The GUI dimension is selected on the *last* axis,
    after removing the loader's singleton axis.
    """
    if not isinstance(raw_state, torch.Tensor):
        raise TypeError(
            f"raw_state must be a torch.Tensor, got {type(raw_state).__name__}."
        )
    if raw_state.ndim == 3:
        if raw_state.shape[1] != 1:
            raise ValueError(
                "Episode route state must have shape [N, 1, D] when 3D, "
                f"got {tuple(raw_state.shape)}."
            )
        frame_state = raw_state[:, 0, :]
    elif raw_state.ndim == 2:
        frame_state = raw_state
    else:
        raise ValueError(
            "Episode route state must have shape [N, D] or [N, 1, D], "
            f"got {tuple(raw_state.shape)}."
        )

    dim = int(route_dim_index)
    if dim < 0 or dim >= int(frame_state.shape[-1]):
        raise ValueError(
            f"route_dim_index={dim} is out of range for state dim "
            f"{int(frame_state.shape[-1])}."
        )
    if expected_starts is not None and int(frame_state.shape[0]) != int(expected_starts):
        raise ValueError(
            "Episode action/state start count mismatch while building route "
            f"labels: state={int(frame_state.shape[0])}, "
            f"action={int(expected_starts)}."
        )

    route_per_frame = frame_state[:, dim:dim + 1]
    return _sliding_window_with_replication(
        route_per_frame, int(window_size)
    ).bool()
