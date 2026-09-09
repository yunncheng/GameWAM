"""GameWAM video dataset wrapper.

Adds frame sub-sampling, camera assembly, cached T5/VAE loading, and
final model-space normalization on top of the LeRobot-backed base dataset.
"""

import hashlib
import json
import os
import traceback
import threading
from types import MappingProxyType
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from accelerate import PartialState

from .base_game_dataset import BaseGameDataset
from .utils.cache_utils import compute_vae_cache_hash, resolve_image_video_size
from .utils.compact_index import load_compact_json_index
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from .transforms.image import Normalize
from gamewam.utils.logging_config import get_logger
from gamewam.utils import misc
from gamewam.utils.misc import LRUDict
from gamewam.models.history.history_module import (
    compute_history_starts,
    derive_history_geometry,
    derive_memory_write_capacity_segments,
)

logger = get_logger(__name__)

_SHARED_INDEX_LOCK = threading.Lock()
_SHARED_JSON_INDICES = {}

def _load_shared_json_index(index_path: Path):
    """Load one immutable cache index per process and share it across splits."""
    stat = index_path.stat()
    signature = (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_ino),
    )
    cache_key = str(index_path.resolve())
    with _SHARED_INDEX_LOCK:
        cached = _SHARED_JSON_INDICES.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            metadata, mapping = load_compact_json_index(index_path)
            index = dict(metadata)
        except OSError as exc:
            logger.warning(
                "Cannot create/read compact cache index for %s; "
                "falling back to the JSON mapping: %s",
                index_path,
                exc,
            )
            with open(index_path, "r", encoding="utf-8") as stream:
                index = json.load(stream)
            raw_mapping = index.get("mapping")
            if not isinstance(raw_mapping, dict):
                raise ValueError(f"Cache index {index_path} has no mapping object.")
            mapping = MappingProxyType(raw_mapping)
        index["mapping"] = mapping
        immutable_index = MappingProxyType(index)
        _SHARED_JSON_INDICES[cache_key] = (signature, immutable_index)
        return immutable_index


def _align_proprio_sequence_fields(
    proprio,
    proprio_field_valid: Optional[torch.Tensor],
    proprio_is_pad: torch.Tensor,
    proprio_dim_is_pad: Optional[torch.Tensor],
):
    values = list(proprio.values()) if isinstance(proprio, dict) else [proprio]
    if not values:
        raise ValueError("proprio must contain at least one state tensor.")
    lengths = {int(value.shape[0]) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"proprio state keys have inconsistent lengths: {sorted(lengths)}")
    sequence_length = next(iter(lengths))
    if int(proprio_is_pad.shape[0]) != sequence_length:
        raise ValueError(
            "proprio/proprio_is_pad length mismatch: "
            f"{sequence_length} vs {proprio_is_pad.shape[0]}"
        )
    if (
        proprio_field_valid is not None
        and int(proprio_field_valid.shape[0]) != sequence_length
    ):
        raise ValueError(
            "proprio/proprio_field_valid length mismatch: "
            f"{sequence_length} vs {proprio_field_valid.shape[0]}"
        )
    if proprio_dim_is_pad is not None:
        if proprio_dim_is_pad.ndim == 1:
            proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(0).expand(
                sequence_length, -1
            )
        elif int(proprio_dim_is_pad.shape[0]) != sequence_length:
            raise ValueError(
                "proprio/proprio_dim_is_pad length mismatch: "
                f"{sequence_length} vs {proprio_dim_is_pad.shape[0]}"
            )
    return proprio_dim_is_pad


def _normalize_vae_latent_cache_dtype(dtype: str) -> tuple[str, np.dtype, torch.dtype, int]:
    key = str(dtype).strip().lower()
    if key in {"no", "fp32", "float32"}:
        return "float32", np.dtype(np.float32), torch.float32, 4
    if key in {"fp16", "float16", "half"}:
        return "float16", np.dtype(np.float16), torch.float16, 2
    if key in {"bf16", "bfloat16"}:
        return "bfloat16", np.dtype(np.uint16), torch.bfloat16, 2
    raise ValueError(
        f"Unsupported `vae_latent_cache_dtype`: {dtype}. "
        "Expected one of: ['no', 'fp16', 'bf16']."
    )


def compute_event_occurrence_label(
    *,
    event_frame_idxs_in_clip: Optional[List[int]],
    clip_local_obs_start: int,
    num_obs_frames: int,
    pred_latents: int,
    raw_steps_per_latent: int,
    frame_stride: int,
    plan_window_raw_steps: Optional[int] = None,
    ignore_index: int = -100,
    per_block: bool = True,
) -> torch.Tensor:
    """Per-sample per-block ``event_occurrence`` label."""
    obs_window_last = (
        int(clip_local_obs_start) + max(0, int(num_obs_frames) - 1) * int(frame_stride)
    )
    if event_frame_idxs_in_clip is None:
        evts_rel: List[int] = []
        evt_known = False
    else:
        evts_rel = [int(e) for e in event_frame_idxs_in_clip]
        evt_known = True

    plan_window_raw_steps = (
        int(raw_steps_per_latent)
        if plan_window_raw_steps is None
        else int(plan_window_raw_steps)
    )
    if not per_block:
        if not evt_known:
            return torch.tensor([int(ignore_index)], dtype=torch.long)
        if plan_window_raw_steps <= 0:
            raise ValueError(
                f"plan_window_raw_steps must be positive for non-AR labels, got {plan_window_raw_steps}"
            )
        future_lo = obs_window_last + 1
        future_hi = obs_window_last + plan_window_raw_steps
        any_in_future = any(future_lo <= e <= future_hi for e in evts_rel)
        return torch.tensor([1 if any_in_future else 0], dtype=torch.long)

    if pred_latents <= 0:
        return torch.empty((0,), dtype=torch.long)
    if plan_window_raw_steps <= 0:
        raise ValueError(
            f"plan_window_raw_steps must be positive, got {plan_window_raw_steps}"
        )
    label = torch.full((int(pred_latents),), int(ignore_index), dtype=torch.long)
    if not evt_known:
        return label
    for i in range(int(pred_latents)):
        # Plan block i covers raw frames
        #   [obs_window_last + 1 + i*stride,
        #    obs_window_last + i*stride + plan_window]
        # inclusive on both ends (see docstring for the boundary convention).
        block_stride = int(raw_steps_per_latent)
        block_lo = obs_window_last + 1 + i * block_stride
        block_hi = obs_window_last + i * block_stride + plan_window_raw_steps
        label[i] = 1 if any(block_lo <= e <= block_hi for e in evts_rel) else 0
    return label


class GameVideoDataset(torch.utils.data.Dataset):
    """LeRobot-backed game dataset used by the trainer and precompute jobs."""

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: dict,
        num_frames: int = 33,
        video_size: List[int] = None,
        processor=None,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        pretrained_norm_stats: Optional[str] = None,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        frame_stride: int = 1,
        sample_interval: int = 4,
        action_video_freq_ratio: int = 1,
        num_obs_frames: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: Optional[str] = None,
        event_anchor_enabled: bool = False,
        event_anchor_radius: Optional[int] = None,
        sample_interval_event: Optional[int] = None,
        sample_interval_normal: Optional[int] = None,
        fine_task_resample_enabled: bool = False,
        fine_task_resample_mix_original: float = 0.5,
        fine_task_resample_gamma: float = 0.5,
        fine_task_resample_max_expected_repeat: float = 5.0,
        vae_latent_cache_dir: Optional[str] = None,
        vae_latent_cache_dtype: str = "bf16",
        video_only: bool = False,
        prompt_template: str = None,
        history: Optional[dict] = None,
        prediction_horizon: int = 16,
        execution_horizon: int = 8,
        ar_mode: Optional[bool] = None,
        delay_normalization: bool = False,
        dim_is_continuous: Optional[Sequence[bool]] = None,
        gui_label_dim_index: Optional[int] = None,
        state_schema: Optional[str] = None,
    ):
        if action_video_freq_ratio < 1:
            raise ValueError(f"action_video_freq_ratio must be >= 1, got {action_video_freq_ratio}")
        if num_frames <= 1:
            raise ValueError(f"num_frames must be > 1, got {num_frames}")
        if (num_frames - 1) % action_video_freq_ratio != 0:
            raise ValueError(
                f"num_frames-1 must be divisible by action_video_freq_ratio, "
                f"got {num_frames - 1} and {action_video_freq_ratio}"
            )
        num_video_transitions = (num_frames - 1) // action_video_freq_ratio
        if num_video_transitions % 4 != 0:
            raise ValueError(
                f"video transitions must be divisible by 4 for tokenization, got {num_video_transitions}"
            )
        self.delay_normalization = bool(delay_normalization)
        processor_schema_cfg = processor
        if isinstance(processor_schema_cfg, DictConfig):
            processor_schema_cfg = OmegaConf.to_container(
                processor_schema_cfg, resolve=True
            )
        # When Hydra's ``instantiate`` recurses through a nested
        # ``_target_ = GameWAMProcessor`` (e.g. via
        # ``MixedGameVideoDataset`` instantiating each child with
        # ``instantiate(node_dict, delay_normalization=True)``), the
        # processor arrives as an instantiated object, not a config.
        # Fall back to the instance's attributes for the early
        # ``proprio_encoder_type`` / ``cursor_reference_size`` extraction.
        if not isinstance(processor_schema_cfg, dict):
            processor_schema_cfg = {
                "proprio_encoder_type": getattr(
                    processor_schema_cfg, "proprio_encoder_type", None
                ),
                "cursor_reference_size": getattr(
                    processor_schema_cfg, "cursor_reference_size", None
                ),
            }
        processor_schema_cfg = processor_schema_cfg or {}
        minecraft_cursor_reference_size = None
        if (
            processor_schema_cfg.get("proprio_encoder_type")
            == "minecraft_grouped_cursor"
        ):
            minecraft_cursor_reference_size = processor_schema_cfg.get(
                "cursor_reference_size"
            )
            if minecraft_cursor_reference_size is None:
                raise ValueError(
                    "Minecraft cursor dataset construction requires processor "
                    "cursor_reference_size."
                )
        prediction_horizon = int(prediction_horizon)
        execution_horizon = int(execution_horizon)
        if prediction_horizon <= 0:
            raise ValueError(f"prediction_horizon must be positive, got {prediction_horizon}")
        if execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {execution_horizon}")
        if execution_horizon > prediction_horizon:
            raise ValueError(
                "execution_horizon must not exceed prediction_horizon, "
                f"got {execution_horizon} > {prediction_horizon}"
            )
        if ar_mode is None:
            ar_mode = prediction_horizon != execution_horizon
        ar_mode = bool(ar_mode)
        committed_future_horizon = (int(num_frames) - int(num_obs_frames)) * int(frame_stride)
        if not ar_mode and prediction_horizon != committed_future_horizon:
            raise ValueError(
                "Non-AR geometry must satisfy "
                "prediction_horizon == (num_frames - num_obs_frames) * frame_stride, "
                f"got {prediction_horizon} != {committed_future_horizon}"
            )
        lookahead_action_steps = prediction_horizon - execution_horizon if ar_mode else 0
        read_num_frames = int(num_frames)
        if lookahead_action_steps > 0:
            if prediction_horizon % execution_horizon != 0:
                raise ValueError(
                    "AR prediction_horizon must be a positive multiple of execution_horizon, "
                    f"got {prediction_horizon} and {execution_horizon}"
                )
            if lookahead_action_steps % int(frame_stride) != 0:
                raise ValueError(
                    "AR lookahead must align with frame_stride, got "
                    f"lookahead={lookahead_action_steps}, frame_stride={frame_stride}"
                )
            read_num_frames = int(num_frames) + lookahead_action_steps // int(frame_stride)
        base_action_window = (num_frames - 1) * frame_stride
        dense_action_size = base_action_window + lookahead_action_steps
        self.video_sample_indices = list(range(0, read_num_frames, action_video_freq_ratio))

        self.game_dataset = BaseGameDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            obs_size=read_num_frames,
            action_size=dense_action_size,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            frame_stride=frame_stride,
            sample_interval=sample_interval,
            event_anchor_enabled=event_anchor_enabled,
            event_anchor_radius=event_anchor_radius,
            sample_interval_event=sample_interval_event,
            sample_interval_normal=sample_interval_normal,
            fine_task_resample_enabled=fine_task_resample_enabled,
            fine_task_resample_mix_original=fine_task_resample_mix_original,
            fine_task_resample_gamma=fine_task_resample_gamma,
            fine_task_resample_max_expected_repeat=fine_task_resample_max_expected_repeat,
            video_only=video_only,
            minecraft_cursor_reference_size=minecraft_cursor_reference_size,
            state_schema=state_schema,
        )
        self.game_dataset._set_return_images(True)

        # Keep plain-python metadata for downstream code.
        self.shape_meta = self.game_dataset.shape_meta
        if video_size is None:
            video_size = resolve_image_video_size(self.shape_meta)
        else:
            video_size = [int(video_size[0]), int(video_size[1])]
        self.num_frames = num_frames
        self.read_num_frames = read_num_frames
        self.frame_stride = int(frame_stride)
        self.action_video_freq_ratio = action_video_freq_ratio
        self.num_obs_frames = num_obs_frames
        self.prediction_horizon = int(prediction_horizon)
        self.execution_horizon = int(execution_horizon)
        self.ar_mode = ar_mode

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        if self.text_embedding_cache_dir is not None:
            os.makedirs(self.text_embedding_cache_dir, exist_ok=True)
        self.context_len = context_len
        self.is_training_set = bool(is_training_set)
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        if prompt_template is None or "{task}" not in str(prompt_template):
            raise ValueError(f"`prompt_template` is required and must contain `{{task}}`, got {prompt_template}")
        self.prompt_template = str(prompt_template)
        self.video_only = video_only
        self.vae_latent_cache_dtype = _normalize_vae_latent_cache_dtype(vae_latent_cache_dtype)[0]

        self.vae_latent_cache_dir = vae_latent_cache_dir
        # AR-history val also needs latent cache to produce history_segment_latents
        ar_history = (history is not None)
        self.latent_only = self.vae_latent_cache_dir is not None and (is_training_set or ar_history) and not video_only
        self.game_dataset._set_return_images(not self.latent_only)
        # History geometry must exist before cache coverage validation.  A
        # change to ``num_obs_frames`` can leave the main sample keys reusable
        # while requiring a different set of history-window keys.
        self.history_config = history
        self.ar_history_mode = history is not None
        if self.ar_history_mode:
            self._history_geometry = self._derive_history_geometry(
                num_frames=num_frames,
                num_obs_frames=num_obs_frames,
                frame_stride=frame_stride,
                action_video_freq_ratio=action_video_freq_ratio,
            )
        self._vae_mmap = None
        self._vae_mapping = None
        self._vae_record_indices = None
        if self.vae_latent_cache_dir is not None:
            self._init_vae_latent_cache(
                dataset_dirs=dataset_dirs,
                video_size=video_size,
                num_frames=read_num_frames,
                frame_stride=frame_stride,
                sample_interval=sample_interval,
                action_video_freq_ratio=action_video_freq_ratio,
                event_anchor_enabled=event_anchor_enabled,
                event_anchor_radius=event_anchor_radius,
                sample_interval_event=sample_interval_event,
                sample_interval_normal=sample_interval_normal,
                latent_dtype=self.vae_latent_cache_dtype,
            )
            if self.latent_only:
                self._validate_vae_latent_cache_coverage()

        # In-memory cache for text embeddings (avoids repeated torch.load for same prompt).
        # LRU-bounded to prevent unbounded growth across many unique tasks.
        self._text_embed_cache = LRUDict(max_size=int(os.environ.get("GAMEWAM_TEXT_CACHE_SIZE", "200")))

        self._event_occurrence_geometry: Optional[dict] = None
        if int(num_obs_frames) < int(num_frames):
            try:
                geom = derive_history_geometry(
                    num_frames=int(num_frames),
                    num_obs_frames=int(num_obs_frames),
                    frame_stride=int(frame_stride),
                    action_video_freq_ratio=int(action_video_freq_ratio),
                )
                self._event_occurrence_geometry = {
                    "pred_latents": int(geom.pred_latents),
                    "raw_steps_per_latent": int(geom.raw_steps_per_latent),
                }
            except Exception:  # pragma: no cover - unsupported auxiliary geometry
                self._event_occurrence_geometry = None
        self._event_occurrence_per_block = self._event_occurrence_geometry is not None

        # Final model-space normalization. Spatial resizing is owned by processor transforms.
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )

        if processor is not None:
            if isinstance(processor, dict) and not isinstance(processor, DictConfig):
                processor = OmegaConf.create(processor)
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if getattr(processor, "num_frames", None) != self.read_num_frames:
                processor.num_frames = self.read_num_frames
            if not self.video_only:
                if delay_normalization:
                    pass
                elif not pretrained_norm_stats:
                    if not is_training_set:
                        raise ValueError(
                            "pretrained_norm_stats must be provided for validation/test sets "
                            "since we don't want to calculate stats on them."
                        )

                    logger.info(
                        "Calculating dataset stats for normalization "
                        "(distributed: each rank handles a slice)..."
                    )
                    dataset_stats = self.game_dataset.get_dataset_stats(processor)
                    if PartialState().is_main_process:
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(
                            dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                        )
                else:
                    dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                    logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                    if PartialState().is_main_process:
                        work_dir = misc.get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

                if not delay_normalization:
                    processor.set_normalizer_from_stats(dataset_stats)
            self.game_dataset.set_processor(processor)

    @property
    def processor(self):
        return self.game_dataset.processor

    def _finalize_video(self, video: torch.Tensor) -> torch.Tensor:
        target_h, target_w = int(self.video_size[0]), int(self.video_size[1])
        if tuple(video.shape[-2:]) != (target_h, target_w):
            raise ValueError(
                "Processor image transforms must produce `video_size` before GameVideoDataset finalization: "
                f"got spatial={tuple(video.shape[-2:])}, expected={(target_h, target_w)}."
            )
        return self.normalize_transform(video)

    def _build_event_occurrence_label(self, sample: dict) -> Optional[torch.Tensor]:
        """Per-sample event_occurrence label (``[pred_latents]`` or ``[1]``) or None for video_only."""
        if self.video_only:
            return None
        clip_local_obs_start = int(sample.get("clip_frame_idx") or 0)
        # Three cases:
        #   - no anchor and no pre-anchor list → non-event data
        #     (contractor / GUI / etc). Pass None so the label is
        #     ignore (-100) rather than all-zero (which would teach
        #     the head to predict 0 for the vast majority of
        #     non-event data and drown out the positives).
        #   - anchor only → list = [anchor].
        #   - anchor + pre-anchor list → list = sorted unique union.
        pre = sample.get("pre_anchor_event_frame_idxs_in_clip")
        anchor = sample.get("event_anchor_frame_idx_in_clip")
        if pre is None and anchor is None:
            evt_idxs: Optional[List[int]] = None
        else:
            evt_idxs = sorted({int(x) for x in (pre or [])} | {int(anchor)})
        if not self._event_occurrence_per_block:
            return compute_event_occurrence_label(
                event_frame_idxs_in_clip=evt_idxs,
                clip_local_obs_start=clip_local_obs_start,
                num_obs_frames=int(self.num_obs_frames),
                pred_latents=0,
                raw_steps_per_latent=int(self.prediction_horizon),
                frame_stride=int(self.frame_stride),
                plan_window_raw_steps=int(self.prediction_horizon),
                per_block=False,
            )
        geom = self._event_occurrence_geometry
        if geom is None or int(geom["pred_latents"]) <= 0:
            return None
        return compute_event_occurrence_label(
            event_frame_idxs_in_clip=evt_idxs,
            clip_local_obs_start=clip_local_obs_start,
            num_obs_frames=int(self.num_obs_frames),
            pred_latents=int(geom["pred_latents"]),
            raw_steps_per_latent=int(geom["raw_steps_per_latent"]),
            frame_stride=int(self.frame_stride),
            plan_window_raw_steps=int(self.prediction_horizon),
            per_block=True,
        )

    def __len__(self):
        return len(self.game_dataset)

    def get_sampler_weights(self):
        return self.game_dataset.get_sampler_weights()

    def get_task_sampling_spec(self):
        return self.game_dataset.get_task_sampling_spec()

    def _prepare_video(self, sample):
        video = sample["pixel_values"]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, t_video, channels, height, width = video.shape
        else:
            if video.ndim != 4:
                raise ValueError(f"Expected video shape [T, C, H, W], got {tuple(video.shape)}")
            video = video[self.video_sample_indices, :, :, :]
            t_video, channels, height, width = video.shape

        video = video.view(num_cameras, t_video, channels, height, width)
        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical."
                )
        else:
            video = video.squeeze(0)
        return self._finalize_video(video).permute(1, 0, 2, 3)

    def get_raw_video(self, idx: int, sample_idx=None):
        position = int(idx)
        if sample_idx is not None:
            if isinstance(sample_idx, torch.Tensor):
                sample_idx = int(sample_idx.reshape(-1)[0].item())
            else:
                sample_idx = int(sample_idx)
            if int(self.game_dataset.sample_starts[position]) != sample_idx:
                positions = np.flatnonzero(
                    np.asarray(self.game_dataset.sample_starts, dtype=np.int64) == sample_idx
                )
                if positions.size == 0:
                    raise KeyError(f"Sample index {sample_idx} is not present in this dataset.")
                position = int(positions[0])

        return_images = bool(self.game_dataset.return_images)
        if not return_images:
            self.game_dataset._set_return_images(True)
        try:
            sample = self.game_dataset[position]
        finally:
            if not return_images:
                self.game_dataset._set_return_images(False)
        if sample_idx is not None and int(sample["idx"]) != sample_idx:
            raise RuntimeError(
                f"Raw-video sample mismatch: expected idx={sample_idx}, got {int(sample['idx'])}."
            )
        return self._prepare_video(sample)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.game_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample.get("action_is_pad")
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample.get("proprio_is_pad")
            has_pad = (
                (action_is_pad is not None and bool(action_is_pad.any().item()))
                or bool(image_is_pad.any().item())
                or (proprio_is_pad is not None and bool(proprio_is_pad.any().item()))
            )
            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.game_dataset))

        image_is_pad = sample["image_is_pad"]

        # In delay_normalization mode, action/proprio are kept as dicts
        # (before normalizer/merger). Shape assertions use .shape so
        # they are skipped; MixedGameVideoDataset handles normalization.
        _delay = self.delay_normalization

        if self.latent_only:
            image_is_pad = image_is_pad[self.video_sample_indices]
            action = sample["action"]
            proprio = sample["proprio"]
            proprio_field_valid = sample.get("proprio_field_valid")
            proprio_is_pad = sample["proprio_is_pad"][:-1]
            if _delay:
                # proprio is a dict {key: tensor[T, dim]}, crop to action horizon
                proprio = {k: v[:-1, :] for k, v in proprio.items()}
                if proprio_field_valid is not None:
                    proprio_field_valid = proprio_field_valid[:-1]
                proprio_dim_is_pad = sample.get("state_dim_is_pad", sample.get("proprio_dim_is_pad"))
                if proprio_dim_is_pad is not None:
                    if proprio_dim_is_pad.ndim > 1:
                        proprio_dim_is_pad = proprio_dim_is_pad[:-1]
            else:
                proprio = proprio[:-1, :]
                if proprio_field_valid is not None:
                    proprio_field_valid = proprio_field_valid[:-1]
                proprio_dim_is_pad = sample.get("proprio_dim_is_pad")
                if proprio_dim_is_pad is not None:
                    if proprio_dim_is_pad.ndim == 1:
                        proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(0).expand(proprio.shape)
                    else:
                        proprio_dim_is_pad = proprio_dim_is_pad[:-1, :]
            if self.ar_history_mode:
                max_proprio_steps = (
                    (int(self.num_obs_frames) - 1) * int(self.frame_stride)
                    + 1
                    + (int(self._history_geometry["pred_latents"]) - 1)
                    * int(self.execution_horizon)
                )
                if isinstance(proprio, dict):
                    proprio = {
                        key: value[:max_proprio_steps]
                        for key, value in proprio.items()
                    }
                else:
                    proprio = proprio[:max_proprio_steps]
                if proprio_field_valid is not None:
                    proprio_field_valid = proprio_field_valid[:max_proprio_steps]
                proprio_is_pad = proprio_is_pad[:max_proprio_steps]
                if proprio_dim_is_pad is not None and proprio_dim_is_pad.ndim > 1:
                    proprio_dim_is_pad = proprio_dim_is_pad[:max_proprio_steps]
            proprio_dim_is_pad = _align_proprio_sequence_fields(
                proprio,
                proprio_field_valid,
                proprio_is_pad,
                proprio_dim_is_pad,
            )
            num_video_frames = len(self.video_sample_indices)
            if not _delay:
                if action.shape[0] % (num_video_frames - 1) != 0:
                    raise ValueError(
                        f"`action` horizon must be divisible by `video` transitions, "
                        f"got {action.shape[0]} and {num_video_frames - 1}"
                    )

            task = sample["instruction"]
            instruction = self.prompt_template.format(task=task)
            context, context_mask = self._get_cached_text_context(instruction)

            data = {
                "action": action,
                "proprio": proprio,
                "proprio_dim_is_pad": proprio_dim_is_pad,
                "proprio_field_valid": proprio_field_valid,
                "prompt": instruction,
                "context": context,
                "context_mask": context_mask,
                "image_is_pad": image_is_pad,
                "action_is_pad": sample["action_is_pad"],
                "proprio_is_pad": proprio_is_pad,
                "idx": sample["idx"],
                "vae_cache_key": sample["vae_cache_key"],
                "input_latents": self._load_vae_latent_for_sample(sample_idx, sample),
            }
            for key in (
                "target_discrete_action",
                "target_continuous_action",
                "raw_continuous_action",
                "action_route_before_action",
                "action_route_valid",
            ):
                if key in sample:
                    data[key] = sample[key]
            event_occurrence_label = self._build_event_occurrence_label(sample)
            if event_occurrence_label is not None:
                data["event_occurrence_label"] = event_occurrence_label
            if self.is_training_set:
                empty_instruction = self.prompt_template.format(task="")
                uncond_context, uncond_context_mask = self._get_cached_text_context(empty_instruction)
                data["uncond_context"] = uncond_context
                data["uncond_context_mask"] = uncond_context_mask
            if self.ar_history_mode:
                # AR-history mode: load segments from main VAE cache
                history_data = self._load_history_segments_for_sample(sample["idx"])
                data.update(history_data)
            return data

        video = self._prepare_video(sample)
        image_is_pad = image_is_pad[self.video_sample_indices]

        if self.video_only:
            data = {"video": video, "idx": sample["idx"], "vae_cache_key": sample["vae_cache_key"]}
            if "instruction" in sample:
                task = sample["instruction"]
                data["prompt"] = self.prompt_template.format(task=task)
            return data

        action = sample["action"]  # [T-1, action_dim] or dict in delay mode
        proprio = sample["proprio"]
        proprio_field_valid = sample.get("proprio_field_valid")
        proprio_is_pad = sample["proprio_is_pad"][:-1]
        if _delay:
            proprio = {k: v[:-1, :] for k, v in proprio.items()}
            if proprio_field_valid is not None:
                proprio_field_valid = proprio_field_valid[:-1]
            proprio_dim_is_pad = sample.get("state_dim_is_pad", sample.get("proprio_dim_is_pad"))
            if proprio_dim_is_pad is not None:
                if proprio_dim_is_pad.ndim > 1:
                    proprio_dim_is_pad = proprio_dim_is_pad[:-1]
        else:
            proprio = proprio[:-1, :]  # [T-1, state_dim], align with action
            if proprio_field_valid is not None:
                proprio_field_valid = proprio_field_valid[:-1]
            proprio_dim_is_pad = sample.get("proprio_dim_is_pad")
            if proprio_dim_is_pad is not None:
                if proprio_dim_is_pad.ndim == 1:
                    proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(0).expand(proprio.shape)
                else:
                    proprio_dim_is_pad = proprio_dim_is_pad[:-1, :]

        if self.ar_history_mode:
            max_proprio_steps = (
                (int(self.num_obs_frames) - 1) * int(self.frame_stride)
                + 1
                + (int(self._history_geometry["pred_latents"]) - 1)
                * int(self.execution_horizon)
            )
            if isinstance(proprio, dict):
                proprio = {
                    key: value[:max_proprio_steps]
                    for key, value in proprio.items()
                }
            else:
                proprio = proprio[:max_proprio_steps]
            if proprio_field_valid is not None:
                proprio_field_valid = proprio_field_valid[:max_proprio_steps]
            proprio_is_pad = proprio_is_pad[:max_proprio_steps]
            if proprio_dim_is_pad is not None and proprio_dim_is_pad.ndim > 1:
                proprio_dim_is_pad = proprio_dim_is_pad[:max_proprio_steps]
        proprio_dim_is_pad = _align_proprio_sequence_fields(
            proprio,
            proprio_field_valid,
            proprio_is_pad,
            proprio_dim_is_pad,
        )
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if not _delay:
            if action.shape[0] % (video.shape[1] - 1) != 0:
                raise ValueError(
                    f"`action` horizon must be divisible by `video` transitions, "
                    f"got {action.shape[0]} and {video.shape[1] - 1}"
                )

        task = sample["instruction"]
        instruction = self.prompt_template.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "proprio_field_valid": proprio_field_valid,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": proprio_is_pad,
            "idx": sample["idx"],
            "vae_cache_key": sample["vae_cache_key"],
        }
        for key in (
            "target_discrete_action",
            "target_continuous_action",
            "raw_continuous_action",
            "action_route_before_action",
            "action_route_valid",
        ):
            if key in sample:
                data[key] = sample[key]
        event_occurrence_label = self._build_event_occurrence_label(sample)
        if event_occurrence_label is not None:
            data["event_occurrence_label"] = event_occurrence_label

        if self.text_embedding_cache_dir is not None:
            context, context_mask = self._get_cached_text_context(instruction)
            data["context"] = context
            data["context_mask"] = context_mask

            if self.is_training_set:
                empty_instruction = self.prompt_template.format(task="")
                uncond_context, uncond_context_mask = self._get_cached_text_context(empty_instruction)
                data["uncond_context"] = uncond_context
                data["uncond_context_mask"] = uncond_context_mask

        if self.vae_latent_cache_dir is not None:
            data["input_latents"] = self._load_vae_latent_for_sample(sample_idx, sample)

        return data

    def _init_vae_latent_cache(
        self,
        dataset_dirs,
        video_size,
        num_frames,
        frame_stride,
        sample_interval,
        action_video_freq_ratio,
        event_anchor_enabled,
        event_anchor_radius,
        sample_interval_event,
        sample_interval_normal,
        latent_dtype,
    ):
        """Open mmap file for VAE latent cache."""
        config_hash = compute_vae_cache_hash(
            video_size=video_size,
            num_frames=num_frames,
            frame_stride=frame_stride,
            sample_interval=sample_interval,
            action_video_freq_ratio=action_video_freq_ratio,
            event_anchor_enabled=event_anchor_enabled,
            event_anchor_radius=event_anchor_radius,
            sample_interval_event=sample_interval_event,
            sample_interval_normal=sample_interval_normal,
            latent_dtype=latent_dtype,
        )

        cache_dir = Path(self.vae_latent_cache_dir)
        bin_path = cache_dir / f"{config_hash}.latents.bin"
        idx_path = cache_dir / f"{config_hash}.latents.idx"

        if not bin_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"Missing VAE latent cache: {bin_path} or {idx_path}. "
                "Run scripts/precompute/vae_latents.py first."
            )

        index = _load_shared_json_index(idx_path)

        index_dataset_dirs = index.get("dataset_dirs")
        if index_dataset_dirs is not None:
            current_dataset_dirs = [str(d) for d in dataset_dirs]
            cached_dataset_dirs = [str(d) for d in index_dataset_dirs]
            current_counts = Counter(current_dataset_dirs)
            cached_counts = Counter(cached_dataset_dirs)
            missing_dirs = list((current_counts - cached_counts).elements())
            if missing_dirs:
                raise ValueError(
                    "VAE latent cache dataset_dirs mismatch: "
                    f"missing {missing_dirs} in cache; current={current_dataset_dirs}, cached={cached_dataset_dirs}. "
                    "Re-run scripts/precompute/vae_latents.py after changing dataset_dirs."
                )
            if cached_counts != current_counts:
                logger.warning(
                    "VAE latent cache was built for a superset of current dataset_dirs; "
                    "cache lookup remains valid because records are keyed by physical dataset/episode/frame."
                )
            elif cached_dataset_dirs != current_dataset_dirs:
                logger.warning(
                    "VAE latent cache dataset_dirs order differs from current config; "
                    "cache lookup remains valid because records are keyed by physical dataset/episode/frame. "
                    "Training sample order may still change."
                )

        index_dtype = str(index.get("dtype", "bfloat16"))
        self._vae_cache_dtype, self._vae_numpy_dtype, self._vae_torch_dtype, self._vae_bytes_per_element = (
            _normalize_vae_latent_cache_dtype(index_dtype)
        )
        if self._vae_cache_dtype != latent_dtype:
            raise ValueError(
                f"VAE latent cache dtype mismatch: expected {latent_dtype}, got {self._vae_cache_dtype}. "
                "Re-run scripts/precompute/vae_latents.py with matching mixed_precision."
            )
        index_hash = str(index.get("config_hash", ""))
        if index_hash and index_hash != config_hash:
            raise ValueError(f"VAE latent cache hash mismatch: expected {config_hash}, got {index_hash}.")
        self._vae_latent_shape = tuple(index["latent_shape"])
        self._vae_bytes_per_sample = int(index["bytes_per_sample"])
        self._vae_mapping = index["mapping"]
        self._vae_elements_per_sample = 1
        for d in self._vae_latent_shape:
            self._vae_elements_per_sample *= d
        if self._vae_bytes_per_sample != self._vae_elements_per_sample * self._vae_bytes_per_element:
            raise ValueError(
                f"VAE latent cache bytes_per_sample mismatch: index has {self._vae_bytes_per_sample}, "
                f"expected {self._vae_elements_per_sample * self._vae_bytes_per_element}."
            )
        expected_bin_size = int(index.get("num_records", len(self._vae_mapping))) * self._vae_bytes_per_sample
        actual_bin_size = bin_path.stat().st_size
        if actual_bin_size != expected_bin_size:
            raise ValueError(
                f"VAE latent cache bin size mismatch: {bin_path} has {actual_bin_size} bytes, "
                f"expected {expected_bin_size}."
            )
        self._vae_bin_path = str(bin_path)
        self._vae_mmap = np.memmap(str(bin_path), dtype=self._vae_numpy_dtype, mode="r")

        logger.info(
            "VAE latent cache loaded: %s, config_hash=%s, dtype=%s, %d records, shape=%s",
            bin_path, config_hash, self._vae_cache_dtype, len(self._vae_mapping), self._vae_latent_shape,
        )

    def _load_vae_latent_by_record(self, record_idx: int) -> torch.Tensor:
        record_idx = int(record_idx)
        start_elem = record_idx * self._vae_elements_per_sample
        end_elem = start_elem + self._vae_elements_per_sample
        raw = self._vae_mmap[start_elem:end_elem].copy()
        tensor = torch.from_numpy(raw)
        if self._vae_cache_dtype == "bfloat16":
            tensor = tensor.view(torch.bfloat16)
        return tensor.reshape(self._vae_latent_shape)

    def _load_vae_latent_prefix_by_record(self, record_idx: int, t_prefix: int) -> torch.Tensor:
        """Load only the first ``t_prefix`` latent frames of a record.

        Bitwise identical to ``_load_vae_latent_by_record(...)[:, :t_prefix]``
        but copies only the needed prefix out of the mmap (records are
        contiguous ``[C, T, H, W]``, so the prefix is a strided read).
        """
        record_idx = int(record_idx)
        t_prefix = int(t_prefix)
        shape = self._vae_latent_shape
        C = int(shape[0])
        T_total = int(shape[1])
        H = int(shape[2])
        W = int(shape[3])
        if t_prefix <= 0:
            raise ValueError(f"t_prefix must be positive, got {t_prefix}")
        if t_prefix > T_total:
            raise IndexError(
                f"latent has {T_total} frames, expected at least {t_prefix}"
            )
        if t_prefix == T_total:
            return self._load_vae_latent_by_record(record_idx)
        records_view = getattr(self, "_vae_records_view", None)
        if records_view is None:
            num_records = self._vae_mmap.shape[0] // self._vae_elements_per_sample
            records_view = self._vae_mmap.reshape(num_records, C, T_total, H, W)
            self._vae_records_view = records_view
        raw = records_view[record_idx, :, :t_prefix].copy()
        tensor = torch.from_numpy(raw)
        if self._vae_cache_dtype == "bfloat16":
            tensor = tensor.view(torch.bfloat16)
        return tensor.reshape(C, t_prefix, H, W)

    def _load_vae_latent(self, cache_key: str) -> torch.Tensor:
        """Load a precomputed VAE latent from the mmap cache."""
        record_idx = self._vae_mapping.get(cache_key)
        if record_idx is None:
            raise KeyError(
                f"VAE latent cache miss: cache_key={cache_key} not found. "
                "Re-run scripts/precompute/vae_latents.py to regenerate the cache."
            )
        return self._load_vae_latent_by_record(record_idx)

    def _load_vae_latent_for_sample(self, sample_position: int, sample: dict) -> torch.Tensor:
        """Use the coverage-built O(1) record table on the ordinary sample path."""
        sample_position = int(sample_position)
        if (
            self._vae_record_indices is not None
            and 0 <= sample_position < len(self._vae_record_indices)
            and int(sample["idx"])
            == int(self.game_dataset.sample_starts[sample_position])
        ):
            record_idx = int(self._vae_record_indices[sample_position])
            if record_idx >= 0:
                return self._load_vae_latent_by_record(record_idx)
        return self._load_vae_latent(str(sample["vae_cache_key"]))

    def _required_history_cache_keys(self, sample_idx: int) -> list[str]:
        if not self.ar_history_mode:
            return []
        cache_key = self.game_dataset.get_vae_cache_key(int(sample_idx))
        try:
            dataset_part, episode_part, frame_part = cache_key.rsplit("::", 2)
            window_start_raw = int(frame_part.split("=", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid VAE cache key format: {cache_key!r}") from exc

        geom = self._history_geometry
        current_raw = window_start_raw + (self.num_obs_frames - 1) * self.frame_stride
        recent_num = int(self.history_config.get("recent_num_segments", 2))
        max_depth = recent_num + derive_memory_write_capacity_segments(self.history_config)
        if max_depth <= 0:
            return []
        starts = compute_history_starts(current_raw, 0, int(geom["cycle_stride_raw"]))
        starts = starts[-max_depth:]
        return [
            f"{dataset_part}::{episode_part}::frame={int(start_raw)}"
            for start_raw in starts
        ]

    def _validate_vae_latent_cache_coverage(self):
        """Validate coverage and cache current-sample record indices compactly."""
        if self._vae_mapping is None:
            return

        sample_starts = self.game_dataset.sample_starts
        record_indices = np.full(len(sample_starts), -1, dtype=np.int64)
        missing_examples = []
        missing_count = 0
        missing_current_keys = set()
        missing_history_keys = set()
        current_prefix = None
        checked_history = set()

        lookup_many = getattr(self._vae_mapping, "get_many", None)
        lookup_chunk_size = 65_536

        for chunk_start in range(0, len(sample_starts), lookup_chunk_size):
            chunk_stop = min(chunk_start + lookup_chunk_size, len(sample_starts))
            chunk_starts = [
                int(sample_starts[position])
                for position in range(chunk_start, chunk_stop)
            ]
            current_keys = [
                self.game_dataset.get_vae_cache_key(sample_start)
                for sample_start in chunk_starts
            ]
            if callable(lookup_many):
                current_records = lookup_many(current_keys, None)
            else:
                current_records = [
                    self._vae_mapping.get(cache_key)
                    for cache_key in current_keys
                ]

            history_keys = []
            for offset, (sample_start, cache_key, found) in enumerate(
                zip(chunk_starts, current_keys, current_records)
            ):
                sample_position = chunk_start + offset
                prefix = cache_key.rsplit("::frame=", 1)[0]
                if prefix != current_prefix:
                    current_prefix = prefix
                    checked_history.clear()

                record_idx = -1 if found is None else int(found)
                if record_idx < 0 and cache_key not in missing_current_keys:
                    missing_current_keys.add(cache_key)
                    missing_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(cache_key)
                record_indices[sample_position] = record_idx

                for history_key in self._required_history_cache_keys(sample_start):
                    if history_key in checked_history:
                        continue
                    checked_history.add(history_key)
                    history_keys.append(history_key)

            if callable(lookup_many):
                history_records = lookup_many(history_keys, None)
            else:
                history_records = [
                    self._vae_mapping.get(history_key)
                    for history_key in history_keys
                ]
            for history_key, found in zip(history_keys, history_records):
                if found is None and history_key not in missing_history_keys:
                    missing_history_keys.add(history_key)
                    if len(missing_examples) < 5:
                        missing_examples.append(history_key)

        missing_history_count = len(missing_history_keys)
        if missing_count or missing_history_count:
            examples = ", ".join(missing_examples)
            raise KeyError(
                "VAE latent cache does not cover the current dataset sampling plan: "
                f"missing_current={missing_count}/{len(sample_starts)}, "
                f"missing_history_references={missing_history_count}. "
                f"Examples: {examples}. "
                "Re-run scripts/precompute/vae_latents.py for this task/config. "
                "The precompute is append-only when possible, so changing "
                "num_obs_frames normally only adds the newly required history keys."
            )

        self._vae_record_indices = record_indices
        logger.info(
            "VAE latent cache coverage validated: %d sample starts.",
            len(sample_starts),
        )

    # AR-history: history segment loading from main VAE cache

    @staticmethod
    def _derive_history_geometry(num_frames, num_obs_frames, frame_stride, action_video_freq_ratio):
        """Derive AR geometry for history segment loading (delegates to shared implementation)."""
        geom = derive_history_geometry(num_frames, num_obs_frames, frame_stride, action_video_freq_ratio)
        return {
            "cycle_stride_raw": geom.cycle_stride_raw,
            "pred_latents": geom.pred_latents,
            "history_segment_latents": geom.history_segment_latents,
            "total_latents": geom.total_latents,
            "video_step_raw": geom.video_step_raw,
            "raw_steps_per_latent": geom.raw_steps_per_latent,
        }

    def _load_history_segments_for_sample(self, sample_idx: int) -> dict:
        """Load history segments for AR-history mode.

        Returns dict with:
            history_segment_latents: [max_depth, C, T_seg, H, W]
            history_segment_mask: [max_depth] bool
            history_segment_start_raw: [max_depth]
            history_segment_next_cycle_start_raw: [max_depth]
            sample_start_raw: scalar
        """
        geom = self._history_geometry
        cycle_stride_raw = geom["cycle_stride_raw"]
        T_seg = geom["history_segment_latents"]
        recent_num = self.history_config.get("recent_num_segments", 2)
        memory_write_capacity = derive_memory_write_capacity_segments(self.history_config)
        max_depth = recent_num + memory_write_capacity

        # Parse sample's episode and frame position
        game_ds = self.game_dataset
        cache_key = game_ds.get_vae_cache_key(sample_idx)
        parts = cache_key.split("::")
        dataset_part = parts[0]
        episode_part = parts[1]
        frame_part = parts[2]
        frame_in_ep = int(frame_part.split("=")[1])

        # frame_in_ep is the cached VAE window start. History is relative to
        # the current decision raw step at the end of the observation context.
        window_start_raw = frame_in_ep
        obs_context_raw_span = (self.num_obs_frames - 1) * self.frame_stride
        sample_start_raw = window_start_raw + obs_context_raw_span
        episode_start_raw = 0  # episodes always start at raw frame 0

        history_starts = []
        p = sample_start_raw - cycle_stride_raw
        while p >= episode_start_raw:
            history_starts.insert(0, p)
            p -= cycle_stride_raw

        if len(history_starts) > max_depth:
            history_starts = history_starts[-max_depth:]

        # Load segment latents from main VAE cache
        # Each segment starts at history_starts[i] and covers T_seg latent frames
        seg_latents_list = []
        seg_mask_list = []
        seg_start_raw_list = []
        seg_next_raw_list = []

        # Get latent shape from main cache
        if self._vae_latent_shape is not None:
            C = self._vae_latent_shape[0]
            H_lat = self._vae_latent_shape[2]
            W_lat = self._vae_latent_shape[3]
        else:
            C, H_lat, W_lat = 48, 14, 14  # fallback

        for start_raw in history_starts:
            # Build cache key for this history start position
            hist_cache_key = f"{dataset_part}::{episode_part}::frame={start_raw}"
            try:
                record_idx = self._vae_mapping.get(hist_cache_key)
                if record_idx is None:
                    raise KeyError(
                        f"VAE latent cache miss: cache_key={hist_cache_key} not found. "
                        "Re-run scripts/precompute/vae_latents.py to regenerate the cache."
                    )
                # Only the T_seg prefix of the record is consumed; copy just
                # that slice out of the mmap (identical values to loading the
                # full record and slicing).
                seg_latent = self._load_vae_latent_prefix_by_record(record_idx, T_seg)  # [C, T_seg, H, W]
                seg_latents_list.append(seg_latent)
                seg_mask_list.append(True)
                seg_start_raw_list.append(start_raw)
                seg_next_raw_list.append(start_raw + cycle_stride_raw)
            except (KeyError, IndexError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"Missing VAE cache for history segment {hist_cache_key}; "
                    f"sample_start_raw={sample_start_raw}, start_raw={start_raw}. "
                    "Re-run VAE precompute with AR-history injection enabled."
                ) from e

        num_valid = min(len(seg_latents_list), max_depth)
        # Pad the mask metadata to max_depth
        while len(seg_mask_list) < max_depth:
            seg_mask_list.append(False)
            seg_start_raw_list.append(0)
            seg_next_raw_list.append(0)

        # Truncate if somehow we have more
        seg_mask_list = seg_mask_list[:max_depth]
        seg_start_raw_list = seg_start_raw_list[:max_depth]
        seg_next_raw_list = seg_next_raw_list[:max_depth]

        stacked = torch.zeros(
            max_depth, C, T_seg, H_lat, W_lat, dtype=self._vae_torch_dtype
        )
        for idx in range(num_valid):
            stacked[idx].copy_(seg_latents_list[idx])

        return {
            "history_segment_latents": stacked,  # [max_depth, C, T_seg, H, W]
            "history_segment_mask": torch.tensor(seg_mask_list, dtype=torch.bool),
            "history_segment_start_raw": torch.tensor(seg_start_raw_list, dtype=torch.long),
            "history_segment_next_cycle_start_raw": torch.tensor(seg_next_raw_list, dtype=torch.long),
            "sample_start_raw": torch.tensor(sample_start_raw, dtype=torch.long),
        }

    def _get_cached_text_context(self, prompt: str):
        """Load pre-computed T5 text embedding from cache."""
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cached = self._text_embed_cache.get(prompt)
        if cached is not None:
            return cached
        cache_dir = self.text_embedding_cache_dir
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute/text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        if not bool(context_mask.all()):
            context = context.clone()
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
        self._text_embed_cache[prompt] = (context, context_mask)
        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            if self.vae_latent_cache_dir is not None or self.video_only:
                raise
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
