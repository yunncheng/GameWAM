"""Precompute VAE latent mmap caches.

Records are keyed by physical dataset/episode/frame identity, so train/val
subsets and dataset order changes can reuse the same cache when their windows
are present.
"""

import importlib.util
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Subset, get_worker_info
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gamewam.datasets.game_video_dataset import GameVideoDataset
from gamewam.datasets.utils.cache_utils import compute_vae_cache_hash, resolve_image_video_size
from gamewam.models.wan22.helpers.loader import (
    _load_registered_model,
    _resolve_configs,
)
from gamewam.utils.cfg import register_default_resolvers
from gamewam.utils.logging_config import get_logger, setup_logging
from utils import (
    TorchcodecDecoderCache,
    apply_image_transforms,
    atomic_write_json,
    build_overlap_blocks,
    build_append_only_cache_mapping,
    build_unique_cache_mapping,
    distributed_barrier,
    init_gloo_distributed,
    iter_split_aware_dataset_nodes,
    rank_slice,
    to_bool,
    validate_existing_bin_size,
)

from gamewam.datasets.materialize import _ensure_container

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_DECODE_BLOCK_SIZE = 512
DEFAULT_VAE_BATCH_SIZE = 64
DEFAULT_VAE_ENCODE_MODE = "batch_fast"
DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_FACTOR = 2


def _resolve_num_workers(cfg: DictConfig, world_size: int) -> int:
    configured = cfg.get("num_workers")
    if configured is not None:
        return int(configured)
    cpu_count = os.cpu_count() or 1
    return max(2, min(DEFAULT_NUM_WORKERS, cpu_count // max(world_size, 1) // 2))


def _resolve_cache_dtype(mixed_precision: Any) -> tuple[str, torch.dtype, int]:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key in {"no", "fp32", "float32"}:
        return "float32", torch.float32, 4
    if key in {"fp16", "float16", "half"}:
        return "float16", torch.float16, 2
    if key in {"bf16", "bfloat16"}:
        return "bfloat16", torch.bfloat16, 2
    raise ValueError(
        f"Unsupported mixed_precision/cache dtype: {mixed_precision}. "
        "Expected one of: ['no', 'fp16', 'bf16']."
    )


def _resolve_vae_encode_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"`vae_encode_mode` must be str, got {type(value)}")
    mode = value.strip().lower()
    if mode not in {"batch_fast", "exact"}:
        raise ValueError(
            f"Unsupported vae_encode_mode: {value}. Expected one of: ['batch_fast', 'exact']."
        )
    return mode


def _node_int(node: Any, key: str, default: int) -> int:
    value = node.get(key, default)
    return int(default if value is None else value)


def _node_bool(node: Any, key: str, default: bool = False) -> bool:
    value = node.get(key, default)
    return bool(default if value is None else value)


def _resolve_read_num_frames(node: Any, ar_mode: bool) -> int:
    num_frames = _node_int(node, "num_frames", 17)
    num_obs_frames = _node_int(node, "num_obs_frames", 1)
    prediction_horizon = _node_int(node, "prediction_horizon", 16)
    execution_horizon = _node_int(node, "execution_horizon", prediction_horizon)
    frame_stride = _node_int(node, "frame_stride", 1)
    if prediction_horizon <= 0:
        raise ValueError(f"prediction_horizon must be positive, got {prediction_horizon}")
    if execution_horizon <= 0:
        raise ValueError(f"execution_horizon must be positive, got {execution_horizon}")
    if execution_horizon > prediction_horizon:
        raise ValueError(
            "execution_horizon must not exceed prediction_horizon, "
            f"got {execution_horizon} > {prediction_horizon}"
        )
    if not ar_mode:
        committed_future_horizon = (num_frames - num_obs_frames) * frame_stride
        if prediction_horizon != committed_future_horizon:
            raise ValueError(
                "Non-AR geometry must satisfy "
                "prediction_horizon == (num_frames - num_obs_frames) * frame_stride, "
                f"got {prediction_horizon} != {committed_future_horizon}"
            )
        return num_frames
    lookahead = prediction_horizon - execution_horizon
    if lookahead <= 0:
        return num_frames
    if prediction_horizon % execution_horizon != 0:
        raise ValueError(
            "AR prediction_horizon must be divisible by execution_horizon, "
            f"got {prediction_horizon} and {execution_horizon}"
        )
    if lookahead % frame_stride != 0:
        raise ValueError(
            "AR VAE read window must align with frame_stride: "
            f"prediction_horizon={prediction_horizon}, "
            f"execution_horizon={execution_horizon}, frame_stride={frame_stride}"
        )
    return num_frames + lookahead // frame_stride


def _collect_dataset_jobs(
    data_cfg: DictConfig,
    latent_dtype: str,
    ar_mode: bool,
):
    """Collect one job per unique (dataset_dirs, config_hash) combination."""
    results = []
    seen = set()
    for node_path, node in iter_split_aware_dataset_nodes(data_cfg, split="train"):
        node = _ensure_container(node)
        node["ar_mode"] = bool(ar_mode)
        cache_dir = node.get("vae_latent_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `vae_latent_cache_dir` for dataset node `{node_path}`. "
                "Please add it to your data config."
            )
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        shape_meta_dict = node.get("shape_meta")
        if OmegaConf.is_config(shape_meta_dict):
            shape_meta = OmegaConf.to_container(shape_meta_dict, resolve=True)
        else:
            shape_meta = shape_meta_dict
        video_size = resolve_image_video_size(shape_meta)
        num_frames = _resolve_read_num_frames(node, ar_mode=node.get("ar_mode"))
        frame_stride = _node_int(node, "frame_stride", 1)
        sample_interval = _node_int(node, "sample_interval", 4)
        action_video_freq_ratio = _node_int(node, "action_video_freq_ratio", 1)
        event_anchor_enabled = _node_bool(node, "event_anchor_enabled", False)
        event_anchor_radius = node.get("event_anchor_radius")
        sample_interval_event = node.get("sample_interval_event")
        sample_interval_normal = node.get("sample_interval_normal")

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

        dataset_dirs = [str(d) for d in raw_dirs]
        dedup_key = (tuple(dataset_dirs), config_hash)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        results.append({
            "node_path": node_path,
            "node": node,
            "cache_dir": str(cache_dir),
            "config_hash": config_hash,
        })

    return results


def _bin_filename(config_hash: str) -> str:
    return f"{config_hash}.latents.bin"


def _idx_filename(config_hash: str) -> str:
    return f"{config_hash}.latents.idx"


def _ensure_tchw_frames(frames: torch.Tensor) -> torch.Tensor:
    """Normalize decoded video frames to [T, C, H, W] uint8."""
    if frames.ndim != 4:
        raise ValueError(f"Expected decoded frames to be 4D, got {tuple(frames.shape)}")
    if frames.shape[1] in {1, 3, 4}:
        out = frames
    elif frames.shape[-1] in {1, 3, 4}:
        out = frames.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "Cannot infer decoded frame layout; expected [T,C,H,W] or [T,H,W,C], "
            f"got {tuple(frames.shape)}"
        )
    return out.contiguous()

class _OverlapWindowIterableDataset(IterableDataset):
    """Decode overlap blocks and return raw uint8 frames."""

    def __init__(self, dataset: GameVideoDataset, blocks: list[dict[str, Any]]):
        self.dataset = dataset
        self.blocks = blocks
        frame_stride = int(dataset.game_dataset.frame_stride)
        read_num_frames = int(getattr(dataset, "read_num_frames", dataset.num_frames))
        self.window_offsets = [i * frame_stride for i in range(read_num_frames)]
        self.output_frame_positions = [int(i) for i in dataset.video_sample_indices]

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            blocks = self.blocks
        else:
            chunk = (len(self.blocks) + worker.num_workers - 1) // worker.num_workers
            start = worker.id * chunk
            blocks = self.blocks[start:start + chunk]
        decoder_cache = TorchcodecDecoderCache()
        for block in blocks:
            yield self._load_block(block, decoder_cache)

    def _decode_union_frames(
        self,
        decoder_cache: TorchcodecDecoderCache,
        video_path: Path,
        frame_indices: list[int],
    ) -> torch.Tensor:
        if not frame_indices:
            raise ValueError("frame_indices must not be empty.")
        decoder = decoder_cache.get(video_path)
        frames = decoder.get_frames_at(indices=frame_indices).data
        if frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)
        return frames

    def _load_block(self, block: dict[str, Any], decoder_cache: TorchcodecDecoderCache) -> dict[str, Any]:
        game_dataset = self.dataset.game_dataset

        source_dataset = game_dataset.multi_dataset._datasets[block["dataset_idx"]]
        ep_len = int(block["ep_to"] - block["ep_from"])
        decoded_by_key = {}
        frame_pos_by_key = {}

        for meta in game_dataset.image_meta:
            image_key = meta["key"]
            video_key = meta["lerobot_key"]
            required_frames = sorted({
                max(0, min(ep_len - 1, int(item["frame_idx"]) + offset))
                for item in block["items"]
                for offset in self.window_offsets
            })
            needed = list(range(required_frames[0], required_frames[-1] + 1))
            video_path = source_dataset.root / source_dataset.meta.get_video_file_path(block["episode_idx"], video_key)
            decoded_by_key[image_key] = self._decode_union_frames(decoder_cache, video_path, needed)
            frame_pos_by_key[image_key] = {frame_idx: pos for pos, frame_idx in enumerate(needed)}

        videos = []
        idxs = []
        cache_keys = []
        for item in block["items"]:
            per_camera = []
            for meta in game_dataset.image_meta:
                image_key = meta["key"]
                frame_to_pos = frame_pos_by_key[image_key]
                positions = [
                    frame_to_pos[max(0, min(ep_len - 1, int(item["frame_idx"]) + offset))]
                    for offset in self.window_offsets
                ]
                frames = decoded_by_key[image_key][positions]
                frames = frames[self.output_frame_positions]
                frames = _ensure_tchw_frames(frames)
                per_camera.append(frames)

            videos.append(torch.stack(per_camera, dim=0))
            idxs.append(int(item["sample_start"]))
            cache_keys.append(str(item["cache_key"]))

        return {
            "pixel_values": torch.stack(videos, dim=0),
            "idx": torch.tensor(idxs, dtype=torch.long),
            "vae_cache_key": cache_keys,
        }


def _assemble_processed_videos(dataset: GameVideoDataset, pixel_values: torch.Tensor) -> torch.Tensor:
    """Assemble cameras and normalize to [B, C, T, H, W]."""
    processor = dataset.game_dataset.processor
    if processor is None:
        raise RuntimeError("Overlap-aware VAE precompute requires an attached processor.")

    bsz = int(pixel_values.shape[0])
    num_output_cameras = int(processor.num_output_cameras)
    if num_output_cameras > pixel_values.shape[1]:
        out = torch.zeros(
            (bsz, num_output_cameras) + tuple(pixel_values.shape[2:]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        out[:, :pixel_values.shape[1]] = pixel_values
        pixel_values = out
    elif num_output_cameras < pixel_values.shape[1]:
        pixel_values = pixel_values[:, :num_output_cameras]

    num_cameras = int(pixel_values.shape[1])
    if num_cameras > 1:
        if dataset.concat_multi_camera == "horizontal":
            video = torch.cat([pixel_values[:, i] for i in range(num_cameras)], dim=-1)
        elif dataset.concat_multi_camera == "vertical":
            video = torch.cat([pixel_values[:, i] for i in range(num_cameras)], dim=-2)
        else:
            raise ValueError(
                f"Invalid concat_multi_camera: {dataset.concat_multi_camera}. "
                "Expected one of: horizontal, vertical."
            )
    else:
        video = pixel_values.squeeze(1)

    target_h, target_w = int(dataset.video_size[0]), int(dataset.video_size[1])
    if tuple(video.shape[-2:]) != (target_h, target_w):
        raise ValueError(
            "Processor image transforms must produce `video_size` before GameVideoDataset finalization: "
            f"got spatial={tuple(video.shape[-2:])}, expected={(target_h, target_w)}."
        )
    video = (video.to(torch.float32) - 0.5) / 0.5
    return video.permute(0, 2, 1, 3, 4).contiguous()


def _prepare_videos_fast(
    dataset: GameVideoDataset,
    raw_pixel_values: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Tensor/GPU path for resize and normalization."""
    if raw_pixel_values.ndim != 6:
        raise ValueError(f"Expected raw pixel shape [B,N,T,C,H,W], got {tuple(raw_pixel_values.shape)}")

    bsz, num_cameras, t_video, c, h, w = raw_pixel_values.shape
    target_h, target_w = int(dataset.video_size[0]), int(dataset.video_size[1])
    pixel_values = raw_pixel_values.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
    if (h, w) != (target_h, target_w):
        flat = pixel_values.reshape(bsz * num_cameras * t_video, c, h, w)
        flat = F.interpolate(
            flat,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixel_values = flat.reshape(bsz, num_cameras, t_video, c, target_h, target_w)
    return _assemble_processed_videos(dataset, pixel_values)


def _prepare_videos_with_processor(dataset: GameVideoDataset, raw_pixel_values: torch.Tensor) -> torch.Tensor:
    """CPU processor path for compatibility checks."""
    if raw_pixel_values.ndim != 6:
        raise ValueError(f"Expected raw pixel shape [B,N,T,C,H,W], got {tuple(raw_pixel_values.shape)}")

    bsz, num_cameras, t_video, c, h, w = raw_pixel_values.shape
    processed_images = []
    for camera_idx, meta in enumerate(dataset.game_dataset.image_meta):
        if camera_idx >= num_cameras:
            break
        image_key = meta["key"]
        flat = raw_pixel_values[:, camera_idx].reshape(bsz * t_video, c, h, w)
        image = apply_image_transforms(dataset.game_dataset.processor, image_key, flat)
        if image.ndim != 4:
            raise ValueError(
                f"Image transforms for key {image_key!r} must return [B*T,C,H,W], got {tuple(image.shape)}"
            )
        image = image.reshape(bsz, t_video, image.shape[-3], image.shape[-2], image.shape[-1])
        processed_images.append(image)

    if not processed_images:
        raise RuntimeError("No image streams were available for VAE precompute.")

    return _assemble_processed_videos(dataset, torch.stack(processed_images, dim=1))


def _compute_far_latent_cache(
    node,
    dataset,
    vae,
    device,
    torch_dtype,
    latent_dtype_name,
    latent_bytes_per_element,
    is_distributed,
    rank,
    world_size,
    overwrite,
    vae_batch_size,
    num_workers,
    prefetch_factor,
):
    """Compute per-episode-frame far latent cache (single-frame VAE encode).

    Each video frame in each episode is independently encoded by VAE to produce
    a single-frame latent [z_dim, 1, H_lat, W_lat]. These are stored contiguously
    and indexed by "{dataset}::episode={ep}::video_frame={idx}".
    """
    from gamewam.datasets.utils.cache_utils import compute_far_latent_cache_hash, resolve_image_video_size

    shape_meta = _ensure_container(node.get("shape_meta"))
    video_size = resolve_image_video_size(shape_meta)
    action_video_freq_ratio = int(node.get("action_video_freq_ratio", 1))

    far_config_hash = compute_far_latent_cache_hash(
        video_size=video_size,
        action_video_freq_ratio=action_video_freq_ratio,
        latent_dtype=latent_dtype_name,
    )

    far_cache_dir = Path(str(node.get("far_latent_cache_dir")))
    far_bin_path = far_cache_dir / f"{far_config_hash}.far_latents.bin"
    far_idx_path = far_cache_dir / f"{far_config_hash}.far_latents.idx"

    if far_bin_path.exists() and far_idx_path.exists() and not overwrite:
        logger.info("[far-%s] Far latent cache already exists, skipping.", far_config_hash)
        return

    logger.info("[far-%s] Computing per-episode-frame far latent cache...", far_config_hash)

    dataset_dirs = _ensure_container(node.get("dataset_dirs"))

    from gamewam.datasets.base_game_dataset import BaseGameDataset
    base_ds = BaseGameDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        obs_size=int(node.get("num_frames", 33)),
        action_size=int(node.get("num_frames", 33)) - 1,
        val_set_proportion=0.0,
        is_training_set=True,
        frame_stride=int(node.get("frame_stride", 1)),
        sample_interval=int(node.get("sample_interval", 8)),
        event_anchor_enabled=bool(node.get("event_anchor_enabled", False)),
        event_anchor_radius=node.get("event_anchor_radius"),
        sample_interval_event=node.get("sample_interval_event"),
        sample_interval_normal=node.get("sample_interval_normal"),
        video_only=True,
    )

    all_far_keys = []
    for ds_idx, sub_dataset in enumerate(base_ds.multi_dataset._datasets):
        repo_id = sub_dataset.repo_id
        sub_episodes = sub_dataset.episodes if sub_dataset.episodes is not None else list(range(sub_dataset.meta.total_episodes))
        ep_from = sub_dataset.episode_data_index["from"]
        ep_to = sub_dataset.episode_data_index["to"]

        for local_ep_pos, ep_idx in enumerate(sub_episodes):
            ep_len = int(ep_to[local_ep_pos].item()) - int(ep_from[local_ep_pos].item())
            num_video_frames_in_ep = ep_len // action_video_freq_ratio
            for vf_idx in range(num_video_frames_in_ep):
                key = f"{repo_id}::episode={ep_idx}::video_frame={vf_idx}"
                all_far_keys.append({
                    "key": key,
                    "ds_idx": ds_idx,
                    "episode_idx": ep_idx,
                    "local_ep_pos": local_ep_pos,
                    "vf_idx": vf_idx,
                    "raw_frame_idx": vf_idx * action_video_freq_ratio,
                    "ep_from": int(ep_from[local_ep_pos].item()),
                })

    num_total_frames = len(all_far_keys)
    logger.info("[far-%s] Total video frames to encode: %d", far_config_hash, num_total_frames)

    H, W = video_size
    z_dim = vae.z_dim
    h_lat = H // vae.upsampling_factor
    w_lat = W // vae.upsampling_factor
    far_latent_shape = (z_dim, 1, h_lat, w_lat)
    elements_per_frame = z_dim * 1 * h_lat * w_lat
    bytes_per_frame = elements_per_frame * latent_bytes_per_element

    os.makedirs(str(far_cache_dir), exist_ok=True)
    total_bin_size = num_total_frames * bytes_per_frame

    if (not is_distributed) or rank == 0:
        logger.info("[far-%s] Pre-allocating far bin: %.2f GB (%d frames)",
                    far_config_hash, total_bin_size / 1e9, num_total_frames)
        with open(far_bin_path, "wb") as f:
            f.truncate(total_bin_size)
    distributed_barrier(is_distributed)

    far_mapping = {item["key"]: idx for idx, item in enumerate(all_far_keys)}

    chunk_start, chunk_end = rank_slice(num_total_frames, rank, world_size)
    my_items = all_far_keys[chunk_start:chunk_end]

    logger.info("[far-%s] Rank %d/%d: encoding %d frames (slice=%d:%d)",
                far_config_hash, rank, world_size, len(my_items), chunk_start, chunk_end)

    written = 0

    has_torchcodec = False
    try:
        from torchcodec.decoders import VideoDecoder
        has_torchcodec = True
    except ImportError:
        pass

    from collections import defaultdict
    ep_groups = defaultdict(list)
    for item in my_items:
        ep_key = (item["ds_idx"], item["episode_idx"], item["local_ep_pos"])
        ep_groups[ep_key].append(item)

    with open(far_bin_path, "r+b") as bin_f:
        with tqdm(
            total=len(my_items),
            desc=f"[far-{far_config_hash}] rank={rank}",
            unit="frame",
            dynamic_ncols=True,
            disable=is_distributed and rank != 0,
        ) as pbar:
            for (ds_idx, ep_idx, local_ep_pos), items in ep_groups.items():
                sub_dataset = base_ds.multi_dataset._datasets[ds_idx]
                video_key = sub_dataset.meta.video_keys[0] if sub_dataset.meta.video_keys else "observation.images"
                video_path = sub_dataset.root / sub_dataset.meta.get_video_file_path(ep_idx, video_key)

                raw_frame_indices = sorted(set(item["raw_frame_idx"] for item in items))

                if has_torchcodec:
                    decoder = VideoDecoder(str(video_path))
                    frames_data = decoder.get_frames_at(indices=raw_frame_indices).data
                    if frames_data.shape[-1] == 3:
                        frames_data = frames_data.permute(0, 3, 1, 2)
                else:
                    import torchvision
                    reader = torchvision.io.VideoReader(str(video_path), "video")
                    all_frames = []
                    for frame_dict in reader:
                        all_frames.append(frame_dict["data"])
                    frames_tensor = torch.stack(all_frames)
                    frames_data = frames_tensor[raw_frame_indices]

                idx_to_pos = {fi: pos for pos, fi in enumerate(raw_frame_indices)}

                import torchvision.transforms.functional as TF
                target_h, target_w = video_size

                batch_frames = []
                batch_record_indices = []
                for item in items:
                    pos = idx_to_pos[item["raw_frame_idx"]]
                    frame = frames_data[pos].float() / 255.0
                    frame = TF.resize(frame, [target_h, target_w], antialias=True)
                    frame = frame * 2.0 - 1.0
                    frame = frame.unsqueeze(1)
                    batch_frames.append(frame)
                    batch_record_indices.append(far_mapping[item["key"]])

                    if len(batch_frames) >= vae_batch_size or item is items[-1]:
                        batch_tensor = torch.stack(batch_frames, dim=0).to(dtype=torch_dtype, device=device)
                        with torch.inference_mode():
                            batch_latents = vae.model.encode(batch_tensor, vae.scale)
                        batch_latents_cpu = batch_latents.to(dtype=torch_dtype, device="cpu").contiguous()

                        for i, record_idx in enumerate(batch_record_indices):
                            latent_bytes = batch_latents_cpu[i].view(torch.uint8).numpy().tobytes()
                            bin_f.seek(record_idx * bytes_per_frame)
                            bin_f.write(latent_bytes)
                            written += 1

                        pbar.update(len(batch_frames))
                        batch_frames = []
                        batch_record_indices = []

    logger.info("[far-%s] Rank %d: wrote %d frames", far_config_hash, rank, written)
    distributed_barrier(is_distributed)

    if (not is_distributed) or rank == 0:
        index = {
            "latent_shape": list(far_latent_shape),
            "dtype": latent_dtype_name,
            "bytes_per_element": latent_bytes_per_element,
            "bytes_per_frame": bytes_per_frame,
            "config_hash": far_config_hash,
            "action_video_freq_ratio": action_video_freq_ratio,
            "num_records": num_total_frames,
            "key_format": "{dataset}::episode={episode_index}::video_frame={frame_index}",
            "mapping": far_mapping,
        }
        atomic_write_json(far_idx_path, index)
        logger.info("[far-%s] Wrote far index: %s (%d records)", far_config_hash, far_idx_path, num_total_frames)

    distributed_barrier(is_distributed)
    del base_ds


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = init_gloo_distributed(timeout=timedelta(hours=24))
    if is_distributed and rank == 0:
        logger.info("Distributed enabled with gloo control barriers: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute/vae_latents.py",
            torch.cuda.device_count(),
        )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    overwrite = to_bool(cfg.get("overwrite", False))
    vae_batch_size = int(cfg.get("vae_batch_size", DEFAULT_VAE_BATCH_SIZE))
    decode_block_size = int(cfg.get("decode_block_size", DEFAULT_DECODE_BLOCK_SIZE))
    if vae_batch_size < 1:
        raise ValueError(f"vae_batch_size must be >= 1, got {vae_batch_size}")
    if decode_block_size < 1:
        raise ValueError(f"decode_block_size must be >= 1, got {decode_block_size}")
    num_workers = _resolve_num_workers(cfg, world_size)
    prefetch_factor = int(cfg.get("prefetch_factor", DEFAULT_PREFETCH_FACTOR))
    vae_encode_mode = _resolve_vae_encode_mode(str(cfg.get("vae_encode_mode", DEFAULT_VAE_ENCODE_MODE)))
    fast_image_transforms = to_bool(cfg.get("fast_image_transforms", True))
    latent_dtype_name, torch_dtype, latent_bytes_per_element = _resolve_cache_dtype(
        str(cfg.get("mixed_precision", "bf16"))
    )

    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")
    if cfg.model is None:
        raise ValueError("`cfg.model` is required.")

    ar_mode = (
        str(OmegaConf.select(
            cfg, "model.video_dit_config.video_attention_mask_mode"
        ))
        == "ar_teacher_forcing"
    )
    dataset_jobs = _collect_dataset_jobs(
        cfg.data,
        latent_dtype=latent_dtype_name,
        ar_mode=ar_mode,
    )
    if not dataset_jobs:
        raise ValueError("No dataset nodes with `vae_latent_cache_dir` found.")

    model_cfg = cfg.model
    model_id = str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    logger.info("Using VAE/cache dtype from mixed_precision=%s: %s", cfg.get("mixed_precision", "bf16"), latent_dtype_name)

    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    logger.info("VAE loaded on %s (z_dim=%d, upsampling_factor=%d, temporal_factor=%d)",
                device, vae.z_dim, vae.upsampling_factor, vae.temporal_downsample_factor)

    for job in dataset_jobs:
        config_hash = job["config_hash"]
        cache_dir = job["cache_dir"]
        node = job["node"]

        bin_path = Path(cache_dir) / _bin_filename(config_hash)
        idx_path = Path(cache_dir) / _idx_filename(config_hash)

        existing_mapping: dict[str, int] = {}
        if bin_path.exists() and idx_path.exists() and not overwrite:
            with open(idx_path, "r", encoding="utf-8") as _f:
                _old_idx = json.load(_f)
            existing_mapping = _old_idx.get("mapping", {})
            logger.info(
                "[%s] Existing cache found: %d records. Will skip already-computed keys.",
                config_hash, len(existing_mapping),
            )
        elif not overwrite and (bin_path.exists() != idx_path.exists()):
            logger.warning("[%s] Partial cache (bin/idx mismatch); will recompute from scratch.", config_hash)

        if vae_encode_mode == "batch_fast":
            logger.warning(
                "[%s] Using batch_fast VAE encode: faster but not bit-exact with online per-sample VAE encode.",
                config_hash,
            )
        else:
            logger.info("[%s] Using exact VAE encode matching online training path.", config_hash)

        logger.info("[%s] Building GameVideoDataset (all episodes, video_only)...", config_hash)

        dataset = GameVideoDataset(
            dataset_dirs=_ensure_container(node.get("dataset_dirs")),
            shape_meta=_ensure_container(node.get("shape_meta")),
            num_frames=_node_int(node, "num_frames", 17),
            processor=node.get("processor"),
            text_embedding_cache_dir=None,
            context_len=_node_int(node, "context_len", 128),
            val_set_proportion=0.0,
            is_training_set=True,
            frame_stride=_node_int(node, "frame_stride", 1),
            sample_interval=_node_int(node, "sample_interval", 4),
            action_video_freq_ratio=_node_int(node, "action_video_freq_ratio", 1),
            num_obs_frames=_node_int(node, "num_obs_frames", 1),
            skip_padding_as_possible=False,
            concat_multi_camera=node.get("concat_multi_camera"),
            event_anchor_enabled=_node_bool(node, "event_anchor_enabled", False),
            event_anchor_radius=node.get("event_anchor_radius"),
            sample_interval_event=node.get("sample_interval_event"),
            sample_interval_normal=node.get("sample_interval_normal"),
            vae_latent_cache_dir=None,
            video_only=True,
            prompt_template=str(node.get("prompt_template")),
            history=node.get("history"),
            prediction_horizon=_node_int(node, "prediction_horizon", 16),
            execution_horizon=_node_int(node, "execution_horizon", _node_int(node, "prediction_horizon", 16)),
            ar_mode=_node_bool(node, "ar_mode", False),
        )

        unique_indices, sorted_cache_keys, cache_key_to_record = build_unique_cache_mapping(dataset)

        history_cfg = node.get("history")
        if history_cfg is not None:
            from gamewam.models.history.history_module import (
                compute_history_starts,
                derive_memory_write_capacity_segments,
                derive_history_geometry,
            )
            num_frames_cfg = int(node.get("num_frames", 33))
            num_obs_frames_cfg = int(node.get("num_obs_frames", 17))
            frame_stride_cfg = int(node.get("frame_stride", 1))
            action_video_freq_ratio_cfg = int(node.get("action_video_freq_ratio", 1))
            geom = derive_history_geometry(
                num_frames=num_frames_cfg,
                num_obs_frames=num_obs_frames_cfg,
                frame_stride=frame_stride_cfg,
                action_video_freq_ratio=action_video_freq_ratio_cfg,
            )
            cycle_stride_raw = geom.cycle_stride_raw
            recent_num = int(history_cfg.get("recent_num_segments", 2))
            memory_write_capacity = derive_memory_write_capacity_segments(history_cfg)
            max_history_segments = recent_num + memory_write_capacity
            obs_context_raw_span = (num_obs_frames_cfg - 1) * frame_stride_cfg

            game_ds = dataset.game_dataset
            sample_starts = [int(s) for s in game_ds.sample_starts]
            existing_starts_set = set(sample_starts)

            all_history_positions = set()
            for ss in sample_starts:
                cache_key = game_ds.get_vae_cache_key(ss)
                parts = cache_key.split("::")
                window_start_raw = int(parts[2].split("=")[1])
                current_raw = window_start_raw + obs_context_raw_span
                ep_global_base = ss - window_start_raw
                history_starts = compute_history_starts(current_raw, 0, cycle_stride_raw)
                history_starts = history_starts[-max_history_segments:]
                for hs in history_starts:
                    global_pos = ep_global_base + hs
                    if global_pos not in existing_starts_set:
                        all_history_positions.add(global_pos)

            if all_history_positions:
                new_positions = sorted(all_history_positions)
                original_len = len(game_ds.sample_starts)
                old_positions = [int(s) for s in game_ds.sample_starts]
                game_ds.sample_starts = old_positions + [int(p) for p in new_positions]

                logger.info(
                    "[%s] AR-history: injected %d history positions into sample_starts "
                    "(total: %d → %d). Rebuilding cache mapping.",
                    config_hash, len(new_positions), original_len, len(game_ds.sample_starts),
                )
                unique_indices, sorted_cache_keys, cache_key_to_record = build_unique_cache_mapping(dataset)
            else:
                logger.info(
                    "[%s] AR-history: all history positions already covered by sample_starts.",
                    config_hash,
                )

        cache_key_to_record, missing_cache_keys = build_append_only_cache_mapping(
            desired_cache_keys=sorted_cache_keys,
            existing_mapping=existing_mapping if existing_mapping and not overwrite else None,
        )
        missing_cache_key_set = set(missing_cache_keys)
        if cache_key_to_record:
            num_records = max(cache_key_to_record.values()) + 1
        else:
            num_records = 0
        total_samples = len(dataset)
        logger.info(
            "[%s] Dataset length: %d, desired physical samples: %d, cache records: %d, missing: %d",
            config_hash, total_samples, len(sorted_cache_keys), num_records, len(missing_cache_keys),
        )

        video_size = dataset.video_size
        num_video_frames = len(dataset.video_sample_indices)
        dummy_video = torch.randn(1, 3, num_video_frames, video_size[0], video_size[1], dtype=torch_dtype)
        with torch.no_grad():
            dummy_latent = vae.encode(dummy_video, device=device)
        latent_shape = tuple(dummy_latent.shape[1:])
        elements_per_sample = 1
        for d in latent_shape:
            elements_per_sample *= d
        bytes_per_sample = elements_per_sample * latent_bytes_per_element
        logger.info("[%s] Latent shape: %s, bytes_per_sample: %d",
                    config_hash, latent_shape, bytes_per_sample)
        del dummy_video, dummy_latent

        total_bin_size = num_records * bytes_per_sample
        if num_records == 0:
            raise RuntimeError(f"[{config_hash}] No unique physical samples found; cannot build latent cache.")
        os.makedirs(cache_dir, exist_ok=True)
        if existing_mapping and not overwrite:
            validate_existing_bin_size(
                bin_path=bin_path,
                existing_mapping=existing_mapping,
                bytes_per_sample=bytes_per_sample,
            )

        if (not is_distributed) or rank == 0:
            if existing_mapping and not overwrite:
                logger.info(
                    "[%s] Expanding bin from %d to %d records (%.2f GB).",
                    config_hash, len(existing_mapping), num_records, total_bin_size / 1e9,
                )
                with open(bin_path, "r+b") as f:
                    f.seek(0, 2)  # seek to end
                    current_size = f.tell()
                    if total_bin_size > current_size:
                        f.truncate(total_bin_size)
            else:
                logger.info("[%s] Pre-allocating bin file: %.2f GB (%d records)",
                            config_hash, total_bin_size / 1e9, num_records)
                with open(bin_path, "wb") as f:
                    f.truncate(total_bin_size)

        distributed_barrier(is_distributed)

        if existing_mapping and not overwrite:
            new_indices = [
                idx for idx in unique_indices
                if dataset.game_dataset.get_vae_cache_key(
                    int(dataset.game_dataset.sample_starts[idx])
                ) in missing_cache_key_set
            ]
            skipped = len(unique_indices) - len(new_indices)
            logger.info("[%s] Skipping %d already-cached records, encoding %d new.",
                        config_hash, skipped, len(new_indices))
            unique_indices = new_indices

        chunk_start, chunk_end = rank_slice(len(unique_indices), rank, world_size)
        my_indices = unique_indices[chunk_start:chunk_end]

        logger.info(
            "[%s] Rank %d/%d: %d unique samples (slice=%d:%d, num_workers=%d, "
            "vae_batch_size=%d, decode_block_size=%d)",
            config_hash, rank, world_size, len(my_indices), chunk_start, chunk_end,
            num_workers, vae_batch_size, decode_block_size,
        )

        if importlib.util.find_spec("torchcodec") is not None:
            blocks = build_overlap_blocks(dataset, my_indices, decode_block_size)
            logger.info(
                "[%s] Rank %d: overlap-aware decode enabled: %d sample blocks, "
                "decoder cache is per worker.",
                config_hash, rank, len(blocks),
            )
            loader = DataLoader(
                _OverlapWindowIterableDataset(dataset, blocks),
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
            )
        else:
            logger.info("[%s] torchcodec unavailable; falling back to per-sample decode.", config_hash)

            def _vae_collate_fn(batch):
                """Custom collate that only stacks fields needed for VAE precompute."""
                videos = torch.stack([item["video"] for item in batch], dim=0)
                idxs = torch.tensor([int(item["idx"]) for item in batch], dtype=torch.long)
                cache_keys = [str(item["vae_cache_key"]) for item in batch]
                return {"video": videos, "idx": idxs, "vae_cache_key": cache_keys}

            subset = Subset(dataset, my_indices)
            loader = DataLoader(
                subset,
                batch_size=vae_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
                collate_fn=_vae_collate_fn,
            )

        written = 0

        with open(bin_path, "r+b") as bin_f:
            with tqdm(
                total=len(my_indices),
                desc=f"[{config_hash}] rank={rank}",
                unit="sample",
                dynamic_ncols=True,
                disable=is_distributed and rank != 0,
            ) as pbar:
                # Encode each block independently because larger accumulated
                # batches increase peak activation and resize memory without
                # improving GPU-bound VAE throughput.
                for batch in loader:
                    if "pixel_values" in batch:
                        if fast_image_transforms:
                            block_videos = _prepare_videos_fast(dataset, batch["pixel_values"], device=device)
                        else:
                            block_videos = _prepare_videos_with_processor(dataset, batch["pixel_values"])
                    else:
                        block_videos = batch["video"]
                    block_cache_keys = batch["vae_cache_key"]

                    for batch_start in range(0, int(block_videos.shape[0]), vae_batch_size):
                        videos = block_videos[batch_start:batch_start + vae_batch_size]
                        batch_cache_keys = block_cache_keys[batch_start:batch_start + vae_batch_size]

                        batch_tensor = videos.to(dtype=torch_dtype, device=device, non_blocking=True)
                        with torch.inference_mode():
                            if vae_encode_mode == "exact":
                                batch_latents = vae.encode(batch_tensor, device=device)
                            elif vae_encode_mode == "batch_fast":
                                batch_latents = vae.model.encode(batch_tensor, vae.scale)
                            else:
                                raise ValueError(f"Unsupported vae_encode_mode: {vae_encode_mode}")

                        record_indices = [cache_key_to_record[cache_key] for cache_key in batch_cache_keys]
                        batch_latents_cpu = batch_latents.to(dtype=torch_dtype, device="cpu").contiguous()
                        raw_bytes = batch_latents_cpu.view(torch.uint8).numpy().tobytes()

                        first_record = record_indices[0]
                        expected_records = list(range(first_record, first_record + len(record_indices)))
                        if record_indices == expected_records:
                            bin_f.seek(first_record * bytes_per_sample)
                            bin_f.write(raw_bytes)
                            written += len(record_indices)
                        else:
                            for i, record_idx in enumerate(record_indices):
                                start = i * bytes_per_sample
                                bin_f.seek(record_idx * bytes_per_sample)
                                bin_f.write(raw_bytes[start:start + bytes_per_sample])
                                written += 1
                        pbar.update(videos.shape[0])

        logger.info("[%s] Rank %d: wrote %d", config_hash, rank, written)

        distributed_barrier(is_distributed)

        if (not is_distributed) or rank == 0:
            index = {
                "latent_shape": list(latent_shape),
                "dtype": latent_dtype_name,
                "bytes_per_element": latent_bytes_per_element,
                "bytes_per_sample": bytes_per_sample,
                "config_hash": config_hash,
                "vae_encode_mode": vae_encode_mode,
                "dataset_dirs": _ensure_container(node.get("dataset_dirs")),
                "num_records": num_records,
                "key_format": "{dataset_dir}::episode={episode_index}::frame={frame_index}",
                "mapping": cache_key_to_record,
            }
            atomic_write_json(idx_path, index)
            logger.info("[%s] Wrote index: %s (%d records)", config_hash, idx_path, num_records)

        distributed_barrier(is_distributed)

        model_cfg = cfg.model
        video_dit_cfg = model_cfg.get("video_dit_config", {})
        is_ar = (str(video_dit_cfg.get("video_attention_mask_mode", "")).strip() == "ar_teacher_forcing")
        far_cache_dir_cfg = node.get("far_latent_cache_dir")
        has_history = node.get("history") is not None
        if is_ar and far_cache_dir_cfg and not has_history:
            _compute_far_latent_cache(
                node=node,
                dataset=dataset if 'dataset' in dir() else None,
                vae=vae,
                device=device,
                torch_dtype=torch_dtype,
                latent_dtype_name=latent_dtype_name,
                latent_bytes_per_element=latent_bytes_per_element,
                is_distributed=is_distributed,
                rank=rank,
                world_size=world_size,
                overwrite=overwrite,
                vae_batch_size=vae_batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

        del dataset

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

    logger.info("Done.")


if __name__ == "__main__":
    main()
