import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

try:
    from omegaconf import DictConfig, ListConfig
    _DICT_CONFIG_TYPES = (DictConfig,)
    _LIST_CONFIG_TYPES = (ListConfig,)
except ModuleNotFoundError:
    DictConfig = ListConfig = ()
    _DICT_CONFIG_TYPES = ()
    _LIST_CONFIG_TYPES = ()


def init_gloo_distributed(timeout=None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        kwargs = {"backend": "gloo", "init_method": "env://"}
        if timeout is not None:
            kwargs["timeout"] = timeout
        dist.init_process_group(**kwargs)
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def distributed_barrier(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()


def rank_slice(total: int, rank: int, world_size: int) -> tuple[int, int]:
    chunk = (int(total) + int(world_size) - 1) // int(world_size)
    start = int(rank) * chunk
    end = min(start + chunk, int(total))
    return start, end


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def iter_split_aware_dataset_nodes(cfg: Any, split: str = "train"):
    """Iterate over dataset nodes (common+train merged) for precompute.

    Delegates to materialize_dataset_nodes() — the single source of truth
    for both training and precompute. Supports regular and mixed configs.
    """
    if cfg is None:
        return
    # Some callers pass cfg, others pass cfg.data. Normalize.
    data_cfg = cfg["data"] if hasattr(cfg, "get") and cfg.get("data") is not None else cfg
    if not data_cfg:
        return

    try:
        from gamewam.datasets.materialize import materialize_dataset_nodes
    except ModuleNotFoundError:
        # Fallback: import the module via importlib
        import importlib
        try:
            materialize_dataset_nodes = importlib.import_module(
                "gamewam.datasets.materialize"
            ).materialize_dataset_nodes
        except ModuleNotFoundError:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "Cannot import materialize_dataset_nodes; "
                "iter_split_aware_dataset_nodes will return nothing."
            )
            return

    for node_path, node in materialize_dataset_nodes(data_cfg, split=split):
        yield node_path, node


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def build_unique_cache_mapping(dataset) -> tuple[list[int], list[str], dict[str, int]]:
    sample_starts = [int(s) for s in dataset.game_dataset.sample_starts]
    start_to_first_index = {}
    for sample_index, sample_start in enumerate(sample_starts):
        start_to_first_index.setdefault(sample_start, sample_index)
    sorted_starts = sorted(start_to_first_index)
    unique_indices = [start_to_first_index[s] for s in sorted_starts]
    sorted_cache_keys = [dataset.game_dataset.get_vae_cache_key(s) for s in sorted_starts]
    if len(set(sorted_cache_keys)) != len(sorted_cache_keys):
        raise RuntimeError("Duplicate cache keys detected.")
    return unique_indices, sorted_cache_keys, {key: i for i, key in enumerate(sorted_cache_keys)}


def build_append_only_cache_mapping(
    desired_cache_keys: list[str],
    existing_mapping: dict[str, int] | None,
) -> tuple[dict[str, int], list[str]]:
    """Build a stable mapping that preserves existing record indices.

    Existing cache records are immutable: new keys are appended after the
    highest existing record index so incremental precompute cannot silently
    reinterpret old latent bytes as different cache keys.
    """
    desired_unique = sorted(set(str(k) for k in desired_cache_keys))
    if existing_mapping is None:
        existing_mapping = {}

    cache_key_to_record = {str(k): int(v) for k, v in existing_mapping.items()}
    used_records = set(cache_key_to_record.values())
    if len(used_records) != len(cache_key_to_record):
        raise RuntimeError("Existing VAE cache mapping has duplicate record indices.")

    next_record = max(used_records) + 1 if used_records else 0
    missing_keys = []
    for key in desired_unique:
        if key not in cache_key_to_record:
            cache_key_to_record[key] = next_record
            missing_keys.append(key)
            next_record += 1

    return cache_key_to_record, missing_keys


def validate_existing_bin_size(
    bin_path: str | Path,
    existing_mapping: dict[str, int],
    bytes_per_sample: int,
) -> None:
    """Ensure existing bin contains every record referenced by existing mapping."""
    if not existing_mapping:
        return
    old_records = [int(v) for v in existing_mapping.values()]
    if not old_records:
        return

    min_expected_size = (max(old_records) + 1) * int(bytes_per_sample)
    actual_size = Path(bin_path).stat().st_size if Path(bin_path).exists() else 0
    if actual_size < min_expected_size:
        raise RuntimeError(
            "Existing VAE bin is smaller than existing mapping requires: "
            f"actual={actual_size}, expected_at_least={min_expected_size}. "
            "The cache is incomplete or corrupted. Re-run with overwrite=true."
        )


class TorchcodecDecoderCache:
    """LRU cache of torchcodec VideoDecoder objects, keyed by video path."""

    def __init__(self, max_size: int = 8):
        self.max_size = int(max_size)
        self._cache = OrderedDict()

    def get(self, video_path: Path):
        key = str(video_path)
        decoder = self._cache.get(key)
        if decoder is not None:
            self._cache.move_to_end(key)
            return decoder
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(key, device="cpu", seek_mode="approximate")
        self._cache[key] = decoder
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return decoder


def apply_image_transforms(processor: Any, image_key: str, image: torch.Tensor) -> torch.Tensor:
    transforms = processor.train_transforms if processor.is_train else processor.val_transforms
    current_transforms = transforms[image_key] if isinstance(transforms, dict) else transforms
    if current_transforms is not None:
        for trans in current_transforms:
            image = trans(image)
    return image


def locate_sample(game_dataset: Any, sample_start: int) -> dict[str, Any]:
    """Map a global sample_start offset to its dataset/episode/frame location."""
    local_idx = int(sample_start)
    dataset_offset = 0
    for dataset_idx, dataset in enumerate(game_dataset.multi_dataset._datasets):
        dataset_frames = int(dataset.num_frames)
        if local_idx < dataset_frames:
            ep_to = dataset.episode_data_index["to"]
            ep_pos = int(torch.searchsorted(
                ep_to,
                torch.tensor(local_idx, dtype=ep_to.dtype),
                right=True,
            ).item())
            if ep_pos >= len(ep_to):
                raise IndexError(f"sample_start={sample_start} maps outside dataset {dataset.repo_id}.")
            ep_from = int(dataset.episode_data_index["from"][ep_pos].item())
            ep_to_value = int(dataset.episode_data_index["to"][ep_pos].item())
            episode_idx = int(dataset.episodes[ep_pos]) if dataset.episodes is not None else ep_pos
            return {
                "dataset_idx": dataset_idx,
                "dataset": dataset,
                "dataset_offset": dataset_offset,
                "local_idx": local_idx,
                "ep_pos": ep_pos,
                "episode_idx": episode_idx,
                "ep_from": ep_from,
                "ep_to": ep_to_value,
                "frame_idx": local_idx - ep_from,
            }
        local_idx -= dataset_frames
        dataset_offset += dataset_frames
    raise IndexError(f"sample_start={sample_start} out of bounds.")


def build_overlap_blocks(
    dataset,
    sample_indices: list[int],
    batch_size: int,
    cache_key_fn=None,
) -> list[dict[str, Any]]:
    """Group sample indices into contiguous episode blocks for overlap-aware decoding.

    Args:
        dataset: GameVideoDataset instance.
        sample_indices: ordered list of unique sample indices to process.
        batch_size: max items per block before flushing.
        cache_key_fn: callable(game_dataset, sample_start) -> str. Defaults to get_vae_cache_key.
    """
    blocks = []
    cur_key = None
    cur_items = []
    game_dataset = dataset.game_dataset
    sample_starts = game_dataset.sample_starts
    if cache_key_fn is None:
        cache_key_fn = lambda gds, s: gds.get_vae_cache_key(s)

    def flush():
        nonlocal cur_items, cur_key
        if cur_items:
            first = cur_items[0]["loc"]
            blocks.append({
                "dataset_idx": first["dataset_idx"],
                "ep_pos": first["ep_pos"],
                "episode_idx": first["episode_idx"],
                "ep_from": first["ep_from"],
                "ep_to": first["ep_to"],
                "items": cur_items,
            })
            cur_items = []
            cur_key = None

    for sample_index in sample_indices:
        sample_start = int(sample_starts[int(sample_index)])
        loc = locate_sample(game_dataset, sample_start)
        key = (loc["dataset_idx"], loc["ep_pos"])
        if cur_key is None:
            cur_key = key
        if key != cur_key or len(cur_items) >= batch_size:
            flush()
            cur_key = key
        cur_items.append({
            "sample_start": sample_start,
            "cache_key": cache_key_fn(game_dataset, sample_start),
            "frame_idx": loc["frame_idx"],
            "local_idx": loc["local_idx"],
            "loc": loc,
        })
    flush()
    return blocks
