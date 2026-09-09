"""MixedGameVideoDataset — combines multiple GameVideoDataset children with
global normalizer and weighted sampling.
"""

import bisect
import logging
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from .game_video_dataset import GameVideoDataset
from .base_game_dataset import (
    _all_gather_tensor_list,
    _build_episode_route_window,
    _quantile_dims_for_meta,
)
from .utils.normalizer import (
    LinearNormalizer,
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from gamewam.utils import misc
from .materialize import _ensure_container

logger = logging.getLogger(__name__)
DEFAULT_MIXED_NORM_STATS_WORKERS = 8
MIXED_NORM_STATS_PENDING_FACTOR = 4


def _compute_local_stats(
    tensor_list: List[torch.Tensor],
    *,
    is_stepwise_action: bool,
    quantile_dims: Optional[List[int]],
) -> Optional[Dict[str, torch.Tensor]]:
    if not tensor_list:
        return None

    stacked = torch.cat(tensor_list, dim=0)
    if not is_stepwise_action:
        stacked = stacked.reshape(-1, stacked.shape[-1])

    data = stacked.double()
    count = torch.tensor([stacked.shape[0]], dtype=torch.float64)
    # Round-robin empty list keeps the collectives balanced across ranks —
    # every rank must contribute to the same number of all_gathers.
    stats = {
        "min": stacked.min(dim=0).values,
        "max": stacked.max(dim=0).values,
        "sum": data.sum(dim=0),
        "sumsq": (data * data).sum(dim=0),
        "count": count,
    }
    if quantile_dims:
        dims = torch.as_tensor(quantile_dims, dtype=torch.long, device=stacked.device)
        stats["quantile_data"] = stacked.index_select(-1, dims)
    return stats



class _StreamingStats:
    """Incremental sufficient statistics with optional exact quantile columns."""

    def __init__(
        self,
        *,
        is_stepwise_action: bool,
        quantile_dims: Optional[List[int]],
    ) -> None:
        self.is_stepwise_action = bool(is_stepwise_action)
        self.quantile_dims = [int(dim) for dim in (quantile_dims or [])]
        self.minimum: Optional[torch.Tensor] = None
        self.maximum: Optional[torch.Tensor] = None
        self.total: Optional[torch.Tensor] = None
        self.total_square: Optional[torch.Tensor] = None
        self.count = 0
        self.quantile_chunks: List[torch.Tensor] = []

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        rows = tensor if self.is_stepwise_action else tensor.reshape(-1, tensor.shape[-1])
        if rows.shape[0] == 0:
            return
        chunk_min = rows.min(dim=0).values
        chunk_max = rows.max(dim=0).values
        data = rows.double()
        chunk_sum = data.sum(dim=0)
        chunk_sumsq = (data * data).sum(dim=0)
        if self.minimum is None:
            self.minimum = chunk_min
            self.maximum = chunk_max
            self.total = chunk_sum
            self.total_square = chunk_sumsq
        else:
            self.minimum = torch.minimum(self.minimum, chunk_min)
            self.maximum = torch.maximum(self.maximum, chunk_max)
            self.total = self.total + chunk_sum
            self.total_square = self.total_square + chunk_sumsq
        self.count += int(rows.shape[0])
        if self.quantile_dims:
            dims = torch.as_tensor(
                self.quantile_dims, dtype=torch.long, device=rows.device
            )
            self.quantile_chunks.append(rows.index_select(-1, dims))

    def finalize(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.minimum is None:
            return None
        result = {
            "min": self.minimum,
            "max": self.maximum,
            "sum": self.total,
            "sumsq": self.total_square,
            "count": torch.tensor([self.count], dtype=torch.float64),
        }
        if self.quantile_dims:
            result["quantile_data"] = torch.cat(self.quantile_chunks, dim=0)
        return result

def _gather_and_reduce_stats(
    local_stats: Optional[Dict[str, torch.Tensor]],
    *,
    action_dim: int = 0,
    quantile_dims: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    # Empty ranks still execute every all_gather; the union over ranks
    # carries non-empty contributions. Using a zero placeholder would
    # corrupt min/max/quantile statistics.
    local = local_stats or {}
    quantile_dims = [int(d) for d in (quantile_dims or [])]

    def gather(name: str) -> List[torch.Tensor]:
        value = local.get(name)
        return _all_gather_tensor_list([] if value is None else [value])

    mins = gather("min")
    maxs = gather("max")
    sums = gather("sum")
    sumsqs = gather("sumsq")
    counts = gather("count")
    if not mins:
        # Globally empty key: keep the two tail-gather slots balanced.
        # action_dim is the only signal we have for the fallback shape.
        _all_gather_tensor_list([])
        _all_gather_tensor_list([])
        return {
            "min": torch.zeros(action_dim),
            "max": torch.zeros(action_dim),
            "q01": torch.zeros(action_dim),
            "q99": torch.zeros(action_dim),
            "mean": torch.zeros(action_dim),
            "std": torch.zeros(action_dim),
            "count": 0,
        }

    global_min = torch.stack(mins).amin(dim=0)
    global_max = torch.stack(maxs).amax(dim=0)
    global_sum = torch.stack([x.double() for x in sums]).sum(dim=0)
    global_sumsq = torch.stack([x.double() for x in sumsqs]).sum(dim=0)
    global_count = torch.stack([x.double() for x in counts]).sum()

    # Exact distributed q01/q99: each rank sends only its local 1%/99% tails
    # (top-k), avoiding an all-gather of every action row. The two gathers
    # are unconditional so sparse ranks remain collective-safe.
    local_quantile_data = local.get("quantile_data")
    n_global = int(global_count.item())
    if quantile_dims:
        if local_quantile_data is not None and local_quantile_data.shape[-1] != len(quantile_dims):
            raise ValueError(
                "Local quantile data width does not match quantile_dims: "
                f"data={tuple(local_quantile_data.shape)}, dims={quantile_dims}."
            )

        q01_pos = 0.01 * (n_global - 1)
        q01_lo = int(np.floor(q01_pos))
        q01_hi = int(np.ceil(q01_pos))
        q01_weight = q01_pos - q01_lo
        low_k = q01_hi + 1

        q99_pos = 0.99 * (n_global - 1)
        q99_lo = int(np.floor(q99_pos))
        q99_hi = int(np.ceil(q99_pos))
        q99_weight = q99_pos - q99_lo
        high_k = n_global - q99_lo

        if local_quantile_data is None:
            local_low_tail: List[torch.Tensor] = []
            local_high_tail: List[torch.Tensor] = []
        else:
            local_rows = int(local_quantile_data.shape[0])
            local_low_tail = [
                torch.topk(
                    local_quantile_data.float(),
                    k=min(low_k, local_rows),
                    dim=0,
                    largest=False,
                    sorted=False,
                ).values
            ]
            local_high_tail = [
                torch.topk(
                    local_quantile_data.float(),
                    k=min(high_k, local_rows),
                    dim=0,
                    largest=True,
                    sorted=False,
                ).values
            ]

        gathered_low = _all_gather_tensor_list(local_low_tail)
        gathered_high = _all_gather_tensor_list(local_high_tail)
        if not gathered_low or not gathered_high:
            raise ValueError("Distributed quantile tails are empty for a non-empty key.")

        low_candidates = torch.cat(gathered_low, dim=0)
        low_tail = torch.topk(
            low_candidates,
            k=low_k,
            dim=0,
            largest=False,
            sorted=True,
        ).values
        q01_selected = low_tail[q01_lo]
        if q01_hi != q01_lo:
            q01_selected = q01_selected + (
                low_tail[q01_hi] - q01_selected
            ) * q01_weight

        high_candidates = torch.cat(gathered_high, dim=0)
        high_tail = torch.topk(
            high_candidates,
            k=high_k,
            dim=0,
            largest=True,
            sorted=False,
        ).values.sort(dim=0).values
        q99_selected = high_tail[0]
        if q99_hi != q99_lo:
            q99_selected = q99_selected + (
                high_tail[q99_hi - q99_lo] - q99_selected
            ) * q99_weight

        dims_t = torch.as_tensor(
            quantile_dims, dtype=torch.long, device=global_min.device
        )
        global_q01 = global_min.clone()
        global_q99 = global_max.clone()
        global_q01.index_copy_(-1, dims_t, q01_selected.to(global_min.dtype))
        global_q99.index_copy_(-1, dims_t, q99_selected.to(global_max.dtype))
    else:
        _all_gather_tensor_list([])
        _all_gather_tensor_list([])
        global_q01 = global_min.clone()
        global_q99 = global_max.clone()

    mean = global_sum / global_count.clamp_min(1.0)
    var_num = (global_sumsq - (global_sum * global_sum) / global_count.clamp_min(1.0)).clamp_min(0.0)
    if global_count.item() > 1:
        std = torch.sqrt(var_num / (global_count - 1.0))
    else:
        std = torch.zeros_like(mean)

    out_dtype = global_min.dtype
    return {
        "min": global_min,
        "max": global_max,
        "q01": global_q01.to(out_dtype),
        "q99": global_q99.to(out_dtype),
        "mean": mean.to(out_dtype),
        "std": std.to(out_dtype),
        # Cross-rank total count (not rank-local) so sparse ranks don't
        # write 0 when other ranks hold real data.
        "count": int(global_count.item()),
    }


def _build_global_normalizer_from_children(
    children: List[GameVideoDataset],
    processor_cfg: dict,
    shape_meta: dict,
    use_stepwise_action_norm: bool,
    norm_default_mode: str,
    norm_exception_mode: Optional[dict],
) -> Tuple[LinearNormalizer, dict]:
    """Compute global normalizer and return (normalizer, raw_stats_dict).

    Iterates ALL episodes via _get_episode_data, applies
    action_state_transform, pools tensors, and computes global stats.
    Equivalent to what regular ``BaseGameDataset.get_dataset_stats``
    does, but aggregated across children.
    """
    dist_state = PartialState()
    rank = dist_state.process_index
    world_size = dist_state.num_processes
    child_episode_counts = [(child, child.game_dataset.multi_dataset.num_episodes) for child in children]
    total_episodes = sum(num_ep for _, num_ep in child_episode_counts)
    local_total_episodes = max((total_episodes + world_size - 1 - rank) // world_size, 0)
    logger.info(
        "Computing global normalizer stats from %d children (rank %d/%d, local episodes %d/%d) ...",
        len(children), rank, world_size, local_total_episodes, total_episodes,
    )

    global_ep_idx = 0
    local_jobs = []
    for child, num_ep in child_episode_counts:
        gd = child.game_dataset
        processor = gd.processor
        if processor is None:
            raise ValueError("Child must have processor set even with delay_normalization.")
        for ep_idx in range(num_ep):
            if global_ep_idx % world_size == rank:
                local_jobs.append((gd, processor, ep_idx))
            global_ep_idx += 1
    local_total_episodes = len(local_jobs)

    # Route-aware config is a single common-level field shared by all
    # children via hydra interpolation; reading it from processor_cfg
    # is the canonical place to look it up.
    route_aware = bool(
        processor_cfg.get("route_aware_action_normalization", False)
    )
    route_state_key = processor_cfg.get("gui_label_state_key", "default")
    if route_state_key is not None:
        route_state_key = str(route_state_key)
    route_dim_index = processor_cfg.get("gui_label_dim_index", None)
    if route_dim_index is not None:
        route_dim_index = int(route_dim_index)

    if route_aware and bool(use_stepwise_action_norm):
        raise NotImplementedError(
            "Mixed dataset route-aware action normalization with stepwise norm "
            "is not supported. Set use_stepwise_action_norm=false or "
            "route_aware_action_normalization=false."
        )
    if route_aware and (
        route_dim_index is None
        or route_state_key is None
        or not route_state_key.strip()
    ):
        raise ValueError(
            "`route_aware_action_normalization=True` requires "
            "`gui_label_dim_index` and `gui_label_state_key` to be set "
            "in the common block of the data config."
        )

    accumulators: Dict[str, Dict[str, _StreamingStats]] = {
        "action": {},
        "state": {},
    }
    for field in ("action", "state"):
        for meta in shape_meta.get(field, []):
            quantile_dims = _quantile_dims_for_meta(
                field, meta, norm_default_mode, norm_exception_mode
            )
            accumulators[field][meta["key"]] = _StreamingStats(
                is_stepwise_action=bool(
                    use_stepwise_action_norm and field == "action"
                ),
                quantile_dims=quantile_dims,
            )
    route_accumulators: Dict[str, Dict[str, _StreamingStats]] = {
        route: {
            meta["key"]: _StreamingStats(
                is_stepwise_action=False,
                quantile_dims=_quantile_dims_for_meta(
                    "action", meta, norm_default_mode, norm_exception_mode
                ),
            )
            for meta in shape_meta.get("action", [])
        }
        for route in ("gameplay", "gui")
    } if route_aware else {}

    def process_job(job):
        gd, processor, ep_idx = job
        batch = gd._get_episode_data(ep_idx)
        action_values = list((batch.get("action") or {}).values())
        if not action_values:
            raise ValueError(f"Episode {ep_idx} contains no action tensors.")
        action_starts = int(action_values[0].shape[0])
        if any(int(value.shape[0]) != action_starts for value in action_values[1:]):
            raise ValueError(f"Episode {ep_idx} has inconsistent action start counts.")
        if action_starts <= 0:
            return None
        # Snapshot raw (pre-transform) route label so per-route action
        # stats align with the action token at index [0:T].
        route_label = None
        route_valid = None
        if route_aware:
            state_dict = batch.get("state") or {}
            raw_state = state_dict.get(route_state_key)
            if raw_state is None:
                raise KeyError(
                    f"Route state key {route_state_key!r} is missing from "
                    f"episode {ep_idx}; available keys={list(state_dict)}."
                )
            full_route_label = _build_episode_route_window(
                raw_state,
                route_dim_index=int(route_dim_index),
                window_size=int(gd.action_size),
                expected_starts=int(raw_state.shape[0]),
            )
            route_label = full_route_label[:action_starts]
            state_field_valid = batch.get("state_field_valid")
            if isinstance(state_field_valid, dict):
                state_field_valid = state_field_valid.get(route_state_key)
            if state_field_valid is not None:
                full_route_valid = _build_episode_route_window(
                    state_field_valid,
                    route_dim_index=int(route_dim_index),
                    window_size=int(gd.action_size),
                    expected_starts=int(raw_state.shape[0]),
                )
                route_valid = full_route_valid[:action_starts]
        batch = processor.action_state_transform(batch)
        return batch, route_label, route_valid

    def consume_batch(result):
        if result is None:
            return
        batch, route_label, route_valid = result
        for meta in shape_meta.get("action", []):
            key = meta["key"]
            action_tensor = batch["action"][key]
            accumulators["action"][key].update(action_tensor)
            if not route_aware or route_label is None:
                continue
            resolved_route_valid = (
                torch.ones_like(route_label, dtype=torch.bool)
                if route_valid is None
                else route_valid.to(dtype=torch.bool)
            )
            if action_tensor.dim() == 3 and route_label.dim() == 3:
                if action_tensor.shape[:2] != route_label.shape[:2]:
                    raise ValueError(
                        "Mixed route stats action/label window mismatch: "
                        f"action={tuple(action_tensor.shape)}, "
                        f"route={tuple(route_label.shape)}."
                    )
                if resolved_route_valid.shape != route_label.shape:
                    raise ValueError(
                        "Mixed route stats route-valid window mismatch: "
                        f"label={tuple(route_label.shape)}, "
                        f"valid={tuple(resolved_route_valid.shape)}."
                    )
                route_bits = route_label[:, :, 0].bool().reshape(-1)
                valid_bits = resolved_route_valid[:, :, 0].bool().reshape(-1)
                flat_action = action_tensor.reshape(-1, action_tensor.shape[-1])
            elif action_tensor.dim() == 2 and route_label.dim() == 2:
                if action_tensor.shape[0] != route_label.shape[0] or route_label.shape[-1] != 1:
                    raise ValueError(
                        "Mixed route stats action/label token mismatch: "
                        f"action={tuple(action_tensor.shape)}, "
                        f"route={tuple(route_label.shape)}."
                    )
                if resolved_route_valid.shape != route_label.shape:
                    raise ValueError(
                        "Mixed route stats route-valid token mismatch: "
                        f"label={tuple(route_label.shape)}, "
                        f"valid={tuple(resolved_route_valid.shape)}."
                    )
                route_bits = route_label[:, 0].bool()
                valid_bits = resolved_route_valid[:, 0].bool()
                flat_action = action_tensor
            else:
                raise ValueError(
                    "Mixed route stats expects both action and route label "
                    "to be 2D or both to be 3D, got "
                    f"action={tuple(action_tensor.shape)}, "
                    f"route={tuple(route_label.shape)}."
                )
            for route in ("gameplay", "gui"):
                selected = (
                    route_bits if route == "gui" else ~route_bits
                ) & valid_bits
                if bool(selected.any()):
                    route_accumulators[route][key].update(flat_action[selected])
        for meta in shape_meta.get("state", []):
            key = meta["key"]
            accumulators["state"][key].update(batch["state"][key])

    local_processed = 0
    progress = tqdm(
        total=local_total_episodes,
        desc=f"Mixed norm stats rank {rank}/{world_size}",
        disable=not dist_state.is_main_process,
    )
    max_workers = min(DEFAULT_MIXED_NORM_STATS_WORKERS, max(len(local_jobs), 1))
    max_pending = max_workers * MIXED_NORM_STATS_PENDING_FACTOR
    logger.info(
        "Mixed normalizer stats rank %d/%d: using %d worker threads",
        rank, world_size, max_workers,
    )

    job_iter = iter(local_jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = set()

        def submit_next():
            try:
                job = next(job_iter)
            except StopIteration:
                return False
            pending.add(executor.submit(process_job, job))
            return True

        for _ in range(max_pending):
            if not submit_next():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                consume_batch(future.result())
                local_processed += 1
                progress.update(1)
                if local_processed % 10000 == 0:
                    logger.info(
                        "Mixed normalizer stats rank %d/%d: processed %d/%d local episodes",
                        rank, world_size, local_processed, local_total_episodes,
                    )
                submit_next()
    progress.close()
    logger.info(
        "Mixed normalizer stats rank %d/%d: finished episode transforms; computing local stats.",
        rank, world_size,
    )

    # Build global stats from pooled tensors
    stats: Dict[str, Any] = {"action": {}, "state": {}}

    for field in ("action", "state"):
        for meta in shape_meta.get(field, []):
            key = meta["key"]
            accumulator = accumulators[field][key]
            is_stepwise_action = bool(
                use_stepwise_action_norm and field == "action"
            )
            prefix = "stepwise_" if is_stepwise_action else "global_"
            quantile_dims = _quantile_dims_for_meta(
                field, meta, norm_default_mode, norm_exception_mode
            )
            logger.info(
                "Mixed normalizer stats rank %d/%d: reducing streamed %s/%s "
                "(%d rows, quantile_dims=%s).",
                rank,
                world_size,
                field,
                key,
                accumulator.count,
                quantile_dims,
            )
            reduce_start = time.perf_counter()
            key_dim = int(meta.get("raw_shape", 0))
            reduced = _gather_and_reduce_stats(
                accumulator.finalize(),
                action_dim=key_dim,
                quantile_dims=quantile_dims,
            )
            logger.info(
                "Mixed normalizer stats rank %d/%d: reduced %s/%s stats in %.1fs.",
                rank,
                world_size,
                field,
                key,
                time.perf_counter() - reduce_start,
            )
            stats[field][key] = {
                f"{prefix}min": reduced["min"],
                f"{prefix}max": reduced["max"],
                f"{prefix}q01": reduced["q01"],
                f"{prefix}q99": reduced["q99"],
                f"{prefix}mean": reduced["mean"],
                f"{prefix}std": reduced["std"],
            }

    normalizer = LinearNormalizer(
        shape_meta=shape_meta,
        use_stepwise_action_norm=use_stepwise_action_norm,
        default_mode=norm_default_mode,
        exception_mode=norm_exception_mode,
        stats=stats,
    )

    if route_aware:
        stats["action_by_route"] = {}
        action_meta_list = shape_meta.get("action", [])
        if len(action_meta_list) != 1:
            raise ValueError(
                "Mixed route-aware normalization currently requires exactly "
                f"one action key, got {[m.get('key') for m in action_meta_list]}."
            )
        for route in ("gameplay", "gui"):
            route_block: Dict[str, Any] = {}
            total_count = 0
            for meta in action_meta_list:
                key = meta["key"]
                action_dim = int(meta.get("raw_shape", 0))
                quantile_dims = _quantile_dims_for_meta(
                    "action", meta, norm_default_mode, norm_exception_mode
                )
                reduced = _gather_and_reduce_stats(
                    route_accumulators[route][key].finalize(),
                    action_dim=action_dim,
                    quantile_dims=quantile_dims,
                )
                prefix = "stepwise_" if use_stepwise_action_norm else "global_"
                route_block[key] = {
                    f"{prefix}min": reduced["min"],
                    f"{prefix}max": reduced["max"],
                    f"{prefix}q01": reduced["q01"],
                    f"{prefix}q99": reduced["q99"],
                    f"{prefix}mean": reduced["mean"],
                    f"{prefix}std": reduced["std"],
                }
                total_count += int(reduced.get("count", 0))
            route_block["count"] = total_count
            route_block["empty"] = total_count == 0
            stats["action_by_route"][route] = route_block
        stats["route_aware_action_normalization"] = {
            "enabled": True,
            "route_state_key": route_state_key,
            "gui_label_dim_index": (
                int(route_dim_index) if route_dim_index is not None else None
            ),
            "continuous_dims": [
                int(index)
                for index, is_continuous in enumerate(
                    processor_cfg.get("dim_is_continuous", []) or []
                )
                if bool(is_continuous)
            ],
            "action_dim": sum(
                int(meta["raw_shape"]) for meta in shape_meta.get("action", [])
            ),
        }

    return normalizer, stats


class MixedGameVideoDataset(torch.utils.data.Dataset):
    """Dataset that combines multiple GameVideoDataset children.

    Each child maintains its own sampling (anchor, fine-task resampling etc.)
    but shares a single global normalizer. Group-weighted sampling is provided
    via ``get_sampler_weights()``.
    """

    def __init__(
        self,
        child: DictConfig,
        weights: Optional[Dict[str, float]] = None,
        split: str = "train",
        shape_meta: Optional[Any] = None,
        processor: Optional[Any] = None,
        max_expected_repeat: Optional[float] = None,
        pretrained_norm_stats: Optional[str] = None,
        replacement: bool = False,
        **kwargs,
    ):
        self.split = str(split)
        self.names: List[str] = []
        self.children: List[GameVideoDataset] = []
        self.target_weights: List[float] = []
        self.shape_meta = _ensure_container(shape_meta) if shape_meta is not None else None

        # Children arrive as complete GameVideoDataset configs,
        # materialized by ``materialize_dataset_nodes`` (via
        # ``build_datasets``); merging and consistency checks are
        # owned by the materialization step.
        for name, node_dict in child.items():
            ds = instantiate(node_dict, delay_normalization=True)
            self.names.append(name)
            self.children.append(ds)
            self.target_weights.append(float((weights or {}).get(name, 1.0)))
            logger.info(
                "Child %s: %d samples, target_weight=%.2f",
                name, len(ds), self.target_weights[-1],
            )

        # Build index offsets for global idx -> (child_idx, local_idx) mapping
        self.offsets = []
        cumulative = 0
        for ds in self.children:
            self.offsets.append(cumulative)
            cumulative += len(ds)
        self._offset_ends = self.offsets[1:] + [cumulative]

        self.effective_weights = self._apply_repeat_caps(max_expected_repeat or 10.0)
        self.replacement = bool(replacement)

        self._print_group_info()

        processor_dict = _ensure_container(processor) if processor is not None else None

        self._setup_normalizer_and_merger(
            processor_dict=processor_dict,
            shape_meta=shape_meta,
            pretrained_norm_stats=pretrained_norm_stats,
        )

        if processor_dict is not None:
            proc = instantiate(processor_dict)
            proc._normalizer = self.normalizer
            proc.action_state_merger = self.action_state_merger
            # Build the processor's per-route normalizer from the
            # SAME raw stats used for self.normalizer. Without this,
            # train/val would stay in global-normalization space while
            # inference uses route-aware denorm (silent drift).
            if getattr(self, "_raw_stats", None) is not None:
                proc.set_normalizer_from_stats(self._raw_stats)
                proc._normalizer = self.normalizer
            self._processor_instance = proc

    def _setup_normalizer_and_merger(
        self,
        processor_dict: Optional[dict],
        shape_meta: Optional[Any],
        pretrained_norm_stats: Optional[str],
    ):
        """Build or load global normalizer, and set up action_state_merger."""
        if processor_dict is None:
            raise ValueError("MixedGameVideoDataset requires 'processor' in common config.")

        shape_meta_dict = _ensure_container(shape_meta)

        # Extract normalizer params from processor config
        use_stepwise_action_norm = bool(processor_dict.get("use_stepwise_action_norm", False))
        norm_default_mode = processor_dict.get("norm_default_mode", "min/max")
        norm_exception_mode = _ensure_container(processor_dict.get("norm_exception_mode"))

        if self.split == "train":
            if pretrained_norm_stats is not None:
                stats = load_dataset_stats_from_json(pretrained_norm_stats)
                self.normalizer = LinearNormalizer(
                    shape_meta=shape_meta_dict,
                    use_stepwise_action_norm=use_stepwise_action_norm,
                    default_mode=norm_default_mode,
                    exception_mode=norm_exception_mode,
                    stats=stats,
                )
                raw_stats = stats
                logger.info("Loaded global normalizer from %s (train split)", pretrained_norm_stats)
            else:
                self.normalizer, raw_stats = _build_global_normalizer_from_children(
                    children=self.children,
                    processor_cfg=processor_dict,
                    shape_meta=shape_meta_dict,
                    use_stepwise_action_norm=use_stepwise_action_norm,
                    norm_default_mode=norm_default_mode,
                    norm_exception_mode=norm_exception_mode,
                )
            # Save raw stats (with global_/stepwise_ prefix) for val reuse.
            # Do NOT use self.normalizer.get_stats() — it strips prefixes.
            work_dir = misc.get_work_dir()
            stats_path = os.path.join(work_dir, "dataset_stats.json")
            dist_state = PartialState()
            if dist_state.is_main_process:
                if pretrained_norm_stats is not None:
                    # Preserve explicitly supplied stats byte-for-byte. Loading turns
                    # every numeric list into a float32 tensor, so serializing the
                    # loaded object would change schema fields such as dimension
                    # indices from [0, 1] to [0.0, 1.0].
                    source_path = os.path.realpath(os.fspath(pretrained_norm_stats))
                    destination_path = os.path.realpath(stats_path)
                    if source_path != destination_path:
                        shutil.copyfile(source_path, destination_path)
                else:
                    save_dataset_stats_to_json(raw_stats, stats_path)
                logger.info("Global normalizer stats saved to %s", stats_path)
            dist_state.wait_for_everyone()
        else:
            if pretrained_norm_stats is None:
                raise ValueError(
                    "MixedGameVideoDataset split='val' requires pretrained_norm_stats"
                )
            stats = load_dataset_stats_from_json(pretrained_norm_stats)
            self.normalizer = LinearNormalizer(
                shape_meta=shape_meta_dict,
                use_stepwise_action_norm=use_stepwise_action_norm,
                default_mode=norm_default_mode,
                exception_mode=norm_exception_mode,
                stats=stats,
            )
            raw_stats = stats
            logger.info("Loaded global normalizer from %s", pretrained_norm_stats)
        # Stash raw stats so the processor can later build its per-route
        # normalizer from the same source of truth (train/inference drift).
        self._raw_stats = raw_stats

        # Setup action_state_merger from processor config
        action_state_merger_cfg = processor_dict.get("action_state_merger")
        if action_state_merger_cfg is not None:
            self.action_state_merger = instantiate(action_state_merger_cfg)
            self.action_state_merger.set_shape_meta(shape_meta_dict)
        else:
            self.action_state_merger = None

    def _apply_repeat_caps(self, max_expected_repeat: float) -> List[float]:
        """Apply repeat caps to prevent individual samples from being repeated too often.

        Each child's per-epoch sample budget is:
            child_budget = target_weight / sum(target_weights) * global_epoch_size
            per_sample_exposure = budget / len(child)

        If per_sample_exposure > max_expected_repeat, the budget is capped at:
            max_expected_repeat * len(child)
        """
        target = torch.tensor(self.target_weights, dtype=torch.float32)
        total_target = target.sum()
        norm_target = target / total_target

        # global_epoch_size is typically sum(len(child)) (default epoch size)
        global_epoch_size = sum(len(ds) for ds in self.children)
        budgets = norm_target * global_epoch_size
        per_sample_exposure = budgets / torch.tensor([len(ds) for ds in self.children], dtype=torch.float32)

        capped_budgets = budgets.clone()
        for i in range(len(self.children)):
            if per_sample_exposure[i] > max_expected_repeat:
                capped_budgets[i] = max_expected_repeat * len(self.children[i])
                logger.info(
                    "Child %s: per-sample exposure %.2f > cap %.2f, capping budget to %.0f",
                    self.names[i], per_sample_exposure[i].item(), max_expected_repeat, capped_budgets[i].item(),
                )

        # Update effective weights from capped budgets
        effective = capped_budgets / capped_budgets.sum()
        return effective.tolist()

    def _print_group_info(self):
        """Log group statistics."""
        logger.info("=== MixedGameVideoDataset Group Info ===")
        for i, name in enumerate(self.names):
            child_len = len(self.children[i])
            target_w = self.target_weights[i]
            eff_w = self.effective_weights[i]
            budget = eff_w * sum(len(ds) for ds in self.children)
            per_sample = budget / max(child_len, 1)
            logger.info(
                "  %s: %d samples, target=%.2f, effective=%.4f, per-sample=%.2f",
                name, child_len, target_w, eff_w, per_sample,
            )
        total = sum(len(ds) for ds in self.children)
        logger.info("  Total: %d samples", total)
        logger.info("========================================")

    def _map_index(self, idx: int) -> tuple:
        """Map global index to (child_idx, local_idx)."""
        ends = getattr(self, "_offset_ends", None)
        if ends is None:
            for child_idx in range(len(self.offsets) - 1, -1, -1):
                if idx >= self.offsets[child_idx]:
                    return child_idx, idx - self.offsets[child_idx]
            raise IndexError(f"Index {idx} out of range [0, {len(self)}).")
        if idx < 0 or idx >= ends[-1]:
            raise IndexError(f"Index {idx} out of range [0, {ends[-1]}).")
        child_idx = bisect.bisect_right(ends, idx)
        return child_idx, idx - self.offsets[child_idx]

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.children)

    def get_raw_video(self, idx: int, sample_idx=None):
        child_idx, local_idx = self._map_index(idx)
        return self.children[child_idx].get_raw_video(local_idx, sample_idx=sample_idx)

    @property
    def processor(self):
        return self._processor_instance

    def __getitem__(self, idx: int):
        child_idx, local_idx = self._map_index(idx)
        sample = self.children[child_idx][local_idx]

        # sample at this point has:
        #   - action as dict (after action_state_transform, before normalizer)
        #   - proprio as dict (delay_normalization mode)
        #   - instruction, pixel_values, etc.

        # LinearNormalizer uses the shape_meta key "state" while child
        # datasets expose the same tensor as "proprio".
        if "proprio" in sample:
            sample["state"] = sample.pop("proprio")
        if "proprio_is_pad" in sample:
            sample["state_is_pad"] = sample.pop("proprio_is_pad")
        if "proprio_dim_is_pad" in sample:
            sample["state_dim_is_pad"] = sample.pop("proprio_dim_is_pad")

        # Apply the same route-aware action normalization as the regular
        # GameWAMProcessor path. Children run with delay_normalization=True,
        # so this is the only place where their action enters model space.
        proc = self._processor_instance
        if bool(getattr(proc, "route_aware_action_normalization", False)):
            route_mask = sample.get("action_route_before_action")
            route_valid = sample.get("action_route_valid")
            if not isinstance(route_mask, torch.Tensor) or not isinstance(route_valid, torch.Tensor):
                raise RuntimeError(
                    "Mixed route-aware normalization requires `action_route_before_action` "
                    "and `action_route_valid` from the child processor."
                )
            if proc.route_norm is None:
                raise RuntimeError(
                    "Mixed route-aware normalization is enabled but the common "
                    "processor's `route_norm` was not initialized."
                )
            sample["action"] = self.normalizer.forward_action_by_route(
                sample["action"],
                route_mask.bool(),
                proc.route_norm,
                proc.route_norm,
                route_valid=route_valid.bool(),
            )
            sample = self.normalizer.forward_state(sample)
        else:
            sample = self.normalizer.forward(sample)
        if self.action_state_merger is not None:
            sample = self.action_state_merger.forward(sample)

        # Rename back to proprio for downstream model code
        if "state" in sample:
            sample["proprio"] = sample.pop("state")
        if "state_is_pad" in sample:
            sample["proprio_is_pad"] = sample.pop("state_is_pad")
        if "state_dim_is_pad" in sample:
            sample["proprio_dim_is_pad"] = sample.pop("state_dim_is_pad")

        # Attach the child identity used by per-group metrics and logging.
        sample["dataset_group"] = self.names[child_idx]
        sample["processor_id"] = self.names[child_idx]

        return sample

    def get_sampler_weights(self) -> torch.Tensor:
        """Return per-sample fine-task-resample weights only (Layer 2).

        Each child's weights are normalised to mean 1 so they act as
        within-child priority multipliers without altering the per-child
        budget.  Layer 1 (inter-child mixing) is handled separately via
        ``self.effective_weights``, exposed through
        ``get_child_sampling_specs()``.

        The order must match __len__ and __getitem__ global index order.
        """
        weights = []
        for child in self.children:
            if hasattr(child, "get_sampler_weights"):
                inner = child.get_sampler_weights()
                if inner is not None:
                    inner = inner.float()
                    inner_mean = inner.mean().clamp_min(1e-8)
                    inner = inner / inner_mean
                else:
                    inner = torch.ones(len(child), dtype=torch.float32)
            else:
                inner = torch.ones(len(child), dtype=torch.float32)
            weights.append(inner)

        result = torch.cat(weights, dim=0)
        assert len(result) == len(self), (
            f"get_sampler_weights() length {len(result)} != __len__() {len(self)}"
        )
        return result

    def get_child_sampling_specs(self) -> list:
        """Return ChildSamplingSpec list for the task-quota-cyclic sampler."""
        from gamewam.utils.samplers import ChildSamplingSpec

        specs = []
        for i, name in enumerate(self.names):
            start = self.offsets[i]
            end = self._offset_ends[i]
            child = self.children[i]
            task_spec = child.get_task_sampling_spec() if hasattr(child, "get_task_sampling_spec") else None
            specs.append(ChildSamplingSpec(
                name=name,
                global_start=start,
                global_end=end,
                normalized_weight=self.effective_weights[i],
                fine_task_spec=task_spec,
            ))
        return specs
