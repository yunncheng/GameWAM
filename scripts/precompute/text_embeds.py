import hashlib
import json
import logging
import re
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from gamewam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from gamewam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from gamewam.utils.cfg import register_default_resolvers
from gamewam.utils.logging_config import get_logger, setup_logging
from utils import atomic_torch_save, init_gloo_distributed, iter_split_aware_dataset_nodes, to_bool

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _collect_dataset_settings(data_cfg: DictConfig):
    """Collect dataset settings using split-aware iteration.

    Returns:
        per_node: list of (node_path, dataset_dirs, cache_dir, context_len, prompt_template)
    """
    per_node = []

    for node_path, node in iter_split_aware_dataset_nodes(data_cfg, split="train"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}`."
            )

        dataset_dirs = [str(ds) for ds in raw_dirs]

        context_len = node.get("context_len")
        prompt_template = node.get("prompt_template")
        if prompt_template is None:
            raise ValueError(f"Missing `prompt_template` for dataset node `{node_path}`.")

        per_node.append((
            node_path,
            dataset_dirs,
            Path(str(cache_dir)).expanduser(),
            int(context_len) if context_len is not None else DEFAULT_CONTEXT_LEN,
            str(prompt_template),
        ))
        logger.info(
            "Discovered dataset node `%s` with %d dataset_dirs, cache=%s.",
            node_path, len(raw_dirs), cache_dir,
        )

    return per_node


def _resolve_context_len(context_lens: set[int]) -> int:
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _load_tasks_map(ds_dir: str) -> dict[int, str]:
    """Load {task_index: task_string} from a dataset's tasks.jsonl."""
    tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
    tasks_map: dict[int, str] = {}
    if not tasks_path.exists():
        return tasks_map
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tasks_map[record["task_index"]] = str(record["task"])
    return tasks_map


def _read_unique_coarse_fine_pairs(ds_dir: str) -> set[tuple[str, str]]:
    """Scan parquet files for unique (coarse_task, fine_task) string pairs."""
    import glob
    import pandas as pd

    tasks_map = _load_tasks_map(ds_dir)
    if not tasks_map:
        return set()

    parquet_files = sorted(glob.glob(str(Path(ds_dir) / "data" / "chunk-*" / "*.parquet")))
    if not parquet_files:
        return set()

    pairs: set[tuple[str, str]] = set()
    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path, columns=["coarse_task_index", "task_index"])
        if "coarse_task_index" not in df.columns:
            # No dual-task info; skip — fine-only prompts are already
            # covered by _read_unique_prompts.
            continue
        for _, row in df[["coarse_task_index", "task_index"]].drop_duplicates().iterrows():
            ci, fi = int(row["coarse_task_index"]), int(row["task_index"])
            coarse = tasks_map.get(ci, "")
            fine = tasks_map.get(fi, "")
            if coarse and fine and coarse != fine:
                pairs.add((coarse, fine))

    return pairs


def _read_unique_prompts(dataset_dirs: list[str], prompt_template: str) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for ds_dir in dataset_dirs:
        # 1. Fine-only / coarse-only prompts from tasks.jsonl
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")

        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                task = str(record["task"])
                prompt = prompt_template.format(task=task)
                total_task_rows += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)

        # 2. Dual-task prompts from parquet coarse/fine pairs
        coarse_fine_pairs = _read_unique_coarse_fine_pairs(ds_dir)
        dual_count = 0
        for coarse, fine in coarse_fine_pairs:
            dual_task = f"[High]: {coarse}, [Low]: {fine}"
            prompt = prompt_template.format(task=dual_task)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
                dual_count += 1

        if dual_count > 0:
            logger.info("Added %d dual-task prompts from %s", dual_count, ds_dir)

    logger.info(
        "Loaded %d task rows from %d datasets, deduplicated to %d prompts.",
        total_task_rows,
        len(dataset_dirs),
        len(prompts),
    )

    # Always include empty prompt for CFG training (template with empty task).
    empty_prompt = prompt_template.format(task="")
    if empty_prompt not in seen:
        seen.add(empty_prompt)
        prompts.append(empty_prompt)
        logger.info("Added empty prompt for CFG: hash=%s", hashlib.sha256(empty_prompt.encode("utf-8")).hexdigest())

    return prompts


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = init_gloo_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute/text_embeds.py",
            torch.cuda.device_count(),
        )

    overwrite = to_bool(cfg.get("overwrite", False))
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")

    per_node = _collect_dataset_settings(cfg.data)
    if not per_node:
        raise ValueError("No dataset nodes with `text_embedding_cache_dir` found under `cfg.data`.")

    # Collect all unique prompts per cache_dir
    cache_dir_to_prompts: dict[Path, list[str]] = {}
    cache_dir_to_context_lens: dict[Path, set[int]] = {}

    for node_path, node_dataset_dirs, cache_dir, context_len, prompt_template in per_node:
        prompts = _read_unique_prompts(node_dataset_dirs, prompt_template)
        cache_dir_to_prompts.setdefault(cache_dir, [])
        cache_dir_to_context_lens.setdefault(cache_dir, set()).add(context_len)
        seen = set(cache_dir_to_prompts[cache_dir])
        for p in prompts:
            if p not in seen:
                seen.add(p)
                cache_dir_to_prompts[cache_dir].append(p)

    # Resolve context_lens — per cache_dir
    context_lens_map: dict[Path, int] = {}
    for cache_dir, lens in cache_dir_to_context_lens.items():
        if len(lens) != 1:
            one_node = [n for n, _, cd, _, _ in per_node if cd == cache_dir][0]
            raise ValueError(
                f"Multiple context_len values for cache_dir={cache_dir} (node={one_node}): {sorted(lens)}. "
                "Each cache_dir should map to a single context_len."
            )
        context_lens_map[cache_dir] = next(iter(lens))

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    enc_id = _model_id_to_enc_id(model_id)

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=max(context_lens_map.values()),
        clean="whitespace",
    )

    # Process each cache_dir with its own prompts
    for cache_dir, prompts in cache_dir_to_prompts.items():
        context_len = context_lens_map[cache_dir]
        logger.info(
            "Processing cache_dir=%s context_len=%d prompts=%d",
            cache_dir, context_len, len(prompts),
        )

        if not prompts:
            logger.warning("No prompts for cache_dir=%s; skipping.", cache_dir)
            continue

        _encode_prompts_for_cache_dir(
            prompts=prompts,
            cache_dir=cache_dir,
            context_len=context_len,
            enc_id=enc_id,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            overwrite=overwrite,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
        )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _encode_prompts_for_cache_dir(
    prompts,
    cache_dir,
    context_len,
    enc_id,
    tokenizer,
    text_encoder,
    device,
    overwrite,
    is_distributed,
    rank,
    world_size,
):
    stats = {"new": 0, "overwrite": 0, "skip": 0}

    prompts = prompts[rank::world_size] if is_distributed else prompts

    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            cache_path = cache_dir / filename
            if cache_path.exists():
                fully_cached_local += 1
                stats["skip"] += 1
            else:
                prompts_to_encode.append(prompt)
        prompts = prompts_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "Cache %s overwrite=false: fully cached=%d, to encode=%d",
                cache_dir.name, fully_cached_global, to_encode_global,
            )

    logger.info("Encoding %d prompts for %s.", len(prompts), cache_dir)
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding {cache_dir.name}",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                tokenizer.seq_len = context_len
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                    if cache_path.exists() and not overwrite:
                        stats["skip"] += 1
                        continue

                    if cache_path.exists():
                        stats["overwrite"] += 1
                    else:
                        stats["new"] += 1

                    atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [stats["new"], stats["overwrite"], stats["skip"]],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            stats["new"] = int(counts_tensor[0].item())
            stats["overwrite"] = int(counts_tensor[1].item())
            stats["skip"] = int(counts_tensor[2].item())

    if (not is_distributed) or rank == 0:
        logger.info(
            "Cache %s: new=%d overwrite=%d skip=%d over_length=%d/%d",
            cache_dir.name, stats["new"], stats["overwrite"], stats["skip"],
            over_length_global, prompts_encoded_global,
        )


if __name__ == "__main__":
    main()
