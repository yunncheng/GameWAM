from typing import Dict, Callable, Optional
import os
import random
import gc

import torch
import numpy as np


def _resolve_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0"))))


def resolve_train_device() -> str:
    """Pick the per-process torch device based on CUDA availability and ``LOCAL_RANK``."""
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (random, numpy, torch) and optionally
    produces a DataLoader worker init function.
    """
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)

    # Process-specific seeding: offset by global rank so each process gets a different seed
    global_rank = _resolve_global_rank()
    process_seed = seed + global_rank

    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    if not get_worker_init_fn:
        return None

    # Worker init pattern borrowed from PyTorch-Lightning:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    def _worker_init_function(worker_id: int) -> None:
        gc.enable()

        process_seed = torch.initial_seed()
        global_rank = _resolve_global_rank()
        base_seed = process_seed - worker_id
        seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])
        np.random.seed(seed_seq.generate_state(4))
        torch_seed_seq, random_seed_seq, _ = seed_seq.spawn(3)
        torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
        random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
        random.seed(random_seed)

    return _worker_init_function


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
