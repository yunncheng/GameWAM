"""Mixed dataset sampler with task-quota cyclic sampling.

Architecture:

  Phase 1  — Mixed child budgets:  largest-remainder, deterministic.
  Phase 2  — Fill each budget:
      * Plain child / fine without resample → pool-level sampling.
      * Fine with resample → task quotas → per-task pool sampling.

  ``replacement=false`` (default):
      budget ≤ pool → without-replacement (max distinct coverage).
      budget > pool → full cycles then remainder.
  ``replacement=true``:
      weighted multinomial with replacement on each pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sized, Tuple

import torch
from torch.utils.data import Sampler


# ── Data structures ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class FineTaskSamplingSpec:
    """Per-task sampling metadata produced by a fine child dataset."""

    task_names: Tuple[str, ...]
    indices_by_task: Tuple[torch.Tensor, ...]  # each tensor: local indices for that task
    task_counts: torch.Tensor  # [num_tasks], int
    gamma: float
    mix: float
    max_expected_repeat: float

    def __post_init__(self):
        n = len(self.task_names)
        if len(self.indices_by_task) != n:
            raise ValueError("indices_by_task length mismatch.")
        if self.task_counts.shape != (n,):
            raise ValueError("task_counts shape mismatch.")
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0.")
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("mix must be in [0, 1].")
        if self.max_expected_repeat < 1:
            raise ValueError("max_expected_repeat must be >= 1.")


@dataclass(frozen=True)
class ChildSamplingSpec:
    """Per-child sampling metadata produced by MixedGameVideoDataset."""

    name: str
    global_start: int
    global_end: int
    normalized_weight: float
    fine_task_spec: Optional[FineTaskSamplingSpec]

    def __post_init__(self):
        if self.normalized_weight <= 0:
            raise ValueError(f"normalized_weight must be positive, got {self.normalized_weight}")
        if self.global_start >= self.global_end:
            raise ValueError(
                f"global_start ({self.global_start}) must be < global_end ({self.global_end})."
            )


# ── Stable seed ────────────────────────────────────────────────────────────


def stable_seed(
    base_seed: int,
    epoch: int,
    child_id: int,
    task_id: int,
    cycle_id: int,
) -> int:
    """Deterministic seed from (base_seed, epoch, child, task, cycle)."""
    value = int(base_seed) & 0x7FFFFFFFFFFFFFFF
    for item in (epoch, child_id, task_id, cycle_id):
        value ^= int(item) + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2)
        value &= 0x7FFFFFFFFFFFFFFF
    return value


def _make_generator_factory(
    base_seed: int,
    epoch: int,
    child_id: int,
    task_id: int,
) -> Callable[[int], torch.Generator]:
    def _factory(cycle_id: int) -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(stable_seed(base_seed, epoch, child_id, task_id, cycle_id))
        return g
    return _factory


# ── Budget allocation ──────────────────────────────────────────────────────


def allocate_exact_budgets(
    weights: torch.Tensor,
    epoch_size: int,
) -> torch.Tensor:
    """Largest-remainder allocation summing to *epoch_size*."""
    if epoch_size <= 0:
        raise ValueError("epoch_size must be positive.")
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("weights must be a non-empty 1D tensor.")
    if not torch.isfinite(weights).all():
        raise ValueError("weights contain non-finite values.")
    if (weights < 0).any():
        raise ValueError("weights must be non-negative.")

    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must have positive sum.")

    probabilities = weights / total
    raw = probabilities * epoch_size
    budgets = raw.long()
    remaining = epoch_size - int(budgets.sum().item())

    if remaining > 0:
        fractions = raw - budgets
        order = torch.argsort(fractions, descending=True, stable=True)
        budgets[order[:remaining]] += 1

    if int(budgets.sum().item()) != epoch_size:
        raise RuntimeError("Allocated child budgets do not sum to epoch_size.")
    return budgets


# ── Pool sampling ──────────────────────────────────────────────────────────


def sample_pool_without_replacement_cycles(
    pool: torch.Tensor,
    budget: int,
    *,
    make_generator: Callable[[int], torch.Generator],
) -> torch.Tensor:
    """Fill *budget* from *pool* with cyclic without-replacement."""
    pool_size = int(pool.numel())
    if pool_size <= 0:
        raise ValueError("Cannot sample from an empty pool.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if budget == 0:
        return torch.empty(0, dtype=torch.long)

    full_passes, remainder = divmod(budget, pool_size)
    outputs: List[torch.Tensor] = []

    for cycle_id in range(full_passes):
        g = make_generator(cycle_id)
        permutation = torch.randperm(pool_size, generator=g)
        outputs.append(pool[permutation])

    if remainder > 0:
        g = make_generator(full_passes)
        permutation = torch.randperm(pool_size, generator=g)
        outputs.append(pool[permutation[:remainder]])

    return torch.cat(outputs)


def sample_pool_with_replacement(
    pool: torch.Tensor,
    budget: int,
    *,
    generator: torch.Generator,
    probabilities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fill *budget* from *pool* with weighted multinomial replacement."""
    pool_size = int(pool.numel())
    if pool_size <= 0:
        raise ValueError("Cannot sample from an empty pool.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if budget == 0:
        return torch.empty(0, dtype=torch.long)

    if probabilities is None:
        probabilities = torch.ones(pool_size, dtype=torch.float64)
    else:
        probabilities = probabilities.to(dtype=torch.float64, device="cpu")

    probabilities = probabilities / probabilities.sum()
    local_indices = torch.multinomial(
        probabilities, num_samples=budget, replacement=True, generator=generator,
    )
    return pool[local_indices]


def sample_pool(
    pool: torch.Tensor,
    budget: int,
    *,
    replacement: bool,
    generator_factory: Callable[[int], torch.Generator],
    probabilities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified pool sampler dispatching on *replacement*."""
    if replacement:
        return sample_pool_with_replacement(
            pool, budget, generator=generator_factory(0), probabilities=probabilities,
        )
    return sample_pool_without_replacement_cycles(
        pool, budget, make_generator=generator_factory,
    )


# ── Task target distribution ──────────────────────────────────────────────


def compute_target_task_probabilities(
    task_counts: torch.Tensor,
    *,
    gamma: float,
    mix: float,
) -> torch.Tensor:
    """Gamma-compressed, mix-blended task target distribution."""
    counts = task_counts.to(dtype=torch.float64)
    total = counts.sum().item()
    if total <= 0:
        raise ValueError("task_counts must have positive total.")

    p_data = counts / total
    scores = p_data ** gamma
    p_balanced = scores / scores.sum()
    p_target = (1.0 - mix) * p_balanced + mix * p_data
    p_target = p_target / p_target.sum()
    return p_target


# ── Task quota allocation ──────────────────────────────────────────────────


def apply_task_quota_caps(
    target_probabilities: torch.Tensor,
    task_counts: torch.Tensor,
    fine_budget: int,
    max_repeat: float,
) -> torch.Tensor:
    """Water-filling: cap tasks at *max_repeat × task_count*, redistribute."""
    target_probabilities = target_probabilities.to(dtype=torch.float64)
    max_quotas = task_counts.to(dtype=torch.float64) * float(max_repeat)
    num_tasks = len(task_counts)

    active = torch.ones(num_tasks, dtype=torch.bool)
    quota_float = torch.zeros(num_tasks, dtype=torch.float64)
    remaining_budget = float(fine_budget)

    while active.any():
        active_prob = target_probabilities[active]
        prob_sum = active_prob.sum()
        if prob_sum <= 0:
            raise RuntimeError(
                "No positive task probability remains while budget is still unallocated."
            )
        proposal = remaining_budget * active_prob / prob_sum
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        active_caps = max_quotas[active]
        exceeds = proposal > active_caps

        if not exceeds.any():
            quota_float[active_indices] = proposal
            remaining_budget = 0.0
            break

        capped_indices = active_indices[exceeds]
        quota_float[capped_indices] = max_quotas[capped_indices]
        used = quota_float[capped_indices].sum()
        remaining_budget -= float(used.item())
        active[capped_indices] = False

    if remaining_budget > 1e-6:
        raise ValueError("Task repeat caps cannot accommodate the requested fine budget.")
    return quota_float


def integerize_quotas(
    quota_float: torch.Tensor,
    total_budget: int,
) -> torch.Tensor:
    """Largest-remainder integerisation of float quotas."""
    quotas = quota_float.long()
    remaining = total_budget - int(quotas.sum().item())
    if remaining < 0:
        raise RuntimeError("Floored quotas exceed total budget.")
    if remaining > 0:
        fractions = quota_float - quotas
        order = torch.argsort(fractions, descending=True, stable=True)
        quotas[order[:remaining]] += 1
    if int(quotas.sum().item()) != total_budget:
        raise RuntimeError("Task quotas do not sum to fine budget.")
    return quotas


# ── Epoch plan builder ─────────────────────────────────────────────────────


def build_epoch_indices(
    specs: List[ChildSamplingSpec],
    *,
    epoch_size: int,
    epoch: int,
    seed: int,
    replacement: bool,
) -> torch.Tensor:
    """Build complete epoch index tensor."""

    weights = torch.tensor(
        [s.normalized_weight for s in specs], dtype=torch.float64,
    )
    child_budgets = allocate_exact_budgets(weights, epoch_size)
    child_outputs: List[torch.Tensor] = []

    for child_id, (spec, b_tensor) in enumerate(zip(specs, child_budgets)):
        budget = int(b_tensor.item())

        if spec.fine_task_spec is not None:
            ts = spec.fine_task_spec
            target_probabilities = compute_target_task_probabilities(
                task_counts=ts.task_counts, gamma=ts.gamma, mix=ts.mix,
            )
            quota_float = apply_task_quota_caps(
                target_probabilities, ts.task_counts, budget, ts.max_expected_repeat,
            )
            task_quotas = integerize_quotas(quota_float, budget)

            task_outputs: List[torch.Tensor] = []
            for task_id, q_tensor in enumerate(task_quotas):
                quota = int(q_tensor.item())
                if quota <= 0:
                    continue
                task_pool = ts.indices_by_task[task_id]
                selected = sample_pool_without_replacement_cycles(
                    task_pool, quota,
                    make_generator=_make_generator_factory(seed, epoch, child_id, task_id),
                )
                task_outputs.append(selected + spec.global_start)
            child_indices = torch.cat(task_outputs) if task_outputs else torch.empty(0, dtype=torch.long)

        else:
            child_pool = torch.arange(
                spec.global_start, spec.global_end, dtype=torch.long,
            )
            child_indices = sample_pool(
                child_pool, budget,
                replacement=replacement,
                generator_factory=_make_generator_factory(seed, epoch, child_id, -1),
            )

        if len(child_indices) != budget:
            raise RuntimeError(
                f"Child {spec.name} produced {len(child_indices)} indices, expected {budget}."
            )
        child_outputs.append(child_indices)

    all_indices = torch.cat(child_outputs)
    if len(all_indices) != epoch_size:
        raise RuntimeError(f"Epoch indices length {len(all_indices)} != epoch_size {epoch_size}.")

    # Global shuffle
    shuffle_g = torch.Generator(device="cpu")
    shuffle_g.manual_seed(stable_seed(seed, epoch, -1, -1, 0))
    permutation = torch.randperm(len(all_indices), generator=shuffle_g)
    return all_indices[permutation]


# ── Sampler ────────────────────────────────────────────────────────────────


class ResumableEpochSampler(Sampler[int]):
    """Mixed-dataset epoch sampler with task-quota cyclic sampling."""

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        replacement: bool = False,
        sampling_specs: Optional[List[ChildSamplingSpec]] = None,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.replacement = bool(replacement)
        self.sampling_specs = sampling_specs
        self.epoch = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        if self.sampling_specs is None:
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            all_indices = build_epoch_indices(
                self.sampling_specs,
                epoch_size=len(self.dataset),
                epoch=self.epoch,
                seed=self.seed,
                replacement=self.replacement,
            )
            indices = all_indices.tolist()

        if self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
