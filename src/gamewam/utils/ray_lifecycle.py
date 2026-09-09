"""Helpers for terminating Ray actor pools without leaking driver references."""

from __future__ import annotations

from collections.abc import MutableSequence
import math
from typing import Any


def packed_gpu_actor_capacity(gpu_count: int, gpus_per_actor: float) -> int:
    """Return Ray's per-node actor capacity for a uniform GPU request.

    Fractional requests at or below one GPU must fit on each physical GPU;
    unused fractions on different devices cannot be combined. Requests above
    one GPU instead consume from the node-wide pool.
    """
    gpu_count = int(gpu_count)
    gpus_per_actor = float(gpus_per_actor)
    if gpu_count < 0:
        raise ValueError(f"gpu_count must be non-negative, got {gpu_count}")
    if not math.isfinite(gpus_per_actor) or gpus_per_actor <= 0:
        raise ValueError(
            f"gpus_per_actor must be finite and positive, got {gpus_per_actor}"
        )
    if gpu_count == 0:
        return 0

    # The small tolerance avoids under-counting exact decimal divisors such
    # as 0.2 due only to their binary floating-point representation.
    tolerance = 1e-12
    if gpus_per_actor <= 1.0:
        actors_per_gpu = math.floor((1.0 + tolerance) / gpus_per_actor)
        return gpu_count * actors_per_gpu
    return math.floor((gpu_count + tolerance) / gpus_per_actor)


def wait_actor_pool_ready(
    ray_module: Any,
    actors: MutableSequence[Any],
    *,
    timeout_s: float,
) -> None:
    """Wait until every actor constructor has completed successfully."""
    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be finite and positive, got {timeout_s}")
    if not actors:
        return

    ready_refs = [actor.ready.remote() for actor in actors]
    ready, pending = ray_module.wait(
        ready_refs,
        num_returns=len(ready_refs),
        timeout=timeout_s,
    )
    if pending:
        raise TimeoutError(
            f"Timed out after {timeout_s:g}s waiting for "
            f"{len(pending)}/{len(ready_refs)} Ray actor(s) to initialize"
        )
    for ref in ready:
        if ray_module.get(ref) is not True:
            raise RuntimeError("Ray actor readiness probe returned an invalid response")


def shutdown_actor_pool(
    ray_module: Any,
    actors: MutableSequence[Any],
    *,
    close_timeout_s: float = 30.0,
    logger: Any = None,
) -> None:
    """Close actor-owned resources, terminate actors, then stop Ray."""
    actor_handles = list(actors)
    close_refs = []
    close_timeout_s = max(float(close_timeout_s), 0.0)

    if close_timeout_s > 0:
        for actor in actor_handles:
            try:
                close_refs.append(actor.shutdown.remote())
            except Exception as exc:
                if logger is not None:
                    logger.warning("Failed to request Ray actor shutdown: %s", exc)

    if close_refs:
        try:
            ready, pending = ray_module.wait(
                close_refs,
                num_returns=len(close_refs),
                timeout=close_timeout_s,
            )
            for ref in ready:
                try:
                    ray_module.get(ref)
                except Exception as exc:
                    if logger is not None:
                        logger.warning("Ray actor resource cleanup failed: %s", exc)
            if pending and logger is not None:
                logger.warning(
                    "Timed out waiting for %d Ray actor(s) to release resources; "
                    "forcing termination.",
                    len(pending),
                )
        except Exception as exc:
            if logger is not None:
                logger.warning("Ray actor cleanup wait failed: %s", exc)

    for actor in actor_handles:
        try:
            ray_module.kill(actor, no_restart=True)
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to terminate Ray actor: %s", exc)

    actor = None
    actors.clear()
    actor_handles.clear()
    close_refs.clear()
    ray_module.shutdown()
