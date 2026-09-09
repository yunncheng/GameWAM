"""Minecraft evaluation lifecycle and cleanup contracts.

The Ray driver must not infer failure provenance from ``RayTaskError`` text:
that text also contains Ray, NCCL, filesystem, and CUDA errors. Classification
therefore happens inside the episode worker while the original exception and
failing operation are still known. Only an explicit JSON-safe sentinel is sent
back to the driver.
"""

from __future__ import annotations

import os
import time
from typing import Any, MutableMapping, MutableSequence, TypeVar


RETRYABLE_MC_ENV_ERROR = "retryable_mc_env_error"

_JobT = TypeVar("_JobT")

_MC_ENV_STAGES = frozenset({"build_environment", "reset", "env_step"})
_FATAL_ERROR_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "could not detect an x server",
    "could not detect any opengl libraries",
    "action space does not cover",
    "has no keyboard mapping",
)
_TRANSIENT_ERROR_MARKERS = (
    "attempted to step an environment server with done=true",
    "something went wrong resetting the environment",
    "minecraft process finished unexpectedly",
    "malmo failed to start",
    "too long waiting for first observation",
    "connection timed out",
    "failed to reset (socket error)",
    "failed to start interactor",
    "no available instances",
)


def cleanup_minecraft_instances(
    *,
    parent_pid: int | None = None,
    logger: Any = None,
    attempts: int = 3,
    _database_manager: Any = None,
) -> int:
    """Remove MineStudio instances owned by a process or by dead processes."""
    if attempts <= 0:
        raise ValueError("attempts must be positive")

    if _database_manager is None:
        from minestudio.simulator.minerl.env import database_manager
    else:
        database_manager = _database_manager

    removed: set[str] = set()
    remaining: list[str] = []
    for attempt in range(attempts):
        candidates = []
        for instance_id in list(database_manager.database["instance_list"]):
            record = database_manager.database.get(instance_id)
            if record is None:
                continue
            if parent_pid is not None:
                should_remove = int(record.parent_pid) == int(parent_pid)
            else:
                owner_pid = int(record.parent_pid)
                if owner_pid < 1:
                    should_remove = True
                else:
                    try:
                        os.kill(owner_pid, 0)
                    except OSError:
                        should_remove = True
                    else:
                        should_remove = False
            if should_remove:
                candidates.append(str(instance_id))

        if not candidates:
            remaining = []
            break
        for instance_id in candidates:
            database_manager.check_instance(instance_id, force_remove=True)
            if instance_id not in database_manager.database:
                removed.add(instance_id)
        remaining = [
            instance_id
            for instance_id in candidates
            if instance_id in database_manager.database
        ]
        if not remaining:
            break
        if attempt + 1 < attempts:
            time.sleep(0.2)

    if remaining:
        raise RuntimeError(
            f"Failed to remove {len(remaining)} MineStudio instance(s): {remaining}"
        )
    if removed and logger is not None:
        logger.debug("Removed %d stale Minecraft instance(s).", len(removed))
    return len(removed)


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_retryable_minecraft_error(exc: BaseException, stage: str) -> bool:
    """Return whether ``exc`` is a transient MineStudio/Malmo failure."""
    if stage not in _MC_ENV_STAGES:
        return False

    chain = tuple(_exception_chain(exc))
    messages = tuple(str(item).lower() for item in chain)
    if any(marker in message for message in messages for marker in _FATAL_ERROR_MARKERS):
        return False

    if any(
        isinstance(item, (ConnectionError, TimeoutError, BrokenPipeError, EOFError))
        for item in chain
    ):
        return True
    if any(
        type(item).__name__ in {"MissionInitException", "MinecraftEnvironmentError"}
        for item in chain
    ):
        return True
    return any(
        marker in message
        for message in messages
        for marker in _TRANSIENT_ERROR_MARKERS
    )


def make_retryable_mc_result(
    *,
    task_name: str,
    episode_id: str,
    stage: str,
    exc: BaseException,
    steps: int,
    traceback_text: str,
) -> dict[str, Any]:
    """Build the explicit worker-to-driver transient-failure sentinel."""
    return {
        "status": RETRYABLE_MC_ENV_ERROR,
        "task_name": str(task_name),
        "episode_id": str(episode_id),
        "stage": str(stage),
        "steps": int(steps),
        "error_type": type(exc).__name__,
        "error_msg": str(exc),
        "traceback": str(traceback_text),
    }


def is_retryable_mc_result(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == RETRYABLE_MC_ENV_ERROR


def is_completed_episode_result(result: Any) -> bool:
    """Validate an ``end.json`` row and decide whether it is terminal.

    Rows shaped as ``[false, 0, episode_id]`` represent infrastructure or
    bootstrap failures rather than completed rollouts. Completed rollouts
    execute at least one environment step, so zero-step rows remain pending.
    """
    if not isinstance(result, (list, tuple)) or len(result) != 3:
        raise ValueError(f"Malformed Minecraft end.json row: {result!r}")
    success, steps, episode_id = result
    if not isinstance(success, bool):
        raise ValueError(f"Invalid success value in Minecraft end.json row: {result!r}")
    try:
        steps_i = int(steps)
        int(str(episode_id))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed Minecraft end.json row: {result!r}") from exc
    if steps_i < 0:
        raise ValueError(f"Negative step count in Minecraft end.json row: {result!r}")
    return steps_i > 0


def consume_retry(
    counts: MutableMapping[tuple[str, str], int],
    task_name: str,
    episode_id: str,
    *,
    max_retries: int,
) -> tuple[bool, int]:
    """Consume retry budget and return ``(requeue, failure_count)``."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    key = (str(task_name), str(episode_id))
    failures = counts.get(key, 0) + 1
    counts[key] = failures
    return failures <= max_retries, failures


def enqueue_retry_next(
    queue: MutableSequence[_JobT],
    next_index: int,
    job: _JobT,
) -> None:
    """Place a retry at the next unconsumed position in an indexed queue."""
    if not 0 <= next_index <= len(queue):
        raise IndexError(
            f"next_index={next_index} is outside queue insertion range "
            f"[0, {len(queue)}]"
        )
    queue.insert(next_index, job)
