"""Minecraft task discovery and deterministic configuration."""
import random
import re
import logging
from typing import Dict, Any

from .mine_block import (
    get_available_mine_block_tasks, gen_mine_block_task_config)
from .craft_item import (
    get_available_craft_item_tasks, gen_craft_item_task_config)
from .kill_entity import (
    get_available_kill_entity_tasks, gen_kill_entity_task_config)
from .smelt_item import (
    get_available_smelt_item_tasks, gen_smelt_item_task_config)
from .interact_block import (
    get_available_interact_block_tasks, gen_interact_block_task_config)

logger = logging.getLogger(__name__)


def get_available_tasks():
    """Get all available tasks across all registered types."""
    all_tasks = []
    all_tasks.extend(get_available_mine_block_tasks())
    all_tasks.extend(get_available_kill_entity_tasks())
    all_tasks.extend(get_available_smelt_item_tasks())
    all_tasks.extend(get_available_craft_item_tasks())
    all_tasks.extend(get_available_interact_block_tasks())
    return all_tasks


def gen_task_config(task_name_pattern: str, difficulty: str = "zero", seed: int = 0):
    """Generate a deterministic task configuration."""
    if task_name_pattern in get_available_craft_item_tasks():
        return gen_craft_item_task_config(task_name_pattern, difficulty=difficulty, seed=seed)
    if task_name_pattern in get_available_smelt_item_tasks():
        return gen_smelt_item_task_config(task_name_pattern, difficulty=difficulty, seed=seed)

    if "smelt_item" in task_name_pattern:
        return gen_smelt_item_task_config(task_name_pattern, difficulty=difficulty, seed=seed)
    if "kill_entity" in task_name_pattern:
        return gen_kill_entity_task_config(task_name_pattern, difficulty=difficulty, seed=seed)
    if "mine_block" in task_name_pattern:
        return gen_mine_block_task_config(task_name_pattern, difficulty=difficulty, seed=seed)
    if "interact_with" in task_name_pattern:
        return gen_interact_block_task_config(task_name_pattern, difficulty=difficulty, seed=seed)

    raise NameError(f"Unknown task pattern: {task_name_pattern}")


def choose_available_task(
    task_name_pattern: str,
    difficulty: str = "zero",
    seed: int = 0,
) -> Dict[str, Any]:
    """Choose a task matching the given pattern.

    Supports:
    - "random" — pick any task
    - Comma-separated patterns — pick one pattern randomly
    - Wildcard "*" — glob matching (e.g. "mine_block:*")

    Every random choice uses a local generator, so global RNG state cannot
    affect task descriptions, spawn positions, or tool selection.
    """
    if not isinstance(task_name_pattern, str):
        raise TypeError(f"task_name_pattern must be a string, got {type(task_name_pattern).__name__}")

    rng = random.Random(int(seed))

    all_available_tasks = get_available_tasks()

    if task_name_pattern == "random":
        task_name_pattern = rng.choice(all_available_tasks)
    else:
        patterns = [p.strip() for p in task_name_pattern.split(",")]
        chosen_pattern = rng.choice(patterns)

        matched_tasks = []
        regex = "^" + re.escape(chosen_pattern).replace(r"\*", ".*") + "$"
        compiled = re.compile(regex)
        for task_name in all_available_tasks:
            if compiled.match(task_name):
                matched_tasks.append(task_name)

        if chosen_pattern == "smelt_item*":
            matched_tasks = [
                t.replace("craft_item", "smelt_item")
                for t in get_available_smelt_item_tasks()
            ]

        if not matched_tasks:
            raise ValueError(f"No task matched pattern(s): {task_name_pattern}")

        task_name_pattern = rng.choice(matched_tasks)

    return gen_task_config(task_name_pattern, difficulty=difficulty, seed=seed)
