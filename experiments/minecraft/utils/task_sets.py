"""Minecraft evaluation task-set definitions and helpers."""

import json
from typing import Iterable, List, Optional

from gamewam.envs.minecraft.tasks.manager import get_available_tasks

TASK_SET_CHOICES = ("custom", "mini", "all")

# Task-type → category mapping
TASK_CATEGORIES = {
    "mine_block": "Embodied",
    "smelt_item": "GUI",
    "craft_item": "GUI",
    "interact_block": "GUI",
    "custom": "GUI",  # interact_with_xxx tasks use 'custom:' prefix
    "kill_entity": "Combat",
}

# Mini benchmark set: 10 mine_block + 10 craft_item + 10 kill_entity
MINI_TASKS = (
    "mine_block:crafting_table", "mine_block:sugar_cane", "mine_block:hay_block",
    "mine_block:oak_log", "mine_block:grass_block", "mine_block:sand",
    "mine_block:dirt", "mine_block:oak_leaves", "mine_block:birch_log",
    "mine_block:chest",
    "kill_entity:salmon", "kill_entity:parrot", "kill_entity:sheep",
    "kill_entity:zombie", "kill_entity:cod", "kill_entity:chicken",
    "kill_entity:iron_golem", "kill_entity:slime", "kill_entity:villager",
    "kill_entity:cow",
    "craft_item:honeycomb_block", "craft_item:sea_lantern", "smelt_item:netherite_scrap",
    "smelt_item:cooked_salmon", "craft_item:paper", "craft_item:enchanting_table",
    "smelt_item:cooked_cod", "craft_item:emerald", "craft_item:arrow",
    "craft_item:shulker_box",
)
MINI_TASKS_SET = set(MINI_TASKS)

_SMELT_KEYS = None


def _get_smelt_keys():
    global _SMELT_KEYS
    if _SMELT_KEYS is None:
        _SMELT_KEYS = set()
        try:
            from gamewam.envs.minecraft.assets import SMELT_ITEM_SPAWN_FILE
            with open(str(SMELT_ITEM_SPAWN_FILE), "r", encoding="utf-8") as f:
                _SMELT_KEYS = set(json.load(f).keys())
        except (ImportError, FileNotFoundError):
            pass
    return _SMELT_KEYS



def task_category(task_name: str) -> str:
    """Return the paper category for a task name."""
    prefix = task_name.split(":")[0]
    return TASK_CATEGORIES.get(prefix, "Other")



def normalize_tasks(tasks: Iterable[str]) -> List[str]:
    """Strip empty task names while preserving order and uniqueness."""
    normalized = []
    seen = set()
    for task_name in tasks:
        task_name = task_name.strip()
        if not task_name or task_name in seen:
            continue
        normalized.append(task_name)
        seen.add(task_name)
    return normalized



def get_mini_tasks() -> List[str]:
    """Return the canonical mini benchmark task list in a stable order."""
    return list(MINI_TASKS)



def get_all_tasks() -> List[str]:
    """Return all registered Minecraft evaluation tasks."""
    return normalize_tasks(get_available_tasks())



def resolve_tasks(task_set: str, tasks_arg: Optional[str] = None) -> List[str]:
    """Resolve task selection from a named task set or explicit CSV."""
    if task_set not in TASK_SET_CHOICES:
        raise ValueError(f"Unknown task_set={task_set!r}. Expected one of {TASK_SET_CHOICES}.")
    if task_set == "mini":
        return get_mini_tasks()
    if task_set == "all":
        return get_all_tasks()
    if tasks_arg is None or not tasks_arg.strip():
        raise ValueError("--tasks is required when --task_set=custom")
    return normalize_tasks(tasks_arg.split(","))
