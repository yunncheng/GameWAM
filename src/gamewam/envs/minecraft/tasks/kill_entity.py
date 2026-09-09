"""Minecraft kill entity task generator."""
import json
import random

from gamewam.envs.minecraft.assets import KILL_ENTITY_SPAWN_FILE
from gamewam.envs.minecraft.tasks.catalog import EVENT_DESCRIPTION

with open(KILL_ENTITY_SPAWN_FILE, "r") as _f:
    _kill_entity_spawn_dict = json.load(_f)


def get_available_kill_entity_tasks():
    available = []
    for name, data in _kill_entity_spawn_dict.items():
        if "normal" in data.get("label", []):
            available.append(name)
    return available


def gen_kill_entity_task_config(task_name, difficulty="zero", seed: int = 0):
    spawn_data = _kill_entity_spawn_dict[task_name]
    rng = random.Random(int(seed))

    if task_name in EVENT_DESCRIPTION:
        task_description = rng.choice(EVENT_DESCRIPTION[task_name])
    else:
        task_description = task_name.replace(":", " ").replace("_", " ")

    seed_config = rng.choice(spawn_data['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']

    commands = [f"/tp @s {' '.join(str(p) for p in seed_position)}"]

    init_inventory = []
    tool_list = spawn_data['tool']
    if tool_list:
        if difficulty == "zero":
            tool_list = [t for t in tool_list if t != ""]
            assert tool_list, f"No valid tools for {task_name} at difficulty=zero"
        tool = rng.choice(tool_list)
        if tool and tool != "air":
            slot = 0 if difficulty != "hard" else rng.choice(list(range(0, 9)))
            init_inventory = [{'slot': slot, 'type': tool, 'quantity': 1}]

    mobs = [{
        "name": task_name.split(":")[-1],
        "number": 1,
        "range_x": [-1, 1],
        "range_z": [2, 7],
    }]
    commands.append(spawn_data["commands"])

    reward_cfg = [{
        "event": "kill_entity",
        "max_reward_times": 1,
        "reward": 1.0,
        "objects": [task_name.split(":")[-1]],
        "identity": task_name,
    }]

    return {
        "task_name": task_name,
        "task_description": task_description,
        "seed": seed_number,
        "init_actions": [],
        "maximum_steps": 600,
        "callback": {
            "init_inventory": {
                "init_inventory": init_inventory,
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": "normal",
            },
            "commands": commands,
            "mobs": mobs,
        },
        "rewards": reward_cfg,
    }
