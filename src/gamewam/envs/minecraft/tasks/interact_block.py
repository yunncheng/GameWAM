"""Minecraft interact block task generator."""
import json
import random

from gamewam.envs.minecraft.assets import INTERACT_BLOCK_SPAWN_FILE
from gamewam.envs.minecraft.tasks.catalog import EVENT_DESCRIPTION

with open(INTERACT_BLOCK_SPAWN_FILE, "r") as _f:
    _interact_block_spawn_dict = json.load(_f)


def get_available_interact_block_tasks():
    return list(_interact_block_spawn_dict.keys())


def gen_interact_block_task_config(task_name, difficulty="zero", seed: int = 0):
    spawn_data = _interact_block_spawn_dict[task_name]
    rng = random.Random(int(seed))

    if task_name in EVENT_DESCRIPTION:
        task_description = rng.choice(EVENT_DESCRIPTION[task_name])
    else:
        task_description = task_name.replace(":", " ").replace("_", " ")

    seed_config = rng.choice(spawn_data['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']

    commands = [f"/tp @s {' '.join(str(p) for p in seed_position)}"]
    for block in spawn_data["block"]:
        block_name = list(block.keys())[0]
        pos = list(block.values())[0]
        commands.append(f"/setblock ^{pos[0]} ^{pos[1]} ^{pos[2]} minecraft:{block_name}")

    reward_cfg = [{
        "event": "custom",
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
                "init_inventory": [],
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": difficulty,
                "forbidden_slots": [0],
            },
            "commands": commands,
        },
        "rewards": reward_cfg,
    }
