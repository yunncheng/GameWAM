"""Minecraft mine-block task generator."""
import json
import random

from gamewam.envs.minecraft.assets import MINE_BLOCK_SPAWN_FILE
from gamewam.envs.minecraft.tasks.catalog import EVENT_DESCRIPTION

with open(MINE_BLOCK_SPAWN_FILE, "r") as _f:
    _mine_block_spawn_dict = json.load(_f)


def get_available_mine_block_tasks():
    return list(_mine_block_spawn_dict.keys())


def gen_mine_block_task_config(task_name, difficulty="zero", seed: int = 0):
    rng = random.Random(int(seed))

    if task_name in EVENT_DESCRIPTION:
        task_description = rng.choice(EVENT_DESCRIPTION[task_name])
    else:
        task_description = task_name.replace(":", " ").replace("_", " ")

    seed_config = rng.choice(_mine_block_spawn_dict[task_name]['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']

    place_cfg = _mine_block_spawn_dict[task_name].get("place")

    if place_cfg:
        commands = [f"/tp @s {' '.join(str(p) for p in seed_position)} -90 0"]

        target = place_cfg["block"]
        support = place_cfg.get("support")
        # Relative coordinates remain valid if terrain changes the landing height.
        if support:
            commands.append(f"/setblock ^0 ^0 ^2 minecraft:{support}")
            if target == "vine":
                commands.append(f"/setblock ^0 ^1 ^3 minecraft:{support}")
            elif target == "kelp":
                commands.append("/setblock ^0 ^1 ^2 minecraft:water")
        if target.endswith("_bed"):
            # Complete both bed halves so mining updates the task statistic.
            commands.append(f"/setblock ^0 ^1 ^2 minecraft:{target}[part=foot,facing=east]")
            commands.append(f"/setblock ^0 ^1 ^3 minecraft:{target}[part=head,facing=east]")
            if support:
                commands.append(f"/setblock ^0 ^0 ^3 minecraft:{support}")
        elif target == "lilac":
            # Tall flowers require support plus both halves.
            commands.append(f"/setblock ^0 ^0 ^2 minecraft:dirt")
            commands.append(f"/setblock ^0 ^1 ^2 minecraft:{target}[half=lower]")
            commands.append(f"/setblock ^0 ^2 ^2 minecraft:{target}[half=upper]")
        elif target == "vine":
            commands.append(f"/setblock ^0 ^1 ^2 minecraft:{target}[east=true]")
        else:
            commands.append(f"/setblock ^0 ^1 ^2 minecraft:{target}")
    else:
        commands = [f"/tp @s {' '.join(str(p) for p in seed_position)}"]

    init_inventory = []
    tool_list = _mine_block_spawn_dict[task_name]['tool']
    if tool_list:
        if difficulty == "zero":
            tool_list = [t for t in tool_list if t != ""]
            assert tool_list, f"No valid tools for {task_name} at difficulty=zero"
        tool = rng.choice(tool_list)
        if tool and tool != "air":
            slot = 0 if difficulty != "hard" else rng.choice(list(range(0, 9)))
            init_inventory = [{'slot': slot, 'type': tool, 'quantity': 1}]

    reward_objects = [task_name.split(':')[-1]]

    reward_cfg = [{
        "event": "mine_block",
        "identity": f"mine {task_name.split(':')[-1]} blocks",
        "objects": reward_objects,
        "reward": 1.0,
        "max_reward_times": 1,
    }]

    noop = {
        "attack": 0, "back": 0, "forward": 0, "jump": 0, "left": 0, "right": 0,
        "sneak": 0, "sprint": 0, "use": 0, "drop": 0, "inventory": 0,
        "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0,
        "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0, "camera": [0.0, 0.0],
    }

    return {
        "task_name": task_name,
        "task_description": task_description,
        "seed": seed_number,
        "init_actions": [noop] * 2,
        "maximum_steps": 600,
        "callback": {
            "init_inventory": {
                "init_inventory": init_inventory,
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": "normal",
            },
            "commands": commands,
        },
        "rewards": reward_cfg,
    }
