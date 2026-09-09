"""Minecraft craft item task generator."""
import json
import random

from gamewam.envs.minecraft.assets import CRAFT_ITEM_SPAWN_FILE, OPEN_GUI_ACTIONS_FILE
from gamewam.envs.minecraft.tasks.catalog import EVENT_DESCRIPTION

with open(CRAFT_ITEM_SPAWN_FILE, "r") as _f:
    _craft_item_spawn_dict = json.load(_f)

with open(OPEN_GUI_ACTIONS_FILE, "r") as _f:
    _open_gui_actions = json.load(_f)


def get_available_craft_item_tasks():
    return list(_craft_item_spawn_dict.keys())


def gen_craft_item_task_config(task_name, difficulty="zero", seed: int = 0):
    rng = random.Random(int(seed))

    if task_name in EVENT_DESCRIPTION:
        task_description = rng.choice(EVENT_DESCRIPTION[task_name])
    else:
        task_description = task_name.replace(":", " ").replace("_", " ")

    seed_config = rng.choice(_craft_item_spawn_dict[task_name]['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']

    init_actions = []
    forbidden_slots = []
    if _craft_item_spawn_dict[task_name]["need_crafting_table"]:
        key = f"{seed_number}_{seed_position[0]}_{seed_position[1]}_{seed_position[2]}"
        init_actions = _open_gui_actions["crafting_table"].get(key, [])
        forbidden_slots = [0]
    else:
        init_actions = _open_gui_actions["inventory"]["init"]

    goal_object = _craft_item_spawn_dict[task_name]["goal"]
    commands = [f"/tp @s {' '.join(str(p) for p in seed_position)}"]
    init_inventory = _craft_item_spawn_dict[task_name].get('init_inventory', [])

    reward_cfg = [{
        "event": "craft_item",
        "identity": f"craft {goal_object}",
        "objects": [goal_object],
        "reward": 1.0,
        "max_reward_times": 1,
    }]

    return {
        "task_name": task_name,
        "task_description": task_description,
        "seed": seed_number,
        "init_actions": init_actions,
        "maximum_steps": 600,
        "callback": {
            "init_inventory": {
                "init_inventory": init_inventory,
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": difficulty,
                "forbidden_slots": forbidden_slots,
            },
            "commands": commands,
        },
        "rewards": reward_cfg,
    }
