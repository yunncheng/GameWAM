"""Initialize the Minecraft inventory after environment reset."""
import json
import random
import re
import logging
from copy import deepcopy
from pathlib import Path
from time import sleep
from typing import Sequence, Union

from minestudio.simulator.callbacks.callback import MinecraftCallback

from gamewam.envs.minecraft.assets import (
    ITEM_PREFIX,
    ITEM_PREFIX_LENGTH,
    TAG_ITEMS_FILE,
    get_mcd,
    load_json,
)

TAG_INFO = load_json(TAG_ITEMS_FILE)

logger = logging.getLogger(__name__)

EQUIP_SLOTS = {
    "mainhand": 0, "offhand": 40,
    "head": 39, "chest": 38, "legs": 37, "feet": 36,
}
REVERSE_EQUIP_SLOTS_MAP = {v: k for k, v in EQUIP_SLOTS.items()}
MIN_SLOT_IDX = 0
MAX_INVENTORY_IDX = 35
MAX_SLOT_IDX = 40
SLOT_IDX2NAME = {v: k for k, v in EQUIP_SLOTS.items()}
MIN_ITEMS_NUM = 0
MAX_ITEMS_NUM = 64

INVENTORY_DISTRACTION_LEVEL = {
    "zero": [0], "one": [1],
    "easy": list(range(3, 7)), "middle": list(range(7, 16)),
    "hard": list(range(16, 36)), "normal": list(range(0, 36)),
}
EQUIP_DISTRACTION_LEVEL = {
    "zero": [0], "normal": list(range(0, 6)),
}


class InitInventoryCallback(MinecraftCallback):

    def __init__(
        self,
        init_inventory: list,
        inventory_distraction_level: Union[Sequence[int], str] = (0,),
        equip_distraction_level: Union[Sequence[int], str] = (0,),
        forbidden_slots=None,
        *,
        seed: int,
    ):
        self.init_inventory = init_inventory
        self.forbidden_slots = set(forbidden_slots or [])
        self.rng = random.Random(int(seed))

        if isinstance(inventory_distraction_level, str):
            if inventory_distraction_level not in INVENTORY_DISTRACTION_LEVEL:
                raise ValueError(
                    "Unknown inventory_distraction_level "
                    f"{inventory_distraction_level!r}."
                )
            self.inventory_distraction_level = INVENTORY_DISTRACTION_LEVEL[
                inventory_distraction_level
            ]
        else:
            self.inventory_distraction_level = inventory_distraction_level

        if isinstance(equip_distraction_level, str):
            if equip_distraction_level not in EQUIP_DISTRACTION_LEVEL:
                raise ValueError(
                    f"Unknown equip_distraction_level {equip_distraction_level!r}."
                )
            self.equip_distraction_level = EQUIP_DISTRACTION_LEVEL[
                equip_distraction_level
            ]
        else:
            self.equip_distraction_level = equip_distraction_level

        mcd = get_mcd()
        if mcd is not None:
            self.items_library = mcd.items_name
            self.items_names = list(mcd.items_name.keys())
        else:
            logger.warning("minecraft_data not installed. InitInventoryCallback item library unavailable.")
            self.items_library = {}
            self.items_names = []
        self.equipments_library = self._get_equipments_library()

    def after_reset(self, sim, obs, info):
        return self._set_inventory(sim, obs, info)

    def _set_inventory(self, sim, obs, info):
        chats = []
        visited_slots = set()
        uncertain_slots = []
        init_inventory = []

        for slot_info in self.init_inventory:
            slot = slot_info["slot"]
            if slot == "random":
                uncertain_slots.append(deepcopy(slot_info))
                continue
            visited_slots.add(int(slot))
            init_inventory.append(slot_info)

        unvisited_slots = set(range(MIN_SLOT_IDX, MAX_INVENTORY_IDX + 1)) - visited_slots
        unvisited_slots -= self.forbidden_slots

        for uncertain_slot in uncertain_slots:
            slot = int(self.rng.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            visited_slots.add(slot)
            uncertain_slot["slot"] = slot
            init_inventory.append(uncertain_slot)

        init_inventory, visited_slots, unvisited_slots = self._sample_inventory(
            init_inventory, visited_slots, unvisited_slots)
        init_inventory, visited_slots = self._sample_equipments(
            init_inventory, visited_slots)

        self.slot_num = len(init_inventory)
        expected_slots = {}

        for item_dict in init_inventory:
            slot = int(item_dict["slot"])
            mc_slot = self._map_slot_number_to_cmd_slot(slot)
            item_type = item_dict["type"]

            if item_type not in self.items_names:
                item_type = self.rng.choice(
                    TAG_INFO.get(ITEM_PREFIX + item_type, [ITEM_PREFIX + item_type])
                )[ITEM_PREFIX_LENGTH:]
                item_dict["type"] = item_type

            stack_size = int(self.items_library.get(item_type, {}).get("stackSize", 64))
            item_quantity = self._item_quantity_parser(item_dict["quantity"], stack_size)
            expected_slots[slot] = {
                "type": str(item_type).removeprefix("minecraft:"),
                "quantity": int(item_quantity),
            }

            chat = f"/replaceitem entity @p {mc_slot} minecraft:{item_type} {item_quantity}"
            if "metadata" in item_dict:
                chat += f" {item_dict['metadata']}"
            chats.append(chat)

        obs, reward, done, info = sim.env.execute_cmd("/gamerule sendCommandFeedback false")
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        obs, info = sim._wrap_obs_info(obs, info)

        # Keep a deterministic minimum tick budget before polling failures.
        _MIN_VALIDATION_STEPS = 20
        init_flag = False
        kdx = 0
        current_slot_num = 0
        mismatches = []
        for kdx in range(300):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)
            if not init_flag:
                init_flag, current_slot_num, mismatches = self._check(obs, expected_slots)
            if init_flag and kdx + 1 >= _MIN_VALIDATION_STEPS:
                break
            if kdx % 40 == 0 and not init_flag:
                sleep(1)

        obs, info = sim._wrap_obs_info(obs, info)

        if not init_flag:
            raise RuntimeError(
                "Failed to set exact init inventory: "
                f"expected_slots={expected_slots}, occupied_slots={current_slot_num}, "
                f"mismatches={mismatches[:8]} after {kdx + 1} steps."
            )

        return obs, info

    def _get_equipments_library(self):
        mc_equipments_file = Path(__file__).resolve().parents[1] / "assets" / "mc_equipments.1.16.json"
        if mc_equipments_file.exists():
            with mc_equipments_file.open("r") as f:
                return json.load(f)
        logger.debug("mc_equipments.1.16.json not found, equipment distraction disabled")
        return {}

    def _sample_inventory(self, init_inventory, visited_slots, unvisited_slots):
        if not self.items_names:
            return init_inventory, visited_slots, unvisited_slots
        distraction_num = min(self.rng.choice(self.inventory_distraction_level), len(unvisited_slots))
        for _ in range(distraction_num):
            item_type = self.rng.choice(self.items_names)
            slot = int(self.rng.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            visited_slots.add(slot)
            init_inventory.append({"slot": slot, "type": item_type, "quantity": "random"})
        return init_inventory, visited_slots, unvisited_slots

    def _sample_equipments(self, init_inventory, visited_slots):
        if not self.equipments_library:
            return init_inventory, visited_slots
        unvisited_equipments = set(range(MAX_INVENTORY_IDX + 1, MAX_SLOT_IDX + 1)) - visited_slots
        sample_num = min(self.rng.choice(self.equip_distraction_level), len(unvisited_equipments))
        for _ in range(sample_num):
            slot = self.rng.choice(list(unvisited_equipments))
            equipment_name = REVERSE_EQUIP_SLOTS_MAP[slot]
            unvisited_equipments.remove(slot)
            visited_slots.add(slot)
            item_type = self.rng.choice(self.equipments_library.get(equipment_name, ["air"]))
            if slot == 40 and self.rng.uniform(0, 1) > 0.25 and self.items_names:
                item_type = self.rng.choice(self.items_names)
            init_inventory.append({"slot": slot, "type": item_type, "quantity": 1})
        return init_inventory, visited_slots

    @staticmethod
    def _map_slot_number_to_cmd_slot(slot_number: Union[int, str]) -> str:
        slot_number = int(slot_number)
        assert MIN_SLOT_IDX <= slot_number <= MAX_SLOT_IDX, f"Slot index out of range: {slot_number}"
        if slot_number in {0, 40}:
            return f"weapon.{SLOT_IDX2NAME[slot_number]}"
        elif 36 <= slot_number <= 39:
            return f"armor.{SLOT_IDX2NAME[slot_number]}"
        elif 1 <= slot_number <= 8:
            return f"hotbar.{slot_number}"
        else:
            return f"inventory.{slot_number - 9}"

    def _item_quantity_parser(self, item_quantity: Union[int, str], max_items_num: int, one_p: float = 0.7) -> int:
        if isinstance(item_quantity, str):
            candidate_nums = set(range(MIN_ITEMS_NUM + 1, max_items_num + 1))
            if item_quantity == "random":
                one_flag = self.rng.choices([True, False], weights=[one_p, 1 - one_p], k=1)[0]
                return 1 if one_flag else self.rng.choice(list(candidate_nums))

            def apply_command(op, val):
                return {
                    '<': set(range(MIN_ITEMS_NUM, val)),
                    '<=': set(range(MIN_ITEMS_NUM, val + 1)),
                    '>': set(range(val + 1, max_items_num + 1)),
                    '>=': set(range(val, max_items_num + 1)),
                    '==': {val},
                }[op]

            for cmd in item_quantity.split(","):
                match = re.search(r'([<>]=?|==)\s*(\d+)', cmd.strip())
                if match:
                    operator, number = match.groups()
                    candidate_nums &= apply_command(operator, int(number))
            return self.rng.choice(list(candidate_nums)) if candidate_nums else 1

        elif not isinstance(item_quantity, int):
            raise TypeError("item_quantity must be an integer or a string")
        return item_quantity

    @staticmethod
    def _check(obs, expected_slots):
        """Validate exact slot, item type and quantity after reset commands."""
        current_slot_num = 0
        for slot_dict in obs["inventory"].values():
            if slot_dict["type"] != "none":
                current_slot_num += 1
        for slot_name, slot_dict in obs["equipped_items"].items():
            if slot_name != "mainhand" and slot_dict["type"] != "air":
                current_slot_num += 1
        mismatches = []
        for slot, expected in sorted(expected_slots.items()):
            if 0 <= slot <= MAX_INVENTORY_IDX:
                actual = obs["inventory"].get(slot)
                if actual is None:
                    actual = obs["inventory"].get(str(slot))
            else:
                slot_name = REVERSE_EQUIP_SLOTS_MAP.get(slot)
                actual = obs["equipped_items"].get(slot_name) if slot_name else None
            if actual is None:
                mismatches.append({"slot": slot, "reason": "missing observation"})
                continue
            actual_type = str(actual.get("type", "")).removeprefix("minecraft:")
            # Equipped-item observations do not report stack quantity.
            quantity_observable = slot <= MAX_INVENTORY_IDX
            actual_quantity = (
                int(actual.get("quantity", 0)) if quantity_observable else None
            )
            quantity_mismatch = (
                quantity_observable
                and actual_quantity != expected["quantity"]
            )
            if actual_type != expected["type"] or quantity_mismatch:
                mismatches.append({
                    "slot": slot,
                    "expected": expected,
                    "actual": {
                        "type": actual_type,
                        "quantity": actual_quantity,
                    },
                })

        if current_slot_num != len(expected_slots):
            mismatches.append({
                "reason": "occupied slot count mismatch",
                "expected": len(expected_slots),
                "actual": current_slot_num,
            })
        return not mismatches, current_slot_num, mismatches
