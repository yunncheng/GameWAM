"""Paths and loaders for Minecraft environment resources."""

import json
from pathlib import Path
from typing import Any

FILE_DIR = Path(__file__).parent
SPAWNS_DIR = FILE_DIR / "spawns"

MINE_BLOCK_SPAWN_FILE = SPAWNS_DIR / "mine_block.json"
KILL_ENTITY_SPAWN_FILE = SPAWNS_DIR / "kill_entity.json"
INTERACT_BLOCK_SPAWN_FILE = SPAWNS_DIR / "interact_block.json"
CRAFT_ITEM_SPAWN_FILE = SPAWNS_DIR / "craft_item.json"
SMELT_ITEM_SPAWN_FILE = SPAWNS_DIR / "smelt_item.json"

TAG_ITEMS_FILE = FILE_DIR / "tag_items.json"
EVENT_DESCRIPTION_FILE = FILE_DIR / "event_description.json"
OPEN_GUI_ACTIONS_FILE = FILE_DIR / "open_gui_actions.json"
GUI_CURSOR_FILE = FILE_DIR / "mouse_cursor_white_16x16.png"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


_minecraft_data = None


def get_mcd():
    global _minecraft_data
    if _minecraft_data is None:
        import minecraft_data

        _minecraft_data = minecraft_data("1.16")
    return _minecraft_data


ITEM_PREFIX = "minecraft:"
ITEM_PREFIX_LENGTH = len(ITEM_PREFIX)
