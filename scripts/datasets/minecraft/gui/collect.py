"""Scripted expert GUI data collection for Minecraft craft_item and smelt_item tasks."""

import argparse
import copy
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MINESTUDIO_DIR", str(PROJECT_ROOT / ".minestudio"))

from gamewam.envs.minecraft.env import MinecraftGameEnv
from gamewam.envs.minecraft.action_space import action_to_vector, ENV_NULL_ACTION
from gamewam.envs.minecraft.observations.cursor import (
    DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL,
    DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL,
)
from gamewam.envs.minecraft.observations.state import (
    RAW_STATE_DIM,
    normalize_screen_size,
)
from gamewam.datasets.schemas.minecraft import (
    attach_or_validate_minecraft_state_schema,
    minecraft_state_features,
    resolve_cursor_reference_size,
)

# GUI cursor calibration measured at the baseline render size below. Runtime
# values are derived from --render_size unless explicitly overridden.
GUI_CAMERA_PITCH_DEG_PER_PX = DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL
GUI_CAMERA_YAW_DEG_PER_PX = DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL
from gamewam.envs.minecraft.assets import (
    CRAFT_ITEM_SPAWN_FILE,
    SMELT_ITEM_SPAWN_FILE,
    TAG_ITEMS_FILE,
    load_json,
)

TAG_INFO = load_json(TAG_ITEMS_FILE)

logger = logging.getLogger(__name__)


def _scripted_gui_episode_metadata(
    task_name: str,
    episode_index: int,
    event_anchor_frame_idx_in_clip: int,
) -> tuple[str, dict[str, Any]]:
    prefix, separator, suffix = task_name.partition(":")
    if prefix not in {"craft_item", "smelt_item"} or not separator or not suffix:
        raise ValueError(f"Unsupported ScriptedGUI task name: {task_name!r}")

    raw_file_name = f"scripted_gui_{task_name}_{episode_index:06d}"
    extra_fields = {
        "task_name": task_name,
        "event_key": f"craft_item:{suffix}",
        "event_anchor_frame_idx_in_clip": event_anchor_frame_idx_in_clip,
    }
    return raw_file_name, extra_fields


def _count_collected_tasks(
    episodes_jsonl: Path,
    task_names: list[str],
) -> dict[str, int]:
    import jsonlines

    counts = {task_name: 0 for task_name in task_names}
    with jsonlines.open(episodes_jsonl) as reader:
        for episode in reader:
            task_name = episode["task_name"]
            if task_name in counts:
                counts[task_name] += 1
    return counts


# Load recipe book card indices for mixed-mode recipe_book support.
_RECIPE_BOOK_CARD_INDICES_PATH = (
    Path(__file__).resolve().parent / "assets" / "recipe_book_card_indices.json"
)
_RECIPE_BOOK_CARD_INDICES: dict[str, int] = {}
if _RECIPE_BOOK_CARD_INDICES_PATH.exists():
    with open(_RECIPE_BOOK_CARD_INDICES_PATH) as _f:
        _RECIPE_BOOK_CARD_INDICES = json.load(_f)
    logger.info("Loaded %d recipe_book card indices from %s",
                len(_RECIPE_BOOK_CARD_INDICES), _RECIPE_BOOK_CARD_INDICES_PATH)
else:
    logger.info("No recipe_book card indices found at %s (recipe_book mode disabled)",
                _RECIPE_BOOK_CARD_INDICES_PATH)

# GUI geometry calibration (Minecraft 1.16). These baseline measurements are
# scaled to the configured render size by _configure_gui_geometry(). They are
# not the cursor-state reference coordinate system.
GUI_CALIBRATION_SIZE = (640, 360)
SCREEN_W, SCREEN_H = GUI_CALIBRATION_SIZE
_GUI_SCALE_X = 1.0
_GUI_SCALE_Y = 1.0
GUI_TEXTURE_W, GUI_TEXTURE_H = 176, 166

CT_GUI_LEFT = (SCREEN_W - GUI_TEXTURE_W) // 2   # 232
CT_GUI_TOP = (SCREEN_H - GUI_TEXTURE_H) // 2    # 97

INV_GUI_LEFT = CT_GUI_LEFT
INV_GUI_TOP = CT_GUI_TOP

# Furnace GUI slot positions measured empirically from real furnace
# recordings. The wiki-based coordinates were systematically off because
# MineStudio's MC 1.16 GUI rendering places slot centers differently than
# wiki texture-local coordinates assume.
FUR_GUI_LEFT = CT_GUI_LEFT
FUR_GUI_TOP  = CT_GUI_TOP
FURNACE_SMELT_WAIT_FRAMES = 220
FURNACE_COLLECT_RETRY_INTERVAL_FRAMES = 20
SMELT_TASK_MAX_STEPS = 600

FURNACE_INGREDIENT_CENTER = (295, 115)
FURNACE_FUEL_CENTER       = (295, 155)
FURNACE_OUTPUT_CENTER     = (365, 135)


def _furnace_ingredient_center() -> tuple[int, int]:
    return FURNACE_INGREDIENT_CENTER


def _furnace_fuel_center() -> tuple[int, int]:
    return FURNACE_FUEL_CENTER


def _furnace_output_center() -> tuple[int, int]:
    return FURNACE_OUTPUT_CENTER

CURSOR_INIT = (SCREEN_W // 2, SCREEN_H // 2)  # (320, 180)


def _crafting_table_grid_center(row: int, col: int) -> tuple[int, int]:
    tx, ty = 30 + col * 18, 17 + row * 18
    return (
        CT_GUI_LEFT + int(round((tx + 8) * _GUI_SCALE_X)),
        CT_GUI_TOP + int(round((ty + 8) * _GUI_SCALE_Y)),
    )


def _crafting_table_output_center() -> tuple[int, int]:
    return (
        CT_GUI_LEFT + int(round((124 + 8) * _GUI_SCALE_X)),
        CT_GUI_TOP + int(round((35 + 8) * _GUI_SCALE_Y)),
    )


def _inventory_2x2_grid_center(row: int, col: int) -> tuple[int, int]:
    tx, ty = 98 + col * 18, 18 + row * 18
    return (
        INV_GUI_LEFT + int(round((tx + 8) * _GUI_SCALE_X)),
        INV_GUI_TOP + int(round((ty + 8) * _GUI_SCALE_Y)),
    )


def _inventory_2x2_output_center() -> tuple[int, int]:
    return (
        INV_GUI_LEFT + int(round((154 + 8) * _GUI_SCALE_X)),
        INV_GUI_TOP + int(round((28 + 8) * _GUI_SCALE_Y)),
    )


# Recipe book GUI geometry (MC 1.16, 640×360 render)
RECIPE_BOOK_BTN_2x2 = (344, 164)
RECIPE_BOOK_BTN_3x3 = (246, 139)

# When the recipe book panel is open, the crafting output slot shifts by the
# panel width (78px) to make room for the recipe card list.
RECIPE_BOOK_OUTPUT_SHIFT_X = 78 + 3  # = 81 (panel width + manual correction for 2x2)


def _recipe_book_output_center_2x2() -> tuple[int, int]:
    x, y = _inventory_2x2_output_center()
    return (x + RECIPE_BOOK_OUTPUT_SHIFT_X, y)


def _recipe_book_output_center_3x3() -> tuple[int, int]:
    x, y = _crafting_table_output_center()
    return (x + RECIPE_BOOK_OUTPUT_SHIFT_X, y)


def _callback_slot_to_screen_center(slot_idx: int) -> tuple[int, int]:
    if 0 <= slot_idx <= 8:
        tx, ty = 8 + slot_idx * 18, 142
    elif 9 <= slot_idx <= 35:
        inv_idx = slot_idx - 9
        row, col = divmod(inv_idx, 9)
        tx, ty = 8 + col * 18, 84 + row * 18
    else:
        raise ValueError(f"Unsupported slot index: {slot_idx}")
    return (
        CT_GUI_LEFT + int(round((tx + 8) * _GUI_SCALE_X)),
        CT_GUI_TOP + int(round((ty + 8) * _GUI_SCALE_Y)),
    )


# Auto-generate RECIPE_LAYOUTS from craft_item.json + minecraft_data
def _build_recipe_layouts() -> dict[str, dict]:
    import minecraft_data as mc_data

    spawn_cfg = json.load(open(CRAFT_ITEM_SPAWN_FILE))
    mcd = mc_data('1.16')
    items_name = mcd.items_name
    id_to_name = {v['id']: k for k, v in items_name.items()}

    tag_to_items: dict[str, set[str]] = {}
    for tag_with_prefix, item_list in TAG_INFO.items():
        tag = tag_with_prefix.replace('minecraft:', '')
        tag_to_items[tag] = {i.replace('minecraft:', '') for i in item_list}

    def candidates_for_id(slot_id: int) -> set[str]:
        direct = id_to_name.get(slot_id)
        if direct:
            return {direct}
        result = set()
        for items in tag_to_items.values():
            for item in items:
                if items_name.get(item, {}).get('id') == slot_id:
                    result.update(items)
        return result

    def match_material(slot_id: int, inv_items: set[str]) -> str | None:
        cands = candidates_for_id(slot_id)
        match = cands & inv_items
        if match:
            return next(iter(match))
        return id_to_name.get(slot_id)

    layouts: dict[str, dict] = {}
    skipped: list[str] = []

    for task_name, cfg in sorted(spawn_cfg.items()):
        if ':' not in task_name:
            continue
        goal = cfg['goal']
        goal_id = items_name.get(goal, {}).get('id')
        if not goal_id:
            skipped.append(f'{task_name}: goal {goal!r} not in minecraft_data')
            continue

        recipes = mcd.recipes.get(str(goal_id), [])
        if not recipes:
            skipped.append(f'{task_name}: no recipe for {goal!r}')
            continue

        r = recipes[0]
        need_ct = cfg['need_crafting_table']
        inv_items = {x['type'] for x in cfg['init_inventory']}

        if 'inShape' in r:
            grid = []
            failed = False
            for row in r['inShape']:
                grid_row = []
                for slot_id in row:
                    if slot_id is None:
                        grid_row.append(None)
                    else:
                        mat = match_material(slot_id, inv_items)
                        if mat is None:
                            skipped.append(f'{task_name}: cannot map slot_id={slot_id}')
                            failed = True
                            break
                        grid_row.append(mat)
                grid.append(grid_row)
                if failed:
                    break
            if not failed:
                layouts[task_name] = {"need_crafting_table": need_ct, "grid": grid}

        elif 'ingredients' in r:
            grid_size = 3 if need_ct else 2
            grid = [[None] * grid_size for _ in range(grid_size)]
            failed = False
            ingredients = [i for i in r['ingredients'] if i is not None]
            for idx, slot_id in enumerate(ingredients):
                row, col = divmod(idx, grid_size)
                if row >= grid_size:
                    skipped.append(f'{task_name}: too many ingredients')
                    failed = True
                    break
                mat = match_material(slot_id, inv_items)
                if mat is None:
                    skipped.append(f'{task_name}: cannot map ingredient slot_id={slot_id}')
                    failed = True
                    break
                grid[row][col] = mat
            if not failed:
                layouts[task_name] = {"need_crafting_table": need_ct, "grid": grid}
        else:
            skipped.append(f'{task_name}: unknown recipe type')

    if skipped:
        logging.getLogger(__name__).warning(
            "RECIPE_LAYOUTS: skipped %d tasks: %s", len(skipped), skipped[:5]
        )
    logging.getLogger(__name__).info("RECIPE_LAYOUTS: loaded %d tasks", len(layouts))
    return layouts


RECIPE_LAYOUTS: dict[str, dict] = _build_recipe_layouts()


# Smelt layouts (furnace recipes from SMELT_ITEM_SPAWN_FILE)
def _build_smelt_layouts() -> dict[str, dict]:
    import json as _json
    data = _json.load(open(SMELT_ITEM_SPAWN_FILE))
    layouts = {}
    for task_name, cfg in data.items():
        if not cfg.get("need_furnace", False):
            continue
        inv = cfg.get("init_inventory", [])
        ingredient = None
        fuel = None
        for slot in inv:
            item = slot["type"].replace("minecraft:", "")
            if item in ("coal", "charcoal", "wood", "oak_planks", "spruce_planks",
                        "birch_planks", "jungle_planks", "acacia_planks"):
                fuel = item
            else:
                ingredient = item
        if ingredient is None or fuel is None:
            if len(inv) >= 2:
                ingredient = inv[0]["type"].replace("minecraft:", "")
                fuel = inv[1]["type"].replace("minecraft:", "")
            else:
                continue
        layouts[task_name] = {
            "ingredient": ingredient,
            "fuel": fuel,
            "goal": cfg["goal"].replace("minecraft:", ""),
        }
    logging.getLogger(__name__).info("SMELT_LAYOUTS: loaded %d tasks", len(layouts))
    return layouts


SMELT_LAYOUTS: dict[str, dict] = _build_smelt_layouts()

ALL_TASK_LAYOUTS: dict[str, dict] = {
    **{k: {**v, "task_type": "craft"} for k, v in RECIPE_LAYOUTS.items()},
    **{k: {**v, "task_type": "smelt"} for k, v in SMELT_LAYOUTS.items()},
}


# Action sequence builder
def _make_noop() -> dict:
    a = copy.deepcopy(ENV_NULL_ACTION)
    a["camera"] = np.array([0.0, 0.0], dtype=np.float32)
    return a


def _make_move_actions(from_pos: tuple, to_pos: tuple, steps: int = 6) -> list[dict]:
    """Camera-move actions to slide cursor from from_pos to to_pos.

    Default steps=6 is calibrated on the craft collection dataset.
    Two trailing noops let Malmo apply the delta before next action reads state.
    """
    import math
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    pixel_dist = math.sqrt(dx*dx + dy*dy)
    actual_steps = max(6, int(pixel_dist / 12))
    actual_steps = min(actual_steps, 30)

    actions = []
    for _ in range(actual_steps):
        a = _make_noop()
        a["camera"] = np.array([
            (dy / actual_steps) * GUI_CAMERA_PITCH_DEG_PER_PX,
            (dx / actual_steps) * GUI_CAMERA_YAW_DEG_PER_PX,
        ], dtype=np.float32)
        actions.append(a)
    actions.append(_make_noop())
    actions.append(_make_noop())
    return actions


def _move_to(last_pos: tuple, target: tuple, steps: int = 6) -> tuple[list[dict], tuple]:
    """Move cursor directly from last_pos to target. Returns (actions, target)."""
    return _make_move_actions(last_pos, target, steps), target


def _make_click(button: str = "attack", hold_frames: int = 4) -> list[dict]:
    """Hold a button for hold_frames then release.

    4 frames matches the craft collection calibration.
    """
    actions = []
    for _ in range(hold_frames):
        a = _make_noop()
        a[button] = 1
        actions.append(a)
    actions.append(_make_noop())
    return actions


def build_crafting_action_sequence(
    recipe: dict,
    item_slot_map: dict[str, list[int]],
) -> list[dict]:
    """Build action sequence for one crafting operation.

    Requires item_slot_map to have one slot per item instance (not stacks).
    Uses absolute positioning from screen center to avoid drift.
    """
    need_ct = recipe["need_crafting_table"]
    grid = recipe["grid"]

    grid_placements: list[tuple[str, int, int]] = []
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell is not None:
                grid_placements.append((cell, row_idx, col_idx))

    actions: list[dict] = []
    last_pos = CURSOR_INIT
    item_slot_usage: dict[str, int] = {}

    for item, row_idx, col_idx in grid_placements:
        slots = item_slot_map.get(item, [])
        use_idx = item_slot_usage.get(item, 0)
        if use_idx >= len(slots):
            logger.warning("Not enough slots for item %s (need %d, have %d)", item, use_idx + 1, len(slots))
            use_idx = len(slots) - 1 if slots else 0
        item_slot_usage[item] = use_idx + 1

        if not slots:
            logger.warning("Item %s not found in inventory", item)
            continue
        src = _callback_slot_to_screen_center(slots[use_idx])
        move_actions, last_pos = _move_to(last_pos, src)
        actions.extend(move_actions)
        actions.extend(_make_click("attack"))

        if need_ct:
            dst = _crafting_table_grid_center(row_idx, col_idx)
        else:
            dst = _inventory_2x2_grid_center(row_idx, col_idx)
        move_actions, last_pos = _move_to(last_pos, dst)
        actions.extend(move_actions)
        actions.extend(_make_click("attack"))

    for _ in range(8):
        actions.append(_make_noop())

    if need_ct:
        output = _crafting_table_output_center()
    else:
        output = _inventory_2x2_output_center()
    move_actions, last_pos = _move_to(last_pos, output)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    occupied_slots: set[int] = set()
    for slots in item_slot_map.values():
        occupied_slots.update(slots)
    free_slots = [s for s in range(36) if s not in occupied_slots]
    if not free_slots:
        free_slots = [0]
    drop_slot = random.choice(free_slots)
    drop_target = _callback_slot_to_screen_center(drop_slot)
    move_actions, last_pos = _move_to(last_pos, drop_target)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    return actions


# Furnace action sequence builder
def build_furnace_setup_actions(
    smelt_layout: dict,
    item_slot_map: dict[str, list[int]],
) -> tuple[list[dict], tuple[int, int]]:
    """Place ingredient and fuel into the furnace. Returns (actions, final_cursor_pos)."""
    ingredient = smelt_layout["ingredient"]
    fuel = smelt_layout["fuel"]

    actions: list[dict] = []
    last_pos = CURSOR_INIT

    ing_slots = item_slot_map.get(ingredient, [])
    if not ing_slots:
        logger.warning("Ingredient %s not found in inventory", ingredient)
        return actions, last_pos
    ing_src = _callback_slot_to_screen_center(ing_slots[0])

    move_actions, last_pos = _move_to(last_pos, ing_src)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    ing_dst = _furnace_ingredient_center()
    move_actions, last_pos = _move_to(last_pos, ing_dst)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    fuel_slots = item_slot_map.get(fuel, [])
    if not fuel_slots:
        logger.warning("Fuel %s not found in inventory", fuel)
    else:
        fuel_src = _callback_slot_to_screen_center(fuel_slots[0])

        move_actions, last_pos = _move_to(last_pos, fuel_src)
        actions.extend(move_actions)
        actions.extend(_make_click("attack"))

        fuel_dst = _furnace_fuel_center()
        move_actions, last_pos = _move_to(last_pos, fuel_dst)
        actions.extend(move_actions)
        actions.extend(_make_click("attack"))

    return actions, last_pos


def build_crafting_action_sequence_stack_split(
    recipe: dict,
    inventory_with_qty: dict[int, tuple[str, int]],
) -> list[dict]:
    """Build crafting actions for stacked inventory.

    Items come from stacked slots (e.g. 6 honeycomb in one slot).
    Left-click source to pick up entire stack, right-click grid cells to place
    1 each, left-click free slot to return leftovers.
    """
    need_ct = recipe["need_crafting_table"]
    grid = recipe["grid"]

    placements: list[tuple[str, int, int]] = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell is not None:
                placements.append((cell, r, c))
    # Sort by item type so same-material cells are placed consecutively,
    # avoiding repeated pick-up/return cycles for interleaved materials.
    placements.sort(key=lambda x: x[0])

    actions: list[dict] = []
    last_pos = CURSOR_INIT

    remaining: dict[int, int] = {slot: qty for slot, (_t, qty) in inventory_with_qty.items()}
    cursor_count = 0
    cursor_item: str | None = None
    cursor_src_slot: int | None = None

    def find_stack(item: str) -> int | None:
        candidates = [s for s, (t, q) in inventory_with_qty.items()
                      if t == item and remaining.get(s, 0) > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda s: remaining[s])

    for item, row_idx, col_idx in placements:
        if need_ct:
            cell_dst = _crafting_table_grid_center(row_idx, col_idx)
        else:
            cell_dst = _inventory_2x2_grid_center(row_idx, col_idx)

        if cursor_count <= 0 or cursor_item != item:
            if cursor_count > 0:
                occupied = {s for s, (_t, q) in inventory_with_qty.items() if q > 0}
                free = [s for s in range(36) if s not in occupied]
                if not free:
                    free = [0]
                return_slot = random.choice(free)
                return_center = _callback_slot_to_screen_center(return_slot)
                move_actions, last_pos = _move_to(last_pos, return_center)
                actions.extend(move_actions)
                actions.extend(_make_click("attack"))
                cursor_count = 0
                cursor_item = None
                cursor_src_slot = None

            src_slot = find_stack(item)
            if src_slot is None:
                logger.warning("[stack_split] item %s not in inventory for cell (%d,%d)",
                               item, row_idx, col_idx)
                return []
            src_center = _callback_slot_to_screen_center(src_slot)
            move_actions, last_pos = _move_to(last_pos, src_center)
            actions.extend(move_actions)
            actions.extend(_make_click("attack"))
            cursor_count = remaining[src_slot]
            cursor_item = item
            cursor_src_slot = src_slot
            remaining[src_slot] = 0

        move_actions, last_pos = _move_to(last_pos, cell_dst)
        actions.extend(move_actions)
        actions.extend(_make_click("use"))
        cursor_count -= 1

    if cursor_count > 0:
        occupied = {s for s, (_t, q) in inventory_with_qty.items() if q > 0}
        free = [s for s in range(36) if s not in occupied]
        if not free:
            free = [0]
        return_slot = random.choice(free)
        return_center = _callback_slot_to_screen_center(return_slot)
        move_actions, last_pos = _move_to(last_pos, return_center)
        actions.extend(move_actions)
        actions.extend(_make_click("attack"))

    for _ in range(8):
        actions.append(_make_noop())

    if need_ct:
        output = _crafting_table_output_center()
    else:
        output = _inventory_2x2_output_center()
    move_actions, last_pos = _move_to(last_pos, output)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    occupied_slots = {s for s, (_t, q) in inventory_with_qty.items() if q > 0}
    free_slots = [s for s in range(36) if s not in occupied_slots]
    if not free_slots:
        free_slots = [0]
    drop_slot = random.choice(free_slots)
    drop_target = _callback_slot_to_screen_center(drop_slot)
    move_actions, last_pos = _move_to(last_pos, drop_target)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    return actions


# Recipe book action sequence builder
def _recipe_book_open_gui_init_actions(is_3x3: bool, has_crafting_table_init: bool) -> list[dict]:
    """Click the recipe book button, wait for panel to render."""
    actions: list[dict] = []

    book_xy = RECIPE_BOOK_BTN_3x3 if is_3x3 else RECIPE_BOOK_BTN_2x2
    move_actions, _ = _move_to(CURSOR_INIT, book_xy)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    for _ in range(8):
        actions.append(_make_noop())

    return actions


def _recipe_book_card_click_position(is_3x3: bool) -> tuple[int, int]:
    """First recipe card center in the open book panel (smoke-measured)."""
    return (
        int(round(180 * _GUI_SCALE_X)),
        int(round(140 * _GUI_SCALE_Y)),
    )


RECIPE_BOOK_CARDS_PER_ROW = 5
RECIPE_BOOK_CARDS_PER_PAGE = 20
RECIPE_BOOK_CARD_DX = 24  # smoke-measured horizontal spacing
RECIPE_BOOK_CARD_DY = 26  # smoke-measured vertical spacing
# Page button offsets from first-card center (smoke-measured)
RECIPE_BOOK_FORWARD_BTN_DX = 82
RECIPE_BOOK_BACK_BTN_DX = 27
RECIPE_BOOK_PAGE_BTN_DY = 106


def _configure_gui_geometry(render_size: tuple[int, int]) -> None:
    """Derive GUI click geometry from the configured render resolution."""
    global SCREEN_W, SCREEN_H, _GUI_SCALE_X, _GUI_SCALE_Y
    global GUI_TEXTURE_W, GUI_TEXTURE_H, CT_GUI_LEFT, CT_GUI_TOP
    global INV_GUI_LEFT, INV_GUI_TOP, FUR_GUI_LEFT, FUR_GUI_TOP
    global FURNACE_INGREDIENT_CENTER, FURNACE_FUEL_CENTER, FURNACE_OUTPUT_CENTER
    global CURSOR_INIT, RECIPE_BOOK_BTN_2x2, RECIPE_BOOK_BTN_3x3
    global RECIPE_BOOK_OUTPUT_SHIFT_X, RECIPE_BOOK_CARD_DX, RECIPE_BOOK_CARD_DY
    global RECIPE_BOOK_FORWARD_BTN_DX, RECIPE_BOOK_BACK_BTN_DX, RECIPE_BOOK_PAGE_BTN_DY

    width, height = normalize_screen_size(render_size, name="render_size")
    SCREEN_W, SCREEN_H = int(round(width)), int(round(height))
    base_width, base_height = GUI_CALIBRATION_SIZE
    _GUI_SCALE_X = width / base_width
    _GUI_SCALE_Y = height / base_height
    sx = lambda value: int(round(value * _GUI_SCALE_X))
    sy = lambda value: int(round(value * _GUI_SCALE_Y))

    GUI_TEXTURE_W, GUI_TEXTURE_H = sx(176), sy(166)
    CT_GUI_LEFT = (SCREEN_W - GUI_TEXTURE_W) // 2
    CT_GUI_TOP = (SCREEN_H - GUI_TEXTURE_H) // 2
    INV_GUI_LEFT, INV_GUI_TOP = CT_GUI_LEFT, CT_GUI_TOP
    FUR_GUI_LEFT, FUR_GUI_TOP = CT_GUI_LEFT, CT_GUI_TOP
    FURNACE_INGREDIENT_CENTER = (sx(295), sy(115))
    FURNACE_FUEL_CENTER = (sx(295), sy(155))
    FURNACE_OUTPUT_CENTER = (sx(365), sy(135))
    CURSOR_INIT = (SCREEN_W // 2, SCREEN_H // 2)
    RECIPE_BOOK_BTN_2x2 = (sx(344), sy(164))
    RECIPE_BOOK_BTN_3x3 = (sx(246), sy(139))
    RECIPE_BOOK_OUTPUT_SHIFT_X = sx(81)
    RECIPE_BOOK_CARD_DX = sx(24)
    RECIPE_BOOK_CARD_DY = sy(26)
    RECIPE_BOOK_FORWARD_BTN_DX = sx(82)
    RECIPE_BOOK_BACK_BTN_DX = sx(27)
    RECIPE_BOOK_PAGE_BTN_DY = sy(106)


def _recipe_book_target_card_index(goal_item: str, inventory_items: set[str], is_3x3: bool = False) -> int:
    """Compute the 0-based index of the goal recipe card in the recipe book.

    Mirrors MC 1.16 ``ClientRecipeBook.categorize()`` + ``RecipeBookGui.updateCollections()``
    ("Show All" mode):

    1. For every recipe in ``minecraft_data.recipes``, check if its ingredients
       intersect the player's inventory and if it fits the current grid.
    2. Sort the shown recipes by their **internal item name** (``item.name``
       in ``minecraft_data.items_name``), matching MC's ``Item.toString()``
       which returns ``"minecraft:item_name"`` → dictionary-order on the
       item_name portion.
    3. Return the index of ``goal_item`` in that sorted list.
    """
    import minecraft_data as mc_data
    mcd = mc_data('1.16')
    items_name = mcd.items_name

    # Build item-name lookups from minecraft_data
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for name, info in items_name.items():
        if isinstance(info, dict) and 'id' in info:
            id_to_name[info['id']] = name
            name_to_id[name] = info['id']

    if not inventory_items:
        return 0

    inv_ids: set[int] = {name_to_id[n] for n in inventory_items if n in name_to_id}
    max_dim = 3 if is_3x3 else 2

    shown: set[str] = set()

    for output_id_str, recipe_list in mcd.recipes.items():
        output_id = int(output_id_str)
        output_name = id_to_name.get(output_id)
        if not output_name:
            continue

        for recipe in recipe_list:
            ing_ids: set[int] = set()
            max_row, max_col = 0, 0

            if 'inShape' in recipe:
                for r, row in enumerate(recipe['inShape']):
                    for c, slot in enumerate(row):
                        if isinstance(slot, int) and slot > 0:
                            ing_ids.add(slot)
                            max_row = max(max_row, r + 1)
                            max_col = max(max_col, c + 1)
            elif 'ingredients' in recipe:
                for ing in recipe['ingredients']:
                    if isinstance(ing, int) and ing > 0:
                        ing_ids.add(ing)
                max_row = 1
                max_col = len(ing_ids)

            if not ing_ids:
                continue

            # Must fit the current grid AND have ingredient overlap with inventory.
            # MC 1.16 "Show All" does NOT require ALL ingredients present — only
            # at least one ingredient type must be in the inventory for the
            # recipe to appear (the card is red if some are missing, but still
            # shown).
            if max_row <= max_dim and max_col <= max_dim and (ing_ids & inv_ids):
                shown.add(output_name)

    # Sort by internal item name (matching MC's Item.toString() → dictionary order).
    sorted_shown = sorted(shown, key=lambda n: n)

    for i, name in enumerate(sorted_shown):
        if name == goal_item:
            return i

    return 0


def _recipe_book_card_xy(is_3x3: bool, card_index: int) -> tuple[int, int]:
    """Pixel center of the (card_index)-th recipe card (smoke-measured)."""
    base_x, base_y = _recipe_book_card_click_position(is_3x3)
    col = card_index % RECIPE_BOOK_CARDS_PER_ROW
    row = card_index // RECIPE_BOOK_CARDS_PER_ROW
    return (base_x + col * RECIPE_BOOK_CARD_DX,
            base_y + row * RECIPE_BOOK_CARD_DY)


def _recipe_book_page_forward_xy(is_3x3: bool) -> tuple[int, int]:
    """Center of the forward (next page) button."""
    base_x, base_y = _recipe_book_card_click_position(is_3x3)
    return (base_x + RECIPE_BOOK_FORWARD_BTN_DX,
            base_y + RECIPE_BOOK_PAGE_BTN_DY)


def _build_recipe_book_card_click_actions(
    is_3x3: bool, card_index: int, cursor_start: tuple[int, int],
) -> tuple[list[dict], tuple[int, int]]:
    """Move cursor to the (card_index)-th card and left-click it.

    If the card is on a later page, clicks the forward button first.
    Returns (actions, cursor_after).
    """
    actions: list[dict] = []
    last_pos = cursor_start

    # Navigate pages: each page holds RECIPE_BOOK_CARDS_PER_PAGE cards.
    # Card 0-19 → page 0, card 20-39 → page 1, etc.
    pages_needed = card_index // RECIPE_BOOK_CARDS_PER_PAGE
    local_index = card_index % RECIPE_BOOK_CARDS_PER_PAGE

    for _ in range(pages_needed):
        fwd_xy = _recipe_book_page_forward_xy(is_3x3)
        move_actions, last_pos = _move_to(last_pos, fwd_xy)
        actions.extend(move_actions)
        actions.extend(_make_click("attack"))
        # Wait for page transition
        for _ in range(3):
            actions.append(_make_noop())

    card_xy = _recipe_book_card_xy(is_3x3, local_index)
    move_actions, last_pos = _move_to(last_pos, card_xy)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))
    return actions, card_xy


def _build_recipe_book_output_pickup_drop_actions(
    is_3x3: bool,
    cursor_start: tuple[int, int],
    item_slot_map: dict[str, list[int]],
) -> tuple[list[dict], tuple[int, int]]:
    """Move to the shifted output slot, pickup, then drop to a free inventory slot.

    When the recipe book panel is open, the output slot is shifted by
    RECIPE_BOOK_OUTPUT_SHIFT_X.
    """
    output = _recipe_book_output_center_3x3() if is_3x3 else _recipe_book_output_center_2x2()
    move_actions, last_pos = _move_to(cursor_start, output)
    actions: list[dict] = list(move_actions) + list(_make_click("attack"))

    occupied: set[int] = set()
    for slots in item_slot_map.values():
        occupied.update(slots)
    free_slots = [s for s in range(36) if s not in occupied]
    if not free_slots:
        free_slots = [0]
    drop_slot = random.choice(free_slots)
    drop_target = _callback_slot_to_screen_center(drop_slot)
    drop_target = (drop_target[0] + RECIPE_BOOK_OUTPUT_SHIFT_X, drop_target[1])
    move_actions, last_pos = _move_to(last_pos, drop_target)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))
    return actions, last_pos


def build_furnace_collect_actions(
    item_slot_map: dict[str, list[int]] | None = None,
    start_pos: tuple[int, int] | None = None,
) -> list[dict]:
    """Click the furnace output slot and place result in inventory."""
    actions: list[dict] = []
    last_pos = start_pos if start_pos is not None else _furnace_fuel_center()

    out_dst = _furnace_output_center()
    move_actions, last_pos = _move_to(last_pos, out_dst)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    occupied_slots: set[int] = set()
    if item_slot_map:
        for slots in item_slot_map.values():
            occupied_slots.update(slots)
    free_slots = [s for s in range(36) if s not in occupied_slots]
    if not free_slots:
        free_slots = [0]
    drop_slot = random.choice(free_slots)
    drop_target = _callback_slot_to_screen_center(drop_slot)
    move_actions, last_pos = _move_to(last_pos, drop_target)
    actions.extend(move_actions)
    actions.extend(_make_click("attack"))

    return actions


# Inventory reader
def read_inventory_from_obs(info: dict) -> dict[str, list[int]]:
    """Parse MineStudio obs inventory into {item_type: [slot_idx, ...]}."""
    raw_inv = info.get("inventory", {})
    item_slots: dict[str, list[int]] = {}
    for slot_idx, slot_content in raw_inv.items():
        if not slot_content or not isinstance(slot_content, dict):
            continue
        item_type = slot_content.get("type", "none")
        if item_type in ("none", "air", ""):
            continue
        item_type = item_type.replace("minecraft:", "")
        item_slots.setdefault(item_type, []).append(int(slot_idx))
    return item_slots


def read_inventory_with_quantities(info: dict) -> dict[int, tuple[str, int]]:
    """Parse MineStudio obs inventory into {slot_idx: (type, quantity)}."""
    raw_inv = info.get("inventory", {})
    out: dict[int, tuple[str, int]] = {}
    for slot_idx, slot_content in raw_inv.items():
        if not slot_content or not isinstance(slot_content, dict):
            continue
        item_type = slot_content.get("type", "none")
        if item_type in ("none", "air", ""):
            continue
        item_type = item_type.replace("minecraft:", "")
        out[int(slot_idx)] = (item_type, int(slot_content.get("quantity", 0)))
    return out


def _wait_for_inventory(env, expected_items: set[str], max_wait_steps: int = 200,
                        warmup_steps: int = 5) -> tuple[dict[str, list[int]], Any]:
    """Step noops until inventory is ready and return the latest observation."""
    item_slot_map: dict[str, list[int]] = {}
    latest_obs = None
    for _ in range(warmup_steps):
        latest_obs = env.step(_noop_dict())
    for _ in range(max_wait_steps):
        info = env._last_info if hasattr(env, "_last_info") else {}
        item_slot_map = read_inventory_from_obs(info)
        if expected_items.issubset(item_slot_map.keys()):
            return item_slot_map, latest_obs
        latest_obs = env.step(_noop_dict())
    return item_slot_map, latest_obs


def _noop_dict() -> dict:
    return copy.deepcopy(ENV_NULL_ACTION)


DEBUG_SAVE_FAIL_VIDEO: bool = False
DEBUG_VIDEO_DIR: str | None = None
DEBUG_VIDEO_CAPTURE_EVERY_N: int = 1


def _save_debug_fail_video(frames: list, task_name: str, attempt_idx: int) -> None:
    """Save a downsampled MP4 of a failed episode for visual inspection."""
    if not DEBUG_VIDEO_DIR:
        return
    import os, time
    try:
        os.makedirs(DEBUG_VIDEO_DIR, exist_ok=True)
        ts = int(time.time()) % 100000
        safe_task = task_name.replace(":", "_")
        path = Path(DEBUG_VIDEO_DIR) / f"debug_fail_{safe_task}_att{attempt_idx}_{ts}.mp4"
        subsampled = frames[::DEBUG_VIDEO_CAPTURE_EVERY_N]
        _write_mp4(subsampled, path, fps=20)
        logger.info("[DEBUG] saved fail video to %s (%d frames sampled to %d)",
                    path, len(frames), len(subsampled))
    except Exception as e:
        logger.warning("[DEBUG] failed to save debug video: %s", e)


# Distractor pool shared across all inventory-patching functions.
_DISTRACTOR_POOL = [
    "dirt", "cobblestone", "stone", "sand", "gravel", "sandstone",
    "oak_planks", "spruce_planks", "birch_planks", "jungle_planks",
    "oak_log", "spruce_log", "birch_log", "jungle_log",
    "oak_stairs", "spruce_stairs", "birch_stairs",
    "oak_slab", "spruce_slab", "birch_slab",
    "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wooden_axe", "stone_axe", "iron_axe",
    "wooden_shovel", "stone_shovel", "iron_shovel",
    "coal", "iron_ore", "gold_ore", "iron_ingot", "gold_ingot", "diamond",
    "bread", "apple", "cooked_beef", "beef", "carrot", "potato",
    "torch", "bucket", "bowl", "flint", "string", "feather", "leather",
    "paper", "book", "writable_book",
]


def _add_distractors(
    new_inv: list,
    required_items: set[str],
    rng: random.Random | None = None,
) -> None:
    """Add distractor items to new_inv with 80% easy / 10% mid / 10% hard distribution.

    - easy: 0 distractors (clean inventory, only required items)
    - mid: 1-4 distractors
    - hard: 5-12 distractors
    """
    rng = rng or random
    roll = rng.random()
    if roll < 0.8:
        return  # easy: no distractors
    elif roll < 0.9:
        num = rng.randint(1, 4)  # mid
    else:
        num = rng.randint(5, 12)  # hard

    available = [item for item in _DISTRACTOR_POOL if item not in required_items]
    if not available:
        return
    for _ in range(num):
        item = rng.choice(available)
        qty = rng.randint(1, 3)
        new_inv.append({"slot": "random", "type": item, "quantity": qty})


def _patch_init_inventory_single_items(task_config: dict, recipe: dict,
                                        add_distractors: bool = True) -> dict:
    """One item per slot for each required ingredient, plus distractors (80/10/10 split)."""
    import copy
    task_config = copy.deepcopy(task_config)

    item_counts: dict[str, int] = {}
    for row in recipe["grid"]:
        for cell in row:
            if cell is not None:
                item_counts[cell] = item_counts.get(cell, 0) + 1

    new_inv = []
    required_items = set()
    for item_type, count in item_counts.items():
        required_items.add(item_type)
        for _ in range(count):
            new_inv.append({"slot": "random", "type": item_type, "quantity": 1})

    if add_distractors:
        _add_distractors(new_inv, required_items)

    task_config["callback"]["init_inventory"]["init_inventory"] = new_inv
    return task_config


def _patch_init_inventory_recipe_book(task_config: dict, recipe: dict, rng: random.Random | None = None) -> dict:
    """Override init_inventory for recipe_book mode: only required items, no distractors."""
    import copy
    rng = rng or random
    task_config = copy.deepcopy(task_config)

    item_counts: dict[str, int] = {}
    for row in recipe["grid"]:
        for cell in row:
            if cell is not None:
                item_counts[cell] = item_counts.get(cell, 0) + 1

    new_inv = []
    for item_type, count in item_counts.items():
        if rng.random() < 0.5 and count >= 2:
            qty = count + rng.randint(0, count)
            new_inv.append({"slot": "random", "type": item_type, "quantity": qty})
        else:
            for _ in range(count):
                new_inv.append({"slot": "random", "type": item_type, "quantity": 1})

    task_config["callback"]["init_inventory"]["init_inventory"] = new_inv
    return task_config


def _patch_init_inventory_stack_split(task_config: dict, recipe: dict, rng: random.Random | None = None) -> dict:
    """Stacked inventory: each multi-copy ingredient in a single slot, plus distractors."""
    import copy
    rng = rng or random
    task_config = copy.deepcopy(task_config)

    item_counts: dict[str, int] = {}
    for row in recipe["grid"]:
        for cell in row:
            if cell is not None:
                item_counts[cell] = item_counts.get(cell, 0) + 1

    new_inv = []
    required_items = set(item_counts.keys())
    for item_type, count in item_counts.items():
        if count >= 2:
            qty = count + rng.randint(0, 2)
            new_inv.append({"slot": "random", "type": item_type, "quantity": qty})
        else:
            new_inv.append({"slot": "random", "type": item_type, "quantity": 1})

    _add_distractors(new_inv, required_items, rng=rng)
    task_config["callback"]["init_inventory"]["init_inventory"] = new_inv
    return task_config


def _patch_init_inventory_smelt(task_config: dict, smelt_layout: dict) -> dict:
    """Smelt inventory: ingredient + fuel + distractors (80/10/10 split)."""
    import copy
    task_config = copy.deepcopy(task_config)

    ingredient = smelt_layout["ingredient"]
    fuel = smelt_layout["fuel"]
    required_items = {ingredient, fuel}

    new_inv = [
        {"slot": "random", "type": ingredient, "quantity": 1},
        {"slot": "random", "type": fuel, "quantity": 4},
    ]

    _add_distractors(new_inv, required_items)
    task_config["callback"]["init_inventory"]["init_inventory"] = new_inv
    return task_config


def _get_render_frame(info: dict) -> np.ndarray | None:
    """Extract the full-resolution pov frame from info['pov']."""
    pov = info.get("pov")
    if pov is not None:
        return pov
    return None


def collect_one_episode(
    task_name: str,
    recipe: dict,
    render_size: tuple[int, int] = (640, 360),
    obs_size: tuple[int, int] = (224, 224),
    difficulty: str = "zero",
    max_steps: int = 800,
    bias: int = 0,
    collect_mode: str = "expanded",
    attempt_idx: int = 0,
    cursor_reference_size: tuple[float, float] | None = None,
    cursor_pitch_deg_per_pixel: float = DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL,
    cursor_yaw_deg_per_pixel: float = DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL,
) -> dict | None:
    """Collect one scripted expert episode. Returns episode dict or None on failure.

    Supports craft (crafting table/inventory) and smelt (furnace) tasks.
    collect_mode: "expanded", "stack_split", "recipe_book", or "mixed".
    With bias > 0, the episode window is extended after the reward event.
    """
    global GUI_CAMERA_PITCH_DEG_PER_PX, GUI_CAMERA_YAW_DEG_PER_PX
    _configure_gui_geometry(render_size)
    GUI_CAMERA_PITCH_DEG_PER_PX = float(cursor_pitch_deg_per_pixel)
    GUI_CAMERA_YAW_DEG_PER_PX = float(cursor_yaw_deg_per_pixel)

    task_type = recipe.get("task_type", "craft")

    if task_type == "smelt":
        from gamewam.envs.minecraft.tasks.smelt_item import gen_smelt_item_task_config
        task_config = gen_smelt_item_task_config(task_name, difficulty=difficulty)
        task_config = _patch_init_inventory_smelt(task_config, recipe)
        max_steps = max(max_steps, SMELT_TASK_MAX_STEPS)
    else:
        from gamewam.envs.minecraft.tasks.craft_item import gen_craft_item_task_config
        task_config = gen_craft_item_task_config(task_name, difficulty=difficulty)

        effective_mode = collect_mode
        if collect_mode == "mixed" and task_type == "craft":
            if task_name in _RECIPE_BOOK_CARD_INDICES:
                effective_mode = random.choices(
                    ["stack_split", "recipe_book", "expanded"],
                    weights=[0.6, 0.2, 0.2], k=1,
                )[0]
            else:
                effective_mode = random.choices(
                    ["expanded", "stack_split"], weights=[0.2, 0.8], k=1
                )[0]

        if effective_mode == "expanded":
            task_config = _patch_init_inventory_single_items(task_config, recipe)
        elif effective_mode == "stack_split":
            task_config = _patch_init_inventory_stack_split(task_config, recipe)
        elif effective_mode == "recipe_book":
            task_config = _patch_init_inventory_recipe_book(task_config, recipe)
        else:
            raise NotImplementedError(
                f"[collect_one_episode] unknown collect_mode={collect_mode!r}; "
                "expected one of 'expanded', 'stack_split', 'recipe_book', 'mixed'"
            )

        collect_mode = effective_mode

    env = MinecraftGameEnv(
        task_config=task_config,
        obs_size=obs_size,
        render_size=render_size,
        cursor_reference_size=cursor_reference_size or render_size,
        cursor_pitch_deg_per_pixel=cursor_pitch_deg_per_pixel,
        cursor_yaw_deg_per_pixel=cursor_yaw_deg_per_pixel,
    )
    try:
        game_obs = env.reset()

        # Wait for required items to settle in inventory after reset
        expected_items = set()
        if task_type == "smelt":
            expected_items = {recipe["ingredient"], recipe["fuel"]}
        else:
            for row in recipe["grid"]:
                for cell in row:
                    if cell is not None:
                        expected_items.add(cell)
        item_slot_map, latest_obs = _wait_for_inventory(
            env, expected_items, max_wait_steps=120
        )
        if latest_obs is not None:
            game_obs = latest_obs
        logger.info("[%s] Inventory after reset+wait: %s", task_name, item_slot_map)

        missing = expected_items - set(item_slot_map.keys())
        if missing:
            logger.warning("[%s] Missing expected items after init: %s (have: %s)",
                           task_name, missing, list(item_slot_map.keys()))

        first_frame = _get_render_frame(game_obs.info)
        if first_frame is None:
            first_frame = game_obs.image
        frames = [first_frame]
        action_vecs = []
        states = [game_obs.proprio]
        state_field_valids = [game_obs.proprio_field_valid]
        rewards = []
        total_reward = 0.0
        success_frame_idx: int | None = None

        def _step(action_dict):
            nonlocal total_reward, game_obs, success_frame_idx
            action_vec = action_to_vector(action_dict)
            action_vecs.append(action_vec)
            game_obs = env.step(action_dict)
            frame = _get_render_frame(game_obs.info)
            frames.append(frame if frame is not None else game_obs.image)
            states.append(game_obs.proprio)
            state_field_valids.append(game_obs.proprio_field_valid)
            rewards.append(game_obs.reward)
            prev_total_reward = total_reward
            total_reward += game_obs.reward
            if success_frame_idx is None and prev_total_reward <= 0.0 < total_reward:
                success_frame_idx = len(action_vecs) - 1
            return game_obs.done

        def _succeeded() -> bool:
            return total_reward > 0.0

        if task_type == "smelt":
            # Place ingredient + fuel into furnace
            setup_seq, _ = build_furnace_setup_actions(recipe, item_slot_map)
            if not setup_seq:
                logger.warning("[%s] empty setup_seq (ingredient=%s fuel=%s map=%s)",
                               task_name, recipe["ingredient"], recipe["fuel"], item_slot_map)
            done = False
            for action_dict in setup_seq:
                if len(frames) >= max_steps:
                    logger.warning("[%s] hit max_steps=%d during furnace setup",
                                   task_name, max_steps)
                    break
                done = _step(action_dict)
                if done:
                    break

            # Wait for smelting
            if not done and not _succeeded():
                wait_frames = min(FURNACE_SMELT_WAIT_FRAMES, max(0, max_steps - len(frames)))
                logger.info("[%s] smelt wait %d frames (setup_seq=%d done=%s)",
                            task_name, wait_frames, len(setup_seq), done)
                for _ in range(wait_frames):
                    done = _step(_make_noop())
                    if done or _succeeded():
                        break

            # Collect output
            if not done and not _succeeded():
                collect_seq = build_furnace_collect_actions(item_slot_map,
                                                            start_pos=_furnace_fuel_center())
                logger.info("[%s] collect_seq has %d actions, output_dst=%s",
                            task_name, len(collect_seq), _furnace_output_center())
                attempts = 0
                while len(frames) < max_steps:
                    for i, action_dict in enumerate(collect_seq):
                        if len(frames) >= max_steps:
                            break
                        done = _step(action_dict)
                        if done:
                            logger.info("[%s] collect step %d: episode done", task_name, i)
                            break
                    logger.info("[%s] collect attempt %d: done=%s succeeded=%s total_reward=%.2f",
                                task_name, attempts, done, _succeeded(), total_reward)
                    if done or _succeeded() or len(frames) >= max_steps:
                        break

                    for _ in range(FURNACE_COLLECT_RETRY_INTERVAL_FRAMES):
                        if len(frames) >= max_steps:
                            break
                        done = _step(_make_noop())
                        if done or _succeeded():
                            break
                    if done or _succeeded():
                        break
                    attempts += 1
                    if attempts > 5:
                        logger.warning("[%s] collect retries exhausted after %d attempts",
                                       task_name, attempts)
                        break
                logger.info("[%s] smelt collect done", task_name)
        else:
            if collect_mode == "stack_split":
                inv_with_qty = read_inventory_with_quantities(env._last_info)
                action_seq = build_crafting_action_sequence_stack_split(recipe, inv_with_qty)
                if not action_seq:
                    logger.warning("[%s] stack_split action sequence empty (missing items)", task_name)
                    return None
            elif collect_mode == "expanded":
                action_seq = build_crafting_action_sequence(recipe, item_slot_map)
            elif collect_mode == "recipe_book":
                is_3x3 = bool(recipe.get("need_crafting_table", False))

                grid = recipe["grid"]
                required_items: set[str] = set()
                for row in grid:
                    for cell in row:
                        if cell is not None:
                            required_items.add(cell)
                missing = required_items - set(item_slot_map.keys())
                if missing:
                    logger.warning(
                        "[recipe_book] task needs %s but inventory only has %s",
                        sorted(required_items), sorted(item_slot_map.keys()),
                    )
                    return None

                goal_item = task_name.split(':', 1)[1]
                inventory_set = set(item_slot_map.keys())
                if task_name in _RECIPE_BOOK_CARD_INDICES:
                    target_card_idx = _RECIPE_BOOK_CARD_INDICES[task_name]
                else:
                    target_card_idx = _recipe_book_target_card_index(goal_item, inventory_set, is_3x3)
                logger.info("[recipe_book] %s: goal=%s target_card_idx=%d",
                            task_name, goal_item, target_card_idx)

                # Open book panel
                book_xy = RECIPE_BOOK_BTN_3x3 if is_3x3 else RECIPE_BOOK_BTN_2x2
                open_actions, cursor_after_open = _move_to(CURSOR_INIT, book_xy)
                open_actions = list(open_actions) + list(_make_click("attack"))
                for a in open_actions:
                    if len(frames) >= max_steps:
                        break
                    done = _step(a)
                    if done:
                        break
                logger.info("[recipe_book] %s: book button clicked, done=%s", task_name, done)
                if not done:
                    # Click the target recipe card
                    card_xy_full = _recipe_book_card_xy(is_3x3, target_card_idx)
                    logger.info("[recipe_book] %s: card_idx=%d card_xy=%s cursor_from=%s",
                                task_name, target_card_idx, card_xy_full, cursor_after_open)
                    card_actions, card_xy = _build_recipe_book_card_click_actions(
                        is_3x3, target_card_idx, cursor_after_open)
                    for a in card_actions:
                        if len(frames) >= max_steps:
                            break
                        done = _step(a)
                        if done:
                            break
                    logger.info("[recipe_book] %s: card clicked, done=%s succeeded=%s",
                                task_name, done, _succeeded())
                    if not done and not _succeeded():
                        # Wait for auto-fill
                        for _ in range(5):
                            if len(frames) >= max_steps or _succeeded():
                                break
                            done = _step(_make_noop())
                            if done:
                                break
                        if not done and not _succeeded():
                            # Pickup output + drop to free slot
                            drop_actions, _cursor_after_drop = _build_recipe_book_output_pickup_drop_actions(
                                is_3x3, card_xy, item_slot_map)
                            for a in drop_actions:
                                if len(frames) >= max_steps:
                                    break
                                done = _step(a)
                                if done:
                                    break
                    logger.info("[recipe_book] %s: after card click loop, succeeded=%s",
                                task_name, _succeeded())
                action_seq = []
            else:
                raise NotImplementedError(
                    f"[collect_one_episode] unknown collect_mode={collect_mode!r}; "
                    "expected one of 'expanded', 'stack_split', 'recipe_book'"
                )
            for action_dict in action_seq:
                if len(frames) >= max_steps:
                    break
                done = _step(action_dict)
                if done:
                    break

        event_frame_idx = success_frame_idx if success_frame_idx is not None else len(action_vecs) - 1

        if bias > 0:
            for _ in range(bias):
                if len(frames) >= max_steps:
                    break
                done = _step(_make_noop())
                if done:
                    break

        success = total_reward > 0
        if not success:
            logger.info("Episode failed for %s (total_reward=%.2f)", task_name, total_reward)
            if DEBUG_SAVE_FAIL_VIDEO and frames:
                _save_debug_fail_video(frames, task_name, attempt_idx)
            return None

        event_frame_idx_in_clip = event_frame_idx

        return {
            "task_name": task_name,
            "task_description": task_config["task_description"],
            "frames": frames,
            "action_vecs": action_vecs,
            "states": states,
            "state_field_valids": state_field_valids,
            "rewards": rewards,
            "success": success,
            "event_frame_idx_in_clip": event_frame_idx_in_clip,
        }
    finally:
        env.close()
        # Kill lingering Minecraft Java processes that env.close() may have
        # failed to terminate, to prevent orphan processes from accumulating.
        try:
            import signal as _signal
            for inst in getattr(getattr(env, '_env', None), 'instances', []):
                mc_proc = getattr(inst, 'minecraft_process', None)
                if mc_proc is not None:
                    try:
                        os.killpg(os.getpgid(mc_proc.pid), _signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass


# Video writer helper
def _video_frames_for_action_rows(
    frames: list[np.ndarray], action_count: int
) -> list[np.ndarray]:
    """Return exactly the N observations aligned with N action rows."""
    if action_count < 0:
        raise ValueError(f"action_count must be non-negative, got {action_count}")
    expected = action_count + 1
    if len(frames) != expected:
        raise ValueError(
            f"collector must hold N+1 observations for N actions before video "
            f"encoding, got {len(frames)} observations for {action_count} actions"
        )
    return frames[:action_count]


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int = 20):
    import av
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for frame_np in frames:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# Main
def parse_args():
    p = argparse.ArgumentParser(description="Collect scripted GUI expert trajectories")
    p.add_argument("--tasks", type=str, default=",".join(ALL_TASK_LAYOUTS.keys()),
                   help="Comma-separated task names (default: all craft + smelt tasks)")
    p.add_argument("--episodes_per_task", type=int, default=50,
                   help="Target successful episodes per task")
    p.add_argument("--max_attempts_per_task", type=int, default=200,
                   help="Max env resets per task before giving up")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output LeRobot dataset directory")
    p.add_argument("--render_size", type=int, nargs=2, default=[640, 360],
                   metavar=("W", "H"), help="Render resolution (default: 640 360)")
    p.add_argument(
        "--cursor_reference_size",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Cursor state reference size; defaults to --render_size.",
    )
    p.add_argument(
        "--cursor_pitch_deg_per_pixel",
        type=float,
        default=None,
        help=(
            "GUI cursor pitch calibration in degrees per render pixel. "
            "Defaults to the 640x360 calibration scaled for --render_size."
        ),
    )
    p.add_argument(
        "--cursor_yaw_deg_per_pixel",
        type=float,
        default=None,
        help=(
            "GUI cursor yaw calibration in degrees per render pixel. "
            "Defaults to the 640x360 calibration scaled for --render_size."
        ),
    )
    p.add_argument("--obs_size", type=int, nargs=2, default=[224, 224],
                   metavar=("W", "H"), help="Observation resolution (default: 224 224)")
    p.add_argument("--difficulty", type=str, default="zero",
                   choices=["zero", "one", "easy", "middle", "hard", "normal"])
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel collection workers (default: 4)")
    p.add_argument("--bias", type=int, default=0,
                   help="Extra noop frames after scripted completion")
    p.add_argument("--collect_mode", type=str, default="mixed",
                   choices=["expanded", "stack_split", "recipe_book", "mixed"],
                   help="Inventory seeding mode: 'mixed' = random expanded/stack_split per episode, "
                        "'recipe_book' = recipe book panel auto-fill, "
                        "'expanded' = one slot per item, "
                        "'stack_split' = single stacked slot per multi-copy ingredient. "
                        "Ignored for smelt tasks.")
    p.add_argument("--debug_save_fail_video", action="store_true",
                   help="Save a debug MP4 of failed attempts for visual inspection")
    p.add_argument("--debug_capture_every_n", type=int, default=1,
                   help="When --debug_save_fail_video is set, save every Nth frame")
    p.add_argument("--dry_run", action="store_true",
                   help="Print action sequences without running the simulator")
    return p.parse_args()


# Worker function for multiprocessing
def _worker_loop(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    render_size: tuple[int, int],
    obs_size: tuple[int, int],
    difficulty: str,
    seed: int,
    temp_video_dir: str,
    fps: int,
    bias: int = 0,
    collect_mode: str = "expanded",
    debug_save_fail_video: bool = False,
    debug_video_dir: str | None = None,
    debug_video_capture_every_n: int = 1,
    cursor_reference_size: tuple[float, float] | None = None,
    cursor_pitch_deg_per_pixel: float = DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL,
    cursor_yaw_deg_per_pixel: float = DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL,
):
    """Worker process: pulls tasks from queue, runs collect_one_episode, pushes results."""

    global DEBUG_SAVE_FAIL_VIDEO, DEBUG_VIDEO_DIR, DEBUG_VIDEO_CAPTURE_EVERY_N
    global GUI_CAMERA_PITCH_DEG_PER_PX, GUI_CAMERA_YAW_DEG_PER_PX
    _configure_gui_geometry(render_size)
    GUI_CAMERA_PITCH_DEG_PER_PX = float(cursor_pitch_deg_per_pixel)
    GUI_CAMERA_YAW_DEG_PER_PX = float(cursor_yaw_deg_per_pixel)
    DEBUG_SAVE_FAIL_VIDEO = bool(debug_save_fail_video)
    DEBUG_VIDEO_DIR = debug_video_dir
    DEBUG_VIDEO_CAPTURE_EVERY_N = int(debug_video_capture_every_n)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [W{worker_id}] %(levelname)s: %(message)s",
    )
    logging.getLogger(__name__).info(
        "[W%d] DEBUG_SAVE_FAIL_VIDEO=%s, dir=%s, every_n=%d",
        worker_id, DEBUG_SAVE_FAIL_VIDEO, DEBUG_VIDEO_DIR, DEBUG_VIDEO_CAPTURE_EVERY_N
    )

    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)

    wlog = logging.getLogger(f"worker-{worker_id}")
    temp_dir = Path(temp_video_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Per-worker MineStudio directory to avoid SQLite lock contention
    import os, subprocess
    original_minestudio = os.environ["MINESTUDIO_DIR"]
    worker_minestudio = os.path.join(original_minestudio, f"worker_{worker_id}")
    os.makedirs(worker_minestudio, exist_ok=True)
    for shared_subdir in ("engine", "runtime"):
        src = os.path.join(original_minestudio, shared_subdir)
        dst = os.path.join(worker_minestudio, shared_subdir)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    os.environ["MINESTUDIO_DIR"] = worker_minestudio

    # Per-worker virtual display to avoid GLFW/OpenGL context contention.
    # Try a range of display numbers in case some are still occupied by
    # orphan Xvfb processes from a previous crashed run.
    xvfb_proc = None
    for offset in range(10):
        display_no = 100 + worker_id + offset * 64
        try:
            xvfb_proc = subprocess.Popen(
                ["Xvfb", f":{display_no}", "-screen", "0", "1024x768x24", "-ac"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            if xvfb_proc.poll() is None:
                break
            xvfb_proc = None
        except Exception:
            xvfb_proc = None
    if xvfb_proc is None:
        result_queue.put(("worker_exit", worker_id, "Xvfb bootstrap failed: no free display"))
        return
    os.environ["DISPLAY"] = f":{display_no}"
    wlog.info("Started Xvfb on :%d", display_no)

    def _stop_xvfb() -> None:
        if xvfb_proc.poll() is not None:
            return
        xvfb_proc.terminate()
        try:
            xvfb_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb_proc.kill()
            xvfb_proc.wait(timeout=5)

    # multiprocessing.Process.terminate() sends SIGTERM, which normally skips
    # Python-level finally blocks.  Register both an ordinary-exit hook and
    # signal handlers so a parent-side failure cannot orphan this worker's
    # Xvfb process.
    import atexit
    import signal
    atexit.register(_stop_xvfb)

    def _handle_worker_signal(signum, _frame) -> None:
        _stop_xvfb()
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _handle_worker_signal)
    signal.signal(signal.SIGINT, _handle_worker_signal)

    while True:
        try:
            item = task_queue.get(timeout=2)
        except queue.Empty:
            continue
        if item is None:
            wlog.info("Received shutdown signal, exiting.")
            _stop_xvfb()
            result_queue.put(("worker_exit", worker_id, None))
            break

        task_name, recipe_dict, attempt_idx = item
        try:
            ep = collect_one_episode(
                task_name=task_name,
                recipe=recipe_dict,
                render_size=render_size,
                obs_size=obs_size,
                difficulty=difficulty,
                bias=bias,
                collect_mode=collect_mode,
                attempt_idx=attempt_idx,
                cursor_reference_size=cursor_reference_size,
                cursor_pitch_deg_per_pixel=cursor_pitch_deg_per_pixel,
                cursor_yaw_deg_per_pixel=cursor_yaw_deg_per_pixel,
            )
            if ep is not None:
                frames = ep["frames"]
                action_vecs = ep["action_vecs"]
                states = ep["states"]
                state_field_valids = ep["state_field_valids"]
                ep_len = len(action_vecs)
                expected_observation_len = ep_len + 1
                if not (
                    len(frames) == expected_observation_len
                    and len(states) == expected_observation_len
                    and len(state_field_valids) == expected_observation_len
                ):
                    raise RuntimeError(
                        "Scripted GUI state-before-action buffers must contain "
                        "exactly N+1 observations for N actions, got "
                        f"actions={ep_len}, frames={len(frames)}, "
                        f"states={len(states)}, "
                        f"validity={len(state_field_valids)}"
                    )

                video_path = temp_dir / f"attempt_{attempt_idx:06d}.mp4"
                # LeRobot has one video frame per parquet row/action. The
                # terminal observation has no matching action and must not be
                # encoded into the episode MP4.
                _write_mp4(_video_frames_for_action_rows(frames, ep_len), video_path, fps=fps)

                result_queue.put(("success", task_name, {
                    "attempt_idx": attempt_idx,
                    "task_description": ep["task_description"],
                    "task_name": ep["task_name"],
                    "action_vecs": action_vecs,
                    "states": states,
                    "state_field_valids": state_field_valids,
                    "length": ep_len,
                    "video_path": str(video_path),
                    "event_frame_idx_in_clip": ep["event_frame_idx_in_clip"],
                }))
            else:
                result_queue.put(("fail", task_name, None))
        except Exception as e:
            wlog.warning("[%s] Exception: %s", task_name, e)
            result_queue.put(("error", task_name, str(e)))


# Main: parallel collection + serial LeRobotDataset writes
def main():
    import shutil
    from gamewam.datasets.lerobot.lerobot_dataset import LeRobotDataset

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in ALL_TASK_LAYOUTS]
    if unknown:
        logger.error("Unknown tasks (not in ALL_TASK_LAYOUTS): %s", unknown)
        logger.info("Available craft tasks: %d, smelt tasks: %d",
                    len(RECIPE_LAYOUTS), len(SMELT_LAYOUTS))
        sys.exit(1)

    render_size = tuple(args.render_size)
    obs_size = tuple(args.obs_size)
    _configure_gui_geometry(render_size)
    cursor_reference_size = resolve_cursor_reference_size(
        args.cursor_reference_size,
        fallback_size=render_size,
    )
    base_width, base_height = GUI_CALIBRATION_SIZE
    cursor_pitch_deg_per_pixel = (
        float(args.cursor_pitch_deg_per_pixel)
        if args.cursor_pitch_deg_per_pixel is not None
        else DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL * base_height / render_size[1]
    )
    cursor_yaw_deg_per_pixel = (
        float(args.cursor_yaw_deg_per_pixel)
        if args.cursor_yaw_deg_per_pixel is not None
        else DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL * base_width / render_size[0]
    )
    logger.info(
        "Cursor config: reference=%sx%s, render=%sx%s, pitch_deg/px=%.6f, yaw_deg/px=%.6f",
        cursor_reference_size[0], cursor_reference_size[1],
        render_size[0], render_size[1],
        cursor_pitch_deg_per_pixel, cursor_yaw_deg_per_pixel,
    )

    global DEBUG_SAVE_FAIL_VIDEO, DEBUG_VIDEO_DIR, DEBUG_VIDEO_CAPTURE_EVERY_N
    DEBUG_SAVE_FAIL_VIDEO = bool(args.debug_save_fail_video)
    DEBUG_VIDEO_DIR = str(Path(args.output_dir) / "debug_fail_videos") if args.debug_save_fail_video else None
    DEBUG_VIDEO_CAPTURE_EVERY_N = args.debug_capture_every_n
    if DEBUG_SAVE_FAIL_VIDEO:
        logger.info("[DEBUG] fail videos will be saved to %s (every %d frames)",
                    DEBUG_VIDEO_DIR, DEBUG_VIDEO_CAPTURE_EVERY_N)

    if args.dry_run:
        logger.info("DRY RUN — printing expected action counts per task")
        for task in tasks:
            layout = ALL_TASK_LAYOUTS[task]
            if layout["task_type"] == "smelt":
                fake_slots = {layout["ingredient"]: [1], layout["fuel"]: [2]}
                setup_n = len(build_furnace_setup_actions(layout, fake_slots)[0])
                collect_n = len(build_furnace_collect_actions(fake_slots))
                est = setup_n + FURNACE_SMELT_WAIT_FRAMES + collect_n
                logger.info(
                    "  %s: smelt task -> ~%d actions (%d setup + %d wait + %d collect)",
                    task, est, setup_n, FURNACE_SMELT_WAIT_FRAMES, collect_n,
                )
            else:
                n_placements = sum(1 for row in layout["grid"] for cell in row if cell is not None)
                est = n_placements * 12 + 6
                logger.info("  %s: %d placements -> ~%d actions", task, n_placements, est)
        return

    num_workers = args.workers
    total_target = len(tasks) * args.episodes_per_task
    logger.info("Starting %d workers for %d tasks x %d episodes = %d total",
                num_workers, len(tasks), args.episodes_per_task, total_target)

    output_dir = Path(args.output_dir)
    temp_video_dir = output_dir.parent / (f"_temp_videos_{output_dir.name}")
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    task_collected: dict[str, int] = {t: 0 for t in tasks}
    task_attempts: dict[str, int] = {t: 0 for t in tasks}
    task_failures: dict[str, int] = {t: 0 for t in tasks}
    task_done: dict[str, bool] = {t: False for t in tasks}
    next_attempt_idx = 0
    total_collected = 0
    workers_exited = 0

    state_dim = RAW_STATE_DIM
    action_dim = 22
    render_h, render_w = render_size[1], render_size[0]

    meta_complete = (
        (output_dir / "meta" / "info.json").exists() and
        (output_dir / "meta" / "tasks.jsonl").exists() and
        (output_dir / "meta" / "episodes.jsonl").exists()
    )

    if meta_complete:
        dataset = LeRobotDataset(
            repo_id=str(output_dir),
            root=output_dir,
            is_compute_episode_stats_image=False,
        )
        logger.info("Resumed existing dataset: %d episodes", dataset.meta.total_episodes)
        attach_or_validate_minecraft_state_schema(
            dataset,
            cursor_reference_size=cursor_reference_size,
            cursor_source_metadata={
                "kind": "online_camera_tracker",
                "render_size": list(render_size),
                "pitch_deg_per_pixel": cursor_pitch_deg_per_pixel,
                "yaw_deg_per_pixel": cursor_yaw_deg_per_pixel,
            },
        )

        episodes_jsonl = output_dir / "meta" / "episodes.jsonl"
        if episodes_jsonl.exists():
            task_collected = _count_collected_tasks(episodes_jsonl, tasks)
            for t in tasks:
                if task_collected[t] >= args.episodes_per_task:
                    task_done[t] = True
            done_count = sum(task_done.values())
            logger.info("Resumed task counts: %d/%d tasks already done",
                        done_count, len(tasks))
    else:
        import shutil as _shutil
        if output_dir.exists():
            logger.warning("Incomplete dataset dir found at %s, removing and starting fresh.", output_dir)
            _shutil.rmtree(output_dir)
        features = {
            "observation.images": {
                "dtype": "video",
                "shape": (3, render_h, render_w),
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": render_h,
                    "video.width": render_w,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": args.fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            **minecraft_state_features(),
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"action_{i}" for i in range(action_dim)],
            },
        }
        dataset = LeRobotDataset.create(
            repo_id=str(output_dir),
            fps=args.fps,
            features=features,
            root=output_dir,
            robot_type="minecraft",
            use_videos=True,
            is_compute_episode_stats_image=False,
        )
        attach_or_validate_minecraft_state_schema(
            dataset,
            cursor_reference_size=cursor_reference_size,
            cursor_source_metadata={
                "kind": "online_camera_tracker",
                "render_size": list(render_size),
                "pitch_deg_per_pixel": cursor_pitch_deg_per_pixel,
                "yaw_deg_per_pixel": cursor_yaw_deg_per_pixel,
            },
        )
        logger.info("Created new dataset at %s", output_dir)

    dummy_img = np.zeros((render_h, render_w, 3), dtype=np.uint8)

    # Start workers. Each worker gets its own Xvfb display, so parallel
    # launch is safe (no GLFW context contention).
    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    workers_procs = []
    debug_video_dir = str(Path(args.output_dir) / "debug_fail_videos") if args.debug_save_fail_video else None
    for wid in range(num_workers):
        p = mp.Process(
            target=_worker_loop,
            args=(wid, task_queue, result_queue, render_size, obs_size,
                  args.difficulty, args.seed,
                  str(temp_video_dir), args.fps, args.bias, args.collect_mode,
                  args.debug_save_fail_video, debug_video_dir,
                  args.debug_capture_every_n, cursor_reference_size,
                  cursor_pitch_deg_per_pixel, cursor_yaw_deg_per_pixel),
            daemon=True,
        )
        p.start()
        workers_procs.append(p)

    # Pre-fill task queue (skip already-completed tasks)
    prefill_per_task = min(max(num_workers * 2 // len(tasks), 2), args.max_attempts_per_task)
    for task in tasks:
        if task_done[task]:
            continue
        recipe = ALL_TASK_LAYOUTS[task]
        for _ in range(prefill_per_task):
            task_queue.put((task, recipe, next_attempt_idx))
            next_attempt_idx += 1
            task_attempts[task] += 1

    t0 = time.time()
    last_log_time = t0
    write_ok = 0

    def _write_episode(ep_data: dict) -> bool:
        """Incrementally write one episode to the dataset."""
        current_ep_idx = dataset.meta.total_episodes
        task_desc = ep_data["task_description"]
        task_name = ep_data["task_name"]
        action_vecs = ep_data["action_vecs"]
        states = ep_data["states"]
        state_field_valids = ep_data["state_field_valids"]
        ep_len = ep_data["length"]
        src_video = Path(ep_data["video_path"])

        # Validate the payload before moving its only video file into the
        # dataset tree, preventing orphan final-path files.
        if ep_len < 2:
            logger.warning("Episode payload too small (%d frames), skipping", ep_len)
            return False
        if (
            len(action_vecs) < ep_len
            or len(states) < ep_len
            or len(state_field_valids) < ep_len
        ):
            logger.warning(
                "Episode payload length mismatch: declared=%d actions=%d states=%d validity=%d, skipping",
                ep_len, len(action_vecs), len(states), len(state_field_valids),
            )
            return False
        if not src_video.is_file():
            logger.warning("Missing temp video %s for ep %d, skipping", src_video, current_ep_idx)
            return False

        for vkey in dataset.meta.video_keys:
            dst_mp4 = dataset.root / dataset.meta.get_video_file_path(current_ep_idx, vkey)
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            if dst_mp4.exists() or dst_mp4.is_symlink():
                dst_mp4.unlink()
            shutil.move(str(src_video), str(dst_mp4))

        dataset.episode_buffer = dataset.create_episode_buffer()
        for idx in range(ep_len):
            frame_data = {
                "observation.images": dummy_img,
                "observation.state": states[idx].astype(np.float32),
                "observation.state_field_valid": state_field_valids[idx].astype(bool),
                "observation.state_dim_is_pad": np.zeros(state_dim, dtype=bool),
                "action": action_vecs[idx].astype(np.float32),
            }
            dataset.add_frame(frame_data, task=[task_desc, task_desc])

        raw_file_name, extra_fields = _scripted_gui_episode_metadata(
            task_name,
            current_ep_idx,
            ep_data["event_frame_idx_in_clip"],
        )
        dataset.save_episode(
            raw_file_name=raw_file_name,
            extra_fields=extra_fields,
        )
        return True

    while workers_exited < num_workers:
        try:
            result = result_queue.get(timeout=5)
        except queue.Empty:
            if workers_procs and all(p.exitcode is not None for p in workers_procs):
                raise RuntimeError(
                    "All GUI collection workers exited before collection completed; "
                    f"exitcodes={[p.exitcode for p in workers_procs]}"
                )
            if all(task_done.values()) and task_queue.empty():
                for _ in range(num_workers - workers_exited):
                    task_queue.put(None)
            continue

        status, task_name, payload = result
        if status == "worker_exit":
            workers_exited += 1
            if payload is not None:
                for proc in workers_procs:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError(
                    f"GUI collection worker {task_name} failed during bootstrap: {payload}"
                )
            continue

        if task_done.get(task_name, True):
            if status == "success" and payload:
                stale = Path(payload.get("video_path", ""))
                if stale.exists():
                    stale.unlink(missing_ok=True)
            continue

        if status == "success":
            if _write_episode(payload):
                write_ok += 1
                task_collected[task_name] += 1
                total_collected += 1
            else:
                task_failures[task_name] += 1
                stale = Path(payload.get("video_path", ""))
                if stale.exists():
                    stale.unlink(missing_ok=True)

            now = time.time()
            if now - last_log_time > 10:
                elapsed = now - t0
                rate = total_collected / elapsed * 3600 if elapsed > 0 else 0
                logger.info("[%s] %d/%d | total %d/%d | %.0f eps/hr | %d workers active",
                            task_name, task_collected[task_name],
                            args.episodes_per_task, total_collected, total_target,
                            rate, num_workers - workers_exited)
                last_log_time = now

            if task_collected[task_name] >= args.episodes_per_task:
                task_done[task_name] = True
                logger.info("[%s] DONE (%d attempts, %d failures)",
                            task_name, task_attempts[task_name], task_failures[task_name])
        else:
            task_failures[task_name] += 1

        if (task_attempts[task_name] >= args.max_attempts_per_task
                and not task_done[task_name]):
            task_done[task_name] = True
            logger.warning("[%s] Hit max attempts (%d). Collected %d/%d.",
                           task_name, args.max_attempts_per_task,
                           task_collected[task_name], args.episodes_per_task)

        if not task_done.get(task_name, True):
            if task_attempts[task_name] < args.max_attempts_per_task:
                recipe = ALL_TASK_LAYOUTS[task_name]
                task_queue.put((task_name, recipe, next_attempt_idx))
                next_attempt_idx += 1
                task_attempts[task_name] += 1

        if all(task_done.values()) and task_queue.empty():
            for _ in range(num_workers - workers_exited):
                task_queue.put(None)

    for p in workers_procs:
        p.join(timeout=30)

    elapsed = time.time() - t0
    logger.info("\n=== Collection Summary (%.1f min, %.0f eps/hr) ===",
                elapsed / 60, total_collected / elapsed * 3600 if elapsed > 0 else 0)
    for task in tasks:
        collected = task_collected[task]
        target = args.episodes_per_task
        logger.info("  %s: %d/%d eps (attempts: %d)",
                    task, collected, target, task_attempts[task])

    if write_ok == 0 and dataset.meta.total_episodes == 0:
        logger.error("No successful episodes collected!")
        sys.exit(1)
    logger.info("Written %d new episodes this run (dataset total: %d)",
                write_ok, dataset.meta.total_episodes)

    if temp_video_dir.exists():
        shutil.rmtree(temp_video_dir, ignore_errors=True)

    import os as _os
    base_minestudio = _os.environ.get("MINESTUDIO_DIR", "")
    if base_minestudio:
        for wid in range(num_workers):
            worker_dir = Path(base_minestudio) / f"worker_{wid}"
            if worker_dir.exists():
                shutil.rmtree(worker_dir, ignore_errors=True)

    logger.info("Dataset complete: %d episodes, %d total frames -> %s",
                dataset.meta.total_episodes, dataset.meta.total_frames, output_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
