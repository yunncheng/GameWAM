"""Calibration script for scripted GUI data collection."""

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gamewam.envs.minecraft.env import MinecraftGameEnv
from gamewam.envs.minecraft.action_space import ENV_NULL_ACTION, CAMERA_SCALER

# Camera scale for the six-subframe cursor movements used by GUI collection.
GUI_CAMERA_PITCH_DEG_PER_PX = 0.144
GUI_CAMERA_YAW_DEG_PER_PX = 0.144
from gamewam.envs.minecraft.tasks.craft_item import gen_craft_item_task_config

logger = logging.getLogger(__name__)

SCREEN_W, SCREEN_H = 640, 360
GUI_TEXTURE_W, GUI_TEXTURE_H = 176, 166
CT_GUI_LEFT = (SCREEN_W - GUI_TEXTURE_W) // 2   # 232
CT_GUI_TOP  = (SCREEN_H - GUI_TEXTURE_H) // 2   # 97


def _make_noop():
    a = copy.deepcopy(ENV_NULL_ACTION)
    a["camera"] = np.array([0.0, 0.0], dtype=np.float32)
    return a


def _save_frame(frame: np.ndarray, path: Path):
    try:
        from PIL import Image
        Image.fromarray(frame).save(str(path))
        logger.info("Saved %s", path)
    except ImportError:
        logger.warning("Pillow not installed; skipping frame save")


def calibrate_camera_pixel_mapping(
    task_name: str = "craft_item:paper",
    save_dir: Path | None = None,
):
    """Send known camera deltas and print what the frame looks like.

    The cursor is a white pixel overlay; visually inspect whether moving
    camera=[0, yaw] by known degrees shifts the cursor by the expected pixels.

    Expected: 1 pixel = CAMERA_SCALER degrees → sending camera=[0, CAMERA_SCALER*N]
    should move cursor N pixels to the right.
    """
    logger.info("=== Camera calibration for %s ===", task_name)
    task_config = gen_craft_item_task_config(task_name, difficulty="zero")

    env = MinecraftGameEnv(
        task_config=task_config,
        render_size=(SCREEN_W, SCREEN_H),
    )
    try:
        obs = env.reset()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            _save_frame(obs.image, save_dir / "00_initial.png")

        # Test moves: 18 pixels (one slot width) in each direction
        slot_px = 18
        moves = [
            ("right_1slot", [0.0, slot_px * CAMERA_SCALER]),
            ("down_1slot",  [slot_px * CAMERA_SCALER, 0.0]),
            ("right_2slot", [0.0, slot_px * 2 * CAMERA_SCALER]),
            ("up_1slot",    [-slot_px * CAMERA_SCALER, 0.0]),
        ]

        for i, (label, camera) in enumerate(moves, start=1):
            a = _make_noop()
            a["camera"] = np.array(camera, dtype=np.float32)
            obs = env.step(a)
            obs = env.step(_make_noop())  # settle
            if save_dir:
                _save_frame(obs.image, save_dir / f"{i:02d}_{label}.png")
            logger.info("  After %s (camera=%s): frame shape=%s", label, camera, obs.image.shape)

        logger.info("Camera calibration done. Inspect frames in %s", save_dir)
    finally:
        env.close()


def check_slot_coordinates(
    task_name: str = "craft_item:paper",
    save_dir: Path | None = None,
):
    """Move cursor to each key GUI slot and save screenshots.

    Visually verify that:
    - cursor moves to the expected inventory slots
    - cursor moves to crafting grid positions correctly
    - cursor reaches the output slot
    """
    from scripts.datasets.minecraft.gui.collect import (
        _callback_slot_to_screen_center,
        _crafting_table_grid_center,
        _crafting_table_output_center,
        _make_move_actions,
        CURSOR_INIT,
        read_inventory_from_obs,
    )

    logger.info("=== Slot coordinate check for %s ===", task_name)
    task_config = gen_craft_item_task_config(task_name, difficulty="zero")

    env = MinecraftGameEnv(
        task_config=task_config,
        render_size=(SCREEN_W, SCREEN_H),
    )
    try:
        obs = env.reset()
        item_slot_map = read_inventory_from_obs(obs.info)
        logger.info("Inventory: %s", item_slot_map)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            _save_frame(obs.image, save_dir / "00_after_reset.png")

        cursor = list(CURSOR_INIT)
        frame_idx = 1

        # Visit each inventory slot that has items
        for item, slots in item_slot_map.items():
            for slot_idx in slots[:1]:  # only first slot per item
                target = _callback_slot_to_screen_center(slot_idx)
                for a in _make_move_actions(tuple(cursor), target):
                    obs = env.step(a)
                cursor[:] = target
                if save_dir:
                    _save_frame(obs.image, save_dir / f"{frame_idx:02d}_inv_slot{slot_idx}_{item}.png")
                    frame_idx += 1
                logger.info("  Moved to inv slot %d (%s) → screen %s", slot_idx, item, target)

        # Visit crafting grid positions
        grid_positions = [
            ((0, 0), "grid_0_0"), ((0, 1), "grid_0_1"), ((0, 2), "grid_0_2"),
            ((1, 0), "grid_1_0"), ((1, 1), "grid_1_1"), ((2, 2), "grid_2_2"),
        ]
        for (r, c), label in grid_positions:
            target = _crafting_table_grid_center(r, c)
            for a in _make_move_actions(tuple(cursor), target):
                obs = env.step(a)
            cursor[:] = target
            if save_dir:
                _save_frame(obs.image, save_dir / f"{frame_idx:02d}_{label}.png")
                frame_idx += 1

        # Visit output slot
        target = _crafting_table_output_center()
        for a in _make_move_actions(tuple(cursor), target):
            obs = env.step(a)
        if save_dir:
            _save_frame(obs.image, save_dir / f"{frame_idx:02d}_output.png")

        logger.info("Slot check done. Inspect frames in %s", save_dir)
    finally:
        env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="craft_item:paper")
    p.add_argument("--save_frames", default="calibration_frames",
                   help="Directory to save PNG screenshots")
    p.add_argument("--check", choices=["camera", "slots", "both"], default="both")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    save_dir = Path(args.save_frames)

    if args.check in ("camera", "both"):
        calibrate_camera_pixel_mapping(
            task_name=args.task,
            save_dir=save_dir / "camera",
        )

    if args.check in ("slots", "both"):
        check_slot_coordinates(
            task_name=args.task,
            save_dir=save_dir / "slots",
        )


if __name__ == "__main__":
    main()
