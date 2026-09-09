#!/usr/bin/env python3
"""Probe recipe book card positions for all craft tasks."""

import argparse
import atexit
import copy
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault(
    "MINESTUDIO_DIR",
    str(PROJECT_ROOT / ".minestudio"),
)

from gamewam.envs.minecraft.env import MinecraftGameEnv

logger = logging.getLogger(__name__)

# Recipe book geometry (must match collect.py)
RECIPE_BOOK_BTN_2x2 = (344, 164)
RECIPE_BOOK_BTN_3x3 = (246, 139)
CURSOR_INIT = (320, 180)
RECIPE_BOOK_CARDS_PER_ROW = 5
RECIPE_BOOK_CARD_DX = 24
RECIPE_BOOK_CARD_DY = 26
FIRST_CARD_CENTER = (180, 140)
MAX_CARDS_TO_PROBE = 25

GUI_CAMERA_PITCH_DEG_PER_PX = 0.145
GUI_CAMERA_YAW_DEG_PER_PX = 0.144

# Output slot centers (from collect.py)
INV_2X2_OUTPUT_CENTER = (394, 133)
CT_OUTPUT_CENTER = (364, 140)
RECIPE_BOOK_OUTPUT_SHIFT_X = 78 + 3


def _make_noop() -> dict:
    return {
        "attack": 0, "back": 0, "forward": 0, "jump": 0, "left": 0, "right": 0,
        "sneak": 0, "sprint": 0, "use": 0, "drop": 0, "inventory": 0,
        "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0,
        "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0,
        "hotbar.9": 0, "camera": [0.0, 0.0],
    }


def _make_move_actions(from_pos: tuple, to_pos: tuple) -> list[dict]:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    a = _make_noop()
    a["camera"] = np.array([
        dy * GUI_CAMERA_PITCH_DEG_PER_PX,
        dx * GUI_CAMERA_YAW_DEG_PER_PX,
    ], dtype=np.float32)
    return [a, _make_noop(), _make_noop()]


def _make_click(button: str = "attack", hold_frames: int = 2) -> list[dict]:
    actions = []
    for _ in range(hold_frames):
        a = _make_noop()
        a[button] = 1
        actions.append(a)
    actions.append(_make_noop())
    return actions


def _output_center(is_3x3: bool) -> tuple[int, int]:
    if is_3x3:
        return (CT_OUTPUT_CENTER[0] + RECIPE_BOOK_OUTPUT_SHIFT_X,
                CT_OUTPUT_CENTER[1])
    else:
        return (INV_2X2_OUTPUT_CENTER[0] + RECIPE_BOOK_OUTPUT_SHIFT_X,
                INV_2X2_OUTPUT_CENTER[1])


def _card_xy(card_index: int) -> tuple[int, int]:
    col = card_index % RECIPE_BOOK_CARDS_PER_ROW
    row = card_index // RECIPE_BOOK_CARDS_PER_ROW
    return (FIRST_CARD_CENTER[0] + col * RECIPE_BOOK_CARD_DX,
            FIRST_CARD_CENTER[1] + row * RECIPE_BOOK_CARD_DY)


def _build_task_config(task_name: str, recipe: dict) -> dict:
    from gamewam.envs.minecraft.tasks.craft_item import gen_craft_item_task_config

    base_config = gen_craft_item_task_config(task_name, difficulty="zero")

    item_counts: dict[str, int] = {}
    for row in recipe["grid"]:
        for cell in row:
            if cell is not None:
                item_counts[cell] = item_counts.get(cell, 0) + 1

    new_inv = []
    for item_type, count in item_counts.items():
        for _ in range(count):
            new_inv.append({"slot": "random", "type": item_type, "quantity": 1})

    base_config["callback"]["init_inventory"]["init_inventory"] = new_inv
    return base_config


def _try_one_card(
    task_name: str,
    task_config: dict,
    recipe: dict,
    card_idx: int,
    save_video: bool = False,
    debug_video_dir: str | None = None,
) -> bool:
    is_3x3 = bool(recipe.get("need_crafting_table", False))
    book_btn = RECIPE_BOOK_BTN_3x3 if is_3x3 else RECIPE_BOOK_BTN_2x2

    env = MinecraftGameEnv(
        task_config=task_config,
        render_size=(640, 360),
    )
    try:
        obs = env.reset()
        frames: list = []

        for _ in range(30):
            obs = env.step(_make_noop())
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])

        for act in _make_move_actions(CURSOR_INIT, book_btn):
            obs = env.step(act)
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])
        for act in _make_click("attack"):
            obs = env.step(act)
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])
        for _ in range(4):
            obs = env.step(_make_noop())
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])

        total_reward = 0.0

        card_xy = _card_xy(card_idx)
        for act in _make_move_actions(book_btn, card_xy):
            obs = env.step(act)
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])

        for act in _make_click("attack"):
            obs = env.step(act)
            total_reward += obs.reward
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])

        # Wait for auto-fill (recipe book fills grid, item appears in output)
        for _ in range(5):
            obs = env.step(_make_noop())
            total_reward += obs.reward
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])
            if total_reward > 0:
                if save_video:
                    _save_probe_video(frames, task_name, debug_video_dir, f"card{card_idx}_success")
                return True

        # Pickup from output slot (reward fires when item enters inventory)
        out_xy = _output_center(is_3x3)
        for act in _make_move_actions(card_xy, out_xy):
            obs = env.step(act)
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])

        for act in _make_click("attack"):
            obs = env.step(act)
            total_reward += obs.reward
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])

        # Wait for reward after pickup (item enters inventory, craft_item event fires)
        for _ in range(8):
            obs = env.step(_make_noop())
            total_reward += obs.reward
            if save_video and obs.info.get("pov") is not None:
                frames.append(obs.info["pov"])
            if total_reward > 0:
                if save_video:
                    _save_probe_video(frames, task_name, debug_video_dir, f"card{card_idx}_success")
                return True

        if save_video:
            _save_probe_video(frames, task_name, debug_video_dir, f"card{card_idx}_fail")
        return False
    finally:
        env.close()
        try:
            for inst in getattr(getattr(env, '_env', None), 'instances', []):
                mc_proc = getattr(inst, 'minecraft_process', None)
                if mc_proc is not None:
                    try:
                        os.killpg(os.getpgid(mc_proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass


def _save_probe_video(frames: list, task_name: str,
                      debug_video_dir: str | None, stage: str) -> None:
    if not debug_video_dir or not frames:
        return
    import cv2
    out_dir = Path(debug_video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = task_name.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"probe_{safe_name}_{stage}.mp4"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp.name, fourcc, 20, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp.name,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path),
    ], capture_output=True)
    Path(tmp.name).unlink(missing_ok=True)


def _probe_worker(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    debug_video_dir: str | None,
    seed: int,
) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)

    wlog = logging.getLogger(f"probe-worker-{worker_id}")

    # Per-worker MineStudio directory
    original_minestudio = os.environ["MINESTUDIO_DIR"]
    worker_minestudio = os.path.join(original_minestudio, f"probe_worker_{worker_id}")
    os.makedirs(worker_minestudio, exist_ok=True)
    for shared_subdir in ("engine", "runtime"):
        src = os.path.join(original_minestudio, shared_subdir)
        dst = os.path.join(worker_minestudio, shared_subdir)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    os.environ["MINESTUDIO_DIR"] = worker_minestudio

    # Per-worker Xvfb
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
        result_queue.put(("worker_exit", worker_id, "Xvfb bootstrap failed"))
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

    atexit.register(_stop_xvfb)

    def _handle_signal(signum, _frame) -> None:
        _stop_xvfb()
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

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

        task_name, recipe = item

        try:
            card_idx = _probe_one_task(task_name, recipe, debug_video_dir)
            if card_idx is not None:
                result_queue.put(("success", task_name, card_idx))
            else:
                result_queue.put(("fail", task_name, None))
        except Exception as e:
            wlog.warning("[%s] Exception: %s", task_name, e)
            result_queue.put(("error", task_name, str(e)))


def _probe_one_task(
    task_name: str,
    recipe: dict,
    debug_video_dir: str | None = None,
) -> int | None:
    task_type = recipe.get("task_type") or "craft"
    if task_type != "craft":
        return None

    base_config = _build_task_config(task_name, recipe)

    for card_idx in range(MAX_CARDS_TO_PROBE):
        ok = _try_one_card(task_name, copy.deepcopy(base_config), recipe, card_idx)
        if ok:
            return card_idx

    # All cards failed — save debug video for card 0
    _try_one_card(task_name, copy.deepcopy(base_config), recipe, 0, save_video=True,
                  debug_video_dir=debug_video_dir)
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Probe recipe book card positions")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSON file")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names (default: all craft tasks)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug_video_dir", type=str, default=None,
                        help="If set, save probe videos to this directory")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    from scripts.datasets.minecraft.gui.collect import RECIPE_LAYOUTS

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = sorted(RECIPE_LAYOUTS.keys())

    unknown = [t for t in tasks if t not in RECIPE_LAYOUTS]
    if unknown:
        logger.error("Unknown tasks: %s", unknown)
        sys.exit(1)

    # Load existing results
    output_path = Path(args.output)
    results: dict[str, int] = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        logger.info("Loaded %d existing results from %s", len(results), output_path)

    remaining = [t for t in tasks if t not in results]
    if not remaining:
        logger.info("All %d tasks already probed", len(tasks))
        return

    num_workers = min(args.workers, len(remaining))
    logger.info("Probing %d tasks (%d already known) with %d workers",
                len(remaining), len(results), num_workers)

    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    # Prefill queue
    for task_name in remaining:
        recipe = RECIPE_LAYOUTS[task_name]
        task_queue.put((task_name, recipe))

    # Start workers
    workers_procs = []
    for wid in range(num_workers):
        p = mp.Process(
            target=_probe_worker,
            args=(wid, task_queue, result_queue, args.debug_video_dir, args.seed),
            daemon=True,
        )
        p.start()
        workers_procs.append(p)
        time.sleep(0.5)  # stagger to reduce MC boot contention

    # Collect results
    pending = len(remaining)
    success = 0
    fail = 0
    error = 0
    workers_exited = 0

    while pending > 0:
        try:
            result = result_queue.get(timeout=5)
        except queue.Empty:
            if workers_procs and all(p.exitcode is not None for p in workers_procs):
                raise RuntimeError(
                    "All probe workers exited before collection completed; "
                    f"exitcodes={[p.exitcode for p in workers_procs]}"
                )
            continue

        status = result[0]

        if status == "worker_exit":
            workers_exited += 1
            if workers_exited >= num_workers:
                break
            continue

        task_name = result[1]

        if status == "success":
            card_idx = result[2]
            results[task_name] = card_idx
            success += 1
            pending -= 1
            logger.info("[%s] card_idx=%d (%d/%d succeeded, %d remaining)",
                        task_name, card_idx, success, pending + success + fail + error, pending)
        elif status == "fail":
            fail += 1
            pending -= 1
            logger.warning("[%s] No card worked after %d attempts", task_name, MAX_CARDS_TO_PROBE)
        elif status == "error":
            error += 1
            pending -= 1
            logger.error("[%s] Error: %s", task_name, result[2])

        # Save incrementally
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Shutdown workers
    for _ in range(num_workers):
        task_queue.put(None)

    for p in workers_procs:
        p.join(timeout=30)

    logger.info("Done. %d succeeded, %d failed, %d errors. Results saved to %s",
                success, fail, error, output_path)


if __name__ == "__main__":
    main()
