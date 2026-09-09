"""Convert VPT data into event-anchored sub-clip LeRobot datasets.

Events are detected from VPT JSONL ``stats`` via frame-to-frame delta. Each
sub-clip is a win_len-frame window around a detected event, with dual-task
labeling (coarse batch + fine event).
"""

import bisect
import json
import multiprocessing as mp
import shutil
import signal
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

# Disable OpenCV's parallel backend — prevents fork deadlock when worker
# processes inherit the parent's OpenMP/TBB thread pool state, and avoids
# 32 workers × N OpenMP threads contention.
cv2.setNumThreads(0)

# Use spawn start method for ProcessPoolExecutor — spawn creates fresh
# processes that don't inherit parent memory/threads, eliminating the
# fork deadlock caused by cv2/numpy OpenMP mutexes.
_MP_CTX = mp.get_context("spawn")

from gamewam.datasets.lerobot.lerobot_dataset import LeRobotDataset
from gamewam.envs.minecraft.action_space import action_to_vector
from gamewam.envs.minecraft.tasks.catalog import EVENT_DESCRIPTION
from gamewam.envs.minecraft.tasks.catalog import VPT_BATCH_TASKS
from gamewam.envs.minecraft.observations.state import (
    RAW_STATE_DIM,
    extract_state_with_validity,
    normalize_screen_size,
)
from .state_tracking import MinecraftStateTracker

from gamewam.datasets.schemas.minecraft import (
    attach_or_validate_minecraft_state_schema,
    minecraft_state_features,
    resolve_cursor_reference_size,
)
from .task_utils import assert_output_dir_empty
from ...common.video import build_video_info
from .vpt import (
    BATCH_NAMES,
    DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
    _build_null_action_keep_mask,
    _extract_info_from_raw_action,
    _extract_vpt_mouse_pair,
    _infer_batch_name,
    _json_action_to_env_action,
    _map_vpt_cursor_to_reference,
    _load_vpt_jsonl,
    _overlay_gui_cursor,
    _probe_video_meta,
    discover_vpt_batches,
)

# Event prefixes with agent-level goals (drop xp/weather/etc noise).
MEANINGFUL_EVENT_PREFIXES = frozenset({
    "mine_block",
    "kill_entity",
    "craft_item",
    "use_item",
    "pickup",
    "drop",
})

# Functional custom events (concrete player actions) kept for training.
# Names must match the Minecraft 1.16 stat identifier (the part after
# "minecraft.custom:minecraft.") AND exist in EVENT_DESCRIPTION.
MEANINGFUL_CUSTOM_EVENTS = frozenset({
    "animals_bred",
    "bell_ring",
    "damage_absorbed",
    "damage_blocked_by_shield",
    "damage_dealt",
    "damage_resisted",
    "damage_taken",
    "deaths",
    "eat_cake_slice",
    "enchant_item",
    "fill_cauldron",
    "fish_caught",
    "inspect_dispenser",
    "inspect_dropper",
    "inspect_hopper",
    "mob_kills",
    "open_barrel",
    "open_chest",
    "open_enderchest",
    "open_shulker_box",
    "play_noteblock",
    "pot_flower",
    "raid_trigger",
    "raid_win",
    "sleep_in_bed",
    "talked_to_villager",
    "target_hit",
    "traded_with_villager",
    "trigger_trapped_chest",
    "tune_noteblock",
    "use_cauldron",
})

# Custom event prefixes to keep (interact_with_* family).
MEANINGFUL_CUSTOM_PREFIXES = (
    "interact_with_",
)


# JSONL-based event detection (frame-accurate, no LMDB alignment issues)

def detect_events_from_jsonl(
    raw_actions: List[Dict[str, Any]],
) -> List[Tuple[int, str, int]]:
    """Detect events from VPT JSONL stats by computing frame-to-frame deltas.

    Unlike the MineStudio event LMDB (whose ``event_time`` lives in the
    LMDB's internal, possibly-filtered coordinate system), this function
    operates directly on the JSONL so the returned frame indices are
    guaranteed to align 1:1 with the VPT video and action arrays.

    Args:
        raw_actions: Per-frame action dicts loaded by ``_load_vpt_jsonl``.
            Each dict must contain a ``stats`` sub-dict with cumulative
            Minecraft statistics.

    Returns:
        List of ``(frame_index, event_key, delta)`` tuples.
        ``event_key`` follows the ``<prefix>:<item>`` convention used by
        ``EVENT_DESCRIPTION`` (e.g. ``"mine_block:oak_log"``).
    """
    # Stat prefixes that correspond to MEANINGFUL_EVENT_PREFIXES + "custom".
    DETECT_PREFIXES = {
        "minecraft.mine_block": "mine_block",
        "minecraft.kill_entity": "kill_entity",
        "minecraft.craft_item": "craft_item",
        "minecraft.use_item": "use_item",
        "minecraft.pickup": "pickup",
        "minecraft.drop": "drop",
        "minecraft.custom": "custom",
    }

    events: List[Tuple[int, str, int]] = []
    prev_stats: Dict[str, int] = {}

    for t, raw_act in enumerate(raw_actions):
        curr_stats = raw_act.get("stats", {})
        if not curr_stats:
            continue

        for stat_key, curr_val in curr_stats.items():
            prev_val = prev_stats.get(stat_key, 0)
            delta = curr_val - prev_val
            if delta <= 0:
                continue

            # Match stat key to event prefix.
            for mc_prefix, event_prefix in DETECT_PREFIXES.items():
                if stat_key.startswith(mc_prefix + ".") or stat_key.startswith(mc_prefix + ":"):
                    # Extract item name after the prefix + separator.
                    item_part = stat_key[len(mc_prefix) + 1:]
                    # Strip "minecraft." namespace from item.
                    if item_part.startswith("minecraft."):
                        item_part = item_part[len("minecraft."):]
                    event_key = f"{event_prefix}:{item_part}"
                    events.append((t, event_key, delta))
                    break

        prev_stats = dict(curr_stats)

    return events


def filter_jsonl_events(
    raw_events: List[Tuple[int, str, int]],
    win_len: int,
    bias: int = 0,
) -> List[Tuple[int, str]]:
    """Filter events detected from JSONL by semantic gates and window bounds.

    Returns list of ``(frame_index, event_key)`` for events that pass
    both the MEANINGFUL_* gates and the EVENT_DESCRIPTION lookup.
    """
    filtered: List[Tuple[int, str]] = []
    total = len(raw_events)

    for frame_idx, event_key, _delta in raw_events:
        # Validate event_key against MEANINGFUL_* gates.
        colon_idx = event_key.find(":")
        if colon_idx < 0:
            continue
        prefix = event_key[:colon_idx]

        if prefix == "custom":
            item_part = event_key[colon_idx + 1:]
            is_interact = any(
                item_part.startswith(s) for s in MEANINGFUL_CUSTOM_PREFIXES
            )
            is_functional = item_part in MEANINGFUL_CUSTOM_EVENTS
            if not (is_interact or is_functional):
                continue
        else:
            if prefix not in MEANINGFUL_EVENT_PREFIXES:
                continue

        if event_key not in EVENT_DESCRIPTION:
            continue

        start_frame, _ = _compute_window_bounds(frame_idx, win_len, bias)
        if start_frame < 0:
            continue

        filtered.append((frame_idx, event_key))

    return filtered


def _probe_one(mp4_path: str) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    """Module-level worker for ProcessPoolExecutor: probe video metadata."""
    return mp4_path, _probe_video_meta(mp4_path)


def _io_alarm_handler(signum, frame):
    """SIGALRM handler — must raise to prevent kernel auto-retry (PEP 475).

    Interrupts blocked I/O syscalls (e.g. NFS-stalled ``read()``) so the worker
    can return gracefully instead of hanging forever.
    """
    raise TimeoutError("worker I/O timeout (possible NFS stall)")


# Per-file I/O timeout for null-mask workers (seconds).  A 245 MB JSONL at
# ~1 MB/s (heavily contended NFS) takes ~245 s; 300 s gives margin for
# legitimate slow reads while catching true stalls.
_NULL_MASK_IO_TIMEOUT = 300

# Outer watchdog timeout for ``as_completed`` loops (seconds).  If no future
# completes within this window, in-flight files are logged and skipped.
# Set well above ``_NULL_MASK_IO_TIMEOUT`` so SIGALRM handles the common case
# and this only fires when a worker is stuck in a way SIGALRM can't interrupt.
_WATCHDOG_TIMEOUT = 600


def _compute_null_mask(
    jsonl_path: str,
    mp4_path: str,
    video_meta: Optional[Tuple[int, int, int]],
    force_keep_indices: Optional[List[int]],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Module-level worker for ProcessPoolExecutor: compute null-action keep_mask."""
    try:
        # SIGALRM around the only I/O call — interrupts NFS-stalled read().
        # PEP 475: handler must raise, else kernel retries the interrupted
        # syscall and we stay hung.
        old_handler = signal.signal(signal.SIGALRM, _io_alarm_handler)
        signal.alarm(_NULL_MASK_IO_TIMEOUT)
        try:
            raw_acts = _load_vpt_jsonl(jsonl_path)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if not raw_acts:
            return jsonl_path, None
        if video_meta is None:
            return jsonl_path, None
        n = min(video_meta[0], len(raw_acts))
        attack_is_stuck = False
        is_null_flags = []
        for j in range(n):
            _, is_null, attack_is_stuck = _json_action_to_env_action(
                raw_acts[j], attack_is_stuck=attack_is_stuck, frame_idx=j,
            )
            is_null_flags.append(is_null)
        keep_mask = _build_null_action_keep_mask(
            is_null_flags,
            force_keep_indices=force_keep_indices,
        )
        return jsonl_path, {
            "n_original": n,
            "keep_mask": keep_mask,
            "n_filtered": int(keep_mask.sum()),
            "mp4_path": mp4_path,
        }
    except Exception as exc:
        print(f"  [WARN] null-mask failed for {jsonl_path}: {exc}")
        return jsonl_path, None


def _detect_events_from_jsonl_file(
    jsonl_path: str,
    stem: str,
    win_len: int,
    bias: int,
) -> Tuple[List[Tuple[str, int, str]], int]:
    """Module-level worker for ProcessPoolExecutor: detect & filter events.

    Loads JSONL, computes delta events, filters by MEANINGFUL gates and
    window bounds.  Returns ``(events, n_frames)`` where *events* is a list
    of ``(stem, frame_idx, event_key)`` tuples and *n_frames* is the total
    number of frames in the JSONL (needed by step [3/6] without re-loading).
    Designed as a top-level function so it can be pickled for multiprocess.
    """
    raw_actions = _load_vpt_jsonl(jsonl_path)
    raw_events = detect_events_from_jsonl(raw_actions)
    filtered = filter_jsonl_events(raw_events, win_len=win_len, bias=bias)
    events = [(stem, frame_idx, evt_key) for frame_idx, evt_key in filtered]
    return events, len(raw_actions)


def _extract_clips_single_pass(
    src_mp4: str,
    clip_specs: List[Tuple[int, int, int, Path]],
    raw_actions: List[Dict[str, Any]],
    fps: int = 20,
    keep_mask: Optional[np.ndarray] = None,
    mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
) -> Set[int]:
    """Single-pass video decode: apply keep_mask + cursor overlay, extract clips.

    Decodes the source video sequentially with OpenCV, optionally applies
    keep_mask filtering and GUI cursor overlay, then distributes frames
    to per-clip ffmpeg encoders.

    Guarantees 1:1 frame-action alignment: frames are read in order with
    no seeking, filtered only by the boolean keep_mask, and written
    directly to encoders.

    Args:
        src_mp4: Source video path.
        clip_specs: List of (ok_idx, start_frame, num_frames, dst_path).
            When keep_mask is provided, start_frame/num_frames are in
            filtered-frame space; otherwise in original-frame space.
        raw_actions: Per-frame raw action dicts (for cursor overlay).
        fps: Video FPS.
        keep_mask: If provided, only frames where mask[i] is True are kept.

    Returns:
        Set of ok_idx values for clips that succeeded.
    """
    if not clip_specs:
        return set()

    clips_sorted = sorted(clip_specs, key=lambda c: c[1])

    cap = cv2.VideoCapture(src_mp4)
    if not cap.isOpened():
        cap.release()
        return set()

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-an",
    ]

    # active encoders: [start, end, proc, ok_idx, written]
    active: List[List] = []
    succeeded: Set[int] = set()
    next_clip = 0

    frame_idx = 0
    output_idx = 0
    max_output = max(s + n for _, s, n, _ in clips_sorted)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if keep_mask is not None:
                if frame_idx >= len(keep_mask):
                    break
                if not keep_mask[frame_idx]:
                    frame_idx += 1
                    continue

            # Open encoders for clips starting at or before output_idx
            while next_clip < len(clips_sorted):
                ok_idx, clip_start, clip_n, dst_path = clips_sorted[next_clip]
                if clip_start > output_idx:
                    break
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                try:
                    proc = subprocess.Popen(
                        ffmpeg_cmd + [str(dst_path)],
                        stdin=subprocess.PIPE,
                    )
                    active.append([clip_start, clip_start + clip_n, proc, ok_idx, 0])
                except Exception:
                    pass
                next_clip += 1

            # Overlay GUI cursor
            if frame_idx < len(raw_actions):
                _overlay_gui_cursor(
                    frame,
                    raw_actions[frame_idx],
                    mouse_reference_size=mouse_reference_size,
                )

            # Distribute frame to active encoders
            frame_bytes = frame.tobytes()
            still_active = []
            for entry in active:
                start, end, proc, ok_idx, written = entry
                if start <= output_idx < end:
                    try:
                        proc.stdin.write(frame_bytes)
                        written += 1
                    except (BrokenPipeError, OSError):
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass
                        proc.kill()
                        continue
                entry[4] = written

                if written >= end - start:
                    try:
                        proc.stdin.close()
                        proc.wait(timeout=30)
                        if proc.returncode == 0 and written > 0:
                            succeeded.add(ok_idx)
                    except Exception:
                        proc.kill()
                    continue

                still_active.append(entry)
            active = still_active

            if output_idx >= max_output and not active:
                break

            frame_idx += 1
            output_idx += 1
    finally:
        cap.release()
        for entry in active:
            _, _, proc, _, _ = entry
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    return succeeded


# Episode grouping & VPT file resolution

def build_episode_groups(
    filtered_events: List[Tuple[str, int, str]],
) -> Dict[str, List[Tuple[int, str]]]:
    """Group filtered events by episode name, sorted by event_time."""
    groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for ep_name, evt_time, evt_key in filtered_events:
        groups[ep_name].append((evt_time, evt_key))
    for ep_name in groups:
        groups[ep_name].sort(key=lambda x: x[0])
    print(f"Grouped into {len(groups)} episodes with events")
    return dict(groups)


def _compute_window_bounds(anchor_frame: int, win_len: int, bias: int) -> Tuple[int, int]:
    """Return [start, end) bounds with bias=0 ending on the anchor frame."""
    end_frame = anchor_frame + 1 + bias
    start_frame = end_frame - win_len
    return start_frame, end_frame


def _scan_pre_anchor_events(
    events: List[Tuple[int, str]],
    evt_key: str,
    anchor_rel: int,
    start_frame: int,
    *,
    skip_null_actions: bool,
    keep_mask: Optional[np.ndarray] = None,
    cum: Optional[np.ndarray] = None,
    n_raw: int = 0,
) -> Tuple[List[int], List[int]]:
    """Find same-key events in the sub-clip that fall BEFORE the anchor.

    Returns two parallel lists: ``(pre_anchor_event_times,
    pre_anchor_event_frame_idxs_in_clip)``. Both are sorted by
    in-clip position. The anchor itself is excluded.

    Coordinate systems:
      - unfiltered: raw frame indices, ``t - start_frame``.
      - filtered:   translate via cum, ``cum[t] - start_frame``;
        events whose raw frame was null-filtered are skipped.

    Args:
        events: per-episode ``(raw_time, key)`` tuples.
        evt_key: the anchor's event key.
        anchor_rel: anchor's in-clip position (exclusive upper bound
            for the scan).
        start_frame: sub-clip's start in raw (or filtered) coords.
        skip_null_actions: if True, use the cum/keep_mask path.
        keep_mask: required when ``skip_null_actions=True``.
        cum: required when ``skip_null_actions=True``.
        n_raw: required when ``skip_null_actions=True``.
    """
    pre_anchor_event_times: List[int] = []
    pre_anchor_event_frame_idxs_in_clip: List[int] = []
    if skip_null_actions:
        assert keep_mask is not None and cum is not None
        for t, k in events:
            if k != evt_key or t < 0 or t >= n_raw:
                continue
            if not bool(keep_mask[t]):
                continue
            rel = int(cum[t]) - start_frame
            if 0 <= rel < anchor_rel:
                pre_anchor_event_times.append(int(t))
                pre_anchor_event_frame_idxs_in_clip.append(rel)
    else:
        for t, k in events:
            if k != evt_key:
                continue
            rel = t - start_frame
            if 0 <= rel < anchor_rel:
                pre_anchor_event_times.append(int(t))
                pre_anchor_event_frame_idxs_in_clip.append(rel)
    return pre_anchor_event_times, pre_anchor_event_frame_idxs_in_clip


def resolve_vpt_file_index(vpt_root: str) -> Dict[str, Tuple[str, str, str]]:
    """Scan VPT data dirs and build stem -> (mp4_path, jsonl_path, batch_name) index."""
    vpt_path = Path(vpt_root)
    if not vpt_path.is_dir():
        raise FileNotFoundError(f"VPT root directory not found: {vpt_root}")

    def _is_batch_dir(p: Path) -> bool:
        return p.is_dir() and p.name.endswith("xx") and p.name[:-2].isdigit()

    index: Dict[str, Tuple[str, str, str]] = {}
    batch_dirs = [vpt_path] if _is_batch_dir(vpt_path) else []
    if not batch_dirs:
        batch_dirs = sorted(d for d in vpt_path.iterdir() if _is_batch_dir(d))
    for batch_dir in batch_dirs:
        batch_name = _infer_batch_name(str(batch_dir))
        mp4_files = sorted(batch_dir.rglob("*.mp4"))
        jsonl_by_stem = {p.stem: p for p in batch_dir.rglob("*.jsonl")}
        for mp4 in mp4_files:
            jsonl = jsonl_by_stem.get(mp4.stem)
            if jsonl is None:
                sibling = mp4.with_suffix(".jsonl")
                if sibling.exists():
                    jsonl = sibling
            if jsonl is not None:
                index[mp4.stem] = (str(mp4), str(jsonl), batch_name)

    print(f"VPT file index: {len(index)} mp4+jsonl pairs across {len(batch_dirs)} batch(es)")
    return index


def max_min_fair_quotas(
    key_counts: Dict[str, int],
    total_budget: int,
) -> Dict[str, int]:
    """Allocate ``total_budget`` across keys using max-min fairness."""
    quotas: Dict[str, int] = {k: 0 for k in key_counts}
    active: Set[str] = {k for k, c in key_counts.items() if c > 0}
    remaining = int(total_budget)

    if remaining <= 0 or not active:
        return quotas

    while active and remaining > 0:
        n_active = len(active)
        share = remaining // n_active

        if share == 0:
            # Remainder < n_active. Give +1 to the scarcest keys first
            # (deterministic, reproducible). This is the natural extension
            # of max-min fair to the integer-rounding case.
            for k in sorted(active, key=lambda x: (key_counts[x], x)):
                if remaining == 0:
                    break
                if quotas[k] < key_counts[k]:
                    quotas[k] += 1
                    remaining -= 1
            break

        saturated: Set[str] = set()
        for k in active:
            avail = key_counts[k] - quotas[k]
            grant = min(share, avail)
            quotas[k] += grant
            remaining -= grant
            if quotas[k] >= key_counts[k]:
                saturated.add(k)

        if not saturated:
            # No one saturated this round → every active key absorbed
            # exactly `share`. The next iteration will split the tail
            # evenly again, so this loop is guaranteed to make progress
            # (either remaining hits 0 or `share` eventually drops to 0).
            continue
        active -= saturated

    return quotas


def _pick_indices_by_episode(
    indices: List[int],
    valid_tasks: List[Dict[str, Any]],
    n_keep: int,
) -> List[int]:
    """Select ``n_keep`` indices from ``indices`` via episode round-robin.

    Keeps per-key sampling diverse: when one key has many candidates
    drawn from a few dirt-heavy episodes, this prevents those episodes
    from monopolizing the key's quota.
    """
    if n_keep >= len(indices):
        return list(indices)
    per_ep: Dict[str, List[int]] = defaultdict(list)
    ep_order: List[str] = []
    for i in indices:
        ep = valid_tasks[i]["ep_name"]
        if ep not in per_ep:
            ep_order.append(ep)
        per_ep[ep].append(i)

    picked: List[int] = []
    round_idx = 0
    while len(picked) < n_keep:
        progressed = False
        for ep in ep_order:
            if round_idx < len(per_ep[ep]):
                picked.append(per_ep[ep][round_idx])
                progressed = True
                if len(picked) >= n_keep:
                    break
        if not progressed:
            break
        round_idx += 1
    return picked


def stratified_cap_to_max(
    valid_tasks: List[Dict[str, Any]],
    max_episodes: int,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """Cap ``valid_tasks`` to ``max_episodes`` via max-min fair per-key quotas."""
    if len(valid_tasks) <= max_episodes:
        key_counts = defaultdict(int)
        for t in valid_tasks:
            key_counts[t["event_key"]] += 1
        # No capping needed — quota == count for every key.
        return valid_tasks, dict(key_counts), dict(key_counts)

    # Group candidates by evt_key
    by_key: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(valid_tasks):
        by_key[t["event_key"]].append(i)
    key_counts: Dict[str, int] = {k: len(v) for k, v in by_key.items()}

    # Allocate quotas
    quotas = max_min_fair_quotas(key_counts, max_episodes)

    # Select indices within each key
    rng = np.random.default_rng(seed) if shuffle else None
    kept_indices: List[int] = []
    for k, indices in by_key.items():
        n = quotas[k]
        if n <= 0:
            continue
        if n >= len(indices):
            kept_indices.extend(indices)
            continue
        if shuffle:
            picked_rel = rng.choice(len(indices), size=n, replace=False)
            kept_indices.extend(int(indices[i]) for i in picked_rel)
        else:
            kept_indices.extend(
                _pick_indices_by_episode(indices, valid_tasks, n)
            )

    # Sort by original index so downstream sees a deterministic, roughly
    # time-ordered list — useful when eyeballing logs / diffs.
    kept_indices.sort()
    return [valid_tasks[i] for i in kept_indices], quotas, key_counts


# Single-batch driver

def convert_vpt_event_dataset(
    vpt_root: str,
    output_dir: str,
    fps: int = 20,
    win_len: int = 256,
    bias: int = 0,
    min_event_stride: int = 64,
    action_dim: int = 22,
    state_dim: int = RAW_STATE_DIM,
    cursor_reference_size: Optional[Tuple[float, float]] = None,
    vpt_mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
    skip_null_actions: bool = False,
    min_clip_stride: int = 1,
    max_episodes: Optional[int] = None,
    ratio: Optional[float] = None,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 32,
) -> List[str]:
    """Convert VPT data into event-driven sub-clip LeRobot dataset."""
    if state_dim != RAW_STATE_DIM:
        raise ValueError(
            f"Minecraft VPT conversion requires state_dim={RAW_STATE_DIM}, got {state_dim}"
        )
    if cursor_reference_size is not None:
        cursor_reference_size = normalize_screen_size(
            cursor_reference_size, name="cursor_reference_size"
        )
    vpt_mouse_reference_size = normalize_screen_size(
        vpt_mouse_reference_size, name="vpt_mouse_reference_size"
    )

    # Validate stride parameters
    # Both gates in [3/6] operate on per-episode frame distance; their
    # composition has a degenerate regime: if min_clip_stride exceeds
    # min_event_stride, the event-stride gate is strictly dominated by the
    # clip gate (any distance rejected by event is also rejected by clip),
    # so min_event_stride silently becomes a no-op. Fail fast rather than
    # let a misconfigured run burn hours of event detection.
    if min_clip_stride < 1:
        raise ValueError(f"min_clip_stride must be >= 1, got {min_clip_stride}")
    if min_clip_stride > min_event_stride:
        raise ValueError(
            f"min_clip_stride ({min_clip_stride}) > min_event_stride "
            f"({min_event_stride}). In this regime the event-stride gate has "
            f"no effect on the kept set (clip gate strictly dominates). "
            f"Either raise min_event_stride >= min_clip_stride, or set "
            f"min_event_stride=0 if you only want the clip gate."
        )

    # Detect + filter. Aux caches (video_meta_cache, _filtered_cache,
    # jsonl_n_frames) are needed by the writer; they are partitioned by
    # batch in the pooled path and held verbatim in the single-batch path.
    valid_tasks, video_meta_cache, _filtered_cache, jsonl_n_frames = (
        _run_event_detection_and_filtering(
            vpt_root=vpt_root,
            win_len=win_len,
            bias=bias,
            min_event_stride=min_event_stride,
            min_clip_stride=min_clip_stride,
            skip_null_actions=skip_null_actions,
            cursor_reference_size=cursor_reference_size,
            vpt_mouse_reference_size=vpt_mouse_reference_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
            max_episodes=max_episodes,
            ratio=ratio,
        )
    )
    if not valid_tasks:
        print(f"\nDone: 0 converted -> {output_dir}")
        return []

    # Cap the kept set to max_episodes using max-min fair per-key quotas.
    # The buffer (+max(100, max_episodes//1000)) gives the writer some
    # slack so it can hit the exact cap after any extraction failures.
    if max_episodes is not None and len(valid_tasks) > max_episodes:
        before = len(valid_tasks)
        target_with_buffer = min(
            len(valid_tasks),
            max_episodes + max(100, max_episodes // 1000),
        )
        valid_tasks, quotas, key_counts_before = stratified_cap_to_max(
            valid_tasks, target_with_buffer, shuffle=shuffle, seed=seed,
        )
        key_counts_after: Dict[str, int] = defaultdict(int)
        for t in valid_tasks:
            key_counts_after[t["event_key"]] += 1
        num_keys = len(quotas)
        num_saturated = sum(
            1 for k, q in quotas.items()
            if q > 0 and key_counts_before.get(k, 0) == q
        )
        top3 = sorted(key_counts_after.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{k}={v}" for k, v in top3)
        print(
            f"  Stratified cap (target={max_episodes}, with_buffer={target_with_buffer}): "
            f"{before} -> {len(valid_tasks)} sub-clips | "
            f"{num_keys} unique keys, {num_saturated} saturated | "
            f"top-3: {top3_str}"
        )

    return _write_lerobot_dataset(
        valid_tasks=valid_tasks,
        video_meta_cache=video_meta_cache,
        filtered_cache=_filtered_cache,
        jsonl_n_frames=jsonl_n_frames,
        output_dir=output_dir,
        fps=fps,
        win_len=win_len,
        bias=bias,
        action_dim=action_dim,
        state_dim=state_dim,
        cursor_reference_size=cursor_reference_size,
        vpt_mouse_reference_size=vpt_mouse_reference_size,
        skip_null_actions=skip_null_actions,
        num_workers=num_workers,
        max_episodes=max_episodes,
    )


# Multi-batch driver


def convert_vpt_event_multi_batch(
    vpt_root: str,
    output_dir: str,
    fps: int = 20,
    win_len: int = 256,
    bias: int = 0,
    min_event_stride: int = 64,
    min_clip_stride: int = 1,
    action_dim: int = 22,
    state_dim: int = RAW_STATE_DIM,
    cursor_reference_size: Optional[Tuple[float, float]] = None,
    vpt_mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
    skip_null_actions: bool = False,
    batch_filter: Optional[List[str]] = None,
    batch_ratios: Optional[Dict[str, float]] = None,
    batch_max_episodes: Optional[Dict[str, int]] = None,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 32,
) -> List[str]:
    """Convert all VPT batches (6xx→10xx) into event-anchored sub-clips."""
    vpt_batches = discover_vpt_batches(vpt_root)

    target_batches = batch_filter if batch_filter else list(BATCH_NAMES)

    print(f"\nVPT Event Multi-Batch: vpt_root={vpt_root} "
          f"vpt={sorted(vpt_batches.keys())} "
          f"target={target_batches} ratios={batch_ratios} caps={batch_max_episodes}")

    # Fail fast on stale output_dir contents.
    assert_output_dir_empty(output_dir)

    all_created: List[str] = []

    for b_idx, batch_name in enumerate(target_batches):
        if batch_name not in vpt_batches:
            print(f"\n[SKIP] {batch_name}: no VPT data directory")
            continue

        batch_vpt_root = vpt_batches[batch_name]
        per_batch_ratio = None
        if batch_ratios:
            per_batch_ratio = batch_ratios.get(batch_name)
        per_batch_cap = None
        if batch_max_episodes:
            per_batch_cap = batch_max_episodes.get(batch_name)

        print(f"\nBatch: {batch_name} ratio={per_batch_ratio} cap={per_batch_cap}")

        created = convert_vpt_event_dataset(
            vpt_root=batch_vpt_root,
            output_dir=output_dir,
            fps=fps,
            win_len=win_len,
            bias=bias,
            min_event_stride=min_event_stride,
            action_dim=action_dim,
            state_dim=state_dim,
            cursor_reference_size=cursor_reference_size,
            vpt_mouse_reference_size=vpt_mouse_reference_size,
            skip_null_actions=skip_null_actions,
            min_clip_stride=min_clip_stride,
            max_episodes=per_batch_cap,
            ratio=per_batch_ratio,
            shuffle=shuffle,
            seed=seed + b_idx,
            num_workers=num_workers,
        )
        all_created.extend(created)

    print(f"\nAll batches done: {len(all_created)} sub-clips -> {output_dir}")
    return all_created


# Pipeline helpers for the 6-stage single-batch driver:
#   _run_event_detection_and_filtering: steps [1-3/6]
#       Scan VPT files, detect events from JSONL stats, probe video
#       metadata, and run the joint clip/event stride gates. Returns
#       ``valid_tasks`` (with ``batch_name`` set) plus the aux caches
#       that the writer needs.
#   _write_lerobot_dataset: steps [4-6/6]
#       Resolve the per-task cursor reference size, create / append the
#       LeRobot dataset, extract video clips in parallel, and save
#       episodes. Returns the list of created episode paths.
#
# Separating detection/filtering from writing is what lets
# ``convert_vpt_event_pooled`` run a single max-min fair cap over the
# union of all batch subdirs' events, then write per-batch LeRobot
# datasets without re-running the expensive event-detection phase.


def _run_event_detection_and_filtering(
    vpt_root: str,
    win_len: int,
    bias: int,
    min_event_stride: int,
    min_clip_stride: int,
    skip_null_actions: bool,
    cursor_reference_size: Optional[Tuple[float, float]],
    vpt_mouse_reference_size: Tuple[float, float],
    shuffle: bool,
    seed: int,
    num_workers: int,
    max_episodes: Optional[int] = None,
    ratio: Optional[float] = None,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Optional[Tuple[int, int, int]]],
    Dict[str, Dict[str, Any]],
    Dict[str, int],
]:
    """Steps [1-3/6]: scan VPT files, detect events, joint filter.

    Does NOT apply the post-filter max-min cap. Returns the
    ``valid_tasks`` list with ``batch_name`` set, plus the auxiliary
    caches that ``_write_lerobot_dataset`` needs to extract clips.

    When ``vpt_root`` is a parent directory containing multiple batch
    subdirs (6xx/7xx/...), events from all batches are pooled into a
    single ``valid_tasks`` list — the cap and per-batch write are
    handled separately by the caller.

    ``max_episodes`` is optional and only used for the cosmetic
    "(will stop after N out of M total)" diagnostic printed in [3/6];
    the actual cap is applied by the caller between this helper and
    the writer.
    """
    print("\n[1/6] Scanning VPT files and detecting events ...")
    vpt_index = resolve_vpt_file_index(vpt_root)

    all_filtered_events: List[Tuple[str, int, str]] = []
    jsonl_n_frames: Dict[str, int] = {}  # jsonl_path -> len(raw_actions)

    jsonl_to_stem: Dict[str, str] = {}
    for stem, (mp4_path, jsonl_path, _batch_name) in vpt_index.items():
        jsonl_to_stem[jsonl_path] = stem

    actual_workers = min(num_workers, max(len(vpt_index), 1))
    print(f"  Scanning {len(vpt_index)} VPT file(s) with {actual_workers} worker(s) ...")
    with ProcessPoolExecutor(max_workers=actual_workers, mp_context=_MP_CTX) as executor:
        futures = {
            executor.submit(
                _detect_events_from_jsonl_file, jp, stem, win_len, bias,
            ): jp
            for jp, stem in jsonl_to_stem.items()
        }
        with tqdm(total=len(futures), desc="  Event detection", unit="file") as pbar:
            for future in as_completed(futures):
                jp = futures[future]
                try:
                    events, n_frames = future.result()
                    all_filtered_events.extend(events)
                    jsonl_n_frames[jp] = n_frames
                except Exception as exc:
                    stem = jsonl_to_stem.get(jp, jp)
                    print(f"    [WARN] event detection failed for {stem}: {exc}")
                pbar.update(1)

    filtered_events = all_filtered_events
    print(f"  Total events from JSONL: {len(filtered_events)}")

    # The rarity prior orders filtering candidates. The final stratified cap
    # recomputes per-key counts from the filtered tasks, so its counts may be
    # smaller for common event keys.
    global_key_count_raw: Counter = Counter(k for _, _, k in filtered_events)
    print(
        f"  Rarity prior: {len(global_key_count_raw)} unique evt_keys, "
        f"head={global_key_count_raw.most_common(3)}, "
        f"tail_count={sum(1 for _, c in global_key_count_raw.items() if c == 1)} "
        f"key(s) with count=1"
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(filtered_events))
        filtered_events = [filtered_events[i] for i in idx]
        print(f"Shuffled {len(filtered_events)} events with seed={seed}")

    if ratio is not None:
        assert 0.0 < ratio <= 1.0, f"ratio must be in (0, 1], got {ratio}"
        before = len(filtered_events)
        n_keep = max(1, int(len(filtered_events) * ratio))
        filtered_events = filtered_events[:n_keep]
        print(f"Applied ratio={ratio}: {before} -> {len(filtered_events)} events")

    episode_groups = build_episode_groups(filtered_events)
    print(f"  Grouped into {len(episode_groups)} episode(s), {sum(len(v) for v in episode_groups.values())} event(s) total")

    # ── [2/6] Probe video metadata ──
    print("\n[2/6] Probing video metadata ...")
    created: List[str] = []
    failed = 0

    batch_tasks_cache: Dict[str, List[str]] = {}
    video_meta_cache: Dict[str, Optional[Tuple[int, int, int]]] = {}

    valid_tasks: List[Dict[str, Any]] = []
    clip_counter = 0

    # When skip_null_actions=True, we pre-filter each episode's actions
    # and video once, then cut sub-clips from the filtered data.
    _filtered_cache: Dict[str, Dict[str, Any]] = {}

    # resolve episode -> (mp4, jsonl, batch)
    ep_resolve: Dict[str, Tuple[str, str, str]] = {}  # ep_name -> (mp4, jsonl, batch)
    for ep_name in episode_groups:
        matched = vpt_index.get(ep_name)
        if matched is None:
            stem = Path(ep_name).stem
            matched = vpt_index.get(stem)
        if matched is not None:
            ep_resolve[ep_name] = matched

    unique_jsonls: Dict[str, str] = {}  # jsonl_path -> mp4_path
    for mp4_path, jsonl_path, _ in ep_resolve.values():
        unique_jsonls.setdefault(jsonl_path, mp4_path)

    # probe video metadata
    actual_workers = min(num_workers, max(len(unique_jsonls), 1))
    print(f"  Probing {len(unique_jsonls)} video(s) with {actual_workers} worker(s) ...")
    unique_mp4s = set(unique_jsonls.values())
    with ProcessPoolExecutor(max_workers=actual_workers, mp_context=_MP_CTX) as executor:
        futures = {executor.submit(_probe_one, mp4): mp4 for mp4 in unique_mp4s}
        pending = set(futures)
        with tqdm(total=len(futures), desc="  Video probe", unit="file") as pbar:
            while pending:
                try:
                    for future in as_completed(pending, timeout=_WATCHDOG_TIMEOUT):
                        pending.discard(future)
                        try:
                            mp4, meta = future.result()
                            video_meta_cache[mp4] = meta
                        except Exception as exc:
                            print(f"  [WARN] video probe failed for {futures[future]}: {exc}")
                        pbar.update(1)
                except FuturesTimeoutError:
                    # No future completed in _WATCHDOG_TIMEOUT — log and skip
                    stuck = [f for f in pending if not f.done()]
                    for f in stuck:
                        mp4 = futures[f]
                        print(f"  [WARN] video probe no progress in {_WATCHDOG_TIMEOUT}s, skipping {mp4}")
                        f.cancel()
                        pending.discard(f)
                        pbar.update(1)

    # compute null-action masks (only if skip_null)
    if skip_null_actions:
        event_times_by_jsonl: Dict[str, List[int]] = defaultdict(list)
        for ep_name, events in episode_groups.items():
            matched = ep_resolve.get(ep_name)
            if matched is None:
                continue
            _, jsonl_path, _ = matched
            event_times_by_jsonl[jsonl_path].extend(evt_time for evt_time, _ in events)

        print(
            f"  Computing null-action masks for {len(unique_jsonls)} episode(s) "
            f"with {actual_workers} worker(s) ..."
        )
        with ProcessPoolExecutor(max_workers=actual_workers, mp_context=_MP_CTX) as executor:
            futures = {
                executor.submit(
                    _compute_null_mask, jp, mp4,
                    video_meta_cache.get(mp4),
                    event_times_by_jsonl.get(jp),
                ): jp
                for jp, mp4 in unique_jsonls.items()
            }
            with tqdm(total=len(futures), desc="  Null-action mask", unit="file") as pbar:
                pending = set(futures)
                while pending:
                    try:
                        for future in as_completed(pending, timeout=_WATCHDOG_TIMEOUT):
                            pending.discard(future)
                            jp = futures[future]
                            try:
                                _, result = future.result()
                                if result is not None:
                                    _filtered_cache[jp] = result
                            except Exception as exc:
                                print(f"  [WARN] null-mask worker crashed for {jp}: {exc}")
                            pbar.update(1)
                    except FuturesTimeoutError:
                        # No future completed in _WATCHDOG_TIMEOUT — SIGALRM
                        # should have fired by now; if we're here, a worker is
                        # stuck in a way SIGALRM can't interrupt.  Log and skip.
                        stuck = [f for f in pending if not f.done()]
                        for f in stuck:
                            jp = futures[f]
                            print(f"  [WARN] null-mask no progress in {_WATCHDOG_TIMEOUT}s, skipping {jp}")
                            f.cancel()
                            pending.discard(f)
                            pbar.update(1)

    # ── [3/6] Build the valid task list ──
    print("\n[3/6] Building valid task list ...")
    total_subclips = sum(len(v) for v in episode_groups.values())
    display_total = max_episodes if (max_episodes is not None and max_episodes < total_subclips) else total_subclips
    if display_total != total_subclips:
        print(f"  (will stop after {display_total} sub-clips out of {total_subclips} total)")
    joint_clip_removed = 0       # A fail & B pass — cross-key density conflict
    joint_event_removed = 0      # A pass & B fail — same-key density conflict
    joint_both_removed = 0       # A fail & B fail — both (clustered events)

    for ep_name, events in sorted(episode_groups.items()):
        if ep_name not in ep_resolve:
            failed += len(events)
            continue
        mp4_path, jsonl_path, batch_name = ep_resolve[ep_name]

        coarse_tasks = batch_tasks_cache.get(batch_name)
        if coarse_tasks is None:
            coarse_tasks = VPT_BATCH_TASKS[batch_name]
            batch_tasks_cache[batch_name] = coarse_tasks

        meta = video_meta_cache.get(mp4_path)
        n_frames = jsonl_n_frames.get(jsonl_path)
        if meta is None or n_frames is None:
            failed += len(events)
            continue
        video_frame_count, vid_h, vid_w = meta

        # Prioritize rare events within each episode so common events cannot
        # consume their windows. Since this order is not chronological,
        # sorted start-frame lists track all conflicts with O(log n) lookup.
        events_ordered = sorted(
            events, key=lambda x: (global_key_count_raw.get(x[1], 0), x[0])
        )
        kept_starts_sorted: List[int] = []
        kept_by_key_sorted: Dict[str, List[int]] = defaultdict(list)

        for evt_time, evt_key in events_ordered:
            clip_counter += 1

            if skip_null_actions:
                filt = _filtered_cache.get(jsonl_path)
                if filt is None:
                    failed += 1
                    continue
                if filt["n_original"] <= evt_time:
                    print(
                        f"  [SKIP] filtered source too short ({filt['n_original']} <= {evt_time}) "
                        f"for event at t={evt_time}"
                    )
                    failed += 1
                    continue

                # Translate this single event's raw time to the
                # filtered frame rank. The cum map gives
                # ``cum[t]`` = number of kept frames in raw [0, t).
                # When the event's own frame is kept, the rank is
                # exactly cum[evt_time]. (For a kept frame,
                # cum[t+1] = cum[t] + 1, so cum[t+1] - 1 = cum[t].)
                # If the event's own frame was null-filtered, this
                # event has no valid filtered position — skip it.
                keep_mask = filt["keep_mask"]
                n_raw = filt["n_original"]
                cum = np.concatenate([[0], np.cumsum(keep_mask.astype(np.int64))])
                n_filtered = int(filt["n_filtered"])
                if not bool(keep_mask[int(evt_time)]):
                    failed += 1
                    continue
                filtered_evt = int(cum[int(evt_time)])

                start_frame, end_frame = _compute_window_bounds(filtered_evt, win_len, bias)
                if start_frame < 0:
                    failed += 1
                    continue
                # Clamp to available frames instead of skipping —
                # events near the video tail may only need a few
                # frames truncated.
                avail = n_filtered
                if end_frame > avail:
                    end_frame = avail
                n = end_frame - start_frame
            else:
                start_frame, end_frame = _compute_window_bounds(int(evt_time), win_len, bias)
                if start_frame < 0:
                    failed += 1
                    continue
                # Clamp to available frames (use the tighter of
                # jsonl vs video).
                avail = min(n_frames, video_frame_count)
                if end_frame > avail:
                    end_frame = avail
                n = end_frame - start_frame

            if n <= 0:
                failed += 1
                continue

            # Use filtered-space event rank for Gate B when
            # skip_null_actions is on; otherwise evt_time itself
            # is the right reference.
            effective_evt = filtered_evt if skip_null_actions else int(evt_time)

            # Anchor must lie inside the (possibly tail-clamped)
            # sub-clip. If end_frame was clamped to avail because
            # the event was near the video tail, the anchor's
            # in-clip position can be >= n; skip such cases.
            anchor_rel = int(effective_evt) - start_frame
            if not (0 <= anchor_rel < n):
                failed += 1
                continue

            # Gate A (clip_stride): bidirectional nearest-neighbor check
            # With rarity-first iteration, we must check BOTH neighbors of
            # ``start_frame`` in the sorted list of already-kept starts.
            a_pass = True
            if kept_starts_sorted:
                pos = bisect.bisect_left(kept_starts_sorted, start_frame)
                if pos > 0 and start_frame - kept_starts_sorted[pos - 1] < min_clip_stride:
                    a_pass = False
                if a_pass and pos < len(kept_starts_sorted) and \
                   kept_starts_sorted[pos] - start_frame < min_clip_stride:
                    a_pass = False

            # Gate B (event_stride): bidirectional nearest-neighbor per evt_key
            b_pass = True
            key_times = kept_by_key_sorted[evt_key]
            if key_times:
                pos2 = bisect.bisect_left(key_times, effective_evt)
                if pos2 > 0 and effective_evt - key_times[pos2 - 1] <= min_event_stride:
                    b_pass = False
                if b_pass and pos2 < len(key_times) and \
                   key_times[pos2] - effective_evt <= min_event_stride:
                    b_pass = False

            if not (a_pass and b_pass):
                if not a_pass and not b_pass:
                    joint_both_removed += 1
                elif not a_pass:
                    joint_clip_removed += 1
                else:
                    joint_event_removed += 1
                continue

            # Same-key events in the same window that fall BEFORE
            # the anchor — collected into the sub-clip's
            # pre-anchor list. (See _scan_pre_anchor_events for the
            # coordinate-system and exclusion rules.) The list may
            # be empty; the anchor alone is a valid sub-clip.
            pre_anchor_event_times, pre_anchor_event_frame_idxs_in_clip = _scan_pre_anchor_events(
                events=events,
                evt_key=evt_key,
                anchor_rel=anchor_rel,
                start_frame=start_frame,
                skip_null_actions=bool(skip_null_actions),
                keep_mask=keep_mask if skip_null_actions else None,
                cum=cum if skip_null_actions else None,
                n_raw=n_raw if skip_null_actions else 0,
            )

            valid_tasks.append({
                "mp4_path": mp4_path,
                "jsonl_path": jsonl_path,
                "event_key": evt_key,
                # Anchor: the iterated event that passed Gate A/B.
                "event_anchor_time": int(evt_time),
                "event_anchor_frame_idx_in_clip": anchor_rel,
                # Earlier same-key events in the same window. Both
                # lists are parallel and sorted by in-clip position.
                "pre_anchor_event_times": pre_anchor_event_times,
                "pre_anchor_event_frame_idxs_in_clip": pre_anchor_event_frame_idxs_in_clip,
                "start_frame": start_frame,
                "num_frames": n,
                "batch_name": batch_name,
                "coarse_tasks": coarse_tasks,
                "clip_counter": clip_counter,
                "ep_name": ep_name,
                "vid_h": vid_h,
                "vid_w": vid_w,
            })
            bisect.insort(kept_starts_sorted, start_frame)
            bisect.insort(kept_by_key_sorted[evt_key], effective_evt)

    print(f"  Pre-validated: {len(valid_tasks)} valid, {failed} skipped")
    # These mutually exclusive counters identify whether clip spacing,
    # same-event spacing, or both constraints rejected each candidate.
    if joint_clip_removed or joint_event_removed or joint_both_removed:
        print(
            f"  Joint filter (min_clip_stride={min_clip_stride}, "
            f"min_event_stride={min_event_stride}): "
            f"only_clip={joint_clip_removed}, "
            f"only_event={joint_event_removed}, "
            f"both={joint_both_removed}"
        )

    # Report how many unique event keys survive both spacing constraints.
    keys_in_valid = {t["event_key"] for t in valid_tasks}
    n_lost = len(global_key_count_raw) - len(keys_in_valid)
    print(
        f"  Gate key-survival: {len(keys_in_valid)}/{len(global_key_count_raw)} "
        f"unique evt_keys survived (lost {n_lost} entirely)"
    )

    return valid_tasks, video_meta_cache, _filtered_cache, jsonl_n_frames


def _write_lerobot_dataset(
    valid_tasks: List[Dict[str, Any]],
    video_meta_cache: Dict[str, Optional[Tuple[int, int, int]]],
    filtered_cache: Dict[str, Dict[str, Any]],
    jsonl_n_frames: Dict[str, int],
    output_dir: str,
    fps: int,
    win_len: int,
    bias: int,
    action_dim: int,
    state_dim: int,
    cursor_reference_size: Optional[Tuple[float, float]],
    vpt_mouse_reference_size: Tuple[float, float],
    skip_null_actions: bool,
    num_workers: int,
    max_episodes: Optional[int] = None,
) -> List[str]:
    """Steps [4-6/6]: create the LeRobot dataset, extract clips, write.

    Resolves per-task cursor reference size and validates uniformity
    across the input, then creates a fresh LeRobot dataset (or appends
    to one if ``output_dir/meta/info.json`` already exists). All
    per-batch statistics (video resolution, episode metadata,
    ``info.json`` schema) are computed independently here — the
    pooling only governs *which* tasks survive, not how this batch's
    LeRobot dataset is laid out. Returns the list of created episode
    paths.

    The caller is responsible for applying the max-min cap BEFORE
    invoking this helper. ``max_episodes`` here is only used for the
    per-episode early-exit print; when ``max_episodes=None`` (pooled
    path), all post-cap tasks are written.
    """
    for task in valid_tasks:
        task["cursor_reference_size"] = resolve_cursor_reference_size(
            cursor_reference_size,
            fallback_size=(task["vid_w"], task["vid_h"]),
        )
    reference_sizes = {tuple(task["cursor_reference_size"]) for task in valid_tasks}
    if len(reference_sizes) != 1:
        raise ValueError(
            "Selected VPT event clips resolve to different cursor reference sizes: "
            f"{sorted(reference_sizes)}. Pass --cursor_reference_size explicitly "
            "or convert resolutions into separate datasets."
        )
    resolved_cursor_reference_size = tuple(valid_tasks[0]["cursor_reference_size"])

    # NOTE: the cap (``stratified_cap_to_max``) is applied by the
    # caller before the writer is invoked — once for the single-batch
    # path (``convert_vpt_event_dataset``) and once for the pooled path
    # (``convert_vpt_event_pooled``). The writer trusts its input and
    # never re-caps. Keeping the per-batch LeRobot dataset independent
    # is what allows ``output_dir/<batch>/`` to be loaded standalone
    # by the GameWAM Mixed config; the pooling only governs which
    # event-level tasks survive, not how each batch's dataset is
    # written.

    # ── [4/6] Create the LeRobot dataset ──
    print("\n[4/6] Creating LeRobot dataset ...")

    output_path = Path(output_dir)
    if (output_path / "meta" / "info.json").exists():
        dataset = LeRobotDataset(
            repo_id=output_dir,
            root=output_path,
            is_compute_episode_stats_image=False,
        )
    else:
        first_task = valid_tasks[0]
        video_info = build_video_info(Path(first_task["mp4_path"]), fps=fps)
        features = {
            "observation.images": {
                "dtype": "video",
                "shape": (3, first_task["vid_h"], first_task["vid_w"]),
                "names": ["channels", "height", "width"],
                "info": video_info,
            },
            **minecraft_state_features(),
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"action_{i}" for i in range(action_dim)],
            },
        }
        dataset = LeRobotDataset.create(
            repo_id=output_dir,
            fps=fps,
            features=features,
            root=output_path,
            robot_type="minecraft",
            use_videos=True,
            is_compute_episode_stats_image=False,
        )
    attach_or_validate_minecraft_state_schema(
        dataset,
        cursor_reference_size=resolved_cursor_reference_size,
        cursor_source_metadata={
            "kind": "vpt_jsonl_mouse",
            "coordinate_size": list(vpt_mouse_reference_size),
            "out_of_bounds_policy": "invalid",
        },
    )

    # ── [5/6] Extract video clips in parallel ──
    print("\n[5/6] Extracting video clips in parallel ...")
    # (outside dataset tree to avoid save_episode's rglob("*.mp4") picking them up)
    temp_dir = output_path.parent / ("_temp_videos_" + output_path.name)
    temp_dir.mkdir(parents=True, exist_ok=True)

    num_vkeys = len(dataset.meta.video_keys)
    extraction_ok: List[bool] = [False] * (len(valid_tasks) * num_vkeys)

    tasks_by_jsonl: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for i, task in enumerate(valid_tasks):
        tasks_by_jsonl[task["jsonl_path"]].append((i, task))

    def _process_jsonl_group(jsonl_path: str, group: List[Tuple[int, Dict[str, Any]]]) -> Tuple[str, bool]:
        try:
            raw_actions = _load_vpt_jsonl(jsonl_path)
        except Exception as exc:
            print(f"  [WARN] Failed to load {jsonl_path}: {exc}")
            return jsonl_path, False

        src_mp4 = group[0][1]["mp4_path"]
        n_total = len(raw_actions)

        keep_mask: Optional[np.ndarray] = None
        if skip_null_actions:
            filt = filtered_cache.get(jsonl_path)
            if filt is None:
                return jsonl_path, False
            keep_mask = filt["keep_mask"]

        # Compute action/state vectors in one pass
        n_compute = min(n_total, len(keep_mask)) if keep_mask is not None else n_total
        ep_actions_all = np.empty((n_compute, action_dim), dtype=np.float32)
        ep_states_all = np.empty((n_compute, state_dim), dtype=np.float32)
        ep_state_valid_all = np.empty((n_compute, RAW_STATE_DIM), dtype=np.bool_)
        attack_is_stuck = False
        tracker = MinecraftStateTracker()
        group_cursor_reference_size = tuple(group[0][1]["cursor_reference_size"])
        gui_open_frames = 0
        gui_cursor_frames = 0
        cursor_valid_frames = 0
        cursor_missing_frames = 0
        cursor_out_of_bounds_frames = 0
        cursor_source_x: List[float] = []
        cursor_source_y: List[float] = []
        for j in range(n_compute):
            raw_act = raw_actions[j]
            env_action, _, attack_is_stuck = _json_action_to_env_action(
                raw_act, attack_is_stuck=attack_is_stuck, frame_idx=j,
            )
            info = _extract_info_from_raw_action(raw_act)
            gui_open = bool(info.get("is_gui_open", False))
            source_cursor = (
                _extract_vpt_mouse_pair(raw_act) if gui_open else None
            )
            if gui_open:
                gui_open_frames += 1
                if source_cursor is None:
                    cursor_missing_frames += 1
                else:
                    gui_cursor_frames += 1
                    cursor_source_x.append(source_cursor[0])
                    cursor_source_y.append(source_cursor[1])
            cursor, cursor_out_of_bounds = _map_vpt_cursor_to_reference(
                raw_act,
                cursor_reference_size=group_cursor_reference_size,
                mouse_reference_size=vpt_mouse_reference_size,
            )
            cursor_out_of_bounds_frames += int(cursor_out_of_bounds)
            if cursor is not None:
                info["cursor_x"], info["cursor_y"] = cursor
                cursor_valid_frames += 1
            observed_values, observed_valid = extract_state_with_validity(
                info,
                cursor_reference_size=group_cursor_reference_size,
            )
            state_values, state_valid = tracker.state_before_action(
                observed_values,
                observed_valid,
                env_action,
            )
            # Semantic validity override: when GUI is open, pitch/yaw/hotbar
            # are frozen (player can't change them in a menu); when GUI is
            # closed, cursor is not in use. Apply the GUI-state rule on top
            # of the per-source flags so the Minecraft cursor encoder sees a
            # consistent validity distribution across data sources.
            from gamewam.envs.minecraft.observations.state import apply_gui_semantic_validity_numpy
            state_valid = apply_gui_semantic_validity_numpy(
                state_values, state_valid
            )
            tracker.commit_after_action(observed_values, observed_valid, env_action)
            ep_actions_all[j] = action_to_vector(env_action)[:action_dim]
            ep_states_all[j] = state_values[:state_dim]
            ep_state_valid_all[j] = state_valid

        # Filter by keep_mask
        if keep_mask is not None:
            ep_actions = ep_actions_all[keep_mask[:n_compute]]
            ep_states = ep_states_all[keep_mask[:n_compute]]
            ep_state_valid = ep_state_valid_all[keep_mask[:n_compute]]
        else:
            ep_actions = ep_actions_all
            ep_states = ep_states_all
            ep_state_valid = ep_state_valid_all

        # Attach vectors to tasks
        for i, task in group:
            start_frame = task["start_frame"]
            n = task["num_frames"]
            task["action_vectors"] = ep_actions[start_frame:start_frame+n]
            task["state_vectors"] = ep_states[start_frame:start_frame+n]
            task["state_field_valid_vectors"] = ep_state_valid[start_frame:start_frame+n]
            task["cursor_stats"] = {
                "gui_open_frames": int(gui_open_frames),
                "gui_cursor_frames": int(gui_cursor_frames),
                "cursor_valid_frames": int(cursor_valid_frames),
                "cursor_missing_frames": int(cursor_missing_frames),
                "cursor_out_of_bounds_frames": int(
                    cursor_out_of_bounds_frames
                ),
                "cursor_out_of_bounds_ratio": (
                    float(cursor_out_of_bounds_frames)
                    / max(gui_cursor_frames, 1)
                ),
                # Metadata compatibility fields; invalid coordinates are never
                # clamped into valid edge cursor positions.
                "cursor_clipped_frames": 0,
                "cursor_clipped_ratio": 0.0,
                "source_x_min": min(cursor_source_x, default=None),
                "source_x_max": max(cursor_source_x, default=None),
                "source_y_min": min(cursor_source_y, default=None),
                "source_y_max": max(cursor_source_y, default=None),
            }
            task["minecraft_state_stats"] = tracker.stats.as_dict()

        # Single-pass video decode + clip extraction
        clip_specs: List[Tuple[int, int, int, Path]] = []
        for i, task in group:
            start_frame = task["start_frame"]
            n = task["num_frames"]
            for ki, vkey in enumerate(dataset.meta.video_keys):
                ok_idx = i * num_vkeys + ki
                temp_mp4 = temp_dir / f"clip_{i:06d}_vk{ki}.mp4"
                clip_specs.append((ok_idx, start_frame, n, temp_mp4))

        succeeded = _extract_clips_single_pass(
            src_mp4,
            clip_specs,
            raw_actions,
            fps=fps,
            keep_mask=keep_mask,
            mouse_reference_size=vpt_mouse_reference_size,
        )
        for ok_idx in succeeded:
            extraction_ok[ok_idx] = True

        return jsonl_path, True

    actual_workers = min(num_workers, max(len(tasks_by_jsonl), 1))
    print(
        f"  Processing {len(tasks_by_jsonl)} JSONL file(s) for extraction "
        f"with {actual_workers} worker(s) ..."
    )
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {
            executor.submit(_process_jsonl_group, jp, group): jp
            for jp, group in tasks_by_jsonl.items()
        }
        with tqdm(total=len(futures), desc="  Extraction by video", unit="file") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    ok_count = sum(extraction_ok)
    print(f"  Extraction complete: {ok_count}/{len(valid_tasks) * num_vkeys} sub-clips succeeded")

    # ── [6/6] Write the dataset ──
    print("\n[6/6] Writing dataset ...")

    dummy_img = PILImage.new("RGB", (1, 1))
    print(f"  Writing {len(valid_tasks)} sub-clips to dataset ...")

    write_failed = 0
    created: List[str] = []
    # display_total is purely cosmetic for the per-episode progress print.
    display_total = (
        max_episodes
        if (max_episodes is not None and max_episodes < len(valid_tasks))
        else len(valid_tasks)
    )

    for i, task in enumerate(valid_tasks):
        src_mp4 = task["mp4_path"]
        event_time = task["event_anchor_time"]
        event_key = task["event_key"]
        start_frame = task["start_frame"]  # in filtered space if skip_null
        n = task["num_frames"]
        cc = task["clip_counter"]
        ep_name = task["ep_name"]
        coarse_tasks = task["coarse_tasks"]

        current_ep_idx = dataset.meta.total_episodes

        # Move pre-extracted video(s) from temp dir to the dataset path.
        video_ok = True
        for ki, vkey in enumerate(dataset.meta.video_keys):
            temp_mp4 = temp_dir / f"clip_{i:06d}_vk{ki}.mp4"
            arg_idx = i * num_vkeys + ki
            dst_mp4 = output_path / dataset.meta.get_video_file_path(current_ep_idx, vkey)
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            if dst_mp4.exists() or dst_mp4.is_symlink():
                dst_mp4.unlink()
            if extraction_ok[arg_idx] and temp_mp4.exists():
                shutil.move(str(temp_mp4), str(dst_mp4))
            else:
                video_ok = False
                break

        if not video_ok or "action_vectors" not in task or "state_vectors" not in task:
            reason = "missing video" if not video_ok else "missing pre-computed vectors"
            print(f"  [SKIP] {ep_name} {reason}")
            for vkey in dataset.meta.video_keys:
                dst_mp4 = output_path / dataset.meta.get_video_file_path(current_ep_idx, vkey)
                if dst_mp4.exists():
                    dst_mp4.unlink()
            write_failed += 1
            continue

        action_vectors = task["action_vectors"]
        state_vectors = task["state_vectors"]
        state_field_valid_vectors = task["state_field_valid_vectors"]
        n_kept = action_vectors.shape[0]

        dataset.episode_buffer = dataset.create_episode_buffer()

        fine_paraphrases = EVENT_DESCRIPTION.get(event_key, [event_key])
        coarse_task = coarse_tasks[cc % len(coarse_tasks)]
        fine_task = fine_paraphrases[cc % len(fine_paraphrases)]

        null_info = ""
        if skip_null_actions:
            null_info = f" filtered_range=[{start_frame}:{start_frame+n}]"

        display_fine = fine_task[:50]
        progress_num = len(created) + 1
        print(
            f"  [{progress_num}/{display_total}] {ep_name} "
            f"t={event_time} coarse=\"{coarse_task}\" fine=\"{display_fine}\"{null_info}"
        )

        for idx in range(n_kept):
            frame_data = {
                "observation.images": dummy_img,
                "observation.state": state_vectors[idx].astype(np.float32),
                "observation.state_field_valid": state_field_valid_vectors[idx].astype(bool),
                "observation.state_dim_is_pad": np.zeros(state_dim, dtype=bool),
                "action": action_vectors[idx].astype(np.float32),
            }
            dataset.add_frame(
                frame_data,
                task=[coarse_task, fine_task],
            )

        if dataset.episode_buffer["size"] < 2:
            print(f"  [SKIP] buffer too small for sub-clip at t={event_time}")
            for vkey in dataset.meta.video_keys:
                dst_mp4 = output_path / dataset.meta.get_video_file_path(current_ep_idx, vkey)
                if dst_mp4.exists():
                    dst_mp4.unlink()
            write_failed += 1
            continue

        raw_name = Path(src_mp4).stem
        # Per-sub-clip event metadata. The anchor is the
        # rightmost same-key event in the window; the list fields
        # cover earlier same-key events in the same window.
        evt_idxs_list = [int(x) for x in task.get("pre_anchor_event_frame_idxs_in_clip", [])]
        evt_times_list = [int(t) for t in task.get("pre_anchor_event_times", [])]
        anchor_idx = int(task.get("event_anchor_frame_idx_in_clip", 0))
        anchor_time = int(task.get("event_anchor_time", 0))
        dataset.save_episode(
            raw_file_name=raw_name,
            extra_fields={
                "event_key": event_key,
                "event_anchor_time": anchor_time,
                "event_anchor_frame_idx_in_clip": anchor_idx,
                "pre_anchor_event_times": evt_times_list,
                "pre_anchor_event_frame_idxs_in_clip": evt_idxs_list,
                "win_len": int(win_len),
                "minecraft_state_stats": task["minecraft_state_stats"],
                "cursor_mapping_stats": task["cursor_stats"],
            },
        )
        created.append(output_dir)

        if max_episodes is not None and len(created) >= max_episodes:
            print(f"\nReached max_episodes={max_episodes}, stopping.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(
                f"\nDone: {len(created)} converted, "
                f"{write_failed} write failures -> {output_dir}"
            )
            return created

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(
        f"\nDone: {len(created)} converted, {write_failed} write failures "
        f"-> {output_dir}"
    )
    return created


# Pooled multi-batch driver

def convert_vpt_event_pooled(
    vpt_root: str,
    output_dir: str,
    max_episodes: int,
    batch_filter: Optional[List[str]] = None,
    fps: int = 20,
    win_len: int = 256,
    bias: int = 0,
    min_event_stride: int = 64,
    min_clip_stride: int = 1,
    action_dim: int = 22,
    state_dim: int = RAW_STATE_DIM,
    cursor_reference_size: Optional[Tuple[float, float]] = None,
    vpt_mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
    skip_null_actions: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 32,
) -> Dict[str, List[str]]:
    """Convert all VPT batches into per-batch LeRobot datasets under one budget."""
    if state_dim != RAW_STATE_DIM:
        raise ValueError(
            f"Minecraft VPT conversion requires state_dim={RAW_STATE_DIM}, got {state_dim}"
        )
    if cursor_reference_size is not None:
        cursor_reference_size = normalize_screen_size(
            cursor_reference_size, name="cursor_reference_size"
        )
    vpt_mouse_reference_size = normalize_screen_size(
        vpt_mouse_reference_size, name="vpt_mouse_reference_size"
    )
    if min_clip_stride < 1:
        raise ValueError(f"min_clip_stride must be >= 1, got {min_clip_stride}")
    if min_clip_stride > min_event_stride:
        raise ValueError(
            f"min_clip_stride ({min_clip_stride}) > min_event_stride "
            f"({min_event_stride}). In this regime the event-stride gate has "
            f"no effect on the kept set (clip gate strictly dominates). "
            f"Either raise min_event_stride >= min_clip_stride, or set "
            f"min_event_stride=0 if you only want the clip gate."
        )
    if max_episodes is None:
        raise ValueError(
            "convert_vpt_event_pooled requires max_episodes to be set; "
            "the whole point of the pooled driver is to allocate one budget "
            "across all batches."
        )

    available = discover_vpt_batches(vpt_root)
    if not available:
        raise FileNotFoundError(
            f"No VPT batch subdirectories (6xx..10xx) found under {vpt_root}"
        )
    target_batches = (
        [b for b in batch_filter if b in available]
        if batch_filter
        else list(available.keys())
    )
    if not target_batches:
        raise ValueError(
            f"batch_filter={batch_filter} matches none of the discovered "
            f"batches: {sorted(available.keys())}"
        )

    print(
        f"\nVPT Event Pooled: vpt_root={vpt_root} "
        f"available={sorted(available.keys())} target={target_batches} "
        f"max_episodes={max_episodes}"
    )

    # ── Per-batch: event detection + joint filtering ─────────────────────
    all_valid_tasks: List[Dict[str, Any]] = []
    # batch_name -> (video_meta_cache, filtered_cache, jsonl_n_frames)
    batch_caches: Dict[str, Tuple[
        Dict[str, Optional[Tuple[int, int, int]]],
        Dict[str, Dict[str, Any]],
        Dict[str, int],
    ]] = {}
    per_batch_candidate_counts: Dict[str, int] = {}

    for batch_name in target_batches:
        batch_vpt_root = available[batch_name]
        print(f"\n[Batch {batch_name}] detection + filtering ...")
        valid_tasks, vmc, fc, jnf = _run_event_detection_and_filtering(
            vpt_root=batch_vpt_root,
            win_len=win_len,
            bias=bias,
            min_event_stride=min_event_stride,
            min_clip_stride=min_clip_stride,
            skip_null_actions=skip_null_actions,
            cursor_reference_size=cursor_reference_size,
            vpt_mouse_reference_size=vpt_mouse_reference_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
            max_episodes=None,
            ratio=None,
        )
        per_batch_candidate_counts[batch_name] = len(valid_tasks)
        all_valid_tasks.extend(valid_tasks)
        batch_caches[batch_name] = (vmc, fc, jnf)

    print(
        f"\nPooled candidate counts: "
        + ", ".join(f"{b}={per_batch_candidate_counts[b]}" for b in target_batches)
        + f" | total={len(all_valid_tasks)}"
    )

    if not all_valid_tasks:
        print(f"\nDone: 0 converted across {len(target_batches)} batch(es).")
        return {b: [] for b in target_batches}

    # ── Global cap: ONE max-min fair allocation across the pool ──────────
    before = len(all_valid_tasks)
    target_with_buffer = min(
        len(all_valid_tasks),
        max_episodes + max(100, max_episodes // 1000),
    )
    all_valid_tasks, quotas, key_counts_before = stratified_cap_to_max(
        all_valid_tasks, target_with_buffer, shuffle=shuffle, seed=seed,
    )
    key_counts_after: Dict[str, int] = defaultdict(int)
    for t in all_valid_tasks:
        key_counts_after[t["event_key"]] += 1
    num_keys = len(quotas)
    num_saturated = sum(
        1 for k, q in quotas.items()
        if q > 0 and key_counts_before.get(k, 0) == q
    )
    top3 = sorted(key_counts_after.items(), key=lambda x: -x[1])[:3]
    top3_str = ", ".join(f"{k}={v}" for k, v in top3)
    print(
        f"  Pooled stratified cap (target={max_episodes}, with_buffer={target_with_buffer}): "
        f"{before} -> {len(all_valid_tasks)} sub-clips | "
        f"{num_keys} unique keys, {num_saturated} saturated | "
        f"top-3: {top3_str}"
    )

    # ── Partition by batch ───────────────────────────────────────────────
    by_batch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in all_valid_tasks:
        by_batch[t["batch_name"]].append(t)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Per-batch target allocation ──────────────────────────────────────
    # Distribute the buffer (cap_with_buffer - max_episodes) across batches
    # in proportion to each batch's head-key count: batches with more head
    # keys absorb more truncation, and each batch early-stops at its own
    # target so truncation doesn't pile onto the last batch. Within a
    # batch, tasks are still ordered by key tier (rare keys written first)
    # to protect rare keys.
    buffer_total = len(all_valid_tasks) - max_episodes

    def _is_head(event_key: str) -> bool:
        return quotas.get(event_key, 0) >= 200

    batch_head_count: Dict[str, int] = {
        b: sum(1 for t in by_batch[b] if _is_head(t["event_key"]))
        for b in target_batches
    }
    total_head = sum(batch_head_count.values())

    write_rng = np.random.default_rng(seed)

    if buffer_total > 0:
        # Prefer allocating the buffer by head-key count (batches with
        # more head keys are cut more); fall back to cap proportion when
        # no batch has head keys.
        if total_head > 0:
            weight = batch_head_count
        else:
            weight = {b: len(by_batch[b]) for b in target_batches}
        total_weight = sum(weight.values())

        batch_buffer: Dict[str, int] = {
            b: buffer_total * weight[b] // total_weight
            for b in target_batches
        }
        # Fix the rounding remainder: assign to batches with the
        # largest weight, in order.
        allocated = sum(batch_buffer.values())
        remaining_buffer = buffer_total - allocated
        if remaining_buffer > 0:
            sorted_by_weight = sorted(
                target_batches, key=lambda b: -weight[b]
            )
            for b in sorted_by_weight[:remaining_buffer]:
                batch_buffer[b] += 1
    else:
        batch_buffer = {b: 0 for b in target_batches}

    batch_target: Dict[str, int] = {
        b: len(by_batch[b]) - batch_buffer[b] for b in target_batches
    }

    # When buffer_total > 0, sum(target) == max_episodes (exact quota);
    # when buffer_total <= 0, everything is kept and sum(target) == total_cap.
    assert sum(batch_target.values()) == min(max_episodes, len(all_valid_tasks)), (
        f"sum(batch_target)={sum(batch_target.values())} != "
        f"min(max_episodes, total_cap)={min(max_episodes, len(all_valid_tasks))}"
    )

    print(
        f"  Per-batch target (buffer={buffer_total} by head-key proportion):"
    )
    for b in target_batches:
        print(
            f"    {b}: cap={len(by_batch[b])}, head={batch_head_count[b]}, "
            f"buffer={batch_buffer[b]}, target={batch_target[b]}"
        )

    # ── Within-batch key-priority ordering ───────────────────────────────
    # Write rare/mid keys first and head keys last (shuffled within each
    # tier) so early-stop preferentially drops head-key episodes and rare
    # keys survive truncation.
    write_order = list(target_batches)
    write_rng.shuffle(write_order)

    def _key_priority(event_key: str) -> int:
        q = quotas.get(event_key, 0)
        if q >= 200:
            return 2  # head: written last, dropped first by early-stop
        elif q >= 50:
            return 1  # mid: written in between
        else:
            return 0  # rare: written first, protected

    for b in write_order:
        by_batch[b].sort(key=lambda t: _key_priority(t["event_key"]))
        shuffled = []
        for prio, group in groupby(by_batch[b], key=lambda t: _key_priority(t["event_key"])):
            g = list(group)
            write_rng.shuffle(g)
            shuffled.extend(g)
        by_batch[b] = shuffled

    # ── Write per-batch datasets ─────────────────────────────────────────
    result: Dict[str, List[str]] = {}
    per_batch_kept: Dict[str, int] = {}
    per_batch_written: Dict[str, int] = {}
    for batch_name in write_order:
        batch_tasks = by_batch.get(batch_name, [])
        vmc, fc, jnf = batch_caches[batch_name]
        per_batch_kept[batch_name] = len(batch_tasks)
        target = batch_target[batch_name]
        if not batch_tasks or target <= 0:
            per_batch_written[batch_name] = 0
            if not batch_tasks:
                print(f"\n[Batch {batch_name}] 0 sub-clips after pooled cap; "
                      f"skipping write.")
            else:
                print(f"\n[Batch {batch_name}] target=0; skipping write.")
            result[batch_name] = []
            continue
        batch_output_dir = output_path / batch_name
        print(
            f"\n[Batch {batch_name}] writing up to {min(len(batch_tasks), target)} "
            f"sub-clips (cap kept {len(batch_tasks)}, target={target}) "
            f"to {batch_output_dir}"
        )
        created = _write_lerobot_dataset(
            valid_tasks=batch_tasks,
            video_meta_cache=vmc,
            filtered_cache=fc,
            jsonl_n_frames=jnf,
            output_dir=str(batch_output_dir),
            fps=fps,
            win_len=win_len,
            bias=bias,
            action_dim=action_dim,
            state_dim=state_dim,
            cursor_reference_size=cursor_reference_size,
            vpt_mouse_reference_size=vpt_mouse_reference_size,
            skip_null_actions=skip_null_actions,
            num_workers=num_workers,
            max_episodes=target,
        )
        result[batch_name] = created
        per_batch_written[batch_name] = len(created)

    # ── Top-level metadata for inspection ─────────────────────────────────
    pooled_meta = {
        "vpt_root": vpt_root,
        "output_dir": str(output_path),
        "max_episodes": max_episodes,
        "discovered_batches": sorted(available.keys()),
        "target_batches": target_batches,
        "per_batch": {
            b: {
                "candidates": per_batch_candidate_counts.get(b, 0),
                "kept": per_batch_kept.get(b, 0),
                "target": batch_target.get(b, 0),
                "buffer": batch_buffer.get(b, 0),
                "head_count": batch_head_count.get(b, 0),
                "written": per_batch_written.get(b, 0),
            }
            for b in target_batches
        },
        "batch_write_order": write_order,
        "actual_total": sum(per_batch_written.values()),
        "quotas": dict(quotas),
        "key_counts_before": dict(key_counts_before),
        "key_counts_after": dict(key_counts_after),
        "win_len": win_len,
        "bias": bias,
        "min_event_stride": min_event_stride,
        "min_clip_stride": min_clip_stride,
        "skip_null_actions": skip_null_actions,
        "shuffle": shuffle,
        "seed": seed,
    }
    meta_path = output_path / "_pooled_meta.json"
    with meta_path.open("w") as f:
        json.dump(pooled_meta, f, indent=2, sort_keys=True)
    print(f"\nWrote pooled metadata: {meta_path}")

    total = sum(len(v) for v in result.values())
    print(
        f"\nAll batches done: {total} sub-clips across {len(result)} batch(es) "
        f"-> {output_path}"
    )
    return result
