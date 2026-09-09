"""Convert VPT Contractor Data (mp4+jsonl) to LeRobot format.

Output is always copy-based (never symlinks) so the dataset directory is
self-contained and distributable.
"""

import copy
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import subprocess

from gamewam.datasets.lerobot.lerobot_dataset import LeRobotDataset
from gamewam.envs.minecraft.action_space import ENV_NULL_ACTION, action_to_vector
from gamewam.envs.minecraft.tasks.catalog import VPT_BATCH_TASKS
from gamewam.envs.minecraft.assets import GUI_CURSOR_FILE
from gamewam.envs.minecraft.observations.state import (
    RAW_STATE_DIM,
    apply_gui_semantic_validity_numpy,
    extract_state_with_validity,
    normalize_minecraft_info,
    normalize_screen_size,
)
from .state_tracking import MinecraftStateTracker
from PIL import Image as PILImage

from .task_utils import assert_output_dir_empty
from ...common.video import _extract_video_segment, build_video_info
from gamewam.datasets.schemas.minecraft import (
    attach_or_validate_minecraft_state_schema,
    minecraft_state_features,
    resolve_cursor_reference_size,
)


# VPT keyboard key → MC env_action key mapping
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

CAMERA_SCALER = 360.0 / 2400.0
DEFAULT_VPT_MOUSE_REFERENCE_SIZE = (1280.0, 720.0)
_CURSOR_OVERLAY: Optional[Tuple[np.ndarray, np.ndarray]] = None


class InvalidVPTJsonlError(ValueError):
    """Raised when a VPT jsonl file cannot be decoded or parsed."""


def _get_cursor_overlay() -> Tuple[np.ndarray, np.ndarray]:
    global _CURSOR_OVERLAY
    if _CURSOR_OVERLAY is None:
        if not GUI_CURSOR_FILE.exists():
            raise FileNotFoundError(f"Cursor asset not found: {GUI_CURSOR_FILE}")
        cursor = cv2.imread(str(GUI_CURSOR_FILE), cv2.IMREAD_UNCHANGED)
        if cursor is None:
            raise RuntimeError(f"Failed to load GUI cursor image: {GUI_CURSOR_FILE}")
        cursor = cursor[:16, :16, :]
        cursor_alpha = cursor[:, :, 3:] / 255.0
        cursor_image = cursor[:, :, :3]
        _CURSOR_OVERLAY = (cursor_image, cursor_alpha)
    return _CURSOR_OVERLAY


def _composite_images_with_alpha(
    image1: np.ndarray, image2: np.ndarray, alpha: np.ndarray, x: int, y: int,
) -> None:
    """OpenAI VPT-style alpha composite of cursor over a frame."""
    # For negative coordinates (cursor past the left/top edge), skip the
    # corresponding prefix of the source image.
    src_y = max(0, -y)
    src_x = max(0, -x)

    # Coordinates where the destination starts rendering.
    dst_y = max(0, y)
    dst_x = max(0, x)

    # Drawable height/width, clamped to prevent overflow.
    ch = max(0, min(image1.shape[0] - dst_y, image2.shape[0] - src_y))
    cw = max(0, min(image1.shape[1] - dst_x, image2.shape[1] - src_x))

    if ch <= 0 or cw <= 0:
        return

    alpha_view = alpha[src_y:src_y + ch, src_x:src_x + cw]
    if alpha_view.ndim == 2:
        alpha_view = alpha_view[..., np.newaxis]

    if alpha_view.shape[:2] != (ch, cw):
        return

    image1[dst_y:dst_y + ch, dst_x:dst_x + cw, :] = (
        image1[dst_y:dst_y + ch, dst_x:dst_x + cw, :] * (1 - alpha_view)
        + image2[src_y:src_y + ch, src_x:src_x + cw, :] * alpha_view
    ).astype(np.uint8)


def _selected_raw_actions_have_gui(
    raw_actions: List[Dict[str, Any]],
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    keep_mask: Optional[np.ndarray] = None,
) -> bool:
    if keep_mask is not None:
        limit = min(len(raw_actions), len(keep_mask))
        for idx in range(limit):
            if keep_mask[idx] and _extract_info_from_raw_action(raw_actions[idx]).get("is_gui_open", False):
                return True
        return False

    end_frame = len(raw_actions) if num_frames is None else min(len(raw_actions), start_frame + num_frames)
    for idx in range(start_frame, end_frame):
        if _extract_info_from_raw_action(raw_actions[idx]).get("is_gui_open", False):
            return True
    return False


def _map_vpt_cursor_to_reference(
    raw_action: Dict[str, Any],
    *,
    cursor_reference_size: Tuple[float, float],
    mouse_reference_size: Tuple[float, float],
) -> tuple[Optional[tuple[float, float]], bool]:
    """Map a valid VPT GUI cursor into dataset reference coordinates.

    The public OpenAI VPT loader treats mouse.x/y as coordinates in the
    1280x720 recorder content area and scales them by output-frame height. It
    does not clamp out-of-window coordinates into the image. Preserve that
    contract here: report an out-of-bounds pair as invalid instead of turning
    it into a valid edge cursor.
    """
    info = normalize_minecraft_info(raw_action)
    if not bool(info.get("is_gui_open", False)):
        return None, False
    source_cursor = _extract_vpt_mouse_pair(raw_action)
    if source_cursor is None:
        return None, False
    reference_width, reference_height = normalize_screen_size(
        cursor_reference_size, name="cursor_reference_size"
    )
    mouse_width, mouse_height = normalize_screen_size(
        mouse_reference_size, name="vpt_mouse_reference_size"
    )
    raw_x, raw_y = source_cursor
    out_of_bounds = not (
        0.0 <= raw_x < mouse_width and 0.0 <= raw_y < mouse_height
    )
    if out_of_bounds:
        return None, True
    mapped_x = raw_x / mouse_width * reference_width
    mapped_y = raw_y / mouse_height * reference_height
    return (float(mapped_x), float(mapped_y)), False


def _extract_vpt_mouse_pair(
    raw_action: Dict[str, Any],
) -> Optional[tuple[float, float]]:
    """Return an atomic finite VPT mouse pair, or None when it is missing."""
    mouse = raw_action.get("mouse")
    if not isinstance(mouse, dict) or mouse.get("x") is None or mouse.get("y") is None:
        return None
    try:
        raw_x = float(mouse["x"])
        raw_y = float(mouse["y"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VPT mouse x/y must be numeric, got {mouse!r}") from exc
    if not np.isfinite(raw_x) or not np.isfinite(raw_y):
        raise ValueError(f"VPT mouse x/y must be finite, got {(raw_x, raw_y)!r}")
    return raw_x, raw_y


def _overlay_gui_cursor(
    frame: np.ndarray,
    raw_action: Dict[str, Any],
    *,
    mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
) -> None:
    cursor, _ = _map_vpt_cursor_to_reference(
        raw_action,
        cursor_reference_size=(float(frame.shape[1]), float(frame.shape[0])),
        mouse_reference_size=mouse_reference_size,
    )
    if cursor is None:
        return
    cursor_image, cursor_alpha = _get_cursor_overlay()
    cursor_x = int(cursor[0])
    cursor_y = int(cursor[1])
    _composite_images_with_alpha(
        frame, cursor_image, cursor_alpha, cursor_x, cursor_y
    )


def _encode_vpt_video_with_actions(
    src_mp4: str,
    dst_mp4: Path,
    raw_actions: List[Dict[str, Any]],
    fps: int = 20,
    truncate_to: Optional[int] = None,
    keep_mask: Optional[np.ndarray] = None,
    mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
) -> None:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists() or dst_mp4.is_symlink():
        dst_mp4.unlink()

    cap = cv2.VideoCapture(src_mp4)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {src_mp4}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-an",
        str(dst_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    written = 0
    limit = len(raw_actions) if truncate_to is None else min(len(raw_actions), truncate_to)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if keep_mask is not None and frame_idx >= len(keep_mask):
            break
        if frame_idx >= limit:
            break
        keep_frame = keep_mask is None or bool(keep_mask[frame_idx])
        if keep_frame:
            _overlay_gui_cursor(
                frame,
                raw_actions[frame_idx],
                mouse_reference_size=mouse_reference_size,
            )
            proc.stdin.write(frame.tobytes())
            written += 1
        frame_idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with return code {proc.returncode}")
    if written == 0:
        raise RuntimeError("No frames were written during VPT video encoding")


def _json_action_to_env_action(
    json_action: Dict[str, Any],
    attack_is_stuck: bool = False,
    frame_idx: int = 0,
) -> Tuple[Dict[str, Any], bool, bool]:
    """Convert VPT raw JSON action to MC env_action dict.

    Matches the public OpenAI VPT `data_loader.py` handling of the
    `attack_is_stuck` recorder bug:
      - On frame 0, `newButtons == [0]` marks attack as stuck, and that same
        frame also has button 0 removed before conversion.
      - On later frames, while stuck, a fresh `newButtons` press of button 0
        clears the stuck state for that same frame; otherwise button 0 is
        removed from `mouse.buttons` before conversion.

    Returns:
        (env_action, is_null_action, attack_is_stuck)
    """
    mouse = json_action.get("mouse", {})

    mouse_buttons = list(mouse.get("buttons", []))
    new_buttons = mouse.get("newButtons", [])
    if frame_idx == 0:
        if new_buttons == [0]:
            attack_is_stuck = True
    elif attack_is_stuck:
        if 0 in new_buttons:
            attack_is_stuck = False
    if attack_is_stuck:
        mouse_buttons = [b for b in mouse_buttons if b != 0]

    env_action = copy.deepcopy(ENV_NULL_ACTION)
    is_null_action = True

    keyboard_keys = json_action.get("keyboard", {}).get("keys", [])
    is_gui_open = bool(_extract_info_from_raw_action(json_action).get("is_gui_open", False))
    for key in keyboard_keys:
        mapped = KEYBOARD_BUTTON_MAPPING.get(key)
        if mapped == "ESC" and is_gui_open:
            mapped = "inventory"
        if mapped is not None and mapped in env_action:
            env_action[mapped] = 1
            is_null_action = False

    for btn in mouse_buttons:
        if btn == 0:
            env_action["attack"] = 1
            is_null_action = False
        elif btn == 1:
            env_action["use"] = 1
            is_null_action = False

    dx = float(mouse.get("dx", 0.0))
    dy = float(mouse.get("dy", 0.0))
    camera_pitch = dy * CAMERA_SCALER
    camera_yaw = dx * CAMERA_SCALER
    if dx != 0.0 or dy != 0.0:
        is_null_action = False
    else:
        if abs(camera_pitch) > 180:
            camera_pitch = 0.0
        if abs(camera_yaw) > 180:
            camera_yaw = 0.0
    env_action["camera"] = np.array([camera_pitch, camera_yaw], dtype=np.float32)

    return env_action, is_null_action, attack_is_stuck


def _detect_hotbar_change(prev_hotbar: int, cur_hotbar: int, env_action: dict) -> dict:
    if cur_hotbar != prev_hotbar and 0 <= cur_hotbar <= 8:
        hotbar_key = f"hotbar.{cur_hotbar + 1}"
        if hotbar_key in env_action:
            env_action[hotbar_key] = 1
    return env_action


def _build_null_action_keep_mask(
    is_null_flags: List[bool],
    tail_frames: int = 2,
    force_keep_indices: Optional[List[int]] = None,
) -> np.ndarray:
    keep_mask = np.zeros(len(is_null_flags), dtype=bool)
    for idx, is_null in enumerate(is_null_flags):
        if not is_null:
            keep_mask[idx:min(len(is_null_flags), idx + tail_frames + 1)] = True
    if force_keep_indices is not None:
        for idx in force_keep_indices:
            if 0 <= idx < len(is_null_flags):
                keep_mask[idx] = True
    return keep_mask


def _load_vpt_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    actions.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise InvalidVPTJsonlError(
                        f"invalid JSON at line {line_no}"
                    ) from exc
    except UnicodeDecodeError as exc:
        raise InvalidVPTJsonlError("jsonl is not valid UTF-8") from exc
    except OSError as exc:
        raise InvalidVPTJsonlError(f"failed to read jsonl: {exc}") from exc
    return actions


def _extract_info_from_raw_action(raw_act: Dict[str, Any]) -> Dict[str, Any]:
    """Extract semantic state fields used by the Minecraft cursor raw schema.

    Proprio fields go through :func:`normalize_minecraft_info` so the
    rest of the codebase sees canonical keys before
    ``[pitch, yaw, cursor_x, cursor_y, current_hotbar, is_gui_open]`` is built.
    """
    info: Dict[str, Any] = {}
    for key in ("xpos", "ypos", "zpos", "pitch", "yaw"):
        if key in raw_act:
            info[key] = raw_act[key]
    info.update(normalize_minecraft_info(raw_act))
    return info


def _probe_video_meta(video_path: str) -> Optional[Tuple[int, int, int]]:
    """Return (frame_count, height, width) using OpenCV, or None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    if frame_count <= 0 or height <= 0 or width <= 0:
        return None
    return frame_count, height, width


def _reencode_video_with_mask(
    src_mp4: str,
    dst_mp4: Path,
    keep_mask: np.ndarray,
    fps: int = 20,
    raw_actions: Optional[List[Dict[str, Any]]] = None,
    mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
) -> None:
    """Re-encode src_mp4 keeping only frames where keep_mask[i] is True.

    If GUI opens in the kept frames, apply OpenAI VPT-style cursor overlay
    before encoding.
    """
    if raw_actions is not None and _selected_raw_actions_have_gui(raw_actions, keep_mask=keep_mask):
        _encode_vpt_video_with_actions(
            src_mp4,
            dst_mp4,
            raw_actions,
            fps=fps,
            keep_mask=keep_mask,
            mouse_reference_size=mouse_reference_size,
        )
        return

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists() or dst_mp4.is_symlink():
        dst_mp4.unlink()

    cap = cv2.VideoCapture(src_mp4)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {src_mp4}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-an",
        str(dst_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= len(keep_mask):
            break
        if keep_mask[frame_idx]:
            proc.stdin.write(frame.tobytes())
            written += 1
        frame_idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with return code {proc.returncode}")
    if written == 0:
        raise RuntimeError("No frames were written during masked video encoding")


def _place_video(
    src_mp4: str,
    dst_mp4: Path,
    truncate_to: Optional[int] = None,
    raw_actions: Optional[List[Dict[str, Any]]] = None,
    fps: int = 20,
    mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
) -> None:
    """Copy or re-encode source mp4 to dst.

    If GUI opens in the selected frames, apply OpenAI VPT-style cursor overlay
    before encoding.
    """
    if raw_actions is not None and _selected_raw_actions_have_gui(
        raw_actions, start_frame=0, num_frames=truncate_to,
    ):
        _encode_vpt_video_with_actions(
            src_mp4,
            dst_mp4,
            raw_actions,
            fps=fps,
            truncate_to=truncate_to,
            mouse_reference_size=mouse_reference_size,
        )
        return

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists() or dst_mp4.is_symlink():
        dst_mp4.unlink()

    if truncate_to is not None:
        _extract_video_segment(
            Path(src_mp4), dst_mp4, start_frame=0, num_frames=truncate_to
        )
        return

    shutil.copy2(src_mp4, dst_mp4)


def _extract_vpt_video_segment(
    src_mp4: Path,
    dst_mp4: Path,
    raw_actions: List[Dict[str, Any]],
    start_frame: int,
    num_frames: int,
    fps: int = 20,
    mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
) -> None:
    """Extract a contiguous VPT video segment, overlaying GUI cursor if needed."""
    if not _selected_raw_actions_have_gui(raw_actions, start_frame=start_frame, num_frames=num_frames):
        _extract_video_segment(src_mp4, dst_mp4, start_frame=start_frame, num_frames=num_frames)
        return

    temp_plain = dst_mp4.with_name(dst_mp4.stem + "_plain.mp4")
    try:
        _extract_video_segment(src_mp4, temp_plain, start_frame=start_frame, num_frames=num_frames)
        _place_video(
            str(temp_plain),
            dst_mp4,
            truncate_to=None,
            raw_actions=raw_actions[start_frame:start_frame + num_frames],
            fps=fps,
            mouse_reference_size=mouse_reference_size,
        )
    finally:
        if temp_plain.exists():
            temp_plain.unlink()


def _infer_batch_name(data_dir: str) -> str:
    """Read '6xx'..'10xx' from a path like '.../VPT_Contractor/raw/6xx/...'."""
    parts = Path(data_dir).resolve().parts
    for p in parts:
        if p.endswith("xx") and p[:-2].isdigit():
            return p
    raise ValueError(f"Could not infer VPT batch name from path: {data_dir}")


def _resolve_vpt_task_list(data_dir: str, explicit: Optional[str]) -> List[str]:
    """Return the list of task-description paraphrases for a VPT batch."""
    if explicit:
        return [explicit]
    batch_name = _infer_batch_name(data_dir)
    return VPT_BATCH_TASKS[batch_name]


BATCH_NAMES = ("6xx", "7xx", "8xx", "9xx", "10xx")


def discover_vpt_batches(vpt_root: str) -> Dict[str, str]:
    """Discover VPT batch directories (6xx..10xx) under vpt_root.

    If *vpt_root* itself is a batch directory (name matches ``<digits>xx``),
    return it directly. Otherwise scan subdirectories.
    """
    out: Dict[str, str] = {}
    root = Path(vpt_root)
    if not root.is_dir():
        raise FileNotFoundError(f"VPT root directory not found: {vpt_root}")
    # If vpt_root itself looks like a batch dir, use it directly.
    if root.name.endswith("xx") and root.name[:-2].isdigit():
        out[_infer_batch_name(str(root))] = str(root)
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.endswith("xx") and d.name[:-2].isdigit():
            out[_infer_batch_name(str(d))] = str(d)
    return out


def _preprocess_contractor_episode_worker(
    mp4_path: str,
    jsonl_path: str,
    action_dim: int,
    state_dim: int,
    skip_null_actions: bool,
    cursor_reference_size: Optional[Tuple[float, float]],
    mouse_reference_size: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    stem = Path(mp4_path).stem
    try:
        raw_actions = _load_vpt_jsonl(jsonl_path)
    except InvalidVPTJsonlError as exc:
        print(f"    [SKIP] {stem}: {exc}")
        return None
    if not raw_actions:
        print(f"    [SKIP] {stem}: empty jsonl")
        return None

    meta = _probe_video_meta(mp4_path)
    if meta is None:
        print(f"    [SKIP] {stem}: cannot read video metadata")
        return None
    video_frame_count, vid_h, vid_w = meta
    episode_cursor_reference_size = resolve_cursor_reference_size(
        cursor_reference_size,
        fallback_size=(vid_w, vid_h),
    )

    n = min(video_frame_count, len(raw_actions))

    env_actions = []
    state_vectors_list = []
    state_field_valid_list = []
    is_null_flags = []
    tracker = MinecraftStateTracker()
    gui_open_frames = 0
    gui_cursor_frames = 0
    cursor_valid_frames = 0
    cursor_missing_frames = 0
    cursor_out_of_bounds_frames = 0
    cursor_source_x: List[float] = []
    cursor_source_y: List[float] = []
    attack_is_stuck = False
    for j in range(n):
        raw_act = raw_actions[j]
        env_action, is_null, attack_is_stuck = _json_action_to_env_action(
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
            cursor_reference_size=episode_cursor_reference_size,
            mouse_reference_size=mouse_reference_size,
        )
        cursor_out_of_bounds_frames += int(cursor_out_of_bounds)
        if cursor is not None:
            info["cursor_x"], info["cursor_y"] = cursor
            cursor_valid_frames += 1
        observed_values, observed_valid = extract_state_with_validity(
            info,
            cursor_reference_size=episode_cursor_reference_size,
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
        state_valid = apply_gui_semantic_validity_numpy(
            state_values, state_valid
        )
        tracker.commit_after_action(observed_values, observed_valid, env_action)
        env_actions.append(env_action)
        state_vectors_list.append(state_values[:state_dim])
        state_field_valid_list.append(state_valid)
        is_null_flags.append(is_null)

    if len(env_actions) < 2:
        print(f"    [SKIP] {stem}: too few valid frames ({len(env_actions)})")
        return None

    action_vectors = np.stack(
        [action_to_vector(a)[:action_dim] for a in env_actions], axis=0
    )
    state_vectors = np.stack(state_vectors_list, axis=0).astype(np.float32)
    state_field_valid_vectors = np.stack(state_field_valid_list, axis=0).astype(np.bool_)

    if skip_null_actions:
        keep_mask = _build_null_action_keep_mask(is_null_flags)
        kept_count = int(keep_mask.sum())
        if kept_count < 2:
            print(f"    [SKIP] {stem}: <2 kept frames after filtering "
                  f"({kept_count}/{n})")
            return None
        action_vectors = action_vectors[keep_mask]
        state_vectors = state_vectors[keep_mask]
        state_field_valid_vectors = state_field_valid_vectors[keep_mask]
        n_kept = kept_count
        truncate_to = None
    else:
        keep_mask = None
        n_kept = n
        truncate_to = n if video_frame_count != n else None

    return {
        "mp4_path": mp4_path,
        "jsonl_path": jsonl_path,
        "n_original": n,
        "n_kept": n_kept,
        "vid_h": vid_h,
        "vid_w": vid_w,
        "cursor_reference_size": episode_cursor_reference_size,
        "truncate_to": truncate_to,
        "keep_mask": keep_mask,
        "action_vectors": action_vectors,
        "state_vectors": state_vectors,
        "state_field_valid_vectors": state_field_valid_vectors,
        "state_stats": tracker.stats.as_dict(),
        "cursor_stats": {
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
        },
    }


def convert_vpt_contractor_dataset(
    data_dir: str,
    output_dir: str,
    task: Optional[str] = None,
    fps: int = 20,
    max_episodes: Optional[int] = None,
    ratio: Optional[float] = None,
    skip_null_actions: bool = False,
    action_dim: int = 22,
    state_dim: int = RAW_STATE_DIM,
    cursor_reference_size: Optional[Tuple[float, float]] = None,
    vpt_mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 32,
) -> List[str]:
    """Convert a directory of VPT contractor data to LeRobot format."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"VPT data directory not found: {data_dir}")
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

    task_paraphrases = _resolve_vpt_task_list(data_dir, task)
    print(
        f"VPT task (batch={_infer_batch_name(data_dir)}): "
        f"{len(task_paraphrases)} paraphrase(s)"
    )
    for t in task_paraphrases:
        print(f'    task: "{t}"')

    mp4_files = sorted(data_path.rglob("*.mp4"))
    jsonl_by_stem: Dict[str, Path] = {p.stem: p for p in data_path.rglob("*.jsonl")}

    pairs: List[Tuple[str, str]] = []
    skipped_no_jsonl = 0
    for mp4 in mp4_files:
        jsonl = jsonl_by_stem.get(mp4.stem)
        if jsonl is None:
            sibling = mp4.with_suffix(".jsonl")
            if sibling.exists():
                jsonl = sibling
        if jsonl is not None:
            pairs.append((str(mp4), str(jsonl)))
        else:
            skipped_no_jsonl += 1

    if not pairs:
        raise ValueError(f"No valid (mp4 + jsonl) pairs found in {data_dir}")

    print(f"\n[1/4] Found {len(pairs)} valid pairs in {data_dir} "
          f"({len(mp4_files)} mp4s, {skipped_no_jsonl} without jsonl)")

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in idx]

    if ratio is not None:
        assert 0.0 < ratio <= 1.0, f"ratio must be in (0, 1], got {ratio}"
        n_keep = max(1, int(len(pairs) * ratio))
        pairs = pairs[:n_keep]
        print(f"Applied ratio={ratio}: keeping {len(pairs)} pairs")

    if max_episodes is not None:
        # Fetch extra pairs to compensate for validation failures.
        # Typical failure rate is ~3-5%, so 20% headroom is conservative.
        fetch_count = min(len(pairs), int(max_episodes * 1.2) + 10)
        pairs = pairs[:fetch_count]
        print(f"Applied max_episodes={max_episodes}: fetching {len(pairs)} candidates "
              f"(with headroom for validation failures)")

    # ── Phase A: pre-validate & collect tasks ──────────────────────
    print(f"\n[2/4] Pre-validating episodes ...")

    actual_workers = min(num_workers, max(1, len(pairs)))
    valid_tasks: List[Dict[str, Any]] = []
    failed = 0

    from functools import partial
    worker_fn = partial(
        _preprocess_contractor_episode_worker,
        action_dim=action_dim,
        state_dim=state_dim,
        skip_null_actions=skip_null_actions,
        cursor_reference_size=cursor_reference_size,
        mouse_reference_size=vpt_mouse_reference_size,
    )

    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = {
            executor.submit(worker_fn, pair[0], pair[1]): i
            for i, pair in enumerate(pairs)
        }
        results_by_idx: Dict[int, Optional[Dict[str, Any]]] = {}
        with tqdm(total=len(futures), desc="    Pre-validate", unit="file") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                results_by_idx[idx] = future.result()
                pbar.update(1)

    # Collect results in original order for deterministic task assignment.
    for i in range(len(pairs)):
        result = results_by_idx.get(i)
        if result is None:
            failed += 1
            continue
        result["task"] = task_paraphrases[i % len(task_paraphrases)]
        valid_tasks.append(result)

    print(f"  Pre-validated: {len(valid_tasks)} valid, {failed} skipped")

    reference_sizes = {tuple(t["cursor_reference_size"]) for t in valid_tasks}
    if len(reference_sizes) > 1:
        raise ValueError(
            "Selected VPT episodes resolve to different cursor reference sizes: "
            f"{sorted(reference_sizes)}. Pass --cursor_reference_size explicitly "
            "or convert resolutions into separate datasets."
        )

    if max_episodes is not None and len(valid_tasks) > max_episodes:
        valid_tasks = valid_tasks[:max_episodes]
        print(f"  Capped to max_episodes={max_episodes}")
    elif max_episodes is not None and len(valid_tasks) < max_episodes:
        print(f"  [WARN] Only {len(valid_tasks)} valid episodes available, "
              f"less than requested max_episodes={max_episodes}")

    if not valid_tasks:
        print(f"\nDone: 0 converted, {failed} skipped/failed → {output_dir}")
        return []
    resolved_cursor_reference_size = tuple(valid_tasks[0]["cursor_reference_size"])

    # ── Phase B: create / open LeRobot dataset & parallel video placement ──
    output_path = Path(output_dir)
    if (output_path / "meta" / "info.json").exists():
        dataset = LeRobotDataset(
            repo_id=output_dir,
            root=output_path,
            is_compute_episode_stats_image=False,
        )
    else:
        first = valid_tasks[0]
        video_info = build_video_info(Path(first["mp4_path"]), fps=fps)
        features = {
            "observation.images": {
                "dtype": "video",
                "shape": (3, first["vid_h"], first["vid_w"]),
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

    # Place all videos in parallel to a temp directory.
    temp_dir = output_path.parent / ("_temp_videos_" + output_path.name)
    temp_dir.mkdir(parents=True, exist_ok=True)

    num_vkeys = len(dataset.meta.video_keys)
    actual_workers = min(num_workers, max(1, len(valid_tasks)))

    if skip_null_actions:
        # ── skip_null_actions path: re-encode with frame mask ──
        print(
            f"\n[3/4] Re-encoding {len(valid_tasks)} video(s) "
            f"(skipping null actions) with {actual_workers} worker(s) ..."
        )

        def _reencode_task(task: Dict[str, Any], idx: int) -> Tuple[int, bool]:
            src = task["mp4_path"]
            stem = Path(src).stem
            try:
                raw_actions = _load_vpt_jsonl(task["jsonl_path"])
                for ki, vkey in enumerate(dataset.meta.video_keys):
                    temp_mp4 = temp_dir / f"clip_{idx:06d}_vk{ki}.mp4"
                    _reencode_video_with_mask(
                        src,
                        temp_mp4,
                        task["keep_mask"],
                        fps=fps,
                        raw_actions=raw_actions,
                        mouse_reference_size=vpt_mouse_reference_size,
                    )
                return idx, True
            except InvalidVPTJsonlError as exc:
                print(f"    [SKIP] {stem}: {exc}")
                return idx, False
            except Exception as exc:
                print(f"    [WARN] {stem}: re-encode failed: {exc}")
                return idx, False

        placement_ok: List[bool] = [False] * len(valid_tasks)
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_reencode_task, t, i): i
                for i, t in enumerate(valid_tasks)
            }
            with tqdm(total=len(futures), desc="    Re-encode", unit="file") as pbar:
                for future in as_completed(futures):
                    idx, ok = future.result()
                    placement_ok[idx] = ok
                    pbar.update(1)
    else:
        # ── default path: copy or truncate ──
        placement_args: List[Tuple[str, str, Path, Optional[int]]] = []
        for i, t in enumerate(valid_tasks):
            for ki, vkey in enumerate(dataset.meta.video_keys):
                temp_mp4 = temp_dir / f"clip_{i:06d}_vk{ki}.mp4"
                placement_args.append(
                    (t["mp4_path"], t["jsonl_path"], temp_mp4, t["truncate_to"])
                )

        actual_workers = min(num_workers, max(1, len(placement_args)))
        print(
            f"\n[3/4] Placing {len(placement_args)} video(s) "
            f"with {actual_workers} worker(s) ..."
        )

        placement_ok_list: List[bool] = [False] * len(placement_args)
        def _place_task(src: str, jsonl_path: str, dst: Path, trunc: Optional[int]) -> bool:
            stem = Path(src).stem
            try:
                raw_actions = _load_vpt_jsonl(jsonl_path)
                _place_video(
                    src,
                    dst,
                    trunc,
                    raw_actions=raw_actions,
                    fps=fps,
                    mouse_reference_size=vpt_mouse_reference_size,
                )
                return True
            except InvalidVPTJsonlError as exc:
                print(f"    [SKIP] {stem}: {exc}")
                return False
            except Exception as exc:
                print(f"    [WARN] {stem}: video placement failed: {exc}")
                return False

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_idx = {
                executor.submit(_place_task, src, jsonl_path, dst, trunc): idx
                for idx, (src, jsonl_path, dst, trunc) in enumerate(placement_args)
            }
            with tqdm(total=len(future_to_idx), desc="    Video placement", unit="clip") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    placement_ok_list[idx] = future.result()
                    pbar.update(1)

        # Convert per-clip ok list to per-episode ok list
        placement_ok = [
            all(placement_ok_list[i * num_vkeys + ki] for ki in range(num_vkeys))
            for i in range(len(valid_tasks))
        ]

    ok_count = sum(placement_ok)
    print(f"  Video placement complete: {ok_count}/{len(valid_tasks)} succeeded")

    # ── Phase C: write to dataset sequentially ─────────────────────
    dummy_img = PILImage.new("RGB", (1, 1))
    print(f"\n[4/4] Writing {len(valid_tasks)} episodes to dataset ...")

    created: List[str] = []
    write_failed = 0

    for i, t in enumerate(valid_tasks):
        mp4_path = t["mp4_path"]
        stem = Path(mp4_path).stem
        n_kept = t["n_kept"]
        ep_task = t["task"]
        action_vectors = t["action_vectors"]
        state_vectors = t["state_vectors"]
        state_field_valid_vectors = t["state_field_valid_vectors"]

        current_ep_idx = dataset.meta.total_episodes
        video_prepare_failed = False

        # Move pre-placed video(s) from temp dir to the dataset path.
        for ki, vkey in enumerate(dataset.meta.video_keys):
            temp_mp4 = temp_dir / f"clip_{i:06d}_vk{ki}.mp4"
            dst_mp4 = output_path / dataset.meta.get_video_file_path(current_ep_idx, vkey)
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            if dst_mp4.exists() or dst_mp4.is_symlink():
                dst_mp4.unlink()
            try:
                if placement_ok[i] and temp_mp4.exists():
                    shutil.move(str(temp_mp4), str(dst_mp4))
                elif skip_null_actions:
                    _reencode_video_with_mask(
                        mp4_path,
                        dst_mp4,
                        t["keep_mask"],
                        fps=fps,
                        raw_actions=_load_vpt_jsonl(t["jsonl_path"]),
                        mouse_reference_size=vpt_mouse_reference_size,
                    )
                else:
                    _place_video(
                        mp4_path,
                        dst_mp4,
                        truncate_to=t["truncate_to"],
                        raw_actions=_load_vpt_jsonl(t["jsonl_path"]),
                        fps=fps,
                        mouse_reference_size=vpt_mouse_reference_size,
                    )
            except InvalidVPTJsonlError as exc:
                print(f"  [SKIP] {stem}: {exc}")
                video_prepare_failed = True
                break
            except Exception as exc:
                print(f"  [WARN] {stem}: failed to prepare video: {exc}")
                video_prepare_failed = True
                break

        if video_prepare_failed:
            for vkey in dataset.meta.video_keys:
                dst_mp4 = output_path / dataset.meta.get_video_file_path(current_ep_idx, vkey)
                if dst_mp4.exists():
                    dst_mp4.unlink()
            write_failed += 1
            continue

        dataset.episode_buffer = dataset.create_episode_buffer()

        null_info = ""
        if skip_null_actions and t["n_original"] != n_kept:
            null_info = f" (kept {n_kept}/{t['n_original']}, skipped {t['n_original'] - n_kept} null)"

        print(
            f'  [{len(created) + 1}/{len(valid_tasks)}] '
            f'{Path(mp4_path).stem}  task="{ep_task}"{null_info}'
        )

        for idx in range(n_kept):
            frame_data = {
                "observation.images": dummy_img,
                "observation.state": state_vectors[idx].astype(np.float32),
                "observation.state_field_valid": state_field_valid_vectors[idx].astype(bool),
                "observation.state_dim_is_pad": np.zeros(state_dim, dtype=bool),
                "action": action_vectors[idx].astype(np.float32),
            }
            dataset.add_frame(frame_data, task=[ep_task, ep_task])

        if dataset.episode_buffer["size"] < 2:
            print(f"  [WARN] Too few frames for {mp4_path}")
            for vkey in dataset.meta.video_keys:
                dst_mp4 = output_path / dataset.meta.get_video_file_path(current_ep_idx, vkey)
                if dst_mp4.exists():
                    dst_mp4.unlink()
            write_failed += 1
            continue

        dataset.save_episode(
            raw_file_name=Path(mp4_path).stem,
            extra_fields={
                "minecraft_state_stats": t["state_stats"],
                "cursor_mapping_stats": t["cursor_stats"],
            },
        )
        created.append(output_dir)

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    total_failed = failed + write_failed
    print(f"\nDone: {len(created)} converted, {total_failed} skipped/failed → {output_dir}")
    return created

# Multi-batch mixing


def convert_vpt_contractor_multi_batch(
    vpt_root: str,
    output_dir: str,
    fps: int = 20,
    skip_null_actions: bool = False,
    action_dim: int = 22,
    state_dim: int = RAW_STATE_DIM,
    cursor_reference_size: Optional[Tuple[float, float]] = None,
    vpt_mouse_reference_size: Tuple[float, float] = DEFAULT_VPT_MOUSE_REFERENCE_SIZE,
    batch_filter: Optional[List[str]] = None,
    batch_ratios: Optional[Dict[str, float]] = None,
    batch_max_episodes: Optional[Dict[str, int]] = None,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 32,
) -> List[str]:
    """Convert multiple VPT contractor batches (6xx..10xx) into one LeRobot dataset."""
    # Fail fast on stale output_dir contents.
    assert_output_dir_empty(output_dir)

    available_batches = discover_vpt_batches(vpt_root)
    target_batches = batch_filter if batch_filter else list(BATCH_NAMES)

    print(f"\nVPT Contractor Multi-Batch: root={vpt_root} "
          f"available={sorted(available_batches.keys())} "
          f"target={target_batches} ratios={batch_ratios} caps={batch_max_episodes}")

    all_created: List[str] = []

    for b_idx, batch_name in enumerate(target_batches):
        if batch_name not in available_batches:
            print(f"\n[SKIP] {batch_name}: no VPT data directory")
            continue

        data_dir = available_batches[batch_name]
        per_batch_ratio = None
        if batch_ratios:
            per_batch_ratio = batch_ratios.get(batch_name)
        per_batch_cap = None
        if batch_max_episodes:
            per_batch_cap = batch_max_episodes.get(batch_name)

        print(f"\nBatch: {batch_name} ratio={per_batch_ratio} cap={per_batch_cap}")

        created = convert_vpt_contractor_dataset(
            data_dir=data_dir,
            output_dir=output_dir,
            task=None,
            fps=fps,
            max_episodes=per_batch_cap,
            ratio=per_batch_ratio,
            skip_null_actions=skip_null_actions,
            action_dim=action_dim,
            state_dim=state_dim,
            cursor_reference_size=cursor_reference_size,
            vpt_mouse_reference_size=vpt_mouse_reference_size,
            shuffle=shuffle,
            seed=seed + b_idx,
            num_workers=num_workers,
        )
        all_created.extend(created)

    print(f"\nAll batches done: {len(all_created)} episodes → {output_dir}")
    return all_created
