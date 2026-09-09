"""Minecraft proprio state contract and extraction."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


FIELD_NAMES = (
    "pitch",
    "yaw",
    "cursor_x",
    "cursor_y",
    "current_hotbar",
    "is_gui_open",
)
RAW_STATE_DIM = len(FIELD_NAMES)
FIELD_VALID_DIM = RAW_STATE_DIM
STATE_FIELD_VALID_NAMES = tuple(f"{name}_valid" for name in FIELD_NAMES)
FIELD_INDEX = {name: index for index, name in enumerate(FIELD_NAMES)}

PITCH_IDX = FIELD_INDEX["pitch"]
YAW_IDX = FIELD_INDEX["yaw"]
CURSOR_X_IDX = FIELD_INDEX["cursor_x"]
CURSOR_Y_IDX = FIELD_INDEX["cursor_y"]
HOTBAR_IDX = FIELD_INDEX["current_hotbar"]
GUI_OPEN_IDX = FIELD_INDEX["is_gui_open"]

CANONICAL_FIELD_NAMES = (
    "pitch",
    "sin_yaw",
    "cos_yaw",
    "cursor_u",
    "cursor_v",
    *(f"hotbar_{slot}" for slot in range(9)),
    "is_gui_open",
)
CANONICAL_STATE_DIM = len(CANONICAL_FIELD_NAMES)
CANONICAL_PITCH_SLICE = slice(0, 1)
CANONICAL_YAW_SLICE = slice(1, 3)
CANONICAL_CURSOR_SLICE = slice(3, 5)
CANONICAL_HOTBAR_SLICE = slice(5, 14)
CANONICAL_GUI_SLICE = slice(14, 15)


def normalize_minecraft_info(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return canonical state keys without mutating a VPT/env info mapping.

    MineStudio does not expose camera pose at the top level. Its
    ``HumanSurvival`` wrapper returns pose in both ``location_stats`` (the
    monitor payload) and ``player_pos`` (the processed observation payload).
    Accept those native layouts here so offline VPT rows and live MineStudio
    observations meet at the same canonical boundary.
    """
    info = dict(raw)
    for nested_key in ("player_pos", "location_stats"):
        nested = info.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for field_name in ("pitch", "yaw"):
            if info.get(field_name) is None and nested.get(field_name) is not None:
                info[field_name] = nested[field_name]
    if "hotbar" in info:
        if "current_hotbar" not in info:
            info["current_hotbar"] = info["hotbar"]
        info.pop("hotbar")
    if "current_item_index" in info:
        if "current_hotbar" not in info:
            info["current_hotbar"] = info["current_item_index"]
        info.pop("current_item_index")
    if "isGuiOpen" in info:
        if "is_gui_open" not in info:
            info["is_gui_open"] = info["isGuiOpen"]
        info.pop("isGuiOpen")
    return info


def normalize_screen_size(size: Sequence[float], *, name: str) -> tuple[float, float]:
    """Validate a configured ``(width, height)`` screen size."""
    if len(size) != 2:
        raise ValueError(f"{name} must contain exactly (width, height), got {size!r}")
    width = _finite_float(size[0], f"{name}.width")
    height = _finite_float(size[1], f"{name}.height")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{name} must be positive, got {(width, height)!r}")
    return width, height


def _finite_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Minecraft state field {field!r} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Minecraft state field {field!r} must be finite, got {value!r}")
    return out


def extract_gui_open(info: Dict[str, Any]) -> Optional[bool]:
    """Return validated GUI state, or ``None`` when it is missing/null."""
    if "is_gui_open" not in info or info.get("is_gui_open") is None:
        return None
    gui = info.get("is_gui_open")
    if isinstance(gui, (bool, np.bool_)):
        return bool(gui)
    if isinstance(gui, (int, float, np.integer, np.floating)) and gui in (0, 1, 0.0, 1.0):
        return bool(gui)
    raise ValueError(f"Minecraft is_gui_open must be boolean or 0/1, got {gui!r}")


def extract_state_with_validity(
    info: Dict[str, Any],
    *,
    cursor_reference_size: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw 6-D Minecraft state plus semantic field validity.

    Cursor coordinates are an atomic pair. Their bounds come from the caller's
    configured reference size rather than a module-level render-size constant.
    """
    values = np.zeros(RAW_STATE_DIM, dtype=np.float32)
    valid = np.zeros(FIELD_VALID_DIM, dtype=np.bool_)

    if "pitch" in info and info.get("pitch") is not None:
        values[PITCH_IDX] = _finite_float(info.get("pitch"), "pitch")
        valid[PITCH_IDX] = True

    if "yaw" in info and info.get("yaw") is not None:
        values[YAW_IDX] = _finite_float(info.get("yaw"), "yaw")
        valid[YAW_IDX] = True

    if "current_hotbar" in info and info.get("current_hotbar") is not None:
        hotbar_f = _finite_float(info.get("current_hotbar"), "current_hotbar")
        hotbar_i = int(hotbar_f)
        if hotbar_f != float(hotbar_i) or not 0 <= hotbar_i <= 8:
            raise ValueError(f"Minecraft current_hotbar must be an integer in [0, 8], got {hotbar_f!r}")
        values[HOTBAR_IDX] = float(hotbar_i)
        valid[HOTBAR_IDX] = True

    gui_open = extract_gui_open(info)
    if gui_open is not None:
        values[GUI_OPEN_IDX] = float(gui_open)
        valid[GUI_OPEN_IDX] = True

    if gui_open is not True:
        return values, valid

    cursor_x = info.get("cursor_x")
    cursor_y = info.get("cursor_y")
    if cursor_x is None or cursor_y is None:
        return values, valid
    if cursor_reference_size is None:
        raise ValueError(
            "cursor_reference_size=(width, height) is required when a GUI-open "
            "Minecraft state contains cursor coordinates"
        )
    reference_width, reference_height = normalize_screen_size(
        cursor_reference_size,
        name="cursor_reference_size",
    )
    cursor_x = _finite_float(cursor_x, "cursor_x")
    cursor_y = _finite_float(cursor_y, "cursor_y")
    if not 0.0 <= cursor_x <= reference_width:
        raise ValueError(
            f"Minecraft cursor_x must be in [0, {reference_width}], got {cursor_x!r}"
        )
    if not 0.0 <= cursor_y <= reference_height:
        raise ValueError(
            f"Minecraft cursor_y must be in [0, {reference_height}], got {cursor_y!r}"
        )
    values[CURSOR_X_IDX] = cursor_x
    values[CURSOR_Y_IDX] = cursor_y
    valid[CURSOR_X_IDX] = True
    valid[CURSOR_Y_IDX] = True

    return values, valid


def extract_state(
    info: Dict[str, Any],
    *,
    cursor_reference_size: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return raw state values, with missing fields represented as zeros."""
    values, _ = extract_state_with_validity(
        info, cursor_reference_size=cursor_reference_size
    )
    return values


def apply_gui_semantic_validity_numpy(
    state_values: "np.ndarray",
    valid: "np.ndarray",
) -> "np.ndarray":
    """Return source validity constrained by Minecraft GUI semantics."""
    state_values = np.asarray(state_values)
    valid = np.asarray(valid, dtype=np.bool_)
    if state_values.ndim == 0 or valid.ndim == 0:
        raise ValueError("Minecraft state and validity must have rank >= 1")
    if state_values.shape[-1] != RAW_STATE_DIM:
        raise ValueError(
            f"Minecraft state must have last dim {RAW_STATE_DIM}, got "
            f"{state_values.shape}"
        )
    if valid.shape[-1] != FIELD_VALID_DIM:
        raise ValueError(
            f"Minecraft validity must have last dim {FIELD_VALID_DIM}, got "
            f"{valid.shape}"
        )
    if state_values.shape[:-1] != valid.shape[:-1]:
        raise ValueError(
            "Minecraft state and validity leading shapes must match, got "
            f"{state_values.shape[:-1]} and {valid.shape[:-1]}"
        )
    if np.any(valid[..., CURSOR_X_IDX] != valid[..., CURSOR_Y_IDX]):
        raise ValueError("cursor_x/cursor_y validity must be an atomic pair")

    gui_known = valid[..., GUI_OPEN_IDX]
    gui_open = gui_known & (state_values[..., GUI_OPEN_IDX] > 0.5)
    gui_closed = gui_known & ~gui_open

    out = valid.copy()
    out[..., PITCH_IDX] &= gui_closed
    out[..., YAW_IDX] &= gui_closed
    out[..., HOTBAR_IDX] &= gui_closed
    out[..., CURSOR_X_IDX] &= gui_open
    out[..., CURSOR_Y_IDX] &= gui_open
    return out


def apply_gui_semantic_validity(
    raw_state: "torch.Tensor",
    valid: "torch.Tensor",
) -> "torch.Tensor":
    """Torch counterpart of :func:`apply_gui_semantic_validity_numpy`."""
    import torch

    if raw_state.ndim == 0 or valid.ndim == 0:
        raise ValueError("Minecraft state and validity must have rank >= 1")
    if raw_state.shape[-1] != RAW_STATE_DIM:
        raise ValueError(
            f"Minecraft state must have last dim {RAW_STATE_DIM}, got "
            f"{tuple(raw_state.shape)}"
        )
    if valid.shape[-1] != FIELD_VALID_DIM:
        raise ValueError(
            f"Minecraft validity must have last dim {FIELD_VALID_DIM}, got "
            f"{tuple(valid.shape)}"
        )
    if raw_state.shape[:-1] != valid.shape[:-1]:
        raise ValueError(
            "Minecraft state and validity leading shapes must match, got "
            f"{tuple(raw_state.shape[:-1])} and {tuple(valid.shape[:-1])}"
        )

    valid = valid.to(dtype=torch.bool)
    if bool((valid[..., CURSOR_X_IDX] != valid[..., CURSOR_Y_IDX]).any().item()):
        raise ValueError("cursor_x/cursor_y validity must be an atomic pair")

    gui_known = valid[..., GUI_OPEN_IDX]
    gui_open = gui_known & (raw_state[..., GUI_OPEN_IDX] > 0.5)
    gui_closed = gui_known & ~gui_open

    out = valid.clone()
    out[..., PITCH_IDX] &= gui_closed
    out[..., YAW_IDX] &= gui_closed
    out[..., HOTBAR_IDX] &= gui_closed
    out[..., CURSOR_X_IDX] &= gui_open
    out[..., CURSOR_Y_IDX] &= gui_open
    return out
