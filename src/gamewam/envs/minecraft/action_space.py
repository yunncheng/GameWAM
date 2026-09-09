"""Minecraft action space: 22-dim continuous vector serialization.

Vector layout (22 dims):
    [0]  camera_pitch  (continuous, angle delta in degrees)
    [1]  camera_yaw    (continuous, angle delta in degrees)
    [2..21] binary keys from ENV_NULL_ACTION (20 dims, excluding 'camera')
"""

import copy
from typing import Optional

import numpy as np

ENV_NULL_ACTION = {
    'forward': 0, 'back': 0, 'left': 0, 'right': 0,
    'jump': 0, 'sneak': 0, 'sprint': 0,
    'attack': 0, 'use': 0, 'drop': 0,
    'inventory': 0,
    'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0,
    'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0,
    'camera': np.array([0.0, 0.0]),
}

_BINARY_KEYS = [k for k in ENV_NULL_ACTION if k != "camera"]
BINARY_KEYS = tuple(_BINARY_KEYS)
ACTION_DIM = 2 + len(_BINARY_KEYS)  # 2 camera + 20 binary = 22

CAMERA_PITCH_IDX = 0
CAMERA_YAW_IDX = 1
BINARY_START_IDX = 2
_HOTBAR_KEYS = [k for k in _BINARY_KEYS if k.startswith("hotbar.")]
HOTBAR_KEYS = tuple(_HOTBAR_KEYS)
HOTBAR_START_IDX = BINARY_START_IDX + _BINARY_KEYS.index(_HOTBAR_KEYS[0])
HOTBAR_END_IDX = HOTBAR_START_IDX + len(_HOTBAR_KEYS)
LOCOMOTION_KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint")
INTERACTION_KEYS = ("attack", "use", "drop", "inventory")
ACTION_GROUP_INDICES = {
    "continuous": (CAMERA_PITCH_IDX, CAMERA_YAW_IDX),
    "locomotion": tuple(BINARY_START_IDX + _BINARY_KEYS.index(k) for k in LOCOMOTION_KEYS),
    "interaction": tuple(BINARY_START_IDX + _BINARY_KEYS.index(k) for k in INTERACTION_KEYS),
    "hotbar": tuple(range(HOTBAR_START_IDX, HOTBAR_END_IDX)),
}

CAMERA_SCALER = 360.0 / 2400.0


def switched_hotbar_slot(env_action: dict) -> Optional[int]:
    """Return the zero-based selected hotbar slot, if any."""
    for slot in range(1, 10):
        if bool(env_action.get(f"hotbar.{slot}", 0)):
            return slot - 1
    return None


def action_to_vector(env_action: dict) -> np.ndarray:
    """Convert MC env_action dict to 22-dim continuous vector.

    Camera values are in degrees (pitch, yaw).
    Binary keys are 0.0 or 1.0.
    """
    vec = np.zeros(ACTION_DIM, dtype=np.float32)

    camera = env_action.get("camera", np.array([0.0, 0.0]))
    if isinstance(camera, (list, tuple)):
        camera = np.array(camera, dtype=np.float32)
    vec[CAMERA_PITCH_IDX] = float(camera[0])  # pitch
    vec[CAMERA_YAW_IDX] = float(camera[1])    # yaw

    for i, key in enumerate(_BINARY_KEYS):
        vec[BINARY_START_IDX + i] = float(env_action.get(key, 0))

    return vec


def vector_to_action(
    vec: np.ndarray,
    binarize: bool = True,
    binary_threshold: float = 0.5,
    hotbar_argmax: bool = True,
    camera_clip: float = 180.0,
) -> dict:
    """Convert 22-dim vector back to MC env_action dict.

    Args:
        vec: 22-dim action vector.
        binarize: Whether to threshold binary dimensions at 0.5.
        binary_threshold: Threshold for binarization.
        hotbar_argmax: If True, only the hotbar slot with highest activation
                       is set to 1 (and only if it exceeds the threshold).
        camera_clip: Clip camera values to [-clip, clip] degrees.
    """
    vec = np.asarray(vec)
    if vec.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {vec.shape}")
    if not bool(np.isfinite(vec).all()):
        raise ValueError("Minecraft action vector must contain only finite values")

    env_action = copy.deepcopy(ENV_NULL_ACTION)

    pitch = float(np.clip(vec[CAMERA_PITCH_IDX], -camera_clip, camera_clip))
    yaw = float(np.clip(vec[CAMERA_YAW_IDX], -camera_clip, camera_clip))
    env_action["camera"] = np.array([pitch, yaw], dtype=np.float32)

    for i, key in enumerate(_BINARY_KEYS):
        val = float(vec[BINARY_START_IDX + i])
        if binarize:
            val = 1 if val >= binary_threshold else 0
        env_action[key] = int(val) if binarize else val

    if hotbar_argmax and binarize:
        hotbar_vals = vec[HOTBAR_START_IDX:HOTBAR_END_IDX]
        max_idx = int(np.argmax(hotbar_vals))
        max_val = hotbar_vals[max_idx]
        for i in range(HOTBAR_START_IDX, HOTBAR_END_IDX):
            key = _BINARY_KEYS[i - BINARY_START_IDX]
            env_action[key] = 0
        if max_val >= binary_threshold:
            key = _BINARY_KEYS[max_idx + HOTBAR_START_IDX - BINARY_START_IDX]
            env_action[key] = 1

    return env_action


def get_delta_mask() -> np.ndarray:
    """Return boolean mask: True for dimensions that are delta (camera), False otherwise."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[CAMERA_PITCH_IDX] = True
    mask[CAMERA_YAW_IDX] = True
    return mask
