"""VizDoom action space: 9-dim canonical action vector serialization.

Mirrors the 9-dim canonical action used by the APPO data collector
(``scripts/datasets/vizdoom/appo/appo_common.py``) and the training config
(``configs/data/vizdoom.yaml``). Vector layout
(9 dims, order is the single source of truth):
    [0] shoot          (ATTACK, binary)
    [1] move_forward   (MOVE_FORWARD, binary)
    [2] move_backward  (MOVE_BACKWARD, binary)
    [3] strafe_left    (MOVE_LEFT, binary)
    [4] strafe_right   (MOVE_RIGHT, binary)
    [5] speed          (SPEED, binary)
    [6] turn_left      (binary; defend scenarios)
    [7] turn_right     (binary; defend scenarios)
    [8] turn_delta     (continuous [-1,1] -> ±TURN_DELTA_MAX_DEG; battle scenarios)

``turn_left``/``turn_right`` (binary) and ``turn_delta`` (continuous) are
mutually exclusive per scenario but coexist in the canonical vector. They
remain separate at execution time: defend scenarios use VizDoom's native
binary turn buttons (including their native acceleration), while Battle
scenarios use ``TURN_LEFT_RIGHT_DELTA``.
"""

import copy
from typing import Dict, List

import numpy as np

ENV_NULL_ACTION: Dict[str, float] = {
    "shoot": 0.0,
    "move_forward": 0.0,
    "move_backward": 0.0,
    "strafe_left": 0.0,
    "strafe_right": 0.0,
    "speed": 0.0,
    "turn_left": 0.0,
    "turn_right": 0.0,
    "turn_delta": 0.0,
}

ACTION_KEYS: List[str] = list(ENV_NULL_ACTION.keys())
ACTION_DIM = len(ACTION_KEYS)  # 9

VIZDOOM_BUTTONS = [
    "ATTACK",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "SPEED",
    "TURN_LEFT",
    "TURN_RIGHT",
    "TURN_LEFT_RIGHT_DELTA",
]

TURN_DELTA_MAX_DEG = 10.0

SHOOT_IDX = 0
MOVE_FORWARD_IDX = 1
MOVE_BACKWARD_IDX = 2
STRAFE_LEFT_IDX = 3
STRAFE_RIGHT_IDX = 4
SPEED_IDX = 5
TURN_LEFT_IDX = 6
TURN_RIGHT_IDX = 7
TURN_DELTA_IDX = 8


def action_to_vector(env_action: dict) -> np.ndarray:
    """Convert a VizDoom env_action dict to the 9-dim canonical vector."""
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    for i, key in enumerate(ACTION_KEYS):
        vec[i] = float(env_action.get(key, 0.0))
    return vec


def vector_to_action(
    vec: np.ndarray,
    binarize: bool = True,
    binary_threshold: float = 0.5,
) -> dict:
    """Convert a canonical vector to an action dictionary."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.size != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}-dim vector, got {vec.size}")
    if not np.isfinite(vec).all():
        raise ValueError("VizDoom action contains non-finite values.")
    env_action = copy.deepcopy(ENV_NULL_ACTION)
    for i, key in enumerate(ACTION_KEYS):
        val = float(vec[i])
        if i == TURN_DELTA_IDX:
            env_action[key] = float(np.clip(val, -1.0, 1.0))
        elif binarize:
            env_action[key] = 1.0 if val >= binary_threshold else 0.0
        else:
            env_action[key] = val
    return env_action


def vector_to_buttons(
    vec: np.ndarray,
    binary_threshold: float = 0.5,
) -> List[float]:
    """Convert a canonical vector to VizDoom engine-button values."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}-dim vector, got {v.size}")
    if not np.isfinite(v).all():
        raise ValueError("VizDoom action contains non-finite values.")
    buttons = [
        1.0 if float(v[i]) >= binary_threshold else 0.0
        for i in range(TURN_DELTA_IDX)
    ]
    buttons.append(
        float(np.clip(v[TURN_DELTA_IDX], -1.0, 1.0) * TURN_DELTA_MAX_DEG)
    )
    return buttons


def get_delta_mask() -> np.ndarray:
    """Return the mask for continuous delta dimensions."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[TURN_DELTA_IDX] = True
    return mask
