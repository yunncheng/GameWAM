"""Canonical VizDoom proprio state contract."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


STATE_GAME_VARS = ("POSITION_X", "POSITION_Y", "ANGLE", "HEALTH")
STATE_DIM = len(STATE_GAME_VARS)


def extract_state(
    info: Mapping[str, Any], state_dim: int = STATE_DIM
) -> np.ndarray:
    if int(state_dim) != STATE_DIM:
        raise ValueError(
            f"VizDoom state must have {STATE_DIM} dimensions, got {state_dim}."
        )
    game_vars = info.get("game_vars") or {}
    return np.asarray(
        [float(game_vars.get(name, 0.0)) for name in STATE_GAME_VARS],
        dtype=np.float32,
    )
