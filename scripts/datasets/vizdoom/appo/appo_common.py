"""Shared contracts for Sample Factory APPO VizDoom collection.

The public Battle policies execute a rich action space: dropping
backward/strafe/speed/continuous-turn actions would make the video
transition ambiguous given the stored action.  This module therefore
defines one lossless, map-independent action vector for all four maps.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from gamewam.envs.vizdoom.observations.state import STATE_DIM, STATE_GAME_VARS


SCENARIO_TO_SF_ENV = {
    "battle1": "doom_battle",
    "battle2": "doom_battle2",
    "defend_the_center": "doom_defend_the_center",
    "defend_the_line": "doom_defend_the_line",
}

# Hugging Face repositories are deliberately data, not executable code.  The
# preparation script downloads files through snapshot_download and opens the
# checkpoints with torch.load(weights_only=True).
MODEL_REPOS = {
    "battle1": "andrewzhang505/sample-factory-2-doom-battle",
    "battle2": "andrewzhang505/sample-factory-2-doom-battle2",
    "defend_the_center": "edbeeching/doom_defend_the_center_2222",
    "defend_the_line": "execbat/rl_course_vizdoom_doom_defend_the_line",
}

# Canonical GameWAM action vector. turn_delta is normalized to [-1, 1], where
# +/-1 corresponds to the Battle policy's +/-10 degree VizDoom delta action.
ACTION_KEYS = (
    "shoot",
    "move_forward",
    "move_backward",
    "strafe_left",
    "strafe_right",
    "speed",
    "turn_left",
    "turn_right",
    "turn_delta",
)
ACTION_DIM = len(ACTION_KEYS)

STATE_KEYS = STATE_GAME_VARS


def canonical_action_from_flat(scenario: str, flat: Sequence[float]) -> np.ndarray:
    """Map the *actual* VizDoom button vector to the canonical 9-D action.

    Sample Factory's ``VizdoomEnv._convert_actions`` is the source of ``flat``.
    Battle order is ``FWD, BACK, RIGHT, LEFT, ATTACK, SPEED, TURN_DELTA``;
    defend order is ``TURN_LEFT, TURN_RIGHT, ATTACK``.
    """

    values = np.asarray(flat, dtype=np.float32).reshape(-1)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    if scenario in ("battle1", "battle2"):
        if values.size != 7:
            raise ValueError(f"{scenario} expected 7 flattened actions, got {values.size}")
        action[0] = values[4]
        action[1] = values[0]
        action[2] = values[1]
        action[3] = values[3]
        action[4] = values[2]
        action[5] = values[5]
        action[8] = np.clip(values[6] / 10.0, -1.0, 1.0)
    elif scenario in ("defend_the_center", "defend_the_line"):
        if values.size != 3:
            raise ValueError(f"{scenario} expected 3 flattened actions, got {values.size}")
        action[0] = values[2]
        action[6] = values[0]
        action[7] = values[1]
    else:
        raise KeyError(f"Unknown scenario {scenario!r}")
    return action


def humanize_flat_action(
    scenario: str,
    flat: Sequence[float],
    *,
    turn_scale: float = 0.5,
    keep_speed: bool = False,
    repeat_index: int = 0,
) -> np.ndarray:
    """Apply a transparent action governor to an expert action.

    This is not imitation or human-recorded data. The APPO policy still
    chooses the action, while the governor limits execution speed. The
    returned actual VizDoom action must be used both for env.step and for
    the stored GameWAM label.

    Battle maps expose continuous TURN_DELTA and a SPEED button, so we scale
    the former and optionally suppress the latter. Defend maps expose binary
    turn buttons; fractional speed is implemented with deterministic pulse
    density across the four repeated engine ticks.
    """

    if not 0.0 < turn_scale <= 1.0:
        raise ValueError(f"turn_scale must be in (0, 1], got {turn_scale}")
    if repeat_index < 0:
        raise ValueError(f"repeat_index must be non-negative, got {repeat_index}")

    values = np.asarray(flat, dtype=np.float32).reshape(-1).copy()
    if scenario in ("battle1", "battle2"):
        if values.size != 7:
            raise ValueError(f"{scenario} expected 7 flattened actions, got {values.size}")
        if not keep_speed:
            values[5] = 0.0
        values[6] *= float(turn_scale)
    elif scenario in ("defend_the_center", "defend_the_line"):
        if values.size != 3:
            raise ValueError(f"{scenario} expected 3 flattened actions, got {values.size}")
        pulse_on = np.floor((repeat_index + 1) * turn_scale) > np.floor(
            repeat_index * turn_scale
        )
        if not pulse_on:
            values[0] = 0.0
            values[1] = 0.0
    else:
        raise KeyError(f"Unknown scenario {scenario!r}")
    return values


def state_from_info(info: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical proprio and its per-dimension padding mask.

    Defend scenarios do not expose pose variables.  Those values remain zero
    and are marked padded rather than being presented to GameWAM as real zeros.
    """

    state = np.zeros(STATE_DIM, dtype=np.float32)
    is_pad = np.ones(STATE_DIM, dtype=bool)
    for idx, key in enumerate(STATE_KEYS):
        value = info.get(key)
        if value is not None:
            state[idx] = float(value)
            is_pad[idx] = False
    return state, is_pad


def trajectory_quality(
    actions: np.ndarray,
    states: np.ndarray,
    state_is_pad: np.ndarray,
) -> dict[str, float]:
    """Compute policy-agnostic motion quality diagnostics for one trajectory."""

    a = np.asarray(actions, dtype=np.float32)
    s = np.asarray(states, dtype=np.float32)
    pad = np.asarray(state_is_pad, dtype=bool)
    if a.ndim != 2 or a.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions [T,{ACTION_DIM}], got {a.shape}")
    steps = max(1, len(a))
    active = np.any(np.abs(a) > 1e-6, axis=1)

    turn = a[:, 7] - a[:, 6] + a[:, 8]
    turn_sign = np.where(np.abs(turn) >= 0.8, np.sign(turn), 0.0)
    consecutive_turn = (turn_sign[1:] != 0) & (turn_sign[:-1] != 0)
    flip_count = np.count_nonzero(consecutive_turn & (turn_sign[1:] != turn_sign[:-1]))
    turn_pairs = max(1, int(np.count_nonzero(consecutive_turn)))

    forward = a[:-1, 1] > 0.5
    pose_valid = ~(pad[:-1, 0] | pad[:-1, 1] | pad[1:, 0] | pad[1:, 1])
    displacement = np.linalg.norm(s[1:, :2] - s[:-1, :2], axis=1) if len(s) > 1 else np.empty(0)
    forward_valid = forward & pose_valid if len(a) > 1 else np.empty(0, dtype=bool)
    stuck = forward_valid & (displacement < 1.0) if len(displacement) else np.empty(0, dtype=bool)

    valid_pose = ~pad[:, 0] & ~pad[:, 1]
    if np.any(valid_pose):
        cells = np.floor(s[valid_pose, :2] / 64.0).astype(np.int64)
        coverage = float(len(np.unique(cells, axis=0)))
    else:
        coverage = 0.0

    return {
        "idle_ratio": float(1.0 - np.count_nonzero(active) / steps),
        "turn_flip_ratio": float(flip_count / turn_pairs),
        "forward_stuck_ratio": float(np.count_nonzero(stuck) / max(1, np.count_nonzero(forward_valid))),
        "shoot_ratio": float(np.count_nonzero(a[:, 0] > 0.5) / steps),
        "position_cells": coverage,
    }


def passes_quality_gate(
    metrics: Mapping[str, float], scenario: str | None = None
) -> bool:
    """Conservative, scenario-aware gate for known collection failures."""

    # In stationary defense, rapid direction changes are legitimate target
    # tracking. Maze navigation gets a stricter threshold for visible U-turns.
    flip_limit = 0.75 if scenario and scenario.startswith("defend_the_") else 0.35
    return (
        metrics["idle_ratio"] <= 0.30
        and metrics["turn_flip_ratio"] <= flip_limit
        and metrics["forward_stuck_ratio"] <= 0.05
    )

