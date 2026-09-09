"""State-before-action tracking for Minecraft trajectory conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from gamewam.envs.minecraft.action_space import switched_hotbar_slot
from gamewam.envs.minecraft.observations.state import (
    FIELD_NAMES,
    FIELD_VALID_DIM,
    GUI_OPEN_IDX,
    HOTBAR_IDX,
    RAW_STATE_DIM,
)


@dataclass
class MinecraftStateStats:
    observed: np.ndarray = field(
        default_factory=lambda: np.zeros(RAW_STATE_DIM, dtype=np.int64)
    )
    derived: np.ndarray = field(
        default_factory=lambda: np.zeros(RAW_STATE_DIM, dtype=np.int64)
    )
    missing: np.ndarray = field(
        default_factory=lambda: np.zeros(RAW_STATE_DIM, dtype=np.int64)
    )
    invalid: np.ndarray = field(
        default_factory=lambda: np.zeros(RAW_STATE_DIM, dtype=np.int64)
    )
    all_missing: int = 0
    hotbar_switches: int = 0
    hotbar_conflicts: int = 0

    def update(
        self,
        valid: np.ndarray,
        derived_mask: Optional[np.ndarray] = None,
    ) -> None:
        valid = np.asarray(valid, dtype=np.bool_)
        if valid.shape != (FIELD_VALID_DIM,):
            raise ValueError(
                "Minecraft state validity must have shape "
                f"{(FIELD_VALID_DIM,)}, got {valid.shape}"
            )
        if derived_mask is None:
            derived_mask = np.zeros(FIELD_VALID_DIM, dtype=np.bool_)
        else:
            derived_mask = np.asarray(derived_mask, dtype=np.bool_)
            if derived_mask.shape != (FIELD_VALID_DIM,):
                raise ValueError(
                    "Minecraft derived mask must have shape "
                    f"{(FIELD_VALID_DIM,)}, got {derived_mask.shape}"
                )
        self.derived += (valid & derived_mask).astype(np.int64)
        self.observed += (valid & ~derived_mask).astype(np.int64)
        self.missing += (~valid).astype(np.int64)
        if not bool(valid.any()):
            self.all_missing += 1

    def as_dict(self) -> dict[str, Any]:
        total = self.observed + self.derived + self.missing
        fields = {}
        for index, name in enumerate(FIELD_NAMES):
            denominator = max(int(total[index]), 1)
            fields[name] = {
                "observed": int(self.observed[index]),
                "observed_ratio": float(self.observed[index]) / denominator,
                "derived": int(self.derived[index]),
                "derived_ratio": float(self.derived[index]) / denominator,
                "missing": int(self.missing[index]),
                "missing_ratio": float(self.missing[index]) / denominator,
                "invalid": int(self.invalid[index]),
            }
        return {
            "fields": fields,
            "all_fields_missing": int(self.all_missing),
            "hotbar_switches": int(self.hotbar_switches),
            "hotbar_conflicts": int(self.hotbar_conflicts),
        }


class MinecraftStateTracker:
    """Track state-before-action semantics for VPT-style trajectories."""

    def __init__(self) -> None:
        self.known_hotbar: Optional[int] = None
        self.known_gui: Optional[bool] = None
        self.stats = MinecraftStateStats()

    def state_before_action(
        self,
        observed_values: np.ndarray,
        observed_valid: np.ndarray,
        env_action: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(observed_values, dtype=np.float32).copy()
        valid = np.asarray(observed_valid, dtype=np.bool_).copy()
        if values.shape != (RAW_STATE_DIM,) or valid.shape != (FIELD_VALID_DIM,):
            raise ValueError(
                "Minecraft tracker expects values/validity shapes "
                f"{(RAW_STATE_DIM,)}/{(FIELD_VALID_DIM,)}, "
                f"got {values.shape}/{valid.shape}"
            )
        derived = np.zeros(FIELD_VALID_DIM, dtype=np.bool_)

        if self.known_hotbar is not None:
            values[HOTBAR_IDX] = float(self.known_hotbar)
            valid[HOTBAR_IDX] = True
            derived[HOTBAR_IDX] = (
                not bool(observed_valid[HOTBAR_IDX])
                or int(observed_values[HOTBAR_IDX]) != self.known_hotbar
            )
        else:
            values[HOTBAR_IDX] = 0.0
            valid[HOTBAR_IDX] = False

        if not valid[GUI_OPEN_IDX]:
            values[GUI_OPEN_IDX] = 0.0

        self.stats.update(valid, derived)
        return values, valid

    def commit_after_action(
        self,
        observed_values: np.ndarray,
        observed_valid: np.ndarray,
        env_action: Dict[str, Any],
    ) -> None:
        switch_slot = switched_hotbar_slot(env_action)
        if switch_slot is not None:
            self.stats.hotbar_switches += 1

        if bool(observed_valid[HOTBAR_IDX]):
            recorded = int(observed_values[HOTBAR_IDX])
            if switch_slot is not None and recorded != switch_slot:
                self.stats.hotbar_conflicts += 1
            self.known_hotbar = recorded
        elif switch_slot is not None:
            self.known_hotbar = int(switch_slot)

        if bool(observed_valid[GUI_OPEN_IDX]):
            self.known_gui = bool(observed_values[GUI_OPEN_IDX])

    def get_state_before_action(
        self,
        observed_values: np.ndarray,
        observed_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.state_before_action(observed_values, observed_valid)
