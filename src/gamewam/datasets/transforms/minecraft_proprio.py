"""Minecraft cursor-state transforms."""
from __future__ import annotations

from typing import Dict

import torch

from gamewam.envs.minecraft.observations.state import (
    CANONICAL_STATE_DIM,
    CURSOR_X_IDX,
    CURSOR_Y_IDX,
    FIELD_VALID_DIM as CURSOR_FIELD_VALID_DIM,
    GUI_OPEN_IDX,
    HOTBAR_IDX,
    PITCH_IDX,
    RAW_STATE_DIM,
    YAW_IDX,
    apply_gui_semantic_validity,
)

class MinecraftCursorProprioTransform:
    """Convert reordered Minecraft cursor raw [6] state into canonical [15] features.

    Cursor normalization belongs to the model input transform, not the data
    producer. Raw datasets retain continuous reference-pixel coordinates.
    """

    RAW_DIM = RAW_STATE_DIM
    FIELD_VALID_DIM = CURSOR_FIELD_VALID_DIM
    CANONICAL_DIM = CANONICAL_STATE_DIM

    def __init__(self, key: str = "default", cursor_reference_size=None):
        self.key = str(key)
        if cursor_reference_size is None or len(cursor_reference_size) != 2:
            raise ValueError(
                "MinecraftCursorProprioTransform requires "
                "cursor_reference_size=[width, height]."
            )
        self.reference_width = float(cursor_reference_size[0])
        self.reference_height = float(cursor_reference_size[1])
        if not (self.reference_width > 0.0 and self.reference_height > 0.0):
            raise ValueError(
                "cursor_reference_size values must be positive, got "
                f"{cursor_reference_size!r}"
            )

    @staticmethod
    def effective_field_validity(
        raw_state: torch.Tensor,
        field_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.as_tensor(
            field_valid, dtype=torch.bool, device=raw_state.device
        )
        return apply_gui_semantic_validity(raw_state, valid)

    @staticmethod
    def transform_state_and_validity(
        raw_state: torch.Tensor,
        *,
        cursor_reference_size,
        field_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_state.ndim == 0:
            raise ValueError("Minecraft cursor raw state must have rank >= 1")
        if raw_state.shape[-1] != MinecraftCursorProprioTransform.RAW_DIM:
            raise ValueError(
                "Minecraft cursor raw state must have last dim "
                f"{MinecraftCursorProprioTransform.RAW_DIM}, got "
                f"{tuple(raw_state.shape)}"
            )
        if cursor_reference_size is None or len(cursor_reference_size) != 2:
            raise ValueError("cursor_reference_size=[width, height] is required.")
        width = float(cursor_reference_size[0])
        height = float(cursor_reference_size[1])
        if not (width > 0.0 and height > 0.0):
            raise ValueError(
                f"cursor_reference_size must be positive, got {(width, height)!r}"
            )

        effective_valid = MinecraftCursorProprioTransform.effective_field_validity(
            raw_state, field_valid
        )
        state = raw_state.float()
        pitch = torch.clamp(
            state[..., PITCH_IDX : PITCH_IDX + 1], -90.0, 90.0
        ) / 90.0
        yaw_rad = state[..., YAW_IDX : YAW_IDX + 1] * torch.pi / 180.0
        yaw = torch.cat([torch.sin(yaw_rad), torch.cos(yaw_rad)], dim=-1)
        hotbar_idx = torch.clamp(
            state[..., HOTBAR_IDX].round().long(), 0, 8
        )
        hotbar = torch.nn.functional.one_hot(
            hotbar_idx, num_classes=9
        ).to(dtype=state.dtype)
        gui = state[..., GUI_OPEN_IDX : GUI_OPEN_IDX + 1].clamp(0.0, 1.0)
        cursor = torch.stack(
            [
                2.0 * state[..., CURSOR_X_IDX] / width - 1.0,
                2.0 * state[..., CURSOR_Y_IDX] / height - 1.0,
            ],
            dim=-1,
        ).clamp(-1.0, 1.0)

        pitch = torch.where(
            effective_valid[..., PITCH_IDX : PITCH_IDX + 1],
            pitch,
            torch.zeros_like(pitch),
        )
        yaw = torch.where(
            effective_valid[..., YAW_IDX : YAW_IDX + 1],
            yaw,
            torch.zeros_like(yaw),
        )
        cursor = torch.where(
            effective_valid[..., CURSOR_X_IDX : CURSOR_X_IDX + 1],
            cursor,
            torch.zeros_like(cursor),
        )
        hotbar = torch.where(
            effective_valid[..., HOTBAR_IDX : HOTBAR_IDX + 1],
            hotbar,
            torch.zeros_like(hotbar),
        )
        gui = torch.where(
            effective_valid[..., GUI_OPEN_IDX : GUI_OPEN_IDX + 1],
            gui,
            torch.zeros_like(gui),
        )
        canonical = torch.cat([pitch, yaw, cursor, hotbar, gui], dim=-1)
        return canonical, effective_valid

    @staticmethod
    def transform_state(
        raw_state: torch.Tensor,
        *,
        cursor_reference_size,
        field_valid: torch.Tensor,
    ) -> torch.Tensor:
        canonical, _ = MinecraftCursorProprioTransform.transform_state_and_validity(
            raw_state,
            cursor_reference_size=cursor_reference_size,
            field_valid=field_valid,
        )
        return canonical

    @staticmethod
    def transform_dim_is_pad(raw_mask: torch.Tensor) -> torch.Tensor:
        if raw_mask.shape[-1] == MinecraftCursorProprioTransform.CANONICAL_DIM:
            return raw_mask
        if raw_mask.shape[-1] != MinecraftCursorProprioTransform.RAW_DIM:
            raise ValueError(
                "Minecraft cursor state_dim_is_pad must have last dim 6 or "
                f"15, got {tuple(raw_mask.shape)}"
            )
        mask = raw_mask.bool()
        pitch = mask[..., PITCH_IDX : PITCH_IDX + 1]
        yaw = mask[..., YAW_IDX : YAW_IDX + 1].expand(*mask.shape[:-1], 2)
        cursor = mask[..., CURSOR_X_IDX : CURSOR_Y_IDX + 1]
        hotbar = mask[..., HOTBAR_IDX : HOTBAR_IDX + 1].expand(
            *mask.shape[:-1], 9
        )
        gui = mask[..., GUI_OPEN_IDX : GUI_OPEN_IDX + 1]
        return torch.cat([pitch, yaw, cursor, hotbar, gui], dim=-1)

    def _field_valid(self, batch: Dict, raw_state: torch.Tensor) -> torch.Tensor:
        valid = batch.get("state_field_valid")
        if isinstance(valid, dict):
            valid = valid.get(self.key)
        if valid is None:
            raise ValueError(
                "Minecraft cursor data requires observation.state_field_valid; "
                "an all-valid fallback is forbidden."
            )
        return torch.as_tensor(valid, dtype=torch.bool, device=raw_state.device)

    def forward(self, batch: Dict) -> Dict:
        state = batch.get("state")
        if state is None:
            return batch
        if isinstance(state, dict):
            if self.key not in state:
                return batch
            raw_state = state[self.key]
        else:
            raw_state = state

        canonical, effective_valid = self.transform_state_and_validity(
            raw_state,
            cursor_reference_size=(self.reference_width, self.reference_height),
            field_valid=self._field_valid(batch, raw_state),
        )
        if isinstance(state, dict):
            state[self.key] = canonical
        else:
            batch["state"] = canonical

        field_valid = batch["state_field_valid"]
        if isinstance(field_valid, dict):
            field_valid[self.key] = effective_valid
        else:
            batch["state_field_valid"] = effective_valid

        if "state_dim_is_pad" in batch and batch["state_dim_is_pad"] is not None:
            batch["state_dim_is_pad"] = self.transform_dim_is_pad(
                batch["state_dim_is_pad"]
            )
        return batch

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
        return batch
