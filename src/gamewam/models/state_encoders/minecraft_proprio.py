"""Minecraft-specific proprio token encoder."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from gamewam.envs.minecraft.observations.state import (
    CANONICAL_CURSOR_SLICE,
    CANONICAL_GUI_SLICE,
    CANONICAL_HOTBAR_SLICE,
    CANONICAL_PITCH_SLICE,
    CANONICAL_STATE_DIM,
    CANONICAL_YAW_SLICE,
    CURSOR_X_IDX,
    CURSOR_Y_IDX,
    FIELD_VALID_DIM,
    GUI_OPEN_IDX,
    HOTBAR_IDX,
    PITCH_IDX,
    YAW_IDX,
)

from .proprio import proprio_time_embedding


class _MinecraftFieldTokenEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pitch_proj = nn.Linear(1, self.hidden_dim)
        self.yaw_proj = nn.Linear(2, self.hidden_dim)
        self.hotbar_proj = nn.Linear(9, self.hidden_dim)
        self.gui_proj = nn.Linear(1, self.hidden_dim)
        self.shared_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.role_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.norm = nn.LayerNorm(self.hidden_dim)
        nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)


class MinecraftCursorProprioTokenEncoder(_MinecraftFieldTokenEncoder):
    """Encode Minecraft cursor canonical [15] state with atomic 6-D validity."""

    proprio_dim = CANONICAL_STATE_DIM
    field_valid_dim = FIELD_VALID_DIM

    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.cursor_proj = nn.Linear(2, self.hidden_dim)

    def forward(
        self,
        proprio: torch.Tensor,
        field_valid: torch.Tensor,
        is_pad: Optional[torch.Tensor] = None,
        raw_step_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                "canonical Minecraft cursor proprio must be [B,K,15], got "
                f"{tuple(proprio.shape)}"
            )
        expected_valid_shape = proprio.shape[:2] + (self.field_valid_dim,)
        if field_valid.shape != expected_valid_shape:
            raise ValueError(
                "Minecraft cursor proprio_field_valid must be [B,K,6], got "
                f"{tuple(field_valid.shape)} for proprio {tuple(proprio.shape)}"
            )
        proprio = proprio.to(dtype=self.pitch_proj.weight.dtype)
        valid_bool = field_valid.to(device=proprio.device, dtype=torch.bool)
        cursor_validity_is_atomic = ~(
            valid_bool[..., CURSOR_X_IDX]
            != valid_bool[..., CURSOR_Y_IDX]
        ).any()
        if valid_bool.is_cuda:
            torch._assert_async(
                cursor_validity_is_atomic,
                "cursor_x/cursor_y validity must be an atomic pair.",
            )
        elif not bool(cursor_validity_is_atomic):
            raise ValueError("cursor_x/cursor_y validity must be an atomic pair.")
        valid = valid_bool.to(dtype=proprio.dtype)
        cursor_valid = (
            valid_bool[..., CURSOR_X_IDX : CURSOR_X_IDX + 1]
            & valid_bool[..., CURSOR_Y_IDX : CURSOR_Y_IDX + 1]
        )

        pitch = self.pitch_proj(proprio[..., CANONICAL_PITCH_SLICE]) * valid[
            ..., PITCH_IDX : PITCH_IDX + 1
        ]
        yaw = self.yaw_proj(proprio[..., CANONICAL_YAW_SLICE]) * valid[
            ..., YAW_IDX : YAW_IDX + 1
        ]
        cursor = self.cursor_proj(proprio[..., CANONICAL_CURSOR_SLICE]) * cursor_valid.to(
            dtype=proprio.dtype
        )
        hotbar = self.hotbar_proj(proprio[..., CANONICAL_HOTBAR_SLICE]) * valid[
            ..., HOTBAR_IDX : HOTBAR_IDX + 1
        ]
        gui = self.gui_proj(proprio[..., CANONICAL_GUI_SLICE]) * valid[
            ..., GUI_OPEN_IDX : GUI_OPEN_IDX + 1
        ]
        field_sum = pitch + yaw + hotbar + gui + cursor

        time_emb = proprio_time_embedding(
            self.hidden_dim, proprio, raw_step_positions
        )
        tokens = self.norm(
            field_sum + self.shared_mlp(field_sum) + self.role_embedding + time_emb
        )
        token_valid = valid_bool.any(dim=-1)
        if is_pad is not None:
            if is_pad.shape != proprio.shape[:2]:
                raise ValueError(
                    "proprio_is_pad must be [B,K], got "
                    f"{tuple(is_pad.shape)} for proprio {tuple(proprio.shape)}"
                )
            token_valid = token_valid & ~is_pad.to(
                device=proprio.device, dtype=torch.bool
            )
        tokens = tokens * token_valid.to(dtype=tokens.dtype).unsqueeze(-1)
        return tokens, token_valid
