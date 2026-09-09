"""Domain-agnostic proprio token encoders."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..wan22.wan_video_dit import sinusoidal_embedding_1d


def proprio_time_embedding(
    hidden_dim: int,
    proprio: torch.Tensor,
    raw_step_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    batch_size, token_count = proprio.shape[:2]
    if raw_step_positions is None:
        raise ValueError(
            "raw_step_positions is required for gated proprio time encoding."
        )
    positions = raw_step_positions.to(
        device=proprio.device, dtype=proprio.dtype
    )
    if positions.ndim == 1:
        if positions.shape != (token_count,):
            raise ValueError(
                "raw_step_positions must be [K] matching proprio, got "
                f"{tuple(positions.shape)} for K={token_count}"
            )
        flat = positions
        leading_shape = (1, token_count)
    elif positions.ndim == 2:
        if positions.shape != (batch_size, token_count):
            raise ValueError(
                "raw_step_positions must be [B,K] matching proprio, got "
                f"{tuple(positions.shape)} vs {(batch_size, token_count)}"
            )
        flat = positions.reshape(-1)
        leading_shape = (batch_size, token_count)
    else:
        raise ValueError(
            "raw_step_positions must be [K] or [B,K], got "
            f"{tuple(positions.shape)}"
        )
    embedding = sinusoidal_embedding_1d(hidden_dim, flat).reshape(
        *leading_shape, hidden_dim
    )
    return embedding.to(dtype=proprio.dtype)


class GenericProprioTokenEncoder(nn.Module):
    """Encode generic proprio vectors with a linear projection."""

    def __init__(self, proprio_dim: int, hidden_dim: int):
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(self.proprio_dim, self.hidden_dim)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"generic proprio must be [B,K,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        return self.proj(proprio.to(dtype=self.proj.weight.dtype))


class GenericMaskedProprioTokenEncoder(nn.Module):
    """Encode canonical state with an independent field-validity schema."""

    def __init__(self, proprio_dim: int, field_valid_dim: int, hidden_dim: int):
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.field_valid_dim = int(field_valid_dim)
        self.hidden_dim = int(hidden_dim)
        if self.field_valid_dim <= 0:
            raise ValueError(
                f"field_valid_dim must be positive, got {self.field_valid_dim}"
            )
        self.proj = nn.Linear(
            self.proprio_dim + self.field_valid_dim,
            self.hidden_dim,
        )
        self.role_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.norm = nn.LayerNorm(self.hidden_dim)
        nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        proprio: torch.Tensor,
        field_valid: torch.Tensor,
        is_pad: Optional[torch.Tensor] = None,
        raw_step_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"generic masked proprio must be [B,K,{self.proprio_dim}], "
                f"got {tuple(proprio.shape)}"
            )
        expected_valid_shape = proprio.shape[:2] + (self.field_valid_dim,)
        if field_valid.shape != expected_valid_shape:
            raise ValueError(
                "generic masked proprio field_valid must be "
                f"{expected_valid_shape}, got {tuple(field_valid.shape)}"
            )
        proprio = proprio.to(dtype=self.proj.weight.dtype)
        valid_bool = field_valid.to(device=proprio.device, dtype=torch.bool)
        features = torch.cat(
            [proprio, valid_bool.to(dtype=proprio.dtype)], dim=-1
        )
        time_emb = proprio_time_embedding(
            self.hidden_dim, proprio, raw_step_positions
        )
        tokens = self.norm(self.proj(features) + self.role_embedding + time_emb)
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
