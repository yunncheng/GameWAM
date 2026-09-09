"""Proprio encoder construction and capability dispatch."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .minecraft_proprio import MinecraftCursorProprioTokenEncoder
from .proprio import (
    GenericMaskedProprioTokenEncoder,
    GenericProprioTokenEncoder,
)


def build_proprio_token_encoder(
    proprio_dim: int,
    hidden_dim: int,
    encoder_type: str = "generic_linear",
    field_valid_dim: Optional[int] = None,
) -> nn.Module:
    encoder_type = str(encoder_type)
    proprio_dim = int(proprio_dim)
    if encoder_type == "minecraft_grouped_cursor":
        expected_dim = MinecraftCursorProprioTokenEncoder.proprio_dim
        if proprio_dim != expected_dim:
            raise ValueError(
                "minecraft_grouped_cursor proprio encoder requires "
                f"proprio_dim={expected_dim}, got {proprio_dim}"
            )
        return MinecraftCursorProprioTokenEncoder(hidden_dim)
    if encoder_type == "generic_linear":
        return GenericProprioTokenEncoder(proprio_dim, hidden_dim)
    if encoder_type == "generic_masked_linear":
        if field_valid_dim is None:
            raise ValueError(
                "generic_masked_linear proprio encoder requires field_valid_dim"
            )
        return GenericMaskedProprioTokenEncoder(
            proprio_dim,
            int(field_valid_dim),
            hidden_dim,
        )
    raise ValueError(
        f"Unsupported proprio_encoder_type={encoder_type!r}; expected "
        "'generic_linear', 'generic_masked_linear', or "
        "'minecraft_grouped_cursor'."
    )


def is_field_valid_gated_proprio_encoder(module: nn.Module) -> bool:
    return isinstance(
        module,
        (MinecraftCursorProprioTokenEncoder, GenericMaskedProprioTokenEncoder),
    )
