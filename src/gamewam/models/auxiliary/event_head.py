"""Auxiliary event-occurrence classifier head and helpers.

Owns the timestep-level event-effect classifier used as an auxiliary
training objective. An independent module from :mod:`gamewam`: the head
consumes post-MoT action hidden states and exposes a two-class CE
loss contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Ignore sentinel for ``event_occurrence_label`` entries that should be
#: excluded from the auxiliary CE loss (no event metadata, padded
#: blocks, etc.).
EVENT_OCCURRENCE_IGNORE_INDEX: int = -100


class EventOccurrenceHead(nn.Module):
    """Timestep-level event-effect head with MIL window aggregation.

    Inputs are post-MoT action hidden states. The head first predicts one
    scalar event-effect logit per action timestep, then aggregates over the
    timestep axis with ``logsumexp`` to obtain a window-level positive logit.
    The returned two-class logits follow the CE loss contract:

    - ``[B, D]`` -> ``[B, 2]``
    - ``[B, T, D]`` -> ``[B, 2]``
    - ``[B, K, T, D]`` -> ``[B, K, 2]``
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)

    def forward_timestep_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim not in (2, 3, 4):
            raise ValueError(
                f"EventOccurrenceHead expects 2D/3D/4D hidden, got shape {tuple(hidden.shape)}"
            )
        hidden = hidden.to(dtype=self.fc.weight.dtype)
        return self.fc(self.norm(hidden)).squeeze(-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        timestep_logits = self.forward_timestep_logits(hidden)
        if timestep_logits.ndim == 1:
            window_logits = timestep_logits
        elif timestep_logits.ndim == 2:
            window_logits = torch.logsumexp(timestep_logits, dim=1)
        else:
            window_logits = torch.logsumexp(timestep_logits, dim=2)
        negative_logits = torch.zeros_like(window_logits)
        return torch.stack([negative_logits, window_logits], dim=-1)


def event_occurrence_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    compute_metrics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """CE loss + diagnostics for the event-occurrence head.

    ``logits``: ``[B, 2]`` (non-AR) or ``[B, num_blocks, 2]`` (AR).
    ``labels``: matching leading dims; ``EVENT_OCCURRENCE_IGNORE_INDEX``
    entries are masked out. Returns
    ``(loss, {valid_ratio, pos_ratio})``.
    """
    flat_logits = logits.reshape(-1, 2).float()
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != EVENT_OCCURRENCE_IGNORE_INDEX
    per_entry = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=EVENT_OCCURRENCE_IGNORE_INDEX,
        reduction="none",
    )
    num_valid_tensor = valid_mask.sum()
    loss = per_entry.sum() / num_valid_tensor.clamp_min(1).to(per_entry.dtype)
    if not compute_metrics:
        return loss, {}

    with torch.no_grad():
        num_pos_tensor = ((flat_labels == 1) & valid_mask).sum()
        num_valid, num_pos = torch.stack(
            [num_valid_tensor, num_pos_tensor]
        ).detach().cpu().tolist()
    num_valid = int(num_valid)
    num_pos = int(num_pos)
    return loss, {
        "valid_ratio": num_valid / max(1, int(flat_labels.numel())),
        "pos_ratio": num_pos / max(1, num_valid),
    }


__all__ = [
    "EVENT_OCCURRENCE_IGNORE_INDEX",
    "EventOccurrenceHead",
    "event_occurrence_ce_loss",
]
