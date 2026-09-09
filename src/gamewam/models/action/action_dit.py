import os
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from gamewam.utils.logging_config import get_logger

from ..wan22.helpers.gradient import gradient_checkpoint_forward
from ..wan22.wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class ActionHeads(nn.Module):
    """Joint-flow velocity head for either the gameplay or GUI route.

    ``continuous_head`` is a flow-matching velocity head
    (``nn.Linear(hidden_dim, continuous_dim)``) covering the full action
    vector. When ``continuous_dim`` is 0 the head is ``None``; callers must
    handle the ``None`` case.
    """

    def __init__(self, hidden_dim: int, continuous_dim: int):
        super().__init__()
        self.continuous_dim = int(continuous_dim)
        self.continuous_head = (
            nn.Linear(hidden_dim, continuous_dim) if continuous_dim > 0 else None
        )


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "action_heads.",
        "gui_route_head.",
    )
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        dim_is_continuous: Optional[Sequence[bool]] = None,
        gui_head_enabled: bool = False,
        gui_label_dim_index: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        # Semantic action schema. The joint flow still needs this layout for
        # grouped losses, diagnostics, and environment decoding.
        if dim_is_continuous is None:
            self.dim_is_continuous: List[bool] = [True] * int(action_dim)
            self.continuous_dims: List[int] = list(range(int(action_dim)))
            self.discrete_dims: List[int] = []
            self.continuous_action_dim = int(action_dim)
            self.discrete_action_dim = 0
        else:
            flags = list(dim_is_continuous)
            if len(flags) != int(action_dim):
                raise ValueError(
                    f"`dim_is_continuous` length must equal `action_dim` "
                    f"({int(action_dim)}), got {len(flags)}."
                )
            self.dim_is_continuous = [bool(x) for x in flags]
            self.continuous_dims = [i for i, c in enumerate(flags) if c]
            self.discrete_dims = [i for i, c in enumerate(flags) if not c]
            self.continuous_action_dim = len(self.continuous_dims)
            self.discrete_action_dim = len(self.discrete_dims)
            if self.continuous_action_dim == 0:
                raise ValueError(
                    "`dim_is_continuous` must mark at least one dim as continuous; "
                    "otherwise the flow-matching input is empty."
                )

        self.gui_head_enabled = bool(gui_head_enabled)
        self.gui_label_dim_index = gui_label_dim_index

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.action_heads = nn.ModuleDict({
            "gameplay": ActionHeads(
                hidden_dim=hidden_dim,
                continuous_dim=action_dim,
            ),
        })
        if self.gui_head_enabled:
            self.action_heads["gui"] = ActionHeads(
                hidden_dim=hidden_dim,
                continuous_dim=action_dim,
            )
            self.gui_route_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 4),
                nn.Linear(hidden_dim // 4, 1),
            )
        else:
            self.gui_route_head = None

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_auxiliary_sample_keys = (
            "target_discrete_action",
            "action_route_before_action",
            "action_route_valid",
        )

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(
                f"`meta` must be a dict in {action_dit_pretrained_path}, got {type(meta)}"
            )
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def encode_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B,L,D], got shape {tuple(context.shape)}")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        elif context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`context_mask` must be [B,L] for encode_context, got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
            )
        return self.text_embedding(context), context_mask.bool()

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        context_embedding: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        expected_dim = self.action_dim
        if action_tokens.shape[2] != expected_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {expected_dim} (action_dim), "
                f"got {action_tokens.shape[2]}"
            )
        if timestep.ndim not in (1, 2):
            raise ValueError(
                f"`timestep` must be 1D [B]/[1] or 2D [B,T], got shape {tuple(timestep.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if context_embedding is not None and context_embedding.shape != (
            batch_size,
            context.shape[1],
            self.hidden_dim,
        ):
            raise ValueError(
                "context_embedding must be [B,L,hidden_dim], got "
                f"{tuple(context_embedding.shape)} vs {(batch_size, context.shape[1], self.hidden_dim)}"
            )
        if timestep.ndim == 1:
            if timestep.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
                )
            if timestep.shape[0] == 1 and batch_size > 1:
                if self.training:
                    raise ValueError("During training, action timestep length must match batch_size.")
                timestep = timestep.expand(batch_size)
        else:
            if timestep.shape[0] != batch_size or timestep.shape[1] != action_tokens.shape[1]:
                raise ValueError(
                    "`timestep` 2D shape must match action tokens [B,T], "
                    f"got {tuple(timestep.shape)} vs {(batch_size, action_tokens.shape[1])}"
                )

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim not in (2, 3):
                raise ValueError(
                    f"`context_mask` must be 2D [B, L] or 3D [B, T, L], got shape {tuple(context_mask.shape)}"
                )
            if context_mask.shape[0] != batch_size or context_mask.shape[-1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match context batch/length, got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if position_ids is not None:
            if position_offset != 0:
                raise ValueError(
                    "`position_ids` contains absolute Action RoPE positions and "
                    "cannot be combined with a non-zero `position_offset`."
                )
            if position_ids.ndim != 1 or position_ids.shape[0] != seq_len:
                raise ValueError(
                    f"`position_ids` must be 1D with length {seq_len}, got "
                    f"shape {tuple(position_ids.shape)}"
                )
            if position_ids.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError(
                    f"`position_ids` must have an integer dtype, got {position_ids.dtype}"
                )
            rope_position_ids = position_ids.to(
                device=self.freqs.device,
                dtype=torch.long,
            )
            if seq_len > 0:
                positions_in_bounds = (
                    (rope_position_ids >= 0)
                    & (rope_position_ids < self.freqs.shape[0])
                ).all()
                if rope_position_ids.is_cuda:
                    torch._assert_async(
                        positions_in_bounds,
                        "Action RoPE position_ids exceed the cache bounds.",
                    )
                elif not bool(positions_in_bounds):
                    min_position = int(rope_position_ids.min().item())
                    max_position = int(rope_position_ids.max().item())
                    raise ValueError(
                        "Action RoPE position_ids must be within cache bounds "
                        f"[0, {self.freqs.shape[0]}), got min={min_position}, "
                        f"max={max_position}"
                    )
            selected_freqs = self.freqs.index_select(0, rope_position_ids)
        else:
            if position_offset < 0:
                raise ValueError(
                    f"`position_offset` must be non-negative, got {position_offset}"
                )
            end = position_offset + seq_len
            if end > self.freqs.shape[0]:
                raise ValueError(
                    f"Action RoPE range [{position_offset}, {end}) exceeds cache {self.freqs.shape[0]}"
                )
            rope_position_ids = torch.arange(
                position_offset,
                end,
                device=self.freqs.device,
                dtype=torch.long,
            )
            selected_freqs = self.freqs[position_offset:end]

        tokens = self.action_encoder(action_tokens)
        if context_mask.ndim == 3 and context_mask.shape[1] != seq_len:
            raise ValueError(
                f"3D `context_mask` query length must match action sequence length {seq_len}, "
                f"got {context_mask.shape[1]}"
            )

        if timestep.ndim == 1:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        else:
            t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1))
            t = self.time_embedding(t_emb).reshape(batch_size, seq_len, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        if context_mask.ndim == 2:
            if context_embedding is None:
                context_emb, context_mask_2d = self.encode_context(context, context_mask)
            else:
                context_emb = context_embedding
                context_mask_2d = context_mask.bool()
            context_attn_mask = context_mask_2d.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            context_emb = (
                self.text_embedding(context)
                if context_embedding is None
                else context_embedding
            )
            context_attn_mask = context_mask
        freqs = selected_freqs.view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "position_ids": rope_position_ids,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all action heads and return their per-route outputs.

        Returns a dict shaped:
            ``gameplay_continuous``: ``[B, T, action_dim]`` from the gameplay
                head. Always present when ``continuous_dim > 0``.
            ``gui_continuous``: same shape as ``gameplay_continuous``, from
                the GUI route head. ``None`` when GUI route is disabled.
            ``gui_route_logit``: ``[B, T, 1]``. ``None`` when GUI route is
                disabled.

        Both prediction heads are evaluated unconditionally; the
        caller (``GameWAM._route_heads``) does the hard-``torch.where``
        route on the action-timestep ``action_route_before_action``
        label. This way the route head's gradients flow even
        when the batch is mostly gameplay (i.e. GUI head always sees
        input → gradient, but only contributes to the routed output
        for tokens that actually happen in a GUI context).
        """
        gameplay = self.action_heads["gameplay"]
        out: Dict[str, Any] = {
            "gameplay_continuous": (
                gameplay.continuous_head(tokens) if gameplay.continuous_head is not None else None
            ),
        }
        if self.gui_head_enabled:
            gui = self.action_heads["gui"]
            out["gui_continuous"] = (
                gui.continuous_head(tokens) if gui.continuous_head is not None else None
            )
            out["gui_route_logit"] = self.gui_route_head(tokens) if self.gui_route_head is not None else None
        else:
            out["gui_continuous"] = None
            out["gui_route_logit"] = None
        return out

    @staticmethod
    def _route_experts(
        per_expert: Dict[str, Optional[torch.Tensor]],
        route_mask: torch.Tensor,
        *,
        default: str = "gameplay",
    ) -> Optional[torch.Tensor]:
        primary = per_expert.get(default)
        if primary is None:
            return None
        secondary_name = "gui" if default == "gameplay" else "gameplay"
        secondary = per_expert.get(secondary_name)
        if secondary is None:
            return primary
        return torch.where(
            route_mask.to(device=primary.device, dtype=torch.bool).expand_as(primary),
            secondary,
            primary,
        )

    def route_head_enabled(self) -> bool:
        return bool(self.gui_head_enabled)

    def primary_flow_prediction(
        self,
        pred_heads: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        flow = pred_heads.get("gameplay_continuous")
        if flow is None:
            raise ValueError("primary action-flow prediction is required.")
        return flow

    def route_heads_for_training(
        self,
        pred_heads: Dict[str, Optional[torch.Tensor]],
        route_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        primary_flow = self.primary_flow_prediction(pred_heads)
        if not self.route_head_enabled():
            return primary_flow, None
        if route_mask is None:
            raise ValueError("route_mask is required when routed action heads are enabled.")
        expected = (*primary_flow.shape[:2], 1)
        if tuple(route_mask.shape) != expected:
            raise ValueError(f"route_mask must have shape {expected}, got {tuple(route_mask.shape)}")
        route_mask = route_mask.to(device=primary_flow.device, dtype=torch.bool)
        secondary_flow = pred_heads.get("gui_continuous")
        routed_flow = (
            primary_flow
            if secondary_flow is None
            else torch.where(route_mask.expand_as(primary_flow), secondary_flow, primary_flow)
        )
        return routed_flow, pred_heads.get("gui_route_logit")

    def combine_cfg_heads(
        self,
        pred_c: Dict[str, Optional[torch.Tensor]],
        pred_u: Dict[str, Optional[torch.Tensor]],
        text_cfg_scale: float,
    ) -> Dict[str, Optional[torch.Tensor]]:
        scale = float(text_cfg_scale)

        def _cfg_boost(
            v_c: Optional[torch.Tensor],
            v_u: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if v_c is None and v_u is None:
                return None
            if v_u is None:
                return v_c
            if v_c is None:
                return v_u
            return v_c + (scale - 1.0) * (v_c - v_u)

        primary_cont = _cfg_boost(
            pred_c.get("gameplay_continuous"), pred_u.get("gameplay_continuous")
        )
        secondary_cont = _cfg_boost(
            pred_c.get("gui_continuous"), pred_u.get("gui_continuous")
        )
        route_logit = pred_c.get("gui_route_logit")

        per_expert_cont = {
            "gameplay": primary_cont,
            "gui": secondary_cont,
        }
        if route_logit is not None and per_expert_cont["gui"] is not None:
            route_mask = torch.sigmoid(route_logit.float()) > 0.5
            continuous = self._route_experts(per_expert_cont, route_mask, default="gameplay")
        else:
            continuous = per_expert_cont["gameplay"]
            route_mask = None

        return {
            "continuous": continuous,
            "route_logit": route_logit,
            "route_mask": route_mask,
        }

    def route_cond_heads(
        self,
        pred_heads: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self.combine_cfg_heads(pred_heads, pred_heads, 1.0)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for layer_idx, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
