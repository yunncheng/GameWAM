import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from einops import rearrange
from .helpers.gradient import gradient_checkpoint_forward

from gamewam.utils.logging_config import get_logger

logger = get_logger(__name__)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    attn_kwargs = {}
    if ctx_mask is not None:
        if ctx_mask.dtype == torch.bool:
            # Never branch on a CUDA reduction here: this primitive runs many
            # times per layer. An all-True additive mask is mathematically the
            # same as no mask and avoids synchronizing the host.
            float_mask = torch.where(ctx_mask, 0.0, float("-inf")).to(
                dtype=q.dtype, device=q.device
            )
            if float_mask.ndim == 2:
                float_mask = float_mask.unsqueeze(0).unsqueeze(0)
            elif float_mask.ndim == 3:
                float_mask = float_mask.unsqueeze(1)
            attn_kwargs["attn_mask"] = float_mask
        else:
            attn_kwargs["attn_mask"] = ctx_mask
    x = F.scaled_dot_product_attention(q, k, v, **attn_kwargs)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def build_identity_rope_like(reference_freqs: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Return RoPE freqs that make rope_apply(x, freqs) == x (identity rotation).

    The RoPE representation uses complex multiplication: x_out = x * freqs.
    Identity means freqs = 1+0j for all positions.

    Args:
        reference_freqs: [S, 1, rope_dim] complex freqs tensor (used to infer shape/dtype/device).
        n_tokens: Number of history prefix tokens to generate identity freqs for.

    Returns:
        identity_freqs: [n_tokens, 1, rope_dim] complex tensor of all ones.
    """
    rope_dim = reference_freqs.shape[2]
    identity = torch.ones(
        n_tokens, 1, rope_dim,
        dtype=reference_freqs.dtype,
        device=reference_freqs.device,
    )
    return identity


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """Creates a group-based attention mask for scaled dot-product attention with two modes:"""
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)


    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        if ctx_mask is not None:
            if ctx_mask.ndim == 2:
                if ctx_mask.shape != ctx.shape[:2]:
                    raise ValueError(
                        "2D cross-attention mask must be [B,K], got "
                        f"{tuple(ctx_mask.shape)} for context {tuple(ctx.shape)}"
                    )
                ctx_mask = ctx_mask.unsqueeze(1).unsqueeze(1)
            elif ctx_mask.ndim == 3:
                ctx_mask = ctx_mask.unsqueeze(1)
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    # Paired with WanVideoVAE38: causal stride-2 temporal convs ×2 give R = 1 + 4*(L-1).
    VAE_TEMPORAL_FACTOR = 4

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        num_obs_frames: int,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        action_video_freq_ratio: int = 1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        # raw → video (subsampled) → latent conversion:
        #   obs_video_frames = (num_obs_frames - 1) // ratio + 1
        #   num_obs_latents  = (obs_video_frames - 1) // VAE_TEMPORAL_FACTOR + 1
        if num_obs_frames < 1:
            raise ValueError(f"`num_obs_frames` must be >= 1, got {num_obs_frames}")
        if action_video_freq_ratio < 1:
            raise ValueError(f"`action_video_freq_ratio` must be >= 1, got {action_video_freq_ratio}")
        obs_video_frames = (num_obs_frames - 1) // action_video_freq_ratio + 1
        if (obs_video_frames - 1) % self.VAE_TEMPORAL_FACTOR != 0:
            raise ValueError(
                f"After subsampling by ratio={action_video_freq_ratio}, "
                f"obs has {obs_video_frames} video frames, but (obs_video_frames-1) "
                f"must be divisible by VAE temporal factor {self.VAE_TEMPORAL_FACTOR}. "
                f"Got num_obs_frames={num_obs_frames}."
            )
        self.num_obs_frames = int(num_obs_frames)
        self.num_obs_latents = (obs_video_frames - 1) // self.VAE_TEMPORAL_FACTOR + 1

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )

        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode

        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
        action_condition_start_latent_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            start_latent = (
                int(self.num_obs_latents)
                if action_condition_start_latent_idx is None
                else int(action_condition_start_latent_idx)
            )
            if start_latent < 0 or start_latent > num_latent_frames:
                raise ValueError(
                    f"`action_condition_start_latent_idx` must be in [0,{num_latent_frames}], got {start_latent}"
                )
            num_action_conditioned_latents = num_latent_frames - start_latent
            if action is None:
                if num_action_conditioned_latents > 0:
                    raise ValueError(
                        "Action-conditioned video inference requires `action` for predicted latent frames. "
                        f"num_latent_frames={num_latent_frames}, action_condition_start_latent_idx={start_latent}."
                    )
            else:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_action_conditioned_latents <= 0:
                    raise ValueError(
                        "`action` was provided but no latent frame is action-conditioned: "
                        f"num_latent_frames={num_latent_frames}, action_condition_start_latent_idx={start_latent}."
                    )
                if action.shape[1] % num_action_conditioned_latents != 0:
                    raise ValueError(
                        "action horizon must be divisible by the number of action-conditioned latent frames, "
                        f"got action_horizon={action.shape[1]}, num_action_conditioned_latents={num_action_conditioned_latents}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim not in (2, 3):
                raise ValueError(
                    f"`context_mask` must be 2D [B, L] or 3D [B, S, L], got shape {tuple(context_mask.shape)}"
                )
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[-1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match context batch/length, got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_context_payload(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        *,
        query_latent_frames: int,
        tokens_per_frame: int,
        action: Optional[torch.Tensor] = None,
        action_condition_start_latent_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build video cross-attention context and optional action-condition mask."""
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        if context_mask.ndim not in (2, 3):
            raise ValueError(
                f"`context_mask` must be 2D [B,L] or 3D [B,S,L], got shape {tuple(context_mask.shape)}"
            )
        if context_mask.shape[0] != context.shape[0] or context_mask.shape[-1] != context.shape[1]:
            raise ValueError(
                f"`context_mask` shape must match context batch/length, got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
            )
        if query_latent_frames < 1:
            raise ValueError(f"`query_latent_frames` must be positive, got {query_latent_frames}")
        if tokens_per_frame < 1:
            raise ValueError(f"`tokens_per_frame` must be positive, got {tokens_per_frame}")

        batch_size = context.shape[0]
        context = self.text_embedding(context)
        context_len = context.shape[1]
        seq_len = query_latent_frames * tokens_per_frame
        if context_mask.ndim == 2:
            base_context_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            if context_mask.shape[1] != seq_len:
                raise ValueError(
                    f"3D `context_mask` query length must match video query length {seq_len}, "
                    f"got {context_mask.shape[1]}"
                )
            base_context_mask = context_mask

        start_latent = (
            int(self.num_obs_latents)
            if action_condition_start_latent_idx is None
            else int(action_condition_start_latent_idx)
        )
        if start_latent < 0 or start_latent > query_latent_frames:
            raise ValueError(
                f"`action_condition_start_latent_idx` must be in [0,{query_latent_frames}], got {start_latent}"
            )

        if self.action_conditioned and action is not None:
            if action.ndim != 3:
                raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
            if action.shape[0] != batch_size:
                raise ValueError(f"`action` batch size must be {batch_size}, got {action.shape[0]}")
            if action.shape[2] != self.action_dim:
                raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)
            action_pos_embed = sinusoidal_embedding_1d(
                self.hidden_dim,
                torch.arange(action_len, device=action_emb.device),
            )
            action_emb = action_emb + action_pos_embed.unsqueeze(0)
            context = torch.cat([context, action_emb], dim=1)

            num_temporal_groups = query_latent_frames - start_latent
            if num_temporal_groups <= 0:
                raise ValueError(
                    "`action` was provided but no latent frame is action-conditioned: "
                    f"query_latent_frames={query_latent_frames}, action_condition_start_latent_idx={start_latent}."
                )
            if action_len % num_temporal_groups != 0:
                raise ValueError(
                    "Action embedding length must be divisible by the number of action-conditioned latent frames, "
                    f"got action_len={action_len}, num_temporal_groups={num_temporal_groups}."
                )

            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device)

            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device)
            final_context_mask[:, :, :context_len] = base_context_mask
            start_token = start_latent * tokens_per_frame
            final_context_mask[:, start_token:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            num_temporal_groups = query_latent_frames - start_latent
            if num_temporal_groups > 0:
                raise ValueError(
                    "Action-conditioned video inference requires `action` for predicted latent frames. "
                    f"query_latent_frames={query_latent_frames}, action_condition_start_latent_idx={start_latent}."
                )
            context_mask = base_context_mask
        else:
            context_mask = base_context_mask

        return {"context": context, "mask": context_mask}

    def _get_3d_rope_freqs(
        self,
        t_start: int,
        num_frames: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return cached 3D RoPE freqs ``[num_frames*h*w, 1, dim]`` on device.

        Equivalent to the manual cat/expand/reshape/to-device sequence but
        cached by ``(t_start, num_frames, h, w, device)`` across steps.
        RoPE freqs depend only on geometry, not on activations.
        """
        cache = getattr(self, "_rope_freqs_cache", None)
        if cache is None:
            cache = {}
            self._rope_freqs_cache = cache
        key = (int(t_start), int(num_frames), int(h), int(w), device)
        freqs = cache.get(key)
        if freqs is None:
            freqs = torch.cat([
                self.freqs[0][t_start:t_start + num_frames].view(num_frames, 1, 1, -1).expand(num_frames, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(num_frames, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(num_frames, h, w, -1),
            ], dim=-1).reshape(num_frames * h * w, 1, -1).to(device)
            cache[key] = freqs
        return freqs

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_k_frames_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            obs_tokens = min(video_tokens_per_frame * self.num_obs_latents, video_seq_len)
            video_mask[:obs_tokens, obs_tokens:] = False
            return video_mask

        if self.video_attention_mask_mode == "ar_teacher_forcing":
            raise RuntimeError(
                "AR Teacher-Forcing mode builds attention masks in MoT/GameWAM."
            )

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        action_condition_start_latent_idx: Optional[int] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            action_condition_start_latent_idx=action_condition_start_latent_idx,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")

            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, :self.num_obs_latents, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError("Only seperated_timestep=True with fuse_vae_embedding_in_latents=True is supported.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        video_context = self.build_video_context_payload(
            context=context,
            context_mask=context_mask,
            query_latent_frames=f,
            tokens_per_frame=tokens_per_frame,
            action=action,
            action_condition_start_latent_idx=action_condition_start_latent_idx,
        )
        context = video_context["context"]
        context_mask = video_context["mask"]

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = self._get_3d_rope_freqs(0, f, h, w, x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
                )
            else:
                x_tokens = block(x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask)

        return self.post_dit(x_tokens, pre_state)
