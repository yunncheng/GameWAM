"""History Memory Module for AR-history mode.

Provides ``HistorySegment``, ``SegmentTokenDecorator`` (frame/token/time
embeddings shared by recent and long paths), ``MemoryUpdater`` (one direct
cross-attention that compresses overflow segments into fixed-capacity
LongMemory slots), and ``HistoryModule`` (replay + prefix construction).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping as MappingABC
from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


VAE_TEMPORAL_FACTOR = 4


def derive_memory_write_capacity_segments(history_cfg: Mapping[str, Any]) -> int:
    """Derive LongMemory overflow capacity from the longest memory timescale."""
    half_lives = history_cfg.get("memory_half_life_segments", []) or []
    if any(int(hl) <= 0 for hl in half_lives):
        raise ValueError(f"memory_half_life_segments must be positive, got {half_lives}")
    return int(max(half_lives) if half_lives else 0)


def _conv_output_size(
    size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _as_three_tuple(value: Any, name: str) -> tuple[int, int, int]:
    try:
        seq = None if isinstance(value, (str, bytes, MappingABC)) else list(value)
    except TypeError:
        seq = None
    if seq is None or len(seq) != 3:
        raise ValueError(
            f"`{name}` must be a 3-tuple/list, got {value!r}"
        )
    return int(seq[0]), int(seq[1]), int(seq[2])


@dataclass
class HistoryConfig:
    recent_num_segments: int = 2

    history_conv_kernel_size: tuple[int, int, int] = (1, 7, 7)
    history_conv_stride: tuple[int, int, int] = (1, 7, 7)

    memory_num_levels: int = 4
    memory_tokens_per_level: int = 4
    memory_half_life_segments: list[int] = None
    history_horizon_dropout: float = 0.1
    history_segment_tokenize_chunk_size: int = 512
    predictive_loss_weight: float = 0.0

    def __post_init__(self):
        self.history_conv_kernel_size = _as_three_tuple(
            self.history_conv_kernel_size, "history_conv_kernel_size"
        )
        self.history_conv_stride = _as_three_tuple(
            self.history_conv_stride, "history_conv_stride"
        )
        kt, kh, kw = self.history_conv_kernel_size
        st, sh, sw = self.history_conv_stride
        if min(kt, kh, kw) <= 0 or min(st, sh, sw) <= 0:
            raise ValueError(
                "history_conv kernel/stride values must be positive, "
                f"got kernel={self.history_conv_kernel_size}, stride={self.history_conv_stride}"
            )
        if kt != 1 or st != 1:
            raise ValueError(
                "history_conv temporal kernel/stride must be 1, "
                f"got kernel_t={kt}, stride_t={st}"
            )

        if self.memory_half_life_segments is None:
            self.memory_half_life_segments = [2, 4, 8, 16]
        self.memory_half_life_segments = [int(hl) for hl in self.memory_half_life_segments]
        if any(hl <= 0 for hl in self.memory_half_life_segments):
            raise ValueError(f"memory_half_life_segments must be positive, got {self.memory_half_life_segments}")
        if len(self.memory_half_life_segments) != self.memory_num_levels:
            raise ValueError(
                f"memory_half_life_segments length ({len(self.memory_half_life_segments)}) "
                f"must match memory_num_levels ({self.memory_num_levels})"
            )
        self.history_horizon_dropout = float(self.history_horizon_dropout)
        if not 0.0 <= self.history_horizon_dropout <= 1.0:
            raise ValueError(
                f"history_horizon_dropout must be in [0, 1], got {self.history_horizon_dropout}"
            )
        self.history_segment_tokenize_chunk_size = int(self.history_segment_tokenize_chunk_size)
        if self.history_segment_tokenize_chunk_size <= 0:
            raise ValueError(
                "history_segment_tokenize_chunk_size must be positive, "
                f"got {self.history_segment_tokenize_chunk_size}"
            )
        self.predictive_loss_weight = float(self.predictive_loss_weight)
        if self.predictive_loss_weight < 0.0:
            raise ValueError(
                f"predictive_loss_weight must be non-negative, got {self.predictive_loss_weight}"
            )

    @property
    def memory_write_capacity_segments(self) -> int:
        return derive_memory_write_capacity_segments({
            "memory_half_life_segments": self.memory_half_life_segments,
        })

    @property
    def max_history_segments(self) -> int:
        return self.recent_num_segments + self.memory_write_capacity_segments


@dataclass
class HistorySegment:
    latents: torch.Tensor  # [C, T_seg, H, W]
    raw_start: int
    next_cycle_start_raw: int


@dataclass
class MemoryWriteRecord:
    """One overflow segment's cached source tokens for LongMemory updates."""
    source_tokens: torch.Tensor  # [1, S, D] (S = T_seg * compressed_tokens_per_frame)
    duration_raw: int
    raw_start: int
    next_cycle_start_raw: int
    reference_raw: int


@dataclass
class SegmentTokenRequest:
    sample_idx: int
    segment_idx: int
    purpose: str  # "recent_read" or "memory_source"
    raw_start: int
    next_cycle_start_raw: int
    reference_raw: int
    latents: torch.Tensor


@dataclass
class HistoryState:
    """Inference-time mutable state for AR-history rollout."""
    recent_history: list  # list[HistorySegment]
    long_memory: torch.Tensor  # [1, L, K, D]
    long_memory_write_count: torch.Tensor  # [1]
    current_start_raw: int
    obs_buffer: list  # list of raw-indexed observation records
    memory_write_queue: list  # list[MemoryWriteRecord]


@dataclass
class HistoryGeometry:
    """Derived AR latent and history geometry.

    During online block-cycle control, each predicted latent corresponds to
    one planning/execution block because ``raw_steps_per_latent`` equals the
    execution horizon. Therefore, ``pred_latents`` is also the number of
    blocks per cycle. A history segment stores one completed cycle.
    """
    cycle_stride_raw: int
    pred_latents: int
    history_segment_latents: int
    total_latents: int
    obs_latents: int
    video_step_raw: int
    raw_steps_per_latent: int
    frame_stride: int
    num_obs_frames: int
    num_frames: int
    action_video_freq_ratio: int


def derive_history_geometry(
    num_frames: int,
    num_obs_frames: int,
    frame_stride: int,
    action_video_freq_ratio: int,
) -> HistoryGeometry:
    """Derive AR geometry constants and verify invariants."""
    video_step_raw = frame_stride * action_video_freq_ratio
    num_video_frames_total = (num_frames - 1) // action_video_freq_ratio + 1
    num_video_frames_obs = (num_obs_frames - 1) // action_video_freq_ratio + 1

    total_latents = (num_video_frames_total - 1) // VAE_TEMPORAL_FACTOR + 1
    obs_latents = (num_video_frames_obs - 1) // VAE_TEMPORAL_FACTOR + 1
    pred_latents = total_latents - obs_latents

    cycle_stride_raw = (num_frames - num_obs_frames) * frame_stride

    pred_video_frames = num_video_frames_total - num_video_frames_obs
    raw_steps_per_latent = VAE_TEMPORAL_FACTOR * action_video_freq_ratio * frame_stride
    history_segment_latents = pred_latents + 1

    # Geometry invariants
    assert cycle_stride_raw == (num_frames - num_obs_frames) * frame_stride
    assert pred_video_frames % VAE_TEMPORAL_FACTOR == 0, (
        f"pred_video_frames={pred_video_frames} not divisible by VAE_TEMPORAL_FACTOR={VAE_TEMPORAL_FACTOR}"
    )
    assert pred_latents == pred_video_frames // VAE_TEMPORAL_FACTOR
    assert cycle_stride_raw == pred_video_frames * action_video_freq_ratio * frame_stride, (
        f"cycle_stride_raw={cycle_stride_raw} != pred_video_frames*action_video_freq_ratio*frame_stride="
        f"{pred_video_frames}*{action_video_freq_ratio}*{frame_stride}="
        f"{pred_video_frames * action_video_freq_ratio * frame_stride}"
    )
    assert history_segment_latents == pred_latents + 1
    assert history_segment_latents <= total_latents
    assert raw_steps_per_latent == VAE_TEMPORAL_FACTOR * video_step_raw

    return HistoryGeometry(
        cycle_stride_raw=cycle_stride_raw,
        pred_latents=pred_latents,
        history_segment_latents=history_segment_latents,
        total_latents=total_latents,
        obs_latents=obs_latents,
        video_step_raw=video_step_raw,
        raw_steps_per_latent=raw_steps_per_latent,
        frame_stride=frame_stride,
        num_obs_frames=num_obs_frames,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
    )


def compute_history_starts(
    sample_start_raw: int,
    episode_start_raw: int,
    cycle_stride_raw: int,
) -> list[int]:
    """Compute history segment start positions by stepping back from sample_start_raw."""
    starts = []
    p = sample_start_raw - cycle_stride_raw
    while p >= episode_start_raw:
        starts.insert(0, p)
        p -= cycle_stride_raw
    return starts


def init_history_state(
    history_config: 'HistoryConfig',
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> HistoryState:
    """Initialize a fresh HistoryState for inference episode start."""
    return HistoryState(
        recent_history=[],
        long_memory=torch.zeros(
            1, history_config.memory_num_levels, history_config.memory_tokens_per_level,
            hidden_dim, device=device, dtype=dtype,
        ),
        long_memory_write_count=torch.zeros(1, dtype=torch.long, device=device),
        current_start_raw=0,
        obs_buffer=[],
        memory_write_queue=[],
    )


def bucketize_write_count(count: torch.Tensor) -> torch.Tensor:
    """Bucket LongMemory write count into 7 bins: 0, 1, 2, 3-4, 5-8, 9-16, 17+."""
    bucket = torch.zeros_like(count)
    bucket = torch.where(count == 1, torch.ones_like(bucket) * 1, bucket)
    bucket = torch.where(count == 2, torch.ones_like(bucket) * 2, bucket)
    bucket = torch.where((count >= 3) & (count <= 4), torch.ones_like(bucket) * 3, bucket)
    bucket = torch.where((count >= 5) & (count <= 8), torch.ones_like(bucket) * 4, bucket)
    bucket = torch.where((count >= 9) & (count <= 16), torch.ones_like(bucket) * 5, bucket)
    bucket = torch.where(count >= 17, torch.ones_like(bucket) * 6, bucket)
    return bucket


NUM_WRITE_COUNT_BUCKETS = 7


class SegmentTokenDecorator(nn.Module):
    """Add frame-index, spatial-token-id, and time embeddings to per-frame Conv tokens.

    Used by both the recent (Conv-only) and long-source (direct MemoryUpdater)
    paths. Decorator is not a compression layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_segment_latents: int,
        tokens_per_latent: int,
        cycle_stride_raw: int,
    ):
        super().__init__()
        self.cycle_stride_raw = int(cycle_stride_raw)
        self.segment_latent_index_embed = nn.Embedding(max_segment_latents, hidden_dim)
        self.segment_token_id_embed = nn.Embedding(tokens_per_latent, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.segment_latent_index_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.segment_token_id_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.time_mlp[0].weight)
        nn.init.zeros_(self.time_mlp[0].bias)
        nn.init.zeros_(self.time_mlp[2].weight)
        nn.init.zeros_(self.time_mlp[2].bias)

    def forward(
        self,
        frame_tokens: torch.Tensor,         # [B, T_seg, K, D]
        reference_raw: torch.Tensor,         # [B]
        raw_start: torch.Tensor,             # [B]
        next_cycle_start_raw: torch.Tensor,  # [B]
    ) -> torch.Tensor:                       # [B, T_seg, K, D]
        B, T_seg, K, D = frame_tokens.shape
        device = frame_tokens.device

        latent_ids = torch.arange(T_seg, device=device)
        token_ids = torch.arange(K, device=device)
        latent_emb = self.segment_latent_index_embed(latent_ids)[None, :, None, :]
        token_emb = self.segment_token_id_embed(token_ids)[None, None, :, :]
        tokens = frame_tokens + latent_emb + token_emb  # [B, T_seg, K, D]

        relative_start = reference_raw - raw_start
        relative_end = reference_raw - next_cycle_start_raw
        duration = next_cycle_start_raw - raw_start

        age_start = torch.log1p(relative_start.float() / self.cycle_stride_raw)
        age_end = torch.log1p(relative_end.float().clamp_min(0) / self.cycle_stride_raw)
        dur = duration.float() / self.cycle_stride_raw

        time_input = torch.stack([age_start, age_end, dur], dim=-1).to(
            dtype=tokens.dtype, device=device,
        )
        time_emb = self.time_mlp(time_input).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        return tokens + time_emb


class MemoryUpdater(nn.Module):
    """Updates LongMemory from full overflow source tokens via a single gated cross-attention.

    Each LongMemory slot (level, position) directly attends to all source tokens
    of the overflow segment. Different slots learn to select different subsets
    of the source via learned queries; the half-life gate prior keeps short-term
    levels responsive and long-term levels stable.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        tokens_per_level: int,
        half_life_segments: list[int],
        cycle_stride_raw: int,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.tokens_per_level = tokens_per_level
        self.half_life_segments = [int(hl) for hl in half_life_segments]
        self.half_life_raw = [hl * cycle_stride_raw for hl in self.half_life_segments]

        self.level_embed = nn.Parameter(torch.randn(num_levels, hidden_dim) * 0.02)
        self.token_id_embed = nn.Parameter(torch.randn(tokens_per_level, hidden_dim) * 0.02)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        self.gate_delta = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.level_embed.data.copy_(self._build_half_life_level_embedding(self.level_embed))
        nn.init.normal_(self.token_id_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_delta.weight)
        nn.init.zeros_(self.gate_delta.bias)

    def _build_half_life_level_embedding(self, target: torch.Tensor) -> torch.Tensor:
        """Small deterministic level embeddings where nearby half-lives are nearby."""
        device = target.device
        dtype = target.dtype
        hidden_dim = target.shape[1]
        x = torch.log(torch.tensor(self.half_life_segments, device=device, dtype=torch.float32))
        if x.numel() <= 1 or torch.isclose(x.std(unbiased=False), torch.tensor(0.0, device=device)):
            return torch.zeros_like(target)
        x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)

        pair_dim = hidden_dim // 2
        if pair_dim == 0:
            return torch.zeros_like(target)
        freqs = torch.exp(
            torch.linspace(0.0, math.log(16.0), pair_dim, device=device, dtype=torch.float32)
        )
        angles = x[:, None] * freqs[None, :]
        emb = torch.zeros(self.num_levels, hidden_dim, device=device, dtype=torch.float32)
        emb[:, 0:2 * pair_dim:2] = torch.sin(angles)
        emb[:, 1:2 * pair_dim:2] = torch.cos(angles)
        emb = emb - emb.mean(dim=0, keepdim=True)
        return (0.02 * emb).to(dtype=dtype)

    def _gate_prior_logits(
        self,
        duration_raw: int | torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Returns [B, L, 1, 1] half-life prior logit for all levels."""
        duration = torch.as_tensor(duration_raw, device=device, dtype=torch.float32)
        if duration.ndim == 0:
            duration = duration.expand(batch_size)
        elif duration.shape != (batch_size,):
            raise ValueError(
                f"`duration_raw` must be scalar or [B]; got shape {tuple(duration.shape)}"
            )

        half_life_raw = torch.tensor(
            self.half_life_raw, device=device, dtype=torch.float32,
        )
        alpha = 1.0 - torch.pow(2.0, -duration[:, None] / half_life_raw[None, :])
        alpha = alpha.clamp(1e-4, 1.0 - 1e-4)
        logits = torch.log(alpha / (1.0 - alpha))
        return logits.to(dtype=dtype)[:, :, None, None]

    def forward(
        self,
        memory: torch.Tensor,         # [B, L, K, D]
        source_tokens: torch.Tensor,  # [B, S, D]
        duration_raw: int | torch.Tensor,
    ) -> torch.Tensor:
        """Returns updated memory [B, L, K, D]."""
        B, L, K, D = memory.shape

        if L != self.num_levels:
            raise ValueError(
                f"memory num_levels mismatch: got {L}, expected {self.num_levels}"
            )
        if K != self.tokens_per_level:
            raise ValueError(
                f"memory tokens_per_level mismatch: got {K}, expected {self.tokens_per_level}"
            )
        if source_tokens.ndim != 3 or source_tokens.shape[0] != B or source_tokens.shape[2] != D:
            raise ValueError(
                f"source_tokens must be [B, S, D]=[{B}, S, {D}]; got {tuple(source_tokens.shape)}"
            )

        q = (
            memory
            + self.level_embed[None, :, None, :]
            + self.token_id_embed[None, None, :, :]
        )
        q = self.query_norm(q.reshape(B, L * K, D))
        source = self.source_norm(source_tokens)

        candidate, _ = self.cross_attn(
            q, source, source, need_weights=False,
        )
        candidate = candidate.reshape(B, L, K, D)

        gate_prior = self._gate_prior_logits(
            duration_raw=duration_raw,
            batch_size=B,
            device=memory.device,
            dtype=memory.dtype,
        )

        gate_delta = self.gate_delta(torch.cat([memory, candidate], dim=-1))
        gate = torch.sigmoid(gate_delta + gate_prior)

        return self.output_norm(memory + gate * (candidate - memory))


class HistoryModule(nn.Module):
    """Orchestrates history prefix construction for AR-history mode.

    Recent segments keep all decorated Conv tokens. Overflow segments go
    directly into LongMemory via a single MemoryUpdater cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        history_config: HistoryConfig,
        geometry: HistoryGeometry,
        latent_in_dim: int,
        init_patch_embedding: nn.Conv3d | None = None,
        latent_spatial_size: tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfg = history_config
        self.geometry = geometry

        latent_h, latent_w = (int(latent_spatial_size[0]), int(latent_spatial_size[1]))
        kt, kh, kw = history_config.history_conv_kernel_size
        st, sh, sw = history_config.history_conv_stride

        compressed_h = _conv_output_size(latent_h, kh, sh, 0)
        compressed_w = _conv_output_size(latent_w, kw, sw, 0)
        if compressed_h <= 0 or compressed_w <= 0:
            raise ValueError(
                "History Conv3D produces non-positive spatial output; "
                f"latent_spatial_size={latent_spatial_size}, "
                f"kernel={history_config.history_conv_kernel_size}, "
                f"stride={history_config.history_conv_stride}"
            )
        self.compressed_patch_size = (kt, kh, kw)
        self.compressed_stride = (st, sh, sw)
        self.compressed_tokens_per_frame = compressed_h * compressed_w
        self.recent_tokens_per_segment = (
            geometry.history_segment_latents * self.compressed_tokens_per_frame
        )
        self.segment_source_tokens_per_segment = self.recent_tokens_per_segment

        self.patch_embedding_compressed = nn.Conv3d(
            int(latent_in_dim),
            hidden_dim,
            kernel_size=self.compressed_patch_size,
            stride=self.compressed_stride,
        )
        if init_patch_embedding is not None:
            self.initialize_compressed_patch_embedding_from_patch_embedding(init_patch_embedding)

        self.segment_token_decorator = SegmentTokenDecorator(
            hidden_dim=hidden_dim,
            max_segment_latents=geometry.history_segment_latents,
            tokens_per_latent=self.compressed_tokens_per_frame,
            cycle_stride_raw=geometry.cycle_stride_raw,
        )

        # Recent path: small role embedding + LayerNorm so recent tokens
        # are distinguishable from long-source tokens in the DiT prefix.
        self.recent_role_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.recent_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.recent_role_embed)

        # Long source path: same role/norm pair, distinct parameters.
        self.long_source_role_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.long_source_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.long_source_role_embed)

        self.memory_updater = MemoryUpdater(
            hidden_dim=hidden_dim,
            num_levels=history_config.memory_num_levels,
            tokens_per_level=history_config.memory_tokens_per_level,
            half_life_segments=history_config.memory_half_life_segments,
            cycle_stride_raw=geometry.cycle_stride_raw,
        )

        self.write_count_bucket_embed = nn.Embedding(NUM_WRITE_COUNT_BUCKETS, hidden_dim)
        nn.init.zeros_(self.write_count_bucket_embed.weight)

        self.predictive_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.xavier_uniform_(self.predictive_delta[-1].weight)
        nn.init.zeros_(self.predictive_delta[-1].bias)

    @property
    def long_memory_token_count(self) -> int:
        return int(self.cfg.memory_num_levels) * int(self.cfg.memory_tokens_per_level)

    @property
    def segment_source_token_count(self) -> int:
        return self.segment_source_tokens_per_segment

    @property
    def history_prefix_token_count(self) -> int:
        return self.long_memory_token_count + (
            int(self.cfg.recent_num_segments) * self.recent_tokens_per_segment
        )

    @staticmethod
    def _zero_parameter_dependency(
        anchor: torch.Tensor,
        parameters: Iterable[nn.Parameter],
    ) -> torch.Tensor:
        """Return scalar zero connected to the supplied trainable parameters.

        This preserves the ZeRO/DDP per-rank gradient contract when a local
        batch legitimately skips a parameterized path, without executing
        placeholder compute for that path.
        """
        dependency = anchor.new_zeros(())
        for parameter in parameters:
            if parameter.requires_grad:
                dependency = dependency + parameter.reshape(-1)[0] * 0.0
        return dependency

    def _recent_encoding_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters used whenever at least one recent segment is encoded."""
        return (
            self.recent_role_embed,
            *self.recent_norm.parameters(),
            *self.patch_embedding_compressed.parameters(),
            *self.segment_token_decorator.parameters(),
        )

    def _long_memory_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters used whenever at least one overflow segment is replayed."""
        return (
            *self.memory_updater.parameters(),
            self.long_source_role_embed,
            *self.long_source_norm.parameters(),
        )

    @torch.no_grad()
    def initialize_compressed_patch_embedding_from_patch_embedding(
        self,
        patch_embedding: nn.Conv3d,
    ) -> None:
        """Initialize the history latent projection from the video expert projection."""
        src = patch_embedding
        dst = self.patch_embedding_compressed
        if src.weight.shape[:2] != dst.weight.shape[:2]:
            raise ValueError(
                "Cannot initialize history compressed patch embedding from patch_embedding: "
                f"channel shape mismatch src={tuple(src.weight.shape)} dst={tuple(dst.weight.shape)}"
            )

        src_weight = src.weight.detach().float()
        dst_kernel = tuple(int(v) for v in dst.weight.shape[2:])
        flat = src_weight.reshape(-1, 1, *src_weight.shape[2:])
        resized = F.interpolate(flat, size=dst_kernel, mode="trilinear", align_corners=False)
        resized = resized.reshape(dst.weight.shape)

        src_fan_in = int(math.prod(src.weight.shape[1:]))
        dst_fan_in = int(math.prod(dst.weight.shape[1:]))
        resized = resized * math.sqrt(src_fan_in / dst_fan_in)

        dst.weight.copy_(resized.to(device=dst.weight.device, dtype=dst.weight.dtype))
        if dst.bias is not None:
            if src.bias is not None:
                dst.bias.copy_(src.bias.detach().to(device=dst.bias.device, dtype=dst.bias.dtype))
            else:
                dst.bias.zero_()

    def compress_single_frame_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Compress one VAE latent frame into history tokens."""
        x = self.patch_embedding_compressed(latent)  # [B, D, 1, H', W']
        return x.flatten(3).squeeze(2).transpose(1, 2)  # [B, K, D]

    def compress_segment_latents(self, segment_latents: torch.Tensor) -> torch.Tensor:
        """Compress segment latents into per-frame tokens.

        Returns ``[B, T_seg, K, D]`` where ``K = compressed_tokens_per_frame``.
        """
        x = self.patch_embedding_compressed(segment_latents)  # [B, D, T, H', W']
        B, D, T_seg, Hc, Wc = x.shape
        return x.permute(0, 2, 3, 4, 1).contiguous().view(B, T_seg, Hc * Wc, D)

    def _resolve_chunk_size(self, chunk_size: int | None) -> int:
        if chunk_size is None:
            chunk_size = int(self.cfg.history_segment_tokenize_chunk_size)
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        return chunk_size

    def tokenize_segments_batched(
        self,
        latents: torch.Tensor,
        reference_raw: torch.Tensor,
        raw_start: torch.Tensor,
        next_cycle_start_raw: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Conv + decorate + flatten. NO role embedding, NO path-specific norm.

        Returns ``[N, T_seg * compressed_tokens_per_frame, D]``.
        """
        if latents.shape[0] == 0:
            return torch.zeros(
                0, self.segment_source_tokens_per_segment, self.hidden_dim,
                device=device, dtype=dtype,
            )
        chunk_size = self._resolve_chunk_size(chunk_size)

        encoded_chunks = []
        for start in range(0, latents.shape[0], chunk_size):
            end = min(start + chunk_size, latents.shape[0])
            latents_chunk = latents[start:end].to(device=device, dtype=dtype)
            frame_tokens = self.compress_segment_latents(latents_chunk)
            decorated = self.segment_token_decorator(
                frame_tokens=frame_tokens,
                reference_raw=reference_raw[start:end].to(device=device, dtype=torch.long),
                raw_start=raw_start[start:end].to(device=device, dtype=torch.long),
                next_cycle_start_raw=next_cycle_start_raw[start:end].to(device=device, dtype=torch.long),
            )
            flat = decorated.flatten(1, 2)  # [b, T_seg*K, D]
            encoded_chunks.append(flat)
        return torch.cat(encoded_chunks, dim=0)

    def encode_recent_segments_batched(
        self,
        latents: torch.Tensor,
        reference_raw: torch.Tensor,
        raw_start: torch.Tensor,
        next_cycle_start_raw: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Recent path: tokenize + recent role/norm.

        Returns ``[N, recent_tokens_per_segment, D]``.
        """
        tokens = self.tokenize_segments_batched(
            latents=latents,
            reference_raw=reference_raw,
            raw_start=raw_start,
            next_cycle_start_raw=next_cycle_start_raw,
            device=device,
            dtype=dtype,
            chunk_size=chunk_size,
        )
        if tokens.shape[0] == 0:
            return tokens
        return self.recent_norm(tokens + self.recent_role_embed)

    def predictive_history_loss_batch(
        self,
        batch: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        step_cache: dict[str, Any] | None = None,
        compute_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Auxiliary predictive objective for strengthening visual history.

        This predicts the pooled clean feature for the current observation
        boundary from the already-built history prefix only.
        """
        weight = float(getattr(self.cfg, "predictive_loss_weight", 0.0))
        zero = torch.zeros((), device=device, dtype=dtype)
        if weight <= 0.0:
            return zero, (
                {"history_predictive_valid": zero.detach()}
                if compute_metrics
                else {}
            )
        required = (
            "history_segment_latents",
            "history_segment_mask",
            "history_segment_start_raw",
            "history_segment_next_cycle_start_raw",
            "sample_start_raw",
            "input_latents",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(
                "predictive_loss_weight > 0 requires history batch fields, "
                f"missing {missing}"
            )

        metadata_batch = self._history_metadata_on_cpu(batch, step_cache)
        source_mask = metadata_batch["history_segment_mask"]
        if source_mask.ndim != 2:
            raise ValueError(f"history_segment_mask must be [B,N], got {tuple(source_mask.shape)}")
        # Resolve the empty-history branch from the shared CPU metadata view,
        # avoiding a CUDA synchronization on every training step.
        has_valid_rows = bool(source_mask.any())
        mask_all = source_mask.to(device=device, dtype=torch.bool)
        valid_rows = mask_all.any(dim=1)
        if not has_valid_rows:
            # The raw local batch has no usable history. Keep the paths that a
            # non-empty rank executes in the graph so ZeRO sees zero gradients
            # here rather than missing gradients.
            parameters = (
                *self._recent_encoding_parameters(),
                *self.predictive_delta.parameters(),
            )
            zero_with_dep = self._zero_parameter_dependency(zero, parameters)
            zero_metric = zero.detach()
            if not compute_metrics:
                return zero_with_dep, {}
            return zero_with_dep, {
                "history_predictive_valid": zero_metric,
                "history_predictive_loss_raw": zero_metric,
                "history_predictive_cos_loss": zero_metric,
                "history_predictive_mse_loss": zero_metric,
            }

        prefix_tokens, prefix_valid_mask, _ = self.build_history_prefix_batch(
            batch=batch,
            training=False,
            device=device,
            dtype=dtype,
            step_cache=step_cache,
        )
        selected_prefix = prefix_tokens[valid_rows]
        selected_mask = prefix_valid_mask[valid_rows].clone()
        weights = selected_mask.to(dtype=dtype).unsqueeze(-1)
        history_summary = (selected_prefix * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        input_latents = batch["input_latents"].to(device=device, dtype=dtype)
        obs_latent_idx = min(max(int(self.geometry.obs_latents) - 1, 0), int(input_latents.shape[2]) - 1)
        target_latent = input_latents[valid_rows, :, obs_latent_idx:obs_latent_idx + 1]
        with torch.no_grad():
            target_tokens = self.compress_single_frame_latent(target_latent).mean(dim=1)
        pred = self.predictive_delta(history_summary)
        pred_f = F.normalize(pred.float(), dim=-1)
        target_f = F.normalize(target_tokens.float(), dim=-1)
        cos_loss = (1.0 - (pred_f * target_f).sum(dim=-1)).mean()
        mse_loss = F.mse_loss(pred.float(), target_tokens.float())
        loss = cos_loss + 0.1 * mse_loss
        if not compute_metrics:
            return loss.to(dtype=dtype), {}
        return loss.to(dtype=dtype), {
            "history_predictive_valid": torch.tensor(float(selected_prefix.shape[0]), device=device, dtype=dtype),
            "history_predictive_loss_raw": loss.detach().to(dtype=dtype),
            "history_predictive_cos_loss": cos_loss.detach().to(dtype=dtype),
            "history_predictive_mse_loss": mse_loss.detach().to(dtype=dtype),
        }

    def encode_long_sources_batched(
        self,
        latents: torch.Tensor,
        reference_raw: torch.Tensor,
        raw_start: torch.Tensor,
        next_cycle_start_raw: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Long-source path: tokenize + long_source role/norm.

        Returns ``[N, T_seg * compressed_tokens_per_frame, D]`` for direct
        MemoryUpdater consumption.
        """
        tokens = self.tokenize_segments_batched(
            latents=latents,
            reference_raw=reference_raw,
            raw_start=raw_start,
            next_cycle_start_raw=next_cycle_start_raw,
            device=device,
            dtype=dtype,
            chunk_size=chunk_size,
        )
        if tokens.shape[0] == 0:
            return tokens
        return self.long_source_norm(tokens + self.long_source_role_embed)

    def encode_recent_segment(
        self,
        segment: HistorySegment,
        reference_raw: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Single-segment convenience wrapper for the recent path. Returns ``[1, R, D]``."""
        latents = segment.latents.unsqueeze(0).to(device=device, dtype=dtype)
        ref_raw = torch.tensor([reference_raw], device=device, dtype=torch.long)
        raw_start = torch.tensor([segment.raw_start], device=device, dtype=torch.long)
        next_raw = torch.tensor([segment.next_cycle_start_raw], device=device, dtype=torch.long)
        return self.encode_recent_segments_batched(
            latents=latents,
            reference_raw=ref_raw,
            raw_start=raw_start,
            next_cycle_start_raw=next_raw,
            device=device,
            dtype=dtype,
            chunk_size=1,
        )

    def encode_long_source_segment(
        self,
        segment: HistorySegment,
        reference_raw: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Single-segment convenience wrapper for the long-source path.

        Returns ``[1, S, D]`` where S = T_seg * compressed_tokens_per_frame.
        """
        latents = segment.latents.unsqueeze(0).to(device=device, dtype=dtype)
        ref_raw = torch.tensor([reference_raw], device=device, dtype=torch.long)
        raw_start = torch.tensor([segment.raw_start], device=device, dtype=torch.long)
        next_raw = torch.tensor([segment.next_cycle_start_raw], device=device, dtype=torch.long)
        return self.encode_long_sources_batched(
            latents=latents,
            reference_raw=ref_raw,
            raw_start=raw_start,
            next_cycle_start_raw=next_raw,
            device=device,
            dtype=dtype,
            chunk_size=1,
        )

    def build_history_prefix_one_sample(
        self,
        segments: list[HistorySegment],
        current_raw: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build history prefix tokens for one sample by replaying segments.

        Returns ``(history_prefix_tokens [1,H,D], history_valid_mask [1,H],
        long_memory_write_count [1])``.
        """
        cfg = self.cfg
        if len(segments) > cfg.max_history_segments:
            segments = segments[-cfg.max_history_segments:]
        if len(segments) == 0:
            return self._empty_history_prefix(batch_size=1, device=device, dtype=dtype)

        recent_history: list[HistorySegment] = []
        long_memory = torch.zeros(
            1, cfg.memory_num_levels, cfg.memory_tokens_per_level,
            self.hidden_dim, device=device, dtype=dtype,
        )
        long_memory_write_count = torch.zeros(1, dtype=torch.long, device=device)

        for seg in segments:
            recent_history.append(seg)
            overflow_raw = seg.next_cycle_start_raw

            while len(recent_history) > cfg.recent_num_segments:
                old_seg = recent_history.pop(0)

                source_tokens = self.encode_long_source_segment(
                    old_seg,
                    reference_raw=overflow_raw,
                    device=device,
                    dtype=dtype,
                )

                long_memory = self.memory_updater(
                    long_memory,
                    source_tokens,
                    duration_raw=old_seg.next_cycle_start_raw - old_seg.raw_start,
                )
                long_memory_write_count += 1

        # LongMemory readout
        num_mem_tokens = self.long_memory_token_count
        long_memory_tokens = long_memory.flatten(1, 2)  # [1, H_mem, D]

        bucket = bucketize_write_count(long_memory_write_count)
        count_emb = self.write_count_bucket_embed(bucket)  # [1, D]
        long_memory_tokens = long_memory_tokens + count_emb[:, None, :]

        long_memory_valid = (long_memory_write_count > 0).expand(1, num_mem_tokens)  # [1, H_mem]

        # Recent readout
        recent_tokens_list = []
        recent_valid_list = []

        for seg in recent_history:
            tokens = self.encode_recent_segment(
                seg,
                reference_raw=current_raw,
                device=device,
                dtype=dtype,
            )
            recent_tokens_list.append(tokens)
            recent_valid_list.append(True)

        while len(recent_tokens_list) < cfg.recent_num_segments:
            placeholder = torch.zeros(
                1, self.recent_tokens_per_segment, self.hidden_dim,
                device=device, dtype=dtype,
            )
            recent_tokens_list.insert(0, placeholder)
            recent_valid_list.insert(0, False)

        recent_tokens = torch.cat(recent_tokens_list, dim=1)  # [1, H_recent, D]

        recent_valid = torch.tensor(recent_valid_list, dtype=torch.bool, device=device)
        recent_valid_mask = recent_valid.repeat_interleave(
            self.recent_tokens_per_segment,
        ).unsqueeze(0)  # [1, H_recent]

        history_prefix_tokens = torch.cat([long_memory_tokens, recent_tokens], dim=1)
        history_valid_mask = torch.cat([long_memory_valid, recent_valid_mask], dim=1)

        return history_prefix_tokens, history_valid_mask, long_memory_write_count

    def rebuild_long_memory_from_write_queue(
        self,
        write_queue: list[MemoryWriteRecord],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay cached source tokens to rebuild the sliding LongMemory."""
        cfg = self.cfg

        long_memory = torch.zeros(
            1, cfg.memory_num_levels, cfg.memory_tokens_per_level,
            self.hidden_dim, device=device, dtype=dtype,
        )

        for rec in write_queue:
            source_tokens = rec.source_tokens.to(device=device, dtype=dtype)
            long_memory = self.memory_updater(
                long_memory,
                source_tokens,
                duration_raw=rec.duration_raw,
            )

        write_count = torch.tensor([len(write_queue)], dtype=torch.long, device=device)
        return long_memory, write_count

    def _empty_history_prefix(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = torch.zeros(
            batch_size,
            self.history_prefix_token_count,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        mask = torch.zeros(
            batch_size,
            self.history_prefix_token_count,
            device=device,
            dtype=torch.bool,
        )
        write_count = torch.zeros(batch_size, device=device, dtype=torch.long)
        return prefix, mask, write_count

    def _select_training_history_horizon(
        self,
        segments: list[HistorySegment],
        training: bool,
        device: torch.device,
    ) -> list[HistorySegment]:
        if len(segments) > self.cfg.max_history_segments:
            segments = segments[-self.cfg.max_history_segments:]

        if (
            training
            and self.cfg.history_horizon_dropout > 0
            and len(segments) > 0
            and torch.rand((), device=device).item() < self.cfg.history_horizon_dropout
        ):
            keep_segments = int(torch.randint(
                low=0,
                high=len(segments) + 1,
                size=(1,),
                device=device,
                dtype=torch.long,
            ).item())
            if keep_segments == 0:
                return []
            return segments[-keep_segments:]

        return segments

    def _make_segment_token_requests(
        self,
        sample_segments: list[list[HistorySegment]],
        current_raws: list[int],
    ) -> list[SegmentTokenRequest]:
        requests = []
        recent_num = self.cfg.recent_num_segments

        for sample_idx, segments in enumerate(sample_segments):
            num_segments = len(segments)
            num_memory_sources = max(0, num_segments - recent_num)

            for segment_idx in range(num_memory_sources):
                seg = segments[segment_idx]
                reference_seg = segments[segment_idx + recent_num]
                requests.append(SegmentTokenRequest(
                    sample_idx=sample_idx,
                    segment_idx=segment_idx,
                    purpose="memory_source",
                    raw_start=seg.raw_start,
                    next_cycle_start_raw=seg.next_cycle_start_raw,
                    reference_raw=reference_seg.next_cycle_start_raw,
                    latents=seg.latents,
                ))

            recent_start = max(0, num_segments - recent_num)
            for segment_idx in range(recent_start, num_segments):
                seg = segments[segment_idx]
                requests.append(SegmentTokenRequest(
                    sample_idx=sample_idx,
                    segment_idx=segment_idx,
                    purpose="recent_read",
                    raw_start=seg.raw_start,
                    next_cycle_start_raw=seg.next_cycle_start_raw,
                    reference_raw=current_raws[sample_idx],
                    latents=seg.latents,
                ))

        return requests

    def _tokenize_segment_requests(
        self,
        requests: list[SegmentTokenRequest],
        device: torch.device,
        dtype: torch.dtype,
        token_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
    ) -> dict[tuple[int, int, str], torch.Tensor]:
        if not requests:
            return {}
        if token_cache is None:
            token_cache = {}

        def cache_key(req: SegmentTokenRequest) -> tuple[Any, ...]:
            # The cache is scoped to one training step. Tensor identity plus
            # role/reference uniquely identifies the exact deterministic
            # encoding even when horizon dropout renumbers segment indices.
            return (
                req.purpose,
                req.reference_raw,
                req.raw_start,
                req.next_cycle_start_raw,
                req.latents.data_ptr(),
                req.latents.storage_offset(),
            )

        result: dict[tuple[int, int, str], torch.Tensor] = {}
        for purpose, encoder in (
            ("recent_read", self.encode_recent_segments_batched),
            ("memory_source", self.encode_long_sources_batched),
        ):
            purpose_requests = [req for req in requests if req.purpose == purpose]
            missing = [req for req in purpose_requests if cache_key(req) not in token_cache]
            if missing:
                latents = torch.stack([req.latents for req in missing], dim=0)
                reference_raw = torch.tensor(
                    [req.reference_raw for req in missing], dtype=torch.long
                )
                raw_start = torch.tensor(
                    [req.raw_start for req in missing], dtype=torch.long
                )
                next_raw = torch.tensor(
                    [req.next_cycle_start_raw for req in missing], dtype=torch.long
                )
                encoded = encoder(
                    latents=latents,
                    reference_raw=reference_raw,
                    raw_start=raw_start,
                    next_cycle_start_raw=next_raw,
                    device=device,
                    dtype=dtype,
                )
                for idx, req in enumerate(missing):
                    token_cache[cache_key(req)] = encoded[idx:idx + 1]
            for req in purpose_requests:
                result[(req.sample_idx, req.segment_idx, req.purpose)] = token_cache[
                    cache_key(req)
                ]
        return result

    @staticmethod
    def _history_metadata_on_cpu(
        batch: Mapping[str, Any],
        step_cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if step_cache is not None and "metadata_batch" in step_cache:
            return step_cache["metadata_batch"]

        metadata_batch = dict(batch)
        metadata_keys = (
            "history_segment_mask",
            "history_segment_start_raw",
            "history_segment_next_cycle_start_raw",
            "sample_start_raw",
        )
        if all(key in batch for key in metadata_keys):
            metadata_tensors = [
                (
                    batch[key].detach()
                    if torch.is_tensor(batch[key])
                    else torch.as_tensor(batch[key])
                )
                for key in metadata_keys
            ]
            if any(tensor.device.type != "cpu" for tensor in metadata_tensors):
                # One packed D2H synchronization per training step. Both the
                # main prefix and predictive objective reuse this CPU view.
                sizes = [tensor.numel() for tensor in metadata_tensors]
                packed = torch.cat(
                    [tensor.reshape(-1).to(dtype=torch.long) for tensor in metadata_tensors]
                ).to(device="cpu")
                offset = 0
                for key, tensor, size in zip(
                    metadata_keys, metadata_tensors, sizes
                ):
                    metadata_batch[key] = packed[offset:offset + size].reshape(
                        tensor.shape
                    )
                    offset += size
                metadata_batch["history_segment_mask"] = metadata_batch[
                    "history_segment_mask"
                ].to(dtype=torch.bool)
            else:
                for key, tensor in zip(metadata_keys, metadata_tensors):
                    metadata_batch[key] = tensor

        if step_cache is not None:
            step_cache["metadata_batch"] = metadata_batch
        return metadata_batch

    def build_history_prefix_batch(
        self,
        batch: dict,
        training: bool,
        device: torch.device,
        dtype: torch.dtype,
        step_cache: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build history prefix for entire batch.

        Returns ``(history_prefix_tokens [B,H,D], history_valid_mask [B,H],
        long_memory_write_count [B])``.
        """
        sample_segments = []
        current_raws = []
        B = batch["input_latents"].shape[0]

        # Accelerator moves the full batch to the training device. Parse the
        # small integral metadata once on CPU and share it across both history
        # objectives in this step; latent payloads stay on their source device.
        metadata_batch = self._history_metadata_on_cpu(batch, step_cache)

        for b in range(B):
            segments = self._unpack_segments_for_sample(metadata_batch, b, device, dtype)
            segments = self._select_training_history_horizon(
                segments=segments,
                training=training,
                device=torch.device("cpu"),
            )
            sample_segments.append(segments)
            current_raws.append(int(metadata_batch["sample_start_raw"][b].item()))

        requests = self._make_segment_token_requests(sample_segments, current_raws)
        token_cache = (
            None if step_cache is None else step_cache.setdefault("tokens", {})
        )
        if token_cache is None:
            encoded_segments = self._tokenize_segment_requests(
                requests, device=device, dtype=dtype
            )
        else:
            encoded_segments = self._tokenize_segment_requests(
                requests,
                device=device,
                dtype=dtype,
                token_cache=token_cache,
            )
        self._last_num_memory_writes = sum(
            1 for r in requests if r.purpose == "memory_source"
        )

        cfg = self.cfg
        long_memory = torch.zeros(
            B,
            cfg.memory_num_levels,
            cfg.memory_tokens_per_level,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        num_memory_sources = [
            max(0, len(segments) - cfg.recent_num_segments)
            for segments in sample_segments
        ]
        long_memory_write_count = torch.tensor(
            num_memory_sources, dtype=torch.long, device=device,
        )
        self._last_memory_updater_batch_sizes = []

        # Replay only real overflow writes: each iteration attends over
        # just the samples that still have memory sources at this index,
        # so batches without overflow skip the attention update entirely.
        max_memory_sources = max(num_memory_sources, default=0)
        for source_idx in range(max_memory_sources):
            active_indices = [
                b for b, count in enumerate(num_memory_sources)
                if source_idx < count
            ]
            source_tokens = []
            source_durations = []
            for b in active_indices:
                seg = sample_segments[b][source_idx]
                source_tokens.append(encoded_segments[(b, source_idx, "memory_source")])
                source_durations.append(seg.next_cycle_start_raw - seg.raw_start)

            source_tokens_b = torch.cat(source_tokens, dim=0)
            source_durations_t = torch.tensor(source_durations, dtype=torch.long, device=device)
            active_indices_t = torch.tensor(active_indices, dtype=torch.long, device=device)
            updated_memory = self.memory_updater(
                long_memory.index_select(0, active_indices_t),
                source_tokens_b,
                duration_raw=source_durations_t,
            )
            long_memory = torch.index_copy(
                long_memory, 0, active_indices_t, updated_memory,
            )
            self._last_memory_updater_batch_sizes.append(len(active_indices))

        num_mem_tokens = self.long_memory_token_count
        long_memory_tokens = long_memory.flatten(1, 2)
        if max_memory_sources == 0:
            long_memory_tokens = long_memory_tokens + self._zero_parameter_dependency(
                long_memory_tokens,
                self._long_memory_parameters(),
            )

        # Any non-empty history produces a recent request, so these parameters
        # are naturally connected on ordinary steps. Only the truly empty case
        # needs an explicit dependency.
        if not requests:
            long_memory_tokens = long_memory_tokens + self._zero_parameter_dependency(
                long_memory_tokens,
                self._recent_encoding_parameters(),
            )

        bucket = bucketize_write_count(long_memory_write_count)
        count_emb = self.write_count_bucket_embed(bucket)
        long_memory_tokens = long_memory_tokens + count_emb[:, None, :]
        long_memory_valid = (long_memory_write_count > 0).unsqueeze(1).expand(B, num_mem_tokens)

        recent_tokens_per_sample = []
        recent_masks = []
        for b, segments in enumerate(sample_segments):
            recent_tokens_list = []
            recent_valid_list = []
            recent_start = max(0, len(segments) - cfg.recent_num_segments)
            for segment_idx in range(recent_start, len(segments)):
                recent_tokens_list.append(
                    encoded_segments[(b, segment_idx, "recent_read")]
                )
                recent_valid_list.append(True)

            while len(recent_tokens_list) < cfg.recent_num_segments:
                recent_tokens_list.insert(0, torch.zeros(
                    1,
                    self.recent_tokens_per_segment,
                    self.hidden_dim,
                    device=device,
                    dtype=dtype,
                ))
                recent_valid_list.insert(0, False)

            recent_tokens_per_sample.append(torch.cat(recent_tokens_list, dim=1))
            recent_valid = torch.tensor(recent_valid_list, dtype=torch.bool, device=device)
            recent_masks.append(
                recent_valid.repeat_interleave(self.recent_tokens_per_segment).unsqueeze(0)
            )

        recent_tokens = torch.cat(recent_tokens_per_sample, dim=0)
        recent_valid_mask = torch.cat(recent_masks, dim=0)
        history_prefix_tokens = torch.cat([long_memory_tokens, recent_tokens], dim=1)
        history_valid_mask = torch.cat([long_memory_valid, recent_valid_mask], dim=1)

        return history_prefix_tokens, history_valid_mask, long_memory_write_count

    def _unpack_segments_for_sample(
        self,
        batch: dict,
        b: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[HistorySegment]:
        """Unpack HistorySegments from batch tensors for sample b."""
        seg_latents = batch["history_segment_latents"]  # [B, max_depth, C, T_seg, H, W]
        seg_mask = batch["history_segment_mask"]         # [B, max_depth]
        seg_start_raw = batch["history_segment_start_raw"]  # [B, max_depth]
        seg_next_raw = batch["history_segment_next_cycle_start_raw"]  # [B, max_depth]

        segments = []
        max_depth = seg_mask.shape[1]
        for d in range(max_depth):
            if seg_mask[b, d].item():
                segments.append(HistorySegment(
                    latents=seg_latents[b, d],  # [C, T_seg, H, W]
                    raw_start=int(seg_start_raw[b, d].item()),
                    next_cycle_start_raw=int(seg_next_raw[b, d].item()),
                ))
        return segments

    def build_prefix_from_state(
        self,
        recent_history: list[HistorySegment],
        long_memory: torch.Tensor,
        long_memory_write_count: torch.Tensor,
        current_raw: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build history prefix from live inference state.

        Returns:
            history_prefix_tokens: [1, H, D]
            history_valid_mask: [1, H]
        """
        cfg = self.cfg
        if len(recent_history) > cfg.recent_num_segments:
            recent_history = recent_history[-cfg.recent_num_segments:]

        # LongMemory readout
        num_mem_tokens = self.long_memory_token_count
        long_memory_tokens = long_memory.flatten(1, 2)  # [1, H_mem, D]

        bucket = bucketize_write_count(long_memory_write_count)
        count_emb = self.write_count_bucket_embed(bucket)
        long_memory_tokens = long_memory_tokens + count_emb[:, None, :]

        long_memory_valid = (long_memory_write_count > 0).expand(1, num_mem_tokens)

        # Recent readout
        recent_tokens_list = []
        recent_valid_list = []

        for seg in recent_history:
            tokens = self.encode_recent_segment(
                seg,
                reference_raw=current_raw,
                device=device,
                dtype=dtype,
            )
            recent_tokens_list.append(tokens)
            recent_valid_list.append(True)

        while len(recent_tokens_list) < cfg.recent_num_segments:
            placeholder = torch.zeros(
                1, self.recent_tokens_per_segment, self.hidden_dim,
                device=device, dtype=dtype,
            )
            recent_tokens_list.insert(0, placeholder)
            recent_valid_list.insert(0, False)

        recent_tokens = torch.cat(recent_tokens_list, dim=1)

        recent_valid = torch.tensor(recent_valid_list, dtype=torch.bool, device=device)
        recent_valid_mask = recent_valid.repeat_interleave(
            self.recent_tokens_per_segment,
        ).unsqueeze(0)

        history_prefix_tokens = torch.cat([long_memory_tokens, recent_tokens], dim=1)
        history_valid_mask = torch.cat([long_memory_valid, recent_valid_mask], dim=1)

        return history_prefix_tokens, history_valid_mask

    def push_history_segment_state(
        self,
        state: HistoryState,
        history_segment: HistorySegment,
        device: torch.device,
        dtype: torch.dtype,
    ) -> HistoryState:
        """Push a segment into inference HistoryState with a sliding source-token queue."""
        cfg = self.cfg
        state.recent_history.append(history_segment)
        overflow_raw = history_segment.next_cycle_start_raw

        while len(state.recent_history) > cfg.recent_num_segments:
            old_seg = state.recent_history.pop(0)

            source_tokens = self.encode_long_source_segment(
                old_seg,
                reference_raw=overflow_raw,
                device=device,
                dtype=dtype,
            )

            state.memory_write_queue.append(MemoryWriteRecord(
                source_tokens=source_tokens.detach(),
                duration_raw=old_seg.next_cycle_start_raw - old_seg.raw_start,
                raw_start=old_seg.raw_start,
                next_cycle_start_raw=old_seg.next_cycle_start_raw,
                reference_raw=overflow_raw,
            ))

        capacity = cfg.memory_write_capacity_segments
        if capacity == 0:
            state.memory_write_queue = []
        elif len(state.memory_write_queue) > capacity:
            state.memory_write_queue = state.memory_write_queue[-capacity:]

        state.long_memory, state.long_memory_write_count = self.rebuild_long_memory_from_write_queue(
            state.memory_write_queue,
            device=device,
            dtype=dtype,
        )
        return state
