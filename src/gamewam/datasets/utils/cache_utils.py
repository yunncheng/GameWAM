import hashlib
from typing import Any

from omegaconf import OmegaConf


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def resolve_image_video_size(shape_meta: Any) -> list[int]:
    shape_meta = _plain(shape_meta)
    images = shape_meta.get("images") if isinstance(shape_meta, dict) else None
    if not images:
        raise ValueError("`shape_meta.images` must contain at least one image entry.")
    shape = images[0].get("shape")
    if shape is None or len(shape) != 3:
        raise ValueError("`shape_meta.images[0].shape` must be [C, H, W].")
    return [int(shape[1]), int(shape[2])]


def compute_vae_cache_hash(
    *,
    video_size: list[int],
    num_frames: int,
    frame_stride: int,
    sample_interval: int,
    action_video_freq_ratio: int,
    event_anchor_enabled: bool,
    event_anchor_radius: Any,
    sample_interval_event: Any,
    sample_interval_normal: Any,
    latent_dtype: str,
) -> str:
    key = (
        f"vs={list(video_size)}_nf={num_frames}_fs={frame_stride}_si={sample_interval}"
        f"_avr={action_video_freq_ratio}_ea={bool(event_anchor_enabled)}"
        f"_ear={event_anchor_radius}_sie={sample_interval_event}_sin={sample_interval_normal}"
        f"_dtype={latent_dtype}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def compute_far_latent_cache_hash(
    *,
    video_size: list[int],
    action_video_freq_ratio: int,
    latent_dtype: str,
) -> str:
    """Hash for per-episode-frame far latent cache.

    Far cache is per-episode and per-video-frame (independent of sample_interval,
    event_anchor, num_frames etc.), so only video_size, ratio, and dtype matter.
    """
    key = (
        f"far_vs={list(video_size)}_avr={action_video_freq_ratio}"
        f"_dtype={latent_dtype}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
