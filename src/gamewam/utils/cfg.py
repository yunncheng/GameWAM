import math
from pathlib import Path
from typing import Any, Callable, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


# cfg conversion (runtime side)

def to_plain_dict(name: str, value, *, default=None, required: bool = False) -> dict:
    """Convert OmegaConf DictConfig to a plain dict at the runtime boundary.

    All downstream constructors expect Python dicts, not DictConfig.
    """
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required.")
        return {} if default is None else dict(default)
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def merge_common_and_split(common_cfg, split_cfg) -> dict:
    """Flatten a Hydra three-level ``common/train/val`` config into one dict.

    Common values are overwritten by split-specific values, so the result is
    ready to pass to Hydra's ``instantiate``.
    """
    merged = to_plain_dict("common_cfg", common_cfg, default={})
    if split_cfg is not None:
        merged.update(to_plain_dict("split_cfg", split_cfg))
    return merged


# OmegaConf resolvers (registered into the Hydra config layer)

def _register(name: str, func: Callable) -> None:
    """Idempotently register a resolver, replacing any existing one."""
    OmegaConf.register_new_resolver(name, func, replace=True)


def _oc_load(path: str, key: Optional[str] = None) -> Any:
    """Load a YAML/JSON config and optionally select a key.

    Uses Hydra's ``to_absolute_path`` to honor the original working directory.
    """
    from hydra.utils import to_absolute_path

    load_path = Path(path)
    if not load_path.is_absolute():
        load_path = Path(to_absolute_path(path))
    cfg = OmegaConf.load(load_path)
    if key is None or key == "":
        return cfg
    return OmegaConf.select(cfg, key)


def sum_shapes(shape_meta_list):
    if not shape_meta_list:
        return 0
    total = sum(int(item["shape"]) for item in shape_meta_list if item["key"] is not None)
    return total


def max_action_dim(embodiment_datasets_cfg):
    max_dim = 0
    for dataset_name, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            action_cfg = dataset_cfg.shape_meta.action
            current_dim = sum_shapes(action_cfg)

            if current_dim > max_dim:
                max_dim = current_dim

    return max_dim


def max_state_dim(embodiment_datasets_cfg):
    max_dim = 0
    for dataset_name, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            state_cfg = dataset_cfg.shape_meta.state
            current_dim = sum_shapes(state_cfg)
            if current_dim > max_dim:
                max_dim = current_dim

    return max_dim


def _contains_key_recursive(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return any(
            str(key) == forbidden_key or _contains_key_recursive(item, forbidden_key)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key_recursive(item, forbidden_key) for item in value)
    return False


def validate_horizon_config(cfg) -> None:
    """Validate the single-source horizon contract.

    ``prediction_horizon`` and ``execution_horizon`` live under
    ``data.common``. The disabled horizon aliases are rejected anywhere
    in the resolved config so invalid runtime/CLI overrides fail before
    construction.
    """
    if cfg is None:
        return
    for forbidden in ("replan_steps", "effective_action_horizon", "action_horizon"):
        if _contains_key_recursive(cfg, forbidden):
            raise ValueError(
                f"Unsupported horizon alias `{forbidden}`. "
                "Use `data.common.prediction_horizon` and "
                "`data.common.execution_horizon` instead."
            )

    data_common = None
    if isinstance(cfg, DictConfig):
        data_common = OmegaConf.select(cfg, "data.common")
    elif isinstance(cfg, dict):
        data_common = cfg.get("data", {}).get("common")
    if data_common is None:
        return

    if isinstance(data_common, DictConfig):
        data_common = OmegaConf.to_container(data_common, resolve=True)
    prediction_horizon = int(data_common.get("prediction_horizon"))
    execution_horizon = int(data_common.get("execution_horizon"))
    if prediction_horizon <= 0:
        raise ValueError(f"prediction_horizon must be > 0, got {prediction_horizon}")
    if execution_horizon <= 0:
        raise ValueError(f"execution_horizon must be > 0, got {execution_horizon}")
    if execution_horizon > prediction_horizon:
        raise ValueError(
            "execution_horizon must not exceed prediction_horizon, "
            f"got {execution_horizon} > {prediction_horizon}"
        )

    num_frames = int(data_common.get("num_frames"))
    num_obs_frames = int(data_common.get("num_obs_frames"))
    frame_stride = int(data_common.get("frame_stride", 1))
    committed_future_horizon = (num_frames - num_obs_frames) * frame_stride
    # AR mode is a property of the video DiT attention mask, not of whether
    # the optional history module is enabled. Using `history is not None`
    # would misclassify an AR model trained without history as Non-AR.
    mask_mode = None
    if isinstance(cfg, DictConfig):
        mask_mode = OmegaConf.select(
            cfg, "model.video_dit_config.video_attention_mask_mode"
        )
    elif isinstance(cfg, dict):
        mask_mode = (
            cfg.get("model", {})
            .get("video_dit_config", {})
            .get("video_attention_mask_mode")
        )
    is_ar = str(mask_mode) == "ar_teacher_forcing"
    # The history module is an AR-only enhancement. Reject the
    # Non-AR + history combination loudly instead of letting the model
    # silently ignore the configured history module.
    # Skip when mask_mode is absent (data-only configs without model section).
    has_history = data_common.get("history") is not None
    if mask_mode is not None and not is_ar and has_history:
        raise ValueError(
            "Non-AR model (model.video_dit_config.video_attention_mask_mode "
            f"= {mask_mode!r}) does not support `data.common.history`. "
            "Either remove `history` from the data config or switch to "
            "`video_attention_mask_mode: ar_teacher_forcing`."
        )
    # Treat history config as an AR indicator when the model section is
    # absent from a data-only configuration.
    effective_ar = is_ar or (mask_mode is None and has_history)
    if effective_ar:
        if prediction_horizon % execution_horizon != 0:
            raise ValueError(
                "AR prediction_horizon must be divisible by execution_horizon, "
                f"got {prediction_horizon} and {execution_horizon}"
            )
        action_video_freq_ratio = int(data_common.get("action_video_freq_ratio", 1))
        raw_steps_per_latent = 4 * action_video_freq_ratio * frame_stride
        if execution_horizon != raw_steps_per_latent:
            raise ValueError(
                "AR execution_horizon must equal raw_steps_per_latent, "
                f"got execution_horizon={execution_horizon}, "
                f"raw_steps_per_latent={raw_steps_per_latent}"
            )
        if committed_future_horizon % execution_horizon != 0:
            raise ValueError(
                "AR committed future horizon must be divisible by execution_horizon, "
                f"got committed={committed_future_horizon}, execution={execution_horizon}"
            )
        if (prediction_horizon - execution_horizon) % frame_stride != 0:
            raise ValueError(
                "AR lookahead must align with frame_stride, got "
                f"prediction_horizon={prediction_horizon}, "
                f"execution_horizon={execution_horizon}, frame_stride={frame_stride}"
            )
    elif committed_future_horizon != prediction_horizon:
        raise ValueError(
            "Non-AR geometry must satisfy "
            "(num_frames - num_obs_frames) * frame_stride == prediction_horizon, "
            f"got {committed_future_horizon} != {prediction_horizon}"
        )


def register_default_resolvers() -> None:
    """Register all resolvers commonly used across entrypoints.

    Safe to call multiple times.
    """
    _register("oc.load", _oc_load)
    _register("eval", eval)
    _register("split", lambda s, idx: s.split("/")[int(idx)])
    _register("max", lambda x: max(x))
    _register("round_up", math.ceil)
    _register("round_down", math.floor)
    _register("sum_shapes", sum_shapes)
    _register("max_action_dim", max_action_dim)
    _register("max_state_dim", max_state_dim)
