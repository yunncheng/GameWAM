"""Template + child override materialization for mixed dataset configs.

Entry point for training and precompute paths — see
``materialize_dataset_nodes`` for the canonical usage pattern.
"""

import copy
import logging
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# ── override rules ─────────────────────────────────────────────────

# Fields a child override MAY set.
_ALLOWED_OVERRIDE_FIELDS = frozenset({
    "dataset_dirs",
    "shape_meta",
    "text_embedding_cache_dir",
    "vae_latent_cache_dir",
    "sample_interval",
    "event_anchor_enabled",
    "event_anchor_radius",
    "sample_interval_event",
    "sample_interval_normal",
    "fine_task_resample_enabled",
    "fine_task_resample_mix_original",
    "fine_task_resample_gamma",
    "fine_task_resample_max_expected_repeat",
    "processor",
})

# Fields a child override MUST provide (non-null).
_REQUIRED_OVERRIDE_FIELDS = frozenset({
    "dataset_dirs",
    "text_embedding_cache_dir",
    "vae_latent_cache_dir",
    "sample_interval",
    "event_anchor_enabled",
    "fine_task_resample_enabled",
})

# Fields NEVER allowed in child override.
_FORBIDDEN_OVERRIDE_FIELDS = frozenset({
    "_target_",
    "num_frames",
    "num_obs_frames",
    "frame_stride",
    "action_video_freq_ratio",
    "val_set_proportion",
    "skip_padding_as_possible",
    "concat_multi_camera",
    "context_len",
    "history",
    "vae_latent_cache_dtype",
    "prompt_template",
})

# Inside shape_meta.images, only raw_shape may differ per child.
_ALLOWED_IMAGE_OVERRIDE_FIELDS = frozenset({"raw_shape"})

# Inside processor, only image transforms may be overridden.
_ALLOWED_PROCESSOR_OVERRIDE_FIELDS = frozenset({
    "train_transforms", "val_transforms",
})


def _validate_override(name: str, override: dict) -> None:
    """Raise if override violates allowed / required / forbidden rules."""
    unknown = set(override.keys()) - _ALLOWED_OVERRIDE_FIELDS
    if unknown:
        raise ValueError(
            f"Child override '{name}' has unknown fields: {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_OVERRIDE_FIELDS)}"
        )
    forbidden = set(override.keys()) & _FORBIDDEN_OVERRIDE_FIELDS
    if forbidden:
        raise ValueError(
            f"Child override '{name}' has forbidden fields: {sorted(forbidden)}"
        )
    # required keys
    for key in _REQUIRED_OVERRIDE_FIELDS:
        if key not in override or override[key] is None:
            raise ValueError(
                f"Child override '{name}' is missing required field: {key}"
            )
    # shape_meta may only set images.*.raw_shape
    if "shape_meta" in override:
        sm = override["shape_meta"]
        if not isinstance(sm, dict):
            raise TypeError(f"Child override '{name}' shape_meta must be dict, got {type(sm)}")
        forbidden_shape = set(sm.keys()) - {"images"}
        if forbidden_shape:
            raise ValueError(
                f"Child override '{name}' shape_meta may only contain 'images', got {sorted(forbidden_shape)}"
            )
        for idx, img_meta in enumerate(sm.get("images", [])):
            forbidden_img = set(img_meta.keys()) - _ALLOWED_IMAGE_OVERRIDE_FIELDS
            if forbidden_img:
                raise ValueError(
                    f"Child override '{name}' shape_meta.images[{idx}] may only contain "
                    f"{sorted(_ALLOWED_IMAGE_OVERRIDE_FIELDS)}, got {sorted(forbidden_img)}"
                )
    # processor may only override image transforms
    if "processor" in override:
        proc = override["processor"]
        if not isinstance(proc, dict):
            raise TypeError(f"Child override '{name}' processor must be dict, got {type(proc)}")
        forbidden_proc = set(proc.keys()) - _ALLOWED_PROCESSOR_OVERRIDE_FIELDS
        if forbidden_proc:
            raise ValueError(
                f"Child override '{name}' processor may only contain "
                f"{sorted(_ALLOWED_PROCESSOR_OVERRIDE_FIELDS)}, got {sorted(forbidden_proc)}"
            )


def _validate_cache_uniqueness(nodes: List[Tuple[str, dict]]) -> None:
    """Raise if any two children share the same cache directory."""
    for kind_name, cache_key in [
        ("text embedding", "text_embedding_cache_dir"),
        ("VAE latent", "vae_latent_cache_dir"),
    ]:
        seen: Dict[str, str] = {}
        for name, node in nodes:
            val = node.get(cache_key)
            if val is not None:
                val = str(val)
                if val in seen:
                    raise ValueError(
                        f"Child '{name}' and '{seen[val]}' share the same {kind_name} cache: {val}"
                    )
                seen[val] = name


def materialize_dataset_nodes(
    data_cfg: Any, split: str = "train"
) -> List[Tuple[str, dict]]:
    """Materialize complete dataset nodes from data config.

    Regular config: returns one merged common+split node.
    Mixed template config: returns one complete GameVideoDataset node per child.
    """
    if data_cfg is None:
        return []

    child_cfg = None
    if hasattr(data_cfg, "get"):
        child_cfg = data_cfg.get("child")

    if child_cfg is None or not child_cfg:
        return _materialize_regular(data_cfg, split)
    return _materialize_mixed(data_cfg, split)


def _materialize_regular(data_cfg: Any, split: str) -> List[Tuple[str, dict]]:
    common = data_cfg.get("common") if hasattr(data_cfg, "get") else None
    if common is None:
        return []
    common_dict = _ensure_container(common)
    split_cfg = data_cfg.get(split)
    split_dict = _ensure_container(split_cfg) if split_cfg is not None else {}
    merged = {**common_dict, **split_dict}
    if merged.get("_target_") is not None and merged.get("dataset_dirs") is not None:
        return [("data", merged)]
    return []


def _materialize_mixed(data_cfg: Any, split: str) -> List[Tuple[str, dict]]:
    common = _ensure_container(data_cfg.get("common", {}))
    child_cfg = data_cfg["child"]

    template = copy.deepcopy(common)
    template.pop("_target_", None)
    template.pop("_recursive_", None)

    nodes: List[Tuple[str, dict]] = []
    for name, child_node in child_cfg.items():
        child_dict = _ensure_container(child_node)
        if not isinstance(child_dict, dict):
            continue  # skip scalars like max_expected_repeat
        weight = child_dict.get("weight")
        override = child_dict.get("override", {})

        _validate_override(name, override)

        node = copy.deepcopy(template)

        # Merge simple overrides
        node.update({
            k: v for k, v in override.items()
            if k not in ("shape_meta", "processor")
        })

        # Strip common-level flags that must not flow to the
        # per-child ``GameVideoDataset`` constructor (the mixed
        # dataset reads them from ``common`` directly).
        for common_only_key in ("route_aware_action_normalization",):
            node.pop(common_only_key, None)

        # shape_meta override: merge images.*.raw_shape by index
        if "shape_meta" in override:
            for idx, img_meta in enumerate(override["shape_meta"].get("images", [])):
                raw = img_meta.get("raw_shape")
                if raw is not None:
                    node["shape_meta"]["images"][idx]["raw_shape"] = raw
                    node["processor"]["shape_meta"]["images"][idx]["raw_shape"] = raw

        # processor override: merge image transforms
        if "processor" in override:
            proc = override["processor"]
            if "train_transforms" in proc:
                node["processor"]["train_transforms"] = proc["train_transforms"]
            if "val_transforms" in proc:
                node["processor"]["val_transforms"] = proc["val_transforms"]

        node["_target_"] = "gamewam.datasets.game_video_dataset.GameVideoDataset"
        node["is_training_set"] = (split == "train")

        logger.info(
            "Materialized child '%s': weight=%s, dataset_dirs=%s, sample_interval=%s",
            name, weight, node.get("dataset_dirs"), node.get("sample_interval"),
        )
        nodes.append((name, node))

    _validate_cache_uniqueness(nodes)
    return nodes


def _ensure_container(cfg: Any) -> Any:
    """Resolve OmegaConf/DictConfig to plain container if needed."""
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg
