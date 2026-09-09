import logging
import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from .inference import GameWAMInference
from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.cfg import to_plain_dict, merge_common_and_split, validate_horizon_config
from .utils.misc import get_work_dir, register_work_dir, normalize_mixed_precision, mixed_precision_to_model_dtype
from .utils.pytorch_utils import resolve_train_device, set_global_seed
from .datasets.materialize import materialize_dataset_nodes, _ensure_container
from .datasets.utils.normalizer import load_dataset_stats_from_json

logger = get_logger(__name__)


def _check_route_aware_consistency(cfg) -> None:
    """Fail-fast guard: the data processor's
    ``route_aware_action_normalization`` flag must match the model
    config's ``action_head.route_aware_action_normalization``.

    The two are coupled: the data side computes per-route stats, and
    the model records the flag in checkpoint metadata. If they
    disagree, the processor per-route-normalizes the action but the
    model silently records ``route_aware=False`` in the checkpoint and the
    route-aware chain breaks. Both flags live in the same Hydra
    config tree, so the check just compares the two paths after
    compose. The model config's flag is normally a Hydra
    interpolation of the data flag; the guard exists for the
    mismatch case (hard-coded model value, or model config without
    the field).
    """
    data_flag = bool(
        _get_cfg_path(cfg, "data.common.processor.route_aware_action_normalization", False)
    )
    model_flag = bool(
        _get_cfg_path(cfg, "model.action_head.route_aware_action_normalization", False)
    )
    gui_head_enabled = bool(
        _get_cfg_path(cfg, "model.action_head.gui_head_enabled", False)
    )
    if data_flag != model_flag:
        raise ValueError(
            "Route-aware normalization flag mismatch between "
            "data and model configs. The data processor's "
            "`route_aware_action_normalization` "
            f"= {data_flag}, but the model config's "
            f"`action_head.route_aware_action_normalization` "
            f"= {model_flag}. Both must agree. The model "
            "config's field is normally a Hydra interpolation "
            "of the data field "
            "(${data.common.processor.route_aware_action_normalization}); "
            "if you hard-coded a value in the model config "
            "remove the hard-coded value and let the "
            "interpolation pull the data flag. If your model "
            "config doesn't have the field at all, this "
            "model is not yet wired up for route-aware "
            "normalization — pair it with a route-aware-capable "
            "data config (or vice versa) deliberately."
        )
    if data_flag and not gui_head_enabled:
        raise ValueError(
            "`route_aware_action_normalization=True` requires "
            "`model.action_head.gui_head_enabled=True`; inference needs the "
            "GUI route head to produce `action_route_mask`."
        )


def _get_cfg_path(cfg, dotted_path: str, default=None):
    """Read a dotted path from a Hydra/OmegaConf config, with
    a default fallback. Avoids hard-coding nested ``.get`` chains
    in the consistency check.
    """
    cur = cfg
    for part in dotted_path.split("."):
        if cur is None:
            return default
        if isinstance(cur, (list, tuple, ListConfig)):
            try:
                index = int(part)
            except (TypeError, ValueError):
                return default
            if index < 0 or index >= len(cur):
                return default
            cur = cur[index]
            continue
        # ``.get`` doesn't raise on missing keys; ``__contains__`` is
        # the DictConfig check.
        if hasattr(cur, "get") and part in cur:
            cur = cur.get(part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    if cur is None:
        return default
    return cur


def _validate_training_proprio_schema(cfg: DictConfig) -> None:
    """Training supports only Minecraft cursor for Minecraft grouped proprio."""
    encoder_type = _get_cfg_path(
        cfg, "data.common.processor.proprio_encoder_type"
    )
    supported = {
        None,
        "generic_linear",
        "generic_masked_linear",
        "minecraft_grouped_cursor",
    }
    if encoder_type not in supported:
        raise ValueError(
            f"Unsupported proprio_encoder_type={encoder_type!r}; expected "
            "'generic_linear', 'generic_masked_linear', or "
            "'minecraft_grouped_cursor'."
        )
    if encoder_type != "minecraft_grouped_cursor":
        return
    raw_dim = int(_get_cfg_path(cfg, "data.common.shape_meta.state.0.raw_shape", -1))
    canonical_dim = int(
        _get_cfg_path(cfg, "data.common.processor.proprio_output_dim", -1)
    )
    gui_index = int(
        _get_cfg_path(cfg, "data.common.processor.gui_label_dim_index", -1)
    )
    reference_size = _get_cfg_path(
        cfg, "data.common.processor.cursor_reference_size"
    )
    if (raw_dim, canonical_dim, gui_index) != (6, 15, 5):
        raise ValueError(
            "Minecraft cursor training requires raw/canonical/gui-index "
            f"(6, 15, 5), got {(raw_dim, canonical_dim, gui_index)}."
        )
    if (
        reference_size is None
        or len(reference_size) != 2
        or float(reference_size[0]) <= 0
        or float(reference_size[1]) <= 0
    ):
        raise ValueError(
            "Minecraft cursor training requires "
            "data.common.processor.cursor_reference_size=[width, height]."
        )


def _bind_model_proprio_runtime_contract(model, common_cfg) -> None:
    """Attach processor-owned schema metadata before save/load validation."""
    processor_cfg = (
        None if common_cfg is None else common_cfg.get("processor")
    )
    reference_size = (
        None if processor_cfg is None else processor_cfg.get("cursor_reference_size")
    )
    model.cursor_reference_size = (
        None
        if reference_size is None
        else (float(reference_size[0]), float(reference_size[1]))
    )


def create_gamewam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    proprio_encoder_type: str = "generic_linear",
    proprio_field_valid_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    action_head=None,
    action_noise_share_across_blocks: bool = False,
    mot_checkpoint_mixed_attn: bool = True,
    video_action_attention_mode: str = "condition_only",
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    history_config=None,
    num_frames: int = 33,
    num_obs_frames: int = 17,
    frame_stride: int = 1,
    action_video_freq_ratio: int = 2,
    prediction_horizon: int = 16,
    execution_horizon: int = 8,
    text_condition_dropout_prob: float = 0.0,
    prompt_template=None,
    coupled_video_action_timesteps: bool = True,
    event_occurrence_aux_enabled: bool = False,
    event_occurrence_aux_loss_weight: float = 0.01,
    image_size: Optional[Sequence[int]] = None,
):
    from .models.gamewam import GameWAM

    video_dit_config = to_plain_dict("video_dit_config", video_dit_config, required=True)
    video_ratio = video_dit_config.get("action_video_freq_ratio", None)
    if video_ratio is not None and int(video_ratio) != int(action_video_freq_ratio):
        raise ValueError(
            "model.video_dit_config.action_video_freq_ratio must match "
            "model.action_video_freq_ratio when both are set: "
            f"{video_ratio} != {action_video_freq_ratio}"
        )
    video_dit_config["action_video_freq_ratio"] = int(action_video_freq_ratio)
    action_dit_config = to_plain_dict("action_dit_config", action_dit_config, default={})
    video_scheduler = to_plain_dict("video_scheduler", video_scheduler, default={})
    loss = to_plain_dict("loss", loss, default={})
    action_head_cfg = to_plain_dict("action_head", action_head, default={})

    action_scheduler = to_plain_dict("action_scheduler", action_scheduler, required=True)
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    # Action-head configuration.
    dim_is_continuous = action_head_cfg.get("dim_is_continuous", None)
    gui_head_enabled = bool(action_head_cfg.get("gui_head_enabled", False))
    gui_label_dim_index = action_head_cfg.get("gui_label_dim_index", None)
    lambda_continuous = float(action_head_cfg.get("lambda_continuous", 1.0))
    lambda_discrete = float(action_head_cfg.get("lambda_discrete", 1.0))
    lambda_gui_route = float(action_head_cfg.get("lambda_gui_route", 0.05))
    # ``route_aware_action_normalization`` lives in the ``action_head``
    # config block (mirroring the data processor's flag). The two must
    # agree, otherwise the checkpoint's ``action_normalization`` block
    # records ``route_aware=false`` regardless. Resolving here keeps
    # the model config self-contained (no need to plumb ``data_cfg``
    # through ``create_gamewam``).
    route_aware_action_normalization = bool(
        action_head_cfg.get("route_aware_action_normalization", False)
    )
    noise_sampling_keys = {
        "share_across_blocks",
        "action_noise_share_across_blocks",
        "action_ordinary_block_noise_shared",
    }
    misplaced_noise_keys = sorted(
        k for k in noise_sampling_keys if k in action_head_cfg
    )
    if misplaced_noise_keys:
        raise ValueError(
            "Action noise sampling fields must live under "
            "`model.action_noise_share_across_blocks`, not "
            f"`model.action_head`: {misplaced_noise_keys}"
        )
    action_noise_share_across_blocks = bool(action_noise_share_across_blocks)

    if history_config is not None:
        history_config = to_plain_dict("history_config", history_config)

    model = GameWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        proprio_encoder_type=str(proprio_encoder_type),
        proprio_field_valid_dim=(
            None
            if proprio_field_valid_dim is None
            else int(proprio_field_valid_dim)
        ),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_action_attention_mode=str(video_action_attention_mode),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        history_config=history_config,
        num_frames=int(num_frames),
        num_obs_frames=int(num_obs_frames),
        frame_stride=int(frame_stride),
        action_video_freq_ratio=int(action_video_freq_ratio),
        prediction_horizon=int(prediction_horizon),
        execution_horizon=int(execution_horizon),
        text_condition_dropout_prob=float(text_condition_dropout_prob),
        prompt_template=prompt_template,
        coupled_video_action_timesteps=bool(coupled_video_action_timesteps),
        event_occurrence_aux_enabled=bool(event_occurrence_aux_enabled),
        event_occurrence_aux_loss_weight=float(event_occurrence_aux_loss_weight),
        action_head_dim_is_continuous=dim_is_continuous,
        action_head_gui_head_enabled=gui_head_enabled,
        action_head_gui_label_dim_index=gui_label_dim_index,
        action_head_lambda_continuous=lambda_continuous,
        action_head_lambda_discrete=lambda_discrete,
        action_head_lambda_gui_route=lambda_gui_route,
        action_head_route_aware_action_normalization=route_aware_action_normalization,
        action_noise_share_across_blocks=action_noise_share_across_blocks,
        image_size=image_size,
    )

    return model


def build_datasets(
    data_cfg: DictConfig,
    *,
    build_val: bool = True,
    ar_mode: Optional[bool] = None,
):
    common_cfg = data_cfg.get("common") if hasattr(data_cfg, "get") else None

    train_nodes = materialize_dataset_nodes(data_cfg, split="train")
    val_nodes = (
        materialize_dataset_nodes(data_cfg, split="val")
        if build_val
        else []
    )

    is_mixed = not (len(train_nodes) == 1 and train_nodes[0][0] == "data")

    train_dict = merge_common_and_split(common_cfg, data_cfg.get("train"))
    if is_mixed:
        train_dict["child"] = {
            name: {**node, **({"ar_mode": bool(ar_mode)} if ar_mode is not None else {})}
            for name, node in train_nodes
        }
        raw_child = _ensure_container(data_cfg.get("child", {}))
        train_dict["weights"] = {
            name: float(raw_child[name]["weight"])
            for name, _ in train_nodes
        }
        train_dict["max_expected_repeat"] = raw_child.get("max_expected_repeat", 10.0)
        train_dict["replacement"] = raw_child.get("replacement", False)
    elif ar_mode is not None:
        train_dict["ar_mode"] = bool(ar_mode)
    train_ds = instantiate(train_dict)

    if not build_val:
        val_ds = None
    elif data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        val_dict = merge_common_and_split(common_cfg, data_cfg.get("val"))
        if is_mixed:
            val_dict["child"] = {
                name: {**node, **({"ar_mode": bool(ar_mode)} if ar_mode is not None else {})}
                for name, node in val_nodes
            }
            raw_child = _ensure_container(data_cfg.get("child", {}))
            val_dict["weights"] = {
                name: float(raw_child[name].get("weight", 1.0))
                for name, _ in val_nodes
            }
            val_dict["max_expected_repeat"] = raw_child.get("max_expected_repeat", 10.0)
            val_dict["replacement"] = raw_child.get("replacement", False)
        elif ar_mode is not None:
            val_dict["ar_mode"] = bool(ar_mode)
        train_stats_path = data_cfg.get("train", {}).get("pretrained_norm_stats")
        val_stats_path = data_cfg.get("val", {}).get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path
        if pretrained_norm_stats is None:
            default_stats_path = os.path.join(get_work_dir(), "dataset_stats.json")
            pretrained_norm_stats = default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_dict["pretrained_norm_stats"] = pretrained_norm_stats
        val_ds = instantiate(val_dict)
    return train_ds, val_ds


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    # Model construction creates randomly initialized trainable modules. Seed
    # before constructing it so cfg.seed controls the initial checkpoint state.
    # Wan22Trainer seeds again later to reset the RNG for the training phase.
    set_global_seed(int(cfg.seed))
    register_work_dir(cfg.output_dir)
    # Fail-fast guard: data and model must agree on
    # route_aware_action_normalization, otherwise the checkpoint
    # metadata records the wrong route-aware value.
    _check_route_aware_consistency(cfg)
    validate_horizon_config(cfg)
    _validate_training_proprio_schema(cfg)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = resolve_train_device()
    mixed_precision = normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    common_cfg = cfg.data.get("common") if hasattr(cfg.data, "get") else None
    _bind_model_proprio_runtime_contract(model, common_cfg)
    prompt_template = (
        None if common_cfg is None else common_cfg.get("prompt_template")
    )
    if not prompt_template or "{task}" not in str(prompt_template):
        raise ValueError(
            "`data.common.prompt_template` is required during training and "
            "must contain `{task}` so the checkpoint runtime contract records "
            "the exact conditioning template."
        )
    configured_template = getattr(model, "prompt_template", None)
    if configured_template is not None and str(configured_template) != str(prompt_template):
        raise ValueError(
            "Model/data prompt-template mismatch during training: "
            f"model={configured_template!r}, data={prompt_template!r}"
        )
    model.prompt_template = str(prompt_template)
    train_ds, val_ds = build_datasets(
        cfg.data,
        build_val=int(cfg.eval_every) > 0,
        ar_mode=bool(getattr(model, "ar_mode", False)),
    )

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

def build_inference(cfg: DictConfig) -> GameWAMInference:
    """Construct a ``GameWAMInference`` from a Hydra ``cfg``.

    Resolves the model, processor, checkpoint, and dataset statistics
    from ``cfg``, loads the checkpoint, and returns the inference
    runtime. Modules that need an inference runtime should call this
    function rather than assembling the pieces themselves.
    """
    setup_logging(log_level=logging.INFO)

    # Data and model must agree on route-aware action normalization.
    _check_route_aware_consistency(cfg)
    validate_horizon_config(cfg)

    inference_cfg = cfg.inference
    mixed_precision = normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = mixed_precision_to_model_dtype(mixed_precision)
    device = str(inference_cfg.get("device", "cuda"))

    model_cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg_copy.load_text_encoder = True
    model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
    model.eval()

    common_cfg = cfg.data.get("common") if hasattr(cfg.data, "get") else None
    _bind_model_proprio_runtime_contract(model, common_cfg)
    prompt_template = inference_cfg.get("prompt_template")
    if prompt_template is None and common_cfg is not None:
        prompt_template = common_cfg.get("prompt_template")
    if not prompt_template or "{task}" not in str(prompt_template):
        raise ValueError(
            "`prompt_template` is required and must contain `{task}`. "
            "Set `cfg.inference.prompt_template` or `cfg.data.common.prompt_template`."
        )
    # The final rollout prompt is part of the checkpoint runtime contract.
    # Set it before load_checkpoint() so an inference-only override cannot
    # pass validation and then silently change the conditioning distribution.
    model.prompt_template = str(prompt_template)

    checkpoint_path = inference_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("`cfg.inference.checkpoint_path` is required for inference.")
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    stats_path = inference_cfg.get("dataset_stats_path")
    if not stats_path:
        raise ValueError("`cfg.inference.dataset_stats_path` is required for inference.")
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Dataset stats not found: {stats_path}")
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    dataset_stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()

    logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
    model.load_checkpoint(
        str(ckpt),
        mode="inference",
        dataset_stats_sha256=dataset_stats_sha256,
        skip_event_occurrence_head=True,
    )

    if common_cfg is None or common_cfg.get("processor") is None:
        raise ValueError(
            "`cfg.data.common.processor` is required for inference "
            "(mixed and non-mixed configs both converge here)."
        )
    processor_cfg = common_cfg.processor
    processor = instantiate(processor_cfg).eval()

    # Normalizer initialization is owned by ``GameWAMInference.__init__``,
    # which calls ``set_normalizer_from_stats`` on the processor.

    return GameWAMInference(
        cfg=cfg,
        model=model,
        processor=processor,
        dataset_stats=dataset_stats,
        prompt_template=str(prompt_template),
        device=device,
        model_dtype=model_dtype,
    )


def run_inference(cfg: DictConfig):
    """Offline image-to-video inference entry point.

    For non-offline orchestration (Minecraft rollout etc.), call
    ``runtime.build_inference(cfg)`` directly inside the worker and
    drive inference yourself — rollout state is managed by the caller.
    """
    setup_logging(log_level=logging.INFO)
    inference_mode = str(cfg.inference.get("mode", "offline")) if cfg.inference is not None else "offline"
    if inference_mode != "offline":
        raise ValueError(
            f"`runtime.run_inference` only supports mode='offline', got {inference_mode!r}. "
            "For rollout orchestration, scripts/experiments should call "
            "`runtime.build_inference(cfg)` directly inside their worker."
        )
    inference = build_inference(cfg)
    try:
        return inference.infer_image_to_video(
            input_image_path=str(cfg.inference.input_image_path),
            prompt=str(cfg.inference.prompt),
            output_mp4=str(cfg.inference.output_mp4),
            num_frames=int(cfg.inference.num_frames),
            width=int(cfg.inference.width),
            height=int(cfg.inference.height),
            text_cfg_scale=float(cfg.inference.get("text_cfg_scale", 1.0)),
            num_inference_steps=int(cfg.inference.num_inference_steps),
            sigma_shift=(
                None
                if cfg.inference.get("sigma_shift") is None
                else float(cfg.inference.sigma_shift)
            ),
            seed=(
                None
                if cfg.inference.get("seed") is None
                else int(cfg.inference.seed)
            ),
            rand_device=str(cfg.inference.get("rand_device", "cpu")),
            tiled=bool(cfg.inference.get("tiled", False)),
        )
    finally:
        inference.close()
