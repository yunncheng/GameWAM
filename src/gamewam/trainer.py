import json
import gc
import hashlib
import inspect
import os
import shutil
from math import ceil
from pathlib import Path
import time
from typing import Optional

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .utils.misc import ensure_dir
from .utils.logging_config import get_logger
from .utils.misc import normalize_mixed_precision
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim, video_tensor_to_pil_frames
from .utils.action_metrics import action_metric_summary, ensure_action_btd, save_action_artifacts

logger = get_logger(__name__)

_TRAINING_UNUSED_SAMPLE_KEYS = frozenset((
    "idx",
    "vae_cache_key",
    "dataset_group",
    "processor_id",
    "history_depth",
    "episode_start_raw",
    # The train loss consumes precomputed ``context`` embeddings only;
    # ``prompt`` strings are read exclusively by eval/inference paths.
    "prompt",
    # ``target_continuous_action`` is emitted by the processor and declared
    # in dataset/action_expert key lists, but no model code reads it.
    # ``raw_continuous_action`` is read in ``training_loss_ar`` but the
    # transformed result is never consumed. Drop both at collate time to
    # avoid a pointless H2D copy and device allocation each step.
    "target_continuous_action",
    "raw_continuous_action",
))

_HISTORY_CPU_METADATA_KEYS = (
    "history_segment_mask",
    "history_segment_start_raw",
    "history_segment_next_cycle_start_raw",
    "sample_start_raw",
)

# Full ``gc.collect()`` runs once per this many log events (automatic cyclic
# GC stays disabled during training; refcounting handles per-step garbage).
_GC_COLLECT_LOG_INTERVAL = 10


def _training_collate(samples: list[dict]):
    """Collate training samples without replicating fixed or CPU-only payloads."""
    if not samples:
        return {}
    samples = [
        {
            key: value
            for key, value in sample.items()
            if key not in _TRAINING_UNUSED_SAMPLE_KEYS
        }
        for sample in samples
    ]
    first = samples[0]
    singleton_keys = ("uncond_context", "uncond_context_mask")
    has_singleton_context = all(key in first for key in singleton_keys)
    if has_singleton_context:
        payload = {key: first[key] for key in singleton_keys}
        collate_samples = [
            {key: value for key, value in sample.items() if key not in singleton_keys}
            for sample in samples
        ]
    else:
        payload = {}
        collate_samples = samples

    batch = default_collate(collate_samples)
    for key, value in payload.items():
        # The empty-prompt embedding is dataset-constant. Preserve a singleton
        # batch axis and rely on broadcasting in the model.
        batch[key] = value.unsqueeze(0)
    for key in _HISTORY_CPU_METADATA_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value) and value.device.type == "cpu":
            # Accelerate recursively moves tensors to CUDA. NumPy keeps these
            # integral parsing fields on CPU; HistoryModule rehydrates a view.
            batch[key] = value.numpy()
    return batch


def _dataset_guarantees_precomputed_latents(dataset) -> bool:
    """Return whether every sample is guaranteed to contain ``input_latents``.

    This is deliberately inferred from the materialized dataset rather than
    from ``data.common.vae_latent_cache_dir``: mixed children can override the
    common value independently.
    """
    explicit = getattr(dataset, "provides_precomputed_latents", None)
    if explicit is not None:
        return bool(explicit() if callable(explicit) else explicit)

    children = getattr(dataset, "children", None)
    if children is not None:
        return bool(children) and all(
            _dataset_guarantees_precomputed_latents(child) for child in children
        )

    # Support transparent wrappers such as torch.utils.data.Subset.
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None and wrapped is not dataset:
        return _dataset_guarantees_precomputed_latents(wrapped)

    return bool(
        getattr(dataset, "vae_latent_cache_dir", None) is not None
        and not getattr(dataset, "video_only", False)
    )


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        requested_eval_predict_video = bool(cfg.get("eval_predict_video", True))
        self.eval_predict_video = bool(
            requested_eval_predict_video
            or getattr(model, "requires_joint_video_prediction", False)
        )
        if self.eval_predict_video and not requested_eval_predict_video:
            logger.info(
                "Forcing eval_predict_video=true because "
                "video_action_attention_mode=joint_bidirectional."
            )
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = normalize_mixed_precision(cfg.mixed_precision)
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            (
                self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
                if self.accelerator.state.deepspeed_plugin is not None else "none"
            ),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional AR-history/proprio modules) as trainable
        # when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        backbone = self._get_trainable_backbone(self.model)
        trainable_params = list(backbone.parameters())
        history_module = getattr(self.model, "history_module", None)
        if getattr(self.model, "ar_history_mode", False) and history_module is not None:
            trainable_params.extend(list(history_module.parameters()))
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        event_occurrence_head = getattr(self.model, "event_occurrence_head", None)
        if getattr(self.model, "event_occurrence_aux_enabled", False) and event_occurrence_head is not None:
            trainable_params.extend(list(event_occurrence_head.parameters()))
        trainable_params = [p for p in trainable_params if p.requires_grad]
        # Deduplicate by parameter id (defense in depth against a module
        # sharing parameters with the backbone).
        seen_ids: set[int] = set()
        deduped_params: list[torch.nn.Parameter] = []
        for param in trainable_params:
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            deduped_params.append(param)
        trainable_params = deduped_params
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self._training_vae_offloaded = False
        self._configure_training_vae_residency()
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        if self.val_dataset is None:
            logger.info(
                "Train dataset size: %d; validation disabled.",
                len(self.train_dataset),
            )
        else:
            logger.info(
                "Train/val dataset size: %d/%d",
                len(self.train_dataset),
                len(self.val_dataset),
            )

    @staticmethod
    def _clear_vae_runtime_cache(vae) -> None:
        vae_model = getattr(vae, "model", None)
        clear_cache = getattr(vae_model, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()

    def _move_vae(self, device) -> None:
        model = self.accelerator.unwrap_model(self.model)
        vae = getattr(model, "vae", None)
        if vae is None:
            return
        if str(device) == "cpu":
            self._clear_vae_runtime_cache(vae)
        vae.to(device=device)

    def _configure_training_vae_residency(self) -> None:
        """Keep VAE off CUDA iff the materialized train dataset never needs it."""
        if not _dataset_guarantees_precomputed_latents(self.train_dataset):
            logger.info(
                "Training dataset can require online VAE encoding; keeping VAE on %s.",
                self.accelerator.device,
            )
            return

        model = self.accelerator.unwrap_model(self.model)
        if getattr(model, "vae", None) is None:
            return
        self._move_vae("cpu")
        self._training_vae_offloaded = True
        logger.info(
            "All training samples provide precomputed latents; VAE is offloaded "
            "to CPU by default and will be restored only around evaluation."
        )

    def _assert_training_latent_contract(self, sample) -> None:
        if bool(getattr(self, "_training_vae_offloaded", False)) and "input_latents" not in sample:
            raise RuntimeError(
                "Training VAE is offloaded because the materialized dataset "
                "declared precomputed latent coverage, but this batch has no "
                "`input_latents`."
            )

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _call_training_loss(self, model, sample, *, compute_metrics: bool = True):
        if not hasattr(model, "training_loss"):
            return model(
                sample,
                global_step=self.global_step,
                compute_metrics=bool(compute_metrics),
            )
        training_loss = model.training_loss
        cache = getattr(self, "_training_loss_call_capabilities", None)
        if cache is None:
            cache = {}
            self._training_loss_call_capabilities = cache
        function_key = getattr(training_loss, "__func__", training_loss)
        capabilities = cache.get(function_key)
        if capabilities is None:
            signature = inspect.signature(training_loss)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            capabilities = (
                "global_step" in signature.parameters or accepts_kwargs,
                "compute_metrics" in signature.parameters or accepts_kwargs,
            )
            cache[function_key] = capabilities
        accepts_global_step, accepts_metrics = capabilities
        kwargs = {}
        if accepts_global_step:
            kwargs["global_step"] = self.global_step
        if accepts_metrics:
            kwargs["compute_metrics"] = bool(compute_metrics)
        return training_loss(sample, **kwargs)

    def _build_loader(self, dataset, worker_init_fn=None):
        sampling_specs = None
        replacement = False
        if hasattr(dataset, "get_child_sampling_specs"):
            sampling_specs = dataset.get_child_sampling_specs()
            replacement = getattr(dataset, "replacement", False)
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            replacement=replacement,
            sampling_specs=sampling_specs,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            prefetch_factor=4,
            persistent_workers=self.num_workers > 0,
            collate_fn=_training_collate,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading strict weight checkpoint: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path),
            mode="resume",
            dataset_stats_sha256=self._dataset_stats_sha256(),
        )
        logger.warning(
            "Loaded .pt weights only; optimizer/scheduler/step were not "
            "restored under ZeRO2."
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _get_trainable_backbone(model):
        backbone = getattr(model, "mot", None)
        if backbone is not None:
            return backbone
        return model.dit

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        backbone = Wan22Trainer._get_trainable_backbone(model)
        backbone.train()
        backbone.requires_grad_(True)
        history_module = getattr(model, "history_module", None)
        if getattr(model, "ar_history_mode", False) and history_module is not None:
            history_module.train()
            history_module.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        event_occurrence_head = getattr(model, "event_occurrence_head", None)
        if getattr(model, "event_occurrence_aux_enabled", False) and event_occurrence_head is not None:
            event_occurrence_head.train()
            event_occurrence_head.requires_grad_(True)
        object.__setattr__(model, "training", True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        input_latents = sample.get("input_latents", None)
        image_is_pad = sample.get("image_is_pad", None)
        action_is_pad = sample.get("action_is_pad", None)
        proprio_dim_is_pad = sample.get("proprio_dim_is_pad", None)
        proprio_field_valid = sample.get("proprio_field_valid", None)
        proprio_is_pad = sample.get("proprio_is_pad", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        action_auxiliary = {}
        for key in (
            "target_discrete_action",
            "action_route_before_action",
            "action_route_valid",
        ):
            value = sample.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`sample[{key!r}]` must be a torch.Tensor, got {type(value)}")
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if value.ndim != 3:
                raise ValueError(
                    f"`sample[{key!r}]` must be [B,T,D], got {tuple(value.shape)}"
                )
            if action is not None and value.shape[:2] != action.shape[:2]:
                raise ValueError(
                    f"`sample[{key!r}]` must match action [B,T], got "
                    f"{tuple(value.shape)} for action {tuple(action.shape)}"
                )
            action_auxiliary[key] = value

        event_occurrence_label = sample.get("event_occurrence_label")
        if event_occurrence_label is not None:
            if not isinstance(event_occurrence_label, torch.Tensor):
                raise TypeError(
                    "`sample['event_occurrence_label']` must be a torch.Tensor, "
                    f"got {type(event_occurrence_label)}"
                )
            if event_occurrence_label.ndim == 0:
                event_occurrence_label = event_occurrence_label.view(1, 1)
            elif event_occurrence_label.ndim == 1:
                event_occurrence_label = event_occurrence_label.unsqueeze(0)
            elif event_occurrence_label.ndim != 2:
                raise ValueError(
                    "`sample['event_occurrence_label']` must be [B,N], got "
                    f"{tuple(event_occurrence_label.shape)}"
                )

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if proprio_field_valid is not None:
            if not isinstance(proprio_field_valid, torch.Tensor):
                raise TypeError(
                    f"`sample['proprio_field_valid']` must be a torch.Tensor, got {type(proprio_field_valid)}"
                )
            if proprio_field_valid.ndim == 2:
                proprio_field_valid = proprio_field_valid.unsqueeze(0)
            invalid_shape = proprio_field_valid.ndim != 3
            if proprio is not None:
                invalid_shape = invalid_shape or (
                    proprio_field_valid.shape[:2] != proprio.shape[:2]
                )
            if proprio_field_valid.ndim == 3:
                invalid_shape = invalid_shape or proprio_field_valid.shape[-1] <= 0
            if invalid_shape:
                proprio_shape = None if proprio is None else tuple(proprio.shape)
                raise ValueError(
                    "`sample['proprio_field_valid']` must match proprio [B,T] "
                    "with a non-empty field dimension [B,T,F], "
                    f"got {tuple(proprio_field_valid.shape)} for proprio "
                    f"{proprio_shape}"
                )

        if proprio_is_pad is not None:
            if not isinstance(proprio_is_pad, torch.Tensor):
                raise TypeError(f"`sample['proprio_is_pad']` must be a torch.Tensor, got {type(proprio_is_pad)}")
            if proprio_is_pad.ndim == 1:
                proprio_is_pad = proprio_is_pad.unsqueeze(0)
            if proprio is not None and proprio_is_pad.shape != proprio.shape[:2]:
                raise ValueError(
                    "`sample['proprio_is_pad']` must match proprio [B,T], "
                    f"got {tuple(proprio_is_pad.shape)} for proprio {tuple(proprio.shape)}"
                )

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        if image_is_pad is not None:
            if not isinstance(image_is_pad, torch.Tensor):
                raise TypeError(f"`sample['image_is_pad']` must be a torch.Tensor, got {type(image_is_pad)}")
            if image_is_pad.ndim == 1:
                image_is_pad = image_is_pad.unsqueeze(0)
            if image_is_pad.ndim != 2 or image_is_pad.shape[0] != video.shape[0] or image_is_pad.shape[1] != num_video_frames:
                raise ValueError(
                    f"`sample['image_is_pad']` must be [B,T], got shape {tuple(image_is_pad.shape)} "
                    f"for video shape {tuple(video.shape)}"
                )

        if action_is_pad is not None:
            if not isinstance(action_is_pad, torch.Tensor):
                raise TypeError(f"`sample['action_is_pad']` must be a torch.Tensor, got {type(action_is_pad)}")
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            if action is not None and (
                action_is_pad.ndim != 2 or action_is_pad.shape[0] != action.shape[0] or action_is_pad.shape[1] != action.shape[1]
            ):
                raise ValueError(
                    f"`sample['action_is_pad']` must match action [B,T], got {tuple(action_is_pad.shape)} "
                    f"for action shape {tuple(action.shape)}"
                )

        if proprio_dim_is_pad is not None:
            if not isinstance(proprio_dim_is_pad, torch.Tensor):
                raise TypeError(f"`sample['proprio_dim_is_pad']` must be a torch.Tensor, got {type(proprio_dim_is_pad)}")
            if proprio is not None:
                if proprio_dim_is_pad.ndim == 1:
                    proprio_dim_is_pad = proprio_dim_is_pad.view(1, 1, -1).expand_as(proprio)
                elif proprio_dim_is_pad.ndim == 2:
                    proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(0).expand_as(proprio)
                if proprio_dim_is_pad.shape != proprio.shape:
                    raise ValueError(
                        f"`sample['proprio_dim_is_pad']` must match proprio shape, got {tuple(proprio_dim_is_pad.shape)} "
                        f"vs {tuple(proprio.shape)}"
                    )
            elif proprio_dim_is_pad.ndim == 1:
                proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(0)

        if input_latents is not None:
            if not isinstance(input_latents, torch.Tensor):
                raise TypeError(f"`sample['input_latents']` must be a torch.Tensor, got {type(input_latents)}")
            if input_latents.ndim == 4:
                input_latents = input_latents.unsqueeze(0)
            if input_latents.ndim != 5:
                raise ValueError(
                    f"`sample['input_latents']` must be 5D [B,C,T,H,W], got shape {tuple(input_latents.shape)}"
                )
            if input_latents.shape[0] != video.shape[0]:
                raise ValueError(
                    f"`sample['input_latents']` batch mismatch: {input_latents.shape[0]} vs video batch={video.shape[0]}"
                )

        result = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }
        if image_is_pad is not None:
            result["image_is_pad"] = image_is_pad
        if action_is_pad is not None:
            result["action_is_pad"] = action_is_pad
        result.update(action_auxiliary)
        if event_occurrence_label is not None:
            result["event_occurrence_label"] = event_occurrence_label
        if proprio_dim_is_pad is not None:
            result["proprio_dim_is_pad"] = proprio_dim_is_pad
        if proprio_field_valid is not None:
            result["proprio_field_valid"] = proprio_field_valid
        if proprio_is_pad is not None:
            result["proprio_is_pad"] = proprio_is_pad
        if input_latents is not None:
            result["input_latents"] = input_latents
        return result

    @torch.no_grad()
    def _evaluate_batch(self, model, eval_indices):
        """Batch eval for both AR and non-AR modes.

        All eval_indices are used for loss/metrics (gathered across ranks).
        Only the first ``_eval_visualize_count`` samples per rank produce
        video/artifact outputs (saved into ``eval/step_XXXXX/rank_NNN/``).
        """
        eval_bs = len(eval_indices)
        is_ar = getattr(model, "ar_mode", False)

        # Step 1: compute val_loss for ALL samples
        all_val_losses = []
        all_val_loss_dicts = []
        all_artifact_metrics = []

        for i, idx in enumerate(eval_indices):
            if is_ar:
                sample = self.val_dataset[idx]
                batched = {}
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        batched[k] = v.unsqueeze(0).to(device=self.accelerator.device)
                    elif isinstance(v, str):
                        batched[k] = v
                    else:
                        batched[k] = v
            else:
                batched = self._to_batched_eval_sample(self.val_dataset[idx])

            with self.accelerator.autocast():
                val_loss, val_loss_dict = self._call_training_loss(model, batched)
                val_loss = val_loss.float().item()
            all_val_losses.append(val_loss)
            all_val_loss_dicts.append(val_loss_dict)

            # Visualisation: only first _eval_visualize_count samples per rank.
            if i < self._eval_visualize_count and self.eval_predict_video:
                if is_ar:
                    raw_video = self.val_dataset.get_raw_video(
                        idx, sample_idx=sample.get("idx"),
                    )
                    batched["raw_video"] = raw_video.unsqueeze(0)
                    with self.accelerator.autocast():
                        artifact_metrics = self._save_ar_eval_artifacts(
                            model, batched, sample_index=i,
                        )
                else:
                    artifact_metrics = self._eval_one_non_ar_visual(
                        model, batched, sample_index=i,
                    )
                all_artifact_metrics.append(artifact_metrics)

        # Step 2: aggregate across samples within this rank
        avg_val_loss = sum(all_val_losses) / max(len(all_val_losses), 1)

        # Merge loss components (average non-count keys, sum count keys).
        merged_components = {}
        count_keys = self._event_count_metric_keys() | {
            key
            for component_dict in all_val_loss_dicts
            for key in component_dict
            if key.endswith("_count")
        }
        for d in all_val_loss_dicts:
            for k, v in d.items():
                vf = float(v.detach().item() if torch.is_tensor(v) else v)
                if k in count_keys:
                    merged_components[k] = merged_components.get(k, 0.0) + vf
                else:
                    merged_components[k] = merged_components.get(k, 0.0) + vf
        n = max(len(all_val_loss_dicts), 1)
        for k in merged_components:
            if k not in count_keys:
                merged_components[k] /= n

        # Step 3: gather across ranks
        global_val_loss = self._gather_scalar_metric(avg_val_loss, self.accelerator)
        global_loss_components = self._merge_val_loss_components(merged_components)

        result = {"val_loss": float(global_val_loss)}
        result.update(global_loss_components)

        if self.eval_predict_video and all_artifact_metrics:
            result.update(self._merge_eval_artifact_metrics(all_artifact_metrics))

        return result

    def _merge_eval_artifact_metrics(self, all_artifact_metrics):
        """Merge action & PSNR/SSIM metrics across the visualized subset."""
        result = {}

        # Aggregate action scalars.
        agg_action = {}
        for m in all_artifact_metrics:
            for k, v in m.items():
                if k in ("action_l1", "action_l2", "action_l1_raw", "action_l2_raw"):
                    agg_action[k] = agg_action.get(k, 0.0) + float(v)
        nv = max(len(all_artifact_metrics), 1)
        agg_action = {k: v / nv for k, v in agg_action.items()}
        if "action_l2" in agg_action:
            agg_action["action_rmse"] = float(max(agg_action["action_l2"], 0.0) ** 0.5)
        result.update(self._merge_action_metric_scalars(agg_action))

        # Pool confusion counts over the existing visualized subset. This adds
        # no inference work: it only replaces biased per-clip F1 averaging with
        # exact TP/FP/TN/FN accumulation across samples and ranks.
        binary_items = [m.get("binary_like") for m in all_artifact_metrics if m.get("binary_like") is not None]
        if binary_items:
            dims = [int(value) for value in binary_items[0]["dims"]]
            local_counts = {name: [0.0] * len(dims) for name in ("tp", "fp", "tn", "fn")}
            for item in binary_items:
                if [int(value) for value in item["dims"]] != dims:
                    raise ValueError("Eval discrete action dimensions differ across samples.")
                for name in local_counts:
                    values = item[name]
                    if len(values) != len(dims):
                        raise ValueError(f"Eval binary {name} count length mismatch.")
                    local_counts[name] = [a + float(b) for a, b in zip(local_counts[name], values)]
            flat = [value for name in ("tp", "fp", "tn", "fn") for value in local_counts[name]]
            global_flat = self._gather_metric_vector(flat).sum(dim=0).detach().cpu().tolist()
            offset = 0
            global_counts = {}
            for name in ("tp", "fp", "tn", "fn"):
                global_counts[name] = global_flat[offset:offset + len(dims)]
                offset += len(dims)
            count_metrics = {}
            for name in ("tp", "fp", "tn", "fn"):
                count_metrics[f"action_discrete_{name}_count"] = sum(global_counts[name])
            for local_idx, action_dim in enumerate(dims):
                stem = f"action_discrete_dim_{action_dim:02d}"
                for name in ("tp", "fp", "tn", "fn"):
                    count_metrics[f"{stem}_{name}_count"] = global_counts[name][local_idx]
            result.update(self._finalize_action_discrete_metrics(
                count_metrics, keep_per_dim=True, drop_raw_counts=True
            ))

        # Gather PSNR/SSIM keys across ranks.
        psnr_ssim_keys = (
            "psnr_ar_tf_vs_gt", "ssim_ar_tf_vs_gt",
            "psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg",
        )
        for key in psnr_ssim_keys:
            vals = [float(m[key]) for m in all_artifact_metrics if key in m]
            if vals:
                avg = sum(vals) / len(vals)
                result[key] = self._gather_scalar_metric(avg, self.accelerator)

        # Artifact paths from first visualized sample.
        first = all_artifact_metrics[0]
        for key, value in first.items():
            if key.endswith("_path") or key == "video_path":
                result[key] = value

        return result

    @torch.no_grad()
    def _eval_one_non_ar_visual(self, model, sample, sample_index: int = 0):
        """Run non-AR joint inference on a single sample and return metrics."""
        model_obj = self.accelerator.unwrap_model(model)

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        _, num_frames, _, _ = video0.shape

        num_obs_latents = int(model_obj.video_expert.num_obs_latents)
        num_obs_frames = int(model_obj.video_expert.num_obs_frames)
        frame_stride = int(getattr(model_obj, "frame_stride", 1))
        obs_video_frames = (num_obs_latents - 1) * model_obj.video_expert.VAE_TEMPORAL_FACTOR + 1
        obs_action_steps = (num_obs_frames - 1) * frame_stride
        future_action_len = sample['action_horizon'] - obs_action_steps if sample['action_horizon'] else None
        if action is not None and future_action_len is not None:
            action = action[obs_action_steps:obs_action_steps + future_action_len]

        input_image = video0[:, 0:obs_video_frames].unsqueeze(0)

        proprio_field_valid = None
        proprio_is_pad = None
        if "proprio" in sample and sample["proprio"] is not None:
            num_obs_proprio = min(obs_action_steps + 1, sample["proprio"].shape[1])
            proprio = sample["proprio"][0:1, :num_obs_proprio, :]
            if sample.get("proprio_field_valid") is not None:
                proprio_field_valid = sample["proprio_field_valid"][0:1, :num_obs_proprio, :]
            if sample.get("proprio_is_pad") is not None:
                proprio_is_pad = sample["proprio_is_pad"][0:1, :num_obs_proprio]
        else:
            proprio = None

        infer_kwargs = {
            "input_image": input_image,
            "num_video_frames": num_frames,
            "action": action,
            "num_action_steps": future_action_len,
            "proprio": proprio,
            "proprio_field_valid": proprio_field_valid,
            "proprio_is_pad": proprio_is_pad,
            "text_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(**infer_kwargs)
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval video prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        pred_future_tensor = pred_video_tensor[:, obs_video_frames:]
        gt_future_tensor = gt_video_tensor[:, obs_video_frames:]

        psnr_pred_vs_gt = video_psnr(pred=pred_future_tensor, target=gt_future_tensor)
        ssim_pred_vs_gt = video_ssim(pred=pred_future_tensor, target=gt_future_tensor)

        # VAE decode GT for comparison
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        vae_future_tensor = vae_video_tensor[:, obs_video_frames:]

        psnr_decode_vs_gt = video_psnr(pred=vae_future_tensor, target=gt_future_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_future_tensor, target=gt_future_tensor)
        psnr_pred_vs_decode = video_psnr(pred=pred_future_tensor, target=vae_future_tensor)
        ssim_pred_vs_decode = video_ssim(pred=pred_future_tensor, target=vae_future_tensor)

        # Save stitched video
        rank_dir = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}",
            f"rank_{self.accelerator.process_index:03d}",
        )
        os.makedirs(rank_dir, exist_ok=True)

        stitched_video_tensor = torch.cat(
            [pred_future_tensor, vae_future_tensor, gt_future_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = video_tensor_to_pil_frames(stitched_video_tensor)
        video_path = os.path.join(rank_dir, f"sample_{sample_index:03d}.mp4")
        save_mp4(stitched_frames, video_path, fps=8)

        action_metrics = None
        if action is not None and pred_action is not None:
            action_metrics = self._compute_eval_action_metrics(
                sample,
                pred_action,
                action,
                pred_route_mask=pred.get("action_route_mask"),
                gt_route_start=obs_action_steps,
                artifact_prefix=os.path.join(rank_dir, f"sample_{sample_index:03d}"),
                block_size=None,
                artifact_context={
                    "video_layout": "top=predicted video, middle=VAE-decoded GT, bottom=raw GT",
                },
            )

        metrics = {
            "psnr_rg": float(psnr_pred_vs_gt),
            "ssim_rg": float(ssim_pred_vs_gt),
            "psnr_rd": float(psnr_pred_vs_decode),
            "ssim_rd": float(ssim_pred_vs_decode),
            "psnr_dg": float(psnr_decode_vs_gt),
            "ssim_dg": float(ssim_decode_vs_gt),
            "video_path": video_path,
        }
        if action_metrics is not None:
            for k in ("action_l1", "action_l2", "action_l1_raw", "action_l2_raw",
                       "action_metrics_path", "action_heatmap_path"):
                if k in action_metrics:
                    metrics[k] = action_metrics[k]
            if action_metrics.get("binary_like") is not None:
                metrics["binary_like"] = action_metrics["binary_like"]
        return metrics

    @property
    def _eval_visualize_count(self) -> int:
        """Number of eval samples per rank to save visual artifacts for."""
        return 8

    @torch.no_grad()
    def _save_ar_eval_artifacts(self, model, sample, sample_index: int = 0):
        if "input_latents" not in sample:
            raise ValueError("AR validation visualization requires cached `input_latents`.")

        history_prefix_tokens = None
        history_valid_mask = None
        if getattr(model, "ar_history_mode", False):
            history_prefix_tokens, history_valid_mask, _ = model.history_module.build_history_prefix_batch(
                batch=sample,
                training=False,
                device=model.device,
                dtype=model.torch_dtype,
            )

        proprio = None
        proprio_dim_is_pad = None
        proprio_field_valid = None
        proprio_is_pad = None
        if getattr(model, "proprio_encoder", None) is not None and sample.get("proprio") is not None:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if sample.get("proprio_dim_is_pad") is not None:
                proprio_dim_is_pad = sample["proprio_dim_is_pad"]
            if sample.get("proprio_field_valid") is not None:
                proprio_field_valid = sample["proprio_field_valid"]
            if sample.get("proprio_is_pad") is not None:
                proprio_is_pad = sample["proprio_is_pad"]
            # Dataset samples keep frame-level proprio_is_pad at raw state length
            # while proprio is action-aligned to T-1. Artifact inference uses
            # the action-aligned proprio sequence, so masks must match it.
            proprio_len = int(proprio.shape[1])
            if proprio_is_pad is not None and proprio_is_pad.shape[1] != proprio_len:
                if proprio_is_pad.shape[1] == proprio_len + 1:
                    proprio_is_pad = proprio_is_pad[:, :proprio_len]
                else:
                    raise ValueError(
                        "`sample[proprio_is_pad]` must match proprio time dim or be one frame longer, "
                        f"got {tuple(proprio_is_pad.shape)} for proprio {tuple(proprio.shape)}"
                    )
            if proprio_field_valid is not None and proprio_field_valid.shape[1] != proprio_len:
                if proprio_field_valid.shape[1] == proprio_len + 1:
                    proprio_field_valid = proprio_field_valid[:, :proprio_len, :]
                else:
                    raise ValueError(
                        "`sample[proprio_field_valid]` must match proprio time dim or be one frame longer, "
                        f"got {tuple(proprio_field_valid.shape)} for proprio {tuple(proprio.shape)}"
                    )

        prompt = None
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None or context_mask is None:
            prompt = sample.get("prompt")
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]

        raw_action_len = sample.get("action_horizon")
        if isinstance(raw_action_len, torch.Tensor):
            raw_action_len = int(raw_action_len.reshape(-1)[0].item())
        elif raw_action_len is None:
            raw_action_len = int(sample["action"].shape[1])
        else:
            raw_action_len = int(raw_action_len)
        num_obs_frames = int(model.video_expert.num_obs_frames)
        frame_stride = int(getattr(model, "frame_stride", 1))
        obs_action_steps = (num_obs_frames - 1) * frame_stride
        committed_blocks = int(model.ar_geometry.pred_latents)
        execution_horizon = int(model.execution_horizon)
        prediction_horizon = int(model.prediction_horizon)
        future_action_len = (committed_blocks - 1) * execution_horizon + prediction_horizon
        available_future_action_len = raw_action_len - obs_action_steps
        if available_future_action_len < future_action_len:
            raise ValueError(
                "AR eval action length must cover the stitched future horizon: "
                f"available={available_future_action_len}, required={future_action_len}, "
                f"action_len={raw_action_len}, obs_action_steps={obs_action_steps}"
            )

        input_image = sample.get("video")
        if input_image is None:
            input_image = torch.empty(1, 3, 1, 16, 16, device=model.device, dtype=model.torch_dtype)

        pred = model.infer(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=int(getattr(model, "num_frames", sample["input_latents"].shape[2])),
            num_action_steps=future_action_len,
            input_latents=sample["input_latents"],
            history_prefix_tokens=history_prefix_tokens,
            history_valid_mask=history_valid_mask,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
            num_inference_steps=self.eval_num_inference_steps,
            seed=42,
            tiled=False,
            ar_eval_clip=True,
        )

        pred_action = pred.get("action")
        gt_action = sample.get("action")
        action_metrics = None
        rank_dir = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}",
            f"rank_{self.accelerator.process_index:03d}",
        )
        os.makedirs(rank_dir, exist_ok=True)
        if gt_action is not None and pred_action is not None:
            gt_action = gt_action[:, obs_action_steps:obs_action_steps + future_action_len]
            prefix = os.path.join(rank_dir, f"sample_{sample_index:03d}")
            action_metrics = self._compute_eval_action_metrics(
                sample,
                pred_action,
                gt_action,
                pred_route_mask=pred.get("action_route_mask"),
                gt_route_start=obs_action_steps,
                artifact_prefix=prefix,
                block_size=int(getattr(model.ar_geometry, "raw_steps_per_latent", 0)) if getattr(model, "ar_geometry", None) is not None else None,
                action_video_freq_ratio=int(model.ar_geometry.action_video_freq_ratio),
                artifact_context={
                    "video_action_coupled": False,
                    "video_layout": "top=predicted video, middle=VAE-decoded GT, bottom=raw GT",
                },
            )

        pred_video_tensor = pil_frames_to_video_tensor(pred["video"])
        decoded_gt_video = model._decode_latents(
            sample["input_latents"].to(device=model.device, dtype=model.torch_dtype),
            tiled=False,
        )
        decoded_gt_video_tensor = pil_frames_to_video_tensor(decoded_gt_video)
        raw_video = sample.get("raw_video")
        if not isinstance(raw_video, torch.Tensor):
            raise ValueError("AR validation visualization requires tensor `raw_video`.")
        if raw_video.ndim == 5:
            if raw_video.shape[0] != 1:
                raise ValueError(f"AR eval raw video batch must have size 1, got {tuple(raw_video.shape)}")
            raw_video = raw_video[0]
        if raw_video.ndim != 4:
            raise ValueError(f"AR eval raw video must be [C,T,H,W], got {tuple(raw_video.shape)}")
        raw_gt_video_tensor = (
            (raw_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
        ).contiguous()

        if pred_video_tensor.shape != decoded_gt_video_tensor.shape:
            raise ValueError(
                "AR eval prediction/decoded-GT shape mismatch: "
                f"pred={tuple(pred_video_tensor.shape)} vs decoded_gt={tuple(decoded_gt_video_tensor.shape)}"
            )
        if decoded_gt_video_tensor.shape != raw_gt_video_tensor.shape:
            raise ValueError(
                "AR eval decoded/raw-GT shape mismatch: "
                f"decoded_gt={tuple(decoded_gt_video_tensor.shape)} vs raw_gt={tuple(raw_gt_video_tensor.shape)}"
            )

        num_obs_latents = int(model.video_expert.num_obs_latents)
        obs_video_frames = (num_obs_latents - 1) * model.video_expert.VAE_TEMPORAL_FACTOR + 1
        pred_future_tensor = pred_video_tensor[:, obs_video_frames:]
        decoded_gt_future_tensor = decoded_gt_video_tensor[:, obs_video_frames:]
        raw_gt_future_tensor = raw_gt_video_tensor[:, obs_video_frames:]

        psnr_rd = video_psnr(pred=pred_future_tensor, target=decoded_gt_future_tensor)
        ssim_rd = video_ssim(pred=pred_future_tensor, target=decoded_gt_future_tensor)
        psnr_rg = video_psnr(pred=pred_future_tensor, target=raw_gt_future_tensor)
        ssim_rg = video_ssim(pred=pred_future_tensor, target=raw_gt_future_tensor)
        psnr_dg = video_psnr(pred=decoded_gt_future_tensor, target=raw_gt_future_tensor)
        ssim_dg = video_ssim(pred=decoded_gt_future_tensor, target=raw_gt_future_tensor)

        stitched_video_tensor = torch.cat(
            [pred_future_tensor, decoded_gt_future_tensor, raw_gt_future_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = video_tensor_to_pil_frames(stitched_video_tensor)

        video_path = os.path.join(rank_dir, f"sample_{sample_index:03d}.mp4")
        save_mp4(stitched_frames, video_path, fps=8)

        metrics = {
            "psnr_ar_tf_vs_gt": float(psnr_rd),
            "ssim_ar_tf_vs_gt": float(ssim_rd),
            "psnr_rg": float(psnr_rg),
            "ssim_rg": float(ssim_rg),
            "psnr_rd": float(psnr_rd),
            "ssim_rd": float(ssim_rd),
            "psnr_dg": float(psnr_dg),
            "ssim_dg": float(ssim_dg),
            "video_path": video_path,
        }
        if action_metrics is not None:
            metrics.update({
                "action_l1": float(action_metrics["action_l1"]),
                "action_l2": float(action_metrics["action_l2"]),
                "action_rmse": float(action_metrics["action_rmse"]),
                "action_metrics_path": action_metrics.get("action_metrics_path"),
                "action_heatmap_path": action_metrics.get("action_heatmap_path"),
                "action_trace_path": action_metrics.get("action_trace_path"),
                "action_binary_mask_path": action_metrics.get("action_binary_mask_path"),
                "action_dim_l1_path": action_metrics.get("action_dim_l1_path"),
            })
            if action_metrics.get("binary_like") is not None:
                metrics["binary_like_mean_recall"] = float(action_metrics["binary_like"]["mean_recall"])
                metrics["binary_like_mean_precision"] = float(action_metrics["binary_like"]["mean_precision"])
                metrics["binary_like_mean_f1"] = float(action_metrics["binary_like"]["mean_f1"])
                metrics["binary_like"] = action_metrics["binary_like"]
        return metrics

    def _denormalize_eval_action(
        self,
        sample,
        raw_action,
        action_name: str,
        route_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_btd = ensure_action_btd(raw_action, action_name)
        processor = self.val_dataset.processor
        shape_meta = self.val_dataset.shape_meta
        action_meta = shape_meta["action"]
        state_meta = shape_meta["state"]

        proprio = sample.get("proprio")
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.detach().to(device="cpu", dtype=torch.float32)
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.shape[1] != action_btd.shape[1]:
                if proprio.shape[1] > action_btd.shape[1]:
                    proprio = proprio[:, : action_btd.shape[1]]
                else:
                    pad = proprio[:, -1:].expand(-1, action_btd.shape[1] - proprio.shape[1], -1)
                    proprio = torch.cat([proprio, pad], dim=1)
        else:
            proprio = torch.zeros(
                action_btd.shape[0],
                action_btd.shape[1],
                processor.proprio_output_dim,
                dtype=torch.float32,
            )

        batch = {
            "action": action_btd,
            "proprio": proprio,
        }
        # Pass the route mask to the processor's postprocess. The mask
        # must be on CPU. Route-aware validation requires it; global-only
        # configs use the global denormalization path.
        route_aware = bool(
            getattr(processor, "route_aware_action_normalization", False)
        )
        if route_aware and route_mask is None:
            raise ValueError(
                f"Route-aware validation requires a route mask for {action_name} action."
            )
        if not route_aware:
            # ``action_route_before_action`` can still be present as a model
            # routing/GUI label when action normalization is global. It must
            # not select the route-aware inverse-normalization path.
            route_mask = None
        if route_mask is not None:
            # ``action_route_before_action`` is shape ``[T, 1]`` in the
            # sample. ``processor.postprocess`` expects ``[1, T, 1]`` for
            # a batched action tensor — the sample's mask doesn't carry
            # the batch dim, so reshape (no copy when already correct).
            route_mask = route_mask.detach().to(device="cpu", dtype=torch.bool)
            if route_mask.ndim == 2:
                route_mask = route_mask.unsqueeze(0)
            expected = (action_btd.shape[0], action_btd.shape[1], 1)
            if tuple(route_mask.shape) != expected:
                raise ValueError(
                    f"Route mask for {action_name} has shape "
                    f"{tuple(route_mask.shape)}, expected {expected}."
                )
        batch = processor.postprocess(batch, route_mask=route_mask)
        merged_batch = {
            "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
            "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
        }
        merged_batch = processor.action_state_merger.forward(merged_batch)
        denorm_action = merged_batch["action"].unsqueeze(0)
        if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
            raise ValueError(
                f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
            )
        return denorm_action

    def _extract_route_mask(self, sample) -> Optional[torch.Tensor]:
        """Extract the per-frame action route mask from a sample."""
        mask = sample.get("action_route_before_action")
        if not isinstance(mask, torch.Tensor):
            return None
        # ``action_route_before_action`` is shape ``[T, 1]`` in the
        # sample. ``processor.postprocess`` expects ``[1, T, 1]`` for
        # a batched action tensor; the sample's mask doesn't carry
        # the batch dim. Reshape (without copying if already in
        # the right shape).
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask.detach().to(device="cpu", dtype=torch.bool)

    @staticmethod
    def _action_display_reference_scale(processor) -> tuple[Optional[list[float]], Optional[str]]:
        """Build per-dim raw-action display scales from training statistics."""
        try:
            action_normalizers = processor.normalizer.normalizers["action"]
        except (AttributeError, KeyError, ValueError):
            return None, None

        scales = []
        sources = []
        for meta in processor.shape_meta.get("action", []):
            key = meta["key"]
            normalizer = action_normalizers.get(key)
            if normalizer is None:
                return None, None
            stats = normalizer.get_stats()
            if "q01" in stats and "q99" in stats:
                lower = torch.as_tensor(stats["q01"], dtype=torch.float32)
                upper = torch.as_tensor(stats["q99"], dtype=torch.float32)
                scale = torch.maximum(lower.abs(), upper.abs())
                source = "global_q01/q99"
            elif "mean" in stats and "std" in stats:
                mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
                std = torch.as_tensor(stats["std"], dtype=torch.float32)
                scale = mean.abs() + 3.0 * std
                source = "global_mean+3std"
            elif "min" in stats and "max" in stats:
                lower = torch.as_tensor(stats["min"], dtype=torch.float32)
                upper = torch.as_tensor(stats["max"], dtype=torch.float32)
                scale = torch.maximum(lower.abs(), upper.abs())
                source = "global_min/max"
            else:
                return None, None

            expected_dim = int(meta["shape"])
            if scale.shape[-1] != expected_dim:
                raise ValueError(
                    "Action display stats dimension mismatch for "
                    f"{key!r}: got {tuple(scale.shape)}, expected last dim {expected_dim}"
                )
            if scale.ndim > 1:
                scale = scale.reshape(-1, expected_dim).amax(dim=0)
            scale = torch.where(
                torch.isfinite(scale) & (scale > 1e-6),
                scale,
                torch.ones_like(scale),
            )
            scales.extend(float(value) for value in scale.tolist())
            sources.append(source)

        if not scales:
            return None, None
        source = sources[0] if len(set(sources)) == 1 else "+".join(sources)
        return scales, source

    def _compute_eval_action_metrics(
        self,
        sample,
        pred_action,
        gt_action,
        pred_route_mask: Optional[torch.Tensor] = None,
        gt_route_start: int = 0,
        artifact_prefix: Optional[str] = None,
        block_size: Optional[int] = None,
        action_video_freq_ratio: Optional[int] = None,
        artifact_context: Optional[dict] = None,
    ):
        if gt_action is None or pred_action is None:
            return None
        processor = getattr(getattr(self, "val_dataset", None), "processor", None)
        route_aware = bool(
            getattr(processor, "route_aware_action_normalization", False)
        )
        if route_aware:
            # GT route mask comes from the sample (processor's
            # ``action_route_before_action``); pred route mask from the
            # model's output (``pred.get("action_route_mask")``). Both must
            # flow through ``processor.postprocess`` so actions are
            # denormalized in the correct route space.
            gt_route_mask = self._extract_route_mask(sample)
            if gt_route_mask is not None:
                gt_action_btd = ensure_action_btd(gt_action, "gt")
                start = int(gt_route_start)
                end = start + int(gt_action_btd.shape[1])
                if start < 0 or end > int(gt_route_mask.shape[1]):
                    raise ValueError(
                        "GT route mask cannot be aligned with the evaluated action: "
                        f"mask_len={int(gt_route_mask.shape[1])}, start={start}, "
                        f"action_len={int(gt_action_btd.shape[1])}."
                    )
                gt_route_mask = gt_route_mask[:, start:end]
        else:
            # The route label may still be emitted for model semantics, but
            # global normalization has only one inverse transform.
            pred_route_mask = None
            gt_route_mask = None
        pred_action_denorm = self._denormalize_eval_action(
            sample, pred_action, "pred", route_mask=pred_route_mask
        )
        gt_action_denorm = self._denormalize_eval_action(
            sample, gt_action, "gt", route_mask=gt_route_mask
        )
        dim_is_continuous = (
            None if processor is None else getattr(processor, "dim_is_continuous", None)
        )
        if processor is None:
            continuous_reference_scale, reference_scale_source = None, None
        else:
            continuous_reference_scale, reference_scale_source = (
                self._action_display_reference_scale(processor)
            )
        metrics = action_metric_summary(
            pred_action_denorm,
            gt_action_denorm,
            dim_is_continuous=dim_is_continuous,
        )
        resolved_artifact_context = dict(artifact_context or {})
        if reference_scale_source is not None:
            resolved_artifact_context["continuous_display_scale_source"] = (
                reference_scale_source
            )
        if resolved_artifact_context:
            metrics["artifact_context"] = resolved_artifact_context
        if artifact_prefix is not None:
            metrics.update(save_action_artifacts(
                artifact_prefix,
                metrics,
                pred_action_denorm,
                gt_action_denorm,
                block_size=block_size,
                dim_is_continuous=dim_is_continuous,
                continuous_reference_scale=continuous_reference_scale,
                continuous_value_floor=None,
                action_video_freq_ratio=action_video_freq_ratio,
                video_action_coupled=resolved_artifact_context.get(
                    "video_action_coupled"
                ),
            ))
        return metrics

    @staticmethod
    def _gather_scalar_metric(value, accelerator) -> float:
        """Gather a single scalar across distributed ranks and return its mean."""
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    f"Expected a scalar metric tensor, got shape {tuple(value.shape)}"
                )
            tensor = value.detach().to(
                device=accelerator.device, dtype=torch.float32
            ).reshape(1)
        else:
            tensor = torch.tensor(
                float(value), device=accelerator.device, dtype=torch.float32
            ).reshape(1)
        return float(accelerator.gather_for_metrics(tensor).mean().item())

    def _gather_metric_vector(self, values) -> torch.Tensor:
        """Gather a vector of scalars across ranks; returns [num_ranks, len(values)]."""
        tensor = torch.tensor(list(values), device=self.accelerator.device, dtype=torch.float32).unsqueeze(0)
        return self.accelerator.gather_for_metrics(tensor)

    @staticmethod
    def _event_count_metric_keys() -> set[str]:
        return {
            "event_occurrence_num_valid",
            "event_occurrence_num_pos",
            "event_occurrence_num_total",
            "event_occurrence_num_neg",
            "event_occurrence_tp",
            "event_occurrence_fp",
            "event_occurrence_tn",
            "event_occurrence_fn",
            "event_occurrence_pos_prob_sum",
            "event_occurrence_neg_prob_sum",
        }

    @staticmethod
    def _finalize_action_discrete_metrics(
        metrics: dict,
        *,
        keep_per_dim: bool = False,
        drop_raw_counts: bool = True,
    ) -> dict:
        """Derive exact binary metrics from additive discrete-action counts."""
        total_keys = {
            name: f"action_discrete_{name}_count"
            for name in ("tp", "fp", "tn", "fn")
        }
        if not set(total_keys.values()).issubset(metrics):
            return metrics

        def _derived(tp: float, fp: float, tn: float, fn: float) -> dict[str, float]:
            valid = tp + fp + tn + fn
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            return {
                "precision": precision,
                "recall": recall,
                "f1": (
                    0.0
                    if precision + recall == 0.0
                    else 2.0 * precision * recall / (precision + recall)
                ),
                "accuracy": (tp + tn) / max(valid, 1.0),
                "gt_pos_ratio": (tp + fn) / max(valid, 1.0),
                "pred_pos_ratio": (tp + fp) / max(valid, 1.0),
                "positive_support": tp + fn,
                "valid_count": valid,
            }

        total = {name: float(metrics[key]) for name, key in total_keys.items()}
        micro = _derived(**total)
        for name, value in micro.items():
            metrics[f"action_discrete_micro_{name}"] = value

        suffix = "_tp_count"
        stems = sorted(
            key[: -len(suffix)]
            for key in metrics
            if key.startswith("action_discrete_dim_") and key.endswith(suffix)
        )
        active_per_dim = []
        predicted_active_dims = 0
        for stem in stems:
            count_keys = {
                name: f"{stem}_{name}_count"
                for name in ("tp", "fp", "tn", "fn")
            }
            if not set(count_keys.values()).issubset(metrics):
                continue
            counts = {name: float(metrics[key]) for name, key in count_keys.items()}
            derived = _derived(**counts)
            if counts["tp"] + counts["fp"] > 0.0:
                predicted_active_dims += 1
            # Recall/F1 is undefined when a key is absent in the window.
            if derived["positive_support"] > 0.0:
                active_per_dim.append(derived)
            if keep_per_dim:
                for name, value in derived.items():
                    metrics[f"{stem}_{name}"] = value

        metrics["action_discrete_supported_dims"] = float(len(active_per_dim))
        metrics["action_discrete_predicted_active_dims"] = float(predicted_active_dims)
        for name in ("precision", "recall", "f1"):
            metrics[f"action_discrete_macro_{name}"] = (
                sum(item[name] for item in active_per_dim) / len(active_per_dim)
                if active_per_dim
                else 0.0
            )

        if drop_raw_counts:
            for key in list(metrics):
                if key.startswith("action_discrete_") and key.endswith("_count"):
                    del metrics[key]
        return metrics

    @staticmethod
    def _finalize_event_metrics(metrics: dict) -> dict:
        required = {
            "event_occurrence_tp",
            "event_occurrence_fp",
            "event_occurrence_tn",
            "event_occurrence_fn",
            "event_occurrence_num_valid",
            "event_occurrence_num_total",
            "event_occurrence_num_pos",
            "event_occurrence_num_neg",
        }
        if not required.issubset(metrics):
            return metrics
        tp = float(metrics["event_occurrence_tp"])
        fp = float(metrics["event_occurrence_fp"])
        tn = float(metrics["event_occurrence_tn"])
        fn = float(metrics["event_occurrence_fn"])
        num_valid = float(metrics["event_occurrence_num_valid"])
        num_total = float(metrics["event_occurrence_num_total"])
        num_pos = float(metrics["event_occurrence_num_pos"])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        metrics["event_occurrence_acc"] = (tp + tn) / max(num_valid, 1.0)
        metrics["event_occurrence_valid_ratio"] = num_valid / max(num_total, 1.0)
        metrics["event_occurrence_pos_ratio"] = num_pos / max(num_valid, 1.0)
        if "event_occurrence_pos_prob_sum" in metrics:
            metrics["event_occurrence_pos_prob_mean"] = (
                float(metrics["event_occurrence_pos_prob_sum"]) / max(num_pos, 1.0)
            )
        if "event_occurrence_neg_prob_sum" in metrics:
            metrics["event_occurrence_neg_prob_mean"] = (
                float(metrics["event_occurrence_neg_prob_sum"]) / max(float(metrics["event_occurrence_num_neg"]), 1.0)
            )
        metrics["event_occurrence_precision"] = precision
        metrics["event_occurrence_recall"] = recall
        metrics["event_occurrence_f1"] = (
            0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        )
        return metrics

    @classmethod
    def _metrics_from_gathered_matrix(cls, keys: list[str], gathered: torch.Tensor) -> dict:
        means = gathered.mean(dim=0).detach().cpu().tolist()
        result = {key: float(means[idx]) for idx, key in enumerate(keys)}

        # Counts are local and additive across ranks.
        count_keys = cls._event_count_metric_keys() | {
            key for key in keys if key.endswith("_count")
        }
        for idx, key in enumerate(keys):
            if key in count_keys:
                result[key] = float(gathered[:, idx].sum().detach().cpu().item())
        result = cls._finalize_event_metrics(result)
        return cls._finalize_action_discrete_metrics(result)

    def _merge_val_loss_components(self, val_loss_dict: dict) -> dict:
        """Gather loss components and recompute event ratios from global counts."""
        keys = sorted(val_loss_dict.keys())
        if not keys:
            return {}
        values = []
        for key in keys:
            value = val_loss_dict[key]
            if torch.is_tensor(value):
                value = float(value.detach().float().reshape(()).item())
            values.append(float(value))
        gathered = self._gather_metric_vector(values)
        return self._metrics_from_gathered_matrix(keys, gathered)

    def _merge_action_metric_scalars(self, action_metrics: Optional[dict]) -> dict:
        """Average scalar action metrics across all validation ranks.

        Artifact paths remain rank-local, but every scalar reported to logs /
        WandB must describe the same distributed sample set as ``val_loss``.
        NaNs represent an unavailable optional metric and are ignored.
        """
        binary = (action_metrics or {}).get("binary_like") or {}
        names = (
            "action_l2",
            "action_l1",
            "binary_like_mean_recall",
            "binary_like_mean_precision",
            "binary_like_mean_f1",
        )
        local = {
            "action_l2": (action_metrics or {}).get("action_l2", float("nan")),
            "action_l1": (action_metrics or {}).get("action_l1", float("nan")),
            "binary_like_mean_recall": binary.get("mean_recall", float("nan")),
            "binary_like_mean_precision": binary.get("mean_precision", float("nan")),
            "binary_like_mean_f1": binary.get("mean_f1", float("nan")),
        }
        gathered = self._gather_metric_vector([float(local[name]) for name in names])
        result = {}
        for idx, name in enumerate(names):
            column = gathered[:, idx]
            valid = ~torch.isnan(column)
            if bool(valid.any().item()):
                result[name] = float(column[valid].mean().item())
        if "action_l2" in result:
            result["action_rmse"] = float(max(result["action_l2"], 0.0) ** 0.5)
        return result

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        restore_offload = bool(getattr(self, "_training_vae_offloaded", False))
        if restore_offload:
            self._move_vae(self.accelerator.device)
        was_dit_training = self._get_trainable_backbone(model).training
        model.eval()

        # Per-rank eval batch size: match training batch_size to avoid OOM.
        eval_bs = self.batch_size
        rng = torch.Generator(device="cpu").manual_seed(
            self.global_step + self.accelerator.process_index
        )
        # Draw a batch of indices (no replacement for better coverage).
        dataset_len = len(self.val_dataset)
        eval_indices = torch.randint(
            0, dataset_len, (min(eval_bs, dataset_len),), generator=rng
        ).tolist()

        try:
            # Both AR and non-AR modes share the same batch eval logic.
            return self._evaluate_batch(model, eval_indices)
        finally:
            if was_dit_training:
                self._set_dit_only_train_mode()
            if restore_offload:
                self._move_vae("cpu")

    def _dataset_stats_sha256(self) -> str:
        stats_path = Path(self.output_dir) / "dataset_stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Dataset stats required for checkpointing were not found: {stats_path}"
            )
        return hashlib.sha256(stats_path.read_bytes()).hexdigest()

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(
            ckpt_path,
            step=self.global_step,
            dataset_stats_sha256=self._dataset_stats_sha256(),
        )
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        latest_dir = os.path.join(self.state_dir, "latest")
        tmp_dir = os.path.join(self.state_dir, "latest_tmp")
        if self.accelerator.is_main_process and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(output_dir=tmp_dir)
        if self.accelerator.is_main_process:
            self._save_trainer_state(tmp_dir)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            prev_dir = os.path.join(self.state_dir, "latest_prev")
            if os.path.isdir(prev_dir):
                shutil.rmtree(prev_dir, ignore_errors=True)
            if os.path.isdir(latest_dir):
                os.rename(latest_dir, prev_dir)
            os.rename(tmp_dir, latest_dir)
            if os.path.isdir(prev_dir):
                shutil.rmtree(prev_dir, ignore_errors=True)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": latest_dir}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        if not state_file.exists():
            raise ValueError(
                f"Training state is missing strict trainer metadata: {state_file}"
            )
        with open(state_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.accelerator.load_state(input_dir=state_dir)
        self.global_step = int(payload["global_step"])

        if "epoch" in payload and "batch_in_epoch" in payload:
            self.epoch = int(payload["epoch"])
            self.batch_in_epoch = int(payload["batch_in_epoch"])
            self.train_sampler.set_epoch(self.epoch)
            self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
            logger.info(
                "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                self.epoch,
                self.batch_in_epoch,
                self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
            )
        else:
            self.epoch = 0
            self.batch_in_epoch = 0
            self.train_sampler.clear_resume_batch_offset()
            logger.warning(
                "State file does not contain `epoch`/`batch_in_epoch`; "
                "optimizer/scheduler were restored, but dataloader progress resume is skipped."
            )
        self.accelerator.wait_for_everyone()

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)

        gc.collect()
        gc.freeze()
        gc.disable()

        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        if torch.cuda.is_available():
            # Peak stats are only read on log steps; reset once here and
            # again after each log read, so the reported peak covers the
            # whole logging window instead of a single step.
            torch.cuda.reset_peak_memory_stats()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue
            self._assert_training_latent_contract(sample)

            with self.accelerator.accumulate(self.model):
                # ``loss_dict`` is consumed only on optimizer steps that are
                # actually logged. Avoid diagnostic reductions and CUDA host
                # synchronizations on all other microsteps.
                compute_metrics = bool(
                    self.accelerator.sync_gradients
                    and self.log_every > 0
                    and (self.global_step + 1) % self.log_every == 0
                )

                with self.accelerator.autocast():
                    loss, loss_dict = self._call_training_loss(
                        self.model, sample, compute_metrics=compute_metrics
                    )

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    if should_log and (self.global_step // self.log_every) % _GC_COLLECT_LOG_INTERVAL == 0:
                        # Full cyclic collection at a reduced cadence; the
                        # per-step objects are handled by refcounting, and
                        # automatic collection stays disabled (gc.disable
                        # above) to avoid pause jitter on hot steps.
                        gc.collect()

                    global_loss = None
                    global_loss_metrics = {}
                    global_grad_norm = None
                    global_peak_memory_allocated_gib = None
                    global_peak_memory_reserved_gib = None
                    if should_log:
                        metric_keys = list(loss_dict.keys())
                        metric_values = [loss.detach().float().reshape(())]
                        for key in metric_keys:
                            metric_value = loss_dict[key]
                            if torch.is_tensor(metric_value):
                                if metric_value.numel() != 1:
                                    raise ValueError(
                                        "loss_dict metrics must be scalar; "
                                        f"{key!r} has shape {tuple(metric_value.shape)}"
                                    )
                                metric_values.append(
                                    metric_value.detach().to(
                                        device=loss.device, dtype=torch.float32
                                    ).reshape(())
                                )
                            else:
                                metric_values.append(
                                    torch.tensor(
                                        float(metric_value),
                                        device=loss.device,
                                        dtype=torch.float32,
                                    )
                                )
                        metric_values.append(torch.as_tensor(grad_norm, device=loss.device, dtype=torch.float32).reshape(()))
                        gathered_matrix = self.accelerator.gather(torch.stack(metric_values).unsqueeze(0))
                        gathered_mean = gathered_matrix.mean(dim=0)
                        gathered_values = gathered_mean.detach().cpu().tolist()
                        global_loss = float(gathered_values[0])
                        global_loss_metrics = self._metrics_from_gathered_matrix(
                            metric_keys,
                            gathered_matrix[:, 1:-1],
                        )
                        global_grad_norm = float(gathered_values[-1])
                        if torch.cuda.is_available():
                            gib = float(1024 ** 3)
                            local_memory = torch.tensor(
                                [[
                                    torch.cuda.max_memory_allocated() / gib,
                                    torch.cuda.max_memory_reserved() / gib,
                                ]],
                                device=loss.device,
                                dtype=torch.float32,
                            )
                            gathered_memory = self.accelerator.gather(local_memory)
                            global_peak_memory_allocated_gib = float(
                                gathered_memory[:, 0].max().detach().cpu().item()
                            )
                            global_peak_memory_reserved_gib = float(
                                gathered_memory[:, 1].max().detach().cpu().item()
                            )
                            # Restart the peak window for the next log interval.
                            torch.cuda.reset_peak_memory_stats()

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        if global_peak_memory_allocated_gib is not None:
                            description += (
                                "peak_allocated_gib=%.2f peak_reserved_gib=%.2f "
                                % (
                                    global_peak_memory_allocated_gib,
                                    global_peak_memory_reserved_gib,
                                )
                            )
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        if global_peak_memory_allocated_gib is not None:
                            wandb_payload["performance/peak_memory_allocated_gib"] = (
                                global_peak_memory_allocated_gib
                            )
                            wandb_payload["performance/peak_memory_reserved_gib"] = (
                                global_peak_memory_reserved_gib
                            )
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            # AR mode: val_loss + loss components + (optional) PSNR/SSIM.
                            if "psnr_rd" not in metrics:
                                description = "[eval] step=%d val_loss=%.4f" % (
                                    self.global_step, metrics["val_loss"],
                                )
                                loss_component_items = sorted(
                                    (k, v) for k, v in metrics.items()
                                    if k.startswith("loss_") and isinstance(v, float)
                                )
                                if loss_component_items:
                                    description += " " + " ".join(
                                        f"{k}={v:.4f}" for k, v in loss_component_items
                                    )
                                for psnr_key in ("psnr_ar_tf_vs_gt",):
                                    if psnr_key in metrics:
                                        description += " %s=%.4f" % (psnr_key, metrics[psnr_key])
                                for ssim_key in ("ssim_ar_tf_vs_gt",):
                                    if ssim_key in metrics:
                                        description += " %s=%.4f" % (ssim_key, metrics[ssim_key])
                                if "action_l2" in metrics:
                                    description += " action_l2=%.4f" % metrics["action_l2"]
                                if "action_l1" in metrics:
                                    description += " action_l1=%.4f" % metrics["action_l1"]
                                if "action_rmse" in metrics:
                                    description += " action_rmse=%.4f" % metrics["action_rmse"]
                                for metric_key in (
                                    "action_discrete_micro_f1",
                                    "action_discrete_macro_f1",
                                    "action_discrete_micro_recall",
                                    "action_discrete_micro_pred_pos_ratio",
                                    "action_discrete_micro_gt_pos_ratio",
                                ):
                                    if metric_key in metrics:
                                        description += " %s=%.4f" % (metric_key, metrics[metric_key])
                                logger.info(description)
                                eval_payload = {"eval/val_loss": float(metrics["val_loss"])}
                                for key, value in loss_component_items:
                                    eval_payload[f"eval/{key}"] = float(value)
                                for key in (
                                    "action_l2",
                                    "action_l1",
                                    "action_rmse",
                                    "binary_like_mean_recall",
                                    "binary_like_mean_precision",
                                    "binary_like_mean_f1",
                                    "psnr_ar_tf_vs_gt",
                                    "ssim_ar_tf_vs_gt",
                                ):
                                    if key in metrics:
                                        eval_payload[f"eval/{key}"] = float(metrics[key])
                                for metric_key, metric_value in metrics.items():
                                    if metric_key.startswith("action_discrete_") and isinstance(metric_value, (int, float)):
                                        eval_payload[f"eval/{metric_key}"] = float(metric_value)
                                self._wandb_log(eval_payload)
                            else:
                                description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                                loss_component_items = sorted(
                                    (k, v) for k, v in metrics.items() if k.startswith("loss_")
                                )
                                if loss_component_items:
                                    description += " " + " ".join(
                                        f"{k}={v:.4f}" for k, v in loss_component_items
                                    )
                                if "action_l2" in metrics:
                                    description += " action_l2=%.4f" % metrics["action_l2"]
                                if "action_l1" in metrics:
                                    description += " action_l1=%.4f" % metrics["action_l1"]
                                if "action_rmse" in metrics:
                                    description += " action_rmse=%.4f" % metrics["action_rmse"]
                                for metric_key in (
                                    "action_discrete_micro_f1",
                                    "action_discrete_macro_f1",
                                    "action_discrete_micro_recall",
                                    "action_discrete_micro_pred_pos_ratio",
                                    "action_discrete_micro_gt_pos_ratio",
                                ):
                                    if metric_key in metrics:
                                        description += " %s=%.4f" % (metric_key, metrics[metric_key])
                                logger.info(description)
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                    "eval/psnr_rg": float(metrics["psnr_rg"]),
                                    "eval/ssim_rg": float(metrics["ssim_rg"]),
                                    "eval/psnr_rd": float(metrics["psnr_rd"]),
                                    "eval/ssim_rd": float(metrics["ssim_rd"]),
                                    "eval/psnr_dg": float(metrics["psnr_dg"]),
                                    "eval/ssim_dg": float(metrics["ssim_dg"]),
                                }
                                for key, value in loss_component_items:
                                    eval_payload[f"eval/{key}"] = float(value)
                                for key in (
                                    "action_l2",
                                    "action_l1",
                                    "action_rmse",
                                    "binary_like_mean_recall",
                                    "binary_like_mean_precision",
                                    "binary_like_mean_f1",
                                ):
                                    if key in metrics:
                                        eval_payload[f"eval/{key}"] = float(metrics[key])
                                for metric_key, metric_value in metrics.items():
                                    if metric_key.startswith("action_discrete_") and isinstance(metric_value, (int, float)):
                                        eval_payload[f"eval/{metric_key}"] = float(metric_value)
                                self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
