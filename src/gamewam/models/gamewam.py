import math
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

from gamewam.utils.logging_config import get_logger

from .action.action_dit import ActionDiT
from .auxiliary.event_head import (
    EVENT_OCCURRENCE_IGNORE_INDEX,
    EventOccurrenceHead,
    event_occurrence_ce_loss,
)
from .wan22.helpers.loader import load_wan22_ti2v_5b_components
from .history.history_module import (
    HistoryConfig,
    HistoryModule,
    HistoryGeometry,
    derive_history_geometry,
)
from .state_encoders import (
    build_proprio_token_encoder,
    is_field_valid_gated_proprio_encoder,
)
from .mot import MoT
from .wan22.wan_video_dit import sinusoidal_embedding_1d
from .wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


# Cache for static AR plan geometry tensors (action RoPE position ids and
# the dense->plan index map). Keys are (geometry..., device) tuples; values
# are small read-only int64 tensors. Entries are only ever read downstream
# (index_select / expand / slice), never mutated in place.
_AR_PLAN_GEOMETRY_CACHE: dict = {}



class GameWAM(torch.nn.Module):
    """MoT world model with video and action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        proprio_encoder_type: str = "generic_linear",
        proprio_field_valid_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        coupled_video_action_timesteps: bool = True,
        video_action_attention_mode: str = "condition_only",
        history_config: Optional[HistoryConfig] = None,
        num_frames: int = 33,
        num_obs_frames: int = 17,
        frame_stride: int = 1,
        action_video_freq_ratio: int = 2,
        prediction_horizon: Optional[int] = None,
        execution_horizon: Optional[int] = None,
        text_condition_dropout_prob: float = 0.0,
        prompt_template: Optional[str] = None,
        event_occurrence_aux_enabled: bool = False,
        event_occurrence_aux_loss_weight: float = 0.01,
        action_head_dim_is_continuous: Optional[Sequence[bool]] = None,
        action_head_gui_head_enabled: bool = False,
        action_head_gui_label_dim_index: Optional[int] = None,
        action_head_lambda_continuous: float = 1.0,
        action_head_lambda_discrete: float = 1.0,
        action_head_lambda_gui_route: float = 0.05,
        action_head_route_aware_action_normalization: bool = False,
        action_noise_share_across_blocks: bool = False,
        image_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        valid_video_action_attention_modes = {
            "condition_only",
            "joint_bidirectional",
        }
        video_action_attention_mode = str(video_action_attention_mode)
        if video_action_attention_mode not in valid_video_action_attention_modes:
            raise ValueError(
                "`video_action_attention_mode` must be one of "
                f"{sorted(valid_video_action_attention_modes)}, got "
                f"{video_action_attention_mode!r}."
            )
        self.video_action_attention_mode = video_action_attention_mode
        if getattr(self.mot, "video_action_attention_mode", video_action_attention_mode) != video_action_attention_mode:
            raise ValueError(
                "GameWAM and MoT video_action_attention_mode mismatch: "
                f"{video_action_attention_mode!r} vs "
                f"{getattr(self.mot, 'video_action_attention_mode', None)!r}."
            )
        self.requires_joint_video_prediction = (
            video_action_attention_mode == "joint_bidirectional"
        )
        if self.requires_joint_video_prediction:
            logger.info(
                "video_action_attention_mode=joint_bidirectional: "
                "all inference calls will denoise video and action jointly."
            )

        # AR mode: derived from video_expert attention mask mode
        self.ar_mode = (getattr(video_expert, "video_attention_mask_mode", "") == "ar_teacher_forcing")
        self.num_frames = int(num_frames)
        self.num_obs_frames = int(num_obs_frames)
        self.frame_stride = int(frame_stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)

        # Text CFG training
        text_condition_dropout_prob = float(text_condition_dropout_prob)
        if not 0.0 <= text_condition_dropout_prob <= 1.0:
            raise ValueError(
                f"text_condition_dropout_prob must be in [0.0, 1.0], got {text_condition_dropout_prob}"
            )
        self.text_condition_dropout_prob = text_condition_dropout_prob
        if prompt_template is not None and "{task}" not in str(prompt_template):
            raise ValueError(f"`prompt_template` must contain `{{task}}`, got {prompt_template}")
        self.prompt_template = str(prompt_template) if prompt_template else None

        # AR-history mode
        if history_config is not None and not isinstance(history_config, HistoryConfig):
            # Convert dict (from Hydra config) to HistoryConfig
            history_config = HistoryConfig(**dict(history_config))
        if history_config is not None and not self.ar_mode:
            # Fail fast instead of silently dropping the configured history
            # module. Mirrors the same guard in `validate_horizon_config`.
            raise ValueError(
                "history_config requires an AR video DiT "
                "(`video_attention_mask_mode='ar_teacher_forcing'`), got "
                f"mask mode "
                f"{getattr(video_expert, 'video_attention_mask_mode', None)!r}. "
                "Either drop `history_config` or switch the video DiT mask mode."
            )
        self.history_config = history_config
        self.ar_history_mode = (self.ar_mode and history_config is not None)
        self.history_module: Optional[HistoryModule] = None
        self.ar_geometry: Optional[HistoryGeometry] = None
        self.history_geometry: Optional[HistoryGeometry] = None

        committed_future_horizon = (int(num_frames) - int(num_obs_frames)) * int(frame_stride)
        if self.ar_mode:
            self.ar_geometry = derive_history_geometry(
                num_frames=num_frames,
                num_obs_frames=num_obs_frames,
                frame_stride=frame_stride,
                action_video_freq_ratio=action_video_freq_ratio,
            )
        if execution_horizon is None:
            execution_horizon = (
                int(self.ar_geometry.raw_steps_per_latent)
                if self.ar_geometry is not None
                else committed_future_horizon
            )
        self.execution_horizon = int(execution_horizon)
        if prediction_horizon is None:
            prediction_horizon = self.execution_horizon if self.ar_mode else committed_future_horizon
        self.prediction_horizon = int(prediction_horizon)
        if self.prediction_horizon <= 0:
            raise ValueError(f"prediction_horizon must be positive, got {self.prediction_horizon}")
        if self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
        if self.execution_horizon > self.prediction_horizon:
            raise ValueError(
                "execution_horizon must not exceed prediction_horizon, "
                f"got {self.execution_horizon} > {self.prediction_horizon}"
            )

        if self.ar_mode:
            if self.prediction_horizon % self.execution_horizon != 0:
                raise ValueError(
                    "AR prediction_horizon must be divisible by execution_horizon, "
                    f"got {self.prediction_horizon} and {self.execution_horizon}"
                )
            if self.execution_horizon != int(self.ar_geometry.raw_steps_per_latent):
                raise ValueError(
                    "AR execution_horizon must equal raw_steps_per_latent, "
                    f"got execution_horizon={self.execution_horizon}, "
                    f"raw_steps_per_latent={self.ar_geometry.raw_steps_per_latent}"
                )
            committed_future_horizon = (int(num_frames) - int(num_obs_frames)) * int(frame_stride)
            if committed_future_horizon % self.execution_horizon != 0:
                raise ValueError(
                    "AR committed future horizon must be divisible by execution_horizon, "
                    f"got {committed_future_horizon} and {self.execution_horizon}"
                )
            if (self.prediction_horizon - self.execution_horizon) % int(frame_stride) != 0:
                raise ValueError(
                    "AR lookahead must align with frame_stride, got "
                    f"prediction_horizon={self.prediction_horizon}, "
                    f"execution_horizon={self.execution_horizon}, frame_stride={frame_stride}"
                )
        else:
            committed_future_horizon = (int(num_frames) - int(num_obs_frames)) * int(frame_stride)
            if committed_future_horizon != self.prediction_horizon:
                raise ValueError(
                    "Non-AR geometry must satisfy "
                    "(num_frames - num_obs_frames) * frame_stride == prediction_horizon, "
                    f"got {committed_future_horizon} != {self.prediction_horizon}"
                )

        if self.ar_history_mode:
            self.history_geometry = self.ar_geometry
            if image_size is None:
                raise ValueError(
                    "AR-history mode requires `image_size` (H, W) to size the "
                    "history module's segment token embedding. Pass it via "
                    "create_gamewam / from_wan22_pretrained (typically interpolated "
                    "from data.common.shape_meta.images.0.shape in the model config)."
                )
            upsampling_factor = int(getattr(vae, "upsampling_factor", 16))
            latent_spatial_size = (
                int(image_size[0]) // upsampling_factor,
                int(image_size[1]) // upsampling_factor,
            )
            self.history_module = HistoryModule(
                hidden_dim=video_expert.hidden_dim,
                history_config=history_config,
                geometry=self.ar_geometry,
                latent_in_dim=video_expert.in_dim,
                init_patch_embedding=video_expert.patch_embedding,
                latent_spatial_size=latent_spatial_size,
            )
            logger.info(
                "AR-history config: recent_num_segments=%d, memory_half_life_segments=%s, "
                "derived_memory_write_capacity_segments=%d, derived_max_history_segments=%d, "
                "history_horizon_dropout=%.3f, latent_spatial_size=%s",
                history_config.recent_num_segments,
                history_config.memory_half_life_segments,
                history_config.memory_write_capacity_segments,
                history_config.max_history_segments,
                history_config.history_horizon_dropout,
                latent_spatial_size,
            )

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder_type = str(proprio_encoder_type)
        self.proprio_field_valid_dim = (
            None
            if proprio_field_valid_dim is None
            else int(proprio_field_valid_dim)
        )
        self.cursor_reference_size: Optional[tuple[float, float]] = None
        if self.proprio_dim is not None:
            self.proprio_encoder = build_proprio_token_encoder(
                self.proprio_dim,
                self.text_dim,
                encoder_type=self.proprio_encoder_type,
                field_valid_dim=self.proprio_field_valid_dim,
            ).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.coupled_video_action_timesteps = bool(coupled_video_action_timesteps)

        # Action heads config (mirrors ``ActionDiT``). The
        # data-schema fields (``dim_is_continuous`` / ``gui_label_dim_index``)
        # are passed in by the caller; the model config interpolates them
        # from the data config (see ``configs/model/wan_uncond_ar.yaml``).
        self.action_head_dim_is_continuous = (
            [bool(x) for x in action_head_dim_is_continuous]
            if action_head_dim_is_continuous is not None
            else None
        )
        self.action_head_gui_head_enabled = bool(action_head_gui_head_enabled)
        self.action_head_gui_label_dim_index = action_head_gui_label_dim_index
        self.action_head_lambda_continuous = float(action_head_lambda_continuous)
        self.action_head_lambda_discrete = float(action_head_lambda_discrete)
        self.action_head_lambda_gui_route = float(action_head_lambda_gui_route)
        self.action_head_route_aware_action_normalization = bool(
            action_head_route_aware_action_normalization
        )
        # Convenience aliases used by training/inference code.
        self.continuous_dims: List[int] = (
            [i for i, c in enumerate(self.action_head_dim_is_continuous) if c]
            if self.action_head_dim_is_continuous is not None
            else list(range(int(action_expert.action_dim)))
        )
        self.discrete_dims: List[int] = (
            [i for i, c in enumerate(self.action_head_dim_is_continuous) if not c]
            if self.action_head_dim_is_continuous is not None
            else []
        )
        self.continuous_action_dim = len(self.continuous_dims)
        self.discrete_action_dim = len(self.discrete_dims)

        # The action expert's total dim must equal
        # ``len(continuous_dims) + len(discrete_dims)`` — catches config
        # drift between the action expert's ``action_dim`` and the
        # model's ``dim_is_continuous`` setting.
        if self.continuous_action_dim + self.discrete_action_dim != int(action_expert.action_dim):
            raise ValueError(
                f"Action expert `action_dim` ({int(action_expert.action_dim)}) does not match "
                f"continuous_dims + discrete_dims = "
                f"{self.continuous_action_dim} + {self.discrete_action_dim}. "
                "Check `dim_is_continuous` length against `action_dim`."
            )

        # ``GameWAM`` and ``ActionDiT`` derive their continuous /
        # discrete dims from the SAME ``dim_is_continuous`` list —
        # catch silent drift between the model and the action expert.
        if list(action_expert.continuous_dims) != list(self.continuous_dims):
            raise ValueError(
                "Action-head layout drift: "
                f"GameWAM.continuous_dims = {self.continuous_dims}, "
                f"but ActionDiT.continuous_dims = "
                f"{list(action_expert.continuous_dims)}. "
                "Both are derived from `dim_is_continuous`; the two "
                "should be identical."
            )
        if list(action_expert.discrete_dims) != list(self.discrete_dims):
            raise ValueError(
                "Action-head layout drift: "
                f"GameWAM.discrete_dims = {self.discrete_dims}, "
                f"but ActionDiT.discrete_dims = "
                f"{list(action_expert.discrete_dims)}. "
                "Both are derived from `dim_is_continuous`; the two "
                "should be identical."
            )
        if bool(action_expert.gui_head_enabled) != bool(action_head_gui_head_enabled):
            raise ValueError(
                "GUI expert on/off mismatch between GameWAM and ActionDiT: "
                f"GameWAM.action_head_gui_head_enabled = "
                f"{action_head_gui_head_enabled}, "
                f"ActionDiT.gui_head_enabled = "
                f"{action_expert.gui_head_enabled}. "
                "Set both to the same value via `action_head.gui_head_enabled` "
                "in the model config (which forwards to ActionDiT as "
                "`action_dit_config.gui_head_enabled`)."
            )
        # ``lambda_gui_route`` / ``gui_head_enabled`` has no silent
        # misconfiguration we can detect — the GUI route loss is
        # only computed when ``gui_head_enabled=True`` and is a
        # no-op otherwise. Models without a GUI route (the common case)
        # keep the default 0.05 weight without spurious warnings.

        self.action_noise_share_across_blocks = bool(
            action_noise_share_across_blocks
        )

        self.event_occurrence_aux_enabled = bool(event_occurrence_aux_enabled)
        self.event_occurrence_aux_loss_weight = float(event_occurrence_aux_loss_weight)
        if self.event_occurrence_aux_enabled and (
            not math.isfinite(self.event_occurrence_aux_loss_weight)
            or self.event_occurrence_aux_loss_weight < 0.0
        ):
            raise ValueError(
                "`event_occurrence_aux_loss_weight` must be finite and >= 0, "
                f"got {event_occurrence_aux_loss_weight}"
            )
        if self.event_occurrence_aux_enabled:
            self.event_occurrence_head = EventOccurrenceHead(hidden_dim=int(action_expert.hidden_dim))
        else:
            self.event_occurrence_head = None

        self.to(self.device)
        self._sync_auxiliary_module_dtypes()

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        proprio_encoder_type: str = "generic_linear",
        proprio_field_valid_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        coupled_video_action_timesteps: bool = True,
        video_action_attention_mode: str = "condition_only",
        history_config=None,
        num_frames: int = 33,
        num_obs_frames: int = 17,
        frame_stride: int = 1,
        action_video_freq_ratio: int = 2,
        prediction_horizon: Optional[int] = None,
        execution_horizon: Optional[int] = None,
        # Text CFG training
        text_condition_dropout_prob: float = 0.0,
        prompt_template: Optional[str] = None,
        # Event-done auxiliary head (representation learning only).
        event_occurrence_aux_enabled: bool = False,
        event_occurrence_aux_loss_weight: float = 0.01,
        action_head_dim_is_continuous: Optional[Sequence[bool]] = None,
        action_head_gui_head_enabled: bool = False,
        action_head_gui_label_dim_index: Optional[int] = None,
        action_head_lambda_continuous: float = 1.0,
        action_head_lambda_discrete: float = 1.0,
        action_head_lambda_gui_route: float = 0.05,
        action_head_route_aware_action_normalization: bool = False,
        action_noise_share_across_blocks: bool = False,
        image_size: Optional[Sequence[int]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for GameWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for GameWAM.")

        # Inject action head config into ``action_dit_config``
        # so the ``ActionDiT`` constructor receives it (keeps the
        # config the single source of truth when hydra-resolved).
        action_dit_config = dict(action_dit_config or {})
        if "dim_is_continuous" not in action_dit_config:
            action_dit_config["dim_is_continuous"] = (
                list(action_head_dim_is_continuous)
                if action_head_dim_is_continuous is not None
                else None
            )
        if "gui_head_enabled" not in action_dit_config:
            action_dit_config["gui_head_enabled"] = bool(action_head_gui_head_enabled)
        if "gui_label_dim_index" not in action_dit_config:
            action_dit_config["gui_label_dim_index"] = action_head_gui_label_dim_index

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            video_action_attention_mode=video_action_attention_mode,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            proprio_encoder_type=proprio_encoder_type,
            proprio_field_valid_dim=proprio_field_valid_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            coupled_video_action_timesteps=coupled_video_action_timesteps,
            video_action_attention_mode=video_action_attention_mode,
            prompt_template=prompt_template,
            history_config=history_config,
            num_frames=num_frames,
            num_obs_frames=num_obs_frames,
            frame_stride=frame_stride,
            action_video_freq_ratio=action_video_freq_ratio,
            prediction_horizon=prediction_horizon,
            execution_horizon=execution_horizon,
            text_condition_dropout_prob=text_condition_dropout_prob,
            event_occurrence_aux_enabled=event_occurrence_aux_enabled,
            event_occurrence_aux_loss_weight=event_occurrence_aux_loss_weight,
            # Data-schema fields for the action head layout.
            action_head_dim_is_continuous=action_dit_config.get("dim_is_continuous"),
            action_head_gui_head_enabled=bool(action_dit_config.get("gui_head_enabled", False)),
            action_head_gui_label_dim_index=action_dit_config.get("gui_label_dim_index"),
            action_head_lambda_continuous=action_head_lambda_continuous,
            action_head_lambda_discrete=action_head_lambda_discrete,
            action_head_lambda_gui_route=action_head_lambda_gui_route,
            action_head_route_aware_action_normalization=bool(
                action_head_route_aware_action_normalization
            ),
            action_noise_share_across_blocks=bool(
                action_noise_share_across_blocks
            ),
            image_size=image_size,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        if hasattr(self, "device") and hasattr(self, "torch_dtype"):
            self._sync_auxiliary_module_dtypes()
        return self

    def _sync_auxiliary_module_dtypes(self) -> None:
        if self.history_module is not None:
            self.history_module.to(device=self.device, dtype=self.torch_dtype)
        if self.proprio_encoder is not None:
            self.proprio_encoder.to(device=self.device, dtype=self.torch_dtype)
        if getattr(self, "event_occurrence_head", None) is not None:
            self.event_occurrence_head.to(device=self.device, dtype=self.torch_dtype)

    def _build_coupled_action_infer_schedule(
        self,
        infer_timesteps_video: torch.Tensor,
        action_dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(infer_timesteps_video.shape[0]),
            device=infer_timesteps_video.device,
            dtype=action_dtype,
            shift_override=shift_override,
        )

    def _runtime_contract(self, dataset_stats_sha256: Optional[str] = None) -> dict[str, Any]:
        """Configuration values that change the meaning of trained weights."""
        encoder_type = str(self.proprio_encoder_type)
        encoder = getattr(self, "proprio_encoder", None)
        field_valid_dim = getattr(encoder, "field_valid_dim", None)
        if encoder_type == "minecraft_grouped_cursor":
            reference_size = self.cursor_reference_size
            if reference_size is None or len(reference_size) != 2:
                raise ValueError(
                    "minecraft_grouped_cursor requires a bound "
                    "cursor_reference_size before saving or loading checkpoints."
                )
            cursor_reference_size = [
                float(reference_size[0]), float(reference_size[1])
            ]
            if cursor_reference_size[0] <= 0 or cursor_reference_size[1] <= 0:
                raise ValueError(
                    "cursor_reference_size must contain positive width/height, "
                    f"got {cursor_reference_size}."
                )
        else:
            cursor_reference_size = None
        return {
            "num_frames": int(self.num_frames),
            "num_obs_frames": int(self.num_obs_frames),
            "frame_stride": int(self.frame_stride),
            "action_video_freq_ratio": int(
                self.action_video_freq_ratio
            ),
            "prediction_horizon": int(self.prediction_horizon),
            "execution_horizon": int(self.execution_horizon),
            "ar_mode": bool(self.ar_mode),
            "video_action_attention_mode": str(
                self.video_action_attention_mode
            ),
            "prompt_template": self.prompt_template,
            "proprio_dim": self.proprio_dim,
            "proprio_encoder_type": encoder_type,
            "field_valid_dim": (
                None if field_valid_dim is None else int(field_valid_dim)
            ),
            "cursor_reference_size": cursor_reference_size,
            "dataset_stats_sha256": dataset_stats_sha256,
        }

    def _action_head_contract(self) -> dict[str, Any]:
        return {
            "dim_is_continuous": (
                list(self.action_head_dim_is_continuous)
                if self.action_head_dim_is_continuous is not None
                else None
            ),
            "gui_head_enabled": bool(
                self.action_head_gui_head_enabled
            ),
            "gui_label_dim_index": self.action_head_gui_label_dim_index,
            "action_normalization": {
                "route_aware": bool(
                    self.action_head_route_aware_action_normalization
                ),
                "continuous_dims": (
                    list(self.continuous_dims)
                    if self.continuous_dims is not None
                    else None
                ),
                "gui_label_dim_index": (
                    int(self.action_head_gui_label_dim_index)
                    if self.action_head_gui_head_enabled
                    and self.action_head_gui_label_dim_index is not None
                    else None
                ),
            },
        }

    def _continuous_action_flow_contract(self) -> dict[str, Any]:
        return {
            "block_horizon": int(self.prediction_horizon),
            "ordinary_block_noise_shared": bool(
                self.action_noise_share_across_blocks
            ),
        }

    def _history_contract(self) -> Optional[dict[str, Any]]:
        if not self.ar_history_mode or self.history_config is None:
            return None
        return {
            "recent_num_segments": self.history_config.recent_num_segments,
            "history_conv_kernel_size": list(self.history_config.history_conv_kernel_size),
            "history_conv_stride": list(self.history_config.history_conv_stride),
            "memory_num_levels": self.history_config.memory_num_levels,
            "memory_tokens_per_level": self.history_config.memory_tokens_per_level,
            "memory_half_life_segments": list(self.history_config.memory_half_life_segments),
            "history_horizon_dropout": self.history_config.history_horizon_dropout,
            "history_segment_tokenize_chunk_size": self.history_config.history_segment_tokenize_chunk_size,
            "predictive_loss_weight": float(
                getattr(self.history_config, "predictive_loss_weight", 0.0)
            ),
        }

    def checkpoint_contract(
        self, dataset_stats_sha256: Optional[str] = None
    ) -> dict[str, Any]:
        """Return the strict model/runtime contract without weight tensors."""
        payload: dict[str, Any] = {
            "runtime_contract": GameWAM._runtime_contract(
                self, dataset_stats_sha256
            ),
            "action_head": GameWAM._action_head_contract(self),
            "continuous_action_flow": GameWAM._continuous_action_flow_contract(self),
        }
        history_contract = GameWAM._history_contract(self)
        if history_contract is not None:
            payload["history_config"] = history_contract
        return payload

    def _checkpoint_payload(
        self,
        step: Optional[int] = None,
        dataset_stats_sha256: Optional[str] = None,
    ) -> dict[str, Any]:
        if not isinstance(dataset_stats_sha256, str) or not dataset_stats_sha256:
            raise ValueError("`dataset_stats_sha256` is required when saving a checkpoint.")
        payload: dict[str, Any] = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            **GameWAM.checkpoint_contract(self, dataset_stats_sha256),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.ar_history_mode and self.history_config is not None:
            payload["history_module"] = self.history_module.state_dict()
        if self.event_occurrence_aux_enabled and self.event_occurrence_head is not None:
            payload["event_occurrence_aux_enabled"] = True
            payload["event_occurrence_aux_loss_weight"] = float(self.event_occurrence_aux_loss_weight)
            payload["event_occurrence_head"] = self.event_occurrence_head.state_dict()
        return payload

    def save_checkpoint(
        self,
        path: str,
        step: Optional[int] = None,
        dataset_stats_sha256: Optional[str] = None,
    ) -> None:
        """Save the trainable GameWAM weights.

        The base Wan weights are loaded from config before a checkpoint is
        applied, so checkpoints only need the modules trained by this trainer.
        """
        torch.save(
            self._checkpoint_payload(
                step=step,
                dataset_stats_sha256=dataset_stats_sha256,
            ),
            path,
        )

    @staticmethod
    def _validate_checkpoint_state_dict(
        module,
        state_dict: Any,
        *,
        module_name: str,
        path: str,
    ) -> None:
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Checkpoint `{module_name}` must be a state-dict mapping: {path}"
            )
        expected_state = module.state_dict()
        expected_keys = set(expected_state)
        saved_keys = set(state_dict)
        missing_keys = sorted(expected_keys - saved_keys)
        unexpected_keys = sorted(saved_keys - expected_keys)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"{module_name} checkpoint state mismatch: "
                f"missing={missing_keys[:10]}, unexpected={unexpected_keys[:10]}. "
                f"Path: {path}"
            )
        for key, expected_value in expected_state.items():
            saved_value = state_dict[key]
            if not isinstance(saved_value, torch.Tensor):
                raise ValueError(
                    f"Checkpoint `{module_name}.{key}` must be a tensor: {path}"
                )
            if tuple(saved_value.shape) != tuple(expected_value.shape):
                raise RuntimeError(
                    f"{module_name} checkpoint shape mismatch for {key}: "
                    f"saved={tuple(saved_value.shape)}, "
                    f"current={tuple(expected_value.shape)}. Path: {path}"
                )

    def load_checkpoint(
        self,
        path: str,
        *,
        mode: str = "resume",
        skip_event_occurrence_head: bool = False,
        dataset_stats_sha256: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load a checkpoint under an explicit lifecycle contract.

        ``resume`` is the strict training default. ``inference`` may skip the
        training-only event head while keeping backbone and runtime contracts
        strict.
        """
        valid_modes = {"resume", "inference"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown checkpoint load mode {mode!r}; expected one of {sorted(valid_modes)}"
            )
        if skip_event_occurrence_head and mode != "inference":
            raise ValueError(
                "`skip_event_occurrence_head=True` is only valid in checkpoint mode='inference'."
            )
        if not isinstance(dataset_stats_sha256, str) or not dataset_stats_sha256:
            raise ValueError("`dataset_stats_sha256` is required when loading a checkpoint.")

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint payload must be a dict: {path}")

        if "mot" not in payload:
            raise ValueError(f"Checkpoint missing `mot` key: {path}")
        if "action_head" not in payload:
            raise ValueError(
                f"Checkpoint missing `action_head` layout metadata: {path}"
            )
        if "runtime_contract" not in payload:
            raise ValueError(f"Checkpoint missing `runtime_contract`: {path}")
        if "continuous_action_flow" not in payload:
            raise ValueError(f"Checkpoint missing `continuous_action_flow`: {path}")

        ckpt_sd = payload["mot"]
        ckpt_layout = payload["action_head"]
        if not isinstance(ckpt_layout, dict):
            raise ValueError(f"Checkpoint `action_head` must be a dict: {path}")
        ckpt_layout = dict(ckpt_layout)
        expected_action = GameWAM._action_head_contract(self)
        missing_action_keys = set(expected_action) - set(ckpt_layout)
        if missing_action_keys:
            raise ValueError(
                "Checkpoint action_head contract is incomplete; missing "
                f"{sorted(missing_action_keys)}: {path}"
            )
        if not isinstance(ckpt_layout["gui_head_enabled"], bool):
            raise ValueError(f"Checkpoint gui_head_enabled must be boolean: {path}")
        for field in (
            "dim_is_continuous",
            "gui_head_enabled",
            "gui_label_dim_index",
        ):
            ckpt_value = ckpt_layout[field]
            current_value = expected_action[field]
            if ckpt_value != current_value:
                raise ValueError(
                    f"Action-head layout mismatch: action_head.{field}: "
                    f"saved={ckpt_value!r}, current={current_value!r}. "
                    f"Path: {path}."
                )

        ckpt_action_norm = ckpt_layout["action_normalization"]
        if not isinstance(ckpt_action_norm, dict):
            raise ValueError(f"Checkpoint action_normalization must be a dict: {path}")
        expected_norm = expected_action["action_normalization"]
        missing_norm_keys = set(expected_norm) - set(ckpt_action_norm)
        if missing_norm_keys:
            raise ValueError(
                "Checkpoint action_normalization contract is incomplete; missing "
                f"{sorted(missing_norm_keys)}: {path}"
            )
        if not isinstance(ckpt_action_norm["route_aware"], bool):
            raise ValueError(f"Checkpoint action_normalization.route_aware must be boolean: {path}")
        for field, expected_value in expected_norm.items():
            ckpt_value = ckpt_action_norm[field]
            if ckpt_value != expected_value:
                raise ValueError(
                    f"Checkpoint action_normalization.{field} mismatch: "
                    f"saved={ckpt_value!r}, current={expected_value!r}. "
                    f"Path: {path}"
                )

        expected_flow = GameWAM._continuous_action_flow_contract(self)
        saved_flow = payload["continuous_action_flow"]
        if not isinstance(saved_flow, dict):
            raise ValueError(f"Checkpoint continuous_action_flow must be a dict: {path}")
        missing_flow_keys = set(expected_flow) - set(saved_flow)
        if missing_flow_keys:
            raise ValueError(
                "Checkpoint continuous_action_flow contract is incomplete; missing "
                f"{sorted(missing_flow_keys)}: {path}"
            )
        for field, expected_value in expected_flow.items():
            if saved_flow[field] != expected_value:
                raise ValueError(
                    f"Checkpoint continuous_action_flow.{field} mismatch: "
                    f"saved={saved_flow[field]!r}, current={expected_value!r}. Path: {path}"
                )

        saved_runtime_contract = payload["runtime_contract"]
        if not isinstance(saved_runtime_contract, dict):
            raise ValueError(f"Checkpoint runtime_contract must be a dict: {path}")
        runtime_fields = set(GameWAM._runtime_contract(self))
        missing_runtime_fields = runtime_fields - set(saved_runtime_contract)
        if missing_runtime_fields:
            raise ValueError(
                "Checkpoint runtime_contract is incomplete; missing "
                f"{sorted(missing_runtime_fields)}: {path}"
            )
        saved_stats_sha256 = saved_runtime_contract["dataset_stats_sha256"]
        if not isinstance(saved_stats_sha256, str) or not saved_stats_sha256:
            raise ValueError(
                f"Checkpoint runtime_contract.dataset_stats_sha256 must be non-empty: {path}"
            )
        current_runtime_contract = GameWAM._runtime_contract(
            self, dataset_stats_sha256
        )
        for field in runtime_fields:
            if saved_runtime_contract[field] != current_runtime_contract[field]:
                raise ValueError(
                    "Checkpoint runtime contract mismatch for "
                    f"{field}: saved={saved_runtime_contract[field]!r}, "
                    f"current={current_runtime_contract[field]!r}. Path: {path}"
                )

        modules_to_load = [("MoT", self.mot, ckpt_sd)]

        if self.proprio_encoder is not None:
            if "proprio_encoder" not in payload:
                raise ValueError(f"Checkpoint missing `proprio_encoder` key: {path}")
            modules_to_load.append(
                ("proprio_encoder", self.proprio_encoder, payload["proprio_encoder"])
            )
        elif "proprio_encoder" in payload:
            raise ValueError(
                "Checkpoint contains proprio weights but current model has proprio_dim=None."
            )

        if self.ar_history_mode and self.history_config is not None:
            if "history_config" not in payload:
                raise ValueError(f"Checkpoint missing `history_config` key: {path}")
            saved_cfg = payload["history_config"]
            if not isinstance(saved_cfg, dict):
                raise ValueError(f"Checkpoint `history_config` must be a dict: {path}")
            required_fields = [
                "recent_num_segments",
                "history_conv_kernel_size",
                "history_conv_stride",
                "memory_num_levels",
                "memory_tokens_per_level",
                "memory_half_life_segments",
            ]
            if mode == "resume":
                required_fields.extend(
                    ["history_horizon_dropout", "predictive_loss_weight"]
                )
            for field in required_fields:
                if field not in saved_cfg:
                    raise ValueError(
                        f"History config mismatch: checkpoint is missing {field!r}; "
                        f"saved_config={saved_cfg}"
                    )
                saved_val = saved_cfg[field]
                current_val = getattr(self.history_config, field)
                if isinstance(saved_val, list):
                    saved_val = tuple(saved_val)
                if isinstance(current_val, list):
                    current_val = tuple(current_val)
                if saved_val != current_val:
                    raise ValueError(
                        f"History config mismatch: checkpoint has {field}={saved_val}, "
                        f"but current config has {field}={current_val}. "
                        "These must match to load weights correctly."
                    )
            if "history_module" not in payload:
                raise ValueError(f"Checkpoint missing `history_module` key: {path}")
            if self.history_module is None:
                raise ValueError(
                    "Current model has history_config but no history_module instance."
                )
            modules_to_load.append(
                ("history_module", self.history_module, payload["history_module"])
            )
        elif "history_module" in payload or "history_config" in payload:
            raise ValueError(
                "Checkpoint contains history weights but current model has no history_config."
            )

        event_keys = {
            "event_occurrence_aux_enabled",
            "event_occurrence_aux_loss_weight",
            "event_occurrence_head",
        }
        if skip_event_occurrence_head:
            if event_keys & set(payload):
                logger.warning(
                    "Skipping event occurrence head while loading checkpoint %s; "
                    "this is only valid for evaluation paths that never use it.",
                    path,
                )
        elif self.event_occurrence_aux_enabled:
            if self.event_occurrence_head is None:
                raise ValueError(
                    "Current model has `event_occurrence_aux_enabled=True` but no "
                    "EventOccurrenceHead instance. This is a configuration error."
                )
            if payload.get("event_occurrence_aux_enabled") is not True:
                raise ValueError(
                    f"Checkpoint missing enabled `event_occurrence_aux_enabled`: {path}"
                )
            if "event_occurrence_head" not in payload:
                raise ValueError(f"Checkpoint missing `event_occurrence_head` key: {path}")
            if "event_occurrence_aux_loss_weight" not in payload:
                raise ValueError(
                    f"Checkpoint missing `event_occurrence_aux_loss_weight` key: {path}"
                )
            try:
                saved_loss_weight = float(payload["event_occurrence_aux_loss_weight"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Checkpoint event_occurrence_aux_loss_weight must be numeric: {path}"
                ) from exc
            current_loss_weight = float(self.event_occurrence_aux_loss_weight)
            if (
                not math.isfinite(saved_loss_weight)
                or not math.isfinite(current_loss_weight)
                or abs(saved_loss_weight - current_loss_weight) >= 1e-9
            ):
                raise ValueError(
                    "event_occurrence_aux_loss_weight mismatch: "
                    f"saved={saved_loss_weight}, current={current_loss_weight}. Path: {path}"
                )
            modules_to_load.append(
                (
                    "event_occurrence_head",
                    self.event_occurrence_head,
                    payload["event_occurrence_head"],
                )
            )
        elif event_keys & set(payload):
            raise ValueError(
                "Checkpoint contains event_occurrence_aux data but the current "
                "model has `event_occurrence_aux_enabled=False`."
            )

        for module_name, module, state_dict in modules_to_load:
            GameWAM._validate_checkpoint_state_dict(
                module,
                state_dict,
                module_name=module_name,
                path=path,
            )

        for _, module, state_dict in modules_to_load:
            module.load_state_dict(state_dict, strict=True)

        self._sync_auxiliary_module_dtypes()
        return payload

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _resolve_cfg_contexts(
        self,
        *,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Resolve both positive (cond) and negative (uncond) contexts.

        Uncond context defaults to empty-string prompt unless precomputed
        negative_context/negative_context_mask is provided.
        Appends proprio to both contexts appropriately.
        """
        # Resolve positive (cond) context
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive for positive context.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided for positive context.")

        if use_prompt:
            ctx_cond, mask_cond = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            ctx_cond = context.clone() if context.ndim == 3 else context.unsqueeze(0).clone()
            mask_cond = context_mask.clone() if context_mask.ndim == 2 else context_mask.unsqueeze(0).clone()
            ctx_cond = ctx_cond.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            mask_cond = mask_cond.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # Resolve negative (uncond) context — always empty/template-empty prompt unless precomputed context is given
        use_neg_context = negative_context is not None or negative_context_mask is not None
        if use_neg_context:
            if negative_context is None or negative_context_mask is None:
                raise ValueError("`negative_context` and `negative_context_mask` must be both provided together.")
            ctx_uncond = negative_context.clone() if negative_context.ndim == 3 else negative_context.unsqueeze(0).clone()
            mask_uncond = negative_context_mask.clone() if negative_context_mask.ndim == 2 else negative_context_mask.unsqueeze(0).clone()
            ctx_uncond = ctx_uncond.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            mask_uncond = mask_uncond.to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            # Template-formatted empty task is the only correct uncond baseline;
            # it must match the training-time condition dropout semantic.
            if self.prompt_template is None:
                raise ValueError(
                    "CFG requires `prompt_template` to construct the uncond baseline "
                    "(must match training-time condition dropout `prompt_template.format(task=\"\")`)."
                )
            uncond_text = self.prompt_template.format(task="")
            ctx_uncond, mask_uncond = self.encode_prompt(uncond_text)

        # Append proprio to both
        if proprio is not None:
            ctx_cond, mask_cond = self._append_proprio_to_context(
                context=ctx_cond,
                context_mask=mask_cond,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )
            ctx_uncond, mask_uncond = self._append_proprio_to_context(
                context=ctx_uncond,
                context_mask=mask_uncond,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )

        return ctx_cond, mask_cond, ctx_uncond, mask_uncond

    def _apply_text_condition_dropout(
        self,
        sample: dict,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple:
        """Apply sample-level text condition dropout for CFG training.

        With probability ``text_condition_dropout_prob``, replaces the
        per-sample context/context_mask with the empty-prompt (uncond) version.

        Returns:
            (context, context_mask, drop_mask):
                ``drop_mask`` is None when dropout is inactive, otherwise a
                ``[B]`` boolean tensor where ``True`` means the sample uses
                unconditional context.
        """
        prob = self.text_condition_dropout_prob
        if not getattr(self, "training", False) or prob <= 0.0:
            return context, context_mask, None

        if "uncond_context" not in sample or "uncond_context_mask" not in sample:
            raise RuntimeError(
                "text_condition_dropout_prob > 0 requires dataset to return "
                "uncond_context / uncond_context_mask"
            )

        B = context.shape[0]
        drop_mask = torch.rand(B, device=context.device) < prob

        uncond_context = sample["uncond_context"].to(
            device=context.device,
            dtype=context.dtype,
            non_blocking=True,
        )
        uncond_context_mask = sample["uncond_context_mask"].to(
            device=context_mask.device,
            dtype=torch.bool,
            non_blocking=True,
        )

        context = torch.where(drop_mask[:, None, None], uncond_context, context)
        context_mask = torch.where(drop_mask[:, None], uncond_context_mask, context_mask)
        return context, context_mask, drop_mask

    def _proprio_obs_action_steps(self) -> int:
        num_obs_frames = int(
            self.num_obs_frames
        )
        return (num_obs_frames - 1) * int(self.frame_stride)

    def _proprio_raw_step_positions(
        self, token_count: int, *, device: torch.device
    ) -> torch.Tensor:
        return torch.arange(
            int(token_count), device=device, dtype=torch.long
        ) - self._proprio_obs_action_steps()

    def _max_ar_visible_proprio_count(self) -> int:
        if self.ar_geometry is None:
            raise RuntimeError("AR geometry is required for proprio visibility.")
        return (
            self._proprio_obs_action_steps()
            + 1
            + (int(self.ar_geometry.pred_latents) - 1)
            * int(self.execution_horizon)
        )

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        if proprio.ndim != 3:
            raise ValueError(f"`proprio` must be 2D [B, D] or 3D [B, K, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[2] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
            )
        proprio = proprio.to(device=self.device, dtype=context.dtype)
        if is_field_valid_gated_proprio_encoder(self.proprio_encoder):
            if proprio_field_valid is None:
                raise ValueError("`proprio_field_valid` is required for gated proprio encoding.")
            if proprio_field_valid.ndim == 2:
                proprio_field_valid = proprio_field_valid.unsqueeze(0) if proprio.shape[0] == 1 else proprio_field_valid.unsqueeze(1)
            expected_field_valid_dim = int(
                getattr(self.proprio_encoder, "field_valid_dim")
            )
            expected_field_valid_shape = proprio.shape[:2] + (
                expected_field_valid_dim,
            )
            if proprio_field_valid.shape != expected_field_valid_shape:
                raise ValueError(
                    "`proprio_field_valid` must match [B,K] with last dim "
                    f"{expected_field_valid_dim}, got "
                    f"{tuple(proprio_field_valid.shape)} for proprio "
                    f"{tuple(proprio.shape)}"
                )
            if proprio_is_pad is not None and proprio_is_pad.ndim == 1:
                proprio_is_pad = proprio_is_pad.unsqueeze(0)
            proprio_tokens, proprio_mask = self.proprio_encoder(
                proprio,
                proprio_field_valid.to(device=self.device, dtype=torch.bool),
                None
                if proprio_is_pad is None
                else proprio_is_pad.to(device=self.device, dtype=torch.bool),
                raw_step_positions=self._proprio_raw_step_positions(
                    proprio.shape[1], device=proprio.device
                ),
            )
        else:
            if proprio_dim_is_pad is not None:
                if proprio_dim_is_pad.ndim == 1:
                    proprio_dim_is_pad = proprio_dim_is_pad.view(1, 1, -1).expand(proprio.shape[0], proprio.shape[1], -1)
                elif proprio_dim_is_pad.ndim == 2:
                    if proprio_dim_is_pad.shape[0] == proprio.shape[0]:
                        proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(1).expand(-1, proprio.shape[1], -1)
                    elif proprio.shape[0] == 1 and proprio_dim_is_pad.shape[0] == proprio.shape[1]:
                        proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(0)
                    else:
                        raise ValueError(
                            "`proprio_dim_is_pad` 2D must be [B, D] or [K, D] for B=1, "
                            f"got {tuple(proprio_dim_is_pad.shape)} for proprio {tuple(proprio.shape)}"
                        )
                elif proprio_dim_is_pad.ndim != 3:
                    raise ValueError(
                        "`proprio_dim_is_pad` must be 1D [D], 2D [B,D]/[K,D], "
                        f"or 3D [B,K,D], got shape {tuple(proprio_dim_is_pad.shape)}"
                    )
                if tuple(proprio_dim_is_pad.shape) != tuple(proprio.shape):
                    raise ValueError(
                        "`proprio_dim_is_pad` must match normalized proprio shape, "
                        f"got {tuple(proprio_dim_is_pad.shape)} vs {tuple(proprio.shape)}"
                    )
                proprio = torch.where(proprio_dim_is_pad.to(device=self.device, dtype=torch.bool), torch.zeros_like(proprio), proprio)
            proprio_tokens = self.proprio_encoder(proprio).to(dtype=context.dtype)
            K = proprio_tokens.shape[1]
            proprio_mask = torch.ones((context_mask.shape[0], K), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_tokens.to(dtype=context.dtype)], dim=1),
            torch.cat([context_mask, proprio_mask.to(device=context_mask.device)], dim=1),
        )

    def _append_sample_proprio_to_context(
        self,
        sample: dict,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask

        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
        if proprio.ndim != 3:
            raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
        if proprio.shape[2] != self.proprio_dim:
            raise ValueError(f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")

        if bool(self.ar_mode):
            num_obs_proprio = min(
                int(proprio.shape[1]), self._max_ar_visible_proprio_count()
            )
        else:
            num_obs_proprio = min(
                self._proprio_obs_action_steps() + 1,
                int(proprio.shape[1]),
            )
        proprio = proprio[:, :num_obs_proprio, :]

        proprio_dim_is_pad = sample.get("proprio_dim_is_pad")
        if proprio_dim_is_pad is not None:
            if proprio_dim_is_pad.ndim == 2:
                proprio_dim_is_pad = proprio_dim_is_pad.unsqueeze(1).expand(-1, num_obs_proprio, -1)
            elif proprio_dim_is_pad.ndim == 3:
                proprio_dim_is_pad = proprio_dim_is_pad[:, :num_obs_proprio, :]
            else:
                raise ValueError(
                    "`sample['proprio_dim_is_pad']` must be 2D [B, d] or "
                    f"3D [B, T, d], got shape {tuple(proprio_dim_is_pad.shape)}"
                )
            proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool)

        proprio_field_valid = sample.get("proprio_field_valid")
        if proprio_field_valid is not None:
            expected_field_valid_dim = int(
                getattr(self.proprio_encoder, "field_valid_dim", 0)
            )
            if (
                proprio_field_valid.ndim != 3
                or (
                    expected_field_valid_dim > 0
                    and proprio_field_valid.shape[-1]
                    != expected_field_valid_dim
                )
            ):
                raise ValueError(
                    "`sample['proprio_field_valid']` must be [B,T,"
                    f"{expected_field_valid_dim}], got "
                    f"{tuple(proprio_field_valid.shape)}"
                )
            proprio_field_valid = proprio_field_valid[
                :, :num_obs_proprio, :
            ].to(device=self.device, dtype=torch.bool)
        proprio_is_pad = sample.get("proprio_is_pad")
        if proprio_is_pad is not None:
            proprio_is_pad = proprio_is_pad[:, :num_obs_proprio].to(device=self.device, dtype=torch.bool)

        return self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
        )

    def _build_runtime_context(
        self,
        static_context: torch.Tensor,
        static_context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio is None:
            return static_context, static_context_mask
        return self._append_proprio_to_context(
            context=static_context,
            context_mask=static_context_mask,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
        )

    def _ar_initial_proprio_count(self, proprio_token_count: int) -> int:
        if proprio_token_count <= 0:
            return 0
        return min(
            int(proprio_token_count), self._proprio_obs_action_steps() + 1
        )

    def _proprio_token_count_from_tensor(self, proprio: Optional[torch.Tensor]) -> int:
        if getattr(self, "proprio_encoder", None) is None or proprio is None:
            return 0
        if proprio.ndim == 1:
            return 1
        if proprio.ndim == 2:
            return 1
        if proprio.ndim == 3:
            return int(proprio.shape[1])
        raise ValueError(f"`proprio` must be 1D, 2D, or 3D, got ndim={proprio.ndim}")

    def _ar_raw_steps_per_latent(self) -> int:
        if self.ar_geometry is None:
            raise RuntimeError("AR geometry is required in AR mode.")
        return int(self.ar_geometry.raw_steps_per_latent)

    def _ar_visible_proprio_count_for_block(self, block_idx: int, proprio_token_count: int) -> int:
        initial = self._ar_initial_proprio_count(proprio_token_count)
        if proprio_token_count <= 0:
            return 0
        visible = initial + max(0, int(block_idx)) * self._ar_raw_steps_per_latent()
        return min(int(proprio_token_count), int(visible))

    def _mask_future_proprio_tokens(
        self,
        context_mask: torch.Tensor,
        proprio_token_count: int,
        visible_proprio_count: int,
        query_len: Optional[int] = None,
    ) -> torch.Tensor:
        if query_len is not None:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D when query_len is provided, got {tuple(context_mask.shape)}")
            mask = context_mask.unsqueeze(1).expand(-1, int(query_len), -1).clone()
        else:
            mask = context_mask.clone()
        if proprio_token_count <= 0:
            return mask
        text_len = mask.shape[-1] - int(proprio_token_count)
        if text_len < 0:
            raise ValueError(
                f"`proprio_token_count`={proprio_token_count} exceeds context length {mask.shape[-1]}"
            )
        visible = max(0, min(int(visible_proprio_count), int(proprio_token_count)))
        mask[..., text_len + visible:] = False
        return mask

    def _build_ar_action_context_mask(
        self,
        context_mask: torch.Tensor,
        proprio_token_count: int,
        *,
        action_seq_len: int,
        num_blocks: int,
    ) -> torch.Tensor:
        action_seq_len = int(action_seq_len)
        num_blocks = int(num_blocks)
        mask = context_mask.unsqueeze(1).expand(-1, action_seq_len, -1).clone()
        if proprio_token_count <= 0 or action_seq_len == 0:
            return mask
        if num_blocks <= 0 or action_seq_len % num_blocks != 0:
            raise ValueError(f"action_seq_len={action_seq_len} must be divisible by num_blocks={num_blocks}")
        text_len = mask.shape[-1] - int(proprio_token_count)
        action_per_block = action_seq_len // num_blocks
        for block_idx in range(num_blocks):
            visible = self._ar_visible_proprio_count_for_block(block_idx, proprio_token_count)
            start = block_idx * action_per_block
            end = start + action_per_block
            mask[:, start:end, text_len + visible:] = False
        return mask

    @staticmethod
    def _build_ar_plan_action_position_ids(
        *,
        committed_blocks: int,
        prediction_horizon: int,
        execution_horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Flatten overlapping plans while retaining raw-timeline RoPE positions."""
        committed_blocks = int(committed_blocks)
        prediction_horizon = int(prediction_horizon)
        execution_horizon = int(execution_horizon)
        if committed_blocks <= 0:
            raise ValueError(
                f"committed_blocks must be positive, got {committed_blocks}"
            )
        if prediction_horizon <= 0:
            raise ValueError(
                f"prediction_horizon must be positive, got {prediction_horizon}"
            )
        if execution_horizon <= 0:
            raise ValueError(
                f"execution_horizon must be positive, got {execution_horizon}"
            )
        # Pure function of (geometry, device); cache the arange/cat result.
        # The cached tensor is only ever read (expanded/gathered downstream),
        # never mutated in place.
        key = (committed_blocks, prediction_horizon, execution_horizon, device)
        position_ids = _AR_PLAN_GEOMETRY_CACHE.get(key)
        if position_ids is None:
            within_plan = torch.arange(
                prediction_horizon,
                device=device,
                dtype=torch.long,
            )
            position_ids = torch.cat(
                [
                    within_plan + block_idx * execution_horizon
                    for block_idx in range(committed_blocks)
                ],
                dim=0,
            )
            _AR_PLAN_GEOMETRY_CACHE[key] = position_ids
        return position_ids

    @staticmethod
    def _get_ar_plan_sequence_index(
        *,
        committed_blocks: int,
        prediction_horizon: int,
        execution_horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Index map that turns the dense future timeline into the flattened
        plan layout: ``cat([tensor[:, a*E : a*E + P] for a in range(C)])`` is
        exactly ``tensor.index_select(1, index)`` (a pure copy of the same
        elements in the same order)."""
        committed_blocks = int(committed_blocks)
        prediction_horizon = int(prediction_horizon)
        execution_horizon = int(execution_horizon)
        key = ("plan_seq", committed_blocks, prediction_horizon, execution_horizon, device)
        index = _AR_PLAN_GEOMETRY_CACHE.get(key)
        if index is None:
            pieces = []
            for anchor_idx in range(committed_blocks):
                start = anchor_idx * execution_horizon
                end = start + prediction_horizon
                pieces.append(torch.arange(start, end, device=device, dtype=torch.long))
            index = torch.cat(pieces, dim=0)
            _AR_PLAN_GEOMETRY_CACHE[key] = index
        return index

    @staticmethod
    def _stitch_ar_video_plan_sequence(
        plan_latents: torch.Tensor,
        *,
        committed_blocks: int,
        plan_subblocks: int,
    ) -> torch.Tensor:
        """Commit replanned video slots with the same policy as action plans.

        Earlier plans contribute only their first execution sub-block. The
        final plan contributes its full horizon. Overlapping forecasts remain
        independent throughout denoising and are never velocity-averaged.
        """
        committed_blocks = int(committed_blocks)
        plan_subblocks = int(plan_subblocks)
        if plan_latents.ndim != 5:
            raise ValueError(
                "AR video plan latents must be [B,C,T,H,W], got "
                f"{tuple(plan_latents.shape)}."
            )
        if committed_blocks <= 0 or plan_subblocks <= 0:
            raise ValueError(
                "AR video plan geometry must be positive, got "
                f"committed_blocks={committed_blocks}, "
                f"plan_subblocks={plan_subblocks}."
            )
        expected_plan_blocks = committed_blocks * plan_subblocks
        if int(plan_latents.shape[2]) != expected_plan_blocks:
            raise ValueError(
                "AR video plan latent count mismatch: "
                f"expected={expected_plan_blocks}, "
                f"got={int(plan_latents.shape[2])}."
            )
        stitch_indices = [
            block_idx * plan_subblocks
            for block_idx in range(max(0, committed_blocks - 1))
        ]
        final_start = (committed_blocks - 1) * plan_subblocks
        stitch_indices.extend(range(final_start, final_start + plan_subblocks))
        index = torch.tensor(
            stitch_indices,
            device=plan_latents.device,
            dtype=torch.long,
        )
        return plan_latents.index_select(2, index)

    def _ar_cycle_video_plan_subblocks(
        self,
        *,
        action_horizon: int,
        predict_video: bool,
    ) -> int:
        if not predict_video:
            return 0
        if not bool(self.requires_joint_video_prediction):
            return 1
        action_horizon = int(action_horizon)
        prediction_horizon = int(self.prediction_horizon)
        execution_horizon = int(self.execution_horizon)
        if action_horizon != prediction_horizon:
            raise ValueError(
                "joint_bidirectional AR cycle inference requires the trained "
                "full action plan horizon: "
                f"action_horizon={action_horizon}, "
                f"prediction_horizon={prediction_horizon}."
            )
        if action_horizon % execution_horizon != 0:
            raise ValueError(
                f"action_horizon={action_horizon} must be divisible by "
                f"execution_horizon={execution_horizon}."
            )
        return action_horizon // execution_horizon

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    # AR cycle-level inference methods (KV cache persistent)

    @torch.no_grad()
    def prefill_cycle(
        self,
        obs_video: torch.Tensor,
        history_prefix_tokens: Optional[torch.Tensor] = None,
        history_valid_mask: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Prefill DiT KV cache at cycle start.

        Encodes initial obs via VAE, builds clean tokens (optional history + obs),
        and caches per-layer K/V for the entire cycle.

        Supports both single-branch and CFG cycle states transparently.
        """
        from einops import rearrange
        from .wan22.wan_video_dit import sinusoidal_embedding_1d, build_identity_rope_like

        self.eval()

        if bool(self.ar_history_mode):
            if history_prefix_tokens is None or history_valid_mask is None:
                raise ValueError(
                    "AR-history prefill_cycle requires both history_prefix_tokens "
                    "and history_valid_mask."
                )

        # Handle CFG: prefill cond and uncond branches using the same VAE-encoded obs_latents
        if text_cfg_scale != 1.0:
            # Mutual exclusivity: same pattern as _resolve_cfg_contexts()
            use_prompt = prompt is not None
            use_context = context is not None or context_mask is not None
            if use_prompt and use_context:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive in prefill_cycle().")
            if not use_prompt and not use_context:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided in prefill_cycle().")
            obs_video = obs_video.to(device=self.device, dtype=self.torch_dtype)
            self.vae.encode_reset()
            first_chunk = obs_video[:, :, :1]
            latent_0 = self.vae.encode_chunk(first_chunk, device=self.device)
            T_video = obs_video.shape[2]
            latent_chunks = [latent_0]
            for i in range(1, 1 + (T_video - 1) // 4):
                chunk = obs_video[:, :, 1 + 4 * (i - 1):1 + 4 * i]
                lat_i = self.vae.encode_chunk(chunk, device=self.device)
                latent_chunks.append(lat_i)
            obs_latents = torch.cat(latent_chunks, dim=2)

            cond_state = self.prefill_cycle_from_latents(
                obs_latents=obs_latents,
                history_prefix_tokens=history_prefix_tokens,
                history_valid_mask=history_valid_mask,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )

            if negative_context is not None or negative_context_mask is not None:
                if negative_context is None or negative_context_mask is None:
                    raise ValueError("`negative_context` and `negative_context_mask` must be both provided together when either is given.")
                prompt_uncond = None
                context_uncond = negative_context
                context_mask_uncond = negative_context_mask
            else:
                if self.prompt_template is None:
                    raise ValueError(
                        "CFG (`text_cfg_scale != 1.0`) requires `prompt_template` to construct the "
                        "uncond baseline in prefill_cycle(). "
                        "Set model.prompt_template to match the training-time `prompt_template.format(task=\"\")`."
                    )
                prompt_uncond = self.prompt_template.format(task="")
                context_uncond = None
                context_mask_uncond = None
            uncond_state = self.prefill_cycle_from_latents(
                obs_latents=obs_latents,
                history_prefix_tokens=history_prefix_tokens,
                history_valid_mask=history_valid_mask,
                prompt=prompt_uncond,
                context=context_uncond,
                context_mask=context_mask_uncond,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )

            return {
                "cfg_enabled": True,
                "cache_static_prompt": prompt,
                "num_latent_frames_in_cache": cond_state.get("num_latent_frames_in_cache", 0),
                "cond": cond_state,
                "uncond": uncond_state,
            }

        # Encode obs to latents
        obs_video = obs_video.to(device=self.device, dtype=self.torch_dtype)
        # Use incremental VAE encoding: reset + encode chunks
        self.vae.encode_reset()
        # First chunk: first video frame → 1 latent
        first_chunk = obs_video[:, :, :1]  # [1, 3, 1, H, W]
        latent_0 = self.vae.encode_chunk(first_chunk, device=self.device)
        # Subsequent chunks: groups of 4 video frames → 1 latent each
        T_video = obs_video.shape[2]
        latent_chunks = [latent_0]
        for i in range(1, 1 + (T_video - 1) // 4):
            chunk = obs_video[:, :, 1 + 4 * (i - 1):1 + 4 * i]
            lat_i = self.vae.encode_chunk(chunk, device=self.device)
            latent_chunks.append(lat_i)
        obs_latents = torch.cat(latent_chunks, dim=2)  # [1, C, T_lat, H_lat, W_lat]
        T_lat = obs_latents.shape[2]

        # Patchify obs latents
        clean_patch = self.video_expert.patch_embedding(obs_latents)
        clean_tokens = rearrange(clean_patch, "b c f h w -> b (f h w) c")
        _, h, w = clean_patch.shape[2:]
        tokens_per_frame = h * w

        # Prepend history prefix (skip when ar_history_mode=False)
        if self.ar_history_mode:
            far_tokens = history_prefix_tokens.to(device=self.device, dtype=self.torch_dtype)
            num_far_tokens = far_tokens.shape[1]
            clean_with_history = torch.cat([far_tokens, clean_tokens], dim=1)
        else:
            far_tokens = None
            num_far_tokens = 0
            clean_with_history = clean_tokens
        S_clean = clean_with_history.shape[1]

        # Time modulation (t=0 for clean)
        clean_t_values = torch.zeros(1, S_clean, device=self.device, dtype=self.torch_dtype)
        clean_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, clean_t_values.reshape(-1))
        clean_t = self.video_expert.time_embedding(clean_t_emb).reshape(1, S_clean, -1)
        clean_t_mod = self.video_expert.time_projection(clean_t).unflatten(2, (6, self.video_expert.hidden_dim))

        # RoPE: identity for history prefix + standard 3D for obs
        clean_near_freqs = torch.cat([
            self.video_expert.freqs[0][:T_lat].view(T_lat, 1, 1, -1).expand(T_lat, h, w, -1),
            self.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(T_lat, h, w, -1),
            self.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(T_lat, h, w, -1),
        ], dim=-1).reshape(T_lat * h * w, 1, -1).to(self.device)
        if self.ar_history_mode:
            identity_freqs = build_identity_rope_like(clean_near_freqs, n_tokens=num_far_tokens)
            clean_freqs = torch.cat([identity_freqs, clean_near_freqs], dim=0)
        else:
            clean_freqs = clean_near_freqs

        # Static cache context and runtime block context are intentionally separate.
        if prompt is not None:
            static_context, static_context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("prefill_cycle requires either prompt or both context and context_mask.")
        else:
            static_context, static_context_mask = context, context_mask
        if static_context.ndim == 2:
            static_context = static_context.unsqueeze(0)
        if static_context_mask.ndim == 1:
            static_context_mask = static_context_mask.unsqueeze(0)
        if static_context.ndim != 3 or static_context_mask.ndim != 2:
            raise ValueError(
                "AR prefill context must be [B,L,D] with mask [B,L], got "
                f"{tuple(static_context.shape)} and {tuple(static_context_mask.shape)}."
            )
        if static_context.shape[:2] != static_context_mask.shape:
            raise ValueError(
                "AR prefill context/mask shape mismatch: "
                f"{tuple(static_context.shape)} vs {tuple(static_context_mask.shape)}."
            )
        if static_context.shape[0] != clean_with_history.shape[0]:
            raise ValueError(
                "AR prefill context batch must match observation batch: "
                f"{static_context.shape[0]} vs {clean_with_history.shape[0]}."
            )
        static_context = static_context.to(device=self.device, dtype=self.torch_dtype)
        static_context_mask = static_context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            if proprio_dim_is_pad is not None:
                proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool)
        runtime_context, runtime_context_mask = self._build_runtime_context(
            static_context=static_context,
            static_context_mask=static_context_mask,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
        )
        clean_video_context_payload = {
            "context": self.video_expert.text_embedding(static_context),
            "mask": static_context_mask,
        }

        # Prefill clean KV cache
        if self.ar_history_mode:
            history_valid_mask = history_valid_mask.to(device=self.device)
        kv_cache = self.mot.prefill_clean_kv_cache(
            clean_video_tokens=clean_with_history,
            clean_video_freqs=clean_freqs,
            clean_video_t_mod=clean_t_mod,
            video_context_payload=clean_video_context_payload,
            far_token_count=num_far_tokens,
            tokens_per_frame=tokens_per_frame,
            history_valid_mask=history_valid_mask,
        )

        state = {
            "kv_cache": kv_cache,
            "clean_cache_context": static_context,
            "clean_cache_context_mask": static_context_mask,
            "clean_video_context_payload": clean_video_context_payload,
            "runtime_context": runtime_context,
            "runtime_context_mask": runtime_context_mask,
            "base_static_context": static_context,
            "base_static_context_mask": static_context_mask,
            "cache_static_prompt": prompt,
            "proprio_context_required": bool(proprio is not None and getattr(self, "proprio_encoder", None) is not None),
            "clean_latents": obs_latents,
            "T_lat": T_lat,
            "h": h,
            "w": w,
            "tokens_per_frame": tokens_per_frame,
            "num_far_tokens": num_far_tokens,
            "num_latent_frames_in_cache": T_lat,
        }
        return state

    @torch.no_grad()
    def _denoise_ar_cycle_step(
        self,
        cycle_state: dict,
        action_horizon: int,
        action_position_offset: int = 0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        predict_video: bool = False,
        tiled: bool = False,
    ) -> dict:
        """Denoise one AR step using cached clean K/V.

        By default this is the fast deployment path and only denoises action.
        When ``predict_video=True``, condition-only mode denoises the next
        video latent, while joint mode denoises the full P/E video plan. The
        prediction is not appended to the KV cache; online deployment still
        appends the real observation after executing one horizon.

        Returns:
            dict with "action": [T, D], and optionally "video"/"latents".
        """
        predict_video = bool(
            predict_video
            or self.requires_joint_video_prediction
        )
        video_plan_subblocks = self._ar_cycle_video_plan_subblocks(
            action_horizon=action_horizon,
            predict_video=predict_video,
        )
        from einops import rearrange
        from .wan22.wan_video_dit import sinusoidal_embedding_1d

        kv_cache = cycle_state["kv_cache"]
        h, w = cycle_state["h"], cycle_state["w"]
        tokens_per_frame = cycle_state["tokens_per_frame"]
        T_lat = cycle_state["num_latent_frames_in_cache"]
        clean_video_context_payload = cycle_state["clean_video_context_payload"]
        runtime_context = cycle_state["runtime_context"]
        runtime_context_mask = cycle_state["runtime_context_mask"]

        noisy_freqs_list = []
        for sub_idx in range(max(1, video_plan_subblocks)):
            t_idx = T_lat + sub_idx
            noisy_freqs_list.append(torch.cat([
                self.video_expert.freqs[0][t_idx:t_idx + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
                self.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
                self.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1),
            ], dim=-1).reshape(h * w, 1, -1).to(self.device))
        noisy_freqs = noisy_freqs_list[0]

        latents_action = self._sample_action_latents(
            action_horizon=action_horizon,
            rand_device=rand_device,
            seed=seed,
        )

        infer_timesteps_a, infer_deltas_a = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        clean_latents = None
        latents_video = None
        infer_timesteps_v = None
        infer_deltas_v = None
        if predict_video:
            clean_latents = cycle_state.get("clean_latents")
            if clean_latents is None:
                raise ValueError("AR video prediction requires cycle_state['clean_latents'].")
            clean_latents = clean_latents.to(device=self.device, dtype=self.torch_dtype)
            # AR inference/deployment has no future GT actions. Keep video
            # prediction conditioned on runtime text/proprio, but do not write
            # those dynamic conditions into clean KV.
            noisy_video_context_payload = {
                "context": self.video_expert.text_embedding(runtime_context),
                "mask": runtime_context_mask,
            }
            video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            C_lat, H_lat, W_lat = clean_latents.shape[1], clean_latents.shape[3], clean_latents.shape[4]
            latents_video = torch.randn(
                (1, C_lat, video_plan_subblocks, H_lat, W_lat),
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            infer_timesteps_v, infer_deltas_v = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
            if self.coupled_video_action_timesteps:
                infer_timesteps_a, infer_deltas_a = self._build_coupled_action_infer_schedule(
                    infer_timesteps_v,
                    action_dtype=latents_action.dtype,
                    shift_override=sigma_shift,
                )
        else:
            noisy_video_placeholder = torch.zeros(
                1, tokens_per_frame, self.video_expert.hidden_dim,
                device=self.device, dtype=self.torch_dtype,
            )
            zero_t = torch.zeros(1, device=self.device, dtype=self.torch_dtype)
            noisy_t_emb_0 = sinusoidal_embedding_1d(self.video_expert.freq_dim, zero_t)
            noisy_t_0 = self.video_expert.time_embedding(noisy_t_emb_0).unsqueeze(1).expand(-1, tokens_per_frame, -1)
            noisy_t_mod_0 = self.video_expert.time_projection(noisy_t_0).unflatten(2, (6, self.video_expert.hidden_dim))

        video_steps = (
            zip(infer_timesteps_v, infer_deltas_v)
            if predict_video
            else [(None, None)] * len(infer_timesteps_a)
        )
        # ``last_route_mask`` is the last step's binary GUI route mask
        # (used by route-aware inference postprocess); ``last_route_logit``
        # is the raw logit for diagnostics.
        last_route_mask: Optional[torch.Tensor] = None
        last_route_logit: Optional[torch.Tensor] = None
        guidance_route_mask: Optional[torch.Tensor] = None
        guidance_blocks = max(1, (int(action_horizon) + int(self.prediction_horizon) - 1) // int(self.prediction_horizon))
        for (step_t_v, step_delta_v), step_t_a, step_delta_a in zip(
            video_steps,
            infer_timesteps_a,
            infer_deltas_a,
        ):
            if predict_video:
                t_val_v = step_t_v.unsqueeze(0)
                noisy_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, t_val_v)
                noisy_t = self.video_expert.time_embedding(noisy_t_emb).unsqueeze(1).expand(-1, tokens_per_frame, -1)
                noisy_video_t_mod = self.video_expert.time_projection(noisy_t).unflatten(2, (6, self.video_expert.hidden_dim))
                noisy_video_tokens_list = []
                noisy_video_t_mod_list = []
                for sub_idx in range(video_plan_subblocks):
                    noisy_patch = self.video_expert.patch_embedding(
                        latents_video[:, :, sub_idx:sub_idx + 1]
                    )
                    noisy_video_tokens_list.append(
                        rearrange(noisy_patch, "b c f h w -> b (f h w) c")
                    )
                    noisy_video_t_mod_list.append(noisy_video_t_mod)
                noisy_video_tokens = noisy_video_tokens_list[0]
            else:
                noisy_video_tokens = noisy_video_placeholder
                noisy_video_t_mod = noisy_t_mod_0

            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=step_t_a.unsqueeze(0),
                context=runtime_context,
                context_mask=runtime_context_mask,
                position_offset=action_position_offset,
            )

            action_context_payload = {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            }
            if predict_video and video_plan_subblocks > 1:
                mot_out = self.mot.forward_ar_from_clean_cache(
                    cache=kv_cache,
                    noisy_video_tokens_list=noisy_video_tokens_list,
                    action_tokens=action_pre["tokens"],
                    noisy_video_freqs_list=noisy_freqs_list,
                    action_freqs=action_pre["freqs"],
                    noisy_video_t_mod_list=noisy_video_t_mod_list,
                    action_t_mod=action_pre["t_mod"],
                    video_context_payload=clean_video_context_payload,
                    action_context_payload=action_context_payload,
                    num_obs_latents=T_lat,
                    noisy_video_context_payloads_list=[
                        noisy_video_context_payload
                    ] * video_plan_subblocks,
                    num_action_blocks=1,
                )
            else:
                mot_out = self.mot.forward_ar_single_step(
                    cache=kv_cache,
                    noisy_video_tokens=noisy_video_tokens,
                    action_tokens=action_pre["tokens"],
                    noisy_video_freqs=noisy_freqs,
                    action_freqs=action_pre["freqs"],
                    noisy_video_t_mod=noisy_video_t_mod,
                    action_t_mod=action_pre["t_mod"],
                    video_context_payload=clean_video_context_payload,
                    action_context_payload=action_context_payload,
                    num_obs_latents=T_lat,
                    noisy_video_context_payload=noisy_video_context_payload if predict_video else None,
                )

            if predict_video:
                t_head_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, t_val_v)
                t_head = self.video_expert.time_embedding(t_head_emb).unsqueeze(1).expand(-1, tokens_per_frame, -1)
                video_outputs = (
                    mot_out["noisy_video"]
                    if isinstance(mot_out["noisy_video"], list)
                    else [mot_out["noisy_video"]]
                )
                pred_video_blocks = []
                for video_output in video_outputs:
                    pred_video_tokens = self.video_expert.head(video_output, t_head)
                    pred_video_blocks.append(rearrange(
                        pred_video_tokens,
                        "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
                        f=1,
                        h=h,
                        w=w,
                        x=self.video_expert.patch_size[0],
                        y=self.video_expert.patch_size[1],
                        z=self.video_expert.patch_size[2],
                    ))
                pred_video = torch.cat(pred_video_blocks, dim=2)
                latents_video = self.infer_video_scheduler.step(pred_video, step_delta_v, latents_video)

            pred_heads = self.action_expert.post_dit(mot_out["action"], action_pre)
            # Reusing the CFG combiner with identical branches applies no
            # guidance while keeping continuous/route outputs consistent.
            routed_heads = self.action_expert.route_cond_heads(pred_heads)
            routed_continuous = routed_heads["continuous"]
            if routed_heads.get("route_mask") is not None:
                last_route_mask = routed_heads["route_mask"]
                guidance_route_mask = last_route_mask.detach()
                last_route_logit = routed_heads.get("route_logit")
            latents_action = self._step_continuous(
                latents_action, routed_continuous, step_delta_a,
            )

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        # Route-aware inference: surface the last step's GUI route
        # mask (and raw logit for debug) so the postprocess side can
        # route-aware denormalize. Skipped when GUI head is disabled.
        if self.action_head_gui_head_enabled and last_route_mask is not None:
            result["action_route_mask"] = last_route_mask[0].detach().to(
                device="cpu", dtype=torch.bool
            )
        if last_route_logit is not None:
            result["action_route_logit"] = last_route_logit[0].detach().to(
                device="cpu", dtype=torch.float32
            )
        if predict_video:
            video_frames, pred_video_latents = self._decode_ar_prediction_frames(
                clean_latents,
                latents_video,
                tiled=tiled,
            )
            result.update({
                "video": video_frames,
                "latents": pred_video_latents.detach().to(device="cpu", dtype=torch.float32),
            })
        return result

    @torch.no_grad()
    def _denoise_ar_cycle_step_cfg(
        self,
        cycle_state_cond: dict,
        cycle_state_uncond: dict,
        text_cfg_scale: float,
        action_horizon: int,
        action_position_offset: int = 0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        predict_video: bool = False,
        tiled: bool = False,
    ) -> dict:
        """CFG denoise: runs cond/uncond branches with shared noise & timesteps.

        CFG combination:
            pred = pred_cond + (text_cfg_scale - 1) * (pred_cond - pred_uncond)

        The scheduler step is applied only once, on the combined prediction.
        """
        predict_video = bool(
            predict_video
            or self.requires_joint_video_prediction
        )
        video_plan_subblocks = self._ar_cycle_video_plan_subblocks(
            action_horizon=action_horizon,
            predict_video=predict_video,
        )
        from einops import rearrange
        from .wan22.wan_video_dit import sinusoidal_embedding_1d

        # shared noise & inference schedule
        latents_action = self._sample_action_latents(
            action_horizon=action_horizon,
            rand_device=rand_device,
            seed=seed,
        )

        infer_timesteps_a, infer_deltas_a = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )


        latents_video = None
        infer_timesteps_v = None
        infer_deltas_v = None
        if predict_video:
            video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            cl = cycle_state_cond.get("clean_latents")
            if cl is None:
                raise ValueError("AR video prediction requires cycle_state_cond['clean_latents'].")
            C_lat, H_lat, W_lat = cl.shape[1], cl.shape[3], cl.shape[4]
            latents_video = torch.randn(
                (1, C_lat, video_plan_subblocks, H_lat, W_lat),
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)

            infer_timesteps_v, infer_deltas_v = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
            if self.coupled_video_action_timesteps:
                infer_timesteps_a, infer_deltas_a = self._build_coupled_action_infer_schedule(
                    infer_timesteps_v,
                    action_dtype=latents_action.dtype,
                    shift_override=sigma_shift,
                )

        # per-step helper: single forward
        guidance_blocks = max(1, (int(action_horizon) + int(self.prediction_horizon) - 1) // int(self.prediction_horizon))
        def _single_branch(cs, latents_a, lv, step_t_a, step_t_v, predict_v, guidance_route_mask=None):
            kv_cache_c = cs["kv_cache"]
            h_c, w_c = cs["h"], cs["w"]
            tokens_per_frame_c = cs["tokens_per_frame"]
            T_lat_c = cs["num_latent_frames_in_cache"]
            clean_vcp_c = cs["clean_video_context_payload"]
            runtime_ctx_c = cs["runtime_context"]
            runtime_mask_c = cs["runtime_context_mask"]

            noisy_freqs_list_c = []
            for sub_idx in range(max(1, video_plan_subblocks)):
                t_idx = T_lat_c + sub_idx
                noisy_freqs_list_c.append(torch.cat([
                    self.video_expert.freqs[0][t_idx:t_idx + 1].view(1, 1, 1, -1).expand(1, h_c, w_c, -1),
                    self.video_expert.freqs[1][:h_c].view(1, h_c, 1, -1).expand(1, h_c, w_c, -1),
                    self.video_expert.freqs[2][:w_c].view(1, 1, w_c, -1).expand(1, h_c, w_c, -1),
                ], dim=-1).reshape(h_c * w_c, 1, -1).to(self.device))
            noisy_freqs_c = noisy_freqs_list_c[0]

            if predict_v:
                t_val_v = step_t_v.unsqueeze(0)
                noisy_t_emb_v = sinusoidal_embedding_1d(self.video_expert.freq_dim, t_val_v)
                noisy_t_v = self.video_expert.time_embedding(noisy_t_emb_v).unsqueeze(1).expand(-1, tokens_per_frame_c, -1)
                noisy_vid_t_mod = self.video_expert.time_projection(noisy_t_v).unflatten(2, (6, self.video_expert.hidden_dim))
                noisy_vid_tokens_list = []
                noisy_vid_t_mod_list = []
                for sub_idx in range(video_plan_subblocks):
                    noisy_patch = self.video_expert.patch_embedding(
                        lv[:, :, sub_idx:sub_idx + 1]
                    )
                    noisy_vid_tokens_list.append(
                        rearrange(noisy_patch, "b c f h w -> b (f h w) c")
                    )
                    noisy_vid_t_mod_list.append(noisy_vid_t_mod)
                noisy_vid_tokens = noisy_vid_tokens_list[0]
            else:
                noisy_vid_tokens = torch.zeros(
                    1, tokens_per_frame_c, self.video_expert.hidden_dim,
                    device=self.device, dtype=self.torch_dtype,
                )
                zero_t = torch.zeros(1, device=self.device, dtype=self.torch_dtype)
                noisy_t_emb_0 = sinusoidal_embedding_1d(self.video_expert.freq_dim, zero_t)
                noisy_t_0 = self.video_expert.time_embedding(noisy_t_emb_0).unsqueeze(1).expand(-1, tokens_per_frame_c, -1)
                noisy_vid_t_mod = self.video_expert.time_projection(noisy_t_0).unflatten(2, (6, self.video_expert.hidden_dim))

            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_a,
                timestep=step_t_a.unsqueeze(0),
                context=runtime_ctx_c,
                context_mask=runtime_mask_c,
                position_offset=action_position_offset,
            )

            action_context_payload = {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            }
            noisy_video_context_payload_c = (
                {
                    "context": self.video_expert.text_embedding(runtime_ctx_c),
                    "mask": runtime_mask_c,
                }
                if predict_v else None
            )
            if predict_v and video_plan_subblocks > 1:
                mot_out = self.mot.forward_ar_from_clean_cache(
                    cache=kv_cache_c,
                    noisy_video_tokens_list=noisy_vid_tokens_list,
                    action_tokens=action_pre["tokens"],
                    noisy_video_freqs_list=noisy_freqs_list_c,
                    action_freqs=action_pre["freqs"],
                    noisy_video_t_mod_list=noisy_vid_t_mod_list,
                    action_t_mod=action_pre["t_mod"],
                    video_context_payload=clean_vcp_c,
                    action_context_payload=action_context_payload,
                    num_obs_latents=T_lat_c,
                    noisy_video_context_payloads_list=[
                        noisy_video_context_payload_c
                    ] * video_plan_subblocks,
                    num_action_blocks=1,
                )
            else:
                mot_out = self.mot.forward_ar_single_step(
                    cache=kv_cache_c,
                    noisy_video_tokens=noisy_vid_tokens,
                    action_tokens=action_pre["tokens"],
                    noisy_video_freqs=noisy_freqs_c,
                    action_freqs=action_pre["freqs"],
                    noisy_video_t_mod=noisy_vid_t_mod,
                    action_t_mod=action_pre["t_mod"],
                    video_context_payload=clean_vcp_c,
                    action_context_payload=action_context_payload,
                    num_obs_latents=T_lat_c,
                    noisy_video_context_payload=noisy_video_context_payload_c,
                )

            pred_heads = self.action_expert.post_dit(mot_out["action"], action_pre)
            pred_vid = None
            if predict_v:
                t_head_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, t_val_v)
                t_head = self.video_expert.time_embedding(t_head_emb).unsqueeze(1).expand(-1, tokens_per_frame_c, -1)
                video_outputs = (
                    mot_out["noisy_video"]
                    if isinstance(mot_out["noisy_video"], list)
                    else [mot_out["noisy_video"]]
                )
                pred_vid_blocks = []
                for video_output in video_outputs:
                    pred_vid_tokens = self.video_expert.head(video_output, t_head)
                    pred_vid_blocks.append(rearrange(
                        pred_vid_tokens,
                        "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
                        f=1,
                        h=h_c,
                        w=w_c,
                        x=self.video_expert.patch_size[0],
                        y=self.video_expert.patch_size[1],
                        z=self.video_expert.patch_size[2],
                    ))
                pred_vid = torch.cat(pred_vid_blocks, dim=2)
            return pred_vid, pred_heads

        # CFG denoise loop
        video_steps = (
            zip(infer_timesteps_v, infer_deltas_v)
            if predict_video
            else [(None, None)] * len(infer_timesteps_a)
        )
        # CFG path: track the cond branch's GUI route mask + logit for
        # the route-aware postprocess side (GUI route uses cond).
        last_route_mask_c: Optional[torch.Tensor] = None
        last_route_logit_c: Optional[torch.Tensor] = None
        guidance_route_mask_c: Optional[torch.Tensor] = None
        guidance_route_mask_u: Optional[torch.Tensor] = None
        for (step_t_v, step_delta_v), step_t_a, step_delta_a in zip(
            video_steps, infer_timesteps_a, infer_deltas_a,
        ):
            pred_video_cond, pred_heads_c = _single_branch(
                cycle_state_cond, latents_action, latents_video, step_t_a, step_t_v, predict_video, guidance_route_mask_c,
            )
            pred_video_uncond, pred_heads_u = _single_branch(
                cycle_state_uncond, latents_action, latents_video, step_t_a, step_t_v, predict_video, guidance_route_mask_u,
            )

            # per-head CFG combination
            pred_heads_combined = self.action_expert.combine_cfg_heads(
                pred_heads_c, pred_heads_u, text_cfg_scale
            )
            pred_action = pred_heads_combined["continuous"]
            latents_action = self._step_continuous(latents_action, pred_action, step_delta_a)

            if predict_video:
                pred_video = pred_video_cond + (text_cfg_scale - 1.0) * (pred_video_cond - pred_video_uncond)
                latents_video = self.infer_video_scheduler.step(pred_video, step_delta_v, latents_video)

            # Capture the routed branch's mask/logit for route-aware postprocess.
            if pred_heads_combined.get("route_mask") is not None:
                last_route_mask_c = pred_heads_combined["route_mask"]
                guidance_route_mask_c = last_route_mask_c.detach()
                guidance_route_mask_u = last_route_mask_c.detach()
                last_route_logit_c = pred_heads_combined.get("route_logit")

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if self.action_head_gui_head_enabled and last_route_mask_c is not None:
            result["action_route_mask"] = last_route_mask_c[0].detach().to(
                device="cpu", dtype=torch.bool
            )
        if last_route_logit_c is not None:
            result["action_route_logit"] = last_route_logit_c[0].detach().to(
                device="cpu", dtype=torch.float32
            )
        if predict_video:
            clean_cond = cycle_state_cond.get("clean_latents")
            if clean_cond is not None:
                video_frames, pred_video_latents = self._decode_ar_prediction_frames(
                    clean_cond,
                    latents_video,
                    tiled=tiled,
                )
                result.update({
                    "video": video_frames,
                    "latents": pred_video_latents.detach().to(device="cpu", dtype=torch.float32),
                })
        return result

    @torch.no_grad()
    def prefill_cycle_from_latents(
        self,
        obs_latents: torch.Tensor,
        history_prefix_tokens: Optional[torch.Tensor] = None,
        history_valid_mask: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Prefill AR clean K/V from already-observed clean latents.

        Supports both single-branch and CFG cycle states transparently.
        """
        from einops import rearrange
        from .wan22.wan_video_dit import sinusoidal_embedding_1d, build_identity_rope_like

        self.eval()

        if bool(self.ar_history_mode):
            if history_prefix_tokens is None or history_valid_mask is None:
                raise ValueError(
                    "AR-history prefill_cycle_from_latents requires both "
                    "history_prefix_tokens and history_valid_mask."
                )

        # Handle CFG: prefill cond and uncond branches using the same clean obs_latents
        if text_cfg_scale != 1.0:
            # Mutual exclusivity: same pattern as _resolve_cfg_contexts()
            use_prompt = prompt is not None
            use_context = context is not None or context_mask is not None
            if use_prompt and use_context:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive in prefill_cycle_from_latents().")
            if not use_prompt and not use_context:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided in prefill_cycle_from_latents().")
            cond_state = self.prefill_cycle_from_latents(
                obs_latents=obs_latents,
                history_prefix_tokens=history_prefix_tokens,
                history_valid_mask=history_valid_mask,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )

            if negative_context is not None or negative_context_mask is not None:
                if negative_context is None or negative_context_mask is None:
                    raise ValueError("`negative_context` and `negative_context_mask` must be both provided together when either is given.")
                prompt_uncond = None
                context_uncond = negative_context
                context_mask_uncond = negative_context_mask
            else:
                if self.prompt_template is None:
                    raise ValueError(
                        "CFG (`text_cfg_scale != 1.0`) requires `prompt_template` to construct the "
                        "uncond baseline in prefill_cycle_from_latents(). "
                        "Set model.prompt_template to match the training-time `prompt_template.format(task=\"\")`."
                    )
                prompt_uncond = self.prompt_template.format(task="")
                context_uncond = None
                context_mask_uncond = None
            uncond_state = self.prefill_cycle_from_latents(
                obs_latents=obs_latents,
                history_prefix_tokens=history_prefix_tokens,
                history_valid_mask=history_valid_mask,
                prompt=prompt_uncond,
                context=context_uncond,
                context_mask=context_mask_uncond,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )

            return {
                "cfg_enabled": True,
                "cache_static_prompt": prompt,
                "num_latent_frames_in_cache": cond_state.get("num_latent_frames_in_cache", 0),
                "cond": cond_state,
                "uncond": uncond_state,
            }

        if obs_latents.ndim == 4:
            obs_latents = obs_latents.unsqueeze(0)
        if obs_latents.ndim != 5 or obs_latents.shape[0] != 1:
            raise ValueError(f"`obs_latents` must be [1,C,T,H,W] or [C,T,H,W], got {tuple(obs_latents.shape)}")
        obs_latents = obs_latents.to(device=self.device, dtype=self.torch_dtype)

        clean_patch = self.video_expert.patch_embedding(obs_latents)
        clean_tokens = rearrange(clean_patch, "b c f h w -> b (f h w) c")
        _, h, w = clean_patch.shape[2:]
        tokens_per_frame = h * w
        T_lat = obs_latents.shape[2]

        if self.ar_history_mode:
            far_tokens = history_prefix_tokens.to(device=self.device, dtype=self.torch_dtype)
            num_far_tokens = far_tokens.shape[1]
            clean_with_history = torch.cat([far_tokens, clean_tokens], dim=1)
        else:
            far_tokens = None
            num_far_tokens = 0
            clean_with_history = clean_tokens
        S_clean = clean_with_history.shape[1]

        clean_t_values = torch.zeros(1, S_clean, device=self.device, dtype=self.torch_dtype)
        clean_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, clean_t_values.reshape(-1))
        clean_t = self.video_expert.time_embedding(clean_t_emb).reshape(1, S_clean, -1)
        clean_t_mod = self.video_expert.time_projection(clean_t).unflatten(2, (6, self.video_expert.hidden_dim))

        clean_near_freqs = torch.cat([
            self.video_expert.freqs[0][:T_lat].view(T_lat, 1, 1, -1).expand(T_lat, h, w, -1),
            self.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(T_lat, h, w, -1),
            self.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(T_lat, h, w, -1),
        ], dim=-1).reshape(T_lat * h * w, 1, -1).to(self.device)
        if self.ar_history_mode:
            identity_freqs = build_identity_rope_like(clean_near_freqs, n_tokens=num_far_tokens)
            clean_freqs = torch.cat([identity_freqs, clean_near_freqs], dim=0)
        else:
            clean_freqs = clean_near_freqs

        if prompt is not None:
            static_context, static_context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("prefill_cycle_from_latents requires either prompt or both context and context_mask.")
        else:
            static_context, static_context_mask = context, context_mask
        if static_context.ndim == 2:
            static_context = static_context.unsqueeze(0)
        if static_context_mask.ndim == 1:
            static_context_mask = static_context_mask.unsqueeze(0)
        if static_context.ndim != 3 or static_context_mask.ndim != 2:
            raise ValueError(
                "AR prefill context must be [B,L,D] with mask [B,L], got "
                f"{tuple(static_context.shape)} and {tuple(static_context_mask.shape)}."
            )
        if static_context.shape[:2] != static_context_mask.shape:
            raise ValueError(
                "AR prefill context/mask shape mismatch: "
                f"{tuple(static_context.shape)} vs {tuple(static_context_mask.shape)}."
            )
        if static_context.shape[0] != clean_with_history.shape[0]:
            raise ValueError(
                "AR prefill context batch must match observation batch: "
                f"{static_context.shape[0]} vs {clean_with_history.shape[0]}."
            )
        static_context = static_context.to(device=self.device, dtype=self.torch_dtype)
        static_context_mask = static_context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            if proprio_dim_is_pad is not None:
                proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool)
        runtime_context, runtime_context_mask = self._build_runtime_context(
            static_context=static_context,
            static_context_mask=static_context_mask,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
        )
        clean_video_context_payload = {
            "context": self.video_expert.text_embedding(static_context),
            "mask": static_context_mask,
        }

        if self.ar_history_mode:
            history_valid_mask = history_valid_mask.to(device=self.device)
        kv_cache = self.mot.prefill_clean_kv_cache(
            clean_video_tokens=clean_with_history,
            clean_video_freqs=clean_freqs,
            clean_video_t_mod=clean_t_mod,
            video_context_payload=clean_video_context_payload,
            far_token_count=num_far_tokens,
            tokens_per_frame=tokens_per_frame,
            history_valid_mask=history_valid_mask,
        )

        state = {
            "kv_cache": kv_cache,
            "clean_cache_context": static_context,
            "clean_cache_context_mask": static_context_mask,
            "clean_video_context_payload": clean_video_context_payload,
            "runtime_context": runtime_context,
            "runtime_context_mask": runtime_context_mask,
            "base_static_context": static_context,
            "base_static_context_mask": static_context_mask,
            "cache_static_prompt": prompt,
            "proprio_context_required": bool(proprio is not None and getattr(self, "proprio_encoder", None) is not None),
            "clean_latents": obs_latents,
            "T_lat": T_lat,
            "h": h,
            "w": w,
            "tokens_per_frame": tokens_per_frame,
            "num_far_tokens": num_far_tokens,
            "num_latent_frames_in_cache": T_lat,
        }
        return state

    @torch.no_grad()
    def encode_and_append_real_obs(
        self,
        cycle_state: dict,
        new_video_frames: torch.Tensor,
        new_proprio: Optional[torch.Tensor] = None,
        new_proprio_dim_is_pad: Optional[torch.Tensor] = None,
        new_proprio_field_valid: Optional[torch.Tensor] = None,
        new_proprio_is_pad: Optional[torch.Tensor] = None,
    ) -> dict:
        """Encode new real observation frames and append to KV cache.

        Called AFTER executing the action chunk and collecting new real frames.
        The new frames are incrementally VAE encoded (using existing feat_cache)
        then patchified and appended to the DiT KV cache.

        Supports both single-branch and CFG cycle states transparently.
        """
        new_video_frames = new_video_frames.to(device=self.device, dtype=self.torch_dtype)

        # Incremental VAE encode: use existing feat_cache (NOT reset)
        new_latent = self.vae.encode_chunk(new_video_frames, device=self.device)
        # new_latent: [1, C, 1, H_lat, W_lat]

        if cycle_state.get("cfg_enabled", False):
            # Text CFG: append same new_latent to cond and uncond KV caches
            cycle_state["cond"] = self.append_real_latent_to_kv_cache(
                cycle_state=cycle_state["cond"],
                new_latent=new_latent,
                proprio=new_proprio,
                proprio_dim_is_pad=new_proprio_dim_is_pad,
                proprio_field_valid=new_proprio_field_valid,
                proprio_is_pad=new_proprio_is_pad,
            )
            cycle_state["uncond"] = self.append_real_latent_to_kv_cache(
                cycle_state=cycle_state["uncond"],
                new_latent=new_latent,
                proprio=new_proprio,
                proprio_dim_is_pad=new_proprio_dim_is_pad,
                proprio_field_valid=new_proprio_field_valid,
                proprio_is_pad=new_proprio_is_pad,
            )
            # Sync top-level counter so policy doesn't need to know about cond/uncond layout
            cycle_state["num_latent_frames_in_cache"] = cycle_state["cond"]["num_latent_frames_in_cache"]
        else:
            # Single branch: check if state is wrapped or is the dict itself
            if "state" in cycle_state:
                cycle_state["state"] = self.append_real_latent_to_kv_cache(
                    cycle_state=cycle_state["state"],
                    new_latent=new_latent,
                    proprio=new_proprio,
                    proprio_dim_is_pad=new_proprio_dim_is_pad,
                    proprio_field_valid=new_proprio_field_valid,
                    proprio_is_pad=new_proprio_is_pad,
                )
            else:
                cycle_state = self.append_real_latent_to_kv_cache(
                    cycle_state=cycle_state,
                    new_latent=new_latent,
                    proprio=new_proprio,
                    proprio_dim_is_pad=new_proprio_dim_is_pad,
                    proprio_field_valid=new_proprio_field_valid,
                    proprio_is_pad=new_proprio_is_pad,
                )

        return cycle_state

    @torch.no_grad()
    def append_real_latent_to_kv_cache(
        self,
        cycle_state: dict,
        new_latent: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
    ) -> dict:
        """Append one clean GT latent frame to the AR clean K/V cache."""
        from einops import rearrange
        from .wan22.wan_video_dit import sinusoidal_embedding_1d

        if new_latent.ndim == 4:
            new_latent = new_latent.unsqueeze(0)
        if new_latent.ndim != 5 or new_latent.shape[0] != 1 or new_latent.shape[2] != 1:
            raise ValueError(
                "`new_latent` must have shape [1,C,1,H,W] or [C,1,H,W], "
                f"got {tuple(new_latent.shape)}"
            )
        new_latent = new_latent.to(device=self.device, dtype=self.torch_dtype)

        if bool(cycle_state.get("proprio_context_required", False)) and proprio is None:
            raise ValueError(
                "AR cycle was initialized with proprio context, so append_real_latent_to_kv_cache "
                "requires updated `proprio` to keep online inference consistent with training."
            )

        if proprio is not None:
            if "base_static_context" not in cycle_state or "base_static_context_mask" not in cycle_state:
                raise ValueError(
                    "Cannot update AR cycle runtime context: "
                    "cycle_state lacks base_static_context/base_static_context_mask."
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            if proprio_dim_is_pad is not None:
                proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool)
            runtime_context, runtime_context_mask = self._build_runtime_context(
                static_context=cycle_state["base_static_context"],
                static_context_mask=cycle_state["base_static_context_mask"],
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )
            cycle_state["runtime_context"] = runtime_context
            cycle_state["runtime_context_mask"] = runtime_context_mask

        new_patch = self.video_expert.patch_embedding(new_latent)
        new_tokens = rearrange(new_patch, "b c f h w -> b (f h w) c")

        h, w = cycle_state["h"], cycle_state["w"]
        tokens_per_frame = cycle_state["tokens_per_frame"]
        T_lat = cycle_state["num_latent_frames_in_cache"]

        new_freqs = torch.cat([
            self.video_expert.freqs[0][T_lat:T_lat + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
            self.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
            self.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1),
        ], dim=-1).reshape(h * w, 1, -1).to(self.device)

        new_t_values = torch.zeros(1, tokens_per_frame, device=self.device, dtype=self.torch_dtype)
        new_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, new_t_values.reshape(-1))
        new_t = self.video_expert.time_embedding(new_t_emb).reshape(1, tokens_per_frame, -1)
        new_t_mod = self.video_expert.time_projection(new_t).unflatten(2, (6, self.video_expert.hidden_dim))

        kv_cache = self.mot.append_to_clean_kv_cache(
            cache=cycle_state["kv_cache"],
            new_tokens=new_tokens,
            new_freqs=new_freqs,
            new_t_mod=new_t_mod,
            video_context_payload=cycle_state["clean_video_context_payload"],
        )
        cycle_state["kv_cache"] = kv_cache
        cycle_state["num_latent_frames_in_cache"] = T_lat + 1
        if "clean_latents" in cycle_state:
            cycle_state["clean_latents"] = torch.cat([cycle_state["clean_latents"], new_latent], dim=2)
        return cycle_state

    @torch.no_grad()
    def _infer_ar_eval_clip_joint(
        self,
        input_latents: Optional[torch.Tensor],
        history_prefix_tokens: Optional[torch.Tensor],
        history_valid_mask: Optional[torch.Tensor],
        action_horizon: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """AR joint sampler for offline validation clips.

        The validation batch already contains the clean future latents, so this
        path can evaluate the whole prediction window with the same visibility
        mask used during training.
        """
        from einops import rearrange
        from .wan22.wan_video_dit import sinusoidal_embedding_1d, build_identity_rope_like

        if not self.ar_mode:
            raise ValueError("AR eval clip joint inference requires ar_mode=True.")
        self.eval()

        if input_latents is None:
            raise ValueError("AR eval clip `infer_joint` requires `input_latents`.")
        if self.ar_history_mode:
            if history_prefix_tokens is None:
                raise ValueError(
                    "history_prefix_tokens must be provided when ar_history_mode=True"
                )
            if history_valid_mask is None:
                raise ValueError(
                    "history_valid_mask must be provided when ar_history_mode=True"
                )

        if input_latents.ndim == 4:
            input_latents = input_latents.unsqueeze(0)
        if input_latents.ndim != 5 or input_latents.shape[0] != 1:
            raise ValueError(f"`input_latents` must be [1,C,T,H,W] or [C,T,H,W], got {tuple(input_latents.shape)}")
        input_latents = input_latents.to(device=self.device, dtype=self.torch_dtype)

        num_obs_latents = int(self.video_expert.num_obs_latents)
        if self.ar_geometry is None:
            raise RuntimeError("AR eval clip requires ar_geometry.")
        if input_latents.shape[2] <= num_obs_latents:
            raise ValueError(
                f"`input_latents` must include future latents beyond obs ({num_obs_latents}), "
                f"got T={input_latents.shape[2]}"
            )
        available_future_latents = input_latents.shape[2] - num_obs_latents
        committed_blocks = int(self.ar_geometry.pred_latents)
        execution_horizon = int(self.execution_horizon)
        prediction_horizon = int(self.prediction_horizon)
        if prediction_horizon % execution_horizon != 0:
            raise ValueError(
                f"prediction_horizon={prediction_horizon} must be divisible by "
                f"execution_horizon={execution_horizon}."
            )
        plan_subblocks = prediction_horizon // execution_horizon
        required_future_latents = committed_blocks + plan_subblocks - 1
        if available_future_latents != required_future_latents:
            raise ValueError(
                "AR eval clip requires exactly the committed plus lookahead future latents: "
                f"available_future_latents={available_future_latents}, "
                f"required={required_future_latents}."
            )
        dense_action_horizon = (committed_blocks - 1) * execution_horizon + prediction_horizon
        if action_horizon != dense_action_horizon:
            raise ValueError(
                "AR eval clip action_horizon must be the stitched future horizon: "
                f"expected={dense_action_horizon}, got {action_horizon}."
            )
        plan_action_horizon = committed_blocks * prediction_horizon
        action_per_block = prediction_horizon
        plan_video_blocks = committed_blocks * plan_subblocks
        action_position_ids = self._build_ar_plan_action_position_ids(
            committed_blocks=committed_blocks,
            prediction_horizon=prediction_horizon,
            execution_horizon=execution_horizon,
            device=self.action_expert.freqs.device,
        )
        if action_position_ids.numel() != plan_action_horizon:
            raise RuntimeError(
                "AR eval action RoPE geometry mismatch: "
                f"position_ids={action_position_ids.numel()} vs "
                f"plan_action_horizon={plan_action_horizon}"
            )
        obs_latents = input_latents[:, :, :num_obs_latents]
        clean_latents = input_latents

        use_cfg = (text_cfg_scale != 1.0)
        if proprio is not None:
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            if proprio_dim_is_pad is not None:
                proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool)
        if use_cfg:
            static_context_cond, static_context_mask_cond, static_context_uncond, static_context_mask_uncond = self._resolve_cfg_contexts(
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                negative_context=negative_context,
                negative_context_mask=negative_context_mask,
                # Proprio is appended exactly once below so CFG and non-CFG
                # share the same per-block visibility mask.
                proprio=None,
            )
        else:
            if prompt is not None:
                static_context_cond, static_context_mask_cond = self.encode_prompt(prompt)
            else:
                if context is None or context_mask is None:
                    raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
                if context.ndim == 2:
                    context = context.unsqueeze(0)
                if context_mask.ndim == 1:
                    context_mask = context_mask.unsqueeze(0)
                static_context_cond = context.to(device=self.device, dtype=self.torch_dtype)
                static_context_mask_cond = context_mask.to(device=self.device, dtype=torch.bool)
        runtime_context_cond, runtime_context_mask_cond = self._build_runtime_context(
            static_context=static_context_cond,
            static_context_mask=static_context_mask_cond,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
        )
        if use_cfg:
            runtime_context_uncond, runtime_context_mask_uncond = self._build_runtime_context(
                static_context=static_context_uncond,
                static_context_mask=static_context_mask_uncond,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )
        proprio_token_count = self._proprio_token_count_from_tensor(proprio)

        clean_patch = self.video_expert.patch_embedding(clean_latents)
        clean_tokens = rearrange(clean_patch, "b c f h w -> b (f h w) c")
        _, h, w = clean_patch.shape[2:]
        tokens_per_frame = h * w
        num_clean = clean_latents.shape[2]

        if self.ar_history_mode:
            far_tokens = history_prefix_tokens.to(device=self.device, dtype=self.torch_dtype)
            num_far_tokens = far_tokens.shape[1]
            history_valid_mask = history_valid_mask.to(device=self.device)
            clean_with_history = torch.cat([far_tokens, clean_tokens], dim=1)
        else:
            far_tokens = None
            num_far_tokens = 0
            clean_with_history = clean_tokens

        S_clean = clean_with_history.shape[1]
        clean_t_values = torch.zeros(1, S_clean, device=self.device, dtype=self.torch_dtype)
        clean_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, clean_t_values.reshape(-1))
        clean_t = self.video_expert.time_embedding(clean_t_emb).reshape(1, S_clean, -1)
        clean_t_mod = self.video_expert.time_projection(clean_t).unflatten(2, (6, self.video_expert.hidden_dim))

        clean_near_freqs = torch.cat([
            self.video_expert.freqs[0][:num_clean].view(num_clean, 1, 1, -1).expand(num_clean, h, w, -1),
            self.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(num_clean, h, w, -1),
            self.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(num_clean, h, w, -1),
        ], dim=-1).reshape(num_clean * h * w, 1, -1).to(self.device)
        if self.ar_history_mode:
            prefix_freqs = build_identity_rope_like(clean_near_freqs, n_tokens=num_far_tokens)
            clean_freqs = torch.cat([prefix_freqs, clean_near_freqs], dim=0)
        else:
            clean_freqs = clean_near_freqs

        video_context_cond = self.video_expert.text_embedding(static_context_cond)
        runtime_video_context_cond = self.video_expert.text_embedding(runtime_context_cond)
        video_context_payload_cond = {
            "context": video_context_cond,
            "mask": static_context_mask_cond,
        }
        action_context_mask_cond = self._build_ar_action_context_mask(
            context_mask=runtime_context_mask_cond,
            proprio_token_count=proprio_token_count,
            action_seq_len=plan_action_horizon,
            num_blocks=committed_blocks,
        )
        use_action_conditioned_video = False  # AR eval clip does not use action-conditioned video

        def _build_noisy_video_context_payloads(runtime_context, runtime_context_mask, runtime_video_context):
            payloads = []
            for plan_idx in range(plan_video_blocks):
                anchor_idx = plan_idx // plan_subblocks
                visible_proprio = self._ar_visible_proprio_count_for_block(anchor_idx, proprio_token_count)
                if use_action_conditioned_video:
                    payloads.append(
                        self.video_expert.build_video_context_payload(
                            context=runtime_context,
                            context_mask=self._mask_future_proprio_tokens(
                                context_mask=runtime_context_mask,
                                proprio_token_count=proprio_token_count,
                                visible_proprio_count=visible_proprio,
                            ),
                            query_latent_frames=1,
                            tokens_per_frame=tokens_per_frame,
                            action=action[:, anchor_idx * action_per_block:(anchor_idx + 1) * action_per_block],
                            action_condition_start_latent_idx=0,
                        )
                    )
                else:
                    payloads.append(
                        {
                            "context": runtime_video_context,
                            "mask": self._mask_future_proprio_tokens(
                                context_mask=runtime_context_mask,
                                proprio_token_count=proprio_token_count,
                                visible_proprio_count=visible_proprio,
                                query_len=tokens_per_frame,
                            ),
                        }
                    )
            return payloads

        noisy_video_context_payloads_list_cond = _build_noisy_video_context_payloads(
            runtime_context_cond,
            runtime_context_mask_cond,
            runtime_video_context_cond,
        )

        video_context_payload_uncond = None
        noisy_video_context_payloads_list_uncond = None
        if use_cfg:
            video_context_uncond = self.video_expert.text_embedding(static_context_uncond)
            runtime_video_context_uncond = self.video_expert.text_embedding(runtime_context_uncond)
            video_context_payload_uncond = {
                "context": video_context_uncond,
                "mask": static_context_mask_uncond,
            }
            action_context_mask_uncond = self._build_ar_action_context_mask(
                context_mask=runtime_context_mask_uncond,
                proprio_token_count=proprio_token_count,
                action_seq_len=plan_action_horizon,
                num_blocks=committed_blocks,
            )
            noisy_video_context_payloads_list_uncond = _build_noisy_video_context_payloads(
                runtime_context_uncond,
                runtime_context_mask_uncond,
                runtime_video_context_uncond,
            )

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        C_lat, H_lat, W_lat = input_latents.shape[1], input_latents.shape[3], input_latents.shape[4]
        # Every overlapping slot is a distinct replan forecast. Keep all
        # C * (P/E) states independent through the complete denoising loop,
        # then commit them with the same execution-window policy as actions.
        latents_video_plan = torch.randn(
            (1, C_lat, plan_video_blocks, H_lat, W_lat),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._sample_action_latents(
            action_horizon=plan_action_horizon,
            rand_device=rand_device,
            seed=seed,
        )

        clean_cache_cond = self.mot.prefill_clean_kv_cache(
            clean_video_tokens=clean_with_history,
            clean_video_freqs=clean_freqs,
            clean_video_t_mod=clean_t_mod,
            video_context_payload=video_context_payload_cond,
            far_token_count=num_far_tokens,
            tokens_per_frame=tokens_per_frame,
            history_valid_mask=history_valid_mask,
        )
        clean_cache_uncond = None
        if use_cfg:
            clean_cache_uncond = self.mot.prefill_clean_kv_cache(
                clean_video_tokens=clean_with_history,
                clean_video_freqs=clean_freqs,
                clean_video_t_mod=clean_t_mod,
                video_context_payload=video_context_payload_uncond,
                far_token_count=num_far_tokens,
                tokens_per_frame=tokens_per_frame,
                history_valid_mask=history_valid_mask,
            )

        infer_timesteps_v, infer_deltas_v = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video_plan.dtype,
            shift_override=sigma_shift,
        )
        if self.coupled_video_action_timesteps:
            infer_timesteps_a, infer_deltas_a = self._build_coupled_action_infer_schedule(
                infer_timesteps_v,
                action_dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        else:
            infer_timesteps_a, infer_deltas_a = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )

        noisy_freqs_list = []
        for plan_idx in range(plan_video_blocks):
            anchor_idx = plan_idx // plan_subblocks
            sub_idx = plan_idx % plan_subblocks
            t_idx = num_obs_latents + anchor_idx + sub_idx
            block_freqs = torch.cat([
                self.video_expert.freqs[0][t_idx:t_idx + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
                self.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
                self.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1),
            ], dim=-1).reshape(h * w, 1, -1).to(self.device)
            noisy_freqs_list.append(block_freqs)

        def _single_branch_joint(clean_cache, ctx, action_ctx_mask, v_payload, noisy_v_payloads, lat_v, lat_a, step_t_a_val, step_t_v_val, guidance_route_mask=None):
            noisy_tokens_list = []
            noisy_t_mod_list = []
            t_val_v = step_t_v_val.unsqueeze(0)
            for plan_idx in range(plan_video_blocks):
                noisy_patch = self.video_expert.patch_embedding(
                    lat_v[:, :, plan_idx:plan_idx + 1]
                )
                noisy_tok = rearrange(noisy_patch, "b c f h w -> b (f h w) c")
                noisy_tokens_list.append(noisy_tok)

                noisy_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, t_val_v)
                noisy_t = self.video_expert.time_embedding(noisy_t_emb).unsqueeze(1).expand(-1, tokens_per_frame, -1)
                noisy_t_mod = self.video_expert.time_projection(noisy_t).unflatten(2, (6, self.video_expert.hidden_dim))
                noisy_t_mod_list.append(noisy_t_mod)

            action_pre = self.action_expert.pre_dit(
                action_tokens=lat_a,
                timestep=step_t_a_val.unsqueeze(0),
                context=ctx,
                context_mask=action_ctx_mask,
                position_ids=action_position_ids,
            )

            action_context_payload = {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            }
            mot_out = self.mot.forward_ar_from_clean_cache(
                cache=clean_cache,
                noisy_video_tokens_list=noisy_tokens_list,
                action_tokens=action_pre["tokens"],
                noisy_video_freqs_list=noisy_freqs_list,
                action_freqs=action_pre["freqs"],
                noisy_video_t_mod_list=noisy_t_mod_list,
                action_t_mod=action_pre["t_mod"],
                video_context_payload=v_payload,
                action_context_payload=action_context_payload,
                num_obs_latents=num_obs_latents,
                noisy_video_context_payloads_list=noisy_v_payloads,
                num_action_blocks=committed_blocks,
            )

            t_head_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, t_val_v)
            t_head = self.video_expert.time_embedding(t_head_emb).unsqueeze(1).expand(-1, tokens_per_frame, -1)
            pred_video_plan = []
            for plan_idx in range(plan_video_blocks):
                pred_video_tokens = self.video_expert.head(mot_out["noisy_video"][plan_idx], t_head)
                pred_video = rearrange(
                    pred_video_tokens,
                    "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
                    f=1,
                    h=h,
                    w=w,
                    x=self.video_expert.patch_size[0],
                    y=self.video_expert.patch_size[1],
                    z=self.video_expert.patch_size[2],
                )
                pred_video_plan.append(pred_video)
            pred_video_all = torch.cat(pred_video_plan, dim=2)
            pred_heads = self.action_expert.post_dit(mot_out["action"], action_pre)
            # Return the full head dict so the caller can do CFG per-head
            # and capture the last step's discrete / route logits for
            # final assembly.
            return pred_video_all, pred_heads

        # Track last step's route logit for route-aware postprocess.
        last_route_mask: Optional[torch.Tensor] = None
        last_route_logit: Optional[torch.Tensor] = None
        guidance_route_mask_cond: Optional[torch.Tensor] = None
        guidance_route_mask_uncond: Optional[torch.Tensor] = None
        for step_t_v, step_delta_v, step_t_a, step_delta_a in zip(
            infer_timesteps_v,
            infer_deltas_v,
            infer_timesteps_a,
            infer_deltas_a,
        ):
            if use_cfg:
                pred_video_cond, pred_heads_cond = _single_branch_joint(
                    clean_cache=clean_cache_cond,
                    ctx=runtime_context_cond,
                    action_ctx_mask=action_context_mask_cond,
                    v_payload=video_context_payload_cond,
                    noisy_v_payloads=noisy_video_context_payloads_list_cond,
                    lat_v=latents_video_plan,
                    lat_a=latents_action,
                    step_t_a_val=step_t_a,
                    step_t_v_val=step_t_v,
                    guidance_route_mask=guidance_route_mask_cond,
                )
                pred_video_uncond, pred_heads_uncond = _single_branch_joint(
                    clean_cache=clean_cache_uncond,
                    ctx=runtime_context_uncond,
                    action_ctx_mask=action_context_mask_uncond,
                    v_payload=video_context_payload_uncond,
                    noisy_v_payloads=noisy_video_context_payloads_list_uncond,
                    lat_v=latents_video_plan,
                    lat_a=latents_action,
                    step_t_a_val=step_t_a,
                    step_t_v_val=step_t_v,
                    guidance_route_mask=guidance_route_mask_uncond,
                )
                pred_video_all = pred_video_cond + (text_cfg_scale - 1.0) * (pred_video_cond - pred_video_uncond)
                # Per-head CFG combine: continuous + discrete (GUI
                # route uses cond branch only).
                routed_heads = self.action_expert.combine_cfg_heads(
                    pred_heads_cond, pred_heads_uncond, text_cfg_scale
                )
                pred_action = routed_heads["continuous"]
            else:
                pred_video_all, pred_heads_cond = _single_branch_joint(
                    clean_cache=clean_cache_cond,
                    ctx=runtime_context_cond,
                    action_ctx_mask=action_context_mask_cond,
                    v_payload=video_context_payload_cond,
                    noisy_v_payloads=noisy_video_context_payloads_list_cond,
                    lat_v=latents_video_plan,
                    lat_a=latents_action,
                    step_t_a_val=step_t_a,
                    step_t_v_val=step_t_v,
                    guidance_route_mask=guidance_route_mask_cond,
                )
                routed_heads = self.action_expert.route_cond_heads(pred_heads_cond)
                pred_action = routed_heads["continuous"]

            latents_video_plan = self.infer_video_scheduler.step(
                pred_video_all,
                step_delta_v,
                latents_video_plan,
            )
            latents_action = self._step_continuous(latents_action, pred_action, step_delta_a)

            if routed_heads.get("route_mask") is not None:
                last_route_mask = routed_heads["route_mask"]
                guidance_route_mask_cond = last_route_mask.detach()
                guidance_route_mask_uncond = last_route_mask.detach()
                last_route_logit = routed_heads.get("route_logit")

        def _stitch_plan_sequence(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            squeeze_feature = False
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
                squeeze_feature = True
            if tensor.shape[1] != plan_action_horizon:
                raise ValueError(
                    "AR eval clip plan sequence length mismatch: "
                    f"expected={plan_action_horizon}, got={tensor.shape[1]}"
                )
            blocks = tensor.reshape(tensor.shape[0], committed_blocks, prediction_horizon, *tensor.shape[2:])
            pieces = [
                blocks[:, i, :execution_horizon]
                for i in range(max(0, committed_blocks - 1))
            ]
            pieces.append(blocks[:, committed_blocks - 1, :prediction_horizon])
            stitched = torch.cat(pieces, dim=1)
            if squeeze_feature:
                stitched = stitched.squeeze(-1)
            return stitched

        latents_action = _stitch_plan_sequence(latents_action)
        last_route_mask = _stitch_plan_sequence(last_route_mask)
        last_route_logit = _stitch_plan_sequence(last_route_logit)

        latents_video = self._stitch_ar_video_plan_sequence(
            latents_video_plan,
            committed_blocks=committed_blocks,
            plan_subblocks=plan_subblocks,
        )
        if int(latents_video.shape[2]) != required_future_latents:
            raise RuntimeError(
                "AR stitched video horizon mismatch: "
                f"expected={required_future_latents}, "
                f"got={int(latents_video.shape[2])}."
            )
        pred_video_latents = torch.cat([obs_latents, latents_video], dim=2)
        result = {
            "video": self._decode_ar_teacher_forced_plan_frames(
                clean_latents,
                latents_video_plan,
                num_obs_latents=num_obs_latents,
                committed_blocks=committed_blocks,
                plan_subblocks=plan_subblocks,
                tiled=tiled,
            ),
            "latents": pred_video_latents.detach().to(device="cpu", dtype=torch.float32),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if self.action_head_gui_head_enabled and last_route_mask is not None:
            result["action_route_mask"] = last_route_mask[0].detach().to(
                device="cpu", dtype=torch.bool
            )
        if last_route_logit is not None:
            result["action_route_logit"] = last_route_logit[0].detach().to(
                device="cpu", dtype=torch.float32
            )
        return result

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        vae_model = getattr(self.vae, "model", None)
        enc_feat_map = None
        enc_conv_idx = None
        if vae_model is not None:
            enc_feat_map = getattr(vae_model, "_enc_feat_map", None)
            enc_conv_idx = getattr(vae_model, "_enc_conv_idx", None)
        try:
            video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        finally:
            if vae_model is not None:
                if enc_feat_map is not None:
                    vae_model._enc_feat_map = enc_feat_map
                if enc_conv_idx is not None:
                    vae_model._enc_conv_idx = enc_conv_idx
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _decode_ar_teacher_forced_plan_frames(
        self,
        clean_latents: torch.Tensor,
        predicted_plan_latents: torch.Tensor,
        *,
        num_obs_latents: int,
        committed_blocks: int,
        plan_subblocks: int,
        tiled: bool = False,
    ):
        """Decode teacher-forced replans with their actual causal prefixes.

        Each replan anchor was denoised against a different clean GT prefix.
        Sub-blocks within the same plan were jointly denoised and must remain a
        continuous predicted sequence at VAE decode time. Earlier plans commit
        only their first execution sub-block; the final plan commits its full
        horizon.
        """
        if clean_latents.ndim != 5 or predicted_plan_latents.ndim != 5:
            raise ValueError(
                "AR teacher-forced decode expects 5D latent tensors, got "
                f"{tuple(clean_latents.shape)} and "
                f"{tuple(predicted_plan_latents.shape)}."
            )
        if clean_latents.shape[:2] != predicted_plan_latents.shape[:2] or (
            clean_latents.shape[3:] != predicted_plan_latents.shape[3:]
        ):
            raise ValueError(
                "AR teacher-forced clean/predicted latent geometry mismatch: "
                f"{tuple(clean_latents.shape)} vs "
                f"{tuple(predicted_plan_latents.shape)}."
            )
        num_obs_latents = int(num_obs_latents)
        committed_blocks = int(committed_blocks)
        plan_subblocks = int(plan_subblocks)
        if num_obs_latents <= 0 or committed_blocks <= 0 or plan_subblocks <= 0:
            raise ValueError(
                "AR teacher-forced decode geometry must be positive: "
                f"num_obs_latents={num_obs_latents}, "
                f"committed_blocks={committed_blocks}, "
                f"plan_subblocks={plan_subblocks}."
            )
        expected_plan_latents = committed_blocks * plan_subblocks
        if int(predicted_plan_latents.shape[2]) != expected_plan_latents:
            raise ValueError(
                "AR teacher-forced plan latent count mismatch: "
                f"expected={expected_plan_latents}, "
                f"got={int(predicted_plan_latents.shape[2])}."
            )
        required_clean_latents = (
            num_obs_latents + committed_blocks - 1 + plan_subblocks
        )
        if required_clean_latents > int(clean_latents.shape[2]):
            raise ValueError(
                "AR teacher-forced clean latent range is incomplete: "
                f"required={required_clean_latents}, "
                f"available={int(clean_latents.shape[2])}."
            )

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                "VAE temporal_downsample_factor must be positive, got "
                f"{temporal_factor}."
            )
        obs_latents = clean_latents[:, :, :num_obs_latents]
        frames = list(self._decode_latents(obs_latents, tiled=tiled))
        expected_obs_frames = (num_obs_latents - 1) * temporal_factor + 1
        if len(frames) != expected_obs_frames:
            raise RuntimeError(
                "AR teacher-forced observation decode length mismatch: "
                f"expected={expected_obs_frames}, got={len(frames)}."
            )

        for anchor_idx in range(committed_blocks):
            clean_prefix_latents = num_obs_latents + anchor_idx
            plan_start = anchor_idx * plan_subblocks
            committed_subblocks = (
                plan_subblocks if anchor_idx == committed_blocks - 1 else 1
            )
            decode_latents = torch.cat(
                [
                    clean_latents[:, :, :clean_prefix_latents],
                    predicted_plan_latents[
                        :, :, plan_start:plan_start + committed_subblocks
                    ],
                ],
                dim=2,
            )
            decoded = list(self._decode_latents(decode_latents, tiled=tiled))
            clean_prefix_frames = (
                clean_prefix_latents - 1
            ) * temporal_factor + 1
            committed_frames = committed_subblocks * temporal_factor
            block_frames = decoded[
                clean_prefix_frames:clean_prefix_frames + committed_frames
            ]
            if len(block_frames) != committed_frames:
                raise RuntimeError(
                    "AR teacher-forced plan decode length mismatch at "
                    f"anchor_idx={anchor_idx}: expected={committed_frames}, "
                    f"got={len(block_frames)}."
                )
            frames.extend(block_frames)
        return frames

    def _decode_ar_prediction_frames(self, clean_latents, latents_video, tiled: bool = False):
        pred_video_latents = torch.cat([clean_latents, latents_video], dim=2)
        frames = self._decode_latents(pred_video_latents, tiled=tiled)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        clean_frame_count = (int(clean_latents.shape[2]) - 1) * temporal_factor + 1
        return frames[clean_frame_count:], pred_video_latents

    def build_inputs(self, sample, tiled: bool = False):
        has_input_latents = "input_latents" in sample
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "GameWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)

        image_is_pad = sample.get("image_is_pad", None)
        if has_input_latents:
            raw_input_latents = sample["input_latents"]
            if raw_input_latents.ndim != 5:
                raise ValueError(
                    f"`sample['input_latents']` must be 5D [B,C,T,H,W], got shape {tuple(raw_input_latents.shape)}"
                )
            batch_size = int(raw_input_latents.shape[0])
            if image_is_pad is not None:
                if image_is_pad.ndim != 2:
                    raise ValueError(
                        f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                    )
                if image_is_pad.shape[0] != batch_size:
                    raise ValueError(
                        f"`sample['image_is_pad']` batch mismatch: {image_is_pad.shape[0]} vs {batch_size}"
                    )
                num_frames = int(image_is_pad.shape[1])
            else:
                num_frames = (int(raw_input_latents.shape[2]) - 1) * int(self.vae.temporal_downsample_factor) + 1
        else:
            video = sample["video"]
            if video.ndim != 5:
                raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
            if video.shape[1] != 3:
                raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

            batch_size, _, num_frames, height, width = video.shape
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(
                    f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
                )

        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for GameWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        if has_input_latents:
            input_latents = raw_input_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_latents(input_video, tiled=tiled)

        obs_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            num_obs_latents = int(self.video_expert.num_obs_latents)
            obs_latents = input_latents[:, :, 0:num_obs_latents]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        context, context_mask = self._append_sample_proprio_to_context(sample, context, context_mask)
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        result = {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "obs_latents": obs_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }
        # Action expert may request extra supervised heads/route labels.
        for key in getattr(self.action_expert, "training_auxiliary_sample_keys", ()):
            if key in sample:
                v = sample[key]
                if torch.is_tensor(v):
                    v = v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                result[key] = v
        return result

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # Segment offsets
        v_start, v_end = 0, video_seq_len
        a_start, a_end = video_seq_len, total_seq_len

        # video -> video (existing causal logic)
        mask[v_start:v_end, v_start:v_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # action -> action (full connectivity)
        mask[a_start:a_end, a_start:a_end] = True

        # action -> obs (clean) video tokens only
        num_obs_latents = int(self.video_expert.num_obs_latents)
        obs_tokens = min(video_tokens_per_frame * num_obs_latents, video_seq_len)
        mask[a_start:a_end, v_start:v_start + obs_tokens] = True

        if self.video_action_attention_mode == "joint_bidirectional":
            # Clean observations remain pure causal conditions. Only the jointly
            # noised prediction regions exchange information in both directions.
            predicted_video_start = v_start + obs_tokens
            mask[predicted_video_start:v_end, a_start:a_end] = True
            mask[a_start:a_end, predicted_video_start:v_end] = True

        return mask

    def _video_latent_is_pad(
        self,
        image_is_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Map causal-VAE pixel-frame padding to latent-frame padding."""
        if image_is_pad.ndim != 2:
            raise ValueError(
                f"`image_is_pad` must be [B,T], got {tuple(image_is_pad.shape)}."
            )
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.reshape(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        return torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)

    @staticmethod
    def _reduce_ar_video_loss(
        video_loss_per_sample_block: torch.Tensor,
        video_weight: torch.Tensor,
        video_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Reduce AR plan losses while excluding padded target latent blocks."""
        try:
            weighted = video_loss_per_sample_block * video_weight
        except RuntimeError as exc:
            raise ValueError(
                "AR video loss/weight shapes are not broadcastable: "
                f"{tuple(video_loss_per_sample_block.shape)} vs "
                f"{tuple(video_weight.shape)}."
            ) from exc
        if weighted.shape != video_loss_per_sample_block.shape:
            raise ValueError(
                "AR video weight broadcast changed the loss geometry: "
                f"{tuple(video_loss_per_sample_block.shape)} -> "
                f"{tuple(weighted.shape)}."
            )
        if video_is_pad is None:
            return weighted.mean()
        if video_is_pad.shape != weighted.shape:
            raise ValueError(
                "AR video-loss mask shape mismatch: "
                f"{tuple(video_is_pad.shape)} vs {tuple(weighted.shape)}."
            )
        valid = (~video_is_pad).to(device=weighted.device, dtype=weighted.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return ((weighted * valid).sum(dim=1) / valid_sum).mean()

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        num_obs_latents: int,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        latent_is_pad = self._video_latent_is_pad(image_is_pad)
        video_is_pad = latent_is_pad[:, num_obs_latents:]

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @staticmethod
    def _binary_action_count_metrics(
        *,
        scores: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        discrete_dims: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Return additive fixed-threshold counts for exact DDP aggregation."""
        if scores.shape != targets.shape:
            raise ValueError(
                "binary action score/target shape mismatch: "
                f"{tuple(scores.shape)} vs {tuple(targets.shape)}"
            )
        if valid.shape != scores.shape[:2]:
            raise ValueError(
                f"binary action valid mask must be {tuple(scores.shape[:2])}, "
                f"got {tuple(valid.shape)}"
            )
        if scores.shape[-1] != len(discrete_dims):
            raise ValueError(
                f"binary action score dim {scores.shape[-1]} does not match "
                f"discrete_dims {len(discrete_dims)}"
            )
        with torch.no_grad():
            pred_pos = scores.detach().float() >= 0.5
            gt_pos = targets.detach().float() >= 0.5
            mask = valid.to(device=scores.device, dtype=torch.bool).unsqueeze(-1)
            tp = (pred_pos & gt_pos & mask).sum(dim=(0, 1)).float()
            fp = (pred_pos & ~gt_pos & mask).sum(dim=(0, 1)).float()
            tn = (~pred_pos & ~gt_pos & mask).sum(dim=(0, 1)).float()
            fn = (~pred_pos & gt_pos & mask).sum(dim=(0, 1)).float()
            metrics = {
                "action_discrete_tp_count": tp.sum(),
                "action_discrete_fp_count": fp.sum(),
                "action_discrete_tn_count": tn.sum(),
                "action_discrete_fn_count": fn.sum(),
            }
            for local_idx, action_dim in enumerate(discrete_dims):
                stem = f"action_discrete_dim_{int(action_dim):02d}"
                metrics[f"{stem}_tp_count"] = tp[local_idx]
                metrics[f"{stem}_fp_count"] = fp[local_idx]
                metrics[f"{stem}_tn_count"] = tn[local_idx]
                metrics[f"{stem}_fn_count"] = fn[local_idx]
            return metrics

    def _compute_action_heads_losses(
        self,
        pred_heads: Dict[str, Optional[torch.Tensor]],
        target_action: torch.Tensor,
        target_discrete_action: Optional[torch.Tensor],
        action_route_before_action: Optional[torch.Tensor],
        action_is_pad: Optional[torch.Tensor],
        timestep_action: torch.Tensor,
        action_route_valid: Optional[torch.Tensor] = None,
        noisy_action: Optional[torch.Tensor] = None,
        compute_metrics: bool = True,
    ) -> tuple:
        """Compute action-head losses without depending on expert-private head keys."""
        device = target_action.device
        dtype = target_action.dtype
        discrete_action_dim = self.discrete_action_dim
        action_dim = int(self.action_expert.action_dim)
        route_head_enabled = self.action_expert.route_head_enabled()
        B, T = target_action.shape[0], target_action.shape[1]

        if target_discrete_action is not None:
            target_discrete_action = target_discrete_action.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        if action_route_before_action is not None:
            action_route_before_action = action_route_before_action.to(
                device=device, dtype=torch.bool, non_blocking=True
            )

        route_mask = None
        if route_head_enabled:
            if action_route_before_action is None:
                raise ValueError(
                    "`action_route_before_action` is required when routed action heads are enabled."
                )
            route_mask = action_route_before_action.bool()
            if action_route_valid is None:
                route_valid = torch.ones(B, T, device=device, dtype=torch.bool)
            else:
                if action_route_valid.shape[-2:] != action_route_before_action.shape[-2:]:
                    raise ValueError(
                        "action_route_valid shape must match route label, "
                        f"got {tuple(action_route_valid.shape)} vs {tuple(action_route_before_action.shape)}"
                    )
                route_valid = action_route_valid.to(device=device, dtype=torch.bool).squeeze(-1)
        else:
            route_valid = torch.ones(B, T, device=device, dtype=torch.bool)

        pred_flow_routed, route_logit = self.action_expert.route_heads_for_training(
            pred_heads, route_mask
        )
        if pred_flow_routed.shape[-1] != action_dim:
            raise ValueError(
                f"routed flow last dim {pred_flow_routed.shape[-1]} != "
                f"action_dim {action_dim}."
            )
        if target_action.shape[-1] != action_dim:
            raise ValueError(
                f"`target_action` last dim {target_action.shape[-1]} != "
                f"action_dim {action_dim}."
            )

        flow_error = F.mse_loss(
            pred_flow_routed.float(), target_action.float(), reduction="none"
        )
        loss_discrete_value: Optional[torch.Tensor] = None
        metrics: Dict[str, Any] = {}
        # Normalize each semantic group independently. A plain mean over
        # all 22 fields would give the 20 binary dimensions 10x the weight
        # of the two camera dimensions merely because of their count.
        loss_cont_token = flow_error[..., self.continuous_dims].mean(dim=-1)
        loss_disc_token = (
            flow_error[..., self.discrete_dims].mean(dim=-1)
            if discrete_action_dim > 0
            else None
        )

        loss_route_value: Optional[torch.Tensor] = None
        if route_head_enabled:
            if route_logit is None:
                raise ValueError("route logit is required when routed action heads are enabled.")
            if route_logit.shape[-2:] != action_route_before_action.shape[-2:]:
                raise ValueError(
                    f"route logit shape {tuple(route_logit.shape)} does not "
                    f"match action_route_before_action shape "
                    f"{tuple(action_route_before_action.shape)}."
                )
            loss_route_token = F.binary_cross_entropy_with_logits(
                route_logit.float(), action_route_before_action.float(), reduction="none"
            ).squeeze(-1)
        else:
            loss_route_token = None

        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=device, dtype=dtype)
        else:
            valid = torch.ones(B, T, device=device, dtype=dtype)
        continuous_valid = valid
        route_loss_valid = valid
        if route_head_enabled:
            route_valid_f = route_valid.to(device=device, dtype=dtype)
            continuous_valid = valid * route_valid_f
            route_loss_valid = valid * route_valid_f
        continuous_valid_sum = continuous_valid.sum().clamp(min=1.0)
        route_loss_valid_sum = route_loss_valid.sum().clamp(min=1.0)

        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(device=device, dtype=dtype)
        if action_weight.ndim == 1:
            action_weight = action_weight.unsqueeze(1)

        if compute_metrics:
            weighted_cont = loss_cont_token * action_weight

            if T % 8 == 0:
                for position in range(8):
                    position_valid = continuous_valid.bool().clone()
                    position_mask = torch.zeros(
                        T, device=device, dtype=torch.bool
                    )
                    position_mask[position::8] = True
                    position_valid &= position_mask.unsqueeze(0)
                    position_count = position_valid.sum()
                    metrics[f"loss_continuous_position_{position}"] = (
                        (weighted_cont * position_valid).sum()
                        / position_count.clamp_min(1)
                    ).detach()

            if route_head_enabled:
                route_true_valid = route_mask.squeeze(-1) & continuous_valid.bool()
                route_false_valid = ~route_mask.squeeze(-1) & continuous_valid.bool()
                for route_name, route_valid_tokens in (
                    ("gameplay", route_false_valid),
                    ("gui", route_true_valid),
                ):
                    route_count = route_valid_tokens.sum()
                    metrics[f"loss_continuous_{route_name}"] = (
                        (weighted_cont * route_valid_tokens).sum()
                        / route_count.clamp_min(1)
                    ).detach()
                    metrics[
                        f"continuous_{route_name}_valid_token_count"
                    ] = route_count.detach()

        loss_continuous = (loss_cont_token * continuous_valid * action_weight).sum() / continuous_valid_sum

        if loss_disc_token is not None:
            discrete_valid = (
                valid * route_valid.to(device=device, dtype=dtype)
                if route_head_enabled
                else valid
            )
            discrete_valid_sum = discrete_valid.sum().clamp(min=1.0)
            loss_discrete_value = (
                loss_disc_token * discrete_valid * action_weight
            ).sum() / discrete_valid_sum
        if loss_route_token is not None:
            loss_route_value = (loss_route_token * route_loss_valid).sum() / route_loss_valid_sum

        if compute_metrics and discrete_action_dim > 0:
            metric_valid = valid.bool()
            if target_discrete_action is None:
                raise ValueError(
                    "Discrete action metrics require target_discrete_action."
                )
            if route_head_enabled:
                metric_valid = metric_valid & route_valid.to(device=device, dtype=torch.bool)
            if noisy_action is None:
                raise ValueError("action metrics require the full noisy_action tensor.")
            pred_x0_full = self._action_x0_from_velocity(
                noisy_action=noisy_action,
                pred_velocity=pred_flow_routed,
                timestep_action=timestep_action,
            )
            discrete_scores = 0.5 * (
                pred_x0_full[..., self.discrete_dims] + 1.0
            )
            metrics.update(self._binary_action_count_metrics(
                scores=discrete_scores,
                targets=target_discrete_action,
                valid=metric_valid,
                discrete_dims=self.discrete_dims,
            ))

        return loss_continuous, loss_discrete_value, loss_route_value, metrics

    def _step_continuous(
        self,
        latents_action: torch.Tensor,
        pred_continuous: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one flow-matching scheduler step over the full action vector."""
        if pred_continuous.shape[-1] != int(self.action_expert.action_dim):
            raise ValueError(
                "scheduler output must cover the full action vector, "
                f"got {pred_continuous.shape[-1]}."
            )
        return self.infer_action_scheduler.step(pred_continuous, delta, latents_action)

    def _action_noise_block_horizon(self) -> int:
        geometry = getattr(self, "ar_geometry", None)
        if geometry is not None and getattr(geometry, "raw_steps_per_latent", None) is not None:
            return int(geometry.raw_steps_per_latent)
        return 8

    def _share_action_noise_across_blocks(
        self,
        noise_action: torch.Tensor,
        *,
        action_per_block: int,
    ) -> torch.Tensor:
        if not bool(self.action_noise_share_across_blocks):
            return noise_action
        if noise_action.ndim != 3:
            raise ValueError(f"noise_action must be [B,T,D], got {tuple(noise_action.shape)}")
        action_per_block = int(action_per_block)
        if action_per_block <= 0:
            raise ValueError(f"action_per_block must be positive, got {action_per_block}")
        batch_size, action_len, action_dim = (
            int(noise_action.shape[0]),
            int(noise_action.shape[1]),
            int(noise_action.shape[2]),
        )
        if action_len % action_per_block != 0:
            raise ValueError(
                "shared block action noise requires action length divisible by "
                f"action_per_block, got action_len={action_len}, "
                f"action_per_block={action_per_block}"
            )
        num_blocks = action_len // action_per_block
        if num_blocks <= 1:
            return noise_action
        first_block = noise_action.reshape(
            batch_size, num_blocks, action_per_block, action_dim
        )[:, :1]
        return first_block.expand(
            batch_size, num_blocks, action_per_block, action_dim
        ).reshape_as(noise_action).clone()

    def _prepare_action_training_timesteps(
        self,
        timesteps_action_blocks: torch.Tensor,
        *,
        action_per_block: int,
    ) -> torch.Tensor:
        """Expand per-block action timesteps to per-token timesteps."""
        if timesteps_action_blocks.ndim != 2:
            raise ValueError(
                "timesteps_action_blocks must be [B,num_blocks], got "
                f"{tuple(timesteps_action_blocks.shape)}"
            )
        action_per_block = int(action_per_block)
        if action_per_block <= 0:
            raise ValueError(
                f"action_per_block must be positive, got {action_per_block}"
            )
        return timesteps_action_blocks.repeat_interleave(action_per_block, dim=1)

    def _action_x0_from_velocity(
        self,
        *,
        noisy_action: torch.Tensor,
        pred_velocity: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.shape != pred_velocity.shape:
            raise ValueError(
                "action x0 reconstruction requires matching state and velocity, "
                f"got {tuple(noisy_action.shape)} vs {tuple(pred_velocity.shape)}."
            )
        sigma = (timestep_action / float(self.train_action_scheduler.num_train_timesteps)).to(
            device=noisy_action.device, dtype=noisy_action.dtype
        )
        while sigma.ndim < noisy_action.ndim:
            sigma = sigma.unsqueeze(-1)
        return noisy_action - sigma * pred_velocity.to(dtype=noisy_action.dtype)

    def _sample_action_latents(
        self,
        *,
        action_horizon: int,
        rand_device: str,
        seed: Optional[int],
    ) -> torch.Tensor:
        """Sample the action prior, with an opt-in camera-DC diagnostic.

        The optional diagnostic attributes preserve iid Gaussian sampling by
        default. ``action_initial_camera_noise_mean_scale=0`` removes
        temporal DC while preserving token variance;
        ``action_initial_camera_noise_scale=0`` makes the whole camera prior
        deterministic.  Both isolate camera-noise effects without silently
        changing deployed behavior.
        """
        # When coupled, video and action share the same diffusion trajectory
        # and should use the same noise stream (same seed) for alignment.
        # When uncoupled, they denoise independently and must not share
        # prefix-identical noise from the same seed.
        if seed is not None and not self.coupled_video_action_timesteps:
            seed = int(seed) ^ 0x5DEECE66D
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(int(seed))
        )
        latents = torch.randn(
            (1, int(action_horizon), int(self.action_expert.action_dim)),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        )
        mean_scale = float(
            getattr(self, "action_initial_camera_noise_mean_scale", 1.0)
        )
        noise_scale = float(
            getattr(self, "action_initial_camera_noise_scale", 1.0)
        )
        if not 0.0 <= noise_scale <= 1.0:
            raise ValueError(
                "action_initial_camera_noise_scale must be in [0, 1], "
                f"got {noise_scale}"
            )
        latents = self._share_action_noise_across_blocks(
            latents,
            action_per_block=self._action_noise_block_horizon(),
        )
        if mean_scale != 1.0:
            if not 0.0 <= mean_scale <= 1.0:
                raise ValueError(
                    "action_initial_camera_noise_mean_scale must be in [0, 1], "
                    f"got {mean_scale}"
                )
            horizon = int(latents.shape[1])
            if horizon <= 1:
                raise ValueError(
                    "Camera-DC suppression requires action_horizon > 1, "
                    f"got {horizon}"
                )
            camera_dims = list(self.continuous_dims[:2])
            if camera_dims != [0, 1]:
                raise ValueError(
                    "Camera-DC diagnostic requires pitch/yaw at continuous "
                    f"dimensions [0, 1], got {camera_dims}"
                )
            camera = latents[..., camera_dims]
            temporal_mean = camera.mean(dim=1, keepdim=True)
            # Var(mean)=1/T and Var(x-mean)=1-1/T for iid N(0,1).  Choose
            # the AC scale so alpha*mean + beta*(x-mean) still has unit
            # per-token variance after changing the DC scale alpha.
            ac_scale = (
                (1.0 - mean_scale * mean_scale / horizon)
                / (1.0 - 1.0 / horizon)
            ) ** 0.5
            latents[..., camera_dims] = (
                mean_scale * temporal_mean
                + ac_scale * (camera - temporal_mean)
            )
        if noise_scale != 1.0:
            camera_dims = list(self.continuous_dims[:2])
            if camera_dims != [0, 1]:
                raise ValueError(
                    "Camera-noise scaling requires pitch/yaw at continuous "
                    f"dimensions [0, 1], got {camera_dims}"
                )
            latents[..., camera_dims] *= noise_scale
        return latents.to(device=self.device, dtype=self.torch_dtype)

    def _slice_to_continuous(
        self,
        action_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Select the semantic continuous fields from a full action vector."""
        if action_tensor.shape[-1] == self.continuous_action_dim:
            return action_tensor
        return action_tensor[..., self.continuous_dims]

    def forward(
        self,
        sample,
        tiled: bool = False,
        global_step: Optional[int] = None,
        compute_metrics: bool = True,
    ):
        return self.training_loss(
            sample,
            tiled=tiled,
            global_step=global_step,
            compute_metrics=compute_metrics,
        )

    def training_loss(
        self,
        sample,
        tiled: bool = False,
        global_step: Optional[int] = None,
        compute_metrics: bool = True,
    ):
        # Route to AR Teacher-Forcing if model is configured in AR mode
        if self.ar_mode:
            return self.training_loss_ar(
                sample,
                tiled=tiled,
                global_step=global_step,
                compute_metrics=compute_metrics,
            )

        # Apply text condition dropout for CFG training.
        drop_mask = None
        if getattr(self, "training", False) and self.text_condition_dropout_prob > 0:
            context = sample.get("context")
            context_mask = sample.get("context_mask")
            if context is not None and context_mask is not None:
                sample = dict(sample)
                context, context_mask, drop_mask = self._apply_text_condition_dropout(
                    sample, context, context_mask
                )
                sample["context"] = context
                sample["context_mask"] = context_mask

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        raw_action_is_pad = action_is_pad
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        if self.coupled_video_action_timesteps:
            training_u = self.train_video_scheduler.sample_training_u(
                batch_size=batch_size,
                device=self.device,
            )
            timestep_video = self.train_video_scheduler.timestep_from_u(training_u, dtype=input_latents.dtype)
        else:
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["obs_latents"] is not None:
            num_obs_latents = inputs["obs_latents"].shape[2]
            latents[:, :, 0:num_obs_latents] = inputs["obs_latents"]
        else:
            num_obs_latents = 1

        noise_action = torch.randn_like(action)
        if self.coupled_video_action_timesteps:
            timestep_action = self.train_action_scheduler.timestep_from_u(training_u, dtype=action.dtype)
        else:
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Strip action steps that fall within the obs region (they correspond to
        # transitions already visible as clean condition — not forward policy signal).
        if inputs["obs_latents"] is not None:
            obs_action_steps = (int(self.video_expert.num_obs_frames) - 1) * int(self.frame_stride)
            obs_action_steps = min(obs_action_steps, noisy_action.shape[1] - 1)
            noisy_action = noisy_action[:, obs_action_steps:]
            target_action = target_action[:, obs_action_steps:]
            if action_is_pad is not None:
                action_is_pad = action_is_pad[:, obs_action_steps:]
            expected_action_len = (int(self.num_frames) - int(self.num_obs_frames)) * int(self.frame_stride)
            if noisy_action.shape[1] != expected_action_len:
                raise ValueError(
                    f"Non-AR action length mismatch after obs slicing: expected {expected_action_len}, "
                    f"got {noisy_action.shape[1]} (num_frames={self.num_frames}, "
                    f"num_obs_frames={self.num_obs_frames}, frame_stride={self.frame_stride})."
                )
            video_action_condition = action[:, obs_action_steps:]
            video_action_condition_start_latent_idx = num_obs_latents
            # Align the auxiliary targets with the same ``obs_action_steps``
            # slice so ``_compute_action_heads_losses`` can rely on shape
            # equality with ``noisy_action``/``target_action``.
            if inputs.get("target_discrete_action") is not None:
                inputs["target_discrete_action"] = inputs["target_discrete_action"][:, obs_action_steps:]
            if inputs.get("action_route_before_action") is not None:
                inputs["action_route_before_action"] = inputs["action_route_before_action"][:, obs_action_steps:]
            if inputs.get("action_route_valid") is not None:
                inputs["action_route_valid"] = inputs["action_route_valid"][:, obs_action_steps:]
        else:
            video_action_condition = action
            video_action_condition_start_latent_idx = 1

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=video_action_condition,
            action_condition_start_latent_idx=video_action_condition_start_latent_idx,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        noisy_action_input = noisy_action

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action_input,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )

        # Build MoT forward inputs
        embeds_all = {"video": video_tokens, "action": action_tokens}
        freqs_all = {"video": video_pre["freqs"], "action": action_pre["freqs"]}
        context_all = {
            "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        }
        t_mod_all = {"video": video_pre["t_mod"], "action": action_pre["t_mod"]}

        tokens_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        action_hidden = tokens_out["action"]
        pred_heads = self.action_expert.post_dit(action_hidden, action_pre)

        if inputs["obs_latents"] is None:
            num_obs_latents = 0
        else:
            num_obs_latents = inputs["obs_latents"].shape[2]
            pred_video = pred_video[:, :, num_obs_latents:]
            target_video = target_video[:, :, num_obs_latents:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            num_obs_latents=num_obs_latents,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # Joint-flow action heads: one continuous flow MSE over the full
        # action vector (grouped per semantic field type) plus an optional
        # GUI route BCE. All slices (obs prefix) for the auxiliary fields
        # are applied at the same index as ``action``/``target_action``.
        #
        # Critical: the flow loss target is the **flow velocity**
        # ``target_action = noise - clean``, not the clean action itself.
        target_discrete_action = inputs.get("target_discrete_action")
        action_route_before_action = inputs.get("action_route_before_action")
        action_route_valid = inputs.get("action_route_valid")
        target_flow_velocity = target_action
        loss_continuous, loss_discrete, loss_gui_route, action_metrics = \
            self._compute_action_heads_losses(
                pred_heads=pred_heads,
                target_action=target_flow_velocity,
                target_discrete_action=target_discrete_action,
                action_route_before_action=action_route_before_action,
                action_is_pad=action_is_pad,
                timestep_action=timestep_action,
                action_route_valid=action_route_valid,
                compute_metrics=compute_metrics,
                noisy_action=noisy_action,
            )

        # Combine per-head losses with their lambdas. ``lambda_action``
        # is the *overall* action-loss weight applied to the sum
        # below; the GUI route loss uses its own ``lambda_gui_route``.
        loss_action = (
            self.action_head_lambda_continuous * loss_continuous
            + (self.action_head_lambda_discrete * loss_discrete
               if loss_discrete is not None else 0.0)
        )
        # GUI route loss is an independent outer term with its own
        # lambda (NOT folded into ``loss_action`` which is then scaled
        # by ``loss_lambda_action``):
        #   loss_total = λ_video * loss_video
        #              + λ_action * (λ_continuous * loss_continuous
        #                          + λ_discrete * loss_discrete)
        #              + λ_gui_route * loss_gui_route
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        if loss_gui_route is not None:
            loss_total = loss_total + self.action_head_lambda_gui_route * loss_gui_route

        loss_event_occurrence_value, event_occurrence_metrics = self._compute_event_occurrence_loss(
            sample=sample,
            action_hidden=action_hidden,
            batch_size=batch_size,
            num_noisy=0,  # non-AR: no block partition
            compute_metrics=compute_metrics,
        )
        loss_total = loss_total + loss_event_occurrence_value

        if not compute_metrics:
            return loss_total, {}
        loss_dict = {
            "loss_video": self.loss_lambda_video * loss_video.detach(),
            "loss_action": self.loss_lambda_action * loss_action.detach(),
            "loss_continuous": self.action_head_lambda_continuous * loss_continuous.detach(),
        }
        if loss_discrete is not None:
            loss_dict["loss_discrete"] = self.action_head_lambda_discrete * loss_discrete.detach()
        if loss_gui_route is not None:
            loss_dict["loss_gui_route"] = self.action_head_lambda_gui_route * loss_gui_route.detach()
        for k, v in action_metrics.items():
            loss_dict[k] = v
        if event_occurrence_metrics["active"]:
            self._add_event_occurrence_metrics(loss_dict, event_occurrence_metrics)
        return loss_total, loss_dict

    # AR Teacher-Forcing Training Loss

    def training_loss_ar(
        self,
        sample,
        tiled: bool = False,
        global_step: Optional[int] = None,
        compute_metrics: bool = True,
    ):
        """AR Teacher-Forcing training loss with history prefix.

        Activated when model config `ar_mode=True`.

        Required sample keys: ``input_latents``, ``context`` / ``context_mask``,
        ``action``. AR-history mode additionally: ``history_segment_latents``,
        ``history_segment_mask``, ``history_segment_start_raw``,
        ``history_segment_next_cycle_start_raw``, ``sample_start_raw``.
        Optional: ``action_is_pad``.
        """
        from .wan22.wan_video_dit import build_identity_rope_like

        # Extract inputs
        input_latents = sample["input_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        batch_size = input_latents.shape[0]
        num_latent_frames = input_latents.shape[2]

        static_context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        static_context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        base_context_len = int(static_context.shape[1])
        static_context, static_context_mask, text_drop_mask = self._apply_text_condition_dropout(
            sample, static_context, static_context_mask
        )
        runtime_context, runtime_context_mask = self._append_sample_proprio_to_context(
            sample, static_context, static_context_mask
        )
        proprio_token_count = int(runtime_context.shape[1]) - base_context_len

        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        # AR split
        if self.ar_geometry is None:
            raise RuntimeError("AR training requires ar_geometry.")
        num_obs_latents = int(self.video_expert.num_obs_latents)
        committed_blocks = int(self.ar_geometry.pred_latents)
        execution_horizon = int(self.execution_horizon)
        prediction_horizon = int(self.prediction_horizon)
        plan_subblocks = prediction_horizon // execution_horizon
        required_future_latents = committed_blocks + plan_subblocks - 1
        available_future_latents = num_latent_frames - num_obs_latents
        if available_future_latents < required_future_latents:
            raise ValueError(
                "AR joint plan requires lookahead video latents: "
                f"available_future_latents={available_future_latents}, "
                f"required={required_future_latents}."
            )

        num_clean = num_latent_frames
        plan_video_blocks = committed_blocks * plan_subblocks
        clean_latents = input_latents

        # Sample video timesteps and add noise.
        if self.coupled_video_action_timesteps:
            training_u = self.train_video_scheduler.sample_training_u(
                batch_size=batch_size * plan_video_blocks,
                device=self.device,
            ).reshape(batch_size, plan_video_blocks)
            timesteps_noisy = self.train_video_scheduler.timestep_from_u(
                training_u,
                dtype=input_latents.dtype,
            )
        else:
            timesteps_noisy = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size * plan_video_blocks,
                device=self.device,
                dtype=input_latents.dtype,
            ).reshape(batch_size, plan_video_blocks)

        gt_latents_list = []
        noise_latents_list = []
        plan_latent_indices = []
        for anchor_idx in range(committed_blocks):
            for sub_idx in range(plan_subblocks):
                latent_idx = num_obs_latents + anchor_idx + sub_idx
                gt_i = input_latents[:, :, latent_idx]
                gt_latents_list.append(gt_i)
                plan_latent_indices.append(latent_idx)
                # Keep one RNG call per block and the original block order so
                # seeded runs consume exactly the same random stream.
                noise_latents_list.append(torch.randn_like(gt_i))
        gt_latents = torch.stack(gt_latents_list, dim=2)
        noise_latents = torch.stack(noise_latents_list, dim=2)
        noisy_latents = self.train_video_scheduler.add_noise(
            gt_latents, noise_latents, timesteps_noisy
        )
        targets = self.train_video_scheduler.training_target(
            gt_latents, noise_latents, timesteps_noisy
        )

        # Prefix construction (AR-history mode)
        history_valid_mask = None
        history_step_cache: dict[str, Any] | None = (
            {} if self.ar_history_mode else None
        )

        if self.ar_history_mode:
            history_prefix_tokens, history_valid_mask, _ = \
                self.history_module.build_history_prefix_batch(
                    batch=sample,
                    training=self.training,
                    device=self.device,
                    dtype=self.torch_dtype,
                    step_cache=history_step_cache,
                )
            far_tokens = history_prefix_tokens  # [B, H_history, D]
            num_far_tokens = far_tokens.shape[1]
        else:
            far_tokens = None
            num_far_tokens = 0

        # Video Expert: prepare clean and noisy tokens separately
        clean_patch = self.video_expert.patch_embedding(clean_latents)  # [B, D, T_clean, h, w]
        from einops import rearrange
        clean_tokens = rearrange(clean_patch, "b c f h w -> b (f h w) c")
        _, h, w = clean_patch.shape[2:]
        tokens_per_frame = h * w

        # Prepend far tokens to clean
        if self.ar_history_mode:
            clean_with_far = torch.cat([far_tokens, clean_tokens], dim=1)  # [B, far+clean, D]
        else:
            clean_with_far = clean_tokens  # [B, clean, D]

        # Build time embeddings for clean (t=0)
        from .wan22.wan_video_dit import sinusoidal_embedding_1d
        S_clean = clean_with_far.shape[1]
        clean_t_value = torch.zeros(1, device=self.device, dtype=self.torch_dtype)
        clean_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, clean_t_value)
        clean_t_single = self.video_expert.time_embedding(clean_t_emb)
        clean_t_mod_single = self.video_expert.time_projection(clean_t_single).unflatten(
            1,
            (6, self.video_expert.hidden_dim),
        )
        clean_t_mod = clean_t_mod_single[:, None].expand(
            batch_size,
            S_clean,
            -1,
            -1,
        )

        # Clean RoPE: prefix RoPE + near RoPE for clean frames
        clean_near_freqs = self.video_expert._get_3d_rope_freqs(0, num_clean, h, w, self.device)

        # Identity RoPE for history prefix: no positional rotation
        if self.ar_history_mode:
            prefix_freqs = build_identity_rope_like(clean_near_freqs, n_tokens=num_far_tokens)
            clean_freqs = torch.cat([prefix_freqs, clean_near_freqs], dim=0)
        else:
            clean_freqs = clean_near_freqs

        # Fold the plan-block axis into batch so patch/time projections launch
        # once instead of once per block. Unbinding below only creates views.
        noisy_blocks = noisy_latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * plan_video_blocks,
            noisy_latents.shape[1],
            1,
            noisy_latents.shape[3],
            noisy_latents.shape[4],
        )
        noisy_patch = self.video_expert.patch_embedding(noisy_blocks)
        noisy_tokens = rearrange(
            noisy_patch,
            "(b p) c f h w -> b p (f h w) c",
            b=batch_size,
            p=plan_video_blocks,
        )
        noisy_tokens_list = list(noisy_tokens.unbind(dim=1))

        noisy_t_emb = sinusoidal_embedding_1d(
            self.video_expert.freq_dim,
            timesteps_noisy.reshape(-1),
        )
        noisy_t_unique = self.video_expert.time_embedding(noisy_t_emb).reshape(
            batch_size,
            plan_video_blocks,
            -1,
        )
        noisy_t = noisy_t_unique[:, :, None, :].expand(
            -1,
            -1,
            tokens_per_frame,
            -1,
        )
        noisy_t_mod_unique = self.video_expert.time_projection(noisy_t_unique).unflatten(
            2,
            (6, self.video_expert.hidden_dim),
        )
        noisy_t_mod = noisy_t_mod_unique[:, :, None].expand(
            -1,
            -1,
            tokens_per_frame,
            -1,
            -1,
        )
        noisy_t_mod_list = list(noisy_t_mod.unbind(dim=1))

        noisy_freqs_list = []
        for i in range(plan_video_blocks):
            anchor_idx = i // plan_subblocks
            sub_idx = i % plan_subblocks
            t_idx = num_obs_latents + anchor_idx + sub_idx
            block_freqs = self.video_expert._get_3d_rope_freqs(t_idx, 1, h, w, self.device)
            noisy_freqs_list.append(block_freqs)

        obs_action_steps = (int(self.video_expert.num_obs_frames) - 1) * int(self.ar_geometry.frame_stride)
        obs_action_steps = min(obs_action_steps, action.shape[1] - 1)
        future_action = action[:, obs_action_steps:]
        expected_future_len = (committed_blocks - 1) * execution_horizon + prediction_horizon
        if future_action.shape[1] != expected_future_len:
            raise ValueError(
                "AR action future length mismatch for joint plans: expected "
                f"(committed_blocks-1)*execution_horizon+prediction_horizon = "
                f"({committed_blocks}-1)*{execution_horizon}+{prediction_horizon} "
                f"= {expected_future_len}, got {future_action.shape[1]}"
            )

        def _build_plan_sequence(tensor):
            if tensor is None:
                return None
            if tensor.shape[1] == expected_future_len:
                # ``cat([tensor[:, a*E:a*E+P] for a in range(C)])`` is exactly
                # this index_select (same elements, same order, pure copy).
                # The index lives on the indexed tensor's own device (the
                # geometry cache is keyed per device, so this is a dict hit
                # after the first call).
                return tensor.index_select(
                    1,
                    self._get_ar_plan_sequence_index(
                        committed_blocks=committed_blocks,
                        prediction_horizon=prediction_horizon,
                        execution_horizon=execution_horizon,
                        device=tensor.device,
                    ),
                )
            pieces = []
            for anchor_idx in range(committed_blocks):
                start = anchor_idx * execution_horizon
                end = start + prediction_horizon
                pieces.append(tensor[:, start:end])
            return torch.cat(pieces, dim=1)

        action_for_denoise = _build_plan_sequence(future_action)
        action_position_ids = self._build_ar_plan_action_position_ids(
            committed_blocks=committed_blocks,
            prediction_horizon=prediction_horizon,
            execution_horizon=execution_horizon,
            device=self.action_expert.freqs.device,
        )
        if action_position_ids.numel() != action_for_denoise.shape[1]:
            raise RuntimeError(
                "AR training action RoPE geometry mismatch: "
                f"position_ids={action_position_ids.numel()} vs "
                f"action_seq_len={action_for_denoise.shape[1]}"
            )
        if action_is_pad is not None:
            action_is_pad = _build_plan_sequence(action_is_pad[:, obs_action_steps:])
        target_discrete_action_ar = sample.get("target_discrete_action")
        action_route_before_action_ar = sample.get("action_route_before_action")
        action_route_valid_ar = sample.get("action_route_valid")
        if target_discrete_action_ar is not None:
            target_discrete_action_ar = _build_plan_sequence(target_discrete_action_ar[:, obs_action_steps:])
        if action_route_before_action_ar is not None:
            action_route_before_action_ar = _build_plan_sequence(action_route_before_action_ar[:, obs_action_steps:])
        if action_route_valid_ar is not None:
            action_route_valid_ar = _build_plan_sequence(action_route_valid_ar[:, obs_action_steps:])

        plan_action_horizon = prediction_horizon
        auxiliary_action_horizon = execution_horizon
        action_per_block = plan_action_horizon
        plan_subblocks = plan_action_horizon // auxiliary_action_horizon
        num_noisy = committed_blocks
        if action_for_denoise.shape[1] % num_noisy != 0:
            raise ValueError(
                "AR action plan length must be divisible by num_noisy, "
                f"got action_len={action_for_denoise.shape[1]}, num_noisy={num_noisy}"
            )

        action_context_mask = self._build_ar_action_context_mask(
            context_mask=runtime_context_mask,
            proprio_token_count=proprio_token_count,
            action_seq_len=action_for_denoise.shape[1],
            num_blocks=committed_blocks,
        )

        runtime_video_context = self.video_expert.text_embedding(runtime_context)
        # runtime_context is static_context followed by proprio tokens. Reuse
        # the row-wise text projection for the clean/static prefix.
        clean_video_context = runtime_video_context[:, :static_context.shape[1]]
        video_context_payload = {
            "context": clean_video_context,
            "mask": static_context_mask,
        }
        use_action_conditioned_video = bool(getattr(self.video_expert, "action_conditioned", False))
        noisy_video_context_payloads_list = []
        for anchor_idx in range(committed_blocks):
            visible_proprio_count = self._ar_visible_proprio_count_for_block(
                anchor_idx,
                proprio_token_count,
            )
            if use_action_conditioned_video:
                i = anchor_idx
                payload = self.video_expert.build_video_context_payload(
                    context=runtime_context,
                    context_mask=self._mask_future_proprio_tokens(
                        context_mask=runtime_context_mask,
                        proprio_token_count=proprio_token_count,
                        visible_proprio_count=visible_proprio_count,
                    ),
                    query_latent_frames=1,
                    tokens_per_frame=tokens_per_frame,
                    action=action_for_denoise[:, i * action_per_block:(i + 1) * action_per_block],
                    action_condition_start_latent_idx=0,
                )
            else:
                payload = {
                    "context": runtime_video_context,
                    "mask": self._mask_future_proprio_tokens(
                        context_mask=runtime_context_mask,
                        proprio_token_count=proprio_token_count,
                        visible_proprio_count=visible_proprio_count,
                        query_len=tokens_per_frame,
                    ),
                }
            noisy_video_context_payloads_list.extend([payload] * plan_subblocks)

        # Both low-frequency parameterizations share the same deep-clean
        # prefill/denoise skeleton; the boolean only switches the continuous
        # action parameterization, not the AR backbone execution path.
        clean_cache = self.mot.prefill_clean_kv_cache(
            clean_video_tokens=clean_with_far,
            clean_video_freqs=clean_freqs,
            clean_video_t_mod=clean_t_mod,
            video_context_payload=video_context_payload,
            far_token_count=num_far_tokens,
            tokens_per_frame=tokens_per_frame,
            history_valid_mask=history_valid_mask,
        )
        noise_action = torch.randn_like(action_for_denoise)
        noise_action = self._share_action_noise_across_blocks(
            noise_action,
            action_per_block=auxiliary_action_horizon,
        )
        if self.coupled_video_action_timesteps:
            timesteps_action_blocks = self.train_action_scheduler.timestep_from_u(
                training_u,
                dtype=action_for_denoise.dtype,
            )
        else:
            timesteps_action_blocks = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size * committed_blocks * plan_subblocks,
                device=self.device,
                dtype=action_for_denoise.dtype,
            ).reshape(batch_size, committed_blocks * plan_subblocks)
        timestep_action = self._prepare_action_training_timesteps(
            timesteps_action_blocks,
            action_per_block=auxiliary_action_horizon,
        )
        noisy_action = self.train_action_scheduler.add_noise(action_for_denoise, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action_for_denoise, noise_action, timestep_action)

        noisy_action_input = noisy_action

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action_input,
            timestep=timestep_action,
            context=runtime_context,
            context_mask=action_context_mask,
            position_ids=action_position_ids,
        )
        action_tokens_out = action_pre["tokens"]
        action_context_payload = {
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        }
        if action_route_before_action_ar is not None:
            action_context_payload["route_mask"] = action_route_before_action_ar.bool()
        if action_route_valid_ar is not None:
            action_context_payload["route_valid"] = action_route_valid_ar.bool()

        mot_out = self.mot.forward_ar_from_clean_cache(
            cache=clean_cache,
            noisy_video_tokens_list=noisy_tokens_list,
            action_tokens=action_tokens_out,
            noisy_video_freqs_list=noisy_freqs_list,
            action_freqs=action_pre["freqs"],
            noisy_video_t_mod_list=noisy_t_mod_list,
            action_t_mod=action_pre["t_mod"],
            video_context_payload=video_context_payload,
            action_context_payload=action_context_payload,
            num_obs_latents=num_obs_latents,
            noisy_video_context_payloads_list=noisy_video_context_payloads_list,
            num_action_blocks=committed_blocks,
        )

        # Compute losses
        # Fold plan blocks into the batch for one head launch and one
        # unpatchify. Blocks remain independent; only the batch geometry changes.
        noisy_out_tokens = torch.stack(mot_out["noisy_video"], dim=1)
        noisy_out_tokens_flat = noisy_out_tokens.reshape(
            batch_size * plan_video_blocks,
            noisy_out_tokens.shape[2],
            noisy_out_tokens.shape[3],
        )
        noisy_t_flat = noisy_t.reshape(
            batch_size * plan_video_blocks, noisy_t.shape[2], noisy_t.shape[3]
        )
        pred_blocks_flat = self.video_expert.head(
            noisy_out_tokens_flat, noisy_t_flat
        )
        pred_blocks_flat = rearrange(
            pred_blocks_flat,
            'bp (f h w) (x y z c) -> bp c (f x) (h y) (w z)',
            f=1,
            h=h,
            w=w,
            x=self.video_expert.patch_size[0],
            y=self.video_expert.patch_size[1],
            z=self.video_expert.patch_size[2],
        )
        pred_video_velocities = rearrange(
            pred_blocks_flat,
            '(b p) c f h w -> b c (p f) h w',
            b=batch_size,
            p=plan_video_blocks,
        )
        video_loss_per_sample_block = F.mse_loss(
            pred_video_velocities.float(), targets.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        video_weight = self.train_video_scheduler.training_weight(
            timesteps_noisy
        ).to(
            device=video_loss_per_sample_block.device,
            dtype=video_loss_per_sample_block.dtype,
        )
        video_plan_is_pad = None
        if image_is_pad is not None:
            latent_is_pad = self._video_latent_is_pad(image_is_pad)
            if latent_is_pad.shape[1] != num_latent_frames:
                raise ValueError(
                    "AR video padding/latent shape mismatch: "
                    f"padding latents={latent_is_pad.shape[1]}, "
                    f"video latents={num_latent_frames}."
                )
            plan_index = torch.tensor(
                plan_latent_indices,
                device=latent_is_pad.device,
                dtype=torch.long,
            )
            video_plan_is_pad = latent_is_pad.index_select(1, plan_index)
        loss_video = self._reduce_ar_video_loss(
            video_loss_per_sample_block,
            video_weight,
            video_plan_is_pad,
        )

        # Action loss
        pred_heads = self.action_expert.post_dit(mot_out["action"], action_pre)
        # Strict AR alignment check: all action-derived targets must
        # share the same post-obs-prefix length.
        if target_discrete_action_ar is not None:
            assert (
                target_discrete_action_ar.shape[1]
                == action_for_denoise.shape[1]
            ), (
                f"AR alignment error: target_discrete_action len "
                f"{target_discrete_action_ar.shape[1]} != action_for_denoise len "
                f"{action_for_denoise.shape[1]}."
            )
        if action_route_before_action_ar is not None:
            assert (
                action_route_before_action_ar.shape[1]
                == action_for_denoise.shape[1]
            ), (
                f"AR alignment error: action_route_before_action len "
                f"{action_route_before_action_ar.shape[1]} != action_for_denoise len "
                f"{action_for_denoise.shape[1]}."
            )
        # Continuous loss target = flow velocity (noise - clean) on
        # continuous dims only.
        target_flow_velocity = target_action
        loss_continuous, loss_discrete, loss_gui_route, action_metrics = \
            self._compute_action_heads_losses(
                pred_heads=pred_heads,
                target_action=target_flow_velocity,
                target_discrete_action=target_discrete_action_ar,
                action_route_before_action=action_route_before_action_ar,
                action_is_pad=action_is_pad,
                timestep_action=timestep_action,
                action_route_valid=action_route_valid_ar,
                compute_metrics=compute_metrics,
                noisy_action=noisy_action,
            )
        loss_action = (
            self.action_head_lambda_continuous * loss_continuous
            + (self.action_head_lambda_discrete * loss_discrete
               if loss_discrete is not None else 0.0)
        )
        # GUI route loss is an independent outer term with its own
        # lambda (NOT folded into ``loss_action`` which is then scaled
        # by ``loss_lambda_action``):
        #   loss_total = λ_video * loss_video
        #              + λ_action * (λ_continuous * loss_continuous
        #                          + λ_discrete * loss_discrete)
        #              + λ_gui_route * loss_gui_route
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        if loss_gui_route is not None:
            loss_total = loss_total + self.action_head_lambda_gui_route * loss_gui_route

        loss_event_occurrence_value, event_occurrence_metrics = self._compute_event_occurrence_loss(
            sample=sample,
            action_hidden=mot_out["action"],
            batch_size=batch_size,
            num_noisy=committed_blocks,
            compute_metrics=compute_metrics,
        )
        loss_total = loss_total + loss_event_occurrence_value

        history_predictive_loss = None
        history_predictive_metrics = {}
        if self.ar_history_mode and self.history_module is not None:
            raw_hist_loss, history_predictive_metrics = self.history_module.predictive_history_loss_batch(
                batch=sample,
                device=self.device,
                dtype=self.torch_dtype,
                step_cache=history_step_cache,
                compute_metrics=compute_metrics,
            )
            hist_weight = float(getattr(self.history_config, "predictive_loss_weight", 0.0))
            if hist_weight > 0.0:
                history_predictive_loss = hist_weight * raw_hist_loss
                loss_total = loss_total + history_predictive_loss

        if not compute_metrics:
            return loss_total, {}
        loss_dict = {
            "loss_video": self.loss_lambda_video * loss_video.detach(),
            "loss_action": self.loss_lambda_action * loss_action.detach(),
            "loss_continuous": self.action_head_lambda_continuous * loss_continuous.detach(),
        }
        if history_predictive_loss is not None:
            loss_dict["loss_history_predictive"] = history_predictive_loss.detach()
            loss_dict["loss_history_predictive_raw"] = history_predictive_metrics[
                "history_predictive_loss_raw"
            ].detach()
        for k, v in history_predictive_metrics.items():
            if k not in loss_dict:
                loss_dict[k] = v.detach()
        if loss_discrete is not None:
            loss_dict["loss_discrete"] = self.action_head_lambda_discrete * loss_discrete.detach()
        if loss_gui_route is not None:
            loss_dict["loss_gui_route"] = self.action_head_lambda_gui_route * loss_gui_route.detach()
        for k, v in action_metrics.items():
            loss_dict[k] = v
        if event_occurrence_metrics["active"]:
            self._add_event_occurrence_metrics(loss_dict, event_occurrence_metrics)
        return loss_total, loss_dict


    @staticmethod
    def _add_event_occurrence_metrics(loss_dict: dict, metrics: dict) -> None:
        loss_dict["loss_event_occurrence"] = metrics["loss_value"]
        loss_dict["loss_event_occurrence_raw"] = metrics["loss_raw"]
        loss_dict["event_occurrence_acc"] = metrics["acc"]
        loss_dict["event_occurrence_valid_ratio"] = metrics["valid_ratio"]
        loss_dict["event_occurrence_pos_ratio"] = metrics["pos_ratio"]
        loss_dict["event_occurrence_precision"] = metrics["precision"]
        loss_dict["event_occurrence_recall"] = metrics["recall"]
        loss_dict["event_occurrence_f1"] = metrics["f1"]
        loss_dict["event_occurrence_num_valid"] = metrics["num_valid"]
        loss_dict["event_occurrence_num_pos"] = metrics["num_pos"]
        loss_dict["event_occurrence_num_total"] = metrics["num_total"]
        loss_dict["event_occurrence_num_neg"] = metrics["num_neg"]
        loss_dict["event_occurrence_tp"] = metrics["tp"]
        loss_dict["event_occurrence_fp"] = metrics["fp"]
        loss_dict["event_occurrence_tn"] = metrics["tn"]
        loss_dict["event_occurrence_fn"] = metrics["fn"]
        loss_dict["event_occurrence_pos_prob_mean"] = metrics["pos_prob_mean"]
        loss_dict["event_occurrence_neg_prob_mean"] = metrics["neg_prob_mean"]
        loss_dict["event_occurrence_pos_prob_sum"] = metrics["pos_prob_sum"]
        loss_dict["event_occurrence_neg_prob_sum"] = metrics["neg_prob_sum"]

    def _compute_event_occurrence_loss(
        self,
        sample: dict,
        action_hidden: torch.Tensor,
        batch_size: int,
        num_noisy: int,
        compute_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the event-done auxiliary CE loss for AR or non-AR mode."""
        metrics = {
            "active": False,
            "loss_value": 0.0,
            "loss_raw": 0.0,
            "acc": 0.0,
            "valid_ratio": 0.0,
            "pos_ratio": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_valid": 0,
            "num_pos": 0,
            "num_total": 0,
            "num_neg": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "pos_prob_mean": 0.0,
            "neg_prob_mean": 0.0,
            "pos_prob_sum": 0.0,
            "neg_prob_sum": 0.0,
        }
        zero = torch.zeros((), device=action_hidden.device, dtype=action_hidden.dtype)

        if not (
            self.event_occurrence_aux_enabled
            and self.event_occurrence_head is not None
            and "event_occurrence_label" in sample
        ):
            return zero, metrics

        event_occurrence_label = sample["event_occurrence_label"].to(device=action_hidden.device)
        # Local "blocked" flag for the event-occurrence head: when the model
        # emits `num_noisy > 0` noisy action blocks, the head operates per
        # block instead of on the whole sequence. Distinct from the
        # video-DiT-level `self.ar_mode`.
        is_blocked = num_noisy > 0

        if is_blocked:
            action_seq_len = action_hidden.shape[1]
            if action_seq_len % num_noisy != 0:
                raise ValueError(
                    "Event-done head expects action hidden length divisible by "
                    f"num_noisy, got {action_seq_len} and {num_noisy}."
                )
            action_per_block = action_seq_len // num_noisy
            block_hidden = action_hidden.view(batch_size, num_noisy, action_per_block, -1)
            event_occurrence_logits = self.event_occurrence_head(block_hidden)
            target_num_blocks = event_occurrence_logits.shape[1]
        else:
            event_occurrence_logits = self.event_occurrence_head(action_hidden)
            target_num_blocks = 1

        # Fail fast on label shape mismatch (see _normalize_event_occurrence_label).
        event_occurrence_label = self._normalize_event_occurrence_label(
            event_occurrence_label=event_occurrence_label,
            batch_size=batch_size,
            target_num_blocks=target_num_blocks,
            is_blocked=is_blocked,
        )

        loss_event_occurrence, aux_metrics = event_occurrence_ce_loss(
            event_occurrence_logits,
            event_occurrence_label,
            compute_metrics=False,
        )
        weighted = self.event_occurrence_aux_loss_weight * loss_event_occurrence
        if not compute_metrics:
            return weighted, metrics

        with torch.no_grad():
            flat_logits = event_occurrence_logits.reshape(-1, 2)
            flat_labels = event_occurrence_label.reshape(-1)
            valid_mask = flat_labels != EVENT_OCCURRENCE_IGNORE_INDEX
            valid_float = valid_mask.float()
            num_total = flat_logits.new_tensor(float(flat_labels.numel()))
            num_valid = valid_float.sum()
            predictions = flat_logits.argmax(dim=-1)
            positive_prob = torch.softmax(flat_logits.float(), dim=-1)[:, 1]
            positives = valid_mask & (flat_labels == 1)
            negatives = valid_mask & (flat_labels == 0)
            predicted_positives = predictions == 1
            num_pos = positives.float().sum()
            num_neg = negatives.float().sum()
            tp = (predicted_positives & positives).float().sum()
            fp = (predicted_positives & negatives).float().sum()
            tn = (~predicted_positives & negatives).float().sum()
            fn = (~predicted_positives & positives).float().sum()
            acc = (
                (predictions == flat_labels).float() * valid_float
            ).sum() / num_valid.clamp_min(1.0)
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            precision_recall_sum = precision + recall
            f1 = torch.where(
                precision_recall_sum > 0,
                2.0 * precision * recall / precision_recall_sum,
                precision_recall_sum,
            )
            pos_prob_sum = (positive_prob * positives.float()).sum()
            neg_prob_sum = (positive_prob * negatives.float()).sum()
            pos_prob_mean = pos_prob_sum / num_pos.clamp_min(1.0)
            neg_prob_mean = neg_prob_sum / num_neg.clamp_min(1.0)

        metrics.update({
            "active": True,
            "loss_value": weighted.detach(),
            "loss_raw": loss_event_occurrence.detach(),
            "acc": acc,
            "valid_ratio": num_valid / num_total.clamp_min(1.0),
            "pos_ratio": num_pos / num_valid.clamp_min(1.0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_valid": num_valid,
            "num_pos": num_pos,
            "num_total": num_total,
            "num_neg": num_neg,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "pos_prob_mean": pos_prob_mean,
            "neg_prob_mean": neg_prob_mean,
            "pos_prob_sum": pos_prob_sum,
            "neg_prob_sum": neg_prob_sum,
        })
        return weighted, metrics

    @staticmethod
    def _normalize_event_occurrence_label(
        event_occurrence_label: torch.Tensor,
        batch_size: int,
        target_num_blocks: int,
        is_blocked: bool,
    ) -> torch.Tensor:
        """Validate and reshape ``event_occurrence_label`` to ``[B, target_num_blocks]``.

        Accepted shapes: ``[B, K]`` (preferred), ``[K]`` or ``[1]`` for
        single-sample broadcast, ``[B]`` for non-AR. Anything else is a
        geometry / collate bug and we fail fast (no silent padding).
        """
        if target_num_blocks <= 0:
            raise ValueError(
                f"target_num_blocks must be positive, got {target_num_blocks}"
            )
        if event_occurrence_label.ndim == 1:
            if event_occurrence_label.shape[0] == target_num_blocks:
                return event_occurrence_label.unsqueeze(0).expand(batch_size, -1).contiguous()
            if event_occurrence_label.shape[0] == 1:
                return event_occurrence_label.unsqueeze(0).expand(batch_size, target_num_blocks).contiguous()
            if not is_blocked and event_occurrence_label.shape[0] == batch_size:
                return event_occurrence_label.unsqueeze(1)
            raise ValueError(
                "Event-done label has unexpected 1D shape "
                f"{tuple(event_occurrence_label.shape)}: for AR mode it must be "
                f"[num_noisy={target_num_blocks}] or [1]; for non-AR mode it "
                f"must be [B={batch_size}] or [1]."
            )
        if event_occurrence_label.ndim == 2:
            if event_occurrence_label.shape == (batch_size, target_num_blocks):
                return event_occurrence_label
            if (
                not is_blocked
                and event_occurrence_label.shape[0] == batch_size
                and event_occurrence_label.shape[1] == 1
                and target_num_blocks == 1
            ):
                return event_occurrence_label
            raise ValueError(
                "Event-done label has unexpected 2D shape "
                f"{tuple(event_occurrence_label.shape)}: expected "
                f"[{batch_size}, {target_num_blocks}]. Refusing to pad or "
                "truncate — likely a dataset geometry, mixed-child, or "
                "collate mismatch."
            )
        raise ValueError(
            "Event-done label must be 1D or 2D, got ndim="
            f"{event_occurrence_label.ndim} with shape {tuple(event_occurrence_label.shape)}."
        )

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        action_condition_start_latent_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            action_condition_start_latent_idx=action_condition_start_latent_idx,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        embeds_all = {
            "video": video_pre["tokens"],
            "action": action_pre["tokens"],
        }

        tokens_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        # ``post_dit`` returns per-route continuous flow / GUI route
        # predictions. The caller does per-head CFG.
        pred_heads = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_heads

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _prepare_ar_cycle_state(
        self,
        *,
        cycle_state: Optional[dict],
        input_image: Optional[torch.Tensor],
        input_latents: Optional[torch.Tensor],
        history_prefix_tokens: Optional[torch.Tensor],
        history_valid_mask: Optional[torch.Tensor],
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        proprio: Optional[torch.Tensor],
        proprio_dim_is_pad: Optional[torch.Tensor],
        proprio_field_valid: Optional[torch.Tensor],
        proprio_is_pad: Optional[torch.Tensor],
        caller: str,
    ) -> dict:
        if cycle_state is not None:
            cfg_in_state = bool(cycle_state.get("cfg_enabled", False))
            cfg_requested = (text_cfg_scale != 1.0)
            if cfg_in_state != cfg_requested:
                raise ValueError(
                    f"cycle_state CFG mismatch: state has cfg_enabled={cfg_in_state} "
                    f"but text_cfg_scale={text_cfg_scale} (requested CFG={cfg_requested}). "
                    "prefill_cycle* and the inference call must use the same text_cfg_scale."
                )
            if prompt is not None:
                cached_prompt = cycle_state.get("cache_static_prompt")
                if cached_prompt is None and cfg_in_state:
                    cached_prompt = cycle_state.get("cond", {}).get(
                        "cache_static_prompt"
                    )
                if cached_prompt != prompt:
                    raise RuntimeError(
                        "AR prompt changed within an active cycle. The clean KV "
                        "cache is prompt-stationary; finish/restart the cycle or "
                        "add an explicit runtime-only text update path."
                    )
            elif context is not None or context_mask is not None:
                if context is None or context_mask is None:
                    raise ValueError(
                        "Existing AR cycle requires context and context_mask together."
                    )
                branch_state = (
                    cycle_state.get("cond", {}) if cfg_in_state else cycle_state
                )
                if branch_state.get("cache_static_prompt") is not None:
                    raise RuntimeError(
                        "AR conditioning source changed from prompt to context "
                        "within an active cycle. Restart the cycle."
                    )
                cached_context = branch_state.get("clean_cache_context")
                cached_mask = branch_state.get("clean_cache_context_mask")
                if cached_context is None or cached_mask is None:
                    raise RuntimeError(
                        "Active AR cycle lacks its cache-static context metadata."
                    )
                actual_context = context.unsqueeze(0) if context.ndim == 2 else context
                actual_mask = (
                    context_mask.unsqueeze(0)
                    if context_mask.ndim == 1
                    else context_mask
                )
                same_context = (
                    tuple(actual_context.shape) == tuple(cached_context.shape)
                    and torch.equal(
                        actual_context.to(
                            device=cached_context.device,
                            dtype=cached_context.dtype,
                        ),
                        cached_context,
                    )
                )
                same_mask = (
                    tuple(actual_mask.shape) == tuple(cached_mask.shape)
                    and torch.equal(
                        actual_mask.to(
                            device=cached_mask.device,
                            dtype=cached_mask.dtype,
                        ),
                        cached_mask,
                    )
                )
                if not (same_context and same_mask):
                    raise RuntimeError(
                        "AR context changed within an active cycle. The clean KV "
                        "cache is context-stationary; restart the cycle."
                    )
            return cycle_state
        if self.ar_history_mode:
            if history_prefix_tokens is None:
                raise ValueError(
                    f"AR `{caller}` requires history_prefix_tokens when ar_history_mode=True."
                )
            if history_valid_mask is None:
                raise ValueError(
                    f"AR `{caller}` requires history_valid_mask when ar_history_mode=True."
                )
        if input_latents is not None:
            return self.prefill_cycle_from_latents(
                obs_latents=input_latents,
                history_prefix_tokens=history_prefix_tokens,
                history_valid_mask=history_valid_mask,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
                text_cfg_scale=text_cfg_scale,
                negative_context=negative_context,
                negative_context_mask=negative_context_mask,
            )
        if input_image is None:
            raise ValueError(f"AR `{caller}` requires input_image or input_latents when cycle_state is not provided.")
        return self.prefill_cycle(
            obs_video=input_image,
            history_prefix_tokens=history_prefix_tokens,
            history_valid_mask=history_valid_mask,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
            text_cfg_scale=text_cfg_scale,
            negative_context=negative_context,
            negative_context_mask=negative_context_mask,
        )

    @torch.no_grad()
    def _infer_ar_cycle_joint_step(
        self,
        *,
        prompt: Optional[str],
        input_image: Optional[torch.Tensor],
        input_latents: Optional[torch.Tensor],
        history_prefix_tokens: Optional[torch.Tensor],
        history_valid_mask: Optional[torch.Tensor],
        action_horizon: int,
        action: Optional[torch.Tensor],
        cycle_state: Optional[dict],
        action_position_offset: int,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        proprio: Optional[torch.Tensor],
        proprio_dim_is_pad: Optional[torch.Tensor],
        proprio_field_valid: Optional[torch.Tensor],
        proprio_is_pad: Optional[torch.Tensor],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
    ) -> dict[str, Any]:
        """One online AR cycle step that denoises action and video."""
        # Future GT action is not part of the AR inference contract. Keep this
        # path identical for video_expert.action_conditioned=True/False.
        action = None
        cycle_state = self._prepare_ar_cycle_state(
            cycle_state=cycle_state,
            input_image=input_image,
            input_latents=input_latents,
            history_prefix_tokens=history_prefix_tokens,
            history_valid_mask=history_valid_mask,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            negative_context=negative_context,
            negative_context_mask=negative_context_mask,
            text_cfg_scale=text_cfg_scale,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
            caller="infer_joint",
        )
        if cycle_state.get("cfg_enabled", False):
            return self._denoise_ar_cycle_step_cfg(
                cycle_state_cond=cycle_state["cond"],
                cycle_state_uncond=cycle_state["uncond"],
                text_cfg_scale=text_cfg_scale,
                action_horizon=action_horizon,
                action_position_offset=action_position_offset,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                predict_video=True,
                tiled=tiled,
            )
        return self._denoise_ar_cycle_step(
            cycle_state=cycle_state["state"] if "state" in cycle_state else cycle_state,
            action_horizon=action_horizon,
            action_position_offset=action_position_offset,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            predict_video=True,
            tiled=tiled,
        )

    @torch.no_grad()
    def _infer_ar_cycle_action_step(
        self,
        *,
        prompt: Optional[str],
        input_image: Optional[torch.Tensor],
        history_prefix_tokens: Optional[torch.Tensor],
        history_valid_mask: Optional[torch.Tensor],
        action_horizon: int,
        cycle_state: Optional[dict],
        action_position_offset: int,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        proprio: Optional[torch.Tensor],
        proprio_dim_is_pad: Optional[torch.Tensor],
        proprio_field_valid: Optional[torch.Tensor],
        proprio_is_pad: Optional[torch.Tensor],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
    ) -> dict[str, Any]:
        """One online AR cycle step that denoises action only."""
        if self.requires_joint_video_prediction:
            raise RuntimeError(
                "_infer_ar_cycle_action_step is unavailable when "
                "video_action_attention_mode=joint_bidirectional; route through "
                "infer(), which forces joint video-action denoising."
            )
        cycle_state = self._prepare_ar_cycle_state(
            cycle_state=cycle_state,
            input_image=input_image,
            input_latents=None,
            history_prefix_tokens=history_prefix_tokens,
            history_valid_mask=history_valid_mask,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            negative_context=negative_context,
            negative_context_mask=negative_context_mask,
            text_cfg_scale=text_cfg_scale,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
            caller="infer_action",
        )
        if cycle_state.get("cfg_enabled", False):
            return self._denoise_ar_cycle_step_cfg(
                cycle_state_cond=cycle_state["cond"],
                cycle_state_uncond=cycle_state["uncond"],
                text_cfg_scale=text_cfg_scale,
                action_horizon=action_horizon,
                action_position_offset=action_position_offset,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                predict_video=False,
                tiled=tiled,
            )
        return self._denoise_ar_cycle_step(
            cycle_state=cycle_state["state"] if "state" in cycle_state else cycle_state,
            action_horizon=action_horizon,
            action_position_offset=action_position_offset,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    @torch.no_grad()
    def _infer_non_ar_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()

        # Normalize input_image to [1, 3, T, H, W]
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0).unsqueeze(2)
        elif input_image.ndim == 4:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` 4D must be [1,3,H,W], got {tuple(input_image.shape)}"
                )
            input_image = input_image.unsqueeze(2)
        elif input_image.ndim == 5:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` 5D must be [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
        else:
            raise ValueError(
                f"`input_image` must be 3D, 4D, or 5D, got ndim={input_image.ndim}"
            )
        _, _, T_obs_img, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # Action conditioning must span the full prediction horizon.
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0).unsqueeze(0)
            elif proprio.ndim == 2:
                if proprio.shape[0] != 1:
                    raise ValueError(f"`proprio` 2D must be [1,D], got shape {tuple(proprio.shape)}")
                proprio = proprio.unsqueeze(1)
            elif proprio.ndim == 3:
                if proprio.shape[0] != 1:
                    raise ValueError(f"`proprio` 3D must be [1,K,D], got shape {tuple(proprio.shape)}")
            else:
                raise ValueError(f"`proprio` must be 1D, 2D, or 3D, got ndim={proprio.ndim}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        if proprio_dim_is_pad is not None:
            proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._sample_action_latents(
            action_horizon=action_horizon,
            rand_device=rand_device,
            seed=seed,
        )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        obs_latents = self._encode_video_latents(input_image, tiled=tiled)
        num_obs_latents = obs_latents.shape[2]
        latents_video[:, :, 0:num_obs_latents] = obs_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # Resolve contexts
        use_cfg = (text_cfg_scale != 1.0)
        if use_cfg:
            context_cond, context_mask_cond, context_uncond, context_mask_uncond = self._resolve_cfg_contexts(
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                negative_context=negative_context,
                negative_context_mask=negative_context_mask,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )
        else:
            # Standard single branch
            use_prompt = prompt is not None
            use_context = context is not None or context_mask is not None
            if use_prompt and use_context:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
            if not use_prompt and not use_context:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

            if use_prompt:
                context_cond, context_mask_cond = self.encode_prompt(prompt)
            else:
                if context is None or context_mask is None:
                    raise ValueError("`context` and `context_mask` must be both provided together.")
                if context.ndim == 2:
                    context = context.unsqueeze(0)
                if context_mask.ndim == 1:
                    context_mask = context_mask.unsqueeze(0)
                if context.ndim != 3 or context_mask.ndim != 2:
                    raise ValueError(
                        f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                    )
                context_cond = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                context_mask_cond = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if proprio is not None:
                context_cond, context_mask_cond = self._append_proprio_to_context(
                    context=context_cond,
                    context_mask=context_mask_cond,
                    proprio=proprio,
                    proprio_dim_is_pad=proprio_dim_is_pad,
                    proprio_field_valid=proprio_field_valid,
                    proprio_is_pad=proprio_is_pad,
                )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        if self.coupled_video_action_timesteps:
            infer_timesteps_action, infer_deltas_action = self._build_coupled_action_infer_schedule(
                infer_timesteps_video,
                action_dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        else:
            infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        # Track the final denoise-step route outputs for route-aware
        # action postprocessing.
        last_route_mask: Optional[torch.Tensor] = None
        last_route_logit: Optional[torch.Tensor] = None
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if use_cfg:
                pred_video_cond, pred_heads_cond = self._predict_joint_noise(
                    latents_video=latents_video,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=context_cond,
                    context_mask=context_mask_cond,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=action,
                    action_condition_start_latent_idx=num_obs_latents,
                        )
                pred_video_uncond, pred_heads_uncond = self._predict_joint_noise(
                    latents_video=latents_video,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=context_uncond,
                    context_mask=context_mask_uncond,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=action,
                    action_condition_start_latent_idx=num_obs_latents,
                        )
                pred_video = pred_video_cond + (text_cfg_scale - 1.0) * (pred_video_cond - pred_video_uncond)
                # Per-head CFG combine: GUI route uses cond branch
                # only; continuous + discrete are CFG-combined.
                routed_heads = self.action_expert.combine_cfg_heads(
                    pred_heads_cond, pred_heads_uncond, text_cfg_scale
                )
                pred_action = routed_heads["continuous"]
            else:
                pred_video_posi, pred_heads_posi = self._predict_joint_noise(
                    latents_video=latents_video,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=context_cond,
                    context_mask=context_mask_cond,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=action,
                    action_condition_start_latent_idx=num_obs_latents,
                        )
                pred_video = pred_video_posi
                routed_heads = self.action_expert.route_cond_heads(pred_heads_posi)
                pred_action = routed_heads["continuous"]

            if routed_heads.get("route_mask") is not None:
                last_route_mask = routed_heads["route_mask"]
                last_route_logit = routed_heads.get("route_logit")

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self._step_continuous(latents_action, pred_action, step_delta_action)
            latents_video[:, :, 0:num_obs_latents] = obs_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)

        result: Dict[str, Any] = {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }
        if self.action_head_gui_head_enabled and last_route_mask is not None:
            result["action_route_mask"] = last_route_mask[0].detach().to(
                device="cpu", dtype=torch.bool
            )
        if last_route_logit is not None:
            result["action_route_logit"] = last_route_logit[0].detach().to(
                device="cpu", dtype=torch.float32
            )
        return result


    @torch.no_grad()
    def _infer_non_ar_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Non-AR action-only denoise (no video branch)."""
        self.eval()
        if self.requires_joint_video_prediction:
            raise RuntimeError(
                "_infer_non_ar_action is unavailable when "
                "video_action_attention_mode=joint_bidirectional; route through "
                "infer(), which forces joint video-action denoising."
            )
        if input_image is None:
            raise ValueError("`infer_action` requires input_image for non-AR models.")
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_k_frames_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_k_frames_causal'`."
            )

        # Normalize input_image to [1, 3, T, H, W]
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0).unsqueeze(2)
        elif input_image.ndim == 4:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` 4D must be [1,3,H,W], got {tuple(input_image.shape)}"
                )
            input_image = input_image.unsqueeze(2)
        elif input_image.ndim == 5:
            if input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` 5D must be [1,3,T,H,W], got {tuple(input_image.shape)}"
                )
        else:
            raise ValueError(
                f"`input_image` must be 3D, 4D, or 5D, got ndim={input_image.ndim}"
            )
        _, _, T_obs, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )

        # Normalize proprio to [1, K, D]
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0).unsqueeze(0)
            elif proprio.ndim == 2:
                if proprio.shape[0] != 1:
                    raise ValueError(f"`proprio` 2D must be [1,D], got shape {tuple(proprio.shape)}")
                proprio = proprio.unsqueeze(1)
            elif proprio.ndim == 3:
                if proprio.shape[0] != 1:
                    raise ValueError(f"`proprio` 3D must be [1,K,D], got shape {tuple(proprio.shape)}")
            else:
                raise ValueError(f"`proprio` must be 1D, 2D, or 3D, got ndim={proprio.ndim}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        if proprio_dim_is_pad is not None:
            proprio_dim_is_pad = proprio_dim_is_pad.to(device=self.device, dtype=torch.bool)

        latents_action = self._sample_action_latents(
            action_horizon=action_horizon,
            rand_device=rand_device,
            seed=seed,
        )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        obs_latents = self._encode_video_latents(input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # Resolve contexts
        use_cfg = (text_cfg_scale != 1.0)
        if use_cfg:
            context_cond, context_mask_cond, context_uncond, context_mask_uncond = self._resolve_cfg_contexts(
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                negative_context=negative_context,
                negative_context_mask=negative_context_mask,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
            )
        else:
            # Standard single branch
            use_prompt = prompt is not None
            use_context = context is not None or context_mask is not None
            if use_prompt and use_context:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
            if not use_prompt and not use_context:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

            if use_prompt:
                context_cond, context_mask_cond = self.encode_prompt(prompt)
            else:
                if context is None or context_mask is None:
                    raise ValueError("`context` and `context_mask` must be both provided together.")
                if context.ndim == 2:
                    context = context.unsqueeze(0)
                if context_mask.ndim == 1:
                    context_mask = context_mask.unsqueeze(0)
                if context.ndim != 3 or context_mask.ndim != 2:
                    raise ValueError(
                        f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                    )
                context_cond = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                context_mask_cond = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if proprio is not None:
                context_cond, context_mask_cond = self._append_proprio_to_context(
                    context=context_cond,
                    context_mask=context_mask_cond,
                    proprio=proprio,
                    proprio_dim_is_pad=proprio_dim_is_pad,
                    proprio_field_valid=proprio_field_valid,
                    proprio_is_pad=proprio_is_pad,
                )

        timestep_video = torch.zeros(
            (obs_latents.shape[0],),
            dtype=obs_latents.dtype,
            device=self.device,
        )

        if use_cfg:
            video_pre_cond = self.video_expert.pre_dit(
                x=obs_latents,
                timestep=timestep_video,
                context=context_cond,
                context_mask=context_mask_cond,
                action=None,
                action_condition_start_latent_idx=obs_latents.shape[2],
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_pre_uncond = self.video_expert.pre_dit(
                x=obs_latents,
                timestep=timestep_video,
                context=context_uncond,
                context_mask=context_mask_uncond,
                action=None,
                action_condition_start_latent_idx=obs_latents.shape[2],
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre_cond["tokens"].shape[1])
        else:
            video_pre_cond = self.video_expert.pre_dit(
                x=obs_latents,
                timestep=timestep_video,
                context=context_cond,
                context_mask=context_mask_cond,
                action=None,
                action_condition_start_latent_idx=obs_latents.shape[2],
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre_cond["tokens"].shape[1])

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=video_pre_cond["tokens"].device,
        )

        if use_cfg:
            video_kv_cache_cond = self.mot.prefill_video_cache(
                video_tokens=video_pre_cond["tokens"],
                video_freqs=video_pre_cond["freqs"],
                video_t_mod=video_pre_cond["t_mod"],
                video_context_payload={
                    "context": video_pre_cond["context"],
                    "mask": video_pre_cond["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
            video_kv_cache_uncond = self.mot.prefill_video_cache(
                video_tokens=video_pre_uncond["tokens"],
                video_freqs=video_pre_uncond["freqs"],
                video_t_mod=video_pre_uncond["t_mod"],
                video_context_payload={
                    "context": video_pre_uncond["context"],
                    "mask": video_pre_uncond["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
        else:
            video_kv_cache_cond = self.mot.prefill_video_cache(
                video_tokens=video_pre_cond["tokens"],
                video_freqs=video_pre_cond["freqs"],
                video_t_mod=video_pre_cond["t_mod"],
                video_context_payload={
                    "context": video_pre_cond["context"],
                    "mask": video_pre_cond["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )


        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        last_route_mask: Optional[torch.Tensor] = None
        last_route_logit: Optional[torch.Tensor] = None
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if use_cfg:
                pred_heads_cond = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context_cond,
                    context_mask=context_mask_cond,
                    video_kv_cache=video_kv_cache_cond,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                        )
                pred_heads_uncond = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context_uncond,
                    context_mask=context_mask_uncond,
                    video_kv_cache=video_kv_cache_uncond,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                        )
                # Per-head CFG combine.
                routed_heads = self.action_expert.combine_cfg_heads(
                    pred_heads_cond, pred_heads_uncond, text_cfg_scale
                )
                pred_action = routed_heads["continuous"]
                if routed_heads.get("route_mask") is not None:
                    last_route_mask = routed_heads["route_mask"]
                    last_route_logit = routed_heads.get("route_logit")
            else:
                pred_heads_posi = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context_cond,
                    context_mask=context_mask_cond,
                    video_kv_cache=video_kv_cache_cond,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                        )
                # Route the single conditional branch with the same helper as
                # every other inference path.
                routed_heads = self.action_expert.route_cond_heads(pred_heads_posi)
                pred_action = routed_heads["continuous"]
                if routed_heads.get("route_mask") is not None:
                    last_route_mask = routed_heads["route_mask"]
                    last_route_logit = routed_heads.get("route_logit")
            latents_action = self._step_continuous(latents_action, pred_action, step_delta_action)

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if self.action_head_gui_head_enabled and last_route_mask is not None:
            result["action_route_mask"] = last_route_mask[0].detach().to(
                device="cpu", dtype=torch.bool
            )
        if last_route_logit is not None:
            result["action_route_logit"] = last_route_logit[0].detach().to(
                device="cpu", dtype=torch.float32
            )
        return result


    @torch.no_grad()
    def infer(
        self,
        *,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        input_latents: Optional[torch.Tensor] = None,
        num_video_frames: Optional[int] = None,
        num_action_steps: int,
        action: Optional[torch.Tensor] = None,
        history_prefix_tokens: Optional[torch.Tensor] = None,
        history_valid_mask: Optional[torch.Tensor] = None,
        cycle_state: Optional[dict] = None,
        action_position_offset: int = 0,
        proprio: Optional[torch.Tensor] = None,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
        proprio_field_valid: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        ar_eval_clip: bool = False,
    ) -> dict[str, Any]:
        """Unified inference entry point.

        ``num_action_steps`` is the number of action steps requested from this
        call. During online block-cycle control it equals ``prediction_horizon``;
        AR eval-clip calls may request a longer stitched sequence.

        Behavior table:

        When ``video_action_attention_mode=joint_bidirectional``, an omitted
        ``num_video_frames`` is replaced by the model-derived sampled video
        frame count before routing,
        so both AR and non-AR calls always use a joint sampler.

        - ``ar_mode=False``:
            * ``cycle_state`` must be ``None`` (else ``ValueError``).
            * ``num_video_frames`` provided → ``_infer_non_ar_joint``.
            * ``num_video_frames=None`` → ``_infer_non_ar_action``.
        - ``ar_mode=True`` and ``ar_eval_clip=True``:
            * ``num_video_frames`` is required (else ``ValueError``).
            * ``cycle_state`` must be ``None`` (else ``ValueError``).
            * Delegates to ``_infer_ar_eval_clip_joint``.
        - ``ar_mode=True`` and ``ar_eval_clip=False``:
            * ``num_video_frames`` provided → ``_infer_ar_cycle_joint_step``.
            * ``num_video_frames=None`` → ``_infer_ar_cycle_action_step``.
        """
        self.eval()
        if (
            self.requires_joint_video_prediction
            and num_video_frames is None
        ):
            # In joint-bidirectional mode predicted-video tokens are part of
            # the action denoising graph. Callers may still request only an
            # action result, but they cannot disable internal video denoising.
            video_ratio = int(self.action_video_freq_ratio)
            if video_ratio <= 0:
                raise ValueError(
                    "action_video_freq_ratio must be positive for joint inference, "
                    f"got {video_ratio}."
                )
            num_video_frames = (int(self.num_frames) - 1) // video_ratio + 1

        if self.ar_mode:
            if ar_eval_clip:
                if num_video_frames is None:
                    raise ValueError(
                        "AR eval clip `infer` requires `num_video_frames`."
                    )
                if cycle_state is not None:
                    raise ValueError(
                        "AR eval clip `infer` must not carry `cycle_state`."
                    )
                return self._infer_ar_eval_clip_joint(
                    input_latents=input_latents,
                    history_prefix_tokens=history_prefix_tokens,
                    history_valid_mask=history_valid_mask,
                    action_horizon=num_action_steps,
                    prompt=prompt,
                    context=context,
                    context_mask=context_mask,
                    negative_context=negative_context,
                    negative_context_mask=negative_context_mask,
                    text_cfg_scale=text_cfg_scale,
                    proprio=proprio,
                    proprio_dim_is_pad=proprio_dim_is_pad,
                    proprio_field_valid=proprio_field_valid,
                    proprio_is_pad=proprio_is_pad,
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    seed=seed,
                    rand_device=rand_device,
                    tiled=tiled,
                )
            if num_video_frames is None:
                return self._infer_ar_cycle_action_step(
                    prompt=prompt,
                    input_image=input_image,
                    history_prefix_tokens=history_prefix_tokens,
                    history_valid_mask=history_valid_mask,
                    action_horizon=num_action_steps,
                    cycle_state=cycle_state,
                    action_position_offset=action_position_offset,
                    context=context,
                    context_mask=context_mask,
                    negative_context=negative_context,
                    negative_context_mask=negative_context_mask,
                    text_cfg_scale=text_cfg_scale,
                    proprio=proprio,
                    proprio_dim_is_pad=proprio_dim_is_pad,
                    proprio_field_valid=proprio_field_valid,
                    proprio_is_pad=proprio_is_pad,
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    seed=seed,
                    rand_device=rand_device,
                    tiled=tiled,
                )
            return self._infer_ar_cycle_joint_step(
                prompt=prompt,
                input_image=input_image,
                input_latents=input_latents,
                history_prefix_tokens=history_prefix_tokens,
                history_valid_mask=history_valid_mask,
                action_horizon=num_action_steps,
                action=action,
                cycle_state=cycle_state,
                action_position_offset=action_position_offset,
                context=context,
                context_mask=context_mask,
                negative_context=negative_context,
                negative_context_mask=negative_context_mask,
                text_cfg_scale=text_cfg_scale,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )

        # non-AR path
        if cycle_state is not None:
            raise ValueError("non-AR `infer` does not accept `cycle_state`.")
        if num_video_frames is not None:
            if input_image is None:
                raise ValueError("non-AR joint `infer` requires `input_image`.")
            return self._infer_non_ar_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=num_action_steps,
                action=action,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
                context=context,
                context_mask=context_mask,
                negative_context=negative_context,
                negative_context_mask=negative_context_mask,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )
        if input_image is None:
            raise ValueError("non-AR action `infer` requires `input_image`.")
        return self._infer_non_ar_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=num_action_steps,
            proprio=proprio,
            proprio_dim_is_pad=proprio_dim_is_pad,
            proprio_field_valid=proprio_field_valid,
            proprio_is_pad=proprio_is_pad,
            context=context,
            context_mask=context_mask,
            negative_context=negative_context,
            negative_context_mask=negative_context_mask,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
