
"""GameWAMInference — runtime for inference / rollout.

Instances are constructed via ``runtime.build_inference(cfg)``, which
resolves the model, processor, checkpoint, and dataset statistics from
``cfg`` and hands them to the constructor. The runtime then owns
normalizer initialization and the inference-time observation buffers.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from .envs.base import GameObservation
from .models.history.history_module import init_history_state as _init_history_state
from .utils.video_io import resize_frame_to_size, save_mp4

logger = logging.getLogger(__name__)


class GameWAMInference:
    """Inference runtime for GameWAM.

    Loads a trained GameWAM model and processor (constructed by
    ``runtime.build_inference``), maintains obs / proprio buffers,
    executes AR-history (or non-AR) chunked action inference, and
    dispatches actions one-at-a-time via an action queue.
    """

    def __init__(
        self,
        *,
        cfg,
        model,
        processor,
        dataset_stats: dict,
        prompt_template: str,
        device: str,
        model_dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.model = model
        # ``processor.eval()`` is idempotent — safe to call even if the
        # caller already called it.
        self.processor = processor.eval()
        self.dataset_stats = dataset_stats
        self.prompt_template = str(prompt_template)
        self.device = str(device)
        self.model_dtype = model_dtype

        self.processor.set_normalizer_from_stats(dataset_stats)
        self.model.prompt_template = self.prompt_template

        self._init_inference_config()
        processor_image_size = self._processor_image_size()
        if self.image_size != processor_image_size:
            raise ValueError(
                "`inference.image_size` must match the image shape used by the "
                "training processor: "
                f"inference={self.image_size}, processor={processor_image_size}."
            )
        self._init_buffers()

        logger.info(
            "GameWAMInference initialized | prediction_horizon=%d | "
            "execution_horizon=%d | num_obs_frames=%d | ratio=%d | "
            "effective_obs_frames=%d | text_cfg_scale=%.2f",
            self.prediction_horizon,
            self.execution_horizon,
            self.num_obs_frames,
            self.action_video_freq_ratio,
            self.effective_obs_frames,
            self.text_cfg_scale,
        )

    # Init helpers
    def _init_inference_config(self) -> None:
        inference_cfg = self.cfg.inference if self.cfg is not None else None
        data_cfg = (
            self.cfg.data.common
            if self.cfg is not None
            and isinstance(self.cfg, DictConfig)
            and "data" in self.cfg
            and self.cfg.data is not None
            and "common" in self.cfg.data
            and self.cfg.data.common is not None
            else None
        )

        def _get_inference(key, default=None):
            if inference_cfg is None:
                return default
            return inference_cfg.get(key, default)

        def _get_data(key, default=None):
            if data_cfg is None:
                return default
            return data_cfg.get(key, default)

        # Detect mode so we can skip rollout-only validation for offline demos
        # (which never go through get_action / chunk inference).
        mode = str(_get_inference("mode", "rollout") or "rollout").lower()

        if inference_cfg is not None:
            for removed_key in ("replan_steps", "action_horizon", "effective_action_horizon"):
                if removed_key in inference_cfg:
                    raise ValueError(
                        f"`inference.{removed_key}` has been removed. Use "
                        "`data.common.prediction_horizon` and "
                        "`data.common.execution_horizon` instead."
                    )

        # Sampling / rollout knobs (from cfg.inference)
        self.num_inference_steps = int(_get_inference("num_inference_steps", 10))
        self.sigma_shift = (
            None
            if _get_inference("sigma_shift") is None
            else float(_get_inference("sigma_shift"))
        )
        self.seed = _get_inference("seed")
        self.text_cfg_scale = float(_get_inference("text_cfg_scale", 1.0))
        self.rand_device = str(_get_inference("rand_device", "cpu"))
        self.tiled = bool(_get_inference("tiled", False))
        requested_eval_predict_video = bool(_get_inference("eval_predict_video", False))
        self.eval_predict_video = bool(
            requested_eval_predict_video
            or getattr(self.model, "requires_joint_video_prediction", False)
        )
        if self.eval_predict_video and not requested_eval_predict_video:
            logger.info(
                "Forcing inference.eval_predict_video=true because "
                "video_action_attention_mode=joint_bidirectional."
            )

        image_size = _get_inference("image_size", [224, 224])
        # Accept plain list / tuple as well as OmegaConf ListConfig.
        if hasattr(image_size, "__iter__") and not isinstance(image_size, str):
            image_size = (int(image_size[0]), int(image_size[1]))
        else:
            raise ValueError(f"`image_size` must be [H, W] or (H, W), got {image_size!r}")
        self.image_size = image_size

        # Geometry knobs are owned by ``cfg.data.common``: ``num_obs_frames``,
        # ``frame_stride``, and ``action_video_freq_ratio`` are dataset-shape
        # properties shared with the training data pipeline, which keeps the
        # rollout / sampling config orthogonal to the data pipeline config.
        num_frames_data = int(_get_data("num_frames", 33))
        num_obs_frames = int(_get_data("num_obs_frames", 1))
        frame_stride = int(_get_data("frame_stride", 1))
        action_video_freq_ratio = int(_get_data("action_video_freq_ratio", 1))
        self.num_obs_frames = num_obs_frames
        self.frame_stride = frame_stride
        self.action_video_freq_ratio = action_video_freq_ratio

        # ``num_video_frames`` is the per-action video count fed to the video
        # expert in non-AR joint inference. It is purely a function of the
        # dataset shape and the frequency ratio — derive it here.
        self._num_video_frames = (
            (num_frames_data - 1) // action_video_freq_ratio + 1
        )

        self.prediction_horizon = int(_get_data("prediction_horizon", 1 if mode == "offline" else 16))
        self.execution_horizon = int(_get_data("execution_horizon", self.prediction_horizon))
        if mode != "offline":
            if self.prediction_horizon <= 0:
                raise ValueError(f"prediction_horizon must be positive, got {self.prediction_horizon}")
            if self.execution_horizon <= 0:
                raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
            if self.execution_horizon > self.prediction_horizon:
                raise ValueError(
                    "execution_horizon must not exceed prediction_horizon, "
                    f"got {self.execution_horizon} > {self.prediction_horizon}"
                )
            if getattr(self.model, "ar_mode", False):
                if self.prediction_horizon % self.execution_horizon != 0:
                    raise ValueError(
                        "AR prediction_horizon must be divisible by execution_horizon, "
                        f"got {self.prediction_horizon} and {self.execution_horizon}"
                    )

        self.effective_obs_frames = (
            (self.num_obs_frames - 1) // self.action_video_freq_ratio + 1
        )
        committed_future_horizon = (num_frames_data - num_obs_frames) * frame_stride
        if mode != "offline" and getattr(self.model, "ar_mode", False):
            geometry = self.model.ar_geometry
            if geometry is None:
                raise ValueError("AR inference requires model.ar_geometry.")
            if self.execution_horizon != int(geometry.raw_steps_per_latent):
                raise ValueError(
                    "AR execution_horizon must equal raw_steps_per_latent, "
                    f"got execution_horizon={self.execution_horizon}, "
                    f"raw_steps_per_latent={geometry.raw_steps_per_latent}"
                )
            if committed_future_horizon % self.execution_horizon != 0:
                raise ValueError(
                    "AR committed future horizon must be divisible by execution_horizon, "
                    f"got {committed_future_horizon} and {self.execution_horizon}"
                )
        elif mode != "offline" and committed_future_horizon != self.prediction_horizon:
            raise ValueError(
                "Non-AR geometry must satisfy "
                "(num_frames - num_obs_frames) * frame_stride == prediction_horizon, "
                f"got {committed_future_horizon} != {self.prediction_horizon}"
            )
        self.obs_action_steps = (
            self.num_obs_frames - 1
        ) * self.frame_stride
        self.num_obs_proprio = self.obs_action_steps + 1

    def _init_buffers(self) -> None:
        obs_raw_span = (self.num_obs_frames - 1) * self.frame_stride + 1
        self._obs_frame_buffer: deque[tuple[int, np.ndarray]] = deque(
            maxlen=max(obs_raw_span, 1)
        )
        self._proprio_buffer: deque[tuple[int, np.ndarray]] = deque(
            maxlen=max(obs_raw_span, 1)
        )

        self._committed_action_queue: deque[np.ndarray] = deque()
        self._last_model_proprio: Optional[np.ndarray] = None
        self._last_model_proprio_field_valid: Optional[np.ndarray] = None
        self.episode_count = 0
        self.step_count = 0
        self._cycle_raw_records: dict[int, np.ndarray] = {}
        self._cycle_proprio_records: dict[int, np.ndarray] = {}
        self._cycle_prefill_start_raw: Optional[int] = None
        self._planned_blocks_in_cycle = 0
        self._cycle_state = None
        self._pending_kv_append = False
        self._pending_cycle_finish = False
        self._history_state = None
        if (
            getattr(self.model, "ar_history_mode", False)
            and self.model.history_module is not None
        ):
            self._history_state = _init_history_state(
                history_config=self.model.history_config,
                hidden_dim=self.model.video_expert.hidden_dim,
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )

        self.inference_logger: Optional[Callable[[Dict[str, Any]], None]] = None
        self._inference_seed_index = 0
        self._cycle_sampling_seed: Optional[int] = None

    # Preprocessing helpers
    @property
    def _expected_raw_state_dim(self) -> int:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one state key in shape_meta['state'].")
        raw_shape = state_meta[0]["raw_shape"]
        return int(
            raw_shape[-1] if isinstance(raw_shape, (list, tuple)) else raw_shape
        )

    def _processor_image_size(self) -> tuple[int, int]:
        image_meta = self.processor.shape_meta.get("images")
        if image_meta is None or len(image_meta) != 1:
            raise ValueError(
                "Online inference requires exactly one image entry in "
                "processor.shape_meta['images']."
            )
        shape = image_meta[0].get("shape")
        if shape is None or len(shape) != 3:
            raise ValueError(
                "processor.shape_meta['images'][0].shape must be [C,H,W], "
                f"got {shape!r}."
            )
        channels, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
        if channels != 3 or height <= 0 or width <= 0:
            raise ValueError(
                "Online inference requires a positive RGB [3,H,W] processor "
                f"image shape, got {(channels, height, width)}."
            )
        return height, width

    def _normalize_state_with_validity(
        self, state: np.ndarray, field_valid: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and normalize a proprioceptive state vector."""
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        raw_shape = state_meta[0]["raw_shape"]
        expected_dim = int(raw_shape[-1] if isinstance(raw_shape, (list, tuple)) else raw_shape)
        state = np.asarray(state)
        if state.shape[-1] != expected_dim:
            raise ValueError(
                "proprio raw state width mismatch: "
                f"runtime={state.shape[-1]}, processor={expected_dim}."
            )
        encoder_type = str(
            getattr(self.processor, "proprio_encoder_type", "generic_linear")
        )
        if field_valid is None and encoder_type == "minecraft_grouped_cursor":
            raise ValueError(
                "Minecraft cursor inference requires proprio_field_valid; an all-valid "
                "fallback would make missing cursor state ambiguous."
            )
        if field_valid is None:
            field_valid = np.ones(state.shape[-1], dtype=np.bool_)
        else:
            field_valid = np.asarray(field_valid)
            if field_valid.shape[-1] != expected_dim:
                raise ValueError(
                    "proprio_field_valid width mismatch: "
                    f"runtime={field_valid.shape[-1]}, processor={expected_dim}."
                )
        state_batch = {
            "state": {
                state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            },
            "state_field_valid": {
                state_key: torch.as_tensor(
                    field_valid, dtype=torch.bool
                ).unsqueeze(0)
            },
        }
        state_batch = self.processor.action_state_transform(state_batch)
        effective_valid = state_batch["state_field_valid"]
        if isinstance(effective_valid, dict):
            effective_valid = effective_valid[state_key]
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key], effective_valid

    def _normalize_state(
        self, state: np.ndarray, field_valid: np.ndarray | None = None,
    ) -> torch.Tensor:
        normalized, _ = self._normalize_state_with_validity(
            state, field_valid=field_valid
        )
        return normalized

    def _runtime_state_debug(self, obs: GameObservation) -> dict:
        """Return the small raw-state payload needed to audit live rollouts."""
        info = obs.info if isinstance(obs.info, dict) else {}
        model_proprio = getattr(self, "_last_model_proprio", None)
        model_field_valid = getattr(
            self, "_last_model_proprio_field_valid", None
        )
        return {
            "proprio_raw": (
                np.asarray(obs.proprio).tolist()
                if obs.proprio is not None else None
            ),
            "proprio_field_valid": (
                np.asarray(obs.proprio_field_valid, dtype=np.bool_).tolist()
                if obs.proprio_field_valid is not None else None
            ),
            "model_proprio": (
                model_proprio.tolist() if model_proprio is not None else None
            ),
            "model_proprio_field_valid": (
                model_field_valid.tolist()
                if model_field_valid is not None else None
            ),
            "canonical_info": {
                key: info.get(key)
                for key in (
                    "pitch",
                    "yaw",
                    "current_hotbar",
                    "is_gui_open",
                    "cursor_x",
                    "cursor_y",
                )
                if key in info
            },
            "minestudio_player_pos": info.get("player_pos"),
            "minestudio_location_stats": info.get("location_stats"),
        }

    def _action_stats(self, action: torch.Tensor) -> dict:
        action = action.detach().to(dtype=torch.float32, device="cpu")
        if action.ndim == 2:
            action = action.unsqueeze(0)
        flat = action.reshape(-1, action.shape[-1])
        return {
            "shape": list(action.shape),
            "mean": flat.mean(dim=0).tolist(),
            "std": flat.std(dim=0, unbiased=False).tolist(),
            "min": flat.min(dim=0).values.tolist(),
            "max": flat.max(dim=0).values.tolist(),
            "ge_0_5_ratio": (flat >= 0.5).to(torch.float32).mean(dim=0).tolist(),
        }

    @staticmethod
    def _camera_values(action: torch.Tensor) -> list:
        action = action.detach().to(dtype=torch.float32, device="cpu")
        if action.shape[-1] < 2:
            return []
        return action[..., :2].tolist()

    def _postprocess_action_with_debug(
        self,
        action: torch.Tensor,
        action_route_mask: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, dict]:
        """Denormalize predicted action tensor and expose intermediate stats.

        When ``action_route_mask`` is provided and the processor is
        configured with route-aware action normalization, the action is
        denormalized per-token with the gameplay or GUI normalizer for
        continuous dims. Discrete dims are still
        produced by the model in ``{-1, +1}`` and map back to
        ``{0, 1}`` through the global normalizer.
        """
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected [B,T,D] action, got {tuple(action.shape)}")

        action_cpu = action.to(dtype=torch.float32, device="cpu")
        state_cpu = torch.zeros(
            action_cpu.shape[0],
            action_cpu.shape[1],
            self.processor.proprio_output_dim,
            dtype=torch.float32,
        )
        data = {
            "action": action_cpu.clone(),
            "state": state_cpu,
        }
        data = self.processor.action_state_merger.backward(data)
        action_key = self.processor.shape_meta["action"][0]["key"]
        debug = {
            "raw_model_action": self._action_stats(action_cpu),
            "after_action_state_merger": self._action_stats(data["action"][action_key]),
            "camera_values": {
                "normalized_model": self._camera_values(action_cpu),
                "after_action_state_merger": self._camera_values(
                    data["action"][action_key]
                ),
            },
        }

        # Route-aware backward: if the model returned a route mask
        # AND the processor was configured with route-aware
        # normalization, route-aware denormalize the action continuous
        # dims. Otherwise fall back to the global backward.
        route_aware = bool(
            getattr(self.processor, "route_aware_action_normalization", False)
        )
        if route_aware:
            if action_route_mask is None:
                raise ValueError(
                    "`route_aware_action_normalization=True` but "
                    "the model output did not include `action_route_mask`. "
                    "Silent fallback to global denormalization would produce "
                    "a wrong-space raw action."
                )
            # Validate mask shape against action shape.
            am = action_route_mask
            if am.ndim == 2:
                am = am.unsqueeze(0)
            B, T, _ = action_cpu.shape
            expected_shape = (B, T, 1)
            if tuple(am.shape) != expected_shape:
                raise ValueError(
                    f"`action_route_mask` shape {tuple(am.shape)} incompatible "
                    f"with action shape {tuple(action_cpu.shape)}; expected {expected_shape}."
                )
            am = am.to(dtype=torch.bool, device=action_cpu.device)
            data["action"] = self.processor.normalizer.backward_action_by_route(
                data["action"], am,
                self.processor.route_norm, self.processor.route_norm,
            )
        else:
            data = self.processor.normalizer.backward(data)
        debug["after_normalizer_backward"] = self._action_stats(data["action"][action_key])
        debug["camera_values"]["after_normalizer_backward"] = self._camera_values(
            data["action"][action_key]
        )

        if self.processor.action_state_transforms is not None:
            for trans in reversed(self.processor.action_state_transforms):
                data = trans.backward(data)
        debug["after_action_transform_backward"] = self._action_stats(data["action"][action_key])
        debug["camera_values"]["after_action_transform_backward"] = self._camera_values(
            data["action"][action_key]
        )
        returned_action = data["action"][action_key].numpy()
        # stage3 and returned_action come from the same
        # ``data["action"][action_key]`` view, so they must be equal.
        # Enforce the invariant here so regressions are caught at the source.
        if returned_action.shape[-1] >= 2:
            stage3_camera = np.asarray(
                debug["camera_values"]["after_action_transform_backward"],
                dtype=np.float32,
            )
            if not np.array_equal(stage3_camera, returned_action[..., :2]):
                raise RuntimeError(
                    "stage3 vs returned_action mismatch: instrumentation invariant "
                    "broken. The postprocess pipeline must not silently transform "
                    "the action after `after_action_transform_backward`."
                )
        return returned_action, debug

    def _build_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert game screenshot [H,W,3] uint8 to model input tensor.

        Returns: [1, C, H, W] float tensor in [-1, 1].
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        target_h, target_w = self.image_size
        pil_img = resize_frame_to_size(image, (target_w, target_h))
        # PIL-backed np.asarray() may be read-only; torch.from_numpy() warns
        # because writes would then be undefined.  Materialize writable RGB
        # storage before moving it to the model device.
        image = np.array(pil_img, dtype=np.uint8, copy=True)

        tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.model.device, dtype=self.model.torch_dtype)
        )
        tensor = tensor * (2.0 / 255.0) - 1.0
        return tensor

    def _build_task_string(
        self, fine_task: str, coarse_task: Optional[str] = None
    ) -> str:
        """Construct the raw task string that goes into the prompt_template."""
        if not coarse_task or coarse_task == fine_task:
            return fine_task
        return f"[High]: {coarse_task}, [Low]: {fine_task}"

    def _next_inference_seed(self) -> Optional[int]:
        """Return the deterministic seed for the next policy replan.

        ``None`` preserves stochastic/global-RNG behaviour.  For an integer
        base seed, each replan gets a keyed, well-mixed sub-seed.  Resetting an
        episode resets the sequence, so fixed-seed rollouts remain exactly
        reproducible without replaying identical diffusion noise within the
        episode.  Mixing also prevents adjacent base seeds from producing the
        same sub-seed sequence shifted by one replan.
        """
        if self.seed is None:
            return None
        # SplitMix64 is used only as a deterministic integer mixer; the model's
        # torch.Generator remains the source of diffusion noise.  Keep the
        # result in the non-negative signed-64 range accepted uniformly by
        # torch.Generator.manual_seed on CPU and CUDA.
        mask64 = (1 << 64) - 1
        index = int(self._inference_seed_index)
        mixed = (
            (int(self.seed) & mask64)
            + (index + 1) * 0x9E3779B97F4A7C15
        ) & mask64
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask64
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask64
        mixed ^= mixed >> 31
        seed = mixed & ((1 << 63) - 1)
        self._inference_seed_index += 1
        return int(seed)

    def _random_inference_seed(self) -> int:
        return int(
            torch.empty((), dtype=torch.int64).random_(0, (1 << 63) - 1).item()
        )

    def _cycle_action_noise_shared_enabled(self) -> bool:
        return bool(
            getattr(self.model, "action_noise_share_across_blocks", False)
        )

    def _next_ar_cycle_sampling_seed(self) -> Optional[int]:
        if not self._cycle_action_noise_shared_enabled():
            return self._next_inference_seed()
        if self._planned_blocks_in_cycle == 0 or self._cycle_sampling_seed is None:
            seed = self._next_inference_seed()
            if seed is None:
                seed = self._random_inference_seed()
            self._cycle_sampling_seed = int(seed)
        return int(self._cycle_sampling_seed)

    # Obs / proprio tensor builders
    def _build_obs_video_tensor(self) -> torch.Tensor:
        """Build [1, 3, T, H, W] obs video from raw-indexed frame buffer."""
        records = self._prepare_initial_obs_records_for_cycle()
        return self._build_obs_video_tensor_from_records(records)

    def _prepare_initial_records(
        self,
        buffer: deque[tuple[int, np.ndarray]],
        *,
        target_count: int,
        value_name: str,
    ) -> list[tuple[int, np.ndarray]]:
        records = list(buffer)
        if not records:
            raise RuntimeError(
                f"{value_name} buffer is empty — call get_action at least once before inference."
            )
        by_raw = {int(raw_idx): value for raw_idx, value in records}
        latest_raw = int(records[-1][0])
        target_indices = [
            latest_raw - (target_count - 1 - i) * self.frame_stride
            for i in range(target_count)
        ]
        first_raw, first_value = records[0]
        selected = []
        for raw_idx in target_indices:
            if raw_idx in by_raw:
                value = by_raw[raw_idx]
            elif raw_idx < int(first_raw):
                value = first_value
            else:
                raise RuntimeError(
                    f"Missing {value_name} record for raw_idx={raw_idx}; "
                    f"available_tail={[int(i) for i, _ in records[-16:]]}"
                )
            selected.append((int(raw_idx), value))
        return selected

    def _prepare_initial_obs_records_for_cycle(self) -> list[tuple[int, np.ndarray]]:
        return self._prepare_initial_records(
            self._obs_frame_buffer,
            target_count=self.num_obs_frames,
            value_name="obs_frame",
        )

    def _build_obs_video_tensor_from_records(
        self, records: list[tuple[int, np.ndarray]]
    ) -> torch.Tensor:
        frames = self._sample_video_frames_from_records(records)
        image_tensors = [self._build_image_tensor(f) for f in frames]
        return torch.stack(image_tensors, dim=2)

    def _sample_video_frames_from_records(
        self, records: list[tuple[int, np.ndarray]]
    ) -> list[np.ndarray]:
        if not records:
            raise RuntimeError(
                "obs_frame_buffer is empty — call get_action at least once before inference."
            )
        phase_start = int(records[0][0])
        video_step_raw = int(self.frame_stride) * int(self.action_video_freq_ratio)
        sampled = [
            (raw_idx, frame)
            for raw_idx, frame in records
            if (int(raw_idx) - phase_start) % video_step_raw == 0
        ]
        if len(sampled) != self.effective_obs_frames:
            raise RuntimeError(
                "Obs video frame count mismatch: "
                f"expected={self.effective_obs_frames}, got={len(sampled)}, "
                f"phase_start={phase_start}, video_step_raw={video_step_raw}, "
                f"records={[int(raw) for raw, _ in records]}"
            )
        return [frame for _, frame in sampled]

    def _prepare_initial_proprio_records_for_cycle(self) -> list[tuple[int, tuple]]:
        """Build the raw-step-resolution proprio prefix for a new cycle."""
        records = list(self._proprio_buffer)
        if not records:
            raise RuntimeError(
                "proprio buffer is empty — call get_action at least once before inference."
            )
        by_raw = {int(raw_idx): value for raw_idx, value in records}
        latest_raw = int(records[-1][0])
        target_indices = list(
            range(latest_raw - self.num_obs_proprio + 1, latest_raw + 1)
        )
        first_raw, first_value = records[0]
        selected = []
        for raw_idx in target_indices:
            if raw_idx in by_raw:
                value = by_raw[raw_idx]
            elif raw_idx < int(first_raw):
                if not isinstance(first_value, tuple) or len(first_value) != 3:
                    raise ValueError(
                        "proprio runtime records must store "
                        "(canonical_state, effective_validity, is_pad)."
                    )
                value = (first_value[0], first_value[1], True)
            else:
                raise RuntimeError(
                    f"Missing proprio record for raw_idx={raw_idx}; "
                    f"available_tail={[int(i) for i, _ in records[-16:]]}"
                )
            selected.append((int(raw_idx), value))
        return selected

    def _build_proprio_tensor_from_records(
        self, records: list[tuple[int, np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build tensors from canonical/effective-validity runtime records."""
        state_dim = self.processor.proprio_output_dim
        expected_valid_dim = self._expected_raw_state_dim
        if not records:
            proprio = torch.zeros(1, self.num_obs_proprio, state_dim, dtype=torch.float32)
            field_valid = torch.zeros(1, self.num_obs_proprio, expected_valid_dim, dtype=torch.bool)
            is_pad = torch.ones(1, self.num_obs_proprio, dtype=torch.bool)
            return proprio, field_valid, is_pad
        state_list = []
        valid_list = []
        pad_list = []
        for _, record in records:
            if not isinstance(record, tuple) or len(record) != 3:
                raise ValueError(
                    "proprio runtime records must store "
                    "(canonical_state, effective_validity, is_pad)."
                )
            state_value, field_valid, is_pad = record
            state_list.append(torch.as_tensor(state_value, dtype=torch.float32))
            valid_list.append(torch.as_tensor(field_valid, dtype=torch.bool))
            pad_list.append(bool(is_pad))
        proprio = torch.stack(state_list, dim=0).unsqueeze(0)
        field_valid = torch.stack(valid_list, dim=0).unsqueeze(0)
        is_pad = torch.tensor(pad_list, dtype=torch.bool).unsqueeze(0)
        return proprio, field_valid, is_pad

    def _build_obs_proprio_tensor(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the observation-window proprio tensor for non-AR inference."""
        state_dim = self.processor.proprio_output_dim
        expected_valid_dim = self._expected_raw_state_dim
        if self.num_obs_proprio == 0:
            return None, None, None

        if not self._proprio_buffer:
            proprio = torch.zeros(1, self.num_obs_proprio, state_dim, dtype=torch.float32)
            field_valid = torch.zeros(1, self.num_obs_proprio, expected_valid_dim, dtype=torch.bool)
            is_pad = torch.ones(1, self.num_obs_proprio, dtype=torch.bool)
            return proprio, field_valid, is_pad

        records = self._prepare_initial_proprio_records_for_cycle()
        return self._build_proprio_tensor_from_records(records)

    # AR inference path
    @torch.no_grad()
    def _infer_action_chunk_ar(
        self,
        obs: GameObservation,
        instruction: str,
        coarse_instruction: Optional[str] = None,
    ) -> np.ndarray:
        geometry = self.model.ar_geometry
        if geometry is None:
            raise RuntimeError("AR inference requires model.ar_geometry.")

        task_str = self._build_task_string(instruction, coarse_instruction)
        prompt = self.prompt_template.format(task=task_str)

        use_cfg = self.text_cfg_scale != 1.0
        if self._planned_blocks_in_cycle == 0:
            initial_records = self._prepare_initial_obs_records_for_cycle()
            self._cycle_raw_records = dict(initial_records)
            self._cycle_prefill_start_raw = int(initial_records[0][0])
            cycle_current_raw = int(initial_records[-1][0])

            initial_proprio_records = self._prepare_initial_proprio_records_for_cycle()
            self._cycle_proprio_records = dict(initial_proprio_records)

            history_prefix = None
            history_valid_mask = None
            if getattr(self.model, "ar_history_mode", False):
                if self._history_state is None or self.model.history_module is None:
                    raise RuntimeError("AR-history inference requires initialized history state and module.")
                state = self._history_state
                history_prefix, history_valid_mask = (
                    self.model.history_module.build_prefix_from_state(
                        recent_history=state.recent_history,
                        long_memory=state.long_memory,
                        long_memory_write_count=state.long_memory_write_count,
                        current_raw=cycle_current_raw,
                        device=self.model.device,
                        dtype=self.model.torch_dtype,
                    )
                )

            obs_video = self._build_obs_video_tensor_from_records(initial_records)
            proprio, proprio_field_valid, proprio_is_pad = self._build_proprio_tensor_from_records(
                initial_proprio_records
            )

            self._cycle_state = self.model.prefill_cycle(
                obs_video=obs_video,
                history_prefix_tokens=history_prefix,
                history_valid_mask=history_valid_mask,
                prompt=prompt,
                proprio=proprio,
                proprio_field_valid=proprio_field_valid,
                proprio_is_pad=proprio_is_pad,
                text_cfg_scale=self.text_cfg_scale,
            )

        execution_block_steps = int(geometry.raw_steps_per_latent)
        prediction_horizon = int(
            getattr(
                self,
                "prediction_horizon",
                getattr(self.model, "prediction_horizon", execution_block_steps),
            )
        )
        block_index_in_cycle = self._planned_blocks_in_cycle
        action_position_offset = block_index_in_cycle * execution_block_steps
        proprio_raw_indices = [int(i) for i in sorted(self._cycle_proprio_records)]
        proprio_expected = int(
            self.num_obs_proprio
            + block_index_in_cycle * execution_block_steps
        )
        proprio_actual = len(proprio_raw_indices)
        diagnostic_cycle_state = self._cycle_state
        if isinstance(diagnostic_cycle_state, dict) and "cond" in diagnostic_cycle_state:
            diagnostic_cycle_state = diagnostic_cycle_state["cond"]
        if isinstance(diagnostic_cycle_state, dict) and "state" in diagnostic_cycle_state:
            diagnostic_cycle_state = diagnostic_cycle_state["state"]
        clean_kv_tokens = None
        clean_latents_in_cache = None
        if isinstance(diagnostic_cycle_state, dict):
            if (
                "kv_cache" in diagnostic_cycle_state
                and proprio_actual != proprio_expected
            ):
                raise RuntimeError(
                    "AR inference proprio diagnostic mismatch before denoise: "
                    f"block={block_index_in_cycle}, expected={proprio_expected}, "
                    f"actual={proprio_actual}, indices={proprio_raw_indices}."
                )
            clean_latents_in_cache = diagnostic_cycle_state.get(
                "num_latent_frames_in_cache"
            )
            clean_kv = diagnostic_cycle_state.get("kv_cache", {}).get("clean_kv", [])
            if clean_kv:
                clean_kv_tokens = int(clean_kv[0][0].shape[1])

        t0 = time.perf_counter()
        global_replan_index = int(getattr(self, "_inference_seed_index", 0))
        sampling_seed = self._next_ar_cycle_sampling_seed()
        with torch.no_grad():
            infer_kwargs = dict(
                prompt=prompt,
                input_image=None,
                num_action_steps=prediction_horizon,
                cycle_state=self._cycle_state,
                action_position_offset=action_position_offset,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=sampling_seed,
                rand_device=self.rand_device,
                tiled=self.tiled,
                text_cfg_scale=self.text_cfg_scale,
            )
            if self.eval_predict_video:
                pred = self.model.infer(
                    **infer_kwargs,
                    num_video_frames=self._num_video_frames,
                )
            else:
                pred = self.model.infer(**infer_kwargs, num_video_frames=None)
        infer_latency_sec = time.perf_counter() - t0

        action_tensor = pred["action"]
        assert action_tensor.shape[0] == prediction_horizon, (
            f"AR step must output prediction_horizon={prediction_horizon}, "
            f"got {action_tensor.shape[0]}"
        )
        postprocessed_action, action_debug = self._postprocess_action_with_debug(
            action_tensor, action_route_mask=pred.get("action_route_mask"),
        )
        predicted_action_plan = postprocessed_action[0]

        self._planned_blocks_in_cycle += 1
        blocks_per_cycle = int(geometry.pred_latents)

        if self._planned_blocks_in_cycle >= blocks_per_cycle:
            self._pending_cycle_finish = True
            self._cycle_state = None
        else:
            self._pending_kv_append = True

        if self.inference_logger is not None:
            try:
                self.inference_logger({
                    "fine_task": instruction,
                    "coarse_task": coarse_instruction,
                    "task_str": task_str,
                    "prompt": prompt,
                    "num_inference_steps": self.num_inference_steps,
                    "text_cfg_scale": self.text_cfg_scale,
                    "cfg_enabled": use_cfg,
                    "sampling_seed": sampling_seed,
                    "base_seed": self.seed,
                    "cycle_action_noise_shared": self._cycle_action_noise_shared_enabled(),
                    "cycle_sampling_seed": self._cycle_sampling_seed,
                    "global_replan_index": global_replan_index,
                    "infer_latency_sec": infer_latency_sec,
                    "action_chunk": predicted_action_plan,
                    "prediction_horizon": int(self.prediction_horizon),
                    "execution_horizon": int(self.execution_horizon),
                    "action_debug": action_debug,
                    "action_route_mask": (
                        pred["action_route_mask"].numpy()
                        if pred.get("action_route_mask") is not None
                        else None
                    ),
                    "action_route_logit": (
                        pred["action_route_logit"].numpy()
                        if pred.get("action_route_logit") is not None
                        else None
                    ),
                    "pred_video": pred.get("video"),
                    "pred_video_skip_initial_frames": 0,
                    "obs_image": obs.image,
                    "runtime_state": self._runtime_state_debug(obs),
                    "eval_predict_video": self.eval_predict_video,
                    "step_count": self.step_count,
                    "episode_count": self.episode_count,
                    "ar_cycle_step": self._planned_blocks_in_cycle,
                    "ar_block_index": int(block_index_in_cycle),
                    "proprio_expected": proprio_expected,
                    "proprio_actual": proprio_actual,
                    "proprio_raw_indices": proprio_raw_indices,
                    "clean_latents_in_cache": (
                        int(clean_latents_in_cache)
                        if clean_latents_in_cache is not None else None
                    ),
                    "clean_kv_tokens": clean_kv_tokens,
                    "ar_history_mode": bool(getattr(self.model, "ar_history_mode", False)),
                })
            except Exception as e:
                logger.warning(
                    "inference_logger callback raised %s; ignored.",
                    type(e).__name__,
                    exc_info=False,
                )

        return predicted_action_plan

    @torch.no_grad()
    def _cycle_video_records_by_raw_index(self) -> list[tuple[int, np.ndarray]]:
        if self._cycle_prefill_start_raw is None:
            raise RuntimeError("AR cycle prefill start is not initialized.")
        video_step_raw = int(self.model.ar_geometry.video_step_raw)
        phase_start = int(self._cycle_prefill_start_raw)
        ordered = sorted(self._cycle_raw_records.items(), key=lambda x: x[0])
        return [
            (raw_idx, frame)
            for raw_idx, frame in ordered
            if raw_idx >= phase_start and (raw_idx - phase_start) % video_step_raw == 0
        ]

    @torch.no_grad()
    def _select_video_frames_for_next_latent(self) -> list:
        latent_idx = int(self._cycle_state["num_latent_frames_in_cache"])
        video_records = self._cycle_video_records_by_raw_index()

        if latent_idx == 0:
            selected_records = video_records[:1]
            expected = 1
        else:
            s = 1 + 4 * (latent_idx - 1)
            e = s + 4
            selected_records = video_records[s:e]
            expected = 4

        if len(selected_records) != expected:
            raw_tail = [int(i) for i in sorted(self._cycle_raw_records)[-16:]]
            if self._cycle_prefill_start_raw is None:
                raise RuntimeError("AR cycle prefill start is not initialized.")
            phase_start = int(self._cycle_prefill_start_raw)
            raise RuntimeError(
                "Not enough frames for VAE latent append: "
                f"latent_idx={latent_idx}, expected={expected}, got={len(selected_records)}, "
                f"phase_start={phase_start}, raw_records_tail={raw_tail}"
            )
        self._last_append_raw_indices = [int(i) for i, _ in selected_records]
        return [frame for _, frame in selected_records]

    @torch.no_grad()
    def _append_real_obs_to_kv_cache(self) -> None:
        video_frames = self._select_video_frames_for_next_latent()
        image_tensors = [self._build_image_tensor(f) for f in video_frames]
        new_video = torch.stack([t.squeeze(0) for t in image_tensors], dim=1).unsqueeze(0)
        proprio, proprio_field_valid, proprio_is_pad = self._build_cycle_proprio_tensor_for_current_cache()
        self._cycle_state = self.model.encode_and_append_real_obs(
            cycle_state=self._cycle_state,
            new_video_frames=new_video,
            new_proprio=proprio,
            new_proprio_field_valid=proprio_field_valid,
            new_proprio_is_pad=proprio_is_pad,
        )

    def _build_cycle_proprio_tensor_for_current_cache(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        state_dim = self.processor.proprio_output_dim
        if self.num_obs_proprio == 0 or state_dim == 0 or not self._cycle_proprio_records:
            return None, None, None

        if self._cycle_prefill_start_raw is None:
            raise RuntimeError("AR cycle prefill start is not initialized.")
        geometry = self.model.ar_geometry
        latent_idx = int(self._cycle_state["num_latent_frames_in_cache"])
        num_obs_latents = int(self.model.video_expert.num_obs_latents)
        visible_steps = self.num_obs_proprio + max(
            0, latent_idx - num_obs_latents + 1
        ) * int(geometry.raw_steps_per_latent)
        phase_start = int(self._cycle_prefill_start_raw)
        ordered = [
            (raw_idx, value)
            for raw_idx, value in sorted(self._cycle_proprio_records.items(), key=lambda x: x[0])
            if int(raw_idx) >= phase_start
        ]
        if len(ordered) != int(visible_steps):
            raise RuntimeError(
                "AR cycle proprio context length mismatch: "
                f"latent_idx={latent_idx}, expected={visible_steps}, "
                f"got={len(ordered)}, phase_start={phase_start}, "
                f"records_tail={[int(i) for i in sorted(self._cycle_proprio_records)[-8:]]}"
            )

        return self._build_proprio_tensor_from_records(ordered)

    @torch.no_grad()
    def _archive_video_records_by_raw_index(self) -> list[tuple[int, np.ndarray]]:
        geometry = self.model.history_geometry
        state = self._history_state

        cycle_start = int(state.current_start_raw)
        cycle_end = cycle_start + geometry.cycle_stride_raw
        video_step_raw = int(geometry.video_step_raw)

        ordered = sorted(self._cycle_raw_records.items(), key=lambda x: x[0])
        return [
            (raw_idx, frame)
            for raw_idx, frame in ordered
            if cycle_start <= raw_idx <= cycle_end
            and (raw_idx - cycle_start) % video_step_raw == 0
        ]

    @torch.no_grad()
    def _archive_cycle_to_history(self) -> None:
        from .models.history.history_module import HistorySegment

        geometry = self.model.history_geometry
        state = self._history_state

        T_seg = geometry.history_segment_latents
        video_frames_needed = (T_seg - 1) * 4 + 1

        cycle_start = int(state.current_start_raw)
        cycle_end = cycle_start + geometry.cycle_stride_raw
        video_records = self._archive_video_records_by_raw_index()

        if len(video_records) != video_frames_needed:
            raw_tail = [int(i) for i in sorted(self._cycle_raw_records)[-16:]]
            raise RuntimeError(
                "Archive frame count mismatch: "
                f"expected={video_frames_needed}, got={len(video_records)}, "
                f"cycle_start={cycle_start}, cycle_end={cycle_end}, raw_records_tail={raw_tail}"
            )
        video_frames = [frame for _, frame in video_records]

        image_tensors = [self._build_image_tensor(f) for f in video_frames]
        obs_video = torch.stack(image_tensors, dim=2)
        cycle_latents = self.model._encode_video_latents(obs_video, tiled=self.tiled)
        actual_T = cycle_latents.shape[2]
        if actual_T != T_seg:
            raise RuntimeError(
                f"Archive latent count mismatch: expected={T_seg}, got={actual_T}"
            )
        segment_latents = cycle_latents[0]

        new_segment = HistorySegment(
            latents=segment_latents,
            raw_start=cycle_start,
            next_cycle_start_raw=cycle_start + geometry.cycle_stride_raw,
        )

        self._history_state = self.model.history_module.push_history_segment_state(
            state=state,
            history_segment=new_segment,
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        state = self._history_state

        state.current_start_raw = cycle_start + geometry.cycle_stride_raw
        self._clear_ar_cycle_state()

    def _clear_ar_cycle_state(self) -> None:
        self._planned_blocks_in_cycle = 0
        self._cycle_raw_records = {}
        self._cycle_proprio_records = {}
        self._cycle_prefill_start_raw = None
        self._cycle_state = None
        self._pending_kv_append = False
        self._cycle_sampling_seed = None

    # Non-AR inference path
    def _infer_action_chunk(
        self,
        obs: GameObservation,
        instruction: str,
        coarse_instruction: Optional[str] = None,
    ) -> np.ndarray:
        sampling_seed = self._next_inference_seed()
        infer_kwargs, ctx = self._build_infer_inputs(
            instruction, coarse_instruction, sampling_seed=sampling_seed
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            if self.eval_predict_video:
                pred = self.model.infer(
                    **infer_kwargs,
                    num_video_frames=self._num_video_frames,
                )
            else:
                pred = self.model.infer(**infer_kwargs, num_video_frames=None)
        infer_latency_sec = time.perf_counter() - t0

        action_tensor = pred["action"]
        postprocessed_action, action_debug = self._postprocess_action_with_debug(
            action_tensor, action_route_mask=pred.get("action_route_mask"),
        )
        action_chunk = postprocessed_action[0]

        if self.inference_logger is not None:
            try:
                self.inference_logger({
                    "fine_task": instruction,
                    "coarse_task": coarse_instruction,
                    "task_str": ctx["task_str"],
                    "prompt": ctx["prompt"],
                    "num_inference_steps": self.num_inference_steps,
                    "text_cfg_scale": self.text_cfg_scale,
                    "sampling_seed": sampling_seed,
                    "prediction_horizon": int(self.prediction_horizon),
                    "execution_horizon": int(self.execution_horizon),
                    "infer_latency_sec": infer_latency_sec,
                    "action_chunk": action_chunk,
                    "action_debug": action_debug,
                    "action_route_mask": (
                        pred["action_route_mask"].numpy()
                        if pred.get("action_route_mask") is not None
                        else None
                    ),
                    "action_route_logit": (
                        pred["action_route_logit"].numpy()
                        if pred.get("action_route_logit") is not None
                        else None
                    ),
                    "pred_video": pred.get("video"),
                    "pred_video_skip_initial_frames": 1,
                    "obs_image": obs.image,
                    "runtime_state": self._runtime_state_debug(obs),
                    "eval_predict_video": self.eval_predict_video,
                    "step_count": self.step_count,
                    "episode_count": self.episode_count,
                })
            except Exception as e:
                logger.warning(
                    "inference_logger callback raised %s; ignored.",
                    type(e).__name__,
                    exc_info=False,
                )

        return action_chunk

    def _build_infer_inputs(
        self,
        instruction: str,
        coarse_instruction: Optional[str] = None,
        *,
        sampling_seed: Optional[int] = None,
    ) -> tuple[dict, dict]:
        obs_video = self._build_obs_video_tensor()
        proprio, proprio_field_valid, proprio_is_pad = self._build_obs_proprio_tensor()

        task_str = self._build_task_string(instruction, coarse_instruction)
        prompt = self.prompt_template.format(task=task_str)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": obs_video,
            "num_action_steps": self.prediction_horizon,
            "proprio": proprio,
            "proprio_field_valid": proprio_field_valid,
            "proprio_is_pad": proprio_is_pad,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": sampling_seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }

        return infer_kwargs, {"task_str": task_str, "prompt": prompt}

    def _replan_and_enqueue_committed_prefix(
        self,
        obs: GameObservation,
        instruction: str,
        coarse_instruction: Optional[str] = None,
    ) -> None:
        """Replan and enqueue only the action prefix committed for execution."""
        if getattr(self.model, "ar_mode", False):
            geometry = self.model.ar_geometry
            if geometry is None:
                raise RuntimeError("AR inference requires model.ar_geometry.")
            if getattr(self, "_pending_cycle_finish", False):
                if getattr(self.model, "ar_history_mode", False):
                    self._archive_cycle_to_history()
                else:
                    self._clear_ar_cycle_state()
                self._pending_cycle_finish = False
            elif (
                getattr(self, "_pending_kv_append", False)
                and self._cycle_state is not None
            ):
                self._append_real_obs_to_kv_cache()
                self._pending_kv_append = False
            predicted_action_plan = self._infer_action_chunk_ar(
                obs, instruction, coarse_instruction
            )
            committed_prefix_length = int(
                getattr(
                    self,
                    "execution_horizon",
                    geometry.raw_steps_per_latent,
                )
            )
        else:
            predicted_action_plan = self._infer_action_chunk(
                obs, instruction, coarse_instruction
            )
            committed_prefix_length = min(
                int(
                    getattr(
                        self,
                        "execution_horizon",
                        predicted_action_plan.shape[0],
                    )
                ),
                predicted_action_plan.shape[0],
            )
        for action_index in range(committed_prefix_length):
            self._committed_action_queue.append(
                np.asarray(predicted_action_plan[action_index], dtype=np.float32)
            )

    # Public API
    def get_action(
        self,
        obs: GameObservation,
        instruction: str,
        coarse_instruction: Optional[str] = None,
    ) -> np.ndarray:
        if obs.image is not None:
            self._obs_frame_buffer.append((self.step_count, obs.image))
            if getattr(self.model, "ar_mode", False):
                self._cycle_raw_records[int(self.step_count)] = obs.image
        if (
            obs.proprio is None
            and str(getattr(self.processor, "proprio_encoder_type", ""))
            in {"generic_masked_linear", "minecraft_grouped_cursor"}
        ):
            raise ValueError(
                "Gated proprio inference requires a proprio observation on every step."
            )
        if obs.proprio is not None:
            proprio, proprio_field_valid = self._normalize_state_with_validity(
                obs.proprio, field_valid=obs.proprio_field_valid
            )
            self._last_model_proprio = (
                proprio.squeeze(0).detach().to(
                    dtype=torch.float32, device="cpu"
                ).numpy()
            )
            self._last_model_proprio_field_valid = (
                proprio_field_valid.squeeze(0).detach().to(device="cpu").numpy()
            )
            proprio_record = (
                proprio.squeeze(0).detach().cpu(),
                proprio_field_valid.squeeze(0).detach().cpu(),
                False,
            )
            self._proprio_buffer.append((self.step_count, proprio_record))
            if getattr(self.model, "ar_mode", False):
                self._cycle_proprio_records[int(self.step_count)] = proprio_record

        if not self._committed_action_queue:
            self._replan_and_enqueue_committed_prefix(
                obs, instruction, coarse_instruction
            )

        if not self._committed_action_queue:
            raise RuntimeError(
                "Model generated no actions for the committed execution prefix."
            )
        action = np.asarray(self._committed_action_queue.popleft(), dtype=np.float32)
        self.step_count += 1
        return action

    def reset(self) -> None:
        """Reset inference state for a new episode."""
        self._committed_action_queue.clear()
        self._last_model_proprio = None
        self._last_model_proprio_field_valid = None
        self._obs_frame_buffer.clear()
        self._proprio_buffer.clear()
        self.episode_count += 1
        self.step_count = 0
        self._inference_seed_index = 0
        self._cycle_sampling_seed = None

        if getattr(self.model, "ar_mode", False):
            self._planned_blocks_in_cycle = 0
            self._cycle_raw_records = {}
            self._cycle_proprio_records = {}
            self._cycle_prefill_start_raw = None
            self._cycle_state = None
            self._pending_kv_append = False
            self._pending_cycle_finish = False
            self._cycle_sampling_seed = None

        if (
            getattr(self.model, "ar_history_mode", False)
            and self.model.history_module is not None
        ):
            from .models.history.history_module import init_history_state

            self._history_state = init_history_state(
                history_config=self.model.history_config,
                hidden_dim=self.model.video_expert.hidden_dim,
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )

    def close(self) -> None:
        """Release inference resources (CUDA cache). Safe to call multiple times."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # Offline single-image → video demo
    @torch.no_grad()
    def infer_image_to_video(
        self,
        *,
        input_image_path: str,
        prompt: str,
        output_mp4: str,
        num_frames: int,
        width: int,
        height: int,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> str:
        """Offline image-to-video demo. Returns the output mp4 path."""

        def _center_crop_resize(img: Image.Image, width: int, height: int) -> Image.Image:
            src_w, src_h = img.size
            scale = max(width / src_w, height / src_h)
            resized = img.resize(
                (round(src_w * scale), round(src_h * scale)),
                resample=Image.BILINEAR,
            )
            rw, rh = resized.size
            left = max((rw - width) // 2, 0)
            top = max((rh - height) // 2, 0)
            return resized.crop((left, top, left + width, top + height))

        input_image = Image.open(input_image_path).convert("RGB")
        input_image = _center_crop_resize(input_image, width=width, height=height)
        arr = np.asarray(input_image, dtype=np.float32)
        x = torch.from_numpy(arr)
        x = x.to(device=self.model.device, dtype=self.model.torch_dtype)
        x = x * (2.0 / 255.0) - 1.0
        x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]

        formatted_prompt = self.prompt_template.format(task=prompt)
        out = self.model.infer(
            prompt=formatted_prompt,
            input_image=x,
            num_video_frames=int(num_frames),
            num_action_steps=1,
            text_cfg_scale=float(text_cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=(
                None if sigma_shift is None else float(sigma_shift)
            ),
            seed=(
                None if seed is None else int(seed)
            ),
            rand_device=str(rand_device),
            tiled=bool(tiled),
        )
        video = out["video"]
        save_mp4(video, str(output_mp4), fps=15)
        logger.info("Saved inference video to %s", output_mp4)
        return str(output_mp4)
