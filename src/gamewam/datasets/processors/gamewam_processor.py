"""GameWAMProcessor — data pre/post-processing for GameWAM."""

from typing import Dict, Any, Optional, List, Sequence
from copy import deepcopy

import torch
import numpy as np
from ..utils.normalizer import NormMode, RouteAwareActionNormalizer
from gamewam.utils.logging_config import get_logger
from .base_processor import BaseProcessor
from gamewam.datasets.transforms.minecraft_proprio import (
    MinecraftCursorProprioTransform,
)
from gamewam.envs.minecraft.observations.state import GUI_OPEN_IDX

logger = get_logger(__name__)


class GameWAMProcessor(BaseProcessor):
    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_frames: int,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,
        proprio_encoder_type: str = "generic_linear",
        action_state_transforms: Optional[List[Any]] = None,
        use_stepwise_action_norm: bool = False,
        norm_default_mode: NormMode = "min/max",
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]] = None,
        action_state_merger=None,
        train_transforms=None,
        val_transforms=None,
        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,
        tokenizer: Optional[Any] = None,
        pad_action_dim_zero_mask: Optional[Dict[str, List[bool]]] = None,
        dim_is_continuous: Optional[Sequence[bool]] = None,
        gui_label_dim_index: Optional[int] = None,
        gui_label_state_key: str = "default",
        route_aware_action_normalization: bool = False,
        cursor_reference_size: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            shape_meta=shape_meta,
            num_frames=num_frames,
            num_output_cameras=num_output_cameras,
            action_output_dim=action_output_dim,
            proprio_output_dim=proprio_output_dim,
            action_state_transforms=action_state_transforms,
            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,
            action_state_merger=action_state_merger,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            drop_high_level_prob=drop_high_level_prob,
            use_zh_instruction=use_zh_instruction,
            tokenizer=tokenizer,
        )
        self.proprio_encoder_type = str(proprio_encoder_type)
        if self.proprio_encoder_type not in {
            "generic_linear",
            "generic_masked_linear",
            "minecraft_grouped_cursor",
        }:
            raise ValueError(
                f"Unsupported proprio_encoder_type={self.proprio_encoder_type!r}; "
                "expected 'generic_linear', 'generic_masked_linear', or "
                "'minecraft_grouped_cursor'."
            )

        if pad_action_dim_zero_mask is None:
            self.pad_action_dim_zero_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(pad_action_dim_zero_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`pad_action_dim_zero_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )
            self.pad_action_dim_zero_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["shape"]
                mask = pad_action_dim_zero_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`pad_action_dim_zero_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.pad_action_dim_zero_mask[key] = torch.as_tensor(mask, dtype=torch.bool)

        self.dim_is_continuous = (
            [bool(x) for x in dim_is_continuous] if dim_is_continuous is not None else None
        )
        if self.dim_is_continuous is not None:
            if len(self.dim_is_continuous) != int(action_output_dim):
                raise ValueError(
                    f"`dim_is_continuous` length must equal `action_output_dim` "
                    f"({int(action_output_dim)}), got {len(self.dim_is_continuous)}."
                )
            if not any(self.dim_is_continuous):
                raise ValueError(
                    "`dim_is_continuous` must mark at least one dim as continuous."
                )
        self.gui_label_dim_index = gui_label_dim_index
        self.gui_label_state_key = (
            None if gui_label_state_key is None else str(gui_label_state_key)
        )
        self.cursor_reference_size: Optional[tuple[float, float]] = None
        self.proprio_field_valid_dim: Optional[int] = None
        schema_state_key = self.gui_label_state_key or "default"
        schema_state_meta = next(
            (
                meta for meta in self.shape_meta.get("state", [])
                if meta.get("key") == schema_state_key
            ),
            None,
        )
        self._uses_minecraft_cursor_transform = any(
            isinstance(transform, MinecraftCursorProprioTransform)
            for transform in (self.action_state_transforms or [])
        )
        if self.proprio_encoder_type == "minecraft_grouped_cursor":
            if schema_state_meta is None:
                raise ValueError(
                    f"proprio_encoder_type={self.proprio_encoder_type!r} requires "
                    f"state key {schema_state_key!r} in shape_meta.state."
                )
            raw_dim = int(schema_state_meta["raw_shape"])
            if (
                raw_dim != MinecraftCursorProprioTransform.RAW_DIM
                or int(proprio_output_dim)
                != MinecraftCursorProprioTransform.CANONICAL_DIM
            ):
                raise ValueError(
                    "minecraft_grouped_cursor schema requires raw state dim 6 "
                    f"and proprio_output_dim=15, got {raw_dim} and "
                    f"{proprio_output_dim}."
                )
            if cursor_reference_size is None or len(cursor_reference_size) != 2:
                raise ValueError(
                    "minecraft_grouped_cursor requires "
                    "cursor_reference_size=[width, height]."
                )
            width, height = map(float, cursor_reference_size)
            if width <= 0.0 or height <= 0.0:
                raise ValueError(
                    "cursor_reference_size values must be positive, got "
                    f"{cursor_reference_size!r}."
                )
            self.cursor_reference_size = (width, height)
            self.proprio_field_valid_dim = (
                MinecraftCursorProprioTransform.FIELD_VALID_DIM
            )
        elif self.proprio_encoder_type == "generic_masked_linear":
            if schema_state_meta is None:
                raise ValueError(
                    "generic_masked_linear requires a state field in shape_meta."
                )
            self.proprio_field_valid_dim = int(schema_state_meta["raw_shape"])
            if self.proprio_field_valid_dim <= 0:
                raise ValueError(
                    "generic_masked_linear requires a positive raw state dimension."
                )
        self.route_aware_action_normalization = bool(route_aware_action_normalization)
        if self.route_aware_action_normalization and self.gui_label_dim_index is None:
            raise ValueError(
                "`route_aware_action_normalization=True` requires "
                "`gui_label_dim_index` to be set."
            )
        if self.route_aware_action_normalization and self.dim_is_continuous is None:
            raise ValueError(
                "`route_aware_action_normalization=True` requires "
                "`dim_is_continuous` to be set."
            )
        if self.route_aware_action_normalization and not self.gui_label_state_key:
            raise ValueError(
                "`route_aware_action_normalization=True` requires a non-empty "
                "`gui_label_state_key`."
            )
        if self.gui_label_dim_index is not None:
            route_state_meta = next(
                (
                    meta for meta in self.shape_meta.get("state", [])
                    if meta.get("key") == self.gui_label_state_key
                ),
                None,
            )
            if route_state_meta is None:
                raise ValueError(
                    f"gui_label_state_key={self.gui_label_state_key!r} is not "
                    "present in shape_meta.state."
                )
            state_dim = int(route_state_meta["raw_shape"])
            if not 0 <= int(self.gui_label_dim_index) < state_dim:
                raise ValueError(
                    f"gui_label_dim_index={self.gui_label_dim_index} is out of "
                    f"range for state key {self.gui_label_state_key!r} with "
                    f"dimension {state_dim}."
                )
            expected_gui_index = (
                GUI_OPEN_IDX
                if self.proprio_encoder_type == "minecraft_grouped_cursor"
                else None
            )
            if (
                expected_gui_index is not None
                and int(self.gui_label_dim_index) != expected_gui_index
            ):
                raise ValueError(
                    f"{self.proprio_encoder_type} requires gui_label_dim_index="
                    f"{expected_gui_index}, got {self.gui_label_dim_index}; using the "
                    "wrong raw-state field as the GUI route label is forbidden."
                )
        # Always defined; ``_post_set_normalizer_hook`` populates these when
        # route-aware action normalization is enabled.
        self.route_norm: Optional[RouteAwareActionNormalizer] = None
        self.route_state: Optional[dict] = None

    def _resolve_action_mode(self, action_key: str):
        """Mirror ``LinearNormalizer.__init__``'s per-key mode
        resolution so the route normalizer honors
        ``norm_exception_mode`` (e.g. ``dims: [0, 1] → z-score``).

        The global ``LinearNormalizer`` resolves the mode per
        action key when it builds its ``SingleFieldLinearNormalizer``
        for that key. The route normalizer must use the same
        resolved mode for **all three** of its inner normalizers
        (gameplay / gui / global) so that forward / backward in
        the route space stay aligned with the trainer's notion of
        "the action space" for that key.
        """
        from ..utils.normalizer import _resolve_mode_spec

        action_meta = next(
            (m for m in self.shape_meta.get("action", []) if m.get("key") == action_key),
            None,
        )
        if action_meta is None:
            return self.norm_default_mode
        if (
            self.norm_exception_mode is not None
            and "action" in self.norm_exception_mode
            and action_key in self.norm_exception_mode["action"]
        ):
            cur_mode = self.norm_exception_mode["action"][action_key]
        else:
            cur_mode = self.norm_default_mode
        return _resolve_mode_spec(cur_mode, self.norm_default_mode, action_meta)

    def _build_route_normalizers(
        self,
        normalizer_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the per-route action normalizer from ``normalizer_stats``.

        ``normalizer_stats`` is the per-key stats dict from
        ``LinearNormalizer.normalizers`` (already with the ``global_``
        prefix stripped). It must contain ``action_by_route.gameplay``
        and ``action_by_route.gui`` when route-aware normalization is
        enabled. Returns a dict with ``"route_norm"`` (a single
        ``RouteAwareActionNormalizer`` that internally dispatches by
        route mask) and ``"continuous_dims"``.
        """
        from ..utils.normalizer import SingleFieldLinearNormalizer

        if not self.route_aware_action_normalization:
            return {}
        metadata = normalizer_stats.get("route_aware_action_normalization")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Route-aware stats are missing the "
                "`route_aware_action_normalization` metadata block; "
                "re-generate dataset_stats.json."
            )
        expected_continuous_dims = [
            int(d) for d, c in enumerate(self.dim_is_continuous or []) if c
        ]
        expected_metadata = {
            "enabled": True,
            "route_state_key": str(self.gui_label_state_key),
            "gui_label_dim_index": int(self.gui_label_dim_index),
            "continuous_dims": expected_continuous_dims,
            "action_dim": int(self.action_output_dim),
        }
        for field, expected in expected_metadata.items():
            actual = metadata.get(field)
            # ``load_dataset_stats_from_json`` rehydrates numeric
            # lists/scalars to tensors; normalize both sides to plain
            # Python types for a well-defined comparison.
            if isinstance(actual, torch.Tensor):
                actual = actual.detach().cpu().tolist()
                if isinstance(actual, list) and len(actual) == 1:
                    actual = actual[0]
            if isinstance(expected, list) and not isinstance(actual, list):
                actual = [actual] if actual is not None else None
            if actual != expected:
                raise ValueError(
                    "Route-aware stats metadata mismatch for "
                    f"{field!r}: stats={actual!r}, processor={expected!r}. "
                    "Re-generate dataset_stats.json with the current config."
                )
        if "action_by_route" not in normalizer_stats:
            raise ValueError(
                "`route_aware_action_normalization=True` requires "
                "`action_by_route` in stats. The dataset stats file "
                "predates route-aware support — re-generate it."
            )
        by_route = normalizer_stats["action_by_route"]
        for route in ("gameplay", "gui"):
            if route not in by_route:
                raise ValueError(
                    f"`action_by_route.{route}` is missing from stats. "
                    "The dataset stats file is incomplete."
                )
            # ``count`` may be int (fresh stats) or a 0-d/1-element
            # tensor (loaded from JSON). Normalize to plain int.
            raw_count = by_route[route].get("count", 0)
            if isinstance(raw_count, torch.Tensor):
                raw_count = int(raw_count.detach().cpu().item())
            else:
                raw_count = int(raw_count)
            if raw_count <= 0:
                raise ValueError(
                    f"`action_by_route.{route}.count` is zero — "
                    f"no {route} tokens were observed during stats generation. "
                    "Route-aware normalization cannot proceed without "
                    f"enough {route} samples."
                )
        continuous_dims = expected_continuous_dims
        if not continuous_dims:
            raise ValueError(
                "`route_aware_action_normalization=True` requires "
                "at least one continuous dim in `dim_is_continuous`."
            )
        discrete_dims = [
            int(d) for d, c in enumerate(self.dim_is_continuous or []) if not c
        ]
        action_keys = [m["key"] for m in self.shape_meta.get("action", [])]
        if len(action_keys) != 1:
            raise ValueError(
                "Route-aware action normalization currently requires exactly "
                f"one action key, got {action_keys}."
            )
        key = action_keys[0]
        for route in ("gameplay", "gui"):
            if key not in by_route[route]:
                raise ValueError(
                    f"`action_by_route.{route}.{key}` is missing from stats."
                )
        # Use the same resolved mode for all three inner normalizers
        # (gameplay / gui / global) so the route forward / backward
        # stays in the trainer's intended action space.
        resolved_mode = self._resolve_action_mode(key)
        # The stats schema uses the ``global_``/``stepwise_`` prefix;
        # ``SingleFieldLinearNormalizer`` expects the unprefixed keys.
        def _strip_prefix(d):
            return {
                (k.removeprefix("global_").removeprefix("stepwise_")): v
                for k, v in d.items()
                if isinstance(v, torch.Tensor)
            }

        gp_raw = _strip_prefix(by_route["gameplay"][key])
        gui_raw = _strip_prefix(by_route["gui"][key])
        glob_raw = _strip_prefix(normalizer_stats.get("action", {}).get(key, {}))
        if not glob_raw:
            raise ValueError(f"Global action stats for key {key!r} are missing.")
        gp_norm = SingleFieldLinearNormalizer(stats=gp_raw, mode=resolved_mode)
        gui_norm = SingleFieldLinearNormalizer(stats=gui_raw, mode=resolved_mode)
        global_norm = SingleFieldLinearNormalizer(stats=glob_raw, mode=resolved_mode)
        route_norm = RouteAwareActionNormalizer(
            global_norm=global_norm,
            gameplay_norm=gp_norm,
            gui_norm=gui_norm,
            continuous_dims=continuous_dims,
            discrete_dims=discrete_dims,
        )
        return {
            "route_norm": route_norm,
            "continuous_dims": continuous_dims,
            "discrete_dims": discrete_dims,
        }

    def _post_set_normalizer_hook(self, dataset_stats: Dict[str, Any]) -> None:
        """Build the per-route normalizer when route-aware is enabled.

        Called automatically by ``set_normalizer_from_stats`` after the
        global normalizer is constructed. ``self.route_norm`` (or
        ``self.route_state`` for the dim lists) is populated here so
        the preprocess/postprocess pipeline can route-aware-normalize
        the action.
        """
        # Discrete dims live in the normalized action tensor as
        # {-1, +1}, so they must use binary min/max normalization even
        # when route-aware continuous normalization is disabled.
        # Validate the invariant for every processor that declares a
        # mixed semantic schema, not only when the route-aware setup
        # below runs.
        discrete_dims = [
            int(d) for d, is_continuous in enumerate(self.dim_is_continuous or [])
            if not is_continuous
        ]
        if discrete_dims:
            action_keys = [m["key"] for m in self.shape_meta.get("action", [])]
            if len(action_keys) != 1:
                raise ValueError(
                    "Discrete-action normalization currently requires "
                    f"exactly one action key, got {action_keys}."
                )
            action_key = action_keys[0]
            _validate_discrete_dims_norm(
                self._resolve_action_mode(action_key), discrete_dims, action_key
            )

        if not self.route_aware_action_normalization:
            self.route_norm = None
            self.route_state = None
            return
        if dataset_stats is None:
            raise ValueError(
                "`route_aware_action_normalization=True` requires "
                "stats, but `set_normalizer_from_stats(None)` was called."
            )
        state = self._build_route_normalizers(dataset_stats)
        self.route_norm = state["route_norm"]
        self.route_state = {
            "continuous_dims": state["continuous_dims"],
            "discrete_dims": state["discrete_dims"],
        }

    def augment_instruction(self, data: Dict[str, Any]) -> str:
        """Process instruction from game data."""
        fine_task = data.get("task", "")
        if self.drop_high_level_prob is None:
            return fine_task

        coarse_task = data.get("coarse_task", "")
        if (
            np.random.rand() < self.drop_high_level_prob
            or not coarse_task
            or coarse_task == fine_task
        ):
            return fine_task
        return f"[High]: {coarse_task}, [Low]: {fine_task}"

    def _get_state_field_valid(
        self,
        data: Dict[str, Any],
        raw_state: torch.Tensor,
    ) -> torch.Tensor:
        valid = data.get("state_field_valid")
        if isinstance(valid, dict):
            valid = valid.get(self.gui_label_state_key or "default")
        expected_dim = self.proprio_field_valid_dim or int(raw_state.shape[-1])
        if (
            self.proprio_field_valid_dim is not None
            and raw_state.shape[-1] != expected_dim
        ):
            raise ValueError(
                f"raw state must have last dim {expected_dim} for "
                f"{self.proprio_encoder_type}, got {tuple(raw_state.shape)}"
            )
        if valid is None:
            if self.proprio_encoder_type == "minecraft_grouped_cursor":
                raise ValueError(
                    "Minecraft cursor data requires state_field_valid; "
                    "an all-valid fallback would make missing cursor state ambiguous."
                )
            valid = torch.ones(
                raw_state.shape[:-1] + (expected_dim,),
                dtype=torch.bool,
                device=raw_state.device,
            )
        valid = torch.as_tensor(
            valid, dtype=torch.bool, device=raw_state.device
        )
        expected_shape = raw_state.shape[:-1] + (expected_dim,)
        if valid.shape == (expected_dim,):
            valid = valid.view(
                *([1] * (len(expected_shape) - 1)), expected_dim
            ).expand(expected_shape)
        if valid.shape != expected_shape:
            raise ValueError(
                f"state_field_valid must be {tuple(expected_shape)}, got "
                f"{tuple(valid.shape)}"
            )
        if (
            self.proprio_encoder_type == "minecraft_grouped_cursor"
            or self._uses_minecraft_cursor_transform
        ):
            valid = MinecraftCursorProprioTransform.effective_field_validity(
                raw_state, valid
            )
        return valid

    def preprocess_state_only(
        self,
        raw_state: torch.Tensor,
        raw_field_valid: torch.Tensor,
        state_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.proprio_encoder_type == "minecraft_grouped_cursor":
            canonical, field_valid = (
                MinecraftCursorProprioTransform.transform_state_and_validity(
                    raw_state,
                    cursor_reference_size=self.cursor_reference_size,
                    field_valid=raw_field_valid,
                )
            )
        else:
            raise ValueError(
                "preprocess_state_only requires minecraft_grouped_cursor, got "
                f"{self.proprio_encoder_type!r}."
            )
        is_pad = torch.as_tensor(
            state_is_pad, dtype=torch.bool, device=canonical.device
        )
        expected_valid_dim = int(self.proprio_field_valid_dim)
        if field_valid.shape != canonical.shape[:-1] + (expected_valid_dim,):
            raise ValueError(
                "raw_field_valid must match state leading dims with last dim "
                f"{expected_valid_dim}, got {tuple(field_valid.shape)}"
            )
        if is_pad.shape != canonical.shape[:-1]:
            raise ValueError(
                f"state_is_pad must match state leading dims, got {tuple(is_pad.shape)} "
                f"for state {tuple(canonical.shape)}"
            )
        return canonical, field_valid, is_pad

    def _emit_auxiliary_action_fields(
        self,
        sample: Dict[str, Any],
        pre_norm_action: Optional[torch.Tensor],
        raw_state: Optional[torch.Tensor],
        raw_state_field_valid: Optional[torch.Tensor] = None,
    ) -> None:
        """Emit the auxiliary action sample keys used by the losses,
        metrics and route supervision."""
        if self.dim_is_continuous is None:
            return

        flags = self.dim_is_continuous
        discrete_indices = [i for i, c in enumerate(flags) if not c]
        continuous_indices = [i for i, c in enumerate(flags) if c]

        if pre_norm_action is not None and len(discrete_indices) > 0:
            sample["target_discrete_action"] = pre_norm_action[
                ..., discrete_indices
            ].float()
        if (
            pre_norm_action is not None
            and "action" in sample
            and isinstance(sample["action"], torch.Tensor)
            and len(continuous_indices) > 0
        ):
            sample["target_continuous_action"] = sample["action"][
                ..., continuous_indices
            ].float()
            sample["raw_continuous_action"] = pre_norm_action[
                ..., continuous_indices
            ].float()
        if (
            raw_state is not None
            and self.gui_label_dim_index is not None
        ):
            # Slice ``[:-1, dim:dim+1]`` to align with action (state is
            # one frame longer in LeRobot convention; the model also
            # slices ``proprio[:-1]``).
            sliced = raw_state[:-1, self.gui_label_dim_index:self.gui_label_dim_index + 1]
            sample["action_route_before_action"] = sliced.float()
            if raw_state_field_valid is not None:
                sample["action_route_valid"] = raw_state_field_valid[
                    :-1, self.gui_label_dim_index:self.gui_label_dim_index + 1
                ].bool()

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess game data for the model"""
        sample = {}
        is_video_only = bool(data.get("video_only", False))
        skip_images = bool(data.get("skip_images", False))

        # 1. instruction
        if "task" in data:
            sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]

        # 2. image
        if not skip_images:
            processed_images = []
            for meta in self.shape_meta["images"]:
                key = meta["key"]
                expected_shape = meta.get("shape", meta.get("raw_shape"))
                if expected_shape is None:
                    raise ValueError(f"Image meta for key {key!r} must define `raw_shape` or `shape`.")
                image = data["images"][key]  # [num_frames, C, H, W]
                assert image.ndim == 4, f"Expected 4D (T, C, H, W), got shape {image.shape}"

                transforms = self.train_transforms if self.is_train else self.val_transforms
                current_transforms = transforms[key] if isinstance(transforms, dict) else transforms
                if current_transforms is not None:
                    for trans in current_transforms:
                        image = trans(image)

                meta_shape = [self.num_frames] + list(expected_shape)
                assert list(image.shape) == meta_shape, \
                    f"Expected shape {meta_shape}, got {list(image.shape)} after transforms for key {key}"

                processed_images.append(image)

            pixel_values = torch.stack(processed_images, dim=0)  # [num_cameras, T, C, H, W]

            if self.num_output_cameras > pixel_values.shape[0]:
                out = torch.zeros(
                    (self.num_output_cameras,) + pixel_values.shape[1:],
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
                out[:pixel_values.shape[0]] = pixel_values
                sample["pixel_values"] = out
            elif self.num_output_cameras < pixel_values.shape[0]:
                sample["pixel_values"] = pixel_values[:self.num_output_cameras]
            else:
                sample["pixel_values"] = pixel_values

        # Copy action before transform for open-loop eval
        if not is_video_only and not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        # Snapshot raw action and state for the auxiliary fields
        # (before any transform).
        pre_norm_action = None
        raw_state_for_route = None
        raw_state_field_valid_for_route = None
        if self.dim_is_continuous is not None and "action" in data:
            try:
                pre_norm_action = torch.cat(
                    [data["action"][m["key"]] for m in self.shape_meta["action"]],
                    dim=-1,
                )
            except (KeyError, TypeError):
                pre_norm_action = None
        if (
            self.dim_is_continuous is not None
            and self.gui_label_dim_index is not None
            and "state" in data
        ):
            try:
                raw_state_for_route = data["state"][self.gui_label_state_key]
                raw_state_field_valid_for_route = self._get_state_field_valid(data, raw_state_for_route)
            except (KeyError, TypeError) as exc:
                available = (
                    list(data["state"].keys())
                    if isinstance(data.get("state"), dict)
                    else type(data.get("state")).__name__
                )
                raise KeyError(
                    f"GUI route state key {self.gui_label_state_key!r} is missing; "
                    f"available state keys={available}."
                ) from exc
        if raw_state_for_route is None and isinstance(data.get("state"), dict) and "default" in data["state"]:
            raw_state_for_route = data["state"]["default"]
            raw_state_field_valid_for_route = self._get_state_field_valid(data, raw_state_for_route)

        # 3. action & state (skip for video_only precompute)
        if not is_video_only and self._normalizer is not None:
            if "action" in data and self.pad_action_dim_zero_mask is not None:
                action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
                if bool(action_is_pad.any().item()):
                    for key, dim_mask in self.pad_action_dim_zero_mask.items():
                        cur_action = data["action"][key]
                        cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                        cur_dim_mask = dim_mask.to(device=cur_action.device)
                        pad_zero_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                        cur_action[pad_zero_mask] = 0.0

            data = self.action_state_transform(data)

            # Route-aware forward: continuous dims normalize per-token
            # by the gameplay/gui normalizer; discrete dims go through
            # the global normalizer.
            if self.route_aware_action_normalization and "action" in data:
                if raw_state_for_route is None or self.gui_label_dim_index is None:
                    raise RuntimeError(
                        "Route-aware forward requested but raw_state or "
                        "gui_label_dim_index is missing — check the "
                        "preprocessor config."
                    )
                route_mask = raw_state_for_route[
                    :-1, int(self.gui_label_dim_index):int(self.gui_label_dim_index) + 1
                ].bool()
                if raw_state_field_valid_for_route is None:
                    raise RuntimeError("Route-aware normalization requires action_route_valid/state_field_valid.")
                route_valid = raw_state_field_valid_for_route[
                    :-1, int(self.gui_label_dim_index):int(self.gui_label_dim_index) + 1
                ].bool()
                data["action"] = self.normalizer.forward_action_by_route(
                    data["action"], route_mask,
                    self.route_norm, self.route_norm,
                    route_valid=route_valid,
                )
                data = self.normalizer.forward_state(data)
            else:
                data = self.normalizer.forward(data)
            data = self.action_state_merger.forward(data)

            if "action" in data:
                sample["action"] = data["action"]
                sample["action_is_pad"] = data["action_is_pad"]
                sample["action_dim_is_pad"] = data["action_dim_is_pad"]
                assert sample["action"].shape[-1] == self.action_output_dim

            sample["proprio"] = data["state"]
            sample["proprio_is_pad"] = data["state_is_pad"]
            sample["proprio_field_valid"] = raw_state_field_valid_for_route
            sample["proprio_dim_is_pad"] = torch.zeros(
                sample["proprio"].shape,
                dtype=torch.bool,
                device=sample["proprio"].device,
            )
            assert sample["proprio"].shape[-1] == self.proprio_output_dim

        if not is_video_only and self._normalizer is None and "action" in data:
            if self.pad_action_dim_zero_mask is not None:
                action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
                if bool(action_is_pad.any().item()):
                    for key, dim_mask in self.pad_action_dim_zero_mask.items():
                        cur_action = data["action"][key]
                        cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                        cur_dim_mask = dim_mask.to(device=cur_action.device)
                        pad_zero_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                        cur_action[pad_zero_mask] = 0.0

            data = self.action_state_transform(data)
            sample["action"] = data["action"]
            sample["action_is_pad"] = data["action_is_pad"]
            sample["proprio"] = data["state"]
            sample["proprio_is_pad"] = data["state_is_pad"]
            sample["proprio_field_valid"] = raw_state_field_valid_for_route
            if "state_dim_is_pad" in data:
                sample["state_dim_is_pad"] = data["state_dim_is_pad"]
                sample["proprio_dim_is_pad"] = data["state_dim_is_pad"]

        self._emit_auxiliary_action_fields(sample, pre_norm_action, raw_state_for_route, raw_state_field_valid_for_route)

        sample["idx"] = data["idx"]
        if "vae_cache_key" in data:
            sample["vae_cache_key"] = data["vae_cache_key"]

        for key in (
            "clip_frame_idx",
            "pre_anchor_event_frame_idxs_in_clip",
            "pre_anchor_event_times",
            "event_anchor_frame_idx_in_clip",
            "event_anchor_time",
            "event_key",
        ):
            if key in data:
                sample[key] = data[key]

        return sample

    def postprocess(
        self,
        data: Dict[str, Any],
        route_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Postprocess model output: unnormalize actions."""
        assert "action" in data, "Action is required in postprocess"
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        if route_mask is None and self.route_aware_action_normalization:
            # Fail-fast: a route-aware processor must receive a route
            # mask. Silent global fallback would denormalize in the
            # wrong space — drift between train/val (global) and
            # inference (route-aware).
            raise ValueError(
                "`postprocess` was called without a `route_mask` "
                "but the processor is in route-aware mode "
                "(`route_aware_action_normalization=True`). "
                "Silent fallback to global backward would produce "
                "a wrong-space raw action. Pass a `route_mask` "
                "(bool tensor of shape `[T, 1]` or `[B, T, 1]` "
                "aligned with the action) or set "
                "`route_aware_action_normalization=False` if you "
                "intentionally want global denormalization."
            )
        if route_mask is None:
            # Global path: route-aware is off, so global backward
            # is the correct denormalization space.
            data = self.normalizer.backward(data)
        else:
            # Route-aware path. Fail-fast if route_norm is missing
            # — caller passed a mask but the processor wasn't
            # properly initialized.
            if self.route_norm is None:
                raise ValueError(
                    "`postprocess` was called with a `route_mask` "
                    "but the processor's `route_norm` is None. "
                    "Either the processor was not initialized "
                    "with `set_normalizer_from_stats(route_aware "
                    "stats)`, or `route_aware_action_normalization` "
                    "is False. To denormalize the action without "
                    "route-aware, set `route_mask=None` for global "
                    "backward (and disable "
                    "`route_aware_action_normalization`), or fix "
                    "the dataset's processor init."
                )
            data["action"] = self.normalizer.backward_action_by_route(
                data["action"], route_mask,
                self.route_norm, self.route_norm,
            )
            data = self.normalizer.backward_state(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)
        return data


def _validate_discrete_dims_norm(
    resolved_mode,
    discrete_dims,
    action_key: str,
) -> None:
    """Fail-fast guard: when the action declares a mixed semantic
    schema (per-dim continuous / discrete split), the discrete dims
    MUST be normalized with ``min/max`` over a binary {0, 1} input.

    The joint-flow objective treats the normalized discrete dims as
    a {-1, +1} flow target, and the training metrics recover the
    binary label via ``0.5 * (x0 + 1)``. Both only hold if the
    forward normalizer maps {0, 1} to {-1, +1}, which is exactly the
    ``min/max`` affine. Any other mode (z-score, q01/q99,
    const_min/const_max, or a per-dim exception that overrides the
    discrete dims) would put those dims in a different space —
    not a numerical error, but a silent logic bug (e.g. z-score
    over {0, 1} maps to a std-scaled range, so the reconstructed
    label no longer lands on {0, 1}).

    ``resolved_mode`` is the output of
    ``LinearNormalizer._resolve_mode_spec``: either a scalar
    string (e.g. ``"min/max"``) or a list of
    ``{"dims": [...], "mode": str}`` dicts.
    """
    if _is_per_dim_mode_spec(resolved_mode):
        for dim in discrete_dims:
            entry = _find_mode_entry_for_dim(resolved_mode, dim)
            if entry is None:
                # Backfill in ``_resolve_mode_spec`` should
                # cover all dims; if not, normalizer bug.
                continue
            if not _is_minmax_mode(entry.get("mode")):
                raise ValueError(
                    f"Mixed action schema: "
                    f"discrete dim {dim} of action key "
                    f"{action_key!r} uses mode "
                    f"{entry.get('mode')!r}, but the joint-flow "
                    f"objective requires discrete dims to live in "
                    f"{{-1, +1}}, which only holds for `min/max` "
                    f"normalization. "
                    f"Either remove this per-dim exception, or "
                    f"mark this dim as continuous (not discrete)."
                )
    else:
        if not _is_minmax_mode(resolved_mode):
            raise ValueError(
                f"Mixed action schema: "
                f"action key {action_key!r} uses scalar mode "
                f"{resolved_mode!r}, but the discrete dims "
                f"({discrete_dims}) require `min/max` so they "
                f"land on {{-1, +1}} in the joint-flow target "
                f"space. Set "
                f"`norm_default_mode` to `min/max` or add a "
                f"per-dim exception for the discrete dims."
            )


def _is_per_dim_mode_spec(mode) -> bool:
    """A per-dim spec is a list of dicts each having a
    ``dims`` key. A scalar mode is a string.
    """
    return isinstance(mode, list) and all(
        isinstance(e, dict) and "dims" in e for e in mode
    )


def _find_mode_entry_for_dim(per_dim_spec, dim: int):
    """Return the per-dim entry that covers ``dim``, or None."""
    for entry in per_dim_spec:
        dims = entry.get("dims", [])
        if dim in dims:
            return entry
    return None


def _is_minmax_mode(mode) -> bool:
    """The default min/max mode name is ``"min/max"`` (with
    a slash). We do NOT accept the const-min/const-max
    ``"a/b"`` form for the discrete dims, even though it is
    also an affine map, because the endpoints wouldn't be
    {-1, +1} so the model's inverse transform would still
    be wrong.
    """
    if not isinstance(mode, str):
        return False
    return mode.strip().lower() == "min/max"
