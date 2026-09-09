"""BaseProcessor — abstract base class for game data processors."""

from abc import ABC
from typing import Dict, Any, Optional, List

from omegaconf import OmegaConf
from ..utils.normalizer import LinearNormalizer, NormMode


class BaseProcessor(ABC):
    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_frames: int,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,
        action_state_transforms: Optional[List[Any]],
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]],
        action_state_merger,
        train_transforms: Dict[str, List[Any]] | None = None,
        val_transforms: Dict[str, List[Any]] | None = None,
        drop_high_level_prob: Optional[float] = None,
        use_zh_instruction: bool = False,
        tokenizer: Optional[Any] = None,
    ):
        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.shape_meta = shape_meta
        self.num_frames = num_frames
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        self.action_state_merger = action_state_merger
        if self.action_state_merger is not None:
            self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer = None

        self.tokenizer = tokenizer

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() or eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("normalizer has not been set. Please call set_normalizer_from_stats() first.")
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
        )
        # Subclasses (e.g. ``GameWAMProcessor``) build extra per-route
        # state from the same stats in this hook.
        self._post_set_normalizer_hook(dataset_stats)

    def augment_instruction(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."

        if self.action_state_transforms is not None:
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)

        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."

        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."

        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
