"""AsinhTransform — compress heavy-tailed action dimensions before min/max normalization.

Forward:  x' = asinh(x / scale)
Backward: x  = sinh(x')  * scale

Typical use: camera-delta dims in Minecraft have teleport/respawn spikes.
asinh compresses the long tail while preserving near-zero resolution.
"""
from typing import Dict, List

import torch


class AsinhTransform:
    def __init__(self, key: str, dims: List[int], scale: float):
        """
        Args:
            key:  action dict key (e.g. "default")
            dims: dimension indices to transform (e.g. [0, 1] for camera)
            scale: asinh scaling factor.  x' = asinh(x / scale).
                   Choose ~std of the dimension for balanced compression.
        """
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.key = key
        self.dims = list(dims)
        self.scale = float(scale)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
        if "action" not in batch:
            return batch
        tensor = batch["action"].get(self.key)
        if tensor is None:
            return batch
        tensor[..., self.dims] = torch.asinh(tensor[..., self.dims] / self.scale)
        return batch

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
        if "action" not in batch:
            return batch
        tensor = batch["action"].get(self.key)
        if tensor is None:
            return batch
        tensor[..., self.dims] = torch.sinh(tensor[..., self.dims]) * self.scale
        return batch
