# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import collections
import os

import torch

_WORK_DIR: str | None = None
_DEFAULT_WORK_DIR = "./runs/"

_MIXED_PRECISION_VALUES = {"no", "fp16", "bf16"}

def register_work_dir(path: str | os.PathLike | None) -> None:
    global _WORK_DIR
    _WORK_DIR = str(path) if path is not None else None
    os.makedirs(path, exist_ok=True)


def get_work_dir() -> str | None:
    return _WORK_DIR if _WORK_DIR is not None else _DEFAULT_WORK_DIR


def normalize_mixed_precision(mixed_precision: str) -> str:
    """Validate and normalize ``mixed_precision`` to one of ``{"no", "fp16", "bf16"}``."""
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in _MIXED_PRECISION_VALUES:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    """Map a normalized ``mixed_precision`` string to its ``torch.dtype``."""
    precision = normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class LRUDict:
    """Bounded LRU dict that evicts oldest entries when full.

    Mirrors the ``dict.get(key, default)`` and ``dict[key] = value`` interface
    so it can replace a plain ``dict`` without changing call sites.

    Args:
        max_size: Maximum number of entries before eviction.
    """

    def __init__(self, max_size: int = 200):
        self._data: collections.OrderedDict = collections.OrderedDict()
        self._max_size = max(1, int(max_size))

    def get(self, key, default=None):
        try:
            val = self._data[key]
        except KeyError:
            return default
        self._data.move_to_end(key)
        return val

    def __setitem__(self, key, value):
        if key in self._data:
            del self._data[key]
        elif len(self._data) >= self._max_size:
            self._data.popitem(last=False)
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key) -> bool:
        return key in self._data
