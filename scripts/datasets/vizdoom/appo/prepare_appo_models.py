#!/usr/bin/env python3
"""Safely download and prepare four public VizDoom APPO checkpoints.

No checkpoint pickle is executed: loading is restricted to tensors and a small
NumPy scalar/dtype allow-list, then the result is sanitized and re-saved.  Two
official Sample Factory 2.0 Battle checkpoints also need an fc_layers ->
mlp_layers state-dict key migration for Sample Factory 2.1.1.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from appo_common import MODEL_REPOS


def _safe_globals() -> list[Any]:
    values: list[Any] = [np.core.multiarray.scalar, np.dtype]
    dtype_cls = getattr(getattr(np, "dtypes", None), "Float64DType", None)
    if dtype_cls is not None:
        values.append(dtype_cls)
    return values


def _safe_load(path: Path) -> dict[str, Any]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError("Preparing downloaded models requires PyTorch with weights_only=True support")
    add_safe = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe is None:
        raise RuntimeError(
            "Preparing legacy checkpoints requires torch.serialization.add_safe_globals "
            "(PyTorch >=2.6 recommended). Do not fall back to weights_only=False."
        )
    add_safe(_safe_globals())
    return torch.load(path, map_location="cpu", weights_only=True)


def _sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, OrderedDict):
        return OrderedDict((key, _sanitize(item)) for key, item in value.items())
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    return value


def _migrate_model_keys(checkpoint: dict[str, Any]) -> int:
    state = checkpoint["model"]
    migrated = 0
    for old_key in list(state):
        if "encoder.basic_encoder.enc.fc_layers." not in old_key:
            continue
        new_key = old_key.replace(
            "encoder.basic_encoder.enc.fc_layers.",
            "encoder.basic_encoder.enc.mlp_layers.",
        )
        if new_key in state:
            raise RuntimeError(f"Both legacy and current keys exist: {old_key}, {new_key}")
        state[new_key] = state.pop(old_key)
        migrated += 1
    return migrated


def _choose_checkpoint(download_dir: Path) -> Path:
    best = sorted((download_dir / "checkpoint_p0").glob("best_*.pth"))
    candidates = best or sorted((download_dir / "checkpoint_p0").glob("checkpoint_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found below {download_dir}")
    return candidates[-1]


def prepare_one(scenario: str, output_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    repo = MODEL_REPOS[scenario]
    # Raw checkpoints go to the default Hugging Face cache; only the
    # sanitized output written below is consumed by collection.
    source = Path(snapshot_download(repo_id=repo))

    config_path = source / "config.json"
    if not config_path.is_file():
        config_path = source / "cfg.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    destination = output_root / scenario
    checkpoint_dir = destination / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["experiment"] = scenario
    config["train_dir"] = str(output_root.resolve())
    config["device"] = "cpu"
    config["actor_worker_gpus"] = []
    config["env_gpu_actions"] = False
    config["env_gpu_observations"] = False
    with (destination / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    checkpoint = _sanitize(_safe_load(_choose_checkpoint(source)))
    migrated = _migrate_model_keys(checkpoint)
    env_steps = int(checkpoint.get("env_steps", 0))
    prepared = checkpoint_dir / f"checkpoint_prepared_{env_steps:012d}.pth"
    torch.save(checkpoint, prepared)
    # Validate that the output no longer needs any pickle allow-list.
    torch.load(prepared, map_location="cpu", weights_only=True)
    shutil.copy2(prepared, checkpoint_dir / f"best_prepared_{env_steps:012d}.pth")
    print(f"[{scenario}] {repo} -> {destination} (env_steps={env_steps}, migrated_keys={migrated})")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=sorted(MODEL_REPOS), default=sorted(MODEL_REPOS))
    parser.add_argument("--output-root", type=Path, default=Path("data/vizdoom_appo_models"))
    args = parser.parse_args()
    for scenario in args.scenarios:
        prepare_one(scenario, args.output_root)


if __name__ == "__main__":
    main()

