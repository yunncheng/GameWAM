"""Shared LeRobot schema helpers for Minecraft raw state datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from gamewam.datasets.lerobot.datasets.utils import write_info
from gamewam.envs.minecraft.observations.state import (
    CANONICAL_FIELD_NAMES,
    FIELD_NAMES,
    RAW_STATE_DIM,
    STATE_FIELD_VALID_NAMES,
    normalize_screen_size,
)


MINECRAFT_STATE_METADATA_KEY = "minecraft_state_schema"


def resolve_cursor_reference_size(
    configured_size: Optional[Sequence[float]],
    *,
    fallback_size: Sequence[float],
) -> tuple[float, float]:
    """Resolve configured cursor reference size, defaulting to source video size."""
    return normalize_screen_size(
        configured_size if configured_size is not None else fallback_size,
        name="cursor_reference_size",
    )


def minecraft_state_features() -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (RAW_STATE_DIM,),
            "names": list(FIELD_NAMES),
        },
        "observation.state_field_valid": {
            "dtype": "bool",
            "shape": (RAW_STATE_DIM,),
            "names": list(STATE_FIELD_VALID_NAMES),
        },
        "observation.state_dim_is_pad": {
            "dtype": "bool",
            "shape": (RAW_STATE_DIM,),
            "names": [f"{name}_is_pad" for name in FIELD_NAMES],
        },
    }


def minecraft_state_metadata(reference_size: Sequence[float]) -> dict[str, Any]:
    width, height = normalize_screen_size(reference_size, name="cursor_reference_size")
    return {
        "field_names": list(FIELD_NAMES),
        "field_valid_names": list(STATE_FIELD_VALID_NAMES),
        "canonical_field_names": list(CANONICAL_FIELD_NAMES),
        "cursor_reference_size": [width, height],
        "cursor_coordinate_convention": "continuous_edge_hotspot",
        "cursor_canonical_normalization": "centered_minus_one_to_one",
    }


def _validate_feature(
    features: dict[str, dict[str, Any]],
    key: str,
    *,
    shape: tuple[int, ...],
    names: Sequence[str],
) -> None:
    feature = features.get(key)
    if feature is None:
        raise ValueError(f"Existing dataset is missing required feature {key!r}")
    actual_shape = tuple(feature.get("shape", ()))
    if actual_shape != shape:
        raise ValueError(
            f"Existing dataset feature {key!r} has shape {actual_shape}, expected {shape}. "
            "Use a new output directory for the 6-D Minecraft state schema."
        )
    actual_names = tuple(feature.get("names", ()))
    if actual_names != tuple(names):
        raise ValueError(
            f"Existing dataset feature {key!r} has names {actual_names}, "
            f"expected {tuple(names)}. Use a new output directory."
        )


def attach_or_validate_minecraft_state_schema(
    dataset: Any,
    *,
    cursor_reference_size: Sequence[float],
    cursor_source_metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Validate feature/schema compatibility and persist cursor reference metadata."""
    expected_features = minecraft_state_features()
    for key, expected in expected_features.items():
        _validate_feature(
            dataset.meta.features,
            key,
            shape=tuple(expected["shape"]),
            names=expected["names"],
        )

    expected_metadata = minecraft_state_metadata(cursor_reference_size)
    existing_metadata = dataset.meta.info.get(MINECRAFT_STATE_METADATA_KEY)
    metadata_changed = False
    if existing_metadata is not None:
        existing_reference = existing_metadata.get("cursor_reference_size")
        if existing_reference is None or not np.allclose(
            np.asarray(existing_reference, dtype=np.float64),
            np.asarray(expected_metadata["cursor_reference_size"], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Existing dataset cursor_reference_size does not match this conversion: "
                f"existing={existing_reference}, "
                f"requested={expected_metadata['cursor_reference_size']}."
            )
        for key in (
            "field_names",
            "field_valid_names",
            "canonical_field_names",
            "cursor_coordinate_convention",
            "cursor_canonical_normalization",
        ):
            if existing_metadata.get(key) != expected_metadata[key]:
                raise ValueError(
                    f"Existing dataset Minecraft state metadata mismatch for {key!r}: "
                    f"existing={existing_metadata.get(key)!r}, expected={expected_metadata[key]!r}"
                )
        if cursor_source_metadata is not None:
            existing_source = existing_metadata.get("cursor_source")
            if existing_source is not None and existing_source != cursor_source_metadata:
                raise ValueError(
                    "Existing dataset cursor source metadata does not match this producer: "
                    f"existing={existing_source!r}, requested={cursor_source_metadata!r}"
                )
            if existing_source is None:
                existing_metadata["cursor_source"] = cursor_source_metadata
                metadata_changed = True
    else:
        if cursor_source_metadata is not None:
            expected_metadata["cursor_source"] = cursor_source_metadata
        dataset.meta.info[MINECRAFT_STATE_METADATA_KEY] = expected_metadata
        metadata_changed = True

    if metadata_changed:
        write_info(dataset.meta.info, Path(dataset.root))


def validate_minecraft_state_schema(
    *,
    features: dict[str, dict[str, Any]],
    info: dict[str, Any],
    cursor_reference_size: Sequence[float],
    dataset_name: str,
) -> None:
    """Read-only validation used by training dataset construction."""
    expected_features = minecraft_state_features()
    for key, expected in expected_features.items():
        try:
            _validate_feature(
                features,
                key,
                shape=tuple(expected["shape"]),
                names=expected["names"],
            )
        except ValueError as exc:
            raise ValueError(f"{dataset_name}: {exc}") from exc

    expected = minecraft_state_metadata(cursor_reference_size)
    actual = info.get(MINECRAFT_STATE_METADATA_KEY)
    if actual is None:
        raise ValueError(
            f"{dataset_name}: missing {MINECRAFT_STATE_METADATA_KEY!r} metadata; "
            "cursor training requires explicit field and reference metadata."
        )
    for key in (
        "field_names",
        "field_valid_names",
        "canonical_field_names",
        "cursor_coordinate_convention",
        "cursor_canonical_normalization",
    ):
        if actual.get(key) != expected[key]:
            raise ValueError(
                f"{dataset_name}: Minecraft state metadata mismatch for {key}: "
                f"actual={actual.get(key)!r}, expected={expected[key]!r}."
            )
    actual_reference = actual.get("cursor_reference_size")
    if actual_reference is None or not np.allclose(
        np.asarray(actual_reference, dtype=np.float64),
        np.asarray(expected["cursor_reference_size"], dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{dataset_name}: cursor_reference_size mismatch: "
            f"dataset={actual_reference!r}, "
            f"processor={expected['cursor_reference_size']!r}."
        )
