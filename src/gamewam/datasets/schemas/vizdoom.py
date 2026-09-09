"""LeRobot feature contract for VizDoom datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from gamewam.datasets.lerobot.datasets.utils import write_info
from gamewam.envs.vizdoom.observations.state import STATE_DIM, STATE_GAME_VARS


VIZDOOM_STATE_METADATA_KEY = "vizdoom_state_schema"


def vizdoom_state_features() -> dict[str, dict[str, Any]]:
    names = list(STATE_GAME_VARS)
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": names,
        },
        "observation.state_dim_is_pad": {
            "dtype": "bool",
            "shape": (STATE_DIM,),
            "names": [f"{name}_is_pad" for name in names],
        },
    }


def vizdoom_state_metadata() -> dict[str, Any]:
    return {
        "field_names": list(STATE_GAME_VARS),
        "units": ["map_unit", "map_unit", "degree", "health_point"],
        "missing_value_mask": "observation.state_dim_is_pad",
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
        raise ValueError(f"Dataset is missing required feature {key!r}")
    actual_shape = tuple(feature.get("shape", ()))
    if actual_shape != shape:
        raise ValueError(
            f"Dataset feature {key!r} has shape {actual_shape}, expected {shape}."
        )
    actual_names = tuple(feature.get("names", ()))
    if actual_names != tuple(names):
        raise ValueError(
            f"Dataset feature {key!r} has names {actual_names}, "
            f"expected {tuple(names)}."
        )


def validate_vizdoom_state_schema(
    features: dict[str, dict[str, Any]], *, dataset_name: str
) -> None:
    for key, expected in vizdoom_state_features().items():
        try:
            _validate_feature(
                features,
                key,
                shape=tuple(expected["shape"]),
                names=expected["names"],
            )
        except ValueError as exc:
            raise ValueError(f"{dataset_name}: {exc}") from exc


def attach_or_validate_vizdoom_state_schema(dataset: Any) -> None:
    validate_vizdoom_state_schema(
        dataset.meta.features,
        dataset_name=str(dataset.root),
    )
    expected = vizdoom_state_metadata()
    existing = dataset.meta.info.get(VIZDOOM_STATE_METADATA_KEY)
    if existing is not None:
        if existing != expected:
            raise ValueError(
                "Existing dataset VizDoom state metadata does not match the "
                f"canonical schema: existing={existing!r}, expected={expected!r}"
            )
        return
    dataset.meta.info[VIZDOOM_STATE_METADATA_KEY] = expected
    write_info(dataset.meta.info, Path(dataset.root))
