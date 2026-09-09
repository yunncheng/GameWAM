"""Loss-minimizing mapping from GameWAM's Minecraft 22-D actions to Craftium."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from gamewam.envs.minecraft.action_space import ENV_NULL_ACTION

CRAFTIUM_ACTION_KEYS = (
    "forward",
    "backward",
    "left",
    "right",
    "jump",
    "aux1",
    "sneak",
    "zoom",
    "dig",
    "place",
    "drop",
    "inventory",
    "slot_1",
    "slot_2",
    "slot_3",
    "slot_4",
    "slot_5",
    "slot_6",
    "slot_7",
    "slot_8",
    "slot_9",
    "mouse",
)

_KEY_MAP = {
    "forward": "forward",
    "back": "backward",
    "left": "left",
    "right": "right",
    "jump": "jump",
    "sneak": "sneak",
    "sprint": "aux1",
    "attack": "dig",
    "use": "place",
    "drop": "drop",
    "inventory": "inventory",
    **{f"hotbar.{slot}": f"slot_{slot}" for slot in range(1, 10)},
}


@dataclass(frozen=True)
class CameraCalibration:
    """Pinned Luanti camera settings and their exact pixel conversion."""

    width: int = 640
    height: int = 360
    vertical_fov_deg: float = 90.0
    mouse_sensitivity: float = 0.2

    def __post_init__(self) -> None:
        if int(self.width) <= 1 or int(self.height) <= 1:
            raise ValueError("Camera dimensions must both be greater than one")
        if not 1.0 <= float(self.vertical_fov_deg) <= 160.0:
            raise ValueError("vertical_fov_deg must be in [1, 160]")
        if not 0.001 <= float(self.mouse_sensitivity) <= 10.0:
            raise ValueError("mouse_sensitivity must be in [0.001, 10]")

    @property
    def sensitivity_scale(self) -> float:
        # Exact formula used by Craftium v0.0.1's Luanti client. The constant
        # makes scale=1 around Luanti's historical 72-degree 16:9 default.
        return math.tan(math.radians(self.vertical_fov_deg) / 2.0) * 1.3763819

    @property
    def degrees_per_pixel(self) -> float:
        return float(self.mouse_sensitivity) * self.sensitivity_scale

    @property
    def yaw_degrees_per_unit(self) -> float:
        return (int(self.width) // 2) * self.degrees_per_pixel

    @property
    def pitch_degrees_per_unit(self) -> float:
        return (int(self.height) // 2) * self.degrees_per_pixel


@dataclass(frozen=True)
class AdaptedAction:
    """Craftium command plus the quantized action actually executed."""

    craftium_action: dict[str, Any]
    effective_minecraft_action: dict[str, Any]
    mouse_pixel_delta: tuple[int, int]
    camera_clipped: bool


class VoxeLibreActionAdapter:
    """Translate Minecraft action semantics without changing model outputs.

    Minecraft camera deltas use ``[pitch_down, yaw_right]`` in degrees.
    Craftium accepts normalized mouse ``[x_right, y_up]`` and internally
    quantizes it to integer pixels. We explicitly reproduce that quantization
    so history records what the environment could actually execute.
    """

    def __init__(self, calibration: CameraCalibration | None = None):
        self.calibration = calibration or CameraCalibration()

    def adapt(self, action: Mapping[str, Any]) -> AdaptedAction:
        unknown = set(action) - set(ENV_NULL_ACTION)
        if unknown:
            raise ValueError(f"Unknown Minecraft action key(s): {sorted(unknown)}")

        camera = np.asarray(action.get("camera", (0.0, 0.0)), dtype=np.float64)
        if camera.shape != (2,) or not np.isfinite(camera).all():
            raise ValueError(
                "Minecraft camera action must be a finite [pitch, yaw] pair, "
                f"got shape={camera.shape}, value={camera!r}"
            )
        pitch_deg, yaw_deg = float(camera[0]), float(camera[1])

        cal = self.calibration
        raw_mouse_x = yaw_deg / cal.yaw_degrees_per_unit
        # Craftium negates its public mouse-y before injecting SDL motion.
        raw_mouse_y = -pitch_deg / cal.pitch_degrees_per_unit
        mouse_x = float(np.clip(raw_mouse_x, -1.0, 1.0))
        mouse_y = float(np.clip(raw_mouse_y, -1.0, 1.0))
        clipped = not (
            math.isclose(mouse_x, raw_mouse_x, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(mouse_y, raw_mouse_y, rel_tol=0.0, abs_tol=1e-12)
        )

        # Mirror CraftiumEnv.step exactly: x is direct; public y is negated.
        pixel_x = int(mouse_x * (int(cal.width) // 2))
        pixel_y = int((-mouse_y) * (int(cal.height) // 2))
        effective_pitch = pixel_y * cal.degrees_per_pixel
        effective_yaw = pixel_x * cal.degrees_per_pixel

        craftium_action: dict[str, Any] = {
            key: 0 for key in CRAFTIUM_ACTION_KEYS if key != "mouse"
        }
        for source, target in _KEY_MAP.items():
            craftium_action[target] = int(bool(action.get(source, 0)))
        craftium_action["mouse"] = np.asarray([mouse_x, mouse_y], dtype=np.float32)

        effective = dict(ENV_NULL_ACTION)
        for source in _KEY_MAP:
            effective[source] = int(bool(action.get(source, 0)))
        effective["camera"] = np.asarray(
            [effective_pitch, effective_yaw], dtype=np.float32
        )

        return AdaptedAction(
            craftium_action=craftium_action,
            effective_minecraft_action=effective,
            mouse_pixel_delta=(pixel_x, pixel_y),
            camera_clipped=clipped,
        )
