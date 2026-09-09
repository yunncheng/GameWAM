"""Runtime GUI cursor tracking for Minecraft."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .state import normalize_screen_size

DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL = 0.145
DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL = 0.144


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


class MinecraftCursorTracker:
    """Track the GUI cursor in a configured reference coordinate system."""

    def __init__(
        self,
        *,
        render_size: Sequence[float],
        reference_size: Sequence[float],
        pitch_deg_per_pixel: float = DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL,
        yaw_deg_per_pixel: float = DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL,
    ) -> None:
        self.render_width, self.render_height = normalize_screen_size(
            render_size, name="render_size"
        )
        self.reference_width, self.reference_height = normalize_screen_size(
            reference_size, name="cursor_reference_size"
        )
        self.pitch_deg_per_pixel = _finite_float(
            pitch_deg_per_pixel, "cursor_pitch_deg_per_pixel"
        )
        self.yaw_deg_per_pixel = _finite_float(
            yaw_deg_per_pixel, "cursor_yaw_deg_per_pixel"
        )
        if self.pitch_deg_per_pixel <= 0.0 or self.yaw_deg_per_pixel <= 0.0:
            raise ValueError("GUI cursor degrees-per-pixel calibration must be positive")
        self.reset()

    @property
    def reference_size(self) -> tuple[float, float]:
        return self.reference_width, self.reference_height

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.valid = False
        self.gui_open: Optional[bool] = None

    def observe_reset(self, gui_open: Optional[bool]) -> None:
        self.gui_open = gui_open
        if gui_open is True:
            self.x = self.reference_width / 2.0
            self.y = self.reference_height / 2.0
            self.valid = True
        else:
            self.x = 0.0
            self.y = 0.0
            self.valid = False

    def update_after_action(
        self,
        env_action: Dict[str, Any],
        next_gui_open: Optional[bool],
    ) -> None:
        previous_gui_open = self.gui_open
        self.gui_open = next_gui_open
        if next_gui_open is not True:
            self.x = 0.0
            self.y = 0.0
            self.valid = False
            return
        if previous_gui_open is not True or not self.valid:
            self.x = self.reference_width / 2.0
            self.y = self.reference_height / 2.0
            self.valid = True
            return

        camera = np.asarray(env_action.get("camera", (0.0, 0.0)), dtype=np.float64)
        if camera.shape != (2,) or not bool(np.isfinite(camera).all()):
            raise ValueError(
                f"Minecraft GUI camera action must be finite [pitch, yaw], got {camera!r}"
            )
        pitch_delta, yaw_delta = float(camera[0]), float(camera[1])
        self.x += (
            yaw_delta / self.yaw_deg_per_pixel
            * self.reference_width / self.render_width
        )
        self.y += (
            pitch_delta / self.pitch_deg_per_pixel
            * self.reference_height / self.render_height
        )
        self.x = float(np.clip(self.x, 0.0, self.reference_width))
        self.y = float(np.clip(self.y, 0.0, self.reference_height))

    def inject(self, info: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(info)
        result.pop("cursor_x", None)
        result.pop("cursor_y", None)
        if self.valid and self.gui_open is True:
            result["cursor_x"] = float(self.x)
            result["cursor_y"] = float(self.y)
        return result
