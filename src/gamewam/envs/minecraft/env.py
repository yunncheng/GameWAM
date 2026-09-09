"""Minecraft environment wrapper for GameWAM deployment."""
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np

from gamewam.envs.base import GameEnvBase, GameObservation
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RewardsCallback
from gamewam.envs.minecraft.callbacks.init_inventory import InitInventoryCallback
from gamewam.envs.minecraft.callbacks.commands import CommandsCallback
from gamewam.envs.minecraft.callbacks.summon_mobs import SummonMobsCallback
from gamewam.envs.minecraft.action_space import ENV_NULL_ACTION, switched_hotbar_slot
from gamewam.envs.minecraft.observations.cursor import (
    DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL,
    DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL,
    MinecraftCursorTracker,
)
from gamewam.envs.minecraft.observations.state import (
    extract_gui_open,
    extract_state_with_validity,
    normalize_minecraft_info,
)

logger = logging.getLogger(__name__)


class MinecraftEnvironmentError(RuntimeError):
    """Transient MineStudio transport/process failure during an episode."""


def _raise_for_environment_error(info: Any, operation: str) -> None:
    """Turn MineStudio's swallowed socket failure into a retryable exception."""
    if not isinstance(info, Mapping):
        return
    error = info.get("error")
    if error:
        raise MinecraftEnvironmentError(f"MineStudio {operation} failed: {error}")


def _ensure_minestudio_action_contract() -> None:
    """Ensure MineStudio exposes every action in the GameWAM contract."""
    from minestudio.simulator.minerl.herobraine.env_specs import human_survival_specs

    simple_actions = human_survival_specs.SIMPLE_KEYBOARD_ACTION
    inverse_keymap = human_survival_specs.INVERSE_KEYMAP
    required_simple = [
        key
        for key in ENV_NULL_ACTION
        if key != "camera" and not key.startswith("hotbar.")
    ]
    unavailable = [key for key in required_simple if key not in inverse_keymap]
    if unavailable:
        raise RuntimeError(
            "MineStudio has no keyboard mapping for GameWAM actions: "
            f"{unavailable}."
        )
    for key in required_simple:
        if key not in simple_actions:
            simple_actions.append(key)


def _validate_minestudio_action_contract(sim: Any) -> None:
    action_space = getattr(getattr(sim, "env", None), "action_space", None)
    spaces = getattr(action_space, "spaces", None)
    if not isinstance(spaces, Mapping):
        return
    missing = sorted(set(ENV_NULL_ACTION) - set(spaces))
    if missing:
        raise RuntimeError(
            "MineStudio action space does not cover GameWAM actions: "
            f"missing={missing}."
        )


# Settings required to avoid unmeshed chunks after fast camera turns.
_PINNED_OPTIFINE_SETTINGS = {
    "ofLazyChunkLoading": "false",  # mesh all chunks in render distance
    "ofChunkUpdates": "5",          # max mesh rebuilds per frame
    "ofSmoothWorld": "false",       # do not throttle chunk updates
}


def _ensure_optifine_chunk_settings() -> None:
    """Pin OptiFine chunk-mesh settings into the instance runtime template."""
    from minestudio.utils.temp import get_mine_studio_dir

    runtime_dir = os.path.join(get_mine_studio_dir(), "runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    path = os.path.join(runtime_dir, "optionsof.txt")
    settings: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                key, sep, value = line.rstrip("\n").partition(":")
                if sep:
                    settings[key] = value
    if all(settings.get(k) == v for k, v in _PINNED_OPTIFINE_SETTINGS.items()):
        return
    settings.update(_PINNED_OPTIFINE_SETTINGS)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for key, value in settings.items():
            f.write(f"{key}:{value}\n")
    os.replace(tmp_path, path)


def _extract_image(obs: Any, info: Optional[Dict[str, Any]] = None):
    """Extract the best available image from Minestudio outputs."""
    if isinstance(obs, dict):
        for key in ("image", "pov", "rgb"):
            image = obs.get(key)
            if image is not None:
                return image
    if isinstance(info, dict):
        for key in ("pov", "rgb", "image"):
            image = info.get(key)
            if image is not None:
                return image
    if not isinstance(obs, dict):
        return obs
    return None


class MinecraftGameEnv(GameEnvBase):
    """Run MineStudio through the shared GameWAM interface."""

    def __init__(
        self,
        task_config: dict,
        render_size: tuple = (640, 360),
        fps: int = 20,
        seed: int = 0,
        camera_cfg=None,
        cursor_reference_size: Optional[tuple[float, float]] = None,
        cursor_pitch_deg_per_pixel: float = DEFAULT_GUI_CAMERA_PITCH_DEG_PER_PIXEL,
        cursor_yaw_deg_per_pixel: float = DEFAULT_GUI_CAMERA_YAW_DEG_PER_PIXEL,
        **kwargs,
    ):
        self.task_config = task_config
        self.render_size = render_size
        self.fps = fps
        self.seed = seed
        self.camera_cfg = camera_cfg
        self.cursor_reference_size = tuple(cursor_reference_size or render_size)
        self._cursor_tracker = MinecraftCursorTracker(
            render_size=self.render_size,
            reference_size=self.cursor_reference_size,
            pitch_deg_per_pixel=cursor_pitch_deg_per_pixel,
            yaw_deg_per_pixel=cursor_yaw_deg_per_pixel,
        )
        self._env = None
        self._step_count = 0
        self._last_obs = None
        self._last_info: Dict[str, Any] = {}
        self._current_hotbar = 0
        self._current_hotbar_valid = False

    def _inject_runtime_hotbar(
        self,
        info: Dict[str, Any],
        *,
        env_action: Optional[Dict[str, Any]] = None,
        previous_gui_open: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Inject selected hotbar state absent from MineStudio observations."""
        out = dict(info)
        observed = out.get("current_hotbar")
        if observed is not None:
            observed_f = float(observed)
            observed_i = int(observed_f)
            if observed_f != float(observed_i) or not 0 <= observed_i <= 8:
                raise ValueError(
                    "MineStudio current hotbar must be an integer in [0, 8], "
                    f"got {observed!r}."
                )
            self._current_hotbar = observed_i
            self._current_hotbar_valid = True
        elif env_action is not None and previous_gui_open is not True:
            switched = switched_hotbar_slot(env_action)
            if switched is not None:
                self._current_hotbar = int(switched)
                self._current_hotbar_valid = True

        if self._current_hotbar_valid:
            out["current_hotbar"] = int(self._current_hotbar)
        else:
            out.pop("current_hotbar", None)
        return out

    def reset(self, **kwargs) -> GameObservation:
        """Initialize and reset the Minecraft environment."""
        seed = self.task_config.get("seed", self.seed)
        init_actions = self.task_config.get("init_actions", [])
        callback_cfg = self.task_config.get("callback", {})
        inv_cb_cfg = callback_cfg.get("init_inventory", {})
        init_inventory = inv_cb_cfg.get("init_inventory", [])
        commands = callback_cfg.get("commands", [])
        reward_cfg = self.task_config.get("rewards", {})

        callbacks = [
            InitInventoryCallback(
                init_inventory,
                inventory_distraction_level=inv_cb_cfg.get("inventory_distraction_level", [0]),
                equip_distraction_level=inv_cb_cfg.get("equip_distraction_level", [0]),
                forbidden_slots=inv_cb_cfg.get("forbidden_slots", []),
                seed=inv_cb_cfg.get("seed", seed),
            ),
            RewardsCallback(reward_cfg),
            CommandsCallback(commands),
        ]

        mobs = callback_cfg.get("mobs")
        if mobs:
            callbacks.append(
                SummonMobsCallback(mobs, seed=callback_cfg.get("mobs_seed", seed))
            )

        _ensure_minestudio_action_contract()
        _ensure_optifine_chunk_settings()
        if self._env is not None:
            self._env.close()
            self._env = None
        self._env = MinecraftSim(
            action_type="env",
            obs_size=self.render_size,
            render_size=self.render_size,
            seed=seed,
            preferred_spawn_biome=None,
            camera_config=self.camera_cfg,
            callbacks=callbacks,
        )
        _validate_minestudio_action_contract(self._env)

        obs, info = self._env.reset()
        _raise_for_environment_error(info, "reset")
        normalized_info = normalize_minecraft_info(info) if isinstance(info, dict) else {}
        # MineStudio omits the selected slot, so track it from sent actions.
        self._current_hotbar = 0
        self._current_hotbar_valid = True
        normalized_info = self._inject_runtime_hotbar(normalized_info)
        self._cursor_tracker.reset()
        self._cursor_tracker.observe_reset(extract_gui_open(normalized_info))

        for action in init_actions:
            time.sleep(0.1)
            previous_gui_open = self._cursor_tracker.gui_open
            obs, reward, terminated, truncated, info = self._env.step(action)
            _raise_for_environment_error(info, "initialization step")
            normalized_info = normalize_minecraft_info(info) if isinstance(info, dict) else {}
            normalized_info = self._inject_runtime_hotbar(
                normalized_info,
                env_action=action,
                previous_gui_open=previous_gui_open,
            )
            self._cursor_tracker.update_after_action(
                action,
                extract_gui_open(normalized_info),
            )

        self._step_count = 0
        self._last_obs = obs
        self._last_info = self._cursor_tracker.inject(normalized_info)

        image = _extract_image(obs, self._last_info)
        proprio, proprio_field_valid = extract_state_with_validity(
            self._last_info,
            cursor_reference_size=self.cursor_reference_size,
        )

        return GameObservation(
            image=image,
            proprio=proprio,
            proprio_field_valid=proprio_field_valid,
            info=self._last_info,
            reward=0.0,
            done=False,
            timestamp=0,
        )

    def step(self, action: Any) -> GameObservation:
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if isinstance(action, list):
            env_actions = action
        else:
            env_actions = [action]

        total_reward = 0.0
        done = False
        for env_action in env_actions:
            previous_gui_open = self._cursor_tracker.gui_open
            obs, reward, terminated, truncated, info = self._env.step(env_action)
            _raise_for_environment_error(info, "step")
            total_reward += reward
            done = terminated or truncated
            self._step_count += 1
            self._last_obs = obs
            normalized_info = normalize_minecraft_info(info) if isinstance(info, dict) else {}
            normalized_info = self._inject_runtime_hotbar(
                normalized_info,
                env_action=env_action,
                previous_gui_open=previous_gui_open,
            )
            self._cursor_tracker.update_after_action(
                env_action,
                extract_gui_open(normalized_info),
            )
            self._last_info = self._cursor_tracker.inject(normalized_info)
            if done:
                break

        image = _extract_image(self._last_obs, self._last_info)
        proprio, proprio_field_valid = extract_state_with_validity(
            self._last_info,
            cursor_reference_size=self.cursor_reference_size,
        )

        return GameObservation(
            image=image,
            proprio=proprio,
            proprio_field_valid=proprio_field_valid,
            info=self._last_info,
            reward=total_reward,
            done=done,
            timestamp=self._step_count,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def get_screenshot(self) -> np.ndarray:
        if self._last_obs is not None:
            image = _extract_image(self._last_obs, self._last_info)
            if image is not None:
                return image
        return np.zeros((*self.render_size[::-1], 3), dtype=np.uint8)

    @property
    def game_type(self) -> str:
        return "minecraft"
