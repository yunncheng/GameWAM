"""VizDoom environment wrapper for GameWAM deployment."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from gamewam.envs.base import GameEnvBase, GameObservation
from gamewam.envs.vizdoom.observations import depth as depth_rendering
from gamewam.envs.vizdoom.action_space import (
    ACTION_DIM,
    TURN_DELTA_MAX_DEG,
    VIZDOOM_BUTTONS,
    vector_to_buttons,
)
from gamewam.envs.vizdoom.assets import DEFAULT_SCENARIO, resolve_scenario
from gamewam.envs.vizdoom.observations.state import STATE_DIM, STATE_GAME_VARS, extract_state

INFO_GAME_VARS = list(STATE_GAME_VARS) + ["KILLCOUNT", "AMMO2"]

_NON_MONSTER_NAMES = frozenset({
    "DoomPlayer", "BulletPuff", "Blood", "Clip", "CustomMedikit", "Medikit",
    "Stimpack", "ExplosiveBarrel", "BurningBarrel", "DoomImpBall", "Rocket",
    "PlasmaBall", "ArachnotronPlasma", "Shell", "ClipBox", "RocketAmmo",
    "Cell", "CellPack", "ShellBox", "Backpack",
})


def _is_monster(object_name: str) -> bool:
    """True if a VizDoom label refers to a living enemy (not an item/effect)."""
    return object_name not in _NON_MONSTER_NAMES

logger = logging.getLogger(__name__)

class VizDoomGameEnv(GameEnvBase):
    """Run a VizDoom scenario through the shared GameWAM interface."""

    def __init__(
        self,
        scenario: str = DEFAULT_SCENARIO,
        frame_skip: int = 1,
        cell: int = depth_rendering.DEFAULT_CELL,
        state_dim: int = STATE_DIM,
        seed: int = 0,
        window_visible: bool = False,
        enable_labels: bool = False,
        obs_mode: str = "rgb",
        screen_size: tuple = (480, 640),
        **kwargs,
    ):
        self.scenario = scenario
        self.frame_skip = int(frame_skip)
        if self.frame_skip <= 0:
            raise ValueError(f"frame_skip must be positive, got {self.frame_skip}.")
        self.cell = int(cell)
        self.state_dim = int(state_dim)
        if self.state_dim != STATE_DIM:
            raise ValueError(
                f"VizDoom APPO proprio must have {STATE_DIM} dimensions, "
                f"got {self.state_dim}."
            )
        self.seed = int(seed)
        self.window_visible = bool(window_visible)
        self.enable_labels = bool(enable_labels)
        if obs_mode not in ("rgb", "depth"):
            raise ValueError(f"obs_mode must be rgb|depth, got {obs_mode}")
        self.obs_mode = obs_mode
        if len(screen_size) != 2:
            raise ValueError(f"screen_size must be (height, width), got {screen_size!r}")
        self.screen_h, self.screen_w = int(screen_size[0]), int(screen_size[1])
        if self.screen_h <= 0 or self.screen_w <= 0:
            raise ValueError(f"screen_size must be positive, got {screen_size!r}")
        self._game = None
        self._step_count = 0
        self._last_depth_grid: Optional[np.ndarray] = None
        self._last_screen: Optional[np.ndarray] = None
        self._last_info: Dict[str, Any] = {}

    # ── lifecycle ────────────────────────────────────────────────────
    def _make_game(self):
        import vizdoom as vzd

        game = vzd.DoomGame()

        cfg_path, wad_path = resolve_scenario(self.scenario)
        game.load_config(cfg_path)
        if wad_path is not None:
            game.set_doom_scenario_path(wad_path)

        game.set_seed(self.seed)
        game.set_window_visible(self.window_visible)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.set_render_hud(True)
        game.set_render_weapon(True)
        # Battle checkpoints were trained with the crosshair enabled.
        game.set_render_crosshair(self.scenario in ("battle1", "battle2"))
        game.set_depth_buffer_enabled(True)
        if self.enable_labels:
            game.set_labels_buffer_enabled(True)

        game.set_available_buttons([getattr(vzd.Button, b) for b in VIZDOOM_BUTTONS])
        game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, TURN_DELTA_MAX_DEG)

        existing = set(game.get_available_game_variables())
        for name in INFO_GAME_VARS:
            gv = getattr(vzd.GameVariable, name)
            if gv not in existing:
                game.add_available_game_variable(gv)

        game.init()
        return game

    def reset(self, **kwargs) -> GameObservation:
        if self._game is None:
            self._game = self._make_game()
        self._game.new_episode()
        self._step_count = 0
        return self._observe(reward=0.0, done=False)

    def step(self, action: Any) -> GameObservation:
        if self._game is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        buttons = self._coerce_action(action)
        reward = float(self._game.make_action(buttons, self.frame_skip))
        self._step_count += 1
        done = bool(self._game.is_episode_finished())
        return self._observe(reward=reward, done=done)

    def get_game_variable(self, name: str) -> float:
        """Read a VizDoom game variable by name (e.g. 'KILLCOUNT'). Valid even
        after the episode finished — returns the final value."""
        import vizdoom as vzd

        return float(self._game.get_game_variable(getattr(vzd.GameVariable, name)))

    def get_total_reward(self) -> float:
        """Return VizDoom's cumulative reward for the current episode."""
        if self._game is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return float(self._game.get_total_reward())

    def close(self) -> None:
        if self._game is not None:
            self._game.close()
            self._game = None

    def get_screenshot(self) -> np.ndarray:
        zeros = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        if self.obs_mode == "rgb":
            return self._last_screen if self._last_screen is not None else zeros
        if self._last_depth_grid is not None:
            return depth_rendering.depth_grid_to_rgb(self._last_depth_grid, cell=self.cell)
        return zeros

    @property
    def game_type(self) -> str:
        return "vizdoom"

    @property
    def proprio_dim(self) -> int:
        return self.state_dim

    # ── helpers ──────────────────────────────────────────────────────
    def _coerce_action(self, action: Any) -> List[float]:
        """Accept a 9-dim canonical vector or canonical action dictionary."""
        if isinstance(action, dict):
            from gamewam.envs.vizdoom.action_space import action_to_vector
            vec = action_to_vector(action)
            return vector_to_buttons(vec)
        arr = np.asarray(action).ravel().astype(np.float32)
        if arr.size != ACTION_DIM:
            raise ValueError(
                f"VizDoom action must have {ACTION_DIM} canonical dimensions, "
                f"got {arr.size}."
            )
        return vector_to_buttons(arr)

    def _observe(self, reward: float, done: bool) -> GameObservation:
        """Build an observation from the current engine state."""
        import cv2

        state = None if done else self._game.get_state()

        game_vars: Dict[str, float] = {}
        if state is not None:
            depth = getattr(state, "depth_buffer", None)
            if depth is not None:
                self._last_depth_grid = depth_rendering.depth_buffer_to_grid(depth)
            else:
                self._last_depth_grid = None
            screen = getattr(state, "screen_buffer", None)
            if screen is not None:
                self._last_screen = cv2.resize(
                    screen, (self.screen_w, self.screen_h),
                    interpolation=cv2.INTER_AREA,
                )
            import vizdoom as vzd
            for name in INFO_GAME_VARS:
                try:
                    game_vars[name] = float(
                        self._game.get_game_variable(getattr(vzd.GameVariable, name))
                    )
                except Exception:
                    pass

        enemies: List[float] = []
        enemies_w: List[float] = []
        health_packs: List[float] = []
        health_packs_w: List[float] = []
        if state is not None and self.enable_labels:
            labels = getattr(state, "labels", None)
            screen_w = self._game.get_screen_width()
            if labels:
                for lab in labels:
                    if _is_monster(lab.object_name):
                        enemies.append((lab.x + lab.width / 2.0) / screen_w)
                        enemies_w.append(lab.width / screen_w)
                    elif lab.object_name in ("Medikit", "Stimpack", "CustomMedikit"):
                        health_packs.append((lab.x + lab.width / 2.0) / screen_w)
                        health_packs_w.append(lab.width / screen_w)

        zeros = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        if self.obs_mode == "rgb":
            image = self._last_screen if self._last_screen is not None else zeros
        else:  # depth
            image = (
                depth_rendering.depth_grid_to_rgb(self._last_depth_grid, cell=self.cell)
                if self._last_depth_grid is not None else zeros
            )

        self._last_info = {
            "depth_grid": self._last_depth_grid,
            "game_vars": game_vars,
            "enemies_x": enemies,
            "enemies_w": enemies_w,
            "health_packs": health_packs,
            "health_packs_w": health_packs_w,
        }
        proprio = extract_state(self._last_info, self.state_dim)
        proprio_field_valid = np.ones(self.state_dim, dtype=bool)
        if self.scenario in ("defend_the_center", "defend_the_line"):
            # Defend checkpoints expose health but not pose.
            pose_dims = min(3, self.state_dim)
            proprio[:pose_dims] = 0.0
            proprio_field_valid[:pose_dims] = False

        return GameObservation(
            image=image,
            proprio=proprio,
            proprio_field_valid=proprio_field_valid,
            info=self._last_info,
            reward=reward,
            done=done,
            timestamp=self._step_count,
        )
