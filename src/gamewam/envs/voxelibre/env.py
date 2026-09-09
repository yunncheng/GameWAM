"""Craftium-backed VoxeLibre environment with Minecraft-compatible semantics."""

from __future__ import annotations

import ctypes
import errno
import math
import os
import secrets
import shutil
import tempfile
from contextlib import contextmanager, suppress
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gamewam.envs.base import GameEnvBase, GameObservation
from gamewam.envs.minecraft.observations.state import extract_state_with_validity

from .action_space import (
    CRAFTIUM_ACTION_KEYS,
    CameraCalibration,
    VoxeLibreActionAdapter,
)
from .tasks import VoxeLibreEpisodeScene, VoxeLibreTaskSpec


class VoxeLibreEnvironmentError(RuntimeError):
    """Raised when the optional Craftium/VoxeLibre runtime is unavailable."""


SUPPORTED_CRAFTIUM_VERSION = "0.0.1"
SUPPORTED_VOXELIBRE_VERSION = "0.89.4"
CRAFTIUM_PLACE_EDGE_MARKER = "GAMEWAM_VIRTUAL_PLACE_EDGE"
CRAFTIUM_PLACE_EDGE_MARKER_VALUE = "craftium-v0.0.1-gamewam-place-edge-v1"
CRAFTIUM_XVFB_FRAMEBUFFER_MARKER = "GAMEWAM_XVFB_FRAMEBUFFER"
CRAFTIUM_XVFB_FRAMEBUFFER_MARKER_VALUE = (
    "craftium-v0.0.1-gamewam-xvfb-back-buffer-v1"
)


class _CraftiumSyncSemaphores:
    """Pre-create Craftium v0.0.1 sync semaphores to remove its startup race."""

    def __init__(self, port: int, names: tuple[bytes, bytes]):
        self.port = int(port)
        self.names = names
        self._closed = False

    @classmethod
    def reserve(cls, attempts: int = 256) -> "_CraftiumSyncSemaphores":
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sem_open.restype = ctypes.c_void_p
        libc.sem_close.argtypes = (ctypes.c_void_p,)
        libc.sem_unlink.argtypes = (ctypes.c_char_p,)
        sem_failed = ctypes.c_void_p(-1).value
        flags = os.O_CREAT | os.O_EXCL
        for _ in range(attempts):
            port = 49152 + secrets.randbelow(65536 - 49152)
            names = (f"/crftA{port}".encode(), f"/crftB{port}".encode())
            handles: list[int] = []
            for name in names:
                ctypes.set_errno(0)
                handle = libc.sem_open(name, flags, 0o644, 0)
                if handle == sem_failed:
                    break
                handles.append(handle)
            failure_code = ctypes.get_errno() if len(handles) != len(names) else 0
            for handle in handles:
                libc.sem_close(handle)
            if len(handles) == len(names):
                return cls(port, names)
            for name in names[: len(handles)]:
                libc.sem_unlink(name)
            if failure_code not in (0, errno.EEXIST):
                raise OSError(failure_code, "Could not create Craftium semaphore")
        raise RuntimeError("Could not reserve a free Craftium sync port/semaphore pair")

    def close(self) -> None:
        if self._closed:
            return
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sem_unlink.argtypes = (ctypes.c_char_p,)
        for name in self.names:
            libc.sem_unlink(name)
        self._closed = True


def _load_craftium_env_class():
    try:
        from craftium.craftium_env import ACTION_ORDER, CraftiumEnv
    except ImportError as exc:
        raise VoxeLibreEnvironmentError(
            f"VoxeLibre evaluation requires Craftium {SUPPORTED_CRAFTIUM_VERSION}. "
            "Install the pinned runtime with scripts/setup_voxelibre_eval.sh."
        ) from exc

    try:
        craftium_version = metadata.version("craftium")
    except metadata.PackageNotFoundError as exc:
        raise VoxeLibreEnvironmentError(
            "Craftium is importable but has no installed distribution metadata."
        ) from exc
    if craftium_version != SUPPORTED_CRAFTIUM_VERSION:
        raise VoxeLibreEnvironmentError(
            f"This suite requires Craftium {SUPPORTED_CRAFTIUM_VERSION}, got "
            f"{craftium_version}."
        )
    marker = resources.files("craftium").joinpath(CRAFTIUM_PLACE_EDGE_MARKER)
    if not marker.is_file():
        raise VoxeLibreEnvironmentError(
            "Craftium v0.0.1 is missing the required GameWAM virtual PLACE "
            "edge patch. Reinstall it with scripts/setup_voxelibre_eval.sh."
        )
    marker_value = marker.read_text(encoding="utf-8").strip()
    if marker_value != CRAFTIUM_PLACE_EDGE_MARKER_VALUE:
        raise VoxeLibreEnvironmentError(
            "Craftium has an unsupported GameWAM virtual PLACE edge patch "
            f"marker {marker_value!r}; expected "
            f"{CRAFTIUM_PLACE_EDGE_MARKER_VALUE!r}. Reinstall it with "
            "scripts/setup_voxelibre_eval.sh."
        )
    actual = tuple(ACTION_ORDER)
    if actual != CRAFTIUM_ACTION_KEYS:
        raise VoxeLibreEnvironmentError(
            "Unsupported Craftium action contract. Expected Craftium v0.0.1 "
            f"{CRAFTIUM_ACTION_KEYS}, got {actual}."
        )
    return CraftiumEnv


def _validate_craftium_build_root(value: str | Path | None) -> tuple[Path, Path]:
    """Resolve the exact patched Craftium data/binary root used at runtime."""

    root = (
        Path(str(resources.files("craftium")))
        if value is None
        else Path(value).expanduser()
    ).resolve()
    binary = root / "bin" / "minetest"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VoxeLibreEnvironmentError(
            f"Invalid Craftium build root {root}: expected executable {binary}."
        )
    markers = (
        root / CRAFTIUM_PLACE_EDGE_MARKER,
        root / "craftium" / CRAFTIUM_PLACE_EDGE_MARKER,
    )
    values = [
        marker.read_text(encoding="utf-8").strip()
        for marker in markers
        if marker.is_file()
    ]
    if CRAFTIUM_PLACE_EDGE_MARKER_VALUE not in values:
        raise VoxeLibreEnvironmentError(
            f"Craftium build root {root} is not marked with the required "
            f"PLACE-edge patch {CRAFTIUM_PLACE_EDGE_MARKER_VALUE!r}."
        )
    framebuffer_marker = root / CRAFTIUM_XVFB_FRAMEBUFFER_MARKER
    if (
        not framebuffer_marker.is_file()
        or framebuffer_marker.read_text(encoding="utf-8").strip()
        != CRAFTIUM_XVFB_FRAMEBUFFER_MARKER_VALUE
    ):
        raise VoxeLibreEnvironmentError(
            f"Craftium build root {root} is missing the required Xvfb "
            "framebuffer patch. Rebuild it with "
            "scripts/setup_voxelibre_eval.sh."
        )
    return root, binary


def _read_voxelibre_version(game_dir: Path) -> str:
    game_conf = game_dir / "game.conf"
    if not game_conf.is_file() or not (game_dir / "mods").is_dir():
        raise VoxeLibreEnvironmentError(
            f"Invalid VoxeLibre game directory {game_dir}: expected game.conf and mods/."
        )
    for line in game_conf.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "version":
            return value.strip()
    return "unknown"


def _write_text(path: Path, value: str) -> None:
    """Write a small generated runtime file."""
    path.write_text(value, encoding="utf-8")


@contextmanager
def _preserve_template_symlinks(template_root: Path):
    """Avoid copying the entire VoxeLibre game for every Craftium episode."""
    original_copytree = shutil.copytree
    template_root = template_root.resolve()

    def copytree(src: Any, dst: Any, *args: Any, **kwargs: Any):
        try:
            Path(src).resolve().relative_to(template_root)
        except (OSError, ValueError):
            pass
        else:
            if not args and "symlinks" not in kwargs:
                kwargs["symlinks"] = True
        return original_copytree(src, dst, *args, **kwargs)

    shutil.copytree = copytree
    try:
        yield
    finally:
        shutil.copytree = original_copytree


class VoxeLibreGameEnv(GameEnvBase):
    """One controlled VoxeLibre episode exposed through GameWAM's env API.

    A fresh Luanti process and world are created for every episode. The model
    still receives the six raw Minecraft state fields and the same 22-D action
    semantics used during training. Craftium is imported only on construction,
    so this module has no effect on existing training or Minecraft evaluation.
    """

    GAME_ID = "mineclone2"
    WORLD_NAME = "world"

    def __init__(
        self,
        *,
        task: VoxeLibreTaskSpec,
        scene: VoxeLibreEpisodeScene,
        voxelibre_game_dir: str | Path,
        craftium_minetest_dir: str | Path | None = None,
        render_size: Sequence[int] = (640, 360),
        cursor_reference_size: Sequence[float] | None = None,
        fps: int = 20,
        max_steps: int | None = None,
        run_dir_prefix: str | Path | None = None,
        keep_runtime_dir: bool = False,
        init_frames: int = 30,
        mt_listen_timeout_ms: int = 180_000,
    ):
        width, height = (int(render_size[0]), int(render_size[1]))
        if width <= 1 or height <= 1:
            raise ValueError(
                f"render_size must be positive (width, height), got {render_size}"
            )
        if int(fps) <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if int(task.selected_hotbar) != 0:
            raise ValueError("VoxeLibre formal suite requires zero-based hotbar slot 0")
        if task.name != scene.task_name:
            raise ValueError(
                f"Task/scene mismatch: {task.name!r} != {scene.task_name!r}"
            )
        self.task = task
        self.scene = scene
        self.render_size = (width, height)
        if cursor_reference_size is None:
            cursor_reference_size = self.render_size
        if len(cursor_reference_size) != 2:
            raise ValueError(
                "cursor_reference_size must be [width, height], got "
                f"{cursor_reference_size!r}"
            )
        reference_width, reference_height = map(float, cursor_reference_size)
        if (
            not math.isfinite(reference_width)
            or not math.isfinite(reference_height)
            or reference_width <= 1
            or reference_height <= 1
        ):
            raise ValueError(
                "cursor_reference_size values must be finite and greater than one, "
                f"got {cursor_reference_size!r}"
            )
        self.cursor_reference_size = (reference_width, reference_height)
        self.fps = int(fps)
        self.max_steps = int(task.max_steps if max_steps is None else max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        self.voxelibre_game_dir = Path(voxelibre_game_dir).expanduser().resolve()
        self.voxelibre_version = _read_voxelibre_version(self.voxelibre_game_dir)
        if self.voxelibre_version != SUPPORTED_VOXELIBRE_VERSION:
            raise VoxeLibreEnvironmentError(
                f"This suite requires VoxeLibre {SUPPORTED_VOXELIBRE_VERSION}, "
                f"got {self.voxelibre_version} at {self.voxelibre_game_dir}."
            )
        self.craftium_minetest_dir = craftium_minetest_dir
        self.run_dir_prefix = (
            str(Path(run_dir_prefix).expanduser().resolve())
            if run_dir_prefix is not None
            else None
        )
        if self.run_dir_prefix is not None:
            Path(self.run_dir_prefix).mkdir(parents=True, exist_ok=True)
        self.keep_runtime_dir = bool(keep_runtime_dir)
        self.init_frames = int(init_frames)
        self.mt_listen_timeout_ms = int(mt_listen_timeout_ms)
        if self.init_frames < 0 or self.mt_listen_timeout_ms <= 0:
            raise ValueError(
                "init_frames must be non-negative and timeout must be positive"
            )

        self.camera = CameraCalibration(width=width, height=height)
        self.action_adapter = VoxeLibreActionAdapter(self.camera)
        self._template_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._env: Any = None
        self._sync_semaphores: _CraftiumSyncSemaphores | None = None
        self._last_image: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._step_count = 0
        self._current_hotbar = int(task.selected_hotbar)
        self._gui_open = False
        self._inventory_pressed = False
        self._cursor_x = reference_width / 2.0
        self._cursor_y = reference_height / 2.0
        self.last_effective_action: dict[str, Any] | None = None
        self.last_camera_clipped = False

    def _create_template(self) -> Path:
        self._template_tmp = tempfile.TemporaryDirectory(
            prefix="gamewam_voxelibre_env_"
        )
        root = Path(self._template_tmp.name)
        game_link = root / "games" / self.GAME_ID
        world_dir = root / "worlds" / self.WORLD_NAME
        mod_dir = root / "mods" / "craftium_env"
        game_link.parent.mkdir(parents=True)
        world_dir.mkdir(parents=True)
        mod_dir.parent.mkdir(parents=True)
        game_link.symlink_to(self.voxelibre_game_dir, target_is_directory=True)

        asset_root = resources.files("gamewam.envs.voxelibre").joinpath(
            "assets", "craftium_env"
        )
        with resources.as_file(asset_root) as asset_path:
            shutil.copytree(asset_path, mod_dir)

        _write_text(
            world_dir / "world.mt",
            "\n".join(
                (
                    "enable_damage = true",
                    "creative_mode = false",
                    "mod_storage_backend = sqlite3",
                    "auth_backend = sqlite3",
                    "player_backend = sqlite3",
                    "backend = sqlite3",
                    f"gameid = {self.GAME_ID}",
                    f"world_name = {self.WORLD_NAME}",
                    "server_announce = false",
                    "load_mod_craftium_env = mods/craftium_env",
                    "",
                )
            ),
        )
        _write_text(
            world_dir / "map_meta.txt",
            f"mg_name = singlenode\nmg_flags = light\nseed = {self.scene.environment_seed}\n[end_of_params]\n",
        )
        return root

    def _minetest_conf(self) -> dict[str, Any]:
        return {
            "fov": self.camera.vertical_fov_deg,
            "mouse_sensitivity": self.camera.mouse_sensitivity,
            "mg_name": "singlenode",
            "time_speed": 0,
            "mobs_spawn": False,
            "mcl_doWeatherCycle": False,
            "enable_particles": False,
            "enable_clouds": False,
            "smooth_lighting": False,
            "performance_tradeoffs": True,
            "console_alpha": 0,
            "viewing_range": 64,
            "max_block_generate_distance": 2,
            "max_block_send_distance": 2,
            "gamewam_task_kind": self.task.kind,
            "gamewam_target": self.task.target,
            "gamewam_structure": self.task.structure,
            "gamewam_inventory": "|".join(self.task.inventory),
            "gamewam_selected_hotbar": int(self.task.selected_hotbar) + 1,
            "gamewam_target_x": self.scene.target_x,
            "gamewam_target_z": self.scene.target_z,
            "gamewam_view_yaw_offset_deg": self.scene.view_yaw_offset_deg,
        }

    def _build_env(self) -> Any:
        CraftiumEnv = _load_craftium_env_class()
        craftium_root, _ = _validate_craftium_build_root(self.craftium_minetest_dir)
        template = self._create_template()
        sync_semaphores = _CraftiumSyncSemaphores.reserve()
        kwargs: dict[str, Any] = {
            "env_dir": str(template),
            "obs_width": self.render_size[0],
            "obs_height": self.render_size[1],
            "init_frames": self.init_frames,
            # Craftium maps rgb_array to SDL's offscreen EGL backend, which is
            # unavailable on many headless CUDA nodes. Under Xvfb, human mode
            # uses X11/GLX and still returns pixels via rgb_observations.
            "render_mode": "human" if os.environ.get("DISPLAY") else "rgb_array",
            "max_timesteps": self.max_steps,
            "run_dir_prefix": self.run_dir_prefix,
            "game_id": self.GAME_ID,
            "world_name": self.WORLD_NAME,
            "minetest_conf": self._minetest_conf(),
            "pipe_proc": True,
            "mt_listen_timeout": self.mt_listen_timeout_ms,
            "mt_port": sync_semaphores.port,
            "frameskip": 1,
            "rgb_observations": True,
            "seed": self.scene.environment_seed,
            "sync_mode": True,
            "fps_max": self.fps,
            "pmul": 1,
            "soft_reset": False,
            # Pass this even for the installed package. Craftium otherwise
            # silently prefers cwd when cwd happens to be a build directory.
            "minetest_dir": str(craftium_root),
        }
        try:
            with _preserve_template_symlinks(template):
                env = CraftiumEnv(**kwargs)
        except Exception:
            sync_semaphores.close()
            raise
        self._sync_semaphores = sync_semaphores
        return env

    def _image(self, raw: Any) -> np.ndarray:
        image = np.asarray(raw)
        width, height = self.render_size
        transport_shape = (height, width, 3)
        documented_shape = (width, height, 3)
        if image.dtype != np.uint8:
            raise VoxeLibreEnvironmentError(
                f"Unexpected Craftium RGB dtype: expected uint8, got {image.dtype}"
            )
        if image.shape == transport_shape:
            return np.ascontiguousarray(image)
        if image.shape == documented_shape:
            return np.ascontiguousarray(image.transpose(1, 0, 2))
        raise VoxeLibreEnvironmentError(
            "Unexpected Craftium RGB observation: expected transport "
            f"{transport_shape} or documented {documented_shape}, got {image.shape}"
        )

    @staticmethod
    def _canonical_yaw(raw_yaw: Any) -> float:
        yaw = -float(raw_yaw)
        return (yaw + 180.0) % 360.0 - 180.0

    def _state_info(self, raw_info: Mapping[str, Any]) -> dict[str, Any]:
        try:
            pitch = float(raw_info["player_pitch"])
            yaw = self._canonical_yaw(raw_info["player_yaw"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VoxeLibreEnvironmentError(
                f"Craftium observation is missing numeric camera pose: {raw_info}"
            ) from exc
        if not math.isfinite(pitch) or not math.isfinite(yaw):
            raise VoxeLibreEnvironmentError(
                f"Craftium returned non-finite camera pose: pitch={pitch}, yaw={yaw}"
            )
        info = dict(raw_info)
        info.update(
            {
                "pitch": pitch,
                "yaw": yaw,
                "current_hotbar": self._current_hotbar,
                "is_gui_open": self._gui_open,
                "voxelibre_task": self.task.name,
                "voxelibre_version": self.voxelibre_version,
            }
        )
        if self._gui_open:
            info["cursor_x"] = self._cursor_x
            info["cursor_y"] = self._cursor_y
        return info

    def _observation(
        self,
        raw_image: Any,
        raw_info: Mapping[str, Any],
        *,
        reward: float,
        done: bool,
    ) -> GameObservation:
        image = self._image(raw_image)
        info = self._state_info(raw_info)
        proprio, valid = extract_state_with_validity(
            info, cursor_reference_size=self.cursor_reference_size
        )
        self._last_image = image
        self._last_info = info
        return GameObservation(
            image=image,
            proprio=proprio,
            proprio_field_valid=valid,
            info=info,
            reward=float(reward),
            done=bool(done),
            timestamp=self._step_count,
        )

    def reset(self, **kwargs: Any) -> GameObservation:
        if kwargs:
            raise TypeError(
                f"VoxeLibreGameEnv.reset takes no overrides, got {sorted(kwargs)}"
            )
        self.close()
        self._env = self._build_env()
        self._step_count = 0
        self._current_hotbar = int(self.task.selected_hotbar)
        self._gui_open = False
        self._inventory_pressed = False
        self._cursor_x = self.cursor_reference_size[0] / 2.0
        self._cursor_y = self.cursor_reference_size[1] / 2.0
        self.last_effective_action = None
        self.last_camera_clipped = False
        raw_image, raw_info = self._env.reset(seed=self.scene.environment_seed)
        return self._observation(raw_image, raw_info, reward=0.0, done=False)

    def _update_tracked_state(
        self, action: Mapping[str, Any], mouse: tuple[int, int]
    ) -> None:
        was_gui_open = self._gui_open
        inventory_pressed = bool(action.get("inventory", 0))
        if inventory_pressed and not self._inventory_pressed:
            self._gui_open = not self._gui_open
            if self._gui_open:
                self._cursor_x = self.cursor_reference_size[0] / 2.0
                self._cursor_y = self.cursor_reference_size[1] / 2.0
        if was_gui_open and self._gui_open:
            scale_x = self.cursor_reference_size[0] / self.render_size[0]
            scale_y = self.cursor_reference_size[1] / self.render_size[1]
            self._cursor_x = float(
                np.clip(
                    self._cursor_x + mouse[0] * scale_x,
                    0,
                    self.cursor_reference_size[0],
                )
            )
            self._cursor_y = float(
                np.clip(
                    self._cursor_y + mouse[1] * scale_y,
                    0,
                    self.cursor_reference_size[1],
                )
            )
        self._inventory_pressed = inventory_pressed
        if not was_gui_open:
            selected = [
                slot for slot in range(1, 10) if bool(action.get(f"hotbar.{slot}", 0))
            ]
            if selected:
                self._current_hotbar = selected[-1] - 1

    def step(self, action: Any) -> GameObservation:
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        actions = action if isinstance(action, list) else [action]
        if not actions:
            raise ValueError("VoxeLibreGameEnv.step action list must not be empty")

        total_reward = 0.0
        done = False
        raw_image: Any = None
        raw_info: Mapping[str, Any] = {}
        for minecraft_action in actions:
            if not isinstance(minecraft_action, Mapping):
                raise TypeError(
                    f"VoxeLibre action must be a mapping, got {type(minecraft_action).__name__}"
                )
            raw_inventory_pressed = bool(minecraft_action.get("inventory", 0))
            craftium_input = dict(minecraft_action)
            craftium_input["inventory"] = int(
                raw_inventory_pressed and not self._inventory_pressed
            )
            adapted = self.action_adapter.adapt(craftium_input)
            raw_image, reward, terminated, truncated, raw_info = self._env.step(
                adapted.craftium_action
            )
            self._step_count += 1
            total_reward += float(reward)
            self._update_tracked_state(minecraft_action, adapted.mouse_pixel_delta)
            self.last_effective_action = adapted.effective_minecraft_action
            self.last_camera_clipped = adapted.camera_clipped
            done = bool(terminated or truncated)
            if done:
                break
        return self._observation(raw_image, raw_info, reward=total_reward, done=done)

    def capture_post_success_frames(self, count: int) -> list[np.ndarray]:
        """Advance neutral transport frames for a settled success video tail.

        Craftium v0.0.1 reports termination as a one-shot flag without stopping
        its Luanti process. A success callback and the corresponding client map
        update can therefore arrive in the same observation, before the client
        has rendered the changed node/entity. Advancing neutral frames lets
        saved videos show the completed world state.

        This is deliberately separate from :meth:`step`: it is a rendering
        flush after the policy episode has ended, not an agent action. It does
        not advance GameWAM's step counter or alter the executed-action history.
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        count = int(count)
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        neutral = self.action_adapter.adapt({})
        frames: list[np.ndarray] = []
        for _ in range(count):
            raw_image, _reward, _terminated, _truncated, _raw_info = self._env.step(
                neutral.craftium_action
            )
            frames.append(self._image(raw_image))
        self._last_image = frames[-1]
        return frames

    def _force_cleanup(self, craftium: Any) -> None:
        channel = getattr(craftium, "mt_chann", None)
        mt = getattr(craftium, "mt", None)
        if channel is not None:
            with suppress(Exception):
                channel.close_conn()
            with suppress(Exception):
                os.close(channel.sockfd)
        if mt is not None:
            process = getattr(mt, "proc", None)
            if process is not None and process.poll() is None:
                with suppress(Exception):
                    process.terminate()
                    process.wait(timeout=5)
                if process.poll() is None:
                    with suppress(Exception):
                        process.kill()
                        process.wait(timeout=5)
            with suppress(Exception):
                mt.close_pipes()
            if not self.keep_runtime_dir:
                with suppress(Exception):
                    mt.clear()

    def close(self) -> None:
        if self._env is not None:
            craftium = self._env
            self._env = None
            try:
                craftium.close(clear=not self.keep_runtime_dir)
            except Exception:
                self._force_cleanup(craftium)
        if self._sync_semaphores is not None:
            self._sync_semaphores.close()
            self._sync_semaphores = None
        if self._template_tmp is not None:
            self._template_tmp.cleanup()
            self._template_tmp = None

    def get_screenshot(self) -> np.ndarray:
        if self._last_image is not None:
            return self._last_image
        return np.zeros((self.render_size[1], self.render_size[0], 3), dtype=np.uint8)

    @property
    def game_type(self) -> str:
        return "voxelibre"

    @property
    def proprio_dim(self) -> int:
        return 6
