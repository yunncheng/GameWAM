"""Canonical non-GUI tasks for the VoxeLibre zero-shot paper table."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VoxeLibreTaskSpec:
    """Static definition of one formal zero-shot task."""

    name: str
    table_label: str
    instruction: str
    kind: str
    target: str
    inventory: tuple[str, ...]
    selected_hotbar: int = 0
    structure: str = "single"
    max_steps: int = 600

    def __post_init__(self) -> None:
        if self.kind not in {"dig_node", "kill_entity", "place_node"}:
            raise ValueError(f"Unsupported VoxeLibre task kind: {self.kind!r}")
        if not self.name or not self.instruction or not self.target:
            raise ValueError("Task name, instruction and target must be non-empty")
        if not 0 <= int(self.selected_hotbar) <= 8:
            raise ValueError("selected_hotbar must be a zero-based slot in [0, 8]")
        if not self.inventory:
            raise ValueError(f"Task {self.name!r} must provide initial inventory")
        if int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class VoxeLibreEpisodeScene:
    """Deterministic geometry and RNG identity for one task episode."""

    task_name: str
    episode_id: str
    task_seed: int
    environment_seed: int
    model_seed: int
    target_x: float
    target_z: float
    view_yaw_offset_deg: float

    @property
    def target_distance(self) -> float:
        return math.hypot(self.target_x, self.target_z)

    @property
    def target_bearing_deg(self) -> float:
        return math.degrees(math.atan2(self.target_x, self.target_z))


# All prompt strings are deliberately taken verbatim from the existing
# Minecraft event-description vocabulary. No VoxeLibre-specific prompt tuning
# or paraphrase selection is performed.
TASK_SPECS: dict[str, VoxeLibreTaskSpec] = {
    "chop_tree": VoxeLibreTaskSpec(
        name="chop_tree",
        table_label="Chop Tree",
        instruction="Chop down the oak log.",
        kind="dig_node",
        target="mcl_core:tree",
        inventory=("mcl_tools:axe_iron",),
        structure="tree",
    ),
    "mine_stone": VoxeLibreTaskSpec(
        name="mine_stone",
        table_label="Mine Stone",
        instruction="Break the stone block.",
        kind="dig_node",
        target="mcl_core:stone",
        inventory=("mcl_tools:pick_iron",),
        structure="ore_outcrop",
    ),
    "mine_iron_ore": VoxeLibreTaskSpec(
        name="mine_iron_ore",
        table_label="Mine Iron Ore",
        instruction="Use a pickaxe to gather iron ore.",
        kind="dig_node",
        target="mcl_core:stone_with_iron",
        inventory=("mcl_tools:pick_iron",),
        structure="ore_outcrop",
    ),
    "kill_cow": VoxeLibreTaskSpec(
        name="kill_cow",
        table_label="Kill Cow",
        instruction="Kill the cow.",
        kind="kill_entity",
        target="mobs_mc:cow",
        inventory=("mcl_tools:sword_iron",),
    ),
    "kill_zombie": VoxeLibreTaskSpec(
        name="kill_zombie",
        table_label="Kill Zombie",
        instruction="Kill a zombie.",
        kind="kill_entity",
        target="mobs_mc:zombie",
        inventory=("mcl_tools:sword_iron",),
    ),
    "place_block": VoxeLibreTaskSpec(
        name="place_block",
        table_label="Place Block",
        instruction="Place the cobblestone on the ground.",
        kind="place_node",
        target="mcl_core:cobble",
        inventory=("mcl_core:cobble 16",),
        structure="placement_pedestal",
    ),
}
TASK_NAMES: tuple[str, ...] = tuple(TASK_SPECS)


def derive_episode_seed(
    base_seed: int,
    task_name: str,
    episode_id: str | int,
    domain: str,
    *,
    bits: int = 63,
) -> int:
    """Return a deterministic independent sub-seed.

    Task identity is always included so episode ``0`` of two tasks cannot
    share environment or model RNG streams.
    """

    if not 1 <= int(bits) <= 64:
        raise ValueError(f"bits must be in [1, 64], got {bits}")
    digest = hashlib.blake2b(
        f"{int(base_seed)}|{task_name}|{episode_id}|{domain}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big", signed=False) & ((1 << bits) - 1)


def build_episode_scene(
    task_name: str,
    episode_id: str | int,
    base_seed: int,
) -> VoxeLibreEpisodeScene:
    """Generate a controlled, reproducible visible-target scene."""

    get_task_spec(task_name)
    episode_id = str(episode_id)
    task_seed = derive_episode_seed(base_seed, task_name, episode_id, "task")
    rng = random.Random(task_seed)

    distance = rng.choice((4.0, 5.0, 6.0))
    bearing_deg = rng.choice((-30.0, -15.0, 0.0, 15.0, 30.0))
    bearing = math.radians(bearing_deg)
    # Structures are node-aligned in Luanti. Store the exact integer location
    # that Lua receives so reports never describe a pre-quantized target.
    target_x = float(round(math.sin(bearing) * distance))
    target_z = float(round(math.cos(bearing) * distance))

    return VoxeLibreEpisodeScene(
        task_name=task_name,
        episode_id=episode_id,
        task_seed=task_seed,
        environment_seed=derive_episode_seed(
            base_seed, task_name, episode_id, "environment", bits=31
        ),
        model_seed=derive_episode_seed(
            base_seed, task_name, episode_id, "model", bits=63
        ),
        target_x=target_x,
        target_z=target_z,
        view_yaw_offset_deg=rng.choice((-12.0, -6.0, 0.0, 6.0, 12.0)),
    )


def get_task_spec(task_name: str) -> VoxeLibreTaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown VoxeLibre task {task_name!r}; expected one of {TASK_NAMES}"
        ) from exc


def resolve_tasks(value: str | Iterable[str]) -> list[str]:
    """Resolve ``all`` or an ordered, duplicate-free task subset."""

    if isinstance(value, str):
        if value.strip().lower() == "all":
            return list(TASK_NAMES)
        values = value.split(",")
    else:
        values = list(value)

    tasks = [str(item).strip() for item in values if str(item).strip()]
    if not tasks:
        raise ValueError("VoxeLibre task selection must not be empty")
    unknown = [name for name in tasks if name not in TASK_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown VoxeLibre task(s) {unknown}; expected a subset of {TASK_NAMES}"
        )
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"Duplicate VoxeLibre task in selection: {tasks}")
    return tasks
