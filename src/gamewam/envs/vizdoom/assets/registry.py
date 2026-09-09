"""VizDoom scenario registry and resource resolution."""
from pathlib import Path
from typing import Optional, Tuple

ASSETS_DIR = Path(__file__).parent
SCENARIOS_DIR = ASSETS_DIR / "scenarios"

DEFAULT_SCENARIO = "defend_the_center"

GAMETARS_SCENARIOS = ["battle1", "battle2", "defend_the_line", "defend_the_center"]

SCENARIO_CFG_NAME = {
    "defend_the_center": "defend_the_center.cfg",
    "defend_the_line": "defend_the_line.cfg",
}

REPO_SCENARIO_FILES = {
    "battle1": ("D3_battle.cfg", "D3_battle.wad"),
    "battle2": ("D4_battle2.cfg", "D4_battle2.wad"),
}


def resolve_scenario(scenario: str) -> Tuple[str, Optional[str]]:
    """Resolve a built-in or bundled scenario to config and optional WAD paths."""
    if scenario in REPO_SCENARIO_FILES:
        cfg_name, wad_name = REPO_SCENARIO_FILES[scenario]
        cfg_path = SCENARIOS_DIR / cfg_name
        wad_path = SCENARIOS_DIR / wad_name
        if not cfg_path.is_file() or not wad_path.is_file():
            raise FileNotFoundError(
                f"Repo scenario '{scenario}' missing files under {SCENARIOS_DIR}"
            )
        return str(cfg_path), str(wad_path)

    import vizdoom as vzd

    cfg_name = SCENARIO_CFG_NAME.get(scenario, f"{scenario}.cfg")
    return vzd.scenarios_path + "/" + cfg_name, None
