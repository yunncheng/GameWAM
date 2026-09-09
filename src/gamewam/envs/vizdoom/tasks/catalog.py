"""Episode-level task descriptions for VizDoom scenarios."""

from ..assets import DEFAULT_SCENARIO

TASK_PARAPHRASES = {
    "defend_the_center": [
        "Defend the center: shoot the enemies closing in from all directions.",
        "Stay in the middle of the arena and gun down approaching monsters.",
        "Survive as long as possible by shooting enemies attacking from every side.",
        "Turn to face incoming demons and fire to keep them away from the center.",
        "Hold your ground in the circular arena and kill enemies before they reach you.",
    ],
    "defend_the_line": [
        "Defend the line: shoot the monsters advancing toward you from across the room.",
        "Hold the line and gun down every enemy charging in from the front.",
        "Stop the incoming monsters before they reach your side of the arena.",
        "Stand your ground and fire at the demons crossing the line.",
        "Keep the enemies back by shooting them as they approach the line.",
    ],
    "battle1": [
        "Fight off the monsters and stay alive as long as you can.",
        "Survive the battle: kill enemies while collecting health and ammo.",
        "Roam the arena, gun down monsters, and grab supplies to keep fighting.",
        "Eliminate as many enemies as possible while keeping your health up.",
        "Battle the demons and survive by managing your health and ammo.",
    ],
    "battle2": [
        "Fight off the monsters in the maze and stay alive as long as you can.",
        "Survive the battle: kill enemies while collecting health and ammo.",
        "Explore the arena, gun down monsters, and grab supplies to keep fighting.",
        "Eliminate as many enemies as possible while keeping your health up.",
        "Battle the demons across the map and survive by managing health and ammo.",
    ],
}


def get_task_paraphrases(scenario: str = DEFAULT_SCENARIO) -> list[str]:
    """Return the task descriptions for a scenario."""
    return TASK_PARAPHRASES[scenario]
