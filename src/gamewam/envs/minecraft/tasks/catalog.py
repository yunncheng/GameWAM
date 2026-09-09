"""Minecraft task descriptions shared by collection and evaluation."""

from gamewam.envs.minecraft.assets import EVENT_DESCRIPTION_FILE, load_json


EVENT_DESCRIPTION = load_json(EVENT_DESCRIPTION_FILE)

VPT_BATCH_TASKS = {
    "6xx": [
        "Play minecraft freely in survival mode.",
        "Explore the minecraft world in survival mode.",
        "Survive and gather resources in minecraft.",
    ],
    "7xx": [
        "Play minecraft in the early game.",
        "Gather basic resources during the first 30 minutes of minecraft.",
        "Start a new minecraft world and explore the early game.",
    ],
    "8xx": [
        "Build a simple house in minecraft.",
        "Construct a small shelter in minecraft.",
        "Build a basic wooden house within a few minutes.",
    ],
    "9xx": [
        "Build a structure in minecraft using the given resources.",
        "Construct a building with random provided materials.",
        "Use the supplied blocks to build something creative.",
    ],
    "10xx": [
        *EVENT_DESCRIPTION["craft_item:diamond_pickaxe"],
        "Obtain a diamond pickaxe in minecraft.",
        "Mine, smelt, and craft your way to a diamond pickaxe.",
    ],
}
