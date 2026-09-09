#!/bin/bash
# Full-scale scripted GUI data collection: all craft_item tasks (~550 recipes)
# --tasks omitted → defaults to ALL tasks in RECIPE_LAYOUTS

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${PROJECT_ROOT}/.minestudio}"

# Edit this path before running.
OUTPUT_DIR="/path/to/MC_ScriptedGUI_Lerobot/gui_craft"

python "${SCRIPT_DIR}/collect.py" \
    --episodes_per_task 100 \
    --output_dir "${OUTPUT_DIR}" \
    --workers 100 \
    --max_attempts_per_task 200 \
    --bias 8 \
    --collect_mode mixed
