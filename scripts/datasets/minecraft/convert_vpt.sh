#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Edit these paths before running.
VPT_RAW_ROOT="/path/to/vpt/raw"
OUTPUT_ROOT="/path/to/VPT_Contractor_Lerobot"

python "${SCRIPT_DIR}/convert_game_data.py" \
    --source_type vpt_contractor \
    --source_dir "${VPT_RAW_ROOT}/6xx/all_6xx_Jun_29" \
    --output_dir "${OUTPUT_ROOT}/6xx" \
    --max_episodes 1800 \
    --skip_null_actions \
    --shuffle

python "${SCRIPT_DIR}/convert_game_data.py" \
    --source_type vpt_contractor \
    --source_dir "${VPT_RAW_ROOT}/7xx/all_7xx_Apr_6" \
    --output_dir "${OUTPUT_ROOT}/7xx" \
    --max_episodes 600 \
    --skip_null_actions \
    --shuffle

python "${SCRIPT_DIR}/convert_game_data.py" \
    --source_type vpt_contractor \
    --source_dir "${VPT_RAW_ROOT}/8xx/all_8xx_Jun_29" \
    --output_dir "${OUTPUT_ROOT}/8xx" \
    --max_episodes 600 \
    --skip_null_actions \
    --shuffle

python "${SCRIPT_DIR}/convert_game_data.py" \
    --source_type vpt_contractor \
    --source_dir "${VPT_RAW_ROOT}/9xx/all_9xx_Jun_29" \
    --output_dir "${OUTPUT_ROOT}/9xx" \
    --max_episodes 900 \
    --skip_null_actions \
    --shuffle

python "${SCRIPT_DIR}/convert_game_data.py" \
    --source_type vpt_contractor \
    --source_dir "${VPT_RAW_ROOT}/10xx/all_10xx_Jun_29" \
    --output_dir "${OUTPUT_ROOT}/10xx" \
    --max_episodes 1000 \
    --skip_null_actions \
    --shuffle
