#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Edit these paths before running.
VPT_RAW_ROOT="/path/to/vpt/raw"
OUTPUT_ROOT="/path/to/VPT_Event_Lerobot"

python "${SCRIPT_DIR}/convert_game_data.py" \
    --source_type vpt_event \
    --vpt_root "${VPT_RAW_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --skip_null_actions \
    --max_episodes 200000 \
    --win_len 96 \
    --bias 8 \
    --min_event_stride 96 \
    --min_clip_stride 64 \
    --shuffle
