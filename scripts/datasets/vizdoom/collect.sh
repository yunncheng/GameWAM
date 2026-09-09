#!/usr/bin/env bash
set -euo pipefail

# CPU-only, native 35 FPS APPO collection for all four VizDoom scenarios.
#
# Common overrides:
#   TARGET_EPISODES_PER_SCENARIO=10000 WORKERS_PER_SCENARIO=24 \
#     bash scripts/datasets/vizdoom/collect.sh
#
# Extra CLI arguments are appended last and therefore can override defaults:
#   bash scripts/datasets/vizdoom/collect.sh \
#     --scenarios battle1 --target-episodes-per-scenario 20

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
MODEL_ROOT="${MODEL_ROOT:-/path/to/vizdoom_appo_models}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/VizDoom_Lerobot}"
TARGET_EPISODES_PER_SCENARIO="${TARGET_EPISODES_PER_SCENARIO:-12500}"
WORKERS_PER_SCENARIO="${WORKERS_PER_SCENARIO:-24}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"
DEFAULT_WORKERS_BY_SCENARIO='{"battle1":44,"battle2":48,"defend_the_center":19,"defend_the_line":17}'
WORKERS_BY_SCENARIO="${WORKERS_BY_SCENARIO:-${DEFAULT_WORKERS_BY_SCENARIO}}"
VIDEO_CRF="${VIDEO_CRF:-16}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
KEEP_TOP_RATIO="${KEEP_TOP_RATIO:-0.75}"
DETERMINISTIC_FRACTION="${DETERMINISTIC_FRACTION:-0.75}"
MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-600}"
DEFAULT_MIN_SCORES='{"battle1":48,"battle2":22,"defend_the_center":18,"defend_the_line":23}'
MIN_SCORES="${MIN_SCORES:-${DEFAULT_MIN_SCORES}}"

export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n vizdoom_appo \
  python "${SCRIPT_DIR}/appo/collect_appo_parallel.py" \
  --output-dir "${OUTPUT_DIR}" \
  --model "battle1=${MODEL_ROOT}/battle1" \
  --model "battle2=${MODEL_ROOT}/battle2" \
  --model "defend_the_center=${MODEL_ROOT}/defend_the_center" \
  --model "defend_the_line=${MODEL_ROOT}/defend_the_line" \
  --target-episodes-per-scenario "${TARGET_EPISODES_PER_SCENARIO}" \
  --workers-per-scenario "${WORKERS_PER_SCENARIO}" \
  --workers-by-scenario "${WORKERS_BY_SCENARIO}" \
  --threads-per-worker "${THREADS_PER_WORKER}" \
  --video-crf "${VIDEO_CRF}" \
  --video-preset "${VIDEO_PRESET}" \
  --keep-top-ratio "${KEEP_TOP_RATIO}" \
  --min-scores "${MIN_SCORES}" \
  --deterministic-fraction "${DETERMINISTIC_FRACTION}" \
  --max-agent-steps "${MAX_AGENT_STEPS}" \
  "$@"
