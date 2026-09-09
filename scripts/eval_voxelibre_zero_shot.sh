#!/bin/bash
set -euo pipefail

# Formal GameWAM -> VoxeLibre zero-shot table: six fixed non-GUI tasks,
# 20 deterministic episodes per task, one model replica per Ray actor.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CKPT="/path/to/minecraft_checkpoint.pt"
DATASET_STATS="/path/to/minecraft_dataset_stats.json"
VOXELIBRE_DIR="${VOXELIBRE_DIR:-${PROJECT_ROOT}/.voxelibre}"
export VOXELIBRE_DIR
VOXELIBRE_GAME_DIR="${VOXELIBRE_GAME_DIR:-${VOXELIBRE_DIR}/VoxeLibre-0.89.4}"
CRAFTIUM_MINETEST_DIR="${CRAFTIUM_MINETEST_DIR:-${VOXELIBRE_DIR}/craftium-v0.0.1}"
CRAFTIUM_PYTHON_DIR="${VOXELIBRE_DIR}/craftium-python"
RUN_DIR_PREFIX="${VOXELIBRE_RUN_DIR_PREFIX:-${VOXELIBRE_DIR}/runtime}"
TASK_CONFIG=""
TASKS="all"
EPISODES=20
SPLIT=40
GPUS_PER_WORKER="0.2"
SEED=0
OUTPUT=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --dataset_stats) DATASET_STATS="$2"; shift 2 ;;
        --voxelibre_game_dir) VOXELIBRE_GAME_DIR="$2"; shift 2 ;;
        --craftium_minetest_dir) CRAFTIUM_MINETEST_DIR="$2"; shift 2 ;;
        --run_dir_prefix) RUN_DIR_PREFIX="$2"; shift 2 ;;
        --task_config) TASK_CONFIG="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --gpus_per_worker) GPUS_PER_WORKER="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

for value_name in CKPT DATASET_STATS VOXELIBRE_GAME_DIR; do
    if [[ -z "${!value_name}" ]]; then
        echo "ERROR: --${value_name,,} (or $value_name) is required" >&2
        exit 2
    fi
done
if [[ ! -d "$CRAFTIUM_PYTHON_DIR/craftium" ]]; then
    echo "ERROR: Craftium Python binding not found at $CRAFTIUM_PYTHON_DIR" >&2
    echo "Run bash scripts/setup_voxelibre_eval.sh first." >&2
    exit 1
fi
export PYTHONPATH="$CRAFTIUM_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "$OUTPUT" ]]; then
    CKPT_STEM="$(basename "$CKPT" .pt)"
    OUTPUT="$PROJECT_ROOT/evaluate_results/voxelibre_zero_shot/${CKPT_STEM}_$(date +%Y%m%d_%H%M%S)"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD=(
    python "$PROJECT_ROOT/experiments/voxelibre/eval_voxelibre_zero_shot.py"
    --ckpt "$CKPT"
    --dataset_stats_path "$DATASET_STATS"
    --voxelibre_game_dir "$VOXELIBRE_GAME_DIR"
    --tasks "$TASKS"
    --episodes_per_task "$EPISODES"
    --split_number "$SPLIT"
    --gpus_per_worker "$GPUS_PER_WORKER"
    --run_dir_prefix "$RUN_DIR_PREFIX"
    --seed "$SEED"
    --output_dir "$OUTPUT"
)
if [[ -n "$TASK_CONFIG" ]]; then
    CMD+=(--task_config "$TASK_CONFIG")
fi
if [[ -n "$CRAFTIUM_MINETEST_DIR" ]]; then
    CMD+=(--craftium_minetest_dir "$CRAFTIUM_MINETEST_DIR")
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    CMD+=("${EXTRA[@]}")
fi

echo "VoxeLibre zero-shot: tasks=$TASKS episodes/task=$EPISODES actors=$SPLIT seed=$SEED"
echo "Output: $OUTPUT"
echo "Temporary Minetest runs: $RUN_DIR_PREFIX"
mkdir -p "$RUN_DIR_PREFIX"
if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "ERROR: xvfb-run is required for Craftium rendering (install xvfb and xauth)" >&2
    exit 1
fi
xvfb-run -a "${CMD[@]}"
