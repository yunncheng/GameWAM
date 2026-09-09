#!/bin/bash
set -euo pipefail

# GameWAM multi-task Minecraft evaluation
#
# Usage:
#   bash scripts/eval_mc_multi.sh [--mode MODE] [--ckpt PATH] [--dataset_stats PATH]
#                                 [--tasks CSV] [--split N] [--episodes N]
#                                 [--gpus_per_worker N] [-- ...]
#
# --mode:            mini | all | custom                       (default: custom)
# --episodes:        episodes per task
# --gpus_per_worker: Ray GPU resources per worker (default: 1)
#
# Only the frequently-changed knobs are flags. Everything else (device, video,
# ...) is tuned via the constants below or passed through to
# eval_mc_multi.py after `--`.
#
# Examples:
#   bash scripts/eval_mc_multi.sh --mode mini
#   bash scripts/eval_mc_multi.sh --mode mini --split 4
#   bash scripts/eval_mc_multi.sh --mode custom --tasks "mine_block:stone,kill_entity:zombie"
#   bash scripts/eval_mc_multi.sh --mode mini --gpus_per_worker 0.5 --split 16
#   bash scripts/eval_mc_multi.sh --mode all --episodes 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Tune these if needed ----
CKPT="/path/to/minecraft_checkpoint.pt"
DATASET_STATS="/path/to/minecraft_dataset_stats.json"
SPLIT=40
EPISODES=5
GPUS_PER_WORKER="0.2"
CUSTOM_TASKS_DEFAULT="mine_block:stone,kill_entity:zombie,craft_item:crafting_table"
MINESTUDIO_DIR="${MINESTUDIO_DIR:-$PROJECT_ROOT/.minestudio}"
export MINESTUDIO_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Parse ----
MODE="custom"
TASKS=""
OUTPUT=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)           MODE="$2";          shift 2 ;;
        --ckpt)           CKPT="$2";          shift 2 ;;
        --dataset_stats)  DATASET_STATS="$2"; shift 2 ;;
        --tasks)          TASKS="$2";         shift 2 ;;
        --split)          SPLIT="$2";         shift 2 ;;
        --episodes)       EPISODES="$2";      shift 2 ;;
        --gpus_per_worker) GPUS_PER_WORKER="$2"; shift 2 ;;
        --output)         OUTPUT="$2";        shift 2 ;;
        --)       shift; EXTRA+=("$@"); break ;;
        *)        EXTRA+=("$1"); shift ;;
    esac
done

# ---- Derive mode-specific settings ----
case "$MODE" in
    mini|all)
        TASK_SET="$MODE"
        ;;
    custom)
        TASK_SET="custom"
        [[ -z "$TASKS" ]] && TASKS="$CUSTOM_TASKS_DEFAULT"
        ;;
    *)
        echo "ERROR: unknown mode '$MODE' (expected: mini | all | custom)" >&2
        exit 1
        ;;
esac

# ---- Derive output dir ----
CKPT_STEM="$(basename "$CKPT" .pt)"
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="./evaluate_results/minecraft/${CKPT_STEM}_${TASK_SET}_$(date +%Y%m%d_%H%M%S)"
fi

echo "Mode=$TASK_SET  Split=$SPLIT  Episodes/task=$EPISODES  Output=$OUTPUT"
[[ -n "$GPUS_PER_WORKER" ]] && echo "GPUs per worker: $GPUS_PER_WORKER"
[[ "$TASK_SET" == "custom" ]] && echo "Tasks: $TASKS"
echo "MineStudio dir: $MINESTUDIO_DIR"

# ---- Prepare MineStudio simulator engine before Ray workers start ----
mkdir -p "$MINESTUDIO_DIR"
MINESTUDIO_ENGINE_JAR="$MINESTUDIO_DIR/engine/build/libs/mcprec-6.13.jar"
MINESTUDIO_LOCK="$MINESTUDIO_DIR/.engine_download.lock"
(
    flock 9
    if [[ ! -f "$MINESTUDIO_ENGINE_JAR" ]]; then
        echo "MineStudio engine not found, downloading to $MINESTUDIO_DIR ..."
        python - <<'PY'
from minestudio.simulator.entry import check_engine
check_engine(skip_confirmation=True)
PY
    else
        echo "MineStudio engine found: $MINESTUDIO_ENGINE_JAR"
    fi
) 9>"$MINESTUDIO_LOCK"

# ---- Run ----
CMD=(
    python "$PROJECT_ROOT/experiments/minecraft/eval_mc_multi.py"
        --ckpt "$CKPT"
        --dataset_stats_path "$DATASET_STATS"
        --task_set "$TASK_SET"
        --episodes_per_task "$EPISODES"
        --split_number "$SPLIT"
        --output_dir "$OUTPUT"
        --save_video
)
[[ "$TASK_SET" == "custom" ]] && CMD+=(--tasks "$TASKS")
[[ -n "$GPUS_PER_WORKER" ]] && CMD+=(--gpus_per_worker "$GPUS_PER_WORKER")
[[ ${#EXTRA[@]} -gt 0 ]] && CMD+=("${EXTRA[@]}")

xvfb-run -a "${CMD[@]}"
