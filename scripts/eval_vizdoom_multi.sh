#!/bin/bash
set -euo pipefail

# VizDoom four-map evaluation. Change the defaults here or override them with
# flags; arguments after `--` are forwarded to eval_vizdoom_multi.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CKPT="/path/to/vizdoom_checkpoint.pt"
DATASET_STATS="/path/to/vizdoom_dataset_stats.json"
MODE="all"
SCENARIOS=""
EPISODES=50
SPLIT=40
GPUS_PER_WORKER="0.2"
OUTPUT=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --dataset_stats) DATASET_STATS="$2"; shift 2 ;;
        --scenarios) SCENARIOS="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --gpus_per_worker) GPUS_PER_WORKER="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --) shift; EXTRA+=("$@"); break ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

case "$MODE" in
    all)
        SCENARIOS="all"
        ;;
    custom)
        if [[ -z "$SCENARIOS" ]]; then
            echo "ERROR: --mode custom requires --scenarios MAP1,MAP2,..." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unknown mode '$MODE' (expected: all | custom)" >&2
        exit 1
        ;;
esac

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$DATASET_STATS" ]] || { echo "ERROR: dataset stats not found: $DATASET_STATS" >&2; exit 1; }
python -c 'import vizdoom' >/dev/null 2>&1 || {
    echo "ERROR: VizDoom is not installed in the active Python environment." >&2
    echo "Install the project-compatible package, for example: pip install vizdoom==1.2.4" >&2
    exit 1
}

if [[ -z "$OUTPUT" ]]; then
    CKPT_STEM="$(basename "$CKPT" .pt)"
    OUTPUT="$PROJECT_ROOT/evaluate_results/vizdoom/${CKPT_STEM}_$(date +%Y%m%d_%H%M%S)"
fi

echo "Scenarios=$SCENARIOS  Episodes/map=$EPISODES  Split=$SPLIT  Output=$OUTPUT"
echo "Primary metric: average episode reward (reported independently per map)"

CMD=(
    python "$PROJECT_ROOT/experiments/vizdoom/eval_vizdoom_multi.py"
        --ckpt "$CKPT"
        --dataset_stats_path "$DATASET_STATS"
        --scenarios "$SCENARIOS"
        --episodes_per_map "$EPISODES"
        --split_number "$SPLIT"
        --gpus_per_worker "$GPUS_PER_WORKER"
        --output_dir "$OUTPUT"
        --save_video
)
[[ ${#EXTRA[@]} -gt 0 ]] && CMD+=("${EXTRA[@]}")

cd "$PROJECT_ROOT"
xvfb-run -a "${CMD[@]}"
