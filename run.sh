#!/usr/bin/env bash
set -euo pipefail

# Command catalog for data caching, training, and evaluation.
# Uncomment the blocks needed for the current run and replace /path/to values.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

GPUS="${GPUS:-8}"


# =============================================================================
# One-time model initialization
# =============================================================================

# Initialize the shared ActionDiT backbone from Wan2.2.
# python scripts/preprocess_action_dit_backbone.py \
#   --task-config mc_uncond_ar_224_4e-5 \
#   --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt


# =============================================================================
# Minecraft
# =============================================================================

# Cache task text embeddings.
# python scripts/precompute/text_embeds.py \
#   task=mc_uncond_ar_224_4e-5

# Cache VAE latents on multiple GPUs.
# torchrun --standalone --nproc_per_node="$GPUS" \
#   scripts/precompute/vae_latents.py \
#   task=mc_uncond_ar_224_4e-5 \
#   num_workers=8

# Train with DeepSpeed ZeRO-2.
# bash scripts/train_zero2.sh "$GPUS" \
#   task=mc_uncond_ar_224_4e-5

# Evaluate the compact Minecraft task set.
# bash scripts/eval_mc_multi.sh \
#   --mode mini \
#   --episodes 10 \
#   --ckpt /path/to/minecraft_checkpoint.pt \
#   --dataset_stats /path/to/minecraft_dataset_stats.json \
#   --output /path/to/evaluate_results/minecraft_mini

# Evaluate the complete Minecraft task set.
# bash scripts/eval_mc_multi.sh \
#   --mode all \
#   --episodes 5 \
#   --ckpt /path/to/minecraft_checkpoint.pt \
#   --dataset_stats /path/to/minecraft_dataset_stats.json \
#   --output /path/to/evaluate_results/minecraft_all


# =============================================================================
# ViZDoom
# =============================================================================

# Cache task text embeddings.
# python scripts/precompute/text_embeds.py \
#   task=vizdoom_uncond_ar_224_2e-4

# Cache VAE latents on multiple GPUs.
# torchrun --standalone --nproc_per_node="$GPUS" \
#   scripts/precompute/vae_latents.py \
#   task=vizdoom_uncond_ar_224_2e-4 \
#   num_workers=8

# Train with DeepSpeed ZeRO-2.
# bash scripts/train_zero2.sh "$GPUS" \
#   task=vizdoom_uncond_ar_224_2e-4

# Evaluate all four ViZDoom scenarios.
# bash scripts/eval_vizdoom_multi.sh \
#   --mode all \
#   --episodes 50 \
#   --ckpt /path/to/vizdoom_checkpoint.pt \
#   --dataset_stats /path/to/vizdoom_dataset_stats.json \
#   --output /path/to/evaluate_results/vizdoom


# =============================================================================
# Minecraft to VoxeLibre zero-shot transfer
# =============================================================================

# Prepare the pinned Craftium and VoxeLibre environment once.
# bash scripts/setup_voxelibre_eval.sh

# Evaluate all six zero-shot transfer tasks.
# bash scripts/eval_voxelibre_zero_shot.sh \
#   --tasks all \
#   --ckpt /path/to/minecraft_checkpoint.pt \
#   --dataset_stats /path/to/minecraft_dataset_stats.json \
#   --output /path/to/evaluate_results/voxelibre_zero_shot
