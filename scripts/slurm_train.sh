#!/bin/bash
# slurm_train.sh — Submit IL training job to agate (UMN MSI SLURM cluster).
#
# Usage:
#   sbatch scripts/slurm_train.sh --policy smolvla --dataset thakk100/so101_duck_push
#   sbatch scripts/slurm_train.sh --policy groot --lora --dataset thakk100/so101_duck_push --gres=gpu:a100:1
#
# Or run directly:
#   ./scripts/slurm_train.sh --policy act --dataset thakk100/so101_duck_push --test

#SBATCH --job-name=so101-il
#SBATCH --output=logs/il_%j.out
#SBATCH --error=logs/il_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
POLICY="smolvla"
DATASET="thakk100/so101_duck_push"
STEPS=10000
BATCH_SIZE=""
LORA=""
OUTPUT_DIR="outputs/sbatch"
TEST_MODE=false

# ── Parse args (handles both #SBATCH directives and command-line) ─────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --policy)     POLICY="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lora)       LORA="--lora"; shift ;;
    --gres)       # Override GPU request
      SLURM_JOB_GPU="$2"; shift 2 ;;
    --test)       TEST_MODE=true; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Environment ───────────────────────────────────────────────────────────────
module purge 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true

echo "═══════════════════════════════════════════════════════════════"
echo "  SLURM Job:    $SLURM_JOB_ID"
echo "  Node:         $SLURM_NODELIST"
echo "  GPUs:         $SLURM_GPUS_ON_NODE"
echo "  Policy:       $POLICY"
echo "  Dataset:      $DATASET"
echo "  Steps:        $STEPS"
echo "  Output dir:   $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"

# ── Activate venv if exists ──────────────────────────────────────────────────
VENV_DIR="${HOME}/projects/so101_mnri/.venv"
if [[ -d "$VENV_DIR" ]]; then
  source "$VENV_DIR/bin/activate"
fi

# ── Build and run training ───────────────────────────────────────────────────
TRAIN_CMD="python -m vla.finetune --backend $POLICY --dataset $DATASET --steps $STEPS --output-dir $OUTPUT_DIR"
[[ -n "$BATCH_SIZE" ]] && TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
[[ -n "$LORA" ]] && TRAIN_CMD="$TRAIN_CMD $LORA"

if $TEST_MODE; then
  echo "  [TEST] Would run: $TRAIN_CMD"
  exit 0
fi

echo "→ Running: $TRAIN_CMD"
mkdir -p logs "$OUTPUT_DIR"
eval "$TRAIN_CMD"

echo ""
echo "→ Job $SLURM_JOB_ID completed."
echo "  Outputs at: $OUTPUT_DIR"
