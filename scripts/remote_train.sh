#!/usr/bin/env bash
# remote_train.sh — Deploy and train on a remote GPU machine via SSH + Docker.
#
# Usage:
#   ./scripts/remote_train.sh --host dl --policy smolvla --dataset thakk100/so101_duck_push
#   ./scripts/remote_train.sh --host lambda --policy groot --lora --steps 20000
#   ./scripts/remote_train.sh --host zz-bw --policy act --batch-size 16
#   ./scripts/remote_train.sh --host agate --slurm --policy groot  # SLURM on agate

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST=""
POLICY=""
DATASET="thakk100/so101_duck_push"
STEPS=10000
BATCH_SIZE=""
LORA=""
SLURM=false
DRY_RUN=false
REMOTE_DIR="~/projects/so101_mnri"
IMAGE="so101-il:latest"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)      HOST="$2"; shift 2 ;;
    --policy)    POLICY="$2"; shift 2 ;;
    --dataset)   DATASET="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lora)      LORA="--lora"; shift ;;
    --slurm)     SLURM=true; shift ;;
    --dry-run)   DRY_RUN=true; shift ;;
    --image)     IMAGE="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$HOST" || -z "$POLICY" ]]; then
  echo "Usage: $0 --host <dl|lambda|zz-bw|zz-dt> --policy <smolvla|act|groot|pi0|diffusion|vqbet> [options]"
  exit 1
fi

# ── Build training command ────────────────────────────────────────────────────
TRAIN_CMD="python3 -m vla.finetune --backend $POLICY --dataset $DATASET --steps $STEPS"
[[ -n "$BATCH_SIZE" ]] && TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
[[ -n "$LORA" ]] && TRAIN_CMD="$TRAIN_CMD $LORA"

echo "═══════════════════════════════════════════════════════════════"
echo "  Remote Training"
echo "  Host:    $HOST"
echo "  Policy:  $POLICY"
echo "  Dataset: $DATASET"
echo "  Steps:   $STEPS"
echo "  Command: $TRAIN_CMD"
echo "═══════════════════════════════════════════════════════════════"

# ── Sync code to remote ───────────────────────────────────────────────────────
echo ""
echo "→ Syncing code to $HOST:$REMOTE_DIR ..."
rsync -avz --delete \
  --exclude '.venv*' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'vla/runs' \
  --exclude 'data' \
  --exclude 'docker' \
  ./ "$HOST:$REMOTE_DIR/"

echo ""
echo "→ Syncing dataset to $HOST:$REMOTE_DIR/data/ ..."
ssh "$HOST" "mkdir -p $REMOTE_DIR/data"
rsync -avz "data/$DATASET/" "$HOST:$REMOTE_DIR/data/$DATASET/" || \
  echo "  (dataset not found locally — will use HuggingFace Hub on remote)"

# ── Run training on remote ───────────────────────────────────────────────────
if $SLURM; then
  echo ""
  echo "→ Submitting SLURM job on $HOST ..."
  ssh "$HOST" "cd $REMOTE_DIR && sbatch --job-name=il_$POLICY \
    --gres=gpu:1 --time=24:00:00 --mem=32G \
    --wrap=\"$TRAIN_CMD\""
else
  echo ""
  echo "→ Running training on $HOST ..."
  ssh -t "$HOST" "cd $REMOTE_DIR && \
    if command -v docker &>/dev/null && docker image inspect $IMAGE &>/dev/null; then \
      docker run --gpus all --rm \
        -v $REMOTE_DIR/data:/workspace/data \
        -v $REMOTE_DIR/vla/runs:/workspace/vla/runs \
        -e PYTHONPATH=/workspace \
        $IMAGE $TRAIN_CMD; \
    else \
      $TRAIN_CMD; \
    fi"
fi

echo ""
echo "→ Done! Check outputs on $HOST:$REMOTE_DIR/vla/runs/"
