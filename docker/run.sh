#!/usr/bin/env bash
# =============================================================================
# Host-side runner for the SO-101 training container.
#
# Usage:
#   docker/run.sh [options] [-- extra rl.train args...]
#
# Options:
#   --image NAME      image to run (default $SO101_IMAGE or docker.io/$USER/so101-isaac-train:3.0.0b2)
#   --branch NAME     git branch to clone (default feature/mjlab)
#   --env-file FILE   secrets file (default docker/.env; created from .env.example if missing)
#   --source          bind-mount THIS checkout instead of cloning (SO101_USE_MOUNT=1)
#   --backend NAME    isaaclab | mujoco | mjlab   (default isaaclab)
#   --task TASK       task id (default SO101-PickLift-Single-v0)
#   --algo NAME       skrl | rsl_rl               (default skrl)
#   --num-envs N      parallel envs
#   --iters N         --max-iterations
#   --film            pass the FiLM policy flag through
#   --name NAME       container name (default so101-train-<ts>)
#   --network-host    use host networking (wandb behind VPN/proxy)
#   -h | --help
#
# Examples:
#   docker/run.sh --task SO101-PickLift-Single-v0 --num-envs 4096 --iters 1500
#   docker/run.sh --backend mujoco --task push_t --iters 200
#   docker/run.sh --source -- --backend isaaclab --task SO101-CylReach-Single-v0 --algo skrl
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # workspace root

IMAGE="${SO101_IMAGE:-thdhyan/so101-isaac-train:3.0.0b2}"
BRANCH="feature/mjlab"
ENV_FILE="docker/.env"
SOURCE_MOUNT=0
CONT_NAME="so101-train-$(date +%m%d-%H%M%S)"
NET_ARGS=()
BACKEND="isaaclab"; TASK="SO101-PickLift-Single-v0"; ALGO="skrl"
NUM_ENVS=""; ITERS=""; FILM=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --source) SOURCE_MOUNT=1; shift ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --algo) ALGO="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --film) FILM=1; shift ;;
        --name) CONT_NAME="$2"; shift 2 ;;
        --network-host) NET_ARGS=(--network host); shift ;;
        -h|--help) grep '^#' "$0" | head -30; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

if [ ! -f "$ENV_FILE" ] && [ -f "docker/.env.example" ]; then
    cp docker/.env.example "$ENV_FILE"
    echo "[run.sh] created $ENV_FILE from example — add WANDB_API_KEY / GITHUB_TOKEN"
fi

TRAIN_ARGS=(--backend "$BACKEND" --task "$TASK" --algo "$ALGO")
[ -n "$NUM_ENVS" ] && TRAIN_ARGS+=(--num-envs "$NUM_ENVS")
[ -n "$ITERS" ] && TRAIN_ARGS+=(--max-iterations "$ITERS")
[ "$FILM" = 1 ] && TRAIN_ARGS+=(--film)

MOUNT_ARGS=(-v "$PWD/logs:/workspace/mounts/logs" -v "$PWD/runs:/workspace/mounts/runs")
SOURCE_ARGS=()
if [ "$SOURCE_MOUNT" = 1 ]; then
    MOUNT_ARGS+=(-v "$PWD:/workspace/so101_mnri")
    SOURCE_ARGS=(-e SO101_USE_MOUNT=1)
fi

mkdir -p logs runs
echo "[run.sh] ${CONT_NAME}: ${TRAIN_ARGS[*]} ${EXTRA[*]:+${EXTRA[*]}}"
exec docker run --gpus all --rm --env-file "$ENV_FILE" \
    "${NET_ARGS[@]}" "${MOUNT_ARGS[@]}" "${SOURCE_ARGS[@]}" \
    -e SO101_GIT_BRANCH="$BRANCH" -e MUJOCO_GL=egl \
    --name "$CONT_NAME" \
    "$IMAGE" "${TRAIN_ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
