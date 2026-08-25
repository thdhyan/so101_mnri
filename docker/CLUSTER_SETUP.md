# CLUSTER SETUP — so101 training on cs-zhang-net-01 (`zz-bw`)

Briefing for the cluster-side agent. Everything writable lives under
**`/xtra/thakk100/so101/`** (7.6 TB free NFS tank). Do NOT rely on `$HOME`
(quota). Server: 2× RTX PRO 6000 Blackwell (96 GB), 64 cores, 503 GB RAM,
`singularity-ce 4.1.1`, **no Slurm** (direct runs / tmux).

```
/xtra/thakk100/so101/            ← everything happens here
├── so101_mnri/                  ← git clone of this repo (build context)
├── so101-train.sif              ← built SIF (~25 GB unpacked; cache elsewhere)
├── docker.env                   ← secrets, chmod 600, NEVER committed
├── runs/                        ← bind → container rl/runs (checkpoints, TB)
├── logs/                        ← bind → container /workspace/mounts/logs
└── cache/                       ← APPTAINER_CACHEDIR + TMPDIR (build blobs)
```

## Step 0 — preflight (run as-is, expect all green)

```bash
ssh zz-bw
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv   # driver ≥ 580 for Blackwell+kit
singularity --version                                                   # 4.1.1
df -h /xtra/thakk100 | tail -1                                          # ~7.6T free
curl -sI https://pypi.nvidia.com -o /dev/null -w '%{http_code}\n'       # 200
```

## Step 1 — layout + cache redirection (home quota!)

```bash
mkdir -p /xtra/thakk100/so101/{runs,logs,cache}
cat >> ~/.bashrc <<'EOF'
export APPTAINER_CACHEDIR=/xtra/thakk100/so101/cache
export APPTAINER_TMPDIR=/xtra/thakk100/so101/cache
export SINGULARITY_CACHEDIR=/xtra/thakk100/so101/cache
export SINGULARITY_TMPDIR=/xtra/thakk100/so101/cache
EOF
source ~/.bashrc
```

## Step 2 — get the repo onto the cluster

```bash
cd /xtra/thakk100/so101
git clone https://github.com/thdhyan/so101_mnri.git so101_mnri
cd so101_mnri && git checkout feature/mjlab && git submodule update --init
```

If the clone 404s, the repo is private → create a fine-grained token at
https://github.com/settings/tokens (repo read scope), then either
`git clone https://<TOKEN>@github.com/thdhyan/so101_mnri.git` once, or put
`GITHUB_TOKEN=` in docker.env (the container entrypoint uses it the same way).
**The laptop side has ~40 uncommitted files on `feature/mjlab` — they must be
committed and pushed BEFORE this step or the cluster gets stale code.**

## Step 3 — secrets (`docker.env`)

```bash
cat > /xtra/thakk100/so101/docker.env <<'EOF'
WANDB_API_KEY=<key from the laptop: grep -A3 api.wandb.ai ~/.netrc>
GITHUB_TOKEN=<only if the repo is private>
EOF
chmod 600 /xtra/thakk100/so101/docker.env
```

## Step 4 — build the SIF (~30 GB downloads → cache on /xtra; ~30–60 min)

```bash
cd /xtra/thakk100/so101/so101_mnri
singularity build /xtra/thakk100/so101/so101-train.sif docker/apptainer/so101_train.def
singularity test /xtra/thakk100/so101/so101-train.sif     # version-parity check
```

The build asserts exact versions (isaaclab 3.0.0b2 / isaacsim 6.0.1 / rsl-rl
5.4.0 / warp 3.10.0.1) and fails loudly on drift — do not "fix" by loosening.

## Step 5 — smoke tests (in order; each proves one more layer)

```bash
cd /xtra/thakk100/so101
S="singularity run --nv --cleanenv --bind $PWD/runs:/workspace/mounts/runs --bind $PWD/logs:/workspace/mounts/logs --env-file docker.env so101-train.sif"

# 5a. clone + custom-skrl install + mujoco backend + a new env (CPU, ~3 min)
$S --backend mujoco --task push_t --algo skrl --max-iterations 2 --no-wandb --device cpu

# 5b. Isaac backend, tiny (needs GPU + kit boot, ~4 min)
$S --backend isaaclab --task SO101-CylReach-Single-v0 --algo skrl --num-envs 64 --max-iterations 3 --no-wandb
```

5a proves: runtime clone, submodule fetch, `pip install -e third_party/skrl`
(custom dataclass skrl), env import. 5b proves: Kit boots on Blackwell,
Isaac task registration, full PPO loop. If 5a fails on submodule fetch →
GITHUB_TOKEN missing (repo private). If 5b dies silently (exit 0, log just
stops) → check `free -g`/`nvidia-smi`, retry (known Isaac flake, see repo
test.md).

## Step 6 — real training runs (tmux; no Slurm on this box)

```bash
tmux new -s so101
cd /xtra/thakk100/so101
S="singularity run --nv --cleanenv --bind $PWD/runs:/workspace/mounts/runs --bind $PWD/logs:/workspace/mounts/logs --env-file docker.env so101-train.sif"

# suggested first real run (HANDOFF open work #1):
$S --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs 4096
#   (defaults to 1500 iterations; add --max-iterations N to shorten)
# wandb lands in project "so101-rl"; checkpoints in $PWD/runs/isaaclab/...
```

Second GPU: same command with `--nv` uses GPU 0 by default; pin via
`CUDA_VISIBLE_DEVICES=1 $S ...`. Run two different experiments concurrently
(96 GB each — no contention).

## Outputs & monitoring

- checkpoints/TensorBoard: `/xtra/thakk100/so101/runs/<backend>/<task>_<algo>/<stamp>/`
- wandb: project `so101-rl` (entity thakk100-dhyan-home), live during training
- container-internal clone: `/tmp/so101_mnri` (node-local disk — kit-friendly;
  wiped on reboot, re-clones automatically on next run; do NOT bind /tmp to NFS)

## Troubleshooting

| Symptom | Fix |
|---|---|
| build: "No space left" | APPTAINER_CACHEDIR/TMPDIR not exported (Step 1) |
| clone 404 in entrypoint | repo private → GITHUB_TOKEN in docker.env |
| `submodules not fetched` fatal | same — token missing/expired |
| 5b silent death (exit 0) | free RAM/GPU, retry; check `~/.local/share/...` no — kit log is inside container: rerun with `--bind /xtra/thakk100/so101/logs/kit:/root/.nvidia-omniverse/logs` |
| wandb prompts for login interactively | WANDB_API_KEY missing in docker.env (or add WANDB_MODE=offline) |
| Blackwell kernel errors in torch | confirm the SIF torch is cu128 (`singularity exec SIF python3 -c "import torch;print(torch.__version__)"`) |
