# CLUSTER SETUP — so101 training on cs-zhang-net-01 (`zz-bw`)

Briefing for the cluster-side agent. Proven machine recipe (validated on the
cosmos3-gen container — see ~/Projects/cosmos3-gen/COSMOS-bw.MD):

- **No sudo, no /etc/subuid** → builds go through **proot at `~/bin/proot`**
  (`export PATH="$HOME/bin:$PATH"` first). Never pass `--fakeroot`; expect
  harmless chown/setgroups warnings.
- **Staging MUST be local ext4** (`/export/scratch/thakk100`, ~825 G free):
  `SINGULARITY_TMPDIR` on NFS (`/xtra`) fails with `unpriv.lremovexattr`.
  `/tmp` (9.8 G) is too small for builds.
- **Home = 10 GiB quota** — code only, never caches/SIF (`csequota -s`).
- **sm_120 Blackwell**: cu128 wheels only (our stack pins torch 2.11.0+cu128 ✓).
  Driver 580.159.03 / CUDA 13.0.
- GPU 1 is often busy with another user's job — check `nvidia-smi` first,
  never kill PIDs you don't own.
- Scratch is shared + purge-eligible: durable outputs go to `/xtra/thakk100`,
  the SIF lives on scratch (rebuild is cheap once cached).

```
/export/scratch/thakk100/        ← SIF + singularity staging (local NVMe, fast)
├── so101-train.sif
├── sing-tmp/  sing-cache/
/xtra/thakk100/so101/            ← durable outputs (NFS, 7.6 T)
├── runs/  logs/  docker.env
/home/thakk100/Projects/so101_mnri   ← repo clone (code only)
```

## Step 0 — preflight (run as-is, expect all green)

```bash
ssh zz-bw
export PATH="$HOME/bin:$PATH"          # proot lives here
proot --version                        # 5.3.1
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv   # pick a free GPU
singularity --version                  # 4.1.1
df -h /export/scratch/thakk100 | tail -1   # staging space (~825 G)
csequota -s                            # home quota check (NOT `quota`)
curl -sI https://pypi.nvidia.com -o /dev/null -w '%{http_code}\n'   # 200
```

## Step 1 — staging + cache redirection (load-bearing — NFS tmpdir breaks builds)

```bash
export PATH="$HOME/bin:$PATH"                      # proot — required for build
export SCRATCH=/export/scratch/thakk100            # local ext4 — NOT /xtra (NFS xattr bug)
export SINGULARITY_TMPDIR=$SCRATCH/sing-tmp
export SINGULARITY_CACHEDIR=$SCRATCH/sing-cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" /xtra/thakk100/so101/{runs,logs}
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

## Step 4 — build the SIF (~20–40 min under proot; run inside tmux)

```bash
export PATH="$HOME/bin:$PATH"
export SCRATCH=/export/scratch/thakk100
export SINGULARITY_TMPDIR=$SCRATCH/sing-tmp SINGULARITY_CACHEDIR=$SCRATCH/sing-cache
tmux new -s so101build     # builds survive disconnects
cd /home/thakk100/Projects/so101_mnri && git pull
singularity build $SCRATCH/so101-train.sif docker/apptainer/so101_train.def
singularity test $SCRATCH/so101-train.sif     # version-parity check
```

No `--fakeroot` (it cannot work here); proot emits harmless
chown/setgroups warnings. The build asserts exact versions (isaaclab
3.0.0b2 / isaacsim 6.0.1 / rsl-rl 5.4.0 / warp 3.10.0.1) and fails loudly
on drift — do not "fix" by loosening.

## Step 5 — smoke tests (in order; each proves one more layer)

```bash
export PATH="$HOME/bin:$PATH"
SCRATCH=/export/scratch/thakk100
cd /xtra/thakk100/so101
S="singularity run --nv --cleanenv --bind $PWD/runs:/workspace/mounts/runs --bind $PWD/logs:/workspace/mounts/logs --env-file docker.env $SCRATCH/so101-train.sif"

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
export PATH="$HOME/bin:$PATH"
cd /xtra/thakk100/so101
S="singularity run --nv --cleanenv --bind $PWD/runs:/workspace/mounts/runs --bind $PWD/logs:/workspace/mounts/logs --env-file docker.env /export/scratch/thakk100/so101-train.sif"

# suggested first real run (HANDOFF open work #1):
$S --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs 4096
#   (defaults to 1500 iterations; add --max-iterations N to shorten)
# wandb lands in project "so101-rl"; checkpoints in $PWD/runs/isaaclab/...
```

Second GPU: same command with `--nv` uses GPU 0 by default; pin via
`CUDA_VISIBLE_DEVICES=1 $S ...`. **Check `nvidia-smi` first** — GPU 1 is
often busy with another user's job; never kill PIDs you don't own. Both
cards have 96 GB, so two concurrent experiments fit easily when free.

## Outputs & monitoring

- checkpoints/TensorBoard: `/xtra/thakk100/so101/runs/<backend>/<task>_<algo>/<stamp>/`
- wandb: project `so101-rl` (entity thakk100-dhyan-home), live during training
- container-internal clone: `/tmp/so101_mnri` (node-local ext4 — kit-friendly;
  wiped on reboot, re-clones automatically; do NOT bind /tmp to NFS)
- the SIF lives on scratch (purge-eligible) — rebuild is cheap once
  `$SINGULARITY_CACHEDIR` is warm; durable outputs are on /xtra

## Troubleshooting

| Symptom | Fix |
|---|---|
| build: `no mapping entry found in /etc/subuid` | you passed `--fakeroot` — don't; proot mode is the only path here |
| build: `unpriv.lremovexattr: invalid argument` | SINGULARITY_TMPDIR is on NFS — must be /export/scratch (local ext4) |
| build: `disk quota exceeded` | caches/SIF landed on home — redo Step 1 exports |
| build: `proot: command not found` | `export PATH="$HOME/bin:$PATH"` |
| clone 404 in entrypoint | repo private → GITHUB_TOKEN in docker.env |
| `submodules not fetched` fatal | same — token missing/expired |
| 5b silent death (exit 0) | free RAM/GPU, retry; kit log: rerun with `--bind /xtra/thakk100/so101/logs/kit:/root/.nvidia-omniverse/logs` |
| wandb prompts for login interactively | WANDB_API_KEY missing in docker.env (or add WANDB_MODE=offline) |
| Blackwell kernel errors in torch | confirm cu128 inside SIF: `singularity exec --nv $SCRATCH/so101-train.sif python3 -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"` — want 2.11.0+cu128 and sm_120 |
