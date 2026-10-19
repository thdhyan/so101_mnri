# docker/ — containerized SO-101 training (Docker + Apptainer/Singularity)

Parity-pinned to the local `.venv` stack: python 3.12, torch 2.11.0+cu128,
isaacsim 6.0.1.0, isaaclab 3.0.0b2.post1, rsl-rl-lib 5.4.0, mujoco
3.10.0 + mujoco-warp 3.10.0.1 (known-good pin, see test.md). The **custom
skrl fork is installed at runtime** from the repo's `third_party/skrl`
submodule — one image serves any fork revision.

## One-time build (workstation with docker; needs ~45 GB disk)

```bash
docker build -f docker/Dockerfile.base -t thdhyan/so101-isaac-base:3.0.0b2 .
docker push thdhyan/so101-isaac-base:3.0.0b2          # optional but recommended
docker build -f docker/Dockerfile -t thdhyan/so101-isaac-train:3.0.0b2 .
docker push thdhyan/so101-isaac-train:3.0.0b2
```

## Local runs

```bash
echo 'WANDB_API_KEY=...' > docker/.env                # from docker/.env.example

docker/run.sh --task SO101-PickLift-Single-v0 --num-envs 4096 --iters 1500
docker/run.sh --backend mujoco --task push_t --iters 200          # CPU-ish smoke
docker/run.sh --source -- --task SO101-CylReach-Single-v0         # uncommitted code
```

Outputs land in `./runs` (bind-mounted to the container's `rl/runs`) and
`./logs`. WandB needs `WANDB_API_KEY` in `docker/.env` (or `WANDB_MODE=offline`).

## Cluster (Singularity/Apptainer, no docker daemon)

```bash
# preferred — pull the pushed image:
apptainer pull so101-train.sif docker://docker.io/thdhyan/so101-isaac-train:3.0.0b2

# or build from the def:
apptainer build so101-train.sif docker/apptainer/so101_train.def

# run (basic PPO, custom skrl):
apptainer run --nv --cleanenv \
    --bind "$PWD/runs:/workspace/mounts/runs" \
    --bind "$PWD/logs:/workspace/mounts/logs" \
    --env-file docker/.env \
    so101-train.sif \
    --backend isaaclab --task SO101-PickLift-Single-v0 --algo skrl --num-envs 4096
```

Slurm: wrap the `apptainer run` line in a batch script with
`#SBATCH --gres=gpu:1` + `#SBATCH --time=24:00:00`. Bind a scratch dir over
`/tmp` (`--bind /scratch/$USER/so101tmp:/tmp`) to cache the runtime clone.

## Notes

- The entrypoint clones the repo at runtime (`SO101_REPO_URL` /
  `SO101_GIT_BRANCH`, default `feature/mjlab`) with submodules, then
  `pip install -e third_party/skrl`. Private repos need `GITHUB_TOKEN` in
  `.env`. **Push your branch before cluster runs** — the image contains no
  source. `docker/run.sh --source` bind-mounts this checkout instead
  (local iteration on uncommitted code).
- `SO101_INSTALL_MJLAB=1` additionally installs `third_party/mjlab` and
  re-pins `mujoco-warp==3.10.0.1` (mjlab deps otherwise upgrade it and break
  indexing — test.md).
- The base image asserts version parity at build time and fails loudly on
  drift; don't bump pins without re-validating locally first.
- Kit/EULA: `OMNI_KIT_ACCEPT_EULA=YES` is baked into the image.
