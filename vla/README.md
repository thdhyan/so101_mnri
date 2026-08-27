# vla/ — VLA finetuning + Imitation Learning pipeline

Train any LeRobot-supported policy on SO-101 demonstration data with a
single entry point, swappable backends, and remote SSH training.

```bash
# ── Supported backends (any LeRobot policy) ──
python -m vla.finetune --list-backends

# ── Local training ──
python -m vla.finetune --backend smolvla --dataset thakk100/so101_duck_push
python -m vla.finetune --backend act --dataset thakk100/so101_duck_push --steps 5000
python -m vla.finetune --backend diffusion --dataset thakk100/so101_duck_push
python -m vla.finetune --backend pi0 --dataset thakk100/so101_duck_push
python -m vla.finetune --backend groot --lora --dataset thakk100/so101_duck_push

# ── Remote training (SSH to GPU server) ──
python -m vla.finetune --backend groot --dataset thakk100/so101_duck_push \
    --remote-host dl --steps 20000

python -m vla.finetune --backend act --dataset thakk100/so101_duck_push \
    --remote-host lambda --batch-size 32
```

Add `--dry-run` to print the underlying command without running it.

## Supported Backends

| Backend | Policy Type | Params | Base Model | Default Steps | Notes |
|---|---|---|---|---|---|
| `act` | ACT | 52M | — | 5,000 | Fastest, good baseline |
| `diffusion` | Diffusion Policy | 263M | — | 10,000 | Multimodal actions |
| `vqbet` | VQ-BeT | — | — | 10,000 | Discrete action tokens |
| `smolvla` | SmolVLA | 450M | `lerobot/smolvla_base` | 10,000 | Language-conditioned, fast |
| `pi0` | Pi0 | 3B | `lerobot/pi0` | 15,000 | VLA generalist |
| `pi0fast` | Pi0Fast | 3B | `lerobot/pi0fast` | 15,000 | Fast Pi0 variant |
| `groot` | GR00T N1.7 | 3B | `nvidia/GR00T-N1.7-3B` | 20,000 | NVIDIA foundation model |
| `xvla` | XVLA | — | — | 20,000 | Cross-embodiment VLA |
| `evo1` | EVO1 | — | — | 15,000 | Evolution-based |
| `tdmpc` | TDMPC | — | — | 50,000 | RL-based |

## Compute Fleet

| Machine | SSH Alias | GPUs | Use For |
|---|---|---|---|
| dl | `dl` | 4× RTX 6000 Ada (192GB) | GR00T, multi-GPU |
| lambda | `lambda` | Cloud GPU | Single-GPU training |
| zz-bw | `zz-bw` | GPU box | Training, eval |
| zz-dt | `zz-dt` | Desktop GPU | Quick experiments |
| agate | `agate` | SLURM cluster | Large batch jobs |

```bash
# Remote training (auto-syncs code + dataset)
python -m vla.finetune --backend groot --dataset thakk100/so101_duck_push \
    --remote-host dl --steps 20000 --lora

# Dry run (print command without executing)
python -m vla.finetune --backend act --dataset thakk100/so101_duck_push \
    --remote-host zz-bw --dry-run
```

## Docker / Singularity

Build containers for reproducible training on any machine:

```bash
# Build Docker image
docker build -t so101-il:latest -f docker/Dockerfile.il .

# Train inside container
docker run --gpus all --rm \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/vla/runs:/workspace/vla/runs \
    so101-il:latest \
    --backend act --dataset thakk100/so101_duck_push

# Convert to Singularity for agate/SLURM
singularity build so101-il.sif docker://so101-il:latest
```

## Environment Setup

### Quick install (uv)
```bash
# Main venv (for ACT, SmolVLA, Diffusion, VQ-BeT)
uv pip install -e ".[all]"

# GR00T needs flash-attn
uv pip install flash-attn --no-build-isolation
```

### Per-backend venvs (if dependency conflicts)
```bash
# ACT / SmolVLA / Diffusion (Python 3.12)
uv venv --python 3.12 .venv-vla
.venv-vla/bin/pip install torch torchvision
git clone https://github.com/huggingface/lerobot third_party/lerobot
.venv-vla/bin/pip install -e "third_party/lerobot[all]"

# GR00T N1.7 (Python 3.12, needs transformers>=5.4)
uv venv --python 3.12 .venv-vla-groot
.venv-vla-groot/bin/pip install -e "third_party/lerobot[groot]"
.venv-vla-groot/bin/pip install flash-attn --no-build-isolation
```

## Dataset spec

LeRobotDataset episodes with:

- cameras: `observation.images.wrist` + `observation.images.global` (or `front`)
- state: `observation.state` (6 joint angles + gripper, rad)
- action: `action` (next-step joint targets, rad)
- `task` string per episode (the language instruction)

## Pipeline

| Step | Command | Notes |
|---|---|---|
| Record demos | `python -m vla.record_demos` | Leader arm teleop → LeRobotDataset |
| (Optional) Augment | `python -m vla.augment` | Multiply episodes 3-4× |
| Train policy | `python -m vla.finetune` | Any LeRobot backend |
| Evaluate sim | `python -m vla.rollout` | Closed-loop eval in MuJoCo |
| Deploy real | `python -m vla.deploy` | Real follower arm |
| Record video | `python -m vla.record_video` | Success/failure videos |

## Files

| File | Purpose |
|---|---|
| `finetune.py` | Unified wrapper — all LeRobot backends + remote SSH |
| `record_demos.py` | Record real-world teleop to LeRobotDataset |
| `augment.py` | Dataset augmentation transforms |
| `rollout.py` | Closed-loop sim evaluation |
| `deploy.py` | Real robot policy deployment |
| `record_video.py` | Success video recording |
| `gr00t_prepare_dataset.py` | GR00T v3→v2 dataset conversion |

## Status / roadmap

- [x] `vla/finetune.py` — unified wrapper (all LeRobot backends)
- [x] `vla/gr00t_prepare_dataset.py` — v3→v2 + modality.json + camera remap
- [x] `vla/record_demos.py` — real-world teleop recorder
- [x] `vla/augment.py` — augmentation transforms
- [x] `vla/rollout.py` — closed-loop eval in sim
- [x] `vla/deploy.py` — real robot deployment
- [x] `vla/record_video.py` — success video recording
- [x] Remote SSH training support
- [x] Docker / Singularity container support
- [ ] First end-to-end run: record 100 demos → train → deploy → record video
- [ ] Create duck-push sim env (optional, for pre-training)
