# vla/ — VLA finetuning harness (SmolVLA / GR00T / π0)

Train vision-language-action models on SO-101 demonstration data (sim
teleop or real arm) with a single entry point and swappable backends.

```bash
# SmolVLA (450M — runs on the 8 GB laptop GPU)
python -m vla.finetune --backend smolvla --dataset <hf-user>/so101_pick_lift --wandb

# GR00T N1.6-3B (LoRA + frozen DiT for <=8 GB; full FT needs 24 GB+)
python -m vla.finetune --backend groot --dataset vla/data/groot_converted/<ds> --lora

# π0 (3B, openpi weights via lerobot)
python -m vla.finetune --backend pi0 --dataset <hf-user>/so101_pick_lift
```

Add `--dry-run` to print the underlying command without running it.

## Backend notes (researched Aug 2026)

| | SmolVLA | GR00T N1.6 | π0 / π0.5 (openpi) |
|---|---|---|---|
| Size | 450M | 3B (Eagle VLM + DiT flow head) | 3B (PaliGemma + flow expert) |
| SO-101 fit | pretrained on SO-100/101 community data; beats ACT on real SO-101 | official NVIDIA SO-101 tutorial + sim-to-real course | generalist; needs new-embodiment finetune |
| Entry point | `lerobot-train` | Isaac-GR00T `scripts/gr00t_finetune.py` | `lerobot-train` (openpi repo for π0.5-native) |
| 8 GB GPU | full FT, batch 8–16 | LoRA rank 16 + `--no-tune_diffusion_model`, batch 4–8 (slow) | LoRA via openpi recommended |
| Data | LeRobotDataset v3 | LeRobotDataset **v2** + `modality.json` (auto-prepared) | LeRobotDataset v3 |

**π\*0.7** (pi.website/blog/pi07, Apr 2026) is PI's steerable generalist
(language + metadata + control-modality + world-model subgoal prompting,
compositional generalization) but is **closed — no open weights**. Treat it
as a frontier reference; the open PI path is π0/π0.5 via
[openpi](https://github.com/Physical-Intelligence/openpi) (its LoRA finetune
is the better π route on small GPUs; wire up `vla/finetune.py` similarly if
needed).

## Environment setup (one venv per backend — do NOT mix)

The main repo venv (py3.12, isaacsim) is incompatible with all three stacks
(GR00T needs py3.10 + flash-attn; the pip lerobot here is currently broken by
a `huggingface-hub>=1.0` upgrade anyway — see test.md).

```bash
# lerobot env (smolvla + pi0)
uv venv --python 3.12 .venv-vla-lerobot
.venv-vla-lerobot/bin/pip install "huggingface-hub<1.0" torch torchvision
git clone https://github.com/huggingface/lerobot third_party/lerobot
.venv-vla-lerobot/bin/pip install -e "third_party/lerobot[smolvla]"

# GR00T env (py3.10, flash-attn — see Isaac-GR00T README for CUDA specifics)
uv venv --python 3.10 .venv-vla-groot
.venv-vla-groot/bin/pip install -e "third_party/Isaac-GR00T[base]"
.venv-vla-groot/bin/pip install --no-build-isolation flash-attn==2.7.1.post4
git clone https://github.com/NVIDIA/Isaac-GR00T third_party/Isaac-GR00T
```

## Dataset spec (what the recorder must produce)

LeRobotDataset episodes with:

- cameras: `observation.images.wrist` + `observation.images.front` (SO-100
  convention — matches SmolVLA's standardized OBS_IMAGE scheme and GR00T's
  `so100_dualcam` config)
- state: `observation.state` (6 joint angles + gripper, rad)
- action: `action` (next-step joint targets, rad — matches the envs' absolute
  joint-position action space)
- `task` string per episode (the language instruction, e.g. "pick up the cube
  and lift it") — required by all three backends

Sim episodes come from the teleop pipeline (see `scripts/`); real episodes
from `teleop_leader_lerobot.py` / `lerobot-record`.

## Status / roadmap

- [x] `vla/finetune.py` — unified wrapper (smolvla / groot / pi0)
- [x] `vla/gr00t_prepare_dataset.py` — v3→v2 + modality.json + camera remap
- [ ] sim episode recorder (LeRobotDataset writer fed by teleop + env cams)
- [ ] `vla/rollout.py` — closed-loop eval of a finetuned checkpoint in sim
- [ ] first end-to-end run: teleop ~50 pick-lift episodes → smolvla finetune → sim eval
