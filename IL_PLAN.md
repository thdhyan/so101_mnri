# Imitation Learning Plan: Duck-Push-to-Square Task

## Task Definition

**Goal**: Use a single SO-101 follower arm to push a small rubber duck into a square taped area on the table.

**Approach**: Imitation Learning (Behavior Cloning) from human teleoperation demonstrations.

**Why IL over RL?**
- You already have the leader arm hardware — data collection is fast
- Pushing a duck is a simple, low-dimensional task
- 300-400 demonstrations is plenty for behavior cloning on a 6-DOF arm
- No reward engineering needed — the policy learns directly from your demonstrations
- Can record videos immediately after training (no 6-hour RL wallclock wait)

---

## Hardware Setup

```
┌─────────────────────────────────────────────────────────────┐
│                        TABLE                                 │
│                                                             │
│   ┌─────────┐                                               │
│   │  DUCK   │        ┌───────────┐                          │
│   └─────────┘        │  SQUARE   │                          │
│        ↑             │   TAPE    │                          │
│        │             └───────────┘                          │
│     SO-101                                                   │
│     FOLLOWER                                                 │
│        ┌───┐                                                 │
│        │   │◄── wrist camera (on follower gripper)          │
│        └───┘                                                 │
│                                                             │
│   GLOBAL CAMERA (position-invariant — any tripod spot)      │
└─────────────────────────────────────────────────────────────┘
```

**Sensors used per timestep**:
- `observation.images.wrist` — 640×480 from follower's wrist camera
- `observation.images.global` — 640×480 from global camera (any position)
- `observation.state` — 6 joint angles of follower (radians)
- `action` — 6 target joint positions (radians)

---

## Why Camera-Position Independent?

The IL policy sees the global camera as just another image stream. You can:

1. **Randomize global camera position during data collection** — record 100 demos from position A, 100 from position B, etc. The wrist camera provides a fixed, ego-centric view that anchors the policy regardless of where the global camera is.

2. **Crop augmentation** — during training, randomly crop and resize the global camera image. This simulates different camera positions and forces the policy to learn from local features (duck shape, tape corners) rather than memorizing pixel coordinates.

3. **The wrist camera is already position-invariant** — it moves with the gripper, so it always sees the same relative view. This is your primary visual input; the global camera is supplementary context.

**Recommendation**: Position the global camera somewhere overhead (60-90° from table) for the best view. Don't worry about the exact spot — augmentation handles the rest.

---

## Data Collection

### Current Teleop Pipeline

You already have `scripts/teleop_leader_lerobot.py` which connects a real SO-101 leader arm to the MuJoCo sim follower. For the real-world IL task, use the same pipeline but record to LeRobotDataset format.

### New Script: `vla/record_demos.py`

Record real-world teleop demonstrations directly to a LeRobotDataset:

```bash
# Single leader arm -> follower + record
python -m vla.record_demos \
    --leader-port /dev/ttyACM0 \
    --follower-port /dev/ttyACM1 \
    --global-cam 0 \
    --dataset thakk100/so101_duck_push \
    --task "push the duck into the square tape" \
    --episodes 400 \
    --hz 30
```

**Each episode**:
1. Start: reset duck to random position in front of the arm
2. Teleop: push duck into the square using leader arm
3. Stop: press button to end recording
4. Auto-save: episode saved to LeRobotDataset

### Data Volume Estimate

| Episodes | Steps/ep (30Hz, 10s avg) | Total frames | Disk (~1MB/frame) |
|---|---|---|---|
| 200 | 300 | 60K | ~60 GB |
| 400 | 300 | 120K | ~120 GB |

**With augmentation, 200 raw episodes → ~800 augmented episodes** is enough.

---

## Data Augmentation

### Visual Augmentations (on global camera images)

1. **Random crop + resize** (crop 80% of image, resize back) — simulates camera position variation
2. **Color jitter** (brightness ±20%, contrast ±20%, saturation ±15%) — simulates lighting changes
3. **Gaussian blur** (kernel 3-5, p=0.2) — simulates out-of-focus global camera
4. **Random horizontal flip** (p=0.5, only if symmetric task) — duck-in-square is symmetric left-right if square is centered
5. **Random rotation** (±15°) — simulates camera tilt variation

### State Augmentations (on joint angles)

6. **Gaussian noise** on joint angles (σ=0.02 rad) — simulates joint backlash
7. **Time warping** — slightly speed up/slow down trajectories (×0.9 to ×1.1) — makes policy robust to timing

### Implementation

```python
# In the dataset or DataLoader
from torchvision import transforms

global_cam_augment = transforms.Compose([
    transforms.RandomResizedCrop(480, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomRotation(degrees=15),
])
```

**Wrist camera augmentations** — lighter touch (it's already ego-centric):
- Color jitter only (no crop/flip — the wrist view geometry matters)

---

## Model Selection: What to Use?

### Option A: SmolVLA (450M) — **RECOMMENDED for quick iteration**

| | |
|---|---|
| **Why** | Smallest VLA, trains on 8GB GPU in ~30 min, already has SO-101 pretrained weights |
| **Training** | `python -m vla.finetune --backend smolvla --dataset thakk100/so101_duck_push --steps 10000` |
| **Inference** | `python -m vla.rollout --backend smolvla --checkpoint <path>` |
| **Hardware** | Your laptop GPU (8GB) or `dl` server |
| **Pros** | Fast iteration, language-conditioned ("push the duck into the square"), wrist+global camera natively supported |
| **Cons** | 450M params — may struggle with very precise fine control; needs LeRobot v3 dataset |

### Option B: GR00T N1.6 (3B) — **Best accuracy, needs more VRAM**

| | |
|---|---|
| **Why** | NVIDIA's official SO-101 model, best sim-to-real transfer, Flow Matching action head |
| **Training** | `python -m vla.finetune --backend groot --dataset thakk100/so101_duck_push --lora` |
| **Hardware** | `dl` server (4× RTX 6000 Ada, 192GB VRAM) or spark with 128GB unified |
| **Pros** | Best accuracy, official NVIDIA support, LoRA keeps VRAM low |
| **Cons** | Needs py3.10 venv + flash-attn, v2 dataset conversion, 30-60 min training |

### Option C: ACT (Action Chunking with Transformers) — **Simplest, no VLM overhead**

| | |
|---|---|
| **Why** | Proven for SO-101, lightweight, no language conditioning needed for single-task |
| **Training** | Use `lerobot-train` with ACT policy config |
| **Hardware** | Your laptop GPU |
| **Pros** | Simple, fast, action chunking handles temporal consistency |
| **Cons** | Not language-conditioned (single task only), no pretrained weights to leverage |

### Recommendation: Start with SmolVLA

**Why SmolVLA first?**
1. You can train on the laptop — no need to SSH to `dl`
2. 10-15 min training time — iterate fast
3. Already understands wrist+global camera (LeRobot convention)
4. If quality is good enough, you're done. If not, graduate to GR00T.

---

## End-to-End Pipeline

### Step 1: Create the Duck-Push Environment (if needed for sim)
```bash
# Option A: Use existing push env with modified object
# Modify envs/mujoco/so101_single_arm_push_t/ to have a duck instead of T-block

# Option B: Record directly in real world (skip sim)
# Just use the leader arm + follower + cameras
```

### Step 2: Record Demonstrations
```bash
# Calibrate leader arm
python scripts/teleop_leader_arm.py --calibrate --port /dev/ttyACM0

# Record 200-400 episodes to LeRobotDataset
python -m vla.record_demos \
    --leader-port /dev/ttyACM0 \
    --follower-port /dev/ttyACM1 \
    --global-cam 0 \
    --dataset thakk100/so101_duck_push \
    --task "push the duck into the square tape" \
    --episodes 400
```

### Step 3: Train the Policy
```bash
# SmolVLA on laptop (~15 min)
source .venv-vla-lerobot/bin/activate
python -m vla.finetune \
    --backend smolvla \
    --dataset thakk100/so101_duck_push \
    --steps 10000 \
    --batch-size 8 \
    --wandb

# OR GR00T on dl (~30 min)
ssh dl
source .venv-vla-groot/bin/activate
python -m vla.finetune \
    --backend groot \
    --dataset thakk100/so101_duck_push \
    --lora \
    --steps 15000
```

### Step 4: Evaluate in Sim
```bash
python -m vla.rollout \
    --backend smolvla \
    --checkpoint vla/runs/smolvla/last \
    --env single \
    --render
```

### Step 5: Deploy on Real Robot
```bash
python -m vla.deploy \
    --backend smolvla \
    --checkpoint vla/runs/smolvla/last \
    --follower-port /dev/ttyACM1 \
    --global-cam 0
```

### Step 6: Record Success Videos
```bash
python -m vla.record_video \
    --backend smolvla \
    --checkpoint vla/runs/smolvla/last \
    --follower-port /dev/ttyACM1 \
    --global-cam 0 \
    --episodes 10 \
    --output videos/duck_push_success.mp4
```

---

## Quick Win Strategy (24 hours to first video)

| Hour | Task | Output |
|---|---|---|
| 0-1 | Set up cameras, tape square, place duck | Physical setup ready |
| 1-3 | Record 100 demos (fast: 2 min/episode) | LeRobotDataset |
| 3-4 | Train SmolVLA (15 min on laptop) | Trained checkpoint |
| 4-5 | Record 10 success videos | Demo video |
| 5+ | If good: collect 300 more demos. If bad: augment + retrain | Improved policy |

**100 demos × 2 min/episode = ~3.5 hours** (including resets and bad takes).

---

## Cosmos / GR00T / Small IL — Decision Tree

```
Do you need language conditioning? (e.g., "push duck" vs "push cube")
├── YES → SmolVLA (quick) or GR00T (best)
└── NO (single task, just push)
    ├── Want fastest training? → ACT
    ├── Want best accuracy? → GR00T
    └── Want easiest setup? → SmolVLA

Do you want to use synthetic data augmentation (Cosmos)?
├── YES → Use Cosmos to generate 1000s of synthetic duck-push frames
│         then pretrain GR00T on synthetic + finetune on your 200 real demos
└── NO  → Just use real demos + standard augmentations (crop, color, blur)
```

**My recommendation**: Skip Cosmos for now. 200 real demos + SmolVLA + standard augmentations will get you a working policy fast. Cosmos is useful if you need to generalize to many duck positions or table configurations — that's a Day 2 problem.

---

## Files to Create

| File | Purpose |
|---|---|
| `vla/record_demos.py` | Record real-world teleop to LeRobotDataset |
| `vla/rollout.py` | Closed-loop eval of trained checkpoint |
| `vla/deploy.py` | Deploy policy to real robot |
| `vla/record_video.py` | Record success/failure videos |
| `vla/augment.py` | Dataset augmentation transforms |
| `envs/mujoco/so101_duck_push/` | Sim env with duck + square tape (optional) |

---

## Status

- [ ] Create `vla/record_demos.py` — real-world teleop recorder
- [ ] Create `vla/augment.py` — augmentation transforms
- [ ] Create `vla/rollout.py` — sim evaluation loop
- [ ] Create `vla/deploy.py` — real robot deployment
- [ ] Create duck-push sim env (optional, for pre-training)
- [ ] Record 100 initial demos
- [ ] Train SmolVLA baseline
- [ ] Record success videos
- [ ] Iterate: augment data, retrain, improve
