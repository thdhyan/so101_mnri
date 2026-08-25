# rl/ — backend-agnostic RL training

skrl and rsl_rl are the algorithm layer; Isaac Lab, MuJoCo, and mjlab are
interchangeable simulation backends. Shared PPO hyperparameters live in
`agents/` so both libraries train comparable policies on the same tasks.

## Train

```bash
# Isaac Lab (Isaac Sim 6.0.1) — GPU required
python -m rl.train --backend isaaclab --task SO101-PickLift-Single-v0  --algo skrl   --num-envs 4096
python -m rl.train --backend isaaclab --task SO101-PickPlace-Single-v0 --algo rsl_rl --num-envs 4096
python -m rl.train --backend isaaclab --task SO101-CylGrasp-Dual-v0    --algo skrl   --num-envs 2048
python -m rl.train --backend isaaclab --task SO101-CylReach-Dual-v0    --algo rsl_rl --num-envs 2048

# mjlab (MuJoCo Warp GPU batching) — pick-lift only (older plan, MJLAB_INTEGRATION.md)
python -m rl.train --backend mjlab --task so101_pick_lift --algo skrl --num-envs 256

# plain MuJoCo gymnasium (single env, CPU-friendly)
python -m rl.train --backend mujoco --task pick_lift --algo skrl --max-iterations 200 --device cpu
```

Flags: `--num-envs`, `--max-iterations`, `--seed`, `--device`,
`--wandb/--no-wandb` (project `so101-rl`), `--log-dir` (default `rl/runs`).

## Play (checkpoint playback, skrl agents)

```bash
python -m rl.play --backend mujoco  --task pick_lift --checkpoint rl/runs/mujoco/.../agent.pt
python -m rl.play --backend isaaclab --task SO101-PickLift-Single-v0 --checkpoint rl/runs/isaaclab/.../agent.pt
```

## Shared PPO hyperparameters (rl/agents/)

actor/critic MLP `[256, 128, 64]` ELU · rollouts 24 · epochs 5 · minibatches
4 · lr 1e-3 adaptive (KL target 0.01) · γ 0.99 · λ 0.95 · clip 0.2 ·
entropy 0 · grad-norm clip 1.0 · checkpoints every 100 iterations.

## Notes

- rsl_rl requires a vectorized env (isaaclab / mjlab backends only).
- Isaac training strips camera sensors from the env cfg (state-based RL);
  cameras remain available for image-obs work and screenshots
  (`envs/isaac/scripts/render_cameras.py`).
- The skrl fork (third_party/skrl) uses the new dataclass API — see test.md
  before writing custom models.
- Logs: TensorBoard under `rl/runs/<backend>/<task>_<algo>/<stamp>/`;
  WandB when `--wandb` (default on).
