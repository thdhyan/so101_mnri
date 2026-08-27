"""Unified IL/RL finetuning for SO-101 — all LeRobot-supported policies.

One entry point, any LeRobot backend via `--policy.type`. Wraps LeRobot's
native `lerobot-train` command with SO-101 presets and remote SSH support.

Supported backends (via LeRobot):
    smolvla, act, diffusion, vqbet, pi0, pi0fast, groot, tdmpc,
    xvla, evo1, molmoact2, wall-oss, multitask_dit

Local training:
    python -m vla.finetune --backend smolvla --dataset thakk100/so101_duck_push
    python -m vla.finetune --backend act --dataset thakk100/so101_duck_push --steps 5000
    python -m vla.finetune --backend diffusion --dataset thakk100/so101_duck_push
    python -m vla.finetune --backend groot --lora --dataset thakk100/so101_duck_push
    python -m vla.finetune --backend pi0 --dataset thakk100/so101_duck_push

Remote SSH training:
    python -m vla.finetune --backend groot --dataset thakk100/so101_duck_push \
        --remote-host dl --steps 20000

    python -m vla.finetune --backend act --dataset thakk100/so101_duck_push \
        --remote-host lambda --batch-size 32
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass


# ── Backend registry ──────────────────────────────────────────────────────────

@dataclass
class Backend:
    """Metadata for a LeRobot-supported policy."""
    name: str
    policy_type: str          # maps to --policy.type=<value>
    base_model: str | None    # HuggingFace base model path (None = use default)
    default_steps: int
    default_batch: int
    default_lr: float | None  # None = use LeRobot default
    needs_groot_prepare: bool = False  # GR00T needs v2 conversion + modality.json


# All backends supported by LeRobot (as of Jul 2026)
BACKENDS: dict[str, Backend] = {
    # ── Imitation Learning ──
    "act": Backend(
        name="act",
        policy_type="act",
        base_model=None,
        default_steps=5000,
        default_batch=16,
        default_lr=None,
    ),
    "diffusion": Backend(
        name="diffusion",
        policy_type="diffusion",
        base_model=None,
        default_steps=10000,
        default_batch=8,
        default_lr=None,
    ),
    "vqbet": Backend(
        name="vqbet",
        policy_type="vqbet",
        base_model=None,
        default_steps=10000,
        default_batch=16,
        default_lr=None,
    ),
    # ── VLAs ──
    "smolvla": Backend(
        name="smolvla",
        policy_type="smolvla",
        base_model="lerobot/smolvla_base",
        default_steps=10000,
        default_batch=16,
        default_lr=None,
    ),
    "pi0": Backend(
        name="pi0",
        policy_type="pi0",
        base_model="lerobot/pi0",
        default_steps=15000,
        default_batch=4,
        default_lr=None,
    ),
    "pi0fast": Backend(
        name="pi0fast",
        policy_type="pi0fast",
        base_model="lerobot/pi0fast",
        default_steps=15000,
        default_batch=4,
        default_lr=None,
    ),
    "groot": Backend(
        name="groot",
        policy_type="groot",
        base_model="nvidia/GR00T-N1.7-3B",
        default_steps=20000,
        default_batch=64,
        default_lr=1e-4,
        needs_groot_prepare=True,
    ),
    "xvla": Backend(
        name="xvla",
        policy_type="xvla",
        base_model=None,
        default_steps=20000,
        default_batch=8,
        default_lr=None,
    ),
    "evo1": Backend(
        name="evo1",
        policy_type="evo1",
        base_model=None,
        default_steps=15000,
        default_batch=4,
        default_lr=None,
    ),
    # ── RL ──
    "tdmpc": Backend(
        name="tdmpc",
        policy_type="tdmpc",
        base_model=None,
        default_steps=50000,
        default_batch=256,
        default_lr=None,
    ),
}


# ── Command builder ───────────────────────────────────────────────────────────

def build_lerobot_train_cmd(backend: Backend, args) -> list[str]:
    """Build the `lerobot-train` command with all arguments."""
    cmd = ["uv", "run", "lerobot-train"]

    # Policy
    cmd += [f"--policy.type={backend.policy_type}"]
    if backend.base_model:
        cmd += [f"--policy.base_model_path={backend.base_model}"]
    cmd += [f"--policy.device={args.device}"]

    # GR00T-specific options
    if backend.name == "groot":
        cmd += [
            "--policy.embodiment_tag=new_embodiment",
            "--policy.chunk_size=16",
            "--policy.n_action_steps=16",
            "--policy.use_relative_actions=true",
            '--policy.relative_exclude_joints=["gripper"]',
        ]
        if args.lora:
            cmd += ["--policy.lora_rank=16", "--policy.lora_alpha=32"]
        else:
            # Full finetune (default for GR00T on big GPUs)
            pass
    elif args.lora:
        cmd += ["--policy.lora_rank=16", "--policy.lora_alpha=32"]

    # Dataset
    cmd += [
        f"--dataset.repo_id={args.dataset}",
        "--dataset.image_transforms.enable=true",
    ]

    # Training
    steps = args.steps or backend.default_steps
    batch = args.batch_size or backend.default_batch
    cmd += [
        f"--steps={steps}",
        f"--batch_size={batch}",
        "--save_checkpoint=true",
        "--save_freq=5000",
        "--use_policy_training_preset=true",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--log_freq=10",
        f"--output_dir={args.output_dir}/{backend.name}",
        f"--job_name={args.dataset.split('/')[-1]}_{backend.name}",
    ]

    # WandB
    if args.wandb:
        cmd += [
            "--wandb.enable=true",
            f"--wandb.project={args.wandb_project}",
            "--wandb.disable_artifact=true",
        ]
    else:
        cmd += ["--wandb.enable=false"]

    # Seed
    cmd += [f"--seed={args.seed}"]

    return cmd


# ── Local training ────────────────────────────────────────────────────────────

def run_local(args):
    """Run training locally."""
    backend = BACKENDS[args.backend]

    # GR00T dataset preparation
    if backend.needs_groot_prepare and not args.skip_prepare:
        prep_cmd = [
            sys.executable, "-m", "vla.gr00t_prepare_dataset",
            "--dataset", args.dataset,
        ]
        print("+", shlex.join(prep_cmd))
        if not args.dry_run:
            subprocess.run(prep_cmd, check=True)

    cmd = build_lerobot_train_cmd(backend, args)
    print("+", shlex.join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(cmd).returncode)


# ── Remote SSH training ───────────────────────────────────────────────────────

def run_remote(args):
    """Train on a remote GPU machine via SSH."""
    backend = BACKENDS[args.backend]
    host = args.remote_host
    remote_dir = args.remote_dir

    # Build the remote command
    train_cmd = build_lerobot_train_cmd(backend, args)
    # Replace 'uv run' with direct python (remote may not have uv)
    train_cmd_str = " ".join(shlex.quote(c) for c in train_cmd)
    train_cmd_str = train_cmd_str.replace("uv run ", "")

    # Full remote command: cd to project dir and run
    remote_full = f"cd {remote_dir} && {train_cmd_str}"

    print(f"\n{'='*60}")
    print(f"  Remote Training via SSH")
    print(f"  Host:      {host}")
    print(f"  Remote dir: {remote_dir}")
    print(f"  Policy:    {args.backend}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Steps:     {args.steps or backend.default_steps}")
    print(f"{'='*60}")
    print(f"\n  Full command:\n  {remote_full}\n")

    if args.dry_run:
        return

    # Sync code first
    print("→ Syncing code to remote ...")
    rsync_cmd = [
        "rsync", "-avz", "--delete",
        "--exclude", ".venv*",
        "--exclude", "__pycache__",
        "--exclude", ".git",
        "--exclude", "vla/runs",
        "--exclude", "data",
        "--exclude", "docker",
        "./", f"{host}:{remote_dir}/",
    ]
    subprocess.run(rsync_cmd, check=True)

    # Sync dataset if it exists locally
    import os
    local_dataset = os.path.join("data", args.dataset.replace("/", "_"))
    if os.path.exists(local_dataset):
        print(f"→ Syncing dataset to remote ...")
        ds_rsync = [
            "rsync", "-avz",
            f"{local_dataset}/",
            f"{host}:{remote_dir}/data/{args.dataset.replace('/', '_')}/",
        ]
        subprocess.run(ds_rsync, check=True)

    # Run training on remote
    print("→ Starting training on remote ...")
    ssh_cmd = ["ssh", "-t", host, remote_full]
    raise SystemExit(subprocess.run(ssh_cmd).returncode)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Backend
    p.add_argument("--backend", choices=list(BACKENDS), required=True,
                   help="Policy backend (any LeRobot-supported policy)")

    # Dataset
    p.add_argument("--dataset", required=True,
                   help="LeRobotDataset repo id (HF) or local path")

    # Training
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lora", action="store_true",
                   help="LoRA finetune (required for small GPUs)")
    p.add_argument("--device", default="cuda")

    # Remote
    p.add_argument("--remote-host", default=None,
                   help="SSH host for remote training (dl, lambda, zz-bw, zz-dt)")
    p.add_argument("--remote-dir", default="~/projects/so101_mnri",
                   help="Project directory on remote host")

    # Output
    p.add_argument("--output-dir", default="vla/runs")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="so101-il")
    p.add_argument("--seed", type=int, default=42)

    # GR00T
    p.add_argument("--skip-prepare", action="store_true",
                   help="Skip GR00T v2/modality.json dataset preparation")

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running")
    p.add_argument("--list-backends", action="store_true",
                   help="List all supported backends and exit")

    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.list_backends:
        print(f"{'Backend':<12} {'Policy Type':<16} {'Base Model':<35} {'Default Steps':<14}")
        print("-" * 80)
        for name, b in BACKENDS.items():
            base = b.base_model or "(LeRobot default)"
            print(f"{name:<12} {b.policy_type:<16} {base:<35} {b.default_steps:<14}")
        return

    if args.remote_host:
        run_remote(args)
    else:
        run_local(args)


if __name__ == "__main__":
    main()
