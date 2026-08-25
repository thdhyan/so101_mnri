"""Unified VLA finetuning for the SO-101 tasks.

One entry point, three swappable backends — SmolVLA, GR00T (N1.5/N1.6), and
pi0/pi0.5 — each in its own dedicated virtualenv (their dependency stacks are
mutually incompatible; see vla/README.md). This wrapper only BUILDS the
backend's training command with SO-101 presets; it never imports the backend.

    python -m vla.finetune --backend smolvla --dataset thakk100/so101_pick_lift \
        --steps 20000 --wandb
    python -m vla.finetune --backend groot  --dataset ./data/so101_pick_lift \
        --steps 20000 --lora
    python -m vla.finetune --backend pi0    --dataset thakk100/so101_pick_lift \
        --steps 15000 --lora

Datasets are LeRobotDataset repos (v3), recorded from sim teleop or the real
arm. GR00T additionally needs a v2 conversion + modality.json — run
`python -m vla.gr00t_prepare_dataset` first (done automatically by this
wrapper unless --skip-prepare).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass


@dataclass
class Backend:
    name: str
    venv: str  # python/entry-point prefix; see vla/README.md for env setup
    base_model: str
    default_steps: int
    default_batch: int  # tuned for an 8 GB laptop GPU; raise on bigger cards


BACKENDS = {
    "smolvla": Backend(
        name="smolvla",
        venv=".venv-vla-lerobot",
        base_model="lerobot/smolvla_base",
        default_steps=20000,
        default_batch=16,
    ),
    # pi0 via lerobot's integrated policy (openpi weights: lerobot/pi0).
    # For pi0.5 / openpi-native training see vla/README.md "openpi" section.
    "pi0": Backend(
        name="pi0",
        venv=".venv-vla-lerobot",
        base_model="lerobot/pi0",
        default_steps=15000,
        default_batch=4,
    ),
    # GR00T uses NVIDIA's standalone repo + its own py3.10 env (flash-attn).
    "groot": Backend(
        name="groot",
        venv=".venv-vla-groot",
        base_model="nvidia/GR00T-N1.6-3B",
        default_steps=20000,
        default_batch=8,
    ),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SO-101 unified VLA finetuner")
    p.add_argument("--backend", choices=list(BACKENDS), required=True)
    p.add_argument("--dataset", required=True,
                   help="LeRobotDataset repo id (HF) or local path")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lora", action="store_true",
                   help="LoRA finetune (required on <=8 GB GPUs for groot/pi0)")
    p.add_argument("--no-wandb", dest="wandb", action="store_false", default=True)
    p.add_argument("--wandb-project", default="so101-vla")
    p.add_argument("--output-dir", default="vla/runs")
    p.add_argument("--task-instruction", default="pick up the cube",
                   help="language instruction stored with the run")
    p.add_argument("--groot-repo", default="third_party/Isaac-GR00T",
                   help="path to a cloned NVIDIA/Isaac-GR00T checkout (groot backend)")
    p.add_argument("--groot-data-config", default="so100_dualcam",
                   help="GR00T data config (so100_dualcam matches wrist+overhead cams)")
    p.add_argument("--skip-prepare", action="store_true",
                   help="skip the GR00T v2/modality.json dataset preparation")
    p.add_argument("--dry-run", action="store_true", help="print the command, don't run")
    return p.parse_args(argv)


def lerobot_cmd(backend: Backend, args) -> list[str]:
    """lerobot-train command (smolvla / pi0 backends)."""
    # Prefer the installed console script; fall back to the module form.
    script = f"{backend.venv}/bin/lerobot-train"
    cmd = [script] if _exists(script) else [f"{backend.venv}/bin/python", "-m", "lerobot.train"]

    cmd += [
        f"--policy.path={backend.base_model}",
        f"--dataset.repo_id={args.dataset}",
        f"--batch_size={args.batch_size or backend.default_batch}",
        f"--steps={args.steps or backend.default_steps}",
        f"--output_dir={args.output_dir}/{backend.name}",
        f"--job_name=so101_{backend.name}",
        "--policy.device=cuda",
    ]
    if args.wandb:
        cmd += ["--wandb.enable=true", f"--wandb.project={args.wandb_project}"]
    return cmd


def groot_cmd(backend: Backend, args) -> list[str]:
    """NVIDIA Isaac-GR00T standalone finetune (official SO-101 path)."""
    py = f"{backend.venv}/bin/python"
    cmd = [
        py, f"{args.groot_repo}/scripts/gr00t_finetune.py",
        f"--base-model-path={backend.base_model}",
        f"--dataset-path={args.dataset}",
        f"--embodiment-tag=NEW_EMBODIMENT",  # SO-101 is not in GR00T pretraining
        f"--data-config={args.groot_data_config}",
        f"--output-dir={args.output_dir}/{backend.name}",
        f"--max-steps={args.steps or backend.default_steps}",
        f"--batch-size={args.batch_size or backend.default_batch}",
        "--video-backend=torchvision_av",
        # 8 GB GPUs: freeze the DiT action head + flash-attn offload
        "--no-tune_diffusion_model",
    ]
    if args.lora:
        cmd += ["--lora_rank=16", "--lora_alpha=32"]
    return cmd


def _exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()


def main(argv=None):
    args = parse_args(argv)
    backend = BACKENDS[args.backend]

    if args.backend == "groot" and not args.skip_prepare:
        prep = [
            f"{backend.venv}/bin/python", "-m", "vla.gr00t_prepare_dataset",
            "--dataset", args.dataset,
        ]
        print("+", shlex.join(prep))
        if not args.dry_run:
            subprocess.run(prep, check=True)

    cmd = groot_cmd(backend, args) if args.backend == "groot" else lerobot_cmd(backend, args)
    print("+", shlex.join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
