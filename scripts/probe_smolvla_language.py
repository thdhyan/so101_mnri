#!/usr/bin/env python3
"""Offline language probe for pretrained SmolVLA: does the instruction change
what it would do? No robot, nothing moves.

For frames of one recorded episode, ask `lerobot/smolvla_base` (or any SmolVLA
checkpoint) for an action chunk under several instructions, holding the
flow-matching noise FIXED across instructions, so any difference comes from
the language alone. Compare that with the spread from noise alone (same
instruction, different noise seeds):

    language effect  = mean |chunk(instr_i) - chunk(instr_j)|     (fixed noise)
    noise floor      = mean |chunk(instr, seed a) - chunk(instr, seed b)|
    ratio            = language effect / noise floor

ratio ~1  -> the words change the output no more than random noise does:
             the model is ignoring language on this data.
ratio >>1 -> the instruction measurably steers the predicted motion.

Caveats, read before trusting numbers:
  * smolvla_base ships no usable normalisation stats (its normaliser has no
    observation.state entry and legacy-named action keys), so this uses the
    DATASET's own stats -- the same thing finetuning does.
  * Its pretraining SO-100 data used an older joint calibration (different
    zero/sign per joint), so absolute actions are not meaningful on this arm.
    The instruction-vs-instruction COMPARISON is the signal; the error against
    the recorded teleop action is reported for context only.
  * The first action of a chunk sits close to the CURRENT joint state, so the
    predicted curve tracking the recorded one mostly shows it reads
    proprioception, not that it understands the task.

    .venv/bin/python scripts/probe_smolvla_language.py \\
        --root data/thakk100_pink_duck_in_a_box --episode 0
    # any LeRobot v3 dataset; map its cameras onto camera1/camera2/camera3:
    .venv/bin/python scripts/probe_smolvla_language.py \\
        --repo-id lerobot/svla_so101_pickplace --cameras up,side
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_INSTRUCTIONS = [
    "{task}",                                # the dataset's own instruction
    "put the pink duck in the box",
    "push the object to the left",
    "lift the arm straight up",
    "do nothing and stay still",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default="thakk100/pink_duck_in_a_box")
    ap.add_argument("--root", default=None, help="local dataset dir (default: HF cache / Hub)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cameras", default="global,wrist",
                    help="dataset camera names, in order, mapped to camera1,camera2,camera3")
    ap.add_argument("--policy", default="lerobot/smolvla_base")
    ap.add_argument("--instructions", nargs="+", default=DEFAULT_INSTRUCTIONS,
                    help="'{task}' is replaced by the dataset's own task string")
    ap.add_argument("--frames", type=int, default=12, help="evenly spaced frames to probe")
    ap.add_argument("--seeds", type=int, default=2, help="noise seeds for the noise floor (>=2)")
    ap.add_argument("--out", default="data/probe_smolvla", help="output directory (gitignored)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    missing = [c for c in cams if f"observation.images.{c}" not in ds.meta.features]
    if missing:
        have = [k.split(".")[-1] for k in ds.meta.features if k.startswith("observation.images.")]
        raise SystemExit(f"cameras {missing} not in dataset; it has {have} (use --cameras)")

    policy = SmolVLAPolicy.from_pretrained(args.policy).to(args.device).eval()
    rename = {f"observation.images.{c}": f"observation.images.camera{i + 1}"
              for i, c in enumerate(cams)}
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": rename},
            "normalizer_processor": {"stats": ds.meta.stats},
        },
        postprocessor_overrides={"unnormalizer_processor": {"stats": ds.meta.stats}},
    )

    ep = ds.meta.episodes[args.episode]
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    # With episodes=[k] the loaded dataset is re-indexed from 0.
    n = hi - lo
    frame_ids = np.linspace(0, n - 1, min(args.frames, n)).astype(int)
    task = ds[0]["task"]
    instrs = [s.replace("{task}", task) for s in args.instructions]
    chunk, adim = policy.config.chunk_size, policy.config.max_action_dim

    def predict(sample: dict, instruction: str, seed: int) -> np.ndarray:
        batch = {k: v for k, v in sample.items() if k.startswith("observation.")}
        batch["task"] = instruction
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, chunk, adim), generator=g).to(args.device)
        policy.reset()
        out = policy.predict_action_chunk(pre(batch), noise=noise)
        return post(out).squeeze(0).cpu().numpy()  # (chunk, action_dim), dataset units

    # preds[instr][frame] -> (chunk, 6); noise[seed][frame] for instruction 0
    preds = {s: [] for s in instrs}
    noise_runs = {k: [] for k in range(args.seeds)}
    recorded = []
    for f in frame_ids:
        sample = ds[int(f)]
        recorded.append(sample["action"].numpy())
        for s in instrs:
            preds[s].append(predict(sample, s, seed=0))
        for k in range(args.seeds):
            noise_runs[k].append(preds[instrs[0]][-1] if k == 0 else predict(sample, instrs[0], seed=k))
        print(f"  frame {f:4d}/{n - 1} done", flush=True)

    P = {s: np.stack(v) for s, v in preds.items()}          # (F, chunk, 6)
    N = [np.stack(v) for v in noise_runs.values()]
    lang = {f"{a} | {b}": float(np.abs(P[a] - P[b]).mean())
            for a, b in itertools.combinations(instrs, 2)}
    floor = float(np.mean([np.abs(x - y).mean() for x, y in itertools.combinations(N, 2)]))
    lang_mean = float(np.mean(list(lang.values())))
    rec = np.stack(recorded)                                  # (F, 6)
    err_first = {s: float(np.abs(P[s][:, 0] - rec).mean()) for s in instrs}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy": args.policy, "dataset": args.repo_id, "episode": args.episode,
        "cameras": rename, "frames": frame_ids.tolist(), "instructions": instrs,
        "language_effect_mean": lang_mean, "noise_floor": floor,
        "ratio": lang_mean / floor if floor > 0 else float("inf"),
        "language_effect_pairs": lang,
        "first_action_abs_err_vs_recorded": err_first,
        "units": "dataset action units (degrees for this repo's arms)",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(out / "actions.png", frame_ids, rec, P, ds.meta.features["action"]["names"], task)

    print(f"\nlanguage effect  {lang_mean:7.2f}  (mean |diff| between instructions, fixed noise)")
    print(f"noise floor      {floor:7.2f}  (same instruction, different noise)")
    print(f"ratio            {summary['ratio']:7.2f}  (~1 = ignores language, >>1 = steered by it)")
    print("\nper instruction pair:")
    for k, v in sorted(lang.items(), key=lambda kv: -kv[1]):
        print(f"  {v:7.2f}  {k}")
    print(f"\nwrote {out / 'summary.json'} and {out / 'actions.png'}")
    return 0


def _plot(path: Path, frames, rec, P: dict, names, task: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(frames, rec[:, j], "k-", lw=2, label="recorded (teleop)")
        for s, arr in P.items():
            ax.plot(frames, arr[:, 0, j], marker=".", lw=1, label=s[:40])
        ax.set_title(names[j] if names else f"joint {j}")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"SmolVLA first predicted action per instruction  |  dataset task: {task!r}")
    fig.supxlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
