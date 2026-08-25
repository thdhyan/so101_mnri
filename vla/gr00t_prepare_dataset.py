"""Prepare a LeRobot v3 dataset for GR00T finetuning.

Mirrors NVIDIA's official SO-101 post-training steps
(docs.nvidia.com "Isaac GR00T: Vision-Language-Action Models" / HF blog
"Post-Training Isaac GR00T N1.5 for LeRobot SO-101 Arm"):

1. convert LeRobot v3 -> v2 (GR00T's loader requires v2)
2. copy the SO-100 ``modality.json`` template into the dataset meta/
3. remap the template's camera keys to this dataset's camera names

    python -m vla.gr00t_prepare_dataset --dataset <repo_id-or-path> \
        [--groot-repo third_party/Isaac-GR00T] \
        [--cameras wrist=observation.images.wrist overhead=observation.images.front]

Must run inside the GR00T venv (Python 3.10, see vla/README.md).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

# SO-100 template keys -> our dataset keys (edit via --cameras)
DEFAULT_CAMERA_MAP = {
    "observation.images.front": "observation.images.front",
    "observation.images.wrist": "observation.images.wrist",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GR00T dataset preparation")
    p.add_argument("--dataset", required=True, help="LeRobot repo id or local path")
    p.add_argument("--groot-repo", default="third_party/Isaac-GR00T")
    p.add_argument(
        "--cameras", nargs="*", default=None,
        help="key remaps as template_key=dataset_key (default: SO-100 names pass through)",
    )
    p.add_argument("--conversion-root", default="vla/data/groot_converted")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    groot = Path(args.groot_repo).resolve()
    assert groot.exists(), f"Isaac-GR00T checkout not found at {groot} (see vla/README.md)"

    repo_id = args.dataset
    root = Path(args.conversion_root).resolve()
    out_dir = root / repo_id.replace("/", "_")

    # 1. v3 -> v2 conversion (GR00T's own script, run in its venv)
    convert = [
        "python", str(groot / "scripts" / "lerobot_conversion" / "convert_v3_to_v2.py"),
        "--repo-id", repo_id,
        "--root", str(root),
    ]
    print("+", " ".join(convert))
    subprocess.run(convert, check=True)

    # 2. SO-100 modality template
    modality_src = groot / "examples" / "SO100" / "modality.json"
    modality_dst = out_dir / "meta" / "modality.json"
    shutil.copy(modality_src, modality_dst)

    # 3. remap camera keys
    remaps = dict(DEFAULT_CAMERA_MAP)
    for pair in args.cameras or []:
        template_key, dataset_key = pair.split("=", 1)
        remaps[template_key] = dataset_key
    text = modality_dst.read_text()
    for template_key, dataset_key in remaps.items():
        if template_key != dataset_key:
            text = text.replace(f'"original_key": "{template_key}"',
                                f'"original_key": "{dataset_key}"')
    modality_dst.write_text(text)

    print(f"[ok] GR00T-ready dataset at {out_dir}")
    print(f"     finetune with: python -m vla.finetune --backend groot "
          f"--dataset {out_dir}")


if __name__ == "__main__":
    main()
