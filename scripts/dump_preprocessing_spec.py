#!/usr/bin/env python3
"""Dump the authoritative preprocessing spec for a recorded SO-101 dataset.

Reading the prose in docs/PREPROCESSING.md is fine for orientation, but the
values that actually matter at training time are the ones LeRobot wrote into
`meta/info.json` and `meta/stats.json` when it recorded. This script reads
those and prints the exact pipeline, so training and inference can be checked
against the real numbers instead of against documentation.

    python scripts/dump_preprocessing_spec.py
    python scripts/dump_preprocessing_spec.py --root data/thakk100_so101_duck_push
    python scripts/dump_preprocessing_spec.py --json > spec.json
"""

import argparse
import json
from pathlib import Path


def find_dataset(root: Path) -> Path:
    """Accept either the dataset dir itself or a parent containing several."""
    if (root / "meta" / "info.json").is_file():
        return root
    candidates = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "meta" / "info.json").is_file()
    ) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"ERROR: no dataset with meta/info.json under {root}")
    raise SystemExit(
        "ERROR: several datasets found, pick one:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    ds = find_dataset(Path(args.root))
    info = json.load(open(ds / "meta" / "info.json"))
    stats_path = ds / "meta" / "stats.json"
    stats = json.load(open(stats_path)) if stats_path.is_file() else {}

    # --- capture-side spec, as resolved at record time -----------------------
    fps = info.get("fps")
    code_version = info.get("codebase_version")
    feat = info.get("features", {})

    images = {
        k: v for k, v in feat.items() if v.get("dtype") == "video" or "video" in (v.get("info") or {})
    }
    numeric = {
        k: v for k, v in feat.items() if k not in images
    }

    # --- training-side deltas, derived exactly as lerobot does ---------------
    # lerobot/datasets/factory.py:resolve_delta_timestamps
    chunk_size = 100
    action_delta = [i / fps for i in range(chunk_size)] if fps else None

    spec = {
        "dataset_root": str(ds),
        "repo_id": info.get("repo_id"),
        "codebase_version": code_version,
        "fps": fps,
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "video_features": {
            k: {
                "shape_hwc": [v.get("height"), v.get("width"), v.get("channels")],
                "encoding": (v.get("info") or {}).get("video"),
            }
            for k, v in images.items()
        },
        "numeric_features": {
            k: {"dtype": v.get("dtype"), "shape": v.get("shape")} for k, v in numeric.items()
        },
        "train_time": {
            "observation_delta_indices": None,
            "action_delta_indices": list(range(chunk_size)),
            "action_delta_timestamps_s": action_delta,
            "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        },
        "stats_present": sorted(stats.keys()),
    }

    if args.json:
        print(json.dumps(spec, indent=2))
        return 0

    # --- human-readable report ----------------------------------------------
    w = 66
    print("=" * w)
    print("SO-101 dataset preprocessing spec")
    print("=" * w)
    print(f"root            {ds}")
    print(f"repo_id         {spec['repo_id']}")
    print(f"lerobot version {code_version}")
    print(f"fps             {fps}")
    print(f"episodes/frames {spec['total_episodes']} / {spec['total_frames']}")

    print("\n--- CAPTURE (must match at train and deploy time) ---")
    print("  pixel format at capture   MJPG (V4L2 fourcc) - NOT the default YUYV")
    print("  colour conversion         cv2.COLOR_BGR2RGB, uint8, HWC")
    for k, v in spec["video_features"].items():
        enc = v["encoding"] or {}
        h, w_, c = v["shape_hwc"]
        print(f"  {k}")
        print(f"    shape (HxWxC)   {h}x{w_}x{c}")
        print(f"    vcodec          {enc.get('vcodec')}")
        print(f"    pix_fmt         {enc.get('pix_fmt')}   <- 4:2:0, chroma subsampled")
        print(f"    crf / g         {enc.get('crf')} / {enc.get('g')}")
        print(f"    preset          {enc.get('preset')}")

    print("\n--- JOINTS ---")
    for k, v in spec["numeric_features"].items():
        print(f"  {k:34s} {v['dtype']} {v['shape']}")
    print("  body joints (5)  DEGREES")
    print("  gripper           RANGE_0_100")
    print("  order             shoulder_pan, shoulder_lift, elbow_flex,")
    print("                    wrist_flex, wrist_roll, gripper")

    print("\n--- TRAIN (lerobot recomputes this from dataset + policy cfg) ---")
    t = spec["train_time"]
    print(f"  observation_delta_indices  {t['observation_delta_indices']}  (current frame only)")
    print(f"  action_delta_indices       0..{chunk_size - 1}  (chunk of {chunk_size} future actions)")
    if action_delta:
        print(f"  action_delta_timestamps    [{action_delta[0]:.4f} .. {action_delta[-1]:.4f}] s"
              f"  ({chunk_size / fps:.2f}s horizon @ {fps}fps)")
    print(f"  normalization             {t['normalization_mapping']}")

    if stats:
        print("\n--- normalization statistics (meta/stats.json) ---")
        for k in sorted(stats):
            if k.startswith("observation.images"):
                st = stats[k]
                for stat_name in ("mean", "std"):
                    if stat_name in st:
                        vals = st[stat_name]
                        flat = [round(float(x), 4) for x in (vals if isinstance(vals, list) else [vals])]
                        print(f"  {k} {stat_name} {flat}")
            elif "mean" in stats[k]:
                vals = stats[k]["mean"]
                flat = [round(float(x), 2) for x in (vals if isinstance(vals, list) else [vals])]
                print(f"  {k} mean {flat}")
    else:
        print("\n  (no meta/stats.json yet - computed at the end of recording)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
