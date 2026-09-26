#!/usr/bin/env python3
"""Tune SO-101 collection cameras (V4L2) and verify the result with a saved frame.

Both collection cameras (Innomaker U20CAM wrist, generic USB global) ship with
`auto_exposure=3` (Aperture Priority). In a bright room that blows the image out
to flat white, which is useless as training data. This script switches each
camera to Manual Mode (`auto_exposure=1`), sweeps `exposure_time_absolute`,
saves one frame per step, and reports the brightness so you can pick a value.

    # see what each camera currently is
    python scripts/tune_collection_cameras.py --list

    # sweep exposure on the global cam (default 4) and the wrist cam (default 2)
    python scripts/tune_collection_cameras.py --device 4 --device 2 --sweep

    # lock in a known-good setting
    python scripts/tune_collection_cameras.py --device 4 --exposure 400 --apply

    # also lock white balance and kill the 50Hz flicker
    python scripts/tune_collection_cameras.py --device 4 --exposure 400 \
        --apply --lock-whitebalance --powerline 1
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def v4l2(device: str, *args: str) -> tuple[int, str]:
    """Run v4l2-ctl against a device. Returns (rc, combined output)."""
    if not shutil.which("v4l2-ctl"):
        print("ERROR: v4l2-ctl not installed (apt install v4l-utils)", file=sys.stderr)
        raise SystemExit(2)
    proc = subprocess.run(
        ["v4l2-ctl", "-d", device, *args], capture_output=True, text=True, check=False
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def show(device: str) -> None:
    rc, _ = v4l2(device, "--list-ctrls")
    if rc != 0:
        print(f"  {device}: <cannot open>")
        return
    _, out = v4l2(device, "--get-ctrl=auto_exposure,exposure_time_absolute")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    for ln in lines:
        print(f"  {device}: {ln.strip()}")


def apply(device: str, exposure: int | None, lock_wb: bool, powerline: int | None,
          brightness: int | None, gamma: int | None) -> None:
    """Turn off auto-exposure first, then set the manual values it gates."""
    # auto_exposure is a menu: 1 = Manual Mode, 3 = Aperture Priority (auto).
    rc, out = v4l2(device, "--set-ctrl=auto_exposure=1")
    print(f"  {device}: auto_exposure=1 (manual) -> {out or 'ok'}")

    if exposure is not None:
        rc, out = v4l2(device, f"--set-ctrl=exposure_time_absolute={exposure}")
        print(f"  {device}: exposure_time_absolute={exposure} -> {out or 'ok'}")
    if brightness is not None:
        _, out = v4l2(device, f"--set-ctrl=brightness={brightness}")
        print(f"  {device}: brightness={brightness} -> {out or 'ok'}")
    if gamma is not None:
        _, out = v4l2(device, f"--set-ctrl=gamma={gamma}")
        print(f"  {device}: gamma={gamma} -> {out or 'ok'}")
    if lock_wb:
        _, out = v4l2(device, "--set-ctrl=white_balance_automatic=0")
        print(f"  {device}: white_balance_automatic=0 (locked) -> {out or 'ok'}")
    if powerline is not None:
        _, out = v4l2(device, f"--set-ctrl=power_line_frequency={powerline}")
        print(f"  {device}: power_line_frequency={powerline} -> {out or 'ok'}")


def shoot(device: str, outdir: Path, tag: str) -> Path | None:
    """Grab one frame through OpenCV and report mean brightness."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  (opencv unavailable, skipping frame grab)", file=sys.stderr)
        return None

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  {device}: could not open for capture", file=sys.stderr)
        return None
    # Let AE/manual settle and discard the first few frames.
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    blown = float((gray >= 250).mean() * 100)
    dark = float((gray <= 5).mean() * 100)
    verdict = "BLOWN OUT" if blown > 20 else ("too dark" if mean < 40 else "ok")
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{tag}.jpg"
    cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(
        f"  {device}: mean={mean:6.1f}  blown={blown:5.1f}%  dark={dark:5.1f}%  "
        f"[{verdict}]  -> {p}"
    )
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", action="append", default=[],
                    help="repeatable, e.g. --device 4 (global) --device 2 (wrist)")
    ap.add_argument("--list", action="store_true", help="show current controls and exit")
    ap.add_argument("--apply", action="store_true", help="set the values and exit")
    ap.add_argument("--sweep", action="store_true", help="sweep exposure, saving a frame each step")
    ap.add_argument("--exposure", type=int, default=None, help="exposure_time_absolute (1-5000)")
    ap.add_argument("--brightness", type=int, default=None)
    ap.add_argument("--gamma", type=int, default=None)
    ap.add_argument("--lock-whitebalance", action="store_true")
    ap.add_argument("--powerline", type=int, default=None, choices=[0, 1, 2],
                    help="0=disabled 1=50Hz 2=60Hz (set 1 to match the 50Hz default)")
    ap.add_argument("--sweep-values", default="100,200,300,400,600,900",
                    help="comma-separated exposure values for --sweep")
    ap.add_argument("--outdir", default="data/camera_tune")
    args = ap.parse_args()

    devices = [f"/dev/video{d}" for d in args.device]
    if not devices:
        devices = ["/dev/video4", "/dev/video2"]  # global, wrist
        print("No --device given, defaulting to global=/dev/video4 wrist=/dev/video2\n")

    if args.list:
        for d in devices:
            print(f"=== {d} ===")
            show(d)
        return 0

    outdir = Path(args.outdir)

    if args.sweep:
        values = [int(v) for v in args.sweep_values.split(",") if v.strip()]
        for d in devices:
            print(f"=== sweeping {d} ===")
            apply(d, None, args.lock_whitebalance, args.powerline, args.brightness, args.gamma)
            for v in values:
                v4l2(d, f"--set-ctrl=exposure_time_absolute={v}")
                shoot(d, outdir, f"dev{d.replace('/dev/video', '')}_exp{v:04d}")
        return 0

    if args.apply:
        for d in devices:
            print(f"=== {d} ===")
            apply(d, args.exposure, args.lock_whitebalance, args.powerline,
                  args.brightness, args.gamma)
            shoot(d, outdir, f"dev{d.replace('/dev/video', '')}_applied")
        return 0

    # Default: just report.
    for d in devices:
        print(f"=== {d} ===")
        show(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
