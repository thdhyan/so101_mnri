#!/usr/bin/env python3
"""Flip upside-down cameras the right way up, inside LeRobot's camera thread.

The global camera is now mounted inverted (new rig, 2026-09-26), so its frames
come out upside down. Two things rule out the obvious fixes:

- Neither UVC camera on this rig has a flip control. `v4l2-ctl -d /dev/video4
  -L` and `-d /dev/video2 -L` list brightness/contrast/exposure and nothing
  else -- no flip, no mirror, no rotate.
- `OpenCVCameraConfig.rotation` is not a flip. It only accepts 0/90/180/270,
  and 180 is a vertical AND horizontal mirror, which is the wrong transform
  when only the vertical axis is backwards.

So the frame is flipped in software, in `OpenCVCamera._postprocess_image`.

Why that seam, and why it is safe for the record loop:

- `_postprocess_image` is called from the camera's own background thread
  (`_read_loop`), which the recorder only peeks at via `read_latest()`. So the
  cost lands on the camera thread, not on the 15 Hz record loop. Measured
  `cv2.flip` on 640x480x3 at **0.026 ms/frame** -- 4x cheaper than the
  BGR->RGB convert already in that same function, against a 47.5 ms loop budget.
- Every consumer sees the corrected frame: the recorded dataset, the Rerun live
  view, and anything that reads the camera later. Flipping in the recorder
  instead would leave the live view and every other consumer inverted.
- The shape check in `_postprocess_image` (`h != self.capture_height`) runs
  before the flip, and a flip does not change the shape, so nothing downstream
  -- including the video encoder -- can notice.

Flip last, after the colour conversion and any `rotation`, so the semantics are
"the image the operator sees is the right way up".

    from so101_camera_flip import install
    install({4})            # flip /dev/video4 only

Config lives in `scripts/so101_env.sh` as `VFLIP_CAMS` (camera NAMES, not
indices); `scripts/record_so101.sh` resolves the names to indices and passes
`--vflip=`.

To eyeball the result before a session (no arms touched):

    .venv/bin/python scripts/so101_camera_flip.py --snap 2,4 --snap-out /tmp/flip

The flip set comes from `VFLIP_CAMS` in `scripts/so101_env.sh` unless `--flip`
overrides it, so those PNGs are the frames exactly as they would be recorded.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Video indices whose frames are upside down. Empty means "flip nothing", which
# is a valid installed state -- install() must not be a no-op, or a run with
# VFLIP_CAMS=none would silently keep a stale flip from an earlier import.
_FLIP: set[int] = set()
_ORIGINAL = None
_CONFIG_FLIP: set[int] | None = None


def video_index(value: object) -> int | None:
    """Normalise a camera's `index_or_path` to a /dev/videoN index.

    LeRobot stores whatever the config gave it: `2` from `--robot.cameras`,
    but `"/dev/video2"` from a hand-written config or a find-cameras dump.
    """
    if isinstance(value, bool):  # bool is an int subclass; not a camera
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
        if s.startswith("/dev/video"):
            tail = s.removeprefix("/dev/video")
            if tail.isdigit():
                return int(tail)
    return None


def is_flipped(value: object) -> bool:
    """Whether a camera with this `index_or_path` is being flipped."""
    idx = video_index(value)
    return idx is not None and idx in _FLIP


def install(indices) -> bool:
    """Flip the frames of the given /dev/videoN indices. Idempotent.

    Returns False only if LeRobot's OpenCVCamera cannot be patched, so the
    caller can refuse to record rather than quietly write inverted data.
    """
    global _ORIGINAL
    wanted = {video_index(i) for i in indices}
    _FLIP.clear()
    _FLIP.update(i for i in wanted if i is not None)

    try:
        import cv2
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    except ImportError as e:  # pragma: no cover - environment problem
        print(f"ERROR: cannot patch OpenCVCamera: {e}", file=sys.stderr)
        return False

    if _ORIGINAL is not None:
        # Already wrapped (re-install in the same process, e.g. a test run).
        # The index set is authoritative and was just updated above.
        return True

    original = OpenCVCamera._postprocess_image

    def _postprocess_image(self, image):
        out = original(self, image)
        if is_flipped(self.index_or_path):
            # 0 = flip about the x axis = vertical flip. cv2.flip, not
            # image[::-1]: a reversed view has negative strides, which the
            # encoder and cvtColor both cope badly with.
            out = cv2.flip(out, 0)
        return out

    _postprocess_image.__wrapped__ = original
    OpenCVCamera._postprocess_image = _postprocess_image
    _ORIGINAL = original
    return True


def describe() -> str:
    """One line for the launcher banner / dry-run output."""
    if not _FLIP:
        return "none (frames recorded as the camera delivers them)"
    return ", ".join(f"/dev/video{i}" for i in sorted(_FLIP))


def _snap(out_dir: Path, indices: list[int], width: int, height: int,
          fourcc: str, warmup_s: float) -> int:
    """Write a PNG per camera through the PATCHED path, so the orientation in
    the file is the orientation that would be recorded."""
    import time

    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    for dev in indices:
        cap = cv2.VideoCapture(f"/dev/video{dev}", cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[FAIL] /dev/video{dev} did not open")
            return 1
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(warmup_s)
        frame = None
        for _ in range(10):
            ok, f = cap.read()
            if ok:
                frame = f
                break
            time.sleep(0.3)
        cap.release()
        if frame is None:
            print(f"[FAIL] /dev/video{dev} produced no frame in 3s")
            return 1

        # Same transform the patch applies, on the same frame: this file is
        # byte-for-byte what add_frame would have received.
        if dev in _FLIP:
            frame = cv2.flip(frame, 0)
        path = out_dir / f"video{dev}_as_recorded.png"
        cv2.imwrite(str(path), frame)
        state = "VFLIPPED" if dev in _FLIP else "as delivered"
        print(f"[ok]   /dev/video{dev} {frame.shape[1]}x{frame.shape[0]} "
              f"mean={frame.mean():.1f} std={frame.std():.1f} [{state}] -> {path}")

    print("\nOpen the PNGs and check they are the right way up. A vertical flip "
          "is not\ndetectable from mean/std -- a flip only reorders pixels -- so "
          "this is a\nlook-at-it check. (A picture of a duck and its tape square "
          "makes it obvious.)")
    return 0


def config_env() -> dict[str, str]:
    """Read scripts/so101_env.sh, the way check_record_config.py does."""
    import re

    env_file = Path(__file__).resolve().parent / "so101_env.sh"
    out: dict[str, str] = {}
    if not env_file.is_file():
        return out
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#", 1)[0].strip().strip("\"'")
        out[key.strip()] = val
    return out


def config_flip(refresh: bool = False) -> set[int]:
    """The flip set the launcher would install, resolved from the config.

    Same name -> index resolution as record_so101.sh, so `--snap` with no
    `--flip` shows the frames exactly as they would be recorded. Parsed here
    rather than shelled out to, so the check cannot disagree with the run.
    Cached, because the inference path calls this once per frame.
    """
    global _CONFIG_FLIP
    if _CONFIG_FLIP is None or refresh:
        env = config_env()
        wanted = {n for n in env.get("VFLIP_CAMS", "none").replace(" ", "").split(",") if n}
        idx: set[int] = set()
        for name, key in (("global", "GLOBAL_CAM"), ("wrist", "WRIST_CAM")):
            if wanted & {name, "all", "both"} and env.get(key, "").isdigit():
                idx.add(int(env[key]))
        _CONFIG_FLIP = idx
    return set(_CONFIG_FLIP)


def apply_to_frame(frame, device_id, indices=None):
    """Flip a raw frame the way the recorder would. For camera paths that do
    NOT go through LeRobot's OpenCVCamera.

    `vla/deploy.py:RealCamera` (used by both deployment and
    `vla/record_video.py`) and `vla/record_demos.py` open cameras with raw
    `cv2.VideoCapture`, so `install()`'s patch never reaches them. Without this
    the policy is trained on upright frames and served inverted ones, which
    fails silently -- the policy just gets worse.

    `indices=None` means "use VFLIP_CAMS from scripts/so101_env.sh", so there is
    one source of truth for orientation across recording and inference.
    """
    idx = video_index(device_id)
    if idx is None:
        return frame
    flip_set = config_flip() if indices is None else {video_index(i) for i in indices}
    if idx in flip_set:
        import cv2

        return cv2.flip(frame, 0)
    return frame


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snap", metavar="INDEXES", default=None,
                    help="comma-separated /dev/videoN indices to snapshot, e.g. 2,4")
    ap.add_argument("--flip", metavar="INDEXES", default=None,
                    help="override which of those are flipped; default is VFLIP_CAMS "
                         "from scripts/so101_env.sh, i.e. what would be recorded")
    ap.add_argument("--snap-out", default="/tmp/so101_flip", help="output directory")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--warmup-s", type=float, default=2.0)
    args = ap.parse_args(argv)

    if args.snap is None:
        print(f"VFLIP_CAMS would install: {describe() or 'nothing'}", file=sys.stderr)
        return 0
    indices = [int(x) for x in args.snap.split(",") if x.strip()]
    if args.flip is not None:
        flip_idx = {int(x) for x in args.flip.split(",") if x.strip()}
        source = "--flip"
    else:
        flip_idx = config_flip()
        source = "VFLIP_CAMS in scripts/so101_env.sh"
    if not install(flip_idx):
        return 1
    print(f"[flip] flipping {describe()} (from {source})")
    return _snap(Path(args.snap_out), indices, args.width, args.height,
                 args.fourcc, args.warmup_s)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
