#!/usr/bin/env python3
"""Offline checks for scripts/so101_camera_flip.py. No hardware, no cameras.

The patch is a monkeypatch, so "it imported" proves nothing. What matters:

- a frame from a listed camera comes out vertically flipped
- a frame from an unlisted camera comes out **exactly** as LeRobot produced it
- the flip is the only added step, and it composes with LeRobot's own rotation
- installing twice does not flip twice, and an empty set really is pass-through

"Exactly as LeRobot produced it" is checked against LeRobot's own unpatched
`_postprocess_image`, not against the raw input: that function also converts
BGR->RGB and applies `rotation`, so the untouched output is NOT the input.

    .venv/bin/python scripts/test_camera_flip.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import so101_camera_flip as flip

import cv2

FAILED: list[str] = []

# Recorded (post-rotation) image is always 8 rows x 6 cols for every rotation.
REC_H, REC_W = 8, 6


def check(cond: bool, label: str) -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILED.append(label)


def sample(h: int, w: int) -> np.ndarray:
    """An (h, w, 3) frame that is NOT vertically symmetric, so a flip shows.

    Row index drives red, column index drives green. A constant or a symmetric
    test pattern would come out identical flipped or not.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (np.arange(h, dtype=np.int32)[:, None] * (200 // max(h, 1))).astype(np.uint8)
    img[:, :, 1] = (np.arange(w, dtype=np.int32)[None, :] * (200 // max(w, 1))).astype(np.uint8)
    img[:, :, 2] = 7
    return img


def make_cam(index, rotation: int = 0):
    """An OpenCVCamera that will accept a synthetic frame.

    The real _postprocess_image rejects any frame whose shape differs from
    capture_width/capture_height, which are only set from the config until
    connect(). So the config carries the size. `rotation` 90/-90 transposes the
    capture buffer, so declare the recorded size (REC_W x REC_H) and let
    _reset_connection_settings do the swap.
    """
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    return OpenCVCamera(OpenCVCameraConfig(
        index_or_path=index, width=REC_W, height=REC_H, rotation=rotation))


def capture_frame(cam) -> np.ndarray:
    """A frame shaped the way this camera's capture buffer is.

    The original _postprocess_image insists on exactly capture_height x
    capture_width, so the test frame has to match it.
    """
    return sample(cam.capture_height, cam.capture_width)


def unpatched(cam, img: np.ndarray) -> np.ndarray:
    """What LeRobot returns with no flip installed -- the reference."""
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    original = flip._ORIGINAL or OpenCVCamera._postprocess_image
    return original(cam, img)


def main() -> int:
    print("index normalisation")
    for value, want in [(4, 4), (0, 0), ("4", 4), ("/dev/video2", 2),
                        ("/dev/video12", 12), (None, None), (True, None)]:
        check(flip.video_index(value) == want, f"video_index({value!r}) == {want}")

    print("\ninstall")
    check(flip.install({4}), "install({4}) succeeds")
    check(flip._FLIP == {4}, f"flip set is {{4}} (got {flip._FLIP})")
    check(flip._ORIGINAL is not None, "the original method was captured for reference")

    cam_global = make_cam(4)
    cam_wrist = make_cam(2)
    cam_path = make_cam("/dev/video4")
    img = capture_frame(cam_global)
    base = unpatched(cam_global, img)

    print("\nflip behaviour")
    got = cam_global._postprocess_image(img)
    check(np.array_equal(got, cv2.flip(base, 0)),
          "listed camera (int 4) frame is LeRobot's frame, vertically flipped")
    check(got.shape == (REC_H, REC_W, 3), f"flipped frame keeps shape {got.shape}")
    check(got.flags["C_CONTIGUOUS"], "flipped frame is C-contiguous (no negative strides)")
    check(np.array_equal(got[::-1, :, :], base),
          "the flip is exactly vertical (rows reversed, columns kept)")

    got = cam_path._postprocess_image(img)
    check(np.array_equal(got, cv2.flip(base, 0)), "'/dev/video4' is recognised as the same device")

    got = cam_wrist._postprocess_image(capture_frame(cam_wrist))
    check(np.array_equal(got, unpatched(cam_wrist, capture_frame(cam_wrist))),
          "unlisted camera (2) frame is byte-identical to LeRobot's own output")
    check(flip.is_flipped(2) is False and flip.is_flipped(4) is True,
          "is_flipped() agrees with the config")

    print("\nidempotence")
    check(flip.install({4}), "re-install succeeds")
    got = cam_global._postprocess_image(img)
    check(np.array_equal(got, cv2.flip(base, 0)), "re-install does not flip a second time")

    print("\nempty config is a real state, not a no-op")
    check(flip.install(set()), "install(set()) succeeds")
    check(flip._FLIP == set(), "flip set is empty")
    check(np.array_equal(cam_global._postprocess_image(img), base),
          "with nothing listed, frames pass through untouched")
    check("none" in flip.describe(), f"describe() says none (got {flip.describe()!r})")

    print("\ncomposes with LeRobot's own rotation")
    # Cv2Rotation is 0 / 90 / 180 / -90. LeRobot applies it inside the original
    # _postprocess_image; the flip is applied after, so the recorded image is
    # right way up whichever way round it came in.
    from lerobot.cameras.utils import get_cv2_rotation

    check(flip.install({4}), "flip re-armed for the rotation checks")
    for rot in (0, 90, 180, -90):
        cam = make_cam(4, rotation=rot)
        src = capture_frame(cam)
        out = cam._postprocess_image(src)
        ref = unpatched(cam, src)
        name = f"cv2 rotate code {get_cv2_rotation(rot)}" if rot else "unrotated"
        check(out.shape == (REC_H, REC_W, 3),
              f"rotation={rot}: recorded shape is still {REC_H}x{REC_W} (got {out.shape})")
        check(np.array_equal(out, cv2.flip(ref, 0)),
              f"rotation={rot}: recorded frame is LeRobot's rotated frame, flipped")
        check(np.array_equal(cv2.flip(out, 0), ref),
              f"rotation={rot}: unflipping gives back the {name} frame")

    print("\nboth cameras at once")
    flip.install({2, 4})
    check(np.array_equal(cam_global._postprocess_image(img), cv2.flip(base, 0)),
          "global flipped")
    check(np.array_equal(cam_wrist._postprocess_image(capture_frame(cam_wrist)),
                         cv2.flip(unpatched(cam_wrist, capture_frame(cam_wrist)), 0)),
          "wrist flipped")
    check(flip.describe() == "/dev/video2, /dev/video4",
          f"describe() lists both (got {flip.describe()!r})")

    flip.install(set())  # leave the process clean for anything imported later

    print("\nraw-cv2 path (vla/deploy.py RealCamera, vla/record_demos.py)")
    # These do not go through OpenCVCamera, so install()'s patch never reaches
    # them. apply_to_frame() is the only thing keeping inference oriented the
    # same way as the recording.
    grad = sample(8, 6)
    check(not np.array_equal(flip.apply_to_frame(grad, 2, indices={2}), grad),
          "explicit indices: a listed camera is flipped")
    check(np.array_equal(flip.apply_to_frame(grad, 4, indices={2}), grad),
          "explicit indices: video4 left alone")
    check(np.array_equal(flip.apply_to_frame(grad, 4, indices=set()), grad),
          "empty explicit set is pass-through")
    check(np.array_equal(flip.apply_to_frame(grad, "not-a-camera", indices={2}), grad),
          "an unparseable device id is left alone rather than guessed at")
    check(np.array_equal(flip.apply_to_frame(grad, 2, indices={2}), cv2.flip(grad, 0)),
          "explicit indices: the flip matches cv2.flip exactly")

    cfg = flip.config_flip(refresh=True)
    check(isinstance(cfg, set) and all(isinstance(i, int) for i in cfg),
          f"config_flip() resolves VFLIP_CAMS to indices (got {cfg})")
    for dev in (2, 4):
        got = flip.apply_to_frame(grad, dev)          # no indices= -> from config
        expect = cv2.flip(grad, 0) if dev in cfg else grad
        check(np.array_equal(got, expect),
              f"config-driven: video{dev} {'flipped' if dev in cfg else 'untouched'}, "
              f"matching the recorder")

    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
