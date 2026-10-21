"""Data augmentation transforms for imitation learning.

Visual augmentations for global camera (position-invariant) and wrist camera
(ego-centric), plus state noise for joint angles.

Usage:
    from vla.augment import GlobalCamAugment, WristCamAugment, StateAugment

    global_aug = GlobalCamAugment()
    wrist_aug = WristCamAugment()
    state_aug = StateAugment(noise_std=0.02)

    aug_global_img = global_aug(raw_global_img)  # (H, W, 3) uint8
    aug_wrist_img = wrist_aug(raw_wrist_img)
    aug_state = state_aug(raw_joint_state)
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torchvision import transforms


class GlobalCamAugment:
    """Aggressive augmentations for the global camera.

    Simulates different camera positions, lighting, and focus.
    The global camera can be anywhere — these transforms force the policy
    to learn from duck/tape features rather than memorizing pixel locations.
    """

    def __init__(
        self,
        crop_scale: tuple[float, float] = (0.7, 1.0),
        color_jitter: float = 0.2,
        blur_prob: float = 0.3,
        blur_kernel: int = 5,
        rotation_deg: float = 15.0,
        h_flip_prob: float = 0.5,
        enable: bool = True,
    ):
        self.enable = enable
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                (480, 640), scale=crop_scale, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter * 0.75,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))
            ], p=blur_prob),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=rotation_deg)
            ], p=0.5),
            transforms.RandomHorizontalFlip(p=h_flip_prob),
            transforms.ToTensor(),  # (3, H, W) float32 [0, 1]
        ])

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """img: (H, W, 3) uint8 RGB → (3, H, W) float32 [0, 1]"""
        if not self.enable:
            return transforms.ToTensor()(img)
        return self.transform(img)


class WristCamAugment:
    """Light augmentations for the wrist camera.

    The wrist camera is ego-centric (moves with gripper), so geometry matters.
    Only color/lighting changes — no crops or flips.
    """

    def __init__(
        self,
        color_jitter: float = 0.15,
        noise_std: float = 5.0,
        enable: bool = True,
    ):
        self.enable = enable
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter * 0.5,
        )
        self.noise_std = noise_std

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """img: (H, W, 3) uint8 RGB → (3, H, W) float32 [0, 1]"""
        t = transforms.ToTensor()(img)  # (3, H, W) float32 [0, 1]

        if not self.enable:
            return t

        # Color jitter (operates on tensor)
        t = self.color_jitter(t)

        # Light Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(t) * (self.noise_std / 255.0)
            t = torch.clamp(t + noise, 0.0, 1.0)

        return t


class StateAugment:
    """Gaussian noise on joint angles.

    Simulates joint backlash and sensor noise in the real robot.
    """

    def __init__(self, noise_std: float = 0.02, enable: bool = True):
        self.noise_std = noise_std
        self.enable = enable

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """state: (N,) float32 joint angles in radians → same shape, noised."""
        if not self.enable:
            return state
        noise = np.random.normal(0, self.noise_std, size=state.shape).astype(np.float32)
        return state + noise


class TimeWarp:
    """Slightly speed up or slow down a trajectory.

    Resamples the trajectory at a different rate, making the policy robust
    to timing variations. Operates on arrays of shape (T, ...).
    """

    def __init__(self, warp_range: tuple[float, float] = (0.9, 1.1), enable: bool = True):
        self.warp_range = warp_range
        self.enable = enable

    def __call__(
        self,
        images_wrist: np.ndarray,
        images_global: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """All inputs have T as first dim. Returns resampled versions."""
        if not self.enable:
            return images_wrist, images_global, states, actions

        T = len(actions)
        warp = random.uniform(*self.warp_range)
        new_T = max(10, int(T * warp))
        src_idx = np.linspace(0, T - 1, new_T).astype(int)
        src_idx = np.clip(src_idx, 0, T - 1)

        return (
            images_wrist[src_idx],
            images_global[src_idx],
            states[src_idx],
            actions[src_idx],
        )


class EpisodeAugmenter:
    """Full augmentation pipeline for one episode.

    Usage:
        augmenter = EpisodeAugmenter()
        aug_data = augmenter(
            images_wrist,  # (T, H, W, 3) uint8
            images_global, # (T, H, W, 3) uint8
            states,        # (T, 6) float32
            actions,       # (T, 6) float32
        )
    """

    def __init__(
        self,
        global_aug: GlobalCamAugment | None = None,
        wrist_aug: WristCamAugment | None = None,
        state_aug: StateAugment | None = None,
        time_warp: TimeWarp | None = None,
    ):
        self.global_aug = global_aug or GlobalCamAugment()
        self.wrist_aug = wrist_aug or WristCamAugment()
        self.state_aug = state_aug or StateAugment()
        self.time_warp = time_warp or TimeWarp()

    def __call__(
        self,
        images_wrist: np.ndarray,
        images_global: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Augment one episode. Returns dict with augmented tensors."""
        T = len(actions)

        # Time warp (resample all temporal data)
        images_wrist, images_global, states, actions = self.time_warp(
            images_wrist, images_global, states, actions
        )

        # Per-frame visual augmentation
        aug_wrist = torch.stack([
            self.wrist_aug(images_wrist[i]) for i in range(len(images_wrist))
        ])
        aug_global = torch.stack([
            self.global_aug(images_global[i]) for i in range(len(images_global))
        ])

        # State augmentation (per timestep)
        aug_states = np.stack([
            self.state_aug(states[i]) for i in range(len(states))
        ])

        return {
            "images.wrist": aug_wrist,       # (T, 3, H, W) float32
            "images.global": aug_global,     # (T, 3, H, W) float32
            "observation.state": aug_states,  # (T, 6) float32
            "action": actions,               # (T, 6) float32 — no augmentation on actions
        }


def augment_dataset(
    input_path: str,
    output_path: str,
    n_copies: int = 3,
    enable_time_warp: bool = True,
):
    """Augment an entire dataset, producing n_copies augmented variants per episode.

    Args:
        input_path: Path to raw LeRobotDataset (npz episodes)
        output_path: Path to write augmented dataset
        n_copies: Number of augmented copies per original episode
        enable_time_warp: Whether to apply time warping
    """
    from pathlib import Path

    input_dir = Path(input_path) / "episodes"
    output_dir = Path(output_path) / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)

    augmenter = EpisodeAugmenter(
        time_warp=TimeWarp(enable=enable_time_warp),
    )

    ep_files = sorted(input_dir.glob("episode_*.npz"))
    print(f"Augmenting {len(ep_files)} episodes × {n_copies} copies = {len(ep_files) * n_copies} total")

    aug_idx = 0
    for ep_file in ep_files:
        data = np.load(ep_file)
        images_wrist = data["images.wrist"]
        images_global = data["images.global"]
        states = data["observation.state"]
        actions = data["action"]

        # Save original
        np.savez_compressed(
            output_dir / f"episode_{aug_idx:06d}.npz",
            images_wrist=images_wrist,
            images_global=images_global,
            observation_state=states,
            action=actions,
        )
        aug_idx += 1

        # Save augmented copies
        for copy_i in range(n_copies):
            aug = augmenter(images_wrist, images_global, states, actions)
            np.savez_compressed(
                output_dir / f"episode_{aug_idx:06d}.npz",
                images_wrist=aug["images.wrist"].permute(0, 2, 3, 1).numpy().astype(np.uint8),
                images_global=aug["images.global"].permute(0, 2, 3, 1).numpy().astype(np.uint8),
                observation_state=aug["observation.state"],
                action=aug["action"],
            )
            aug_idx += 1

        if (aug_idx // (n_copies + 1)) % 50 == 0:
            print(f"  Processed {aug_idx // (n_copies + 1)}/{len(ep_files)} episodes")

    print(f"Done: {aug_idx} total episodes in {output_path}")
