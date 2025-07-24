"""
datasets.py
===========

PyTorch dataset for Lumina‑STM that loads Sony / Nikon RAW files
via RawLoader and returns a 4‑channel linear tensor suitable
for training Transformer / Diffusion models.

Features
--------
* Random crop with Bayer‑phase alignment (even coordinates).
* Optional white‑balance jitter to improve robustness.
* Multi‑process safe: each dataloader worker owns its own RawLoader.
Pixel values are linear‑normalised by white‑level and **may be larger than 1.0** after white‑balance jitter.
* Optional online augmentation via configs/augmentations.yaml.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import yaml
from .augmentations import build_pipeline
from torch.utils.data import Dataset, get_worker_info

from .raw_loader import RawLoader, raw_to_tensor


def _aligned_random_crop(
    tensor: np.ndarray, crop_size: int
) -> np.ndarray:
    """Return an even‑aligned random crop of size (S, S, 4)."""
    h, w, _ = tensor.shape
    if h < crop_size or w < crop_size:
        raise ValueError("Crop size larger than image.")
    # enforce even coordinates to preserve RGGB phase
    y0 = np.random.randint(0, (h - crop_size) // 2 + 1) * 2
    x0 = np.random.randint(0, (w - crop_size) // 2 + 1) * 2
    return tensor[y0 : y0 + crop_size, x0 : x0 + crop_size, :]


class RawTensorDataset(Dataset):
    """PyTorch Dataset that yields 4‑channel RAW tensors."""

    def __init__(
        self,
        files: Sequence[Path] | Sequence[str],
        crop_size: int = 512,
        wb_jitter: float = 0.1,
        random_crop: bool = True,
        *,
        augment_cfg: Path | str | None = None,
        augment_mode: str = "none",
        rng_seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        files : list[str or Path]
            Absolute paths to RAW files.
        crop_size : int
            Square crop size, must be even.
        wb_jitter : float
            White‑balance multiplicative jitter range ±wb_jitter.
        random_crop : bool
            If *False*, return centre crop; otherwise random crop.
        """
        self.files: List[Path] = [Path(f) for f in files]
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.wb_jitter = wb_jitter

        # augmentation pipeline (online mode only)
        self.pipeline = None
        if augment_mode not in {"online", "offline", "none"}:
            raise ValueError("augment_mode must be 'online', 'offline', or 'none'")
        if augment_mode == "online":
            if augment_cfg is None:
                raise ValueError("augment_cfg is required when augment_mode='online'")
            with Path(augment_cfg).expanduser().open("r") as f:
                _cfg = yaml.safe_load(f)
            self.pipeline = build_pipeline(_cfg, mode="online", seed=rng_seed)

        if crop_size % 2 != 0:
            raise ValueError("crop_size must be even to keep Bayer phase.")

        # placeholder for per‑worker RawLoader, will be set in __getitem__
        self._loader: RawLoader | None = None

    # -------------------------------------------------- #
    def __len__(self) -> int:  # noqa: Dunder
        return len(self.files)

    # -------------------------------------------------- #
    def _get_loader(self) -> RawLoader:
        """Return a worker‑local RawLoader instance."""
        if self._loader is None:
            # Each worker constructs its own RawLoader to avoid shared state.
            self._loader = RawLoader()
        return self._loader

    # -------------------------------------------------- #
    def __getitem__(self, idx: int) -> torch.Tensor:  # noqa: Dunder
        file_path = self.files[idx]
        tensor = self._get_loader().load(file_path)  # (H/2,W/2,4), float32

        # random / centre crop
        if self.random_crop:
            tensor = _aligned_random_crop(tensor, self.crop_size)
        else:
            # centre crop
            h, w, _ = tensor.shape
            y0 = (h - self.crop_size) // 2 // 2 * 2
            x0 = (w - self.crop_size) // 2 // 2 * 2
            tensor = tensor[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size, :]

        # white‑balance jitter augmentation
        if self.wb_jitter > 0:
            gains = np.random.uniform(
                1.0 - self.wb_jitter, 1.0 + self.wb_jitter, size=(4,)
            ).astype(np.float32)
            tensor = tensor * gains[None, None, :]

        # apply online augmentation pipeline if present
        if self.pipeline is not None:
            tensor = self.pipeline(tensor)

        # Note: we intentionally keep values >1.0 (HDR headroom) for the model to learn tone‑mapping.

        # to CHW torch tensor
        tensor = torch.from_numpy(tensor).permute(2, 0, 1)  # (C,H,W)
        return tensor