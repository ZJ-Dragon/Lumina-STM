

"""
Lumina‑STM – Metrics utility helpers
===================================

This module contains small, dependency‑light helpers that are reused by
multiple metric implementations (PSNR/SSIM/LPIPS/FID) and by the CLI
`scripts/eval_metrics.py`.

Key features
------------
* ``batch_iter``      – memory‑friendly iterator over paired image paths.
* ``load_image``      – Pillow‑based loader → float32 numpy array in range 0‑1.
* ``align_tensors``   – broadcast / pad tensors so they share height & width.
* ``distributed_mean``– torch.distributed all‑reduce helper that gracefully
                         falls back to single‑process mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch


# ------------------------------------------------------------------------- #
# IO helpers                                                                #
# ------------------------------------------------------------------------- #
def load_image(path: str | Path, mode: str = "RGB") -> np.ndarray:
    """
    Load an image as float32 numpy array in range [0, 1].

    Parameters
    ----------
    path : str | Path
        Image file path.
    mode : str
        Pillow mode to convert to (default RGB).

    Returns
    -------
    np.ndarray
        Array shape (H, W, C) with dtype float32.
    """
    img = Image.open(path).convert(mode)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def batch_iter(
    preds: Sequence[Path] | Sequence[str],
    gts: Sequence[Path] | Sequence[str],
    batch_size: int = 16,
) -> Iterator[Tuple[List[Path], List[Path]]]:
    """
    Yield batches of prediction / ground‑truth path pairs.

    Notes
    -----
    * Lists must be equal length and **aligned by index**.
    * No shuffling is performed – caller decides if random order is needed.
    """
    if len(preds) != len(gts):
        raise ValueError("preds and gts must have the same length")

    total = len(preds)
    for i in range(0, total, batch_size):
        yield list(preds[i : i + batch_size]), list(gts[i : i + batch_size])


# ------------------------------------------------------------------------- #
# Tensor helpers                                                            #
# ------------------------------------------------------------------------- #
def align_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If tensors differ in H/W, centrally pad the smaller one with reflection
    so that both tensors share identical spatial dimensions.

    Returns a **copy** (never in‑place) to preserve gradient graph safety.
    """
    if a.shape[-2:] == b.shape[-2:]:
        return a, b

    def _pad(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = target_h - h
        pad_w = target_w - w
        # reflection padding – symmetrical
        return torch.nn.functional.pad(
            x,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            mode="reflect",
        )

    target_h = max(a.shape[-2], b.shape[-2])
    target_w = max(a.shape[-1], b.shape[-1])
    return _pad(a, target_h, target_w), _pad(b, target_h, target_w)


# ------------------------------------------------------------------------- #
# Distributed helpers                                                       #
# ------------------------------------------------------------------------- #
def distributed_mean(value: torch.Tensor) -> torch.Tensor:
    """
    All‑reduce average across **world_size** GPUs if torch.distributed is
    initialised; otherwise returns value unchanged.

    Parameters
    ----------
    value : torch.Tensor
        Scalar tensor (any shape works, but typical use is scalar metric).

    Returns
    -------
    torch.Tensor
        Tensor containing the averaged value across processes (or original).
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        value = value.clone()
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        value /= torch.distributed.get_world_size()
    return value