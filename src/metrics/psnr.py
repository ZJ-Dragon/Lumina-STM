"""
Lumina-STM – Peak‑Signal‑to‑Noise Ratio (PSNR)

Implemented for both NumPy and PyTorch tensors. Supports:
* custom peak value (default HDR peak = **4.0** – same as dataset range)
* optional boolean mask (same shape as image, True = include)
* flexible reduction: "mean" | "sum" | "none"

API
---
>>> from src.metrics.psnr import compute_psnr
>>> psnr = compute_psnr(pred, gt, peak=4.0)        # float scalar
"""

from __future__ import annotations

from typing import Literal, Union, overload

import numpy as np
import torch

__all__ = ["compute_psnr"]


ArrayLike = Union[np.ndarray, torch.Tensor]
Reduction = Literal["mean", "sum", "none"]


def _mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    diff = (a - b) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean() if diff.size else 0.0)


def _mse_torch(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    diff = (a - b) ** 2
    if mask is not None:
        diff = diff[mask]
    return diff.mean() if diff.numel() else torch.tensor(0.0, dtype=a.dtype, device=a.device)


@overload
def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    /,
    *,
    peak: float = 4.0,
    mask: np.ndarray | None = None,
) -> float:
    ...


@overload
def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    peak: float = 4.0,
    mask: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> float:
    ...


def compute_psnr(
    pred: ArrayLike,
    target: ArrayLike,
    /,
    *,
    peak: float = 4.0,
    mask: ArrayLike | None = None,
    reduction: Reduction = "mean",
) -> float:
    """
    Compute PSNR between **pred** and **target**.

    Parameters
    ----------
    pred, target : ndarray | Tensor
        Predicted and ground‑truth images. Shape (H, W, C) or (C, H, W) accepted.
    peak : float, default 4.0
        Maximum possible pixel value in the images.
    mask : ndarray | Tensor, optional
        Boolean mask selecting pixels to include. Must broadcast to image shape.
    reduction : {'mean', 'sum', 'none'}
        How to reduce PSNR if `pred` is batched (only for torch version).

    Returns
    -------
    float
        PSNR value in dB. If multiple images, returns average according to `reduction`.
    """
    if isinstance(pred, np.ndarray):
        mse_val = _mse(pred.astype(np.float64), target.astype(np.float64), mask)
        if mse_val == 0:
            return float("inf")
        return 20.0 * np.log10(peak) - 10.0 * np.log10(mse_val)

    # torch branch – supports batches
    if not isinstance(pred, torch.Tensor):
        raise TypeError("Inputs must be either numpy.ndarray or torch.Tensor")

    pred_f = pred.float()
    target_f = target.float()
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=pred_f.device)

    if pred_f.dim() == 4:  # NCHW
        psnr_list = []
        for i in range(pred_f.size(0)):
            mse = _mse_torch(pred_f[i], target_f[i], mask[i] if mask is not None else None)
            psnr_list.append(
                torch.tensor(float("inf")) if mse == 0 else 20 * torch.log10(torch.tensor(peak)) - 10 * torch.log10(mse)
            )
        psnr_stack = torch.stack(psnr_list)
        if reduction == "mean":
            return float(psnr_stack.mean().item())
        if reduction == "sum":
            return float(psnr_stack.sum().item())
        return psnr_stack  # type: ignore[return-value]

    # single image
    mse_val = _mse_torch(pred_f, target_f, mask)
    return float("inf") if mse_val == 0 else float(20 * torch.log10(torch.tensor(peak)) - 10 * torch.log10(mse_val))
