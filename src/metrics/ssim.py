"""
Lumina-STM – Structural Similarity Index (SSIM)

Torch‑native 11×11 Gaussian window implementation.

Features
--------
* Works on **RGB** (default) or **grayscale** images.
* Accepts single image (C,H,W) or batches (N,C,H,W).
* Peak value default 4.0 to match HDR range.
* Returns scalar float (average over batch & channels) unless `reduction='none'`.

References
----------
Wang et al., “Image Quality Assessment: From Error Visibility to Structural
Similarity,” IEEE TIP 2004.
"""
from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F

__all__ = ["compute_ssim"]

Reduction = Literal["mean", "sum", "none"]


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Generate 1‑D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.reshape(1, 1, -1)  # shape (1,1,W)


def _create_window(window_size: int, channel: int, device, dtype) -> torch.Tensor:
    """Create 2‑D Gaussian window of shape (C,1,H,W) for depth‑wise conv."""
    _1d = _gaussian_kernel(window_size, 1.5, device, dtype)
    _2d = _1d.transpose(-1, -2) @ _1d  # outer prod → (1,1,H,W)
    window = _2d.expand(channel, 1, window_size, window_size)
    return window


def _ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    peak: float,
    C1: float,
    C2: float,
) -> torch.Tensor:
    """Core SSIM calc on NCHW tensors (assumes range [0, peak])."""
    mu1 = F.conv2d(img1, window, groups=img1.size(1), padding=window_size // 2)
    mu2 = F.conv2d(img2, window, groups=img2.size(1), padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=img1.size(1), padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=img2.size(1), padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=img1.size(1), padding=window_size // 2) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return ssim_map.mean(dim=[2, 3])  # keep N,C


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    window_size: int = 11,
    peak: float = 4.0,
    color: Literal["rgb", "gray"] = "rgb",
    reduction: Reduction = "mean",
) -> float | torch.Tensor:
    """
    Compute SSIM between `pred` and `target`.

    Parameters
    ----------
    pred, target : torch.Tensor
        Image tensors. Allowed shapes:
        * (N,C,H,W) batch
        * (C,H,W) single image
    window_size : int
        Gaussian kernel size (odd, default 11).
    peak : float
        Maximum possible pixel value (HDR default 4.0).
    color : {'rgb','gray'}
        If 'gray', convert via luminance Y = 0.299R+0.587G+0.114B before metric.
    reduction : {'mean','sum','none'}
        Reduction across batch & channels.

    Returns
    -------
    float or torch.Tensor
        Scalar metric (float) if reduction ≠ 'none'; else per‑image tensor (N,).
    """
    if pred.shape != target.shape:
        raise ValueError("Input shapes must match")

    if pred.dim() == 3:  # C,H,W → add batch dim
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    if color == "gray":
        if pred.size(1) == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=pred.device, dtype=pred.dtype).view(1, 3, 1, 1)
            pred = (pred * weights).sum(dim=1, keepdim=True)
            target = (target * weights).sum(dim=1, keepdim=True)
        # if already single‑channel, no change
    elif color != "rgb":
        raise ValueError("color must be 'rgb' or 'gray'")

    channel = pred.size(1)
    window = _create_window(window_size, channel, pred.device, pred.dtype)

    K1, K2 = 0.01, 0.03
    C1 = (K1 * peak) ** 2
    C2 = (K2 * peak) ** 2

    ssim_per_channel = _ssim_torch(pred, target, window, window_size, peak, C1, C2)  # shape (N,C)

    if reduction == "none":
        return ssim_per_channel.mean(dim=1)  # (N,)
    score = ssim_per_channel.mean()
    if reduction == "sum":
        return float(score.sum().item())
    return float(score.item())
