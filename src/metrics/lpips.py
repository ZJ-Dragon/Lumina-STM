

"""
Lumina-STM – Learned Perceptual Image Patch Similarity (LPIPS)

Thin wrapper around the official lpips PyPI package.

Highlights
----------
* Automatic download of network weights (VGG/AlexNet/Squeeze) to
  `~/.cache/lpips/` or a custom `$LPIPS_CACHE_DIR`.
* Supports batched NCHW tensors in range **[0, 1]** or **[0, 4]** (will auto‑scale).
* Returns the **mean** LPIPS over the batch as Python `float`.

Install dependency
------------------
`pip install lpips>=0.1.4`
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import lpips as _lpips  # type: ignore


__all__ = ["compute_lpips"]

_Net = Literal["vgg", "alex", "squeeze"]


def _get_cache_dir() -> str:
    """Resolve lpips cache dir, defaulting to ~/.cache/lpips/."""
    return os.getenv("LPIPS_CACHE_DIR", str(Path.home() / ".cache" / "lpips"))


def _get_model(net: _Net, device: torch.device) -> _lpips.LPIPS:  # type: ignore
    cache_dir = Path(_get_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    # lpips uses torch.hub under the hood; we intercept HOME
    torch_home = os.environ.get("TORCH_HOME", "")
    os.environ["TORCH_HOME"] = str(cache_dir)

    try:
        model = _lpips.LPIPS(net=net).to(device)
    finally:
        if torch_home:
            os.environ["TORCH_HOME"] = torch_home  # restore env

    model.eval()
    return model


_models: dict[tuple[_Net, str], _lpips.LPIPS] = {}


@torch.inference_mode()
def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    /,
    *,
    net: _Net = "vgg",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Compute LPIPS score between `pred` and `target`.

    Parameters
    ----------
    pred, target : torch.Tensor
        Images in shape (N, C, H, W) or (C, H, W). Range can be [0,1] or [0,4].
    net : {'vgg','alex','squeeze'}
        Backbone network.
    device : str | torch.device
        Device to run on.

    Returns
    -------
    float
        Mean LPIPS score over the batch.
    """
    if pred.shape != target.shape:
        raise ValueError("Input shapes must match")
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    device_t = torch.device(device)
    key = (net, str(device_t))
    if key not in _models:
        _models[key] = _get_model(net, device_t)

    model = _models[key]

    # LPIPS expects inputs in [-1,1]
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        x = x / x.max()  # scale HDR 0–4 to 0–1 if needed
        return x * 2 - 1

    pred_n = _normalize(pred.to(device_t).float())
    target_n = _normalize(target.to(device_t).float())

    score = model(pred_n, target_n, normalize=False)  # returns N×1×1×1
    return float(score.mean().item())