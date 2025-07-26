"""
Lumina-STM – Metrics public API
Expose unified callable interfaces for PSNR / SSIM / LPIPS / FID and a
pluggable registry so downstream code (CLI / trainers) can request metrics
by name.
"""
from __future__ import annotations

from typing import Callable, Dict

from .psnr import compute_psnr
from .ssim import compute_ssim
from .lpips import compute_lpips
from .fid import compute_fid

# -------------------------------------------------------------------------
# Public registry
# -------------------------------------------------------------------------
MetricFn = Callable[..., float]

METRIC_REGISTRY: Dict[str, MetricFn] = {
    "psnr": compute_psnr,
    "ssim": compute_ssim,
    "lpips": compute_lpips,
    "fid": compute_fid,
}

def register_metric(name: str, fn: MetricFn) -> None:
    """Register a custom metric at runtime.

    Parameters
    ----------
    name : str
        Metric key (case-insensitive externally – we store as lower-case).
    fn : Callable[..., float]
        A function that returns a single scalar (float). Can accept **kwargs.
    """
    key = name.lower()
    if key in METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' already exists in registry.")
    METRIC_REGISTRY[key] = fn

def get_metric(name: str) -> MetricFn:
    """Fetch a metric function by name (case-insensitive)."""
    try:
        return METRIC_REGISTRY[name.lower()]
    except KeyError as e:
        raise KeyError(f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}") from e

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_fid",
    "METRIC_REGISTRY",
    "register_metric",
    "get_metric",
]
