"""
Lumina-STM – Fréchet Inception Distance (FID)

Wrapper around `torch-fidelity` with .npz cache for reference stats.

Features
--------
* Automatic download (or reuse) of InceptionV3 2048‑d pool3 features.
* Supports:
  1. `compute_fid(pred_dir, gt_dir, cache_npz="...")` – directory trees.
  2. `compute_fid_tensor(pred_tensor, gt_tensor)` – batched tensors in memory.
* Caches ground‑truth activations to `.npz` for fast re‑runs.

Install
-------
pip install torch-fidelity>=0.3
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch_fidelity as tf  # type: ignore

__all__ = ["compute_fid", "compute_fid_tensor"]


def _calc_stats_npz(img_dir: Path, npz_path: Path) -> Dict[str, Any]:
    """Compute stats and save to npz; returns dict compatible with torch-fidelity."""
    metrics = tf.calculate_metrics(
        input1=str(img_dir),
        input2=None,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        feature_extractor=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_cpu_ram=True,
        verbose=False,
    )
    # torch-fidelity returns {'frechet_inception_distance': value}
    mu = metrics["frechet_inception_distance_feature_activations"]
    sigma = metrics["frechet_inception_distance_feature_activations_cov"]
    np.savez(npz_path, mu=mu, sigma=sigma)
    return {"mu": mu, "sigma": sigma}


def _load_stats(npz_path: Path) -> Dict[str, Any]:
    """Load stats dict (mu, sigma) from npz."""
    data = np.load(npz_path)
    return {"mu": data["mu"], "sigma": data["sigma"]}


def compute_fid(
    pred_dir: str | Path,
    gt_dir: str | Path,
    *,
    cache_npz: str | Path | None = None,
) -> float:
    """
    Compute FID between two directories of images.

    Parameters
    ----------
    pred_dir : str | Path
        Directory containing model-generated images.
    gt_dir : str | Path
        Directory containing ground-truth images.
    cache_npz : str | Path, optional
        Path to ground‑truth .npz statistics file. If provided and exists,
        will be loaded; otherwise computed and saved.

    Returns
    -------
    float
        FID score.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    if cache_npz is not None:
        cache_npz = Path(cache_npz)
        if cache_npz.exists():
            gt_stats = _load_stats(cache_npz)
        else:
            gt_stats = _calc_stats_npz(gt_dir, cache_npz)
    else:
        gt_stats = None

    kwargs = dict(
        input1=str(pred_dir),
        input2=str(gt_dir),
        batch_size=50,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        feature_extractor=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_cpu_ram=True,
        verbose=False,
    )
    if gt_stats is not None:
        kwargs["input2"] = gt_stats  # torch‑fidelity accepts dict

    metrics = tf.calculate_metrics(**kwargs)
    return float(metrics["frechet_inception_distance"])


@torch.inference_mode()
def compute_fid_tensor(
    preds: torch.Tensor,
    gts: torch.Tensor,
) -> float:
    """
    Compute FID directly from tensors (N,C,H,W) in [0,1] or [0,4].

    Notes
    -----
    * This is slower for large datasets; intended for small batches or testing.
    """
    if preds.shape != gts.shape:
        raise ValueError("preds and gts must have same shape")

    def _norm(x: torch.Tensor) -> torch.Tensor:
        x = x / x.max()
        return x.clamp(0, 1)

    metrics = tf.calculate_metrics(
        input1=_norm(preds),
        input2=_norm(gts),
        isc=False,
        fid=True,
        kid=False,
        feature_extractor=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])
