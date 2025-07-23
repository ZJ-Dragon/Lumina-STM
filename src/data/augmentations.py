

"""
Lumina‑STM – Data Augmentation Utilities
---------------------------------------
This module builds image‑space augmentation pipelines based on
`configs/augmentations.yaml`.

* All images are assumed to be float32 numpy arrays in **H x W x C**
  format, linear HDR space with a nominal range **[0, 4.0]**.
* Each transform is implemented as a class with a configurable
  probability **p**. Whether it triggers is decided *inside* the
  pipeline so the same RNG governs the entire chain.
* The public helper :func:`build_pipeline` turns a parsed YAML config
  + a selected *mode* (“offline”, “online”, “none”) into a single
  callable that can be plugged into Dataset ``__getitem__`` or used
  by offline preprocessing scripts.

NOTE: This file is intentionally *lightweight*: heavy CV ops are kept
to straightforward numpy / OpenCV calls so as not to introduce large
dependencies. Extend or swap implementations as project demands grow.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List
import random
import math

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

# ---------------------------------------------------------------------------
# Base Transform
# ---------------------------------------------------------------------------

class Transform:
    """
    Minimal transform interface: subclasses override :meth:`_apply`.
    The wrapper handles probability gating and RNG injection.
    """

    def __init__(self, p: float):
        self.p = float(p)

    # Sub‑classes must override this
    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray: ...

    # Do *not* override
    def __call__(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        return self._apply(img, rng) if rng.random() < self.p else img


# ---------------------------------------------------------------------------
# Individual Transforms
# ---------------------------------------------------------------------------

class ExposureShift(Transform):
    def __init__(self, p: float, ev_range: List[float]):
        super().__init__(p)
        self.ev_min, self.ev_max = ev_range

    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        ev = rng.uniform(self.ev_min, self.ev_max)
        gain = math.pow(2.0, ev)
        return np.clip(img * gain, 0.0, 4.0, out=img.copy())


class WhiteBalanceJitter(Transform):
    def __init__(self, p: float, rgb_gain_range: List[float]):
        super().__init__(p)
        self.g_min, self.g_max = rgb_gain_range

    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        r_gain = rng.uniform(self.g_min, self.g_max)
        b_gain = rng.uniform(self.g_min, self.g_max)
        gains = np.array([r_gain, 1.0, b_gain], dtype=img.dtype)
        return np.clip(img * gains, 0.0, 4.0, out=img.copy())


class CurvePerturb(Transform):
    """
    Applies a simple toe/shoulder linear curve modification controlled
    by *pivot* and *slope*. Suitable as a lightweight tone‑mapping
    perturbation.
    """

    def __init__(self, p: float, slope_range: List[float], pivot_range: List[float]):
        super().__init__(p)
        self.s_min, self.s_max = slope_range
        self.pv_min, self.pv_max = pivot_range

    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        slope = rng.uniform(self.s_min, self.s_max)
        pivot = rng.uniform(self.pv_min, self.pv_max)
        out = img.copy()
        mask = out > pivot
        out[mask] = pivot + (out[mask] - pivot) * slope
        return np.clip(out, 0.0, 4.0, out=out)


class RandomNoise(Transform):
    def __init__(self, p: float, sigma_range: List[float], color_noise_ratio: float):
        super().__init__(p)
        self.s_min, self.s_max = sigma_range
        self.color_ratio = color_noise_ratio

    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        sigma = rng.uniform(self.s_min, self.s_max)
        h, w, c = img.shape
        noise = rng.normalvariate  # shortcut
        # White component
        n_white = np.array([[noise(0, sigma) for _ in range(c)] for _ in range(w * h)],
                           dtype=img.dtype).reshape((h, w, c))
        # Colour‑dependent component
        if self.color_ratio > 0.0:
            n_color = np.stack([rng.normalvariate(0, sigma) for _ in range(c)], axis=0)
            n_color = n_color.reshape((1, 1, c)).astype(img.dtype)
            n_white += self.color_ratio * n_color
        return np.clip(img + n_white, 0.0, 4.0, out=img.copy())


class GaussianBlur(Transform):
    def __init__(self, p: float, sigma_range: List[float]):
        super().__init__(p)
        self.s_min, self.s_max = sigma_range
        if cv2 is None:
            raise ImportError("opencv‑python is required for GaussianBlur")

    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        sigma = rng.uniform(self.s_min, self.s_max)
        ksize = int(max(3, round(sigma * 4) | 1))  # kernel size odd
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)


class HFlip(Transform):
    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        return np.flip(img, axis=1).copy()


class RandomCrop(Transform):
    def __init__(self, p: float, size: List[int]):
        super().__init__(p)
        self.crop_h, self.crop_w = size

    def _apply(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        h, w, _ = img.shape
        if h < self.crop_h or w < self.crop_w:
            raise ValueError("Crop size larger than input image.")
        top = rng.randint(0, h - self.crop_h)
        left = rng.randint(0, w - self.crop_w)
        return img[top : top + self.crop_h, left : left + self.crop_w, :].copy()


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

_NAME_TO_CLASS = {
    "exposure_shift": ExposureShift,
    "white_balance_jitter": WhiteBalanceJitter,
    "curve_perturb": CurvePerturb,
    "random_noise": RandomNoise,
    "gaussian_blur": GaussianBlur,
    "hflip": HFlip,
    "random_crop": RandomCrop,
}


def _construct(name: str, cfg: Dict[str, Any]) -> Transform:
    cls = _NAME_TO_CLASS[name]
    kwargs = {k: v for k, v in cfg.items() if k != "enable" and k != "p"}
    return cls(p=cfg.get("p", 1.0), **kwargs)  # type: ignore[arg-type]


def build_pipeline(cfg: Dict[str, Any], mode: str = "online", seed: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a deterministic augmentation pipeline.

    Parameters
    ----------
    cfg   : Parsed YAML dict from ``configs/augmentations.yaml``.
    mode  : One of ``offline``, ``online``, ``none``.
    seed  : Optional RNG seed overriding the config‑level ``global.rng_seed``.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that applies the chained transforms.
    """

    include = cfg["modes"][mode]["include"]
    rng_seed = cfg.get("global", {}).get("rng_seed", None) if seed is None else seed
    rng = random.Random(rng_seed)

    transforms: List[Transform] = []
    for name in include:
        t_cfg = cfg["transforms"][name]
        if not t_cfg.get("enable", True):
            continue
        transforms.append(_construct(name, t_cfg))

    def _apply(img: np.ndarray) -> np.ndarray:
        out = img
        for t in transforms:
            out = t(out, rng)
        return out

    return _apply


# Convenience shortcut for quick tests
if __name__ == "__main__":  # pragma: no cover
    import yaml, pathlib
    cfg_path = pathlib.Path(__file__).parents[2] / "configs" / "augmentations.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    pipeline = build_pipeline(cfg, mode="online", seed=0)
    dummy = np.ones((1024, 1024, 3), dtype=np.float32)
    out = pipeline(dummy)
    print("Pipeline applied, mean =", out.mean())