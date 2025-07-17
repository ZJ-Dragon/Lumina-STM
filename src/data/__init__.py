

"""
Lumina-STM data sub‑package
===========================

This package provides:
    * RawLoader          – low‑level RAW → 4‑ch tensor converter
    * raw_to_tensor()    – convenience helper around RawLoader
    * RawTensorDataset   – PyTorch Dataset wrapping RawLoader with
                           Bayer‑aligned random cropping & WB jitter
"""

from .raw_loader import RawLoader, raw_to_tensor
from .datasets import RawTensorDataset

__all__ = [
    "RawLoader",
    "raw_to_tensor",
    "RawTensorDataset",
]