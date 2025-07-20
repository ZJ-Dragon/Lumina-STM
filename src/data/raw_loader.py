"""
raw_loader.py
=============

Utility to convert Sony ARW / Nikon NEF (and other Bayer‑based RAW files)
into a 4‑channel linear tensor that can be fed directly to the Lumina
Transformer / Diffusion models.

The pipeline is:
    1. Read RAW with rawpy (LibRaw backend)
    2. Subtract per–channel black level
    3. Clip by white level and normalise to 0‑1
    4. Optionally apply camera white‑balance gains
    5. Pack the RGGB mosaic as (H/2, W/2, 4) tensor
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import rawpy


class RawLoader:
    """Decode RAW file and return 4‑channel linear tensor."""

    def __init__(
        self,
        *,
        subtract_black: bool = True,
        apply_white_balance: bool = True,
        cfa_pattern: str = "RGGB",
    ) -> None:
        """
        Parameters
        ----------
        subtract_black : bool
            If *True* subtract per‑channel black level before normalisation.
        apply_white_balance : bool
            If *True* multiply pixel values by the camera white‑balance gains.
        cfa_pattern : str
            CFA pattern string, currently only 'RGGB' is supported.
        """
        self.subtract_black = subtract_black
        self.apply_white_balance = apply_white_balance
        self.cfa_pattern = cfa_pattern.upper()
        if self.cfa_pattern != "RGGB":
            raise NotImplementedError("Only RGGB pattern is supported currently")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def load(self, file: Union[str, Path]) -> np.ndarray:
        """
        Load RAW file and return a packed 4‑channel tensor.

        Returns
        -------
        np.ndarray
            Array of shape (H/2, W/2, 4), dtype float32. Values are linear‑normalised by
            camera white‑level and **may exceed 1.0** after white‑balance gains.
            Channel order = [R, G(top), G(bottom), B].
        """
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"RAW file not found: {file}")

        with rawpy.imread(str(file)) as raw:
            mosaic = raw.raw_image_visible.astype(np.float32)
            raw_colors = raw.raw_colors_visible  # per‑pixel colour plane index

            # 1) subtract black level
            if self.subtract_black and raw.black_level_per_channel is not None:
                for ch_id, blk in enumerate(raw.black_level_per_channel):
                    mosaic[raw_colors == ch_id] -= blk

            # 2) normalise by white level (retain HDR > 1 values)
            white = (
                max(raw.camera_white_level_per_channel)
                if raw.camera_white_level_per_channel
                else raw.white_level
            )
            if white is None:
                white = 16383.0  # fallback for 14‑bit RAW
            # Ensure no negative values, but **do not clip the high end**.
            mosaic = np.maximum(mosaic, 0.0) / white

            # 3) apply camera white balance
            if self.apply_white_balance:
                gains = raw.camera_whitebalance or [1.0, 1.0, 1.0, 1.0]
                for ch_id, gain in enumerate(gains):
                    mosaic[raw_colors == ch_id] *= gain

            # 4) pack RGGB ➜ 4 channels (R, G1, G2, B)
            H, W = mosaic.shape
            tensor = np.empty((H // 2, W // 2, 4), dtype=np.float32)
            tensor[..., 0] = mosaic[0::2, 0::2]  # R
            tensor[..., 1] = mosaic[0::2, 1::2]  # G (row 0)
            tensor[..., 2] = mosaic[1::2, 0::2]  # G (row 1)
            tensor[..., 3] = mosaic[1::2, 1::2]  # B
        return tensor


# ------------------------------------------------------------------------- #
# Convenience wrapper
# ------------------------------------------------------------------------- #
def raw_to_tensor(file: Union[str, Path], **kwargs) -> np.ndarray:
    """
    Shorthand to load a RAW file into a 4‑channel tensor.

    Example
    -------
    >>> tensor = raw_to_tensor("DSC_0001.NEF")
    >>> print(tensor.shape)  # (H/2, W/2, 4)
    """
    return RawLoader(**kwargs).load(file)
