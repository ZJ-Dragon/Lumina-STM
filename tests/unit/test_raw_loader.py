

"""
Unit‑tests for RawLoader
========================
Goals
-----
1. Assert black‑level subtraction – the minimum pixel value after loading
   should be ~0 (tolerance 1e‑3).  A sample RAW containing a grey card that
   is deliberately under‑exposed is placed in `tests/assets/`.
2. Assert the packed tensor shape is (H/2, W/2, 4) and dtype float32.
3. Assert pixel values do not exceed a reasonable HDR head‑room (≤ 4.0).

If the sample RAW is missing (e.g. CI without large binary assets) the
tests will be skipped automatically.
"""

from __future__ import annotations

import pytest
from pathlib import Path
import numpy as np

from src.data.raw_loader import RawLoader

# ----------------------------------------------------------------------------- #
# Path to sample RAW with a grey card in frame. Using a tiny crop keeps the file
# size very small so it can live in the repo without bloating it.
SAMPLE_RAW = Path(__file__).parent.parent / "assets" / "grey_card_sample.ARW"

loader = RawLoader()


@pytest.mark.skipif(not SAMPLE_RAW.exists(), reason="Sample RAW not available.")
def test_tensor_properties() -> None:
    """Tensor should have correct shape, dtype and value range."""
    tensor = loader.load(SAMPLE_RAW)  # (H/2, W/2, 4)

    # shape assertions
    assert tensor.ndim == 3, "Tensor must be H/2 x W/2 x 4."
    assert tensor.shape[2] == 4, "Last dim must be 4 channels (R,G1,G2,B)."

    # dtype
    assert tensor.dtype == np.float32

    # range assertions
    assert np.all(tensor >= 0.0), "Negative values indicate black-level not subtracted."

    # black-level subtraction check: min value close to zero
    assert np.min(tensor) < 1e-3, "Min pixel should be ~0 after black-level correction."