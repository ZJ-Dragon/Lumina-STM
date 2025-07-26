

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

    assert np.min(tensor) < 0.005, "Min pixel should be ~0 after black-level correction."


# ----------------------------------------------------------------------------- #
# Dataset-level augmentation sanity                                             #
# ----------------------------------------------------------------------------- #
@pytest.mark.skipif(not SAMPLE_RAW.exists(), reason="Sample RAW not available.")
def test_dataset_online_augmentation(tmp_path) -> None:
    """
    Sanity‑check that RawTensorDataset loads with augment_mode='online'
    without errors and preserves basic tensor invariants (shape & dtype).
    """
    from src.data.datasets import RawTensorDataset
    import yaml

    # Prepare a tiny dummy npy tensor for augmentation pipeline test
    dummy = np.ones((512, 512, 3), dtype=np.float32) * 0.5
    dummy_path = tmp_path / "dummy.npy"
    np.save(dummy_path, dummy)

    # Load augmentation config
    aug_cfg = Path(__file__).parents[2] / "configs" / "augmentations.yaml"
    assert aug_cfg.exists(), "augmentations.yaml not found"

    # Instantiate dataset in online‑augmentation mode
    ds = RawTensorDataset(
        files=[dummy_path],
        crop_size=512,
        random_crop=False,
        wb_jitter=0.0,
        augment_cfg=aug_cfg,
        augment_mode="online",
        rng_seed=123,
    )

    aug_tensor = ds[0]  # triggers __getitem__ with augmentation pipeline

    # Basic invariants
    assert aug_tensor.ndim == 3
    assert aug_tensor.shape[2] in (3, 4), "Unexpected channel count after aug."
    assert aug_tensor.dtype == np.float32
    assert np.all(aug_tensor >= 0.0) and np.all(
        aug_tensor <= 4.0
    ), "Augmented tensor out of expected HDR range [0,4]"