

"""
Unit tests for src.data.augmentations
------------------------------------
1. Per–transform sanity:
   * Output shape must equal input shape.
   * Output pixel range must remain within [0, 4.0].
2. Pipeline determinism:
   * With the same RNG seed, the *online* pipeline must produce identical
     results across multiple invocations on the same input.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

AUG_CFG_PATH = Path(__file__).parents[2] / "configs" / "augmentations.yaml"
AUG_MOD = importlib.import_module("src.data.augmentations")


@pytest.fixture(scope="module")
def dummy_img():
    rng = np.random.default_rng(0)
    return rng.random((256, 256, 3), dtype=np.float32) * 4.0  # HDR range 0‑4


# --------------------------------------------------------------------------- #
# 1. Per‑transform sanity                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(AUG_MOD._NAME_TO_CLASS.keys()))
def test_transform_shape_value_range(dummy_img, name):
    """Each individual transform keeps shape & range."""
    cls = AUG_MOD._NAME_TO_CLASS[name]
    # Grab default kwargs from YAML
    import yaml

    with AUG_CFG_PATH.open("r") as f:
        cfg_yaml = yaml.safe_load(f)
    t_cfg = cfg_yaml["transforms"][name]
    # Force always‑on for test
    t_cfg = {**t_cfg, "p": 1.0}
    transform = AUG_MOD._construct(name, t_cfg)

    out = transform(dummy_img, np.random.RandomState(42))

    assert out.shape == dummy_img.shape
    assert out.dtype == dummy_img.dtype
    assert np.all(out >= 0.0) and np.all(out <= 4.0), f"Range violated for {name}"


# --------------------------------------------------------------------------- #
# 2. Pipeline determinism                                                     #
# --------------------------------------------------------------------------- #

def test_pipeline_determinism(dummy_img):
    import yaml

    with AUG_CFG_PATH.open("r") as f:
        cfg_yaml = yaml.safe_load(f)

    pipe_a = AUG_MOD.build_pipeline(cfg_yaml, mode="online", seed=123)
    pipe_b = AUG_MOD.build_pipeline(cfg_yaml, mode="online", seed=123)

    out1 = pipe_a(dummy_img)
    out2 = pipe_b(dummy_img)

    np.testing.assert_array_equal(out1, out2, err_msg="Pipeline not deterministic")