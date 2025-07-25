#!/usr/bin/env python3
"""
Lumina‑STM – Offline Augmentation Generator
===========================================
Convert *raw_master* .npy tensors into an *aug_train* set using the
**offline** augmentation pipeline defined in `configs/augmentations.yaml`.

The script is designed to be *DVC‑friendly*: deterministic given the same
inputs + config, and supports multi‑process execution via the ``-j/--workers``
flag.

Example
-------
```
python scripts/gen_aug_train.py \
        datasets/pipeline/raw_master \
        datasets/pipeline/aug_train \
        --config configs/augmentations.yaml \
        -j 8
```
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

# project imports – ensure PYTHONPATH includes repo root
from src.data.augmentations import build_pipeline

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _process_one(
    src_path: Path,
    dst_root: Path,
    pipeline_cfg: dict,
    seed_offset: int,
) -> None:
    """Load a single .npy tensor, apply augmentation, save to aug_train dir."""
    rng_seed = seed_offset + hash(src_path.name) % 2_000_000_000
    pipeline = build_pipeline(pipeline_cfg, mode="offline", seed=rng_seed)

    out_rel = src_path.relative_to(src_path.parents[1])  # keep camera sub‑dir
    dst_path = dst_root / out_rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.load(src_path)
    arr_aug = pipeline(arr)

    np.save(dst_path, arr_aug)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def _gather_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.npy"))


def main(
    src_dir: str | Path,
    dst_dir: str | Path,
    cfg_path: str | Path,
    workers: int,
    seed: int | None,
) -> None:
    src_root = Path(src_dir).expanduser().resolve()
    dst_root = Path(dst_dir).expanduser().resolve()
    with Path(cfg_path).open("r") as f:
        pipeline_cfg = yaml.safe_load(f)

    files = _gather_files(src_root)
    if not files:
        raise RuntimeError(f"No .npy files found under {src_root}")

    # partial func for pool
    proc = partial(
        _process_one,
        dst_root=dst_root,
        pipeline_cfg=pipeline_cfg,
        seed_offset=0 if seed is None else seed,
    )

    workers = workers or mp.cpu_count()
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(proc, files))

    print(f"[gen_aug_train] Done. {len(files)} files ➜ {dst_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate offline‑augmented training tensors."
    )
    parser.add_argument(
        "src",
        type=str,
        help="Source directory (raw_master).",
    )
    parser.add_argument(
        "dst",
        type=str,
        help="Destination directory (aug_train).",
    )
    parser.add_argument(
        "--config",
        default="configs/augmentations.yaml",
        type=str,
        help="Augmentation YAML config.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0 ➜ CPU count).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed for deterministic output.",
    )

    args = parser.parse_args()
    main(args.src, args.dst, args.config, args.workers, args.seed)
