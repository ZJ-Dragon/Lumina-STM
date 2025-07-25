#!/usr/bin/env python3
"""
raw_preprocess.py
-----------------
CLI utility that scans `$DATASETS_ROOT/filters/*/raw/` for ARW / NEF files,
converts them to:

    1. pipeline/raw_master/<filter>/<stem>.npy    (H,W,1) linear Bayer
    2. pipeline/space2depth/<filter>/<stem>.npy   (H/2,W/2,4) packed tensor

Usage
-----
$ python scripts/raw_preprocess.py --cfg configs/dataloader_raw.yaml
    --augment {offline,online,none}   # offline ➜ run gen_aug_train.py

The YAML must provide:
    dataset_root : str
    crop_size    : int (ignored here, but parsed for completeness)
    wb_jitter    : float (ignored here)
    num_workers  : int
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from src.data.raw_loader import RawLoader


# ------------------------------------------------------------------ #
def parse_cfg(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if "dataset_root" not in cfg:
        raise KeyError("dataset_root missing in cfg")
    return cfg


def gather_raw_files(dataset_root: Path) -> List[Tuple[Path, str]]:
    """Return list of (file_path, filter_name) tuples."""
    files: List[Tuple[Path, str]] = []
    filter_dirs = (dataset_root / "filters").glob("*")
    for f_dir in filter_dirs:
        raw_dir = f_dir / "raw"
        if not raw_dir.exists():
            continue
        for raw_file in raw_dir.iterdir():
            if raw_file.suffix.lower() in {".arw", ".nef", ".dng"}:
                files.append((raw_file, f_dir.name))
    return files


# ------------------------------------------------------------------ #
_loader = RawLoader()  # one per worker


def _process(args: Tuple[Path, str, Path, Path]) -> None:
    raw_path, filter_name, out_master_root, out_packed_root = args
    try:
        tensor = _loader.load(raw_path)  # (H/2,W/2,4)

        # Save raw_master (single channel linear) for reference / debugging
        master_dir = out_master_root / filter_name
        master_dir.mkdir(parents=True, exist_ok=True)
        # Reconstruct full-resolution single-channel Bayer image
        bayer = np.zeros((tensor.shape[0] * 2, tensor.shape[1] * 2), dtype=np.float32)
        bayer[0::2, 0::2] = tensor[..., 0]  # R
        bayer[0::2, 1::2] = tensor[..., 1]  # G1
        bayer[1::2, 0::2] = tensor[..., 2]  # G2
        bayer[1::2, 1::2] = tensor[..., 3]  # B
        np.save(master_dir / f"{raw_path.stem}.npy", bayer.astype(np.float16))

        # Save packed 4‑channel tensor
        packed_dir = out_packed_root / filter_name
        packed_dir.mkdir(parents=True, exist_ok=True)
        np.save(packed_dir / f"{raw_path.stem}.npy", tensor.astype(np.float16))
    except Exception as e:
        tqdm.write(f"[ERROR] {raw_path.name}: {e}")


# ------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True, help="Path to YAML config")
    ap.add_argument(
        "--overwrite", action="store_true", help="Delete previous outputs first"
    )
    ap.add_argument(
        "--augment",
        choices=["offline", "online", "none"],
        default="none",
        help="Data augmentation mode: "
             "'offline' runs scripts/gen_aug_train.py, "
             "'online' handled inside Dataset, "
             "'none' disables augmentation.",
    )
    args = ap.parse_args()

    cfg = parse_cfg(args.cfg)
    root = Path(cfg["dataset_root"]).expanduser().resolve()

    out_master_root = (root / "pipeline" / "raw_master").resolve()
    out_packed_root = (root / "pipeline" / "space2depth").resolve()

    if args.overwrite:
        for d in (out_master_root, out_packed_root):
            if d.exists():
                shutil.rmtree(d)

    raw_files = gather_raw_files(root)
    print(f"Found {len(raw_files)} RAW files")

    tasks = [
        (path, filt, out_master_root, out_packed_root) for path, filt in raw_files
    ]
    with mp.Pool(cfg.get("num_workers", mp.cpu_count())) as pool:
        list(
            tqdm(
                pool.imap_unordered(_process, tasks),
                total=len(tasks),
                ncols=80,
                desc="processing",
            )
        )

    print("Finished preprocessing.")

    # ------------------------------------------------------------------
    # Trigger offline augmentation if requested
    # ------------------------------------------------------------------
    if args.augment == "offline":
        print("[raw_preprocess] ▶ Running offline augmentation ...")
        cmd = [
            sys.executable,
            "scripts/gen_aug_train.py",
            str(out_master_root),
            str(out_master_root.parent / "aug_train"),
            "--config",
            "configs/augmentations.yaml",
            "-j",
            str(cfg.get("num_workers", mp.cpu_count())),
        ]
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()