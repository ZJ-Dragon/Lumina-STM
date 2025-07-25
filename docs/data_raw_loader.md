# Lumina‑STM

**Lumina‑STM** (Style‑Transform Module) is an end‑to‑end pipeline that learns to turn **Sony / Nikon RAW photos** into high‑quality, stylised colour renditions—locally with a lightweight Transformer and in the cloud with a diffusion refiner.

---

## ✨ What’s inside — branch `feature/data‑raw‑loader`

| Module | Purpose |
|--------|---------|
| `src/data/raw_loader.py` | Decode `.ARW` / `.NEF` → subtract black‑level → white‑level normalise → optional WB → **4‑ch RGGB tensor** |
| `src/data/datasets.py` | `RawTensorDataset` with Bayer‑aligned random crop & white‑balance jitter |
| `scripts/raw_preprocess.py` | CLI to batch convert **`filters/*/raw/`** into `pipeline/raw_master/` + `pipeline/space2depth/` |
| `tests/unit/test_raw_loader.py` | Unit tests: shape / range / black‑level correctness |
| `configs/dataloader_raw.yaml` | Dataset root, crop size, WB jitter, DataLoader workers |
| `src/data/augmentations.py` | Compose exposure/WB/noise transforms (shared YAML) |
| `scripts/gen_aug_train.py` | Multiprocess offline augmentation into `pipeline/aug_train/` |
| `configs/augmentations.yaml` | Global switchboard for both offline & online aug |
| `dvc.yaml` (stage augment_train) | Reproducible offline augmentation via DVC |

---

## 🛠 Quick‑start

```bash
git clone https://github.com/your‑org/Lumina‑STM.git
cd Lumina‑STM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. copy & edit environment file
cp .env.example .env           # edit paths / keys
source .env                    # or use python‑dotenv

# 2. generate tensors from RAW + (optionally) offline augmentation
dvc repro preprocess_raw                    # base tensors
dvc repro augment_train                     # extra augmented tensors
# or in one go:
python scripts/raw_preprocess.py --cfg configs/dataloader_raw.yaml --augment offline -j 8

# 3. smoke‑test dataloader
pytest tests/unit/test_raw_loader.py -q
```

---

## 🗂 Directory layout (simplified)

```
Lumina‑STM/
 ├─ filters/               # raw photos organised by filter name
 │   └─ <filter>/raw/*.ARW
 ├─ pipeline/              # autogenerated by raw_preprocess.py
 │   ├─ raw_master/        # linear Bayer (H,W,1)
 │   ├─ space2depth/       # packed 4‑ch tensors (H/2,W/2,4)
 │   └─ aug_train/         # offline‑augmented 4‑ch tensors
 ├─ src/
 │   └─ data/              # RawLoader & Dataset
 └─ configs/               # YAML configs
```

---

## ⚙️ Environment variables

All runtime configuration lives in a `.env` file (ignored by Git).  
` .env.example ` documents every key—copy & customise.

| Key | Example | Notes |
|-----|---------|-------|
| **DATASETS_ROOT** | `/mnt/raw_pool/Lumina‑STM` | **Must point to the folder that contains `filters/` & `pipeline/`** |
| OUTPUT_ROOT | `./pipeline` | Where checkpoints / logs are stored |
| CUDA_VISIBLE_DEVICES | `0,1` | GPU selection (empty = CPU) |
| NUM_WORKERS | `8` | Default DataLoader worker count |
| AWS_* | *optional* | Needed only if you push data / ckpt to S3 |
| WANDB_* | *optional* | Set if you use Weights‑and‑Biases tracking |

> **Tip:** edit `.env.example`, commit it; keep your real `.env` local & private.

---

## 🧪 Testing & CI

```bash
pytest -q                      # unit tests
dvc repro preprocess_raw       # data pipeline reproducibility
```

Add a GitHub Action (see `.github/workflows/ci.yml`) to run both commands on every PR.

---

## 🖍 Data augmentation hook

*Offline* and *online* augmentation now share a single YAML (`configs/augmentations.yaml`).  
*Offline* tensors live in **`pipeline/aug_train/`** and are created by:

```bash
python scripts/gen_aug_train.py pipeline/raw_master pipeline/aug_train -j 8
```

For the theory and tuning tips, see **`docs/data_augmentation.md`** (EN) or **`docs/data_augmentation.zh‑CN.md`** (CN).

---

## 📜 License

This repository is released under the **Apache 2.0** licence (see `LICENSE`).