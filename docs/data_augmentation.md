<!--
  Lumina‑STM | Data Augmentation Handbook
  ======================================
  This document explains *why* and *how* we augment RAW‑tensor data for both
  Transformer and Diffusion branches, and offers practical tuning advice.
-->

## Overview

Lumina‑STM trains on linear‑HDR tensors whose numeric range spans **0 – 4** EV.
To prevent over‑fitting and make the network robust to exposure mistakes, white‑balance drift and sensor noise, we adopt *two* complementary augmentation paths:

| Mode | When applied | Saved to disk? | Typical use‑case |
|------|--------------|----------------|------------------|
| **Offline** | Once, right after preprocessing | ✔ `datasets/pipeline/aug_train/` | Large‑scale training on fixed clusters (less CPU load during epochs) |
| **Online** | On‑the‑fly inside `Dataset.__getitem__` | ✖ | Rapid prototyping or overfit‑checks on a small sample set |

The same YAML (`configs/augmentations.yaml`) governs *both* modes, ensuring identical math and parameter spaces.

---

## 1. Algorithmic Principles

### 1.1 Exposure Shift  
Camera exposure errors translate to a scalar gain *G = 2^EV*.  
We sample **EV ∈ [–2, +2]** (configurable) and clip back to `0–4` afterwards, preserving highlight roll‑off.

### 1.2 White‑Balance Jitter  
We multiply R & B channels by gains sampled from **[0.9, 1.1]** while keeping G constant.  
This teaches the network to handle atypical CCTs without colour casts in the output.

### 1.3 Tone‑Curve Perturbation  
A light‑weight toe/shoulder modification:  
```text
if x > pivot:  y = pivot + slope · (x – pivot)
else:          y = x
```  
Parameters **pivot ∈ [0.3, 0.7]**, **slope ∈ [0.8, 1.2]** emulate different in‑camera tone profiles yet remain invertible.

### 1.4 Noise Injection  
We add *white* Gaussian noise (σ range **5 e‑4 – 1 e‑2**) plus an optional chroma‑correlated term (ratio ≤ 0.5).  
This mimics sensor/ISP patterns while keeping SNR realistic for RAW space.

### 1.5 Geometric & Blur  
Horizontal flip (50 %) and mild Gaussian blur (σ **0.2–1.0 px**) enrich local texture statistics without hurting Bayer alignment.  
Vertical flip is disabled by default to keep orientation metadata intact, but can be toggled in YAML.

---

## 2. Offline Workflow

```
raw_master/*.npy        ┐
                        │
        gen_aug_train.py│  (multiprocess)
                        ▼
aug_train/*.npy         ┘
```

1. **Trigger**:  
   ```bash
   python scripts/raw_preprocess.py --cfg ... --augment offline -j 8
   ```
2. **Determinism**: Each file gets a seed derived from its filename hash → output is fully reproducible via DVC (`dvc repro augment_train`).
3. **Storage Footprint**: Expect ×1.0 – 1.3 disk growth vs. `raw_master` depending on enabled transforms.

---

## 3. Online Workflow

1. `RawTensorDataset` builds the pipeline once per worker using `build_pipeline(cfg, mode="online")`.
2. During `__getitem__`, the pipeline is executed **after** white‑balance jitter and **before** Tensor → Torch conversion.
3. Set in the dataloader config:
   ```yaml
   use_augmentation: true
   augmentation_cfg: configs/augmentations.yaml
   ```
4. Useful for quick hyper‑parameter sweeps where CPU overhead is negligible compared to GPU idling.

---

## 4. Tuning Guide

| Symptom | Likely Fix | YAML Knob |
|---------|-----------|-----------|
| Model over‑exposes highlights | Increase **pivot** upper bound or reduce **slope** | `curve_perturb.pivot_range` / `slope_range` |
| Colour cast on daylight shots | Narrow **rgb_gain_range** to around [0.95, 1.05] | `white_balance_jitter.rgb_gain_range` |
| Training diverges early | Lower **EV range** to ±1 | `exposure_shift.ev_range` |
| Too smooth / lack detail | Disable **gaussian_blur** or drop its `p` to 0.1 | `gaussian_blur.p`, `sigma_range` |
| GPU under‑utilised (online mode) | Switch to **offline** and regenerate `aug_train/` once | CLI `--augment offline` |

> **Tip:** Always run `pytest -k augment` after tweaking YAML to catch out‑of‑range params early.

---

## 5. Future Extensions

* **Hue/Sat randomisation** in perceptual colour spaces (e.g. JzAzBz) once we migrate to spectral‑aware training.
* **Learnable LUT** augmentation driven by a small MLP to mimic proprietary film simulations.
* **Synthetic sensor defects** (hot pixels, CFA mis‑alignment) gated by hardware model ID.

---

Happy training! 🚀 Feel free to open a GitHub issue whenever new augmentation ideas surface.
