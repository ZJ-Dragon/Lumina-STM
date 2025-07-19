# Lumina‑STM

**Lumina‑STM**（Style‑Transform Module）是一条端到端管道，可在本地使...量级 Transformer，并在云端通过 diffusion refiner 完成。

---

## ✨ 内部内容 — 分支 `feature/data‑raw‑loader`

| 模块 | 作用 |
|--------|---------|
| `src/data/raw_loader.py` | 解码 `.ARW` / `.NEF` → 扣黑电平 → 白电平归一化 → 可选 WB → **4‑通道 RGGB 张量** |
| `src/data/datasets.py` | `RawTensorDataset`：Bayer 对齐随机裁剪 & 白平衡抖动 |
| `scripts/raw_preprocess.py` | CLI 将 **`filte.../raw/`** 批量转换为 `pipeline/raw_master/` + `pipeline/space2depth/` |
| `tests/unit/test_raw_loader.py` | 单元测试：形状 / 数值范围 / 黑电平正确性 |
| `configs/dataloader_raw.yaml` | 数据根目录、裁剪尺寸、WB 抖动、DataLoader 线程数 |
| `.env.example` | 运行时环境变量模板 |
| `dvc.yaml`（stage `preprocess_raw`） | 单命令可复现的 **DVC** 预处理 |

---

## 🛠 快速开始

```bash
git clone https://github.com/your‑org/Lumina‑STM.git
cd Lumina‑STM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. 复制并编辑环境文件
cp .env.example .env           # 修改路径 / 密钥
source .env                    # 或使用 python‑dotenv

# 2. 从 RAW 生成张量
dvc repro preprocess_raw       # 或：python scripts/raw_preprocess.py --cfg configs/dataloader_raw.yaml

# 3. DataLoader 烟雾测试
pytest tests/unit/test_raw_loader.py -q
```

---

## 🗂 目录结构（简化版）

```
Lumina‑STM/
 ├─ filters/               # 按滤镜名组织 raw 照片
 │   └─ <filter>/raw/*.ARW
 ├─ pipeline/              # raw_preprocess.py 自动生成
 │   ├─ raw_master/        # 线性 Bayer (H,W,1)
 │   └─ space2depth/       # 打包 4‑通道张量 (H/2,W/2,4)
 ├─ src/
 │   └─ data/              # RawLoader & Dataset
 └─ configs/               # YAML 配置
```

---

## ⚙️ 环境变量

| 键 | 示例 | 说明 |
|-----|---------|-------|
| **DATASETS_ROOT** | `/mnt/raw_pool/Lumina‑STM` | **必须指向同时含有 `filters/` & `pipeline/` 的目录** |
| OUTPUT_ROOT | `./pipeline` | 存放 checkpoint / 日志 |
| CUDA_VISIBLE_DEVICES | `0,1` | GPU 选择（空 = CPU） |
| NUM_WORKERS | `8` | DataLoader 工作进程 |
| AWS_* | *可选* | 仅在将数据 / 权重推送到 S3 时需要 |
| WANDB_* | *可选* | 若使用 Weights‑and‑Biases 进行实验跟踪 |

---

## 🧪 测试与 CI

```bash
pytest -q                      # 单元测试
dvc repro preprocess_raw       # 数据管线可复现
```

请在 `.github/workflows/ci.yml` 中添加 GitHub Action，以在每次 PR 时运行这两个命令。

---

## 📜 许可证

本仓库采用 **Apache 2.0** 许可（见 `LICENSE`）。