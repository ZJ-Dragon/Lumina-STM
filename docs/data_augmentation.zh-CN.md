

<!--
  Lumina‑STM | 数据增强手册（中文版）
  ==================================
  本文档阐述我们为什么以及如何对线性 HDR RAW 张量进行数据增强，
  同时给出离线 / 在线两条流程及实战调参建议。
-->

## 概览

Lumina‑STM 训练的数据为线性 HDR 张量，数值范围 **0 – 4** EV。  
为避免过拟合，并让网络能适应曝光误差、白平衡漂移及传感器噪声，我们采用 **两条互补** 的增强路径：

| 模式 | 何时应用 | 是否落盘 | 典型场景 |
|------|----------|----------|----------|
| **离线** | 在预处理完成后一次性执行 | ✔ `datasets/pipeline/aug_train/` | 集群大规模训练（节省训练时 CPU） |
| **在线** | `Dataset.__getitem__` 里即时执行 | ✖ | 小批量试验 / 过拟合检查 |

二者均由同一份 YAML（`configs/augmentations.yaml`）控制，保证数学逻辑与参数空间一致。

---

## 1. 算法原理

### 1.1 曝光偏移  
相机曝光误差可简化为标量增益 *G = 2^EV*。  
我们在 **[‑2, +2]** EV 内采样，并在增益后裁剪回 `0–4`，保持高光过渡。

### 1.2 白平衡抖动  
对 R、B 通道乘以 **[0.9, 1.1]** 的增益，G 通道固定。  
可训练网络在非常规色温下也不偏色。

### 1.3 曲线扰动  
轻量级 toe / shoulder 调整：

```text
if x > pivot:  y = pivot + slope · (x – pivot)
else:          y = x
```

参数 **pivot ∈ [0.3, 0.7]**，**slope ∈ [0.8, 1.2]**。  
模拟不同机内色调曲线且保持可逆。

### 1.4 噪声注入  
添加 *白噪声*（σ **5 e‑4 – 1 e‑2**），并可混入一定比例的色彩相关噪声（≤ 0.5）。  
逼真地复现传感器 / ISP 噪声特性，同时保证 RAW 空间的信噪比。

### 1.5 几何 & 模糊  
水平翻转（50 %）及轻度高斯模糊（σ **0.2–1.0 px**）丰富纹理统计，但不破坏拜耳相位。  
垂直翻转默认关闭以保持方向元数据一致，可在 YAML 中开启。

---

## 2. 离线流程

```
raw_master/*.npy        ┐
                        │
        gen_aug_train.py│  (多进程)
                        ▼
aug_train/*.npy         ┘
```

1. **触发命令**  

   ```bash
   python scripts/raw_preprocess.py --cfg ... --augment offline -j 8
   ```

2. **确定性**  
   每个文件的随机种子取自文件名哈希 → 通过 DVC (`dvc repro augment_train`) 可完全复现。  
3. **存储膨胀**  
   视启用算子不同，磁盘占用约为 `raw_master` 的 ×1.0 – 1.3。

---

## 3. 在线流程

1. 每个 DataLoader worker 通过 `build_pipeline(cfg, mode="online")` 构造一次管线。  
2. 在 `__getitem__` 中，该管线于白平衡抖动 **之后**、张量转 Torch **之前** 执行。  
3. Dataloader 配置示例  

   ```yaml
   use_augmentation: true
   augmentation_cfg: configs/augmentations.yaml
   ```

4. 当 CPU 相对空闲、需快速调参时尤为方便。

---

## 4. 调参指北

| 症状 | 可能方案 | YAML 参数 |
|------|----------|-----------|
| 高光过曝 | 提高 **pivot** 上限或降低 **slope** | `curve_perturb.pivot_range` / `slope_range` |
| 日光场景偏色 | 将 **rgb_gain_range** 收窄到 [0.95, 1.05] | `white_balance_jitter.rgb_gain_range` |
| 训练早期发散 | 将 **EV 范围** 缩小到 ±1 | `exposure_shift.ev_range` |
| 画面过于平滑 | 关闭 **gaussian_blur** 或将其 `p` 降至 0.1 | `gaussian_blur.p`, `sigma_range` |
| 在线模式 GPU 吃不满 | 切换到 **离线**，重新生成 `aug_train/` | CLI `--augment offline` |

> **提示：** 每次修改 YAML 后都建议先运行 `pytest -k augment`，可及早捕获参数越界。

---

## 5. 后续扩展

* 在感知色彩空间（如 JzAzBz）中加入 **Hue/Sat 随机化**，待后续升级至光谱感知训练后启用。  
* 引入 **可学习 LUT** 由小型 MLP 驱动，模拟专有胶片模拟预设。  
* 根据硬件型号控制的 **传感器缺陷**（热像素、CFA 偏位）随机注入。

---

愿训练顺利！🚀  
若有新的增强想法，欢迎随时在 GitHub 提 Issue。