# data-raw-loader 功能分支目标

在整个 **Lumina-STM** 项目中，`data-raw-loader` 功能分支的目标是：  
把分散在外部磁盘 `$DATASETS_ROOT` 中的 Sony ARW / Nikon NEF 文件，批量转换成“模型可直接 ingest 的 4 通道线性张量”，并把这一转换流程做成 **可配置、可复现、可被 DataLoader 即刻消费** 的模块。  
它既要读懂 RAW 元数据（黑/白电平、白平衡、Bayer pattern），又要完成归一化、打包 space-to-depth 4 通道张量等预处理；同时还要和 DVC、CI 管线、后续增广流程无缝衔接。  

---

## 总览

1. **核心职责**  
   - 解码 ARW/NEF ⇒ NumPy  
   - 校正 黑电平、白电平、白平衡  
   - 打包 Bayer → `(H/2, W/2, 4)` 张量（R/G/G/B）  
   - 输出缓存为 `.npy` / `.pt`，并生成索引 JSON  
   - 提供 PyTorch `Dataset`+`DataLoader`，支持多进程读取、随机裁剪、RGB 预览  
   - 可选写入 DVC stage，复现命令一键 `dvc repro`

2. **与目录结构关系**  
   - 处理后文件落到早前规划的 `pipeline/raw_master/`、`pipeline/space2depth/`  
   - 代码全部放在 `src/data/`  
   - 不依赖项目根目录是否包含 `datasets/` —— 只要在 `.env` 或 YAML 中声明 `DATASETS_ROOT` 即可，无影响

---

## 分步文件/目录说明

| 步骤 | 新建/修改文件         | 位置              | 作用                                                                                              |
| ---- | ---------------------- | ----------------- | ------------------------------------------------------------------------------------------------- |
| 1    | `raw_loader.py`        | `src/data/`       | ① 读 RAW（`rawpy`/`libraw`）<br>② 校正黑白电平<br>③ 应用白平衡<br>④ 输出 4-ch 张量；<br>提供类 `RawLoader` + 函数 `raw_to_tensor()` |
| 2    | `datasets.py`          | `src/data/`       | 继承 `torch.utils.data.Dataset` 实现 `RawTensorDataset`，内部调用 `RawLoader`；<br>支持随机裁剪、白平衡扰动、多进程安全 |
| 3    | `__init__.py`          | `src/data/`       | 将 `RawLoader`、`RawTensorDataset` 暴露为包接口                                                     |
| 4    | `dataloader_raw.yaml`  | `configs/`        | 参数化：数据根路径、裁剪大小、增益范围、线程数；<br>供 `train.py` 读取                              |
| 5    | `raw_preprocess.py`    | `scripts/`        | CLI 脚本：批量扫描 `filters/*/raw/` → 生成 `pipeline/raw_master/` & `space2depth/`；<br>可写入 DVC stage |
| 6    | `test_raw_loader.py`   | `tests/unit/`     | 验证 ① 黑白电平扣除正确（灰卡像素≈0）<br>② 输出张量四通道形状 & 范围 0–1                              |
| 7    | `requirements.txt` (更新) | 根目录         | 添加 `rawpy`, `numpy`, `tqdm`；<br>Linux 下注明 `libraw` 动态库依赖                                  |
| 8    | `.env.example`         | 项目根            | 示范 `DATASETS_ROOT=/mnt/raw_pool`，让脚本通过 `os.getenv` 读取                                     |
| 9    | `dvc.yaml` (追加 stage) | 项目根           | `stages:` `preprocess_raw:` →<br>`cmd: python scripts/raw_preprocess.py --cfg configs/dataloader_raw.yaml`，<br>`outs` 指向 `pipeline/*` |
| 10   | README 片段            | `docs/`           | 写“使用 raw_loader”说明：安装依赖 → 设置数据根 → 运行预处理 → 训练                                 |

> **命名规范**：所有脚本/模块前缀保持小写短划线；配置文件用蛇形；测试文件与被测模块同名。

---

## 其他重要事项

1. **线程与多进程安全**  
   - `rawpy` 释放 C 资源要在 `__del__` 或 `with` 语句中关闭；避免 DataLoader 多进程复制句柄导致崩溃  
   - 建议在 `RawLoader.__enter__` 内使用  
     ```python
     rawpy.imread(..., threaded=False)
     ```  
     显式禁用内部 OpenMP，防止与 PyTorch DataLoader 线程抢 CPU

2. **Bayer 对齐裁剪**  
   - `RawTensorDataset` 的 `__getitem__` 在裁剪时必须保证左上角坐标为偶数，使 RGGB 相位不乱

3. **黑电平/白电平容差**  
   - 某些 ARW/NEF 通道黑电平不同；务必按 `raw.raw_colors_visible` 面罩逐像素扣除黑电平，而不是用单一值。参考 LibRaw 官方讨论

4. **可选 CCache**  
   - 若批量处理巨量 RAW，可在 `raw_preprocess.py` 后端加 `joblib` disk cache，避免重复解码

5. **单元测试基准**  
   - 放 1 张 Sony、1 张 Nikon RAW 在 `tests/assets/`；对生成张量计算 MD5，保证升级后 loader 行为一致

6. **错误日志**  
   - 对读取失败或损坏 RAW，记录 `logger.error` 并继续，生成 `failed_files.txt` 供人工核查

7. **未来扩展**  
   - 为适配 Quad-Bayer 手机 RAW 及压缩 RAW (CR3/HEIF-RAW)，仅需在 `RawLoader` 中新增 pattern 对应和压缩标志解析

---

按照上表实施 `data-raw-loader` 分支，即可在不破坏既有目录规划的前提下，为 Transformer / Diffusion 双线模型提供 **一致、可复现、效率高** 的 RAW→张量数据流。