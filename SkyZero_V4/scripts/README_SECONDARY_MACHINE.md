# 副机运行 selfplay 指南

适用于**显卡架构与主机不同**的副机（worker machine），按本指南重新编译 C++ 端即可让 `selfplay.sh` 在副机上跑起来并通过 Syncthing 向主机贡献对弈数据。

> 主机当前使用 **RTX 4090（Ada Lovelace, sm_89）**。如果副机也是 4090 或其他 Ada 卡（4070/4080/4090），则**不需要**重编，直接跑 `scripts/selfplay.sh`。只有架构不同（Blackwell / Ampere / Hopper 等）时才需要本文档。

---

## 1. 环境要求

副机需要满足：

| 组件 | 要求 | 如何查 |
|------|------|--------|
| NVIDIA 驱动 | 支持你这张卡的版本 | `nvidia-smi` 第一行 |
| CUDA Toolkit | 能覆盖你卡的 compute capability | `/usr/local/cuda/bin/nvcc --version` |
| PyTorch | 其 `get_arch_list()` 包含你卡的 `sm_XX` | 见下 |
| conda 环境 | 与主机一致，本项目统一用 `pytorch` 环境 | `conda activate pytorch` |

验证 PyTorch 端 kernel 覆盖：

```bash
conda activate pytorch
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

输出里必须看到你目标卡的 `sm_XX`。例如 RTX 5060 需要看到 `sm_120`：

```
2.11.0+cu130 13.0 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

如果不包含，你需要升级 PyTorch（Blackwell 卡需要 ≥ 2.7 的 cu128 / cu13x wheel）。

### 常见显卡 → compute capability 映射

| GPU | 架构 | `TORCH_CUDA_ARCH_LIST` |
|-----|------|-----------------------|
| RTX 30xx（3060 / 3070 / 3080 / 3090） | Ampere | `8.6` |
| RTX A6000 / A100 | Ampere | `8.0` / `8.6` |
| H100 | Hopper | `9.0` |
| RTX 40xx（4070 / 4080 / 4090） | Ada Lovelace | `8.9` ← 主机 |
| RTX 50xx（5060 / 5070 / 5080 / 5090） | Blackwell | `12.0` |

不确定时：

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"
# 输出 (12, 0) 就填 "12.0"，输出 (8, 9) 就填 "8.9"
```

---

## 2. 需要修改的代码点

**唯一**改动位置：`cpp/CMakeLists.txt` 第 8 行的 `TORCH_CUDA_ARCH_LIST`。

```cmake
# cpp/CMakeLists.txt:5-9
# Set CUDA arch: override via -DTORCH_CUDA_ARCH_LIST="X.Y" if needed
# 8.9 = RTX 4090, 12.0 = RTX 5060/5090
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "8.9")
endif()
```

有两种做法，**任选其一**：

### 做法 A — 命令行覆盖（推荐，不动源文件）

长期和主机共享同一个 git 仓库时用这种，cmake 时加 `-DTORCH_CUDA_ARCH_LIST=...` 即可，源文件保持 `"8.9"` 不变，主副机都能干净地 checkout。

### 做法 B — 直接改源文件

如果这台副机是你独占的工作机，且你不想每次 cmake 都敲长命令，就把第 8 行的 `"8.9"` 改成你的目标值（例如 `"12.0"`）。注意**别把这个改动 commit 回主分支**。

---

## 3. 编译 C++ 端

```bash
conda activate pytorch
cd /path/to/SkyZero_V4/cpp

# 1) 清掉主机编译的残留（重要：主机 build 目录里的 CMake cache 锁定了 sm_89）
rm -rf build && mkdir build && cd build

# 2) cmake 配置阶段
#    把下面的 "12.0" 换成你副机对应的值
cmake -DTORCH_CUDA_ARCH_LIST="12.0" ..

# 若报 "Could not find a package configuration file provided by Torch"，
# 把 conda 里 libtorch 的 cmake 路径显式传给 -DCMAKE_PREFIX_PATH：
#   cmake -DTORCH_CUDA_ARCH_LIST="12.0" \
#         -DCMAKE_PREFIX_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake'))") \
#         ..

# 3) 编译
make -j$(nproc) 2>&1 | tee build.log
```

### 验证编译结果

```bash
# a) 编译时确实带了你的 arch（把 120 换成你对应的值）
grep -E "compute_120|sm_120" build.log | head

# b) 二进制产出
ls -lh selfplay

# c) 动态库能找到
ldd selfplay | grep -E "libtorch|libcuda"
```

第 a) 条没有输出就说明 arch 没吃进去，回去检查 `TORCH_CUDA_ARCH_LIST` 是否传对了。

---

## 4. 运行前准备

`selfplay.sh` 启动后会到 `data/models/` 下找 `*.pt`，找不到会一直等。

**项目默认目录结构**（`BASEDIR` 默认 `<项目根>/data`）：

```
data/
├── models/                     # 主机导出的模型 (*.pt) 放这里
└── selfplay/
    └── <NODE_ID>/              # 本副机生成的对弈数据
```

### 4.1 指定 NODE_ID（决定数据落在哪个子目录）

**不需要改 `selfplay.sh`**。脚本会根据 `NODE_ID` 自动在 `data/selfplay/` 下建出对应的子目录并把 `.npz` 写进去（出处 `scripts/selfplay.sh:30,99`）。

三种方式（优先级由高到低）：

1. **命令行环境变量**（临时指定）：
   ```bash
   NODE_ID=node2 GPU=0 bash scripts/selfplay.sh
   ```

2. **写进 `scripts/run.cfg`**（长期固定用这个 node_id，推荐）：
   ```bash
   # scripts/run.cfg 里加一行
   NODE_ID=node2
   ```
   注意 `selfplay.sh:87-91` 的加载顺序是先用脚本内默认值、再 `source run.cfg`；但因为 `NODE_ID="${NODE_ID:-$(hostname)}"` 已经在第 30 行展开过，**命令行环境变量优先级高于 run.cfg**。

3. **什么都不设**：默认用 `$(hostname)`。如果副机 hostname 本身就有区分度（例如 `worker-01`），可以直接用默认值。

**建议**：多副机场景下给每台机器起一个稳定、易辨认的 `NODE_ID`（如 `node2`、`node3`），方便后面 Syncthing 共享只这台机器的子目录。

### 4.2 Syncthing 配置（两个单向 Folder）

很关键：**不要**给整个 `data/` 或 `data/selfplay/` 建一个双向同步 Folder，否则多台副机的子目录会互相覆盖或触发冲突。正确姿势是建**两个独立的、方向相反的单向 Folder**。

#### Folder 1 — 主机 → 副机：`data/models/`

| 端   | 路径                         | Folder Type  |
|------|------------------------------|--------------|
| 主机 | `<项目根>/data/models`       | Send Only    |
| 副机 | `<项目根>/data/models`       | Receive Only |

作用：主机训练导出的最新 `*.pt` 自动推到副机，`selfplay.sh` 启动时在这里找权重（见 `scripts/selfplay.sh:15`）。

#### Folder 2 — 副机 → 主机：`data/selfplay/<NODE_ID>/`

| 端   | 路径                                          | Folder Type  |
|------|-----------------------------------------------|--------------|
| 副机 | `<项目根>/data/selfplay/<NODE_ID>`            | Send Only    |
| 主机 | `<项目根>/data/selfplay/<NODE_ID>`            | Receive Only |

作用：副机产出的对弈 `.npz` 回流到主机。主机 `shuffle.py` 用 `os.walk` 递归读 `data/selfplay/` 下所有子目录，所以只要落进 `data/selfplay/<NODE_ID>/` 就会被训练吃到。

**配置步骤建议**：

1. 先在副机上跑一次（可以用第 5 节的冒烟参数），让 `selfplay.sh` 自动建出 `data/selfplay/<NODE_ID>/` 目录。
2. 再去 Syncthing UI 上共享这个已存在的目录，省得手工 `mkdir` 或对不齐路径。
3. 两个 Folder 的 Folder ID 建议起有意义的名字，例如 `skyzero-models` 和 `skyzero-selfplay-node2`，多台副机一目了然。
4. Send/Receive Only 的方向性不是可有可无的装饰 —— 它能防止副机本地误改的 `data/models/` 被推回主机覆盖正在导出的权重。

#### scp（临时验证用）

如果只是想先冒烟测试一下，不想立即碰 Syncthing：

```bash
mkdir -p data/models
scp <主机用户>@<主机IP>:/path/to/SkyZero_V4/data/models/'*.pt' data/models/
```

冒烟跑通后再按上面两个 Folder 的结构接入 Syncthing。

---

## 5. 冒烟测试（第一次运行务必先跑这个）

RTX 5060 等 8 GB 卡显存比 4090（24 GB）小很多，直接跑默认参数大概率 OOM。先用极小参数跑两局确认链路通：

```bash
cd /path/to/SkyZero_V4
GPU=0 \
  MAX_GAMES=2 \
  NUM_WORKERS=4 \
  NUM_SIMULATIONS=64 \
  INFERENCE_BATCH=32 \
  LEAF_BATCH=8 \
  bash scripts/selfplay.sh
```

### 预期观察

1. stdout 打印头部：
   ```
   === SkyZero V4 Selfplay Worker ===
   Node: <hostname> | GPU: 0 | Board: 15x15 | Renju: true
   ```
2. 接着打印 `| Device: cuda`（来自 `cpp/selfplay_main.cpp:166`）。
3. 另开一个终端 `watch -n1 nvidia-smi` —— 应看到 `selfplay` 进程占用目标 GPU 的显存（几百 MB ~ 几 GB）。
4. 在 `data/selfplay/<NODE_ID>/` 下开始出现 `.npz` 文件。
5. 两局跑完后程序不会自己退出（`selfplay.sh` 是无限循环），Ctrl-C 停掉即可，这就算冒烟通过。

### 失败信号

- `no kernel image available for execution on the device` / `CUDA error: no kernel image` → arch 没编进去，回到第 3 节清 `build/` 重编。
- `CUDA out of memory` → 参数还是太大，继续下调（见第 7 节）。
- `libtorch.so: cannot open shared object file` → 缺库路径：
  ```bash
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
  ```
  `selfplay.sh:102-104` 已经帮你设了一份，但如果是手动直接跑 `./cpp/build/selfplay` 可能要自己 export。

---

## 6. 正式运行

冒烟通过后用默认参数起：

```bash
cd /path/to/SkyZero_V4
GPU=0 bash scripts/selfplay.sh
```

常用的环境变量覆盖：

```bash
NODE_ID=node2 \          # 区分多台副机，数据落到 data/selfplay/node2/
BASEDIR=/mnt/ssd/skyzero \  # 换数据目录（默认 <项目根>/data）
GPU=0 \                  # 选卡（多卡机器指定 GPU 索引）
bash scripts/selfplay.sh
```

更系统化的参数可以写到 `scripts/run.cfg`，`selfplay.sh` 会自动 source 这个文件（见 `scripts/selfplay.sh:88-91`）。环境变量优先级高于 `run.cfg`。

---

## 7. 8 GB 卡（5060 / 3070 等）参数建议

作为起点，显存紧张时可以先用这组：

```bash
NUM_WORKERS=16
NUM_SIMULATIONS=128
INFERENCE_BATCH=64
LEAF_BATCH=16
```

再根据 `nvidia-smi` 的显存占用慢慢往上调。主要的显存消费来自 `NUM_WORKERS × 棋盘张量` 和 `INFERENCE_BATCH × 模型`，从这两个下手最有效。

默认值（`selfplay.sh:44-59`）是给 24 GB 的 4090 调的：`NUM_WORKERS=32`、`NUM_SIMULATIONS=256`、`INFERENCE_BATCH=256`、`LEAF_BATCH=32`。

---

## 8. 常见问题速查

| 症状 | 原因 | 解决 |
|------|------|------|
| `no kernel image available for execution on the device` | 二进制里没编目标 arch 的 kernel | 清 `cpp/build/` 重编，确认 `-DTORCH_CUDA_ARCH_LIST` 传对 |
| `torch.cuda.get_arch_list()` 里没有目标 `sm_XX` | PyTorch 版本太旧 | 升级 PyTorch 到支持你架构的版本 |
| `CUDA out of memory` | 参数默认是为 4090 24GB 调的 | 按第 7 节调小 `NUM_WORKERS` / `INFERENCE_BATCH` / `NUM_SIMULATIONS` |
| `libtorch.so: cannot open shared object file` | `LD_LIBRARY_PATH` 不含 conda lib | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`，或直接 `bash scripts/selfplay.sh`（脚本自己会设） |
| cmake 报 `Could not find a package configuration file provided by Torch` | libtorch 的 cmake 路径没传 | 加 `-DCMAKE_PREFIX_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake'))")` |
| 一直卡在 `No model found in .../models — waiting...` | Syncthing 还没把模型同步过来 | 等 Syncthing，或先 scp 一个 `.pt` 过去验证链路 |
| 主机看不到副机贡献的数据 | `data/selfplay/<NODE_ID>/` 没被 Syncthing 同步回主机 | 检查 Syncthing 配置，确认该目录在共享文件夹里 |
