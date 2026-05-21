# SETUP — 训练服务器从零跑起来

目标：装好依赖 → 填两个路径 → `bash scripts/build.sh && bash scripts/run.sh` 直接跑。

适用：Ubuntu / Debian 系（其他发行版自行对应包名）。需要 NVIDIA GPU。

---

## 0. 一句话流程

```bash
# 1. 装 NVIDIA driver + CUDA toolkit + PyTorch + libzip + numpy/scipy
# 2. 在 scripts/paths.cfg.local 填本机的 DATA_DIR / LIBTORCH / NVCC
# 3. bash scripts/build.sh
# 4. bash scripts/run.sh
```

下面是每一步的细节。

---

## 1. 系统前置

### 1.1 NVIDIA 驱动

云 GPU 实例通常自带，本地机器需自己装。检查：

```bash
nvidia-smi   # 能输出 GPU 列表就 OK
```

没装的话（Ubuntu）：

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

或手动从 https://www.nvidia.com/Download/index.aspx 选对应卡的驱动 `.run` 文件。

### 1.2 编译工具链

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
cmake --version   # ≥ 3.18
gcc --version     # ≥ 9
```

---

## 2. CUDA Toolkit（提供 `nvcc`）

驱动 ≠ Toolkit。`nvidia-smi` 能跑不代表 `nvcc` 能用。

### 下载

官网：https://developer.nvidia.com/cuda-downloads
（选 Linux → x86_64 → 你的发行版 → 版本 → `deb (network)` 最省心）

**版本选择**：要和你将要装的 PyTorch 的 CUDA 版本**主版本一致**（例如 PyTorch CUDA 12.x → CUDA Toolkit 12.x）。版本不匹配大概率能编过但运行时崩。

### 安装后

CUDA 一般装到 `/usr/local/cuda-XX.Y/`，并提供 `/usr/local/cuda` 软链。验证：

```bash
/usr/local/cuda/bin/nvcc --version
```

如果你的 `nvcc` 不在这个路径，**记下绝对路径**，等会儿填到 `paths.cfg.local`。

可选：把 CUDA 加进 PATH（不加也行，因为我们在 paths.cfg 用绝对路径）：

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## 3. PyTorch（同时提供 LibTorch）

**重点**：本项目的 C++ 端用的就是 PyTorch 安装包里自带的 LibTorch（在 `site-packages/torch/`），**不用单独下 LibTorch tarball**。这样可以彻底避免 ABI 不匹配（`_GLIBCXX_USE_CXX11_ABI`、CUDA 运行时版本等问题）。

### 装 PyTorch

去 https://pytorch.org/get-started/locally/ 选对应 OS / Package / CUDA 版本，复制一行命令跑。比如 CUDA 12.4：

```bash
# pip
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 或 conda
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 找出 LibTorch 路径

```bash
python -c 'import torch, os; print(os.path.dirname(torch.__file__))'
```

输出类似：
- conda：`/root/miniconda3/lib/python3.12/site-packages/torch`
- venv：`/home/user/venv/lib/python3.12/site-packages/torch`
- 系统 pip：`/usr/lib/python3/dist-packages/torch`

**记下来**，等会儿填到 `paths.cfg.local`。

验证 PyTorch 看到 GPU：

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
# 期望：True N
```

---

## 4. 其余 Python 依赖

项目用到的纯 Python 库就两个（PyTorch 已装）：

```bash
pip install numpy scipy
```

---

## 5. libzip

C++ 端的 NPZ 写出依赖 libzip：

```bash
sudo apt install -y libzip-dev
```

---

## 6. 配置本机路径（`scripts/paths.cfg.local`）

仓库里 `scripts/paths.cfg` 是版本控制的默认值。**不要改它**，改了会跟同事 conflict。
每台服务器自己写一份 `scripts/paths.cfg.local`（已被 `.gitignore` 忽略）：

```bash
cat > scripts/paths.cfg.local << 'EOF'
# 数据目录（self-play / shuffled / models / checkpoints / logs 都在这下面）
DATA_DIR=/srv/training/skyzero_runA

# LibTorch = PyTorch 安装目录（步骤 3 里查到的）
LIBTORCH=/root/miniconda3/lib/python3.12/site-packages/torch

# nvcc 绝对路径（步骤 2 里确认的）
NVCC=/usr/local/cuda/bin/nvcc
EOF
```

或者懒人版（让 `LIBTORCH` 自动从当前 Python 探测）：

```bash
cat > scripts/paths.cfg.local << 'EOF'
DATA_DIR=/srv/training/skyzero_runA
LIBTORCH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
NVCC=/usr/local/cuda/bin/nvcc
EOF
```

### 优先级（同名变量）

`paths.cfg.local`（裸 `=` 赋值）> 调用前导出的 env 变量 > `paths.cfg` 默认值

一次性试别的路径：`DATA_DIR=/tmp/test bash scripts/run.sh`（不动 `.local`）。

---

## 7. 训练超参的本机覆盖（可选，`scripts/run.cfg.local`）

`scripts/run.cfg` 是训练超参（MCTS 模拟数、batch、GPU 分配……）。每台机器不同的话，同理写 `scripts/run.cfg.local`：

```bash
cat > scripts/run.cfg.local << 'EOF'
# 这台机器有 4 张卡
INFERENCE_SERVER_DEVICES=0,1,2,3
NUM_INFERENCE_SERVERS=4

# 每 iter 多跑点
GAMES_PER_ITER=8000
EOF
```

shell 和 C++ 两边都会读到这个覆盖（C++ 端在 `parse_cfg()` 里做了一样的尾接逻辑）。

**注意**：`MAX_BOARD_SIZE` 改了的话会触发 cmake 重 configure + C++ 重编（`run.sh` 每 iter 都 `cmake --build` 一次，自动处理）。

---

## 8. 构建 + 跑

```bash
bash scripts/build.sh
bash scripts/run.sh           # 一直跑，Ctrl+C 停
# 或限定 iter 数：
bash scripts/run.sh 100
```

`build.sh` 第一次会 cmake configure 一次，之后只增量编。如果路径写错，`build.sh` 会直接报错并提示怎么改 `paths.cfg.local`。

`run.sh` 自动：
- 探测 GPU 数量；>1 卡时启动 self-play daemon 利用副卡
- 从 `$DATA_DIR/checkpoints/state.json` 续跑
- 没有初始模型时自动 random-init
- 每个 iter 跑 selfplay → shuffle → train → export → mcts probe

---

## 9. 常见坑

| 症状 | 多半是 |
|---|---|
| `cmake` 报 `Could NOT find Torch` | `LIBTORCH` 路径不对，或这个目录里没 `share/cmake/Torch/TorchConfig.cmake`（说明 PyTorch 装坏了/装的是不带 LibTorch 的纯 wheel） |
| 编出来运行 `undefined symbol __cxa_*` / `GLIBCXX_*` | 你用了系统 g++ 但 PyTorch 编译时用的是更新的 ABI。要么升 g++，要么换 conda 里的 g++ |
| 运行时 `CUDA error: no kernel image for device` | `CUDA_ARCH` 不匹配。`scripts/build.sh` 默认从 `nvidia-smi` 探测；强制指定：`CUDA_ARCH=89 bash scripts/build.sh`（Ada=89, Hopper=90, Blackwell=120, Turing=75, Ampere=80/86） |
| `nvcc fatal: unsupported gpu architecture` | 你的 CUDA Toolkit 太旧不认识你卡的 arch，升 CUDA |
| `nvidia-smi` 能跑但 `torch.cuda.is_available()` False | PyTorch 装的是 CPU 版（pip 默认就是 CPU）。重装时务必带 `--index-url .../whl/cuXXX` |
| `bash scripts/run.sh` 报找不到 `DATA_DIR` 下某文件 | 第一次跑会自己 bootstrap；如果之前删过文件造成中间状态，干脆 `rm -rf $DATA_DIR` 重头来 |
| 想在不同目录跑多个 run | 给每个 worktree / clone 单独写一份 `paths.cfg.local`，`DATA_DIR` 指向不同路径即可 |

---

## 10. 检查清单

跑之前对一下：

- [ ] `nvidia-smi` 列出预期 GPU
- [ ] `/usr/local/cuda/bin/nvcc --version`（或你的 `NVCC` 路径）能跑
- [ ] `python -c 'import torch; print(torch.cuda.is_available())'` → `True`
- [ ] `python -c 'import numpy, scipy'` 无报错
- [ ] `pkg-config --modversion libzip` 有版本号
- [ ] `scripts/paths.cfg.local` 三个变量都填好
- [ ] `bash scripts/build.sh` 一路绿
- [ ] `bash scripts/run.sh 1` 跑通一个 iter
