#!/usr/bin/env bash
# 一键在 WSL2(Windows 下的 Ubuntu)上装齐 SkyZero 依赖。
#
# 和裸机 Ubuntu(SETUP.md §0)的关键区别 —— 这也是本脚本存在的理由:
#   * GPU 驱动装在 *Windows 宿主机* 上,WSL 里 **绝不能** 装 NVIDIA 驱动。
#   * 因此 CUDA 只装 Toolkit(给 nvcc 编 C++ 用),并且要用 NVIDIA 专门的
#     `wsl-ubuntu` 仓库 + `cuda-toolkit-XX-Y` 包(不是 `cuda` 元包,后者会拉驱动,
#     把宿主机的 GPU 直通搞坏)。
#
# 用法(在 WSL2 的 Ubuntu 里,项目根目录下):
#   bash scripts/setup_wsl2.sh
#
# 老卡 / 宿主机驱动较旧导致 torch.cuda.is_available() 为 False 时,把 CUDA 版本一起降:
#   CUDA_VER=12.4 TORCH_CUDA=cu124 bash scripts/setup_wsl2.sh
#
# 幂等:已装的步骤会自动跳过,可重复运行。
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ---- 可调版本 ----------------------------------------------------------------
# 默认覆盖较新的卡(含 Blackwell / RTX 5090,需 >=12.8)。CUDA_VER 与 TORCH_CUDA
# 的主版本必须一致(都是 12.x)。老卡或老驱动就一起降到 12.4 / cu124。
CUDA_VER="${CUDA_VER:-12.8}"        # CUDA Toolkit 次版本(给 nvcc 编 C++)
TORCH_CUDA="${TORCH_CUDA:-cu128}"   # PyTorch wheel 的 CUDA tag(主版本须同 CUDA_VER)
CONDA_ENV="${CONDA_ENV:-pytorch}"   # conda 环境名(项目约定用 pytorch)
PY_VER="${PY_VER:-3.12}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

log() { printf '\n\033[1;34m[setup_wsl2]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup_wsl2] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 0. 必须是 WSL2 ----------------------------------------------------------
grep -qi microsoft /proc/version \
    || die "这不是 WSL2 环境。本脚本只用于 Windows 下的 WSL2 Ubuntu;裸机请照 SETUP.md。"

# ---- 1. 宿主机 GPU 驱动(装在 Windows 上,不在 WSL 里)-----------------------
if ! { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }; then
    die "nvidia-smi 跑不起来 → 请在 *Windows* 上安装/更新 NVIDIA 显卡驱动(WSL 里不要装驱动),
       关掉再重开 WSL 后重试。WSL 的 GPU 直通完全依赖宿主机驱动。"
fi
log "GPU 驱动 OK:"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

# ---- 2. apt 基础工具链 -------------------------------------------------------
log "apt 安装 build-essential / cmake / libzip-dev ..."
sudo -E apt-get update
sudo -E apt-get install -y build-essential cmake git pkg-config libzip-dev wget ca-certificates

# ---- 3. CUDA Toolkit(wsl-ubuntu 仓库,仅 toolkit,绝不装驱动)--------------
if [[ -x "/usr/local/cuda-${CUDA_VER}/bin/nvcc" || -x /usr/local/cuda/bin/nvcc ]]; then
    log "已检测到 nvcc,跳过 CUDA Toolkit 安装。"
else
    log "安装 CUDA Toolkit ${CUDA_VER}(wsl-ubuntu 仓库,仅 toolkit)..."
    tmpdeb="$(mktemp --suffix=.deb)"
    wget -qO "$tmpdeb" \
        https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "$tmpdeb"
    rm -f "$tmpdeb"
    sudo -E apt-get update
    sudo -E apt-get install -y "cuda-toolkit-${CUDA_VER//./-}"
fi
NVCC_BIN="/usr/local/cuda-${CUDA_VER}/bin/nvcc"
[[ -x "$NVCC_BIN" ]] || NVCC_BIN="/usr/local/cuda/bin/nvcc"
[[ -x "$NVCC_BIN" ]] || die "装完仍找不到 nvcc(查 /usr/local/cuda*/bin/)。"

# ---- 4. Miniconda ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    if [[ ! -d "$HOME/miniconda3" ]]; then
        log "安装 Miniconda 到 ~/miniconda3 ..."
        wget -qO /tmp/miniconda.sh \
            https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
        rm -f /tmp/miniconda.sh
    fi
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda init bash >/dev/null   # 让以后新开的 shell 也能用 conda(会改 ~/.bashrc)
else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ---- 5. pytorch conda 环境 ---------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    log "创建 conda 环境 '$CONDA_ENV'(python $PY_VER)..."
    conda create -y -n "$CONDA_ENV" "python=$PY_VER"
fi
set +u                       # conda activate 在 set -u 下会引用未绑定变量
conda activate "$CONDA_ENV"
set -u

# ---- 6. PyTorch(自带 LibTorch)+ numpy/scipy -------------------------------
if python -c 'import torch' 2>/dev/null; then
    log "环境里已有 torch,跳过(要重装先 pip uninstall torch)。"
else
    log "安装 PyTorch ($TORCH_CUDA) + numpy/scipy ..."
    pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch
    pip install numpy scipy
fi

# ---- 7. 写 env_paths.cfg.local(build.sh / run.sh 读这个)-------------------
LIBTORCH_DIR="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
PY_BIN="$(which python)"
LOCAL="$SCRIPT_DIR/env_paths.cfg.local"
if [[ -f "$LOCAL" ]]; then
    cp "$LOCAL" "$LOCAL.bak"
    log "原 env_paths.cfg.local 已备份为 env_paths.cfg.local.bak"
fi
cat > "$LOCAL" <<EOF
# 由 scripts/setup_wsl2.sh 在 WSL2 上自动生成。
LIBTORCH="$LIBTORCH_DIR"
NVCC=$NVCC_BIN
PY="$PY_BIN"
EOF
log "已写 $LOCAL:"
sed 's/^/    /' "$LOCAL"

# ---- 8. 验证 -----------------------------------------------------------------
log "==== 验证 ===="
"$NVCC_BIN" --version | grep -i release || true
python -c 'import torch; print("torch", torch.__version__, "| cuda.is_available =", torch.cuda.is_available(), "| device_count =", torch.cuda.device_count())'
python -c 'import numpy, scipy; print("numpy/scipy OK")'
printf 'libzip '; pkg-config --modversion libzip

cat <<EOF

==== 装好了。下一步 ====
  conda activate $CONDA_ENV        # 若提示找不到 conda,先重开 WSL 终端或 source ~/.bashrc
  bash scripts/build.sh            # 首次 cmake configure + 编译(3-5 分钟)
  bash scripts/run.sh 1            # 跑通一个 iter

若上面 cuda.is_available 为 False:多半是 Windows 宿主机驱动太旧,不支持 $TORCH_CUDA。
更新 Windows 上的 NVIDIA 驱动,或把 CUDA 版本一起降后重跑本脚本:
  CUDA_VER=12.4 TORCH_CUDA=cu124 bash scripts/setup_wsl2.sh
EOF
