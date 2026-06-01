#!/usr/bin/env bash
# Build the 2048 C++ binaries (selfplay2048, selfplay2048_par).
#
# CUDA / CMake on this box:
#   * nvcc is at /usr/local/cuda/bin/nvcc (not on PATH)
#   * CUDA arch auto-detect can fail, so we pass it explicitly
# LIBTORCH / NVCC / PY live in scripts/env_paths.cfg (+ .local overrides).
#
# Usage:
#   bash scripts/build.sh                          # configure (if needed) + build all
#   bash scripts/build.sh --target selfplay2048_par
#   CUDA_ARCH=120 bash scripts/build.sh
#   rm -rf cpp/build && bash scripts/build.sh       # force fresh configure
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
SRC_DIR="$ROOT/cpp"
BUILD_DIR="$SRC_DIR/build"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_paths.cfg"

if [[ ! -d "$LIBTORCH" ]]; then
    echo "[build.sh] ERROR: LIBTORCH=$LIBTORCH does not exist. Set it in scripts/env_paths.cfg.local:" >&2
    echo "[build.sh]   LIBTORCH=\"\$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')\"" >&2
    exit 1
fi
if [[ ! -x "$NVCC" ]]; then
    echo "[build.sh] ERROR: NVCC=$NVCC not found. Install CUDA toolkit / set NVCC in env_paths.cfg.local." >&2
    exit 1
fi

if [[ -z "${CUDA_ARCH:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    fi
    CUDA_ARCH="${CUDA_ARCH:-89}"
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "[build.sh] configure: LIBTORCH=$LIBTORCH NVCC=$NVCC CUDA_ARCH=$CUDA_ARCH"
    cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
        -DCMAKE_PREFIX_PATH="$LIBTORCH" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$NVCC"
fi

echo "[build.sh] build"
cmake --build "$BUILD_DIR" -j "$@"
