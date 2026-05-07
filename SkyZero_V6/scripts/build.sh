#!/usr/bin/env bash
# Build all V6 C++ targets.
#
# CUDA 13 + CMake 3.28 quirks on this box:
#   * nvcc isn't on PATH (it's at /usr/local/cuda/bin/nvcc)
#   * CUDA architecture auto-detection fails
# So we pass both explicitly. Override via env vars if your setup differs.
#
# Usage:
#   bash scripts/build.sh                     # configure (if needed) + build all
#   bash scripts/build.sh --target selfplay_main   # extra args go to cmake --build
#   LIBTORCH=/path/to/libtorch bash scripts/build.sh
#   CUDA_ARCH=75 bash scripts/build.sh
#   rm -rf cpp/build && bash scripts/build.sh # force fresh configure

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
SRC_DIR="$ROOT/cpp"
BUILD_DIR="$SRC_DIR/build"

LIBTORCH="${LIBTORCH:-/home/sky/libtorch}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-89}"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "[build.sh] configure: BUILD_DIR=$BUILD_DIR LIBTORCH=$LIBTORCH NVCC=$NVCC CUDA_ARCH=$CUDA_ARCH"
    cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
        -DCMAKE_PREFIX_PATH="$LIBTORCH" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$NVCC"
fi

echo "[build.sh] build"
cmake --build "$BUILD_DIR" -j "$@"
